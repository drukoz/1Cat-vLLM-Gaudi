# SPDX-License-Identifier: Apache-2.0
"""Torch reference for the fused hyper-connection kernels (``csrc/hc_fused``).

The first five functions are verbatim copies of the pure-torch glue functions the HPU model runs
today, ``vllm_gaudi/models/qwen4_exp/hyperconnection.py``.  They are copied because the TPC
simulator container has no ``vllm_gaudi``; ``test_reference_copy.py`` (CPU, needs vLLM) asserts the
copies return the same bits as the plugin's functions.  Do not edit them here: change the plugin
file and copy again.

The ``*_reference`` functions below them define what each fused op returns, in terms of those
copies.  Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Verbatim copies (vllm_gaudi/models/qwen4_exp/hyperconnection.py)
# ---------------------------------------------------------------------------
def grouped_gemma_rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int) -> torch.Tensor:
    """Grouped Gemma RMSNorm (torch replacement of ``grouped_gemma_rmsnorm``).

    Each group of ``x.shape[-1] // num_groups`` elements is normalized
    independently with Gemma's ``(1 + w)`` affine; ``weight`` holds either
    one shared affine per group (``numel == group_dim``) or a distinct
    affine for every element of the flat layout (``numel == DIM``).
    """
    dim = x.shape[-1]
    group_dim = dim // num_groups
    assert dim % num_groups == 0
    assert weight.numel() in (group_dim, dim)

    input_dtype = x.dtype
    xg = x.float().unflatten(-1, (num_groups, group_dim))
    rrms = torch.rsqrt(xg.square().mean(dim=-1, keepdim=True) + eps)
    y = xg * rrms
    w = weight.float()
    w = w.view(group_dim) if weight.numel() == group_dim else w.view(num_groups, group_dim)
    # Gemma's (1 + w) affine, written like the kernel to stay an FMA.
    y = y + y * w
    return y.flatten(-2).to(input_dtype)


def hc_silu_torch(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Silu with the ``/ hc_count`` prescale (torch replacement of ``hc_silu``)."""
    input_dtype = x.dtype
    xf = x.float() / hc_count
    return (xf * torch.sigmoid(xf)).to(input_dtype)


def hc_gate_mix_torch(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Sigmoid-gated mean over the HC streams (torch replacement of ``hc_gate_mix``).

    ``x`` and ``gate`` share the flat ``[..., HC*HS]`` layout; the output is
    the single ``[..., HS]`` block input.
    """
    assert x.shape == gate.shape
    dim = gate.shape[-1]
    assert dim % hc_count == 0
    hc_dim = dim // hc_count

    input_dtype = x.dtype
    xg = x.float().unflatten(-1, (hc_count, hc_dim))
    g = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, hc_dim))
    acc = (g * xg).sum(dim=-2) / hc_count
    return acc.to(input_dtype)


def hc_combine_torch(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Inject a block output into the HC streams (torch replacement of ``hc_combine``).

    The injection logits become weights via ``2 * sigmoid(logits / hc_count)``,
    exactly as the combine kernel computes them.
    """
    dim = residual.shape[-1]
    assert dim % hc_count == 0
    hc_dim = dim // hc_count
    assert block_output.shape[-1] == hc_dim
    assert injection_logits.shape[-1] == hc_count

    input_dtype = residual.dtype
    w = (2.0 * torch.sigmoid(injection_logits.float() / hc_count)).unsqueeze(-1)
    res = residual.float().unflatten(-1, (hc_count, hc_dim))
    block = block_output.float().unsqueeze(-2)
    return (res + block * w).flatten(-2).to(input_dtype)


def hc_combine_norm_torch(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused combine + grouped Gemma RMSNorm (torch replacement of ``hc_combine_norm``).

    Returns the combined multi-stream residual **and** its grouped norm.
    The combine result is rounded to the input dtype before normalization,
    matching the fused kernel's materialized combine -> RMSNorm boundary.
    """
    dim = residual.shape[-1]
    assert dim % hc_count == 0
    hc_dim = dim // hc_count
    assert block_output.shape[-1] == hc_dim
    assert injection_logits.shape[-1] == hc_count
    assert norm_weight.numel() in (hc_dim, dim)

    input_dtype = residual.dtype
    w = (2.0 * torch.sigmoid(injection_logits.float() / hc_count)).unsqueeze(-1)
    res = residual.float().unflatten(-1, (hc_count, hc_dim))
    block = block_output.float().unsqueeze(-2)
    out = (res + block * w).flatten(-2).to(input_dtype)

    xn = grouped_gemma_rmsnorm_torch(out, norm_weight, eps, hc_count)
    return out, xn


COPIED_FUNCTIONS = ("grouped_gemma_rmsnorm_torch", "hc_silu_torch", "hc_gate_mix_torch", "hc_combine_torch",
                    "hc_combine_norm_torch")


# ---------------------------------------------------------------------------
# What each fused op returns
# ---------------------------------------------------------------------------
def hc_norm_reference(x: torch.Tensor, norm_weight: torch.Tensor, eps: float, hc_count: int) -> torch.Tensor:
    """``hc_norm``: the grouped Gemma RMSNorm of the multi-stream residual (``mix``)."""
    return grouped_gemma_rmsnorm_torch(x, norm_weight, eps, hc_count)


def hc_silu_reference(down: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``hc_silu``: ``hc_silu_torch`` of the whole down projection (final mixer, ``use_combine=False``)."""
    return hc_silu_torch(down, hc_count)


def hc_silu_split_reference(down_and_injection: torch.Tensor, hc_count: int,
                            lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_silu_split``: the ``split`` of the merged down+inject GEMM output and ``hc_silu_torch`` of its lora part.

    Returns ``(silu(lora / hc_count), injection)``; ``injection`` is the exact copy of columns
    ``[lora_rank, lora_rank + hc_count)``.  The padding columns after them are never read.
    """
    lora = down_and_injection[..., :lora_rank]
    injection = down_and_injection[..., lora_rank:lora_rank + hc_count]
    return hc_silu_torch(lora, hc_count), injection.clone()


def hc_gate_mix_reference(xn: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``hc_gate_mix``: sigmoid gate and gated mean over the streams, the block input."""
    return hc_gate_mix_torch(xn, gate, hc_count)


def hc_combine_reference(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
                         hc_count: int) -> torch.Tensor:
    """``hc_combine``: the standalone combine (PLE layers, deepstack, pipeline-parallel hand-off)."""
    return hc_combine_torch(residual, block_output, injection, hc_count)


def hc_combine_norm_reference(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
                              norm_weight: torch.Tensor, eps: float,
                              hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_combine_norm``: combine, rounded to the input dtype, then the grouped norm of the rounded result."""
    return hc_combine_norm_torch(residual, block_output, injection, norm_weight, eps, hc_count)


def hc_combine_norm_unrounded(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
                              norm_weight: torch.Tensor, eps: float, hc_count: int) -> torch.Tensor:
    """The WRONG norm for ``hc_combine_norm``: normalizes the fp32 combine without rounding it first.

    Only the rounding-point test uses it, to find the elements on which a kernel that skips the
    oracle's bf16 rounding between combine and norm would differ from the oracle.
    """
    dim = residual.shape[-1]
    hc_dim = dim // hc_count
    w = (2.0 * torch.sigmoid(injection.float() / hc_count)).unsqueeze(-1)
    res = residual.float().unflatten(-1, (hc_count, hc_dim))
    block = block_output.float().unsqueeze(-2)
    combined = (res + block * w).flatten(-2)  # fp32, not rounded
    return grouped_gemma_rmsnorm_torch(combined, norm_weight.float(), eps, hc_count).to(residual.dtype)
