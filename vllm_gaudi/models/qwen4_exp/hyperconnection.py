# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — HPU pure-torch variant.

Drop-in replacement for upstream
``vllm/models/qwen4_exp/nvidia/hyperconnection.py::GatedResidual``: same
constructor, weight names and ``mix`` / ``combine_and_mix`` / ``combine``
API, but the five HC glue kernels of ``nvidia/ops/hc.py`` are replaced by
plain PyTorch implementations (no Triton, no custom-op registration, no
CUDA checks). The math mirrors the Triton kernel bodies: fp32 inside,
cast back to the input dtype at the end.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout). Both 2-D
``[tokens, HC*HS]`` and 3-D ``[batch, seq, HC*HS]`` inputs are supported.
"""

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.qwen4_exp.common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)


# ---------------------------------------------------------------------------
# Pure-torch replacements for the five glue kernels in ``nvidia/ops/hc.py``
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


# ---------------------------------------------------------------------------
# Gated-residual variant (pure torch)
# ---------------------------------------------------------------------------
class HpuGatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the input RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch applies through the standard quant_method mechanism. The
    five glue ops are pure PyTorch (see module docstring).
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size)
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count] + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm_torch(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu_torch(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [..., D]
        block_input = hc_gate_mix_torch(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        hidden_states, xn = hc_combine_norm_torch(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu_torch(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [..., D]
        block_input = hc_gate_mix_torch(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine_torch(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "HpuGatedResidual",
    "grouped_gemma_rmsnorm_torch",
    "hc_combine_norm_torch",
    "hc_combine_torch",
    "hc_gate_mix_torch",
    "hc_silu_torch",
]
