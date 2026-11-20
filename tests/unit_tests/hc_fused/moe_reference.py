# SPDX-License-Identifier: Apache-2.0
"""Torch reference for the fused MoE glue kernels (``csrc/hc_fused``, the ``moe_*`` ops).

The four ``*_torch`` functions are the torch code the HPU model runs today for the element-wise
stages of one Flash-Next MoE block, copied line for line (same variable names) from the plugin and
the pinned engine.  They are copied because the TPC simulator container has neither ``vllm`` nor
``vllm_gaudi``; ``test_moe_reference_copy.py`` (CPU, needs vLLM) asserts that every line of a copy
is in the plugin's source and that the copies return the plugin's bits.  Do not edit them here:
change the plugin and copy again.

    moe_route_torch      vllm_gaudi/ops/hpu_fused_moe.py  HPUUnquantizedFusedMoEMethod.apply_monolithic
                         vllm_gaudi/ops/hpu_fp8.py        HPUFp8MoEMethod.apply_monolithic
    moe_row_quant_torch  vllm_gaudi/extension/ops.py      dynamic_quant (the default branch)
    moe_silu_mul_torch   vllm_gaudi/ops/hpu_silu_and_mul.py  HPUSiluAndMul.forward_oot (the default branch)
    moe_combine_torch    vllm/model_executor/models/qwen2_moe.py  Qwen2MoeMLP.forward (the gate line) and
                         vllm_gaudi/ops/hpu_fused_moe.py  patched_fused_moe_forward (the add)

The ``*_reference`` functions below them define what each fused op returns, in terms of those
copies.  Contract: ``docs/design/flash-next/hc-fused.md``, "MoE glue kernels".
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GAUDI2_FP8_MAX = 240.0  # torch.finfo(torch.float8_e4m3fnuz).max: FP8_MAX of vllm_gaudi/extension/ops.py on Gaudi2


# ---------------------------------------------------------------------------
# Copies of the torch path
# ---------------------------------------------------------------------------
def moe_route_torch(x: torch.Tensor, router_logits: torch.Tensor, layer) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing of the default (softmax, renormalized) branch: ``layer.top_k`` experts per token."""
    topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int64)
    topk_weights = topk_weights.to(x.dtype)
    return topk_ids, topk_weights


def moe_row_quant_torch(data: torch.Tensor, FP8_MAX: float, cast_to_fp8_v2) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic FP8 quantization of the routed experts' input: ``(data_fp8, scale)``."""
    scale = ((torch.abs(data)).max(dim=-1).values + 1e-8) / FP8_MAX
    scale = scale.unsqueeze(-1)
    data_fp8 = cast_to_fp8_v2(data, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    return data_fp8, scale.float()


def moe_silu_mul_torch(x: torch.Tensor) -> torch.Tensor:
    """SiLU of the first half of the columns times the second half (the shared expert's activation)."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def moe_combine_torch(out: torch.Tensor, expert_gate: torch.Tensor, fused_output: torch.Tensor) -> torch.Tensor:
    """The gated shared-expert output plus the routed experts' output (``expert_gate`` is the gate logit)."""
    out = F.sigmoid(expert_gate) * out
    shared_output = out
    combined = (shared_output + fused_output) if shared_output is not None else fused_output
    return combined


COPIED_FUNCTIONS = ("moe_route_torch", "moe_row_quant_torch", "moe_silu_mul_torch", "moe_combine_torch")


# ---------------------------------------------------------------------------
# The vendor cast the row quantization ends with
# ---------------------------------------------------------------------------
def cast_to_fp8_reference(data: torch.Tensor, scale_inv: torch.Tensor, fp8_max: float = GAUDI2_FP8_MAX):
    """What ``torch.ops.hpu.cast_to_fp8_v2(data, scale_inv, False, False, float8_e4m3fn)`` is taken to do.

    A multiplication in the input dtype (the vendor op is a ``mult_fwd`` node followed by a cast,
    ``docs/features/supported_features.md``), then round-to-nearest-even to the 1-4-3 format with
    the result clipped to the largest value of the device's format: 240 on Gaudi2, whose format
    keeps the top exponent for infinity.  Up to 240 the Gaudi2 format and ``torch.float8_e4m3fn``
    have the same bit patterns.  Returns a 1-tuple, like the op.
    """
    product = data * scale_inv
    return (product.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn), )


# ---------------------------------------------------------------------------
# What a Gaudi2 card computes differently from CPU torch
# ---------------------------------------------------------------------------
def hpu_div_by_scalar(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """``x / divisor`` for a bf16 tensor and a Python number, as a Gaudi2 card computes it.

    The card multiplies by the reciprocal rounded to the tensor's dtype: ``bf16(x * bf16(1 / divisor))``.
    CPU torch divides, ``bf16(x / divisor)``.  For 240 the constant is ``0x1.12p-8``, 0.34 % above 1/240,
    and the card's result is CPU torch's or one bf16 step above it.  Found on the first card run of the
    MoE glue kernels (2026-11-20); the evidence and the scope are in ``hpu_arithmetic.py``.
    """
    return x * torch.tensor(1.0 / divisor, dtype=x.dtype)


# ---------------------------------------------------------------------------
# The fused ops
# ---------------------------------------------------------------------------
class _TopK:

    def __init__(self, top_k: int):
        self.top_k = top_k


def moe_route_reference(gates: torch.Tensor, experts: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``moe_route``: ``(ids int32 [T, K], weights [T, K] in the dtype of gates)``.

    ``gates`` is the merged gate GEMM output ``[T, C]`` with the router logits in columns
    ``[0, experts)``; later columns (the shared-gate logit) are not read.  The ids are the torch
    path's, as int32: the caller widens them for the vendor op.  Which expert wins among equal
    probabilities at the cut is whatever ``torch.topk`` does here; ``moe_case.compare`` accepts
    every valid choice and pins the kernel's own rule (the lowest expert id).
    """
    ids, weights = moe_route_torch(gates, gates[:, :experts], _TopK(top_k))
    return ids.to(torch.int32), weights


def moe_row_quant_reference(x: torch.Tensor, fp8_max: float = GAUDI2_FP8_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    """``moe_row_quant``: ``(x_fp8 float8_e4m3fn [T, H], scale float32 [T, 1])``."""

    def cast(data, scale_inv, *_):
        return cast_to_fp8_reference(data, scale_inv, fp8_max)

    return moe_row_quant_torch(x, fp8_max, cast)


def moe_silu_mul_reference(gate_up: torch.Tensor) -> torch.Tensor:
    """``moe_silu_mul``: ``[T, 2 * D] -> [T, D]``."""
    return moe_silu_mul_torch(gate_up)


def moe_combine_reference(shared: torch.Tensor, gates: torch.Tensor, routed: torch.Tensor,
                          gate_column: int) -> torch.Tensor:
    """``moe_combine``: ``sigmoid(gates[:, gate_column]) * shared + routed``, ``[T, H]``."""
    return moe_combine_torch(shared, gates[:, gate_column:gate_column + 1], routed)
