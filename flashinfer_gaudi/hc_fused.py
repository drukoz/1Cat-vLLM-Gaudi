# SPDX-License-Identifier: Apache-2.0
"""The fused hyper-connection ops of Flash-Next: schemas and fake (Meta) kernels.

Six ops replace the element-wise torch code of one hyper-connection block
(``vllm_gaudi/models/qwen4_exp/hyperconnection.py``); the two GEMMs of a block stay torch linears.
This module is the only place that defines the schemas.  It registers no kernel that computes
anything: the HPU kernels come from the C++ extension ``hc_fused_ops*.so`` (``csrc/hc_fused/pytorch``),
which ``flashinfer_gaudi._native.load_hc_fused_extension`` loads after the kernel database
``libhc_fused_gaudi2.so`` is registered.  A process without that extension has no kernel for any
device, ``hc_fused_missing_ops`` names all six ops, and the layer runs its torch path.

Nothing imports this module unless ``VLLM_HPU_HC_FUSED=1`` or ``VLLM_HPU_MOE_GLUE=1``: importing it defines
the six schemas, and the four of the MoE glue ops (``flashinfer_gaudi/moe_glue.py``), because the one
extension implements the HPU kernels of both families and every schema must exist before it is loaded.

Schemas (all tensors bf16, ``[tokens, columns]``):

    hc_norm(Tensor x, Tensor norm_weight, float eps, int hc_count) -> Tensor
    hc_silu(Tensor down, int hc_count) -> Tensor
    hc_silu_split(Tensor down, int hc_count, int lora_rank) -> (Tensor, Tensor)
    hc_gate_mix(Tensor xn, Tensor gate, int hc_count) -> Tensor
    hc_combine(Tensor residual, Tensor block_output, Tensor injection, int hc_count) -> Tensor
    hc_combine_norm(Tensor residual, Tensor block_output, Tensor injection, Tensor norm_weight,
                    float eps, int hc_count) -> (Tensor, Tensor)

Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import torch

NAMESPACE = "flashinfer_gaudi"
OP_NAMES = ("hc_norm", "hc_silu", "hc_silu_split", "hc_gate_mix", "hc_combine", "hc_combine_norm")
SCHEMAS = {
    "hc_norm":
    "hc_norm(Tensor x, Tensor norm_weight, float eps, int hc_count) -> Tensor",
    "hc_silu":
    "hc_silu(Tensor down, int hc_count) -> Tensor",
    "hc_silu_split":
    "hc_silu_split(Tensor down, int hc_count, int lora_rank) -> (Tensor, Tensor)",
    "hc_gate_mix":
    "hc_gate_mix(Tensor xn, Tensor gate, int hc_count) -> Tensor",
    "hc_combine":
    "hc_combine(Tensor residual, Tensor block_output, Tensor injection, int hc_count) -> Tensor",
    "hc_combine_norm": ("hc_combine_norm(Tensor residual, Tensor block_output, Tensor injection, Tensor norm_weight, "
                        "float eps, int hc_count) -> (Tensor, Tensor)"),
}

# What the HPU kernels accept, as the op's validation callback (csrc/hc_fused/pytorch/hpu_hc_fused.cpp) and the
# kernel glue (csrc/hc_fused/host/hc_fused_gaudi2.cpp) check it.  The layer checks the same before it calls an
# op, because a rejected input raises inside the op.
VECTOR_LANES = 128  # bf16 lanes of one TPC vector: hidden_size must be a multiple of it
MAX_STREAMS = 64  # 1 <= hc_count <= MAX_STREAMS
FP32_MIN_NORMAL = 1.1754943508222875e-38  # eps must be a positive normal fp32 value
FP32_MAX = 3.4028234663852886e+38

_LIBRARY = torch.library.Library(NAMESPACE, "FRAGMENT")
for _schema in SCHEMAS.values():
    _LIBRARY.define(_schema)


# Fake kernels: allocation only, no data access.
@torch.library.register_fake(f"{NAMESPACE}::hc_norm", lib=_LIBRARY)
def _hc_norm_fake(x, norm_weight, eps, hc_count):
    return x.new_empty(x.shape)


@torch.library.register_fake(f"{NAMESPACE}::hc_silu", lib=_LIBRARY)
def _hc_silu_fake(down, hc_count):
    return down.new_empty(down.shape)


@torch.library.register_fake(f"{NAMESPACE}::hc_silu_split", lib=_LIBRARY)
def _hc_silu_split_fake(down, hc_count, lora_rank):
    lead = tuple(down.shape[:-1])
    return down.new_empty(lead + (lora_rank, )), down.new_empty(lead + (hc_count, ))


@torch.library.register_fake(f"{NAMESPACE}::hc_gate_mix", lib=_LIBRARY)
def _hc_gate_mix_fake(xn, gate, hc_count):
    return xn.new_empty(tuple(xn.shape[:-1]) + (xn.shape[-1] // hc_count, ))


@torch.library.register_fake(f"{NAMESPACE}::hc_combine", lib=_LIBRARY)
def _hc_combine_fake(residual, block_output, injection, hc_count):
    return residual.new_empty(residual.shape)


@torch.library.register_fake(f"{NAMESPACE}::hc_combine_norm", lib=_LIBRARY)
def _hc_combine_norm_fake(residual, block_output, injection, norm_weight, eps, hc_count):
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


def _owned(x: torch.Tensor) -> torch.Tensor:
    """``x`` as a contiguous tensor that is not a view.

    The Habana lazy bridge cannot hand a view to a custom op: the first servers that called custom ops
    on views died with "ValidateSyncInputTensors tensor_data is empty ... Strided Params Has Value: 1"
    (rental 2026-11-20). The activations a layer passes are often views (a reshape of the block output,
    the model's first hidden state), so each one is copied; the weights are parameters and are not.
    """
    return x.clone(memory_format=torch.contiguous_format)


def hc_norm(x: torch.Tensor, norm_weight: torch.Tensor, eps: float, hc_count: int) -> torch.Tensor:
    """Grouped Gemma RMSNorm of the multi-stream residual ``[T, hc_count * H]``."""
    return torch.ops.flashinfer_gaudi.hc_norm(_owned(x), norm_weight, float(eps), int(hc_count))


def hc_silu(down: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``silu(down / hc_count)`` of the down GEMM output ``[T, R]`` (final mixer)."""
    return torch.ops.flashinfer_gaudi.hc_silu(_owned(down), int(hc_count))


def hc_silu_split(down: torch.Tensor, hc_count: int, lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(silu(down[:, :R] / hc_count), down[:, R:R + hc_count])`` of the merged down+inject GEMM output."""
    return torch.ops.flashinfer_gaudi.hc_silu_split(_owned(down), int(hc_count), int(lora_rank))


def hc_gate_mix(xn: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Sigmoid gate and gated mean over the streams: the block input ``[T, H]``."""
    return torch.ops.flashinfer_gaudi.hc_gate_mix(_owned(xn), _owned(gate), int(hc_count))


def hc_combine(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
               hc_count: int) -> torch.Tensor:
    """``residual + block_output * 2 * sigmoid(injection / hc_count)`` per stream."""
    return torch.ops.flashinfer_gaudi.hc_combine(_owned(residual), _owned(block_output), _owned(injection), int(hc_count))


def hc_combine_norm(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
                    norm_weight: torch.Tensor, eps: float, hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The combine, rounded to the input dtype, and the grouped Gemma RMSNorm of the rounded result."""
    return torch.ops.flashinfer_gaudi.hc_combine_norm(_owned(residual), _owned(block_output), _owned(injection),
                                                      norm_weight, float(eps),
                                                      int(hc_count))


_DISPATCH_KEYS_BY_DEVICE_TYPE = {"cpu": "CPU", "hpu": "HPU"}


def hc_fused_missing_ops(device: torch.device | str) -> tuple[str, ...]:
    """The ops of ``OP_NAMES`` that have no kernel for the device's type (all of them for an unknown type)."""
    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    dispatch_key = _DISPATCH_KEYS_BY_DEVICE_TYPE.get(device_type)
    if dispatch_key is None:
        return OP_NAMES
    return tuple(name for name in OP_NAMES
                 if not torch._C._dispatch_has_kernel_for_dispatch_key(f"{NAMESPACE}::{name}", dispatch_key))


def hc_fused_available(device: torch.device | str) -> bool:
    """True when every fused hyper-connection op has a kernel for the device's type."""
    return not hc_fused_missing_ops(device)


# hc_fused_ops*.so implements the HPU kernels of the MoE glue ops too: their schemas are defined with these.
from flashinfer_gaudi import moe_glue as _moe_glue  # noqa: E402,F401
