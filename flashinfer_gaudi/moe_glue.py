# SPDX-License-Identifier: Apache-2.0
"""The fused MoE glue ops of Flash-Next: schemas and fake (Meta) kernels.

Four ops replace the element-wise torch code around the GEMMs and the vendor expert op of one MoE
block (``vllm_gaudi/ops/hpu_fused_moe.py``, ``_moe_glue_forward``).  This module is the only place
that defines the schemas.  It registers no kernel that computes anything: the HPU kernels come from
the C++ extension of the fused hyper-connection ops, ``hc_fused_ops*.so`` (``csrc/hc_fused/pytorch``),
whose kernel database ``libhc_fused_gaudi2.so`` carries the four MoE kernels next to the six
hyper-connection ones.  A process without that extension has no kernel for any device,
``moe_glue_missing_ops`` names all four ops, and the layer runs its torch path.

Nothing imports this module unless ``VLLM_HPU_MOE_GLUE=1`` or ``VLLM_HPU_HC_FUSED=1`` (the shared
extension implements both families, so its loader defines both sets of schemas first).

Schemas (activations are bf16 ``[tokens, columns]``):

    moe_route(Tensor gates, int experts, int top_k) -> (Tensor, Tensor)
    moe_silu_mul(Tensor gate_up) -> Tensor
    moe_combine(Tensor shared, Tensor gates, Tensor routed, int gate_column) -> Tensor
    moe_row_quant(Tensor x, float fp8_max) -> (Tensor, Tensor)

Contract: ``docs/design/flash-next/hc-fused.md``, "MoE glue kernels".
"""

from __future__ import annotations

import torch

NAMESPACE = "flashinfer_gaudi"
OP_NAMES = ("moe_route", "moe_silu_mul", "moe_combine", "moe_row_quant")
SCHEMAS = {
    "moe_route": "moe_route(Tensor gates, int experts, int top_k) -> (Tensor, Tensor)",
    "moe_silu_mul": "moe_silu_mul(Tensor gate_up) -> Tensor",
    "moe_combine": "moe_combine(Tensor shared, Tensor gates, Tensor routed, int gate_column) -> Tensor",
    "moe_row_quant": "moe_row_quant(Tensor x, float fp8_max) -> (Tensor, Tensor)",
}
# The kernel of each op in libhc_fused_gaudi2.so (csrc/hc_fused/hc_fused_params.h).  Each kernel is optional
# at build time, so the loader asks the library which of them it serves.
GUIDS = {
    "moe_route": "moe_glue_route_bf16_gaudi2",
    "moe_silu_mul": "moe_glue_silu_mul_bf16_gaudi2",
    "moe_combine": "moe_glue_combine_bf16_gaudi2",
    "moe_row_quant": "moe_glue_row_quant_bf16_gaudi2",
}

# What the HPU kernels accept, as the op's validation callback (csrc/hc_fused/pytorch/hpu_hc_fused.cpp) and the
# kernel glue (csrc/hc_fused/host/moe_glue_gaudi2.cpp) check it.  The layer checks the same before it calls an
# op, because a rejected input raises inside the op.
VECTOR_LANES = 128  # bf16 lanes of one TPC vector: the hidden size must be a multiple of it
MAX_TOP_K = 64  # 1 <= top_k <= MAX_TOP_K: one int32 vector holds the ids of a row
MAX_EXPERTS = 32767  # top_k <= experts <= MAX_EXPERTS: int16 lane ids
GAUDI2_FP8_MAX = 240.0  # the only fp8_max moe_row_quant takes: the largest value of Gaudi2's 1-4-3 format

_LIBRARY = torch.library.Library(NAMESPACE, "FRAGMENT")
for _schema in SCHEMAS.values():
    _LIBRARY.define(_schema)


# Fake kernels: allocation only, no data access.
@torch.library.register_fake(f"{NAMESPACE}::moe_route", lib=_LIBRARY)
def _moe_route_fake(gates, experts, top_k):
    shape = tuple(gates.shape[:-1]) + (top_k, )
    return gates.new_empty(shape, dtype=torch.int32), gates.new_empty(shape)


@torch.library.register_fake(f"{NAMESPACE}::moe_silu_mul", lib=_LIBRARY)
def _moe_silu_mul_fake(gate_up):
    return gate_up.new_empty(tuple(gate_up.shape[:-1]) + (gate_up.shape[-1] // 2, ))


@torch.library.register_fake(f"{NAMESPACE}::moe_combine", lib=_LIBRARY)
def _moe_combine_fake(shared, gates, routed, gate_column):
    return shared.new_empty(shared.shape)


@torch.library.register_fake(f"{NAMESPACE}::moe_row_quant", lib=_LIBRARY)
def _moe_row_quant_fake(x, fp8_max):
    return (x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
            x.new_empty(tuple(x.shape[:-1]) + (1, ), dtype=torch.float32))


def moe_route(gates: torch.Tensor, experts: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over ``gates[:, :experts]``, the ``top_k`` experts and their renormalized weights.

    Returns ``(ids int32 [T, top_k], weights [T, top_k])``, the largest logit first; among equal logits
    the lowest expert id wins.
    """
    return torch.ops.flashinfer_gaudi.moe_route(gates, int(experts), int(top_k))


def moe_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(gate_up[:, :D]) * gate_up[:, D:]`` of the shared expert's gate_up GEMM output ``[T, 2 * D]``."""
    return torch.ops.flashinfer_gaudi.moe_silu_mul(gate_up)


def moe_combine(shared: torch.Tensor, gates: torch.Tensor, routed: torch.Tensor, gate_column: int) -> torch.Tensor:
    """``sigmoid(gates[:, gate_column]) * shared + routed``, ``[T, H]``."""
    return torch.ops.flashinfer_gaudi.moe_combine(shared, gates, routed, int(gate_column))


def moe_row_quant(x: torch.Tensor, fp8_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic FP8 quantization: ``(x_fp8 float8_e4m3fn [T, H], scale float32 [T, 1])``."""
    return torch.ops.flashinfer_gaudi.moe_row_quant(x, float(fp8_max))


_DISPATCH_KEYS_BY_DEVICE_TYPE = {"cpu": "CPU", "hpu": "HPU"}


def moe_glue_missing_ops(device: torch.device | str) -> tuple[str, ...]:
    """The ops of ``OP_NAMES`` that have no kernel for the device's type (all of them for an unknown type)."""
    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    dispatch_key = _DISPATCH_KEYS_BY_DEVICE_TYPE.get(device_type)
    if dispatch_key is None:
        return OP_NAMES
    return tuple(name for name in OP_NAMES
                 if not torch._C._dispatch_has_kernel_for_dispatch_key(f"{NAMESPACE}::{name}", dispatch_key))


def moe_glue_available(device: torch.device | str) -> bool:
    """True when every fused MoE glue op has a kernel for the device's type."""
    return not moe_glue_missing_ops(device)
