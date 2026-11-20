# SPDX-License-Identifier: Apache-2.0
"""The fused hyper-connection ops: schema, CPU kernel and fake (Meta) kernel.

Staged here because the track may only add files under ``tests/unit_tests/hc_fused``; at
integration this module becomes ``flashinfer_gaudi/hc_fused.py`` (with ``hc_reference.py`` next to
it) and this file a one-line re-export.  It is the only place that defines the six schemas and the
fake kernels.  The HPU kernels come from the C++ extension ``csrc/hc_fused/pytorch`` (``hc_fused_ops*.so``),
which must not define them.  The CPU kernels are the torch oracle, so a CPU run of the fused path
is the oracle by construction.

Schemas (all tensors bf16 on HPU; the CPU kernels take any float dtype):

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

from tests.unit_tests.hc_fused.hc_reference import (hc_combine_norm_reference, hc_combine_reference,
                                                    hc_gate_mix_reference, hc_norm_reference, hc_silu_reference,
                                                    hc_silu_split_reference)

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

_LIBRARY = torch.library.Library(NAMESPACE, "FRAGMENT")
for _schema in SCHEMAS.values():
    _LIBRARY.define(_schema)

_LIBRARY.impl("hc_norm", hc_norm_reference, "CPU")
_LIBRARY.impl("hc_silu", hc_silu_reference, "CPU")
_LIBRARY.impl("hc_silu_split", hc_silu_split_reference, "CPU")
_LIBRARY.impl("hc_gate_mix", hc_gate_mix_reference, "CPU")
_LIBRARY.impl("hc_combine", hc_combine_reference, "CPU")
_LIBRARY.impl("hc_combine_norm", hc_combine_norm_reference, "CPU")


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


def hc_norm(x: torch.Tensor, norm_weight: torch.Tensor, eps: float, hc_count: int) -> torch.Tensor:
    """Grouped Gemma RMSNorm of the multi-stream residual ``[T, hc_count * H]``."""
    return torch.ops.flashinfer_gaudi.hc_norm(x, norm_weight, float(eps), int(hc_count))


def hc_silu(down: torch.Tensor, hc_count: int) -> torch.Tensor:
    """``silu(down / hc_count)`` of the down GEMM output ``[T, R]`` (final mixer)."""
    return torch.ops.flashinfer_gaudi.hc_silu(down, int(hc_count))


def hc_silu_split(down: torch.Tensor, hc_count: int, lora_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(silu(down[:, :R] / hc_count), down[:, R:R + hc_count])`` of the merged down+inject GEMM output."""
    return torch.ops.flashinfer_gaudi.hc_silu_split(down, int(hc_count), int(lora_rank))


def hc_gate_mix(xn: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    """Sigmoid gate and gated mean over the streams: the block input ``[T, H]``."""
    return torch.ops.flashinfer_gaudi.hc_gate_mix(xn, gate, int(hc_count))


def hc_combine(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
               hc_count: int) -> torch.Tensor:
    """``residual + block_output * 2 * sigmoid(injection / hc_count)`` per stream."""
    return torch.ops.flashinfer_gaudi.hc_combine(residual, block_output, injection, int(hc_count))


def hc_combine_norm(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
                    norm_weight: torch.Tensor, eps: float, hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The combine, rounded to the input dtype, and the grouped Gemma RMSNorm of the rounded result."""
    return torch.ops.flashinfer_gaudi.hc_combine_norm(residual, block_output, injection, norm_weight, float(eps),
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
