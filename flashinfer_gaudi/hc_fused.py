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

    The bridge hands a view to a custom op as a graph input without data: the first servers that called
    custom ops on views died with "ValidateSyncInputTensors tensor_data is empty ... Strided Params Has
    Value: 1" (rental 2026-11-20), and with the accumulation thread off seven of seven view cases (row and
    column slices, reshapes, of a tensor and of a pending op) raised "Empty tensor optional ... Strided Params
    Has Value: 1" on every call (card job 00043, tools/flash_next/kernel_probes/probe_view_inputs.py), while
    a fresh tensor and a copy of a slice passed. The activations a layer is handed are often views (a reshape
    of the block output, the model's first hidden state), so they are copied; the weights are parameters and
    are not. Which activations: ``COPY_MODES`` below.
    """
    return x.clone(memory_format=torch.contiguous_format)


# Which activations are copied before an op (the ``copy_mode`` of the wrappers below; the layer passes its
# VLLM_HPU_HC_FUSED_COPY_MODE). A view must never reach a custom op (card job 00043, 2026-11-20: 7 of 7 view
# cases raised with the accumulation thread off), but only a tensor that is handed to the layer can be one:
#   COPY_ALL           every activation (the default; what ran on the cards on 2026-11-20)
#   COPY_HANDED_IN     not the tensors the hyper-connection itself makes inside the call: the norm output and
#                      the outputs of its two GEMMs. The residual, the block output and the injection are copied.
#   COPY_BLOCK_OUTPUT  only the block output (an attention or MoE block may return a view of its result) and
#                      hc_norm's input. The residual and the injection must then be op outputs, as the target
#                      model's layer chain makes them (test_fewer_owned_copies_change_no_bit_and_hand_no_op_a_view).
#                      There is no fallback for a caller that hands in a view: in lazy mode the op is only
#                      recorded when it is called, and the refusal comes when the graph is launched, outside the
#                      layer's try (the first servers that did this died).
COPY_ALL, COPY_HANDED_IN, COPY_BLOCK_OUTPUT = 0, 1, 2
COPY_MODES = (COPY_ALL, COPY_HANDED_IN, COPY_BLOCK_OUTPUT)


def _made_here(x: torch.Tensor, copy_mode: int) -> torch.Tensor:
    """An output of the caller's own op of this call (its norm, one of its GEMMs)."""
    return _owned(x) if copy_mode == COPY_ALL else x


def _chained(x: torch.Tensor, copy_mode: int) -> torch.Tensor:
    """The residual stream or the injection: a fused op's output wherever the model's layers chain them."""
    return x if copy_mode == COPY_BLOCK_OUTPUT else _owned(x)


def hc_norm(x: torch.Tensor, norm_weight: torch.Tensor, eps: float, hc_count: int, *,
            copy_mode: int = COPY_ALL) -> torch.Tensor:
    """Grouped Gemma RMSNorm of the multi-stream residual ``[T, hc_count * H]``."""
    return torch.ops.flashinfer_gaudi.hc_norm(_owned(x), norm_weight, float(eps), int(hc_count))


def hc_silu(down: torch.Tensor, hc_count: int, *, copy_mode: int = COPY_ALL) -> torch.Tensor:
    """``silu(down / hc_count)`` of the down GEMM output ``[T, R]`` (final mixer)."""
    return torch.ops.flashinfer_gaudi.hc_silu(_made_here(down, copy_mode), int(hc_count))


def hc_silu_split(down: torch.Tensor, hc_count: int, lora_rank: int, *,
                  copy_mode: int = COPY_ALL) -> tuple[torch.Tensor, torch.Tensor]:
    """``(silu(down[:, :R] / hc_count), down[:, R:R + hc_count])`` of the merged down+inject GEMM output."""
    return torch.ops.flashinfer_gaudi.hc_silu_split(_made_here(down, copy_mode), int(hc_count), int(lora_rank))


def hc_gate_mix(xn: torch.Tensor, gate: torch.Tensor, hc_count: int, *, copy_mode: int = COPY_ALL) -> torch.Tensor:
    """Sigmoid gate and gated mean over the streams: the block input ``[T, H]``."""
    return torch.ops.flashinfer_gaudi.hc_gate_mix(_made_here(xn, copy_mode), _made_here(gate, copy_mode),
                                                  int(hc_count))


def hc_combine(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor, hc_count: int, *,
               copy_mode: int = COPY_ALL) -> torch.Tensor:
    """``residual + block_output * 2 * sigmoid(injection / hc_count)`` per stream."""
    return torch.ops.flashinfer_gaudi.hc_combine(_chained(residual, copy_mode), _owned(block_output),
                                                 _chained(injection, copy_mode), int(hc_count))


def hc_combine_norm(residual: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor,
                    norm_weight: torch.Tensor, eps: float, hc_count: int, *,
                    copy_mode: int = COPY_ALL) -> tuple[torch.Tensor, torch.Tensor]:
    """The combine, rounded to the input dtype, and the grouped Gemma RMSNorm of the rounded result."""
    return torch.ops.flashinfer_gaudi.hc_combine_norm(_chained(residual, copy_mode), _owned(block_output),
                                                      _chained(injection, copy_mode), norm_weight, float(eps),
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
