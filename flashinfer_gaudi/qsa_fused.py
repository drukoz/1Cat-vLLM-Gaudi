# SPDX-License-Identifier: Apache-2.0
"""The fused QSA decode ops of Flash-Next: schemas, fake (Meta) kernels and wrappers.

Two ops replace the per-row glue of a sparse-QSA decode step (``vllm_gaudi/models/qwen4_exp/qsa.py``,
``qsa_hpu_attention.py``); the GEMMs, the request-major key scores (``qsa_fused_path``), the top-k and the
attention stay torch. This module is the only place that defines the schemas. It registers no kernel that
computes anything: the HPU kernels come from the C++ extension ``qsa_fused_ops*.so``
(``csrc/qsa_fused/pytorch``), which ``flashinfer_gaudi._native.load_qsa_fused_extension`` loads after the
kernel database ``libqsa_fused_gaudi2.so`` is registered. Nothing imports this module unless
``VLLM_HPU_QSA_FUSED=2``.

Schemas (torch shapes; every integer tensor int32):

    qsa_index_prepare(Tensor(a!) key_ring, Tensor(b!) key_cache, Tensor qk, Tensor rows, Tensor q_norm_weight,
                      Tensor k_norm_weight, Tensor cos_sin, float eps, int heads, int ring_rows, int ratio) -> Tensor
        key_ring [S * R, D] bf16 and key_cache [C, D] bf16 are written in place; qk [T, (H + 1) * D] bf16;
        rows [T, 4] int32 (position, attention slot, ring slot read, ring slot written; -1 slots on padding);
        weights [D] bf16; cos_sin [P, 2 * R2] bf16. Returns q_idx [T, H * D] bf16.
    qsa_index_emit(Tensor blocks, Tensor positions, Tensor block_table, int ratio, int block_size)
        -> (Tensor, Tensor)
        blocks [T, topk] int32 (select_blocks), positions [T] int32, block_table [T, MB] int32. Returns
        slots and valid, int32 [T, topk * ratio + ratio - 1].

Contract and rounding points: ``docs/design/flash-next/qsa-fused.md``.
"""

from __future__ import annotations

import torch

NAMESPACE = "flashinfer_gaudi"
OP_NAMES = ("qsa_index_prepare", "qsa_index_emit")
GUIDS = {"qsa_index_prepare": "qsa_fused_prepare_bf16_gaudi2", "qsa_index_emit": "qsa_fused_emit_i32_gaudi2"}
SCHEMAS = {
    "qsa_index_prepare": ("qsa_index_prepare(Tensor(a!) key_ring, Tensor(b!) key_cache, Tensor qk, Tensor rows, "
                          "Tensor q_norm_weight, Tensor k_norm_weight, Tensor cos_sin, float eps, int heads, "
                          "int ring_rows, int ratio) -> Tensor"),
    "qsa_index_emit": ("qsa_index_emit(Tensor blocks, Tensor positions, Tensor block_table, int ratio, "
                       "int block_size) -> (Tensor, Tensor)"),
}

# What the HPU kernels accept (csrc/qsa_fused/qsa_fused_params.h, checked by the op's validation callback and the
# kernel glue). The layer checks the same before it calls an op (qsa_fused_path.kernel_shape_reason).
MAX_DIM = 128
MAX_HEADS = 8
MAX_RATIO = 8
MAX_RING_ROWS = 64

_LIBRARY = torch.library.Library(NAMESPACE, "FRAGMENT")
for _schema in SCHEMAS.values():
    _LIBRARY.define(_schema)


# Fake kernels: allocation only, no data access and no mutation.
@torch.library.register_fake(f"{NAMESPACE}::qsa_index_prepare", lib=_LIBRARY)
def _qsa_index_prepare_fake(key_ring, key_cache, qk, rows, q_norm_weight, k_norm_weight, cos_sin, eps, heads,
                            ring_rows, ratio):
    return qk.new_empty((qk.shape[0], heads * key_cache.shape[1]))


@torch.library.register_fake(f"{NAMESPACE}::qsa_index_emit", lib=_LIBRARY)
def _qsa_index_emit_fake(blocks, positions, block_table, ratio, block_size):
    width = blocks.shape[1] * ratio + ratio - 1
    return (blocks.new_empty((blocks.shape[0], width), dtype=torch.int32),
            blocks.new_empty((blocks.shape[0], width), dtype=torch.int32))


def qsa_index_prepare(key_ring: torch.Tensor, key_cache: torch.Tensor, qk: torch.Tensor, rows: torch.Tensor,
                      q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, cos_sin: torch.Tensor, eps: float,
                      heads: int, ring_rows: int, ratio: int) -> torch.Tensor:
    """K1 as ``torch.ops.flashinfer_gaudi.qsa_index_prepare`` (mutates ``key_ring`` and ``key_cache``)."""
    return torch.ops.flashinfer_gaudi.qsa_index_prepare(key_ring, key_cache, qk, rows, q_norm_weight, k_norm_weight,
                                                        cos_sin, float(eps), int(heads), int(ring_rows), int(ratio))


def qsa_index_emit(blocks: torch.Tensor, positions: torch.Tensor, block_table: torch.Tensor, ratio: int,
                   block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K4 as ``torch.ops.flashinfer_gaudi.qsa_index_emit``: int32 ``(slots, valid)``."""
    return torch.ops.flashinfer_gaudi.qsa_index_emit(blocks, positions, block_table, int(ratio), int(block_size))


def qsa_fused_missing_ops(device_type: str = "hpu") -> tuple[str, ...]:
    """The ops without a kernel for ``device_type`` (``"hpu"`` or ``"cpu"``) in this process."""
    key = {"hpu": "HPU", "cpu": "CPU"}[device_type]
    missing = []
    for name in OP_NAMES:
        try:
            has = torch._C._dispatch_has_kernel_for_dispatch_key(f"{NAMESPACE}::{name}", key)
        except RuntimeError:
            has = False
        if not has:
            missing.append(name)
    return tuple(missing)


__all__ = ["GUIDS", "OP_NAMES", "SCHEMAS", "qsa_fused_missing_ops", "qsa_index_emit", "qsa_index_prepare"]
