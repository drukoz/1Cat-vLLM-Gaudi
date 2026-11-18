# SPDX-License-Identifier: Apache-2.0
"""Sizing helpers for the Flash-Next sparse QSA side caches (M2).

Every QSA layer owns two plugin-side tensors in sparse mode: a compressed
key cache of shape ``[key_cache_rows // compress_ratio, index_head_dim]``
sized from the layer's attention KV cache (the flat K cache has one row per
slot, the compressed cache one row per ``compress_ratio`` slots), and a
raw-key ring of shape ``[gdn_state_rows, R, index_head_dim]`` sized from the
first GDN group's state tensors, with ``R = ring_rows(compress_ratio,
num_speculative_tokens)``.  Pure functions only (``math`` and ``torch``);
the model runner allocates, this module only does the arithmetic.

The ring's LAST row (index ``gdn_state_rows - 1``) is the scratch row: it is
never a request's slot, because the runner's state tensors reserve their last
row as padding (slot 0 is vLLM's null block, the last row the padding row), so
this holds for the compact, pool and baseline GDN layouts alike.  The sparse
decode branch of ``HpuQsaIndexer.select`` parks the padding rows of a ragged
multi-token decode step on it (both as load and as store slot), which keeps
those rows out of every request's ring window.
"""

from __future__ import annotations

import math

import torch


def ring_rows(compress_ratio: int, num_speculative_tokens: int) -> int:
    """Ring rows per GDN state slot: one full block per in-flight token.

    Same formula as ``qsa_sparse_ops.ring_rows`` (duplicated here so this
    module stays import-light; T3 may later import one from the other).
    """
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if num_speculative_tokens < 0:
        raise ValueError(f"num_speculative_tokens must be >= 0, got {num_speculative_tokens}")
    return compress_ratio * math.ceil((compress_ratio + num_speculative_tokens) / compress_ratio)


def compressed_cache_shape(key_cache_rows: int, compress_ratio: int, index_head_dim: int) -> tuple[int, int]:
    """[key_cache_rows // compress_ratio, index_head_dim]; one row per block."""
    if key_cache_rows % compress_ratio:
        raise ValueError(
            f"key_cache_rows={key_cache_rows} is not divisible by compress_ratio={compress_ratio}")
    return (key_cache_rows // compress_ratio, index_head_dim)


def ring_shape(gdn_state_rows: int, rows: int, index_head_dim: int) -> tuple[int, int, int]:
    """[gdn_state_rows, rows, index_head_dim]; rows come from ring_rows()."""
    return (gdn_state_rows, rows, index_head_dim)


def attention_budget_fraction(
    num_kv_heads: int,
    head_size: int,
    kv_dtype_bytes: int,
    compress_ratio: int,
    index_head_dim: int,
    side_dtype_bytes: int = 2,
) -> float:
    """Share of the KV budget left for attention blocks once the compressed
    key cache is funded alongside (per token: kv bytes vs kv + comp bytes)."""
    kv = 2 * num_kv_heads * head_size * kv_dtype_bytes
    comp = index_head_dim * side_dtype_bytes / compress_ratio
    return kv / (kv + comp)


def ring_bytes(
    num_qsa_layers: int,
    gdn_state_rows: int,
    rows: int,
    index_head_dim: int,
    side_dtype_bytes: int = 2,
) -> int:
    """Total bytes of every QSA layer's raw-key ring."""
    return num_qsa_layers * gdn_state_rows * rows * index_head_dim * side_dtype_bytes


def allocate_side_caches(
    key_cache_rows: int,
    gdn_state_rows: int,
    compress_ratio: int,
    index_head_dim: int,
    num_speculative_tokens: int,
    device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-init (compressed_key_cache, key_ring) with the shapes above."""
    compressed = torch.zeros(
        compressed_cache_shape(key_cache_rows, compress_ratio, index_head_dim),
        dtype=dtype,
        device=device,
    )
    ring = torch.zeros(
        ring_shape(gdn_state_rows, ring_rows(compress_ratio, num_speculative_tokens), index_head_dim),
        dtype=dtype,
        device=device,
    )
    return compressed, ring
