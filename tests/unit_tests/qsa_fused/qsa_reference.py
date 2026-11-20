# SPDX-License-Identifier: Apache-2.0
"""Torch references of the fused QSA decode ops (the executable definition; docs/design/flash-next/qsa-fused.md).

Each reference computes, with the torch path's own functions, exactly what the torch path of
``HpuQsaIndexer._select_decode`` and ``HpuQsaSparseAttention`` compute for the same inputs, so the
references equal the torch path bit for bit on CPU (``test_qsa_reference.py``) and the TPC kernels are
compared with the references (``tests/standalone/qsa_fused``):

* ``qsa_index_prepare_ref`` (K1): the indexer queries after norm and RoPE, the step's raw keys written
  into the ring, and the compressed key of every block a row completes written into the cache.
  Padding rows (a negative ring slot in ``rows``) write nothing: on the torch path they write the
  ring's scratch slot and the cache's scratch row, whose contents are unspecified on both paths.
* ``qsa_index_emit_ref`` (K4): the K/V cache slot and the 0/1 valid flag of every attended column of
  every row, from the selected blocks (``expand_selection``, ``decode_selected_slots``, the valid
  mask with ``rows_without_selection``).

The helpers below the references are verbatim copies of the plugin's pure-torch QSA functions
(``vllm_gaudi/models/qwen4_exp/qsa_sparse_ops.py`` and ``qsa_hpu_attention.py``): the TPC simulator container
has no ``vllm_gaudi``. ``test_reference_copy.py`` (CPU, needs vLLM) asserts the same source text and the same
bits. Do not edit them here: change the plugin file and copy again.
"""

from __future__ import annotations

import math

import torch


def rows_table(positions: torch.Tensor, slot_mapping: torch.Tensor, load_rows: torch.Tensor,
               store_rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """int32 ``[T, 4]`` row table of a decode step: position, attention slot, ring slot read, ring slot written.

    ``load_rows``/``store_rows`` are the per-row ring slots of ``_select_decode``; a padding row
    (``valid`` False) gets ``-1`` as both ring slots.
    """
    neg = torch.full_like(load_rows, -1)
    return torch.stack((positions.to(torch.int32), slot_mapping.to(torch.int32),
                        torch.where(valid, load_rows, neg).to(torch.int32),
                        torch.where(valid, store_rows, neg).to(torch.int32)), dim=1)


def qsa_index_prepare_ref(key_ring: torch.Tensor, key_cache: torch.Tensor, qk: torch.Tensor, rows: torch.Tensor,
                          q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, cos_sin: torch.Tensor, eps: float,
                          heads: int, ring_rows: int, ratio: int) -> torch.Tensor:
    """K1. ``key_ring`` ``[S * R, D]`` and ``key_cache`` ``[C, D]`` are written in place; returns ``q_idx`` ``[T, H * D]``.

    ``qk`` ``[T, (H + 1) * D]`` (queries, then the raw key), ``rows`` int32 ``[T, 4]`` (``rows_table``),
    ``cos_sin`` ``[P, 2 * R2]`` (cosines, then sines). Write-then-read: every real row's raw key is in the
    ring before any block is compressed, as on the torch path.
    """
    T = int(qk.shape[0])
    dim = int(key_cache.shape[1])
    slots = int(key_ring.shape[0]) // ring_rows
    pos = rows[:, 0].long()
    sm = rows[:, 1].long()
    load = rows[:, 2].long()
    store = rows[:, 3].long()
    cos, sin = cos_sin[pos.clamp_min(0)].chunk(2, dim=-1)
    q = index_queries(qk[:, :heads * dim].reshape(T, heads, dim), q_norm_weight, eps, cos, sin)
    raw_k = qk[:, heads * dim:(heads + 1) * dim]
    real = (store >= 0) & (store < slots)
    if bool(real.any()):
        idx = (store * ring_rows + pos % ring_rows)[real]
        key_ring.index_copy_(0, idx, raw_k[real].to(key_ring.dtype))
    row = torch.div(sm, ratio, rounding_mode="floor")
    complete = real & ((pos % ratio) == ratio - 1) & (row >= 0) & (row < key_cache.shape[0])
    if bool(complete.any()):
        sel = complete.nonzero().reshape(-1)
        p = pos[sel]
        ls = load[sel]
        blk = [key_ring.index_select(0, ls * ring_rows + (p - (ratio - 1) + j) % ring_rows) for j in range(ratio - 1)]
        blk.append(raw_k[sel].to(key_ring.dtype))
        cs, sn = cos_sin[(p - (ratio - 1)).clamp_min(0)].chunk(2, dim=-1)
        keys = compress_block_keys(torch.stack(blk, dim=1), k_norm_weight, eps, cs, sn)
        key_cache.index_copy_(0, row[sel], keys.to(key_cache.dtype))
    return q.reshape(T, heads * dim)


def qsa_index_emit_ref(blocks: torch.Tensor, positions: torch.Tensor, block_table: torch.Tensor, ratio: int,
                       block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K4. int32 ``(slots, valid)`` ``[T, topk * ratio + ratio - 1]`` of the selected blocks ``[T, topk]``."""
    selected = expand_selection(blocks.to(torch.int32), positions, ratio)
    slots = decode_selected_slots(selected, block_table, block_size).to(torch.int32)
    valid = ((selected >= 0) | rows_without_selection(selected)).to(torch.int32)
    return slots, valid


# ---------------------------------------------------------------------------
# Verbatim copies (qsa_sparse_ops.py, qsa_hpu_attention.py)
# ---------------------------------------------------------------------------
def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF ``Qwen4ExpTextRMSNorm`` over the last dim, Gemma-style.

    ``x`` ``[..., D]`` (fp32/bf16/fp16), ``weight`` ``[D]`` (same dtype):
    ``(x.float() * rsqrt(mean(x.float()**2, -1, keepdim=True) + eps) *
    (1 + weight.float())).type_as(x)``, output ``[..., D]`` in ``x``'s dtype.
    """
    x_fp32 = x.float()
    normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).type_as(x)


def rope_neox_partial(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial neox-style RoPE: ``x`` ``[..., D]``, ``cos``/``sin`` ``[..., R2]``
    (``R = 2 * R2 <= D``), broadcastable to ``x[..., :R2]`` (unsqueeze the head
    dim yourself).

    With ``x1, x2 = x[..., :R2], x[..., R2:R]`` returns ``cat((x1 * cos - x2 *
    sin, x2 * cos + x1 * sin, x[..., R:]), -1)``: equals HF
    ``apply_rotary_pos_emb`` with ``cos_full = cat(cos, cos)``, ``sin_full =
    cat(sin, sin)``. Computed in ``x``'s dtype, output ``[..., D]``.
    """
    r2 = cos.shape[-1]
    r = 2 * r2
    x1, x2 = x[..., :r2], x[..., r2:r]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., r:]), dim=-1)


def index_queries(q: torch.Tensor, q_norm_weight: torch.Tensor, eps: float, cos: torch.Tensor,
                  sin: torch.Tensor) -> torch.Tensor:
    """Indexer queries: ``q`` ``[T, H, D]`` -> ``gemma_rms_norm`` over ``D`` ->
    ``rope_neox_partial`` with ``cos``/``sin`` ``[T, R2]`` unsqueezed over ``H``
    -> ``[T, H, D]`` in ``q``'s dtype.
    """
    qn = gemma_rms_norm(q, q_norm_weight, eps)
    return rope_neox_partial(qn, cos[:, None, :], sin[:, None, :])


def compress_block_keys(raw_blocks: torch.Tensor, k_norm_weight: torch.Tensor, eps: float, cos_start: torch.Tensor,
                        sin_start: torch.Tensor) -> torch.Tensor:
    """Compressed block keys: ``raw_blocks`` ``[N, ratio, D]`` -> ``pooled =
    raw_blocks.float().mean(1).to(raw_blocks.dtype)`` -> ``gemma_rms_norm`` ->
    ``rope_neox_partial(pooled, cos_start, sin_start)`` with ``cos``/``sin``
    ``[N, R2]`` at each block's FIRST position -> ``[N, D]`` in ``raw_blocks``'
    dtype.
    """
    pooled = raw_blocks.float().mean(1).to(raw_blocks.dtype)
    pooled = gemma_rms_norm(pooled, k_norm_weight, eps)
    return rope_neox_partial(pooled, cos_start, sin_start)


def visible_blocks(positions: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """``(positions + 1) // compress_ratio`` as int64 ``[T]``: the number of
    complete blocks the query at each position sees.
    """
    return ((positions + 1) // compress_ratio).to(torch.int64)


def select_blocks(scores: torch.Tensor, visible: torch.Tensor, block_topk: int) -> torch.Tensor:
    """Top-k blocks: ``scores`` ``[T, N]`` fp32, ``visible`` ``[T]`` int64 ->
    int32 ``[T, block_topk]`` block indices, -1 padded.

    Columns ``n >= visible[t]`` are masked to -inf, ``k = min(block_topk, N)``
    and ``topk(k, dim=-1)``; entries whose value is -inf become -1, and the
    width is padded to ``block_topk`` with -1. The set per row equals HF's
    ``topk(min(block_topk, nc))``.
    """
    t, n = scores.shape
    k = min(block_topk, n)
    cols = torch.arange(n, device=scores.device)
    masked = scores.masked_fill(cols[None, :] >= visible[:, None], float("-inf"))
    vals, idx = torch.topk(masked, k, dim=-1)
    picked = idx.to(torch.int32).masked_fill(vals == float("-inf"), -1)
    if k < block_topk:
        pad = torch.full((t, block_topk - k), -1, dtype=torch.int32, device=scores.device)
        picked = torch.cat((picked, pad), dim=1)
    return picked


def expand_selection(blocks: torch.Tensor, positions: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Tokens of the selected blocks: ``blocks`` ``[T, K]`` int32 (-1 padded),
    ``positions`` ``[T]`` -> int32 ``[T, K * ratio + ratio - 1]`` absolute token
    positions.

    The ``ratio`` tokens of every selected block (``-1`` where the block is
    ``-1``), then the tail of the incomplete current block: ``start =
    visible_blocks(positions) * ratio``, ``tail_j = start + j`` for ``j`` in
    ``0..ratio-2``, kept where ``tail_j <= pos`` else ``-1``. The SET per row
    equals HF's ``selected_token_indices`` row; the order is free.
    """
    neg = torch.tensor(-1, dtype=torch.int32, device=blocks.device)
    offs = torch.arange(compress_ratio, dtype=torch.int32, device=blocks.device)
    tokens = torch.where(blocks[..., None] >= 0, blocks[..., None] * compress_ratio + offs, neg).flatten(1)
    start = visible_blocks(positions, compress_ratio).to(torch.int32) * compress_ratio
    tail_offs = torch.arange(compress_ratio - 1, dtype=torch.int32, device=blocks.device)
    tail = start[:, None] + tail_offs[None, :]
    tail = torch.where(tail <= positions[:, None].to(torch.int32), tail, neg)
    return torch.cat((tokens, tail), dim=1)


def rows_without_selection(selected: torch.Tensor) -> torch.Tensor:
    """Bool ``[..., 1]``: True for the rows of ``selected`` (``[..., W]``, -1 padded) with no position at all.

    One max over the row (a tensor op, no host read): such rows are the padding rows of a bucket.
    """
    return selected.amax(dim=-1, keepdim=True) < 0


def decode_selected_slots(selected: torch.Tensor, block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Map ``selected`` positions to flat cache slots through ``block_table``.

    ``selected`` is ``[bs, W]`` int32 of absolute positions (-1 padded) and
    ``block_table`` is ``[bs, MB]`` of kernel block ids. Returns ``[bs, W]``
    int64 slots ``block_table.gather(1, pos // block_size) * block_size +
    pos % block_size`` (``pos = clamp_min(selected, 0)``). Entries with
    ``selected < 0`` map to a valid slot (row 0's first block) so the gather is
    always in range; the caller masks them out.
    """
    pos = selected.to(torch.long).clamp_min(0)
    block_id = block_table.to(torch.long).gather(1, pos // block_size)
    return block_id * block_size + (pos % block_size)


COPIED = ("gemma_rms_norm", "rope_neox_partial", "index_queries", "compress_block_keys", "visible_blocks",
          "select_blocks", "expand_selection", "rows_without_selection", "decode_selected_slots")

__all__ = ["COPIED", "qsa_index_emit_ref", "qsa_index_prepare_ref", "rows_table", *COPIED]
