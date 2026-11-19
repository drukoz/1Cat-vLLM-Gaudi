# SPDX-License-Identifier: Apache-2.0
"""Pure-torch selection ops for the Flash-Next QSA indexer (M2-T1).

Reproduces the token selection of HF ``transformers`` ``Qwen4ExpTextQSAIndexer``
(``models/qwen4_exp/modeling_qwen4_exp.py``, lines 670-773): for every query
token, score the complete blocks of ``compress_ratio`` tokens in the visible
prefix, keep the top ``block_topk`` blocks by ``sum_h relu(q_h . k_b) / sqrt(D)``
in fp32, and select the 4 tokens of every kept block plus the tail of the
incomplete current block.

Shapes and dtypes (``T`` tokens, ``H = index_n_heads``, ``D = index_head_dim``,
``N`` block keys, ``ratio = indexer_compress_ratio``, ``R = rotary_dim < D``):
``cos``/``sin`` are the half-width rotary tables ``[T, R2]`` (``R = 2 * R2``),
i.e. vLLM's ``cos_sin_cache[positions].chunk(2, -1)``; HF duplicates the half to
its full ``[T, R]`` layout inside ``apply_rotary_pos_emb``.

No HPU code and no vLLM imports: the layer (M2-T3) feeds these from its
projection outputs, and the tests compare against the HF module on CPU.
"""

import math

import torch


def ring_rows(compress_ratio: int, num_speculative_tokens: int) -> int:
    """Number of raw-key ring rows per slot: ``compress_ratio *
    ceil((compress_ratio + num_speculative_tokens) / compress_ratio)``.

    ``ring_rows(4, 0) == 4`` without speculation, ``ring_rows(4, 3) == 8`` with
    3 speculative tokens (int -> int).
    """
    return compress_ratio * -(-(compress_ratio + num_speculative_tokens) // compress_ratio)


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


def score_blocks(q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Block scores: ``q`` ``[T, H, D]``, ``keys`` ``[N, D]`` -> fp32 ``[T, N]``:
    ``relu(q.float() @ keys.float().T)`` summed over ``H``, divided by
    ``sqrt(D)``.
    """
    scores = torch.relu(torch.matmul(q.float(), keys.float().transpose(0, 1)))  # [T, H, N]
    return scores.sum(dim=1) / math.sqrt(q.shape[-1])


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


def selection_mask(selected: torch.Tensor, num_columns: int) -> torch.Tensor:
    """Selection mask: ``selected`` ``[T, W]`` int32 -> bool ``[T,
    num_columns]``, True at the selected positions.

    Scatters True into a ``[T, num_columns + 1]`` buffer with -1 entries
    redirected to the extra column, then drops it (HF's trick).
    """
    buf = torch.zeros(selected.shape[0], num_columns + 1, dtype=torch.bool, device=selected.device)
    idx = torch.where(selected >= 0, selected, torch.tensor(num_columns, dtype=torch.int32, device=selected.device))
    return buf.scatter(-1, idx.to(torch.int64), True)[..., :num_columns]


def compressed_row_slots(slot_mapping: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Compressed-cache rows: ``slot_mapping`` ``[T]`` int64 (attention slot of
    each token) -> ``slot_mapping // compress_ratio``, the compressed row of the
    token's block (int64 ``[T]``).
    """
    return slot_mapping // compress_ratio


def ring_write(ring: torch.Tensor, slot: torch.Tensor, positions: torch.Tensor, raw_k: torch.Tensor) -> None:
    """Write raw keys into the ring (in place): ``ring`` ``[S, R, D]`` (contiguous),
    ``slot`` ``[bs]`` int64, ``positions`` ``[bs]`` int64, ``raw_k`` ``[bs, D]``:
    ``ring.view(-1, D).index_copy_(0, slot * R + positions % R, raw_k)``, the
    flat-index form of writing the two-axis row ``(slot, positions % R)`` (the
    HPU-compilable gather/write form; no multi-index write).
    """
    ring.view(-1, ring.shape[2]).index_copy_(0, slot * ring.shape[1] + positions % ring.shape[1],
                                             raw_k.to(ring.dtype))


def ring_block_rows(ring: torch.Tensor, slot: torch.Tensor, positions: torch.Tensor, raw_k: torch.Tensor,
                    compress_ratio: int) -> torch.Tensor:
    """The ``ratio`` raw keys of the blocks that the tokens at ``positions``
    (``positions % ratio == ratio - 1``) complete.

    ``ring`` ``[S, R, D]`` (contiguous), ``slot`` ``[bs]`` int64, ``positions``
    ``[bs]`` int64, ``raw_k`` ``[bs, D]``: rows ``j = 0 .. ratio-2`` come from
    ``ring.view(-1, D).index_select(0, slot * R + (positions - (ratio - 1) + j) % R)``,
    the flat-index form of the two-axis row ``(slot, (positions - (ratio - 1) +
    j) % R)``, the last row is ``raw_k`` -> ``[bs, ratio, D]`` in ``ring``'s
    dtype. Rows whose position does not complete a block are garbage and the
    caller ignores them.
    """
    r = ring.shape[1]
    rows = [
        ring.view(-1, ring.shape[2]).index_select(0, slot * r + (positions - (compress_ratio - 1) + j) % r)
        for j in range(compress_ratio - 1)
    ]
    rows.append(raw_k)
    return torch.stack(rows, dim=1)
