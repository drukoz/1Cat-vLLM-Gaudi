# SPDX-License-Identifier: Apache-2.0
"""HPU sparse QSA attention for Flash-Next (M2).

The QSA layer (M2-T3) calls ``self.sparse_attn(q, k, v, selected, positions, md)``
where the indexer (M2-T1) has already picked, for every query row, a set of
absolute token positions. This module runs the layer's attention over exactly
those positions on HPU:

* prefill mirrors the dense ``HPUAttentionImpl`` prompt path (write the chunk's
  K/V, then ``ops.prompt_attention``) but with an explicit selection bias and
  ``is_causal=False``; every query tile attends to the whole context + chunk
  and the bias keeps only the selected columns,
* decode gathers the selected K/V rows through the decode ``block_table`` and
  runs a small masked SDPA over them.

The three module-level helpers (``prefill_selection_bias``,
``decode_selected_slots``, ``sparse_decode_attention``) are pure torch so the
CPU unit tests can check them against dense attention with a mask. The
``HpuQsaSparseAttention`` wrapper pulls in the HPU-only ``ops`` /
``_set_fetch_by_id`` modules lazily inside the prefill branch, so importing
this file on a machine without HPU libraries succeeds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm_gaudi.envs as envs


def rows_without_selection(selected: torch.Tensor) -> torch.Tensor:
    """Bool ``[..., 1]``: True for the rows of ``selected`` (``[..., W]``, -1 padded) with no position at all.

    One max over the row (a tensor op, no host read): such rows are the padding rows of a bucket.
    """
    return selected.amax(dim=-1, keepdim=True) < 0


def prefill_selection_bias(selected: torch.Tensor, context_lens: torch.Tensor, n_ctx_cols: int, num_chunk_cols: int,
                           dtype: torch.dtype) -> torch.Tensor:
    """Bias selecting the indexer's positions inside the prompt-attention view.

    ``selected`` is ``[bs, T, W]`` of absolute token positions (-1 padded) for
    one query tile of ``T`` rows, ``context_lens`` is ``[bs]`` int64,
    ``n_ctx_cols = n_ctx_blocks * block_size`` is the context-column width that
    ``_include_past`` prepends and ``num_chunk_cols`` is the **full chunk
    length** (all ``seq_len`` keys the kernel sees, not this tile's height).
    Returns a ``[bs, 1, T, n_ctx_cols + num_chunk_cols]`` bias in ``dtype``: 0
    where column ``c`` is selected for that query, ``-inf`` elsewhere. Column
    of position ``p`` for row ``b`` is ``p`` when ``p < context_lens[b]``
    (context block, logical order) and ``n_ctx_cols + (p - context_lens[b])``
    for the chunk. The query-row count is ``selected.shape[1]`` (the tile
    height), decoupled from the column count so every tile of a long-context
    prefill shares the same ``n_ctx_cols + seq_len`` key width. ``-1`` entries
    are redirected to an extra dump column that is dropped. A query row with no
    selected position at all (a padding row of the bucket, whose position is
    ``-1``) gets an all-zero bias row instead of an all ``-inf`` one: the
    softmax of an all ``-inf`` row is NaN, and the layers after this one carry
    a NaN on a padding row into the recurrent state they save. Pure torch, no
    loops over tokens.
    """
    bs, num_queries, _ = selected.shape
    cols = n_ctx_cols + num_chunk_cols
    sel = selected.long()
    ctx = context_lens.to(torch.long).view(bs, 1, 1)
    valid = sel >= 0
    pos = sel.clamp_min(0)
    col = torch.where(pos < ctx, pos, n_ctx_cols + (pos - ctx))
    # Redirect padding (-1) entries to the dump column at index `cols`.
    col = torch.where(valid, col, torch.full_like(col, cols))
    # The zeros are scattered straight into the bias, in its own float dtype and with a numeric
    # value. A bool mask filled with ``scatter_(2, col, True)`` does not run on a card: the Habana
    # graph has no scatter_ overload for a bool value ("We don't have an op for aten::scatter_ ...
    # Argument types: Tensor, int, Tensor, bool", rental 2026-11-20).
    bias = torch.full((bs, num_queries, cols + 1), float("-inf"), dtype=dtype, device=selected.device)
    bias.scatter_(2, col, 0.0)
    bias = bias[:, :, :cols]  # drop the dump column
    nothing_selected = rows_without_selection(selected)  # [bs, T, 1]
    return bias.masked_fill(nothing_selected, 0.0).unsqueeze(1)


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


def _gather(cache: torch.Tensor, flat: torch.Tensor, dequant=None) -> torch.Tensor:
    """The cache rows at ``flat`` (one ``index_select``), dequantized when the cache holds FP8 values."""
    rows = cache.index_select(0, flat)
    return rows if dequant is None else dequant(rows)


def bf16_attention_math(impl, modules=None) -> bool:
    """Put the backend's attention math of one layer back to bf16 when ``fp8_inc`` switched it to FP8.

    With ``--kv-cache-dtype fp8_inc`` the fork's ``HPUAttentionImpl`` stores K/V as E4M3 (``VLLMFP8KVCache``:
    quantize on write, dequantize on fetch) and also computes its attention with FP8 matmuls and an FP8 fused
    SDPA. A QSA layer under ``VLLM_HPU_QSA_KV_FP8`` keeps the storage and gets the bf16 math back, so only the
    K/V values are rounded. ``modules`` provides ``Matmul``, ``B2BMatmul``, ``ModuleFusedSDPA`` and ``fsdpa``
    (default: ``vllm_gaudi.extension``; the CPU tests pass stand-ins). Returns whether anything was replaced.
    """
    if not getattr(impl, "enable_fp8_attn", False):
        return False
    if modules is None:
        import vllm_gaudi.extension.kernels as kernels
        from vllm_gaudi.extension import utils as ext_utils

        class modules:  # noqa: N801  (a namespace)
            Matmul = ext_utils.Matmul
            B2BMatmul = ext_utils.B2BMatmul
            ModuleFusedSDPA = ext_utils.ModuleFusedSDPA
            fsdpa = staticmethod(kernels.fsdpa)

    impl.matmul_qk = modules.Matmul()
    impl.matmul_av = modules.Matmul()
    impl.batch2block_matmul = modules.B2BMatmul()
    impl.block2batch_matmul = modules.B2BMatmul()
    fused = modules.fsdpa()
    impl.fused_scaled_dot_product_attention = None if fused is None else modules.ModuleFusedSDPA(fused)
    impl.enable_fp8_attn = False
    return True


def sparse_decode_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                            slots: torch.Tensor, valid: torch.Tensor, num_heads: int, num_kv_heads: int,
                            head_dim: int, scale: float, k_dequant=None, v_dequant=None) -> torch.Tensor:
    """Gathered-K/V masked SDPA over the selected decode positions.

    ``query`` is ``[bs, H * D]``, ``key_cache`` / ``value_cache`` are
    ``[S, KV, D]``, ``slots`` is ``[bs, W]`` int64 and ``valid`` is ``[bs, W]``
    bool. Returns ``[bs, H * D]``: gather the selected rows with a flat
    ``index_select`` (the HPU-compilable form), broadcast KV heads to query
    heads for GQA, score ``q @ k^T * scale`` with ``-inf`` on invalid columns,
    softmax in fp32 cast back to the query dtype, and take the weighted sum of
    the gathered values. ``k_dequant`` / ``v_dequant`` (``VLLM_HPU_QSA_KV_FP8``) turn the gathered rows of an
    FP8 cache into the query's dtype right after the gather.
    """
    bs, width = slots.shape
    flat = slots.reshape(-1)
    k = _gather(key_cache, flat, k_dequant).view(bs, width, num_kv_heads, head_dim)
    v = _gather(value_cache, flat, v_dequant).view(bs, width, num_kv_heads, head_dim)
    repeat = num_heads // num_kv_heads
    if repeat > 1:
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
    # [bs, H, 1, D] @ [bs, H, D, W] -> [bs, H, 1, W]
    q = query.view(bs, num_heads, 1, head_dim)
    scores = torch.matmul(q, k.permute(0, 2, 3, 1)) * scale
    scores = scores.masked_fill(~valid.view(bs, 1, 1, width), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    # [bs, H, 1, W] @ [bs, H, W, D] -> [bs, H, 1, D] -> [bs, H * D]
    out = torch.matmul(probs, v.transpose(1, 2))
    return out.reshape(bs, num_heads * head_dim)


# The grouped decode attention from this many query rows up (a static choice per bucket). Card job 00011
# (reports/qsa-fused/card/qsa-bench1): grouped against repeated K/V, device ms per layer at W = 2051, 3 query
# heads per KV head: 0.51 vs 0.36 at 4 rows, 0.51 vs 0.78 at 32, 0.68 vs 1.85 at 128; bit-exact at every size.
GROUPED_MIN_ROWS = 8


def sparse_decode_attention_grouped(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                                    slots: torch.Tensor, valid: torch.Tensor, num_heads: int, num_kv_heads: int,
                                    head_dim: int, scale: float, k_dequant=None, v_dequant=None) -> torch.Tensor:
    """``sparse_decode_attention`` with the query heads of each KV head in the GEMM's ``M`` dimension.

    Same inputs and result. The gathered ``[bs, W, KV, D]`` rows are not repeated for the
    ``G = num_heads // num_kv_heads`` query heads of a KV head: ``q`` is viewed as ``[bs, KV, G, D]``
    and multiplied with ``k`` as ``[bs, KV, D, W]`` (``[bs, KV, G, W]`` scores), the probabilities with
    ``v`` as ``[bs, KV, W, D]``. Query head ``h`` uses KV head ``h // G``, as ``repeat_interleave`` does.
    With one KV head per rank (Flash-Next at TP8, TP4 and TP2) the K and V operands are plain views of
    the gathered rows (I21, G5-1; the grouped form of ``extension/ops.py`` ``flat_pa``).
    """
    bs, width = slots.shape
    groups = num_heads // num_kv_heads
    flat = slots.reshape(-1)
    k = _gather(key_cache, flat, k_dequant).view(bs, width, num_kv_heads, head_dim)
    v = _gather(value_cache, flat, v_dequant).view(bs, width, num_kv_heads, head_dim)
    if num_kv_heads == 1:
        k, v = k.view(bs, 1, width, head_dim), v.view(bs, 1, width, head_dim)
    else:
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    q = query.view(bs, num_kv_heads, groups, head_dim)
    # [bs, KV, G, D] @ [bs, KV, D, W] -> [bs, KV, G, W]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~valid.view(bs, 1, 1, width), float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    # [bs, KV, G, W] @ [bs, KV, W, D] -> [bs, KV, G, D] -> [bs, H * D]
    out = torch.matmul(probs, v)
    return out.reshape(bs, num_heads * head_dim)


class HpuQsaSparseAttention(torch.nn.Module):
    """Sparse QSA attention over the indexer's selection, wrapping an Attention.

    Built as ``HpuQsaSparseAttention(attn)`` where ``attn`` is the QSA layer's
    ``Attention`` child: its ``.impl`` is the fork's ``HPUAttentionImpl`` and
    its bound KV cache is ``attn.kv_cache``, the ``(key_cache, value_cache,
    k_scales, v_scales)`` tuple the runner allocates and ``bind_kv_cache``
    assigns (the fork's dense path reads it the same way,
    ``vllm_gaudi/ops/hpu_attention.py``). The pinned vLLM has no
    ``ForwardContext.virtual_engine`` and no per-engine cache list.
    """

    def __init__(self, attn) -> None:
        super().__init__()
        self.attn = attn
        impl = attn.impl
        self.num_heads = int(impl.num_heads)
        self.num_kv_heads = int(impl.num_kv_heads)
        self.head_dim = int(impl.head_size)
        # VLLM_HPU_QSA_FUSED >= 1: the grouped decode attention (no K/V repeat); read once.
        from vllm_gaudi.models.qwen4_exp.qsa_fused_path import QsaFusedMode, qsa_fused_mode
        self.grouped_decode = qsa_fused_mode() is not QsaFusedMode.OFF
        # VLLM_HPU_QSA_KV_FP8 with --kv-cache-dtype fp8_inc (the layer refuses fp8_inc without the flag): E4M3
        # K/V storage through the backend's caches, bf16 attention math, the decode gather dequantized.
        self.kv_fp8 = getattr(impl, "kv_cache_dtype", None) == "fp8_inc"
        self.k_dequant = self.v_dequant = None
        if self.kv_fp8:
            bf16_attention_math(impl)
            self.k_dequant = getattr(impl.k_cache, "dequant_output", None) or (lambda t: t.to(torch.bfloat16))
            self.v_dequant = getattr(impl.v_cache, "dequant_output", None) or (lambda t: t.to(torch.bfloat16))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, selected: torch.Tensor,
                positions: torch.Tensor, md) -> torch.Tensor:
        """Run the layer's attention over ``selected`` positions, return ``[T, H * D]``.

        ``q`` is ``[T, H * D]``, ``k`` / ``v`` are ``[T, KV * D]`` (already
        normed and RoPE'd), ``selected`` is int32 ``[T, W]`` of absolute token
        positions (-1 padded) and ``md`` is the forward's
        ``HPUAttentionMetadata``.
        """
        impl = self.attn.impl
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        kv_cache = self.attn.kv_cache
        # Split the (key, value, k_scales, v_scales) cache tuple; keep it local
        # so decode needs no HPU-only import (split_kv_cache is pure indexing).
        key_cache, value_cache, k_scales, v_scales = kv_cache[0], kv_cache[1], kv_cache[2], kv_cache[3]
        impl.do_kv_cache_update(self.attn, k, v, kv_cache, md.slot_mapping)

        if md.is_prompt:
            # HPU-only modules; the CPU tests only exercise the pure helpers.
            import vllm_gaudi.extension.ops as ops
            from vllm_gaudi.attention.backends.hpu_attn import _set_fetch_by_id

            tokens = q.shape[0]
            rows = int(md.context_lens_tensor.numel()) if md.context_lens_tensor is not None else 1
            seq_len = tokens // rows
            n_ctx_cols = 0 if md.block_list is None else int(md.block_list.numel()) // rows * int(md.block_size)
            if md.context_lens_tensor is not None:
                context_lens = md.context_lens_tensor.to(torch.long)
            else:
                context_lens = torch.zeros(rows, dtype=torch.long, device=q.device)

            # Prefill context blocks are raw scattered ids: gather by id.
            _set_fetch_by_id(impl.k_cache, True)
            _set_fetch_by_id(impl.v_cache, True)

            tile = int(envs.VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE)
            selected_rows = selected.view(rows, seq_len, selected.shape[-1])
            key_full = k.view(rows, seq_len, num_kv_heads, head_dim)
            value_full = v.view(rows, seq_len, num_kv_heads, head_dim)
            query_full = q.view(rows, seq_len, num_heads, head_dim)
            common_args = impl.common_attention_args(md.block_list, key_cache, value_cache, md.block_size, k_scales,
                                                     v_scales)

            out_tiles = []
            for start in range(0, seq_len, tile):
                end = min(start + tile, seq_len)
                bias = prefill_selection_bias(selected_rows[:, start:end, :], context_lens, n_ctx_cols, seq_len,
                                              q.dtype)
                out_tile = ops.prompt_attention(
                    impl=impl.prefill_impl,
                    query=query_full[:, start:end, :, :],
                    key=key_full,
                    value=value_full,
                    is_causal=False,
                    attn_bias=bias,
                    position_bias=None,
                    valid_seq_lengths=None,
                    **common_args)
                out_tiles.append(out_tile)
            out = torch.cat(out_tiles, dim=1)
            return out.reshape(tokens, num_heads * head_dim)

        from vllm_gaudi.models.qwen4_exp.qsa_fused_path import QsaDecodeSelection
        if isinstance(selected, QsaDecodeSelection):  # mode 2: the slots and flags come from qsa_index_emit
            attend = (sparse_decode_attention_grouped if int(q.shape[0]) >= GROUPED_MIN_ROWS else
                      sparse_decode_attention)
            out = attend(q, key_cache, value_cache, selected.slots, selected.valid != 0, num_heads, num_kv_heads,
                         head_dim, impl.scale, k_dequant=self.k_dequant, v_dequant=self.v_dequant)
            return out.reshape(q.shape[0], num_heads * head_dim)
        slots = decode_selected_slots(selected, md.block_table, int(md.block_size))
        # A row with no selected position (a padding row of the batch under M-RoPE: position -1)
        # attends to its gathered pad-block rows instead of to nothing: the softmax of an all
        # -inf row is NaN.
        valid = (selected >= 0) | rows_without_selection(selected)
        attend = (sparse_decode_attention_grouped if self.grouped_decode and int(q.shape[0]) >= GROUPED_MIN_ROWS else
                  sparse_decode_attention)
        out = attend(q, key_cache, value_cache, slots, valid, num_heads, num_kv_heads, head_dim, impl.scale,
                     k_dequant=self.k_dequant, v_dequant=self.v_dequant)
        return out.reshape(q.shape[0], num_heads * head_dim)


__all__ = [
    "HpuQsaSparseAttention",
    "bf16_attention_math",
    "decode_selected_slots",
    "prefill_selection_bias",
    "rows_without_selection",
    "sparse_decode_attention",
    "sparse_decode_attention_grouped",
    "GROUPED_MIN_ROWS",
]
