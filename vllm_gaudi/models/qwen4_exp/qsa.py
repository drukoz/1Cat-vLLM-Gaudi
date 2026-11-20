# SPDX-License-Identifier: Apache-2.0
"""HPU QSA attention for Flash-Next: dense paged attention behind QsaMode.

The upstream ``vllm/models/qwen4_exp/nvidia/qsa.py`` owner runs sparse paged
attention through a FlashAttention implementation, registers itself as the
KV-cache owner and keeps side caches plus a topk buffer.  The HPU class
replaces all of that for M0: dense causal attention through a standard
``Attention`` child (exact while the indexer selects the full prefix), the
same checkpoint parameter names, and the platform KV-cache owner.  Mode
selection is the integer ``QsaMode`` enum read from
``VLLM_HPU_FLASH_NEXT_QSA_MODE`` (default ``0`` = dense); ``SPARSE`` (1) is
the M2 indexer top-k path: ``HpuQsaIndexer.select`` picks the attended
positions and the layer hands them to ``self.sparse_attn`` (the HPU wrapper
M2-T5c, a CPU stand-in in tests).
"""

from __future__ import annotations

import enum
from typing import Callable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.models.qwen4_exp.nvidia import model
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
from vllm.v1.attention.backend import AttentionBackend

import vllm_gaudi.envs as envs
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import (
    compress_block_keys,
    compressed_row_slots,
    expand_selection,
    index_queries,
    ring_block_rows,
    ring_write,
    score_blocks,
    select_blocks,
    visible_blocks,
)


class QsaMode(enum.IntEnum):
    """QSA attention mode for the Flash-Next Qwen4-Exp layers (next free: 2)."""

    DENSE = 0
    SPARSE = 1


def platform_is_hpu() -> bool:
    """True when vLLM's current platform is the Gaudi (HPU) platform.

    No platform defines ``is_hpu`` (``Platform.__getattr__`` answers an unknown name with ``None``
    and the log line "Current platform hpu does not have 'is_hpu' attribute."), so the test reads
    the two attributes ``HpuPlatform`` declares. ``device_name`` is checked next to ``device_type``
    because ``HPUModelRunner.load_model`` sets ``device_type`` to the load device (``"cpu"`` for an
    INC quantization run) while the model is being built.
    """
    return "hpu" in (current_platform.device_type, current_platform.device_name)


def prefill_row_lengths(md: object, rows: int, seq: int, device: torch.device) -> torch.Tensor:
    """Real token count of every row of a prompt bucket: int64 ``[rows]``, at most ``seq``.

    ``query_start_loc_p`` is the runner's int32 cumulative sum of the real query lengths of the
    bucket's rows (``_form_prefill_batch``; a chunk of a chunked prefill is its own bucket) and
    ``seq_lens_tensor`` the lengths themselves. Metadata with neither describes unpadded rows.
    """
    query_start_loc = getattr(md, "query_start_loc_p", None)
    if query_start_loc is not None and query_start_loc.numel() == rows + 1:
        qsl = query_start_loc.to(device=device, dtype=torch.long)
        return (qsl[1:] - qsl[:-1]).clamp(min=0, max=seq)
    seq_lens = getattr(md, "seq_lens_tensor", None)
    if seq_lens is not None and seq_lens.numel() == rows:
        return seq_lens.to(device=device, dtype=torch.long).clamp(min=0, max=seq)
    return torch.full((rows,), seq, dtype=torch.long, device=device)


def indexer_positions(positions: torch.Tensor, mrope: bool = True) -> torch.Tensor:
    """The one position per query row the indexer works with: ``[num_tokens]``, ``-1`` on bucket padding.

    Without M-RoPE (``--language-model-only``: the engine strips ``mrope_section``, ``uses_mrope`` is
    False) the runner sends ``[num_tokens]`` or ``[rows, bucket]`` positions, which flatten to one per
    row. With M-RoPE it sends ``[3, num_tokens]`` (temporal, height and width rows), and row 0 is taken.

    Row 0 is exact for text tokens only. The engine (``nvidia/indexer_qsa.py``) rotates the indexer
    queries and block keys with the attention's M-RoPE composition (each rotary section from its own
    row) and does the block bookkeeping on the request's logical token index, not on a RoPE row. For a
    text token the three rows are equal and equal to that index, so both reduce to row 0. Tokens of an
    image or video (rows that differ, and the text after them, whose index and RoPE position differ by
    ``mrope_position_delta``) are not supported by this indexer.
    """
    if mrope and positions.dim() == 2:
        return positions[0]
    return positions.reshape(-1)


class HpuQsaIndexer(nn.Module):
    """Flash-Next QSA indexer: checkpoint weights plus top-k token selection.

    Holds the three checkpoint modules of the upstream ``QSAIndexer``
    (``index_qk_proj``, ``q_layernorm``, ``k_layernorm``) under the indexer
    prefix so ``load_weights`` resolves every indexer tensor, and reproduces
    HF ``Qwen4ExpTextQSAIndexer`` selection in :meth:`select` (M2, sparse mode).
    The two side caches (``compressed_key_cache`` of block keys, ``key_ring`` of
    raw keys) are plain instance attributes bound by the model runner (HPU) or
    the CPU harness; ``None`` means "not bound", so dense mode and profiling runs
    never touch the sparse path.
    """

    SCORE_TILE = 512  # query rows scored against the block keys in one pass

    def __init__(
        self,
        *,
        config: Qwen4ExpTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.prefix = prefix
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = int(config.indexer_budget) // self.compress_ratio
        self.output_width = self.block_topk * self.compress_ratio + self.compress_ratio - 1
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=self.eps)
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=self.eps)

        self.compressed_key_cache: torch.Tensor | None = None
        self.key_ring: torch.Tensor | None = None
        self.ring_group_idx: torch.Tensor | None = None

    def _state_indices(self, md: object, name: str) -> torch.Tensor | None:
        """Resolve one state-index tensor for this layer's ring group (see ``HpuQwen4ExpPLELayer``).

        Three shapes are accepted (the runner's tensors are ``[num_groups, ...]``
        and the indexer's ``ring_group_idx`` picks this layer's group row):

        * ``[groups, bs]``: one slot per state row (plain decode, the drafter's
          later steps) -> ``[bs]``;
        * ``[groups, bs, 1]``: the same with the GDN trailing 1 -> ``[bs]``;
        * ``[groups, bs, T]``: the M3 verification lattice (``store`` is
          identical to ``load``).  The ring and the compressed rows are
          addressed by the request's slot, not by the lattice column (the ring
          rows are position-addressed inside a slot and the compressed rows
          come from ``slot_mapping``), so column 0 is the request's slot and
          the result is ``[bs, T] -> [:, 0].repeat_interleave(T)``, i.e. the
          ``bs * T`` per-row slots directly.
        """
        indices = getattr(md, name, None)
        if indices is None:
            return None
        if indices.dim() > 1:
            if self.ring_group_idx is None:
                raise RuntimeError("QSA indexer ring_group_idx not set")
            indices = indices.index_select(0, self.ring_group_idx.to(device=indices.device).view(1)).squeeze(0)
        if indices.dim() == 2:
            if indices.shape[-1] > 1:
                return indices[:, 0].repeat_interleave(int(indices.shape[-1])).long()
            indices = indices.squeeze(-1)
        return indices.reshape(-1).long()

    @torch.no_grad()
    def select(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_at: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        md: object,
    ) -> torch.Tensor:
        """Attended positions per query token: int32 ``[T, output_width]`` (absolute, -1 padded)."""
        H, D, ratio = self.index_n_heads, self.index_head_dim, self.compress_ratio
        topk, width = self.block_topk, self.output_width
        T = hidden.shape[0]
        qk, _ = self.index_qk_proj(hidden)
        q, raw_k = qk.split([H * D, self.index_kv_heads * D], dim=-1)
        q = q.view(T, H, D)
        cos, sin = cos_sin_at(positions)
        q = index_queries(q, self.q_layernorm.weight, self.eps, cos, sin)

        load = self._state_indices(md, "load_indices_tensor")
        store = self._state_indices(md, "store_indices_tensor")
        if store is None:
            store = load
        slot_mapping = md.slot_mapping.reshape(-1)[:T].long()
        rpb = int(md.block_size) // ratio
        cache = self.compressed_key_cache

        if md.is_prompt:
            rows = load.numel()
            seq = T // rows
            ctx = md.context_lens_tensor
            ctx = torch.zeros(rows, dtype=torch.long, device=hidden.device) if ctx is None else ctx.long()
            # ctx is 4-aligned by construction (the base scheduler's mamba-block-aligned chunking,
            # the runner's assert); no device check here: a host sync breaks HPU graph capture.
            positions2 = positions.view(rows, seq)
            raw2 = raw_k.view(rows, seq, D)
            n_full = seq // ratio  # complete blocks inside the chunk (ctx 4-aligned => blocks start at t = 0, 4, ...)
            blocks = raw2[:, :n_full * ratio].reshape(rows * n_full, ratio, D)
            starts = positions2[:, :n_full * ratio:ratio].reshape(-1)
            cs, sn = cos_sin_at(starts)
            keys = compress_block_keys(blocks, self.k_layernorm.weight, self.eps, cs, sn)
            last_slots = compressed_row_slots(slot_mapping.view(rows, seq)[:, ratio - 1:n_full * ratio:ratio].
                                              reshape(-1), ratio)
            # ``seq`` is the bucket length: a block whose last token is bucket padding (the block
            # that straddles the end of the prompt, and every block after it) is not complete. Its
            # key is built from padding and its last token's slot is the runner's pad slot (-1), so
            # it writes the cache's scratch row (its last row, see qsa_side_caches) and the write
            # keeps one fixed shape.
            lens = prefill_row_lengths(md, rows, seq, hidden.device)
            block_ends = (torch.arange(n_full, device=hidden.device) + 1) * ratio
            in_prompt = (block_ends[None, :] <= lens[:, None]).reshape(-1)
            scratch_row = torch.full_like(last_slots, cache.shape[0] - 1)
            cache.index_copy_(0, torch.where(in_prompt, last_slots, scratch_row), keys.to(cache.dtype))
            # The prompt's tail: the up to ``ratio - 1`` real tokens after the row's last complete
            # block. Their raw keys go to the ring, where the decode step that completes the block
            # reads them (``ring_block_rows``). The tail sits at the row's REAL length, not at the
            # end of the bucket, so it is gathered per row; the entries past the real length (and
            # every entry of a row without a tail) are parked on the ring's scratch slot, which
            # keeps the write at one fixed shape, ``rows * (ratio - 1)``.
            tail_cols = (lens // ratio * ratio)[:, None] + torch.arange(ratio - 1, device=hidden.device)[None, :]
            tail_real = (tail_cols < lens[:, None]).reshape(-1)
            tail_rows = (torch.arange(rows, device=hidden.device)[:, None] * seq
                         + tail_cols.clamp(max=seq - 1)).reshape(-1)
            ring_scratch = torch.full_like(tail_rows, self.key_ring.shape[0] - 1)
            ring_write(self.key_ring, torch.where(tail_real, store.repeat_interleave(ratio - 1), ring_scratch),
                       positions.index_select(0, tail_rows).clamp_min(0), raw_k.index_select(0, tail_rows))
            # Block keys per row in absolute block order (column i = tokens 4*i ..): the context
            # rows come from the row's kernel blocks (block_list, rpb compressed rows each), the
            # chunk's blocks go at ctx // ratio + j (overwriting a partial last context block).
            n_ctx = 0 if md.block_list is None else md.block_list.numel() // rows
            ctx_keys = (cache.view(-1, rpb, D).index_select(0, md.block_list.long().clamp_min(0).reshape(-1)).reshape(
                rows, n_ctx * rpb, D) if n_ctx else cache.new_zeros((rows, 0, D)))
            all_rows = torch.cat((ctx_keys, cache.new_zeros((rows, n_full, D))), dim=1)
            dst = (ctx // ratio)[:, None] + torch.arange(n_full, device=hidden.device)[None, :]
            all_rows.scatter_(1, dst[..., None].expand(-1, -1, D), keys.view(rows, n_full, D).to(cache.dtype))
            q4 = q.view(rows, seq, H, D)
            out = cache.new_full((rows, seq, width), -1, dtype=torch.int32)
            for b in range(rows):
                row_keys = all_rows[b]
                for lo in range(0, seq, self.SCORE_TILE):
                    hi = min(lo + self.SCORE_TILE, seq)
                    pos_tile = positions2[b, lo:hi]
                    scores = score_blocks(q4[b, lo:hi], row_keys)
                    blocks_sel = select_blocks(scores, visible_blocks(pos_tile, ratio), topk)
                    out[b, lo:hi] = expand_selection(blocks_sel, pos_tile, ratio)
            return out.view(T, width)

        return self._select_decode(q, raw_k, positions, cos_sin_at, md, load, store, slot_mapping, cache, rpb)

    def _select_decode(
        self,
        q: torch.Tensor,
        raw_k: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_at: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        md: object,
        load: torch.Tensor,
        store: torch.Tensor,
        slot_mapping: torch.Tensor,
        cache: torch.Tensor,
        rpb: int,
    ) -> torch.Tensor:
        """Decode selection over ``T`` query rows of ``bs`` requests (M3: ``T_cols >= 1`` tokens each).

        Write-then-score: the step's raw keys go into the ring and the
        completed blocks' compressed rows into ``compressed_key_cache``
        BEFORE any row is scored.  Both caches are position-addressed
        (the request slot's ring row ``pos % R``, ``compressed_row_slots(slot_mapping)``), so a
        token re-processed at a position overwrites the row a rejected draft
        left there: after a verification step the caches hold the accepted
        tokens' rows and rollback needs no extra state (no checkpoint, no
        undo log).  Scoring a row before writing would let a query see the
        block key of a draft that is about to be replaced.

        Rows are the padded M3 layout: ``bs`` requests of ``T_cols = T // bs``
        rows each, the real length of request ``b`` is
        ``md.query_start_loc_p[b + 1] - md.query_start_loc_p[b]``; the padding
        rows of a ragged request (and the rows of a step without
        ``query_start_loc_p``: every row is valid) are parked on the ring's
        scratch row ``key_ring.shape[0] - 1``, which is never a request's slot
        (see ``qsa_side_caches``), and never write a compressed row.  The
        single-token decode is the ``T_cols == 1`` case: there the ring write
        only ever adds the current token's key (earlier keys were written by
        earlier steps), so moving it before ``ring_block_rows`` leaves it
        unchanged and one code path serves both.
        """
        D, ratio = self.index_head_dim, self.compress_ratio
        topk = self.block_topk
        T = int(positions.numel())
        qsl = getattr(md, "query_start_loc_p", None)
        if qsl is not None and int(qsl.numel()) >= 2:
            bs = int(qsl.numel()) - 1
            lens = (qsl[1:].long() - qsl[:bs].long()).to(device=positions.device)
        else:
            bs, lens = int(load.numel()), None
        T_cols = T // bs
        if lens is None:
            valid = torch.ones(T, dtype=torch.bool, device=positions.device)
        else:
            cols = torch.arange(T, device=positions.device)
            valid = (cols % T_cols) < lens.repeat_interleave(T_cols)
        # ``load``/``store`` carry one slot per state row: the [groups, bs] form
        # repeats each request's slot over its rows, the lattice form already
        # is one slot per row (see ``_state_indices``).
        n_slots = int(load.numel())
        if n_slots == T:
            load_rows, store_rows = load, store
        elif n_slots * T_cols == T:
            load_rows, store_rows = load.repeat_interleave(T_cols), store.repeat_interleave(T_cols)
        else:
            raise NotImplementedError(f"QSA decode got {n_slots} state slots for {T} rows of {bs} requests")
        scratch = torch.full((T,), self.key_ring.shape[0] - 1, dtype=torch.long, device=positions.device)
        ring_write(self.key_ring, torch.where(valid, store_rows, scratch), positions, raw_k)
        blk = ring_block_rows(self.key_ring, torch.where(valid, load_rows, scratch), positions, raw_k, ratio)
        complete = valid & ((positions % ratio) == ratio - 1)
        cs, sn = cos_sin_at(positions - (ratio - 1))
        keys = compress_block_keys(blk, self.k_layernorm.weight, self.eps, cs, sn)
        # One fixed-shape write (HPU graphs replay the captured shapes): a row that completes a
        # block writes its compressed row; every other row writes the scratch row, the cache's
        # last row, which no request's block table reaches (qsa_side_caches). Writing a block's
        # row from a row that does not complete it would race the completing row's write.
        row_slots = compressed_row_slots(slot_mapping, ratio)
        scratch_row = torch.full_like(row_slots, cache.shape[0] - 1)
        cache.index_copy_(0, torch.where(complete, row_slots, scratch_row), keys.to(cache.dtype))
        MB = int(md.block_table.shape[1])
        n = MB * rpb
        bt = md.block_table.long()
        blk_idx = torch.arange(n, device=positions.device)
        idx = bt.gather(1, (blk_idx // rpb)[None, :].expand(T, -1)) * rpb + (blk_idx % rpb)[None, :]
        all_rows = cache.index_select(0, idx.reshape(-1)).view(T, n, D)
        scores = torch.relu(torch.einsum("thd,tnd->thn", q.float(), all_rows.float())).sum(1) / (D**0.5)
        blocks_sel = select_blocks(scores, visible_blocks(positions, ratio), topk)
        return expand_selection(blocks_sel, positions, ratio)


# Backwards-compatible alias for the checkpoint-carrying indexer (M0 name).
HpuQsaIndexerWeights = HpuQsaIndexer


class HpuQwen4ExpQSAAttention(Qwen4ExpQSAAttention):
    """HPU QSA owner: dense paged attention behind the ``QsaMode`` flag.

    Does not call the upstream ``__init__``.  It keeps the checkpoint-shape
    construction (config checks, head counts, gated QKV projection, output
    projection, RoPE, QK norms), drops the CUDA-only fused
    qk-norm/rope/gate path, and replaces the FlashAttention implementation,
    the side caches, the topk buffer and the self-registration with a
    standard ``Attention`` child registered as ``f"{prefix}.attn"``.  That
    child is the KV-cache owner: the platform selects ``HPUAttentionBackend``
    on HPU and the CPU backend in tests.  ``get_kv_cache_spec`` is inherited
    from the parent only for API compatibility; the registered ``Attention``
    child is the layer the engine collects.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.qsa_mode = QsaMode(envs.VLLM_HPU_FLASH_NEXT_QSA_MODE)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Qwen4Exp QSA requires a BF16 main KV cache")
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1):
            raise NotImplementedError("Qwen4Exp QSA does not support context parallelism")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(config, "dual_chunk_attention_config", None)
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # The fused qk-norm + rope + gate kernel is CUDA-only; the HPU
        # forward below is always the eager split path.
        self.use_fused_qk_norm_rope_gate = False

        # HPU tail: a standard Attention child owns the KV cache and
        # registers itself in the static forward context as
        # f"{prefix}.attn"; upstream registers the owner itself.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.indexer = HpuQsaIndexer(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

        # Sparse mode runs the indexer's selection through a dedicated
        # attention wrapper. On HPU that is ``HpuQsaSparseAttention`` (M2-T5c)
        # and a wrapper that cannot be built stops the model build: without it
        # the forward below would run dense attention, which is wrong beyond
        # about 2050 tokens of context. Off HPU (the CPU tests) it stays
        # ``None`` and the harness installs a stand-in.
        self.prefix = prefix
        self.sparse_attn = None
        if self.qsa_mode is QsaMode.SPARSE and platform_is_hpu():
            try:
                from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import HpuQsaSparseAttention
                self.sparse_attn = HpuQsaSparseAttention(self.attn)
            except Exception as exc:
                raise RuntimeError(
                    f"VLLM_HPU_FLASH_NEXT_QSA_MODE={int(QsaMode.SPARSE)} (sparse QSA) needs the HPU sparse "
                    f"attention wrapper, which could not be built for {prefix or 'the QSA layer'}: {exc!r}. "
                    "Dense attention is not a fallback for sparse mode.") from exc

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn.get_attn_backend()

    def _forward_metadata(self):
        """The layer's attention metadata from the forward context, dict-flattened (see the PLE layer)."""
        try:
            md = get_forward_context().attn_metadata
        except AssertionError:  # no active forward context (e.g. a bare unit test)
            return None
        if isinstance(md, dict):
            md = md.get(f"{self.prefix}.attn", md)
        return md

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors _hpu_qwen3next_attention_forward: the upstream forward
        # assumes 2-D tokens for the attention output while keeping 3-D gate
        # shapes, so everything runs flattened to 2-D and the caller's
        # layout is restored at the end.
        orig_shape = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        gate_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*gate_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*gate_shape, -1)
        gate = gate.reshape(*gate_shape, -1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

        q, k = self.rotary_emb(positions, q, k)

        md = self._forward_metadata()
        if (self.qsa_mode is QsaMode.SPARSE and self.indexer.compressed_key_cache is not None
                and self.sparse_attn is not None and md is not None):
            pos = indexer_positions(positions, bool(getattr(self.rotary_emb, "mrope_section", None)))
            cos_sin_at = lambda p: self.rotary_emb.cos_sin_cache[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
            selected = self.indexer.select(hidden_states.reshape(-1, self.hidden_size), pos, cos_sin_at, md)
            attn_output = self.sparse_attn(q, k, v, selected, pos, md)
        else:
            attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(-1, attn_output.shape[-1])
        attn_output = attn_output * torch.sigmoid(gate.view(-1, gate.shape[-1]))

        output, _ = self.o_proj(attn_output)
        return output.view(*orig_shape[:-1], output.shape[-1])


__all__ = [
    "HpuQsaIndexer",
    "HpuQsaIndexerWeights",
    "HpuQwen4ExpQSAAttention",
    "QsaMode",
    "indexer_positions",
    "platform_is_hpu",
]
