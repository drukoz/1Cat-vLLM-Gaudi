# SPDX-License-Identifier: Apache-2.0
"""Restructured sparse-QSA decode data flow for Flash-Next (``VLLM_HPU_QSA_FUSED``).

The torch path (mode 0, ``HpuQsaIndexer._select_decode``) scores every query row against **all**
``N = MB * rpb`` compressed keys its block table can address: it builds an int64 ``[T, N]`` index,
gathers ``[T, N, D]`` keys (8 MiB per row and layer at ``--max-model-len 131072``, whatever the
row's context, because ``MB`` is fixed per engine), widens them to fp32 and scores. The ``1 + k``
rows of one request in a verification step gather the same keys ``1 + k`` times, and all TP ranks
repeat the same work. Mode 1 (``QsaFusedMode.TORCH``) changes the data flow, not the arithmetic:

* **request-major keys (I14).** One block-table row per request (the row of its last real token,
  ``b * T_cols + clamp(len_b - 1, 0)``, the longest causal prefix of the request) and one gather of
  whole kernel blocks, ``cache.view(-1, rpb, D).index_select(0, table)``: ``[bs, N, D]`` for the
  ``T = bs * T_cols`` rows, scored as one ``[bs, T_cols * H, D] @ [bs, D, N]`` product. Exact: a
  row's visible columns ``n < (p + 1) // ratio`` lie on pages its own table shares with the longest
  one; every other column is masked before the top-k (``select_blocks``). No ``[T, N]`` index.
* **by request across the TP ranks (I18, ``VLLM_HPU_QSA_INDEXER_SHARD=1``).** Rank ``r`` scores
  requests ``[r * bs / tp, (r + 1) * bs / tp)`` only; one all-gather of the int32 ``[rows, topk]``
  block selections per layer gives every rank the same selection (the one its owner computed). The
  cache writes stay replicated (every rank keeps complete caches; the request-to-rank map follows
  the batch order of each step). A bucket whose ``bs`` is not a multiple of ``tp`` stays replicated.
* **grouped attention (I21, G5-1).** ``sparse_decode_attention_grouped`` scores the ``G`` query
  heads of a KV head as the ``M`` dimension of one GEMM against the gathered rows instead of
  ``repeat_interleave``-ing K and V ``G`` times (``qsa_hpu_attention``).

The score of a row is the same fp32 sum of the same products as in mode 0; only the GEMM that forms
it differs (``[T_cols * H]`` rows per request instead of ``[H]`` per row). Selections are therefore
equal to mode 0 except where two scores at the cut are equal within the GEMM's summation-order
rounding (``tests/unit_tests/models/flash_next/test_qsa_fused_torch.py``).

Mode 2 (``QsaFusedMode.KERNELS``) adds the TPC kernels of ``flashinfer_gaudi.qsa_fused`` on top of
this data flow; see ``docs/design/flash-next/qsa-fused.md``.
"""

from __future__ import annotations

import enum
import math
from typing import NamedTuple

import torch

import vllm_gaudi.envs as envs
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import select_blocks, visible_blocks


class QsaFusedMode(enum.IntEnum):
    """``VLLM_HPU_QSA_FUSED`` (next free: 4)."""

    OFF = 0  # the torch path of today
    TORCH = 1  # the restructured torch data flow (this module)
    KERNELS = 2  # TORCH plus the TPC kernels (flashinfer_gaudi.qsa_fused), each with its torch fallback
    KERNELS_TOPK = 3  # KERNELS plus the top-k kernels (threshold + select): ties at the cut to the lowest block


class QsaShardMode(enum.IntEnum):
    """``VLLM_HPU_QSA_INDEXER_SHARD`` (next free: 2)."""

    REPLICATED = 0
    BY_REQUEST = 1


def qsa_fused_mode() -> QsaFusedMode:
    """The configured mode; any value that is not a member is ``OFF``."""
    try:
        return QsaFusedMode(int(envs.VLLM_HPU_QSA_FUSED))
    except ValueError:
        return QsaFusedMode.OFF


def qsa_shard_mode() -> QsaShardMode:
    """The configured indexer sharding; any value that is not a member is ``REPLICATED``."""
    try:
        return QsaShardMode(int(envs.VLLM_HPU_QSA_INDEXER_SHARD))
    except ValueError:
        return QsaShardMode.REPLICATED


# What the QSA layers of one forward share: keyed by the step's attention metadata object (one per forward; the
# runner's trimmed namedtuple cannot carry attributes, so the cache holds the object until the next step's).
_STEP_MD: object | None = None
_STEP_CACHE: dict = {}


def step_cached(md: object, key: tuple, build):
    """``build()`` once per step metadata ``md`` and ``key``: the layer-independent glue of a decode step.

    Every QSA layer of a forward gets the same ``md`` and the same positions, so the row validity, the request
    tables, the int32 copies the kernels take and the visible-block counts are built by the first layer and
    reused by the other eleven (fewer graph nodes; the values are the same). A new ``md`` (the next step, a draft
    step, a warmup run) starts a new cache. Modes 1 and 2 only.
    """
    global _STEP_MD, _STEP_CACHE
    if md is not _STEP_MD:
        _STEP_MD, _STEP_CACHE = md, {}
    if key not in _STEP_CACHE:
        _STEP_CACHE[key] = build()
    return _STEP_CACHE[key]


def request_table_rows(lens: torch.Tensor | None, bs: int, t_cols: int, device: torch.device) -> torch.Tensor:
    """int32 ``[bs]``: the block-table row of each request's last real token (its longest prefix).

    Rows are the padded decode layout, ``T_cols`` per request. ``lens`` is the real length of each
    request (``query_start_loc_p`` differences; ``None`` = every row real). A request without real
    rows (bucket padding, ``len = 0``) takes its first row, whose table is the runner's pad row.
    """
    base = torch.arange(bs, dtype=torch.int32, device=device) * t_cols
    if lens is None:
        return base + (t_cols - 1)
    return base + (lens.to(device=device, dtype=torch.int32) - 1).clamp(min=0, max=t_cols - 1)


def score_request_major(q: torch.Tensor, cache: torch.Tensor, table: torch.Tensor, rpb: int,
                        t_cols: int) -> torch.Tensor:
    """fp32 ``[bs * T_cols, N]`` block scores: ``sum_h relu(q_h . k_n) / sqrt(D)``.

    ``q`` ``[bs * T_cols, H, D]`` (bf16 or fp32, the indexer queries after norm and RoPE), ``cache``
    the compressed key cache ``[C, D]`` (``C`` a multiple of ``rpb``), ``table`` int32 ``[bs, MB]``
    kernel block ids, one row per request. One gather of ``MB`` whole kernel blocks per request (the
    form the prefill branch uses, ``qsa.py``), one batched product for all rows of the request.
    """
    bs, mb = int(table.shape[0]), int(table.shape[1])
    heads, dim = int(q.shape[1]), int(q.shape[2])
    n = mb * rpb
    keys = cache.view(-1, rpb, dim).index_select(0, table.reshape(-1)).view(bs, n, dim)
    prod = torch.matmul(q.reshape(bs, t_cols * heads, dim).float(), keys.float().transpose(1, 2))
    return torch.relu(prod).view(bs * t_cols, heads, n).sum(1) / math.sqrt(dim)


def tp_rank_and_size() -> tuple[int, int]:
    """``(rank, world)`` of the tensor-parallel group; ``(0, 1)`` when it is not initialized."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        return int(get_tensor_model_parallel_rank()), int(get_tensor_model_parallel_world_size())
    except (AssertionError, RuntimeError, ImportError):
        return 0, 1


def shard_requests(bs: int, shard: QsaShardMode) -> tuple[int, int, int]:
    """``(rank, world, per_rank)`` when the indexer of a ``bs``-request bucket is sharded, else ``(0, 1, bs)``.

    A bucket shards only when ``bs`` is a multiple of the TP size and the TP size is above 1: every
    rank then gives the all-gather the same ``[per_rank * T_cols, topk]`` shape.
    """
    if shard is not QsaShardMode.BY_REQUEST:
        return 0, 1, bs
    rank, world = tp_rank_and_size()
    if world <= 1 or bs % world:
        return 0, 1, bs
    return rank, world, bs // world


def all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    """All-gather along dim 0 over the TP group (rank-major order, as ``tensor_model_parallel_all_gather``)."""
    from vllm.distributed import tensor_model_parallel_all_gather
    return tensor_model_parallel_all_gather(x, dim=0)


def topk_kernels(scores: torch.Tensor, positions32: torch.Tensor, ratio: int, topk: int) -> torch.Tensor:
    """Mode 3: ``select_blocks`` as the two TPC kernels (threshold, select): ascending ids, ties to the lowest block."""
    from flashinfer_gaudi.qsa_fused import qsa_index_select, qsa_index_threshold
    return qsa_index_select(qsa_index_threshold(owned(scores), positions32, ratio, topk), topk)


def select_blocks_request_major(q: torch.Tensor, positions: torch.Tensor, block_table: torch.Tensor,
                                lens: torch.Tensor | None, bs: int, t_cols: int, cache: torch.Tensor, rpb: int,
                                ratio: int, topk: int, shard: QsaShardMode, md: object | None = None,
                                use_topk_kernels: bool = False) -> torch.Tensor:
    """int32 ``[T, topk]`` selected block indices (``-1`` padded) of the ``T = bs * T_cols`` decode rows.

    The scoring and top-k of ``_select_decode`` in the request-major form (``score_request_major``),
    on this rank's share of the requests when the bucket shards (``shard_requests``), followed by the
    all-gather that gives every rank the selection of every request.
    """
    T = int(positions.numel())
    heads, dim = int(q.shape[1]), int(q.shape[2])

    def shared(key, build):
        return build() if md is None else step_cached(md, key, build)

    def request_table():
        if t_cols == 1:  # one row per request: its own table row is the request's table
            return block_table.to(torch.int32)
        return block_table.to(torch.int32).index_select(0, request_table_rows(lens, bs, t_cols, positions.device))

    n_keys = int(block_table.shape[1]) * rpb
    use_topk_kernels = use_topk_kernels and n_keys % 128 == 0  # the threshold kernel's gate

    rank, world, per = shard_requests(bs, shard)
    if world == 1:
        table = shared(("request_table", bs, t_cols), request_table)
        scores = score_request_major(q, cache, table, rpb, t_cols)
        if use_topk_kernels:
            positions32 = shared(("positions32", T), lambda: positions.to(torch.int32, copy=True))
            return topk_kernels(scores, positions32, ratio, topk)
        visible = shared(("visible", T, ratio), lambda: visible_blocks(positions, ratio))
        return select_blocks(scores, visible, topk)

    def my_share():
        mine = torch.arange(rank * per, (rank + 1) * per, dtype=torch.int32, device=positions.device)
        pos_mine = positions.reshape(bs, t_cols).index_select(0, mine).reshape(-1)
        return mine, request_table().index_select(0, mine), visible_blocks(pos_mine, ratio), pos_mine.to(torch.int32)

    mine, table_mine, visible_mine, pos32_mine = shared(("share", bs, t_cols, rank, world, ratio), my_share)
    q_mine = q.reshape(bs, t_cols, heads, dim).index_select(0, mine).reshape(per * t_cols, heads, dim)
    scores = score_request_major(q_mine, cache, table_mine, rpb, t_cols)
    if use_topk_kernels:
        local = topk_kernels(scores, owned(pos32_mine), ratio, topk)
    else:
        local = select_blocks(scores, visible_mine, topk)  # [per * T_cols, topk]
    gathered = all_gather_rows(local)
    if int(gathered.shape[0]) != T:
        raise RuntimeError(f"QSA sharded indexer gathered {int(gathered.shape[0])} rows for {T}")
    return gathered


# ----------------------------------------------------------------------------------- mode 2: the TPC kernels
class QsaDecodeSelection(NamedTuple):
    """What the mode-2 indexer hands the decode attention: the attended K/V cache slots and their 0/1 flags.

    Both int32 ``[T, W]`` (``W = block_topk * ratio + ratio - 1``), the output of ``qsa_index_emit``: the slots
    ``decode_selected_slots`` gives the expanded selection, and ``valid`` as ``(selected >= 0) |
    rows_without_selection(selected)``.
    """

    slots: torch.Tensor
    valid: torch.Tensor


# Device types whose qsa_fused ops count as kernels (the CPU tests add "cpu" with the references registered).
KERNEL_DEVICE_TYPES = ["hpu"]


def owned(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself when it owns its storage and is contiguous, else a copy (card rule 1: no view into a custom op)."""
    return t if (t._base is None and t.is_contiguous() and t.storage_offset() == 0) else t.contiguous().clone()


def rows_table(positions: torch.Tensor, slot_mapping: torch.Tensor, load_rows: torch.Tensor, store_rows: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    """int32 ``[T, 4]`` row table of a decode step: position, attention slot, ring slot read, ring slot written.

    A padding row (``valid`` False) gets ``-1`` as both ring slots, so ``qsa_index_prepare`` writes nothing for it.
    """
    neg = torch.full_like(load_rows, -1)
    return torch.stack((positions.to(torch.int32), slot_mapping.to(torch.int32),
                        torch.where(valid, load_rows, neg).to(torch.int32),
                        torch.where(valid, store_rows, neg).to(torch.int32)), dim=1)


def kernel_ops_reason(device: torch.device) -> str | None:
    """Why the qsa_fused ops cannot run on ``device`` in this process, or None when they can."""
    if device.type not in KERNEL_DEVICE_TYPES:
        return f"device {device.type} has no QSA fused kernel"
    if device.type == "hpu":
        try:
            from flashinfer_gaudi._native import qsa_fused_extension_errors, qsa_fused_extension_loaded
        except ImportError as exc:  # pragma: no cover - flashinfer_gaudi is part of this repository
            return f"flashinfer_gaudi is not importable: {exc}"
        if qsa_fused_extension_loaded() is not True:
            errors = "; ".join(qsa_fused_extension_errors()) or "the loader did not run"
            return f"the qsa_fused_ops extension is not loaded ({errors})"
    try:
        from flashinfer_gaudi.qsa_fused import qsa_fused_missing_ops
    except ImportError as exc:  # pragma: no cover
        return f"flashinfer_gaudi.qsa_fused is not importable: {exc}"
    missing = qsa_fused_missing_ops(device.type)
    if missing:
        return f"no {device.type} kernel for {', '.join(missing)}"
    return None


def kernel_shape_reason(*, dim: int, heads: int, ratio: int, ring_rows: int, block_size: int,
                        cos_sin: torch.Tensor | None, cache: torch.Tensor | None, ring: torch.Tensor | None) -> str | None:
    """Why this layer's geometry is outside the kernels' gates (``csrc/qsa_fused/qsa_fused_params.h``), or None."""
    def power_of_two(v: int) -> bool:
        return v > 0 and (v & (v - 1)) == 0

    if cos_sin is None or cache is None or ring is None:
        return "the side caches or the RoPE table are not bound"
    if not 1 <= dim <= 128:
        return f"indexer head dim {dim} is not in [1, 128]"
    if not 1 <= heads <= 8:
        return f"indexer heads {heads} is not in [1, 8]"
    if ratio not in (2, 4, 8):
        return f"compress ratio {ratio} is not 2, 4 or 8"
    if not ratio <= ring_rows <= 64:
        return f"ring rows {ring_rows} is not in [ratio, 64]"
    if not (power_of_two(block_size) and block_size >= ratio):
        return f"block size {block_size} is not a power of two >= ratio"
    if cos_sin.dtype != torch.bfloat16 or cos_sin.dim() != 2 or cos_sin.shape[1] % 2 or cos_sin.shape[1] > dim or \
            not power_of_two(cos_sin.shape[1] // 2):
        return f"the RoPE table {tuple(cos_sin.shape)} {cos_sin.dtype} is not bf16 [P, 2 * R2], R2 a power of two"
    if cache.dtype != torch.bfloat16 or ring.dtype != torch.bfloat16:
        return "the side caches are not bf16"
    return None


__all__ = [
    "KERNEL_DEVICE_TYPES",
    "QsaDecodeSelection",
    "QsaFusedMode",
    "QsaShardMode",
    "all_gather_rows",
    "kernel_ops_reason",
    "kernel_shape_reason",
    "owned",
    "qsa_fused_mode",
    "qsa_shard_mode",
    "request_table_rows",
    "rows_table",
    "score_request_major",
    "select_blocks_request_major",
    "shard_requests",
    "step_cached",
    "topk_kernels",
    "tp_rank_and_size",
]
