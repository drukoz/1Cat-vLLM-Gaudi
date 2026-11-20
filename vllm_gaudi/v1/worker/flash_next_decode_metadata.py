# SPDX-License-Identifier: Apache-2.0
"""Flash-Next decode metadata for sparse QSA (``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1``): pure CPU helpers.

In sparse QSA mode (``VLLM_HPU_FLASH_NEXT_QSA_MODE=1``) a Flash-Next decode or verification step reads, of the
paged-attention metadata, only ``block_table``, ``slot_mapping`` and ``block_size``:

* the QSA indexer's decode selection gathers the compressed keys through ``md.block_table``
  (``models/qwen4_exp/qsa.py`` ``HpuQsaIndexer._select_decode``) and writes through ``slot_mapping``;
* the sparse attention writes K/V at ``slot_mapping`` and gathers the selected rows through ``md.block_table``
  (``models/qwen4_exp/qsa_hpu_attention.py`` ``HpuQsaSparseAttention.forward``, decode branch);
* ``block_list`` is read only by the prompt branches of both, and the GDN and PLE layers use their own state
  indices.

So the runner's dense paged-attention buffers of a decode step (``block_list``, ``block_groups``,
``block_usage`` from ``get_habana_paged_attn_buffers``, and the ``block_mapping`` one-hot and ``attn_bias``
that ``HPUAttentionMetadataProcessor`` builds from them in the graph) are not read by anything. With the flag
on they are not built: the decode metadata carries ``None`` for them, the block table is one row per request
built with numpy (repeated over the request's verification rows) instead of per-row Python lists, and the
decode bucket depends on the batch size only (the block buckets of ``block_list`` were the only input whose
shape followed the context length).

One block-table row per request is exact for a verification step: a row of request ``r`` at position ``p``
sees only the blocks below ``visible_blocks(p)`` (the indexer masks the other columns to ``-inf``,
``qsa_sparse_ops.select_blocks``) and attends only to selected positions ``<= p``, so the pages of the
request's later rows that the full row exposes are never selected (``tests/unit_tests/models/flash_next/
test_decode_metadata_sparse.py`` checks the tokens and logits against the dense-buffer path).

This module imports only ``torch`` and ``numpy`` (the runner cannot be imported on a machine without an HPU).
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]


def _as_numpy(table: TensorLike) -> np.ndarray:
    if isinstance(table, torch.Tensor):
        return table.detach().cpu().numpy()
    return np.asarray(table)


def request_block_table(block_table_cpu: TensorLike, num_blocks: Sequence[int], rows_per_request: int,
                        padded_rows: int, max_blocks: int, pad_block_id: int) -> torch.Tensor:
    """int32 ``[padded_rows, max_blocks]``: request ``r``'s first ``num_blocks[r]`` kernel blocks, then
    ``pad_block_id``, on its ``rows_per_request`` consecutive rows; rows past the requests are all
    ``pad_block_id``.

    ``block_table_cpu`` is the input batch's kernel-block table (row ``r`` = the ``r``-th request of the step).
    For one row per request this equals ``build_block_table([table[r, :num_blocks[r]] ...], ...)`` of
    ``flash_next_inputs``. Raises ``ValueError`` if a request needs more than ``max_blocks`` blocks or more
    than the table holds.
    """
    nb = np.asarray(num_blocks, dtype=np.int64).reshape(-1)
    n = int(nb.shape[0])
    rows_per_request = int(rows_per_request)
    out = np.full((int(padded_rows), int(max_blocks)), int(pad_block_id), dtype=np.int32)
    if n == 0:
        return torch.from_numpy(out)
    if n * rows_per_request > out.shape[0]:
        raise ValueError(f"{n} requests x {rows_per_request} rows do not fit {out.shape[0]} rows")
    source = _as_numpy(block_table_cpu)
    if source.ndim == 1:
        source = source.reshape(1, -1)
    if int(nb.max()) > max_blocks:
        raise ValueError(f"a request needs {int(nb.max())} blocks, max_blocks={max_blocks}")
    if int(nb.max()) > source.shape[1]:
        raise ValueError(f"a request needs {int(nb.max())} blocks, the block table holds {source.shape[1]}")
    width = min(int(source.shape[1]), int(max_blocks))
    table = np.full((n, int(max_blocks)), int(pad_block_id), dtype=np.int32)
    table[:, :width] = source[:n, :width]
    table[np.arange(max_blocks)[None, :] >= nb[:, None]] = int(pad_block_id)
    if rows_per_request > 1:
        table = np.repeat(table, rows_per_request, axis=0)
    out[:n * rows_per_request] = table
    return torch.from_numpy(out)


def fill_pad_slots(slot_mapping: torch.Tensor, query_lens: Sequence[int], pad_slot_base: int, block_size: int) -> None:
    """In place: the entries ``slot_mapping[r, q_r:]`` (``q_r = query_lens[r]``) get the pad block's slots
    ``pad_slot_base + (k % block_size)``, ``k`` counting those entries in row-major order.

    The vectorised form of the runner's per-request ``apply_`` over an ``itertools.cycle`` of the pad block's
    slots (``_create_decode_input_data``): the same values in the same order. ``slot_mapping`` is a contiguous
    2-D CPU tensor ``[rows, T]``; ``query_lens`` has one entry per row.
    """
    rows, width = slot_mapping.shape
    lens = np.asarray(query_lens, dtype=np.int64).reshape(-1)
    if lens.shape[0] != rows:
        raise ValueError(f"{lens.shape[0]} query lengths for {rows} slot rows")
    mask = np.arange(width)[None, :] >= lens[:, None]
    count = int(mask.sum())
    if count == 0:
        return
    values = torch.from_numpy(int(pad_slot_base) + np.arange(count, dtype=np.int64) % int(block_size))
    slot_mapping[torch.from_numpy(mask)] = values.to(slot_mapping.dtype)


def draft_step_block_table_and_slots(block_table_cpu: TensorLike, positions: torch.Tensor, batch_size: int,
                                     block_size: int, max_model_len: int, max_blocks: int, pad_block_id: int,
                                     pad_slot_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The MTP drafter's per-step ``(block_table, slot_mapping)`` without per-row Python lists.

    The same values as ``HpuFlashNextMtpProposer.prepare_attn_metadata`` builds with the defragmenter off:
    ``positions`` (CPU ``[batch_size]``) are the draft tokens' positions, ``block_table_cpu`` holds the
    ``num_seq`` real requests (``num_seq <= batch_size``). Returns the int32 table ``[batch_size,
    max_blocks]`` (row ``i < num_seq``: its first ``ceil((p_i + 1) / block_size)`` blocks, then pad; the other
    rows all pad) and the int64 slots ``[batch_size, 1]`` (row ``i < num_seq``: its block at
    ``clamp(p_i) // block_size`` times ``block_size`` plus ``clamp(p_i) % block_size`` with positions at or
    past ``max_model_len`` clamped to 0; row ``i >= num_seq``: ``pad_slot_id + (i - num_seq) % block_size``).
    """
    source = _as_numpy(block_table_cpu)
    if source.ndim == 1:
        source = source.reshape(1, -1)
    num_seq = int(source.shape[0])
    exceeds = positions >= max_model_len
    clamped = torch.where(exceeds, 0, positions).view(-1, 1)
    num_blocks = torch.ceil((positions + 1) / block_size).int()[:num_seq].numpy()
    table = request_block_table(source, num_blocks, 1, batch_size, max_blocks, pad_block_id)
    block_numbers = (clamped // block_size).to(torch.int64)[:num_seq]
    block_ids = torch.full((batch_size, 1), int(pad_block_id), dtype=torch.int32)
    block_ids[:num_seq] = torch.from_numpy(np.ascontiguousarray(source)).gather(dim=1, index=block_numbers)
    slot_mapping = block_ids * block_size + clamped % block_size
    if batch_size > num_seq:
        slot_mapping[num_seq:, 0] = (int(pad_slot_id) +
                                     torch.arange(batch_size - num_seq, dtype=slot_mapping.dtype) % block_size)
    return table, slot_mapping.to(torch.int64)


def compact_state_indices(base_slots: np.ndarray, group_offset: int, max_reqs: int, slots_per_req: int,
                          state_width: int, checkpoint_offsets: Optional[Sequence[int]] = None) -> np.ndarray:
    """int32 ``[len(base_slots), state_width]`` compact-GDN state slots of one cache group: the vectorised loop
    body of ``HPUModelRunner.prepare_mamba_state_idxs``.

    A request's first slot is ``(group_offset * max_reqs + base_slot) * slots_per_req + 1``; row ``i`` holds that
    slot plus ``0 .. state_width - 1``, or (with ``checkpoint_offsets``, ``state_width == 1``) plus its offset
    clamped to ``[0, slots_per_req - 1]``.
    """
    first = (int(group_offset) * int(max_reqs) + np.asarray(base_slots, dtype=np.int64)) * int(slots_per_req) + 1
    if checkpoint_offsets is None:
        return (first[:, None] + np.arange(int(state_width), dtype=np.int64)[None, :]).astype(np.int32)
    offsets = np.clip(np.asarray(checkpoint_offsets, dtype=np.int64), 0, int(slots_per_req) - 1)
    return (first + offsets)[:, None].astype(np.int32)


def sparse_decode_metadata_refusal(*, flag: int, needs_block_table: bool, qsa_sparse: bool, framework_layout: bool,
                                   sliding_window: bool, chunked_attention: bool, defragmenter_enabled: bool,
                                   speculative_method: Optional[str], dense_qsa_layers: int) -> Optional[str]:
    """``None`` when the sparse-QSA decode metadata may be used, else the reason it is not (logged once)."""
    if flag != 1:
        return f"VLLM_HPU_FLASH_NEXT_DECODE_METADATA={flag} (only 1 selects it)"
    if not needs_block_table:
        return "the model does not declare requires_block_table (not Flash-Next)"
    if not qsa_sparse:
        return "VLLM_HPU_FLASH_NEXT_QSA_MODE is not 1 (dense QSA reads the paged-attention buffers)"
    if dense_qsa_layers:
        return f"{dense_qsa_layers} QSA layer(s) would run dense attention (no sparse wrapper or side cache)"
    if framework_layout:
        return "the framework KV-cache layout builds its own decode metadata"
    if sliding_window or chunked_attention:
        return "the model has sliding-window or chunked attention"
    if defragmenter_enabled:
        return "the KV-cache defragmenter is on (block ids need resolving)"
    if speculative_method not in (None, "mtp", "ngram"):
        return f"speculative method {speculative_method!r} builds its own decode metadata"
    return None


__all__ = [
    "compact_state_indices",
    "draft_step_block_table_and_slots",
    "fill_pad_slots",
    "request_block_table",
    "sparse_decode_metadata_refusal",
]
