# SPDX-License-Identifier: Apache-2.0
"""min_tokens inside the fused greedy region (``VLLM_HPU_FUSED_GREEDY_LOGITS=1``): pure helpers.

The fused path (``HpuModelAdapter.select_compute_logits_and_greedy``) computes the LM head and the argmax in one
lazy region, without the sampler's separate graph, its two ``mark_step`` calls and the per-step sampling
metadata. A request with ``min_tokens`` used to keep the general sampler path. Here its rule runs in the fused
region instead: until a request has ``min_tokens`` output tokens, every id of ``all_stop_token_ids`` gets
``-inf`` before the argmax (vLLM's ``MinTokensLogitsProcessor``, ``vllm/v1/sample/logits_processor/builtin.py``),
with fixed shapes and numeric ops only (a gather, a where and a scatter of numbers; no ``index_put_``, no
boolean indexing, which have no HPU kernel):

* ``stop_ids`` int64 ``[rows, width]``: a masking row lists its stop ids, the last one repeated up to ``width``
  (masking an id twice changes nothing); every other row (a request past ``min_tokens``, a padding row) is all
  ``0``;
* ``stop_rows`` bool ``[rows]``: whether a row masks. A row that does not mask writes its own column-0 logit
  back to column 0.

The ``width`` is fixed per server (``STOP_ID_WIDTH``), so the region has one shape per batch bucket.

Integer ids and the card (rule 9, 2026-11-20: ``torch.topk`` ids narrowed to int32 in the graph that made them
came back wrong): the stop ids are made on the host as int64, the dtype ``gather`` and ``scatter`` take, and are
used as they are, so the region has no integer cast of its own; the runner reads the arg max ids back as int64
and narrows them on the host unless async scheduling needs them on the device
(``HpuModelAdapter.select_compute_logits_and_greedy``). ``tools/flash_next/engine_step/hpu_sampling_ops.py``
checks every such pattern on one card against the CPU.
"""

from typing import Optional, Sequence

import torch

STOP_ID_WIDTH = 8  # stop ids masked per row; a request with more keeps the general sampler path


def min_tokens_stop_ids(stop_token_ids: Sequence[Optional[Sequence[int]]], rows: int,
                        width: int = STOP_ID_WIDTH) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """``(stop_ids, stop_rows)`` for the fused region, or ``None`` when no row masks anything.

    ``stop_token_ids[r]`` is the ids row ``r`` must not produce this step (``None`` or empty: none); rows past
    ``len(stop_token_ids)`` are padding. Raises ``ValueError`` when a row has more than ``width`` ids or more
    rows are given than ``rows``.
    """
    if len(stop_token_ids) > rows:
        raise ValueError(f"{len(stop_token_ids)} rows of stop ids for {rows} rows")
    ids = torch.zeros((rows, width), dtype=torch.int64)
    active = torch.zeros((rows, ), dtype=torch.bool)
    for row, stops in enumerate(stop_token_ids):
        if not stops:
            continue
        stops = sorted(int(token) for token in stops)
        if len(stops) > width:
            raise ValueError(f"row {row} has {len(stops)} stop ids, the fused region masks {width}")
        ids[row] = torch.tensor(stops + [stops[-1]] * (width - len(stops)), dtype=torch.int64)
        active[row] = True
    if not bool(active.any()):
        return None
    return ids, active


def mask_stop_tokens(logits: torch.Tensor, stop_ids: torch.Tensor, stop_rows: torch.Tensor) -> torch.Tensor:
    """``logits`` with ``-inf`` at ``stop_ids`` of the rows where ``stop_rows`` is set (see the module doc).

    ``stop_ids`` is int64 as ``min_tokens_stop_ids`` makes it (any other integer dtype is widened here, which
    on a card is a cast inside the graph: see the module doc). ``stop_rows`` may be bool or int32 (the runner
    sends int32: no bool buffer goes to the device)."""
    index = stop_ids if stop_ids.dtype == torch.int64 else stop_ids.to(torch.int64)
    current = logits.gather(1, index)
    masked = torch.where(stop_rows.view(-1, 1) != 0, torch.full_like(current, float("-inf")), current)
    return logits.scatter(1, index, masked)


__all__ = ["STOP_ID_WIDTH", "mask_stop_tokens", "min_tokens_stop_ids"]
