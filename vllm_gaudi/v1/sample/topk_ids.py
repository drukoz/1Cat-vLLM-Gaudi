# SPDX-License-Identifier: Apache-2.0
"""Token ids of a ``torch.topk`` that the host reads as data, in the form the card gets right.

Card evidence (engine-step job 00093, one card, ``[rows, 248320]``, rows 1 to 32; collectives job 00108):

* ``topk(...).indices.to(torch.int32)`` in the graph that made them: every second id of the true table on the
  first run of a new graph (card rule 9); this is what vLLM's ``gather_logprobs`` does, and a ``logprobs=5``
  request then killed the server's output handler with ``OverflowError`` (spec-lookup job 00086).
* ``topk(...).indices`` read back as int64, or narrowed in a second graph: right in one graph sequence, but in
  another (the same process had run ``sort`` and ``topk(64).values`` graphs first) every read returned the top-k
  VALUES instead of the indices.
* the indices consumed inside the graph, as the index of a ``gather`` into an fp32 id table, and that fp32 id
  narrowed to int32: equal to the CPU at k = 20 and 64, widths 31,040 and 248,320, rows 1 to 128, lazy and
  replayed, tied and untied (00108 ``topk_gather_fp32@k``, ``vp_topk@k``). The fp32 id is exact below 2**24.

So: never read top-k indices back as data; read ids out of an fp32 id table with the indices as top-k made them.
This module imports only ``torch``.
"""

import torch

FP32_EXACT_IDS = 1 << 24  # every integer below it is exact in fp32


def id_table(width: int, rows: int, device) -> torch.Tensor:
    """The ids ``0 .. width - 1`` as fp32 ``[rows, width]`` (a broadcast of one row, no copy)."""
    if not 0 < width <= FP32_EXACT_IDS:
        raise ValueError(f"fp32 ids are exact for widths up to {FP32_EXACT_IDS}, got {width}")
    return torch.arange(width, dtype=torch.float32, device=device).expand(rows, width)


def ids_from_topk_index(index: torch.Tensor, width: int) -> torch.Tensor:
    """int32 ids ``[rows, k]`` of the top-k ``index`` ``[rows, k]`` over rows of ``width`` columns.

    The index is consumed inside the graph (a ``gather`` into the fp32 id table) and the fp32 ids are narrowed:
    no top-k index is cast or read back as data.
    """
    return torch.gather(id_table(width, index.shape[0], index.device), -1, index).to(torch.int32)


def topk_with_ids(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(top-k values [rows, k] largest first, their int32 ids [rows, k])`` of a 2-D tensor, last dimension."""
    top, index = torch.topk(values, k, dim=-1)
    return top, ids_from_topk_index(index, values.shape[-1])


__all__ = ["FP32_EXACT_IDS", "id_table", "ids_from_topk_index", "topk_with_ids"]
