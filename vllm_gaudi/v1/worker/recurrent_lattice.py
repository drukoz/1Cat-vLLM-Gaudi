# SPDX-License-Identifier: Apache-2.0
"""Align-mode verification lattices for MTP on the recurrent state pool (M3-T10a).

Pure torch/int helpers, importable on CPU, so they can be unit-tested without an HPU.

Under align mode (prefix caching on) the request's recurrent state lives in the position
blocks of the block table and the ``k`` speculative scratch blocks the MambaManager keeps
right after them (``mamba_get_block_table_tensor``: "align: the last 1 + num_speculative_blocks
of each request"). Loading and storing differ for token 0: the committed state is read from the
block of the last computed token, but when ``context_len`` is a multiple of ``block_size`` that
block is complete and may be a prefix-cache checkpoint, so token 0's state must be written to the
block of token 0's position. The runner keeps only thin call sites (T10b).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["align_lattice", "align_lattices"]


def align_lattice(block_row: Sequence[int], context_len: int, num_scheduled: int, block_size: int, k: int,
                  null_block: int = 0) -> tuple[list[int], list[int]]:
    """(load, store) of length 1 + k. load[0] = block_row[(context_len - 1) // block_size] or null_block when
    context_len == 0; store[0] = block_row[context_len // block_size]; load[1 + j] = store[1 + j] = block_row[n_pos + j]
    with n_pos = ceil((context_len + num_scheduled) / block_size). ValueError when block_row is too short."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}.")
    if num_scheduled < 0:
        raise ValueError(f"num_scheduled must be non-negative, got {num_scheduled}.")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    n_pos = -(-(context_len + num_scheduled) // block_size)
    store_pos = context_len // block_size
    required = max(n_pos + k, store_pos + 1)
    if len(block_row) < required:
        raise ValueError(f"block row has {len(block_row)} blocks, needs at least {required} for context_len="
                         f"{context_len}, num_scheduled={num_scheduled}, block_size={block_size}, k={k}.")
    load_first = int(null_block) if context_len == 0 else int(block_row[(context_len - 1) // block_size])
    store = [int(block_row[store_pos])]
    load = [load_first]
    for j in range(k):
        scratch = int(block_row[n_pos + j])
        load.append(scratch)
        store.append(scratch)
    return load, store


def align_lattices(block_table_cpu: torch.Tensor, req_indices: Sequence[int], context_lens: Sequence[int],
                   num_scheduled: Sequence[int], block_size: int, k: int, padded_batch_size: int,
                   pad_block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 [padded_batch_size, 1 + k] load and store lattices; padded rows are pad_block_id."""
    if block_table_cpu.ndim != 2:
        raise ValueError(f"block_table_cpu must be a 2-D CPU tensor, got shape {tuple(block_table_cpu.shape)}.")
    if block_table_cpu.device.type != "cpu":
        raise ValueError(f"block_table_cpu must live on the CPU, got device {block_table_cpu.device}.")
    batch = len(req_indices)
    if len(context_lens) != batch or len(num_scheduled) != batch:
        raise ValueError("req_indices, context_lens and num_scheduled must have equal length, got "
                         f"{batch}, {len(context_lens)}, {len(num_scheduled)}.")
    if padded_batch_size < batch:
        raise ValueError(f"padded_batch_size={padded_batch_size} is below the real batch {batch}.")
    table = block_table_cpu.tolist()
    load = torch.full((padded_batch_size, 1 + k), int(pad_block_id), dtype=torch.int32)
    store = torch.full((padded_batch_size, 1 + k), int(pad_block_id), dtype=torch.int32)
    for i in range(batch):
        lat_load, lat_store = align_lattice(table[int(req_indices[i])], int(context_lens[i]), int(num_scheduled[i]),
                                            block_size, k)
        load[i] = torch.tensor(lat_load, dtype=torch.int32)
        store[i] = torch.tensor(lat_store, dtype=torch.int32)
    return load, store
