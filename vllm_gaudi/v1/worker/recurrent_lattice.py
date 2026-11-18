# SPDX-License-Identifier: Apache-2.0
"""Align-mode verification lattices for MTP on the recurrent state pool (M3-T10a, T10c).

Pure torch/int helpers, importable on CPU, so they can be unit-tested without an HPU.

Under align mode (prefix caching on) the request's recurrent state lives in the position
blocks of the block table and the ``k`` speculative scratch blocks the MambaManager keeps
right after them (``mamba_get_block_table_tensor``: "align: the last 1 + num_speculative_blocks
of each request"). Following upstream vLLM (``preprocess_mamba``) an align-mode step keeps the
running recurrent state in the block of the LAST SCHEDULED token, ``curr = cdiv(context_len +
num_scheduled, block_size) - 1``, so the load and store lattices are the same window
``row[curr : curr + 1 + k]``. Block-table rows are index-stable (RUK-2847), so a step that
crosses a block boundary just moves the window right by a block; the state the step starts
from is moved into ``row[curr]`` by the pre-step copy plan of ``recurrent_commit``. The runner
keeps only thin call sites (T10b, T10c-2).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["align_running_block", "align_lattice", "align_lattices"]


def align_running_block(context_len: int, num_scheduled: int, block_size: int) -> int:
    """cdiv(context_len + num_scheduled, block_size) - 1: the block of the last scheduled token, where an
    align-mode step keeps the running recurrent state (upstream vLLM preprocess_mamba).
    ValueError: block_size <= 0, context_len < 0, num_scheduled < 1."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}.")
    if num_scheduled < 1:
        raise ValueError(f"num_scheduled must be at least 1, got {num_scheduled}.")
    return -(-(context_len + num_scheduled) // block_size) - 1


def align_lattice(block_row: Sequence[int], context_len: int, num_scheduled: int, block_size: int, k: int,
                  null_block: int = 0) -> tuple[list[int], list[int]]:
    """(load, store), two equal lists block_row[c : c + 1 + k] with c = align_running_block(...).
    ValueError: k < 0; block_row shorter than c + 1 + k (message names the needed length); any lattice entry
    equal to null_block (message names the index)."""
    curr = align_running_block(context_len, num_scheduled, block_size)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}.")
    required = curr + 1 + k
    if len(block_row) < required:
        raise ValueError(f"block row has {len(block_row)} blocks, needs at least {required} for context_len="
                         f"{context_len}, num_scheduled={num_scheduled}, block_size={block_size}, k={k}.")
    lattice = [int(block_row[idx]) for idx in range(curr, curr + 1 + k)]
    for idx, slot in enumerate(lattice):
        if slot == int(null_block):
            raise ValueError(f"lattice entry {idx} (block row index {curr + idx}) is the null block "
                             f"{null_block} (context_len={context_len}, num_scheduled={num_scheduled}, "
                             f"block_size={block_size}, k={k}).")
    return lattice, list(lattice)


def align_lattices(block_table_cpu: torch.Tensor, req_indices: Sequence[int], context_lens: Sequence[int],
                   num_scheduled: Sequence[int], block_size: int, k: int, padded_batch_size: int,
                   pad_block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 [padded_batch_size, 1 + k] load and store lattices with load == store; padded rows are pad_block_id."""
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
