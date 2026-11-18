# SPDX-License-Identifier: Apache-2.0
"""Boundary commits of accepted verification tokens (M3-T10b).

Pure torch/int helpers, importable on CPU, so they can be unit-tested without an HPU.

Under align mode the verification writes the state after candidate ``t`` into the store
lattice slot ``store[i, t]`` (``recurrent_lattice.align_lattice``), while the accepted
checkpoint a request must expose for prefix-cache hits lives in the POSITION blocks of
its block table. An accepted token at position ``context_len + t`` completes the block
holding that position when ``(context_len + t + 1) % block_size == 0``; token 0 is the
store lattice's own job (``store[0]`` is the block of token 0's position when the context
is block-aligned), so the runner copies the states of the COMMITTED tokens ``1 <= t <
accepted`` that complete a block from the store lattice into that block's slot. The
copies run at sample time, once the accepted length of the step is known.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["boundary_commits", "commit_ssm", "commit_conv"]


def boundary_commits(context_lens: Sequence[int],
                     accepted: Sequence[int],
                     block_size: int,
                     lattices_store: torch.Tensor,
                     block_rows: Sequence[Sequence[int]],
                     lattices_load: torch.Tensor | None = None) -> list[tuple[int, int, int, int]]:
    """(request, t, src_slot, dst_slot) for accepted tokens ``1 <= t < accepted[i]`` that complete a block.

    ``lattices_store`` is the int32 ``[batch, 1 + k]`` store lattice of the step (row ``i``
    aligned with ``context_lens[i]``/``accepted[i]``), ``block_rows[i]`` the block-table row
    of request ``i`` (the same rows the lattices were built from) and ``lattices_load`` an
    optional load lattice checked for a matching shape. ``src_slot`` is the store-lattice
    slot the kernel wrote the state after token ``t`` into; ``dst_slot`` is the block-table
    entry of the block that token ``t`` completes.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    batch = len(context_lens)
    if len(accepted) != batch or len(block_rows) != batch:
        raise ValueError("context_lens, accepted and block_rows must have equal length, got "
                         f"{batch}, {len(accepted)}, {len(block_rows)}.")
    if lattices_store.ndim != 2 or lattices_store.shape[0] < batch:
        raise ValueError(f"lattices_store must be [batch, 1 + k], got shape {tuple(lattices_store.shape)}.")
    if lattices_load is not None and tuple(lattices_load.shape) != tuple(lattices_store.shape):
        raise ValueError("lattices_load and lattices_store must have equal shapes, got "
                         f"{tuple(lattices_load.shape)} and {tuple(lattices_store.shape)}.")
    store = lattices_store.tolist()
    pairs: list[tuple[int, int, int, int]] = []
    for i in range(batch):
        ctx = int(context_lens[i])
        if ctx < 0:
            raise ValueError(f"context_lens[{i}] must be non-negative, got {ctx}.")
        a = int(accepted[i])
        for t in range(1, max(0, a)):
            if (ctx + t + 1) % block_size != 0:
                continue
            if t >= lattices_store.shape[1]:
                raise ValueError(f"accepted token t={t} exceeds the store lattice width "
                                 f"{lattices_store.shape[1]} (request {i}).")
            block_idx = (ctx + t) // block_size
            if block_idx >= len(block_rows[i]):
                raise ValueError(f"block row {i} has {len(block_rows[i])} blocks, needs index {block_idx} "
                                 f"for context_len={ctx}, t={t}, block_size={block_size}.")
            src = int(store[i][t])
            dst = int(block_rows[i][block_idx])
            if src < 0:
                continue
            pairs.append((i, t, src, dst))
    return pairs


def commit_ssm(state: torch.Tensor, pairs: Sequence[tuple[int, int, int, int]]) -> None:
    """Copy ``state[src_slot]`` to ``state[dst_slot]`` for every pair (the ssm row of token ``t``)."""
    if not pairs:
        return
    srcs = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=state.device)
    dsts = torch.tensor([p[3] for p in pairs], dtype=torch.long, device=state.device)
    state.index_copy_(0, dsts, state.index_select(0, srcs))


def commit_conv(state: torch.Tensor,
                pairs: Sequence[tuple[int, int, int, int]],
                window_len: int,
                t_per_pair: Sequence[int] | None = None) -> None:
    """Copy the conv window of token ``t`` from the src slot into the head of the dst slot.

    The spec-path conv state is the expanded window ``[history; x_0 .. x_T]`` (rows
    ``window_len + k``), written by the kernel into the src slot (the store lattice's
    column-0 slot). The plain-decode window of the state after token ``t`` is the row range
    ``[t, t + window_len)``; the position block stores the plain layout, so the copy lands
    in rows ``[0, window_len)`` of ``dst_slot``. Supports ``[slots, S, dim]`` (rows on dim
    1, the GDN/PLE layout) and ``[slots, dim, S]`` (rows on dim 2). ``t_per_pair`` overrides
    the ``pair[1]`` token index (all pairs share one ``t`` in the common case).
    """
    if not pairs:
        return
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}.")
    if state.ndim != 3:
        raise ValueError(f"commit_conv expects a 3-D [slots, ...] state, got shape {tuple(state.shape)}.")
    if state.shape[1] >= window_len and state.shape[2] < window_len:
        row_dim = 1
    elif state.shape[2] >= window_len and state.shape[1] < window_len:
        row_dim = 2
    elif state.shape[1] >= window_len:
        row_dim = 1
    else:
        raise ValueError(f"state shape {tuple(state.shape)} has no row axis of length >= window_len="
                         f"{window_len}.")
    ts = list(t_per_pair) if t_per_pair is not None else [p[1] for p in pairs]
    if len(ts) != len(pairs):
        raise ValueError(f"t_per_pair has {len(ts)} entries for {len(pairs)} pairs.")
    # Two-phase: read every window (clone) before writing any. src == dst is the common
    # case (the store lattice's column-0 slot commits into its own head) and consecutive
    # pairs can share slots, so an in-order copy-read copy would read corrupted rows.
    windows = []
    for (i, _t, src, dst), t in zip(pairs, ts):
        if t < 0:
            raise ValueError(f"pair token index must be non-negative, got {t} (request {i}).")
        windows.append(state[src].narrow(row_dim - 1, t, window_len).clone())
    for (_i, _t, _src, dst), window in zip(pairs, windows):
        state[dst].narrow(row_dim - 1, 0, window_len).copy_(window)
