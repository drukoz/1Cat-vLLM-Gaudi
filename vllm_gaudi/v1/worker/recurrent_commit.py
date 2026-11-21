# SPDX-License-Identifier: Apache-2.0
"""Boundary commits and pre-step state copies for align-mode MTP (M3-T10b, T10c).

Pure torch/int helpers, importable on CPU, so they can be unit-tested without an HPU.

Under align mode the verification writes the state after candidate ``t`` into the store
lattice slot ``store[i, t]`` (``recurrent_lattice.align_lattice``, ``load == store``), while
the accepted checkpoint a request must expose for prefix-cache hits lives in the POSITION
blocks of its block table. An accepted token at position ``context_len + t`` completes the
block holding that position when ``(context_len + t + 1) % block_size == 0``, so the runner
copies the states of the COMMITTED tokens ``0 <= t < accepted`` that complete a block from the
store lattice into that block's slot (``boundary_commits``); token 0 is committed like any
other token because the store lattice no longer special-cases it. The copies run at sample
time, once the accepted length of the step is known.

Before the step, ``align_precopy_plan`` moves the running recurrent state into the request's
running block ``curr`` when the window has moved (in either direction: a step that schedules
fewer tokens moves it back), or, with ``normalize``, when the previous step left more than one
accepted token in it; ``precopy_slot_pairs`` maps a plan onto one cache group's slots for
``commit_conv``/``commit_ssm``, and ``self_commit_requests`` finds the requests whose boundary
commit writes the running block itself (their next accepted count is 1).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_gaudi.v1.worker.recurrent_lattice import align_running_block

__all__ = [
    "boundary_commits",
    "commit_ssm",
    "commit_conv",
    "align_precopy_plan",
    "precopy_slot_pairs",
    "self_commit_requests",
    "apply_align_precopy",
]


def boundary_commits(context_lens: Sequence[int],
                     accepted: Sequence[int],
                     block_size: int,
                     lattices_store: torch.Tensor,
                     block_rows: Sequence[Sequence[int]],
                     lattices_load: torch.Tensor | None = None) -> list[tuple[int, int, int, int]]:
    """(request, t, src_slot, dst_slot) for accepted tokens ``0 <= t < accepted[i]`` that complete a block.

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
        for t in range(0, max(0, a)):
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
    for (i, _t, _src, _dst), t in zip(pairs, ts):
        if t < 0:
            raise ValueError(f"pair token index must be non-negative, got {t} (request {i}).")
    # Two pairs on one dst: the later one wins (all of them write the same head rows).
    last_of_dst = {int(p[3]): j for j, p in enumerate(pairs)}
    keep = sorted(last_of_dst.values())
    # Two-phase: every source and destination row is read (index_select makes copies) before
    # the one write. src == dst is the common case (the store lattice's column-0 slot commits
    # into its own head) and consecutive pairs can share slots.
    srcs = torch.tensor([int(pairs[j][2]) for j in keep], dtype=torch.long, device=state.device)
    dsts = torch.tensor([int(pairs[j][3]) for j in keep], dtype=torch.long, device=state.device)
    src_rows = state.index_select(0, srcs)
    dst_rows = state.index_select(0, dsts)
    windows = torch.stack([src_rows[n].narrow(row_dim - 1, ts[j], window_len) for n, j in enumerate(keep)])
    rest = dst_rows.narrow(row_dim, window_len, dst_rows.shape[row_dim] - window_len)
    # Whole rows on the pool itself, never a write through a slice of it: the lazy bridge turns a slice
    # write into hpu::slice_insert, and an HPU graph captured after it found the GDN conv pool without data
    # (card job 00133: "ValidateSyncInputTensors tensor_data is empty" at gdn_core_verify's first input,
    # on the request after a block commit). commit_ssm writes whole rows the same way.
    state.index_copy_(0, dsts, torch.cat((windows, rest), dim=row_dim))


def align_precopy_plan(prev_blocks: Sequence[int | None], context_lens: Sequence[int], num_scheduled: Sequence[int],
                       accepted: Sequence[int], block_size: int, normalize: bool = False
                       ) -> tuple[list[tuple[int, int, int, int]], list[int], list[int]]:
    """Pre-step copies that move each request's running state into its running block.
    Returns (plan, new_prev_blocks, new_accepted). plan entries are (request, bias, src_block, dst_block)
    (block INDICES, not slots). new_prev_blocks[i] = curr_i; new_accepted[i] = 1 when request i is planned,
    else accepted[i]. prev may be past curr (this step schedules fewer tokens than the previous one);
    the copy then runs backward. ValueError: unequal lengths, accepted[i] < 1."""
    batch = len(context_lens)
    if len(prev_blocks) != batch or len(num_scheduled) != batch or len(accepted) != batch:
        raise ValueError("prev_blocks, context_lens, num_scheduled and accepted must have equal length, got "
                         f"{len(prev_blocks)}, {batch}, {len(num_scheduled)}, {len(accepted)}.")
    plan: list[tuple[int, int, int, int]] = []
    new_prev_blocks: list[int] = []
    new_accepted: list[int] = []
    for i in range(batch):
        ctx = int(context_lens[i])
        n = int(accepted[i])
        if n < 1:
            raise ValueError(f"accepted[{i}] must be at least 1, got {n}.")
        curr = align_running_block(ctx, int(num_scheduled[i]), block_size)
        prev = (ctx - 1) // block_size if prev_blocks[i] is None else int(prev_blocks[i])
        new_prev_blocks.append(curr)
        if ctx > 0 and (prev != curr or (normalize and n > 1)):
            plan.append((i, n - 1, prev, curr))
            new_accepted.append(1)
        else:
            new_accepted.append(n)
    return plan, new_prev_blocks, new_accepted


def precopy_slot_pairs(plan: Sequence[tuple[int, int, int, int]], block_rows: Sequence[Sequence[int]],
                       null_block: int = 0
                       ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[int]]:
    """Map a plan to one cache group's slots: (conv_pairs, ssm_pairs, ts) with
    conv (i, bias, row[src], row[dst]), ssm (i, bias, row[src + bias], row[dst]), ts[j] = bias.
    ValueError: an index outside block_rows[i]; any of the three slots equal to null_block."""
    conv_pairs: list[tuple[int, int, int, int]] = []
    ssm_pairs: list[tuple[int, int, int, int]] = []
    ts: list[int] = []
    for (i, bias, src_block, dst_block) in plan:
        if i < 0 or i >= len(block_rows):
            raise ValueError(f"plan entry references request {i}, outside block_rows (len {len(block_rows)}).")
        row = block_rows[i]
        for name, idx in (("src_block", src_block), ("src_block + bias", src_block + bias),
                          ("dst_block", dst_block)):
            if idx < 0 or idx >= len(row):
                raise ValueError(f"precopy {name} index {idx} of request {i} is outside block_rows[{i}] "
                                 f"(len {len(row)}).")
        conv_src, ssm_src, dst = int(row[src_block]), int(row[src_block + bias]), int(row[dst_block])
        for name, slot in (("conv source", conv_src), ("ssm source", ssm_src), ("destination", dst)):
            if slot == int(null_block):
                raise ValueError(f"precopy {name} slot of request {i} is the null block {null_block} "
                                 f"(block indices {src_block} + {bias} -> {dst_block}).")
        conv_pairs.append((i, bias, conv_src, dst))
        ssm_pairs.append((i, bias, ssm_src, dst))
        ts.append(int(bias))
    return conv_pairs, ssm_pairs, ts


def self_commit_requests(pairs: Sequence[tuple[int, int, int, int]], lattices_store: torch.Tensor) -> set[int]:
    """Requests i with a boundary-commit pair whose dst_slot == int(lattices_store[i][0])."""
    return {int(i) for (i, _t, _src, dst) in pairs if int(lattices_store[int(i)][0]) == int(dst)}


def apply_align_precopy(registry: dict, block_rows_by_group: dict, req_ids: Sequence[str],
                        running_blocks: dict, accepted_by_req: dict, context_lens: Sequence[int],
                        num_scheduled: Sequence[int], block_size: int, normalize: bool
                        ) -> list[tuple[int, int, int, int]]:
    """Pre-step align-mode copies for the decode requests req_ids (upstream vLLM preprocess_mamba, M3-T10c).

    registry is the runner's {group_id: [(tensor, "conv" | "ssm", window_rows)]}; block_rows_by_group[gid][i]
    is request i's block-table row of that group. Reads running_blocks.get(req_id) and
    accepted_by_req.get(req_id, 1), copies the states in place with commit_ssm/commit_conv, then writes the new
    running block and accepted count of every request back into the two dicts. Returns the plan."""
    prev = [running_blocks.get(req_id) for req_id in req_ids]
    acc = [accepted_by_req.get(req_id, 1) for req_id in req_ids]
    plan, new_prev, new_acc = align_precopy_plan(prev, context_lens, num_scheduled, acc, block_size, normalize)
    if plan:
        for gid, entries in registry.items():
            conv_pairs, ssm_pairs, ts = precopy_slot_pairs(plan, block_rows_by_group[gid])
            for tensor, kind, window_len in entries:
                if kind == "ssm":
                    commit_ssm(tensor, ssm_pairs)
                else:
                    commit_conv(tensor, conv_pairs, window_len, t_per_pair=ts)
    for i, req_id in enumerate(req_ids):
        running_blocks[req_id] = new_prev[i]
        accepted_by_req[req_id] = new_acc[i]
    return plan
