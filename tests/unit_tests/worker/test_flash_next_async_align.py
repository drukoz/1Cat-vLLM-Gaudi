# SPDX-License-Identifier: Apache-2.0
"""Stage 2 M3 (VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN): the prefix-cache align mode of a verification step made on the
device (``flash_next_async_spec.align_*``, ``device_commit_*``) equals the host's (``recurrent_lattice``,
``recurrent_commit``) for random contexts, previous running blocks, load counts and acceptance."""

import random

import pytest
import torch

from vllm_gaudi.v1.worker.flash_next_async_spec import (align_commits, align_lattice_rows, align_precopy,
                                                        device_commit_conv, device_commit_ssm)
from vllm_gaudi.v1.worker.recurrent_commit import (align_precopy_plan, boundary_commits, commit_conv, commit_ssm,
                                                   precopy_slot_pairs, self_commit_requests)
from vllm_gaudi.v1.worker.recurrent_lattice import align_lattices

PAD = 0


def _case(rng, k, bs, batch, padded):
    width = 12
    blocks = list(range(1, 1 + batch * width))
    rng.shuffle(blocks)
    rows = torch.tensor([blocks[i * width:(i + 1) * width] for i in range(batch)], dtype=torch.int32)
    ctx = [rng.randrange(0, (width - 2 - k) * bs) for _ in range(batch)]
    prev = [None if rng.random() < 0.3 else max(0, (c - 1) // bs + rng.choice([0, 0, -1, 1])) for c in ctx]
    n = [rng.randrange(1, k + 2) for _ in range(batch)]
    full = torch.zeros((padded, width), dtype=torch.int32)
    full[:batch] = rows
    real = torch.arange(padded) < batch
    pad = lambda xs, fill: torch.tensor(list(xs) + [fill] * (padded - batch), dtype=torch.int32)
    return rows, full, real, ctx, prev, n, pad


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_precopy_plan_and_lattices_equal_the_hosts(seed, k):
    rng = random.Random(seed * 7 + k)
    bs, batch, padded = 8, rng.randrange(1, 6), 6
    rows, full, real, ctx, prev, n, pad = _case(rng, k, bs, batch, padded)
    plan, new_prev, new_acc = align_precopy_plan(prev, ctx, [1 + k] * batch, n, bs)
    curr, planned, src, ssm_src, n_after = align_precopy(pad(ctx, 0), pad([-1 if p is None else p for p in prev], 0),
                                                         pad(n, 1), k, bs, real)
    assert curr[:batch].tolist() == new_prev
    assert n_after[:batch].tolist() == new_acc
    assert [i for i in range(batch) if planned[i]] == [p[0] for p in plan] and not planned[batch:].any()
    conv_pairs, ssm_pairs, _ = precopy_slot_pairs(plan, rows.tolist()) if plan else ([], [], [])
    for (i, _bias, conv_slot, dst), (_, _, ssm_slot, _) in zip(conv_pairs, ssm_pairs):
        assert int(rows[i, src[i]]) == conv_slot and int(rows[i, ssm_src[i]]) == ssm_slot
        assert int(rows[i, curr[i]]) == dst
    load, store = align_lattices(rows, list(range(batch)), ctx, [1 + k] * batch, bs, k, padded, PAD)
    lattice = align_lattice_rows(full, curr, k, real, PAD)
    assert torch.equal(lattice, load) and torch.equal(lattice, store)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_boundary_commits_and_self_commits_equal_the_hosts(seed, k):
    rng = random.Random(seed * 11 + k)
    bs, batch, padded = 8, rng.randrange(1, 6), 6
    rows, full, real, ctx, _, _, pad = _case(rng, k, bs, batch, padded)
    committed = [rng.randrange(1, k + 2) for _ in range(batch)]
    load, store = align_lattices(rows, list(range(batch)), ctx, [1 + k] * batch, bs, k, padded, PAD)
    pairs = boundary_commits(ctx, committed, bs, store[:batch], rows.tolist(), load[:batch])
    selfs = self_commit_requests(pairs, store)
    mask, t, ssm_src, conv_src, dst, self_commit = align_commits(pad(ctx, 0), pad(committed, 0), bs, store, full, real)
    assert [(i, int(t[i]), int(ssm_src[i]), int(dst[i])) for i in range(padded) if mask[i]] == pairs
    assert {i for i in range(padded) if self_commit[i]} == selfs
    assert all(int(conv_src[i]) == int(store[i, 0]) for i in range(batch))


@pytest.mark.parametrize("layout", ["rows_dim1", "rows_dim2"])
def test_device_commits_write_what_the_host_commits_write(layout):
    torch.manual_seed(0)
    window, k, slots = 3, 2, 20
    shape = (slots, window + k, 5) if layout == "rows_dim1" else (slots, 5, window + k)
    conv = torch.randn(shape)
    ssm = torch.randn((slots, 4, 4))
    pairs = [(0, 2, 5, 9), (1, 0, 7, 7), (3, 1, 11, 2)]  # (request, t, src, dst), one src == dst
    mask = torch.tensor([True, True, False, True])
    src = torch.tensor([5, 7, 13, 11], dtype=torch.int32)
    dst = torch.tensor([9, 7, 14, 2], dtype=torch.int32)
    t = torch.tensor([2, 0, 1, 1], dtype=torch.int32)
    host_conv, host_ssm = conv.clone(), ssm.clone()
    commit_conv(host_conv, pairs, window)
    commit_ssm(host_ssm, pairs)
    device_commit_conv(conv, src, dst, t, mask, window, PAD)
    device_commit_ssm(ssm, src, dst, mask, PAD)
    keep = [s for s in range(slots) if s != PAD]
    assert torch.equal(conv[keep], host_conv[keep]) and torch.equal(ssm[keep], host_ssm[keep])
