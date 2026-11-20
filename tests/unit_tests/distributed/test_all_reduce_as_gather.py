# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS: small all-reduces as an all-gather of the partials and a local fp32 sum.

Runs on CPU via ``tests/unit_tests/hpu_cpu_stubs``; the collectives are replaced by functions that play the other
ranks (the whole-model parity on gloo ranks is ``models/flash_next/test_collective_cuts.py``).
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()

from vllm_gaudi.distributed.device_communicators import hpu_communicator  # noqa: E402

ENV = "VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS"
WORLD = 4


@pytest.fixture
def ranks(monkeypatch):
    """The partial tensors of ranks 1..3 for the next collective; rank 0's is the caller's input."""
    seen = []
    others = {}

    def all_gather_into_tensor(out, t, group=None):
        seen.append("all_gather")
        parts = [t] + [others["parts"][r] for r in range(1, WORLD)]
        out.copy_(torch.cat(parts, dim=0))

    def all_reduce(t, group=None):
        seen.append("all_reduce")
        for r in range(1, WORLD):
            t.add_(others["parts"][r])

    monkeypatch.setattr(hpu_communicator.htorch.core, "mark_step", lambda: seen.append("mark_step"), raising=False)
    monkeypatch.setattr(hpu_communicator.dist, "all_gather_into_tensor", all_gather_into_tensor)
    monkeypatch.setattr(hpu_communicator.dist, "all_reduce", all_reduce)
    monkeypatch.setenv("VLLM_HPU_COLLECTIVE_MARK_STEP", "0")
    return seen, others


def _communicator():
    comm = SimpleNamespace(device_group=object(), world_size=WORLD)
    comm._all_reduce_as_gather = MethodType(hpu_communicator.HpuCommunicator._all_reduce_as_gather, comm)
    return comm


def _parts(rows, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, 8, generator=gen).to(dtype) for _ in range(WORLD)]


def test_small_all_reduce_gathers_and_sums_in_fp32(ranks, monkeypatch):
    seen, others = ranks
    monkeypatch.setenv(ENV, "4")
    parts = _parts(3)
    others["parts"] = parts
    mine = parts[0].clone()
    out = hpu_communicator.HpuCommunicator.all_reduce(_communicator(), mine)
    assert seen == ["all_gather"]
    expected = torch.stack(parts).float().sum(0).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.equal(mine, parts[0])  # out of place: the input is not written


def test_rows_above_the_limit_keep_the_all_reduce(ranks, monkeypatch):
    seen, others = ranks
    monkeypatch.setenv(ENV, "4")
    others["parts"] = _parts(5)
    hpu_communicator.HpuCommunicator.all_reduce(_communicator(), others["parts"][0].clone())
    assert seen == ["all_reduce"]


def test_rows_count_every_dim_but_the_last(ranks, monkeypatch):
    """A [1, 6, 8] payload has 6 rows: above a limit of 4 it stays an all-reduce, at 6 it gathers."""
    seen, others = ranks
    monkeypatch.setenv(ENV, "4")
    others["parts"] = [p.view(1, 6, 8) for p in _parts(6)]
    hpu_communicator.HpuCommunicator.all_reduce(_communicator(), others["parts"][0].clone())
    monkeypatch.setenv(ENV, "6")
    out = hpu_communicator.HpuCommunicator.all_reduce(_communicator(), others["parts"][0].clone())
    assert seen == ["all_reduce", "all_gather"] and out.shape == (1, 6, 8)


@pytest.mark.parametrize("value", ["0", None])
def test_off_by_default(ranks, monkeypatch, value):
    seen, others = ranks
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    others["parts"] = _parts(1)
    hpu_communicator.HpuCommunicator.all_reduce(_communicator(), others["parts"][0].clone())
    assert seen == ["all_reduce"]


def test_integer_payloads_keep_the_all_reduce(ranks, monkeypatch):
    seen, others = ranks
    monkeypatch.setenv(ENV, "4")
    others["parts"] = [torch.ones(2, 8, dtype=torch.int32) for _ in range(WORLD)]
    out = hpu_communicator.HpuCommunicator.all_reduce(_communicator(), others["parts"][0].clone())
    assert seen == ["all_reduce"] and int(out[0, 0]) == WORLD


def test_one_owner_sums_are_exact(ranks, monkeypatch):
    """A vocabulary-parallel lookup's all-reduce (one nonzero partial per element) is the owner's value exactly."""
    seen, others = ranks
    monkeypatch.setenv(ENV, "8")
    owner = torch.randn(2, 8).to(torch.bfloat16)
    others["parts"] = [torch.zeros_like(owner), torch.zeros_like(owner), owner, torch.zeros_like(owner)]
    out = hpu_communicator.HpuCommunicator.all_reduce(_communicator(), torch.zeros_like(owner))
    assert torch.equal(out, owner)
