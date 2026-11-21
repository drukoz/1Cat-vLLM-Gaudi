# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_COLLECTIVE_MARK_STEP: the mark_step() before every HPU collective can be switched off.

``HpuCommunicator`` calls ``htorch.core.mark_step()`` before each all-reduce and all-gather, a
workaround for a Habana bridge bug with lazy collectives. A Flash-Next decode step at TP8 has 99
collectives, so that is 99 graph cuts per step. The switch exists to measure, on a card, what the
step costs without them; the default keeps the workaround.

Runs on CPU via ``tests/unit_tests/hpu_cpu_stubs`` (the communicator imports habana_frameworks).
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()

from vllm_gaudi.distributed.device_communicators import hpu_communicator  # noqa: E402

ENV = "VLLM_HPU_COLLECTIVE_MARK_STEP"


@pytest.fixture
def calls(monkeypatch):
    """Record the order of mark_step and the collectives; the collectives themselves do nothing."""
    seen = []
    monkeypatch.setattr(hpu_communicator.htorch.core, "mark_step", lambda: seen.append("mark_step"), raising=False)
    monkeypatch.setattr(hpu_communicator.dist, "all_reduce", lambda t, group=None: seen.append("all_reduce"))
    monkeypatch.setattr(hpu_communicator.dist, "all_gather_into_tensor",
                        lambda out, t, group=None: seen.append("all_gather"))
    return seen


def _communicator(world_size=2):
    return SimpleNamespace(device_group=object(), world_size=world_size)


def test_all_reduce_marks_a_step_by_default(calls, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    hpu_communicator.HpuCommunicator.all_reduce(_communicator(), torch.ones(1, 8))
    assert calls == ["mark_step", "all_reduce"]


def test_all_gather_marks_a_step_by_default(calls, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    hpu_communicator.HpuCommunicator.all_gather(_communicator(), torch.ones(1, 8))
    assert calls == ["mark_step", "all_gather"]


def test_the_switch_removes_the_mark_step_before_an_all_reduce(calls, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    hpu_communicator.HpuCommunicator.all_reduce(_communicator(), torch.ones(1, 8))
    assert calls == ["all_reduce"]


def test_the_switch_removes_the_mark_step_before_an_all_gather(calls, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    out = hpu_communicator.HpuCommunicator.all_gather(_communicator(world_size=2), torch.ones(1, 8))
    assert calls == ["all_gather"]
    assert out.shape == (1, 16)  # the reshape after the collective is unchanged
