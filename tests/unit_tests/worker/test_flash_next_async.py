# SPDX-License-Identifier: Apache-2.0
"""Async scheduling of Flash-Next on HPU: the carried PLE n-gram context, the output-token ledger and the
platform's decision (``vllm_gaudi/v1/worker/flash_next_async.py``, ``VLLM_HPU_FLASH_NEXT_ASYNC_SCHED``).

The carry is checked against the oracle it replaces: under async scheduling the host never learns a decode row's
input token, so the host-built context of a row that decoded in the previous step is wrong, and the carried one
must equal ``build_ple_inputs`` over the true token history, whatever the row order, the requests that join,
leave or skip a step, and the padding.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gaudi.v1.worker.flash_next_async import (AsyncOutputLedger, PleContextCarry, needs_output_token_values)
from vllm_gaudi.v1.worker.flash_next_inputs import build_ple_inputs

EOS = 7
NGRAM = 3  # ngram_size: a context of two tokens
STALE = 0  # what the host holds where it never learned a token


class _Sim:
    """True token histories of several requests and the host's view of them under async scheduling."""

    def __init__(self, seed: int, num_requests: int, max_len: int = 64):
        rng = random.Random(seed)
        self.rng = rng
        self.max_len = max_len
        self.truth = {}
        self.prompt_len = {}
        self.pos = {}  # next input position = num_computed_tokens
        for index in range(num_requests):
            req_id = f"r{index}"
            prompt = rng.randint(1, 5)
            self.truth[req_id] = [rng.randrange(1, 100) for _ in range(max_len)]
            # Some histories carry an EOS inside, so the context's EOS cut is part of the check.
            if rng.random() < 0.3:
                self.truth[req_id][rng.randrange(0, max_len)] = EOS
            self.prompt_len[req_id] = prompt
            self.pos[req_id] = prompt  # the prompt is done: the first decode's input is the prefill's sample
        self.host = {req_id: self._host_row(req_id, upto=self.prompt_len[req_id]) for req_id in self.truth}

    def _host_row(self, req_id, upto):
        row = [STALE] * self.max_len
        row[:upto] = self.truth[req_id][:upto]
        return row

    def readd(self, req_id):
        """The request comes back into the persistent batch: the scheduler sends every token it knows."""
        self.host[req_id] = self._host_row(req_id, upto=self.pos[req_id])

    def host_context(self, req_ids, padded):
        token_ids = np.array([self.host[r] for r in req_ids] + [[STALE] * self.max_len] * (padded - len(req_ids)),
                             dtype=np.int32)
        computed = np.array([self.pos[r] for r in req_ids] + [0] * (padded - len(req_ids)), dtype=np.int32)
        return build_ple_inputs(token_ids, computed, list(range(len(req_ids))), padded, 1, EOS, NGRAM)[1]

    def oracle_context(self, req_ids, padded):
        token_ids = np.array([self.truth[r] for r in req_ids] + [[STALE] * self.max_len] * (padded - len(req_ids)),
                             dtype=np.int32)
        computed = np.array([self.pos[r] for r in req_ids] + [0] * (padded - len(req_ids)), dtype=np.int32)
        return build_ple_inputs(token_ids, computed, list(range(len(req_ids))), padded, 1, EOS, NGRAM)[1]

    def input_ids(self, req_ids, padded):
        ids = [self.truth[r][self.pos[r]] for r in req_ids] + [0] * (padded - len(req_ids))
        return torch.tensor(ids, dtype=torch.int32).view(-1, 1)

    def advance(self, req_ids):
        for r in req_ids:
            self.pos[r] += 1


def _padded(n):
    return max(1, 1 << (n - 1).bit_length())


@pytest.mark.parametrize("seed", range(12))
def test_carried_context_equals_the_true_history(seed):
    """Random steps: rows reorder, requests join (first decode after the prompt), skip a step and come back
    (re-added with every known token), leave, and some steps have no decode batch at all."""
    sim = _Sim(seed, num_requests=9)
    carry = PleContextCarry()
    active = []
    previous = set()  # requests that decoded in the step before
    carried_rows = 0
    for step in range(40):
        carry.begin_step()
        rng = sim.rng
        if rng.random() < 0.1:
            previous = set()  # a step without a decode batch (only prompts): nothing carries past it
            continue
        pool = [r for r in sim.truth if sim.pos[r] < sim.max_len - 1]
        if not pool:
            break
        active = rng.sample(pool, k=rng.randint(1, min(6, len(pool))))
        for req_id in active:
            if req_id not in previous and sim.pos[req_id] > sim.prompt_len[req_id]:
                sim.readd(req_id)  # it decoded before, then left the batch for a step or more
        padded = _padded(len(active))
        host = sim.host_context(active, padded)
        got = carry.decode_context(host.clone(), active, sim.input_ids(active, padded))
        want = sim.oracle_context(active, padded)
        assert torch.equal(got, want), f"step {step}: rows {active}"
        assert got.dtype == torch.int32 and tuple(got.shape) == tuple(host.shape)
        carried_rows += len(set(active) & previous)
        sim.advance(active)
        previous = set(active)
    assert carried_rows > 20, "the random schedule carried too few rows to test anything"


def test_without_the_carry_the_host_context_is_wrong():
    """Negative control: the host's own context of a row that decoded in the previous step is stale."""
    sim = _Sim(3, num_requests=4)
    rows = ["r0", "r1", "r2", "r3"]
    sim.advance(rows)  # one decode step happened: the host never learned its input tokens
    assert not torch.equal(sim.host_context(rows, 4), sim.oracle_context(rows, 4))


def test_common_case_keeps_rows_and_padding():
    carry = PleContextCarry()
    carry.begin_step()
    first = torch.tensor([[1, 2], [3, 4], [EOS, EOS], [EOS, EOS]], dtype=torch.int32)
    carry.decode_context(first, ["a", "b"], torch.tensor([[10], [20], [0], [0]], dtype=torch.int32))
    carry.begin_step()
    host = torch.tensor([[0, 0], [0, 0], [EOS, EOS], [EOS, EOS]], dtype=torch.int32)
    got = carry.decode_context(host, ["a", "b"], torch.tensor([[11], [21], [0], [0]], dtype=torch.int32))
    assert got.tolist() == [[2, 10], [4, 20], [EOS, EOS], [EOS, EOS]]
    assert carry.carried_requests == ("a", "b")


def test_reordered_and_new_rows():
    carry = PleContextCarry()
    carry.begin_step()
    carry.decode_context(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), ["a", "b"],
                         torch.tensor([[10], [20]], dtype=torch.int32))
    carry.begin_step()
    host = torch.tensor([[5, 6], [0, 0], [0, 0], [EOS, EOS]], dtype=torch.int32)  # "c" is new: the host knows it
    got = carry.decode_context(host, ["c", "b", "a"], torch.tensor([[30], [21], [11], [0]], dtype=torch.int32))
    assert got.tolist() == [[5, 6], [4, 20], [2, 10], [EOS, EOS]]


def test_a_step_without_decodes_ends_the_carry():
    carry = PleContextCarry()
    carry.begin_step()
    carry.decode_context(torch.tensor([[1, 2]], dtype=torch.int32), ["a"], torch.tensor([[10]], dtype=torch.int32))
    carry.begin_step()  # a step with prompts only: no decode batch, nothing recorded
    carry.begin_step()
    host = torch.tensor([[8, 9]], dtype=torch.int32)
    assert carry.decode_context(host, ["a"], torch.tensor([[11]], dtype=torch.int32)).tolist() == [[8, 9]]


def test_several_tokens_per_row_commit_every_scheduled_token():
    """Not a plain decode step (no speculation, several tokens): the next context is the last two inputs."""
    carry = PleContextCarry()
    carry.begin_step()
    context = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    inputs = torch.tensor([[10, 11, 12], [20, 21, 0]], dtype=torch.int32).view(-1, 1)  # [rows * T, 1]
    carry.decode_context(context, ["a", "b"], inputs, num_tokens_per_req=[3, 2])
    carry.begin_step()
    got = carry.decode_context(torch.zeros((2, 2), dtype=torch.int32), ["a", "b"],
                               torch.zeros((2, 1), dtype=torch.int32))
    assert got.tolist() == [[11, 12], [20, 21]]


def test_no_context_width_is_a_no_op():
    carry = PleContextCarry()
    carry.begin_step()
    empty = torch.zeros((2, 0), dtype=torch.int32)
    assert carry.decode_context(empty, ["a"], torch.tensor([[1], [0]], dtype=torch.int32)).shape == (2, 0)
    assert carry.carried_requests == ()


class _Event:

    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


def _requests(*names):
    return {name: SimpleNamespace(output_token_ids=[]) for name in names}


def test_ledger_placeholders_then_values():
    requests = _requests("a", "b", "c")
    ledger = AsyncOutputLedger()
    first = _Event()
    ledger.record(requests, ["a", "b", "c"], invalid_rows=[2], sampled_cpu=torch.tensor([[5], [6], [9]]),
                  event=first)
    assert [requests[n].output_token_ids for n in "abc"] == [[-1], [-1], []]
    second = _Event()
    ledger.record(requests, ["b", "a"], invalid_rows=[], sampled_cpu=torch.tensor([[7], [8]]), event=second)
    assert requests["a"].output_token_ids == [-1, -1] and requests["b"].output_token_ids == [-1, -1]
    assert ledger.settle(keep=1) == 2 and first.waits == 1 and second.waits == 0
    assert requests["a"].output_token_ids == [5, -1] and requests["b"].output_token_ids == [6, -1]
    assert ledger.settle() == 2 and second.waits == 1 and len(ledger) == 0
    assert requests["a"].output_token_ids == [5, 8] and requests["b"].output_token_ids == [6, 7]


def test_ledger_leaves_replaced_and_rewritten_lists_alone():
    requests = _requests("a", "b")
    ledger = AsyncOutputLedger()
    ledger.record(requests, ["a", "b", "gone"], invalid_rows=None, sampled_cpu=[[5], [6], [7]], event=None)
    old_a = requests["a"].output_token_ids
    requests["a"].output_token_ids = [42]  # resumed: the scheduler's tokens replaced the list
    requests["b"].output_token_ids[0] = 99  # someone wrote the real token already
    assert ledger.settle() == 0
    assert old_a == [-1]  # no longer the request's list: not touched
    assert requests["a"].output_token_ids == [42] and requests["b"].output_token_ids == [99]


def test_needs_output_token_values():
    assert not needs_output_token_values(SimpleNamespace(no_penalties=True, bad_words_token_ids={}))
    assert needs_output_token_values(SimpleNamespace(no_penalties=False, bad_words_token_ids={}))
    assert needs_output_token_values(SimpleNamespace(no_penalties=True, bad_words_token_ids={0: [[1, 2]]}))


@pytest.fixture
def platform_module(monkeypatch):
    from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (habana_frameworks for the platform import)
    import vllm_gaudi.platform as platform
    monkeypatch.delenv("VLLM_HPU_FLASH_NEXT_ASYNC_SCHED", raising=False)
    return platform


def _config(async_scheduling, speculative=False):
    return SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
                           speculative_config=SimpleNamespace(method="mtp") if speculative else None)


@pytest.mark.parametrize("value", [None, "0", "2"])
def test_platform_default_turns_async_off(platform_module, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_ASYNC_SCHED", value)
    config = _config(True)
    assert platform_module.resolve_flash_next_async_scheduling(config) is False
    assert config.scheduler_config.async_scheduling is False


def test_platform_flag_keeps_async(platform_module, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_ASYNC_SCHED", "1")
    config = _config(True)
    assert platform_module.resolve_flash_next_async_scheduling(config) is True
    assert config.scheduler_config.async_scheduling is True


@pytest.mark.parametrize("speculative", [False, True])
def test_platform_flag_does_not_turn_async_on(platform_module, monkeypatch, speculative):
    """--no-async-scheduling, or a speculative config (turned off just before), stays off."""
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_ASYNC_SCHED", "1")
    config = _config(False, speculative=speculative)
    assert platform_module.resolve_flash_next_async_scheduling(config) is False
    assert config.scheduler_config.async_scheduling is False
