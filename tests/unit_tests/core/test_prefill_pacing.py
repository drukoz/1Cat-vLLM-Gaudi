# SPDX-License-Identifier: Apache-2.0
"""Prefill pacing (``vllm_gaudi/v1/core/sched/prefill_pacing.py``): the policy and the wrapper's contract.

The policy is plain Python and is driven here with a fake clock. The wrapper is run against a stand-in
scheduler that records what the wrapped ``schedule`` was called with (step budget, per-request prompt
cap, the defer flag); ``test_prefill_pacing_scheduler.py`` runs the same wrapper on vLLM's real
``Scheduler``.
"""

import json
from types import SimpleNamespace

import pytest

from vllm_gaudi.v1.core.sched import prefill_pacing as pp
from vllm_gaudi.v1.core.sched.prefill_pacing import Knobs, KnobsFile, PrefillPacer, PrefillPacingMode, StepKind

PACED = Knobs(mode=int(PrefillPacingMode.TIME_SHARE), share_percent=25, slice_tokens=2048)


class Clock:

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_defaults_are_off():
    knobs = pp.knobs_from_env({})
    assert knobs.mode == PrefillPacingMode.OFF and knobs.is_noop and not knobs.paced
    assert knobs.share_percent == 25 and knobs.slice_tokens == 2048
    assert not pp.pacing_requested({})
    assert pp.pacing_requested({pp.ENV_MODE: "1"}) and pp.pacing_requested({pp.ENV_KNOBS_FILE: "/x/knobs.json"})


@pytest.mark.parametrize("bad", [dict(mode=2), dict(share_percent=0), dict(share_percent=101), dict(slice_tokens=-1),
                                 dict(max_num_scheduled_tokens=-1), dict(long_prefill_token_threshold=-2)])
def test_knobs_reject_values_out_of_range(bad):
    with pytest.raises(ValueError):
        Knobs(**bad)


def test_merged_rejects_unknown_keys_and_non_integers():
    assert PACED.merged({"share_percent": 10}).share_percent == 10
    with pytest.raises(ValueError, match="unknown scheduler knobs"):
        PACED.merged({"share": 10})
    for value in (True, "10", 2.5, None):
        with pytest.raises(ValueError, match="must be an integer"):
            PACED.merged({"share_percent": value})


def test_prompt_step_is_followed_by_decode_only_steps_for_three_times_its_extra_time():
    """25% share: a prompt step that took 1.03 s where a decode step takes 0.03 s owes 3.0 s of decode-only steps."""
    clock = Clock()
    pacer = PrefillPacer(clock)
    for _ in range(5):  # five decode-only steps of 30 ms teach the decode step time
        assert not pacer.should_defer(PACED, 8, False)
        pacer.record(StepKind.DECODE, True)
        clock.advance(0.03)
    assert not pacer.should_defer(PACED, 8, True)  # the first prompt step is admitted at once
    pacer.record(StepKind.PROMPT, True)
    clock.advance(1.03)
    deferred = 0
    while pacer.should_defer(PACED, 8, True):
        pacer.record(StepKind.DECODE, True)
        clock.advance(0.03)
        deferred += 1
    assert pacer.debt_s == pytest.approx(3.0, abs=1e-6)
    assert deferred == 100  # 3.0 s of 30 ms decode steps
    assert pacer.decode_step_s == pytest.approx(0.03)


@pytest.mark.parametrize("share, decode_steps", [(50, 34), (20, 134), (10, 300), (100, 0)])
def test_share_sets_the_ratio(share, decode_steps):
    clock = Clock()
    pacer = PrefillPacer(clock)
    knobs = Knobs(mode=1, share_percent=share, slice_tokens=2048)
    pacer.should_defer(knobs, 4, False)
    pacer.record(StepKind.DECODE, True)
    clock.advance(0.03)
    assert not pacer.should_defer(knobs, 4, True)
    pacer.record(StepKind.PROMPT, True)
    clock.advance(1.03)
    count = 0
    while pacer.should_defer(knobs, 4, True):
        pacer.record(StepKind.DECODE, True)
        clock.advance(0.03)
        count += 1
    assert count == decode_steps


def test_nothing_is_paced_while_no_request_decodes():
    clock = Clock()
    pacer = PrefillPacer(clock)
    for _ in range(6):  # a cold start: prompt steps back to back
        assert not pacer.should_defer(PACED, 0, True)
        pacer.record(StepKind.PROMPT, False)
        clock.advance(2.0)
    assert pacer.debt_s == 0.0
    # the first stream starts decoding: the prompt step that follows is admitted, the one after waits
    assert not pacer.should_defer(PACED, 1, True)
    pacer.record(StepKind.PROMPT, True)
    clock.advance(2.0)
    assert pacer.should_defer(PACED, 1, True)


def test_debt_is_dropped_when_the_streams_end_or_the_engine_idles():
    clock = Clock()
    pacer = PrefillPacer(clock)
    pacer.should_defer(PACED, 2, True)
    pacer.record(StepKind.PROMPT, True)
    clock.advance(4.0)
    assert pacer.should_defer(PACED, 2, True)  # 12 s owed
    pacer.record(StepKind.DECODE, True)
    clock.advance(0.03)
    assert not pacer.should_defer(PACED, 0, True)  # the streams finished: prompt work runs
    pacer.record(StepKind.PROMPT, False)
    clock.advance(pp.MAX_ACCOUNTED_STEP_S + 1.0)  # no schedule call for two minutes
    assert not pacer.should_defer(PACED, 2, True)
    assert pacer.debt_s == 0.0 and pacer.paid_s == 0.0


def test_mode_off_never_defers():
    clock = Clock()
    pacer = PrefillPacer(clock)
    off = Knobs()
    for _ in range(4):
        assert not pacer.should_defer(off, 8, True)
        pacer.record(StepKind.PROMPT, True)
        clock.advance(5.0)


@pytest.mark.parametrize("slice_tokens, block, expected", [(0, 512, 0), (2048, 512, 2048), (2000, 512, 1536),
                                                           (100, 512, 512), (2048, 896, 1792), (896, 896, 896),
                                                           (2048, 0, 2048)])
def test_effective_slice_is_whole_scheduler_blocks(slice_tokens, block, expected):
    assert pp.effective_slice(slice_tokens, block) == expected


def test_knobs_file_is_reread_on_change_and_survives_garbage(tmp_path):
    clock = Clock()
    path = tmp_path / "knobs.json"
    source = KnobsFile(str(path), Knobs(), clock)
    assert source.get() == Knobs()  # no file: the base knobs
    path.write_text(json.dumps({"max_num_scheduled_tokens": 16384, "long_prefill_token_threshold": 4096}))
    assert source.get() == Knobs()  # inside the poll interval
    clock.advance(pp.KNOBS_POLL_S + 0.01)
    assert source.get() == Knobs(max_num_scheduled_tokens=16384, long_prefill_token_threshold=4096)
    path.write_text("{ not json")
    clock.advance(pp.KNOBS_POLL_S + 0.01)
    assert source.get().max_num_scheduled_tokens == 16384  # the last good knobs stay
    path.write_text(json.dumps({"mode": 1, "share_percent": 10, "slice_tokens": 512, "extra": 1}))
    clock.advance(pp.KNOBS_POLL_S + 0.01)
    assert source.get().max_num_scheduled_tokens == 16384  # an unknown key rejects the whole file
    path.write_text(json.dumps({"mode": 1, "share_percent": 10, "slice_tokens": 512}))
    clock.advance(pp.KNOBS_POLL_S + 0.01)
    assert source.get() == Knobs(mode=1, share_percent=10, slice_tokens=512)
    path.unlink()
    clock.advance(pp.KNOBS_POLL_S + 0.01)
    assert source.get() == Knobs()


# ----- the wrapper against a stand-in scheduler -----


def request(request_id: str, prompt: bool):
    return SimpleNamespace(request_id=request_id, is_prefill_chunk=prompt)


class FakeScheduler:
    """The attributes ``paced_schedule`` reads and writes, with a ``schedule`` that records its inputs."""

    def __init__(self, budget=8192, threshold=2048, block=512, spec=0):
        self.running = []
        self.waiting = []
        self.skipped_waiting = []
        self.max_num_scheduled_tokens = budget
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=threshold, async_scheduling=False)
        self.cache_config = SimpleNamespace(block_size=block)
        self.num_spec_tokens = spec
        self.prefill_capacity_bound = True
        self.calls = []
        self.next_scheduled = {}

    def base_schedule(self, throttle_prefills=False):
        self.calls.append(
            SimpleNamespace(budget=self.max_num_scheduled_tokens,
                            threshold=self.scheduler_config.long_prefill_token_threshold,
                            throttle=throttle_prefills,
                            capacity_bound=self.prefill_capacity_bound))
        return SimpleNamespace(num_scheduled_tokens=dict(self.next_scheduled))


def run(scheduler, clock, scheduled):
    scheduler.next_scheduled = scheduled
    pp.paced_schedule(scheduler, FakeScheduler.base_schedule, False, clock)
    return scheduler.calls[-1]


@pytest.fixture()
def paced_env(monkeypatch):
    monkeypatch.setenv(pp.ENV_MODE, "1")
    monkeypatch.setenv(pp.ENV_SHARE, "25")
    monkeypatch.setenv(pp.ENV_SLICE, "2048")
    monkeypatch.delenv(pp.ENV_KNOBS_FILE, raising=False)


def test_wrapper_is_a_pass_through_with_the_flags_unset(monkeypatch):
    for name in (pp.ENV_MODE, pp.ENV_SHARE, pp.ENV_SLICE, pp.ENV_KNOBS_FILE):
        monkeypatch.delenv(name, raising=False)
    scheduler = FakeScheduler()
    scheduler.running = [request("d0", False), request("p0", True)]
    call = run(scheduler, Clock(), {"d0": 1, "p0": 2048})
    assert (call.budget, call.threshold, call.throttle, call.capacity_bound) == (8192, 2048, False, True)
    assert scheduler.max_num_scheduled_tokens == 8192


def test_wrapper_caps_the_prompt_tokens_of_a_step_while_streams_decode(paced_env):
    clock = Clock()
    scheduler = FakeScheduler(budget=65536, threshold=0, spec=3)
    scheduler.running = [request(f"d{i}", False) for i in range(8)] + [request("p0", True)]
    call = run(scheduler, clock, {"d0": 4, "p0": 2048})
    assert call.budget == 8 * 4 + 2048  # eight streams keep one token plus three drafts each
    assert call.threshold == 2048 and not call.throttle
    assert scheduler.max_num_scheduled_tokens == 65536  # restored
    assert scheduler.scheduler_config.long_prefill_token_threshold == 0


def test_wrapper_gives_a_cold_start_the_full_budget(paced_env):
    scheduler = FakeScheduler(budget=65536, threshold=16384)
    scheduler.running = [request("p0", True), request("p1", True)]
    scheduler.waiting = [request("p2", True)]
    call = run(scheduler, Clock(), {"p0": 16384, "p1": 16384, "p2": 16384})
    assert (call.budget, call.threshold, call.throttle) == (65536, 16384, False)


def test_wrapper_defers_prompt_work_and_clears_the_capacity_flag(paced_env):
    clock = Clock()
    scheduler = FakeScheduler()
    scheduler.running = [request("d0", False), request("p0", True)]
    first = run(scheduler, clock, {"d0": 1, "p0": 2048})
    assert not first.throttle and first.budget == 1 + 2048
    clock.advance(1.0)  # the prompt step took a second
    second = run(scheduler, clock, {"d0": 1})
    assert second.throttle and not second.capacity_bound
    assert second.budget == 8192 and second.threshold == 2048  # a deferred step needs no slice
    state = getattr(scheduler, pp.STATE_ATTR)
    assert (state.steps, state.deferred_steps, state.prompt_steps) == (2, 1, 1)


def test_wrapper_counts_a_newly_admitted_request_as_prompt_work(paced_env):
    clock = Clock()
    scheduler = FakeScheduler()
    scheduler.running = [request("d0", False)]
    scheduler.waiting = [request("new", True)]
    run(scheduler, clock, {"d0": 1, "new": 2048})  # "new" was not running before the call
    clock.advance(0.5)
    assert run(scheduler, clock, {"d0": 1}).throttle


def test_wrapper_applies_the_runtime_budget_and_cap_without_pacing(monkeypatch, tmp_path):
    path = tmp_path / "knobs.json"
    path.write_text(json.dumps({"max_num_scheduled_tokens": 16384, "long_prefill_token_threshold": 0}))
    monkeypatch.delenv(pp.ENV_MODE, raising=False)
    monkeypatch.setenv(pp.ENV_KNOBS_FILE, str(path))
    scheduler = FakeScheduler(budget=65536, threshold=2048)
    scheduler.running = [request("d0", False), request("p0", True)]
    call = run(scheduler, Clock(), {"d0": 1, "p0": 16383})
    assert (call.budget, call.threshold, call.throttle) == (16384, 0, False)
    path.write_text(json.dumps({"max_num_scheduled_tokens": 1 << 20}))  # above the server's own budget
    clock = Clock()
    clock.advance(1.0)
    scheduler2 = FakeScheduler(budget=65536, threshold=2048)
    assert run(scheduler2, clock, {}).budget == 65536


def test_async_scheduling_switches_the_pacing_off(paced_env):
    scheduler = FakeScheduler()
    scheduler.scheduler_config.async_scheduling = True
    scheduler.running = [request("d0", False), request("p0", True)]
    call = run(scheduler, Clock(), {"d0": 1, "p0": 2048})
    assert (call.budget, call.threshold, call.throttle) == (8192, 2048, False)


def test_install_wraps_once_and_only_on_request(monkeypatch):
    class Base:

        def schedule(self, throttle_prefills=False):
            return "base"

    for name in (pp.ENV_MODE, pp.ENV_KNOBS_FILE):
        monkeypatch.delenv(name, raising=False)
    original = Base.schedule
    assert not pp.install_prefill_pacing(Base)
    assert Base.schedule is original
    monkeypatch.setenv(pp.ENV_MODE, "1")
    assert pp.install_prefill_pacing(Base)
    wrapped = Base.schedule
    assert wrapped is not original and wrapped._hpu_base_schedule is original
    assert pp.install_prefill_pacing(Base) and Base.schedule is wrapped  # idempotent
    pp.uninstall_prefill_pacing(Base)
    assert Base.schedule is original
