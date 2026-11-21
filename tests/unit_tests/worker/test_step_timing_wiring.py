# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_STEP_TIMING: where the worker, the model runner, the communicator and the engine core call the timer.

Each test drives the real method with stand-ins around it and a hand clock, then reads the record the timer wrote.
With the flag at 0 no timer exists and the same methods run their original statements; that is checked too.

Runs on CPU via ``tests/unit_tests/hpu_cpu_stubs``.
"""

import contextlib
import inspect
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()

from vllm_gaudi.distributed.device_communicators import hpu_communicator  # noqa: E402
from vllm_gaudi.extension import step_timing  # noqa: E402
from vllm_gaudi.extension.step_timing import StepTimer  # noqa: E402
from vllm_gaudi.v1.worker import hpu_model_runner  # noqa: E402
from vllm_gaudi.v1.worker.hpu_model_runner import AsyncHPUModelRunnerOutput, HPUModelRunner  # noqa: E402
from vllm_gaudi.v1.worker.hpu_worker import HPUWorker  # noqa: E402

MS = 1_000_000


class Clock:

    def __init__(self):
        self.ns = 5 * 1_000_000_000

    def __call__(self):
        return self.ns

    def wall(self):
        return self.ns / 1e9

    def ms(self, milliseconds):
        self.ns += int(milliseconds * MS)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def timer(tmp_path, clock, monkeypatch):
    made = StepTimer("worker", rank=0, mode=step_timing.MODE_SPANS, every=1, out_dir=str(tmp_path), clock=clock,
                     wall=clock.wall)
    monkeypatch.setattr(step_timing, "TIMER", made)
    yield made
    made.close()


@pytest.fixture
def no_timer(monkeypatch):
    monkeypatch.setattr(step_timing, "TIMER", None)


def records(timer):
    timer.flush()
    with open(timer.path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def p50(record, name):
    return record["spans_ms"][name]["p50"]


# ------------------------------------------------------------------------------------------- worker
class FakeRunner:

    def __init__(self, clock, timer, execute_output=None):
        self.clock = clock
        self.timer = timer
        self.execute_output = execute_output
        self.calls = []

    def execute_model(self, scheduler_output):
        self.calls.append("execute_model")
        self.clock.ms(0.3)
        return self.execute_output

    def sample_tokens(self, grammar_output):
        self.calls.append("sample_tokens")
        if self.timer is not None:
            self.timer.tag("decode", 4, tokens=4)
        self.clock.ms(2.0)
        if self.timer is not None:
            self.timer.lap("forward")
        self.clock.ms(0.1)
        return "sampled"


def make_worker(runner, timer):
    worker = object.__new__(HPUWorker)
    worker.step_debug = None
    worker.step_profiler = None
    worker.gc_track_recompiles = False
    worker.step = 0
    worker.model_runner = runner
    if timer is not None:
        worker.step_timer = timer
    return worker


def test_a_worker_step_is_execute_model_the_engine_turn_and_sample_tokens(timer, clock):
    runner = FakeRunner(clock, timer)
    worker = make_worker(runner, timer)
    for _ in range(2):
        assert HPUWorker.execute_model(worker, object()) is None
        clock.ms(0.4)  # the engine reads the reply and sends sample_tokens
        assert HPUWorker.sample_tokens(worker, None) == "sampled"
        clock.ms(1.0)  # the engine's turn between two steps
    first, second = records(timer)
    assert p50(first, "execute_model") == pytest.approx(0.3)
    assert p50(first, "rpc_gap") == pytest.approx(0.4)
    assert p50(first, "forward") == pytest.approx(2.0)
    assert p50(first, "post") == pytest.approx(0.1)
    assert p50(first, "step") == pytest.approx(2.8)
    assert first["raw"]["gap"] == [None] and second["raw"]["gap"] == [1.0]
    assert p50(second, "period") == pytest.approx(3.8)
    assert (first["kind"], first["rows"], first["role"]) == ("decode", 4, "worker")
    assert worker.step == 2


def test_a_step_with_nothing_to_sample_is_not_recorded(timer, clock):
    runner = FakeRunner(clock, timer, execute_output="empty")
    worker = make_worker(runner, timer)
    assert HPUWorker.execute_model(worker, object()) == "empty"
    assert timer.active is False
    timer.flush()
    assert not os.path.exists(timer.path)


def test_without_the_flag_the_worker_only_calls_the_runner(no_timer, clock):
    runner = FakeRunner(clock, None)
    worker = make_worker(runner, None)
    assert HPUWorker.step_timer is None and "step_timer" not in vars(worker)
    assert HPUWorker.execute_model(worker, object()) is None
    assert HPUWorker.sample_tokens(worker, None) == "sampled"
    assert runner.calls == ["execute_model", "sample_tokens"]


def test_the_worker_creates_its_timer_from_the_flag():
    source = inspect.getsource(HPUWorker.__init__)
    assert 'step_timing.init_timer("worker", rank' in source
    assert "self.step_timer.bind_runtime(htorch, torch)" in inspect.getsource(HPUWorker.init_device)


# ------------------------------------------------------------------------------------------- runner
def make_runner(timer, clock, forward_ms=3.0, logits_ms=0.5):

    def forward(**kwargs):
        clock.ms(forward_ms)
        return torch.zeros(2, 1, 8)

    def select_and_compute_logits(hidden_states, logits_indices):
        clock.ms(logits_ms)
        return hidden_states, torch.zeros(2, 16)

    runner = object.__new__(HPUModelRunner)
    runner.seen_configs = set()
    runner.model_has_chunked_attention = False
    runner.is_driver_worker = True
    runner.use_aux_hidden_state_outputs = False
    runner.profiler = SimpleNamespace(record_event=lambda *args, **kwargs: contextlib.nullcontext())
    runner.model = SimpleNamespace(forward=forward, select_and_compute_logits=select_and_compute_logits)
    if timer is not None:
        runner.step_timer = timer
    return runner


def run_generic(runner):
    token_ids = torch.zeros(2, 1, dtype=torch.int32)
    return HPUModelRunner._execute_model_generic(runner, token_ids, token_ids, {}, torch.zeros(2, dtype=torch.int32),
                                                 None, None, None)


def test_the_forward_and_the_logits_are_two_spans(timer, clock):
    runner = make_runner(timer, clock)
    timer.begin_step()
    timer.tag("decode", 2)
    clock.ms(0.2)  # whatever the caller did since its last boundary
    outputs = run_generic(runner)
    timer.end_step()
    record = records(timer)[0]
    assert p50(record, "setup") == pytest.approx(0.2)
    assert p50(record, "forward") == pytest.approx(3.0)
    assert p50(record, "logits") == pytest.approx(0.5)
    assert outputs[3].shape == (2, 16) and outputs[4] is None
    assert timer.phase == "logits"  # a collective issued by compute_logits belongs to the logits span


def test_without_the_flag_the_forward_runs_unchanged(no_timer, clock):
    runner = make_runner(None, clock)
    assert HPUModelRunner.step_timer is None and "step_timer" not in vars(runner)
    outputs = run_generic(runner)
    assert outputs[0].shape == (2, 1, 8) and outputs[3].shape == (2, 16)


def sampling_runner(timer, clock, marks):
    runner = object.__new__(HPUModelRunner)
    runner._prepare_sampling = lambda *args: "metadata"

    def sampler(logits, sampling_metadata):
        clock.ms(0.7)
        return "sampled"

    runner.sampler = sampler
    if timer is not None:
        runner.step_timer = timer
    return runner


def test_the_sampler_and_its_launch_are_two_spans(timer, clock, monkeypatch):
    marks = []

    def mark_step():
        marks.append("mark_step")
        clock.ms(0.25)

    monkeypatch.setattr(hpu_model_runner.htorch.core, "mark_step", mark_step, raising=False)
    runner = sampling_runner(timer, clock, marks)
    timer.begin_step()
    timer.tag("decode", 1)
    assert HPUModelRunner._run_sampling(runner, False, torch.zeros(1, 16), ["a"], 1) == ("sampled", "metadata")
    timer.end_step()
    record = records(timer)[0]
    assert p50(record, "sample") == pytest.approx(0.25 + 0.7)  # the mark_step in front of it, and the sampler
    assert p50(record, "launch_sample") == pytest.approx(0.25)
    assert "device_sample" not in record["spans_ms"]
    assert marks == ["mark_step", "mark_step"]


def test_mode_3_waits_for_the_device_after_the_sampler(tmp_path, clock, monkeypatch):
    calls = []
    monkeypatch.setattr(hpu_model_runner.htorch.core, "mark_step", lambda: calls.append("mark_step"), raising=False)

    def synchronize():
        calls.append("synchronize")
        clock.ms(1.25)

    fake_torch = SimpleNamespace(hpu=SimpleNamespace(synchronize=synchronize))
    timer = StepTimer("worker", mode=step_timing.MODE_SYNC, every=1, out_dir=str(tmp_path), clock=clock,
                      wall=clock.wall)
    timer.bind_runtime(hpu_model_runner.htorch, fake_torch)
    runner = sampling_runner(timer, clock, calls)
    timer.begin_step()
    timer.tag("decode", 1)
    HPUModelRunner._run_sampling(runner, False, torch.zeros(1, 16), ["a"], 1)
    timer.end_step()
    record = records(timer)[0]
    assert p50(record, "device_sample") == pytest.approx(1.25)
    assert calls == ["mark_step", "mark_step", "mark_step", "synchronize"]


def test_without_the_flag_the_sampler_runs_unchanged(no_timer, clock, monkeypatch):
    marks = []
    monkeypatch.setattr(hpu_model_runner.htorch.core, "mark_step", lambda: marks.append("mark_step"), raising=False)
    runner = sampling_runner(None, clock, marks)
    assert HPUModelRunner._run_sampling(runner, False, torch.zeros(1, 16), ["a"], 1) == ("sampled", "metadata")
    assert marks == ["mark_step", "mark_step"]


def test_sample_tokens_calls_the_timer_at_every_boundary_of_a_step():
    """The whole method needs a loaded model; its boundaries are checked in the source, in step order."""
    source = inspect.getsource(HPUModelRunner.sample_tokens)
    order = [
        'step_timer = None if warmup_mode else self.step_timer',
        'step_timer.tag(step_kind, num_reqs, scheduler_output.total_num_scheduled_tokens)',
        'step_timer.lap("prepare")',
        'step_timer.lap("launch_inputs")',  # the prefill loop
        'step_timer.device_wait("device_forward", (logits_device, sample_hidden_states))',
        'step_timer.bucket(decode_data.token_ids.size(0))',
        'step_timer.lap("launch_inputs")',  # the decode batch
        'step_timer.lap("launch_model")',
        'step_timer.device_wait("device_forward", (logits_device, fused_sampled_token_ids, sample_hidden_states))',
        'step_timer.lap("wait")',  # a verification step reads the accepted tokens here
        'step_timer.lap("wait")',  # every other step reads the sampled tokens here
        'step_timer.lap("draft")',
    ]
    position = 0
    for text in order:
        found = source.find(text, position)
        assert found >= 0, text
        position = found + len(text)
    assert source.count("if step_timer is not None:") >= 10


def test_the_async_output_wait_is_credited_to_the_timer(timer, clock, monkeypatch):
    monkeypatch.setattr(torch, "hpu", MagicMock(name="torch.hpu"), raising=False)
    model_runner_output = SimpleNamespace(sampled_token_ids=[[], []])
    monkeypatch.setattr(hpu_model_runner.time, "perf_counter_ns", clock)
    output = AsyncHPUModelRunnerOutput(model_runner_output, torch.tensor([[7], [9]]), [], MagicMock())
    output._async_copy_ready_event.synchronize = lambda: clock.ms(6.0)
    assert output.get_output().sampled_token_ids == [[7], [9]]
    timer.begin_step()
    timer.tag("decode", 2)
    timer.end_step()
    assert records(timer)[0]["nested_ms"]["async_wait"]["p50"] == pytest.approx(6.0)


# ------------------------------------------------------------------------------------- communicator
@pytest.fixture
def collectives(monkeypatch):
    seen = []
    monkeypatch.setattr(hpu_communicator.htorch.core, "mark_step", lambda: seen.append("mark_step"), raising=False)
    monkeypatch.setattr(hpu_communicator.dist, "all_reduce", lambda t, group=None: seen.append("all_reduce"))
    monkeypatch.setattr(hpu_communicator.dist, "all_gather_into_tensor",
                        lambda out, t, group=None: seen.append("all_gather"))
    return seen


def communicator(world_size=2):
    return SimpleNamespace(device_group=object(), world_size=world_size)


def test_collectives_of_a_step_are_counted(timer, clock, collectives, monkeypatch):
    monkeypatch.delenv("VLLM_HPU_COLLECTIVE_MARK_STEP", raising=False)
    hidden = torch.ones(8, 2560, dtype=torch.bfloat16)
    hpu_communicator.HpuCommunicator.all_reduce(communicator(), hidden)  # outside a step: not counted
    timer.begin_step()
    timer.tag("decode", 8)
    for _ in range(3):
        assert hpu_communicator.HpuCommunicator.all_reduce(communicator(), hidden) is hidden
    logits = hpu_communicator.HpuCommunicator.all_gather(communicator(world_size=8), torch.ones(8, 4))
    timer.lap("forward")
    timer.end_step()
    record = records(timer)[0]
    assert record["counters"]["all_reduce"]["p50"] == 3
    assert record["counters"]["all_reduce_bytes"]["p50"] == 3 * 8 * 2560 * 2
    assert record["counters"]["all_gather"]["p50"] == 1
    assert record["counters"]["all_gather_bytes"]["p50"] == 8 * 32 * 4  # the gathered tensor, float32
    assert logits.shape == (8, 32)
    assert collectives == ["mark_step", "all_reduce"] * 4 + ["mark_step", "all_gather"]


def test_mode_3_waits_for_the_device_around_every_collective(tmp_path, clock, collectives, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_COLLECTIVE_MARK_STEP", "0")

    def synchronize():
        collectives.append("synchronize")
        clock.ms(2.0)

    fake_torch = SimpleNamespace(hpu=SimpleNamespace(synchronize=synchronize))
    timer = StepTimer("worker", mode=step_timing.MODE_SYNC, every=1, out_dir=str(tmp_path), clock=clock,
                      wall=clock.wall)
    timer.bind_runtime(hpu_communicator.htorch, fake_torch)
    monkeypatch.setattr(step_timing, "TIMER", timer)
    timer.begin_step()
    timer.tag("decode", 1)
    hpu_communicator.HpuCommunicator.all_reduce(communicator(), torch.ones(1, 8))
    hpu_communicator.HpuCommunicator.all_gather(communicator(), torch.ones(1, 8))
    timer.end_step()
    assert collectives == ["mark_step", "synchronize", "all_reduce", "mark_step", "synchronize", "mark_step",
                           "synchronize", "all_gather", "mark_step", "synchronize"]
    record = records(timer)[0]
    assert record["segments"]["names"] == ["all_reduce", "all_gather"]
    assert record["segments"]["compute_p50_ms"] == [2.0, 2.0]
    assert record["segments"]["collective_p50_ms"] == [2.0, 2.0]


def test_without_the_flag_a_collective_is_the_original_two_calls(no_timer, collectives, monkeypatch):
    monkeypatch.delenv("VLLM_HPU_COLLECTIVE_MARK_STEP", raising=False)
    hpu_communicator.HpuCommunicator.all_reduce(communicator(), torch.ones(1, 8))
    out = hpu_communicator.HpuCommunicator.all_gather(communicator(), torch.ones(1, 8))
    assert collectives == ["mark_step", "all_reduce", "mark_step", "all_gather"]
    assert out.shape == (1, 16)


# ------------------------------------------------------------------------------------- engine core
class FakeFuture:

    def __init__(self, clock, value, wait_ms):
        self.clock = clock
        self.value = value
        self.wait_ms = wait_ms

    def result(self, timeout=None):
        self.clock.ms(self.wait_ms)
        return self.value


class FakeEngineCore:
    """``EngineCore.step`` of the pinned vLLM, with stand-ins for the scheduler and the executor."""

    def __init__(self, clock, spec=False, tokens_per_request=1):
        self.clock = clock
        self.requests = True
        scheduled = {"a": tokens_per_request, "b": tokens_per_request}
        output = SimpleNamespace(num_scheduled_tokens=scheduled, total_num_scheduled_tokens=sum(scheduled.values()),
                                 scheduled_spec_decode_tokens={"a": [1, 2]} if spec else {})

        def schedule(throttle=False):
            clock.ms(0.3)
            return output

        def update_from_output(scheduler_output, model_output):
            clock.ms(0.4)
            return {0: "outputs"}

        def execute_model(scheduler_output, non_block=False):
            clock.ms(0.05)
            return FakeFuture(clock, None, 0.2)

        def sample_tokens(grammar_output, non_block=False):
            clock.ms(30.0)
            return "model output"

        self.scheduler = SimpleNamespace(has_requests=lambda: self.requests, schedule=schedule,
                                         get_grammar_bitmask=lambda scheduler_output: None,
                                         update_from_output=update_from_output)
        self.model_executor = SimpleNamespace(execute_model=execute_model, sample_tokens=sample_tokens)

    def step(self):
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(False)
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        self.clock.ms(0.01)
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        return engine_core_outputs, True


@pytest.fixture
def engine_env(monkeypatch, tmp_path, clock):
    """A process whose 'vllm.v1.engine.core' holds a fresh copy of the stand-in engine core class."""
    engine_class = type("EngineCore", (FakeEngineCore, ), {})
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", SimpleNamespace(EngineCore=engine_class))
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING_EVERY", "1")
    timers = []

    def make_timer(mode, every, out_dir):
        timers.append(StepTimer("engine", mode=mode, every=every, out_dir=out_dir, clock=clock, wall=clock.wall))
        return timers[-1]

    return engine_class, make_timer, timers


def test_the_engine_core_step_is_cut_at_its_four_calls(engine_env, clock, monkeypatch):
    engine_class, make_timer, timers = engine_env
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING", "1")
    assert step_timing.install_engine_step_timing(make_timer=make_timer) is True
    assert step_timing.install_engine_step_timing(make_timer=make_timer) is True  # a second call changes nothing
    engine = engine_class(clock)
    for _ in range(2):
        assert engine.step() == ({0: "outputs"}, True)
        clock.ms(0.6)  # the engine loop between two steps
    assert len(timers) == 1
    first, second = records(timers[0])
    assert p50(first, "schedule") == pytest.approx(0.3)
    assert p50(first, "execute_model_send") == pytest.approx(0.05)
    assert p50(first, "execute_model_wait") == pytest.approx(0.2)
    assert p50(first, "sample_tokens_rpc") == pytest.approx(30.0)
    assert p50(first, "update_from_output") == pytest.approx(0.4)
    assert p50(first, "other") == pytest.approx(0.01)
    assert p50(first, "step") == pytest.approx(30.96)
    assert (first["role"], first["kind"], first["rows"]) == ("engine", "decode", 2)
    assert first["counters"]["tokens"]["p50"] == 2
    assert second["raw"]["gap"] == [0.6]


def test_the_engine_tells_verification_and_prefill_steps_apart(engine_env, clock, monkeypatch):
    engine_class, make_timer, timers = engine_env
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING", "1")
    step_timing.install_engine_step_timing(make_timer=make_timer)
    engine_class(clock, spec=True, tokens_per_request=4).step()
    engine_class(clock, tokens_per_request=2048).step()
    timers[0].flush()
    kinds = [record["kind"] for record in records(timers[1])]  # two engines of one process share the file
    assert kinds == ["verify", "prefill"]


def test_an_engine_step_without_requests_is_not_recorded(engine_env, clock, monkeypatch):
    engine_class, make_timer, timers = engine_env
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING", "1")
    step_timing.install_engine_step_timing(make_timer=make_timer)
    engine = engine_class(clock)
    engine.requests = False
    assert engine.step() == ({}, False)
    timers[0].flush()
    assert not os.path.exists(timers[0].path)


def test_without_the_flag_the_engine_core_is_not_touched(engine_env, clock, monkeypatch):
    engine_class, make_timer, timers = engine_env
    monkeypatch.delenv("VLLM_HPU_STEP_TIMING", raising=False)
    assert step_timing.install_engine_step_timing(make_timer=make_timer) is False
    assert engine_class.step is FakeEngineCore.step
    from vllm_gaudi import patches
    patches._patch_engine_core_step_timing()
    assert engine_class.step is FakeEngineCore.step and timers == []


def test_a_timer_that_cannot_be_built_leaves_the_engine_running(engine_env, clock, monkeypatch):
    engine_class, _make_timer, _timers = engine_env
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING", "1")

    def broken(mode, every, out_dir):
        raise RuntimeError("no timer today")

    said = []
    step_timing.install_engine_step_timing(log=said.append, make_timer=broken)
    engine = engine_class(clock)
    assert engine.step() == ({0: "outputs"}, True)
    assert engine.step() == ({0: "outputs"}, True)
    assert sum("the engine core is not timed" in line for line in said) == 1


def test_the_pinned_engine_core_has_the_calls_the_timer_wraps():
    """The wrapper relies on names of the pinned vLLM; this fails when an upgrade renames one of them."""
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.multiproc_executor import FutureWrapper

    for method in ("step", "step_with_batch_queue"):
        assert callable(getattr(EngineCore, method))
    source = inspect.getsource(EngineCore.step)
    for call in ("self.scheduler.schedule(", "self.model_executor.execute_model(", "future.result()",
                 "self.model_executor.sample_tokens(", "self.scheduler.update_from_output("):
        assert call in source, call
    assert "load_general_plugins()" in inspect.getsource(EngineCore.__init__)
    assert not hasattr(FutureWrapper, "__slots__")  # its result method can be wrapped on one object
