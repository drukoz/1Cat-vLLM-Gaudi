# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_STEP_TIMING: the step timer itself (vllm_gaudi/extension/step_timing.py).

The timer cuts a step into spans with one clock read per boundary, groups steps by kind and row count, and writes
one log line and one JSON record per N steps. These tests drive it with a hand clock, so every number is exact.
"""

import json

import pytest
import torch

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()

from vllm_gaudi.extension import step_timing  # noqa: E402
from vllm_gaudi.extension.step_timing import StepTimer  # noqa: E402

MS = 1_000_000  # nanoseconds


class Clock:
    """A clock the test moves by hand; ``wall`` follows it."""

    def __init__(self):
        self.ns = 10 * 1_000_000_000

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
def lines():
    return []


def make(tmp_path, clock, lines, mode=step_timing.MODE_SPANS, every=2, role="worker", rank=0):
    return StepTimer(role, rank=rank, mode=mode, every=every, out_dir=str(tmp_path), log=lines.append, clock=clock,
                     wall=clock.wall)


def records(timer):
    with open(timer.path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def one_step(timer, clock, spans, kind="decode", rows=1, tokens=None):
    """Run one step whose spans take the given milliseconds, in order."""
    timer.begin_step()
    timer.tag(kind, rows, tokens)
    for name, milliseconds in spans:
        clock.ms(milliseconds)
        timer.lap(name)
    timer.end_step()


def test_the_laps_partition_the_step(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=1)
    timer.begin_step()
    clock.ms(0.5)
    timer.lap("execute_model")
    timer.tag("decode", 8, tokens=8)
    clock.ms(2.0)
    timer.lap("forward")
    clock.ms(1.0)
    timer.lap("forward")  # a span may be entered twice in one step: the parts add up
    clock.ms(4.0)
    timer.lap("wait")
    clock.ms(0.25)
    timer.end_step()
    record = records(timer)[0]
    spans = record["spans_ms"]
    assert spans["execute_model"]["p50"] == pytest.approx(0.5)
    assert spans["forward"]["p50"] == pytest.approx(3.0)
    assert spans["wait"]["p50"] == pytest.approx(4.0)
    assert spans["post"]["p50"] == pytest.approx(0.25)
    assert spans["step"]["p50"] == pytest.approx(7.75)
    parts = sum(spans[name]["p50"] for name in spans if name not in ("step", "gap", "period"))
    assert parts == pytest.approx(spans["step"]["p50"])
    assert (record["kind"], record["rows"], record["steps"]) == ("decode", 8, 1)
    assert record["counters"]["tokens"]["p50"] == 8


def test_a_step_that_was_never_tagged_is_dropped(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=1)
    timer.begin_step()
    clock.ms(1.0)
    timer.lap("execute_model")
    timer.end_step()
    timer.flush()
    assert not (tmp_path / timer.path).exists()
    assert lines == []


def test_laps_outside_a_step_are_ignored(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=1)
    timer.lap("forward")
    timer.count("all_reduce")
    timer.add("collective_host", 5)
    timer.tag("decode", 1)
    timer.end_step()
    assert timer.active is False
    timer.flush()
    assert lines == []


def test_steps_are_grouped_by_kind_and_rows_and_flushed_every_n(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=2)
    one_step(timer, clock, [("forward", 1.0)], rows=1)
    one_step(timer, clock, [("forward", 10.0)], rows=8)
    assert lines == []  # neither group has two steps yet
    one_step(timer, clock, [("forward", 3.0)], rows=1)
    assert len(lines) == 1 and "rows=1" in lines[0] and "steps=2" in lines[0]
    one_step(timer, clock, [("forward", 5.0)], kind="prefill", rows=1, tokens=2048)
    timer.flush()
    got = {(r["kind"], r["rows"]): r for r in records(timer)}
    assert sorted(got) == [("decode", 1), ("decode", 8), ("prefill", 1)]
    assert got[("decode", 1)]["raw"]["forward"] == [1.0, 3.0]
    assert got[("decode", 8)]["steps"] == 1
    assert got[("prefill", 1)]["counters"]["tokens"]["p50"] == 2048


def test_gap_is_the_time_between_two_steps_and_period_adds_it(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=3)
    one_step(timer, clock, [("forward", 5.0)])
    clock.ms(1.5)
    one_step(timer, clock, [("forward", 6.0)])
    clock.ms(2.5)
    one_step(timer, clock, [("forward", 7.0)])
    record = records(timer)[0]
    assert record["raw"]["gap"] == [None, 1.5, 2.5]  # the first step follows an idle process: no gap
    assert record["raw"]["period"] == [None, 7.5, 9.5]
    assert record["spans_ms"]["gap"]["n"] == 2


def test_an_idle_gap_is_not_a_gap_and_flushes_what_is_collected(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=100)
    one_step(timer, clock, [("forward", 5.0)])
    clock.ms(1.0)
    one_step(timer, clock, [("forward", 5.0)])
    assert lines == []
    clock.ms(3000.0)  # the server was idle
    one_step(timer, clock, [("forward", 5.0)])
    assert len(lines) == 1 and "steps=2" in lines[0]
    timer.flush()
    assert [r["raw"]["gap"] for r in records(timer)] == [[None, 1.0], [None]]


def test_percentiles_are_nearest_rank(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=10)
    for milliseconds in range(1, 11):
        one_step(timer, clock, [("forward", float(milliseconds))])
    forward = records(timer)[0]["spans_ms"]["forward"]
    assert (forward["p50"], forward["p90"], forward["min"], forward["max"], forward["n"]) == (5.0, 9.0, 1.0, 10.0, 10)
    assert forward["mean"] == pytest.approx(5.5)


def test_the_log_line_gives_p50_and_p90_per_span_in_step_order(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=2)
    for forward in (2.0, 4.0):
        timer.begin_step()
        timer.tag("decode", 32)
        timer.bucket(32)
        clock.ms(0.5)
        timer.lap("prepare")
        clock.ms(forward)
        timer.lap("forward")
        clock.ms(1.0)
        timer.lap("wait")
        timer.count("all_reduce", 98)
        timer.end_step()
    line = lines[0]
    assert line.startswith("step timing: worker rank 0 decode rows=32 bucket=32 steps=2 mode=1 | ")
    assert "prepare 0.50/0.50 forward 2.00/4.00" in line and "wait 1.00/1.00" in line
    assert line.index("prepare") < line.index("forward") < line.index("wait")
    assert "all_reduce 98" in line
    assert line.endswith("(ms, p50/p90)")


def test_a_span_that_only_some_steps_have_is_none_in_the_others(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=3)
    one_step(timer, clock, [("forward", 1.0)])
    one_step(timer, clock, [("forward", 1.0), ("draft", 2.0)])
    one_step(timer, clock, [("forward", 1.0)])
    record = records(timer)[0]
    assert record["raw"]["draft"] == [None, 2.0, None]
    assert record["spans_ms"]["draft"]["n"] == 1
    assert all(len(column) == 3 for column in record["raw"].values())


def test_nested_time_is_reported_next_to_the_spans(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=1)
    timer.begin_step()
    timer.tag("decode", 1)
    clock.ms(3.0)
    timer.add("collective_host", 2 * MS)
    timer.lap("forward")
    timer.end_step()
    record = records(timer)[0]
    assert record["nested_ms"]["collective_host"]["p50"] == pytest.approx(2.0)
    assert record["spans_ms"]["step"]["p50"] == pytest.approx(3.0)  # nested time is not added to the step


def test_time_from_another_thread_goes_to_the_step_that_ends_next(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=1)
    timer.add_async("async_wait", 4 * MS)
    one_step(timer, clock, [("forward", 1.0)])
    assert records(timer)[0]["nested_ms"]["async_wait"]["p50"] == pytest.approx(4.0)


def test_collectives_are_counted_with_their_bytes_and_host_time(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, every=1)
    timer.begin_step()
    timer.tag("decode", 1)
    payload = torch.zeros(1, 2560, dtype=torch.bfloat16)
    for _ in range(3):
        token = timer.collective_begin()
        clock.ms(0.01)
        timer.collective_end(token, "all_reduce", payload)
    token = timer.collective_begin()
    clock.ms(0.02)
    timer.collective_end(token, "all_gather", torch.zeros(1, 248320, dtype=torch.bfloat16))
    timer.lap("forward")
    timer.end_step()
    record = records(timer)[0]
    assert record["counters"]["all_reduce"]["p50"] == 3
    assert record["counters"]["all_reduce_bytes"]["p50"] == 3 * 2560 * 2
    assert record["counters"]["all_gather_bytes"]["p50"] == 248320 * 2
    assert record["nested_ms"]["collective_host"]["p50"] == pytest.approx(0.05)
    assert "segments" not in record


class FakeRuntime:
    """Stands in for (htorch, torch): mark_step and synchronize are recorded and take time on the hand clock."""

    def __init__(self, clock, calls, device_ms):
        self.clock = clock
        self.calls = calls
        self.device_ms = list(device_ms)
        self.core = self
        self.hpu = self

    def mark_step(self):
        self.calls.append("mark_step")

    def synchronize(self):
        self.calls.append("synchronize")
        self.clock.ms(self.device_ms.pop(0) if self.device_ms else 0.0)


def test_mode_3_measures_every_segment_and_every_collective_on_its_own(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_SYNC, every=1)
    calls = []
    # the waits, in order: compute before collective 0, collective 0, compute before 1, collective 1, the rest
    runtime = FakeRuntime(clock, calls, [3.0, 0.4, 5.0, 0.6, 1.5])
    timer.bind_runtime(runtime, runtime)
    timer.begin_step()
    timer.tag("decode", 8)
    timer.phase = "forward"
    for name in ("all_reduce", "all_gather"):
        clock.ms(0.2)  # host accumulation in front of the collective
        token = timer.collective_begin()
        timer.collective_end(token, name)
    clock.ms(0.1)
    timer.lap("forward")
    timer.device_wait("device_forward")
    timer.end_step()
    record = records(timer)[0]
    spans = record["spans_ms"]
    assert spans["forward"]["p50"] == pytest.approx(0.5)  # host time only: 0.2 + 0.2 + 0.1
    assert spans["seg_compute"]["p50"] == pytest.approx(8.0)
    assert spans["seg_collective"]["p50"] == pytest.approx(1.0)
    assert spans["device_forward"]["p50"] == pytest.approx(1.5)
    assert record["segments"] == {
        "names": ["all_reduce", "all_gather"],
        "compute_p50_ms": [3.0, 5.0],
        "collective_p50_ms": [0.4, 0.6],
        "steps": 1,
    }
    assert calls == ["mark_step", "synchronize"] * 5
    assert record["counters"]["all_reduce"]["p50"] == 1


def test_device_wait_does_nothing_outside_mode_3(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_SPANS, every=1)
    calls = []
    runtime = FakeRuntime(clock, calls, [])
    timer.bind_runtime(runtime, runtime)
    timer.begin_step()
    timer.tag("decode", 1)
    token = timer.collective_begin()
    timer.collective_end(token, "all_reduce")
    timer.device_wait("device_forward")
    timer.end_step()
    assert calls == []
    assert "device_forward" not in records(timer)[0]["spans_ms"]


def test_mode_2_counts_host_synchronisation_points_with_their_source_lines(tmp_path, clock, lines, monkeypatch):
    original_item, original_to = torch.Tensor.item, torch.Tensor.to
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_COUNTERS, every=1)
    # No device on this machine: int16 tensors play the device, everything else is the host.
    monkeypatch.setattr(timer, "_on_device", lambda tensor: tensor.dtype == torch.int16)
    device = torch.arange(4, dtype=torch.int16)
    host = torch.arange(4, dtype=torch.int32)
    device[0].item()  # outside a step: not counted
    timer.begin_step()
    timer.tag("decode", 1)
    device[0].item()
    device.tolist()
    device.cpu()
    device.cpu()
    host.cpu()  # a host tensor: not a synchronisation point
    host.tolist()
    host.to(torch.int16)  # host to device
    device.to(torch.int32)  # device to host
    device.to(torch.int16)  # stays on the device
    device.clone().copy_(host)  # host to device
    host.clone().copy_(device)  # device to host
    if device[1]:  # a tensor used as a Python bool
        pass
    timer.end_step()
    record = records(timer)[0]
    counters = {name: summary["p50"] for name, summary in record["counters"].items()}
    assert counters == {"item": 1, "tolist": 1, "cpu": 2, "h2d": 2, "d2h_to": 1, "d2h_copy": 1, "scalar": 1}
    sites = {(site["what"], site["where"].split(":")[0]) for site in record["sync_sites"]}
    assert ("cpu", "test_step_timing.py") in sites and ("h2d", "test_step_timing.py") in sites
    timer.close()
    assert torch.Tensor.item is original_item and torch.Tensor.to is original_to


def test_mode_2_counts_mark_step_and_synchronize_of_the_bound_runtime(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_COUNTERS, every=1)
    calls = []
    runtime = FakeRuntime(clock, calls, [])
    timer.bind_runtime(runtime, runtime)
    timer.begin_step()
    timer.tag("decode", 1)
    runtime.core.mark_step()
    runtime.core.mark_step()
    runtime.hpu.synchronize()
    timer.end_step()
    counters = records(timer)[0]["counters"]
    assert counters["mark_step"]["p50"] == 2 and counters["synchronize"]["p50"] == 1
    assert calls == ["mark_step", "mark_step", "synchronize"]  # the calls themselves still happen
    timer.close()
    runtime.core.mark_step()
    assert calls[-1] == "mark_step"


def test_mode_4_counts_the_ops_dispatched_inside_a_block(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_OPS, every=1)
    x = torch.ones(2, 4)
    timer.begin_step()
    timer.tag("decode", 2)
    with timer.ops("forward"):
        y = x * 2.0 + 1.0
        y = torch.nn.functional.silu(y)
    with timer.ops("logits"):
        y.sum()
    timer.end_step()
    record = records(timer)[0]
    assert record["counters"]["ops_forward"]["p50"] == 3
    assert record["counters"]["ops_logits"]["p50"] == 1
    assert record["op_names_per_step"]["aten::mul"] == 1.0 and record["op_names_per_step"]["aten::silu"] == 1.0


def test_ops_is_a_null_context_in_the_other_modes_and_without_a_timer(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_SPANS, every=1)
    timer.begin_step()
    assert timer.ops("forward") is step_timing.ops(None, "forward")
    with step_timing.ops(timer, "forward"):
        pass
    timer.discard_step()


def test_the_mode_file_is_reread_when_a_step_starts_after_an_idle_gap(tmp_path, clock, lines):
    timer = make(tmp_path, clock, lines, mode=step_timing.MODE_SPANS, every=100)
    one_step(timer, clock, [("forward", 1.0)])
    (tmp_path / step_timing.MODE_FILE).write_text("3\n", encoding="utf-8")
    clock.ms(1.0)
    one_step(timer, clock, [("forward", 1.0)])
    assert timer.mode == step_timing.MODE_SPANS  # no idle gap yet: the file is not read
    clock.ms(2500.0)
    one_step(timer, clock, [("forward", 1.0)])
    assert timer.mode == step_timing.MODE_SYNC
    assert any("mode 1 -> 3" in line for line in lines)
    (tmp_path / step_timing.MODE_FILE).write_text("0", encoding="utf-8")
    clock.ms(2500.0)
    one_step(timer, clock, [("forward", 1.0)])  # off: nothing is recorded
    assert timer.mode == step_timing.MODE_OFF and timer.active is False
    (tmp_path / step_timing.MODE_FILE).write_text("1", encoding="utf-8")
    clock.ms(2500.0)
    one_step(timer, clock, [("forward", 1.0)])
    timer.flush()
    modes = [(r["mode"], r["steps"]) for r in records(timer)]
    assert modes == [(1, 2), (3, 1), (1, 1)]


@pytest.mark.parametrize("text, mode", [("0", 0), ("1", 1), (" 2\n", 2), ("3", 3), ("4", 4), ("5", 0), ("-1", 0),
                                        ("on", 0), ("", 0), (None, 0)])
def test_parse_mode(text, mode):
    assert step_timing.parse_mode(text) == mode


def test_an_unwritable_folder_stops_the_file_but_not_the_log(tmp_path, clock, lines):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    timer = StepTimer("worker", mode=1, every=1, out_dir=str(blocker / "sub"), log=lines.append, clock=clock,
                      wall=clock.wall)
    one_step(timer, clock, [("forward", 1.0)])
    one_step(timer, clock, [("forward", 1.0)])
    assert sum("cannot write" in line for line in lines) == 1
    assert sum(line.startswith("step timing: worker rank 0 decode") for line in lines) == 2


def test_init_timer_is_off_by_default(tmp_path, monkeypatch):
    for name in ("VLLM_HPU_STEP_TIMING", "VLLM_HPU_STEP_TIMING_EVERY", "VLLM_HPU_STEP_TIMING_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(step_timing, "TIMER", None)
    assert step_timing.init_timer("worker", 0) is None
    assert step_timing.TIMER is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["0", "", "7", "true"])
def test_init_timer_stays_off_for_zero_and_unknown_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING", value)
    monkeypatch.setattr(step_timing, "TIMER", None)
    assert step_timing.init_timer("worker", 0) is None


def test_init_timer_reads_the_three_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING", "2")
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING_EVERY", "7")
    monkeypatch.setenv("VLLM_HPU_STEP_TIMING_DIR", str(tmp_path / "timing"))
    monkeypatch.setattr(step_timing, "TIMER", None)
    said = []
    timer = step_timing.init_timer("worker", 3, log=said.append)
    try:
        assert timer is step_timing.TIMER
        assert (timer.mode, timer.every, timer.rank, timer.role) == (2, 7, 3, "worker")
        assert timer.path.startswith(str(tmp_path / "timing")) and "step_timing_worker_r3_" in timer.path
        assert said and said[0].startswith("step timing: on, mode 2, every 7 steps")
    finally:
        timer.close()
        monkeypatch.setattr(step_timing, "TIMER", None)


def test_the_plugin_documents_the_three_variables():
    from vllm_gaudi import envs

    for name in ("VLLM_HPU_STEP_TIMING", "VLLM_HPU_STEP_TIMING_EVERY", "VLLM_HPU_STEP_TIMING_DIR"):
        assert name in envs.environment_variables
    assert envs.environment_variables["VLLM_HPU_STEP_TIMING"]() in (0, 1, 2, 3, 4)
