# SPDX-License-Identifier: Apache-2.0
"""summarize_timing.py reads what the step timer writes and matches it to the probe's rounds."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "flash_next" / "anatomy"))

import summarize_timing  # noqa: E402
from vllm_gaudi.extension.step_timing import MODE_SPANS, StepTimer  # noqa: E402


class FakeClock:

    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance(self, ms):
        self.ns += int(ms * 1e6)


def _run_worker(out_dir, rank, rows, steps, wall_t, forward_ms):
    clock = FakeClock()
    timer = StepTimer("worker", rank=rank, mode=MODE_SPANS, every=10_000, out_dir=str(out_dir), clock=clock,
                      wall=lambda: wall_t, idle_s=2.0)
    for _ in range(steps):
        timer.begin_step()
        timer.tag("decode", rows, rows)
        clock.advance(1.0)
        timer.lap("prepare")
        clock.advance(forward_ms)
        timer.lap("forward")
        for _ in range(3):
            token = timer.collective_begin()
            timer.collective_end(token, "all_reduce")
        clock.advance(20.0)
        timer.lap("wait")
        timer.end_step()
        clock.advance(5.0)  # the engine's turn: the next step's gap
    timer.close()


def test_summarize_matches_rounds_and_ranks(tmp_path):
    timing = tmp_path / "timing"
    _run_worker(timing, 0, 4, 30, 1000.5, 2.0)
    _run_worker(timing, 1, 4, 30, 1000.5, 3.0)
    _run_worker(timing, 0, 1, 3, 1000.7, 2.0)  # too few steps: not a steady record
    engine_clock = FakeClock()
    engine = StepTimer("engine", rank=0, mode=MODE_SPANS, every=10_000, out_dir=str(timing), clock=engine_clock,
                       wall=lambda: 1000.6)
    for _ in range(25):
        engine.begin_step()
        engine.tag("decode", 4, 4)
        engine_clock.advance(0.5)
        engine.lap("schedule")
        engine_clock.advance(24.0)
        engine.lap("execute_model_rpc")
        engine.end_step("other")
    engine.close()
    probe = {"phases": [{"mode": "1", "rounds": [{"streams": 4, "tokens": 32, "t_start": 1000.0, "t_end": 1001.0,
                                                   "ms_per_token_p50": 28.1}]}], "traces": []}
    (tmp_path / "probe.json").write_text(json.dumps(probe))

    result = summarize_timing.summarize(str(tmp_path))
    assert [c["label"] for c in result["columns"]] == ["1/N=4"]
    col = result["columns"][0]
    assert col["ranks"] == [0, 1] and col["steps"] == 30
    assert abs(col["worker"]["forward"] - 2.0) < 1e-6
    assert abs(col["worker_rank_max"]["forward"] - 3.0) < 1e-6
    assert abs(col["worker"]["gap"] - 5.0) < 1e-6
    assert col["counters"]["all_reduce"] == 3
    assert abs(col["engine"]["execute_model_rpc"] - 24.0) < 1e-6
    text = summarize_timing.markdown(result)
    assert "| worker forward |" in text and "| engine schedule |" in text and "28.10" in text


def test_summarize_without_probe_groups_by_mode_and_rows(tmp_path):
    _run_worker(tmp_path / "timing", 0, 8, 25, 50.0, 4.0)
    result = summarize_timing.summarize(str(tmp_path))
    assert [c["label"] for c in result["columns"]] == ["mode1/N=8"]
    assert summarize_timing.main([str(tmp_path)]) == 0


def test_summarize_reads_records_in_the_run_folder_itself(tmp_path):
    _run_worker(tmp_path, 0, 32, 30, 70.0, 3.0)  # a farm-serve server writes into its working directory
    result = summarize_timing.summarize(str(tmp_path))
    assert [c["label"] for c in result["columns"]] == ["mode1/N=32"]


class _Runtime:
    """mark_step and synchronize of a device that needs 2 ms per wait."""

    def __init__(self, clock):
        self.clock = clock
        self.core = self
        self.hpu = self

    def mark_step(self):
        pass

    def synchronize(self):
        self.clock.advance(2.0)


def test_summarize_tabulates_the_components_of_a_mode_3_round(tmp_path):
    from vllm_gaudi.extension.step_timing import MODE_SYNC

    for rank, host_ms in ((0, 0.5), (1, 0.7)):
        clock = FakeClock()
        timer = StepTimer("worker", rank=rank, mode=MODE_SYNC, every=10_000, out_dir=str(tmp_path / "timing"),
                          clock=clock, wall=lambda: 2000.5, idle_s=2.0)
        timer.bind_runtime(_Runtime(clock), _Runtime(clock))
        for _ in range(25):
            timer.begin_step()
            timer.tag("decode", 32, 32)
            for name in ("hc", "qsa"):
                timer.component_begin(name)
                clock.advance(host_ms)
                timer.component_end()
            timer.end_step()
            clock.advance(5.0)
        timer.close()
    probe = {"phases": [{"mode": "3", "min_tokens": True, "ctx": 4000,
                         "rounds": [{"streams": 32, "tokens": 48, "t_start": 2000.0, "t_end": 2001.0,
                                     "ms_per_token_p50": 300.0}]}], "traces": []}
    (tmp_path / "probe.json").write_text(json.dumps(probe))
    result = summarize_timing.summarize(str(tmp_path))
    col = result["columns"][0]
    assert col["label"] == "3/N=32/min/ctx=4000"
    assert col["components"]["qsa"] == {"host": 0.5, "device": 2.0, "collective": 0.0, "calls": 1}
    assert col["components_rank_max"]["qsa"]["host"] == 0.7
    text = summarize_timing.markdown(result)
    assert "| qsa | 1 | 2.00 (2.00) | 0.00 (0.00) | 0.50 (0.70) |" in text
    assert "| all components | | 8.00 | 0.00 | 1.00 |" in text  # other: the two waits in front of the components
