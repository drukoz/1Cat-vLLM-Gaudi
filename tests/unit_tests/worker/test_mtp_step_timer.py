# SPDX-License-Identifier: Apache-2.0
"""The speculative step's timer (``mtp_step_timer.MtpStepTimer``, VLLM_HPU_MTP_TIMING): its spans, laps and records.

A scripted clock stands in for ``time.perf_counter``, so every span is exact.
"""
import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402

from vllm_gaudi.v1.spec_decode import mtp_step_timer  # noqa: E402
from vllm_gaudi.v1.spec_decode.mtp_step_timer import MtpStepTimer  # noqa: E402


class Clock:
    """``perf_counter`` that advances by the scripted seconds before each reading."""

    def __init__(self):
        self.now = 0.0
        self.pending = 0.0

    def advance(self, milliseconds):
        self.pending += milliseconds / 1000.0

    def __call__(self):
        self.now += self.pending
        self.pending = 0.0
        return self.now


def one_step(timer, clock, verify=40.0, pass0=3.0, pass0_wait=9.0, later=((2.0, 3.0), (2.0, 3.0)), final_wait=20.0,
             rows=4, requests=1, k=3, waits=None):
    timer.step_start()
    clock.advance(verify)
    timer.draft_start("verify", rows, requests, k)
    clock.advance(pass0)
    timer.lap("pass0")
    clock.advance(pass0_wait)
    timer.lap("pass0_wait")
    for index, (meta, issue) in enumerate(later, start=1):
        clock.advance(meta)
        timer.lap("pass%d_meta" % index)
        clock.advance(issue)
        timer.lap("pass%d" % index)
    if waits is not None:
        waits.append(final_wait)
    clock.advance(final_wait)
    timer.draft_end()


def test_spans_laps_and_the_two_step_costs(tmp_path):
    clock = Clock()
    lines = []
    waited = []
    timer = MtpStepTimer(rank=0, log=lambda fmt, *args: lines.append(fmt % args), clock=clock,
                         device_wait=lambda: waited.append(1), every=2, directory=str(tmp_path))
    one_step(timer, clock)
    assert lines == [] and not os.path.exists(timer.path)  # one step of two
    one_step(timer, clock, verify=44.0)
    record = json.loads(open(timer.path).read().splitlines()[0])
    assert record["kind"] == "verify" and record["rows"] == 4 and record["requests"] == 1 and record["k"] == 3
    assert record["steps"] == 2 and record["total_steps"] == 2
    assert record["ms"]["verify"] == pytest.approx(42.0)
    assert record["ms"]["draft"] == pytest.approx(3 + 9 + 2 + 3 + 2 + 3 + 20)
    assert record["ms"]["pass0"] == pytest.approx(3.0)
    assert record["ms"]["pass0_wait"] == pytest.approx(9.0)
    assert record["ms"]["pass1_meta"] == pytest.approx(2.0) and record["ms"]["pass2"] == pytest.approx(3.0)
    assert record["ms"]["final_wait"] == pytest.approx(20.0)
    assert record["Ft_ms"] == pytest.approx(42.0)
    assert record["Fd_ms"] == pytest.approx(42.0 / 3)
    assert len(waited) == 2  # one device wait per step, after the drafter call
    assert len(lines) == 1 and "verify 42.0 ms, draft 42.0 ms" in lines[0] and "pass0_wait 9.0" in lines[0]


def test_records_are_kept_per_row_count_and_start_again_after_a_write(tmp_path):
    clock = Clock()
    timer = MtpStepTimer(clock=clock, every=2, directory=str(tmp_path))
    one_step(timer, clock, verify=10.0, rows=4)
    one_step(timer, clock, verify=100.0, rows=128, requests=32)
    one_step(timer, clock, verify=30.0, rows=4)  # closes the first record of rows=4
    one_step(timer, clock, verify=50.0, rows=4)
    one_step(timer, clock, verify=70.0, rows=4)  # closes the second
    records = [json.loads(line) for line in open(timer.path).read().splitlines()]
    assert [(r["rows"], r["steps"], r["total_steps"]) for r in records] == [(4, 2, 2), (4, 2, 4)]
    assert records[0]["ms"]["verify"] == pytest.approx(20.0)
    assert records[1]["ms"]["verify"] == pytest.approx(60.0)  # the means of the steps since the last write
    assert ("verify", 128, 32, 3) in timer.records and timer.records[("verify", 128, 32, 3)]["steps"] == 1


def test_a_step_without_a_draft_and_laps_outside_a_draft_are_ignored(tmp_path):
    clock = Clock()
    timer = MtpStepTimer(clock=clock, every=1, directory=str(tmp_path))
    timer.lap("pass0")  # no step open
    timer.draft_end()
    timer.step_start()
    clock.advance(5.0)
    timer.step_start()  # the step before made no draft (no speculative config path): its start is dropped
    clock.advance(7.0)
    timer.draft_start("prefill", 128, 1, 3)
    clock.advance(4.0)
    timer.draft_end()
    record = json.loads(open(timer.path).read().splitlines()[0])
    assert record["kind"] == "prefill" and record["ms"]["verify"] == pytest.approx(7.0)
    assert record["ms"]["draft"] == pytest.approx(4.0) and record["ms"]["final_wait"] == pytest.approx(4.0)


def test_from_env_is_none_by_default_and_a_timer_with_the_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_HPU_MTP_TIMING", raising=False)
    assert mtp_step_timer.from_env() is None
    monkeypatch.setenv("VLLM_HPU_MTP_TIMING", "1")
    monkeypatch.chdir(tmp_path)
    timer = mtp_step_timer.from_env(rank=3, is_driver=False, log=print, device_type="cpu")
    assert isinstance(timer, MtpStepTimer) and timer.rank == 3
    assert timer.log is None and timer.device_wait is None  # only the driver logs; a CPU device waits for nothing
    assert os.path.dirname(timer.path) == str(tmp_path)


def test_per_step_records_with_the_tokens_committed_in_the_step_and_the_one_before(tmp_path):
    """VLLM_HPU_MTP_TIMING=2: one line per step; ``prev_committed`` is each request's count of its previous step."""
    clock = Clock()
    timer = MtpStepTimer(clock=clock, every=25, directory=str(tmp_path), per_step=True)

    def step(req_ids, counts, verify, draft, rows=8, kind="verify"):
        timer.step_start()
        timer.step_tokens(req_ids, counts)
        clock.advance(verify)
        timer.draft_start(kind, rows, len(req_ids), 3)
        clock.advance(draft)
        timer.draft_end()

    step(["a", "b"], [3, 1], verify=50.0, draft=12.0)
    step(["a", "b"], [1, 4], verify=70.0, draft=11.0)
    step([], [], verify=30.0, draft=5.0, rows=128, kind="prefill")  # no decodes: the counts of a and b are kept
    step(["a", "c"], [2, 0], verify=55.0, draft=10.0)  # c is new; 0: not known (async scheduling)
    lines = [json.loads(line) for line in open(timer.step_path).read().splitlines()]
    assert [line["step"] for line in lines] == [0, 1, 2, 3]
    assert lines[0]["committed"] == {"3": 1, "1": 1} and lines[0]["prev_committed"] == {"none": 2}
    assert lines[0]["verify_ms"] == pytest.approx(50.0) and lines[0]["draft_ms"] == pytest.approx(12.0)
    assert lines[0]["rows"] == 8 and lines[0]["requests"] == 2 and lines[0]["k"] == 3 and lines[0]["kind"] == "verify"
    assert lines[1]["committed"] == {"1": 1, "4": 1} and lines[1]["prev_committed"] == {"3": 1, "1": 1}
    assert lines[2]["kind"] == "prefill" and lines[2]["committed"] == {} and lines[2]["prev_committed"] == {}
    assert lines[3]["committed"] == {"2": 1, "none": 1} and lines[3]["prev_committed"] == {"1": 1, "none": 1}
    assert lines[3]["verify_ms"] == pytest.approx(55.0)
    assert not os.path.exists(timer.path)  # the 25-step means are written as before, at 25 steps of a key


def test_no_per_step_records_without_the_mode(tmp_path):
    clock = Clock()
    timer = MtpStepTimer(clock=clock, every=1, directory=str(tmp_path))
    timer.step_start()
    timer.step_tokens(["a"], [2])  # ignored without per_step
    timer.draft_start("verify", 4, 1, 3)
    timer.draft_end()
    assert os.path.exists(timer.path) and not os.path.exists(timer.step_path)
    timer.per_step = True
    timer.step_tokens(["a"], [2])  # outside a step: ignored
    assert timer._committed is None


def test_from_env_mode_two_records_every_step(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VLLM_HPU_MTP_TIMING", "1")
    assert mtp_step_timer.from_env(device_type="cpu").per_step is False
    monkeypatch.setenv("VLLM_HPU_MTP_TIMING", "2")
    timer = mtp_step_timer.from_env(rank=2, device_type="cpu")
    assert timer.per_step is True
    assert os.path.basename(timer.step_path) == "mtp_step_record_r2_%d.jsonl" % os.getpid()
    assert os.path.dirname(timer.step_path) == str(tmp_path)
    monkeypatch.setenv("VLLM_HPU_MTP_TIMING", "3")
    assert mtp_step_timer.from_env(device_type="cpu") is None
