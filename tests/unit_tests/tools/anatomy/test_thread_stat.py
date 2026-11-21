# SPDX-License-Identifier: Apache-2.0
"""thread_stat.py: per-thread scheduling figures from /proc (Linux only; the arithmetic is tested everywhere)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "flash_next" / "anatomy"))

import thread_stat  # noqa: E402


def test_measure_gives_deltas_per_thread_and_the_cpus_of_busy_threads(monkeypatch):
    clock = {"t": 0.0}
    # two threads of process 7: the main one works and moves from CPU 3 to CPU 5, the other sleeps
    state = {(7, 7): {"cpu_ns": 0, "runq_ns": 0, "slices": 0, "voluntary": 0, "involuntary": 0, "processor": 3},
             (7, 9): {"cpu_ns": 0, "runq_ns": 0, "slices": 0, "voluntary": 5, "involuntary": 0, "processor": 1}}

    def sleep(seconds):
        clock["t"] += seconds
        main = state[(7, 7)]
        main["cpu_ns"] += int(seconds * 0.9e9)
        main["runq_ns"] += int(seconds * 0.05e9)
        main["slices"] += 2
        main["involuntary"] += 1
        main["processor"] = 3 if clock["t"] < 0.5 else 5

    monkeypatch.setattr(thread_stat, "thread_ids", lambda pid: [7, 9])
    monkeypatch.setattr(thread_stat, "snapshot", lambda pid, tid: dict(state[(pid, tid)], comm="worker"))
    monkeypatch.setattr(thread_stat, "processor", lambda pid, tid: state[(pid, tid)]["processor"])
    result = thread_stat.measure([7], seconds=1.0, interval=0.1, probe_s=0.3, clock=lambda: clock["t"], sleep=sleep)
    main, other = result["threads"]
    assert main["main"] and main["tid"] == 7 and other["tid"] == 9
    assert abs(main["cpu_ms"] - 900.0) < 100.0 and abs(main["runq_ms"] - 50.0) < 6.0
    assert main["involuntary"] >= 7 and other["cpu_ms"] == 0.0 and other["voluntary"] == 0
    assert set(main["cpus"]) == {"3", "5"} and main["moves"] == 1 and "cpus" not in other
    text = thread_stat.table(result)
    assert "7 7 worker* " in text and " 3,5 1" in text


def test_processor_reads_the_field_after_a_comm_with_spaces(monkeypatch):
    stat = "42 (VLLM::Worker TP0) S " + " ".join(str(i) for i in range(4, 60))
    monkeypatch.setattr(thread_stat, "read", lambda path: stat)
    assert thread_stat.processor(42, 42) == 39  # field 39 of the line
