# SPDX-License-Identifier: Apache-2.0
"""trace_device.py on a synthetic Chrome trace with HPU device lanes."""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "flash_next" / "anatomy"))

import trace_device  # noqa: E402


def _trace(steps=3, nodes=10, node_us=5.0, gap_us=3.0, step_gap_us=4000.0):
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "python3"}},
        {"ph": "M", "name": "process_name", "pid": 9, "args": {"name": "HPU 0"}},
        {"ph": "M", "name": "thread_name", "pid": 9, "tid": 1, "args": {"name": "TPC"}},
        {"ph": "M", "name": "thread_name", "pid": 9, "tid": 2, "args": {"name": "MME"}},
        {"ph": "M", "name": "thread_name", "pid": 9, "tid": 3, "args": {"name": "HCL"}},
    ]
    t = 1000.0
    for step in range(steps):
        events.append({"ph": "X", "pid": 1, "tid": 1, "name": "aten::copy_", "ts": t - 50.0, "dur": 20.0})
        for node in range(nodes):
            tid = 1 if node % 2 == 0 else 2
            events.append({"ph": "X", "pid": 9, "tid": tid, "name": "node%d" % node, "ts": t, "dur": node_us,
                           "args": {"HW event name": "TPC_SPU" if tid == 1 else "MMEH_WB"}})
            t += node_us + gap_us
        events.append({"ph": "X", "pid": 9, "tid": 3, "name": "all_reduce", "ts": t, "dur": 20.0, "args": {}})
        t += 20.0 + step_gap_us
    return {"traceEvents": events}


def test_per_step_budget(tmp_path):
    res = trace_device.analyze(_trace(), min_burst_ms=0.05)
    assert res["steps"] == 3 and res["steps_from"] == "bursts"
    ps = res["per_step_ms"]
    assert abs(ps["busy_any"] - (10 * 5.0 + 20.0) / 1e3) < 1e-9
    assert abs(ps["busy_compute_tpc_or_mme"] - 50.0 / 1e3) < 1e-9
    assert abs(ps["comm_not_under_compute"] - 20.0 / 1e3) < 1e-9
    assert res["device_events_per_step"] == {"TPC": 5.0, "MME": 5.0, "COMM": 1.0}
    small = next(g for g in res["gaps_per_step"] if g["from_us"] == 2.0)
    assert small["count"] == 10.0  # nine gaps between nodes and one before the collective
    assert res["top_host_names"][0]["name"] == "aten::copy_"


def test_steps_argument_and_gzip_and_cli(tmp_path):
    path = tmp_path / "rank0.pt.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(_trace(steps=4), fh)
    res = trace_device.analyze(trace_device.load(str(path)), steps=2)
    assert res["steps"] == 2 and res["steps_from"] == "argument"
    assert trace_device.main([str(path), "--json", str(tmp_path / "out.json")]) == 0
    assert trace_device.main([str(path), "--probe"]) == 0
    assert json.loads((tmp_path / "out.json").read_text())[str(path)]["device_events"] == 4 * 11


def test_no_device_events_is_an_error():
    res = trace_device.analyze({"traceEvents": [{"ph": "X", "pid": 1, "tid": 1, "name": "a", "ts": 0, "dur": 1}]})
    assert "error" in res
