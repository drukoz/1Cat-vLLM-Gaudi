# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/step_anatomy.py on synthetic high-level profiler traces."""

import json

import pytest

from tools.flash_next.step_anatomy import load_events, main

PERIOD_US = 10000.0
DECODE_ROW = "decode (forward + sampler, host)"
POST_ROW = "sampler post-processing"
DRAFTER_ROW = "drafter"
PREP_ROW = "input prep (next step)"
GAP_ROW = "untracked gap"


def ev(name, ts, dur):
    return {"pid": 1, "tid": 3, "ph": "X", "name": name, "ts": ts, "dur": dur, "args": None}


def step_events(base, decode_us=6000.0):
    return [
        ev("decode", base, decode_us),
        ev("sampler_postprocessing", base + decode_us + 100.0, 1000.0),
        ev("propose_draft_tokens", base + decode_us + 1200.0, 1500.0),
        ev("prepare_input_tensors", base + 8700.0, 500.0),
    ]


def steady_events(n_decodes=7, decode_us=6000.0):
    """n_decodes decode events, each at i * 10 ms: n_decodes - 1 windows (the last decode has none)."""
    events = []
    for i in range(n_decodes):
        events.extend(step_events(i * PERIOD_US, decode_us))
    return events


def write_trace(tmp_path, events, name="trace.json"):
    """Write the events in the server's format: unterminated array, every event followed by a comma."""
    path = tmp_path / name
    path.write_text("[" + "".join(json.dumps(e) + "," for e in events), encoding="utf-8")
    return path


def run_json(tmp_path, events, capsys, extra_args=()):
    path = write_trace(tmp_path, events)
    assert main([str(path), "--json", *extra_args]) == 0
    return json.loads(capsys.readouterr().out)


def test_six_steps_means(tmp_path, capsys):
    """Period 10 ms = decode 6 + post 1 + drafter 1.5 + prep 0.5 + gap 1 ms, exactly."""
    payload = run_json(tmp_path, steady_events(), capsys)
    assert payload["steady_steps"] == 6
    assert payload["mixed_steps_skipped"] == 0
    rows = payload["rows"]
    expected_ms = {
        "period": 10.0,
        DECODE_ROW: 6.0,
        POST_ROW: 1.0,
        DRAFTER_ROW: 1.5,
        PREP_ROW: 0.5,
        GAP_ROW: 1.0,
    }
    for label, mean_ms in expected_ms.items():
        assert rows[label]["mean_ms"] == pytest.approx(mean_ms, abs=1e-9), label
    phase_sum = sum(rows[label]["mean_ms"] for label in (DECODE_ROW, POST_ROW, DRAFTER_ROW, PREP_ROW, GAP_ROW))
    assert phase_sum == pytest.approx(rows["period"]["mean_ms"], abs=1e-9)
    assert rows[DECODE_ROW]["share_of_period_pct"] == pytest.approx(60.0, abs=1e-9)
    assert rows[GAP_ROW]["share_of_period_pct"] == pytest.approx(10.0, abs=1e-9)
    assert list(rows) == [
        "period", DECODE_ROW, "\u21b3 model_forward", "\u21b3 compute_logits", "\u21b3 sampler",
        POST_ROW, DRAFTER_ROW, PREP_ROW, GAP_ROW,
    ]


def test_unterminated_trace_with_null_dur_loads(tmp_path):
    events = steady_events()
    text = "[" + "".join(json.dumps(e) + "," for e in events)
    text += json.dumps(ev("decode", 123000.0, None)) + ","
    path = tmp_path / "trace.json"
    path.write_text(text, encoding="utf-8")
    loaded = load_events(path)
    assert len(loaded) == len(events)
    assert all(e["dur"] is not None for e in loaded)


def test_warmup_hides_earlier_steps(tmp_path, capsys):
    events = [ev("warmup", 0.0, 55000.0)]
    events.extend(ev("decode", i * PERIOD_US, 6000.0) for i in range(6))
    for i in range(7):
        events.extend(step_events(60000.0 + i * PERIOD_US))
    payload = run_json(tmp_path, events, capsys)
    assert payload["steady_steps"] == 6
    assert payload["mixed_steps_skipped"] == 0
    assert payload["rows"][DECODE_ROW]["mean_ms"] == pytest.approx(6.0, abs=1e-9)


def test_prefill_window_skipped_and_counted(tmp_path, capsys):
    events = steady_events()
    events.append(ev("prefill", 25000.0, 3000.0))  # starts inside window [20 ms, 30 ms)
    payload = run_json(tmp_path, events, capsys)
    assert payload["steady_steps"] == 5
    assert payload["mixed_steps_skipped"] == 1
    path = write_trace(tmp_path, events)
    assert main([str(path)]) == 0
    assert "steady steps: 5, mixed steps skipped: 1" in capsys.readouterr().out


def test_last_n_steps(tmp_path, capsys):
    events = []
    for i in range(7):
        events.extend(step_events(i * PERIOD_US, decode_us=5000.0 + i * 100.0))
    path = write_trace(tmp_path, events)
    assert main([str(path), "--json"]) == 0
    full = json.loads(capsys.readouterr().out)
    assert full["steady_steps"] == 6
    assert full["rows"][DECODE_ROW]["mean_ms"] == pytest.approx(5.25, abs=1e-9)
    assert main([str(path), "--json", "--last", "2"]) == 0
    last2 = json.loads(capsys.readouterr().out)
    assert last2["steady_steps"] == 2
    assert last2["rows"][DECODE_ROW]["mean_ms"] == pytest.approx(5.45, abs=1e-9)


def test_json_output_parses(tmp_path, capsys):
    payload = run_json(tmp_path, steady_events(), capsys)
    assert set(payload) == {"trace", "steady_steps", "mixed_steps_skipped", "rows"}
    assert payload["rows"]["period"]["p50_ms"] == pytest.approx(10.0, abs=1e-9)


def test_exit_2_on_single_decode(tmp_path, capsys):
    path = write_trace(tmp_path, [ev("decode", 0.0, 6000.0)])
    assert main([str(path)]) == 2
    assert "decode" in capsys.readouterr().err
