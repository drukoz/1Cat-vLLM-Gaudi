# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/check_fp8_layout.py (the M4 FP8 layout preflight, RUK-2709).

The five expected layouts from the plan (TP8/TP4 without EP fail, TP8/TP4 with EP pass, TP1 passes), the
row content for tp 8 without EP, and the CLI exit codes 0 / 1 / 2. Runs on CPU from the worktree root
with ``PYTHONPATH=$PWD``.
"""

import json
from pathlib import Path

from tools.flash_next.check_fp8_layout import check_layout, main

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flash_next_fp8_config.json"


def _config() -> dict:
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _row(rows: list[dict], module: str) -> dict:
    return next(row for row in rows if row["module"] == module)


def test_tp8_no_ep_fails():
    rows = check_layout(_config(), tp=8, ep=False)
    assert _row(rows, "experts.w13")["ok"] is False
    assert _row(rows, "experts.w2")["ok"] is False
    assert _row(rows, "experts.hidden")["ok"] is True


def test_tp4_no_ep_fails():
    rows = check_layout(_config(), tp=4, ep=False)
    assert _row(rows, "experts.w13")["ok"] is False
    assert _row(rows, "experts.w2")["ok"] is False
    assert _row(rows, "experts.hidden")["ok"] is True


def test_tp8_ep_passes():
    rows = check_layout(_config(), tp=8, ep=True)
    assert all(row["ok"] for row in rows)
    assert _row(rows, "experts.count")["per_rank"] == 64


def test_tp4_ep_passes():
    rows = check_layout(_config(), tp=4, ep=True)
    assert all(row["ok"] for row in rows)
    assert _row(rows, "experts.count")["per_rank"] == 128


def test_tp1_passes_both_ways():
    assert all(row["ok"] for row in check_layout(_config(), tp=1, ep=False))
    assert all(row["ok"] for row in check_layout(_config(), tp=1, ep=True))


def test_row_content_tp8_no_ep():
    rows = check_layout(_config(), tp=8, ep=False)
    assert [row["module"] for row in rows] == ["experts.w13", "experts.w2", "experts.hidden"]
    assert _row(rows, "experts.w13") == {
        "module": "experts.w13",
        "dim": "intermediate",
        "size": 1280,
        "per_rank": 80,
        "block": 128,
        "ok": False,
    }


def test_cli_exit_0_ep_pass(capsys):
    assert main(["--config", str(FIXTURE), "--tp", "8", "--ep"]) == 0
    out = capsys.readouterr().out
    assert "experts.w13 intermediate 1280 640 128 OK" in out
    assert "experts.count experts 512 64 1 OK" in out
    assert "fp8 layout tp=8 ep=1: PASS" in out


def test_cli_exit_1_no_ep_fail(capsys):
    assert main(["--config", str(FIXTURE), "--tp", "8"]) == 1
    out = capsys.readouterr().out
    assert "experts.w13 intermediate 1280 80 128 FAIL" in out
    assert "fp8 layout tp=8 ep=0: FAIL (2 rows)" in out


def test_cli_exit_2_bad_input(tmp_path, capsys):
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps({"text_config": {"hidden_size": 2560}}), encoding="utf-8")
    assert main(["--config", str(bad), "--tp", "8"]) == 2
    assert "quantization_config" in capsys.readouterr().err
