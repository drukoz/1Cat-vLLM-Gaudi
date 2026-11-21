# SPDX-License-Identifier: Apache-2.0
"""End-to-end stub-mode checks for the rental orchestrator (M4-V).

Covers the two spec checks the module tests do not assert: the variant-level
STOP rule (A fails -> the run stops before B and C) and --with-profile
adding variant P.
"""

import json
from pathlib import Path

from tools.flash_next.rental import main as rental_main
from tools.flash_next.rental.schema import GateStatus

from .test_main import STUB_ARGS, make_model, read_json, run_stub


def test_variant_a_failure_stops_the_run(tmp_path, capsys, monkeypatch):
    async def fake_run_load(workload, servers, model, out_path, **kwargs):
        return 1

    monkeypatch.setattr(rental_main.agent_load, "run_load", fake_run_load)
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "stop1", "--variants", "A,B,C")
    out = capsys.readouterr().out

    assert rc == 1
    assert out.splitlines()[-1] == "RUN stop1: FAILED at A"
    run = tmp_path / "results" / "stop1"

    status = read_json(run / "A" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert "non-OK rows" in status["reason"]

    assert not (run / "B" / "status.json").exists()
    assert not (run / "C" / "status.json").exists()

    summary_md = (run / "summary.md").read_text(encoding="utf-8")
    assert "| A |" in summary_md and "| B |" in summary_md
    summary = read_json(run / "summary.json")
    assert summary["variants"]["A"]["status"] == int(GateStatus.FAILED)
    assert summary["variants"]["B"]["status"] is None


def test_with_profile_adds_variant_p(tmp_path, capsys):
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "prof1", "--variants", "A,B,C", "--with-profile")
    out = capsys.readouterr().out

    assert rc == 0
    assert out.splitlines()[-1] == "RUN prof1: PASSED"
    run = tmp_path / "results" / "prof1"

    manifest = read_json(run / "manifest.json")
    assert [v["id"] for v in manifest["variants"]] == ["A", "B", "C", "P"]
    assert read_json(run / "P" / "status.json")["status"] == int(GateStatus.PASSED)

    summary_md = (run / "summary.md").read_text(encoding="utf-8")
    assert "| P |" in summary_md and "## Variant P" in summary_md
