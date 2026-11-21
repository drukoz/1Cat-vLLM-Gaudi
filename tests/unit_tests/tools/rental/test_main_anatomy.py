# SPDX-License-Identifier: Apache-2.0
"""The orchestrator's optional step-anatomy stage (--anatomy): the step timer in every server of a variant, the
probe and the summary after the first token, its record in anatomy.json, and nothing of it without the flag. The
stage is diagnostic: a probe that fails (the stub server does not stream) does not fail the variant."""

import subprocess

from tools.flash_next.rental import main as rental_main

from .test_main import make_model, read_json, run_stub
from .test_rental_scripts import SCRIPTS_DIR


def test_anatomy_stage_in_a_stub_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rental_main, "ANATOMY_IDLE_S", 0.05)
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "an1", "--variants", "A", "--anatomy")
    out = capsys.readouterr().out
    assert rc == 0, out
    vdir = tmp_path / "results" / "an1" / "A"
    record = read_json(vdir / rental_main.ANATOMY_STATUS_FILE)
    probe = record["probe_command"]
    assert probe[1].endswith(rental_main.ANATOMY_PROBE)
    assert probe[probe.index("--phases") + 1] == rental_main.ANATOMY_PHASES
    assert probe[probe.index("--timing-dir") + 1] == str(vdir / "timing")
    assert record["summary_command"][1].endswith(rental_main.ANATOMY_SUMMARY)
    assert "probe_exit" in record and "summary_exit" in record
    assert (vdir / "anatomy_probe.log").exists() and (vdir / "anatomy_summary.md").exists()
    env = read_json(vdir / "status.json")["servers"][0]["planned_env"]
    assert env["VLLM_HPU_STEP_TIMING"] == "1"
    assert env["VLLM_HPU_STEP_TIMING_DIR"] == str(vdir / "timing")
    assert read_json(tmp_path / "results" / "an1" / "manifest.json")["anatomy"] is True
    assert "variant A: anatomy probe exit" in out


def test_no_anatomy_without_the_flag(tmp_path):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "an0", "--variants", "A") == 0
    vdir = tmp_path / "results" / "an0" / "A"
    assert not (vdir / rental_main.ANATOMY_STATUS_FILE).exists()
    env = read_json(vdir / "status.json")["servers"][0]["planned_env"]
    assert "VLLM_HPU_STEP_TIMING" not in env
    assert read_json(tmp_path / "results" / "an0" / "manifest.json")["anatomy"] is False


def test_run_rental_passes_anatomy_to_the_entry(tmp_path):
    (tmp_path / "models").mkdir()
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "run_rental.sh"), "--dry-run", "--model", str(tmp_path / "models" / "X"),
         "--anatomy"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--anatomy" in proc.stdout
    help_text = subprocess.run(["bash", str(SCRIPTS_DIR / "run_rental.sh"), "--help"], capture_output=True,
                               text=True, timeout=60).stdout
    assert "--anatomy" in help_text


def test_container_entry_hands_anatomy_to_the_orchestrator():
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert 'if [ "$ANATOMY" -eq 1 ]; then MAIN_ARGS+=(--anatomy); fi' in text
    assert "        --anatomy)\n            ANATOMY=1\n" in text
