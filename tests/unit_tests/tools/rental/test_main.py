# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental orchestrator (stub mode) and its helpers."""

import json
import shutil
from pathlib import Path

from tools.flash_next.rental import main as rental_main
from tools.flash_next.rental.main import gdn_summary_lines, server_env
from tools.flash_next.rental.schema import GateStatus

REPO = Path(__file__).resolve().parents[4]
FIXTURE = REPO / "tests" / "unit_tests" / "tools" / "rental" / "fixtures" / "flash_next_fp8_config.json"
STUB_ARGS = ["--agents", "4", "--shared-tokens", "512", "--initial-tokens", "4000", "--max-tokens", "16"]


def make_model(tmp_path: Path, fp8: bool = True) -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    if fp8:
        shutil.copyfile(FIXTURE, model / "config.json")
    else:
        (model / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 64}), encoding="utf-8")
    (model / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    return model


def run_stub(tmp_path: Path, model: Path, run_id: str, *extra: str) -> int:
    argv = [
        "--mode",
        "stub",
        "--model",
        str(model),
        "--results",
        str(tmp_path / "results"),
        "--run-id",
        run_id,
    ] + list(extra) + STUB_ARGS
    return rental_main.main(argv)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_stub_run(tmp_path, capsys):
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "r1", "--variants", "A,B")
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[-1] == "RUN r1: PASSED"
    run = tmp_path / "results" / "r1"

    manifest = read_json(run / "manifest.json")
    assert manifest["run_id"] == "r1"
    assert manifest["mode"] == "stub"
    assert manifest["workload_sha"]
    assert [v["id"] for v in manifest["variants"]] == ["A", "B"]
    assert manifest["gdn_core_decode_mode"] is None
    for key in manifest["env"]:
        assert key.startswith(("VLLM_", "HABANA_", "PT_", "HF_"))

    workload = read_json(run / "workload.json")
    assert len(workload["agents"]) == 4
    assert manifest["workload"]["sha256"] == manifest["workload_sha"]

    for vid in ("A", "B"):
        status = read_json(run / vid / "status.json")
        assert status["status"] == int(GateStatus.PASSED), status
        startup = read_json(run / vid / "startup.json")
        assert startup[0]["max_concurrency"] == 64.0
        assert (run / vid / "requests.jsonl").read_text(encoding="utf-8").strip()
        assert (run / vid / "metrics.jsonl").read_text(encoding="utf-8").strip()

    fused = read_json(run / "fused_gdn" / "status.json")
    assert fused["status"] == int(GateStatus.SKIPPED)
    coherence = read_json(run / "coherence" / "status.json")
    assert coherence["status"] == int(GateStatus.SKIPPED)

    summary = (run / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[-1] == "GDN decode: not checked (stub mode)"


def test_c_inherits_b_spec(tmp_path, capsys):
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "rinh", "--variants", "A,B,C")
    capsys.readouterr()
    assert rc == 0
    run = tmp_path / "results" / "rinh"

    a = read_json(run / "A" / "status.json")
    c = read_json(run / "C" / "status.json")
    assert c["status"] == int(GateStatus.PASSED), c
    a_dump = json.dumps(a["servers"])
    c_dump = json.dumps(c["servers"])
    assert "--speculative-config" not in a_dump
    assert "--speculative-config" in c_dump
    assert "mtp" in c_dump
    assert "VLLM_FLASH_NEXT_DRAFT_RANK_LOG" in c_dump
    assert c["reason"] and "B" in c["reason"]


def test_from_b_runs_only_b_and_c(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "rf", "--variants", "A,B") == 0
    capsys.readouterr()
    a_status = tmp_path / "results" / "rf" / "A" / "status.json"
    mtime = a_status.stat().st_mtime_ns

    rc = run_stub(tmp_path, model, "rf", "--from", "B", "--variants", "A,B,C")
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[-1] == "RUN rf: PASSED"
    assert a_status.stat().st_mtime_ns == mtime

    manifest = read_json(tmp_path / "results" / "rf" / "manifest.json")
    assert [v["id"] for v in manifest["variants"]] == ["A", "B", "C"]
    assert manifest["phases"] == ["B", "C", "report"]
    c = read_json(tmp_path / "results" / "rf" / "C" / "status.json")
    assert c["status"] == int(GateStatus.PASSED), c
    assert "--speculative-config" in json.dumps(c["servers"])


def test_preflight_fail_stops_with_report(tmp_path, capsys):
    model = make_model(tmp_path, fp8=False)
    rc = run_stub(tmp_path, model, "rfail", "--variants", "A")
    out = capsys.readouterr().out
    assert rc == 1
    assert out.splitlines()[-1] == "RUN rfail: FAILED at preflight"
    run = tmp_path / "results" / "rfail"
    pre = read_json(run / "preflight" / "status.json")
    assert pre["status"] == int(GateStatus.FAILED)
    assert (run / "summary.md").is_file()
    assert not (run / "A" / "status.json").exists()


def test_dry_run_prints_commands_and_starts_nothing(tmp_path, capsys):
    model = make_model(tmp_path)
    results = tmp_path / "results-dry"
    rc = rental_main.main(["--model", str(model), "--results", str(results), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    for phase in ("preflight", "build", "fused_gdn", "tiny", "A", "B", "C", "report"):
        assert "phase: %s" % phase in out
    assert out.count("vllm serve") == 4
    assert "--speculative-config" in out
    assert not results.exists()


def test_run_load_raises_leaves_complete_run(tmp_path, capsys, monkeypatch):
    model = make_model(tmp_path)

    def boom(*args, **kwargs):
        raise ConnectionResetError("stub connection reset")

    monkeypatch.setattr(rental_main.agent_load, "run_load", boom)
    rc = run_stub(tmp_path, model, "rboom", "--variants", "A")
    out = capsys.readouterr().out
    assert rc == 1
    assert out.splitlines()[-1] == "RUN rboom: FAILED at A"
    run = tmp_path / "results" / "rboom"
    assert (run / "summary.md").is_file()
    assert (run / "summary.json").is_file()
    status = read_json(run / "A" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert "ConnectionResetError" in status["reason"]


def test_from_c_summary_keeps_all_variants(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "rsum", "--variants", "A,B,C") == 0
    capsys.readouterr()
    rc = run_stub(tmp_path, model, "rsum", "--from", "C", "--variants", "A,B,C")
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[-1] == "RUN rsum: PASSED"
    run = tmp_path / "results" / "rsum"
    summary = read_json(run / "summary.json")
    assert sorted(summary["variants"]) == ["A", "B", "C"]
    manifest = read_json(run / "manifest.json")
    assert [v["id"] for v in manifest["variants"]] == ["A", "B", "C"]
    assert manifest["phases"] == ["C", "report"]


def test_from_without_run_id_is_an_error(tmp_path, capsys):
    model = make_model(tmp_path)
    rc = rental_main.main([
        "--mode",
        "stub",
        "--model",
        str(model),
        "--results",
        str(tmp_path / "results"),
        "--from",
        "C",
    ] + STUB_ARGS)
    out = capsys.readouterr().out
    assert rc == 2
    assert "--run-id" in out


def test_server_env_gdn_wins():
    env = server_env({"A": "1", "VLLM_HPU_GDN_CORE_DECODE": "9"}, {"VLLM_HPU_GDN_CORE_DECODE": "0"})
    assert env == {"A": "1", "VLLM_HPU_GDN_CORE_DECODE": "0"}


def test_gdn_summary_lines_pytorch_path(tmp_path):
    run = tmp_path / "run"
    (run / "fused_gdn").mkdir(parents=True)
    (run / "fused_gdn" / "details.json").write_text(
        json.dumps({"mode": 0, "reason": "no fused kernel", "env": {}}), encoding="utf-8")
    (run / "A").mkdir()
    (run / "A" / "gdn_fallback.txt").write_text("server0: fallback one\nserver0: fallback two\n", encoding="utf-8")
    (run / "B").mkdir()
    (run / "B" / "gdn_fallback.txt").write_text("", encoding="utf-8")
    lines = gdn_summary_lines(run, ["A", "B"])
    assert lines == [
        "GDN decode: PyTorch path, VLLM_HPU_GDN_CORE_DECODE=0 (no fused kernel)",
        "GDN fallback lines in A: 2 (the kernel did not run on every layer)",
    ]
