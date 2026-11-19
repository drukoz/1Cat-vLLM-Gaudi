# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental orchestrator (stub mode) and its helpers."""

import copy
import json
import shutil
import socket
from pathlib import Path

from tools.flash_next.rental import main as rental_main, serve
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
    assert manifest["skip_warmup"] is True
    assert manifest["server_timeout_s"] == rental_main.SERVER_HEALTH_TIMEOUT_S
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
    card = read_json(run / "card_tests" / "status.json")
    assert card["status"] == int(GateStatus.SKIPPED)
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


def hold_ports(*ports):
    """Listening sockets on 127.0.0.1 for each port we can bind (one already held by someone else is held anyway)."""
    held = []
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            held.append(sock)
        except OSError:
            sock.close()
    return held


def test_stub_run_ignores_held_config_ports(tmp_path, capsys):
    held = hold_ports(8000, 8001)
    try:
        model = make_model(tmp_path)
        rc = run_stub(tmp_path, model, "rheld", "--variants", "A,C")
        capsys.readouterr()
    finally:
        for sock in held:
            sock.close()
    assert rc == 0
    c = read_json(tmp_path / "results" / "rheld" / "C" / "status.json")
    assert c["status"] == int(GateStatus.PASSED), c
    urls = [srv["url"] for srv in c["servers"]]
    assert len(set(urls)) == 2
    assert not any(u.endswith((":8000", ":8001")) for u in urls)
    assert "8000" in c["servers"][0]["planned_command"]


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
    for phase in ("preflight", "build", "fused_gdn", "tiny", "card_tests", "A", "B", "C", "report"):
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


def test_summary_lines_toolchain(tmp_path):
    run = tmp_path / "run"
    (run / "fused_gdn").mkdir(parents=True)
    (run / "fused_gdn" / "details.json").write_text(json.dumps({"mode": 1, "reason": None, "env": {}}), encoding="utf-8")
    (run / "toolchain.json").write_text(json.dumps({
        "tpc_clang": True, "tpc_clang_path": "/usr/bin/tpc-clang", "tpc_clang_version": "tpc-clang 2.7.33",
        "habanalabs_tools_version": "1.24.1-482", "habanalabs_tools_installed_by_script": True, "native_build": 1,
    }), encoding="utf-8")
    assert gdn_summary_lines(run, []) == [
        "GDN decode: fused kernel (G1b passed)",
        "Native kernels: compiled (tpc-clang: tpc-clang 2.7.33; habanalabs-tools: 1.24.1-482)",
    ]
    (run / "toolchain.json").write_text(json.dumps({
        "tpc_clang": False, "tpc_clang_path": None, "tpc_clang_version": None,
        "habanalabs_tools_version": None, "habanalabs_tools_installed_by_script": False, "native_build": 0,
    }), encoding="utf-8")
    assert gdn_summary_lines(run, [])[-1] == (
        "Native kernels: NOT compiled, PyTorch GDN path (tpc-clang: absent; habanalabs-tools: absent)")


def test_tiny_env_matches_variant_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    args = rental_main._parse_args([
        "--mode", "node", "--model", str(tmp_path / "m"), "--results", str(tmp_path / "r"), "--run-id", "t"
    ])
    runner = rental_main.Runner(args, rental_main.load_variants())
    runner.gdn_env = {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    variant = runner.variant_by_id("C")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = server_env(eff["env"], runner.gdn_env)
    _, venv = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    tiny = runner.tiny_env()
    for key in ("PT_HPU_LAZY_MODE", "VLLM_HPU_FLASH_NEXT_QSA_MODE", "VLLM_HPU_RECURRENT_STATE_POOL",
                "VLLM_HPU_GDN_CORE_DECODE", "VLLM_HPU_GDN_CORE_VERIFY", "VLLM_COMPACT_GDN"):
        assert tiny.get(key) == venv.get(key)
    assert tiny["PT_HPU_LAZY_MODE"] == "1"
    assert "VLLM_COMPACT_GDN" not in tiny


def test_variant_server_env_skips_warmup_by_default(tmp_path):
    runner = node_runner(tmp_path)
    runner.gdn_env = {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    env = runner.variant_server_env({"VLLM_X": "1"})
    assert env["VLLM_SKIP_WARMUP"] == "true"
    assert env["VLLM_X"] == "1"
    assert env["VLLM_HPU_GDN_CORE_DECODE"] == "0"
    assert runner.request_timeout_s == 3600.0
    assert runner.server_timeout_s == rental_main.SERVER_HEALTH_TIMEOUT_S


def test_variant_server_env_full_warmup(tmp_path):
    args = rental_main._parse_args([
        "--mode", "node", "--model", str(tmp_path / "model"), "--results", str(tmp_path / "results"),
        "--run-id", "r", "--variants", "A", "--full-warmup", "--server-timeout", "7200"
    ])
    runner = rental_main.Runner(args, rental_main.load_variants())
    env = runner.variant_server_env({"VLLM_X": "1"})
    assert "VLLM_SKIP_WARMUP" not in env
    assert env["VLLM_X"] == "1"
    assert runner.server_timeout_s == 7200.0
    assert runner.request_timeout_s == rental_main.gates.HTTP_TIMEOUT_S


def node_runner(tmp_path):
    args = rental_main._parse_args(["--mode", "node", "--model", str(tmp_path / "model"),
                                    "--results", str(tmp_path / "results"), "--run-id", "r",
                                    "--variants", "A"])
    return rental_main.Runner(args, rental_main.load_variants())


def test_tiny_gate_refuses_held_port(tmp_path, monkeypatch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    held_port = int(sock.getsockname()[1])
    try:

        def no_run(*args, **kwargs):
            raise AssertionError("the tiny checkpoint must not be built")

        monkeypatch.setattr(rental_main, "TINY_PORT", held_port)
        monkeypatch.setattr(rental_main.subprocess, "run", no_run)
        runner = node_runner(tmp_path)
        status, details = runner.gate_tiny(tmp_path / "tiny")
        assert status == GateStatus.FAILED
        assert "already in use" in details["reason"]
    finally:
        sock.close()


def test_node_variant_refuses_held_port(tmp_path, monkeypatch):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = int(sock.getsockname()[1])
    try:

        def no_start(*args, **kwargs):
            raise AssertionError("no server must start on a held port")

        monkeypatch.setattr(rental_main.serve, "start_server", no_start)
        runner = node_runner(tmp_path)
        v = copy.deepcopy(runner.variant_by_id("A"))
        v["servers"][0]["port"] = port
        v["on_fail"] = 1
        runner.run_variant(v)
        status = read_json(tmp_path / "results" / "r" / "A" / "status.json")
        assert status["status"] == int(GateStatus.FAILED)
        assert "port %d already in use" % port in status["reason"]
    finally:
        sock.close()


def make_node_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    args = rental_main._parse_args([
        "--mode", "node", "--model", str(tmp_path / "m"), "--results", str(tmp_path / "r"), "--run-id", "ct"
    ])
    return rental_main.Runner(args, rental_main.load_variants())


def test_card_tests_failure_is_recorded_and_run_continues(tmp_path, monkeypatch):
    def fake_card_tests(plugin_root, out_dir, python):
        return GateStatus.FAILED, {"files": [], "reason": "test_hpu_fused_moe.py failed (exit 1)"}

    monkeypatch.setattr(rental_main.gates, "card_tests", fake_card_tests)
    runner = make_node_runner(tmp_path, monkeypatch)
    assert runner.run_gate("card_tests") is True
    status = read_json(runner.run_dir / "card_tests" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert status["reason"] == "test_hpu_fused_moe.py failed (exit 1)"


def test_card_tests_stub_mode_skips(tmp_path, monkeypatch):
    args = rental_main._parse_args([
        "--mode", "stub", "--model", str(tmp_path / "m"), "--results", str(tmp_path / "r"), "--run-id", "ct2"
    ] + STUB_ARGS)
    runner = rental_main.Runner(args, rental_main.load_variants())
    assert runner.run_gate("card_tests") is True
    status = read_json(runner.run_dir / "card_tests" / "status.json")
    assert status["status"] == int(GateStatus.SKIPPED)
    assert status["reason"] == "stub mode"
