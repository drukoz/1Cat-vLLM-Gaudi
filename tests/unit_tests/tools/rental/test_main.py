# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental orchestrator (stub mode) and its helpers."""

import ast
import copy
import hashlib
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from tools.flash_next.rental import gates, main as rental_main, serve
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
    assert manifest["warm_pass"] is True
    assert manifest["server_timeout_s"] == rental_main.SERVER_HEALTH_TIMEOUT_S
    assert manifest["recipe_cache"]["dir"] is None
    assert manifest["recipe_cache"]["variants"] == {"A": "none", "B": "none"}
    for key in manifest["env"]:  # the kernel-loader prefixes and the fault handler joined the list on purpose
        assert key.startswith(rental_main.ENV_PREFIXES) or key in rental_main.ENV_KEYS
    assert set(rental_main.ENV_PREFIXES) >= {"VLLM_", "HABANA_", "PT_", "HF_", "FLASHINFER_", "GC_", "GDN_"}

    workload = read_json(run / "workload.json")
    assert len(workload["agents"]) == 4
    assert manifest["workload"]["sha256"] == manifest["workload_sha"]

    warm = read_json(run / "warm-workload.json")
    assert warm["turns"] == 1
    assert warm["file_offset"] == 1
    assert len(warm["agents"]) == 4

    for vid in ("A", "B"):
        status = read_json(run / vid / "status.json")
        assert status["status"] == int(GateStatus.PASSED), status
        startup = read_json(run / vid / "startup.json")
        assert startup[0]["max_concurrency"] == 64.0
        assert (run / vid / "requests.jsonl").read_text(encoding="utf-8").strip()
        assert (run / vid / "metrics.jsonl").read_text(encoding="utf-8").strip()

        warm_pass = read_json(run / vid / "warm-pass.json")
        assert warm_pass["requests"] == 4
        assert warm_pass["non_ok"] == 0
        assert warm_pass["max_tokens"] == rental_main.WARM_PASS_MAX_TOKENS
        assert warm_pass["file_offset"] == rental_main.WARM_PASS_FILE_OFFSET
        warm_rows = (run / vid / "warm-requests.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(warm_rows) == 4

    fused = read_json(run / "fused_gdn" / "status.json")
    assert fused["status"] == int(GateStatus.SKIPPED)
    card = read_json(run / "card_tests" / "status.json")
    assert card["status"] == int(GateStatus.SKIPPED)
    coherence = read_json(run / "coherence" / "status.json")
    assert coherence["status"] == int(GateStatus.SKIPPED)

    summary = (run / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[-1] == "GDN decode: not checked (stub mode)"


def test_no_warm_pass_writes_no_warm_files(tmp_path, capsys):
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "rnowarm", "--variants", "A", "--no-warm-pass")
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[-1] == "RUN rnowarm: PASSED"
    run = tmp_path / "results" / "rnowarm"

    manifest = read_json(run / "manifest.json")
    assert manifest["warm_pass"] is False
    assert not (run / "warm-workload.json").exists()
    assert not (run / "A" / "warm-pass.json").exists()
    assert not (run / "A" / "warm-requests.jsonl").exists()
    assert read_json(run / "A" / "status.json")["status"] == int(GateStatus.PASSED)


def test_warm_pass_non_ok_rows_fails_the_variant(tmp_path, capsys, monkeypatch):
    real_run_load = rental_main.agent_load.run_load

    async def fake_run_load(workload, servers, model, out_path, **kwargs):
        if Path(out_path).name == "warm-requests.jsonl":
            return 1
        return await real_run_load(workload, servers, model, out_path, **kwargs)

    monkeypatch.setattr(rental_main.agent_load, "run_load", fake_run_load)
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "rwarm", "--variants", "A")
    out = capsys.readouterr().out

    assert rc == 1
    assert out.splitlines()[-1] == "RUN rwarm: FAILED at A"
    run = tmp_path / "results" / "rwarm"

    status = read_json(run / "A" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert status["reason"] == "warm pass: 1 non-OK rows"
    warm_pass = read_json(run / "A" / "warm-pass.json")
    assert warm_pass["non_ok"] == 1
    assert warm_pass["requests"] == 4
    assert not (run / "A" / "requests.jsonl").exists()  # the measured load never started


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


def recipe_runner(tmp_path):
    args = rental_main._parse_args([
        "--mode", "stub", "--model", str(tmp_path / "model"), "--results", str(tmp_path / "results"),
        "--run-id", "r", "--variants", "A", "--recipe-cache", "/rc", "--recipe-cache-mb", "123"
    ] + STUB_ARGS)
    return rental_main.Runner(args, rental_main.load_variants())


def test_variant_server_env_recipe_cache_pytorch_gdn(tmp_path):
    runner = recipe_runner(tmp_path)
    runner.gdn_env = dict(gates.GDN_FALLBACK_ENV)
    runner.gdn_mode = gates.GDN_MODE_PYTORCH
    env = runner.variant_server_env({})
    key = (runner.bundle_commit or "unknown")[:12]
    assert env["PT_HPU_RECIPE_CACHE_CONFIG"] == "/rc/%s,false,123" % key
    assert runner.recipe_cache_note == runner.recipe_cache_dir


def test_variant_server_env_recipe_cache_off_for_fused_gdn(tmp_path):
    runner = recipe_runner(tmp_path)
    runner.gdn_env = {"VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    runner.gdn_mode = gates.GDN_MODE_FUSED
    env = runner.variant_server_env({})
    assert "PT_HPU_RECIPE_CACHE_CONFIG" not in env
    assert "FLASHINFER_GAUDI_COMBINED_KERNEL_DB" not in env
    assert runner.recipe_cache_note == serve.RECIPE_CACHE_OFF_REASON


def test_variant_server_env_recipe_cache_combined_db(tmp_path):
    runner = recipe_runner(tmp_path)
    runner.combined_db_installed = True
    runner.gdn_env = {"VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_GDN_CORE_VERIFY": "1"}
    runner.gdn_mode = gates.GDN_MODE_FUSED
    env = runner.variant_server_env({})
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in env  # the combined database is safe next to a recipe cache
    assert "FLASHINFER_GAUDI_COMBINED_KERNEL_DB" not in env
    runner.gdn_mode = gates.GDN_MODE_PYTORCH  # installed, but G1b did not pass fused: the database stays out
    env = runner.variant_server_env({})
    assert "PT_HPU_RECIPE_CACHE_CONFIG" not in env
    assert env["FLASHINFER_GAUDI_COMBINED_KERNEL_DB"] == "0"


def test_variant_server_env_no_recipe_cache_flag(tmp_path):
    runner = node_runner(tmp_path)
    runner.gdn_env = dict(gates.GDN_FALLBACK_ENV)
    env = runner.variant_server_env({})
    assert "PT_HPU_RECIPE_CACHE_CONFIG" not in env
    assert runner.recipe_cache_note == "none"


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


# --------------------------------------------------------------------------- the tiny checkpoint and its server

GENERATOR = REPO / "tools" / "flash_next" / "make_tiny_checkpoint.py"
COMMITTED_REFERENCE = REPO / "tests" / "unit_tests" / "models" / "flash_next" / "reference.json"


def generator_constants() -> dict:
    """The generator's module-level literals, read from its source: importing it would import torch."""
    out = {}
    for node in ast.parse(GENERATOR.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def serve_length(cmd: list) -> int:
    return int(cmd[cmd.index("--max-model-len") + 1])


def write_tiny_reference(ckpt: Path, lengths, build=None) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    prompts = [{"prompt_ids": list(range(n)), "generated_ids": list(range(16))} for n in lengths]
    reference = {"schema": 2, "max_new_tokens": 16, "prompts": prompts}
    if build is not None:
        reference["build"] = build
    (ckpt / "reference.json").write_text(json.dumps(reference), encoding="utf-8")


def test_tiny_serve_length_covers_the_longest_reference_prompt(tmp_path):
    # Session 1: the server ran with --max-model-len 512 and answered HTTP 400 to the 700 and 2600 token prompts.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    shutil.copyfile(COMMITTED_REFERENCE, ckpt / "reference.json")
    reference = read_json(ckpt / "reference.json")
    longest = max(len(p["prompt_ids"]) for p in reference["prompts"])
    assert longest == max(generator_constants()["PROMPT_LENGTHS"]) == 2600
    length = serve_length(rental_main.tiny_serve_command(ckpt, 8010))
    assert length >= longest + reference["max_new_tokens"]
    assert length >= longest + gates.TINY_MAX_TOKENS
    fixture = read_json(REPO / rental_main.TINY_FIXTURE)
    assert length <= fixture["text_config"]["max_position_embeddings"]


def test_tiny_serve_length_follows_the_reference(tmp_path):
    ckpt = tmp_path / "ckpt"
    write_tiny_reference(ckpt, [1, 7])
    assert serve_length(rental_main.tiny_serve_command(ckpt, 8010)) == 512
    write_tiny_reference(ckpt, [1, 496])  # 496 + 16 fits 512 exactly
    assert serve_length(rental_main.tiny_serve_command(ckpt, 8010)) == 512
    write_tiny_reference(ckpt, [1, 497])
    assert serve_length(rental_main.tiny_serve_command(ckpt, 8010)) == 1024
    write_tiny_reference(ckpt, [5000, 3])
    assert serve_length(rental_main.tiny_serve_command(ckpt, 8010)) == 8192


def test_tiny_serve_length_without_a_reference_covers_the_generator_prompts(tmp_path):
    constants = generator_constants()
    length = serve_length(rental_main.tiny_serve_command(tmp_path / "no-ckpt", 8010))
    assert length == rental_main.TINY_DEFAULT_MODEL_LEN
    assert length >= max(constants["PROMPT_LENGTHS"]) + constants["MAX_NEW_TOKENS"]
    assert gates.TINY_MAX_TOKENS == constants["MAX_NEW_TOKENS"]


def test_tiny_checkpoint_command_builds_the_gdn_layers_at_head_dim_128(tmp_path):
    # The HPU fused GDN kernel needs head dim 128; the committed fixture has 16 and stays as it is.
    cmd = rental_main.tiny_checkpoint_command(tmp_path / "ckpt")
    assert cmd[cmd.index("--gdn-head-dim") + 1] == "128"
    assert rental_main.TINY_GDN_HEAD_DIM == 128
    fixture = read_json(REPO / rental_main.TINY_FIXTURE)["text_config"]
    assert (fixture["linear_key_head_dim"], fixture["linear_value_head_dim"]) == (16, 16)


def test_tiny_build_settings_name_the_generator_and_the_fixture():
    assert rental_main.tiny_build_settings() == {
        "seed": 0,
        "gdn_head_dim": 128,
        "fixture_sha256": hashlib.sha256((REPO / rental_main.TINY_FIXTURE).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
    }


def test_tiny_checkpoint_stale_reasons(tmp_path):
    ckpt = tmp_path / "ckpt"
    wanted = rental_main.tiny_build_settings()
    assert "no reference.json" in rental_main.tiny_checkpoint_stale_reason(ckpt)
    write_tiny_reference(ckpt, [1, 7])  # a reference from before the build settings were recorded
    assert "no build settings" in rental_main.tiny_checkpoint_stale_reason(ckpt)
    build = dict(wanted, schema=2, prompt_lengths=[1, 7], max_new_tokens=16)
    write_tiny_reference(ckpt, [1, 7], build)
    assert rental_main.tiny_checkpoint_stale_reason(ckpt) is None
    write_tiny_reference(ckpt, [1, 7], dict(build, gdn_head_dim=16))
    assert "gdn_head_dim" in rental_main.tiny_checkpoint_stale_reason(ckpt)
    write_tiny_reference(ckpt, [1, 7], dict(build, generator_sha256="0" * 64))
    assert "generator_sha256" in rental_main.tiny_checkpoint_stale_reason(ckpt)
    write_tiny_reference(ckpt, [1, 7], dict(build, prompt_lengths=[1, 7, 33]))  # prompts cut out of the file
    assert "prompt_lengths" in rental_main.tiny_checkpoint_stale_reason(ckpt)
    write_tiny_reference(ckpt, [1, 7], dict(build, schema=1))
    assert "schema" in rental_main.tiny_checkpoint_stale_reason(ckpt)


def fake_builder(tmp_path: Path, build: dict) -> list:
    """A checkpoint command that counts its runs in <ckpt>/builds and writes a reference with `build`."""
    script = tmp_path / "fake_builder.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)\n"
        "count = out / 'builds'\n"
        "count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))\n"
        "prompts = [{'prompt_ids': [1], 'generated_ids': list(range(16))}]\n"
        "(out / 'reference.json').write_text(json.dumps({'schema': 2, 'prompts': prompts, "
        "'build': json.loads(sys.argv[2])}))\n", encoding="utf-8")
    return [sys.executable, str(script)]


def test_ensure_tiny_checkpoint_rebuilds_on_other_settings(tmp_path):
    ckpt = tmp_path / "ckpt"
    wanted = dict(rental_main.tiny_build_settings(), schema=2, prompt_lengths=[1], max_new_tokens=16)
    good = fake_builder(tmp_path, wanted) + [str(ckpt), json.dumps(wanted)]
    said = []
    assert rental_main.ensure_tiny_checkpoint(ckpt, good, say=said.append) is None
    assert (ckpt / "builds").read_text() == "1"
    assert "building the tiny checkpoint" in said[-1]
    assert rental_main.ensure_tiny_checkpoint(ckpt, good, say=said.append) is None
    assert (ckpt / "builds").read_text() == "1"  # same settings: reused
    assert "reusing the tiny checkpoint" in said[-1]

    reference = read_json(ckpt / "reference.json")
    reference["build"]["gdn_head_dim"] = 16  # as built by the smoke test of an earlier bundle
    (ckpt / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    assert rental_main.ensure_tiny_checkpoint(ckpt, good, say=said.append) is None
    assert (ckpt / "builds").read_text() == "2"
    assert "gdn_head_dim" in said[-1] and "building the tiny checkpoint" in said[-1]
    assert read_json(ckpt / "reference.json")["build"]["gdn_head_dim"] == 128


def test_ensure_tiny_checkpoint_reports_a_failed_build(tmp_path):
    ckpt = tmp_path / "ckpt"
    write_tiny_reference(ckpt, [1])  # stale: it must not survive the failed build either
    failing = [sys.executable, "-c", "import sys; print('tokenizer: download failed', file=sys.stderr); sys.exit(1)"]
    reason = rental_main.ensure_tiny_checkpoint(ckpt, failing, say=lambda text: None)
    assert reason.startswith("tiny checkpoint failed (exit 1)") and "tokenizer: download failed" in reason
    assert not (ckpt / "reference.json").exists()
    reason = rental_main.ensure_tiny_checkpoint(ckpt, [str(tmp_path / "no-such-interpreter")], say=lambda text: None)
    assert reason.startswith("tiny checkpoint could not start")


def test_the_tiny_checkpoint_build_runs_on_one_cpu_thread(tmp_path, monkeypatch):
    # The tiny model's tensors are too small for CPU threads to pay off: measured on the CPU venv, the
    # head-dim-128 build takes 13 s on one thread and 172 s on the default count, with the same reference.
    ckpt = tmp_path / "ckpt"
    code = ("import os, sys; from pathlib import Path; out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); "
            "(out / 'threads').write_text(os.environ.get('OMP_NUM_THREADS', 'unset') + ' ' + "
            "os.environ.get('MKL_NUM_THREADS', 'unset'))")
    cmd = [sys.executable, "-c", code, str(ckpt)]
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    assert rental_main.ensure_tiny_checkpoint(ckpt, cmd, say=lambda text: None) is None
    assert (ckpt / "threads").read_text() == "1 1"
    monkeypatch.setenv("OMP_NUM_THREADS", "4")  # a count the operator set is kept
    assert rental_main.ensure_tiny_checkpoint(ckpt, cmd, say=lambda text: None) is None
    assert (ckpt / "threads").read_text() == "4 1"


# --------------------------------------------------------------------------- the combined kernel database switch


def fake_combined_db_lookup(tmp_path: Path, monkeypatch) -> Path:
    """An installed flashinfer_gaudi/lib with the combined kernel database, found by the orchestrator's lookup."""
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    (lib / serve.COMBINED_KERNEL_DB).write_bytes(b"\0")
    real_run_step = gates.run_step

    def run_step(cmd, cwd, timeout_s, env=None):
        if list(cmd[1:3]) == ["-c", gates.GDN_LIB_LOOKUP_CODE]:
            return subprocess.CompletedProcess(cmd, 0, "%s\n" % lib, "")
        return real_run_step(cmd, cwd, timeout_s, env)

    monkeypatch.setattr(gates, "run_step", run_step)
    return lib


def test_combined_kernel_db_installed_honours_the_operator_switch(tmp_path, monkeypatch):
    fake_combined_db_lookup(tmp_path, monkeypatch)
    monkeypatch.delenv(serve.COMBINED_KERNEL_DB_ENV, raising=False)
    assert rental_main.combined_kernel_db_installed() is True
    monkeypatch.setenv(serve.COMBINED_KERNEL_DB_ENV, "1")
    assert rental_main.combined_kernel_db_installed() is True
    monkeypatch.setenv(serve.COMBINED_KERNEL_DB_ENV, "0")  # run_rental.sh --no-combined-kernel-db
    assert rental_main.combined_kernel_db_installed() is False


def test_fused_server_gets_no_recipe_cache_when_the_operator_switched_the_database_off(tmp_path, monkeypatch):
    # With the switch at 0 the loader registers the kernel libraries as a colon list, which the bridge
    # rejects next to a recipe cache: a fused server that got the cache would run the torch path.
    fake_combined_db_lookup(tmp_path, monkeypatch)
    monkeypatch.setenv(serve.COMBINED_KERNEL_DB_ENV, "0")
    args = rental_main._parse_args([
        "--mode", "node", "--model", str(tmp_path / "model"), "--results", str(tmp_path / "results"),
        "--run-id", "r", "--variants", "A", "--recipe-cache", str(tmp_path / "rc")])
    runner = rental_main.Runner(args, rental_main.load_variants())
    assert runner.combined_db_installed is False
    assert runner.combined_db_disabled_by == "operator"
    runner.gdn_env = {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    runner.gdn_mode = gates.GDN_MODE_FUSED
    variant = runner.variant_by_id("A")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = runner.variant_server_env(eff["env"])
    _, env = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    assert serve.RECIPE_CACHE_ENV not in env
    assert env[serve.COMBINED_KERNEL_DB_ENV] == "0"
    assert runner.recipe_cache_note == serve.RECIPE_CACHE_OFF_REASON

    monkeypatch.delenv(serve.COMBINED_KERNEL_DB_ENV)  # the same run without the switch: the cache is safe
    runner = rental_main.Runner(args, rental_main.load_variants())
    runner.gdn_env = {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    runner.gdn_mode = gates.GDN_MODE_FUSED
    assert runner.combined_db_disabled_by is None
    assert serve.RECIPE_CACHE_ENV in runner.variant_server_env({})
