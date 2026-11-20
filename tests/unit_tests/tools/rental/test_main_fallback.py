# SPDX-License-Identifier: Apache-2.0
"""The orchestrator's automatic fallback for the combined kernel database (RUK-3248: gate G1b and the
tiny gate G2), and the flags that skip the recorded-only steps (--quick-gates, --no-verify-probe)."""

import json
import socket
import sys
from pathlib import Path

import pytest

from tools.flash_next.rental import gates, report, serve
from tools.flash_next.rental import main as rental_main
from tools.flash_next.rental.schema import GateStatus

from .test_main import make_model, read_json, run_stub
from .test_tiny_smoke import FAKE_SERVER

KILL_SWITCH = serve.COMBINED_KERNEL_DB_ENV
FUSED_ENV = {"VLLM_HPU_GDN_CORE_VERIFY": "1"}

# A tiny server that cannot start with the combined kernel database and serves the reference without it.
BROKEN_WITH_COMBINED_DB = '''
import os, sys, time
if os.environ.get("FLASHINFER_GAUDI_COMBINED_KERNEL_DB") != "0":
    if os.environ.get("FAKE_HANG_WITH_COMBINED_DB"):
        time.sleep(600)
    print("(Worker_TP0 pid=5) RuntimeError: synStatus=26 [Generic failure] the adapter returned no kernel", flush=True)
    sys.exit(4)
''' + FAKE_SERVER


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def fused_node(tmp_path, monkeypatch):
    """A node-mode runner whose G1b passed with the fused kernel; `combined_db` says whether the combined
    kernel database is installed. The tiny gate "builds" a two-prompt reference and starts a fake server."""
    monkeypatch.setattr(rental_main, "CARD_FREE_WAIT_S", 0.0)
    monkeypatch.setattr(rental_main, "FAILURE_EVIDENCE", ())
    monkeypatch.setattr(rental_main, "TINY_PORT", free_port())
    monkeypatch.setattr(serve, "_HEALTH_POLL_S", 0.1)
    monkeypatch.delenv(KILL_SWITCH, raising=False)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (KILL_SWITCH, serve.RECIPE_CACHE_ENV))
    for name in ("FAKE_SERVER_EXIT", "FAKE_SERVER_LINE", "FAKE_SERVER_WRONG", "FAKE_HANG_WITH_COMBINED_DB"):
        monkeypatch.delenv(name, raising=False)
    builds = []

    def ensure(ckpt, **kwargs):
        builds.append(str(ckpt))
        Path(ckpt).mkdir(parents=True, exist_ok=True)
        prompts = [{"prompt_ids": list(range(1, n + 1)), "generated_ids": list(range(100, 116))} for n in (1, 7)]
        (Path(ckpt) / "reference.json").write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
        return None

    monkeypatch.setattr(rental_main, "ensure_tiny_checkpoint", ensure)

    def make(combined_db: bool = True, server: str = BROKEN_WITH_COMBINED_DB, *extra):
        script = tmp_path / "fake_tiny_server.py"
        script.write_text(server, encoding="utf-8")
        monkeypatch.setattr(rental_main, "tiny_serve_command",
                            lambda ckpt, port: [sys.executable, str(script), "--port", str(port), "--reference",
                                                str(Path(ckpt) / "reference.json")])
        monkeypatch.setattr(rental_main, "combined_kernel_db_installed", lambda *args, **kwargs: combined_db)
        args = rental_main._parse_args(["--mode", "node", "--model", str(tmp_path / "model"), "--results",
                                        str(tmp_path / "results"), "--run-id", "r", "--variants", "A",
                                        "--recipe-cache", str(tmp_path / "rc"), *extra])
        runner = rental_main.Runner(args, rental_main.load_variants())
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        runner.gdn_env = dict(FUSED_ENV)
        runner.gdn_mode = gates.GDN_MODE_FUSED
        runner.builds = builds
        return runner

    return make


# --------------------------------------------------------------------------- the tiny gate G2


def test_tiny_gate_runs_once_more_without_the_combined_database(fused_node, capsys):
    runner = fused_node()
    assert runner.combined_db_active() is True
    assert runner.run_gate("tiny") is True  # the second attempt passed: the run goes on
    tiny = runner.run_dir / "tiny"

    kept = tiny / "attempt-combined"
    first_status = read_json(kept / "status.json")
    assert first_status["status"] == int(GateStatus.FAILED)
    assert first_status["reason"].startswith("tiny server exited with code 4 after")
    assert first_status["errors"] == ["RuntimeError: synStatus=26 [Generic failure] the adapter returned no kernel"]
    first_details = read_json(kept / "details.json")
    assert first_details["combined_kernel_db"] is True
    assert "synStatus=26" in (kept / "server0.log").read_text(encoding="utf-8")

    status = read_json(tiny / "status.json")
    assert status["status"] == int(GateStatus.PASSED)
    details = read_json(tiny / "details.json")
    assert details["combined_kernel_db"] is False
    assert details["combined_kernel_db_disabled_by"] == "tiny"
    assert details["attempt_combined"]["dir"] == "attempt-combined"
    assert details["attempt_combined"]["status"] == int(GateStatus.FAILED)
    assert details["n_prompts"] == 2
    assert "synStatus=26" not in (tiny / "server0.log").read_text(encoding="utf-8")  # the second server's own log

    second_env = read_json(tiny / "server_env.json")  # written by the second server only
    assert second_env[KILL_SWITCH] == "0"
    assert second_env[serve.RECIPE_CACHE_ENV] is None  # fused without the database: the colon-list path

    assert runner.combined_db_disabled_by == "tiny"
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["recipe_cache"]["combined_kernel_db_disabled_by"] == "tiny"
    assert manifest["recipe_cache"]["combined_kernel_db"] is True  # it is installed

    env = runner.variant_server_env({})  # every later server of the run
    assert env[KILL_SWITCH] == "0"
    assert serve.RECIPE_CACHE_ENV not in env
    assert runner.recipe_cache_note == serve.RECIPE_CACHE_OFF_REASON
    assert len(runner.builds) == 2  # the checkpoint step ran for both attempts (the real one reuses the folder)

    out = capsys.readouterr().out
    assert "tiny: FAILED with the combined kernel database" in out and "running the gate once more" in out
    assert "tiny: PASSED" in out

    runner.write_manifest()
    assert report.main([str(runner.run_dir)]) == 0
    md = (runner.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "- Combined kernel database: switched off by the tiny gate" in md


def test_tiny_gate_first_attempt_gets_the_database_and_the_cache(fused_node):
    # A server that works with the combined database: one attempt, the cache stays on, nothing is switched off.
    runner = fused_node(True, FAKE_SERVER)
    assert runner.run_gate("tiny") is True
    tiny = runner.run_dir / "tiny"
    assert not (tiny / "attempt-combined").exists()
    env = read_json(tiny / "server_env.json")
    assert env[KILL_SWITCH] is None and env[serve.RECIPE_CACHE_ENV] is not None
    details = read_json(tiny / "details.json")
    assert details["combined_kernel_db"] is True and "combined_kernel_db_disabled_by" not in details
    assert runner.combined_db_disabled_by is None
    assert len(runner.builds) == 1


def test_tiny_gate_timeout_with_the_combined_database_also_runs_once_more(fused_node, monkeypatch):
    monkeypatch.setenv("FAKE_HANG_WITH_COMBINED_DB", "1")
    monkeypatch.setattr(rental_main, "TINY_HEALTH_TIMEOUT_S", 2.0)
    runner = fused_node()
    assert runner.run_gate("tiny") is True
    first = read_json(runner.run_dir / "tiny" / "attempt-combined" / "status.json")
    assert first["status"] == int(GateStatus.TIMED_OUT)
    assert first["reason"] == "tiny server not healthy within 2 s"
    assert read_json(runner.run_dir / "tiny" / "status.json")["status"] == int(GateStatus.PASSED)
    assert runner.combined_db_disabled_by == "tiny"


def test_tiny_gate_failing_twice_fails_the_gate(fused_node, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")  # wrong tokens with and without the database
    runner = fused_node(True, FAKE_SERVER)
    assert runner.run_gate("tiny") is False
    tiny = runner.run_dir / "tiny"
    assert read_json(tiny / "attempt-combined" / "status.json")["status"] == int(GateStatus.FAILED)
    status = read_json(tiny / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert "the shortest failing prompt has 1 tokens" in status["reason"]
    details = read_json(tiny / "details.json")
    assert details["combined_kernel_db_disabled_by"] == "tiny"
    # one rerun without the database, one on the PyTorch GDN path (the last fallback of a fused run)
    assert len(runner.builds) == 3
    assert details["gdn_fused_disabled_by"] == "tiny" and (tiny / "attempt-gdn-fused").is_dir()


def test_tiny_gate_does_not_rerun_without_the_combined_database(fused_node, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    runner = fused_node(False, FAKE_SERVER)  # not installed
    assert runner.run_gate("tiny") is False
    assert not (runner.run_dir / "tiny" / "attempt-combined").exists()
    # one more attempt: the last fallback of a fused run is the PyTorch GDN path (attempt-gdn-fused)
    assert (runner.run_dir / "tiny" / "attempt-gdn-fused").is_dir()
    assert len(runner.builds) == 2
    assert runner.gdn_mode == gates.GDN_MODE_PYTORCH and runner.gdn_fused_disabled_by == "tiny"
    assert runner.variant_server_env({})[gates.GDN_MODE_ENV] == "0"  # every later server
    assert runner.combined_db_disabled_by is None

    runner = fused_node(True, FAKE_SERVER)  # installed, but G1b turned the fused kernel off: never loaded
    runner.gdn_env = dict(gates.GDN_FALLBACK_ENV)
    runner.gdn_mode = gates.GDN_MODE_PYTORCH
    assert runner.run_gate("tiny") is False
    assert not (runner.run_dir / "tiny" / "attempt-combined").exists()
    assert runner.combined_db_disabled_by is None


def test_tiny_gate_does_not_rerun_when_its_server_never_started(fused_node, monkeypatch):
    runner = fused_node()
    monkeypatch.setattr(rental_main, "ensure_tiny_checkpoint", lambda ckpt, **kwargs: "tiny checkpoint failed (exit 1)")
    assert runner.run_gate("tiny") is False
    assert not (runner.run_dir / "tiny" / "attempt-combined").exists()
    assert runner.combined_db_disabled_by is None  # the database was not what failed
    assert read_json(runner.run_dir / "tiny" / "status.json")["reason"] == "tiny checkpoint failed (exit 1)"


# --------------------------------------------------------------------------- gate G1b's decision in the run


def g1b_disabled_details() -> dict:
    return {"mode": 1, "verify_mode": 1, "env": {"VLLM_HPU_GDN_CORE_VERIFY": "1", KILL_SWITCH: "0"},
            "combined_kernel_db": "disabled_by_g1b", "combined_attempt": {"status": "failed", "logs": []}}


def test_g1b_fallback_keeps_the_database_and_the_cache_out_of_every_fused_server(fused_node, monkeypatch):
    runner = fused_node()
    runner.gdn_env, runner.gdn_mode = {}, None
    monkeypatch.setattr(gates, "fused_gdn", lambda *args, **kwargs: (GateStatus.PASSED, g1b_disabled_details()))
    assert runner.run_gate("fused_gdn") is True
    assert runner.gdn_mode == gates.GDN_MODE_FUSED
    assert runner.combined_db_disabled_by == "g1b"
    assert runner.combined_db_active() is False

    variant = runner.variant_by_id("A")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = runner.variant_server_env(eff["env"])
    _, env = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    assert env[KILL_SWITCH] == "0"
    assert serve.RECIPE_CACHE_ENV not in env  # fused, colon-list GC_KERNEL_PATH: the path proven on a card
    assert "VLLM_HPU_GDN_CORE_DECODE" not in env  # the fused kernel stays on
    tiny = runner.tiny_env()
    assert tiny[KILL_SWITCH] == "0" and serve.RECIPE_CACHE_ENV not in tiny

    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["recipe_cache"]["combined_kernel_db_disabled_by"] == "g1b"
    assert manifest["gdn_core_decode_mode"] == 1
    assert report.main([str(runner.run_dir)]) == 0
    md = (runner.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "- Combined kernel database: switched off by gate G1b" in md


def test_g1b_pass_through_the_database_keeps_it_and_the_cache(fused_node, monkeypatch):
    runner = fused_node()
    details = {"mode": 1, "verify_mode": 1, "env": {"VLLM_HPU_GDN_CORE_VERIFY": "1"}, "combined_kernel_db": "active"}
    monkeypatch.setattr(gates, "fused_gdn", lambda *args, **kwargs: (GateStatus.PASSED, details))
    assert runner.run_gate("fused_gdn") is True
    assert runner.combined_db_disabled_by is None and runner.combined_db_active() is True
    env = runner.variant_server_env({})
    assert KILL_SWITCH not in env and serve.RECIPE_CACHE_ENV in env


def test_a_resumed_run_keeps_the_decision_of_the_gates_that_ran(fused_node, monkeypatch):
    runner = fused_node()
    runner.combined_db_disabled_by = "tiny"
    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")

    resumed = fused_node(True, BROKEN_WITH_COMBINED_DB, "--from", "A")
    assert resumed.combined_db_disabled_by is None
    resumed.restore_combined_db(manifest)
    assert resumed.combined_db_disabled_by == "tiny"
    assert KILL_SWITCH in resumed.variant_server_env({})

    # G1b's own decision also comes back with its details.json (the kill switch is in its env)
    write = runner.run_dir / "fused_gdn"
    write.mkdir(parents=True, exist_ok=True)
    (write / "details.json").write_text(json.dumps(g1b_disabled_details()), encoding="utf-8")
    resumed = fused_node(True, BROKEN_WITH_COMBINED_DB, "--from", "A")
    resumed.restore_gdn()
    assert resumed.combined_db_disabled_by == "g1b"
    assert resumed.gdn_env[KILL_SWITCH] == "0"

    # a gate that runs again decides again
    resumed.combined_db_disabled_by = "tiny"
    monkeypatch.setattr(rental_main, "tiny_serve_command",
                        lambda ckpt, port: [sys.executable, "-c", "import sys; sys.exit(9)"])
    resumed.gdn_env, resumed.gdn_mode = dict(FUSED_ENV), gates.GDN_MODE_FUSED
    resumed.run_gate("tiny")
    assert (resumed.run_dir / "tiny" / "attempt-combined" / "status.json").is_file()  # it tried the database again


# --------------------------------------------------------------------------- --quick-gates and --no-verify-probe


def capture_fused_gdn(monkeypatch) -> list:
    calls = []

    def fused_gdn(plugin_root, out_dir, python, **kwargs):
        calls.append(kwargs)
        return GateStatus.PASSED, {"mode": 1, "verify_mode": 1, "env": {"VLLM_HPU_GDN_CORE_VERIFY": "1"},
                                   "skipped_steps": {}}

    monkeypatch.setattr(gates, "fused_gdn", fused_gdn)
    return calls


def test_quick_gates_skips_the_recorded_only_steps_and_the_card_tests(fused_node, monkeypatch, capsys):
    calls = capture_fused_gdn(monkeypatch)
    monkeypatch.setattr(gates, "card_tests", lambda *args, **kwargs: pytest.fail("card_tests must not run"))
    runner = fused_node(False, FAKE_SERVER, "--quick-gates")
    assert runner.run_gate("fused_gdn") is True
    assert calls == [{"skip_recorded_only": "--quick-gates", "skip_verify_probe": "--quick-gates"}]
    assert runner.run_gate("card_tests") is True
    status = read_json(runner.run_dir / "card_tests" / "status.json")
    assert status["status"] == int(GateStatus.SKIPPED) and status["reason"] == "--quick-gates"
    assert read_json(runner.run_dir / "card_tests" / "details.json") == {"reason": "--quick-gates"}

    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["quick_gates"] is True
    assert manifest["skipped_steps"] == {
        "fused_gdn.replay_eager": "--quick-gates",
        "fused_gdn.bench": "--quick-gates",
        "fused_gdn.verify_crash_probe": "--quick-gates",
        "card_tests": "--quick-gates",
    }
    assert report.main([str(runner.run_dir)]) == 0
    md = (runner.run_dir / "summary.md").read_text(encoding="utf-8")
    assert ("- Skipped steps: fused_gdn.replay_eager, fused_gdn.bench, fused_gdn.verify_crash_probe, card_tests "
            "(--quick-gates)") in md


def test_no_verify_probe_skips_only_the_probe(fused_node, monkeypatch):
    calls = capture_fused_gdn(monkeypatch)
    ran = []
    monkeypatch.setattr(gates, "card_tests",
                        lambda *args, **kwargs: (ran.append(1), (GateStatus.PASSED, {"files": []}))[1])
    runner = fused_node(False, FAKE_SERVER, "--no-verify-probe")
    assert runner.run_gate("fused_gdn") is True
    assert calls == [{"skip_recorded_only": None, "skip_verify_probe": "--no-verify-probe"}]
    assert runner.run_gate("card_tests") is True and ran == [1]  # the card tests still run
    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["quick_gates"] is False
    assert manifest["skipped_steps"] == {"fused_gdn.verify_crash_probe": "--no-verify-probe"}
    assert report.main([str(runner.run_dir)]) == 0
    md = (runner.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "- Skipped steps: fused_gdn.verify_crash_probe (--no-verify-probe)" in md


def test_without_the_flags_nothing_is_skipped(fused_node, monkeypatch):
    calls = capture_fused_gdn(monkeypatch)
    runner = fused_node(False, FAKE_SERVER)
    assert runner.run_gate("fused_gdn") is True
    assert calls == [{"skip_recorded_only": None, "skip_verify_probe": None}]
    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["quick_gates"] is False and manifest["skipped_steps"] == {}
    assert report.main([str(runner.run_dir)]) == 0
    assert "Skipped steps" not in (runner.run_dir / "summary.md").read_text(encoding="utf-8")


def test_quick_gates_stub_run_records_the_skips(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "quick", "--variants", "A", "--quick-gates") == 0
    capsys.readouterr()
    run = tmp_path / "results" / "quick"
    manifest = read_json(run / "manifest.json")
    assert manifest["args"]["quick_gates"] is True and manifest["args"]["no_verify_probe"] is False
    assert sorted(manifest["skipped_steps"]) == ["card_tests", "fused_gdn.bench", "fused_gdn.replay_eager",
                                                 "fused_gdn.verify_crash_probe"]
    assert "(--quick-gates)" in (run / "summary.md").read_text(encoding="utf-8")
    assert manifest["phases"][:5] == ["preflight", "build", "fused_gdn", "tiny", "card_tests"]  # same phases
