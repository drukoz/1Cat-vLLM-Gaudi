# SPDX-License-Identifier: Apache-2.0
"""What the orchestrator tells the operator and leaves as evidence: a crashed server is not a timeout,
the first token of every variant server, the timeouts in use, the bundle commit, the failure
snapshots, and the refusal of a `--from` that has nothing to resume."""

import copy
import json
import os
import re
import socket
import sys
from pathlib import Path

import pytest

from tools.flash_next.rental import gates, report, serve
from tools.flash_next.rental import main as rental_main
from tools.flash_next.rental.schema import GateStatus

from .test_main import STUB_ARGS, make_model, read_json, run_stub

CRASH = ("import sys; print('(Worker_TP1 pid=7) ERROR 11-18 22:36:00 [multiproc_executor.py:1055] "
         "IndexError: tuple index out of range', flush=True); sys.exit(3)")
HANG = "import time; time.sleep(600)"
SHA = "0123456789abcdef0123456789abcdef01234567"
BUNDLE_TEXT = ("plugin sha: %s\nbranch: feat/flash-next-gaudi\nvllm commit: 8f816a3f\n"
               "build date: 2026-11-19T10:00:00Z\n" % SHA)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A node-mode runner on a machine without cards: no kernel database lookup, no card wait, no real
    evidence commands, a fast health poll."""
    monkeypatch.setattr(rental_main, "combined_kernel_db_installed", lambda *args, **kwargs: False)
    monkeypatch.setattr(rental_main, "CARD_FREE_WAIT_S", 0.0)
    monkeypatch.setattr(rental_main, "FAILURE_EVIDENCE", ())
    monkeypatch.setattr(serve, "_HEALTH_POLL_S", 0.1)
    monkeypatch.delenv(serve.COMBINED_KERNEL_DB_ENV, raising=False)

    def make(*extra):
        args = rental_main._parse_args(["--mode", "node", "--model", str(tmp_path / "model"), "--results",
                                        str(tmp_path / "results"), "--run-id", "r", "--variants", "A", *extra])
        runner = rental_main.Runner(args, rental_main.load_variants())
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        return runner

    return make


def tiny_server(monkeypatch, code: str) -> None:
    """The tiny gate starts `python -c code` as its server and builds no checkpoint."""
    monkeypatch.setattr(rental_main, "TINY_PORT", free_port())
    monkeypatch.setattr(rental_main, "ensure_tiny_checkpoint",
                        lambda ckpt, **kwargs: Path(ckpt).mkdir(parents=True, exist_ok=True))  # "built": None
    monkeypatch.setattr(rental_main, "tiny_serve_command", lambda ckpt, port: [sys.executable, "-c", code])


# --------------------------------------------------------------------------- a crashed server is not a timeout


def test_tiny_gate_reports_a_crashed_server_as_failed_with_its_error_lines(node, monkeypatch, capsys):
    # Session 1: "TIMED_OUT, tiny server not healthy within 1200 s", written 220 s after a Python exception.
    tiny_server(monkeypatch, CRASH)
    runner = node()
    assert runner.run_gate("tiny") is False
    status = read_json(runner.run_dir / "tiny" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert re.fullmatch(r"tiny server exited with code 3 after \d+ s", status["reason"])
    assert status["errors"] == ["IndexError: tuple index out of range"]
    assert status["seconds"] < 60  # far below the 1200 s health limit
    out = capsys.readouterr().out
    assert "tiny: FAILED" in out and "tiny server exited with code 3" in out
    assert "    IndexError: tuple index out of range" in out
    assert str(runner.run_dir / "tiny" / "server0.log") in out

    assert report.main([str(runner.run_dir)]) == 0
    capsys.readouterr()
    md = (runner.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "| tiny | FAILED | tiny server exited with code 3 after" in md
    assert "IndexError: tuple index out of range" in md
    assert read_json(runner.run_dir / "summary.json")["gates"]["tiny"]["errors"] == status["errors"]


def test_tiny_gate_times_out_only_when_the_server_is_still_running(node, monkeypatch, capsys):
    tiny_server(monkeypatch, HANG)
    monkeypatch.setattr(rental_main, "TINY_HEALTH_TIMEOUT_S", 1.0)
    runner = node()
    assert runner.run_gate("tiny") is False
    status = read_json(runner.run_dir / "tiny" / "status.json")
    assert status["status"] == int(GateStatus.TIMED_OUT)
    assert status["reason"] == "tiny server not healthy within 1 s"
    assert "tiny: TIMED_OUT" in capsys.readouterr().out


def crashing_variant(runner, monkeypatch, code: str):
    variant = copy.deepcopy(runner.variant_by_id("A"))
    variant["servers"][0]["port"] = free_port()
    monkeypatch.setattr(rental_main.serve, "build_command",
                        lambda eff, srv, model, name: ([sys.executable, "-c", code], dict(os.environ)))
    return variant


def test_variant_reports_a_crashed_server_as_failed_with_its_error_lines(node, monkeypatch, capsys):
    runner = node()
    variant = crashing_variant(runner, monkeypatch, CRASH)
    assert runner.run_variant(variant) is False  # variant A stops the run
    status = read_json(runner.run_dir / "A" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert re.fullmatch(r"server 0 exited with code 3 after \d+ s", status["reason"])
    assert status["errors"] == ["server0: IndexError: tuple index out of range"]
    out = capsys.readouterr().out
    assert "variant A: FAILED" in out and "server 0 exited with code 3" in out
    assert "    server0: IndexError: tuple index out of range" in out
    assert str(runner.run_dir / "A" / "server0.log") in out

    assert report.main([str(runner.run_dir)]) == 0
    md = (runner.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "- reason: server 0 exited with code 3 after" in md
    assert "- last error line: server0: IndexError: tuple index out of range" in md


def test_variant_times_out_only_when_the_server_is_still_running(node, monkeypatch, capsys):
    runner = node("--server-timeout", "1")
    variant = crashing_variant(runner, monkeypatch, HANG)
    assert runner.run_variant(variant) is False
    status = read_json(runner.run_dir / "A" / "status.json")
    assert status["status"] == int(GateStatus.TIMED_OUT)
    assert status["reason"] == "server 0 not healthy within 1 s"


# --------------------------------------------------------------------------- the first token


def test_first_token_runs_before_the_workload_and_the_warm_pass(tmp_path, capsys, monkeypatch):
    events = []
    real_first = gates.first_token
    real_workload = rental_main.Runner.ensure_workload
    real_warm = rental_main.Runner.run_warm_pass

    def first_token(*args, **kwargs):
        events.append("first_token")
        return real_first(*args, **kwargs)

    def ensure_workload(self, *args, **kwargs):
        events.append("workload")
        return real_workload(self, *args, **kwargs)

    def run_warm_pass(self, *args, **kwargs):
        events.append("warm_pass")
        return real_warm(self, *args, **kwargs)

    monkeypatch.setattr(gates, "first_token", first_token)
    monkeypatch.setattr(rental_main.Runner, "ensure_workload", ensure_workload)
    monkeypatch.setattr(rental_main.Runner, "run_warm_pass", run_warm_pass)
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "ft", "--variants", "A,C") == 0
    out = capsys.readouterr().out
    assert events == ["first_token", "workload", "warm_pass", "first_token", "first_token", "workload", "warm_pass"]

    run = tmp_path / "results" / "ft"
    records = read_json(run / "A" / "first-token.json")
    assert len(records) == 1
    record = records[0]
    assert record["server"] == 0 and record["status"] == int(GateStatus.PASSED)
    assert record["prompt"] == gates.FIRST_TOKEN_PROMPT and record["max_tokens"] == 8
    assert record["text"] and record["tokens"] == 3 and record["seconds"] >= 0
    assert [r["server"] for r in read_json(run / "C" / "first-token.json")] == [0, 1]
    assert re.search(r"^variant A server0: first token after \d+\.\d s: '.+'$", out, re.MULTILINE)
    assert re.search(r"^variant C server1: first token after \d+\.\d s: '.+'$", out, re.MULTILINE)


def test_first_token_failure_fails_the_variant_like_a_startup_failure(tmp_path, capsys, monkeypatch):
    def no_token(base_url, model, timeout_s):
        return GateStatus.FAILED, {"prompt": gates.FIRST_TOKEN_PROMPT, "max_tokens": 8, "text": None, "tokens": None,
                                   "seconds": 12.5, "error": "HTTP 500 from /v1/completions: EngineDeadError"}

    def no_workload(self, *args, **kwargs):
        raise AssertionError("nothing may run after a failed first token")

    monkeypatch.setattr(gates, "first_token", no_token)
    monkeypatch.setattr(rental_main.Runner, "ensure_workload", no_workload)
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "ftfail", "--variants", "A,B")
    out = capsys.readouterr().out
    assert rc == 1
    assert out.splitlines()[-1] == "RUN ftfail: FAILED at A"
    assert "variant A server0: no first token after 12.5 s: HTTP 500 from /v1/completions: EngineDeadError" in out
    run = tmp_path / "results" / "ftfail"
    status = read_json(run / "A" / "status.json")
    assert status["status"] == int(GateStatus.FAILED)
    assert status["reason"] == "server 0 first token: HTTP 500 from /v1/completions: EngineDeadError"
    assert read_json(run / "A" / "first-token.json")[0]["error"].startswith("HTTP 500")
    assert not (run / "workload.json").exists() and not (run / "A" / "warm-requests.jsonl").exists()
    assert not (run / "B" / "status.json").exists()  # A stops the run, as a startup failure does


def test_first_token_uses_the_coherence_request_timeout(node, monkeypatch):
    seen = []

    def first_token(base_url, model, timeout_s):
        seen.append((base_url, model, timeout_s))
        return GateStatus.PASSED, {"prompt": "p", "max_tokens": 8, "text": " Paris", "tokens": 8, "seconds": 41.23}

    monkeypatch.setattr(gates, "first_token", first_token)
    runner = node()
    assert runner.run_first_token("A", ["http://127.0.0.1:8000"], runner.run_dir / "A") is None
    assert seen == [("http://127.0.0.1:8000", "flash-next", rental_main.COHERENCE_REQUEST_TIMEOUT_SKIP_WARMUP_S)]


# --------------------------------------------------------------------------- the timeouts in use


def test_timeout_flags_default_to_the_constants():
    args = rental_main._parse_args(["--model", "m", "--results", "r"])
    assert args.load_read_timeout == rental_main.LOAD_REQUEST_TIMEOUT_S == 1800.0
    assert args.warm_request_timeout == rental_main.WARM_PASS_REQUEST_TIMEOUT_S == 3600.0


def test_timeout_flags_reach_the_load_the_manifest_and_the_summary(tmp_path, capsys, monkeypatch):
    seen = {}
    real_run_load = rental_main.agent_load.run_load

    async def run_load(workload, servers, model, out_path, **kwargs):
        seen[Path(out_path).name] = kwargs["request_timeout_s"]
        return await real_run_load(workload, servers, model, out_path, **kwargs)

    monkeypatch.setattr(rental_main.agent_load, "run_load", run_load)
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "to", "--variants", "A", "--load-read-timeout", "77", "--warm-request-timeout", "88")
    capsys.readouterr()
    assert rc == 0
    assert seen == {"warm-requests.jsonl": 88.0, "requests.jsonl": 77.0}
    run = tmp_path / "results" / "to"
    manifest = read_json(run / "manifest.json")
    assert manifest["load_read_timeout_s"] == 77.0
    assert manifest["warm_request_timeout_s"] == 88.0
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Load read timeout: 77.0 s" in md
    assert "- Warm request timeout: 88.0 s" in md


def test_default_timeouts_are_recorded_too(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "todef", "--variants", "A") == 0
    capsys.readouterr()
    run = tmp_path / "results" / "todef"
    manifest = read_json(run / "manifest.json")
    assert (manifest["load_read_timeout_s"], manifest["warm_request_timeout_s"]) == (1800.0, 3600.0)
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Load read timeout: 1800.0 s" in md and "- Warm request timeout: 3600.0 s" in md


# --------------------------------------------------------------------------- the bundle commit


def test_bundle_commit_falls_back_to_bundle_txt(monkeypatch):
    monkeypatch.setattr(rental_main, "git_commit", lambda root=None: None)  # a bundle has no .git
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: BUNDLE_TEXT)
    assert rental_main.bundle_commit() == SHA
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: "plugin sha: not-a-sha\n")
    assert rental_main.bundle_commit() is None
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: None)
    assert rental_main.bundle_commit() is None
    monkeypatch.setattr(rental_main, "git_commit", lambda root=None: "f" * 40)  # a checkout: git wins
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: BUNDLE_TEXT)
    assert rental_main.bundle_commit() == "f" * 40


def test_bundle_text_is_read_next_to_the_plugin_folder(tmp_path):
    plugin = tmp_path / "bundle" / "plugin"
    plugin.mkdir(parents=True)
    assert rental_main.bundle_text(plugin) is None
    (tmp_path / "bundle" / "BUNDLE.txt").write_text(BUNDLE_TEXT, encoding="utf-8")
    assert rental_main.bundle_text(plugin) == BUNDLE_TEXT
    assert rental_main.git_commit(plugin) is None  # no checkout there


def test_a_bundle_run_records_its_commit_and_keys_the_recipe_cache_with_it(tmp_path, capsys, monkeypatch):
    # Session 1 manifests have "bundle_commit": null, and the cache key was always "unknown".
    monkeypatch.setattr(rental_main, "git_commit", lambda root=None: None)
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: BUNDLE_TEXT)
    model = make_model(tmp_path)
    rc = run_stub(tmp_path, model, "bc", "--variants", "A", "--recipe-cache", str(tmp_path / "rc"))
    capsys.readouterr()
    assert rc == 0
    manifest = read_json(tmp_path / "results" / "bc" / "manifest.json")
    assert manifest["bundle_commit"] == SHA
    assert manifest["recipe_cache"]["dir"] == "%s/%s" % (tmp_path / "rc", SHA[:12])
    assert "- Bundle commit: %s" % SHA in (tmp_path / "results" / "bc" / "summary.md").read_text(encoding="utf-8")


def test_a_resumed_run_keys_the_cache_with_the_bundle_that_runs_now(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(rental_main, "git_commit", lambda root=None: None)
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: BUNDLE_TEXT)
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "bc2", "--variants", "A,B") == 0
    newer = "a" * 40  # a newer bundle extracted over the old one before the resume
    monkeypatch.setattr(rental_main, "bundle_text", lambda root=None: "plugin sha: %s\n" % newer)
    assert run_stub(tmp_path, model, "bc2", "--from", "B", "--variants", "A,B", "--recipe-cache", "/rc") == 0
    capsys.readouterr()
    manifest = read_json(tmp_path / "results" / "bc2" / "manifest.json")
    assert manifest["bundle_commit"] == newer
    assert manifest["recipe_cache"]["dir"] == "/rc/%s" % newer[:12]


# --------------------------------------------------------------------------- evidence for off-node debugging


def test_manifest_environment_keeps_the_kernel_and_loader_variables(tmp_path, capsys, monkeypatch):
    wanted = {
        "FLASHINFER_GAUDI_COMBINED_KERNEL_DB": "1",
        "GC_KERNEL_PATH": "/usr/lib/habanalabs/libtpc_kernels.so",
        "GDN_FUSED_GAUDI_KERNEL_LIBRARY": "/x/libgdn_fused_gaudi2.so",
        "HABANA_VISIBLE_MODULES": "0,1",
        "PT_HPU_LAZY_MODE": "1",
        "VLLM_SKIP_WARMUP": "true",
        "VLLM_GDN_PREFILL_STATE_FP32_MAX_TOKENS": "4096",  # a setting, not a secret
    }
    for key, value in wanted.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("VLLM_API_KEY", "hf_secret_too")
    monkeypatch.setenv("UNRELATED_VARIABLE", "x")
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "env", "--variants", "A") == 0
    capsys.readouterr()
    env = read_json(tmp_path / "results" / "env" / "manifest.json")["env"]
    for key, value in wanted.items():
        assert env[key] == value
    assert "UNRELATED_VARIABLE" not in env
    assert "hf_secret" not in json.dumps(env)  # a token or a key never reaches the evidence
    assert env["HF_TOKEN"] == env["VLLM_API_KEY"] == "<redacted>"


def test_planned_env_records_the_fault_handler(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "fh", "--variants", "A") == 0
    capsys.readouterr()
    status = read_json(tmp_path / "results" / "fh" / "A" / "status.json")
    assert status["servers"][0]["planned_env"]["PYTHONFAULTHANDLER"] == "1"


def test_tiny_env_has_the_fault_handler(node):
    assert node().tiny_env()["PYTHONFAULTHANDLER"] == "1"


def evidence_commands(tmp_path: Path) -> tuple:
    return (
        ("pip-freeze.txt", [sys.executable, "-c", "print('torch==2.9.0')"], 30.0, "w"),
        ("hl-smi.txt", [str(tmp_path / "no-hl-smi-here")], 30.0, "a"),
    )


def test_write_failure_evidence_is_best_effort(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rental_main.write_failure_evidence(run, "tiny", evidence_commands(tmp_path))
    freeze = (run / "pip-freeze.txt").read_text(encoding="utf-8")
    assert "torch==2.9.0" in freeze and "tiny" in freeze.splitlines()[0]
    first = (run / "hl-smi.txt").read_text(encoding="utf-8")
    assert "tiny" in first and "FileNotFoundError" in first  # no hl-smi on this machine: recorded, not raised
    rental_main.write_failure_evidence(run, "A", evidence_commands(tmp_path))
    second = (run / "hl-smi.txt").read_text(encoding="utf-8")
    assert second.startswith(first) and "failure of A" in second  # every failure adds its own snapshot
    rental_main.write_failure_evidence(tmp_path / "missing" / "run", "tiny", evidence_commands(tmp_path))  # no raise


def test_a_failed_gate_and_a_failed_variant_leave_the_evidence(node, tmp_path, monkeypatch, capsys):
    runner = node()
    monkeypatch.setattr(rental_main, "FAILURE_EVIDENCE", evidence_commands(tmp_path))
    monkeypatch.setattr(rental_main.gates, "build_check", lambda *args, **kwargs: (GateStatus.PASSED, {}))
    assert runner.run_gate("build") is True
    assert not (runner.run_dir / "pip-freeze.txt").exists()  # nothing failed yet

    tiny_server(monkeypatch, CRASH)
    assert runner.run_gate("tiny") is False
    assert "torch==2.9.0" in (runner.run_dir / "pip-freeze.txt").read_text(encoding="utf-8")
    assert "failure of tiny" in (runner.run_dir / "hl-smi.txt").read_text(encoding="utf-8")

    variant = crashing_variant(runner, monkeypatch, CRASH)
    assert runner.run_variant(variant) is False
    assert "failure of A" in (runner.run_dir / "hl-smi.txt").read_text(encoding="utf-8")


def test_a_recorded_only_gate_failure_leaves_the_evidence_too(node, tmp_path, monkeypatch):
    runner = node()
    monkeypatch.setattr(rental_main, "FAILURE_EVIDENCE", evidence_commands(tmp_path))
    monkeypatch.setattr(rental_main.gates, "card_tests",
                        lambda *args, **kwargs: (GateStatus.FAILED, {"files": [], "reason": "a card test failed"}))
    assert runner.run_gate("card_tests") is True  # recorded only: the run goes on
    assert "failure of card_tests" in (runner.run_dir / "hl-smi.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- --from needs a run to resume


def from_args(tmp_path: Path, model: Path, *extra: str) -> list:
    return ["--mode", "stub", "--model", str(model), "--results", str(tmp_path / "results"), *extra] + STUB_ARGS


def test_from_refuses_a_run_without_earlier_results(tmp_path, capsys):
    model = make_model(tmp_path)
    run = tmp_path / "results" / "fresh"
    rc = rental_main.main(from_args(tmp_path, model, "--run-id", "fresh", "--from", "B"))
    out = capsys.readouterr().out
    assert rc == 2
    assert str(run) in out and "manifest.json" in out and "--from B" in out
    assert not run.exists()  # no empty run is started


def test_from_refuses_a_run_folder_that_only_has_the_install_logs(tmp_path, capsys):
    # container_entry.sh creates the folder and writes its install logs before the orchestrator starts.
    model = make_model(tmp_path)
    run = tmp_path / "results" / "half"
    run.mkdir(parents=True)
    (run / "install-1.log").write_text("ok\n", encoding="utf-8")
    (run / "toolchain.json").write_text("{}", encoding="utf-8")
    rc = rental_main.main(from_args(tmp_path, model, "--run-id", "half", "--from", "A"))
    out = capsys.readouterr().out
    assert rc == 2
    assert str(run) in out and "manifest.json" in out
    assert sorted(p.name for p in run.iterdir()) == ["install-1.log", "toolchain.json"]  # nothing was written


def test_from_validates_the_phase_name_before_anything_else(tmp_path, capsys, monkeypatch):
    def no_lookup(*args, **kwargs):
        raise AssertionError("nothing may run before the phase name is checked")

    monkeypatch.setattr(rental_main, "combined_kernel_db_installed", no_lookup)
    model = make_model(tmp_path)
    argv = ["--mode", "node", "--model", str(model), "--results", str(tmp_path / "results"), "--from", "tinny"]
    rc = rental_main.main(argv)  # no --run-id either: the phase name is still the first complaint
    out = capsys.readouterr().out
    assert rc == 2
    assert "--from tinny is not a phase of this run" in out
    assert "preflight, build, fused_gdn, tiny, card_tests, A, B, C, report" in out
    assert not (tmp_path / "results").exists()


def test_from_a_variant_that_is_not_selected_is_refused(tmp_path, capsys):
    model = make_model(tmp_path)
    rc = rental_main.main(from_args(tmp_path, model, "--run-id", "x", "--variants", "A,B", "--from", "C"))
    assert rc == 2
    assert "--from C is not a phase of this run" in capsys.readouterr().out
