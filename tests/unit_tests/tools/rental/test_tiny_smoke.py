# SPDX-License-Identifier: Apache-2.0
"""Tests for the tiny smoke test (tools/flash_next/rental/tiny_smoke.py, RUK-2709).

The card and vLLM are replaced by two small executables: a fake checkpoint builder that writes
``reference.json``, and a fake server that answers ``/health``, ``/v1/models`` and ``/v1/completions``
from that reference. Everything else is the real code: the server start, the health wait, the
comparison, the fallback check and the stop. Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools.flash_next.rental import gates
from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental import serve, tiny_smoke

REAL_COMBINED_KERNEL_DB_INSTALLED = orchestrator.combined_kernel_db_installed
BUNDLE_SHA = "abcdef0123456789abcdef0123456789abcdef01"
CACHE_KEY = BUNDLE_SHA[:12]  # the per-bundle subfolder of the recipe cache, as in the run

FAKE_BUILDER = r"""
import json, os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
code = int(os.environ.get("FAKE_BUILD_EXIT", "0"))
if code:
    print("tiny checkpoint: no transformers", file=sys.stderr)
    sys.exit(code)
out.mkdir(parents=True, exist_ok=True)
prompts = []
for i, n in enumerate([1, 7]):
    prompts.append({"prompt_ids": list(range(1, n + 1)),
                    "generated_ids": list(range(100 + 16 * i, 100 + 16 * i + 16))})
reference = {"schema": 2, "prompts": prompts}
if os.environ.get("FAKE_BUILD_SETTINGS"):
    reference["build"] = json.loads(os.environ["FAKE_BUILD_SETTINGS"])
(out / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
"""

FAKE_SERVER = r"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
port = int(sys.argv[sys.argv.index("--port") + 1])
reference = json.loads(Path(sys.argv[sys.argv.index("--reference") + 1]).read_text(encoding="utf-8"))
answers = {tuple(p["prompt_ids"]): p["generated_ids"] for p in reference["prompts"]}
keys = ["PT_HPU_LAZY_MODE", "VLLM_HPU_FLASH_NEXT_QSA_MODE", "VLLM_HPU_RECURRENT_STATE_POOL",
        "VLLM_HPU_GDN_CORE_DECODE", "VLLM_HPU_GDN_CORE_VERIFY", "VLLM_SKIP_WARMUP"]
if os.environ.get("FAKE_SERVER_ENV_KEYS"):
    keys += os.environ["FAKE_SERVER_ENV_KEYS"].split(",")
Path("server_env.json").write_text(json.dumps({k: os.environ.get(k) for k in keys}), encoding="utf-8")
if os.environ.get("FAKE_SERVER_LINE"):
    print(os.environ["FAKE_SERVER_LINE"], flush=True)
if os.environ.get("FAKE_SERVER_EXIT"):
    print("(Worker_TP1 pid=7) ERROR 11-18 22:36:00 [multiproc_executor.py:1055] IndexError: tuple index out of range",
          flush=True)
    sys.exit(int(os.environ["FAKE_SERVER_EXIT"]))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids = list(answers[tuple(payload["prompt"])])
        if os.environ.get("FAKE_SERVER_WRONG"):
            ids[3] = -1
        self._send({"choices": [{"token_ids": ids}]})


try:
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
except KeyboardInterrupt:
    pass
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """Replace the two commands the smoke test starts with the fake builder and the fake server."""
    builder = tmp_path / "fake_builder.py"
    builder.write_text(FAKE_BUILDER, encoding="utf-8")
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER, encoding="utf-8")
    monkeypatch.setattr(tiny_smoke, "checkpoint_command",
                        lambda ckpt: [sys.executable, str(builder), "--out", str(ckpt)])
    monkeypatch.setattr(tiny_smoke, "serve_command",
                        lambda ckpt, port: [sys.executable, str(server), "--port", str(port), "--reference",
                                            str(Path(ckpt) / "reference.json")])
    monkeypatch.setattr(serve, "_HEALTH_POLL_S", 0.1)
    monkeypatch.setattr(orchestrator, "bundle_commit", lambda: BUNDLE_SHA)
    for name in ("FAKE_BUILD_EXIT", "FAKE_SERVER_EXIT", "FAKE_SERVER_LINE", "FAKE_SERVER_WRONG",
                 "FAKE_SERVER_ENV_KEYS"):
        monkeypatch.delenv(name, raising=False)
    # the fake builder records the settings the real generator would record for the smoke test's command
    monkeypatch.setenv("FAKE_BUILD_SETTINGS", json.dumps(dict(
        orchestrator.tiny_build_settings(), schema=2, prompt_lengths=[1, 7], max_new_tokens=16)))
    port = free_port()
    out = tmp_path / "out"
    return out, port


@pytest.fixture(autouse=True)
def no_combined_kernel_db(monkeypatch):
    """The combined kernel database is not installed on the test machine; a test that wants it overrides this."""
    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: False)


def run(out: Path, port: int, *extra: str) -> tuple[int, dict]:
    code = tiny_smoke.main(["--out", str(out), "--port", str(port), "--health-timeout", "20", *extra])
    return code, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def by_mode(summary: dict) -> dict:
    return {entry["mode"]: entry for entry in summary["modes"]}


def test_both_modes_pass_and_the_cards_are_released(fakes, capsys):
    out, port = fakes
    code, summary = run(out, port)
    assert code == 0
    assert summary["passed"] is True
    modes = by_mode(summary)
    assert list(modes) == ["pytorch", "fused"]
    assert modes["pytorch"]["status"] == "passed" and modes["fused"]["status"] == "passed"
    assert not serve.port_in_use(port)  # each server is stopped again
    printed = capsys.readouterr().out
    assert "tiny smoke [pytorch]: PASSED" in printed
    assert "tiny smoke [fused]: PASSED" in printed
    for mode in ("pytorch", "fused"):
        assert (out / mode / "server0.log").is_file()
        assert json.loads((out / mode / "details.json").read_text(encoding="utf-8"))["n_prompts"] == 2


def test_each_mode_gets_a_variant_environment_with_its_gdn_flags(fakes):
    out, port = fakes
    code, _ = run(out, port)
    assert code == 0
    pytorch = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    fused = json.loads((out / "fused" / "server_env.json").read_text(encoding="utf-8"))
    for env in (pytorch, fused):  # what serve.server_environment gives every rental server
        assert env["PT_HPU_LAZY_MODE"] == "1"
        assert env["VLLM_HPU_FLASH_NEXT_QSA_MODE"] == "1"
        assert env["VLLM_HPU_RECURRENT_STATE_POOL"] == "1"
    assert (pytorch["VLLM_HPU_GDN_CORE_DECODE"], pytorch["VLLM_HPU_GDN_CORE_VERIFY"]) == ("0", "0")
    assert (fused["VLLM_HPU_GDN_CORE_DECODE"], fused["VLLM_HPU_GDN_CORE_VERIFY"]) == ("1", "0")


def test_the_servers_skip_the_warmup_by_default(fakes, capsys):
    out, port = fakes
    code, _ = run(out, port)
    assert code == 0
    for mode in ("pytorch", "fused"):
        env = json.loads((out / mode / "server_env.json").read_text(encoding="utf-8"))
        assert env["VLLM_SKIP_WARMUP"] == "true"
    printed = capsys.readouterr().out
    assert "warmup skipped" in printed


def test_full_warmup_leaves_skip_warmup_unset(fakes, capsys):
    out, port = fakes
    code, _ = run(out, port, "--gdn", "pytorch", "--full-warmup")
    assert code == 0
    env = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    assert env["VLLM_SKIP_WARMUP"] is None
    assert "full warmup" in capsys.readouterr().out


def test_env_pairs_reach_the_server_environment(fakes, monkeypatch):
    out, port = fakes
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "A,B")
    code, _ = run(out, port, "--gdn", "pytorch", "--env", "A=1", "--env", "B=2")
    assert code == 0
    env = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    assert (env["A"], env["B"]) == ("1", "2")


def test_env_pairs_are_applied_last(fakes):
    out, port = fakes
    code, _ = run(out, port, "--gdn", "fused", "--env", "VLLM_HPU_GDN_CORE_DECODE=0")
    assert code == 0
    env = json.loads((out / "fused" / "server_env.json").read_text(encoding="utf-8"))
    assert env["VLLM_HPU_GDN_CORE_DECODE"] == "0"


def test_the_pytorch_mode_gets_the_recipe_cache_and_the_fused_mode_does_not(fakes, tmp_path, monkeypatch, capsys):
    # Changed on purpose: the cache folder now has the run's per-bundle subfolder (the plugin commit).
    out, port = fakes
    rc = tmp_path / "rc"
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", serve.RECIPE_CACHE_ENV)
    code, summary = run(out, port, "--recipe-cache", str(rc))
    assert code == 0
    pytorch = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    fused = json.loads((out / "fused" / "server_env.json").read_text(encoding="utf-8"))
    assert pytorch[serve.RECIPE_CACHE_ENV] == "%s/%s,false,8192" % (rc, CACHE_KEY)
    assert (rc / CACHE_KEY).is_dir()
    assert fused[serve.RECIPE_CACHE_ENV] is None
    modes = by_mode(summary)
    assert modes["pytorch"]["recipe_cache"] == "%s/%s" % (rc, CACHE_KEY)
    assert modes["fused"]["recipe_cache"] == serve.RECIPE_CACHE_OFF_REASON
    printed = capsys.readouterr().out
    assert "starting the server (recipe cache on" in printed
    assert "starting the server (recipe cache off" in printed
    assert "PASSED (server up in" in printed
    assert "recipe cache on)" in printed and "recipe cache off)" in printed


def test_without_recipe_cache_no_server_gets_the_variable(fakes, monkeypatch):
    out, port = fakes
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", serve.RECIPE_CACHE_ENV)
    code, summary = run(out, port)
    assert code == 0
    modes = by_mode(summary)
    for mode in ("pytorch", "fused"):
        env = json.loads((out / mode / "server_env.json").read_text(encoding="utf-8"))
        assert env[serve.RECIPE_CACHE_ENV] is None
        assert modes[mode]["recipe_cache"] == serve.RECIPE_CACHE_NONE


def test_the_recipe_cache_key_is_unknown_without_a_commit(fakes, tmp_path, monkeypatch):
    out, port = fakes
    monkeypatch.setattr(orchestrator, "bundle_commit", lambda: None)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", serve.RECIPE_CACHE_ENV)
    code, _ = run(out, port, "--gdn", "pytorch", "--recipe-cache", str(tmp_path / "rc"))
    assert code == 0
    env = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    assert env[serve.RECIPE_CACHE_ENV] == "%s/unknown,false,8192" % (tmp_path / "rc")


def test_env_pairs_override_the_recipe_cache(fakes, tmp_path, monkeypatch):
    out, port = fakes
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", serve.RECIPE_CACHE_ENV)
    code, _ = run(out, port, "--gdn", "pytorch", "--recipe-cache", str(tmp_path / "rc"),
                  "--env", "PT_HPU_RECIPE_CACHE_CONFIG=/x,false,1")
    assert code == 0
    env = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    assert env[serve.RECIPE_CACHE_ENV] == "/x,false,1"


def test_with_the_combined_kernel_database_both_modes_get_the_cache(fakes, tmp_path, monkeypatch):
    out, port = fakes
    rc = tmp_path / "rc"
    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: True)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", serve.RECIPE_CACHE_ENV)
    code, summary = run(out, port, "--recipe-cache", str(rc))
    assert code == 0
    modes = by_mode(summary)
    for mode in ("pytorch", "fused"):
        env = json.loads((out / mode / "server_env.json").read_text(encoding="utf-8"))
        assert env[serve.RECIPE_CACHE_ENV] == "%s/%s,false,8192" % (rc, CACHE_KEY)
        assert modes[mode]["recipe_cache"] == "%s/%s" % (rc, CACHE_KEY)


def test_the_operator_switch_keeps_the_cache_from_the_fused_mode(fakes, tmp_path, monkeypatch):
    # tiny_smoke.sh --no-combined-kernel-db: the database is installed, the switch is 0 in the container.
    out, port = fakes
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    (lib / serve.COMBINED_KERNEL_DB).write_bytes(b"\0")
    real_run_step = gates.run_step

    def run_step(cmd, cwd, timeout_s, env=None):
        if list(cmd[1:3]) == ["-c", gates.GDN_LIB_LOOKUP_CODE]:
            return subprocess.CompletedProcess(cmd, 0, "%s\n" % lib, "")
        return real_run_step(cmd, cwd, timeout_s, env)

    monkeypatch.setattr(gates, "run_step", run_step)
    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", REAL_COMBINED_KERNEL_DB_INSTALLED)
    monkeypatch.setenv(serve.COMBINED_KERNEL_DB_ENV, "0")
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (serve.RECIPE_CACHE_ENV, serve.COMBINED_KERNEL_DB_ENV))
    code, summary = run(out, port, "--recipe-cache", str(tmp_path / "rc"))
    assert code == 0
    fused = json.loads((out / "fused" / "server_env.json").read_text(encoding="utf-8"))
    assert fused[serve.RECIPE_CACHE_ENV] is None
    assert fused[serve.COMBINED_KERNEL_DB_ENV] == "0"
    assert by_mode(summary)["fused"]["recipe_cache"] == serve.RECIPE_CACHE_OFF_REASON
    pytorch = json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))
    assert pytorch[serve.RECIPE_CACHE_ENV] is not None  # the torch path registers no kernel database


def test_env_without_equals_is_a_usage_error(fakes):
    out, port = fakes
    with pytest.raises(SystemExit) as info:
        tiny_smoke.main(["--out", str(out), "--port", str(port), "--env", "NOEQUALS"])
    assert info.value.code == 2


def test_one_mode_on_request(fakes):
    out, port = fakes
    code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 0
    assert list(by_mode(summary)) == ["pytorch"]
    assert not (out / "fused").exists()


def test_server_that_dies_is_a_failure_with_its_error_line(fakes, monkeypatch, capsys):
    out, port = fakes
    monkeypatch.setenv("FAKE_SERVER_EXIT", "3")
    code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 1
    assert summary["passed"] is False
    entry = by_mode(summary)["pytorch"]
    assert entry["status"] == "failed"
    assert "exited" in entry["reason"] and "3" in entry["reason"]
    assert entry["errors"] == ["IndexError: tuple index out of range"]
    printed = capsys.readouterr().out
    assert "tiny smoke [pytorch]: FAILED" in printed
    assert "IndexError: tuple index out of range" in printed
    assert str(out / "pytorch" / "server0.log") in printed


def test_wrong_tokens_are_a_failure(fakes, monkeypatch):
    out, port = fakes
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 1
    entry = by_mode(summary)["pytorch"]
    assert entry["status"] == "failed"
    # the reason's wording changed with the per-prompt record: it names the shortest failing prompt
    assert entry["reason"] == ("2 of 2 prompts failed; the shortest failing prompt has 1 tokens: "
                               "15 of 16 tokens match, first difference at token 3")


def test_fused_mode_fails_when_the_server_fell_back_to_the_torch_path(fakes, monkeypatch):
    out, port = fakes
    line = ("WARNING 11-18 22:26:38 [__init__.py:119] GDN core decode (VLLM_HPU_GDN_CORE_DECODE=1) has no HPU "
            "kernel: the gdn_fused_ops extension is not loaded")
    monkeypatch.setenv("FAKE_SERVER_LINE", line)
    code, summary = run(out, port)
    assert code == 1
    modes = by_mode(summary)
    assert modes["pytorch"]["status"] == "passed"  # the torch path is what this mode asks for
    assert modes["fused"]["status"] == "failed"
    assert "torch" in modes["fused"]["reason"]
    assert modes["fused"]["fallback_lines"] == [line]


def test_a_verify_fallback_line_does_not_fail_the_fused_mode(fakes, monkeypatch):
    out, port = fakes
    monkeypatch.setenv("FAKE_SERVER_LINE",
                       "GDN core verify (VLLM_HPU_GDN_CORE_VERIFY=1) has no HPU kernel: the gdn_fused_ops extension")
    code, summary = run(out, port, "--gdn", "fused")
    assert code == 0  # the smoke test keeps the verify op off: only the decode kernel is checked
    assert by_mode(summary)["fused"]["status"] == "passed"


def test_checkpoint_build_failure_starts_no_server(fakes, monkeypatch, capsys):
    out, port = fakes
    monkeypatch.setenv("FAKE_BUILD_EXIT", "2")
    code, summary = run(out, port)
    assert code == 1
    assert summary["passed"] is False
    assert summary["modes"] == []
    assert "tiny checkpoint failed (exit 2)" in summary["reason"]
    assert "no transformers" in summary["reason"]
    assert not (out / "pytorch").exists()
    assert "tiny smoke: FAILED" in capsys.readouterr().out


def test_a_checkpoint_folder_is_built_once_and_reused(fakes, tmp_path, monkeypatch):
    out, port = fakes
    ckpt = tmp_path / "kept-ckpt"
    code, _ = run(out, port, "--gdn", "pytorch", "--ckpt", str(ckpt))
    assert code == 0
    assert (ckpt / "reference.json").is_file() and not (out / "ckpt").exists()
    monkeypatch.setenv("FAKE_BUILD_EXIT", "2")  # a second build would fail: the folder must be reused
    code, summary = run(tmp_path / "out2", port, "--gdn", "pytorch", "--ckpt", str(ckpt))
    assert code == 0
    assert by_mode(summary)["pytorch"]["status"] == "passed"


def test_a_checkpoint_built_with_other_settings_is_not_reused(fakes, tmp_path, monkeypatch, capsys):
    # A folder kept from an earlier bundle (GDN head dim 16, where the fused kernel can never run) is rebuilt.
    out, port = fakes
    ckpt = tmp_path / "kept-ckpt"
    old = dict(json.loads(os.environ["FAKE_BUILD_SETTINGS"]), gdn_head_dim=16)
    wanted = os.environ["FAKE_BUILD_SETTINGS"]
    monkeypatch.setenv("FAKE_BUILD_SETTINGS", json.dumps(old))
    code, _ = run(out, port, "--gdn", "pytorch", "--ckpt", str(ckpt))
    assert code == 0
    assert json.loads((ckpt / "reference.json").read_text(encoding="utf-8"))["build"]["gdn_head_dim"] == 16
    capsys.readouterr()
    monkeypatch.setenv("FAKE_BUILD_SETTINGS", wanted)
    code, _ = run(tmp_path / "out2", port, "--gdn", "pytorch", "--ckpt", str(ckpt))
    assert code == 0
    assert json.loads((ckpt / "reference.json").read_text(encoding="utf-8"))["build"]["gdn_head_dim"] == 128
    printed = capsys.readouterr().out
    assert "gdn_head_dim" in printed and "reusing" not in printed


def test_port_in_use_is_a_failure_before_any_start(fakes):
    out, port = fakes
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 1
    assert "already in use" in by_mode(summary)["pytorch"]["reason"]
    assert not (out / "pytorch" / "server_env.json").exists()


def test_unknown_mode_is_a_usage_error(fakes):
    out, port = fakes
    with pytest.raises(SystemExit) as info:
        tiny_smoke.main(["--out", str(out), "--port", str(port), "--gdn", "pytorch,turbo"])
    assert info.value.code == 2


def test_the_commands_are_the_tiny_gate_commands(tmp_path):
    # One definition for the gate and the smoke test: the smoke test must start what the run will start.
    ckpt = tmp_path / "ckpt"
    assert tiny_smoke.checkpoint_command(ckpt) == orchestrator.tiny_checkpoint_command(ckpt)
    assert tiny_smoke.serve_command(ckpt, 8010) == orchestrator.tiny_serve_command(ckpt, 8010)
    # Changed on purpose: the model length now follows the reference (4096 while there is none: 512 was
    # below the 700 and 2600 token reference prompts), and the GDN layers are built at head dim 128.
    assert orchestrator.tiny_serve_command(ckpt, 8010) == [
        "vllm", "serve", str(ckpt), "--served-model-name", "tiny", "--tensor-parallel-size", "8",
        "--max-model-len", "4096", "--max-num-seqs", "4", "--language-model-only", "--trust-remote-code",
        "--port", "8010", "--host", "127.0.0.1",
    ]
    assert orchestrator.tiny_checkpoint_command(ckpt) == [
        sys.executable, "tools/flash_next/make_tiny_checkpoint.py", "--out", str(ckpt), "--fixture",
        orchestrator.TINY_FIXTURE, "--gdn-head-dim", "128",
    ]


def test_the_smoke_servers_get_the_fault_handler(fakes, monkeypatch):
    out, port = fakes
    monkeypatch.delenv("PYTHONFAULTHANDLER", raising=False)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "PYTHONFAULTHANDLER")
    code, _ = run(out, port, "--gdn", "pytorch")
    assert code == 0
    assert json.loads((out / "pytorch" / "server_env.json").read_text(encoding="utf-8"))["PYTHONFAULTHANDLER"] == "1"


def test_the_error_lines_are_the_orchestrators():
    assert tiny_smoke.error_lines is serve.error_lines


def test_error_lines_keeps_the_last_distinct_errors():
    log = "\n".join([
        "(Worker_TP0 pid=1) INFO 11-18 22:27:10 [runtime.py:32] skip_warmup: False",
        "(Worker_TP0 pid=1) ERROR 11-18 22:27:11 [multiproc_executor.py:944] TypeError: bad keyword 'bias_vl'",
        "(Worker_TP1 pid=2) ERROR 11-18 22:27:11 [multiproc_executor.py:944] TypeError: bad keyword 'bias_vl'",
        "(APIServer pid=9) RuntimeError: Engine core initialization failed. See root cause above.",
    ])
    assert tiny_smoke.error_lines(log) == [
        "TypeError: bad keyword 'bias_vl'",
        "RuntimeError: Engine core initialization failed. See root cause above.",
    ]
    assert tiny_smoke.error_lines("all fine\n") == []
