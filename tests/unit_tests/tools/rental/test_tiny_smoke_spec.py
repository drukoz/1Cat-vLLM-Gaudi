# SPDX-License-Identifier: Apache-2.0
"""Tests for the speculative mode of the tiny smoke test (tools/flash_next/rental/tiny_smoke.py ``--spec K``).

As in test_tiny_smoke.py the checkpoint builder and the server are small fakes; the fake server here also
records its command line, answers ``/metrics`` with the speculative counters a server started with
``--speculative-config`` exports, and fails on request: at startup when speculation and the fused verification
op are both on (``FAKE_DIE_VERIFY``), or whenever speculation is on (``FAKE_DIE_SPEC``), or with other tokens
under speculation (``FAKE_SPEC_OTHER``: within the reference's top-5 at the last token, so the reference
comparison still passes where it should not). Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental import serve, tiny_smoke

BUNDLE_SHA = "abcdef0123456789abcdef0123456789abcdef01"

FAKE_BUILDER = r"""
import json, os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
prompts = []
for i, n in enumerate([1, 7]):
    prompts.append({"prompt_ids": list(range(1, n + 1)),
                    "generated_ids": list(range(100 + 16 * i, 100 + 16 * i + 16))})
reference = {"schema": 2, "prompts": prompts}
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
spec = "--speculative-config" in sys.argv
k = json.loads(sys.argv[sys.argv.index("--speculative-config") + 1])["num_speculative_tokens"] if spec else 0
verify = os.environ.get("VLLM_HPU_GDN_CORE_VERIFY")
Path("server_cmd.json").write_text(json.dumps({"argv": sys.argv[1:], "verify": verify}), encoding="utf-8")
if spec and (os.environ.get("FAKE_DIE_SPEC") or (os.environ.get("FAKE_DIE_VERIFY") and verify == "1")):
    print("(Worker_TP0 pid=7) ERROR 11-20 18:00:00 [multiproc_executor.py:944] RuntimeError: verify died", flush=True)
    sys.exit(3)
if spec and os.environ.get("FAKE_VERIFY_OFF_LINE") and verify == "1":
    print("WARNING 11-20 18:00:00 [qwen3_5.py:405] GDN core verify (VLLM_HPU_GDN_CORE_VERIFY=1) is off for "
          "layers.0: conv state rows; using the torch verification path", flush=True)
counts = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "pos": [0] * k}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, kind="application/json"):
        body = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/metrics":
            lines = ['vllm:num_requests_running{engine="0",model_name="tiny"} 0.0']
            if spec:
                lines += ['vllm:spec_decode_num_drafts_total{engine="0",model_name="tiny"} %d' % counts["drafts"],
                          'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="tiny"} %d' % counts["draft_tokens"],
                          'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="tiny"} %d' % counts["accepted"]]
                lines += ['vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="tiny",position="%d"} %d'
                          % (p, c) for p, c in enumerate(counts["pos"])]
            self._send("\n".join(lines) + "\n", "text/plain")
            return
        self._send(json.dumps({"ok": True}))

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids = list(answers[tuple(payload["prompt"])])[:payload["max_tokens"]]
        if spec:
            # 5 verification steps per request, 1 draft accepted in each: acceptance length 2.0
            counts["drafts"] += 5
            counts["draft_tokens"] += 5 * k
            counts["accepted"] += 5
            counts["pos"][0] += 5
            if os.environ.get("FAKE_SPEC_OTHER") and len(ids) > 3:
                ids[-1] = ids[-1] + 1000
        self._send(json.dumps({"choices": [{"token_ids": ids}]}))


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
    builder = tmp_path / "fake_builder.py"
    builder.write_text(FAKE_BUILDER, encoding="utf-8")
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER, encoding="utf-8")
    monkeypatch.setattr(tiny_smoke, "checkpoint_command", lambda ckpt: [sys.executable, str(builder), "--out", str(ckpt)])
    monkeypatch.setattr(tiny_smoke, "serve_command",
                        lambda ckpt, port: [sys.executable, str(server), "--port", str(port), "--reference",
                                            str(Path(ckpt) / "reference.json")])
    monkeypatch.setattr(serve, "_HEALTH_POLL_S", 0.1)
    monkeypatch.setattr(orchestrator, "bundle_commit", lambda: BUNDLE_SHA)
    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: False)
    for name in ("FAKE_DIE_SPEC", "FAKE_DIE_VERIFY", "FAKE_SPEC_OTHER", "FAKE_VERIFY_OFF_LINE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAKE_BUILD_SETTINGS", json.dumps(dict(
        orchestrator.tiny_build_settings(), schema=2, prompt_lengths=[1, 7], max_new_tokens=16)))
    return tmp_path / "out", free_port()


def run(out: Path, port: int, *extra: str) -> tuple[int, dict]:
    code = tiny_smoke.main(["--out", str(out), "--port", str(port), "--health-timeout", "20", *extra])
    return code, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def by_mode(summary: dict) -> dict:
    return {entry["mode"]: entry for entry in summary["modes"]}


def test_the_speculative_mode_follows_its_mode_and_passes_on_the_fused_verification(fakes, capsys):
    out, port = fakes
    code, summary = run(out, port, "--gdn", "fused", "--spec", "3")
    assert code == 0 and summary["passed"] is True
    modes = by_mode(summary)
    assert list(modes) == ["fused", "fused+mtp3"]
    spec = modes["fused+mtp3"]
    assert spec["status"] == "passed" and spec["rung"] == tiny_smoke.SPEC_RUNG_VERIFY
    assert [r["rung"] for r in spec["rungs"]] == [tiny_smoke.SPEC_RUNG_VERIFY]
    started = json.loads((out / "fused+mtp3" / "verify" / "server_cmd.json").read_text(encoding="utf-8"))
    assert started["verify"] == "1"
    config = json.loads(started["argv"][started["argv"].index("--speculative-config") + 1])
    assert config == {"method": "mtp", "num_speculative_tokens": 3}
    plain = json.loads((out / "fused" / "server_cmd.json").read_text(encoding="utf-8"))
    assert "--speculative-config" not in plain["argv"]
    # the acceptance comes from the counters around the comparison: 2 prompts x 5 steps, 1 accepted each
    assert spec["acceptance"]["acceptance_length"] == pytest.approx(2.0)
    assert spec["acceptance"]["drafts"] == 10
    assert spec["spec_vs_plain"] == {"equal": True, "first_diff": {"1": None, "7": None}}
    printed = capsys.readouterr().out
    assert "tiny smoke [fused+mtp3]: PASSED on rung verify" in printed
    assert "tiny smoke [fused+mtp3]: acceptance length 2.00" in printed
    assert "tokens equal to the same mode without speculation" in printed
    assert not serve.port_in_use(port)


def test_a_failing_fused_verification_falls_back_to_the_torch_verification(fakes, monkeypatch, capsys):
    out, port = fakes
    monkeypatch.setenv("FAKE_DIE_VERIFY", "1")
    code, summary = run(out, port, "--gdn", "fused", "--spec", "3")
    assert code == 0
    spec = by_mode(summary)["fused+mtp3"]
    assert spec["status"] == "passed" and spec["rung"] == tiny_smoke.SPEC_RUNG_TORCH
    assert [(r["rung"], r["status"]) for r in spec["rungs"]] == [("verify", "failed"), ("torch-verify", "passed")]
    assert spec["reason"].startswith("verify: ")
    started = json.loads((out / "fused+mtp3" / "torch-verify" / "server_cmd.json").read_text(encoding="utf-8"))
    assert started["verify"] == "0" and "--speculative-config" in started["argv"]
    assert "PASSED on rung torch-verify" in capsys.readouterr().out


def test_a_verify_off_line_fails_the_verify_rung(fakes, monkeypatch):
    out, port = fakes
    monkeypatch.setenv("FAKE_VERIFY_OFF_LINE", "1")
    code, summary = run(out, port, "--gdn", "fused", "--spec", "3")
    assert code == 0
    spec = by_mode(summary)["fused+mtp3"]
    assert [(r["rung"], r["status"]) for r in spec["rungs"]] == [("verify", "failed"), ("torch-verify", "passed")]
    assert "fused verification op did not run" in spec["rungs"][0]["reason"]


def test_speculation_that_never_serves_is_a_fallback_and_a_failure(fakes, monkeypatch, capsys):
    out, port = fakes
    monkeypatch.setenv("FAKE_DIE_SPEC", "1")
    code, summary = run(out, port, "--gdn", "fused", "--spec", "3")
    assert code == 1 and summary["passed"] is False
    spec = by_mode(summary)["fused+mtp3"]
    assert spec["status"] == "fallback" and spec["rung"] == tiny_smoke.SPEC_RUNG_OFF
    assert [r["status"] for r in spec["rungs"]] == ["failed", "failed", "passed"]
    assert "speculation fell back to off" in spec["reason"]
    off = json.loads((out / "fused+mtp3" / "no-spec" / "server_cmd.json").read_text(encoding="utf-8"))
    assert "--speculative-config" not in off["argv"]
    printed = capsys.readouterr().out
    assert "tiny smoke [fused+mtp3]: FALLBACK" in printed
    assert "RuntimeError: verify died" in printed


def test_tokens_that_differ_from_the_plain_mode_are_reported(fakes, monkeypatch, capsys):
    out, port = fakes
    monkeypatch.setenv("FAKE_SPEC_OTHER", "1")
    code, summary = run(out, port, "--gdn", "fused", "--spec", "3")
    spec = by_mode(summary)["fused+mtp3"]
    # the fake's last token is not the reference's: the reference comparison fails every rung but no-spec
    assert spec["status"] == "fallback"
    assert code == 1


def test_without_spec_there_is_no_speculative_mode(fakes):
    out, port = fakes
    code, summary = run(out, port, "--gdn", "fused")
    assert code == 0
    assert list(by_mode(summary)) == ["fused"]
    assert "streams" not in by_mode(summary)["fused"]


def test_spec_must_not_be_negative(fakes):
    out, port = fakes
    with pytest.raises(SystemExit) as info:
        tiny_smoke.main(["--out", str(out), "--port", str(port), "--spec", "-1"])
    assert info.value.code == 2


def test_compare_streams():
    assert tiny_smoke.compare_streams({1: [1, 2], 7: [3, 4]}, {1: [1, 2], 7: [3, 4]}) == \
        {"equal": True, "first_diff": {"1": None, "7": None}}
    assert tiny_smoke.compare_streams({1: [1, 5], 7: [3, 4]}, {1: [1, 2], 7: [3, 4]}) == \
        {"equal": False, "first_diff": {"1": 1, "7": None}}
    assert tiny_smoke.compare_streams({1: [1]}, {1: [1, 2]})["first_diff"] == {"1": 1}
    assert tiny_smoke.compare_streams(None, {1: [1]}) is None
    assert tiny_smoke.compare_streams({1: [1]}, None) is None
