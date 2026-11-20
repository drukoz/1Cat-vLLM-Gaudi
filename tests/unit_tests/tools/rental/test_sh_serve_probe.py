# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/serve_probe.sh (the real-model server job of the card farm) and farm/decode_scaling.py
(its decode probe), against a stand-in `vllm` that serves /health and streaming /v1/completions on the
port it is given. Runs on CPU; on macOS a Perl stand-in replaces setsid(1)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FARM = REPO_ROOT / "tools" / "flash_next" / "rental" / "farm"
SERVE_PROBE = FARM / "serve_probe.sh"

SETSID_SHIM = r"""#!/usr/bin/perl
use POSIX qw(setsid getpid getpgrp);
my $fork = 0;
while (@ARGV && $ARGV[0] =~ /^-/) { my $a = shift; $fork = 1 if $a eq '-f'; }
if ($fork || getpid() == getpgrp()) { my $pid = fork(); exit 0 if $pid; }
setsid();
exec @ARGV or exit 127;
"""

FAKE_VLLM = r'''#!/usr/bin/env python3
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
with open(os.environ["FAKE_VLLM_RECORD"], "w") as f:
    json.dump({"argv": args, "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_", "PT_HPU"))}}, f)
mode = os.environ.get("FAKE_VLLM_MODE", "ok")
if mode == "die":
    print("Traceback (most recent call last):\nRuntimeError: fake failure", flush=True)
    sys.exit(1)
if mode == "hang":
    time.sleep(600)
port = int(args[args.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = body["max_tokens"]
        with open(os.environ["FAKE_VLLM_REQUESTS"], "a") as f:
            f.write(json.dumps({k: body.get(k) for k in ("max_tokens", "min_tokens", "ignore_eos")}) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n):
            time.sleep(0.002)
            self.wfile.write(b"data: " + json.dumps({"choices": [{"text": "t%d " % i}]}).encode() + b"\n\n")
            self.wfile.flush()
        usage = {"prompt_tokens": 7, "completion_tokens": n}
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
time.sleep(float(os.environ.get("FAKE_VLLM_START_S", "0.5")))
print("fake vllm: listening on %d" % port, flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Rig:

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.out = tmp_path / "out"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "vllm").write_text(FAKE_VLLM.replace("#!/usr/bin/env python3", "#!" + sys.executable))
        (bin_dir / "vllm").chmod(0o755)
        if shutil.which("setsid") is None:
            (bin_dir / "setsid").write_text(SETSID_SHIM)
            (bin_dir / "setsid").chmod(0o755)
        self.port = free_port()
        self.record = tmp_path / "vllm.json"
        self.requests_file = tmp_path / "requests.jsonl"
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("FARM_", "VLLM_", "PT_HPU"))}
        self.env.update({
            "PATH": "%s:%s" % (bin_dir, os.environ.get("PATH", "/usr/bin:/bin")),
            "FARM_PYTHON": sys.executable, "FARM_OUT": str(self.out), "FARM_NCARDS": "4",
            "FARM_PORT_BASE": str(self.port), "FARM_CARDS": "0,1,2,7", "FARM_MODULES": "0,1,2,3",
            "FAKE_VLLM_RECORD": str(self.record), "FAKE_VLLM_REQUESTS": str(self.requests_file),
        })

    def popen(self, *args, **env):
        return subprocess.Popen(["bash", str(SERVE_PROBE), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=dict(self.env, **env), start_new_session=True)

    def run(self, *args, timeout=120, **env):
        proc = self.popen(*args, **env)
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out

    def server(self):
        return json.loads(self.record.read_text())

    def summary(self):
        return json.loads((self.out / "summary.json").read_text())

    def requests(self):
        """Every completion request the stand-in server got, in arrival order; the file is emptied."""
        lines = self.requests_file.read_text().splitlines() if self.requests_file.exists() else []
        self.requests_file.write_text("")
        return [json.loads(line) for line in lines]


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


QUICK = ("--levels", "1,2", "--tokens", "6", "--warm-tokens", "3", "--repeat-top", "1")


def test_a_healthy_server_is_probed_and_stopped(rig):
    rc, out = rig.run(*QUICK, "--", "--max-num-batched-tokens", "4096")
    assert rc == 0, out
    assert "SERVER-HEALTHY" in out and "serve_probe: DONE (ok)" in out
    argv = rig.server()["argv"]
    assert argv[:2] == ["serve", "/models/Qwen3.8-Flash-Next-FP8"]
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4" and "--enable-expert-parallel" in argv
    assert argv[argv.index("--port") + 1] == str(rig.port) and argv[-2:] == ["--max-num-batched-tokens", "4096"]
    assert "--speculative-config" not in argv
    env = rig.server()["env"]
    assert env["VLLM_HPU_HC_FUSED"] == "1" and env["PT_HPU_LAZY_ACC_PAR_MODE"] == "0"
    assert env["VLLM_SKIP_WARMUP"] == "true" and env["VLLM_HPU_GDN_CORE_DECODE"] == "1"
    scaling = json.loads((rig.out / "scaling.json").read_text())
    levels = [m["streams"] for m in scaling["levels"]]
    assert levels == [1, 2] and all(m["ok"] == m["streams"] and m["errors"] == 0 for m in scaling["levels"])
    assert all(m["ms_per_token_p50"] > 0 and m["tok_s_per_stream_p50"] > 0 for m in scaling["levels"])
    assert scaling["levels"][0]["tokens"] == 6 and "aggregate_tok_s" in scaling["levels"][1]
    summary = rig.summary()
    assert summary["result"] == "ok" and summary["tp"] == 4 and summary["ep"] is True
    assert summary["probe_rc"] == 0 and summary["healthy_after_s"] is not None
    assert summary["env"]["FARM_CARDS"] == "0,1,2,7"
    versions = (rig.out / "versions.txt").read_text()
    assert "cards: 0,1,2,7 (modules 0,1,2,3)" in versions and "--tensor-parallel-size 4" in versions
    assert (rig.out / "probe.log").read_text().count("MEAS") == 2
    assert (rig.out / "server0.log").read_text().startswith("fake vllm: listening")


def test_the_probe_sends_no_min_tokens_and_measures_the_mask_in_one_labelled_round(rig):
    # min_tokens makes vLLM mask the logits of every row on every step (MinTokensLogitsProcessor), so the
    # probe fixes the length with max_tokens + ignore_eos and measures the mask once, at the top level.
    rc, out = rig.run(*QUICK)
    assert rc == 0, out
    reqs = rig.requests()
    assert all(r["ignore_eos"] is True for r in reqs)
    plain = [r for r in reqs if r["min_tokens"] is None]
    masked = [r for r in reqs if r["min_tokens"] is not None]
    assert sorted(r["max_tokens"] for r in plain) == [3, 3, 3, 6, 6, 6]  # levels 1 and 2: warm, then measured
    assert sorted(r["max_tokens"] for r in masked) == [3, 3, 6, 6]  # the MINTOK round at level 2: warm, measured
    assert all(r["min_tokens"] == r["max_tokens"] for r in masked)
    scaling = json.loads((rig.out / "scaling.json").read_text())
    assert [m["min_tokens"] for m in scaling["levels"]] == [False, False]
    (mintok,) = scaling["mintok"]
    assert mintok["streams"] == 2 and mintok["min_tokens"] is True and mintok["ok"] == 2 and mintok["tokens"] == 6
    probe_log = (rig.out / "probe.log").read_text()
    assert probe_log.count("MINTOK N=2") == 1 and "with min_tokens" in probe_log and "(without: " in probe_log
    assert probe_log.count("MEAS") == 2  # the MINTOK round is not one of the measured levels


def test_no_mintok_check_skips_the_round_and_min_tokens_sends_it_in_every_round(rig):
    rc, out = rig.run(*QUICK, "--no-mintok-check")
    assert rc == 0, out
    reqs = rig.requests()
    assert len(reqs) == 6 and all(r["min_tokens"] is None for r in reqs)
    assert json.loads((rig.out / "scaling.json").read_text())["mintok"] == []
    assert "MINTOK" not in (rig.out / "probe.log").read_text()
    rc, out = rig.run(*QUICK, "--min-tokens")
    assert rc == 0, out
    reqs = rig.requests()
    assert len(reqs) == 6 and all(r["min_tokens"] == r["max_tokens"] for r in reqs)  # no extra round
    scaling = json.loads((rig.out / "scaling.json").read_text())
    assert scaling["mintok"] == [] and [m["min_tokens"] for m in scaling["levels"]] == [True, True]
    assert "MINTOK" not in (rig.out / "probe.log").read_text()


def test_the_job_env_wins_over_the_preset_and_bare_drops_it(rig):
    rc, out = rig.run(*QUICK, "--no-probe", VLLM_HPU_HC_FUSED="0")
    assert rc == 0, out
    assert rig.server()["env"]["VLLM_HPU_HC_FUSED"] == "0"
    rc, out = rig.run("--no-probe", "--bare")
    assert rc == 0, out
    assert "VLLM_HPU_HC_FUSED" not in rig.server()["env"] and "VLLM_SKIP_WARMUP" not in rig.server()["env"]
    assert not (rig.out / "scaling.json").exists()  # --no-probe: no probe ran


def test_mtp_tp_ep_and_ctx_options(rig):
    rc, out = rig.run("--mtp", "3", "--tp", "8", "--no-ep", "--levels", "1", "--tokens", "4", "--warm-tokens", "2",
                      "--ctx", "2:40", "--max-num-seqs", "16", "--max-model-len", "8192")
    assert rc == 0, out
    argv = rig.server()["argv"]
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    assert spec == {"method": "mtp", "num_speculative_tokens": 3}
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8" and "--enable-expert-parallel" not in argv
    assert argv[argv.index("--max-num-seqs") + 1] == "16" and argv[argv.index("--max-model-len") + 1] == "8192"
    scaling = json.loads((rig.out / "scaling.json").read_text())
    assert scaling["ctx"] and scaling["ctx"][0]["streams"] == 2 and scaling["ctx"][0]["ok"] == 2
    assert rig.summary()["speculative_config"] == '{"method": "mtp", "num_speculative_tokens": 3}'


def test_a_server_that_dies_is_reported_with_its_error(rig):
    rc, out = rig.run(*QUICK, FAKE_VLLM_MODE="die")
    assert rc == 3 and "SERVER-DIED" in out and "RuntimeError: fake failure" in out
    assert rig.summary()["result"] == "server-died"


def test_a_server_that_never_gets_healthy_is_stopped_at_the_timeout(rig):
    rc, out = rig.run(*QUICK, "--health-timeout", "2", FAKE_VLLM_MODE="hang", timeout=200)
    assert rc == 4 and "SERVER-TIMEOUT" in out
    assert rig.summary()["result"] == "server-timeout"


def test_sigterm_stops_the_server_and_writes_the_summary(rig):
    proc = rig.popen("--levels", "1", "--tokens", "20000", "--warm-tokens", "2")
    deadline = time.time() + 60
    while time.time() < deadline and not rig.record.exists():
        time.sleep(0.1)
    time.sleep(2.0)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=180)
    assert proc.returncode == 143, out
    assert "interrupted (SIGTERM)" in out
    assert rig.summary()["result"] == "interrupted"


@pytest.mark.parametrize("args, text", [
    (["--levels", "a"], "--levels takes"), (["--ctx", "32"], "--ctx takes"), (["--tp", "x"], "--tp takes"),
    (["--frob"], "unknown option"),
])
def test_bad_options_are_refused(rig, args, text):
    rc, out = rig.run(*args)
    assert rc == 2 and text in out
    assert not rig.record.exists()
