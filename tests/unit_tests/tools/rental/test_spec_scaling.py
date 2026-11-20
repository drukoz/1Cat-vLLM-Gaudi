# SPDX-License-Identifier: Apache-2.0
"""``tools/flash_next/rental/spec_scaling.py`` and ``mtp_real_card.sh`` against a stand-in streaming server.

The stand-in streams two tokens per chunk (an acceptance length of 2), counts drafts on ``/metrics`` and records
every request, so the tests see what the probe sent: greedy rounds carry ``temperature: 0``, sampled rounds no
temperature at all, and no round ever sends ``min_tokens``.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.flash_next.rental import spec_scaling

REPO = Path(__file__).resolve().parents[4]
STAND_IN = textwrap.dedent('''
    import json, sys, threading, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def serve(port, speculative=True, refuse_sampled=False, requests=None):
        counts = {"drafts": 0, "accepted": 0}
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):  # noqa: N802
                lines = []
                if self.path == "/metrics" and speculative:
                    with lock:
                        lines = ['vllm:spec_decode_num_drafts_total{engine="0"} %d' % counts["drafts"],
                                 'vllm:spec_decode_num_draft_tokens_total{engine="0"} %d' % (3 * counts["drafts"]),
                                 'vllm:spec_decode_num_accepted_tokens_total{engine="0"} %d' % counts["accepted"],
                                 'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} %d'
                                 % counts["accepted"]]
                body = ("\\n".join(lines) + "\\n").encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if requests is not None:
                    requests.append(request)
                if "temperature" not in request and refuse_sampled:
                    body = b'{"error": {"message": "greedy requests only: VLLM_HPU_SPEC_SAMPLED_VERIFY=1"}}'
                    self.send_response(400)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                total = request["max_tokens"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                sent = 0
                while sent < total:
                    chunk = min(2, total - sent)
                    time.sleep(0.002)
                    event = {"choices": [{"text": "ab"[:chunk]}]}
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\\n\\n")
                    self.wfile.flush()
                    sent += chunk
                    with lock:
                        counts["drafts"] += 1
                        counts["accepted"] += chunk - 1
                usage = {"choices": [], "usage": {"completion_tokens": total, "prompt_tokens": 12}}
                self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\\n\\ndata: [DONE]\\n\\n")
                self.wfile.flush()

        return ThreadingHTTPServer(("127.0.0.1", port), Handler)

    if __name__ == "__main__":
        port = int(sys.argv[sys.argv.index("--port") + 1])
        serve(port, speculative="--speculative-config" in sys.argv).serve_forever()
''')


@pytest.fixture
def stand_in(tmp_path):
    module_path = tmp_path / "stand_in_vllm.py"
    module_path.write_text(STAND_IN)
    namespace: dict = {"__name__": "stand_in_vllm"}
    exec(compile(STAND_IN, str(module_path), "exec"), namespace)  # noqa: S102 - the test's own stand-in
    servers = []

    def start(**kwargs):
        requests: list = []
        server = namespace["serve"](0, requests=requests, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1], requests

    yield start, module_path
    for server in servers:
        server.shutdown()


def probe(tmp_path, port, *options):
    out = tmp_path / "scaling.json"
    assert spec_scaling.main(["--port", str(port), "--out", str(out), "--tokens", "32", "--warm-tokens", "8",
                              "--timeout", "30", *options]) == 0
    return json.loads(out.read_text())


def test_levels_report_step_time_tokens_per_step_and_acceptance(stand_in, tmp_path, capsys):
    start, _ = stand_in
    port, requests = start()
    data = probe(tmp_path, port, "--levels", "1,4")
    assert [(r["streams"], r["rep"], r["ok"]) for r in data["levels"]] == [(1, 0, 1), (4, 0, 4), (4, 1, 4)]
    for record in data["levels"]:
        assert record["tokens_per_chunk_p50"] == pytest.approx(2.0)
        assert record["spec"]["acceptance_length"] == pytest.approx(2.0)
        assert record["spec"]["position_rates"][0] == pytest.approx(1.0)
        assert record["steady_ms_per_token_p50"] == pytest.approx(record["ms_per_step_p50"] / 2.0)
        assert record["min_tokens"] is False and record["sampled"] is False
        assert record["texts"]["0"].startswith("abab")
    assert sorted(data["levels"][1]["texts"]) == ["0", "1", "2", "3"]
    assert all(request["temperature"] == 0 and "min_tokens" not in request for request in requests)
    printed = capsys.readouterr().out
    assert "MEAS  N=4" in printed and "acceptance length=2.00 (per position 1.00)" in printed and "TEXT  N=4" in printed


def test_sampled_rounds_send_no_temperature(stand_in, tmp_path):
    start, _ = stand_in
    port, requests = start()
    data = probe(tmp_path, port, "--levels", "1", "--repeat-top", "1", "--sampled-levels", "2")
    assert [(r["streams"], r["sampled"], r["ok"]) for r in data["sampled"]] == [(2, True, 2)]
    sampled = [request for request in requests if "temperature" not in request]
    assert len(sampled) == 4  # the warm and the measured round of two streams
    assert all("min_tokens" not in request and "top_k" not in request for request in requests)


def test_a_refused_sampled_round_is_recorded_and_the_probe_goes_on(stand_in, tmp_path, capsys):
    start, _ = stand_in
    port, _ = start(refuse_sampled=True)
    data = probe(tmp_path, port, "--levels", "1", "--repeat-top", "1", "--sampled-levels", "2", "--ctx", "2:300")
    assert data["sampled"][0]["warm_failed"] is True and "HTTP 400" in data["sampled"][0]["error_samples"][0]
    assert data["ctx"][0]["ok"] == 2 and data["ctx"][0]["ctx_tokens"] == 300
    assert "the warm round failed" in capsys.readouterr().out


def test_a_server_without_speculation_gives_the_lines_without_acceptance(stand_in, tmp_path, capsys):
    start, _ = stand_in
    port, _ = start(speculative=False)
    data = probe(tmp_path, port, "--levels", "2", "--repeat-top", "1")
    assert data["levels"][0]["spec"] is None and data["levels"][0]["ok"] == 2
    assert "acceptance length=- (per position -)" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_job_script_starts_the_server_runs_the_probe_and_stops(stand_in, tmp_path):
    """mtp_real_card.sh with a stand-in ``vllm`` (and ``setsid``, which macOS lacks) first on PATH."""
    _, module_path = stand_in
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "vllm").write_text("#!/bin/bash\nexec %s %s \"$@\"\n" % (sys.executable, module_path))
    (bin_dir / "setsid").write_text("#!/bin/bash\nexec \"$@\"\n")
    (bin_dir / "python3").write_text("#!/bin/bash\nexec %s \"$@\"\n" % sys.executable)
    for name in ("vllm", "setsid", "python3"):
        path = bin_dir / name
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "results"
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]), FARM_OUT=str(out), FARM_NCARDS="4",
               FARM_PORT_BASE="18941", MTP_MODEL=str(tmp_path), MTP_HEALTH_TIMEOUT="60", PYTHONPATH=str(REPO),
               VLLM_HPU_MTP_TIMING="1")
    done = subprocess.run(["bash", str(REPO / "tools/flash_next/rental/mtp_real_card.sh"), "3", "--", "--levels",
                           "1,2", "--tokens", "16", "--warm-tokens", "4", "--sampled-levels", "2"],
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "SERVER-HEALTHY" in done.stdout and "PROBE-RC 0" in done.stdout and "DONE (ok)" in done.stdout
    data = json.loads((out / "scaling.json").read_text())
    assert [r["streams"] for r in data["levels"]] == [1, 2, 2] and data["sampled"][0]["ok"] == 2
    versions = (out / "versions.txt").read_text()
    assert "--tensor-parallel-size 4" in versions and "num_speculative_tokens" in versions
    assert "VLLM_HPU_MTP_TIMING=1" in versions and "VLLM_HPU_HC_FUSED=1" in versions  # the job's env, the preset
    lines = (out / "probe.log").read_text().splitlines()
    assert sum(line.startswith("MEAS ") for line in lines) == 3 and sum(line.startswith("SAMPL ") for line in lines) == 1


def test_a_probe_that_measured_nothing_exits_1(tmp_path):
    """A server that died at its first prompt (nothing listens): no round measured anything, exit 1."""
    out = tmp_path / "scaling.json"
    assert spec_scaling.main(["--port", "1", "--out", str(out), "--levels", "1", "--tokens", "8", "--warm-tokens",
                              "4", "--timeout", "5"]) == 1
    assert json.loads(out.read_text())["levels"][0]["warm_failed"] is True


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_arms_script_runs_every_arm_in_its_folder_and_a_fallback_after_a_dead_server(tmp_path):
    """mtp_arms.sh with a stand-in card script: arms in order, each in FARM_OUT/NAME with its variables, the
    probe options passed on; an arm whose server died (exit 3) runs once more as NAME-fallback with the
    fallback variables on top."""
    card = tmp_path / "card.sh"
    card.write_text(textwrap.dedent('''\
        #!/bin/bash
        mkdir -p "$FARM_OUT"
        { echo "k=$1"; shift; echo "args=$*"; env | grep -E '^(ARM_|VLLM_HPU_MTP_DRAFT_LOOP)' | sort; } >"$FARM_OUT/seen.txt"
        if [ "${ARM_DIES:-0}" = 1 ] && [ "${VLLM_HPU_MTP_DRAFT_LOOP:-1}" != 0 ]; then exit 3; fi
        exit 0
        '''))
    out = tmp_path / "results"
    env = dict(os.environ, FARM_OUT=str(out), MTP_ARMS_CARD_SCRIPT=str(card), MTP_ARMS_PAUSE="0")
    done = subprocess.run(["bash", str(REPO / "tools/flash_next/rental/mtp_arms.sh"),
                           "k3|3|ARM_A=1 VLLM_HPU_MTP_DRAFT_LOOP=1 ARM_DIES=1|--levels 1,32 --sampled-levels 32",
                           "k0|0||--levels 1"], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    seen_k3 = (out / "k3" / "seen.txt").read_text()
    assert "k=3" in seen_k3 and "args=-- --levels 1,32 --sampled-levels 32" in seen_k3 and "ARM_A=1" in seen_k3
    fallback = (out / "k3-fallback" / "seen.txt").read_text()
    assert "VLLM_HPU_MTP_DRAFT_LOOP=0" in fallback and "ARM_A=1" in fallback
    seen_k0 = (out / "k0" / "seen.txt").read_text()
    assert "k=0" in seen_k0 and "args=-- --levels 1\n" in seen_k0 and "ARM_A" not in seen_k0
    assert not (out / "k0-fallback").exists()
    assert "arm k3 rc=3" in done.stdout and "arm k3-fallback rc=0" in done.stdout and "done, exit 0" in done.stdout
