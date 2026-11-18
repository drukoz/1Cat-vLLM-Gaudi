# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/rental/monitor.py (the M4 metrics/hl-smi sampler, RUK-2709).

``parse_prometheus`` on a canned exposition text (two label sets of ``vllm:prefix_cache_hits_total``,
a ``_per_pos`` metric, HELP/TYPE lines, a ``_created`` sample, a NaN gauge), ``to_sample`` nulls and
key order, and the CLI loop: an end-to-end run against a tiny ``http.server``, the ``hlsmi.csv`` rows
of ``--hlsmi-cmd``, the single warning for a missing command, and SIGTERM exiting 0 within 2 s.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.flash_next.rental.monitor import main, parse_prometheus, to_sample
from tools.flash_next.rental.schema import METRICS_KEYS

REPO_ROOT = Path(__file__).resolve().parents[4]

EXPOSITION = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="flash-next"} 2
vllm:num_requests_running{engine="1",model_name="flash-next"} 1
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="flash-next"} NaN
# HELP vllm:prefix_cache_hits_total Prefix cache hits from KV and processor caching.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="flash-next"} 3
vllm:prefix_cache_hits_total{engine="1",model_name="flash-next"} 4
vllm:prefix_cache_hits_created{engine="0",model_name="flash-next"} 1764000000.0
vllm:prefix_cache_queries_total{engine="0",model_name="flash-next"} 100
vllm:num_preemptions_total{engine="0",model_name="flash-next"} 0
# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="flash-next"} 9
vllm:spec_decode_num_accepted_tokens_per_pos{engine="0",model_name="flash-next",position="0"} 5
vllm:spec_decode_num_accepted_tokens_per_pos{engine="0",model_name="flash-next",position="1"} 3
vllm:spec_decode_num_drafts_total{engine="0",model_name="flash-next"} 7
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="flash-next"} 21
vllm:generation_tokens_total{engine="0",model_name="flash-next"} 1e3
vllm:prompt_tokens_total{engine="0",model_name="with space}inside"} 5000 1764000000000
"""


class _Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        body = EXPOSITION.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_parse_prometheus_sums_label_sets():
    values = parse_prometheus(EXPOSITION)
    assert values["vllm:num_requests_running"] == 3.0
    assert values["vllm:prefix_cache_hits"] == 7.0
    assert values["vllm:prefix_cache_queries"] == 100.0
    assert values["vllm:num_preemptions"] == 0.0
    assert values["vllm:generation_tokens"] == 1000.0
    assert values["vllm:prompt_tokens"] == 5000.0
    assert "vllm:prefix_cache_hits_created" not in values
    assert "vllm:prefix_cache_hits_total" not in values
    assert "vllm:kv_cache_usage_perc" not in values


def test_parse_prometheus_per_pos_is_not_accepted_tokens():
    values = parse_prometheus(EXPOSITION)
    assert values["vllm:spec_decode_num_accepted_tokens"] == 9.0
    assert values["vllm:spec_decode_num_accepted_tokens_per_pos"] == 8.0


def test_to_sample_keys_nulls_and_values():
    sample = to_sample(parse_prometheus(EXPOSITION), 2, 123.5)
    assert list(sample.keys()) == list(METRICS_KEYS)
    assert sample["t"] == 123.5
    assert sample["server"] == 2
    assert sample["running"] == 3.0
    assert sample["prefix_hits"] == 7.0
    assert sample["spec_accepted"] == 9.0
    assert sample["kv_usage"] is None
    assert sample["waiting"] is None


def test_to_sample_all_null():
    sample = to_sample({}, 0, 1.0)
    assert list(sample.keys()) == list(METRICS_KEYS)
    assert all(sample[key] is None for key in METRICS_KEYS if key not in ("t", "server"))


def test_end_to_end_metrics(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rc = main([
            "--servers", f"http://127.0.0.1:{server.server_address[1]}", "--out-dir",
            str(tmp_path), "--interval", "0.2", "--duration", "1", "--no-hlsmi"
        ])
    finally:
        server.shutdown()
        server.server_close()
    assert rc == 0
    rows = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(rows) >= 3
    assert all(list(row.keys()) == list(METRICS_KEYS) for row in rows)
    assert all(row["server"] == 0 for row in rows)
    assert all(row["prefix_hits"] == 7.0 for row in rows)
    assert all(row["spec_accepted"] == 9.0 for row in rows)
    assert not (tmp_path / "hlsmi.csv").exists()


def test_hlsmi_custom_cmd_writes_time_prefixed_rows(tmp_path):
    rc = main([
        "--servers", "http://127.0.0.1:1", "--out-dir",
        str(tmp_path), "--interval", "0.2", "--duration", "0.5", "--hlsmi-cmd", "echo 1,2,3"
    ])
    assert rc == 0
    lines = (tmp_path / "hlsmi.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("t")
    assert len(lines) >= 3
    assert all(re.fullmatch(r"\d+\.\d+,1,2,3", line) for line in lines[1:])
    rows = _read_jsonl(tmp_path / "metrics.jsonl")
    assert rows and all(row["running"] is None for row in rows)


def test_hlsmi_missing_command_warns_once_and_run_exits_zero(tmp_path, capsys):
    rc = main([
        "--servers", "http://127.0.0.1:1", "--out-dir",
        str(tmp_path), "--interval", "0.2", "--duration", "1", "--hlsmi-cmd", "no-such-binary-xyz"
    ])
    assert rc == 0
    stderr = capsys.readouterr().err
    assert stderr.count("hl-smi") == 1
    rows = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(rows) >= 3


def test_sigterm_exits_zero_within_two_seconds(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.Popen([
        sys.executable, "-m", "tools.flash_next.rental.monitor", "--servers", "http://127.0.0.1:1", "--out-dir",
        str(tmp_path), "--interval", "0.2", "--no-hlsmi"
    ],
                            cwd=str(REPO_ROOT),
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    time.sleep(0.8)
    proc.terminate()
    try:
        rc = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise
    assert rc == 0
    assert _read_jsonl(tmp_path / "metrics.jsonl")
