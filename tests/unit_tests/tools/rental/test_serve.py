# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the rental harness server runner (plan M4, RUK-2709).

No vLLM is started: the command and environment are compared against literal
lists, `parse_startup` runs on canned log text, and the process helpers are
exercised with a plain `http.server` child (which answers 404 on /health).
"""

from __future__ import annotations

import os
import socket
import sys
import time

import pytest

from tools.flash_next.rental import schema, serve

MODEL = "/models/flash-next-fp8"
SERVED_NAME = "flash-next"

SERVER_TP8 = {"cards": "0,1,2,3,4,5,6,7", "tp": 8, "ep": True, "port": 8020, "agents": 32}
SERVER_TP4 = {"cards": "0,1,2,3", "tp": 4, "ep": False, "port": 8021, "agents": 16}


def make_variant(**overrides):
    variant = {
        "id": "C",
        "title": "baseline",
        "servers": [],
        "env": {},
        "serve_args": ["--enable-chunked-prefill", "--disable-log-requests"],
        "inherit_spec_from": None,
        "on_fail": 0,
        "optional": False,
    }
    variant.update(overrides)
    return variant


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_server_proc(tmp_path, port):
    cmd = [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]
    return serve.start_server(cmd, os.environ.copy(), tmp_path / "server.log", tmp_path)


class TestBuildCommand:

    def test_tp8_ep_command_is_exact(self):
        cmd, env = serve.build_command(make_variant(), SERVER_TP8, MODEL, SERVED_NAME)
        assert cmd == [
            "vllm",
            "serve",
            MODEL,
            "--served-model-name",
            SERVED_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            "8020",
            "--tensor-parallel-size",
            "8",
            "--enable-expert-parallel",
            "--max-model-len",
            "131072",
            "--max-num-seqs",
            "32",
            "--enable-prefix-caching",
            "--language-model-only",
            "--trust-remote-code",
            "--gpu-memory-utilization",
            "0.92",
            "--max-num-batched-tokens",
            "8192",
            "--long-prefill-token-threshold",
            "2048",
            "--enable-prompt-tokens-details",
            "--enable-chunked-prefill",
            "--disable-log-requests",
        ]
        assert env["HABANA_VISIBLE_MODULES"] == "0,1,2,3,4,5,6,7"

    def test_tp4_without_ep(self):
        cmd, _ = serve.build_command(make_variant(serve_args=[]), SERVER_TP4, MODEL, SERVED_NAME)
        assert "--enable-expert-parallel" not in cmd
        assert cmd == [
            "vllm",
            "serve",
            MODEL,
            "--served-model-name",
            SERVED_NAME,
            "--host",
            "127.0.0.1",
            "--port",
            "8021",
            "--tensor-parallel-size",
            "4",
            "--max-model-len",
            "131072",
            "--max-num-seqs",
            "16",
            "--enable-prefix-caching",
            "--language-model-only",
            "--trust-remote-code",
            "--gpu-memory-utilization",
            "0.92",
            "--max-num-batched-tokens",
            "8192",
            "--long-prefill-token-threshold",
            "2048",
            "--enable-prompt-tokens-details",
        ]

    def test_serve_args_appended_last_in_order(self):
        cmd, _ = serve.build_command(make_variant(serve_args=["--a", "--b", "1", "--c"]), SERVER_TP4, MODEL,
                                     SERVED_NAME)
        assert cmd[-4:] == ["--a", "--b", "1", "--c"]

    def test_env_defaults(self, monkeypatch):
        monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
        monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
        _, env = serve.build_command(make_variant(), SERVER_TP8, MODEL, SERVED_NAME)
        assert env["HABANA_VISIBLE_MODULES"] == "0,1,2,3,4,5,6,7"
        assert env["VLLM_HPU_FLASH_NEXT_QSA_MODE"] == "1"
        assert env["VLLM_HPU_RECURRENT_STATE_POOL"] == "1"
        assert env["PT_HPU_LAZY_MODE"] == "1"
        assert "VLLM_COMPACT_GDN" not in env
        assert env["PATH"] == os.environ["PATH"]

    def test_variant_env_wins_over_defaults(self, monkeypatch):
        monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
        variant = make_variant(env={"VLLM_HPU_FLASH_NEXT_QSA_MODE": "0", "PT_HPU_LAZY_MODE": "1", "MY_FLAG": "on"})
        _, env = serve.build_command(variant, SERVER_TP4, MODEL, SERVED_NAME)
        assert env["VLLM_HPU_FLASH_NEXT_QSA_MODE"] == "0"
        assert env["MY_FLAG"] == "on"
        assert "VLLM_COMPACT_GDN" not in env


CANNED_LOG = """\
(Worker_TP0) GDN layers detected (36): VLLM_USE_HYBRID_CACHE=1, VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=0, VLLM_COMPACT_GDN=0
INFO HPU KV cache size: 2,500,648 tokens, Maximum concurrency for 131,072 tokens per request: 19.08x
Usable num_blocks: 2,862, actual allocated num_blocks (max across layers): 2,863 (_PAD_BLOCK_ID=127, _PAD_SLOT_ID=-1)
(Worker_TP0 pid=44318) INFO 11-17 [hpu_model_runner.py:6893] Loading model weights took 21.6012 GB
INFO 11-17 [hpu_model_runner.py:8651] Warmup finished in 404 secs, allocated 1.2 GiB of device memory
"""


class TestParseStartup:

    def test_keys_are_exactly_startup_keys(self):
        record = serve.parse_startup("", 0, 0.0)
        assert tuple(record.keys()) == schema.STARTUP_KEYS

    def test_canned_log(self):
        record = serve.parse_startup(CANNED_LOG, 1, 512.5)
        assert record == {
            "server": 1,
            "usable_num_blocks": 2862,
            "kv_cache_tokens": 2500648,
            "max_concurrency": 19.08,
            "max_concurrency_len": 131072,
            "weights_gib": 21.6012,
            "warmup_seconds": 404.0,
            "gdn_layers_detected": "(36): VLLM_USE_HYBRID_CACHE=1, VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=0, "
            "VLLM_COMPACT_GDN=0",
            "startup_seconds": 512.5,
        }

    def test_gpu_word_and_first_match_wins(self):
        log = ("(Worker_TP0 pid=1) GPU KV cache size: 1,000 tokens, "
               "Maximum concurrency for 131,072 tokens per request: 1.00x\n"
               "GPU KV cache size: 2,000 tokens, Maximum concurrency for 131,072 tokens per request: 2.00x\n")
        record = serve.parse_startup(log, 0, 0.0)
        assert record["kv_cache_tokens"] == 1000
        assert record["max_concurrency"] == 1.00

    def test_empty_log_is_all_null(self):
        record = serve.parse_startup("", 2, 42.0)
        assert record["server"] == 2
        assert record["startup_seconds"] == 42.0
        for key in schema.STARTUP_KEYS:
            if key not in ("server", "startup_seconds"):
                assert record[key] is None, key


class TestProcessHandling:

    def test_start_server_uses_given_cwd(self, tmp_path):
        log_path = tmp_path / "cwd.log"
        proc = serve.start_server([sys.executable, "-c", "import os; print(os.getcwd())"], os.environ.copy(), log_path,
                                  tmp_path)
        try:
            assert proc.wait(timeout=30) == 0
        finally:
            serve.stop_server(proc)
        assert log_path.read_text().strip() == str(tmp_path)

    def test_start_server_new_session_and_stop_kills_group(self, tmp_path):
        proc = http_server_proc(tmp_path, free_port())
        try:
            assert os.getpgid(proc.pid) == proc.pid  # start_new_session: the group is the child
            serve.stop_server(proc)
            assert proc.poll() is not None
            with pytest.raises(ProcessLookupError):
                os.killpg(proc.pid, 0)  # no process left in the group
        finally:
            serve.stop_server(proc)  # idempotent: must not raise

    @pytest.mark.parametrize("parent_waits", [True, False])
    def test_stop_server_kills_workers_left_in_group(self, tmp_path, parent_waits):
        # A worker that ignores SIGINT outlives its server (which exits on SIGINT, or has already exited).
        worker = ("import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
                  "print('worker up', flush=True); time.sleep(600)")
        code = (f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {worker!r}]); "
                f"time.sleep({600 if parent_waits else 0})")
        log_path = tmp_path / "group.log"
        proc = serve.start_server([sys.executable, "-c", code], os.environ.copy(), log_path, tmp_path)
        try:
            deadline = time.time() + 20
            while "worker up" not in log_path.read_text() and time.time() < deadline:
                time.sleep(0.1)
            assert "worker up" in log_path.read_text()
            started = time.time()
            serve.stop_server(proc, timeout_s=2.0)
            assert time.time() - started < 15
            assert proc.poll() is not None
            with pytest.raises(ProcessLookupError):  # the worker is gone too
                os.killpg(proc.pid, 0)
        finally:
            serve.stop_server(proc, timeout_s=2.0)

    def test_stop_server_on_exited_process_never_raises(self, tmp_path):
        proc = serve.start_server([sys.executable, "-c", "pass"], os.environ.copy(), tmp_path / "x.log", tmp_path)
        assert proc.wait(timeout=30) == 0
        serve.stop_server(proc)

    def test_wait_healthy_fails_on_plain_http_server(self, tmp_path):
        port = free_port()
        proc = http_server_proc(tmp_path, port)
        try:
            started = time.time()
            assert serve.wait_healthy(port, 6.0, proc) is False  # 404 on /health
            assert time.time() - started < 20
        finally:
            serve.stop_server(proc)

    def test_wait_healthy_false_quickly_when_process_exited(self, tmp_path):
        proc = serve.start_server([sys.executable, "-c", "pass"], os.environ.copy(), tmp_path / "x.log", tmp_path)
        started = time.time()
        try:
            assert serve.wait_healthy(free_port(), 60.0, proc) is False
        finally:
            serve.stop_server(proc)
        assert time.time() - started < 5
