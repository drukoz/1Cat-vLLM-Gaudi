# SPDX-License-Identifier: Apache-2.0
"""Tests for the stub server's free-port binding and port-file reporting (RUK-3129)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

from tools.flash_next.rental import serve

REPO_ROOT = Path(__file__).resolve().parents[4]


def stub_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_port_zero_writes_port_file(tmp_path):
    port_file = tmp_path / "s.port"
    cmd = [
        sys.executable,
        "-m",
        "tools.flash_next.rental.stub_server",
        "--port",
        "0",
        "--port-file",
        str(port_file),
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=stub_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    try:
        port = serve.wait_port_file(port_file, 15.0, proc)
        assert isinstance(port, int)
        assert port > 0
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as resp:
            assert resp.status == 200
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_held_port_exits_2(tmp_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = int(sock.getsockname()[1])
    try:
        cmd = [
            sys.executable,
            "-m",
            "tools.flash_next.rental.stub_server",
            "--port",
            str(port),
        ]
        done = subprocess.run(cmd, cwd=str(REPO_ROOT), env=stub_env(), capture_output=True, text=True, timeout=30)
        assert done.returncode == 2
        assert "cannot listen" in done.stderr
    finally:
        sock.close()
