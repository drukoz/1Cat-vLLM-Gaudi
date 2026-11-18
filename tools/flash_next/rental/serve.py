# SPDX-License-Identifier: Apache-2.0
"""Start, health-check, parse and stop one `vllm serve` process (plan M4, RUK-2709).

`build_command` turns one variant and one server entry of `variants.json` into
the exact `vllm serve` command and environment; `start_server` launches it with
its log in a file in its own process group; `wait_healthy` polls `/health` and
`/v1/models`; `parse_startup` reads the startup lines into the `startup.json`
record of `schema.STARTUP_KEYS`; `stop_server` ends the whole process group.

Runs on Python 3.10 with the standard library only (the node's container).
CLI: `python -m tools.flash_next.rental.serve --parse LOG [--server N]`
prints the parsed record as JSON (re-parse a log on the node).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tools.flash_next.rental import schema

_HEALTH_POLL_S = 5.0
_HTTP_TIMEOUT_S = 5.0
_LIVENESS_STEP_S = 0.1

_KV_CACHE_RE = re.compile(
    r"(?:GPU|HPU) KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_NUM_BLOCKS_RE = re.compile(r"Usable num_blocks: ([\d,]+),")
_WEIGHTS_RE = re.compile(r"Loading model weights took ([\d.]+) GB")
_WARMUP_RE = re.compile(r"Warmup finished in ([\d,]+) secs")
_GDN_RE = re.compile(r"GDN layers detected (.*)$", re.MULTILINE)


def _to_int(text: str) -> int:
    return int(text.replace(",", ""))


def build_command(variant: dict, server: dict, model: str, served_name: str) -> tuple[list[str], dict[str, str]]:
    """The exact `vllm serve` command for one server of one variant, and its environment.

    The environment is `os.environ` with the rental defaults set, `VLLM_COMPACT_GDN`
    removed and `variant["env"]` applied on top (it may override any default).
    """
    cmd = [
        "vllm",
        "serve",
        model,
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(server["port"]),
        "--tensor-parallel-size",
        str(server["tp"]),
    ]
    if server["ep"]:
        cmd.append("--enable-expert-parallel")
    cmd += [
        "--max-model-len",
        str(schema.MAX_MODEL_LEN),
        "--max-num-seqs",
        str(server["agents"]),
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
    cmd += list(variant.get("serve_args") or [])

    env = dict(os.environ)
    env["HABANA_VISIBLE_MODULES"] = str(server["cards"])
    env["VLLM_HPU_FLASH_NEXT_QSA_MODE"] = "1"
    env["VLLM_HPU_RECURRENT_STATE_POOL"] = "1"
    env["PT_HPU_LAZY_MODE"] = "1"
    env.pop("VLLM_COMPACT_GDN", None)
    for key, value in (variant.get("env") or {}).items():
        env[key] = value
    return cmd, env


def start_server(cmd: list[str], env: dict[str, str], log_path: Path, cwd: Path) -> subprocess.Popen:
    """Start the server in its own process group, stdout+stderr written to `log_path`.

    `cwd` must be the variant's results folder, never the repository root: running
    from the root would import the source tree's `vllm_gaudi` instead of the
    installed one with the native libraries.
    """
    log_file = open(log_path, "wb", buffering=0)  # noqa: SIM115  # the fd goes to the server child
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_file = log_file  # keep the handle alive for as long as the caller holds proc
    return proc


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _sleep_polling_exit(proc: subprocess.Popen, seconds: float) -> bool:
    """Sleep up to `seconds` in small steps; False if the process exited during it."""
    end = time.monotonic() + seconds
    while True:
        if proc.poll() is not None:
            return False
        remaining = end - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(_LIVENESS_STEP_S, remaining))


def wait_healthy(port: int, timeout_s: float, proc: subprocess.Popen) -> bool:
    """True once GET /health and then GET /v1/models answer 200 on this port.

    Polls every 5 s; returns False when `timeout_s` passes or the process exits first.
    """
    deadline = time.monotonic() + timeout_s
    health_url = f"http://127.0.0.1:{port}/health"
    models_url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if _http_ok(health_url) and _http_ok(models_url):
            return True
        if not _sleep_polling_exit(proc, min(_HEALTH_POLL_S, deadline - time.monotonic())):
            return False
    return False


def parse_startup(log_text: str, server: int, startup_seconds: float) -> dict:
    """The `startup.json` record for one server, parsed from its whole log text.

    Each startup line is searched once (the first match wins, as printed by the
    TP worker that logs it); a line that never appears leaves its key null.
    """
    record: dict = {
        "server": server,
        "usable_num_blocks": None,
        "kv_cache_tokens": None,
        "max_concurrency": None,
        "max_concurrency_len": None,
        "weights_gib": None,
        "warmup_seconds": None,
        "gdn_layers_detected": None,
        "startup_seconds": startup_seconds,
    }
    match = _KV_CACHE_RE.search(log_text)
    if match is not None:
        record["kv_cache_tokens"] = _to_int(match.group(1))
        record["max_concurrency_len"] = _to_int(match.group(2))
        record["max_concurrency"] = float(match.group(3))
    match = _NUM_BLOCKS_RE.search(log_text)
    if match is not None:
        record["usable_num_blocks"] = _to_int(match.group(1))
    match = _WEIGHTS_RE.search(log_text)
    if match is not None:
        record["weights_gib"] = float(match.group(1))
    match = _WARMUP_RE.search(log_text)
    if match is not None:
        record["warmup_seconds"] = float(match.group(1))
    match = _GDN_RE.search(log_text)
    if match is not None:
        record["gdn_layers_detected"] = match.group(1)
    return record


def _killpg_safe(pgid: int, sig: int) -> None:
    with contextlib.suppress(OSError):  # the group is already gone
        os.killpg(pgid, sig)


def stop_server(proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    """SIGINT the server's process group, SIGKILL it after `timeout_s`; never raises.

    The group covers the server and all of its TP workers (one session, see
    `start_server`). An already-exited process is left as it is.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    if pgid is not None:
        _killpg_safe(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            _killpg_safe(pgid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.serve",
                                     description="Rental harness: server runner for one variant's server.")
    parser.add_argument("--parse",
                        metavar="LOG",
                        required=True,
                        help="print the startup record parsed from LOG as JSON")
    parser.add_argument("--server", type=int, default=0, help="server index for the parsed record (default 0)")
    args = parser.parse_args(argv)
    log_text = Path(args.parse).read_text(encoding="utf-8", errors="replace")
    print(json.dumps(parse_startup(log_text, args.server, 0.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
