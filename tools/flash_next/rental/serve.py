# SPDX-License-Identifier: Apache-2.0
"""Start, health-check, parse and stop one `vllm serve` process (plan M4, RUK-2709).

`build_command` turns one variant and one server entry of `variants.json` into
the exact `vllm serve` command and environment; `start_server` launches it with
its log in a file in its own process group; `wait_healthy` polls `/health` and
`/v1/models`; `unhealthy_reason` says whether a server that did not become
healthy exited or ran out of time, and `error_lines` gives the last error texts
of its log; `wait_port_file` reads the port a server wrote to a file once
listening; `port_in_use` reports whether something already accepts TCP
connections on a port; `parse_startup` reads the startup lines into the `startup.json`
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
import socket
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

RENTAL_ENV_DEFAULTS = {
    "VLLM_HPU_FLASH_NEXT_QSA_MODE": "1",
    "VLLM_HPU_RECURRENT_STATE_POOL": "1",
    "PT_HPU_LAZY_MODE": "1",
    "PYTHONFAULTHANDLER": "1",  # a worker that dies on a signal leaves its Python stack in the server log
}

MAX_ERROR_LINES = 5
# "TypeError: ...", "torch._dynamo.exc.TorchRuntimeError: ..." up to the end of the line
_ERROR_RE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception): .*")

RECIPE_CACHE_ENV = "PT_HPU_RECIPE_CACHE_CONFIG"
COMBINED_KERNEL_DB = "libflashinfer_gaudi_combined_gaudi2.so"
COMBINED_KERNEL_DB_ENV = "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"
RECIPE_CACHE_NONE = "none"
RECIPE_CACHE_OFF_REASON = ("off: this server registers a fused kernel's database (the fused GDN kernel's, or the "
                           "library of the fused hyper-connection and MoE glue kernels) in GC_KERNEL_PATH, and the "
                           "Habana bridge accepts only one kernel library next to a recipe cache")


def recipe_cache_env(cache_dir: str | None, size_mb: int, server_env: dict,
                     combined_db: bool = False) -> tuple[dict, str]:
    """The recipe-cache variable for one server and a note for the manifest.

    Returns ({RECIPE_CACHE_ENV: f"{cache_dir},false,{size_mb}"}, cache_dir) when `cache_dir` is set and
    either `combined_db` is True or the server registers no kernel database of its own: `server_env` has
    VLLM_HPU_GDN_CORE_DECODE and VLLM_HPU_GDN_CORE_VERIFY both equal to "0" (a missing key counts as "1",
    the plugin default) and neither VLLM_HPU_HC_FUSED nor VLLM_HPU_MOE_GLUE is "1" (a missing key counts
    as "0", the plugin default; either flag loads the one library that holds the fused hyper-connection
    and the MoE glue kernels). Returns ({}, RECIPE_CACHE_NONE) when `cache_dir` is None or empty, and
    ({}, RECIPE_CACHE_OFF_REASON) otherwise.
    """
    if not cache_dir:
        return {}, RECIPE_CACHE_NONE
    if not combined_db:
        if server_env.get("VLLM_HPU_GDN_CORE_DECODE", "1") != "0" or \
                server_env.get("VLLM_HPU_GDN_CORE_VERIFY", "1") != "0" or \
                server_env.get("VLLM_HPU_HC_FUSED", "0") == "1" or \
                server_env.get("VLLM_HPU_MOE_GLUE", "0") == "1":
            return {}, RECIPE_CACHE_OFF_REASON
    return {RECIPE_CACHE_ENV: "%s,false,%s" % (cache_dir, size_mb)}, cache_dir


def server_environment(overrides: dict, cards: str = None) -> dict[str, str]:
    """os.environ with the rental defaults, VLLM_COMPACT_GDN removed, then `overrides` on top.

    `cards` sets HABANA_VISIBLE_MODULES when given (a variant's server); the tiny gate uses every card
    and passes None. Every `vllm serve` the rental starts gets its environment from here.
    """
    env = dict(os.environ)
    if cards is not None:
        env["HABANA_VISIBLE_MODULES"] = str(cards)
    env.update(RENTAL_ENV_DEFAULTS)
    env.pop("VLLM_COMPACT_GDN", None)
    env.update(overrides or {})
    return env


def _to_int(text: str) -> int:
    return int(text.replace(",", ""))


def build_command(variant: dict, server: dict, model: str, served_name: str) -> tuple[list[str], dict[str, str]]:
    """The exact `vllm serve` command for one server of one variant, and its environment.

    The environment is `server_environment` with `variant["env"]` as the overrides
    (they may override any default) and the server's `cards` in HABANA_VISIBLE_MODULES.
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
    if os.environ.get("FARM_SERVE_NO_PREFIX_CACHING") == "1":  # async-sched job branch: the tiny gate of mode 3
        cmd.append("--no-enable-prefix-caching")

    env = server_environment(variant.get("env") or {}, server["cards"])
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


def unhealthy_reason(proc: subprocess.Popen, seconds: float, timeout_s: float,
                     name: str = "server") -> tuple[schema.GateStatus, str]:
    """Why `wait_healthy` returned False after `seconds`: the process is gone (FAILED, "<name> exited with
    code X after N s") or it is still there and the time ran out (TIMED_OUT, "<name> not healthy within
    N s"). `wait_healthy` returns as soon as the process exits, so the two must not share one text."""
    code = proc.poll()
    if code is None:
        return schema.GateStatus.TIMED_OUT, "%s not healthy within %d s" % (name, int(timeout_s))
    return schema.GateStatus.FAILED, "%s exited with code %d after %d s" % (name, code, int(seconds))


def error_lines(log_text: str | None, limit: int = MAX_ERROR_LINES) -> list[str]:
    """The last `limit` distinct `SomeError: message` texts of a server log, in order of first appearance."""
    seen: list[str] = []
    for line in (log_text or "").splitlines():
        match = _ERROR_RE.search(line)
        if match is not None:
            text = match.group(0).strip()
            if text not in seen:
                seen.append(text)
    return seen[-limit:]


def wait_port_file(path: Path, timeout_s: float, proc: subprocess.Popen) -> int | None:
    """The port a server wrote to `path` (one decimal line).

    None when the process exits or `timeout_s` passes before the file holds a number.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text.isdigit():
            return int(text)
        if proc.poll() is not None or time.monotonic() >= deadline:
            return None
        time.sleep(_LIVENESS_STEP_S)


def port_in_use(port: int) -> bool:
    """True when something accepts TCP connections on 127.0.0.1:`port`."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
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


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # EPERM: macOS answers it for a member that is exiting or a zombie
        return True
    return True


def stop_server(proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    """SIGINT the server's process group, SIGKILL it after `timeout_s`; never raises.

    The group covers the server and all of its TP workers (one session, see
    `start_server`, so the group id is the server's pid). It waits for the whole
    group, not just the server: a server that exited (or crashed) can leave
    workers behind that still hold the cards the next variant needs. When the
    server itself is already gone there is nobody to run a clean shutdown, so
    what is left of the group gets SIGKILL at once, without the SIGINT wait.
    """
    pgid = proc.pid
    lead_exited = proc.poll() is not None  # also reaps an exited server: its zombie must not count as alive
    if not _group_alive(pgid):
        return
    if not lead_exited:
        _killpg_safe(pgid, signal.SIGINT)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            proc.poll()
            if not _group_alive(pgid):
                return
            time.sleep(0.2)
    _killpg_safe(pgid, signal.SIGKILL)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and (proc.poll() is None or _group_alive(pgid)):
        time.sleep(0.1)


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
