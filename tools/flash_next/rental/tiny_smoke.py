# SPDX-License-Identifier: Apache-2.0
"""The tiny smoke test: can a Flash-Next server start and decode on this node at all (RUK-2709).

The first thing to run on a new rental node, before the checkpoint download and the gates. It builds
the tiny checkpoint and its CPU reference (`make_tiny_checkpoint.py`), then, once per GDN mode, starts
the tiny gate's `vllm serve` command with the environment every rental server gets
(`serve.server_environment`), waits for `/health`, compares the served tokens with the reference
(`gates.tiny_compare`) and stops the server, so the cards are free again.

Modes: `pytorch` turns the fused GDN kernel off (the flags gate G1b sets when it fails); `fused` turns
the decode kernel on and keeps the verify op off, and fails when the server log says a layer fell back
to the torch decode path. Each mode writes `<out>/<mode>/server0.log` and `details.json`;
`<out>/summary.json` holds one entry per mode. The exit code is 0 only when every requested mode passed.

Runs on Python 3.10 with the standard library only, from the plugin root of the bundle:
`python -m tools.flash_next.rental.tiny_smoke --out DIR [--ckpt DIR] [--gdn pytorch,fused] [--port N]
[--health-timeout S]`. `--ckpt` names a folder the checkpoint is built in once and reused from then on
(default: `<out>/ckpt`, built every time). tiny_smoke.sh starts this inside the container.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from tools.flash_next.rental import gates, serve
from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental.schema import GateStatus

GDN_MODES = {
    "pytorch": dict(gates.GDN_FALLBACK_ENV),
    "fused": {gates.GDN_MODE_ENV: str(gates.GDN_MODE_FUSED), gates.GDN_VERIFY_ENV: str(gates.GDN_VERIFY_OFF)},
}
DEFAULT_MODES = "pytorch,fused"
HEALTH_TIMEOUT_S = 600.0
MAX_LINES = 5
DECODE_FALLBACK = "GDN core decode"

# "TypeError: ...", "torch._dynamo.exc.TorchRuntimeError: ..." up to the end of the line
_ERROR_RE = re.compile(r"[A-Za-z_][\w.]*(?:Error|Exception): .*")

# The two commands are the tiny gate's own, so the smoke test starts what the run will start.
checkpoint_command = orchestrator.tiny_checkpoint_command
serve_command = orchestrator.tiny_serve_command


def error_lines(log_text: str, limit: int = MAX_LINES) -> list[str]:
    """The last `limit` distinct `SomeError: message` texts of a server log, in order of first appearance."""
    seen: list[str] = []
    for line in (log_text or "").splitlines():
        match = _ERROR_RE.search(line)
        if match is not None:
            text = match.group(0).strip()
            if text not in seen:
                seen.append(text)
    return seen[-limit:]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_checkpoint(ckpt: Path) -> str | None:
    """Build the tiny checkpoint in `ckpt` unless its reference is already there; the failure reason or None."""
    if (ckpt / "reference.json").is_file():
        print("tiny smoke: reusing the tiny checkpoint in %s" % ckpt, flush=True)
        return None
    print("tiny smoke: building the tiny checkpoint in %s (a few minutes)" % ckpt, flush=True)
    timeout_s = orchestrator.TINY_CHECKPOINT_TIMEOUT_S
    try:
        proc = gates.run_step(checkpoint_command(ckpt), orchestrator.repo_root(), timeout_s)
    except subprocess.TimeoutExpired:
        return "tiny checkpoint did not finish in %d s" % int(timeout_s)
    except OSError as e:
        return "tiny checkpoint could not start: %s" % e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return "tiny checkpoint failed (exit %d): %s" % (proc.returncode, tail.strip())
    return None


def run_mode(mode: str, ckpt: Path, out_dir: Path, port: int, health_timeout_s: float) -> dict:
    """Serve the tiny checkpoint in one GDN mode, compare it with the reference, stop the server.

    Returns the mode's summary entry; the comparison's own record goes to `<out_dir>/details.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server0.log"
    entry: dict = {"mode": mode, "status": None, "reason": None, "startup_seconds": None, "errors": [],
                   "fallback_lines": [], "log": str(log_path)}
    details: dict = {}
    status = GateStatus.FAILED
    if serve.port_in_use(port):
        entry["reason"] = "port %d already in use" % port
    else:
        env = serve.server_environment(orchestrator.server_env({}, GDN_MODES[mode]))
        print("tiny smoke [%s]: starting the server (log: %s)" % (mode, log_path), flush=True)
        popen = serve.start_server(serve_command(ckpt, port), env, log_path, out_dir)
        try:
            started = time.monotonic()
            healthy = serve.wait_healthy(port, health_timeout_s, popen)
            entry["startup_seconds"] = round(time.monotonic() - started, 1)
            if healthy:
                status, details = gates.tiny_compare("http://127.0.0.1:%d" % port, ckpt / "reference.json")
                entry["reason"] = details.get("reason") or details.get("error")
            elif popen.poll() is None:
                status = GateStatus.TIMED_OUT
                entry["reason"] = "server not healthy within %d s" % int(health_timeout_s)
            else:
                entry["reason"] = "server exited before it was healthy (exit %d)" % popen.returncode
        finally:
            serve.stop_server(popen)  # also on Ctrl-C: the workers must not keep the cards
        log_text = _read(log_path)
        if status is GateStatus.PASSED and mode == "fused":
            fallback = [ln for ln in gates.gdn_fallback_lines(log_text) if DECODE_FALLBACK in ln]
            if fallback:
                status = GateStatus.FAILED
                entry["reason"] = ("the fused kernel did not run: %d log line(s) say a layer used the torch "
                                   "decode path" % len(fallback))
                entry["fallback_lines"] = fallback[:MAX_LINES]
        if status is not GateStatus.PASSED:
            entry["errors"] = error_lines(log_text)
    entry["status"] = status.name.lower()
    _write_json(out_dir / "details.json", dict(details, mode=mode, status=int(status), reason=entry["reason"]))
    return entry


def report_mode(entry: dict) -> None:
    """One result line for the console, and on a failure the error lines and the log to read."""
    mode = entry["mode"]
    if entry["status"] == "passed":
        print("tiny smoke [%s]: PASSED (server up in %s s, tokens match the reference)" %
              (mode, entry["startup_seconds"]), flush=True)
        return
    word = "TIMED OUT" if entry["status"] == "timed_out" else "FAILED"
    print("tiny smoke [%s]: %s (%s)" % (mode, word, entry["reason"]), flush=True)
    for line in entry["errors"] + entry["fallback_lines"]:
        print("    %s" % line, flush=True)
    print("    log: %s" % entry["log"], flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.tiny_smoke",
                                     description="Serve the tiny Flash-Next checkpoint and compare it with its "
                                     "CPU reference, once per GDN mode.")
    parser.add_argument("--out", required=True, help="results folder of this call")
    parser.add_argument("--ckpt", default=None, help="checkpoint folder, built once and reused (default: OUT/ckpt)")
    parser.add_argument("--gdn", default=DEFAULT_MODES, help="comma-separated modes: pytorch, fused")
    parser.add_argument("--port", type=int, default=orchestrator.TINY_PORT)
    parser.add_argument("--health-timeout", type=float, default=HEALTH_TIMEOUT_S, metavar="S")
    args = parser.parse_args(argv)
    modes = [m for m in args.gdn.split(",") if m]
    if not modes or any(m not in GDN_MODES for m in modes):
        parser.error("--gdn takes a comma-separated list of: %s" % ", ".join(GDN_MODES))

    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt).absolute() if args.ckpt else out / "ckpt"
    summary: dict = {"passed": False, "reason": None, "checkpoint": str(ckpt), "modes": []}
    summary["reason"] = build_checkpoint(ckpt)
    if summary["reason"] is None:
        for mode in modes:
            entry = run_mode(mode, ckpt, out / mode, args.port, args.health_timeout)
            summary["modes"].append(entry)
            report_mode(entry)
        summary["passed"] = all(entry["status"] == "passed" for entry in summary["modes"])
    _write_json(out / "summary.json", summary)
    if summary["passed"]:
        print("tiny smoke: PASSED (%s)" % ", ".join(modes), flush=True)
        return 0
    failed = [entry["mode"] for entry in summary["modes"] if entry["status"] != "passed"]
    print("tiny smoke: FAILED (%s)" % (summary["reason"] or ", ".join(failed)), flush=True)
    print("summary: %s" % (out / "summary.json"), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
