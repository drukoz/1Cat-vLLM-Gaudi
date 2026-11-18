# SPDX-License-Identifier: Apache-2.0
"""Rental harness monitor: Prometheus ``/metrics`` and ``hl-smi`` sampler (plan M4, RUK-2709).

``python -m tools.flash_next.rental.monitor --servers URL[,URL] --out-dir DIR [--interval 5]
[--duration SECONDS] [--hlsmi-cmd CMD | --no-hlsmi]``

Every ``--interval`` seconds it fetches each server's ``/metrics`` with ``urllib.request``
(5 s timeout; a failed fetch writes a row of ``null`` values and continues), writes one
``METRICS_FILE`` row per server (keys exactly ``schema.METRICS_KEYS``, counters as raw
cumulative sums over all label sets) and runs the ``hl-smi`` query, appending every output
line prefixed with the Unix time to ``HLSMI_FILE``. It runs until ``--duration`` passes or it
gets SIGTERM/SIGINT (the current write finishes, exit 0). The orchestrator starts this as a
subprocess and sends SIGTERM when the load finishes.

Prometheus names, checked at the pinned vLLM commit (``8f816a3f``):
``vllm/v1/metrics/loggers.py`` defines ``vllm:num_requests_running`` (line 494),
``vllm:num_requests_waiting`` (504), ``vllm:kv_cache_usage_perc`` (562),
``vllm:prefix_cache_queries`` (585), ``vllm:prefix_cache_hits`` (596), ``vllm:num_preemptions``
(662), ``vllm:prompt_tokens`` (671) and ``vllm:generation_tokens`` (705);
``vllm/v1/spec_decode/metrics.py`` defines ``vllm:spec_decode_num_drafts`` (229),
``vllm:spec_decode_num_draft_tokens`` (230), ``vllm:spec_decode_num_accepted_tokens`` (231)
and ``vllm:spec_decode_num_accepted_tokens_per_pos`` (255). The table in the plan is correct.
``prometheus_client`` appends ``_total`` to counter names in the exposition text, so a trailing
``_total`` is stripped before matching; matching is by exact name, so ``..._per_pos`` is a
different name and never counts into ``spec_accepted``. ``_created`` samples are ignored, and
non-finite values (NaN, Inf) are skipped because JSON cannot carry them.

Standard library only, Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from tools.flash_next.rental.schema import HLSMI_FILE, METRICS_FILE, METRICS_KEYS

#: ``metrics.jsonl`` key -> exact Prometheus metric name (the pin, see the module docstring).
PROMETHEUS_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "prefix_hits": "vllm:prefix_cache_hits",
    "prefix_queries": "vllm:prefix_cache_queries",
    "preemptions": "vllm:num_preemptions",
    "spec_accepted": "vllm:spec_decode_num_accepted_tokens",
    "spec_drafts": "vllm:spec_decode_num_drafts",
    "spec_draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "generation_tokens": "vllm:generation_tokens",
    "prompt_tokens": "vllm:prompt_tokens",
}

HLSMI_FIELDS = ("timestamp", "index", "utilization.aip", "memory.used", "memory.total", "power.draw", "temperature.aip")
DEFAULT_HLSMI_CMD = ["hl-smi", "-Q", ",".join(HLSMI_FIELDS), "-f", "csv,noheader"]

FETCH_TIMEOUT_S = 5.0
HLSMI_TIMEOUT_S = 10.0
HLSMI_MAX_FAILURES = 3
SLEEP_TICK_S = 0.05


def parse_prometheus(text: str) -> dict[str, float]:
    """Sum every exposition-text sample per metric name: ``{name_without_labels: sum}``.

    ``#`` lines are comments; a sample line is ``name{labels} value [timestamp]`` or
    ``name value`` (label values may contain spaces and braces inside quotes). A trailing
    ``_total`` is stripped from names, ``_created`` samples are ignored, and non-finite
    values (``NaN``, ``+Inf``) are skipped; scientific notation is accepted.
    """
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        split = _split_key_and_rest(line)
        if split is None:
            continue
        key, rest = split
        fields = rest.split()
        if not fields:
            continue
        name = key.split("{", 1)[0]
        if name.endswith("_created"):
            continue
        if name.endswith("_total") and len(name) > len("_total"):
            name = name[:-len("_total")]
        try:
            value = float(fields[0])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def _split_key_and_rest(line: str) -> Optional[tuple[str, str]]:
    """Split a sample line at its first whitespace outside the quoted label values."""
    in_quotes = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
        elif in_quotes and char == "\\":
            escaped = True
        elif char == '"':
            in_quotes = not in_quotes
        elif not in_quotes and char in " \t":
            return line[:index], line[index:].strip()
    return None


def to_sample(values: dict[str, float], server: int, t: float) -> dict:
    """One ``metrics.jsonl`` row: keys exactly ``METRICS_KEYS``, ``null`` for absent metrics."""
    sample: dict = {}
    for key in METRICS_KEYS:
        if key == "t":
            sample[key] = float(t)
        elif key == "server":
            sample[key] = int(server)
        else:
            value = values.get(PROMETHEUS_METRICS[key])
            sample[key] = None if value is None else float(value)
    return sample


def _fetch_metrics(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response:
            return response.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, ValueError):
        # A server dying mid-response raises HTTPException (IncompleteRead, BadStatusLine), not OSError.
        return None


def _write_metric_rows(metrics_file, metric_urls: list[str], t: float) -> None:
    for server, url in enumerate(metric_urls):
        text = _fetch_metrics(url)
        values = parse_prometheus(text) if text is not None else {}
        metrics_file.write(json.dumps(to_sample(values, server, t)) + "\n")
    metrics_file.flush()


def _sample_hlsmi(cmd: list[str], hlsmi_path: Path) -> bool:
    """Run the hl-smi command once; append ``<unix time>,<line>`` per output line.

    Returns True on success, False on any failure (missing binary, timeout, non-zero exit).
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=HLSMI_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    t = time.time()
    with open(hlsmi_path, "a", encoding="utf-8") as hlsmi_file:
        for line in lines:
            hlsmi_file.write(f"{t:.3f},{line}\n")
    return True


def _sleep(seconds: float, stop: threading.Event) -> None:
    """Sleep in short ticks so a SIGTERM/SIGINT is seen quickly."""
    end = time.monotonic() + seconds
    while not stop.is_set():
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(SLEEP_TICK_S, remaining))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.monitor",
                                     description="Sample vLLM /metrics and hl-smi into metrics.jsonl and hlsmi.csv.")
    parser.add_argument("--servers", required=True, help="comma-separated server base URLs")
    parser.add_argument("--out-dir", required=True, help="directory for metrics.jsonl and hlsmi.csv")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between samples (default 5)")
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument("--hlsmi-cmd", default=None, help="replace the hl-smi command")
    parser.add_argument("--no-hlsmi", action="store_true", help="do not sample hl-smi")
    args = parser.parse_args(argv)

    servers = [entry.strip() for entry in args.servers.split(",") if entry.strip()]
    if not servers:
        print("monitor: no servers given", file=sys.stderr)
        return 2
    metric_urls = [server.rstrip("/") + "/metrics" for server in servers]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / METRICS_FILE
    hlsmi_path = out_dir / HLSMI_FILE

    if args.hlsmi_cmd is not None:
        hlsmi_cmd = shlex.split(args.hlsmi_cmd)
        hlsmi_header = "t"
    else:
        hlsmi_cmd = list(DEFAULT_HLSMI_CMD)
        hlsmi_header = "t," + ",".join(HLSMI_FIELDS)
    hlsmi_enabled = not args.no_hlsmi
    if hlsmi_enabled and (not hlsmi_path.exists() or hlsmi_path.stat().st_size == 0):
        with open(hlsmi_path, "a", encoding="utf-8") as hlsmi_file:
            hlsmi_file.write(hlsmi_header + "\n")
    hlsmi_failures = 0

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    start = time.monotonic()
    with open(metrics_path, "a", encoding="utf-8") as metrics_file:
        while True:
            _write_metric_rows(metrics_file, metric_urls, time.time())
            if hlsmi_enabled:
                if _sample_hlsmi(hlsmi_cmd, hlsmi_path):
                    hlsmi_failures = 0
                else:
                    hlsmi_failures += 1
                    if hlsmi_failures >= HLSMI_MAX_FAILURES:
                        hlsmi_enabled = False
                        print("monitor: hl-smi failed 3 times in a row; stopping card sampling, metrics continue",
                              file=sys.stderr)
            if stop.is_set():
                break
            if args.duration is not None and time.monotonic() - start >= args.duration:
                break
            _sleep(args.interval, stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
