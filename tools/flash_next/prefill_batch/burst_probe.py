# SPDX-License-Identifier: Apache-2.0
"""What a merged prompt forward costs against the one-request forwards it replaces (``VLLM_HPU_PREFILL_BATCH``).

    python3 -m tools.flash_next.prefill_batch.burst_probe --url http://127.0.0.1:PORT --model fn --out DIR \\
        --switch-file PATH [--rows 8] [--levels 1,2,4,8] [--context 2048] [--new 512] [--rounds 3]

For each level K: K conversations per arm, each a fresh prompt of ``--context`` tokens (sent once, untimed: the
prefix the turns hit), then ``--rounds`` bursts per arm in which every conversation sends one turn at once, its
context plus ``--new`` fresh tokens, for one output token. Both arms send turns of the same shape: a prefix hit
of the context and the same uncached tail. The arm ``off`` writes ``VLLM_HPU_PREFILL_BATCH=0`` to the server's runtime
switch file (every prompt chunk its own forward), the arm ``on`` writes ``--rows``; the arms alternate within each
round. A burst's span is the time from its first send to its last first token. With K rows merged the span is
K one-row forwards minus what merging saves: ``(off - on) / (K - 1)`` per removed forward is the per-forward cost
the merged forward no longer pays (the question of the design note: the about 80 ms intercept of a prompt forward
at 10K context, per forward or per row). The server's rank-0 events (``VLLM_PROFILER_ENABLED=true``) carry each
forward's bucket and time.

Writes ``DIR/bursts.jsonl`` (one row per burst) and ``DIR/burst_probe.json`` (medians per level and arm).
Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SYNTHETIC_ID_RANGE = (1000, 60000)  # ordinary vocabulary ids (prefill_sweep's synthetic range)
SETTLE_S = 1.0  # the workers poll the switch file once per step


def post(url: str, payload: dict, timeout_s: float = 900.0) -> tuple[int, dict | str, float]:
    """``(status, answer, seconds)`` of one completion request."""
    request = urllib.request.Request(url + "/v1/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - the job's own server
            return response.status, json.loads(response.read().decode("utf-8")), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:400], time.monotonic() - started
    except (OSError, ValueError) as error:
        return 0, repr(error)[:400], time.monotonic() - started


def completion(model: str, ids: list[int], max_tokens: int) -> dict:
    return {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}


def write_switch(path: Path, rows: int) -> None:
    staging = Path(str(path) + ".tmp")
    staging.write_text("VLLM_HPU_PREFILL_BATCH=%d\n" % rows, encoding="utf-8")
    os.replace(staging, path)  # a rename: every worker takes it at the same step
    time.sleep(SETTLE_S)


def cached_tokens(answer) -> int | None:
    if not isinstance(answer, dict):
        return None
    details = (answer.get("usage") or {}).get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def burst(url: str, model: str, prompts: list[list[int]], max_tokens: int) -> dict:
    """Send every prompt at once; the span from the first send to the last answer and each request's time."""
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        futures = [pool.submit(post, url, completion(model, p, max_tokens)) for p in prompts]
        results = [f.result() for f in futures]
    span = time.monotonic() - started
    return {"span_s": round(span, 4), "seconds": [round(r[2], 4) for r in results],
            "status": [r[0] for r in results], "cached": [cached_tokens(r[1]) for r in results],
            "errors": [str(r[1])[:200] for r in results if r[0] != 200]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="fn")
    parser.add_argument("--out", required=True)
    parser.add_argument("--switch-file", required=True, help="the server's VLLM_HPU_RUNTIME_SWITCH_FILE")
    parser.add_argument("--rows", type=int, default=8, help="the arm on's VLLM_HPU_PREFILL_BATCH")
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--new", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--nonce", default="burst")
    parser.add_argument("--label", default="")
    parser.add_argument("--knobs-file", default=None, help="the server's VLLM_HPU_SCHED_KNOBS_FILE")
    parser.add_argument("--knobs", default=None, help='JSON written to --knobs-file first, e.g. '
                        '\'{"prompt_graph_max_tokens": 70000}\' (the workers read it at every step)')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.knobs_file and args.knobs:
        knobs = json.loads(args.knobs)
        staging = Path(args.knobs_file + ".tmp")
        staging.write_text(json.dumps(knobs), encoding="utf-8")
        os.replace(staging, args.knobs_file)
        time.sleep(SETTLE_S)
    switch = Path(args.switch_file)
    rng = random.Random("%s-%d-%d" % (args.nonce, args.context, args.new))

    def ids(n: int) -> list[int]:
        return [rng.randrange(*SYNTHETIC_ID_RANGE) for _ in range(n)]

    levels = [int(v) for v in args.levels.split(",") if v]
    arms = (("off", 0), ("on", args.rows))
    rows_file = out / "bursts.jsonl"
    summary: dict = {"context": args.context, "new": args.new, "rows": args.rows, "label": args.label, "levels": {}}
    for k in levels:
        contexts = {arm: [ids(args.context) for _ in range(k)] for arm, _ in arms}
        write_switch(switch, 0)
        setup = {arm: burst(args.url, args.model, contexts[arm], 1) for arm, _ in arms}  # the prefix, untimed
        spans: dict = {arm: [] for arm, _ in arms}
        for rnd in range(args.rounds):
            order = arms if rnd % 2 == 0 else tuple(reversed(arms))
            for arm, value in order:
                write_switch(switch, value)
                prompts = [c + ids(args.new) for c in contexts[arm]]
                row = burst(args.url, args.model, prompts, 1)
                row.update(level=k, arm=arm, round=rnd, context=args.context, new=args.new, label=args.label,
                           setup_span_s=setup[arm]["span_s"] if rnd == 0 else None)
                spans[arm].append(row["span_s"])
                with open(rows_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print("level %d round %d arm %-3s span %.3f s cached %s errors %d" %
                      (k, rnd, arm, row["span_s"], row["cached"][:4], len(row["errors"])), flush=True)
        med = {arm: statistics.median(v) for arm, v in spans.items() if v}
        entry = {"span_s": med, "saved_per_removed_forward_s": None}
        if k > 1 and "off" in med and "on" in med:
            entry["saved_per_removed_forward_s"] = round((med["off"] - med["on"]) / (k - 1), 4)
        summary["levels"][str(k)] = entry
        print("LEVEL %d: span off %.3f s, on %.3f s; saved per removed forward %s s" %
              (k, med.get("off", float("nan")), med.get("on", float("nan")), entry["saved_per_removed_forward_s"]),
              flush=True)
        (out / "burst_probe.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    write_switch(switch, args.rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
