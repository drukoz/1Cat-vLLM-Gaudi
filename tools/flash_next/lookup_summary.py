#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One table from the card jobs of the spec-lookup track (``tools/flash_next/rental/lookup_serve.sh``).

    lookup_summary.py RUN_DIR [RUN_DIR ...] [--out SUMMARY.json]

Each RUN_DIR is a job's results folder: ``probe.json`` (``lookup_probe.py``), ``summary.json`` (the server's
speculative config and environment) and ``server0.log``. Every measured round becomes one row, named by its arm:

  no speculation            a server without ``--speculative-config``
  no speculation (n-gram server, lookup off)
                            the ``ngram`` method, the request's lookup mode 0 (``vllm_xargs {"spec_lookup": 0}``): it
                            drafts nothing, and with ``VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS=1`` its steps are plain
                            decode steps (the same-server reference; a real no-speculation server is preferred)
  n-gram k=K                the ``ngram`` method (ragged rows: a step is as wide as its longest draft)
  MTP k=K                   the ``mtp`` method, the request's lookup mode 0 (or a server without the lookup hook)
  hybrid k=K                mode 1: the lookup continuation where the last tokens match, else the MTP drafts
  lookup only k=K           mode 2: the lookup continuation or k empty drafts, in the MTP shapes

Per row: tokens per verification step per request and the rate of each draft position (the server's
counters), ms per engine step (the median gap between a stream's chunks), ms per token steady (``ms per step /
tokens per step``) and over the decode span, the steady figure against the no-speculation row of the same
prompts and stream count, the rows of one verification step in the MTP shapes, and how many streams equal the
no-speculation streams token for token (greedy speculation is lossless; the first difference otherwise).
For a server with the lookup hook the rows and accepted drafts of each source (lookup, mtp, none) per round
come from the "Flash-Next lookup hybrid" lines of ``server0.log``. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

MODES = {0: "MTP", 1: "hybrid", 2: "lookup only"}
NO_SPEC = "no speculation"
NO_SPEC_NGRAM = "no speculation (n-gram server, lookup off)"
_SOURCE = re.compile(r"(lookup|mtp|none) rows (\d+), accepted (\d+) over (\d+) verified rows")
_BOUNDARY = re.compile(r"Flash-Next lookup hybrid: (.*) \(after (\d+) steps, a new set of requests\)")


def arm_name(spec: dict | None, lookup_env: str | None, mode: int | None) -> str:
    """The arm of a round from the server's speculative config, its lookup flag and the request's mode."""
    if not spec:
        return NO_SPEC
    k = spec.get("num_speculative_tokens")
    if spec.get("method") == "ngram":
        return NO_SPEC_NGRAM if mode == 0 else "n-gram k=%s" % k
    if mode is None:
        try:
            mode = int(lookup_env or 0)
        except ValueError:
            mode = 0
    if lookup_env in (None, "", "0"):
        mode = 0  # no hook installed: the request's mode is ignored
    return "%s k=%s" % (MODES.get(mode, "MTP"), k)


def is_reference(arm: str) -> bool:
    """A round without speculation: a server without it, or an n-gram server's request with its lookup off."""
    return arm in (NO_SPEC, NO_SPEC_NGRAM)


def boundary_counters(log_text: str) -> list[dict]:
    """The cumulative per-source counters at every "new set of requests" line of a server log, in order, once per
    step count (every worker of a server prints the same line)."""
    out, seen = [], set()
    for line in log_text.splitlines():
        match = _BOUNDARY.search(line)
        if match is None or match.group(2) in seen:
            continue
        seen.add(match.group(2))
        counters = {name: {"rows": int(rows), "accepted": int(accepted), "verified": int(verified)}
                    for name, rows, accepted, verified in _SOURCE.findall(match.group(1))}
        out.append({"steps": int(match.group(2)), "sources": counters})
    return out


def executed_rounds(probe: dict) -> list[tuple[str, int]]:
    """``("warm" | "round", index)`` of the rounds that reached the server's decode steps (at least one stream
    answered), in the order ``lookup_probe.py`` ran them: a warm round right before the first round of each
    (kind, streams)."""
    order, warmed, warm_index = [], set(), 0
    warm_rounds = probe.get("warm", [])
    for i, rnd in enumerate(probe.get("rounds", [])):
        key = (rnd.get("kind"), rnd.get("streams"))
        if key not in warmed and warm_index < len(warm_rounds):
            warm = warm_rounds[warm_index]
            if (warm.get("kind"), warm.get("streams")) == key:
                if warm.get("ok"):
                    order.append(("warm", warm_index))
                warm_index += 1
        warmed.add(key)
        if "warm_failed" not in rnd and rnd.get("ok"):
            order.append(("round", i))
    return order


def source_deltas(probe: dict, log_text: str) -> dict[int, dict]:
    """``{round index: {source: {"rows", "accepted", "verified"}}}`` for the measured rounds of a server with the
    lookup hook. The server prints its cumulative counters when a new set of requests arrives, so the line of a
    round comes with the next round (the probe's last request gives the last one): line i belongs to the i-th
    executed round. ``{}`` when the lines do not line up with the rounds (one line less: the last round has no
    counters)."""
    lines = boundary_counters(log_text)
    order = executed_rounds(probe)
    if not lines or len(lines) not in (len(order), len(order) - 1):
        return {}
    out, previous = {}, {}
    for (what, index), line in zip(order, lines):
        delta = {name: {key: value - previous.get(name, {}).get(key, 0) for key, value in counters.items()}
                 for name, counters in line["sources"].items()}
        previous = line["sources"]
        if what == "round":
            out[index] = delta
    return out


def load_run(run_dir: str) -> dict | None:
    """A job's probe record with its server's speculative config, lookup flag and log; None without probe.json."""
    try:
        with open(os.path.join(run_dir, "probe.json"), encoding="utf-8") as f:
            probe = json.load(f)
    except (OSError, ValueError):
        return None
    summary, log_text = {}, ""
    try:
        with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(run_dir, "server0.log"), encoding="utf-8", errors="replace") as f:
            log_text = f.read()
    except OSError:
        pass
    spec = summary.get("speculative_config")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            spec = None
    env = summary.get("env") or {}
    return {"dir": run_dir, "probe": probe, "spec": spec, "tp": summary.get("tp"),
            "lookup_env": env.get("VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP"), "healthy_after_s": summary.get("healthy_after_s"),
            "sources": source_deltas(probe, log_text)}


def reference_streams(runs: list[dict]) -> dict[str, list[int]]:
    """``{prompt hash: the longest stream of token ids}`` over the rounds without speculation (``is_reference``)."""
    best: dict[str, list[int]] = {}
    for run in runs:
        for rnd in run["probe"].get("rounds", []):
            if not is_reference(arm_name(run["spec"], run["lookup_env"], rnd.get("mode"))):
                continue
            for sha, ids in zip(rnd.get("prompt_sha") or [], rnd.get("token_ids") or []):
                if ids and len(ids) > len(best.get(sha, [])):
                    best[sha] = ids
    return best


def lossless(rnd: dict, reference: dict[str, list[int]]) -> dict | None:
    """The round's streams against the no-speculation streams of the same prompts (their common length)."""
    compared, same, first = 0, 0, []
    for sha, ids in zip(rnd.get("prompt_sha") or [], rnd.get("token_ids") or []):
        want = reference.get(sha)
        if not ids or not want:
            continue
        compared += 1
        limit = min(len(ids), len(want))
        diff = next((i for i in range(limit) if ids[i] != want[i]), None)
        if diff is None:
            same += 1
        else:
            first.append(diff)
    if not compared:
        return None
    return {"compared": compared, "identical": same, "first_difference_min": min(first) if first else None,
            "first_difference_p50": statistics.median(first) if first else None}


def rows_of(runs: list[dict]) -> list[dict]:
    reference = reference_streams(runs)
    rows = []
    for run in runs:
        for index, rnd in enumerate(run["probe"].get("rounds", [])):
            if not rnd.get("ok"):
                rows.append({"kind": rnd.get("kind"), "streams": rnd.get("streams"), "failed": True,
                             "arm": arm_name(run["spec"], run["lookup_env"], rnd.get("mode")),
                             "error": (rnd.get("error_samples") or rnd.get("warm_failed") or [None])[0]})
                continue
            spec = rnd.get("spec") or {}
            k = (run["spec"] or {}).get("num_speculative_tokens") or 0
            arm = arm_name(run["spec"], run["lookup_env"], rnd.get("mode"))
            rows.append({
                "kind": rnd["kind"], "streams": rnd["streams"], "tokens": rnd["tokens"], "failed": False,
                "arm": arm, "label": run["probe"].get("label"),
                "tp": run["tp"], "k": k,
                "tokens_per_step": spec.get("acceptance_length") or 1.0,
                "position_rates": spec.get("position_rates") or [],
                "ms_per_step": rnd.get("ms_per_step"), "gap_ms_p90": rnd.get("gap_ms_p90"),
                "steady_ms_per_token": rnd.get("steady_ms_per_token"),
                "span_ms_per_token_p50": rnd.get("ms_per_token_p50"),
                "aggregate_tok_s": rnd.get("aggregate_tok_s"), "ttft_p50_s": rnd.get("ttft_p50_s"),
                "prompt_tokens": rnd.get("prompt_tokens"),
                "rows_per_step_mtp_shapes": rnd["streams"] * (1 + k),
                "step_sizes": rnd.get("step_sizes"),
                "sources": run["sources"].get(index),
                "lossless": None if is_reference(arm) else lossless(rnd, reference)})
    # the reference of a (kind, streams): a server without speculation when there is one, else the n-gram server's
    # rounds with the lookup off (the last such round of each)
    base = {(r["kind"], r["streams"]): r for r in rows if not r["failed"] and r["arm"] == NO_SPEC_NGRAM}
    base.update({(r["kind"], r["streams"]): r for r in rows if not r["failed"] and r["arm"] == NO_SPEC})
    for r in rows:
        ref = base.get((r["kind"], r["streams"]))
        r["speedup_vs_no_speculation"] = (ref["steady_ms_per_token"] / r["steady_ms_per_token"]
                                          if ref and not r["failed"] and r.get("steady_ms_per_token") else None)
    order = {"edit": 0, "editf": 1, "editl": 2, "prose": 3, "mix": 4}
    rows.sort(key=lambda r: (order.get(r["kind"], 9), -(r["streams"] or 0), not is_reference(r["arm"]), r["arm"]))
    return rows


def table(rows: list[dict]) -> list[str]:
    lines = ["| prompts | streams | arm | tokens/step | accepted >= j per step | ms/step | ms/token steady | "
             "ms/token span p50 | vs no speculation | rows/step | equal to no speculation |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["failed"]:
            lines.append("| %s | %s | %s | FAILED: %s | | | | | | | |" % (r["kind"], r["streams"], r["arm"],
                                                                         str(r.get("error"))[:80]))
            continue
        rates = " ".join("%.2f" % x for x in r["position_rates"] if x is not None) or "-"
        speedup = ("%.2fx" % r["speedup_vs_no_speculation"]) if r["speedup_vs_no_speculation"] else "-"
        same = "-"
        if r["lossless"]:
            same = "%d/%d" % (r["lossless"]["identical"], r["lossless"]["compared"])
            if r["lossless"]["first_difference_min"] is not None:
                same += " (first difference at token %d)" % r["lossless"]["first_difference_min"]
        lines.append("| %s | %d | %s | %.2f | %s | %.1f | %.2f | %.2f | %s | %s | %s |" % (
            r["kind"], r["streams"], r["arm"], r["tokens_per_step"], rates, r["ms_per_step"],
            r["steady_ms_per_token"], r["span_ms_per_token_p50"], speedup,
            r["rows_per_step_mtp_shapes"] if r["k"] else r["streams"], same))
        if r.get("sources"):
            parts = []
            for name in ("lookup", "mtp", "none"):
                src = r["sources"].get(name)
                if src and src["rows"]:
                    per = (src["accepted"] / src["verified"]) if src["verified"] else 0.0
                    parts.append("%s rows %d, %.2f accepted per row" % (name, src["rows"], per))
            if parts:
                lines.append("| | | | sources: %s | | | | | | | |" % "; ".join(parts))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", metavar="RUN_DIR")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    runs = [run for run in (load_run(d) for d in args.runs) if run is not None]
    if not runs:
        print("lookup_summary: no probe.json in %s" % ", ".join(args.runs), file=sys.stderr)
        return 1
    rows = rows_of(runs)
    print("\n".join(table(rows)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"runs": [{"dir": r["dir"], "spec": r["spec"], "tp": r["tp"], "lookup_env": r["lookup_env"],
                                 "healthy_after_s": r["healthy_after_s"]} for r in runs], "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
