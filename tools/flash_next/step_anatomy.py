# SPDX-License-Identifier: Apache-2.0
"""Print the anatomy of the steady-state decode steps of a vLLM-Gaudi high-level profiler trace.

``python tools/flash_next/step_anatomy.py <trace.json>`` reads a ``server_events_<instance>.json`` trace
written by ``HabanaHighLevelProfiler`` (``VLLM_PROFILER_ENABLED=true``, rank 0 only), splits the run into
decode steps (the windows between consecutive ``decode`` event starts, after the end of the last ``warmup``
event) and prints, for each host phase, the mean/p50/p90 milliseconds it takes and its share of the step
period: the decode forward + sampler, sampler post-processing, the drafter (``propose_draft_tokens``), the
next step's input prep, the nested ``model_forward_*`` / ``compute_logits*`` / ``sampler`` sub-rows, and the
untracked gap left over. Windows that contain a ``prefill`` event start are mixed steps: counted, not tabulated.

Durations are host time: device work is asynchronous, and its time is absorbed into the first event that
synchronises (usually ``sampler`` / ``sampler_postprocessing``, which copy tokens to the CPU).

The trace file is a JSON array the server leaves unterminated while it runs: ``[`` followed by one event
object per event, each followed by ``,``, with no closing ``]``. Events are ``{"pid", "tid", "ph": "X",
"name", "ts", "dur", "args"}`` with ``ts`` and ``dur`` in microseconds; ``dur`` is ``null`` for events that
never ended (skipped). Counter events (``"ph": "C"``) are ignored.

Percentiles are nearest-rank on the sorted values. ``--last N`` analyses only the last N steady steps;
``--json`` prints the same numbers as one JSON object. Exits 2 with a message if the trace has fewer than
2 ``decode`` events after warmup.
"""

import argparse
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path

FOOTNOTE = ("durations are host time: device work is asynchronous and its time is absorbed into the first "
            "event that synchronises (usually sampler / sampler_postprocessing, which copy tokens to the CPU)")

PERIOD_ROW = "period"
DECODE_ROW = "decode (forward + sampler, host)"
GAP_ROW = "untracked gap"
PHASE_ROWS = (DECODE_ROW, "sampler post-processing", "drafter", "input prep (next step)")
SUBROW_ROWS = ("model_forward", "compute_logits", "sampler")
PHASE_BY_NAME = {
    "decode": DECODE_ROW,
    "sampler_postprocessing": "sampler post-processing",
    "propose_draft_tokens": "drafter",
    "prepare_input_tensors": "input prep (next step)",
}


def load_events(path):
    """Load a (possibly unterminated) high-level profiler trace; keep complete ``ph == "X"`` events only."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.endswith(","):
        text = text[:-1]
    if not text.endswith("]"):
        text += "]"
    events = json.loads(text)
    return [e for e in events if e.get("ph") == "X" and isinstance(e.get("dur"), (int, float))]


def _subrow_of(name):
    if name.startswith("model_forward_"):
        return "model_forward"
    if name.startswith("compute_logits"):
        return "compute_logits"
    if name == "sampler":
        return "sampler"
    return None


def _has_start(sorted_starts, lo, hi):
    index = bisect_left(sorted_starts, lo)
    return index < len(sorted_starts) and sorted_starts[index] < hi


def analyze(events):
    """Split events into decode-step windows. Returns ``(steps, mixed_skipped, n_decodes)``.

    Step ``i`` is the window ``[decode_i.ts, decode_{i+1}.ts)`` after the end of the last ``warmup`` event;
    a window containing a ``prefill`` event start is a mixed step (counted, dropped). Each step is
    ``{"period": us, "rows": {row: us}}`` with a value for every phase, sub-row and the untracked gap.
    """
    events = sorted(events, key=lambda e: e["ts"])
    warmup_ends = [e["ts"] + e["dur"] for e in events if e["name"] == "warmup"]
    if warmup_ends:
        cutoff = max(warmup_ends)
        events = [e for e in events if e["ts"] >= cutoff]
    ts_list = [e["ts"] for e in events]
    decode_starts = [e["ts"] for e in events if e["name"] == "decode"]
    prefill_starts = [e["ts"] for e in events if e["name"] == "prefill"]
    steps, mixed = [], 0
    for lo_ts, hi_ts in zip(decode_starts, decode_starts[1:]):
        if _has_start(prefill_starts, lo_ts, hi_ts):
            mixed += 1
            continue
        rows = {r: 0.0 for r in (*PHASE_ROWS, *SUBROW_ROWS)}
        for event in events[bisect_left(ts_list, lo_ts):bisect_left(ts_list, hi_ts)]:
            phase = PHASE_BY_NAME.get(event["name"])
            if phase is not None:
                rows[phase] += event["dur"]
            subrow = _subrow_of(event["name"])
            if subrow is not None:
                rows[subrow] += event["dur"]
        rows[GAP_ROW] = max(0.0, (hi_ts - lo_ts) - sum(rows[phase] for phase in PHASE_ROWS))
        steps.append({"period": hi_ts - lo_ts, "rows": rows})
    return steps, mixed, len(decode_starts)


def percentile(values, fraction):
    """Nearest-rank percentile of a list of numbers (1 <= ceil(fraction * n) <= n)."""
    ordered = sorted(values)
    rank = min(max(math.ceil(fraction * len(ordered)), 1), len(ordered))
    return ordered[rank - 1]


def _stats(values_us, period_mean_us):
    values_ms = [v / 1000.0 for v in values_us]
    mean_ms = sum(values_ms) / len(values_ms)
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(values_ms, 0.50),
        "p90_ms": percentile(values_ms, 0.90),
        "share_of_period_pct": 100.0 * mean_ms * 1000.0 / period_mean_us,
    }


def summarize(steps):
    """Mean/p50/p90 ms and share of the period, per row, in table order."""
    period_us = [step["period"] for step in steps]
    period_mean_us = sum(period_us) / len(period_us)
    table = {PERIOD_ROW: _stats(period_us, period_mean_us)}
    for phase in PHASE_ROWS:
        table[phase] = _stats([step["rows"][phase] for step in steps], period_mean_us)
        if phase == DECODE_ROW:
            for subrow in SUBROW_ROWS:
                table[f"\u21b3 {subrow}"] = _stats([step["rows"][subrow] for step in steps], period_mean_us)
    table[GAP_ROW] = _stats([step["rows"][GAP_ROW] for step in steps], period_mean_us)
    return table


def _print_markdown(trace, n_steps, mixed, table):
    print(f"# step anatomy: {trace}")
    print(f"steady steps: {n_steps}, mixed steps skipped: {mixed}")
    print()
    print("| phase | mean ms | p50 ms | p90 ms | share of period |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for label, stats in table.items():
        print(f"| {label} | {stats['mean_ms']:.3f} | {stats['p50_ms']:.3f} | {stats['p90_ms']:.3f} | "
              f"{stats['share_of_period_pct']:.1f}% |")
    print()
    print(f"> note: {FOOTNOTE}.")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="step_anatomy.py",
        description="Print the decode-step period of a vLLM-Gaudi high-level profiler trace and how much of "
                    "it each host phase takes (input prep, forward launch, sampler, drafter, untracked gap).",
        epilog=f"note: {FOOTNOTE}.")
    parser.add_argument("trace", help="server_events_<instance>.json trace file to analyze")
    parser.add_argument("--last", type=int, default=None, metavar="N", help="analyze only the last N steady steps")
    parser.add_argument("--json", action="store_true", help="print the same numbers as one JSON object instead")
    args = parser.parse_args(argv)
    if args.last is not None and args.last < 1:
        parser.error("--last needs N >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    steps, mixed, n_decodes = analyze(load_events(args.trace))
    if n_decodes < 2:
        print(f"{args.trace}: only {n_decodes} decode event(s) after warmup; "
              f"need at least 2 to form a step window", file=sys.stderr)
        return 2
    if args.last is not None:
        steps = steps[-args.last:]
    if not steps:
        print(f"{args.trace}: no steady decode steps left ({mixed} mixed step(s) skipped)", file=sys.stderr)
        return 2
    table = summarize(steps)
    if args.json:
        print(json.dumps({
            "trace": args.trace,
            "steady_steps": len(steps),
            "mixed_steps_skipped": mixed,
            "rows": table,
        }))
    else:
        _print_markdown(args.trace, len(steps), mixed, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
