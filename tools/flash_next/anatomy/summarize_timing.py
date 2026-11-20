#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Budget tables of the step timer (``VLLM_HPU_STEP_TIMING``) of one server run of the step-anatomy job.

``python3 -I summarize_timing.py <run dir>`` reads ``<run dir>/timing/step_timing_<role>_r<rank>_<pid>.jsonl``
(one JSON record per window: the steps of one kind and row count between two idle gaps) and, when present,
``<run dir>/probe.json`` of ``anatomy_probe.py`` (the wall-clock span of every round). Each record is matched to
the probe round it falls in, and the steady records (``kind`` decode or verify, ``rows`` equal to the round's
stream count) are tabulated: one column per (timer mode, streams), one row per span, p50 milliseconds of the
driver rank (``--rank``), the spread over the ranks, the engine core's spans and the per-step counters
(collectives, dispatched ops, host sync points). ``--json OUT`` writes the same numbers. Standard library only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

WORKER_ORDER = ("period", "gap", "execute_model", "rpc_gap", "setup", "prepare", "launch_inputs", "forward",
                "logits", "launch_model", "device_forward", "sample", "launch_sample", "device_sample", "wait",
                "draft", "post", "seg_compute", "seg_collective", "step")
ENGINE_ORDER = ("schedule", "execute_model_send", "execute_model_wait", "execute_model_rpc", "sample_tokens_send",
                "sample_tokens_wait", "sample_tokens_rpc", "update_from_output", "other", "step", "gap", "period")
FILE_RE = re.compile(r"step_timing_(?P<role>[a-z]+)_r(?P<rank>\d+)_(?P<pid>\d+)\.jsonl$")


def load_records(run_dir: str) -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "timing", "step_timing_*.jsonl"))):
        match = FILE_RE.search(os.path.basename(path))
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                rec["_file"] = os.path.basename(path)
                if match and "role" not in rec:
                    rec["role"] = match.group("role")
                records.append(rec)
    return records


def load_rounds(run_dir: str) -> list:
    """Every round of probe.json: ``{"label", "mode", "streams", "tokens", "t_start", "t_end", "ms_per_token"}``."""
    path = os.path.join(run_dir, "probe.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        probe = json.load(fh)
    rounds = []
    for phase in probe.get("phases", []):
        for res in phase.get("rounds", []):
            if "t_start" not in res:
                continue
            label = "%s/N=%d%s" % (phase["mode"], res["streams"], "" if phase.get("min_tokens", True) else "/nomin")
            if any(r["label"] == label for r in rounds):  # the same phase twice: number the repeats
                label += "/#%d" % (1 + sum(1 for r in rounds if r["label"].startswith(label)))
            rounds.append({"label": label, "mode": phase["mode"],
                           "streams": res["streams"], "tokens": res["tokens"], "t_start": res["t_start"],
                           "t_end": res["t_end"], "ms_per_token": res.get("ms_per_token_p50"), "trace": False})
    for res in probe.get("traces", []):
        if "t_start" in res:
            rounds.append({"label": "trace/N=%d" % res["streams"], "mode": "trace", "streams": res["streams"],
                           "tokens": res["tokens"], "t_start": res["t_start"], "t_end": res["t_end"],
                           "ms_per_token": res.get("ms_per_token_p50"), "trace": True})
    return rounds


def round_of(rec: dict, rounds: list, slack_s: float = 1.0):
    t = rec.get("t_first")
    for rnd in rounds:
        if t is not None and rnd["t_start"] - slack_s <= t <= rnd["t_end"] + slack_s:
            return rnd
    return None


def p50(rec: dict, group: str, name: str):
    summary = (rec.get(group) or {}).get(name)
    return None if summary is None else summary.get("p50")


def steady(records: list, rounds: list, min_steps: int) -> dict:
    """``{(round label, role): {rank: record}}`` of the steady records: decode or verify at the round's streams."""
    out = {}
    for rec in records:
        rnd = round_of(rec, rounds) if rounds else None
        if rec.get("kind") not in ("decode", "verify") or rec.get("steps", 0) < min_steps:
            continue
        if rnd is not None and rec.get("rows") != rnd["streams"]:
            continue
        label = rnd["label"] if rnd is not None else "mode%s/N=%s" % (rec.get("mode"), rec.get("rows"))
        key = (label, rec.get("role"))
        slot = out.setdefault(key, {})
        rank = rec.get("rank", 0)
        if rank not in slot or rec.get("steps", 0) > slot[rank].get("steps", 0):
            slot[rank] = rec
    return out


def fmt(value, digits=2):
    if value is None:
        return ""
    if isinstance(value, float) and abs(value) >= 1000:
        return "%.0f" % value
    return ("%." + str(digits) + "f") % value if isinstance(value, float) else str(value)


def table(columns: list, rows: list) -> str:
    head = "| | " + " | ".join(c for c, _ in columns) + " |"
    rule = "|---|" + "---|" * len(columns)
    body = []
    for name, getter in rows:
        cells = [getter(col) for _, col in columns]
        if all(cell in ("", None) for cell in cells):
            continue
        body.append("| %s | %s |" % (name, " | ".join(fmt(c) for c in cells)))
    return "\n".join([head, rule] + body)


def summarize(run_dir: str, rank: int = 0, min_steps: int = 20) -> dict:
    records = load_records(run_dir)
    rounds = load_rounds(run_dir)
    groups = steady(records, rounds, min_steps)
    labels = []
    for rnd in rounds:
        if rnd["label"] not in labels:
            labels.append(rnd["label"])
    for label, _role in groups:
        if label not in labels:
            labels.append(label)
    result = {"run_dir": run_dir, "records": len(records), "rounds": rounds, "columns": []}
    for label in labels:
        workers = groups.get((label, "worker"), {})
        engines = groups.get((label, "engine"), {})
        if not workers and not engines:
            continue
        driver = workers.get(rank) or (next(iter(workers.values())) if workers else None)
        engine = next(iter(engines.values())) if engines else None
        col = {"label": label, "ranks": sorted(workers), "steps": driver.get("steps") if driver else None,
               "mode": driver.get("mode") if driver else (engine or {}).get("mode"),
               "bucket_rows": driver.get("bucket_rows") if driver else None, "worker": {}, "worker_rank_max": {},
               "nested": {}, "counters": {}, "engine": {}, "segments": (driver or {}).get("segments"),
               "sync_sites": (driver or {}).get("sync_sites"), "op_names": (driver or {}).get("op_names_per_step")}
        rnd = next((r for r in rounds if r["label"] == label), None)
        col["ms_per_token"] = rnd["ms_per_token"] if rnd else None
        if driver:
            for name in (driver.get("spans_ms") or {}):
                col["worker"][name] = p50(driver, "spans_ms", name)
                values = [p50(rec, "spans_ms", name) for rec in workers.values()]
                values = [v for v in values if v is not None]
                col["worker_rank_max"][name] = max(values) if values else None
            for name in (driver.get("nested_ms") or {}):
                col["nested"][name] = p50(driver, "nested_ms", name)
            for name in (driver.get("counters") or {}):
                col["counters"][name] = p50(driver, "counters", name)
        if engine:
            for name in (engine.get("spans_ms") or {}):
                col["engine"][name] = p50(engine, "spans_ms", name)
        result["columns"].append(col)
    return result


def markdown(result: dict) -> str:
    cols = [(c["label"], c) for c in result["columns"]]
    if not cols:
        return "no steady decode records in %s (%d records)" % (result["run_dir"], result["records"])
    rows = [("probe ms per token (p50 over streams)", lambda c: c.get("ms_per_token")),
            ("steps in the record", lambda c: c.get("steps")), ("decode bucket rows", lambda c: c.get("bucket_rows"))]
    names = [n for n in WORKER_ORDER if any(n in c["worker"] for _, c in cols)]
    names += sorted({n for _, c in cols for n in c["worker"]} - set(names))
    rows += [("worker %s" % n, (lambda n: lambda c: c["worker"].get(n))(n)) for n in names]
    rows += [("worker %s, slowest rank" % n, (lambda n: lambda c: c["worker_rank_max"].get(n))(n))
             for n in ("period", "forward", "wait", "device_forward")]
    rows += [("in %s" % n, (lambda n: lambda c: c["nested"].get(n))(n))
             for n in sorted({n for _, c in cols for n in c["nested"]})]
    rows += [("n %s" % n, (lambda n: lambda c: c["counters"].get(n))(n))
             for n in sorted({n for _, c in cols for n in c["counters"]})]
    enames = [n for n in ENGINE_ORDER if any(n in c["engine"] for _, c in cols)]
    rows += [("engine %s" % n, (lambda n: lambda c: c["engine"].get(n))(n)) for n in enames]
    out = [table(cols, rows)]
    for _, c in cols:
        if c.get("segments"):
            seg = c["segments"]
            out.append("\n%s: %d collective segments per step (mode 3), compute p50 sum %.2f ms, collective p50 sum "
                       "%.2f ms" % (c["label"], len(seg["names"]), sum(seg["compute_p50_ms"]),
                                    sum(seg["collective_p50_ms"])))
        if c.get("sync_sites"):
            out.append("\n%s: host sync points per step (mode 2): %s" % (c["label"], "; ".join(
                "%s at %s x%.1f" % (s["what"], s["where"], s["per_step"]) for s in c["sync_sites"][:12])))
        if c.get("op_names"):
            top = sorted(c["op_names"].items(), key=lambda kv: -kv[1])[:15]
            out.append("\n%s: dispatched ops per step (mode 4), top: %s" % (c["label"], ", ".join(
                "%s %.0f" % (k, v) for k, v in top)))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="summarize_timing.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    results = []
    for run_dir in args.run_dir:
        result = summarize(run_dir, args.rank, args.min_steps)
        results.append(result)
        print("## %s (%d timer records, %d probe rounds)\n" % (run_dir, result["records"], len(result["rounds"])))
        print(markdown(result))
        print()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    return 0 if any(r["columns"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
