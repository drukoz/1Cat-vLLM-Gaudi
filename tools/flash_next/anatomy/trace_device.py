#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device budget of steady decode steps from a torch-profiler trace with HPU activities (step-anatomy, I12).

``python3 -I trace_device.py <trace.json[.gz]> [--steps N]`` reads one Chrome trace written by the HPU worker's
torch profiler (``--profiler-config`` with ``"profiler": "torch"``, ``HABANA_PROFILE=profile_api_light``) and splits
the device timeline into:

* busy time per engine class (TPC, MME, DMA, COMM for collective / network events, OTHER) and their union;
* idle time (no device event at all) and the gaps between busy blocks, as a histogram (how many gaps of which size,
  and how much time they hold): many short gaps are a per-node launch or dependency cost, long gaps are the host;
* bursts: the busy stretches separated by gaps longer than ``--burst-gap-ms`` (one step, or one graph of a step);
* the device events per step (graph nodes as the device sees them) and the names that take the most time;
* the host (CPU) events by name, count and total time.

Per-step figures divide by ``--steps`` (from the probe's capture window) or, without it, by the number of bursts
longer than ``--min-burst-ms``. ``--json OUT`` writes the numbers. ``--probe`` only prints the trace's format (the
processes, threads and sample events), for a trace whose layout this script does not know yet. Standard library.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict

CLASSES = ("TPC", "MME", "DMA", "COMM", "OTHER")
COMM_WORDS = ("hcl", "hccl", "nic", "collective", "allreduce", "all_reduce", "allgather", "all_gather",
              "reduce_scatter", "reducescatter", "scaleout", "scale_out", "scaleup", "scale_up", "rdma", "send", "recv")
DEVICE_WORDS = ("hpu", "device", "gaudi", "habana", "synapse")
GAP_EDGES_US = (2.0, 10.0, 50.0, 200.0, 1000.0)


def load(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    data = json.loads(text)
    if isinstance(data, list):
        data = {"traceEvents": data}
    return data


def names(events: list) -> tuple:
    procs, threads = {}, {}
    for ev in events:
        if ev.get("ph") != "M":
            continue
        args = ev.get("args") or {}
        if ev.get("name") == "process_name":
            procs[ev.get("pid")] = str(args.get("name", ""))
        elif ev.get("name") == "thread_name":
            threads[(ev.get("pid"), ev.get("tid"))] = str(args.get("name", ""))
    return procs, threads


def is_device(ev: dict, procs: dict, threads: dict) -> bool:
    args = ev.get("args") or {}
    if "HW event name" in args:
        return True
    cat = str(ev.get("cat", "")).lower()
    if cat in ("kernel", "gpu_op", "hpu_op", "device", "hpu", "hw"):
        return True
    pname = procs.get(ev.get("pid"), "").lower()
    if any(word in pname for word in DEVICE_WORDS) and "cpu" not in pname:
        return True
    return False


def engine_class(ev: dict, threads: dict) -> str:
    args = ev.get("args") or {}
    text = " ".join([str(args.get("HW event name", "")), threads.get((ev.get("pid"), ev.get("tid")), ""),
                     str(ev.get("name", "")), str(args.get("engine", "")), str(args.get("Engine", ""))]).lower()
    if any(word in text for word in COMM_WORDS):
        return "COMM"
    for cls in ("TPC", "MME", "DMA"):
        if cls.lower() in text:
            return cls
    return "OTHER"


def merge(intervals: list) -> list:
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1][1] = end
        else:
            out.append([start, end])
    return out


def total(intervals: list) -> float:
    return sum(end - start for start, end in intervals)


def clip(intervals: list, lo: float, hi: float) -> list:
    return [[max(a, lo), min(b, hi)] for a, b in intervals if b > lo and a < hi]


def analyze(data: dict, steps: int | None = None, burst_gap_ms: float = 1.0, min_burst_ms: float = 0.5,
            window: tuple | None = None) -> dict:
    events = data.get("traceEvents", [])
    procs, threads = names(events)
    device, host = [], []
    for ev in events:
        if ev.get("ph") != "X" or not isinstance(ev.get("dur"), (int, float)) or ev.get("ts") is None:
            continue
        ts, dur = float(ev["ts"]), float(ev["dur"])
        if window and not (window[0] <= ts <= window[1]):
            continue
        (device if is_device(ev, procs, threads) else host).append((ts, ts + dur, ev))
    result = {"events": len(events), "device_events": len(device), "host_events": len(host)}
    if not device:
        result["error"] = "no device events: the trace has no HPU activity (see --probe)"
        return result
    lo = min(s for s, _, _ in device)
    hi = max(e for _, e, _ in device)
    by_class = defaultdict(list)
    counts = Counter()
    name_time = Counter()
    name_count = Counter()
    for start, end, ev in device:
        cls = engine_class(ev, threads)
        by_class[cls].append((start, end))
        counts[cls] += 1
        key = "%s | %s" % (cls, str(ev.get("name", ""))[:80])
        name_time[key] += end - start
        name_count[key] += 1
    merged = {cls: merge(iv) for cls, iv in by_class.items()}
    compute = merge(by_class.get("TPC", []) + by_class.get("MME", []))
    everything = merge([iv for ivs in by_class.values() for iv in ivs])
    gaps = [b[0] - a[1] for a, b in zip(everything, everything[1:]) if b[0] > a[1]]
    hist = []
    edges = (0.0, ) + GAP_EDGES_US + (float("inf"), )
    for low, high in zip(edges, edges[1:]):
        inside = [g for g in gaps if low <= g < high]
        hist.append({"from_us": low, "to_us": high, "count": len(inside), "total_ms": sum(inside) / 1e3})
    bursts, current = [], None
    for start, end in everything:
        if current is None or start - current[1] > burst_gap_ms * 1e3:
            if current is not None:
                bursts.append(current)
            current = [start, end, 0.0]
        current[1] = max(current[1], end)
        current[2] += end - start
    if current is not None:
        bursts.append(current)
    long_bursts = [b for b in bursts if (b[1] - b[0]) >= min_burst_ms * 1e3]
    n_steps = steps or max(1, len(long_bursts))
    comm_only = total(merge(by_class.get("COMM", []))) - sum(
        total(clip(compute, a, b)) for a, b in merge(by_class.get("COMM", [])))
    span = hi - lo
    result.update({
        "window_ms": span / 1e3,
        "steps": n_steps,
        "steps_from": "argument" if steps else "bursts",
        "per_step_ms": {
            "window": span / 1e3 / n_steps,
            "busy_any": total(everything) / 1e3 / n_steps,
            "busy_compute_tpc_or_mme": total(compute) / 1e3 / n_steps,
            "idle": (span - total(everything)) / 1e3 / n_steps,
            "comm_not_under_compute": max(0.0, comm_only) / 1e3 / n_steps,
            **{"busy_" + cls: total(iv) / 1e3 / n_steps for cls, iv in merged.items()},
        },
        "device_events_per_step": {cls: counts[cls] / n_steps for cls in CLASSES if counts[cls]},
        "gaps_per_step": [{**h, "count": h["count"] / n_steps, "total_ms": h["total_ms"] / n_steps} for h in hist],
        "bursts": {"count": len(bursts), "long": len(long_bursts),
                   "long_ms": [round((b[1] - b[0]) / 1e3, 3) for b in long_bursts][:40],
                   "long_busy_ms": [round(b[2] / 1e3, 3) for b in long_bursts][:40]},
        "top_device_names": [{"name": k, "ms_per_step": v / 1e3 / n_steps, "count_per_step": name_count[k] / n_steps}
                             for k, v in name_time.most_common(25)],
    })
    host_time, host_count = Counter(), Counter()
    for start, end, ev in host:
        host_time[str(ev.get("name", ""))[:80]] += end - start
        host_count[str(ev.get("name", ""))[:80]] += 1
    result["top_host_names"] = [{"name": k, "ms_per_step": v / 1e3 / n_steps, "count_per_step": host_count[k] / n_steps}
                                for k, v in host_time.most_common(20)]
    return result


def probe(data: dict, samples: int = 3) -> str:
    events = data.get("traceEvents", [])
    procs, threads = names(events)
    lanes = Counter()
    first = defaultdict(list)
    for ev in events:
        if ev.get("ph") == "X":
            key = (ev.get("pid"), ev.get("tid"))
            lanes[key] += 1
            if len(first[key]) < samples:
                first[key].append({k: ev.get(k) for k in ("name", "cat", "ts", "dur", "args")})
    out = ["%d events; other top-level keys: %s" % (len(events), sorted(k for k in data if k != "traceEvents"))]
    for (pid, tid), n in lanes.most_common(40):
        out.append("pid %s (%s) tid %s (%s): %d X events" % (pid, procs.get(pid, "?"), tid, threads.get(
            (pid, tid), "?"), n))
        for sample in first[(pid, tid)]:
            out.append("    " + json.dumps(sample)[:300])
    return "\n".join(out)


def report(res: dict) -> str:
    if res.get("error"):
        return res["error"]
    ps = res["per_step_ms"]
    lines = ["window %.2f ms, %d steps (%s), %d device events" % (res["window_ms"], res["steps"], res["steps_from"],
                                                                  res["device_events"]),
             "per step: window %.3f ms, device busy (any engine) %.3f, compute (TPC or MME) %.3f, idle %.3f, "
             "collective not under compute %.3f" % (ps["window"], ps["busy_any"], ps["busy_compute_tpc_or_mme"],
                                                     ps["idle"], ps["comm_not_under_compute"]),
             "busy per class per step: " + ", ".join("%s %.3f" % (k[5:], v) for k, v in sorted(ps.items())
                                                      if k.startswith("busy_") and k[5:] in CLASSES),
             "device events per step: " + ", ".join("%s %.0f" % kv for kv in res["device_events_per_step"].items()),
             "gaps per step (count, ms): " + "; ".join(
                 "%g-%g us: %.1f, %.3f" % (g["from_us"], g["to_us"], g["count"], g["total_ms"])
                 for g in res["gaps_per_step"]),
             "bursts: %d (%d long); long burst ms: %s" % (res["bursts"]["count"], res["bursts"]["long"],
                                                          res["bursts"]["long_ms"][:12]),
             "top device names (ms per step, count per step):"]
    lines += ["  %-90s %8.3f %7.1f" % (t["name"], t["ms_per_step"], t["count_per_step"])
              for t in res["top_device_names"][:15]]
    lines.append("top host names (ms per step, count per step):")
    lines += ["  %-90s %8.3f %7.1f" % (t["name"], t["ms_per_step"], t["count_per_step"])
              for t in res["top_host_names"][:12]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="trace_device.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--burst-gap-ms", type=float, default=1.0)
    ap.add_argument("--min-burst-ms", type=float, default=0.5)
    ap.add_argument("--probe", action="store_true", help="print the trace's layout only")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    results = {}
    for path in args.trace:
        data = load(path)
        print("== %s" % path)
        if args.probe:
            print(probe(data))
            continue
        res = analyze(data, args.steps, args.burst_gap_ms, args.min_burst_ms)
        results[path] = res
        print(report(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    return 0 if args.probe or any(not r.get("error") for r in results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
