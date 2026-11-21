#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What the threads of some processes did for a few seconds, from /proc (Linux; standard library; no privilege).

``python3 thread_stat.py --pids 101,102 --seconds 3 --out stat.json`` reads, for every thread of the given
processes, at the start and at the end of the window: the CPU time and the time it stood runnable in a run queue
(``/proc/<pid>/task/<tid>/schedstat``), its voluntary and involuntary context switches (``status``) and the CPU
it last ran on (``stat``). Threads that used CPU in the first 0.3 s are then sampled every ``--interval`` seconds
for the CPU they run on: how many different CPUs, how many moves. The step-anatomy question it answers: a rank
whose HPU-graph replay call takes longer on the host than the other ranks', was its calling thread preempted
(involuntary switches, run-queue wait), moved between CPUs, or simply busy for longer (CPU time)?

The first thread of a process (tid == pid) is the one that executes the model. Output: one JSON document, and a
table of the busiest threads on standard output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def thread_ids(pid: int) -> list:
    try:
        return sorted(int(name) for name in os.listdir("/proc/%d/task" % pid) if name.isdigit())
    except OSError:
        return []


def snapshot(pid: int, tid: int) -> dict | None:
    base = "/proc/%d/task/%d/" % (pid, tid)
    sched = read(base + "schedstat").split()
    if len(sched) < 3:
        return None
    snap = {"cpu_ns": int(sched[0]), "runq_ns": int(sched[1]), "slices": int(sched[2]), "voluntary": 0,
            "involuntary": 0, "processor": processor(pid, tid), "comm": read(base + "comm").strip()}
    for line in read(base + "status").splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            snap["voluntary"] = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            snap["involuntary"] = int(line.split()[1])
    return snap


def processor(pid: int, tid: int) -> int | None:
    """The CPU the thread last ran on: field 39 of ``stat`` (the comm field may hold spaces and brackets)."""
    text = read("/proc/%d/task/%d/stat" % (pid, tid))
    tail = text.rsplit(")", 1)[-1].split()
    try:
        return int(tail[36])
    except (IndexError, ValueError):
        return None


def measure(pids: list, seconds: float, interval: float = 0.01, probe_s: float = 0.3, clock=time.monotonic,
            sleep=time.sleep) -> dict:
    start = {(pid, tid): snapshot(pid, tid) for pid in pids for tid in thread_ids(pid)}
    start = {key: snap for key, snap in start.items() if snap is not None}
    t0 = clock()
    sleep(min(probe_s, seconds))
    busy = []
    for (pid, tid), first in start.items():
        now = snapshot(pid, tid)
        if now is not None and (now["cpu_ns"] - first["cpu_ns"] > 0.02 * probe_s * 1e9 or tid == pid):
            busy.append((pid, tid))
    seen = {key: {} for key in busy}
    moves = {key: 0 for key in busy}
    last = {key: start[key]["processor"] for key in busy}
    samples = 0
    while clock() - t0 < seconds:
        for key in busy:
            cpu = processor(*key)
            if cpu is None:
                continue
            seen[key][cpu] = seen[key].get(cpu, 0) + 1
            if last[key] is not None and cpu != last[key]:
                moves[key] += 1
            last[key] = cpu
        samples += 1
        sleep(interval)
    elapsed = clock() - t0
    threads = []
    for (pid, tid), first in start.items():
        end = snapshot(pid, tid)
        if end is None:
            continue
        row = {"pid": pid, "tid": tid, "main": pid == tid, "comm": end["comm"],
               "cpu_ms": (end["cpu_ns"] - first["cpu_ns"]) / 1e6, "runq_ms": (end["runq_ns"] - first["runq_ns"]) / 1e6,
               "slices": end["slices"] - first["slices"], "voluntary": end["voluntary"] - first["voluntary"],
               "involuntary": end["involuntary"] - first["involuntary"]}
        key = (pid, tid)
        if key in seen:
            row["cpus"] = {str(cpu): n for cpu, n in sorted(seen[key].items())}
            row["moves"] = moves[key]
        threads.append(row)
    threads.sort(key=lambda row: (-int(row["main"]), -row["cpu_ms"]))
    return {"seconds": elapsed, "samples": samples, "interval_s": interval, "pids": pids, "threads": threads}


def table(result: dict, top: int = 4) -> str:
    """Per process: its main thread and its ``top`` busiest other threads."""
    lines = ["window %.2f s, %d CPU samples per busy thread" % (result["seconds"], result["samples"]),
             "pid tid thread cpu_ms runq_wait_ms voluntary involuntary cpus_seen moves"]
    for pid in result["pids"]:
        rows = [row for row in result["threads"] if row["pid"] == pid]
        rows = [row for row in rows if row["main"]] + sorted((row for row in rows if not row["main"]),
                                                              key=lambda row: -row["cpu_ms"])[:top]
        for row in rows:
            lines.append("%d %d %s%s %.1f %.2f %d %d %s %s" % (
                row["pid"], row["tid"], row["comm"].replace(" ", "_"), "*" if row["main"] else "", row["cpu_ms"],
                row["runq_ms"], row["voluntary"], row["involuntary"],
                ",".join(sorted(row.get("cpus", {}), key=int)) or "-", row.get("moves", "-")))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="thread_stat.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--pids", required=True, help="comma-separated process ids")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=0.01)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    pids = [int(p) for p in args.pids.split(",") if p.strip()]
    result = measure(pids, args.seconds, args.interval)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
    print(table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
