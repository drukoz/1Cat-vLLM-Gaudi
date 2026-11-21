#!/usr/bin/env python3
"""Sibling-aware CPU placement of a served model's HPU workers (native-replay; the pin arm of bridge job 00189).

Section 7.11 of docs/design/flash-next/native-replay.md: at 1 and 8 streams two of the eight ranks run about 2x
slower per launch, which two changes with every server start, and the VM reports one thread per core. INFERRED: two
busy main threads on two vCPUs that are hyperthreads of one host core. ``tools/flash_next/replay/cpu_siblings.py``
measures which vCPUs are siblings; this script uses its rule.

Run once inside the server's container, as root, right after the server is healthy (collectives' place_wrap.sh):

- every worker's main thread (the process's first thread, which runs the model) gets one vCPU of its own, in the
  worker's share of the container's CPUs (the same equal shares as ``VLLM_HPU_WORKER_CPU_PIN=1``);
- that vCPU's sibling (by ``PLACE_SIBLING_RULE``: ``a+N/2``, ``a^1``, or empty when unknown) gets no thread at all;
- the worker's other threads keep to the rest of its share; every other process of the container keeps off the
  reserved vCPUs;
- a background copy re-applies it every 2 s (a thread started later inherits its creator's mask) until the workers
  are gone.

Workers are found from the server log's lines ``(Worker_TP<r>... pid=<pid>) ... HPU worker rank=<r> ...`` (the log of
``$FARM_OUT/server0.log``), or else from the processes whose command line names ``Worker_TP<r>``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

LINE = re.compile(r"\(Worker(?:_\w+)? pid=(\d+)\).*HPU worker rank=(\d+) ")
NAME = re.compile(r"Worker_TP(\d+)")


def sibling(cpu: int, rule: str, n_machine: int) -> int | None:
    """The vCPU that shares ``cpu``'s host core under ``rule``; None when the rule is unknown."""
    half = n_machine // 2
    if rule == "a+N/2":
        return cpu + half if cpu < half else cpu - half
    if rule == "a^1":
        return cpu ^ 1
    return None


def shares(cpus, count: int) -> list[list[int]]:
    """``count`` equal shares of the sorted CPUs (the leftover CPUs stay with nobody), as VLLM_HPU_WORKER_CPU_PIN=1."""
    cpus = sorted(set(cpus))
    if count < 1 or len(cpus) < count:
        return []
    size = len(cpus) // count
    return [cpus[i * size:(i + 1) * size] for i in range(count)]


def plan(ranks, cpus, rule: str, n_machine: int) -> dict:
    """Main vCPU per rank, the reserved vCPUs (mains and their siblings) and each rank's helper vCPUs.

    ``ranks``: the ranks in order (rank r takes share r). A main vCPU is never another main's sibling and its own
    sibling is never another rank's main.
    """
    ranks = sorted(ranks)
    parts = shares(cpus, len(ranks))
    if not parts:
        return {}
    reserved: set[int] = set()
    main: dict[int, int] = {}
    for rank, part in zip(ranks, parts):
        for cpu in part[1:] + part[:1]:  # the share's first CPU last (it takes the share's interrupts first)
            sib = sibling(cpu, rule, n_machine)
            if cpu in reserved or (sib is not None and sib in reserved):
                continue
            main[rank] = cpu
            reserved.add(cpu)
            if sib is not None:
                reserved.add(sib)
            break
    helpers = {rank: [c for c in part if c not in reserved] for rank, part in zip(ranks, parts)}
    return {"main": main, "reserved": sorted(reserved), "helpers": helpers}


def find_workers(log_path: str, proc: str = "/proc") -> dict[int, int]:
    """{rank: pid} from the server log, else from the process names."""
    found: dict[int, int] = {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LINE.search(line)
                if m:
                    found[int(m.group(2))] = int(m.group(1))
    except OSError:
        pass
    alive = {rank: pid for rank, pid in found.items() if os.path.exists(os.path.join(proc, str(pid)))}
    if alive:
        return alive
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc, entry, "cmdline"), "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        m = NAME.search(cmd)
        if m:
            found[int(m.group(1))] = int(entry)
    return found


def thread_ids(pid: int, proc: str = "/proc") -> list[int]:
    try:
        return [int(t) for t in os.listdir(os.path.join(proc, str(pid), "task"))]
    except OSError:
        return []


def all_pids(proc: str = "/proc") -> list[int]:
    return [int(e) for e in os.listdir(proc) if e.isdigit()]


def apply(the_plan: dict, workers: dict[int, int], *, setaffinity=None, getaffinity=None, tids=thread_ids,
          pids=all_pids, allowed=None) -> dict:
    """Puts every thread where the plan says; returns counts (threads moved, failures)."""
    setaffinity = setaffinity or os.sched_setaffinity
    getaffinity = getaffinity or os.sched_getaffinity
    main, helpers, reserved = the_plan["main"], the_plan["helpers"], set(the_plan["reserved"])
    allowed = set(allowed if allowed is not None else os.sched_getaffinity(0)) | reserved
    moved = failed = 0
    worker_pids = set(workers.values())

    def put(tid: int, target: set[int]) -> None:
        nonlocal moved, failed
        try:
            if set(getaffinity(tid)) != target:
                setaffinity(tid, target)
                moved += 1
        except OSError:
            failed += 1

    for rank, pid in workers.items():
        if rank not in main:
            continue
        for tid in tids(pid):
            put(tid, {main[rank]} if tid == pid else set(helpers[rank]))
    free = allowed - reserved
    for pid in pids():
        if pid in worker_pids or pid == os.getpid():
            continue
        for tid in tids(pid):
            try:
                cur = set(getaffinity(tid))
            except OSError:
                continue
            if cur & reserved and cur - reserved:
                put(tid, cur - reserved)
            elif cur & reserved:
                put(tid, free)
    return {"moved": moved, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default=os.path.join(os.environ.get("FARM_OUT", "."), "server0.log"))
    ap.add_argument("--rule", default=os.environ.get("PLACE_SIBLING_RULE", ""))
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--wait-workers", type=float, default=120.0, help="seconds to wait for all workers in the log")
    ap.add_argument("--foreground", action="store_true", help="keep re-applying in this process (default: fork)")
    args = ap.parse_args()

    n_machine = os.cpu_count() or 0
    cpus = sorted(os.sched_getaffinity(0))
    deadline = time.time() + args.wait_workers
    workers = find_workers(args.log)
    while time.time() < deadline and len(workers) < 2:
        time.sleep(2)
        workers = find_workers(args.log)
    the_plan = plan(workers.keys(), cpus, args.rule, n_machine)
    if not the_plan:
        print("place_workers: no plan (workers %s, %d CPUs): nothing placed" % (workers, len(cpus)), flush=True)
        return 0
    print("place_workers: rule %r, %d CPUs of %d, workers %s" % (args.rule, len(cpus), n_machine, workers), flush=True)
    for rank in sorted(the_plan["main"]):
        sib = sibling(the_plan["main"][rank], args.rule, n_machine)
        print("place_workers: rank %d pid %d main thread on CPU %d (sibling %s kept free), %d helper CPUs" % (
            rank, workers[rank], the_plan["main"][rank], sib, len(the_plan["helpers"][rank])), flush=True)
    print("place_workers: first pass %s" % apply(the_plan, workers, allowed=cpus), flush=True)
    if not args.foreground and os.fork() > 0:
        return 0  # the caller goes on; the child keeps the placement
    if not args.foreground:
        os.setsid()
    end = time.time() + args.max_hours * 3600
    passes = 0
    while time.time() < end and any(os.path.exists("/proc/%d" % pid) for pid in workers.values()):
        time.sleep(args.every)
        counts = apply(the_plan, workers, allowed=cpus)
        passes += 1
        if counts["moved"]:
            print("place_workers: pass %d %s" % (passes, counts), flush=True)
    print("place_workers: done after %d passes" % passes, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
