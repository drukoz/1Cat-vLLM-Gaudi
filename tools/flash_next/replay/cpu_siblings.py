#!/usr/bin/env python3
"""Which vCPUs of the node share a physical core (hyperthread siblings on the host).

The rented node is a VM that reports one thread per core, so the kernel and every CPU placement treat each vCPU as
a core of its own. If two vCPUs are in fact two hyperthreads of one host core, two busy threads on them slow each
other down by a third to a half. The decode workers' main threads are such busy threads (design document
docs/design/flash-next/native-replay.md, section 7.11: in each server two of the eight ranks run about 2x slower per
launch for long stretches, and which two changes with every server start).

The probe measures it directly: a small C loop of integer work, pinned to one vCPU, counts its iterations for a
fixed time, first alone and then together with the same loop on a second vCPU. A sibling pair shows a large drop on
both sides; two separate cores show almost none.

1. Baseline: every tested vCPU alone.
2. Hypotheses: for a sample of vCPUs, the partners a+N/2, a-N/2, a^1, a+1, a-1 (N = the number of vCPUs of the
   machine), each pair run together.
3. If one rule explains every tested vCPU, it is checked on every vCPU of the job's CPU set; otherwise a few vCPUs
   are paired with every other vCPU.
4. The pairs found are run again at the end: a hypervisor that does not pin its vCPUs to host threads gives no
   stable map.

Only the CPU is used. Run it inside a farm job that holds all cards (so that no other job's threads share the
CPUs), for example:

    bin/farm job native-replay cpu-siblings all 10 --no-model -- python3 tools/flash_next/replay/cpu_siblings.py

Writes ``cpu_siblings.json`` and the compiled loop under ``$FARM_OUT`` (or ``--out``); prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

LOOP_C = r"""
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* spin CPU START SECONDS: pin to CPU, wait until CLOCK_MONOTONIC reaches START, then run integer work for
   SECONDS and print the iterations per second. */
int main(int argc, char** argv) {
  if (argc != 4) {
    return 2;
  }
  int cpu = atoi(argv[1]);
  double start = atof(argv[2]);
  double seconds = atof(argv[3]);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    perror("sched_setaffinity");
    return 3;
  }
  while (now() < start) {
  }
  volatile uint64_t sink = 0;
  uint64_t x = 88172645463325252ULL ^ (uint64_t)cpu;
  uint64_t iters = 0;
  double end = start + seconds;
  double t = now();
  while (t < end) {
    for (int i = 0; i < 4096; ++i) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      x += (x >> 3) * 0x9E3779B97F4A7C15ULL;
    }
    iters += 4096;
    t = now();
  }
  sink = x;
  (void)sink;
  printf("%.1f\n", iters / (t - start));
  return 0;
}
"""


def build_loop(out_dir: str) -> str:
    """Compile the loop into ``out_dir/spin`` and return its path."""
    src = os.path.join(out_dir, "spin.c")
    exe = os.path.join(out_dir, "spin")
    with open(src, "w") as fh:
        fh.write(LOOP_C)
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("no C compiler in the image")
    subprocess.run([cc, "-O2", "-o", exe, src], check=True)
    return exe


def run_together(exe: str, cpus: list[int], seconds: float) -> dict[int, float]:
    """Run the loop on every CPU of ``cpus`` at the same time; iterations per second per CPU."""
    start = time.monotonic() + 0.15
    procs = {
        cpu: subprocess.Popen([exe, str(cpu), "%.6f" % start, "%.6f" % seconds], stdout=subprocess.PIPE, text=True)
        for cpu in cpus
    }
    rates = {}
    for cpu, proc in procs.items():
        out, _ = proc.communicate(timeout=seconds + 30)
        if proc.returncode != 0:
            raise RuntimeError("loop on CPU %d failed with %d" % (cpu, proc.returncode))
        rates[cpu] = float(out.strip())
    return rates


def candidate_rules(n_machine: int) -> dict[str, callable]:
    """Partner rules to test: the usual ways a host enumerates its hyperthreads."""
    half = n_machine // 2
    return {
        "a+N/2": lambda a: (a + half) % n_machine if a < half else (a - half),
        "a^1": lambda a: a ^ 1,
        "a+1": lambda a: a + 1,
        "a-1": lambda a: a - 1,
    }


def slowdown(base: float, paired: float) -> float:
    """Fraction of the baseline rate lost when paired (0: no effect)."""
    return max(0.0, 1.0 - paired / base) if base > 0 else 0.0


def infer_rule(pair_results: list[dict], threshold: float) -> tuple[str | None, dict[str, int]]:
    """The rule whose pairs all show a slowdown above ``threshold`` on both sides, if exactly one fits.

    ``pair_results``: dicts with ``rule``, ``a``, ``b``, ``slow_a``, ``slow_b``. Returns (rule or None, per rule the
    number of tested vCPUs that it explains).
    """
    explained: dict[str, int] = {}
    tested: dict[str, int] = {}
    for r in pair_results:
        tested[r["rule"]] = tested.get(r["rule"], 0) + 1
        if min(r["slow_a"], r["slow_b"]) >= threshold:
            explained[r["rule"]] = explained.get(r["rule"], 0) + 1
    full = [rule for rule, n in tested.items() if n > 0 and explained.get(rule, 0) == n]
    return (full[0] if len(full) == 1 else None), explained


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.environ.get("FARM_OUT", "."))
    ap.add_argument("--seconds", type=float, default=0.3, help="length of one measurement")
    ap.add_argument("--sample", type=int, default=12, help="vCPUs whose candidate partners are tested")
    ap.add_argument("--threshold", type=float, default=0.2, help="slowdown that marks a sibling pair")
    ap.add_argument("--scan", type=int, default=3, help="vCPUs paired with every other vCPU when no rule fits")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    cpus = sorted(os.sched_getaffinity(0))
    n_machine = os.cpu_count() or (max(cpus) + 1)
    report: dict = {"cpus": cpus, "n_machine": n_machine, "seconds": args.seconds, "threshold": args.threshold}
    for name in ("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "/proc/sys/kernel/numa_balancing"):
        try:
            with open(name) as fh:
                report[name] = fh.read().strip()
        except OSError:
            pass
    try:
        report["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        pass
    exe = build_loop(args.out)

    # 1. baselines of the sample and of every CPU a rule may pair with it
    step = max(1, len(cpus) // args.sample)
    sample = cpus[::step][: args.sample]
    rules = candidate_rules(n_machine)
    base: dict[int, float] = {}

    def baseline(cpu: int) -> float:
        if cpu not in base:
            base[cpu] = run_together(exe, [cpu], args.seconds)[cpu]
        return base[cpu]

    def pair(a: int, b: int, rule: str) -> dict:
        ba, bb = baseline(a), baseline(b)
        rates = run_together(exe, [a, b], args.seconds)
        return {"rule": rule, "a": a, "b": b, "slow_a": round(slowdown(ba, rates[a]), 3),
                "slow_b": round(slowdown(bb, rates[b]), 3)}

    # 2. candidate partners of the sample
    hyp = []
    for a in sample:
        for name, rule in rules.items():
            b = rule(a)
            if b in cpus and b != a:
                hyp.append(pair(a, b, name))
    report["hypotheses"] = hyp
    rule_name, explained = infer_rule(hyp, args.threshold)
    report["rule"] = rule_name
    report["explained"] = explained
    print("cpu_siblings: %d vCPUs in the job (machine %d); sample %s" % (len(cpus), n_machine, sample))
    print("cpu_siblings: candidate rules, tested vCPUs explained: %s; rule: %s" % (explained, rule_name))

    # 3. the rule on every vCPU, or a full scan of a few
    if rule_name is not None:
        rule = rules[rule_name]
        verify = []
        for a in cpus:
            b = rule(a)
            if b in cpus and a < b:
                verify.append(pair(a, b, rule_name))
        report["verify"] = verify
        weak = [r for r in verify if min(r["slow_a"], r["slow_b"]) < args.threshold]
        print("cpu_siblings: rule %s checked on %d pairs, %d below the threshold" % (rule_name, len(verify), len(weak)))
        found = [(r["a"], r["b"]) for r in verify if min(r["slow_a"], r["slow_b"]) >= args.threshold]
    else:
        scan = []
        for a in sample[: args.scan]:
            for b in cpus:
                if b != a:
                    scan.append(pair(a, b, "scan"))
        report["scan"] = scan
        found = sorted({(min(r["a"], r["b"]), max(r["a"], r["b"])) for r in scan
                        if min(r["slow_a"], r["slow_b"]) >= args.threshold})
        print("cpu_siblings: no rule; scan of %s found the pairs %s" % (sample[: args.scan], found))
    report["pairs"] = found

    # 4. stability: the pairs again
    again = [pair(a, b, "again") for a, b in found[:16]]
    report["again"] = again
    stable = sum(1 for r in again if min(r["slow_a"], r["slow_b"]) >= args.threshold)
    print("cpu_siblings: %d of %d pairs found again" % (stable, len(again)))
    slows = sorted(min(r["slow_a"], r["slow_b"]) for r in hyp)
    if slows:
        print("cpu_siblings: slowdowns of the candidate pairs: min %.2f median %.2f max %.2f" % (
            slows[0], slows[len(slows) // 2], slows[-1]))
    report["seconds_total"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "cpu_siblings.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("cpu_siblings: %.0f s, report %s" % (report["seconds_total"], os.path.join(args.out, "cpu_siblings.json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
