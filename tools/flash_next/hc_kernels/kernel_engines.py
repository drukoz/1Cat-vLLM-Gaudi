# SPDX-License-Identifier: Apache-2.0
"""kernel_engines.py TRACE.json[.gz] [NAME_SUBSTR ...]: per execution of each named TPC/MME kernel on the device, the
number of engines that ran it, the per-engine busy time and the wall span (docs/design/flash-next/hc-kernels.md).
Read-only analysis of a torch-profiler trace with HPU activities (HABANA_PROFILE=profile_api_light)."""
import gzip
import json
import sys
from collections import defaultdict
import statistics as st

path = sys.argv[1]
wanted = sys.argv[2:] or ["hc_fused_gate_mix", "hc_fused_combine_norm", "hc_fused_silu", "gdn_fused_core_verify"]
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as fh:
    data = json.load(fh)
events = data["traceEvents"]
procs, threads = {}, {}
for ev in events:
    if ev.get("ph") == "M":
        a = ev.get("args") or {}
        if ev.get("name") == "process_name":
            procs[ev.get("pid")] = str(a.get("name", ""))
        elif ev.get("name") == "thread_name":
            threads[(ev.get("pid"), ev.get("tid"))] = str(a.get("name", ""))
dev = {p for p, n in procs.items() if "/device:" in n}
n_steps = max(1, sum(1 for e in events if e.get("name") == "step_timing:step"))
by = defaultdict(list)
for ev in events:
    if ev.get("ph") != "X" or ev.get("pid") not in dev:
        continue
    nm = str(ev.get("name", ""))
    a = ev.get("args") or {}
    en = str(a.get("EventName", ""))
    for w in wanted:
        if w in nm or w in en:
            s = float(ev["ts"]); d = float(ev.get("dur", 0.0))
            by[w].append((s, s + d, threads.get((ev.get("pid"), ev.get("tid")), "?"), nm, en))
            break
print("steps", n_steps)
for w, lst in by.items():
    lst.sort()
    execs = []
    cur = None
    for s, e, th, nm, en in lst:
        if cur is None or s > cur["end"] + 1.0:
            cur = {"start": s, "end": e, "eng": defaultdict(float), "names": set()}
            execs.append(cur)
        cur["end"] = max(cur["end"], e)
        cur["eng"][th] += e - s
        cur["names"].add(nm)
    walls = [x["end"] - x["start"] for x in execs]
    neng = [len(x["eng"]) for x in execs]
    busy = [sum(x["eng"].values()) for x in execs]
    permax = [max(x["eng"].values()) for x in execs]
    permin = [min(x["eng"].values()) for x in execs]
    print(f"== {w}: {len(execs)} executions ({len(execs)/n_steps:.1f} per step), names {sorted(set().union(*[x['names'] for x in execs]))[:3]}")
    print(f"   wall us: median {st.median(walls):.1f} min {min(walls):.1f} max {max(walls):.1f}; sum per step {sum(walls)/n_steps/1000:.2f} ms")
    print(f"   engines: median {st.median(neng)} min {min(neng)} max {max(neng)}; engine-us per exec median {st.median(busy):.1f}")
    print(f"   per-engine busy us: max median {st.median(permax):.1f}, min median {st.median(permin):.1f}")
    # distribution of wall by engines
    hist = defaultdict(list)
    for x in execs:
        hist[len(x["eng"])].append(x["end"] - x["start"])
    for k in sorted(hist):
        print(f"   {k} engines: {len(hist[k])} execs, wall median {st.median(hist[k]):.1f} us")
    # sample engine names
    print("   engine names (first exec):", sorted(execs[0]["eng"].keys())[:30])
