#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device time per model component of replayed decode steps, from an HPU torch-profiler trace (step-anatomy).

``python3 -I trace_components.py <trace.json[.gz]>... [--json OUT] [--no-assign]`` reads the Chrome traces that an
HPU worker writes with ``--profiler-config '{"profiler": "torch", ...}'`` and ``HABANA_PROFILE=profile_api_light``
(``anatomy_job.sh`` plans M and P). Device kernels carry the graph's module path in ``args.EventName``
(``id_<n>_model/<layer>/linear_attn/... Unique ID:...``); HCL collective DMA runs on the "(HCL dedicated)" engines.
Per decode step (the ``step_timing:step`` ranges of the step timer) it prints:

* device time per component, exclusive: a sweep line over all engines, time where several components run at once
  split equally between them, so the components add up to the device busy time;
* engine time per component and engine class (TPC, MME, DMA, HCL), summed over the engines (work, not wall time);
* the compute engines' idle gaps: those an HCL event overlaps (the compute engines wait for an all-reduce: its
  transfer, its peers, and at one stream the host launch of the next graph) and the others, by size; gaps under
  200 us go to the component whose node ran just before (its per-node overhead);
* device nodes per component: events of one node name that overlap in time are one execution (a TPC node runs on
  many engines at once); layers share recipes, so a name recurs once per layer;
* host: graph launches (``ExecuteCachedGraph``) per step, their host time and spread, the device-to-host read-back.

Components from the module path: ``linear_attn`` -> gdn; ``self_attn/sparse_attn`` -> qsa_attention;
``self_attn/{qkv_proj,o_proj,index_qk_proj,bmm,rotary_emb,...}`` -> qsa_proj; other ``self_attn`` ops ->
qsa_indexer; ``hpu__mixture_of_experts`` and ``MOE/layer_*`` -> moe_experts (the vendor op); other ``moe_op`` ops ->
moe_prep; ``_shared_experts`` -> moe_shared; other ``mlp`` ops -> moe_router; hyper-connection kernels and mixing
weights -> hc; ``ple`` -> ple; ``logits_processor``/``lm_head`` -> logits; ``embed_tokens`` -> embed. Events without
a module path (fused TPC clusters, memsets) take the component of the closest preceding attributed event
(``--no-assign`` keeps them apart as "unattributed"). Standard library only.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

QSA_PROJ = ("qkv_proj", "o_proj", "index_qk_proj", "bmm", "rotary_emb", "q_norm", "k_norm", "index_k_norm", "q_proj",
            "k_proj", "v_proj", "kv_proj", "index_weights", "weights_proj", "q_a_proj", "q_b_proj")
ORDER = ("hc", "gdn", "qsa_indexer", "qsa_attention", "qsa_proj", "moe_router", "moe_prep", "moe_experts", "moe_shared",
         "ple", "embed", "logits", "sampler", "draft_hc", "draft_gdn", "draft_qsa_indexer", "draft_qsa_attention",
         "draft_qsa_proj", "draft_moe_router", "draft_moe_prep", "draft_moe_experts", "draft_moe_shared", "draft_ple",
         "draft_layer_other", "draft_other", "layer_other", "unattributed", "collective")
SMALL_GAP_US = 200.0
NODE_JOIN_US = 1.0  # events of one node name closer than this are one execution on several engines


def component(path: str) -> str | None:
    """The model component of a device event's module path (None: no path). The MTP draft head's modules have
    paths that start with its layer number (``48/self_attn/...``): they get the component of the same path under the
    target with a ``draft_`` prefix."""
    p = path
    if not p:
        return None
    m = re.match(r"(\d+)/(.*)$", p)
    if m:
        sub = component("model/0/" + m.group(2))
        return "draft_" + (sub or "other")
    if p.startswith("sampler") or "/sampler" in p:
        return "sampler"
    if "logits_processor" in p or "lm_head" in p:
        return "logits"
    if "embed_tokens" in p:
        return "embed"
    if "/ple" in p:
        return "ple"
    if "linear_attn" in p:
        return "gdn"
    if p.startswith("MOE/"):
        # the vendor op's chunk events name the module path after "id_<n>_": a draft layer's starts with its number
        return "draft_moe_experts" if re.search(r"/id_\d+_\d+/", p) else "moe_experts"
    if "hpu__mixture_of_experts" in p:
        return "moe_experts"
    if "moe_op" in p:
        return "moe_prep"
    if "_shared_experts" in p:
        return "moe_shared"
    if "/mlp" in p:
        return "moe_router"
    if "self_attn/sparse_attn" in p:
        return "qsa_attention"
    if "self_attn" in p:
        parts = p.split("/")
        i = parts.index("self_attn") if "self_attn" in parts else -1
        sub = parts[i + 1] if 0 <= i < len(parts) - 2 else ""
        return "qsa_proj" if sub in QSA_PROJ else "qsa_indexer"
    if "hc_" in p or "input_mix" in p or "output_mix" in p or "block_inject" in p:
        return "hc"
    if p.startswith("model"):
        return "layer_other"
    return None


def event_path(ev: dict) -> str:
    """The module path of a device event, from its ``EventName``."""
    en = (ev.get("args") or {}).get("EventName", "") or ""
    if en.startswith("MOE/"):
        return en
    m = re.search(r"id_\d+_(.*?)(?: Unique ID|$)", en)
    if m:
        return m.group(1)
    return ""


def engine_class(thread: str) -> str:
    t = thread.lower()
    for word, cls in (("hcl", "HCL"), ("mme", "MME"), ("tpc", "TPC"), ("dma", "DMA")):
        if word in t:
            return cls
    return "OTHER"


def pct(values: list, q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def load(path: str) -> list:
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as fh:
        data = json.load(fh)
    return data["traceEvents"] if isinstance(data, dict) else data


def analyze(events: list, assign: bool = True, name: str = "") -> dict:
    procs, threads = {}, {}
    for ev in events:
        if ev.get("ph") == "M":
            a = ev.get("args") or {}
            if ev.get("name") == "process_name":
                procs[ev.get("pid")] = str(a.get("name", ""))
            elif ev.get("name") == "thread_name":
                threads[(ev.get("pid"), ev.get("tid"))] = str(a.get("name", ""))
    dev_pids = {pid for pid, n in procs.items() if "/device:" in n}
    dev, host = [], []
    for ev in events:
        if ev.get("ph") == "X":
            (dev if ev.get("pid") in dev_pids else host).append(ev)
    steps = sorted((e for e in host if e.get("name") == "step_timing:step"), key=lambda e: e["ts"])
    n = max(1, len(steps))
    launches = sorted((e for e in host if e.get("name") == "ExecuteCachedGraph"), key=lambda e: e["ts"])
    d2h = [e for e in host if e.get("name") == "copy_hpu_lazy_D2H"]
    recs = []
    for ev in dev:
        cls = engine_class(threads.get((ev.get("pid"), ev.get("tid")), ""))
        start = float(ev["ts"])
        comp = "collective" if cls == "HCL" else component(event_path(ev))
        key = (ev.get("args") or {}).get("EventName") or ev.get("name", "")
        recs.append([start, start + float(ev.get("dur", 0.0)), comp, cls, key])
    recs.sort(key=lambda r: r[0])
    last, n_unattr, unattr = "unattributed", 0, 0.0
    for r in recs:
        if r[2] is None:
            r[2] = last if assign else "unattributed"
            n_unattr += 1
            unattr += r[1] - r[0]
        elif r[2] != "collective":
            last = r[2]
    # exclusive device time: sweep line over every engine
    bounds = []
    for r in recs:
        if r[1] > r[0]:
            bounds += [(r[0], 1, r[2]), (r[1], -1, r[2])]
    bounds.sort(key=lambda b: (b[0], b[1]))
    active, excl, busy, prev = Counter(), Counter(), 0.0, None
    for t, kind, comp in bounds:
        if prev is not None and t > prev:
            comps = [c for c, k in active.items() if k > 0]
            if comps:
                busy += t - prev
                for c in comps:
                    excl[c] += (t - prev) / len(comps)
        active[comp] += kind
        prev = t
    work = defaultdict(Counter)
    for r in recs:
        work[r[2]][r[3]] += r[1] - r[0]
    nodes = Counter()
    by_key = defaultdict(list)
    for r in recs:
        if r[3] != "HCL":
            by_key[r[4]].append(r)
    for rows in by_key.values():
        end = None
        for r in sorted(rows, key=lambda x: x[0]):
            if end is None or r[0] > end + NODE_JOIN_US:
                nodes[r[2]] += 1
                end = r[1]
            else:
                end = max(end, r[1])
    # idle gaps of the compute engines (all but HCL)
    merged = []
    for r in recs:
        if r[3] == "HCL":
            continue
        if merged and r[0] <= merged[-1][1]:
            if r[1] > merged[-1][1]:
                merged[-1][1], merged[-1][2] = r[1], r[2]
        else:
            merged.append([r[0], r[1], r[2]])
    hcl = sorted((r[0], r[1]) for r in recs if r[3] == "HCL")
    hcl_starts = [h[0] for h in hcl]
    coll_gap, coll_n, small_gap, other_hist = Counter(), Counter(), Counter(), Counter()
    for a, b in zip(merged, merged[1:]):
        g0, g1 = a[1], b[0]
        if g1 <= g0:
            continue
        hit = False
        for j in range(max(0, bisect.bisect_left(hcl_starts, g0) - 1), len(hcl)):
            if hcl[j][0] >= g1:
                break
            if hcl[j][1] > g0:
                hit = True
                break
        g = g1 - g0
        if hit:
            coll_gap[a[2]] += g
            coll_n[a[2]] += 1
            continue
        other_hist["<10us" if g < 10 else "10-50us" if g < 50 else "50-200us" if g < 200 else ">=200us"] += g
        if g < SMALL_GAP_US:
            small_gap[a[2]] += g
    durs = [e["dur"] for e in launches]
    per = 1000.0 * n
    return {
        "trace": name,
        "steps": len(steps),
        "step_period_ms": (steps[-1]["ts"] + steps[-1]["dur"] - steps[0]["ts"]) / per if steps else None,
        "busy_ms": busy / per,
        "exclusive_ms": {c: v / per for c, v in excl.items()},
        "engine_ms": {c: {k: v / per for k, v in w.items()} for c, w in work.items()},
        "nodes": {c: v / n for c, v in nodes.items()},
        "unattributed_events": n_unattr / n,
        "unattributed_engine_ms": unattr / per,
        "compute_busy_ms": sum(m[1] - m[0] for m in merged) / per,
        "collective_gap_ms": {c: v / per for c, v in coll_gap.items()},
        "collective_gaps": {c: v / n for c, v in coll_n.items()},
        "small_gap_ms": {c: v / per for c, v in small_gap.items()},
        "other_gap_ms": {k: v / per for k, v in other_hist.items()},
        "launches": len(launches) / n,
        "launch_ms": sum(durs) / per,
        "launch_us_p10_p50_p90_max": [pct(durs, 0.1), pct(durs, 0.5), pct(durs, 0.9), max(durs) if durs else 0.0],
        "d2h_ms": sum(e["dur"] for e in d2h) / per,
    }


def report(res: dict) -> str:
    ex, sg, cg = res["exclusive_ms"], res["small_gap_ms"], res["collective_gap_ms"]
    names = [c for c in ORDER if c in ex] + sorted(set(ex) - set(ORDER))
    out = ["== %s" % res["trace"],
           "%d steps, period %.2f ms; device busy %.2f ms per step; host: %.1f graph launches per step, %.2f ms "
           "(us p10/p50/p90/max %s), device-to-host read-back %.2f ms; events without a module path %.0f per step "
           "(%.2f engine-ms)" % (res["steps"], res["step_period_ms"] or 0.0, res["busy_ms"], res["launches"],
                                 res["launch_ms"], "/".join("%.0f" % x for x in res["launch_us_p10_p50_p90_max"]),
                                 res["d2h_ms"], res["unattributed_events"], res["unattributed_engine_ms"]),
           "| component | device ms (exclusive) | + idle gaps < 200 us after it | + all-reduce gaps after it | "
           "device nodes | TPC engine-ms | MME engine-ms | DMA engine-ms | HCL engine-ms |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in names:
        w = res["engine_ms"].get(c, {})
        out.append("| %s | %.2f | %.2f | %.2f | %.0f | %.2f | %.2f | %.2f | %.2f |" % (
            c, ex.get(c, 0.0), sg.get(c, 0.0), cg.get(c, 0.0), res["nodes"].get(c, 0.0), w.get("TPC", 0.0),
            w.get("MME", 0.0), w.get("DMA", 0.0), w.get("HCL", 0.0)))
    out.append("| sum | %.2f | %.2f | %.2f | %.0f | | | | |" % (sum(ex.values()), sum(sg.values()), sum(cg.values()),
                                                         sum(res["nodes"].values())))
    out.append("compute engines busy %.2f ms; idle under all-reduces %.2f ms in %.0f gaps; other idle %s" % (
        res["compute_busy_ms"], sum(cg.values()), sum(res["collective_gaps"].values()),
        ", ".join("%s %.2f ms" % kv for kv in sorted(res["other_gap_ms"].items()))))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-assign", action="store_true", help="keep events without a module path apart")
    a = ap.parse_args(argv)
    results = []
    for path in a.trace:
        res = analyze(load(path), assign=not a.no_assign, name=path)
        results.append(res)
        print(report(res))
        sys.stdout.flush()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
