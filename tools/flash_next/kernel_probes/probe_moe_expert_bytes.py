#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe (idea G2-1, B9/I33): does torch.ops.hpu.mixture_of_experts read every local expert every call?

    python3 probe_moe_expert_bytes.py --out DIR            (timing; one card; about 3 minutes)
    GRAPH_VISUALIZATION=1 python3 probe_moe_expert_bytes.py --out DIR --static      (post-graph dump and count)
    python3 -I probe_moe_expert_bytes.py --parse DIR       (parse a dump again, offline, standard library only)

Flash-Next per rank under EP8: 64 local FP8 experts (global ids 0..63 of 512 on rank 0), H 2560, I 640, top 10,
the keyword arguments of VllmMixtureOfExpertsOpFP8PerChannel.forward (vllm_gaudi/extension/ops.py) with per-channel
weight scales [2I] and [H].

Timing (fixes the two faults the skeptic found in reports/idea-sweep/evidence-lens/moe_expert_bytes_probe.py):
  * K calls of the op in ONE lazy graph, each on its own pre-quantized input (no common subexpression, nothing
    dead), timed as issue (Python), mark (mark_step) and wait (synchronize). Device time per call is bounded by
    (mark + wait) / K: the device starts at the mark_step, so the host issue time is outside it.
  * the routing ids stay inside the 512 global experts: remote ids are 64 + 48 j.
Routing arms at a fixed row count: none (0 local experts hit), one (1), ten (the same 10 for every row), all
(rows spread over the 64). One expert is 4,915,200 B of FP8; 64 are 314.6 MB, 128 us at 2.46 TB/s.

Verdict line "VERDICT expert-bytes: ...": DENSE when 'one' at 1 row costs at least 60 % of 'all' at 128 rows and
at least 60 us per call; SPARSE when 'one' at 1 row is under 25 % of 'all' at 128 rows; otherwise MIXED.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

H, I, LOCAL, GLOBAL, TOPK = 2560, 640, 64, 512, 10
EXPERT_BYTES = 2 * I * H + H * I  # FP8: w12 [2I, H] + w3 [H, I]
ROWS = (1, 4, 32, 128)
MODES = ("none", "one", "ten", "all")
REMOTE = [64 + 48 * j for j in range(TOPK)]  # 64..496: other ranks' experts, inside the 512 global ids


def say(text):
    print(f"PROBE expert-bytes {text}", flush=True)


def ids_for(torch, rows, mode):
    if mode == "none":
        table = [REMOTE] * rows
    elif mode == "one":
        table = [[0] + REMOTE[1:]] * rows
    elif mode == "ten":
        table = [list(range(TOPK))] * rows
    elif mode == "all":
        table = [[(r * TOPK + j) % LOCAL for j in range(TOPK)] for r in range(rows)]
    else:
        raise ValueError(mode)
    ids = torch.tensor(table, dtype=torch.int64)
    local = len({int(v) for v in ids.flatten() if 0 <= int(v) < LOCAL})  # distinct local experts hit
    return ids, local


def setup(torch, dev, seed=7):
    g = torch.Generator().manual_seed(seed)
    fp8 = torch.float8_e4m3fn
    w12 = [(torch.randn(2 * I, H, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(LOCAL)]
    w3 = [(torch.randn(H, I, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(LOCAL)]
    s12 = [torch.full((2 * I, ), 1e-3, dtype=torch.float32).to(dev) for _ in range(LOCAL)]
    s3 = [torch.full((H, ), 1e-3, dtype=torch.float32).to(dev) for _ in range(LOCAL)]
    return g, (tuple(w12), tuple(w3), tuple(s12), tuple(s3))


def quantized(torch, g, rows, dev):
    """A bf16 input quantized per row on the card (outside the timed graphs): (x_fp8, scale [rows, 1] fp32)."""
    x = torch.randn(rows, H, generator=g).to(torch.bfloat16).to(dev)
    scale = ((x.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / 240.0)
    x8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    return x8, scale


def moe(torch, weights, x8, scale, ids, router_weights):
    w12, w3, s12, s3 = weights
    return torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids, router_weights=router_weights,
                                            w12=w12, w3=w3, d_scale_hidden_states=scale, d_scale_w12=s12,
                                            d_scale_w3=s3, permuted_weights=True, activation="silu", experts_min=0,
                                            experts_max=LOCAL - 1)


DEVICE = os.environ.get("PROBE_DEVICE", "hpu")


class _CpuCore:
    """PROBE_DEVICE=cpu: the probe's logic on a CPU with the tests' stand-ins (times mean nothing)."""

    @staticmethod
    def mark_step():
        pass


def backend():
    import torch

    if DEVICE == "cpu":
        from tests.unit_tests.models.flash_next import moe_block

        moe_block.install_vendor_ops(fp8_max=240.0)
        torch.hpu = type("hpu", (), {"synchronize": staticmethod(lambda: None)})
        return torch, _CpuCore, "cpu"
    import habana_frameworks.torch.core as htcore

    return torch, htcore, "hpu"


def timing(out, iters, calls_small, calls_large):
    torch, htcore, dev = backend()
    g, weights = setup(torch, dev)
    htcore.mark_step()
    torch.hpu.synchronize()
    results = {}
    for rows in ROWS:
        calls = calls_small if rows <= 32 else calls_large
        inputs = [quantized(torch, g, rows, dev) for _ in range(calls)]
        router_weights = torch.full((rows, TOPK), 0.1, dtype=torch.bfloat16).to(dev)
        htcore.mark_step()
        torch.hpu.synchronize()
        for mode in MODES:
            if rows == 1 and mode == "all":
                continue  # the same 10 experts as 'ten' at one row
            ids, local = ids_for(torch, rows, mode)
            ids = ids.to(dev)
            htcore.mark_step()
            torch.hpu.synchronize()
            samples = []
            for it in range(iters + 2):  # two untimed: compile, then the first replay of the recipe
                t0 = time.perf_counter()
                outs = [moe(torch, weights, x8, s, ids, router_weights) for x8, s in inputs]
                t1 = time.perf_counter()
                htcore.mark_step()
                t2 = time.perf_counter()
                torch.hpu.synchronize()
                t3 = time.perf_counter()
                if it >= 2:
                    samples.append((t1 - t0, t2 - t1, t3 - t2))
                del outs
            samples.sort(key=lambda s: s[1] + s[2])
            issue, mark, wait = samples[len(samples) // 2]
            per_call_us = (mark + wait) / calls * 1e6
            key = f"rows{rows}_{mode}"
            results[key] = dict(rows=rows, mode=mode, local_experts=local, calls_per_graph=calls,
                                issue_ms=issue * 1e3, mark_ms=mark * 1e3, wait_ms=wait * 1e3,
                                device_bound_us_per_call=per_call_us,
                                min_device_bound_us_per_call=min((s[1] + s[2]) for s in samples) / calls * 1e6,
                                implied_tbs_if_all_local_read=LOCAL * EXPERT_BYTES / (per_call_us * 1e-6) / 1e12,
                                implied_tbs_if_hit_read=(local * EXPERT_BYTES / (per_call_us * 1e-6) / 1e12)
                                if local else 0.0)
            say(f"rows {rows:4d} routing {mode:4s} ({local:2d} local experts hit): {per_call_us:8.1f} us per call "
                f"(K={calls}; issue {issue * 1e3:.2f} ms, mark {mark * 1e3:.2f} ms, wait {wait * 1e3:.2f} ms per graph)")
    (out / "timing.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


def verdict(results):
    one1 = results.get("rows1_one", {}).get("device_bound_us_per_call")
    all128 = results.get("rows128_all", {}).get("device_bound_us_per_call")
    none1 = results.get("rows1_none", {}).get("device_bound_us_per_call")
    ten1 = results.get("rows1_ten", {}).get("device_bound_us_per_call")
    if not one1 or not all128:
        return "INCOMPLETE: the decisive arms did not run"
    ratio = one1 / all128
    if ratio >= 0.6 and one1 >= 60:
        code = "DENSE: the op costs about the same with 1 local expert hit as with 64: it reads every local expert"
    elif ratio < 0.25:
        code = "SPARSE: the op's time follows the experts hit, so it skips the unhit ones (B9 is moot)"
    else:
        code = "MIXED: neither flat nor proportional"
    return (f"{code}. 1 row: none {none1:.1f} us, one {one1:.1f} us, ten {ten1:.1f} us; 128 rows all 64: "
            f"{all128:.1f} us; one/all {ratio:.2f}; dense floor 64 x 4.9152 MB = 314.6 MB = 128 us at 2.46 TB/s")


# ----------------------------------------------------------------------------------------------------------
# Static arm: the compiled post-graph (GRAPH_VISUALIZATION=1), the method of the fork's
# tools/audit_deepseek_v41_physical_nodes.py and tools/audit_deepseek_v41_sram.py.
# ----------------------------------------------------------------------------------------------------------
LOGICAL = {"Placeholder", "OutputTensor", "Reshape", "StaticReshape", "Slice", "Split", "Concatenate", "Squeeze",
           "ExpandDims", "Identity", "Flatten", "Broadcast", "LogicalTranspose", "LogicalBroadcast",
           "ReinterpretCast", "TransposedShape", "Reduction", "StridedView"}
_ATTR = re.compile(r'key: "((?:input|output)Tensor:\d+)"\s+value \{\s+s: "([^"]*)"')


def static(out):
    graphs = out / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    os.environ["PT_HPU_GRAPH_DUMP_PREFIX"] = str(graphs)
    torch, htcore, dev = backend()
    g, weights = setup(torch, dev)
    htcore.mark_step()
    for rows in (1, 128):
        x8, s = quantized(torch, g, rows, dev)
        router_weights = torch.full((rows, TOPK), 0.1, dtype=torch.bfloat16).to(dev)
        ids, _ = ids_for(torch, rows, "one")
        ids = ids.to(dev)
        htcore.mark_step()
        y = moe(torch, weights, x8, s, ids, router_weights)
        htcore.mark_step()
        torch.hpu.synchronize()
        say(f"static: rows {rows} graph run, output {tuple(y.shape)} {y.dtype}")
    return parse(out)


def tinfo(s):

    def grab(p):
        m = re.search(p, s)
        return m.group(1) if m else None

    shape = grab(r"Sizes = (\[[^\]]+\])")
    try:
        shape = json.loads(shape) if shape else None
    except ValueError:
        shape = None
    return dict(name=s.split("  |", 1)[0], shape=shape, bytes=int(grab(r"sizeInBytes = (\d+)") or 0),
                dtype=grab(r"\|\s+([a-z0-9]+)\s+\|"), loc=grab(r"location = in (\w+)"))


def is_expert_weight(t):
    dims = sorted(d for d in (t["shape"] or []) if d != 1)
    return (t["dtype"] or "").startswith(("hfloat8", "fp8", "f8")) and dims in (sorted([2 * I, H]), sorted([H, I]))


def parse(out):
    rows = []
    for path in sorted((out / "graphs").rglob("*PostGraph-symbol.pbtxt")):
        raw = path.read_text(errors="replace")
        phys, mme, weight_reads, weights_seen, ops, locs = 0, 0, 0, set(), {}, {}
        for block in re.split(r"(?m)^node \{", raw)[1:]:
            op = re.search(r'\n  op: "([^"]+)"', block)
            if not op or op.group(1) in LOGICAL:
                continue
            op = op.group(1)
            phys += 1
            ops[op] = ops.get(op, 0) + 1
            mme += ("gemm" in op.lower() or "mme" in op.lower() or "conv" in op.lower())
            for key, value in _ATTR.findall(block):
                if not key.startswith("inputTensor"):
                    continue
                t = tinfo(value)
                if is_expert_weight(t):
                    weight_reads += t["bytes"]
                    weights_seen.add(t["name"])
                    locs[t["loc"]] = locs.get(t["loc"], 0) + t["bytes"]
        rows.append(dict(graph=path.name, physical_nodes=phys, mme_nodes=mme,
                         fp8_expert_weight_bytes_bound=weight_reads, distinct_expert_weight_tensors=len(weights_seen),
                         equivalent_experts=round(weight_reads / EXPERT_BYTES, 2), bytes_by_location=locs,
                         ops=dict(sorted(ops.items(), key=lambda kv: -kv[1]))))
    (out / "graphs.json").write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        say("static: " + json.dumps({k: v for k, v in r.items() if k != "ops"}) + " ops " +
            json.dumps(dict(list(r["ops"].items())[:10])))
    if not rows:
        files = sorted(str(p.relative_to(out)) for p in (out / "graphs").rglob("*"))[:20]
        say(f"static: no *PostGraph-symbol.pbtxt under {out / 'graphs'} (files: {files})")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("moe_expert_bytes"))
    ap.add_argument("--static", action="store_true", help="dump and count the post-graphs instead of timing")
    ap.add_argument("--parse", type=Path, help="only parse an existing output folder")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--calls", type=int, default=32, help="op calls per lazy graph up to 32 rows")
    ap.add_argument("--calls-large", type=int, default=16, help="op calls per lazy graph at 128 rows")
    a = ap.parse_args()
    if a.parse:
        parse(a.parse)
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    if a.static:
        static(a.out)
        return 0
    results = timing(a.out, a.iters, a.calls, a.calls_large)
    line = verdict(results)
    (a.out / "verdict.txt").write_text(line + "\n")
    print("VERDICT expert-bytes: " + line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
