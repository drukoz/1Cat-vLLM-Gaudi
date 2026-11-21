#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe (G2-1 again, coordinator 2026-11-20 23:33 UTC): the vendor MoE op per layer, each call with its OWN
weight set, as in the model, and the op's chunk_size argument.

    python3 probe_moe_layers.py --out DIR [--layers 16] [--iters 6]
    PROBE_DEVICE=cpu python3 probe_moe_layers.py --out DIR --layers 2 --iters 1   (a dry run; times mean nothing)

Why: job 00041 timed K calls of `torch.ops.hpu.mixture_of_experts` that shared ONE weight set and found 64 to 78 us
per call at 1 to 32 rows. step-anatomy's device trace of the served graph (00113, TP4 + EP4) has the op at 677 us
per layer at 32 rows and 90 us at 1 row, 51 device nodes per layer at 1 row and 361 at 32 rows (expert GEMM chunks
with router stages per chunk). Here every one of the K calls in a lazy graph is a different "layer": its own
pre-quantized input and its own FP8 experts (distinct device memory; K x 128 x 4.9 MB = 10 GB at K = 16 and
TP4 geometry), so no call can reuse another's weights.

Per rank: 128 local experts at TP4 + EP4, 64 at TP8 + EP8 (global 512, top 10, H 2560, I 640, per-channel FP8
scales), experts_min 0 and experts_max local - 1 (or 2 and 4 equal sub-ranges, one call each, summed: what
MAX_EXPERTS_PER_SLICE does in the bf16 class; the FP8 per-channel class never slices), the keyword arguments of
VllmMixtureOfExpertsOpFP8PerChannel.forward (vllm_gaudi/extension/ops.py). Routing: `uniform` (each row 10 distinct
experts of the 512 drawn uniformly: about a quarter local at TP4, an eighth at TP8), `none` (no local expert),
`dense` (every row's 10 experts local, spread over all of them). chunk_size: the served model passes none
(VLLM_MOE_CHUNK unset: no kwargs); a value c passes chunk_size=c and total_experts=512, as _get_extra_kwargs does.

Device time per layer = (mark + wait) / K: the device starts at the mark_step, so the host issue time is outside.
Lines `PROBE moe-layers local L rows R routing M chunk C: X us per layer (...)`, then VERDICT lines.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

H, I, GLOBAL, TOPK = 2560, 640, 512, 10
EXPERT_BYTES = 2 * I * H + H * I  # FP8: w12 [2I, H] + w3 [H, I]
TRACE_US = {(128, 1): 90.0, (128, 32): 677.0}  # step-anatomy 00113, TP4 + EP4, the served graph
DEVICE = os.environ.get("PROBE_DEVICE", "hpu")


def say(text):
    print(f"PROBE moe-layers {text}", flush=True)


def backend():
    import torch

    if DEVICE == "cpu":
        from tests.unit_tests.models.flash_next import moe_block

        moe_block.install_vendor_ops(fp8_max=240.0)
        torch.hpu = type("hpu", (), {"synchronize": staticmethod(lambda: None)})

        class _Core:

            @staticmethod
            def mark_step():
                pass

        return torch, _Core, "cpu"
    import habana_frameworks.torch.core as htcore

    return torch, htcore, "hpu"


def build_layers(torch, htcore, dev, layers, local, pool=4):
    """``layers`` weight sets of ``local`` FP8 experts, each expert its own device tensor (clones of a small pool)."""
    g = torch.Generator().manual_seed(11)
    fp8 = torch.float8_e4m3fn
    base12 = [(torch.randn(2 * I, H, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(pool)]
    base3 = [(torch.randn(H, I, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(pool)]
    s12 = torch.full((2 * I, ), 1e-3, dtype=torch.float32).to(dev)
    s3 = torch.full((H, ), 1e-3, dtype=torch.float32).to(dev)
    sets = []
    for layer in range(layers):
        w12 = tuple(base12[(layer * local + e) % pool].clone() for e in range(local))
        w3 = tuple(base3[(layer * local + e + 1) % pool].clone() for e in range(local))
        sc12 = tuple(s12.clone() for _ in range(local))
        sc3 = tuple(s3.clone() for _ in range(local))
        htcore.mark_step()
        sets.append((w12, w3, sc12, sc3))
    return sets


def routing(torch, rows, mode, local, seed):
    """[rows, 10] int64 expert ids and the number of distinct local experts they hit."""
    g = torch.Generator().manual_seed(seed)
    if mode == "uniform":
        ids = torch.stack([torch.randperm(GLOBAL, generator=g)[:TOPK] for _ in range(rows)])
    elif mode == "none":
        ids = torch.stack([local + torch.randperm(GLOBAL - local, generator=g)[:TOPK] for _ in range(rows)])
    elif mode == "dense":
        ids = torch.tensor([[(r * TOPK + j) % local for j in range(TOPK)] for r in range(rows)], dtype=torch.int64)
    else:
        raise ValueError(mode)
    ids = ids.to(torch.int64)
    hit = len({int(v) for v in ids.flatten() if int(v) < local})
    return ids, hit


def quantized_inputs(torch, rows, count, dev, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        x = torch.randn(rows, H, generator=g).to(torch.bfloat16).to(dev)
        scale = (x.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / 240.0
        x8 = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
        out.append((x8, scale))
    return out


def moe(torch, weights, local, x8, scale, ids, router_weights, chunk, slices=1):
    """One layer's routed experts: one op call over the local range, or ``slices`` calls over equal sub-ranges
    summed (the bf16 VllmMixtureOfExpertsOp's moe_n_slice loop under MAX_EXPERTS_PER_SLICE; the FP8 per-channel
    class that Flash-Next runs never slices)."""
    w12, w3, s12, s3 = weights
    kwargs = {} if chunk is None else {"chunk_size": int(chunk), "total_experts": GLOBAL}
    per = local // slices
    out = None
    for k in range(slices):
        lo, hi = k * per, (k + 1) * per
        y = torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids, router_weights=router_weights,
                                             w12=w12[lo:hi], w3=w3[lo:hi], d_scale_hidden_states=scale,
                                             d_scale_w12=s12[lo:hi], d_scale_w3=s3[lo:hi], permuted_weights=True,
                                             activation="silu", experts_min=lo, experts_max=hi - 1, **kwargs)
        out = y if out is None else out + y
    return out


def replay_times(torch, dev, run, iters):
    """Wall time of ``iters`` replays of ``run`` captured in an HPU graph (as the server's decode runs), each one
    synchronized: a replay costs the host one launch, so its wall time is the device time of the K layers plus that
    launch. The lazy timing (mark + wait after issuing the K calls) misses device work that starts during a long
    issue: job 00132 got 6 TB/s "if read once", which no HBM read gives."""

    class Layers(torch.nn.Module):

        def forward(self, dummy):
            return run()

    module = Layers()
    if dev == "hpu":
        import habana_frameworks.torch as htorch

        module = htorch.hpu.wrap_in_hpu_graph(module)
    dummy = torch.zeros(1).to(dev)
    for _ in range(2):  # capture, then the first replay
        module(dummy)
        torch.hpu.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        module(dummy)
        torch.hpu.synchronize()
        samples.append(time.perf_counter() - t0)
    return sorted(samples)


def measure(torch, htcore, dev, sets, local, rows, mode, chunk, iters, results, slices=1, replay=False):
    key = f"local{local}_rows{rows}_{mode}_chunk{chunk}" + (f"_slices{slices}" if slices > 1 else "")
    layers = len(sets)
    try:
        ids, hit = routing(torch, rows, mode, local, seed=1000 + rows)
        ids = ids.to(dev)
        router_weights = torch.full((rows, TOPK), 0.1, dtype=torch.bfloat16).to(dev)
        inputs = quantized_inputs(torch, rows, layers, dev, seed=rows)
        htcore.mark_step()
        torch.hpu.synchronize()

        def run():
            return tuple(moe(torch, sets[layer], local, x8, s, ids, router_weights, chunk, slices)
                         for layer, (x8, s) in enumerate(inputs))

        if replay:
            walls = replay_times(torch, dev, run, iters)
            issue, mark, wait = 0.0, 0.0, walls[len(walls) // 2]
            us = wait / layers * 1e6
            best = walls[0] / layers * 1e6
        else:
            samples = []
            for it in range(iters + 2):  # two untimed: the compile, then the first run of the recipe
                t0 = time.perf_counter()
                outs = run()
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
            us = (mark + wait) / layers * 1e6
            best = min(s[1] + s[2] for s in samples) / layers * 1e6
        results[key] = dict(local=local, rows=rows, mode=mode, chunk=chunk, slices=slices, local_experts_hit=hit,
                            layers=layers,
                            us_per_layer=us, best_us_per_layer=best, issue_ms=issue * 1e3, mark_ms=mark * 1e3,
                            wait_ms=wait * 1e3, hit_bytes_tbs=hit * EXPERT_BYTES / (us * 1e-6) / 1e12 if us else 0.0)
        trace = TRACE_US.get((local, rows))
        say(f"local {local:3d} rows {rows:3d} routing {mode:7s} chunk {str(chunk):>5s}"
            f"{f' slices {slices}' if slices > 1 else ''}: {us:8.1f} us per layer "
            f"(best {best:.1f}; {hit:3d} local experts hit = {hit * EXPERT_BYTES / 1e6:6.1f} MB, "
            f"{results[key]['hit_bytes_tbs']:.2f} TB/s if read once; K={layers}; issue {issue * 1e3:.1f} ms, "
            f"mark {mark * 1e3:.2f} ms, wait {wait * 1e3:.2f} ms per graph)" +
            (f"; the served graph's trace: {trace:.0f} us" if trace and mode == "uniform" and chunk is None else ""))
    except Exception as error:  # noqa: BLE001 - a probe reports every failure
        results[key] = dict(local=local, rows=rows, mode=mode, chunk=chunk, slices=slices,
                            error=" ".join(repr(error).split())[:300])
        say(f"local {local:3d} rows {rows:3d} routing {mode:7s} chunk {str(chunk):>5s}"
            f"{f' slices {slices}' if slices > 1 else ''}: ERROR {results[key]['error']}")
        try:
            htcore.mark_step()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "moe_layers")))
    ap.add_argument("--layers", type=int, default=16, help="distinct weight sets = op calls per lazy graph")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--rows", default="1,8,32,128")
    ap.add_argument("--chunks", default="32,64,128,256,512,1024", help="chunk_size values at 32 and 128 rows")
    ap.add_argument("--served-locals", default="128,64", help="local expert counts of the served-form sweep")
    ap.add_argument("--chunk-locals", default="128", help="local expert counts of the chunk_size sweep")
    ap.add_argument("--chunk-rows", default="32,128", help="row counts of the chunk_size sweep")
    ap.add_argument("--skip-routing", action="store_true", help="leave out the none / dense routing arms")
    ap.add_argument("--skip-slices", action="store_true", help="leave out the 2- and 4-slice arms")
    ap.add_argument("--replay", action="store_true",
                    help="time HPU-graph replays of the K layers (wall per replay / K) instead of mark + wait")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    torch, htcore, dev = backend()
    rows_list = [int(r) for r in a.rows.split(",") if r]
    chunks = [int(c) for c in a.chunks.split(",") if c]
    t0 = time.perf_counter()
    sets = build_layers(torch, htcore, dev, a.layers, 128)
    torch.hpu.synchronize()
    say(f"device {dev}; {a.layers} weight sets of 128 experts built in {time.perf_counter() - t0:.1f} s "
        f"({a.layers * 128 * EXPERT_BYTES / 1e9:.1f} GB); timing: "
        f"{'HPU-graph replay, wall per replay / K' if a.replay else 'lazy, (mark + wait) / K'}")
    results = {}

    def save():
        (a.out / "moe_layers.json").write_text(json.dumps(results, indent=1) + "\n")

    # 1. The served form (no chunk_size) at TP4 (128 local experts) and TP8 (the first 64 of each set).
    served_locals = [int(v) for v in a.served_locals.split(",") if v]
    chunk_locals = [int(v) for v in a.chunk_locals.split(",") if v]
    chunk_rows = [int(v) for v in a.chunk_rows.split(",") if v]
    for local in served_locals:
        for rows in rows_list:
            measure(torch, htcore, dev, sets, local, rows, "uniform", None, a.iters, results, replay=a.replay)
            save()
    # 2. What the routing does at 32 rows, TP4: no local expert hit, and all of them.
    for mode in () if a.skip_routing else ("none", "dense"):
        measure(torch, htcore, dev, sets, 128, 32, mode, None, a.iters, results, replay=a.replay)
        save()
    # 3. chunk_size at 32 and 128 rows, TP4; then the expert range in 2 and 4 slices (64 and 32 experts per call).
    for local in chunk_locals:
        for rows in chunk_rows:
            for chunk in chunks:
                measure(torch, htcore, dev, sets, local, rows, "uniform", chunk, a.iters, results, replay=a.replay)
                save()
    for rows in () if a.skip_slices else (32, 128):
        if rows not in rows_list:
            continue
        for slices in (2, 4):
            measure(torch, htcore, dev, sets, 128, rows, "uniform", None, a.iters, results, slices=slices,
                    replay=a.replay)
            save()

    def us(local, rows, mode="uniform", chunk=None, slices=1):
        key = f"local{local}_rows{rows}_{mode}_chunk{chunk}" + (f"_slices{slices}" if slices > 1 else "")
        return results.get(key, {}).get("us_per_layer")

    def fmt(value):
        return f"{value:.1f}" if value is not None else "n/a"

    line1 = ("served form, us per layer (TP4 128 local / TP8 64 local): " +
             "; ".join(f"{r} rows {fmt(us(128, r))} / {fmt(us(64, r))}" for r in rows_list) +
             f"; the served graph's trace at TP4: 90 at 1 row, 677 at 32 rows")
    line2 = (f"routing at 32 rows, TP4: none {fmt(us(128, 32, 'none'))}, uniform {fmt(us(128, 32))}, "
             f"dense {fmt(us(128, 32, 'dense'))} us per layer")
    parts = []
    for local in chunk_locals:
        for rows in chunk_rows:
            timed = {c: us(local, rows, chunk=c) for c in [None] + chunks}
            timed = {c: v for c, v in timed.items() if v is not None}
            if timed:
                best = min(timed, key=timed.get)
                parts.append(f"{local} local, {rows} rows: " + ", ".join(f"{'none' if c is None else c} {v:.1f}"
                                                                        for c, v in timed.items()) +
                             f" -> best chunk_size {best} ({timed[best]:.1f} us)")
    line3 = "chunk_size (uniform), us per layer: " + ("; ".join(parts) or "n/a")
    line4 = ("expert range in slices (TP4, uniform), us per layer, 1 / 2 / 4 calls: " +
             "; ".join(f"{r} rows {fmt(us(128, r))} / {fmt(us(128, r, slices=2))} / {fmt(us(128, r, slices=4))}"
                       for r in (32, 128) if r in rows_list))
    for name, line in (("served", line1), ("routing", line2), ("chunk", line3), ("slices", line4)):
        print(f"VERDICT moe-layers {name}: {line}", flush=True)
    results["verdicts"] = [line1, line2, line3, line4]
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
