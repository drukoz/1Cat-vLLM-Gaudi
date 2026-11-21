#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card bisect (qsa-fused for kernel-fixes, coordinator 2026-11-21 08:01 UTC): what cuts the served MoE layer
into several graphs.

    python3 moe_cut_probe.py --out DIR [--rows 4,32,128] [--arms s,a,b,c,d,e,f,g1,g2,g3,h] [--budget 540]

Job 00190's trace (build 4, TP8 rank 0, a 32-stream MTP step): every MoE layer is "MoE pre" (HC, router, shared
expert) | 1 to 4 tiny graphs under `mlp/experts/moe_op` (`hpu::index` f32 gathers, abs, min, max, ceil, ne, not;
under 20 us of device time each, about 92 us of host each, 90 per step) | the vendor op's own graph. The served op
(`VllmMixtureOfExpertsOpFP8PerChannel`) has no mark_step, so something in the vendor-op call path flushes the graph.

Each arm runs one MoE layer's path L times in a row (L = 1 and 2, the same weights, a residual chain), captured in an
HPU graph as the server's decode is (`wrap_in_hpu_graph`), then profiles one replay with the torch profiler (CPU +
HPU activities) and counts the graph launches (`ExecuteCachedGraph`). Launches per layer = n(L=2) - n(L=1).

Arms (shapes of one TP8 + EP8 rank: 64 local FP8 experts of 512, top 10, H 2560, I 640, per-channel scales):
  s   the served class: vllm_gaudi's VllmMixtureOfExpertsOpFP8PerChannel.forward (dynamic_quant, the int64 ids, the
      chunk kwargs of VLLM_MOE_CHUNK / VLLM_MOE_TOKEN_BOUNDARY), routing as the served apply (softmax, top-k, renorm),
      its experts and scales VIEWS of stacked [E, ...] tensors as in the served model
  v   c with the experts and scales as views of the stacked tensors (the served layout); a..g give owned tensors
  p2  c with one per-tensor power-of-two activation scale
  a   the vendor op alone on precomputed FP8 inputs and ids (one elementwise op before and after it)
  b   a + the row quantization in the graph (abs, amax, cast_to_fp8_v2: the ordinary dynamic_quant)
  c   b + the routing in the graph (softmax, top-k 10, renorm, ids .to(int64), weights .to(bf16))
  d   c without the chunk kwargs
  e   c with one per-tensor activation scale instead of the per-row scales
  f   c with bf16 experts (the bf16 vendor op, no scales)
  g1  c with chunk_size only (no total_experts)
  g2  c with total_experts only (no chunk_size)
  g3  c with total_experts = 64 (the local count)
  g4  c with chunk_size = the row count (one chunk)
  h   the dense form (kernel-fixes' moe_dense_fp8: two batched FP8 GEMMs over all local experts), routing as c
Lines: `MOECUT arm X rows R: L1 n1, L2 n2 launches, per layer P; recipes ...`, then the device nodes of every
launched recipe of the L=1 replay (when the trace has them), and a JSON file per arm in --out.
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

H, I, GLOBAL, TOPK, LOCAL = 2560, 640, 512, 10, 64
ARMS = "s,v,a,c,d,h,e,p2,g1,g2,g3,g4,b,f"


def say(text):
    print(f"MOECUT {text}", flush=True)


def served_chunk(rows):
    """The chunk kwargs of the served setting VLLM_MOE_CHUNK=0,64,8192 VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000."""
    chunks = [int(v) for v in os.environ.get("VLLM_MOE_CHUNK", "0,64,8192").split(",")]
    bounds = [int(v) for v in os.environ.get("VLLM_MOE_TOKEN_BOUNDARY", "16,512,1000000").split(",")]
    size = chunks[-1]
    for c, b in zip(chunks, bounds):
        if rows <= b:
            size = c
            break
    return {"chunk_size": size, "total_experts": GLOBAL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("FARM_OUT", "."))
    ap.add_argument("--rows", default="4,32,128")
    ap.add_argument("--arms", default=ARMS)
    ap.add_argument("--budget", type=float, default=470.0, help="seconds; arms not started by then are skipped")
    args = ap.parse_args()
    out = Path(args.out) / "moe_cut"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    import torch
    import torch.nn.functional as F
    import habana_frameworks.torch as htorch
    import habana_frameworks.torch.core as htcore
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device("hpu")
    fp8 = torch.float8_e4m3fn
    fp8_max = float(torch.finfo(fp8).max)
    try:
        from vllm_gaudi.extension.ops import FP8_MAX as served_fp8_max
        fp8_max = float(served_fp8_max)
    except Exception:  # noqa: BLE001
        pass
    say(f"torch {torch.__version__}, fp8 max {fp8_max}, PT_HPU_LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')}, "
        f"PT_HPU_LAZY_ACC_PAR_MODE={os.environ.get('PT_HPU_LAZY_ACC_PAR_MODE')}, "
        f"HABANA_PROFILE={os.environ.get('HABANA_PROFILE')}")

    g = torch.Generator().manual_seed(7)
    w12 = [(torch.randn(2 * I, H, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(LOCAL)]
    w3 = [(torch.randn(H, I, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(LOCAL)]
    s12 = [torch.full((2 * I, ), 1e-3, dtype=torch.float32).to(dev) for _ in range(LOCAL)]
    s3 = [torch.full((H, ), 1e-3, dtype=torch.float32).to(dev) for _ in range(LOCAL)]
    htcore.mark_step()
    # The served layout (fp8_block_moe_prepare_weights + _cache_weight_lists): stacked [E, 2I, H] / [E, H, I] experts
    # and [E, 2I] / [E, H] scales; the op gets per-expert VIEWS of them. Arms a..g give it owned per-expert tensors.
    stacked = (torch.stack(w12), torch.stack(w3), torch.stack([v.reshape(-1) for v in s12]),
               torch.stack([v.reshape(-1) for v in s3]))
    htcore.mark_step()
    vw12 = [stacked[0][e] for e in range(LOCAL)]
    vw3 = [stacked[1][e] for e in range(LOCAL)]
    vs12 = [stacked[2][e] for e in range(LOCAL)]
    vs3 = [stacked[3][e] for e in range(LOCAL)]
    bf16_w = None  # the bf16 arm's experts

    def quant_rows(x):
        scale = (x.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / fp8_max
        return torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / scale, False, False, fp8)[0], scale

    def quant_tensor(x):
        scale = (x.abs().amax().float() + 1e-8) / fp8_max
        return torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / scale, False, False, fp8)[0], scale

    def route(logits):
        w = F.softmax(logits.float(), dim=1, dtype=torch.float32)
        w, ids = torch.topk(w, TOPK, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        return w.to(torch.bfloat16), ids.to(torch.int64)

    def vendor(x8, scale, ids, rw, kwargs, views=False):
        a12, a3, b12, b3 = (vw12, vw3, vs12, vs3) if views else (w12, w3, s12, s3)
        return torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids, router_weights=rw, w12=a12,
                                                w3=a3, d_scale_hidden_states=scale, d_scale_w12=b12, d_scale_w3=b3,
                                                permuted_weights=True, activation="silu", experts_min=0,
                                                experts_max=LOCAL - 1, **kwargs)

    def quant_pow2(x):
        """One per-tensor power-of-two activation scale."""
        amax = x.abs().amax().float() + 1e-8
        scale = torch.exp2(torch.ceil(torch.log2(amax / fp8_max)))
        return torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / scale, False, False, fp8)[0], scale

    def dense(x, ids, rw):
        w13s, w2s, s13s, s2s = stacked
        rows = x.shape[0]
        local = ids.to(torch.int32)
        expert_range = torch.arange(LOCAL, dtype=torch.int32, device=x.device)
        routed = (local.unsqueeze(-1) == expert_range).to(torch.float32)
        pair_weight = (routed * rw.to(torch.float32).unsqueeze(-1)).sum(dim=1)
        x8, xs = quant_rows(x)
        a = x8.unsqueeze(0).expand(LOCAL, rows, H).contiguous()
        a_scale = xs.reshape(1, rows, 1).to(torch.float32).expand(LOCAL, rows, 1).contiguous()
        gate_up = torch.ops.hpu.fp8_gemm_v2(a, False, w13s, True, None, torch.bfloat16, a_scale, s13s.unsqueeze(1), None,
                                            False)
        mid = (F.silu(gate_up[..., :I].float()) * gate_up[..., I:].float()).to(x.dtype)
        m8, ms = quant_rows(mid.reshape(LOCAL * rows, I))
        ms = ms.reshape(LOCAL, rows, 1).to(torch.float32) * pair_weight.t().unsqueeze(-1)
        down = torch.ops.hpu.fp8_gemm_v2(m8.reshape(LOCAL, rows, I), False, w2s, True, None, torch.bfloat16, ms,
                                         s2s.unsqueeze(1), None, False)
        return down.sum(dim=0, dtype=torch.float32).to(x.dtype)

    served_op = None

    def make_served():
        nonlocal served_op
        if served_op is None:
            from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpFP8PerChannel
            op = VllmMixtureOfExpertsOpFP8PerChannel(GLOBAL, LOCAL, 0, LOCAL - 1)
            for e in range(LOCAL):
                op.w13_list[e].set_weight(vw12[e])
                op.w13_list[e].set_scale_inv_fp8(vs12[e])
                op.w2_list[e].set_weight(vw3[e])
                op.w2_list[e].set_scale_inv_fp8(vs3[e])
            if hasattr(op, "_cache_weight_lists"):
                op._cache_weight_lists()
            say(f"served class: chunk list {getattr(op, 'chunk_size_list', None)}, "
                f"boundaries {getattr(op, 'token_boundary_list', None)}, "
                f"kwargs at 4 / 32 / 128 rows {op._get_extra_kwargs(4)} / {op._get_extra_kwargs(32)} / "
                f"{op._get_extra_kwargs(128)}")
            served_op = op
        return served_op

    def layer_fn(arm, rows, inputs):
        """One MoE layer of arm ``arm``: acc -> acc + moe(acc)."""
        kw = served_chunk(rows)
        x8c, scc, idsc, rwc, logits = inputs

        if arm == "s":
            op = make_served()

            def f(acc):
                rw, ids = route(logits + acc[:, :GLOBAL] * 0)
                return acc + op(acc, ids, rw)
        elif arm == "a":

            def f(acc):
                y = acc * 0.5
                return y + vendor(x8c, scc, idsc, rwc, kw)
        elif arm == "b":

            def f(acc):
                x8, sc = quant_rows(acc)
                return acc + vendor(x8, sc, idsc, rwc, kw)
        elif arm == "v":

            def f(acc):
                rw, ids = route(logits + acc[:, :GLOBAL] * 0)
                x8, sc = quant_rows(acc)
                return acc + vendor(x8, sc, ids, rw, kw, views=True)
        elif arm == "p2":

            def f(acc):
                rw, ids = route(logits + acc[:, :GLOBAL] * 0)
                x8, sc = quant_pow2(acc)
                return acc + vendor(x8, sc, ids, rw, kw)
        elif arm in ("c", "d", "e", "g1", "g2", "g3", "g4"):
            kwargs = {
                "c": kw,
                "d": {},
                "e": kw,
                "g1": {"chunk_size": kw["chunk_size"]},
                "g2": {"total_experts": GLOBAL},
                "g3": {"chunk_size": kw["chunk_size"], "total_experts": LOCAL},
                "g4": {"chunk_size": rows, "total_experts": GLOBAL},
            }[arm]
            quant = quant_tensor if arm == "e" else quant_rows

            def f(acc):
                rw, ids = route(logits + acc[:, :GLOBAL] * 0)
                x8, sc = quant(acc)
                return acc + vendor(x8, sc, ids, rw, kwargs)
        elif arm == "f":

            def f(acc):
                rw, ids = route(logits + acc[:, :GLOBAL] * 0)
                return acc + torch.ops.hpu.mixture_of_experts(hidden_states=acc, expert_routing_table=ids,
                                                              router_weights=rw, w12=bf16_w[0], w3=bf16_w[1],
                                                              permuted_weights=True, activation="silu", experts_min=0,
                                                              experts_max=LOCAL - 1, **kw)
        elif arm == "h":

            def f(acc):
                rw, ids = route(logits + acc[:, :GLOBAL] * 0)
                return acc + dense(acc, ids, rw)
        else:
            raise ValueError(arm)
        return f

    class Layers(torch.nn.Module):

        def __init__(self, fn, n):
            super().__init__()
            self.fn, self.n = fn, n

        def forward(self, x):
            acc = x * 1.0
            for _ in range(self.n):
                acc = self.fn(acc)
            return acc * 1.0

    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "HPU"):
        activities.append(ProfilerActivity.HPU)

    def count_launches(module, x, trace_path):
        with profile(activities=activities) as prof:
            module(x)
            torch.hpu.synchronize()
        prof.export_chrome_trace(str(trace_path))
        d = json.load(open(trace_path))
        ev = [e for e in d.get("traceEvents", []) if e.get("ph") == "X"]
        launches = sorted((e for e in ev if e.get("name") == "ExecuteCachedGraph"), key=lambda e: e["ts"])
        enq = sorted((e for e in ev if str(e.get("name", "")).startswith("enqueueWithExternalEventsExt")),
                     key=lambda e: e["ts"])
        recipes = [str((e.get("args") or {}).get("recipeName", ""))[-26:] for e in enq]
        # device nodes per recipe handle (when HABANA_PROFILE gave device events)
        nodes = defaultdict(Counter)
        for e in ev:
            a = e.get("args") or {}
            h = a.get("recipeHandle")
            if h and h != "0" and "HW event name" in a:  # a device node's event (HABANA_PROFILE traces)
                n = str(a.get("EventName") or e.get("name", ""))
                n = n.split(" Unique ID")[0]
                if n.startswith("id_"):
                    n = n.split("_", 2)[-1]
                nodes[h][n[:110]] += 1
        order = []
        for e in enq:
            h = str((e.get("args") or {}).get("recipeHandle", ""))
            order.append((str((e.get("args") or {}).get("recipeName", ""))[-26:], h, sorted(nodes.get(h, {}))[:14]))
        host_us = [e.get("dur", 0) for e in launches]
        return len(launches), recipes, order, host_us

    rows_list = [int(r) for r in args.rows.split(",") if r]
    arms = [a for a in args.arms.split(",") if a]
    summary = {}
    for arm in arms:
        for rows in rows_list:
            if time.time() - t_start > args.budget:
                say(f"arm {arm} rows {rows}: SKIPPED (budget {args.budget:.0f} s spent)")
                continue
            key = f"{arm}_rows{rows}"
            res = {"arm": arm, "rows": rows}
            t0 = time.time()
            try:
                if arm == "f" and bf16_w is None:
                    bf16_w = ([w.to(torch.bfloat16) * 1e-3 for w in w12], [w.to(torch.bfloat16) * 1e-3 for w in w3])
                    htcore.mark_step()
                gi = torch.Generator().manual_seed(100 + rows)
                x = (torch.randn(rows, H, generator=gi) * 0.5).to(torch.bfloat16).to(dev)
                logits = torch.randn(rows, GLOBAL, generator=gi).to(torch.bfloat16).to(dev)
                ids = torch.stack([torch.randperm(GLOBAL, generator=gi)[:TOPK] for _ in range(rows)]).to(dev)
                rw = torch.full((rows, TOPK), 0.1, dtype=torch.bfloat16).to(dev)
                x8, sc = quant_rows(x)
                htcore.mark_step()
                torch.hpu.synchronize()
                fn = layer_fn(arm, rows, (x8, sc, ids, rw, logits))
                counts = {}
                for n_layers in (1, 2):
                    module = htorch.hpu.wrap_in_hpu_graph(Layers(fn, n_layers))
                    for _ in range(2):  # the capture, then the first replay
                        y = module(x)
                        torch.hpu.synchronize()
                    finite = bool(torch.isfinite(y.float()).all().item())
                    n, recipes, order, host_us = count_launches(module, x, out / f"{key}_L{n_layers}.json")
                    counts[n_layers] = n
                    res[f"L{n_layers}"] = {"launches": n, "recipes": recipes, "host_us": host_us, "finite": finite}
                    if n_layers == 1:
                        res["order_L1"] = order
                    del module, y
                    gc.collect()
                per_layer = counts[2] - counts[1]
                res["per_layer"] = per_layer
                res["seconds"] = round(time.time() - t0, 1)
                say(f"arm {arm:2s} rows {rows:3d}: L1 {counts[1]}, L2 {counts[2]} launches, per layer {per_layer}; "
                    f"host us L1 {[round(v) for v in res['L1']['host_us']]}; {res['seconds']} s")
                for name, h, nodes in res["order_L1"]:
                    say(f"    arm {arm} rows {rows} recipe {name} nodes {nodes}")
            except Exception as exc:  # noqa: BLE001
                res["error"] = f"{type(exc).__name__}: {exc}"[:600]
                say(f"arm {arm:2s} rows {rows:3d}: ERROR {res['error']}")
                traceback.print_exc()
            summary[key] = res
            json.dump(summary, open(out / "summary.json", "w"), indent=1, default=str)
    say("TABLE arm: per-layer launches at " + " / ".join(str(r) for r in rows_list) + " rows")
    for arm in arms:
        cells = []
        for rows in rows_list:
            r = summary.get(f"{arm}_rows{rows}", {})
            cells.append(str(r.get("per_layer", "ERR" if "error" in r else "-")))
        say(f"TABLE {arm:3s} " + " / ".join(cells))
    say(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
