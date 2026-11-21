#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe: MoE forms at TP8 geometry for the coordinator's 12:35 UTC plan. For each form it reports the
device time per layer, the graph launches per layer, and the error against an fp32 reference.

    python3 probe_moe_forms.py --out DIR [--big-rows 96,128] [--small-rows 1,3,4,8,16] [--layers 8]
    PROBE_DEVICE=cpu python3 probe_moe_forms.py --out DIR --big-rows 8 --small-rows 1,3 --layers 1 --local 8  (dry run)

Big rows (the verify rows of 32 streams):
  dense         the served dense form (vllm_gaudi/extension/ops.py moe_dense_fp8): [E, T, H] copy of x, two
                batched FP8 GEMMs, routing weight on the down GEMM's row scale, sum over E in fp32
  flat13        (a) gate / up as ONE 2-D GEMM over all experts: x8 [T, H] x w13 viewed [E*2I, H] (trans_B), the
                row scale x_scale [T, 1] and the column scale s13 [1, E*2I]: no [E, T, H] copy, one long N; then the
                served activation, quantization, batched down GEMM and sum (mid moved to [E, T, I])
  flat13-mmsum  flat13 with the sum over E as a matmul with a ones row ([1, E] x [E, T*H]; the c1 question: is the
                TPC sum the slow part)
  dense-triton  (b) via Gaudi Triton: the served batched gate / up, then the activation and the mid quantization as
                one fused node (vllm_gaudi/ops/triton_gaudi silu_and_mul_dynamic_quant; ERROR when the image has no
                Gaudi Triton backend)
  flat13-triton the same after the 2-D gate / up GEMM (rows (t, e); the FP8 mid then moved to [E, T, I])
  flat13-bf16act  flat13 with the activation in bf16 (silu(gate) * up on the bf16 GEMM output, no fp32 copies; the
                (b) question: how much of the glue is the fp32 path)
  flatdown      (c2, NOT exact, parity only) W2 requantized once to a per-channel scale shared by the experts, so the
                down projection is ONE 2-D GEMM with K = E*I (the routing weight folded into mid before its per-row
                quantization) and no sum
  vendor-chunk64  the vendor op with chunk_size 64 (the served vendor setting from 17 to 512 rows; reference)
Small rows (a step of 1 to 16 tokens):
  dense         as above (reads all 64 local experts)
  vendor        the vendor op with per-row activation scales (the fallback today; host checks from 2 rows up)
  vendor-t      the vendor op with ONE per-tensor activation scale (qsa-fused's VLLM_HPU_MOE_TENSOR_SCALE_MAX_TOKENS,
                25328606): one launch per layer, reads only the experts the rows hit

Device time per layer:
  - wall: the HPU-graph replay of K layers, each with its own experts, divided by K;
  - busy: with HABANA_PROFILE=profile_api_light, the union of the device events of one profiled replay of a chain of
    L = 1 and L = 2 layers, as busy(2) - busy(1).
Launches per layer: the ExecuteCachedGraph events of the same profiled replays, as n(2) - n(1) (qsa-fused's
moe_cut_probe method).
Accuracy: each form against an fp32 reference on the CPU (the dequantized FP8 experts, exact routing, x in fp32),
for three kinds of x: rows of one size, row sizes spread 0.1 to 10x, and one row at 100x. Reported: the relative L2
error per row with a local expert, mean and max.

Lines `PROBE moe-forms ...`, then VERDICT lines. Results in DIR/moe_forms.json.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_moe_dense as dp  # noqa: E402  (backend, layers, routing, vendor op, served dense form, replay timing)

H, I, GLOBAL, TOPK = dp.H, dp.I, dp.GLOBAL, dp.TOPK
FP8_MAX = dp.FP8_MAX


def say(text):
    print(f"PROBE moe-forms {text}", flush=True)


def quantize_tensor(torch, x):
    """One per-tensor FP8 scale (0-dim fp32) for the whole input."""
    scale = (x.abs().amax().float() + 1e-8) / FP8_MAX
    q = torch.ops.hpu.cast_to_fp8_v2(x, 1.0 / scale, False, False, torch.float8_e4m3fn)[0]
    return q, scale


def vendor_tensor_moe(torch, layer, local, x, ids, weights):
    w13, w2, s13, s2 = layer
    x8, scale = quantize_tensor(torch, x)
    return torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids, router_weights=weights,
                                            w12=tuple(w13[e] for e in range(local)),
                                            w3=tuple(w2[e] for e in range(local)), d_scale_hidden_states=scale,
                                            d_scale_w12=tuple(s13[e] for e in range(local)),
                                            d_scale_w3=tuple(s2[e] for e in range(local)), permuted_weights=True,
                                            activation="silu", experts_min=0, experts_max=local - 1)


def _pair_weights(torch, ids32, weights, expert_ids):
    routed = (ids32.unsqueeze(-1) == expert_ids).to(torch.float32)  # [T, K, E]
    return (routed * weights.float().unsqueeze(-1)).sum(dim=1)  # [T, E]


def _flat_gate_up(torch, layer, local, x):
    """(a): x8 [T, H] x w13 [E*2I, H]^T with the row and the column scales inside the GEMM: [T, E, 2I] bf16."""
    w13, _, s13, _ = layer
    rows = x.shape[0]
    x8, xs = dp.quantize_rows(torch, x)
    gate_up = torch.ops.hpu.fp8_gemm_v2(x8, False, w13.reshape(local * 2 * I, H), True, None, torch.bfloat16, xs,
                                        s13.reshape(1, local * 2 * I), None, False)
    return gate_up.reshape(rows, local, 2 * I)


def flat13_moe(torch, layer, local, x, ids32, weights, expert_ids, mmsum=False, bf16_act=False):
    _, w2, _, s2 = layer
    rows = x.shape[0]
    wmat = _pair_weights(torch, ids32, weights, expert_ids)
    gu = _flat_gate_up(torch, layer, local, x)
    if bf16_act:  # the activation in bf16 (the MoE glue's moe_silu_mul rounding): half the bytes of the fp32 path
        mid = torch.nn.functional.silu(gu[..., :I]) * gu[..., I:]
    else:
        mid = (torch.nn.functional.silu(gu[..., :I].float()) * gu[..., I:].float()).to(torch.bfloat16)  # [T, E, I]
    mid = mid.transpose(0, 1).contiguous()  # [E, T, I]
    m8, ms = dp.quantize_rows(torch, mid)
    m_scale = ms * wmat.t().unsqueeze(-1)
    down = torch.ops.hpu.fp8_gemm_v2(m8, False, w2, True, None, torch.bfloat16, m_scale, s2.unsqueeze(1), None, False)
    if mmsum:
        ones = torch.ones(1, local, dtype=torch.bfloat16, device=x.device)
        return torch.matmul(ones, down.reshape(local, rows * H)).reshape(rows, H)
    return down.sum(dim=0, dtype=torch.float32).to(torch.bfloat16)


def _triton_act_quant():
    """The Gaudi-Triton fused SwiGLU + per-row E4M3 quantization (vllm_gaudi/ops/triton_gaudi, one TPC node), called
    directly: VLLM_HPU_TRITON_MODE is set to strict in this probe process only, to prepare the Triton runtime."""
    os.environ["VLLM_HPU_TRITON_MODE"] = "strict"
    from vllm_gaudi.ops.triton_gaudi import runtime

    runtime.prepare_if_enabled()  # raises FastPathUnavailable when the image has no Gaudi Triton backend
    from vllm_gaudi.ops.triton_gaudi.kernels import silu_and_mul_dynamic_quant

    return silu_and_mul_dynamic_quant


def dense_triton_moe(torch, layer, local, x, ids32, weights, expert_ids, flat=False):
    """(b) with Triton: the activation and the per-(expert, row) quantization of mid as one fused node; batched
    (rows (e, t)) or after the 2-D gate / up GEMM (rows (t, e), then the FP8 mid is moved to [E, T, I])."""
    _, w2, _, s2 = layer
    rows = x.shape[0]
    fused = _triton_act_quant()
    wmat = _pair_weights(torch, ids32, weights, expert_ids)
    if flat:
        gu = _flat_gate_up(torch, layer, local, x)  # [T, E, 2I]
        m8, ms = fused(gu.reshape(rows * local, 2 * I))
        m8 = m8.reshape(rows, local, I).transpose(0, 1).contiguous()
        ms = ms.reshape(rows, local).t().unsqueeze(-1)
    else:
        w13, _, s13, _ = layer
        x8, xs = dp.quantize_rows(torch, x)
        a = x8.unsqueeze(0).expand(local, rows, H).contiguous()
        a_scale = xs.unsqueeze(0).expand(local, rows, 1).contiguous()
        gate_up = torch.ops.hpu.fp8_gemm_v2(a, False, w13, True, None, torch.bfloat16, a_scale, s13.unsqueeze(1), None,
                                            False)  # [E, T, 2I]
        m8, ms = fused(gate_up.reshape(local * rows, 2 * I))
        m8, ms = m8.reshape(local, rows, I), ms.reshape(local, rows, 1)
    m_scale = ms.float() * wmat.t().unsqueeze(-1)
    down = torch.ops.hpu.fp8_gemm_v2(m8, False, w2, True, None, torch.bfloat16, m_scale, s2.unsqueeze(1), None, False)
    return down.sum(dim=0, dtype=torch.float32).to(torch.bfloat16)


def flatdown_prepare(torch, layer, local):
    """(c2) W2 requantized to the per-channel scale S[h] = max over the experts (a second FP8 rounding), as [H, E*I]."""
    _, w2, _, s2 = layer
    shared = s2.max(dim=0).values  # [H]
    ratio = s2 / shared.unsqueeze(0)  # [E, H], at most 1
    w2r = (w2.float() * ratio.unsqueeze(-1)).to(torch.float8_e4m3fn)  # [E, H, I]
    return w2r.permute(1, 0, 2).reshape(H, local * I).contiguous(), shared


def flatdown_moe(torch, layer, prep, local, x, ids32, weights, expert_ids):
    w2all, shared = prep
    rows = x.shape[0]
    wmat = _pair_weights(torch, ids32, weights, expert_ids)
    gu = _flat_gate_up(torch, layer, local, x)
    mid = torch.nn.functional.silu(gu[..., :I].float()) * gu[..., I:].float() * wmat.unsqueeze(-1)  # [T, E, I]
    m8, ms = dp.quantize_rows(torch, mid.to(torch.bfloat16).reshape(rows, local * I))
    return torch.ops.hpu.fp8_gemm_v2(m8, False, w2all, True, None, torch.bfloat16, ms, shared.reshape(1, H), None,
                                     False)


def reference(torch, layer, local, x, ids, weights):
    """fp32 on the CPU: sum over the routed local experts of weight * down(silu(gate) * up), dequantized experts."""
    w13, w2, s13, s2 = (t.cpu() for t in layer)
    xf = x.float().cpu()
    ids_c, w_c = ids.cpu(), weights.float().cpu()
    out = torch.zeros(xf.shape[0], H)
    for e in sorted({int(v) for v in ids_c.flatten().tolist() if int(v) < local}):
        rows, slots = (ids_c == e).nonzero(as_tuple=True)
        w13e = w13[e].float() * s13[e].float().unsqueeze(-1)  # [2I, H]
        w2e = w2[e].float() * s2[e].float().unsqueeze(-1)  # [H, I]
        gu = xf[rows] @ w13e.t()
        mid = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
        out.index_add_(0, rows, (mid @ w2e.t()) * w_c[rows, slots].unsqueeze(-1))
    return out


def make_x(torch, rows, kind, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, H, generator=g) * 0.5
    if kind == "spread":
        x = x * torch.exp(torch.empty(rows, 1).uniform_(math.log(0.1), math.log(10.0), generator=g))
    elif kind == "outlier":
        x[0] = x[0] * 100.0
    return x.to(torch.bfloat16)


def rel_errors(torch, y, ref, ids, local):
    has_local = (ids.cpu() < local).any(dim=1)
    diff = (y.float().cpu() - ref).norm(dim=1)
    norm = ref.norm(dim=1).clamp_min(1e-30)
    errs = (diff / norm)[has_local]
    return (float(errs.mean()), float(errs.max())) if errs.numel() else (float("nan"), float("nan"))


def profile_launches(torch, dev, run_chain, trace_dir, tag):
    """(launches, device busy us) of one profiled replay of a captured HPU graph of run_chain() (0, 0 on a CPU)."""
    if dev != "hpu":
        run_chain()
        return 0, 0.0
    from torch.profiler import ProfilerActivity, profile

    class Chain(torch.nn.Module):

        def forward(self, dummy):
            return run_chain()

    module = Chain()
    if dev == "hpu":
        import habana_frameworks.torch as htorch

        module = htorch.hpu.wrap_in_hpu_graph(module)
    dummy = torch.zeros(1).to(dev)
    for _ in range(2):
        module(dummy)
        torch.hpu.synchronize()
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "HPU"):
        activities.append(ProfilerActivity.HPU)
    with profile(activities=activities) as prof:
        module(dummy)
        torch.hpu.synchronize()
    path = Path(trace_dir) / f"{tag}.json"
    prof.export_chrome_trace(str(path))
    events = [e for e in json.load(open(path)).get("traceEvents", []) if e.get("ph") == "X"]
    launches = sum(1 for e in events if e.get("name") == "ExecuteCachedGraph")
    spans = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in events
                   if "HW event name" in (e.get("args") or {}))
    busy, end = 0.0, -1.0
    for start, stop in spans:
        if stop <= end:
            continue
        busy += stop - max(start, end)
        end = stop
    path.unlink()
    return launches, busy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "moe_forms")))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--big-rows", default="96,128")
    ap.add_argument("--small-rows", default="1,3,4,8,16")
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--budget", type=float, default=780.0, help="seconds; later cells are skipped")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    t_start = time.time()
    torch, htcore, dev = dp.backend()
    local = a.local
    results = {"geometry": dict(H=H, I=I, GLOBAL=GLOBAL, TOPK=TOPK, local=local),
               "habana_profile": os.environ.get("HABANA_PROFILE", "")}

    def save():
        (a.out / "moe_forms.json").write_text(json.dumps(results, indent=1) + "\n")

    sets = dp.build_layers(torch, htcore, dev, max(2, a.layers), local)
    expert_ids = torch.arange(local, dtype=torch.int32).to(dev)
    prep = None
    torch.hpu.synchronize()
    cells = [("big", r) for r in a.big_rows.split(",") if r] + [("small", r) for r in a.small_rows.split(",") if r]
    for size, rows_text in cells:
        rows = int(rows_text)
        if time.time() - t_start > a.budget:
            say(f"rows {rows:3d}: SKIPPED (budget {a.budget:.0f} s spent)")
            continue
        ids, weights, hit = dp.routing(torch, rows, local, seed=3000 + rows)
        ids_d, ids32_d, w_d = ids.to(dev), ids.to(torch.int32).to(dev), weights.to(dev)
        if size == "big":
            if prep is None:
                prep = [flatdown_prepare(torch, layer, local) for layer in sets]
            forms = {
                "dense": lambda layer, x, i: dp.dense_moe_folded(torch, layer, local, x, ids32_d, w_d, expert_ids),
                "flat13": lambda layer, x, i: flat13_moe(torch, layer, local, x, ids32_d, w_d, expert_ids),
                "flat13-mmsum": lambda layer, x, i: flat13_moe(torch, layer, local, x, ids32_d, w_d, expert_ids, True),
                "flat13-bf16act": lambda layer, x, i: flat13_moe(torch, layer, local, x, ids32_d, w_d, expert_ids,
                                                                 bf16_act=True),
                "flatdown": lambda layer, x, i: flatdown_moe(torch, layer, prep[i], local, x, ids32_d, w_d,
                                                             expert_ids),
                "dense-triton": lambda layer, x, i: dense_triton_moe(torch, layer, local, x, ids32_d, w_d, expert_ids),
                "flat13-triton": lambda layer, x, i: dense_triton_moe(torch, layer, local, x, ids32_d, w_d, expert_ids,
                                                                      flat=True),
                "vendor-chunk64": lambda layer, x, i: dp.vendor_moe(torch, layer, local, x, ids_d, w_d, 64),
            }
        else:
            forms = {
                "dense": lambda layer, x, i: dp.dense_moe_folded(torch, layer, local, x, ids32_d, w_d, expert_ids),
                "vendor": lambda layer, x, i: dp.vendor_moe(torch, layer, local, x, ids_d, w_d),
                "vendor-t": lambda layer, x, i: vendor_tensor_moe(torch, layer, local, x, ids_d, w_d),
            }
        say(f"rows {rows:3d}: {hit} local experts hit; forms {', '.join(forms)}")
        # accuracy on layer 0, three kinds of x
        for kind in ("even", "spread", "outlier"):
            x = make_x(torch, rows, kind, seed=rows)
            ref = reference(torch, sets[0], local, x, ids, weights)
            xd = x.to(dev)
            outs = {}
            for name, fn in forms.items():
                try:
                    outs[name] = fn(sets[0], xd, 0).float().cpu()
                    mean, worst = rel_errors(torch, outs[name], ref, ids, local)
                    results[f"acc_rows{rows}_{kind}_{name}"] = dict(mean=mean, max=worst)
                    say(f"rows {rows:3d} x {kind:7s} form {name:14s}: error against fp32 mean {mean:.4f} "
                        f"max {worst:.4f}")
                except Exception as error:  # noqa: BLE001 - a probe reports every failure
                    results[f"acc_rows{rows}_{kind}_{name}"] = dict(error=" ".join(repr(error).split())[:300])
                    say(f"rows {rows:3d} x {kind:7s} form {name:14s}: ERROR {' '.join(repr(error).split())[:300]}")
            base = "vendor" if size == "small" else "dense"
            for name, y in outs.items():
                if name != base and base in outs:
                    mean, worst = rel_errors(torch, y, outs[base].float(), ids, local)
                    say(f"rows {rows:3d} x {kind:7s} form {name:14s} against {base}: mean {mean:.4f} max {worst:.4f}")
        save()
        # time, launches and device busy per layer
        g = torch.Generator().manual_seed(rows)
        xs = [(torch.randn(rows, H, generator=g) * 0.5).to(torch.bfloat16).to(dev) for _ in range(len(sets))]
        htcore.mark_step()
        for name, fn in forms.items():
            key = f"rows{rows}_{name}"
            try:
                us, best = dp.replay_us(torch, dev, lambda fn=fn: tuple(fn(layer, x, i) for i, (layer, x) in
                                                                        enumerate(zip(sets[:a.layers], xs))),
                                        a.layers, a.iters)

                def chain(depth, fn=fn):
                    def run():
                        acc = xs[0]
                        for i in range(depth):
                            acc = acc + fn(sets[i], acc, i)
                        return acc
                    return run

                n1, b1 = profile_launches(torch, dev, chain(1), a.out, f"{key}_L1")
                n2, b2 = profile_launches(torch, dev, chain(2), a.out, f"{key}_L2")
                results[key] = dict(us_per_layer=us, best=best, launches_l1=n1, launches_l2=n2,
                                    launches_per_layer=n2 - n1, busy_l1=b1, busy_l2=b2, busy_per_layer=b2 - b1)
                say(f"rows {rows:3d} form {name:14s}: {us:8.1f} us per layer wall (best {best:.1f}), device busy "
                    f"{b2 - b1:7.1f} us per layer, launches per layer {n2 - n1} (L1 {n1}, L2 {n2})")
            except Exception as error:  # noqa: BLE001
                results[key] = dict(error=" ".join(repr(error).split())[:300])
                say(f"rows {rows:3d} form {name:14s}: ERROR {results[key]['error']}")
            save()
    lines = []
    for size, rows_text in cells:
        parts = []
        for key, value in results.items():
            if key.startswith(f"rows{rows_text}_") and "us_per_layer" in value:
                parts.append(f"{key.split('_', 1)[1]} {value['us_per_layer']:.0f}/{value['busy_per_layer']:.0f} us "
                             f"{value['launches_per_layer']} launches")
        if parts:
            lines.append(f"rows {rows_text}: " + ", ".join(parts))
    print("VERDICT moe-forms (wall/busy us per layer, launches per layer): " + ("; ".join(lines) or "nothing ran"),
          flush=True)
    results["verdict"] = lines
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
