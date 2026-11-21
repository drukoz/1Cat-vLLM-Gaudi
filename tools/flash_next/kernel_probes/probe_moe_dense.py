#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe: a dense batched FP8 MoE (every local expert on every row, two batched GEMMs) against the vendor
op `torch.ops.hpu.mixture_of_experts`, per layer, each layer with its own experts, timed as HPU-graph replays.

    python3 probe_moe_dense.py --out DIR [--layers 8] [--rows 32,128] [--locals 64,128]
    PROBE_DEVICE=cpu python3 probe_moe_dense.py --out DIR --layers 1 --rows 4 --locals 8   (a dry run)

Why: on one card the vendor op costs about 600 us per layer at 32 rows and 1,200 us at 128 rows at TP8 geometry
(jobs 00141, 00149: 361 device nodes per layer at 32 rows in step-anatomy's trace), while reading all 64 local
experts' FP8 weights once is 315 MB = 128 us at 2.45 TB/s, and the dense compute at 32 rows (64 x 32 x 4.9 M MACs)
is about 25 us of FP8 MME time. If one batched GEMM over [E, rows, H] x [E, 2I, H] and one over [E, rows, I] x
[E, H, I] run near those floors, a dense form beats the vendor op wherever most local experts are hit (MTP
verification rows; 32 streams), at the price of compute on unrouted (row, expert) pairs.

The dense form (`bmm`), per layer: x quantized per row (FP8, scale [T, 1]); A = x expanded to [E, T, H];
gate_up = fp8_gemm_v2(A, B = w13 [E, 2I, H], trans_B) without scales, then times the row scale and the expert's
per-channel scale [E, 1, 2I]; silu(gate) * up; quantized per (expert, row); down = fp8_gemm_v2(.., w2 [E, H, I],
trans_B) times its scales; out = sum over E of down * the routing weight of (row, expert) (0 when not routed).
The vendor form gets the same weights as per-expert views and the same routing (ids, weights), as
VllmMixtureOfExpertsOpFP8PerChannel.forward calls it. Parity: max abs difference of the two outputs, relative to
the vendor output's max abs value.

Lines `PROBE moe-dense local L rows R form F: X us per layer (...)`, parity lines, then VERDICT lines.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

H, I, GLOBAL, TOPK = 2560, 640, 512, 10
DEVICE = os.environ.get("PROBE_DEVICE", "hpu")
FP8_MAX = 240.0


def say(text):
    print(f"PROBE moe-dense {text}", flush=True)


def backend():
    import torch

    if DEVICE == "cpu":
        from tests.unit_tests.models.flash_next import moe_block

        moe_block.install_vendor_ops(fp8_max=FP8_MAX)

        def fp8_gemm_v2(A, trans_A, B, trans_B, D, out_dtype, A_scale_inv=None, B_scale_inv=None, bias=None,
                        accumulate=False):
            a = A.float().transpose(-1, -2) if trans_A else A.float()
            b = B.float().transpose(-1, -2) if trans_B else B.float()
            out = torch.matmul(a, b)
            if A_scale_inv is not None:
                out = out * A_scale_inv
            if B_scale_inv is not None:
                out = out * B_scale_inv
            return out.to(out_dtype)

        torch.ops.hpu.fp8_gemm_v2 = fp8_gemm_v2
        torch.hpu = type("hpu", (), {"synchronize": staticmethod(lambda: None)})

        class _Core:

            @staticmethod
            def mark_step():
                pass

        return torch, _Core, "cpu"
    import habana_frameworks.torch.core as htcore

    return torch, htcore, "hpu"


def quantize_rows(torch, x):
    """Per-row FP8 (the last dim): (x_fp8, scale [..., 1] fp32) with x = x_fp8 * scale."""
    scale = (x.abs().amax(dim=-1, keepdim=True).float() + 1e-8) / FP8_MAX
    flat = x.reshape(-1, x.shape[-1])
    flat_scale = scale.reshape(-1, 1)
    q = torch.ops.hpu.cast_to_fp8_v2(flat, 1.0 / flat_scale, False, False, torch.float8_e4m3fn)[0]
    return q.reshape(x.shape), scale


def build_layers(torch, htcore, dev, layers, local, pool=2):
    """Per layer: stacked FP8 experts w13 [E, 2I, H], w2 [E, H, I] (distinct device memory per layer) and per-channel
    scales s13 [E, 2I], s2 [E, H]."""
    g = torch.Generator().manual_seed(17)
    fp8 = torch.float8_e4m3fn
    base13 = [(torch.randn(local, 2 * I, H, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(pool)]
    base2 = [(torch.randn(local, H, I, generator=g) * 60).clamp(-224, 224).to(fp8).to(dev) for _ in range(pool)]
    s13 = (torch.rand(local, 2 * I, generator=g) * 1e-3 + 5e-4).float().to(dev)
    s2 = (torch.rand(local, H, generator=g) * 1e-3 + 5e-4).float().to(dev)
    sets = []
    for layer in range(layers):
        sets.append((base13[layer % pool].clone(), base2[layer % pool].clone(), s13.clone(), s2.clone()))
        htcore.mark_step()
    return sets


def routing(torch, rows, local, seed):
    """Uniform top-10 of 512 global ids per row; the rank's local experts are ids 0 .. local - 1."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(GLOBAL, generator=g)[:TOPK] for _ in range(rows)]).to(torch.int64)
    weights = torch.softmax(torch.randn(rows, TOPK, generator=g), dim=-1).to(torch.bfloat16)
    hit = len({int(v) for v in ids.flatten() if int(v) < local})
    return ids, weights, hit


def vendor_moe(torch, layer, local, x, ids, weights, chunk=None):
    w13, w2, s13, s2 = layer
    x8, scale = quantize_rows(torch, x)
    kwargs = {} if chunk is None else {"chunk_size": int(chunk), "total_experts": GLOBAL}
    return torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids, router_weights=weights,
                                            w12=tuple(w13[e] for e in range(local)),
                                            w3=tuple(w2[e] for e in range(local)), d_scale_hidden_states=scale,
                                            d_scale_w12=tuple(s13[e] for e in range(local)),
                                            d_scale_w3=tuple(s2[e] for e in range(local)), permuted_weights=True,
                                            activation="silu", experts_min=0, experts_max=local - 1, **kwargs)


def dense_moe(torch, layer, local, x, ids32, weights, expert_ids):
    """Every local expert on every row; the routing weight of a (row, expert) pair that is not routed is 0."""
    w13, w2, s13, s2 = layer
    rows = x.shape[0]
    routed = (ids32.unsqueeze(-1) == expert_ids).to(torch.float32)  # [T, 10, E]
    wmat = (routed * weights.float().unsqueeze(-1)).sum(dim=1)  # [T, E]
    x8, xs = quantize_rows(torch, x)  # [T, H], [T, 1]
    a = x8.unsqueeze(0).expand(local, rows, H).contiguous()  # [E, T, H]
    gate_up = torch.ops.hpu.fp8_gemm_v2(a, False, w13, True, None, torch.float32, None, None, None, False)
    gate_up = gate_up * xs.unsqueeze(0) * s13.unsqueeze(1)  # [E, T, 2I]
    gate, up = gate_up[..., :I], gate_up[..., I:]
    mid = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)  # [E, T, I]
    m8, ms = quantize_rows(torch, mid)  # [E, T, I], [E, T, 1]
    down = torch.ops.hpu.fp8_gemm_v2(m8, False, w2, True, None, torch.float32, None, None, None, False)
    down = down * ms * s2.unsqueeze(1)  # [E, T, H]
    return (down * wmat.t().unsqueeze(-1)).sum(dim=0).to(torch.bfloat16)  # [T, H]


def dense_moe_fused(torch, layer, local, x, ids32, weights, expert_ids):
    """The dense form with the scales inside the two GEMMs (A_scale_inv per row [E, T, 1], B_scale_inv per column
    [E, 1, N], as vllm_gaudi/ops/deepseek_v41_grouped_prefill.py passes them) and bf16 GEMM outputs."""
    w13, w2, s13, s2 = layer
    rows = x.shape[0]
    routed = (ids32.unsqueeze(-1) == expert_ids).to(torch.float32)
    wmat = (routed * weights.float().unsqueeze(-1)).sum(dim=1)  # [T, E]
    x8, xs = quantize_rows(torch, x)
    a = x8.unsqueeze(0).expand(local, rows, H).contiguous()
    a_scale = xs.unsqueeze(0).expand(local, rows, 1).contiguous()
    gate_up = torch.ops.hpu.fp8_gemm_v2(a, False, w13, True, None, torch.bfloat16, a_scale, s13.unsqueeze(1), None,
                                        False)
    mid = (torch.nn.functional.silu(gate_up[..., :I].float()) * gate_up[..., I:].float()).to(torch.bfloat16)
    m8, ms = quantize_rows(torch, mid)
    down = torch.ops.hpu.fp8_gemm_v2(m8, False, w2, True, None, torch.bfloat16, ms, s2.unsqueeze(1), None, False)
    return (down.float() * wmat.t().unsqueeze(-1)).sum(dim=0).to(torch.bfloat16)


def dense_moe_folded(torch, layer, local, x, ids32, weights, expert_ids):
    """dense-fused with the routing weight folded into the down GEMM's row scale (a pair that is not routed gets
    scale 0) and the sum over the experts accumulated in fp32 straight from the bf16 GEMM output: no [E, T, H] fp32
    product (job 00177: the served dense form lost on the real model; one suspect is that intermediate)."""
    w13, w2, s13, s2 = layer
    rows = x.shape[0]
    routed = (ids32.unsqueeze(-1) == expert_ids).to(torch.float32)
    wmat = (routed * weights.float().unsqueeze(-1)).sum(dim=1)  # [T, E]
    x8, xs = quantize_rows(torch, x)
    a = x8.unsqueeze(0).expand(local, rows, H).contiguous()
    a_scale = xs.unsqueeze(0).expand(local, rows, 1).contiguous()
    gate_up = torch.ops.hpu.fp8_gemm_v2(a, False, w13, True, None, torch.bfloat16, a_scale, s13.unsqueeze(1), None,
                                        False)
    mid = (torch.nn.functional.silu(gate_up[..., :I].float()) * gate_up[..., I:].float()).to(torch.bfloat16)
    m8, ms = quantize_rows(torch, mid)
    m_scale = ms * wmat.t().unsqueeze(-1)  # [E, T, 1]: the routing weight rides on the row scale
    down = torch.ops.hpu.fp8_gemm_v2(m8, False, w2, True, None, torch.bfloat16, m_scale, s2.unsqueeze(1), None,
                                     False)
    return down.sum(dim=0, dtype=torch.float32).to(torch.bfloat16)


def replay_us(torch, dev, run, layers, iters):
    class Layers(torch.nn.Module):

        def forward(self, dummy):
            return run()

    module = Layers()
    if dev == "hpu":
        import habana_frameworks.torch as htorch

        module = htorch.hpu.wrap_in_hpu_graph(module)
    dummy = torch.zeros(1).to(dev)
    for _ in range(2):
        module(dummy)
        torch.hpu.synchronize()
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        module(dummy)
        torch.hpu.synchronize()
        walls.append(time.perf_counter() - t0)
    walls.sort()
    return walls[len(walls) // 2] / layers * 1e6, walls[0] / layers * 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "moe_dense")))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--rows", default="32,128")
    ap.add_argument("--locals", default="64,128")
    ap.add_argument("--forms", default="vendor,vendor-chunk64,dense,dense-fused,dense-folded")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    torch, htcore, dev = backend()
    results = {}
    for local in [int(v) for v in a.locals.split(",") if v]:
        sets = build_layers(torch, htcore, dev, a.layers, local)
        expert_ids = torch.arange(local, dtype=torch.int32).to(dev)
        torch.hpu.synchronize()
        for rows in [int(v) for v in a.rows.split(",") if v]:
            g = torch.Generator().manual_seed(rows)
            ids, weights, hit = routing(torch, rows, local, seed=2000 + rows)
            ids_d, ids32_d, w_d = ids.to(dev), ids.to(torch.int32).to(dev), weights.to(dev)
            xs = [(torch.randn(rows, H, generator=g) * 0.5).to(torch.bfloat16).to(dev) for _ in range(a.layers)]
            htcore.mark_step()
            # parity on layer 0
            try:
                y_vendor = vendor_moe(torch, sets[0], local, xs[0], ids_d, w_d).float().cpu()
                ref = float(y_vendor.abs().max()) or 1.0
                for name, fn in (("dense", dense_moe), ("dense-fused", dense_moe_fused),
                                 ("dense-folded", dense_moe_folded)):
                    y = fn(torch, sets[0], local, xs[0], ids32_d, w_d, expert_ids).float().cpu()
                    diff = float((y_vendor - y).abs().max())
                    say(f"local {local:3d} rows {rows:3d} parity {name} vs vendor: max abs diff {diff:.3e}, "
                        f"relative {diff / ref:.3e} (vendor max {ref:.3e}; {hit} local experts hit)")
                    results[f"parity_{name}_local{local}_rows{rows}"] = dict(max_abs=diff, rel=diff / ref, hit=hit)
            except Exception as error:  # noqa: BLE001 - a probe reports every failure
                say(f"local {local:3d} rows {rows:3d} parity: ERROR {' '.join(repr(error).split())[:300]}")
            for form in a.forms.split(","):
                key = f"local{local}_rows{rows}_{form}"
                try:
                    if form in ("dense", "dense-fused", "dense-folded"):
                        fn = {"dense": dense_moe, "dense-fused": dense_moe_fused,
                              "dense-folded": dense_moe_folded}[form]

                        def run(sets=sets, xs=xs, fn=fn):
                            return tuple(fn(torch, layer, local, x, ids32_d, w_d, expert_ids)
                                         for layer, x in zip(sets, xs))
                    else:
                        chunk = 64 if form == "vendor-chunk64" else None

                        def run(sets=sets, xs=xs, chunk=chunk):
                            return tuple(vendor_moe(torch, layer, local, x, ids_d, w_d, chunk)
                                         for layer, x in zip(sets, xs))
                    us, best = replay_us(torch, dev, run, a.layers, a.iters)
                    results[key] = dict(us_per_layer=us, best=best, hit=hit)
                    say(f"local {local:3d} rows {rows:3d} form {form:14s}: {us:8.1f} us per layer (best {best:.1f}; "
                        f"{hit} local experts hit; K={a.layers}, HPU-graph replay)")
                except Exception as error:  # noqa: BLE001
                    results[key] = dict(error=" ".join(repr(error).split())[:300])
                    say(f"local {local:3d} rows {rows:3d} form {form:14s}: ERROR {results[key]['error']}")
                (a.out / "moe_dense.json").write_text(json.dumps(results, indent=1) + "\n")
        del sets
    lines = []
    for key in sorted(k for k in results if k.endswith("_vendor")):
        base = key[:-len("vendor")]
        v, c, d, f, g = (results.get(base + n, {}).get("us_per_layer")
                         for n in ("vendor", "vendor-chunk64", "dense", "dense-fused", "dense-folded"))
        if v:
            lines.append(f"{base.rstrip('_')}: vendor {v:.1f}, vendor chunk 64 {c or float('nan'):.1f}, "
                         f"dense {d or float('nan'):.1f}, dense-fused {f or float('nan'):.1f}, "
                         f"dense-folded {g or float('nan'):.1f} us")
    print("VERDICT moe-dense: " + ("; ".join(lines) or "no dense form ran"), flush=True)
    results["verdict"] = lines
    (a.out / "moe_dense.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
