#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe: a grouped FP8 MoE (only the routed (row, expert) pairs, at most C per local expert) against the
served dense form and the vendor op with chunk_size 64, per layer at TP8 geometry, timed as HPU-graph replays.

    python3 probe_moe_grouped.py --out DIR [--layers 8] [--rows 32,96,128] [--caps 8,16,32]
    PROBE_DEVICE=cpu python3 probe_moe_grouped.py --out DIR --layers 1 --rows 8 --caps 2,4 --local 8  (a dry run)

Why: with the dense form on, the MoE is the largest block of device time of a 128-row MTP step (job 00205: 19.4 of
54 ms busy): it computes all 64 local experts for every row, while a row uses about 1.25 local experts per rank
(10 of 512 at EP8). The grouped form computes E x C rows instead of E x T: x is gathered into expert-major slots
[E * C, H] (slot = expert * C + the pair's rank among that expert's pairs, in row order; a cumulative sum over a
one-hot [T * K, E], no sort and no host read), two batched FP8 GEMMs [E, C, H] x w13 [E, 2I, H] and [E, C, I] x
w2 [E, H, I] with the scales inside them and the routing weight on the down GEMM's row scale (as the served dense
form), then index_add of the slots' rows back into [T, H]. Static shapes; the weights are still read in full (all
64 experts), so the floor is the weight bytes (315 MB per layer, about 128 us at 2.45 TB/s).

Exactness: a pair beyond C for its expert is NOT computed; the probe counts those pairs ("overflow") under uniform
and under skewed routing. With C = T no pair can overflow (that is the dense form's compute).

Node count: the aten ops one layer of each form issues (counted on the CPU with a dispatch mode; on the card the
graph compiler fuses some of them).

Lines `PROBE moe-grouped ...`, then VERDICT lines. Results in DIR/moe_grouped.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_moe_dense as dense_probe  # noqa: E402  (the same backend, layers, routing, vendor and timing)

H, I, GLOBAL, TOPK = dense_probe.H, dense_probe.I, dense_probe.GLOBAL, dense_probe.TOPK


def say(text):
    print(f"PROBE moe-grouped {text}", flush=True)


def served_dense(torch, layer, local, x, ids32, weights, expert_ids):
    """The served dense form (vllm_gaudi/extension/ops.py moe_dense_fp8)."""
    return dense_probe.dense_moe_folded(torch, layer, local, x, ids32, weights, expert_ids)


def grouped_moe(torch, layer, local, x, ids32, weights, expert_ids, cap):
    """Only the routed pairs, at most `cap` per local expert. Returns (out [T, H] bf16, overflow pairs (0-d))."""
    w13, w2, s13, s2 = layer
    rows, topk = ids32.shape
    pairs = rows * topk
    slots = local * cap
    flat = ids32.reshape(pairs)  # this rank's local experts are ids 0 .. local - 1 (experts_min 0)
    onehot = (flat.unsqueeze(-1) == expert_ids).to(torch.int32)  # [P, E]; a remote pair is all zeros
    rank = (torch.cumsum(onehot, dim=0) * onehot).sum(dim=-1) - 1  # the pair's rank among its expert's pairs; -1 remote
    local_pair = rank >= 0
    keep = local_pair & (rank < cap)
    overflow = (local_pair & (rank >= cap)).sum()
    slot = torch.where(keep, flat * cap + rank, torch.full_like(rank, slots))  # slot `slots` = the trash slot
    pair_ids = torch.arange(pairs, dtype=torch.int32, device=x.device)
    pair_of_slot = torch.full((slots + 1, ), pairs, dtype=torch.int32, device=x.device).scatter(0, slot.long(),
                                                                                                 pair_ids)
    row_of_pair = torch.cat([pair_ids // topk, torch.full((1, ), rows, dtype=torch.int32, device=x.device)])
    row_of_slot = row_of_pair.index_select(0, pair_of_slot[:slots])  # [E*C]; an empty slot reads row `rows`
    weight_of_pair = torch.cat([weights.float().reshape(pairs), torch.zeros(1, device=x.device)])
    weight_of_slot = weight_of_pair.index_select(0, pair_of_slot[:slots])  # [E*C]; 0 for an empty slot
    x_pad = torch.cat([x, torch.zeros(1, x.shape[1], dtype=x.dtype, device=x.device)])
    xg = x_pad.index_select(0, row_of_slot)  # [E*C, H]
    x8, xs = dense_probe.quantize_rows(torch, xg)
    gate_up = torch.ops.hpu.fp8_gemm_v2(x8.reshape(local, cap, H), False, w13, True, None, torch.bfloat16,
                                        xs.reshape(local, cap, 1), s13.unsqueeze(1), None, False)
    mid = (torch.nn.functional.silu(gate_up[..., :I].float()) * gate_up[..., I:].float()).to(torch.bfloat16)
    m8, ms = dense_probe.quantize_rows(torch, mid)  # [E, C, I], [E, C, 1]
    m_scale = ms * weight_of_slot.reshape(local, cap, 1)
    down = torch.ops.hpu.fp8_gemm_v2(m8, False, w2, True, None, torch.bfloat16, m_scale, s2.unsqueeze(1), None,
                                     False)  # [E, C, H]
    out = torch.zeros(rows + 1, H, dtype=torch.float32, device=x.device)
    out = out.index_add(0, row_of_slot, down.reshape(slots, H).float())
    return out[:rows].to(torch.bfloat16), overflow


def skewed_routing(torch, rows, local, seed, alpha=1.0):
    """Top-10 of 512 drawn with probability proportional to 1 / (rank + 1) ** alpha over a random expert order
    (a few hot experts, some of them local); weights as in the uniform routing."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(GLOBAL, generator=g)
    prob = 1.0 / torch.arange(1, GLOBAL + 1, dtype=torch.float64)**alpha
    ids = torch.stack([order[torch.multinomial(prob, TOPK, replacement=False, generator=g)] for _ in range(rows)])
    weights = torch.softmax(torch.randn(rows, TOPK, generator=g), dim=-1).to(torch.bfloat16)
    hit = len({int(v) for v in ids.flatten() if int(v) < local})
    return ids.to(torch.int64), weights, hit


def max_per_expert(ids, local):
    counts = [0] * local
    for v in ids.flatten().tolist():
        if v < local:
            counts[v] += 1
    return max(counts), sum(counts)


def count_ops(fn):
    """The aten ops one call issues (CPU, dispatch mode)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):

        def __init__(self):
            super().__init__()
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.append(str(func.overloadpacket.__name__))
            return func(*args, **(kwargs or {}))

    with Count() as counter:
        fn()
    return counter.ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "moe_grouped")))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--rows", default="32,96,128")
    ap.add_argument("--caps", default="8,16,32")
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--forms", default="vendor-chunk64,dense,grouped")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    torch, htcore, dev = dense_probe.backend()
    local = a.local
    caps = [int(v) for v in a.caps.split(",") if v]
    forms = a.forms.split(",")
    results = {"geometry": dict(H=H, I=I, GLOBAL=GLOBAL, TOPK=TOPK, local=local)}

    def save():
        (a.out / "moe_grouped.json").write_text(json.dumps(results, indent=1) + "\n")

    # Overflow statistics of the capacities (host side, many routings): uniform and skewed top-10 of 512.
    for rows in [int(v) for v in a.rows.split(",") if v]:
        for kind in ("uniform", "skew1.0", "skew1.2"):
            maxima = []
            for seed in range(40):
                if kind == "uniform":
                    ids, _, _ = dense_probe.routing(torch, rows, local, seed=9000 + seed)
                else:
                    ids, _, _ = skewed_routing(torch, rows, local, seed=9000 + seed, alpha=float(kind[4:]))
                maxima.append(max_per_expert(ids, local))
            worst = max(m for m, _ in maxima)
            mean_pairs = sum(p for _, p in maxima) / len(maxima)
            over = {c: sum(1 for m, _ in maxima if m > c) / len(maxima) for c in caps}
            results[f"overflow_rows{rows}_{kind}"] = dict(max_per_expert=worst, mean_local_pairs=mean_pairs,
                                                          share_of_layers_over_cap=over)
            say(f"rows {rows:3d} routing {kind:7s}: local pairs {mean_pairs:.1f} per layer, max per expert {worst} "
                f"(40 layers); layers with a pair beyond C: " + ", ".join(f"C={c} {over[c]:.2f}" for c in caps))
    save()

    # Op counts (CPU dispatch mode; the same code as the card runs).
    if dev == "cpu" or os.environ.get("PROBE_COUNT_OPS") == "1":
        try:
            g = torch.Generator().manual_seed(5)
            small = dense_probe.build_layers(torch, htcore, dev, 1, local)[0]
            rows = 8
            ids, weights, _ = dense_probe.routing(torch, rows, local, seed=5)
            x = (torch.randn(rows, H, generator=g) * 0.5).to(torch.bfloat16).to(dev)
            ids32, w = ids.to(torch.int32).to(dev), weights.to(dev)
            expert_ids = torch.arange(local, dtype=torch.int32).to(dev)
            for name, fn in (("dense", lambda: served_dense(torch, small, local, x, ids32, w, expert_ids)),
                             ("grouped", lambda: grouped_moe(torch, small, local, x, ids32, w, expert_ids, 4))):
                ops = count_ops(fn)
                results[f"ops_{name}"] = ops
                say(f"form {name}: {len(ops)} aten ops per layer ({', '.join(ops)})")
        except Exception as error:  # noqa: BLE001
            say(f"op count: ERROR {' '.join(repr(error).split())[:300]}")
        save()

    sets = dense_probe.build_layers(torch, htcore, dev, a.layers, local)
    expert_ids = torch.arange(local, dtype=torch.int32).to(dev)
    torch.hpu.synchronize()
    for rows in [int(v) for v in a.rows.split(",") if v]:
        g = torch.Generator().manual_seed(rows)
        ids, weights, hit = dense_probe.routing(torch, rows, local, seed=2000 + rows)
        ids_d, ids32_d, w_d = ids.to(dev), ids.to(torch.int32).to(dev), weights.to(dev)
        xs = [(torch.randn(rows, H, generator=g) * 0.5).to(torch.bfloat16).to(dev) for _ in range(a.layers)]
        htcore.mark_step()
        worst, pairs = max_per_expert(ids, local)
        say(f"rows {rows:3d}: {pairs} local pairs, {hit} local experts hit, max {worst} pairs per expert")
        try:  # parity on layer 0: dense and every capacity against the vendor op
            y_vendor = dense_probe.vendor_moe(torch, sets[0], local, xs[0], ids_d, w_d).float().cpu()
            ref = float(y_vendor.abs().max()) or 1.0
            y = served_dense(torch, sets[0], local, xs[0], ids32_d, w_d, expert_ids).float().cpu()
            diff = float((y_vendor - y).abs().max())
            results[f"parity_dense_rows{rows}"] = dict(max_abs=diff, rel=diff / ref)
            say(f"rows {rows:3d} parity dense vs vendor: relative {diff / ref:.3e}")
            for cap in caps:
                y, over = grouped_moe(torch, sets[0], local, xs[0], ids32_d, w_d, expert_ids, cap)
                diff = float((y_vendor - y.float().cpu()).abs().max())
                over = int(over.cpu())
                results[f"parity_grouped{cap}_rows{rows}"] = dict(max_abs=diff, rel=diff / ref, overflow=over)
                say(f"rows {rows:3d} parity grouped C={cap} vs vendor: relative {diff / ref:.3e}, overflow pairs {over}")
        except Exception as error:  # noqa: BLE001 - a probe reports every failure
            say(f"rows {rows:3d} parity: ERROR {' '.join(repr(error).split())[:300]}")
        save()
        runs = []
        if "vendor-chunk64" in forms:
            runs.append(("vendor-chunk64", lambda sets=sets, xs=xs: tuple(
                dense_probe.vendor_moe(torch, layer, local, x, ids_d, w_d, 64) for layer, x in zip(sets, xs))))
        if "vendor" in forms:
            runs.append(("vendor", lambda sets=sets, xs=xs: tuple(
                dense_probe.vendor_moe(torch, layer, local, x, ids_d, w_d) for layer, x in zip(sets, xs))))
        if "dense" in forms:
            runs.append(("dense", lambda sets=sets, xs=xs: tuple(
                served_dense(torch, layer, local, x, ids32_d, w_d, expert_ids) for layer, x in zip(sets, xs))))
        if "grouped" in forms:
            for cap in caps:
                if cap <= rows:
                    runs.append((f"grouped{cap}", lambda sets=sets, xs=xs, cap=cap: tuple(
                        grouped_moe(torch, layer, local, x, ids32_d, w_d, expert_ids, cap)[0]
                        for layer, x in zip(sets, xs))))
        for name, run in runs:
            key = f"rows{rows}_{name}"
            try:
                us, best = dense_probe.replay_us(torch, dev, run, a.layers, a.iters)
                results[key] = dict(us_per_layer=us, best=best)
                say(f"rows {rows:3d} form {name:14s}: {us:8.1f} us per layer (best {best:.1f}; K={a.layers}, "
                    f"HPU-graph replay)")
            except Exception as error:  # noqa: BLE001
                results[key] = dict(error=" ".join(repr(error).split())[:300])
                say(f"rows {rows:3d} form {name:14s}: ERROR {results[key]['error']}")
            save()
    lines = []
    for rows in [int(v) for v in a.rows.split(",") if v]:
        parts = []
        for name in ["vendor-chunk64", "vendor", "dense"] + [f"grouped{c}" for c in caps]:
            us = results.get(f"rows{rows}_{name}", {}).get("us_per_layer")
            if us is not None:
                parts.append(f"{name} {us:.1f}")
        if parts:
            lines.append(f"rows {rows}: " + ", ".join(parts) + " us per layer")
    print("VERDICT moe-grouped: " + ("; ".join(lines) or "no form ran"), flush=True)
    results["verdict"] = lines
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
