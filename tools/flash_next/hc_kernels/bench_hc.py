#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card bench of the hyper-connection kernels: shipped against interleaved (docs/design/flash-next/hc-kernels.md).

    bash tools/flash_next/kernel_probes/with_kernels.sh python tools/flash_next/hc_kernels/bench_hc.py --out "$FARM_OUT"
    PROBE_DEVICE=cpu PYTHONPATH=<tree> python tools/flash_next/hc_kernels/bench_hc.py --rows 4 --quick   # dry run

At the real geometry (hidden 2560, 4 streams, low rank 320) and each row count (default 4, 32, 96, 128):
  * time per call of hc_gate_mix / hc_gate_mix_il and hc_combine_norm / hc_combine_norm_il: K calls captured in one
    HPU graph (wrap_in_hpu_graph) and replayed with a synchronize; the median wall of a replay at two K (``--k``,
    default 8 and 40), and the slope between them is the device time per call (the replay's fixed host and launch
    cost cancels). The calls are CHAINED as consecutive layers are, so the graph compiler cannot run two side by
    side on idle engines: combine_norm takes the previous call's residual_out as its residual; gate_mix takes its
    own inputs plus 1e-30 x one element of the previous call's output in ``gate`` (an add of one full-width
    tensor per call, measured alone as ``chain_add`` and to be subtracted);
  * the same for one whole hyper-connection (combine_norm -> GEMM -> silu_split -> GEMM -> gate_mix, each with its
    own weights; the next one's residual, block output and injection are this one's residual, block input and
    injection, as in the model), shipped ops against interleaved ops (``--chain-k``, default 4 and 16);
  * ``--trace`` (the process must start with HABANA_PROFILE=profile_api_light): a torch-profiler trace of three
    replays of each op at the larger K, at the largest row count, for the TPC engines each call runs on;
  * parity on the card: each interleaved op against the shipped op on the same inputs (bit-equal fraction) and
    against the CPU oracle (the HC contract, tests/unit_tests/hc_fused/hc_case.compare).
Lines "BENCH hc ..." as it goes, "VERDICT hc-kernels: ..." at the end; everything in <out>/bench_hc.json, written
after every measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

H, STREAMS, RANK, PAD = 2560, 4, 320, 12
WIDE = H * STREAMS
EPS = 1e-6
DEVICE = os.environ.get("PROBE_DEVICE", "hpu")


def say(text: str) -> None:
    print(f"BENCH hc {text}", flush=True)


def backend():
    import torch
    if DEVICE == "cpu":
        from tests.unit_tests.hc_fused import hc_ops  # noqa: F401  the oracle as the CPU kernels

        class _Hpu:

            @staticmethod
            def synchronize():
                return None

        return torch, None, _Hpu, hc_ops
    import habana_frameworks.torch as htorch
    import habana_frameworks.torch.core as htcore  # noqa: F401
    from tests.unit_tests.hc_fused.hpu_loader import hc_fused_extension_errors, load_hc_fused_extension
    if not load_hc_fused_extension():
        raise SystemExit("the fused hyper-connection kernels did not load: " + "; ".join(hc_fused_extension_errors()))
    from tests.unit_tests.hc_fused import hc_ops
    return torch, htorch, torch.hpu, hc_ops


def make_sets(torch, rows: int, count: int, seed: int, dev: str):
    """``count`` independent input sets for one hyper-connection at ``rows`` rows (bf16, on the device)."""
    g = torch.Generator().manual_seed(seed)
    sets = []
    for _ in range(count):
        scale = torch.exp(torch.randn(rows, STREAMS, 1, generator=g))
        sets.append({
            "residual": (torch.randn(rows, STREAMS, H, generator=g) * scale).reshape(rows, WIDE).to(torch.bfloat16),
            "block": torch.randn(rows, H, generator=g).to(torch.bfloat16),
            "injection": (torch.randn(rows, STREAMS, generator=g) * 2).to(torch.bfloat16),
            "xn": torch.randn(rows, WIDE, generator=g).to(torch.bfloat16),
            "gate": (torch.randn(rows, WIDE, generator=g) * 3).to(torch.bfloat16),
            "norm_weight": (0.25 * torch.randn(WIDE, generator=g)).to(torch.bfloat16),
            "down": (torch.randn(RANK + STREAMS + PAD, WIDE, generator=g) * WIDE**-0.5).to(torch.bfloat16),
            "up": (torch.randn(WIDE, RANK, generator=g) * RANK**-0.5).to(torch.bfloat16),
        })
    return [{key: value.to(dev) for key, value in s.items()} for s in sets]


def op_call(ops, kind: str, il: bool, s, previous=None):
    """One call on the set ``s``; ``previous`` (the last call's output) makes it depend on that call."""
    if kind == "gate_mix":
        gate = s["gate"]
        if previous is not None:
            gate = gate + (previous.reshape(-1)[:1] * 1e-30).to(gate.dtype)
        return (ops.hc_gate_mix_il if il else ops.hc_gate_mix)(s["xn"], gate, STREAMS)
    if kind == "chain_add":  # the dependency of gate_mix alone
        gate = s["gate"]
        if previous is not None:
            gate = gate + (previous.reshape(-1)[:1] * 1e-30).to(gate.dtype)
        return gate
    residual = s["residual"] if previous is None else previous[0]
    return (ops.hc_combine_norm_il if il else ops.hc_combine_norm)(residual, s["block"], s["injection"],
                                                                     s["norm_weight"], EPS, STREAMS)


def op_chain(ops, kind: str, il: bool, sets):
    outs = []
    previous = None
    for s in sets:
        previous = op_call(ops, kind, il, s, previous)
        outs.append(previous)
    return tuple(outs)


def hc_chain(torch, ops, il: bool, s, previous=None):
    """One hyper-connection as HpuGatedResidual.combine_and_mix wires it; ``previous`` = the last one's
    (residual, block_input, injection), which become this one's residual, block output and injection."""
    combine_norm = ops.hc_combine_norm_il if il else ops.hc_combine_norm
    gate_mix = ops.hc_gate_mix_il if il else ops.hc_gate_mix
    residual, block, injection = (s["residual"], s["block"], s["injection"]) if previous is None else previous
    residual, xn = combine_norm(residual, block, injection, s["norm_weight"], EPS, STREAMS)
    lora, injection = ops.hc_silu_split(torch.nn.functional.linear(xn, s["down"]), STREAMS, RANK)
    block_input = gate_mix(xn, torch.nn.functional.linear(lora, s["up"]), STREAMS)
    return residual, block_input, injection


def hc_chains(torch, ops, il: bool, sets):
    previous = None
    for s in sets:
        previous = hc_chain(torch, ops, il, s, previous)
    return previous


def replay_wall_ms(torch, htorch, hpu, run, iters: int, trace_path: Path | None = None) -> float:
    """Median wall of one synchronized replay of ``run`` captured in an HPU graph (eager on CPU); with
    ``trace_path`` also a torch-profiler trace of three more replays."""

    class Calls(torch.nn.Module):

        def forward(self, dummy):
            return run()

    module = Calls()
    if htorch is not None:
        module = htorch.hpu.wrap_in_hpu_graph(module)
    dummy = torch.zeros(1).to(DEVICE)
    for _ in range(2):  # the capture and the first replay of the recipe
        module(dummy)
        hpu.synchronize()
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        module(dummy)
        hpu.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    if trace_path is not None and htorch is not None:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]) as profiler:
            for _ in range(3):
                module(dummy)
            hpu.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        say(f"trace {trace_path}")
    return statistics.median(walls)


def bit_equal(torch, a, b) -> float:
    return float((a.cpu().view(torch.int16) == b.cpu().view(torch.int16)).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "hc_bench")))
    ap.add_argument("--rows", default="4,32,96,128")
    ap.add_argument("--k", default="8,40", help="two call counts per graph for the per-op slope")
    ap.add_argument("--chain-k", default="4,16", help="two hyper-connection counts per graph for the chain slope")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="1 iteration, small K (a dry run of the script)")
    ap.add_argument("--trace", action="store_true", help="profiler traces at the largest row count (see above)")
    ap.add_argument("--parts", default="parity,ops,chain", help="what to run: parity, ops (per op), chain (per HC)")
    a = ap.parse_args()
    if a.trace and DEVICE == "hpu" and os.environ.get("HABANA_PROFILE") != "profile_api_light":
        ap.error("--trace needs HABANA_PROFILE=profile_api_light in the environment before torch is imported")
    a.out.mkdir(parents=True, exist_ok=True)
    torch, htorch, hpu, ops = backend()
    k_lo, k_hi = (int(v) for v in a.k.split(","))
    c_lo, c_hi = (int(v) for v in a.chain_k.split(","))
    if a.quick:
        k_lo, k_hi, c_lo, c_hi, a.iters = 1, 2, 1, 2, 1
    from tests.unit_tests.hc_fused.hc_case import HcCase, compare

    results: dict = {"device": DEVICE, "k": [k_lo, k_hi], "chain_k": [c_lo, c_hi], "rows": {}}
    out_file = a.out / "bench_hc.json"

    def save():
        out_file.write_text(json.dumps(results, indent=1) + "\n")

    all_rows = [int(v) for v in a.rows.split(",")]
    for rows in all_rows:
        traced = a.trace and rows == max(all_rows)
        entry = results["rows"].setdefault(str(rows), {})
        sets = make_sets(torch, rows, max(k_hi, c_hi), seed=rows, dev=DEVICE)
        hpu.synchronize()
        parts = set(a.parts.split(","))
        # parity on the card, set 0
        for kind in (("gate_mix", "combine_norm") if "parity" in parts else ()):
            try:
                shipped = op_call(ops, kind, False, sets[0])
                il = op_call(ops, kind, True, sets[0])
                shipped = shipped if isinstance(shipped, tuple) else (shipped, )
                il = il if isinstance(il, tuple) else (il, )
                fractions = [bit_equal(torch, x, y) for x, y in zip(il, shipped)]
                case = HcCase(name=f"bench_{kind}_r{rows}", kind=kind, hidden=H, rows=rows)
                inputs = ({
                    "xn": sets[0]["xn"].cpu(),
                    "gate": sets[0]["gate"].cpu()
                } if kind == "gate_mix" else {
                    "residual": sets[0]["residual"].cpu(),
                    "block_output": sets[0]["block"].cpu(),
                    "injection": sets[0]["injection"].cpu(),
                    "norm_weight": sets[0]["norm_weight"].cpu()
                })
                names = ("block_input", ) if kind == "gate_mix" else ("residual_out", "xn")
                problems = compare(case, inputs, dict(zip(names, (t.cpu() for t in il))))
                entry[f"parity_{kind}"] = dict(bit_equal_to_shipped=fractions, oracle_problems=problems[:3])
                say(f"rows {rows:3d} parity {kind}: bit-equal to the shipped op "
                    f"{', '.join(f'{100 * f:.4f} %' for f in fractions)}; oracle "
                    f"{'ok' if not problems else 'FAIL ' + problems[0]}")
            except Exception as error:  # noqa: BLE001 - a bench reports every failure
                entry[f"parity_{kind}"] = dict(error=" ".join(repr(error).split())[:300])
                say(f"rows {rows:3d} parity {kind}: ERROR {entry[f'parity_{kind}']['error']}")
            save()
        # time per op call (chained calls)
        op_runs = (("chain_add", False), ("gate_mix", False), ("gate_mix", True), ("combine_norm", False),
                   ("combine_norm", True))
        for kind, il in (op_runs if "ops" in parts else ()):
            name = f"{kind}{'_il' if il else ''}"
            try:
                walls = {}
                for k in (k_lo, k_hi):

                    def run(k=k, kind=kind, il=il):
                        return op_chain(ops, kind, il, sets[:k])

                    trace = (a.out / f"trace_{name}_r{rows}_k{k}.json") if traced and k == k_hi else None
                    walls[k] = replay_wall_ms(torch, htorch, hpu, run, a.iters, trace)
                per_call_us = (walls[k_hi] - walls[k_lo]) / (k_hi - k_lo) * 1e3
                entry[name] = dict(wall_ms=walls, us_per_call=per_call_us)
                say(f"rows {rows:3d} {name:17s}: {per_call_us:8.2f} us per call (replay {walls[k_lo]:.3f} ms "
                    f"at K={k_lo}, {walls[k_hi]:.3f} ms at K={k_hi})")
            except Exception as error:  # noqa: BLE001
                entry[name] = dict(error=" ".join(repr(error).split())[:300])
                say(f"rows {rows:3d} {name:17s}: ERROR {entry[name]['error']}")
            save()
        # time per hyper-connection
        for il in ((False, True) if "chain" in parts else ()):
            name = f"chain{'_il' if il else ''}"
            try:
                walls = {}
                for k in (c_lo, c_hi):

                    def run(k=k, il=il):
                        return hc_chains(torch, ops, il, sets[:k])

                    trace = (a.out / f"trace_{name}_r{rows}_k{k}.json") if traced and k == c_hi else None
                    walls[k] = replay_wall_ms(torch, htorch, hpu, run, a.iters, trace)
                per_hc_us = (walls[c_hi] - walls[c_lo]) / (c_hi - c_lo) * 1e3
                entry[name] = dict(wall_ms=walls, us_per_hc=per_hc_us)
                say(f"rows {rows:3d} {name:17s}: {per_hc_us:8.2f} us per hyper-connection (replay "
                    f"{walls[c_lo]:.3f} ms at K={c_lo}, {walls[c_hi]:.3f} ms at K={c_hi})")
            except Exception as error:  # noqa: BLE001
                entry[name] = dict(error=" ".join(repr(error).split())[:300])
                say(f"rows {rows:3d} {name:17s}: ERROR {entry[name]['error']}")
            save()
        del sets

    parts = []
    for rows, entry in results["rows"].items():
        cells = []
        for name, key in (("gate_mix", "us_per_call"), ("combine_norm", "us_per_call"), ("chain", "us_per_hc")):
            old, new = entry.get(name, {}).get(key), entry.get(f"{name}_il", {}).get(key)
            if old is not None and new is not None:
                cells.append(f"{name} {old:.1f} -> {new:.1f} us")
        parts.append(f"rows {rows}: " + ", ".join(cells))
    print("VERDICT hc-kernels: " + ("; ".join(parts) or "nothing measured"), flush=True)
    results["verdict"] = parts
    save()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.exit(main())
