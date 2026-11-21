#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card bench of the fused GDN verify op per variant (docs/design/flash-next/hc-kernels.md, "GDN verify").

    bash tools/flash_next/kernel_probes/with_kernels.sh env VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 \
        VLLM_HPU_GDN_VERIFY_VARIANT=1 python tools/flash_next/hc_kernels/bench_gdn_verify.py --label variant1 \
        --out "$FARM_OUT"
    PROBE_DEVICE=cpu PYTHONPATH=<tree> python tools/flash_next/hc_kernels/bench_gdn_verify.py --quick   # dry run

The glue picks the kernel build and its index-space layout from VLLM_HPU_GDN_KERNEL_INTERLEAVE and
VLLM_HPU_GDN_VERIFY_VARIANT when the graph compiler instantiates the op, so one process measures one variant:
run the script once per layout (``card_gdn_verify.sh`` does) and compare the ``--label``ed results.

At the TP8 geometry (2 key heads, 6 value heads per rank) with T nodes per request (default 4, MTP k = 3) and
each batch size (default 1, 8, 32): K verify calls, EACH ON ITS OWN conv and SSM pools (two calls on one pool in
one graph have no declared order: kernel-fixes' rule-10 note), CHAINED (call i + 1's gate input ``a`` gets
1e-30 x one element of call i's output, so the graph compiler cannot run two calls side by side on idle engines, as
consecutive layers cannot), captured in one HPU graph and replayed with a synchronize; the median wall of a replay at
two K (``--k``, default 4 and 12), and the slope is the device time per call (with one tiny add). The output and both
pools of the first set (one eager call, no chain) are saved for a bit comparison between variants (``--compare-to
DIR``: another run's out folder). ``--trace``: a torch-profiler trace of three replays at the larger K per batch size
(the process must start with HABANA_PROFILE=profile_api_light), for the TPC engines each call runs on
(reports/hc-kernels/tools/kernel_engines.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

DEVICE = os.environ.get("PROBE_DEVICE", "hpu")


def say(text: str) -> None:
    print(f"BENCH gdn-verify {text}", flush=True)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "gdn_bench")))
    ap.add_argument("--label", default="layout")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--k", default="4,12")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--compare-to", type=Path, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    if a.trace and DEVICE == "hpu" and os.environ.get("HABANA_PROFILE") != "profile_api_light":
        ap.error("--trace needs HABANA_PROFILE=profile_api_light in the environment before torch is imported")
    out = a.out / a.label
    out.mkdir(parents=True, exist_ok=True)

    import torch

    from flashinfer_gaudi.gdn_core_decode import GdnGateActivation
    from tools.flash_next.gdn_core_decode_bench import verify_case
    from tools.flash_next.gdn_core_decode_replay import VerifyStep
    from tests.unit_tests.gdn_core_decode.verify_sim_case import make_verify_inputs, verify_scalars

    htorch = None
    if DEVICE == "hpu":
        import habana_frameworks.torch as htorch  # noqa: F811
        from flashinfer_gaudi._native import load_gdn_fused_extension
        if not load_gdn_fused_extension():
            raise SystemExit("gdn_fused_ops extension is not loaded")

    k_lo, k_hi = (int(v) for v in a.k.split(","))
    if a.quick:
        k_lo, k_hi, a.iters = 1, 2, 1
    env = {key: os.environ.get(key) for key in ("VLLM_HPU_GDN_KERNEL_INTERLEAVE", "VLLM_HPU_GDN_VERIFY_VARIANT")}
    results = {"label": a.label, "env": env, "tp": a.tp, "tokens": a.tokens, "k": [k_lo, k_hi], "batches": {}}
    say(f"label {a.label}, env {env}")

    def sync():
        if DEVICE == "hpu":
            torch.hpu.synchronize()

    for batch in (int(v) for v in a.batches.split(",")):
        case = verify_case(a.tp, batch, a.tokens, GdnGateActivation.SIGMOID)
        scale, eps, gate_activation = verify_scalars(case)
        sets = []
        for index in range(k_hi):
            inputs = {name: tensor.to(DEVICE) for name, tensor in make_verify_inputs(case).items()}
            step = VerifyStep(inputs["A_log"], inputs["dt_bias"], inputs["conv_weight_t"], inputs["norm_weight"],
                              inputs["row_indices"], inputs["ssm_store_indices"], scale, eps, gate_activation)
            args = (inputs["conv_state"], inputs["ssm_state"], inputs["mixed_qkv"], inputs["z"], inputs["a"],
                    inputs["b"])
            sets.append((step, args))
        sync()
        entry = results["batches"].setdefault(str(batch), {})
        # the bits of set 0 after one eager call (both pools are written in place)
        step0, args0 = sets[0]
        first = step0(*args0)
        sync()
        digest = {}
        for name, tensor in (("out", first), ("conv_state", args0[0]), ("ssm_state", args0[1])):
            data = tensor.detach().float().cpu().contiguous().numpy().tobytes()
            digest[name] = hashlib.sha256(data).hexdigest()
            (out / f"b{batch}_{name}.bin").write_bytes(data)
        entry["sha256"] = digest
        walls = {}
        for k in (k_lo, k_hi):

            class Calls(torch.nn.Module):

                def forward(self, dummy, k=k):
                    outs = []
                    previous = None
                    for step, args in sets[:k]:
                        if previous is not None:  # a real dependency on the previous call (a layer chain)
                            conv, ssm, mixed, z, gate_a, gate_b = args
                            gate_a = gate_a + (previous.reshape(-1)[:1] * 1e-30).to(gate_a.dtype)
                            args = (conv, ssm, mixed, z, gate_a, gate_b)
                        previous = step(*args)
                        outs.append(previous)
                    return tuple(outs)

            module = Calls()
            if htorch is not None:
                module = htorch.hpu.wrap_in_hpu_graph(module)
            dummy = torch.zeros(1).to(DEVICE)
            for _ in range(2):
                module(dummy)
                sync()
            times = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                module(dummy)
                sync()
                times.append((time.perf_counter() - t0) * 1e3)
            walls[k] = statistics.median(times)
            if a.trace and k == k_hi and DEVICE == "hpu":
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.HPU]) as profiler:
                    for _ in range(3):
                        module(dummy)
                    sync()
                trace_path = out / f"trace_b{batch}_k{k}.json"
                profiler.export_chrome_trace(str(trace_path))
                say(f"{a.label} batch {batch:3d}: trace {trace_path}")
        per_call = (walls[k_hi] - walls[k_lo]) / (k_hi - k_lo) * 1e3
        entry.update(wall_ms=walls, us_per_call=per_call)
        say(f"{a.label} batch {batch:3d} T {a.tokens}: {per_call:8.1f} us per call (replay {walls[k_lo]:.3f} ms at "
            f"K={k_lo}, {walls[k_hi]:.3f} ms at K={k_hi})")
        if a.compare_to is not None:
            for name in ("out", "conv_state", "ssm_state"):
                other = a.compare_to / f"b{batch}_{name}.bin"
                if other.is_file():
                    same = other.read_bytes() == (out / f"b{batch}_{name}.bin").read_bytes()
                    entry.setdefault("same_bits_as_compare", {})[name] = same
                    say(f"{a.label} batch {batch:3d} {name}: {'bit-identical' if same else 'DIFFERS'} to "
                        f"{a.compare_to.name}")
        (out / "bench_gdn_verify.json").write_text(json.dumps(results, indent=1) + "\n")
        del sets
    print("VERDICT gdn-verify " + a.label + ": " + ", ".join(
        f"batch {b} {e['us_per_call']:.1f} us" for b, e in results["batches"].items()), flush=True)
    (out / "bench_gdn_verify.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
