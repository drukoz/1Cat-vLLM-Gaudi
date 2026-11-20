#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card probe: what the owned copies in front of the fused hyper-connection ops cost
(VLLM_HPU_HC_FUSED_COPY_MODE 0, 1, 2), and whether a chain runs without them.

    bash tools/flash_next/kernel_probes/with_kernels.sh python tools/flash_next/kernel_probes/probe_hc_copies.py
    PROBE_DEVICE=cpu PYTHONPATH=<tree> python probe_hc_copies.py        # a dry run of this script (times mean nothing)

A chain of K hyper-connections as a decoder layer wires them (vllm_gaudi/models/qwen4_exp/hyperconnection.py,
`_hc_fused_combine_and_mix`), at the real geometry (H 2560, 4 streams, low rank 320), in ONE lazy graph:

    residual, xn    = hc_combine_norm(residual, block_output, injection)   residual and injection: the previous
    down            = linear(xn, W_down)             [T, 336]             hyper-connection's outputs
    lora, injection = hc_silu_split(down)
    gate            = linear(lora, W_up)             [T, 10240]
    block_input     = hc_gate_mix(xn, gate)          [T, 2560]
    block_output    = linear(block_input, W_block).view(-1, H)            a block's output: a GEMM result, viewed

The wrappers of flashinfer_gaudi.hc_fused copy, per hyper-connection: mode 0 all six activations, mode 1 three
(residual, block output, injection), mode 2 one (the block output). Per mode and row count (1, 32, 128):
  * parity: the chain's last residual against mode 0's, bit for bit (the ops and their inputs are the same);
  * time: the device-bound time per hyper-connection, (mark_step + synchronize) / K, the median of the iterations,
    and the host time to issue the K hyper-connections.
An op that raises (a view reached it) is reported with its message; the mode is then not usable as it is.
Last line: "VERDICT hc-copies: ..." with the microseconds per hyper-connection of each mode; a decode step of the
48-layer model has 97.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

H, STREAMS, RANK, PAD = 2560, 4, 320, 12
WIDE = H * STREAMS
ROWS = (1, 32, 128)
MODES = (0, 1, 2)
DEVICE = os.environ.get("PROBE_DEVICE", "hpu")
EPS = 1e-6


def say(text):
    print(f"PROBE hc-copies {text}", flush=True)


def prepare():
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    os.environ.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)
    os.environ.setdefault("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")
    import torch

    if DEVICE == "cpu":
        from tests.unit_tests.hc_fused import hc_ops  # noqa: F401  the oracle as the CPU kernels
        from flashinfer_gaudi import hc_fused

        def flush():
            return None

        return torch, hc_fused, flush, ""
    import habana_frameworks.torch.core as htcore
    from flashinfer_gaudi import _native

    problem = ""
    if not _native.load_hc_fused_extension():
        problem = "; ".join(_native.hc_fused_extension_errors()) or "load_hc_fused_extension returned False"
    from flashinfer_gaudi import hc_fused

    def flush():
        htcore.mark_step()
        torch.hpu.synchronize()

    return torch, hc_fused, flush, problem


class Weights:

    def __init__(self, torch, seed=4242):
        g = torch.Generator().manual_seed(seed)

        def randn(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(DEVICE)

        self.norm = (1.0 + 0.1 * torch.randn(WIDE, generator=g)).to(torch.bfloat16).to(DEVICE)
        self.down = randn(RANK + STREAMS + PAD, WIDE, scale=WIDE**-0.5)
        self.up = randn(WIDE, RANK, scale=RANK**-0.5)
        self.block = randn(H, H, scale=H**-0.5)
        self.g = g
        self.torch = torch

    def inputs(self, rows):
        torch = self.torch
        residual = torch.randn(rows, WIDE, generator=self.g).to(torch.bfloat16).to(DEVICE)
        block = torch.randn(rows, H, generator=self.g).to(torch.bfloat16).to(DEVICE)
        injection = (torch.randn(rows, STREAMS, generator=self.g) * 2).to(torch.bfloat16).to(DEVICE)
        return residual, block, injection


def chain(torch, ops, w, inputs, calls, mode):
    """K hyper-connections of one lazy graph; the last residual (the chain's result)."""
    linear = torch.nn.functional.linear
    kwargs = {"copy_mode": mode} if mode else {}
    residual, block_output, injection = inputs
    for _ in range(calls):
        residual, xn = ops.hc_combine_norm(residual, block_output, injection, w.norm, EPS, STREAMS, **kwargs)
        down = linear(xn, w.down)
        lora, injection = ops.hc_silu_split(down, STREAMS, RANK, **kwargs)
        gate = linear(lora, w.up)
        block_input = ops.hc_gate_mix(xn, gate, STREAMS, **kwargs)
        block_output = linear(block_input, w.block).view(-1, H)
    return residual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "/tmp/probe_hc_copies")))
    ap.add_argument("--calls", type=int, default=16, help="hyper-connections per lazy graph")
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch, ops, flush, problem = prepare()
    say(f"device {DEVICE}; kernels: {'loaded' if not problem else 'NOT loaded (' + problem + ')'}; "
        f"copy modes of this plugin: {getattr(ops, 'COPY_MODES', 'none (an older tree)')}")
    if problem:
        print("VERDICT hc-copies: NOT RUN - the fused hyper-connection kernels did not load", flush=True)
        return 2
    w = Weights(torch)
    flush()
    results = {}
    for rows in ROWS:
        inputs = w.inputs(rows)
        flush()
        base = None
        for mode in MODES:
            key = f"rows{rows}_mode{mode}"
            try:
                out = chain(torch, ops, w, inputs, a.calls, mode).cpu()
                flush()
                if mode == 0:
                    base = out
                same = bool(base is not None and torch.equal(out.view(torch.int16), base.view(torch.int16)))
                worst = float((out.float() - base.float()).abs().max()) if base is not None else float("nan")
                samples = []
                for it in range(a.iters + 2):  # two untimed: the compile, then the first replay of the recipe
                    t0 = time.perf_counter()
                    result = chain(torch, ops, w, inputs, a.calls, mode)
                    t1 = time.perf_counter()
                    flush_start = time.perf_counter()
                    flush()
                    t2 = time.perf_counter()
                    if it >= 2:
                        samples.append((t1 - t0, t2 - flush_start))
                    del result
                samples.sort(key=lambda sample: sample[1])
                issue, device = samples[len(samples) // 2]
                per_call = device / a.calls * 1e6
                results[key] = dict(rows=rows, mode=mode, calls=a.calls, bit_identical_to_mode0=same,
                                    max_abs_diff_to_mode0=worst, device_bound_us_per_hc=per_call,
                                    min_device_bound_us_per_hc=min(s[1] for s in samples) / a.calls * 1e6,
                                    issue_us_per_hc=issue / a.calls * 1e6)
                say(f"rows {rows:4d} mode {mode}: {per_call:8.1f} us per hyper-connection on the device (bound; best "
                    f"{results[key]['min_device_bound_us_per_hc']:.1f}), host issue {issue / a.calls * 1e6:7.1f} us; "
                    f"bit-identical to mode 0: {same} (max abs diff {worst:.3e})")
            except Exception as error:  # noqa: BLE001 - a refusal is the answer for that mode
                text = " ".join(str(error).split())[:400]
                results[key] = dict(rows=rows, mode=mode, error=f"{type(error).__name__}: {text}")
                say(f"rows {rows:4d} mode {mode}: RAISED {type(error).__name__}: {text}")
                with contextlib.suppress(Exception):  # the leftover graph is the one that just failed
                    flush()
    (a.out / "hc_copies.json").write_text(json.dumps(results, indent=1) + "\n")
    parts = []
    for rows in ROWS:
        cells = []
        for mode in MODES:
            r = results[f"rows{rows}_mode{mode}"]
            cells.append(f"mode {mode} " + ("RAISED" if "error" in r else f"{r['device_bound_us_per_hc']:.1f}" +
                                            ("" if r["bit_identical_to_mode0"] else " (NOT the bits of mode 0)")))
        parts.append(f"{rows} rows: " + ", ".join(cells) + " us")
    print("VERDICT hc-copies: " + "; ".join(parts) + " (device-bound time per hyper-connection; x97 per decode step)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
