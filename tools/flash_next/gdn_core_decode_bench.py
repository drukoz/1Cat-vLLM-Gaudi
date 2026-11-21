# SPDX-License-Identifier: Apache-2.0
"""Time the two GDN decode ops per call against the bandwidth floor, per TP head count and batch size.

Usage: ``PYTHONPATH=$PWD python tools/flash_next/gdn_core_decode_bench.py [--device auto|cpu|hpu]
[--ops core,recurrent] [--tp 2,8] [--batches 1,16,64] [--iters 100] [--warmup 10] [--repeat 8]
[--bandwidth-tbs 2.45] [--gate sigmoid|silu] [--no-hpu-graph] [--verify-tokens 4]``.

``--ops`` also takes ``verify``: the fused verification op (``gdn_core_verify``, MTP) with ``--verify-tokens``
nodes per request (1 + k draft tokens; default 4), every row an in-place chain row. Its floor is the core op's
floor times the nodes, an estimate (each node loads and stores one state as a decode step does).

On this Mac the script runs the CPU kernels as a smoke test only: CPU times mean nothing, the
Gaudi card gives the numbers the rental compares with the target (at most 2x the floor).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

from flashinfer_gaudi.gdn_core_decode import GdnGateActivation
from tests.unit_tests.gdn_core_decode.sim_case import SimCase, make_inputs, scalars
from tests.unit_tests.gdn_core_decode.verify_sim_case import (VerifySimCase, _lattice_rows, make_verify_inputs,
                                                              verify_scalars)
from tools.flash_next.gdn_core_decode_replay import CoreStep, RecurrentStep, VerifyStep, hpu_present

D = 128
OPS = ("core", "recurrent", "verify")
GATES = {"silu": GdnGateActivation.SILU, "sigmoid": GdnGateActivation.SIGMOID}


def heads(tp: int) -> tuple[int, int, int]:
    """Per-rank head counts and conv channels at TP size ``tp``: ``(H, HV, C)``; ``tp`` must divide 16."""
    if tp < 1 or 16 % tp != 0:
        raise ValueError(f"tp must be >= 1 and divide 16, got {tp}")
    H, HV = 16 // tp, 48 // tp
    return H, HV, 2 * H * D + HV * D


def floor_bytes(op: str, tp: int, batch: int, tokens: int = 1) -> int:
    """Bytes the op must move per call for ``batch`` rows: the memory-bandwidth floor numerator.

    ``verify`` with ``tokens`` nodes per row: the core op's bytes for ``batch * tokens`` rows (an estimate)."""
    _, HV, C = heads(tp)
    B = batch
    if op == "verify":
        return floor_bytes("core", tp, batch * tokens)
    if op == "core":
        extra = B * 3 * C * 2 * 2 + B * C * 2 + B * HV * D * 2
    elif op == "recurrent":
        extra = B * C * 2
    else:
        raise ValueError(f"op must be 'core' or 'recurrent', got {op!r}")
    return B * HV * D * D * 4 * 2 + B * HV * D * 4 + B * HV * 2 * 2 + extra


def bench_case(tp: int, batch: int, gate: GdnGateActivation) -> SimCase:
    """Decode case where every row loads and stores its own state slot (slot 0 unused)."""
    H, HV, _ = heads(tp)
    return SimCase(f"bench_tp{tp}_b{batch}", H, HV, batch + 1, tuple(range(1, batch + 1)),
                   tuple(range(1, batch + 1)), gate, 100 + batch)


def verify_case(tp: int, batch: int, tokens: int, gate: GdnGateActivation) -> VerifySimCase:
    """Verification case of ``batch`` in-place chain rows with ``tokens`` nodes each (the lattice of the HPU cases)."""
    H, HV, _ = heads(tp)
    return VerifySimCase(f"bench_ver_tp{tp}_b{batch}_t{tokens}", H, HV, 1 + batch * tokens, tokens + 2,
                         _lattice_rows(batch, tokens), gate, 200 + batch)


class Repeat(nn.Module):
    """Calls the wrapped step module ``repeat`` times with the same args, keeping graph launch out of the clock."""

    def __init__(self, step: nn.Module, repeat: int):
        super().__init__()
        self.step = step
        self.repeat = int(repeat)

    def forward(self, *args):
        out = None
        for _ in range(self.repeat):
            out = self.step(*args)
        return out


def time_op(op: str, tp: int, batch: int, gate: GdnGateActivation, device: torch.device, iters: int, warmup: int,
            repeat: int, hpu_graph: bool, tokens: int = 4) -> float:
    """Time ``iters`` calls of the op (each call runs the op ``repeat`` times); ms per op call."""
    if op == "verify":
        vcase = verify_case(tp, batch, tokens, gate)
        inputs = {name: tensor.to(device) for name, tensor in make_verify_inputs(vcase).items()}
        scale, eps, gate_activation = verify_scalars(vcase)
    else:
        case = bench_case(tp, batch, gate)
        inputs = {name: tensor.to(device) for name, tensor in make_inputs(case).items()}
        scale, eps, gate_activation = scalars(case)
    if op == "verify":
        step = VerifyStep(inputs["A_log"], inputs["dt_bias"], inputs["conv_weight_t"], inputs["norm_weight"],
                          inputs["row_indices"], inputs["ssm_store_indices"], scale, eps, gate_activation)
        args = (inputs["conv_state"], inputs["ssm_state"], inputs["mixed_qkv"], inputs["z"], inputs["a"], inputs["b"])
    elif op == "core":
        step = CoreStep(inputs["A_log"], inputs["dt_bias"], inputs["conv_weight_t"], inputs["norm_weight"],
                        inputs["load_indices"], inputs["store_indices"], scale, eps, gate_activation)
        args = (inputs["conv_state"], inputs["ssm_state"], inputs["mixed_qkv"], inputs["z"], inputs["a"], inputs["b"])
    else:
        step = RecurrentStep(inputs["A_log"], inputs["dt_bias"], inputs["load_indices"], inputs["store_indices"],
                             scale)
        qkv = torch.nn.functional.silu(inputs["mixed_qkv"].float()).to(torch.bfloat16)
        args = (inputs["ssm_state"], qkv, inputs["a"], inputs["b"])
    module: nn.Module = Repeat(step, repeat)
    hpu = device.type == "hpu"
    if hpu:
        import habana_frameworks.torch as htorch
        if hpu_graph:
            module = htorch.hpu.wrap_in_hpu_graph(module)
    for _ in range(warmup):
        module(*args)
    if hpu:
        htorch.core.mark_step()
        torch.hpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        module(*args)
        if hpu:
            htorch.core.mark_step()
    if hpu:
        torch.hpu.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / (iters * repeat) * 1e3


def slope(batches: list[int], ms: list[float]) -> float:
    """Least-squares slope of ``ms`` over ``batches`` (ms per sequence); needs >= 2 distinct batch sizes."""
    xs = [float(b) for b in batches]
    if len(set(xs)) < 2:
        raise ValueError("slope needs at least 2 distinct batch sizes")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ms) / n
    den = sum((x - mx)**2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ms)) / den


def floor_ms(op: str, tp: int, batch: int, bandwidth_tbs: float, tokens: int = 1) -> float:
    """The bandwidth floor for one call, in ms, at ``bandwidth_tbs`` TB/s."""
    return floor_bytes(op, tp, batch, tokens) / (bandwidth_tbs * 1e12) * 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("auto", "cpu", "hpu"), default="auto")
    parser.add_argument("--ops", default="core,recurrent")
    parser.add_argument("--tp", default="2,8")
    parser.add_argument("--batches", default="1,16,64")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--bandwidth-tbs", type=float, default=2.45)
    parser.add_argument("--gate", choices=("sigmoid", "silu"), default="sigmoid")
    parser.add_argument("--no-hpu-graph", action="store_true")
    parser.add_argument("--verify-tokens", type=int, default=4, help="nodes per request of the verify op (1 + k)")
    ns = parser.parse_args(argv)

    ops = [op.strip() for op in ns.ops.split(",") if op.strip()]
    if not ops or any(op not in OPS for op in ops):
        print(f"--ops must be a comma list of {','.join(OPS)}, got {ns.ops!r}", file=sys.stderr)
        return 2
    try:
        tps = [int(tok) for tok in ns.tp.split(",") if tok.strip()]
        batches = [int(tok) for tok in ns.batches.split(",") if tok.strip()]
    except ValueError:
        print(f"--tp and --batches must be comma lists of ints, got tp={ns.tp!r} batches={ns.batches!r}",
              file=sys.stderr)
        return 2
    if not tps or any(tp < 1 or 16 % tp != 0 for tp in tps):
        print(f"--tp values must each be >= 1 and divide 16, got {ns.tp!r}", file=sys.stderr)
        return 2
    if not batches or any(b < 1 for b in batches):
        print(f"--batches values must each be >= 1, got {ns.batches!r}", file=sys.stderr)
        return 2
    if "verify" in ops and not 1 <= ns.verify_tokens <= 6:
        print(f"--verify-tokens must be 1 to 6 (the verify op's node limit), got {ns.verify_tokens}", file=sys.stderr)
        return 2

    device_str = ns.device
    if device_str == "auto":
        device_str = "hpu" if hpu_present() else "cpu"
    elif device_str == "hpu" and not hpu_present():
        print("--device hpu requested but no Gaudi device is visible", file=sys.stderr)
        return 2
    device = torch.device(device_str)
    if device_str == "hpu":
        import habana_frameworks.torch  # noqa: F401  (registers the HPU device with torch)
        from flashinfer_gaudi._native import load_gdn_fused_extension

        if not load_gdn_fused_extension():
            print("the gdn_fused_ops extension did not load: build it with tools/build_flashinfer_gaudi.py "
                  "--gdn-fused-only, or point GDN_FUSED_GAUDI_KERNEL_LIBRARY and GDN_FUSED_GAUDI_NATIVE_LIBRARY "
                  "at the built files", file=sys.stderr)
            return 2
    hpu_graph = not ns.no_hpu_graph
    gate = GATES[ns.gate]
    print(f"device={device_str} ops={','.join(ops)} tp={','.join(str(tp) for tp in tps)} "
          f"batches={','.join(str(b) for b in batches)} iters={ns.iters} repeat={ns.repeat} "
          f"graph={'on' if hpu_graph else 'off'} bandwidth={ns.bandwidth_tbs} TB/s", flush=True)

    for op in ops:
        tokens = ns.verify_tokens if op == "verify" else 1
        label = f"verify(T={tokens})" if op == "verify" else op
        for tp in tps:
            ms = [
                time_op(op, tp, b, gate, device, ns.iters, ns.warmup, ns.repeat, hpu_graph, ns.verify_tokens)
                for b in batches
            ]
            for b, t in zip(batches, ms):
                f_ms = floor_ms(op, tp, b, ns.bandwidth_tbs, tokens)
                print(f"{label} tp{tp} B={b}: {t:.6f} ms/call, floor {f_ms:.6f} ms, {t / f_ms:.2f}x floor",
                      flush=True)
            if len(set(batches)) >= 2:
                s = slope(batches, ms)
                f1 = floor_ms(op, tp, 1, ns.bandwidth_tbs, tokens)
                print(f"{label} tp{tp} slope: {s:.6f} ms/seq/layer, floor {f1:.6f}, "
                      f"{s / f1:.2f}x floor (target <= 2x)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
