#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stock HPU-graph replay against the fork's native joint replay, on the same decode-shaped TP4 body.

STATUS: DRAFT, NEVER RUN (2026-11-21, track native-replay). The native mode needs the fork's patched runtime
(tools/flash_next/native_runtime/build_native_runtime.sh) and its adapter; it follows the fork's own check,
tools/communication/check_dsv4_native_allreduce.py, step for step.

The body: two groups of 24 layers; a layer is two sub-blocks, each an element-wise op on ``[rows, 2560]`` bf16 and
an all-reduce of it, so 96 all-reduces per step in two plans of 48 (the joint plan takes at most 128 consumers;
an all-reduce at 4 ranks is two phases). Each sub-block computes ``x = all_reduce((x + 1) * 0.25) - 1`` (4 ranks):
after the first sub-block every rank holds the ranks' mean of the input and keeps it exactly, so the output of a
step that starts from ``full(step + rank)`` is ``step + 1.5`` on every rank (checked).

    # stock: lazy mode, wrap_in_hpu_graph as the runner does (one process per rank; torchrun or the farm's ranks)
    PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=1 torchrun --nproc-per-node 4 bench_native_tp4.py --mode stock --out DIR
    # native: eager mode, the fork's prepared communication and native decoder replay
    PT_HPU_LAZY_MODE=0 torchrun --nproc-per-node 4 bench_native_tp4.py --mode native --bridge EXT.so --out DIR

Each rank writes DIR/<mode>_rank<r>.json: per row count the step time (call, and call + completion), p50 / p90 /
min over the timed steps, and the value check. Per collective = (step - the same body without collectives) / 96;
``--no-collectives`` gives the second number in the same mode.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

HIDDEN = 2560
GROUPS = (24, 24)  # layers per plan
SUBS = 2  # all-reduces per layer


def make_body(reduce, layers: int):

    def body(x):
        for _ in range(layers * SUBS):
            x = reduce((x + 1) * 0.25) - 1
        return x

    return body


def summary(values):
    values = sorted(values)
    return {"p50": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))], "min": values[0],
            "n": len(values)}


def run_stock(args, rank, world, rows):
    import habana_frameworks.torch as htorch

    def reduce(t):
        if args.no_collectives:
            return t * world
        dist.all_reduce(t)
        return t

    class Step(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.groups = [make_body(reduce, layers) for layers in GROUPS]

        def forward(self, x):
            for group in self.groups:
                x = group(x)
            return x

    module = htorch.hpu.wrap_in_hpu_graph(Step(), disable_tensor_cache=True)
    return time_steps(args, rank, rows, lambda x: module(x), lambda out: out.cpu())


def run_native(args, rank, world, rows):
    from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _RUNTIME_ATTR
    from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
    from vllm_gaudi.ops.tp2_prepared_plan import (collect_prepared_group_replays, record_native_decoder_outputs,
                                                  register_tp2_prepared_group_pass, replay_native_decoder)

    bridge = _load_bridge(Path(args.bridge).resolve())
    bridge.set_use_tensor_ids(True)
    bridge.set_prepared_communication(True)
    backend = dist.group.WORLD._get_backend(torch.device("hpu"))
    setattr(torch, _RUNTIME_ATTR, (bridge, backend, bridge.communicator_id(backend)))
    register_tp2_prepared_group_pass()

    def reduce(t):
        if args.no_collectives:
            return t * world
        return torch.ops.vllm_gaudi.tp2_allreduce_plain(t)

    compiled = [torch.compile(make_body(reduce, layers), backend="hpu_backend", fullgraph=True, dynamic=False)
                for layers in GROUPS]
    owner = torch.nn.Identity()
    topology = DecoderTopology("deepseek_v4", GROUPS, SUBS, False)

    def step(x):
        roots = dict(hidden_states=x, positions=None, residual=None, metadata=SimpleNamespace(is_prompt=False))
        result = replay_native_decoder(owner, **roots)
        if result is None:  # the first call captures the two plans
            with collect_prepared_group_replays(owner=owner, adapter=topology,
                                                snapshot=lambda: SimpleNamespace(restore=lambda: None), **roots):
                out = x
                for group in compiled:
                    out = group(out)
                record_native_decoder_outputs(out)
            result = out, None
        ticket = getattr(roots["metadata"], "native_completion", None)
        return result[0], ticket

    def finish(out_ticket):
        out, ticket = out_ticket
        if ticket is not None and not ticket.query():
            ticket.synchronize()
        return out.cpu()

    return time_steps(args, rank, rows, step, finish)


def time_steps(args, rank, rows, call, finish):
    calls, totals, wrong = [], [], 0
    for step in range(args.warmup + args.steps):
        x = torch.full((rows, HIDDEN), float(step + rank), dtype=torch.bfloat16).to("hpu")
        dist.barrier()
        t0 = time.perf_counter()
        out = call(x)
        t1 = time.perf_counter()
        value = finish(out)
        t2 = time.perf_counter()
        expected = float(step) * args.world_scale + 1.5 if not args.no_collectives else None
        if expected is not None and not torch.allclose(value.float(), torch.full_like(value.float(), expected)):
            wrong += 1
        if step >= args.warmup:
            calls.append(1e3 * (t1 - t0))
            totals.append(1e3 * (t2 - t0))
    return {"rows": rows, "call_ms": summary(calls), "total_ms": summary(totals), "wrong_steps": wrong}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("stock", "native"), required=True)
    ap.add_argument("--bridge", help="the fork's adapter (.so), native mode")
    ap.add_argument("--rows", default="1,32")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-collectives", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.world_scale = 1.0
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.hpu.set_device(local)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise SystemExit("bench_native_tp4: 4 ranks (the native TP4 path), got %d" % world)
    runner = run_native if args.mode == "native" else run_stock
    records = [runner(args, rank, world, int(rows)) for rows in args.rows.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "%s%s_rank%d.json" % (args.mode, "_nocoll" if args.no_collectives else "", rank)
    (out / name).write_text(json.dumps({"mode": args.mode, "world": world, "records": records}, indent=1))
    print(json.dumps({"rank": rank, "mode": args.mode, "records": records}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
