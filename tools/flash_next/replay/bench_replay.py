#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Card microbenchmark: what HPU-graph replay and torch.compile leave of the fixed cost of a TP decode step.

A Flash-Next decode step is served from a replayed HPU graph (lazy mode, ``enforce_eager=False``: the runner
wraps the model with ``wrap_in_hpu_graph(..., disable_tensor_cache=True)``). This script builds a decode-shaped
body of L layers. Each layer has two sub-blocks of (RMS norm, a GEMM down, SiLU, a GEMM up, an all-reduce of
``[rows, 2560]`` bf16, a residual add), the 2 x 48 = 96 layer all-reduces of a Flash-Next step. It times one step
of that body in these regimes on every rank of a tensor-parallel group, each with the collectives and without
them (the same body minus the all-reduces), at ``--rows``:

lazy process (``PT_HPU_LAZY_MODE=1``, ``PT_HPU_ENABLE_LAZY_COLLECTIVES=true``, ``PT_HPU_LAZY_ACC_PAR_MODE=0``):

* ``lazy``          the body issued op by op and one ``mark_step()`` at the end: the host builds the graph
                    every step (what a step outside an HPU graph costs)
* ``lazy_layer``    plus a ``mark_step()`` after every layer (``modify_model_layers`` at
                    ``VLLM_CONFIG_HIDDEN_LAYERS=1``, the serving default)
* ``lazy_coll``     plus a ``mark_step()`` before every collective (``VLLM_HPU_COLLECTIVE_MARK_STEP=1``, the
                    program's default)
* ``replay``, ``replay_layer``, ``replay_coll``  the same three bodies captured once with ``wrap_in_hpu_graph``
                    and replayed, with a new input tensor every call (as the runner hands new inputs each step)

eager process (``PT_HPU_LAZY_MODE=0``):

* ``compile_full``  ``torch.compile(body, backend="hpu_backend", fullgraph=True, dynamic=False)``, functional
                    all-reduces inside
* ``compile_layer`` one layer compiled the same way and called L times per step (regional compilation)

Per call: ``call`` is the time until the call returns on the host, ``total`` until ``torch.hpu.synchronize()``
returns. Derived per regime and row count: one all-reduce (with minus without, over 2L), one cut (``*_layer``
minus the regime without cuts, over L - 1), and the host share of a replay.

Run on a Gaudi2 node, inside the container with ``habana_frameworks``, with ``--ranks`` free cards::

    python3 bench_replay.py --ranks 8 --out replay_r8.json --dump-dir bridge_src
    python3 bench_replay.py --selftest          # no card: gloo ranks on the CPU, checks this script only

Exit code 0 when every rank finished every process, 2 otherwise. Needs only torch and habana_frameworks.
"""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import timedelta

RESULT_TAG = "BENCH_RESULT "
HIDDEN = 2560
PROJ = 320  # a hyper-connection-sized projection: the body's GEMMs are small, as in decode
DEFAULT_LAYERS = 48
DEFAULT_ROWS = (1, 32, 128)
COMPILE_FULL_ROWS = (1, 32)
EXIT_GRACE_S = 20.0
LAZY_REGIMES = ("lazy", "lazy_layer", "lazy_coll", "replay", "replay_layer", "replay_coll", "graph_raw",
                "replay_async", "replay_meta", "replay_tail", "replay_async_tail", "replay_par2", "replay_fused2",
                "replay_coalesce")
# The pair regimes: both sub-blocks of a layer from the same input, so the layer's two all-reduces are independent
# (par2: two calls; fused2: one call over the stacked pair, the same bytes; coalesce: the two calls in a coalescing
# context). Layer.layer_pair. They have the compute of a pair layer, so they compare with each other, not with replay.
PAIR_REGIMES = {"replay_par2": "par2", "replay_fused2": "fused2", "replay_coalesce": "coalesce"}
# derive(): each of these against its reference regime, with the collectives.
COMPARE_WITH = {"graph_raw": "replay", "replay_async": "replay", "replay_meta": "replay",
                "replay_async_tail": "replay_tail", "replay_fused2": "replay_par2", "replay_coalesce": "replay_par2"}
# The sampler stand-in after a replay in the *_tail regimes: lazy ops on [rows, TAIL_COLS] of the output.
TAIL_OPS = 24
TAIL_COLS = 256
# replay_meta passes, besides the input, a namedtuple of META_TENSORS int32 tensors (half [rows, META_COLS], half
# [rows, 1]) and two scalars as a keyword, like the runner's trimmed decode metadata, so the wrapper hashes and
# copies them every call.
META_TENSORS = 20
META_COLS = 1024
EAGER_REGIMES = ("compile_full", "compile_layer")
# The bridge sources whose text decides how a replay runs (read on the card, dumped once by rank 0).
BRIDGE_MODULES = ("habana_frameworks.torch.hpu.graphs", "habana_frameworks.torch.hpu")


def _now() -> float:
    return time.perf_counter()


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def stats_ms(seconds):
    if not seconds:
        return {"p50": None, "p90": None, "min": None, "n": 0}
    ms = sorted(1e3 * s for s in seconds)
    rank = min(max(-(-9 * len(ms) // 10), 1), len(ms))
    return {"p50": _median(ms), "p90": ms[rank - 1], "min": ms[0], "n": len(ms)}


def emit(record: dict) -> None:
    print(RESULT_TAG + json.dumps(record), flush=True)


# ---------------------------------------------------------------------------------------------
# The decode-shaped body.
# ---------------------------------------------------------------------------------------------
class Body:
    """L layers of two sub-blocks: norm, GEMM down, SiLU, GEMM up, optional all-reduce, residual add.

    ``reduce`` is a callable ``tensor -> tensor`` (an in-place all-reduce returning its input, or a functional
    one), or ``None`` for the body without collectives. ``cut`` and ``cut_coll`` are the lazy ``mark_step``
    callables after each layer and before each collective (``None`` = no cut).
    """

    def __init__(self, torch, device, layers: int, world: int, seed: int = 0):
        self.torch = torch
        self.layers = layers
        gen = torch.Generator().manual_seed(seed)
        scale_down = 1.0 / (HIDDEN**0.5)
        # The up projection is scaled by 1/world so the all-reduced value keeps the size of one rank's.
        scale_up = 1.0 / (PROJ**0.5) / world

        def weight(rows, cols, scale):
            return (torch.randn(rows, cols, generator=gen) * scale).to(torch.bfloat16).to(device)

        # Two weight sets shared by every layer: the bytes per step stay small, the graph keeps its shape.
        self.norm_w = [(torch.ones(HIDDEN) + 0.01 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16).to(device)
                       for _ in range(2)]
        self.down = [weight(HIDDEN, PROJ, scale_down) for _ in range(2)]
        self.up = [weight(PROJ, HIDDEN, scale_up) for _ in range(2)]

    def branch(self, x, k: int):
        """RMS norm, GEMM down, SiLU, GEMM up: the part of a sub-block before its all-reduce."""
        torch = self.torch
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        h = (x32 * torch.rsqrt(var + 1e-6)).to(torch.bfloat16) * self.norm_w[k]
        return torch.nn.functional.silu(h @ self.down[k]) @ self.up[k]

    def layer_pair(self, x, reduce, mode: str, coalesce=None):
        """Both sub-blocks from the same input, so their two all-reduces are independent of each other.

        mode ``par2``: two all-reduces; ``fused2``: one all-reduce of the two outputs stacked (the same bytes,
        half the calls); ``coalesce``: the two all-reduces inside ``coalesce()`` (a coalescing context).
        """
        torch = self.torch
        h0 = self.branch(x, 0)
        h1 = self.branch(x, 1)
        if reduce is not None:
            if mode == "fused2":
                both = reduce(torch.cat([h0, h1], 0))
                h0, h1 = both[:x.shape[0]], both[x.shape[0]:]
            elif mode == "coalesce" and coalesce is not None:
                with coalesce():
                    h0 = reduce(h0)
                    h1 = reduce(h1)
            else:
                h0 = reduce(h0)
                h1 = reduce(h1)
        return x + h0 + h1

    def forward_pair(self, x, reduce, mode: str, coalesce=None):
        for _ in range(self.layers):
            x = self.layer_pair(x, reduce, mode, coalesce)
        return x

    def sub_block(self, x, k: int, reduce, cut_coll):
        h = self.branch(x, k)
        if reduce is not None:
            if cut_coll is not None:
                cut_coll()
            h = reduce(h)
        return x + h

    def layer(self, x, reduce, cut_coll=None):
        x = self.sub_block(x, 0, reduce, cut_coll)
        return self.sub_block(x, 1, reduce, cut_coll)

    def forward(self, x, reduce, cut=None, cut_coll=None):
        for _ in range(self.layers):
            x = self.layer(x, reduce, cut_coll)
            if cut is not None:
                cut()
        return x


def count_ops(torch, fn) -> int | None:
    """Dispatched aten ops of one call of fn (CPU tensors only; None when the dispatch mode is unavailable)."""
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:  # noqa: BLE001 - informative only
        return None

    class Counter(TorchDispatchMode):

        def __init__(self):
            super().__init__()
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.n += 1
            return func(*args, **(kwargs or {}))

    counter = Counter()
    with counter:
        fn()
    return counter.n


# ---------------------------------------------------------------------------------------------
# One rank.
# ---------------------------------------------------------------------------------------------
class Rank:

    def __init__(self, args):
        import torch
        import torch.distributed as dist
        self.torch = torch
        self.dist = dist
        self.args = args
        self.rank = args.rank
        self.world = args.ranks
        self.selftest = args.selftest
        self.htorch = None
        if self.selftest:
            self.device, backend = "cpu", "gloo"
        else:
            import habana_frameworks.torch as htorch
            import habana_frameworks.torch.core  # noqa: F401 - mark_step, and the hpu_backend for torch.compile
            try:
                import habana_frameworks.torch.distributed.hccl  # noqa: F401 - registers the backend where needed
            except Exception:  # noqa: BLE001 - importing habana_frameworks.torch registers it on current releases
                pass
            self.htorch = htorch
            torch.hpu.set_device(self.rank)
            self.device, backend = "hpu", "hccl"
        dist.init_process_group(backend=backend, init_method=args.init_method, world_size=self.world,
                                rank=self.rank, timeout=timedelta(seconds=args.timeout))
        self.group = dist.new_group(list(range(self.world)), backend=backend)
        probe = torch.ones(1).to(self.device)
        dist.all_reduce(probe, group=self.group)
        if probe.item() != self.world:
            raise RuntimeError("all-reduce of ones gave %s on rank %d, expected %d" %
                               (probe.item(), self.rank, self.world))
        self.barrier_probe = torch.zeros(1).to(self.device)
        self.body = Body(torch, self.device, args.layers, self.world)

    # ------------------------------------------------------------------------------- primitives
    def mark_step(self) -> None:
        if self.htorch is not None:
            self.htorch.core.mark_step()

    def synchronize(self) -> None:
        if self.htorch is not None:
            self.torch.hpu.synchronize()

    def barrier(self) -> None:
        self.dist.all_reduce(self.barrier_probe, group=self.group)
        self.barrier_probe.item()

    def inplace_reduce(self, tensor):
        self.dist.all_reduce(tensor, group=self.group)
        return tensor

    def functional_reduce(self):
        import torch.distributed._functional_collectives as funcol
        group = self.group

        def reduce(tensor):
            return funcol.all_reduce(tensor, "sum", group)

        return reduce

    def sampler_tail(self, out):
        """TAIL_OPS lazy element-wise ops on a slice of the output, an argmax and a mark_step: int32 [rows]."""
        torch = self.torch
        h = out[:, :TAIL_COLS].float()
        for i in range(TAIL_OPS):
            h = h * 1.0001 + 1e-4 if i % 2 == 0 else h - 1e-4
        tokens = h.argmax(-1).to(torch.int32)
        self.mark_step()
        return tokens

    def coalescing_context(self):
        """torch.distributed's coalescing manager over the TP group (the closest stock form of a grouped call)."""
        manager = getattr(self.dist, "_coalescing_manager", None)
        if manager is None:
            raise RuntimeError("torch.distributed has no _coalescing_manager")
        return manager(group=self.group)

    def meta_inputs(self, rows: int, count: int):
        """Namedtuples shaped like the runner's trimmed decode metadata: META_TENSORS int32 tensors and scalars."""
        torch = self.torch
        fields = ["t%d" % i for i in range(META_TENSORS)] + ["is_prompt", "block_size"]
        meta_type = collections.namedtuple("BenchMetadata", fields)
        metas = []
        for k in range(count):
            tensors = [
                torch.full((rows, META_COLS if i % 2 == 0 else 1), k + i, dtype=torch.int32).to(self.device)
                for i in range(META_TENSORS)
            ]
            metas.append(meta_type(*tensors, False, 128))
        return metas

    def raw_graph(self, step, example):
        """A direct HPUGraph of step with one static input: the replay without the wrapper's hash and lookup."""
        static_in = example.clone()
        if self.htorch is None:  # self-test: no graph, the same call shape

            def call_cpu(x):
                static_in.copy_(x)
                return step(static_in)

            return call_cpu
        hpu = self.htorch.hpu
        step(static_in)  # compile the recipes once outside the capture, as a warmup would
        self.mark_step()
        self.synchronize()
        graph = hpu.HPUGraph()
        stream = hpu.Stream()
        with hpu.stream(stream):
            graph.capture_begin()
            static_out = step(static_in)
            graph.capture_end()
        self.synchronize()

        def call(x):
            static_in.copy_(x)
            graph.replay()
            return static_out

        return call

    def inputs(self, rows: int, count: int = 4):
        torch = self.torch
        gen = torch.Generator().manual_seed(1000 + rows)
        return [(torch.randn(rows, HIDDEN, generator=gen)).to(torch.bfloat16).to(self.device) for _ in range(count)]

    def timed(self, call, inputs, repeats: int, warmup: int) -> dict:
        """call(x) for warmup + repeats inputs; returns first-call, call and total times."""
        self.synchronize()
        self.barrier()
        first = None
        calls, totals = [], []
        out = None
        for i in range(warmup + repeats):
            x = inputs[i % len(inputs)]
            t0 = _now()
            out = call(x)
            t1 = _now()
            self.synchronize()
            t2 = _now()
            if i == 0:
                first = t2 - t0
            elif i >= warmup:
                calls.append(t1 - t0)
                totals.append(t2 - t0)
        finite = None
        if out is not None:
            finite = bool(self.torch.isfinite(out.float()).all().item())
        return {"first_ms": None if first is None else 1e3 * first, "call_ms": stats_ms(calls),
                "total_ms": stats_ms(totals), "finite": finite}

    # ------------------------------------------------------------------------------- lazy process
    def lazy_case(self, regime: str, rows: int, collectives: bool, repeats: int) -> dict:
        body = self.body
        reduce = self.inplace_reduce if collectives else None
        cut = self.mark_step if regime.endswith(("_layer", "_coll")) else None
        cut_coll = self.mark_step if regime.endswith("_coll") else None

        pair_mode = PAIR_REGIMES.get(regime)

        def step(x):
            if pair_mode is not None:
                return body.forward_pair(x, reduce, pair_mode, self.coalescing_context)
            return body.forward(x, reduce, cut, cut_coll)

        inputs = self.inputs(rows)
        torch = self.torch
        if regime in ("replay_tail", "replay_async_tail"):

            class TailStep(torch.nn.Module):

                def forward(self, x):
                    return step(x)

            module = TailStep()
            if self.htorch is not None:
                module = self.htorch.hpu.wrap_in_hpu_graph(module, asynchronous=(regime == "replay_async_tail"),
                                                           disable_tensor_cache=True)

            def call(x):
                # What the runner does after the replayed forward: lazy ops on its output (logits and greedy
                # sampling stand-in), a mark_step, and a read of the tokens to the host.
                return self.sampler_tail(module(x)).cpu()
        elif regime.startswith("lazy"):

            def call(x):
                out = step(x)
                self.mark_step()
                return out
        elif regime == "graph_raw":
            call = self.raw_graph(step, inputs[0])
        elif regime == "replay_meta":
            metas = self.meta_inputs(rows, len(inputs))
            inputs = list(zip(inputs, metas))

            class MetaStep(torch.nn.Module):

                def forward(self, x, meta=None):
                    # Read every metadata tensor once (as the layers read theirs), keep the value unchanged.
                    touch = sum(getattr(meta, name)[:, :1].sum() for name in meta._fields if name.startswith("t"))
                    return step(x + (touch * 0).to(x.dtype))

            module = MetaStep()
            if self.htorch is not None:
                module = self.htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)

            def call(pair):
                return module(pair[0], meta=pair[1])
        else:

            class Step(torch.nn.Module):

                def forward(self, x):
                    return step(x)

            module = Step()
            if self.htorch is not None:
                module = self.htorch.hpu.wrap_in_hpu_graph(module, asynchronous=(regime == "replay_async"),
                                                           disable_tensor_cache=True)
            call = module
        record = self.timed(call, inputs, repeats, warmup=3)
        record.update({"case": regime, "rows": rows, "collectives": collectives, "layers": body.layers,
                       "all_reduces": ((body.layers if pair_mode == "fused2" else 2 * body.layers)
                                       if collectives else 0)})
        return record

    # ------------------------------------------------------------------------------- eager process
    def compile_case(self, regime: str, rows: int, collectives: bool, repeats: int) -> dict:
        torch = self.torch
        body = self.body
        backend = "eager" if self.selftest else "hpu_backend"
        reduce = self.functional_reduce() if collectives else None
        if regime == "compile_full":

            def full(x):
                return body.forward(x, reduce)

            compiled = torch.compile(full, backend=backend, fullgraph=True, dynamic=False)
            call = compiled
        else:

            def one_layer(x):
                return body.layer(x, reduce)

            compiled_layer = torch.compile(one_layer, backend=backend, fullgraph=True, dynamic=False)

            def call(x):
                for _ in range(body.layers):
                    x = compiled_layer(x)
                return x

        inputs = self.inputs(rows)
        record = self.timed(call, inputs, repeats, warmup=3)
        record.update({"case": regime, "rows": rows, "collectives": collectives, "layers": body.layers,
                       "all_reduces": 2 * body.layers if collectives else 0, "backend": backend})
        return record

    def check_values(self) -> dict:
        """The replayed body gives the lazy body's numbers (one input, 1 row, with collectives)."""
        torch = self.torch
        x = self.inputs(1, count=1)[0]
        lazy = self.body.forward(x, self.inplace_reduce)
        self.mark_step()
        lazy = lazy.float().cpu()

        class Step(torch.nn.Module):

            def forward(inner, value):  # noqa: N805 - the body, not the module, holds the state
                return self.body.forward(value, self.inplace_reduce)

        module = Step()
        if self.htorch is not None:
            module = self.htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)
        module(x)
        replay = module(x).float().cpu()
        diff = float((lazy - replay).abs().max())
        record = {"case": "check", "replay_vs_lazy_max_abs": diff, "finite": bool(torch.isfinite(replay).all())}
        # An asynchronous replay followed at once by host-side lazy ops on its output and a read (the runner's
        # sampler pattern) must give the synchronous replay's tokens.
        module_async = Step()
        if self.htorch is not None:
            module_async = self.htorch.hpu.wrap_in_hpu_graph(module_async, asynchronous=True,
                                                             disable_tensor_cache=True)
        module_async(x)
        sync_tokens = self.sampler_tail(module(x)).cpu()
        async_tokens = self.sampler_tail(module_async(x)).cpu()
        record["async_tokens_equal_sync"] = bool(torch.equal(sync_tokens, async_tokens))
        record["async_replay_vs_lazy_max_abs"] = float((lazy - module_async(x).float().cpu()).abs().max())
        return record


def dump_bridge_sources(dump_dir: str) -> dict:
    """Write the bridge's HPU-graph sources and the wrapper's signature (what a replay does is decided there)."""
    import importlib
    os.makedirs(dump_dir, exist_ok=True)
    found = {}
    for name in BRIDGE_MODULES:
        try:
            module = importlib.import_module(name)
            source = inspect.getsource(module)
            path = os.path.join(dump_dir, name.replace(".", "_") + ".py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            found[name] = {"file": getattr(module, "__file__", None), "bytes": len(source)}
        except Exception as exc:  # noqa: BLE001 - informative only
            found[name] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    try:
        import habana_frameworks.torch as htorch
        found["wrap_in_hpu_graph_signature"] = str(inspect.signature(htorch.hpu.wrap_in_hpu_graph))
        found["HPUGraph_members"] = sorted(m for m in dir(htorch.hpu.HPUGraph) if not m.startswith("__"))
    except Exception as exc:  # noqa: BLE001
        found["signature_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
    with open(os.path.join(dump_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(found, fh, indent=1)
    return found


def rank_main(args) -> int:
    try:
        rank = Rank(args)
        repeats = 4 if args.quick else args.repeats
        emit({
            "case": "env", "rank": args.rank, "ranks": args.ranks, "device": rank.device, "process": args.process,
            "lazy_mode_env": os.environ.get("PT_HPU_LAZY_MODE"),
            "acc_par_env": os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE"),
            "lazy_collectives_env": os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES"),
            "selftest": args.selftest
        })
        if args.rank == 0 and args.dump_dir and not args.selftest and args.process == "lazy":
            emit({"case": "bridge_sources", "rank": 0, "found": dump_bridge_sources(args.dump_dir)})
        if args.process == "lazy":
            try:
                record = rank.check_values()
            except Exception as exc:  # noqa: BLE001 - the timings still stand
                record = {"case": "check", "error": "%s: %s" % (type(exc).__name__, str(exc)[:500])}
            record["rank"] = args.rank
            emit(record)
        regimes = LAZY_REGIMES if args.process == "lazy" else EAGER_REGIMES
        for regime in regimes:
            if regime not in args.regimes:
                continue
            for rows in args.rows:
                if regime == "compile_full" and rows not in args.compile_full_rows:
                    continue
                for collectives in (False, True):
                    try:
                        if args.process == "lazy":
                            record = rank.lazy_case(regime, rows, collectives, repeats)
                        else:
                            record = rank.compile_case(regime, rows, collectives, repeats)
                    except Exception as exc:  # noqa: BLE001 - one regime failing must not lose the others
                        record = {"case": regime, "rows": rows, "collectives": collectives,
                                  "error": "%s: %s" % (type(exc).__name__, str(exc)[:600]),
                                  "traceback": traceback.format_exc()[-1500:]}
                        try:
                            rank.mark_step()
                            rank.synchronize()
                        except Exception:  # noqa: BLE001
                            pass
                    record["rank"] = args.rank
                    emit(record)
        rank.barrier()
        emit({"case": "done", "rank": args.rank})
        sys.stdout.flush()
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 - the parent reports it
        emit({"case": "error", "rank": args.rank, "error": "%s: %s" % (type(exc).__name__, str(exc)[:800]),
              "traceback": traceback.format_exc()[-2000:]})
        return 1


# ---------------------------------------------------------------------------------------------
# Parent: start the ranks per process kind, gather their lines, derive, print.
# ---------------------------------------------------------------------------------------------
def rank_environment(args, process: str) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if not args.selftest:
        env["PT_HPU_LAZY_MODE"] = "1" if process == "lazy" else "0"
        env["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"
        if process == "lazy":
            env.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        if args.cards:
            env["HABANA_VISIBLE_MODULES"] = args.cards
    return env


def case_key(record: dict) -> tuple:
    return (record.get("case"), record.get("rows"), record.get("collectives"))


def merge(per_rank: dict) -> list:
    """One entry per case: rank 0's fields plus the slowest rank's and the median rank's p50 of each timing."""
    merged = []
    for record in per_rank.get(0, []):
        if record.get("case") in ("env", "done", "error", "bridge_sources", "check") or "error" in record:
            continue
        entry = dict(record)
        key = case_key(record)
        for field in ("call_ms", "total_ms"):
            medians = [other[field]["p50"] for records in per_rank.values() for other in records
                       if case_key(other) == key and isinstance(other.get(field), dict)
                       and other[field].get("p50") is not None]
            entry[field.replace("_ms", "_p50_max_rank_ms")] = max(medians) if medians else None
            entry[field.replace("_ms", "_p50_median_rank_ms")] = _median(medians) if medians else None
        merged.append(entry)
    return merged


def derive(merged: list) -> list:
    """Per regime and rows: one all-reduce, one cut, and the host share of a call."""

    def total(case, rows, collectives):
        for entry in merged:
            if case_key(entry) == (case, rows, collectives):
                return entry.get("total_p50_max_rank_ms"), entry.get("call_p50_max_rank_ms"), entry.get("layers")
        return None, None, None

    lines = []
    cases = sorted({(entry["case"], entry["rows"]) for entry in merged})
    for case, rows in cases:
        without, _, layers_without = total(case, rows, False)
        with_c, call_with, layers_with = total(case, rows, True)
        layers = layers_with or layers_without
        line = {"case": case, "rows": rows, "total_without_ms": without, "total_with_ms": with_c,
                "call_with_ms": call_with}
        if None not in (without, with_c) and layers:
            calls = next((e.get("all_reduces") for e in merged if case_key(e) == (case, rows, True)), None)
            line["all_reduce_ms_each"] = (with_c - without) / (calls or 2 * layers)
        if with_c and call_with is not None:
            line["host_call_share"] = call_with / with_c
        base = {"lazy_layer": "lazy", "lazy_coll": "lazy_layer", "replay_layer": "replay",
                "replay_coll": "replay_layer"}.get(case)
        if base and layers:
            base_with, _, _ = total(base, rows, True)
            base_without, _, _ = total(base, rows, False)
            cuts = (layers - 1) if case.endswith("_layer") else 2 * layers
            if None not in (base_with, with_c):
                line["cut_ms_each_with_collectives"] = (with_c - base_with) / cuts
            if None not in (base_without, without) and case.endswith("_layer"):
                line["cut_ms_each_without_collectives"] = (without - base_without) / cuts
        ref = COMPARE_WITH.get(case)
        if ref is not None:
            # Against its reference on the same body, with the collectives: the wrapper's own per-call cost
            # (graph_raw), an asynchronous replay (replay_async, replay_async_tail), hashing and copying a
            # decode-metadata-sized input (replay_meta), and one call per pair of all-reduces (fused2, coalesce).
            ref_with, ref_call, _ = total(ref, rows, True)
            line["vs"] = ref
            if None not in (ref_with, with_c):
                line["vs_total_ms"] = with_c - ref_with
            if None not in (ref_call, call_with):
                line["vs_call_ms"] = call_with - ref_call
        lines.append(line)
    return lines


def _fmt(value, digits=3):
    return "n/a" if value is None else ("%.*f" % (digits, value))


def print_summary(merged: list, derived: list, ranks: int, echo=print) -> None:
    echo("== %d ranks; the slowest rank's median per call, ms (call = until the call returns on the host)" % ranks)
    for entry in merged:
        echo("   %-14s rows %3d coll %-5s: total %s  call %s  first %s" %
             (entry["case"], entry["rows"], entry["collectives"], _fmt(entry.get("total_p50_max_rank_ms")),
              _fmt(entry.get("call_p50_max_rank_ms")), _fmt(entry.get("first_ms"), 1)))
    for line in derived:
        echo("   %-14s rows %3d: one all-reduce %s ms; one cut %s ms (with coll) / %s ms (without); host share %s%s" %
             (line["case"], line["rows"], _fmt(line.get("all_reduce_ms_each"), 4),
              _fmt(line.get("cut_ms_each_with_collectives"), 4), _fmt(line.get("cut_ms_each_without_collectives"), 4),
              _fmt(line.get("host_call_share"), 2),
              ("; against %s: total %s ms, call %s ms" %
               (line["vs"], _fmt(line.get("vs_total_ms"), 4), _fmt(line.get("vs_call_ms"), 4)))
              if "vs" in line else ""))


def run_ranks(args, process: str, echo=print) -> dict:
    workdir = tempfile.mkdtemp(prefix="replay-bench-")
    init_method = "file://" + os.path.join(workdir, "dist_init")
    env = rank_environment(args, process)
    procs, per_rank, tails = [], {}, {}
    started = time.monotonic()

    def reader(rank, proc):
        records, tail = per_rank.setdefault(rank, []), tails.setdefault(rank, [])
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(RESULT_TAG):
                try:
                    records.append(json.loads(line[len(RESULT_TAG):]))
                except ValueError:
                    continue
                if rank == 0 and records[-1].get("case") not in ("env", "done"):
                    echo("  [%s rank 0] %s" % (process, json.dumps(records[-1])[:260]))
            else:
                tail.append(line)
                del tail[:-40]

    threads = []
    for rank in range(args.ranks):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--process", process, "--rank", str(rank),
               "--ranks", str(args.ranks), "--init-method", init_method, "--rows",
               ",".join(str(r) for r in args.rows), "--layers", str(args.layers), "--repeats", str(args.repeats),
               "--timeout", str(int(args.timeout)), "--regimes", ",".join(args.regimes),
               "--compile-full-rows", ",".join(str(r) for r in args.compile_full_rows)]
        if args.dump_dir:
            cmd += ["--dump-dir", os.path.abspath(args.dump_dir)]
        if args.selftest:
            cmd.append("--selftest")
        if args.quick:
            cmd.append("--quick")
        proc = subprocess.Popen(cmd, env=env, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        procs.append(proc)
        thread = threading.Thread(target=reader, args=(rank, proc), daemon=True)
        thread.start()
        threads.append(thread)
    deadline = started + args.timeout
    timed_out = False
    all_done_at = None
    while any(proc.poll() is None for proc in procs):
        now = time.monotonic()
        if any(proc.poll() not in (0, None) for proc in procs):
            break
        done_ranks = sum(1 for records in list(per_rank.values()) if any(r.get("case") == "done" for r in records))
        if done_ranks == args.ranks:
            all_done_at = all_done_at or now
            if now - all_done_at > EXIT_GRACE_S:
                break
        if now > deadline:
            timed_out = True
            break
        time.sleep(0.2)
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
    for proc in procs:
        proc.wait()
    for thread in threads:
        thread.join(timeout=10)
    done = sorted(rank for rank, records in per_rank.items() if any(r.get("case") == "done" for r in records))
    errors = {rank: r["error"] for rank, records in per_rank.items() for r in records if r.get("case") == "error"}
    return {"process": process, "ranks": args.ranks, "seconds": round(time.monotonic() - started, 1),
            "timed_out": timed_out, "exit_codes": [proc.returncode for proc in procs], "ranks_done": done,
            "errors": errors, "per_rank": {str(rank): records for rank, records in sorted(per_rank.items())},
            "output_tails": {str(rank): tail[-20:] for rank, tail in sorted(tails.items())}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench_replay.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--ranks", type=int, default=8, help="processes, one card each (default 8)")
    parser.add_argument("--cards", default=None, help="HABANA_VISIBLE_MODULES for the ranks (default: inherited)")
    parser.add_argument("--rows", default=",".join(str(r) for r in DEFAULT_ROWS))
    parser.add_argument("--compile-full-rows", default=",".join(str(r) for r in COMPILE_FULL_ROWS))
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--processes", default="lazy,eager", help="which process kinds to run (default both)")
    parser.add_argument("--regimes", default=",".join(LAZY_REGIMES + EAGER_REGIMES))
    parser.add_argument("--dump-dir", default=None, help="rank 0 writes the bridge's HPU-graph sources here")
    parser.add_argument("--out", default=None, help="write every result to this JSON file")
    parser.add_argument("--timeout", type=float, default=900.0, metavar="S", help="per process kind (default 900)")
    parser.add_argument("--quick", action="store_true", help="few repeats (a smoke run)")
    parser.add_argument("--selftest", action="store_true", help="gloo ranks on the CPU: checks this script only")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--process", default="lazy", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--init-method", default=None, help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    args.rows = [int(token) for token in str(args.rows).split(",") if token.strip()]
    args.compile_full_rows = [int(token) for token in str(args.compile_full_rows).split(",") if token.strip()]
    args.regimes = [token.strip() for token in str(args.regimes).split(",") if token.strip()]
    args.processes = [token.strip() for token in str(args.processes).split(",") if token.strip()]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.child:
        return rank_main(args)
    if args.selftest and args.ranks > 4:
        args.ranks = 2
    runs, merged, derived = [], [], []
    complete = True
    for process in args.processes:
        run = run_ranks(args, process)
        runs.append(run)
        per_rank = {int(rank): records for rank, records in run["per_rank"].items()}
        merged_p = merge(per_rank)
        merged += merged_p
        derived += derive(merged_p)
        if run["ranks_done"] != list(range(args.ranks)):
            complete = False
            print("INCOMPLETE %s: ranks done %s of %d, timed out %s, errors %s" %
                  (process, run["ranks_done"], args.ranks, run["timed_out"], run["errors"]))
            for rank, tail in run["output_tails"].items():
                if int(rank) not in run["ranks_done"]:
                    print("  rank %s last output:" % rank)
                    for line in tail[-10:]:
                        print("     " + line)
    print()
    if args.selftest:
        print("SELFTEST ONLY (gloo on the CPU): the lines below check this script, they are not results.")
    print_summary(merged, derived, args.ranks)
    for run in runs:
        for record in run["per_rank"].get("0", []):
            if record.get("case") == "check":
                print("   check (%s): %s" % (run["process"], json.dumps(record)[:300]))
            if "error" in record and record.get("case") not in ("error",):
                print("   failed %s rows %s coll %s: %s" % (record.get("case"), record.get("rows"),
                                                           record.get("collectives"), record["error"][:300]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tool": "bench_replay", "selftest": args.selftest, "runs": runs, "merged": merged,
                       "derived": derived}, fh, indent=1)
        print("wrote %s" % args.out)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
