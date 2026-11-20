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
* ``replay_seg``    the replayed body cut where each all-reduce would be, without the all-reduces: the recipe
                    launches of a step with collectives, alone (one launch = (replay_seg - replay) / 2L without
                    collectives; one collective alone = (replay with - replay_seg) / 2L)
* ``replay_ag``     the replayed body with every all-reduce written as an all-gather and a local sum (one HCL
                    operation instead of the reduce-scatter and all-gather pair of an all-reduce)
* ``*_cache``       the same regime wrapped with ``disable_tensor_cache=False`` (the runner passes True): the
                    captured tensors keep their device memory between replays
* ``*_work``        the same regime with device work in every sub-block (``--work-n``, ``--work-k``: k products
                    of two n x n matrices that the sub-block's output waits for), so the device needs longer
                    per segment than the host needs to issue it. If the call still returns as early as without
                    the work, the host issues ahead of the device without limit; if the call grows with the
                    work, the host is held back by the device and a cheaper host path saves nothing there
* ``*_busy``        the same regime behind one long device operation (``--busy-n``, ``--busy-k``: k products of
                    two n x n matrices, whose result the body's input waits for). The host has issued every
                    launch and collective of the step before the device starts on them, so total minus the
                    long operation is what the device and the links need for the step when the host costs
                    nothing: the floor that no change of the host path can go below

eager process (``PT_HPU_LAZY_MODE=0``):

* ``compile_full``  ``torch.compile(body, backend="hpu_backend", fullgraph=True, dynamic=False)``, functional
                    all-reduces inside
* ``compile_layer`` one layer compiled the same way and called L times per step (regional compilation)

Per call: ``call`` is the time until the call returns on the host, ``total`` until ``torch.hpu.synchronize()``
returns, ``done`` until one element of the output has been read back to the host (what the runner's read of the
sampled tokens waits for: every launch and collective of the step, on every rank). The derived costs use ``done``.
Derived per regime and row count: one all-reduce (with minus without, over 2L), one cut (``*_layer`` minus the
regime without cuts, over L - 1), and the host share of a replay.

``BENCH_PIN`` in a rank's environment (for example through ``--env-sets pin:BENCH_PIN=core``) places the rank's
threads: ``module`` keeps the whole process on the CPUs of its card's module (module m of 8: the m-th eighth of the
host's CPUs, the slices the card farm gives its containers); ``core`` does the same and then gives the calling
thread the slice's first CPU for itself; ``far`` is ``core`` on the slice of the module four further (the other
half of the host's CPUs: the other NUMA node on a two-node host), which tells whether the node matters. The step ends when the slowest rank has issued, so a rank whose calling
thread is moved or shares its CPU sets the time of all of them.

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
import resource
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
                "replay_coalesce", "replay_seg", "replay_cache", "replay_seg_cache", "replay_layer_cache",
                "replay_busy", "replay_layer_busy", "replay_seg_busy", "replay_ag", "replay_ag_busy",
                "replay_work", "replay_seg_work")
# Regimes that only exist without the collectives (a cut at every collective position and no collective).
NO_COLLECTIVE_REGIMES = ("replay_seg", "replay_seg_cache", "replay_seg_busy", "replay_seg_work")
CACHE_SUFFIX = "_cache"
BUSY_SUFFIX = "_busy"
WORK_SUFFIX = "_work"
DEFAULT_BUSY_N = 8192
DEFAULT_BUSY_K = 4
DEFAULT_WORK_N = 2560
DEFAULT_WORK_K = 1


def split_regime(regime: str) -> tuple:
    """``<shape>[_cache][_busy|_work]`` -> (shape, cache suffix or "", "_busy", "_work" or "")."""
    load = next((suffix for suffix in (BUSY_SUFFIX, WORK_SUFFIX) if regime.endswith(suffix)), "")
    rest = regime[:-len(load)] if load else regime
    cache = CACHE_SUFFIX if rest.endswith(CACHE_SUFFIX) else ""
    shape = rest[:-len(CACHE_SUFFIX)] if cache else rest
    return shape, cache, load
# The pair regimes: both sub-blocks of a layer from the same input, so the layer's two all-reduces are independent
# (par2: two calls; fused2: one call over the stacked pair, the same bytes; coalesce: the two calls in a coalescing
# context). Layer.layer_pair. They have the compute of a pair layer, so they compare with each other, not with replay.
PAIR_REGIMES = {"replay_par2": "par2", "replay_fused2": "fused2", "replay_coalesce": "coalesce"}
# derive(): each of these against its reference regime, with the collectives.
COMPARE_WITH = {"graph_raw": "replay", "replay_async": "replay", "replay_meta": "replay",
                "replay_async_tail": "replay_tail", "replay_fused2": "replay_par2", "replay_coalesce": "replay_par2",
                "replay_cache": "replay", "replay_layer_cache": "replay_layer", "replay_ag": "replay",
                "replay_ag_busy": "replay_busy"}
# An environment set's run of a regime is recorded as "<regime>@<set name>" and compared with the regime itself.
ENV_SEPARATOR = "@"
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


def thread_usage():
    """(CPU seconds of the calling thread, its system seconds, its voluntary context switches).

    The switches and the system time need RUSAGE_THREAD (Linux); elsewhere they are None.
    """
    cpu = time.thread_time()
    who = getattr(resource, "RUSAGE_THREAD", None)
    if who is None:
        return cpu, None, None
    usage = resource.getrusage(who)
    return cpu, usage.ru_stime, usage.ru_nvcsw


TOTAL_MODULES = 8  # the modules of one node: a module's CPUs are the m-th eighth of the host's


def pin_cpus(rank: int, mode: str, allowed=None, total=None, modules=None) -> tuple:
    """(CPUs of the whole process, CPU of the calling thread or None) for ``BENCH_PIN=mode`` on this rank.

    The rank's module is the rank-th of HABANA_VISIBLE_MODULES (ascending), else the rank itself. Its slice is the
    module's eighth of the host's CPUs, cut to the CPUs this process may use; when nothing of it is allowed (a
    container with another CPU set) the allowed CPUs are split evenly over the visible modules instead.
    """
    if allowed is None:
        allowed = sorted(os.sched_getaffinity(0))
    if total is None:
        total = os.cpu_count() or len(allowed)
    if modules is None:
        text = os.environ.get("HABANA_VISIBLE_MODULES", "")
        modules = sorted(int(m) for m in text.split(",") if m.strip().isdigit())
    module = modules[rank] if rank < len(modules) else rank
    if mode == "far":
        module = (module + TOTAL_MODULES // 2) % TOTAL_MODULES
    per = max(total // TOTAL_MODULES, 1)
    cpus = [c for c in allowed if c // per == module]
    if not cpus:
        count = max(len(modules), rank + 1)
        share = max(len(allowed) // count, 1)
        cpus = allowed[rank * share:(rank + 1) * share] or list(allowed)
    if mode in ("core", "far") and len(cpus) > 1:
        return cpus[1:], cpus[0]
    return cpus, None


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
        self.work_w = None  # the operand of the *_work regimes (make_work), used while work_on is set
        self.work_on = False
        self.norm_w = [(torch.ones(HIDDEN) + 0.01 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16).to(device)
                       for _ in range(2)]
        self.down = [weight(HIDDEN, PROJ, scale_down) for _ in range(2)]
        self.up = [weight(PROJ, HIDDEN, scale_up) for _ in range(2)]

    def make_busy(self, device, n: int, k: int, seed: int = 7) -> None:
        """The operand of the long device operation of the ``*_busy`` regimes (n x n bf16)."""
        torch = self.torch
        gen = torch.Generator().manual_seed(seed)
        self.busy_w = (torch.randn(n, n, generator=gen) / (n**0.5)).to(torch.bfloat16).to(device)
        self.busy_k = k

    def make_work(self, device, n: int, k: int, seed: int = 11) -> None:
        """The operand of the device work of the ``*_work`` regimes (n x n bf16), one product chain per sub-block."""
        torch = self.torch
        gen = torch.Generator().manual_seed(seed)
        self.work_w = (torch.randn(n, n, generator=gen) / (n**0.5)).to(torch.bfloat16).to(device)
        self.work_k = k

    def delay(self, x):
        """x after a long device operation it now depends on (k products of n x n matrices); the value is x.

        One element of the last product times zero is added, so every later operation of the step waits for
        the products while the host goes on issuing.
        """
        y = self.busy_w
        for _ in range(self.busy_k):
            y = y @ self.busy_w
        return x + (y[:1, :1].float().sum() * 0.0).to(x.dtype)

    def branch(self, x, k: int):
        """RMS norm, GEMM down, SiLU, GEMM up: the part of a sub-block before its all-reduce."""
        torch = self.torch
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        h = (x32 * torch.rsqrt(var + 1e-6)).to(torch.bfloat16) * self.norm_w[k]
        out = torch.nn.functional.silu(h @ self.down[k]) @ self.up[k]
        if self.work_on and self.work_w is not None:
            # Device work this sub-block's output (and so its all-reduce) waits for; the value is unchanged.
            y = self.work_w
            for _ in range(self.work_k):
                y = y @ self.work_w
            out = out + (y[:1, :1].float().sum() * 0.0).to(out.dtype)
        return out

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

    def sub_block(self, x, k: int, reduce, cut_coll, cut_seg=None):
        h = self.branch(x, k)
        if cut_seg is not None:
            cut_seg()  # a cut where the all-reduce is (or would be), whether or not there is one
        if reduce is not None:
            if cut_coll is not None:
                cut_coll()
            h = reduce(h)
        return x + h

    def layer(self, x, reduce, cut_coll=None, cut_seg=None):
        x = self.sub_block(x, 0, reduce, cut_coll, cut_seg)
        return self.sub_block(x, 1, reduce, cut_coll, cut_seg)

    def forward(self, x, reduce, cut=None, cut_coll=None, cut_seg=None):
        for _ in range(self.layers):
            x = self.layer(x, reduce, cut_coll, cut_seg)
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
        # One rank alone (a one-card run of the launch path): no process group, the reduction is the identity.
        self.group = None
        if self.world > 1:
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
        if any(regime.endswith(BUSY_SUFFIX) for regime in args.regimes):
            self.body.make_busy(self.device, args.busy_n, args.busy_k)
        if any(regime.endswith(WORK_SUFFIX) for regime in args.regimes):
            self.body.make_work(self.device, args.work_n, args.work_k)

    # ------------------------------------------------------------------------------- primitives
    def mark_step(self) -> None:
        if self.htorch is not None:
            self.htorch.core.mark_step()

    def synchronize(self) -> None:
        if self.htorch is not None:
            self.torch.hpu.synchronize()

    def read_back(self, out) -> None:
        """One element of the output on the host (a tensor, or the first tensor of a tuple or list)."""
        torch = self.torch
        while isinstance(out, (tuple, list)) and out:
            out = out[0]
        if torch.is_tensor(out) and out.device.type != "cpu" and out.numel():
            out.reshape(-1)[:1].float().cpu()

    def barrier(self) -> None:
        if self.group is None:
            return
        self.dist.all_reduce(self.barrier_probe, group=self.group)
        self.barrier_probe.item()

    def inplace_reduce(self, tensor):
        if self.group is not None:
            self.dist.all_reduce(tensor, group=self.group)
        return tensor

    def gather_sum_reduce(self, tensor):
        """The all-reduce as an all-gather of every rank's tensor and a sum on the device: the same value."""
        if self.group is None:
            return tensor
        torch = self.torch
        rows = tensor.shape[0]
        out = torch.empty((self.world * rows, ) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        self.dist.all_gather_into_tensor(out, tensor, group=self.group)
        return out.view((self.world, rows) + tuple(tensor.shape[1:])).sum(0)

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
        calls, totals, dones = [], [], []
        # Where the host's time goes: CPU seconds of the calling thread during the call and during call plus
        # synchronize, CPU seconds of the whole process (every thread) during both, the calling thread's system
        # time and its voluntary context switches (a thread that waits sleeps; one that issues or polls burns CPU).
        cpu_calls, cpu_totals, cpu_procs, sys_totals, switches = [], [], [], [], []
        where = getattr(os, "sched_getcpu", None)
        seen_cpus, moves, last_cpu = set(), 0, None
        out = None
        for i in range(warmup + repeats):
            x = inputs[i % len(inputs)]
            p0 = time.process_time()
            c0, s0, v0 = thread_usage()
            t0 = _now()
            out = call(x)
            t1 = _now()
            c1 = time.thread_time()
            self.synchronize()
            t2 = _now()
            c2, s2, v2 = thread_usage()
            p2 = time.process_time()
            # A read of the output, as the runner reads the sampled tokens: the step is complete on every rank
            # when it returns (synchronize alone may return while launches behind a collective still wait), and
            # the next call starts with nothing in flight.
            self.read_back(out)
            t3 = _now()
            if where is not None:
                cpu_now = where()
                moves += int(last_cpu is not None and cpu_now != last_cpu)
                last_cpu = cpu_now
                seen_cpus.add(cpu_now)
            if i == 0:
                first = t2 - t0
            elif i >= warmup:
                calls.append(t1 - t0)
                totals.append(t2 - t0)
                dones.append(t3 - t0)
                cpu_calls.append(c1 - c0)
                cpu_totals.append(c2 - c0)
                cpu_procs.append(p2 - p0)
                if s0 is not None:
                    sys_totals.append(s2 - s0)
                    switches.append(v2 - v0)
        finite = None
        if out is not None:
            finite = bool(self.torch.isfinite(out.float()).all().item())
        return {"first_ms": None if first is None else 1e3 * first, "call_ms": stats_ms(calls),
                "total_ms": stats_ms(totals), "done_ms": stats_ms(dones), "finite": finite,
                "cpu": {"call_ms": _median([1e3 * v for v in cpu_calls]),
                        "total_ms": _median([1e3 * v for v in cpu_totals]),
                        "process_ms": _median([1e3 * v for v in cpu_procs]),
                        "thread_sys_ms": _median([1e3 * v for v in sys_totals]),
                        "thread_voluntary_switches": _median(switches),
                        # the CPUs the calling thread was seen on after the calls, and how often it had moved
                        "cpus_seen": sorted(seen_cpus), "cpu_moves": moves}}

    # ------------------------------------------------------------------------------- lazy process
    def lazy_case(self, regime: str, rows: int, collectives: bool, repeats: int) -> dict:
        body = self.body
        reduce = self.inplace_reduce if collectives else None
        # "<regime>_cache": the same regime with the wrapper's tensor cache on (disable_tensor_cache=False).
        shape, cache_suffix, load_suffix = split_regime(regime)
        busy_suffix = load_suffix == BUSY_SUFFIX
        body.work_on = load_suffix == WORK_SUFFIX  # read while the graph is captured (the first call)
        tensor_cache = bool(cache_suffix)
        if collectives and shape == "replay_ag":
            reduce = self.gather_sum_reduce
        cut = self.mark_step if shape.endswith(("_layer", "_coll")) else None
        cut_coll = self.mark_step if shape.endswith("_coll") else None
        cut_seg = self.mark_step if shape.endswith("_seg") else None

        pair_mode = PAIR_REGIMES.get(regime)

        def step(x):
            if busy_suffix:
                x = body.delay(x)
            if pair_mode is not None:
                return body.forward_pair(x, reduce, pair_mode, self.coalescing_context)
            return body.forward(x, reduce, cut, cut_coll, cut_seg)

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
                                                           disable_tensor_cache=not tensor_cache)
            call = module
        try:
            record = self.timed(call, inputs, repeats, warmup=3)
        finally:
            body.work_on = False
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

    def trace_replay(self, trace_dir: str, rows: int, calls: int) -> dict:
        """A few replays of the body with its all-reduces under torch.profiler (CPU and HPU) on ranks 0 and 1.

        Every rank replays (the collectives need all of them); ranks 0 and 1 record and write a Chrome trace, which
        shows what the host thread does during a replay: issuing launches and collectives, or waiting on them.
        """
        import contextlib
        torch = self.torch
        body = self.body

        class Step(torch.nn.Module):

            def forward(inner, x):  # noqa: N805 - the body, not the module, holds the state
                return body.forward(x, self.inplace_reduce)

        module = Step()
        if self.htorch is not None:
            module = self.htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)
        inputs = self.inputs(rows)
        for i in range(3):
            module(inputs[i % len(inputs)])
        self.synchronize()
        self.barrier()
        record = self.rank in (0, 1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.htorch is not None:
            activities.append(torch.profiler.ProfilerActivity.HPU)
        profiler = torch.profiler.profile(activities=activities) if record else contextlib.nullcontext()
        with profiler as prof:
            for i in range(calls):
                with torch.profiler.record_function("replay_call_%d" % i):
                    module(inputs[i % len(inputs)])
                with torch.profiler.record_function("replay_sync_%d" % i):
                    self.synchronize()
        path = None
        if record:
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, "replay_rows%d_rank%d.json" % (rows, self.rank))
            prof.export_chrome_trace(path)
        return {"case": "trace", "rows": rows, "calls": calls, "path": path}

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
        record.update(self.cpu_reference(x, replay))
        return record

    def cpu_reference(self, x, replay) -> dict:
        """The replayed output against the same body on the CPU, with and without the reductions.

        Every rank holds the same weights and the same input, so an all-reduce multiplies by the world size. The
        CPU body with that factor is what the card must give; the CPU body with factor 1 is what it gives when a
        collective did nothing (a setting that skips the hardware). bf16 GEMMs differ a little between CPU and
        card, so the two distances are reported and the smaller one names the case.
        """
        torch = self.torch
        try:
            cpu_body = Body(torch, "cpu", self.body.layers, self.world)
            x_cpu = x.detach().to("cpu")
            world = self.world
            reduced = cpu_body.forward(x_cpu, lambda t: t * world).float()
            unreduced = cpu_body.forward(x_cpu, lambda t: t).float()
            scale = float(reduced.abs().max()) or 1.0
            return {"cpu_reference_max_abs": scale,
                    "replay_vs_cpu_reduced_max_abs": float((replay - reduced).abs().max()),
                    "replay_vs_cpu_unreduced_max_abs": float((replay - unreduced).abs().max())}
        except Exception as exc:  # noqa: BLE001 - informative only
            return {"cpu_reference_error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}

    def hold_loop(self, path: str, rows: int, calls: int, regime: str = "replay") -> dict:
        """calls replays of the body, back to back, while an outside sampler looks.

        regime ``replay``: the body with its all-reduces; ``replay_seg``: cut at every collective position, no
        collective (the launches alone; the form for one rank).

        Rank 0 writes ``path`` (its pid and the start time) before the loop and ``path + ".done"`` after it, so
        a script outside the process (thread CPU times from /proc, native stacks, a syscall count) knows when the
        process is inside replays and nothing else. Every rank runs the same number of calls: the collectives
        keep the ranks in step.
        """
        torch = self.torch
        body = self.body

        segments = regime == "replay_seg"

        class Step(torch.nn.Module):

            def forward(inner, x):  # noqa: N805 - the body, not the module, holds the state
                if segments:
                    return body.forward(x, None, cut_seg=self.mark_step)
                return body.forward(x, self.inplace_reduce)

        module = Step()
        if self.htorch is not None:
            module = self.htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)
        inputs = self.inputs(rows)
        for i in range(3):
            module(inputs[i % len(inputs)])
        self.synchronize()
        self.barrier()
        if self.rank == 0:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "start": time.time(), "calls": calls, "rows": rows,
                           "regime": regime}, fh)
        started = _now()
        cpu0, sys0, switches0 = thread_usage()
        process0 = time.process_time()
        for i in range(calls):
            out = module(inputs[i % len(inputs)])
            self.synchronize()
            self.read_back(out)
        seconds = _now() - started
        cpu1, sys1, switches1 = thread_usage()
        record = {"case": "hold", "regime": regime, "rows": rows, "calls": calls, "seconds": seconds,
                  "ms_per_call": 1e3 * seconds / max(calls, 1),
                  "thread_cpu_ms_per_call": 1e3 * (cpu1 - cpu0) / max(calls, 1),
                  "process_cpu_ms_per_call": 1e3 * (time.process_time() - process0) / max(calls, 1)}
        if sys0 is not None:
            record["thread_sys_ms_per_call"] = 1e3 * (sys1 - sys0) / max(calls, 1)
            record["thread_voluntary_switches_per_call"] = (switches1 - switches0) / max(calls, 1)
        if self.rank == 0:
            with open(path + ".done", "w", encoding="utf-8") as fh:
                json.dump({"end": time.time(), "seconds": seconds}, fh)
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
        pin_mode = os.environ.get("BENCH_PIN", "")
        pinned, own_cpu = None, None
        if pin_mode in ("module", "core", "far") and hasattr(os, "sched_setaffinity"):
            # Before torch and the bridge start their threads: they inherit the calling thread's CPUs.
            pinned, own_cpu = pin_cpus(args.rank, pin_mode)
            os.sched_setaffinity(0, pinned)
        rank = Rank(args)
        if own_cpu is not None:
            os.sched_setaffinity(0, {own_cpu})  # the calling thread alone (pid 0 names the calling thread)
        repeats = 4 if args.quick else args.repeats
        emit({
            "case": "env", "rank": args.rank, "ranks": args.ranks, "device": rank.device, "process": args.process,
            "lazy_mode_env": os.environ.get("PT_HPU_LAZY_MODE"),
            "acc_par_env": os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE"),
            "lazy_collectives_env": os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES"),
            "selftest": args.selftest, "pin": pin_mode or None, "pin_cpus": pinned, "pin_own_cpu": own_cpu,
            "allowed_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "visible_modules": os.environ.get("HABANA_VISIBLE_MODULES"),
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
                    if collectives and (regime in NO_COLLECTIVE_REGIMES or args.collectives == "off"):
                        continue
                    if not collectives and args.collectives == "on":
                        continue
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
        if args.hold_file and args.hold_calls > 0 and args.process == "lazy":
            try:
                record = rank.hold_loop(args.hold_file, args.hold_rows, args.hold_calls, args.hold_regime)
            except Exception as exc:  # noqa: BLE001 - the timings still stand
                record = {"case": "hold", "error": "%s: %s" % (type(exc).__name__, str(exc)[:500])}
            record["rank"] = args.rank
            emit(record)
        if args.trace_dir and args.process == "lazy":
            try:
                record = rank.trace_replay(args.trace_dir, args.trace_rows, args.trace_calls)
            except Exception as exc:  # noqa: BLE001 - the timings still stand
                record = {"case": "trace", "error": "%s: %s" % (type(exc).__name__, str(exc)[:500])}
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
def rank_environment(args, process: str, extra_env=None) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if not args.selftest:
        env["PT_HPU_LAZY_MODE"] = "1" if process == "lazy" else "0"
        env["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"
        if process == "lazy":
            env.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        if args.cards:
            env["HABANA_VISIBLE_MODULES"] = args.cards
    env.update(extra_env or {})
    return env


def parse_env_sets(text: str) -> list:
    """``name:KEY=VALUE,KEY=VALUE;name2:KEY=VALUE`` -> [(name, {KEY: VALUE})]: runs of the lazy process under
    another environment (bridge, Synapse or HCL settings), each recorded as ``<regime>@<name>``."""
    sets = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, pairs = chunk.partition(":")
        name = name.strip()
        env = {}
        for pair in pairs.split(","):
            key, sep, value = pair.strip().partition("=")
            if not sep or not key:
                raise ValueError("environment set %r: %r is not KEY=VALUE" % (name, pair))
            env[key] = value
        if not name or not env or ENV_SEPARATOR in name:
            raise ValueError("environment set %r needs a name and at least one KEY=VALUE" % chunk)
        sets.append((name, env))
    return sets


def case_key(record: dict) -> tuple:
    return (record.get("case"), record.get("rows"), record.get("collectives"))


def merge(per_rank: dict) -> list:
    """One entry per case: rank 0's fields plus the slowest rank's and the median rank's p50 of each timing."""
    merged = []
    for record in per_rank.get(0, []):
        if record.get("case") in ("env", "done", "error", "bridge_sources", "check", "hold", "trace") \
                or "error" in record:
            continue
        entry = dict(record)
        key = case_key(record)
        for field in ("call_ms", "total_ms", "done_ms"):
            medians = [other[field]["p50"] for records in per_rank.values() for other in records
                       if case_key(other) == key and isinstance(other.get(field), dict)
                       and other[field].get("p50") is not None]
            entry[field.replace("_ms", "_p50_max_rank_ms")] = max(medians) if medians else None
            entry[field.replace("_ms", "_p50_median_rank_ms")] = _median(medians) if medians else None
        merged.append(entry)
    return merged


def derive(merged: list, reference: list | None = None) -> list:
    """Per regime and rows: one all-reduce, one cut, one launch, and the host share of a call.

    reference: entries of other runs to compare with (the baseline, for an environment set's entries).
    """
    lookup = list(merged) + list(reference or [])

    def total(case, rows, collectives):
        # The end of a call: the read of the output when it was measured (done), else synchronize (total).
        for entry in lookup:
            if case_key(entry) == (case, rows, collectives):
                end = entry.get("done_p50_max_rank_ms")
                if end is None:
                    end = entry.get("total_p50_max_rank_ms")
                return end, entry.get("call_p50_max_rank_ms"), entry.get("layers")
        return None, None, None

    lines = []
    cases = sorted({(entry["case"], entry["rows"]) for entry in merged})
    for case, rows in cases:
        without, _, layers_without = total(case, rows, False)
        with_c, call_with, layers_with = total(case, rows, True)
        layers = layers_with or layers_without
        _, call_without, _ = total(case, rows, False)
        line = {"case": case, "rows": rows, "total_without_ms": without, "total_with_ms": with_c,
                "call_with_ms": call_with, "call_without_ms": call_without}
        if None not in (without, with_c) and layers:
            calls = next((e.get("all_reduces") for e in merged if case_key(e) == (case, rows, True)), None)
            line["all_reduce_ms_each"] = (with_c - without) / (calls or 2 * layers)
        if with_c and call_with is not None:
            line["host_call_share"] = call_with / with_c
        name, _, env_name = case.partition(ENV_SEPARATOR)
        suffix = (ENV_SEPARATOR + env_name) if env_name else ""
        shape, cache, load = split_regime(name)
        cache = cache + load  # the uncut body and the base regime of the same family
        if load == BUSY_SUFFIX:
            # Behind the long device operation the host is done issuing before the device starts on the step.
            line["busy_ms"], _, _ = total("replay" + cache + suffix, rows, False)
        if shape == "replay_seg" and layers:
            # A cut at each of the 2L collective positions and no collective: the launches of a step with
            # collectives. One launch = (this - the uncut body) / 2L; one collective alone = (the body with
            # its collectives - this) / 2L, both without the launch the collective forces.
            uncut_without, _, _ = total("replay" + cache + suffix, rows, False)
            uncut_with, _, _ = total("replay" + cache + suffix, rows, True)
            _, call_without, _ = total(case, rows, False)
            if None not in (uncut_without, without):
                line["launch_ms_each"] = (without - uncut_without) / (2 * layers)
            if None not in (uncut_with, without):
                line["collective_alone_ms_each"] = (uncut_with - without) / (2 * layers)
            if without and call_without is not None:
                line["host_call_share"] = call_without / without
        base = {"lazy_layer": "lazy", "lazy_coll": "lazy_layer", "replay_layer": "replay",
                "replay_coll": "replay_layer"}.get(shape)
        if base and layers:
            base = base + cache + suffix
            base_with, _, _ = total(base, rows, True)
            base_without, _, _ = total(base, rows, False)
            cuts = (layers - 1) if shape.endswith("_layer") else 2 * layers
            if None not in (base_with, with_c):
                line["cut_ms_each_with_collectives"] = (with_c - base_with) / cuts
            if None not in (base_without, without) and shape.endswith("_layer"):
                line["cut_ms_each_without_collectives"] = (without - base_without) / cuts
        ref = COMPARE_WITH.get(case)
        if env_name:
            ref = name  # an environment set's run of a regime against the baseline's run of it
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
            ref_without, _, _ = total(ref, rows, False)
            if None not in (ref_without, without):
                line["vs_total_without_ms"] = without - ref_without
        lines.append(line)
    return lines


def _fmt(value, digits=3):
    return "n/a" if value is None else ("%.*f" % (digits, value))


def print_summary(merged: list, derived: list, ranks: int, echo=print) -> None:
    echo("== %d ranks; the slowest rank's median per call, ms (call = until the call returns on the host, total = "
         "until synchronize returns, done = until a read of the output returns)" % ranks)
    echo("   (cpu, rank 0: CPU ms of the calling thread in the call / in call + synchronize, of the whole process,"
         " the thread's system ms, its voluntary context switches)")
    for entry in merged:
        cpu = entry.get("cpu") or {}
        echo("   %-22s rows %3d coll %-5s: done %s  total %s  call %s  first %s  cpu %s / %s  proc %s  sys %s  sw %s" %
             (entry["case"], entry["rows"], entry["collectives"], _fmt(entry.get("done_p50_max_rank_ms")),
              _fmt(entry.get("total_p50_max_rank_ms")),
              _fmt(entry.get("call_p50_max_rank_ms")), _fmt(entry.get("first_ms"), 1), _fmt(cpu.get("call_ms")),
              _fmt(cpu.get("total_ms")), _fmt(cpu.get("process_ms")), _fmt(cpu.get("thread_sys_ms")),
              _fmt(cpu.get("thread_voluntary_switches"), 0)))
    for line in derived:
        echo("   %-22s rows %3d: one all-reduce %s ms; one cut %s ms (with coll) / %s ms (without); host share %s%s%s" %
             (line["case"], line["rows"], _fmt(line.get("all_reduce_ms_each"), 4),
              _fmt(line.get("cut_ms_each_with_collectives"), 4), _fmt(line.get("cut_ms_each_without_collectives"), 4),
              _fmt(line.get("host_call_share"), 2),
              ("; one launch %s ms, one collective alone %s ms" %
               (_fmt(line.get("launch_ms_each"), 4), _fmt(line.get("collective_alone_ms_each"), 4)))
              if "launch_ms_each" in line or "collective_alone_ms_each" in line else "",
              ("; against %s: total %s ms (without coll %s), call %s ms" %
               (line["vs"], _fmt(line.get("vs_total_ms"), 4), _fmt(line.get("vs_total_without_ms"), 4),
                _fmt(line.get("vs_call_ms"), 4)))
              if "vs" in line else "") +
             ("; long operation %s ms, host call %s ms (a device floor only when the call is the shorter)" %
              (_fmt(line.get("busy_ms")), _fmt(line.get("call_with_ms") if line.get("call_with_ms") is not None
                                                 else line.get("call_without_ms")))
              if "busy_ms" in line else ""))


def run_ranks(args, process: str, echo=print, env_set=None) -> dict:
    """Start the ranks of one process kind and gather their records.

    env_set ``(name, env)``: the lazy process under that environment, only ``--env-regimes`` at ``--env-rows``,
    without the hold loop and the trace, within ``--env-timeout`` seconds.
    """
    workdir = tempfile.mkdtemp(prefix="replay-bench-")
    init_method = "file://" + os.path.join(workdir, "dist_init")
    env = rank_environment(args, process, env_set[1] if env_set else None)
    regimes = args.env_regimes if env_set else args.regimes
    rows = args.env_rows if env_set else args.rows
    timeout = args.env_timeout if env_set else args.timeout
    label = process if not env_set else "%s%s%s" % (process, ENV_SEPARATOR, env_set[0])
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
                    echo("  [%s rank 0] %s" % (label, json.dumps(records[-1])[:260]))
            else:
                tail.append(line)
                del tail[:-40]

    threads = []
    for rank in range(args.ranks):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--process", process, "--rank", str(rank),
               "--ranks", str(args.ranks), "--init-method", init_method, "--rows",
               ",".join(str(r) for r in rows), "--layers", str(args.layers), "--repeats", str(args.repeats),
               "--timeout", str(int(timeout)), "--regimes", ",".join(regimes),
               "--compile-full-rows", ",".join(str(r) for r in args.compile_full_rows),
               "--collectives", args.collectives, "--busy-n", str(args.busy_n), "--busy-k", str(args.busy_k),
               "--work-n", str(args.work_n), "--work-k", str(args.work_k)]
        if args.dump_dir and not env_set:
            cmd += ["--dump-dir", os.path.abspath(args.dump_dir)]
        if args.trace_dir and not env_set:
            cmd += ["--trace-dir", os.path.abspath(args.trace_dir), "--trace-rows", str(args.trace_rows),
                    "--trace-calls", str(args.trace_calls)]
        if args.hold_file and args.hold_calls > 0 and not env_set:
            cmd += ["--hold-file", os.path.abspath(args.hold_file), "--hold-calls", str(args.hold_calls),
                    "--hold-rows", str(args.hold_rows), "--hold-regime", args.hold_regime]
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
    deadline = started + timeout
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
    if env_set:
        # The records of an environment set carry its name, so they stand next to the baseline's.
        for records in per_rank.values():
            for record in records:
                if record.get("case") not in ("env", "done", "error", "bridge_sources", "check", None):
                    record["case"] = "%s%s%s" % (record["case"], ENV_SEPARATOR, env_set[0])
                record["env_set"] = env_set[0]
    return {"process": label, "env": env_set[1] if env_set else None,
            "ranks": args.ranks, "seconds": round(time.monotonic() - started, 1),
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
    parser.add_argument("--trace-dir", default=None,
                        help="ranks 0 and 1 write a torch.profiler trace of a few replays (with collectives) here")
    parser.add_argument("--trace-rows", type=int, default=1)
    parser.add_argument("--trace-calls", type=int, default=5)
    parser.add_argument("--collectives", default="both", choices=("both", "off", "on"),
                        help="time each regime without and with its all-reduces (both), or only one of them")
    parser.add_argument("--hold-file", default=None,
                        help="after the regimes, every rank replays --hold-calls times back to back; rank 0 writes "
                        "this file (its pid) before and FILE.done after, for a sampler outside the process")
    parser.add_argument("--hold-calls", type=int, default=0)
    parser.add_argument("--hold-rows", type=int, default=1)
    parser.add_argument("--hold-regime", default="replay", choices=("replay", "replay_seg"))
    parser.add_argument("--busy-n", type=int, default=DEFAULT_BUSY_N,
                        help="the *_busy regimes: size of the matrices of the long device operation")
    parser.add_argument("--busy-k", type=int, default=DEFAULT_BUSY_K, help="and how many products of them")
    parser.add_argument("--work-n", type=int, default=DEFAULT_WORK_N,
                        help="the *_work regimes: size of the matrices of the device work of every sub-block")
    parser.add_argument("--work-k", type=int, default=DEFAULT_WORK_K, help="and how many products of them")
    parser.add_argument("--env-sets", default="",
                        help="more runs of the lazy process under other settings: name:KEY=VALUE,KEY=VALUE;name2:...")
    parser.add_argument("--env-regimes", default="replay", help="the regimes of the --env-sets runs")
    parser.add_argument("--env-rows", default=None, help="their rows (default: --rows)")
    parser.add_argument("--env-timeout", type=float, default=240.0, metavar="S",
                        help="per --env-sets run (default 240)")
    parser.add_argument("--env-pause", type=float, default=15.0, metavar="S",
                        help="wait after an --env-sets run that did not end cleanly (the cards must be released)")
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
    args.env_regimes = [token.strip() for token in str(args.env_regimes).split(",") if token.strip()]
    args.env_rows = ([int(token) for token in str(args.env_rows).split(",") if token.strip()]
                     if args.env_rows else list(args.rows))
    args.env_sets = parse_env_sets(args.env_sets)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.child:
        return rank_main(args)
    if args.selftest and args.ranks > 4:
        args.ranks = 2
    runs, merged, derived = [], [], []
    complete = True

    def write_out():
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump({"tool": "bench_replay", "selftest": args.selftest, "runs": runs, "merged": merged,
                           "derived": derived}, fh, indent=1)

    plan = [(process, None) for process in args.processes]
    if "lazy" in args.processes:
        plan += [("lazy", env_set) for env_set in args.env_sets]
    for process, env_set in plan:
        if env_set:
            print("== environment set %s: %s" % (env_set[0], " ".join("%s=%s" % kv for kv in env_set[1].items())),
                  flush=True)
        run = run_ranks(args, process, env_set=env_set)
        runs.append(run)
        per_rank = {int(rank): records for rank, records in run["per_rank"].items()}
        merged_p = merge(per_rank)
        derived += derive(merged_p, reference=merged)
        merged += merged_p
        clean = run["ranks_done"] == list(range(args.ranks))
        if not clean:
            # An environment set that fails is a result (that setting does not work), not a failed benchmark.
            complete = complete and bool(env_set)
            print("INCOMPLETE %s: ranks done %s of %d, timed out %s, errors %s" %
                  (run["process"], run["ranks_done"], args.ranks, run["timed_out"], run["errors"]))
            for rank, tail in run["output_tails"].items():
                if int(rank) not in run["ranks_done"]:
                    print("  rank %s last output:" % rank)
                    for line in tail[-10:]:
                        print("     " + line)
        write_out()  # after every run: a later run that hangs or is cut off loses nothing
        if not clean and not args.selftest and (process, env_set) != plan[-1]:
            time.sleep(args.env_pause)
    print()
    if args.selftest:
        print("SELFTEST ONLY (gloo on the CPU): the lines below check this script, they are not results.")
    print_summary(merged, derived, args.ranks)
    for run in runs:
        for record in run["per_rank"].get("0", []):
            if record.get("case") == "check":
                print("   check (%s): %s" % (run["process"], json.dumps(record)[:600]))
            if record.get("case") == "hold":
                print("   hold (%s): %s" % (run["process"], json.dumps(record)[:600]))
            if "error" in record and record.get("case") not in ("error",):
                print("   failed %s rows %s coll %s: %s" % (record.get("case"), record.get("rows"),
                                                           record.get("collectives"), record["error"][:300]))
    if args.out:
        write_out()
        print("wrote %s" % args.out)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
