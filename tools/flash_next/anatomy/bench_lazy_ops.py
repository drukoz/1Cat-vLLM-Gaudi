#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Card microbenchmark: what one lazy op, one graph launch and one ``mark_step()`` cost on a Gaudi card.

A Flash-Next decode step is one lazy graph of several thousand ops that the host rebuilds every step. This
script measures the fixed costs of that mechanism with graphs whose device work is negligible, so the times are
the bridge's and the runtime's own:

* ``graph``  a graph of N ops (N from 100 to 10,000), two families: ``chain`` (``v = v * c + d``) and ``mix``
  (layer-like blocks: cast, grouped norm, two small GEMMs, SiLU, a gate, a residual add, with their own
  weights). Per iteration: ``issue`` (Python issues the ops: graph accumulation), ``mark`` (``mark_step()``:
  the graph is closed, looked up or compiled, and launched), ``wait`` (``torch.hpu.synchronize()``), ``read``
  (``.cpu()`` of the result). The first iteration is the one without a recipe in memory (a compile, or a load
  from the recipe cache on disk); the later ones hit the in-memory recipe.
* ``cuts``   the same ops cut into G graphs by ``mark_step()``: what one more launch costs.
* ``mark_empty`` ``mark_step()`` with nothing accumulated.
* ``h2d`` / ``d2h``  small host-to-device copies as the runner issues them, and reading a small tensor back.
* ``head``   the per-rank LM head GEMM (2560 to 31,040) and an argmax over 248,320 logits at 1, 8 and 32 rows:
  the device work of "logits and greedy sampling" on one card.

Every measurement runs in a child process per bridge setting: ``PT_HPU_LAZY_ACC_PAR_MODE`` 0 and 1, and with a
recipe cache on disk (cold, then warm) when ``--recipe-cache DIR`` is given. The bridge reads these variables
when it is loaded, so they cannot be changed inside one process.

Run on a Gaudi2 node, in the container that has ``habana_frameworks``, with one free card::

    python3 bench_lazy_ops.py --out lazy_ops.json --recipe-cache /tmp/anatomy-recipes
    python3 bench_lazy_ops.py --selftest          # no card: checks this script only, its times are not results

The script needs only ``torch`` and ``habana_frameworks`` (no vLLM, no plugin). Exit code 0 when the decisive
child (accumulation mode 0, no recipe cache on disk) gave every ``graph`` result, 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

RESULT_TAG = "BENCH_RESULT "
ACC_FLAG = "PT_HPU_LAZY_ACC_PAR_MODE"
RECIPE_FLAG = "PT_HPU_RECIPE_CACHE_CONFIG"
HIDDEN = 2560
VOCAB = 248320
TP_SHARD = VOCAB // 8  # the logits columns one rank of TP8 computes
MIX_PROJ = 64
DEFAULT_SIZES = (100, 300, 1000, 3000, 10000)
SELFTEST_SIZES = (100, 300)
DEFAULT_ROWS = (1, 32)
ROWS_PROBE_N = 1000  # the graph size at which the other row counts are measured
CUT_OPS = 1200
CUT_SEGMENTS = (1, 2, 4, 12, 50, 100)
HEAD_ROWS = (1, 8, 32)
FAMILIES = ("chain", "mix")


def _now() -> float:
    return time.perf_counter()


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def _percentile(values, fraction):
    """Nearest-rank percentile, as tools/flash_next/step_anatomy.py."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = min(max(int(-(-fraction * len(ordered) // 1)), 1), len(ordered))
    return ordered[rank - 1]


def stats_ms(seconds):
    """{"p50", "p90", "min", "mean", "n"} in milliseconds of a list of durations in seconds."""
    if not seconds:
        return {"p50": None, "p90": None, "min": None, "mean": None, "n": 0}
    ms = [1e3 * s for s in seconds]
    return {"p50": _median(ms), "p90": _percentile(ms, 0.90), "min": min(ms), "mean": sum(ms) / len(ms), "n": len(ms)}


def linear_fit(xs, ys):
    """Least-squares ``y = a + b * x``; returns ``(a, b)`` or ``None`` with fewer than two distinct x."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None]
    if len({x for x, _ in pairs}) < 2:
        return None
    n = len(pairs)
    mean_x = sum(x for x, _ in pairs) / n
    mean_y = sum(y for _, y in pairs) / n
    sxx = sum((x - mean_x)**2 for x, _ in pairs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


# ---------------------------------------------------------------------------------------------
# Child: one process, one bridge setting.
# ---------------------------------------------------------------------------------------------
class Backend:
    """The device and the two bridge calls; CPU stand-ins under ``--selftest``."""

    def __init__(self, selftest: bool):
        import torch
        self.torch = torch
        self.htorch = None
        self.device = "cpu"
        if not selftest:
            import habana_frameworks.torch as htorch
            self.htorch = htorch
            self.device = "hpu"

    def mark_step(self) -> None:
        if self.htorch is not None:
            self.htorch.core.mark_step()

    def synchronize(self) -> None:
        if self.htorch is not None:
            self.torch.hpu.synchronize()

    def is_lazy(self):
        if self.htorch is None:
            return None
        try:
            return bool(self.htorch.utils.internal.is_lazy())
        except Exception:  # noqa: BLE001 - reported as unknown
            return None


def emit(record: dict) -> None:
    print(RESULT_TAG + json.dumps(record), flush=True)


def build_chain(x, n_ops: int):
    value = x
    for _ in range(n_ops // 2):
        value = value * 1.0009765625 + 0.25
    return value


def mix_block(torch, x, w_norm, w_proj, w_out):
    """One layer-like block of 16 dispatched ops on ``x`` [rows, HIDDEN] bf16; the result has the same shape."""
    h = x.float()
    n = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + 1e-6)
    y = (n * w_norm).to(x.dtype)
    z = torch.nn.functional.silu(torch.nn.functional.linear(y, w_proj))
    a, b = z.chunk(2, dim=-1)
    o = torch.nn.functional.linear(torch.sigmoid(a) * b, w_out)
    return x + o * 0.5


def make_mix_weights(torch, device, blocks: int, shared: bool):
    """Per block ``(w_norm [HIDDEN] f32, w_proj [MIX_PROJ, HIDDEN] bf16, w_out [HIDDEN, MIX_PROJ/2] bf16)``."""
    gen = torch.Generator().manual_seed(7)
    count = 1 if shared else blocks
    weights = []
    for _ in range(count):
        w_norm = (1.0 + 0.01 * torch.randn(HIDDEN, generator=gen)).to(device)
        w_proj = (0.02 * torch.randn(MIX_PROJ, HIDDEN, generator=gen)).to(torch.bfloat16).to(device)
        w_out = (0.02 * torch.randn(HIDDEN, MIX_PROJ // 2, generator=gen)).to(torch.bfloat16).to(device)
        weights.append((w_norm, w_proj, w_out))
    return weights if not shared else weights * blocks


def build_mix(torch, x, weights):
    value = x
    for w_norm, w_proj, w_out in weights:
        value = mix_block(torch, value, w_norm, w_proj, w_out)
    return value


def count_dispatched(torch, build):
    """Dispatched op count of ``build()`` under a TorchDispatchMode, and its result; ``(None, error)`` on failure."""
    try:
        from torch.utils._python_dispatch import TorchDispatchMode

        class Counter(TorchDispatchMode):

            def __init__(self):
                super().__init__()
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.count += 1
                return func(*args, **(kwargs or {}))

        counter = Counter()
        with counter:
            out = build()
        return counter.count, out
    except Exception as exc:  # noqa: BLE001 - the timing does not depend on the count
        return None, "%s: %s" % (type(exc).__name__, str(exc)[:200])


def time_graph(be: Backend, build, iterations: int):
    """Per iteration ``(issue, mark, wait, read, total)`` in seconds."""
    rows = []
    for _ in range(iterations):
        t0 = _now()
        out = build()
        t1 = _now()
        be.mark_step()
        t2 = _now()
        be.synchronize()
        t3 = _now()
        out.cpu()
        t4 = _now()
        rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
    return rows


def iterations_for(n_ops: int, quick: bool) -> int:
    """Timed iterations after the first one: fewer for the big graphs, so the job stays short."""
    if quick:
        return 4
    if n_ops <= 1000:
        return 24
    if n_ops <= 3000:
        return 12
    return 6


def bench_graph(be: Backend, family: str, n_ops: int, rows: int, weights_shared: bool, quick: bool) -> dict:
    torch = be.torch
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16).to(be.device)
    if family == "chain":
        ops = 2 * (n_ops // 2)

        def build():
            return build_chain(x, n_ops)
    else:
        blocks = max(1, n_ops // 16)
        weights = make_mix_weights(torch, be.device, blocks, weights_shared)
        ops = 16 * blocks

        def build():
            return build_mix(torch, x, weights)

    be.mark_step()
    be.synchronize()  # the inputs are on the device before the first timed iteration
    timed = time_graph(be, build, 1 + 2 + iterations_for(n_ops, quick))
    first, steady = timed[0], timed[3:]
    dispatched, note = count_dispatched(torch, build)
    if dispatched is not None:
        be.mark_step()
        be.synchronize()
        note = None
    names = ("issue", "mark", "wait", "read", "total")
    return {
        "case": "graph",
        "family": family,
        "ops": ops,
        "ops_dispatched": dispatched,
        "dispatch_note": note if isinstance(note, str) else None,
        "rows": rows,
        "weights": "shared" if weights_shared else "own",
        "first_ms": {name: 1e3 * first[i] for i, name in enumerate(names)},
        "steady_ms": {name: stats_ms([row[i] for row in steady]) for i, name in enumerate(names)},
    }


def bench_cuts(be: Backend, n_ops: int, segments: int, quick: bool) -> dict:
    """``n_ops`` chain ops in ``segments`` graphs: a ``mark_step()`` between the segments, one wait at the end."""
    torch = be.torch
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(1, HIDDEN, generator=gen).to(torch.bfloat16).to(be.device)
    per_segment = max(1, (n_ops // 2) // segments)
    be.mark_step()
    be.synchronize()
    total, inner, issue = [], [], []
    for _ in range(3 + (4 if quick else 16)):
        marks = 0.0
        t0 = _now()
        value = x
        for segment in range(segments):
            if segment:
                m0 = _now()
                be.mark_step()
                marks += _now() - m0
            for _ in range(per_segment):
                value = value * 1.0009765625 + 0.25
        t1 = _now()
        be.mark_step()
        be.synchronize()
        value.cpu()
        t2 = _now()
        total.append(t2 - t0)
        inner.append(marks)
        issue.append(t1 - t0 - marks)
    return {
        "case": "cuts",
        "ops": 2 * per_segment * segments,
        "segments": segments,
        "total_ms": stats_ms(total[3:]),
        "inner_mark_ms": stats_ms(inner[3:]),
        "issue_ms": stats_ms(issue[3:]),
    }


def bench_mark_empty(be: Backend) -> dict:
    be.mark_step()
    be.synchronize()
    calls = []
    for _ in range(220):
        t0 = _now()
        be.mark_step()
        calls.append(_now() - t0)
    waits = []
    for _ in range(50):
        t0 = _now()
        be.synchronize()
        waits.append(_now() - t0)
    return {"case": "mark_empty", "mark_ms": stats_ms(calls[20:]), "synchronize_idle_ms": stats_ms(waits[5:])}


def bench_h2d(be: Backend, copies: int, quick: bool) -> dict:
    """``copies`` small int32 tensors sent as ``async_h2d_copy`` sends them, then used by one op each."""
    torch = be.torch
    sources = [torch.arange(32, dtype=torch.int32) + index for index in range(copies)]
    results = {}
    for style in ("copy_", "to"):
        issue, flush = [], []
        for _ in range(3 + (4 if quick else 20)):
            t0 = _now()
            if style == "copy_":
                outs = [torch.empty_like(src, device=be.device).copy_(src, non_blocking=True) for src in sources]
            else:
                outs = [src.to(be.device, non_blocking=True) for src in sources]
            t1 = _now()
            be.mark_step()
            be.synchronize()
            t2 = _now()
            issue.append(t1 - t0)
            flush.append(t2 - t1)
            del outs
        results[style] = {"issue_ms": stats_ms(issue[3:]), "flush_ms": stats_ms(flush[3:])}
    return {"case": "h2d", "copies": copies, "elements": 32, "styles": results}


def bench_d2h(be: Backend, quick: bool) -> dict:
    """Reading a small computed tensor back: ``.cpu()`` and ``.item()`` once the device is idle, and right after
    a one-op graph was issued (the read then launches and waits for that graph)."""
    torch = be.torch
    x = torch.arange(32, dtype=torch.int32).to(be.device)
    y = x + 1
    be.mark_step()
    be.synchronize()
    repeats = 6 if quick else 40
    idle_cpu, idle_item, fresh_cpu = [], [], []
    scalar = y.sum()
    be.mark_step()
    be.synchronize()
    for _ in range(repeats):
        t0 = _now()
        y.cpu()
        idle_cpu.append(_now() - t0)
        t0 = _now()
        scalar.item()
        idle_item.append(_now() - t0)
    for _ in range(repeats):
        t0 = _now()
        (x + 2).cpu()
        fresh_cpu.append(_now() - t0)
    return {
        "case": "d2h",
        "cpu_idle_ms": stats_ms(idle_cpu[2:]),
        "item_idle_ms": stats_ms(idle_item[2:]),
        "cpu_one_op_graph_ms": stats_ms(fresh_cpu[2:]),
    }


def bench_head(be: Backend, rows: int, quick: bool) -> dict:
    """The LM head of one TP8 rank ([rows, HIDDEN] x [TP_SHARD, HIDDEN]) and an argmax over VOCAB logits."""
    torch = be.torch
    gen = torch.Generator().manual_seed(11)
    shard = TP_SHARD if be.htorch is not None else 512
    vocab = VOCAB if be.htorch is not None else 4096
    hidden = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16).to(be.device)
    weight = (0.02 * torch.randn(shard, HIDDEN, generator=gen)).to(torch.bfloat16).to(be.device)
    logits = torch.randn(rows, vocab, generator=gen).to(torch.bfloat16).to(be.device)
    be.mark_step()
    be.synchronize()
    repeats = 3 + (4 if quick else 20)
    gemm, argmax = [], []
    for _ in range(repeats):
        t0 = _now()
        out = torch.nn.functional.linear(hidden, weight)
        be.mark_step()
        be.synchronize()
        gemm.append(_now() - t0)
        del out
        t0 = _now()
        token = logits.argmax(dim=-1)
        be.mark_step()
        be.synchronize()
        argmax.append(_now() - t0)
        del token
    return {
        "case": "head",
        "rows": rows,
        "shard": shard,
        "vocab": vocab,
        "gemm_ms": stats_ms(gemm[3:]),
        "argmax_ms": stats_ms(argmax[3:]),
    }


def child_main(args) -> int:
    selftest = args.selftest
    try:
        be = Backend(selftest)
        torch = be.torch
        emit({
            "case": "env",
            "device": be.device,
            "lazy": be.is_lazy(),
            "acc_par_env": os.environ.get(ACC_FLAG),
            "recipe_cache_env": os.environ.get(RECIPE_FLAG),
            "lazy_mode_env": os.environ.get("PT_HPU_LAZY_MODE"),
            "torch": torch.__version__,
            "selftest": selftest,
        })
        rows_list = list(args.rows)
        for family in args.families:
            for n_ops in args.sizes:
                for rows in rows_list:
                    if rows != rows_list[0] and n_ops != args.rows_probe_n:
                        continue
                    emit(bench_graph(be, family, n_ops, rows, False, args.quick))
        if "mix" in args.families and args.rows_probe_n in args.sizes:
            # the same graph with one set of weights for every block: does the launch depend on the graph inputs?
            emit(bench_graph(be, "mix", args.rows_probe_n, rows_list[0], True, args.quick))
        if not args.graphs_only:
            for segments in CUT_SEGMENTS:
                emit(bench_cuts(be, CUT_OPS if not selftest else 240, segments, args.quick))
            emit(bench_mark_empty(be))
            emit(bench_h2d(be, 16, args.quick))
            emit(bench_d2h(be, args.quick))
            for rows in HEAD_ROWS:
                emit(bench_head(be, rows, args.quick))
        emit({"case": "done"})
        return 0
    except BaseException as exc:  # noqa: BLE001 - the parent reports it next to the results it already has
        emit({
            "case": "error",
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:800]),
            "traceback": traceback.format_exc()[-2000:]
        })
        return 1


# ---------------------------------------------------------------------------------------------
# Parent: one child per bridge setting, then the tables.
# ---------------------------------------------------------------------------------------------
def child_environment(acc_par: str, recipe_cache: str | None, recipe_cache_mb: int) -> dict:
    env = dict(os.environ)
    env["PT_HPU_LAZY_MODE"] = "1"
    env[ACC_FLAG] = str(acc_par)
    if recipe_cache:
        env[RECIPE_FLAG] = "%s,false,%d" % (recipe_cache, recipe_cache_mb)
    else:
        env.pop(RECIPE_FLAG, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_child(label: str, env: dict, child_args: list, timeout_s: float, echo=print) -> dict:
    """Run this script as ``--child``; keep every result line it printed, also when it is killed at the timeout."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", *child_args]
    started = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    records, tail = [], []

    def reader():
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(RESULT_TAG):
                try:
                    record = json.loads(line[len(RESULT_TAG):])
                except ValueError:
                    continue
                records.append(record)
                echo("  [%s] %s" % (label, describe(record)))
            else:
                tail.append(line)
                del tail[:-40]

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    thread.join(timeout=10)
    return {
        "label": label,
        "acc_par": env.get(ACC_FLAG),
        "recipe_cache": env.get(RECIPE_FLAG),
        "exit": proc.returncode,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - started, 1),
        "records": records,
        "output_tail": tail[-25:],
    }


def _fmt(value, digits=3):
    return "n/a" if value is None else ("%." + str(digits) + "f") % value


def describe(record: dict) -> str:
    """One console line for one result record."""
    case = record.get("case")
    if case == "graph":
        steady, first = record["steady_ms"], record["first_ms"]
        return ("graph %-5s ops %5d rows %2d weights %-6s: issue %s mark %s wait %s read %s total %s ms (p50); "
                "first iteration %s ms" %
                (record["family"], record["ops"], record["rows"], record["weights"], _fmt(steady["issue"]["p50"]),
                 _fmt(steady["mark"]["p50"]), _fmt(steady["wait"]["p50"]), _fmt(steady["read"]["p50"]),
                 _fmt(steady["total"]["p50"]), _fmt(first["total"], 1)))
    if case == "cuts":
        return "cuts ops %d in %3d graphs: total %s ms, the inner mark_step calls %s ms (p50)" % (
            record["ops"], record["segments"], _fmt(record["total_ms"]["p50"]), _fmt(record["inner_mark_ms"]["p50"]))
    if case == "mark_empty":
        return "mark_step with nothing accumulated %s ms, synchronize on an idle device %s ms (p50)" % (
            _fmt(record["mark_ms"]["p50"], 4), _fmt(record["synchronize_idle_ms"]["p50"], 4))
    if case == "h2d":
        parts = [
            "%s: issue %s ms, flush %s ms" % (style, _fmt(values["issue_ms"]["p50"]), _fmt(values["flush_ms"]["p50"]))
            for style, values in record["styles"].items()
        ]
        return "h2d %d copies of %d int32: %s (p50)" % (record["copies"], record["elements"], "; ".join(parts))
    if case == "d2h":
        return "d2h: .cpu() idle %s ms, .item() idle %s ms, .cpu() of a fresh one-op graph %s ms (p50)" % (_fmt(
            record["cpu_idle_ms"]["p50"]), _fmt(record["item_idle_ms"]["p50"]), _fmt(
                record["cpu_one_op_graph_ms"]["p50"]))
    if case == "head":
        return "head rows %2d: GEMM to %d columns %s ms, argmax over %d %s ms (p50, launch and wait included)" % (
            record["rows"], record["shard"], _fmt(record["gemm_ms"]["p50"]), record["vocab"],
            _fmt(record["argmax_ms"]["p50"]))
    if case == "env":
        return "env: device %s, lazy %s, %s=%s, %s=%s" % (record.get("device"), record.get("lazy"), ACC_FLAG,
                                                        record.get("acc_par_env"), RECIPE_FLAG,
                                                        record.get("recipe_cache_env"))
    if case == "error":
        return "ERROR %s" % record.get("error")
    return str(case)


def fits(records: list) -> dict:
    """Per family: the linear fits over the graph size of the steady medians (first row count, own weights)."""
    out = {}
    graphs = [r for r in records if r.get("case") == "graph" and r.get("weights") == "own"]
    if not graphs:
        return out
    base_rows = graphs[0]["rows"]
    for family in sorted({r["family"] for r in graphs}):
        rows = [r for r in graphs if r["family"] == family and r["rows"] == base_rows]
        xs = [r["ops"] for r in rows]
        entry = {"sizes": xs}
        for name in ("issue", "mark", "wait", "read", "total"):
            fit = linear_fit(xs, [r["steady_ms"][name]["p50"] for r in rows])
            entry[name] = None if fit is None else {"fixed_ms": fit[0], "us_per_op": 1e3 * fit[1]}
        fit = linear_fit(xs, [r["first_ms"]["total"] for r in rows])
        entry["first_total"] = None if fit is None else {"fixed_ms": fit[0], "us_per_op": 1e3 * fit[1]}
        out[family] = entry
    return out


def cut_cost(records: list):
    """Milliseconds one more launch adds: the slope of the total over the number of graphs."""
    cuts = [r for r in records if r.get("case") == "cuts"]
    fit_total = linear_fit([r["segments"] for r in cuts], [r["total_ms"]["p50"] for r in cuts])
    fit_mark = linear_fit([r["segments"] for r in cuts], [r["inner_mark_ms"]["p50"] for r in cuts])
    if fit_total is None or fit_mark is None:
        return None
    return {"total_ms_per_launch": fit_total[1], "host_mark_ms_per_launch": fit_mark[1]}


def summarize(run: dict) -> dict:
    records = run["records"]
    errors = [r for r in records if r.get("case") == "error"]
    return {
        "label": run["label"],
        "acc_par": run["acc_par"],
        "recipe_cache": run["recipe_cache"],
        "complete": any(r.get("case") == "done" for r in records),
        "timed_out": run["timed_out"],
        "exit": run["exit"],
        "error": errors[0]["error"] if errors else None,
        "fits": fits(records),
        "cut_cost": cut_cost(records),
    }


def print_summary(summary: dict, echo=print) -> None:
    state = "complete" if summary["complete"] else ("TIMED OUT" if summary["timed_out"] else "INCOMPLETE")
    echo("== %s (%s=%s, recipe cache %s): %s%s" %
         (summary["label"], ACC_FLAG, summary["acc_par"], summary["recipe_cache"] or "off", state,
          "; " + summary["error"] if summary["error"] else ""))
    for family, entry in summary["fits"].items():
        for name in ("issue", "mark", "wait", "total"):
            fit = entry.get(name)
            if fit is not None:
                echo("   %-5s %-5s: %7.3f us per op + %7.3f ms fixed   (sizes %s)" %
                     (family, name, fit["us_per_op"], fit["fixed_ms"], entry["sizes"]))
        fit = entry.get("first_total")
        if fit is not None:
            echo("   %-5s first iteration (no recipe in memory): %.1f us per op + %.1f ms fixed" %
                 (family, fit["us_per_op"], fit["fixed_ms"]))
    if summary["cut_cost"] is not None:
        echo("   one more launch (mark_step cut): %.3f ms in total, %.3f ms of it inside mark_step" %
             (summary["cut_cost"]["total_ms_per_launch"], summary["cut_cost"]["host_mark_ms_per_launch"]))


def parse_int_list(text: str) -> list:
    return [int(token) for token in text.split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench_lazy_ops.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=None, help="write every result and the fits to this JSON file")
    parser.add_argument("--sizes", default=None, help="graph sizes in ops (default %s)" % (DEFAULT_SIZES, ))
    parser.add_argument("--rows", default=",".join(str(r) for r in DEFAULT_ROWS),
                        help="row counts; the first is used at every size, the others at --rows-probe-n only")
    parser.add_argument("--rows-probe-n", type=int, default=ROWS_PROBE_N)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--acc-modes", default="0,1", help="values of %s to measure (default 0,1)" % ACC_FLAG)
    parser.add_argument("--recipe-cache", default=None, metavar="DIR",
                        help="also measure with a recipe cache on disk under DIR (a cold run, then a warm one)")
    parser.add_argument("--recipe-cache-mb", type=int, default=4096)
    parser.add_argument("--recipe-cache-all-modes", action="store_true",
                        help="the recipe-cache runs for every accumulation mode (default: mode 0 only)")
    parser.add_argument("--child-timeout", type=float, default=420.0, metavar="S")
    parser.add_argument("--quick", action="store_true", help="few iterations (a smoke run)")
    parser.add_argument("--graphs-only", action="store_true", help="only the graph cases")
    parser.add_argument("--selftest", action="store_true", help="CPU tensors, no bridge: checks this script only")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sizes = parse_int_list(args.sizes) if args.sizes else list(SELFTEST_SIZES if args.selftest else DEFAULT_SIZES)
    args.sizes = sizes
    args.rows = parse_int_list(args.rows)
    args.families = [name for name in args.families.split(",") if name in FAMILIES]
    if args.selftest and args.rows_probe_n not in sizes:
        args.rows_probe_n = sizes[-1]
    if args.child:
        return child_main(args)

    child_args = ["--sizes", ",".join(str(n) for n in sizes), "--rows", ",".join(str(r) for r in args.rows),
                  "--rows-probe-n", str(args.rows_probe_n), "--families", ",".join(args.families)]
    if args.quick:
        child_args.append("--quick")
    if args.selftest:
        child_args.append("--selftest")
    if args.graphs_only:
        child_args.append("--graphs-only")

    plan = []  # (label, acc mode, recipe cache folder or None, extra child args)
    modes = [m.strip() for m in args.acc_modes.split(",") if m.strip()]
    for mode in modes:
        plan.append(("acc%s" % mode, mode, None, []))
    if args.recipe_cache:
        cache_modes = modes if args.recipe_cache_all_modes else modes[:1]
        for mode in cache_modes:
            folder = os.path.join(args.recipe_cache, "acc%s-%d" % (mode, os.getpid()))
            os.makedirs(folder, exist_ok=True)
            plan.append(("acc%s-cache-cold" % mode, mode, folder, ["--graphs-only"]))
            plan.append(("acc%s-cache-warm" % mode, mode, folder, ["--graphs-only"]))

    runs, summaries = [], []
    for label, mode, folder, extra in plan:
        print("-- child %s: %s=%s, recipe cache %s" % (label, ACC_FLAG, mode, folder or "off"), flush=True)
        run = run_child(label, child_environment(mode, folder, args.recipe_cache_mb),
                        child_args + [a for a in extra if a not in child_args], args.child_timeout)
        runs.append(run)
        if not any(r.get("case") == "done" for r in run["records"]):
            print("   child %s did not finish (exit %s, timed out %s); last output:" %
                  (label, run["exit"], run["timed_out"]))
            for line in run["output_tail"][-12:]:
                print("     " + line)
        summaries.append(summarize(run))

    print()
    if args.selftest:
        print("SELFTEST ONLY (CPU, no bridge): the lines below check this script, they are not results.")
    for summary in summaries:
        print_summary(summary)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tool": "bench_lazy_ops", "selftest": args.selftest, "sizes": sizes, "runs": runs,
                       "summaries": summaries}, fh, indent=1)
        print("wrote %s" % args.out)
    decisive = summaries[0] if summaries else None
    wanted = len(sizes) * len(args.families)
    got = 0 if decisive is None else sum(len(entry["sizes"]) for entry in decisive["fits"].values())
    return 0 if decisive is not None and got >= wanted else 2


if __name__ == "__main__":
    sys.exit(main())
