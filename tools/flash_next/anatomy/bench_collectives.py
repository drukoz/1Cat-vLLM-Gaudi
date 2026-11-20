#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Card microbenchmark: latency of HCCL all-reduce and all-gather at the message sizes of a Flash-Next decode step.

A decode step at TP8 has 98 all-reduces of one hidden row per stream ([rows, 2560] bf16) and one all-gather of
the logits ([rows, 31,040] bf16 per rank, 248,320 columns gathered). This script starts one process per rank,
wires them as vLLM does (``torch.distributed`` with the ``hccl`` backend, a device group over all ranks, each
rank on its own card), and measures at 1, 8 and 32 rows:

* ``own``      one collective as its own launch: the collective, ``mark_step()``, ``synchronize()``. Also the
  time of the call itself (graph accumulation) and of a read-back after the wait (it must be small: the wait
  already covered the collective).
* ``graph``    one lazy graph of 98 all-reduces with two element-wise ops in front of each (the shape of a
  decode step), with and without the logits all-gather at its end, against the same graph without collectives:
  the difference is what the collectives cost inside a graph.
* ``marked``   the same graph with a ``mark_step()`` in front of every collective, as the communicator issues
  it by default (``VLLM_HPU_COLLECTIVE_MARK_STEP=1``).
* ``gather_hidden`` an all-gather of the hidden rows ([rows, 2560] per rank), the message of a fused collective
  that sums the ranks in a kernel.

Run on a Gaudi2 node, in the container that has ``habana_frameworks``, with ``--ranks`` free cards::

    python3 bench_collectives.py --ranks 8 --out collectives_r8.json
    python3 bench_collectives.py --ranks 4 --cards 0,1,2,3 --out collectives_r4.json
    python3 bench_collectives.py --selftest       # no card: two gloo ranks on the CPU, checks this script only

Needs only ``torch`` and ``habana_frameworks``. Every rank prints its own numbers; the summary gives rank 0 and
the slowest rank. Exit code 0 when every rank finished, 2 otherwise.
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
from datetime import timedelta

RESULT_TAG = "BENCH_RESULT "
HIDDEN = 2560
VOCAB = 248320
DECODE_ALL_REDUCES = 98
DEFAULT_ROWS = (1, 8, 32)
EXIT_GRACE_S = 20.0


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
        return {"p50": None, "p90": None, "min": None, "mean": None, "n": 0}
    ms = sorted(1e3 * s for s in seconds)
    rank = min(max(-(-9 * len(ms) // 10), 1), len(ms))
    return {"p50": _median(ms), "p90": ms[rank - 1], "min": ms[0], "mean": sum(ms) / len(ms), "n": len(ms)}


def emit(record: dict) -> None:
    print(RESULT_TAG + json.dumps(record), flush=True)


# ---------------------------------------------------------------------------------------------
# One rank.
# ---------------------------------------------------------------------------------------------
class Rank:

    def __init__(self, args):
        import torch
        import torch.distributed as dist
        self.torch = torch
        self.dist = dist
        self.rank = args.rank
        self.world = args.ranks
        self.selftest = args.selftest
        self.htorch = None
        if self.selftest:
            self.device, backend = "cpu", "gloo"
        else:
            import habana_frameworks.torch as htorch
            try:
                import habana_frameworks.torch.distributed.hccl  # noqa: F401 - registers the backend where needed
            except Exception:  # noqa: BLE001 - importing habana_frameworks.torch registers it on current releases
                pass
            self.htorch = htorch
            torch.hpu.set_device(self.rank)  # as HPUWorker.init_device: one visible card per rank
            self.device, backend = "hpu", "hccl"
        dist.init_process_group(backend=backend, init_method=args.init_method, world_size=self.world,
                                rank=self.rank, timeout=timedelta(seconds=args.timeout))
        probe = torch.ones(1).to(self.device)
        dist.all_reduce(probe)
        if probe.item() != self.world:
            raise RuntimeError("all-reduce of ones gave %s on rank %d, expected %d" %
                               (probe.item(), self.rank, self.world))
        # the tensor-parallel device group, as vLLM's GroupCoordinator creates it
        self.group = dist.new_group(list(range(self.world)), backend=backend)
        self.barrier_probe = torch.zeros(1).to(self.device)

    def mark_step(self) -> None:
        if self.htorch is not None:
            self.htorch.core.mark_step()

    def synchronize(self) -> None:
        if self.htorch is not None:
            self.torch.hpu.synchronize()

    def barrier(self) -> None:
        """Every rank passes here together: a collective that is read back."""
        self.dist.all_reduce(self.barrier_probe, group=self.group)
        self.barrier_probe.item()

    def zeros(self, *shape):
        return self.torch.zeros(*shape, dtype=self.torch.bfloat16).to(self.device)

    # ----------------------------------------------------------------------------------- the cases
    def own_launch(self, kind: str, rows: int, width: int, repeats: int) -> dict:
        torch, dist = self.torch, self.dist
        x = self.zeros(rows, width)
        out = None
        self.mark_step()
        self.synchronize()
        self.barrier()
        issue, total, read = [], [], []
        for _ in range(3 + repeats):
            t0 = _now()
            if kind == "all_reduce":
                dist.all_reduce(x, group=self.group)
                result = x
            else:
                out = torch.empty((rows * self.world, width), dtype=x.dtype, device=x.device)
                dist.all_gather_into_tensor(out, x, group=self.group)
                result = out
            t1 = _now()
            self.mark_step()
            self.synchronize()
            t2 = _now()
            result.cpu()
            t3 = _now()
            issue.append(t1 - t0)
            total.append(t2 - t0)
            read.append(t3 - t2)
        return {
            "case": "own",
            "kind": kind,
            "rows": rows,
            "width": width,
            "bytes_per_rank": rows * width * 2,
            "issue_ms": stats_ms(issue[3:]),
            "latency_ms": stats_ms(total[3:]),
            "read_after_ms": stats_ms(read[3:]),
        }

    def graph(self, rows: int, collectives: bool, marked: bool, logits_gather: bool, repeats: int,
              shard: int) -> dict:
        """One lazy graph shaped like a decode step: 98 times (two element-wise ops, an all-reduce)."""
        torch, dist = self.torch, self.dist
        base = (self.zeros(rows, HIDDEN) + 1.0)
        logits_local = self.zeros(rows, shard)
        scale = 1.0 / self.world  # the reduced value stays 1.0, whatever the number of steps
        self.mark_step()
        self.synchronize()
        self.barrier()
        issue, mark, wait, total = [], [], [], []
        for _ in range(3 + repeats):
            t0 = _now()
            value = base
            for _ in range(DECODE_ALL_REDUCES):
                value = value * scale + 0.0
                if collectives:
                    if marked:
                        self.mark_step()
                    dist.all_reduce(value, group=self.group)
            gathered = None
            if logits_gather:
                local = logits_local + value[:, :1]
                gathered = torch.empty((rows * self.world, shard), dtype=local.dtype, device=local.device)
                if marked:
                    self.mark_step()
                dist.all_gather_into_tensor(gathered, local, group=self.group)
            t1 = _now()
            self.mark_step()
            t2 = _now()
            self.synchronize()
            t3 = _now()
            value.cpu()
            t4 = _now()
            issue.append(t1 - t0)
            mark.append(t2 - t1)
            wait.append(t3 - t2)
            total.append(t4 - t0)
            del gathered
        name = "marked" if marked else "graph"
        return {
            "case": name if collectives else "graph_without_collectives",
            "rows": rows,
            "all_reduces": DECODE_ALL_REDUCES if collectives else 0,
            "logits_gather": bool(logits_gather),
            "issue_ms": stats_ms(issue[3:]),
            "mark_ms": stats_ms(mark[3:]),
            "wait_ms": stats_ms(wait[3:]),
            "total_ms": stats_ms(total[3:]),
        }


def rank_main(args) -> int:
    try:
        rank = Rank(args)
        shard = VOCAB // args.ranks if not args.selftest else 256
        repeats_own = 6 if args.quick else 40
        repeats_graph = 4 if args.quick else 15
        emit({
            "case": "env",
            "rank": args.rank,
            "ranks": args.ranks,
            "device": rank.device,
            "acc_par_env": os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE"),
            "lazy_collectives_env": os.environ.get("PT_HPU_ENABLE_LAZY_COLLECTIVES"),
            "visible_modules": os.environ.get("HABANA_VISIBLE_MODULES"),
            "selftest": args.selftest,
        })
        for rows in args.rows:
            for record in (
                    rank.own_launch("all_reduce", rows, HIDDEN, repeats_own),
                    rank.own_launch("all_gather", rows, HIDDEN, repeats_own),
                    rank.own_launch("all_gather", rows, shard, repeats_own),
                    rank.graph(rows, False, False, False, repeats_graph, shard),
                    rank.graph(rows, True, False, False, repeats_graph, shard),
                    rank.graph(rows, True, False, True, repeats_graph, shard),
                    rank.graph(rows, True, True, True, repeats_graph, shard),
            ):
                record["rank"] = args.rank
                emit(record)
        rank.barrier()
        emit({"case": "done", "rank": args.rank})
        sys.stdout.flush()
        # Every number is out. Leave without tearing the process group down: a teardown that waits for a peer
        # would cost card time and adds nothing to a benchmark.
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 - the parent reports it
        emit({
            "case": "error",
            "rank": args.rank,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:800]),
            "traceback": traceback.format_exc()[-2000:]
        })
        return 1


# ---------------------------------------------------------------------------------------------
# Parent: start the ranks, gather their lines, print the summary.
# ---------------------------------------------------------------------------------------------
def rank_environment(args) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if not args.selftest:
        env["PT_HPU_LAZY_MODE"] = "1"
        env["PT_HPU_LAZY_ACC_PAR_MODE"] = str(args.acc_mode)
        env.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")  # as the plugin sets it for lazy tensor parallel
        if args.cards:
            env["HABANA_VISIBLE_MODULES"] = args.cards
    return env


def case_key(record: dict) -> tuple:
    return (record.get("case"), record.get("kind"), record.get("rows"), record.get("width"),
            record.get("logits_gather"))


def merge(per_rank: dict) -> list:
    """One entry per case: the fields of rank 0 plus, for every timing, the slowest rank's median."""
    merged = []
    rank0 = per_rank.get(0, [])
    for record in rank0:
        if record.get("case") in ("env", "done", "error"):
            continue
        entry = dict(record)
        key = case_key(record)
        for field in [name for name in record if name.endswith("_ms")]:
            medians = []
            for records in per_rank.values():
                for other in records:
                    if case_key(other) == key and other.get(field, {}).get("p50") is not None:
                        medians.append(other[field]["p50"])
            entry[field.replace("_ms", "_p50_max_rank_ms")] = max(medians) if medians else None
            entry[field.replace("_ms", "_p50_median_rank_ms")] = _median(medians) if medians else None
        merged.append(entry)
    return merged


def derive(merged: list) -> list:
    """Per row count: the cost of the collectives inside a graph, from the graphs with and without them."""
    lines = []
    for rows in sorted({entry["rows"] for entry in merged}):

        def total(case, logits_gather):
            for entry in merged:
                if (entry["case"], entry["rows"], entry.get("logits_gather")) == (case, rows, logits_gather):
                    return entry["total_p50_max_rank_ms"]
            return None

        without = total("graph_without_collectives", False)
        with_ar = total("graph", False)
        with_ag = total("graph", True)
        marked = total("marked", True)
        own = {(entry["kind"], entry["width"]): entry["latency_p50_max_rank_ms"]
               for entry in merged if entry["case"] == "own" and entry["rows"] == rows}
        line = {"rows": rows, "own_ms": {"%s_%d" % key: value for key, value in own.items()}}
        if None not in (without, with_ar):
            line["in_graph_all_reduce_ms_each"] = (with_ar - without) / DECODE_ALL_REDUCES
            line["in_graph_98_all_reduces_ms"] = with_ar - without
        if None not in (with_ar, with_ag):
            line["in_graph_logits_gather_ms"] = with_ag - with_ar
        if None not in (with_ag, marked):
            line["mark_step_before_each_adds_ms"] = marked - with_ag
        line["graph_total_ms"] = {"without_collectives": without, "all_reduces": with_ar,
                                  "all_reduces_and_gather": with_ag, "marked": marked}
        lines.append(line)
    return lines


def _fmt(value):
    return "n/a" if value is None else "%.3f" % value


def print_summary(merged: list, derived: list, ranks: int, echo=print) -> None:
    echo("== %d ranks, the slowest rank's median in ms (rank 0 in brackets)" % ranks)
    for entry in merged:
        if entry["case"] == "own":
            echo("   own launch %-10s rows %2d x %6d (%7d bytes per rank): %s [%s]; the call itself %s; read after %s" %
                 (entry["kind"], entry["rows"], entry["width"], entry["bytes_per_rank"],
                  _fmt(entry["latency_p50_max_rank_ms"]), _fmt(entry["latency_ms"]["p50"]),
                  _fmt(entry["issue_p50_max_rank_ms"]), _fmt(entry["read_after_p50_max_rank_ms"])))
        else:
            echo("   %-25s rows %2d gather %-5s: total %s [%s] = issue %s + mark %s + wait %s (+ read)" %
                 (entry["case"], entry["rows"], entry["logits_gather"], _fmt(entry["total_p50_max_rank_ms"]),
                  _fmt(entry["total_ms"]["p50"]), _fmt(entry["issue_p50_max_rank_ms"]),
                  _fmt(entry["mark_p50_max_rank_ms"]), _fmt(entry["wait_p50_max_rank_ms"])))
    for line in derived:
        echo("   rows %2d: one all-reduce inside a graph %s ms (98 of them %s ms); the logits all-gather %s ms; "
             "a mark_step before each collective adds %s ms" %
             (line["rows"], _fmt(line.get("in_graph_all_reduce_ms_each")),
              _fmt(line.get("in_graph_98_all_reduces_ms")), _fmt(line.get("in_graph_logits_gather_ms")),
              _fmt(line.get("mark_step_before_each_adds_ms"))))


def run_ranks(args, echo=print) -> dict:
    workdir = tempfile.mkdtemp(prefix="anatomy-coll-")
    init_method = "file://" + os.path.join(workdir, "dist_init")
    env = rank_environment(args)
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
                    echo("  [rank 0] %s" % json.dumps(records[-1])[:230])
            else:
                tail.append(line)
                del tail[:-30]

    threads = []
    for rank in range(args.ranks):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--rank", str(rank), "--ranks", str(args.ranks),
               "--init-method", init_method, "--rows", ",".join(str(r) for r in args.rows), "--timeout",
               str(int(args.timeout))]
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
            break  # a rank that failed leaves the others waiting in a collective
        done_ranks = sum(1 for records in list(per_rank.values()) if any(r.get("case") == "done" for r in records))
        if done_ranks == args.ranks:
            all_done_at = all_done_at or now
            if now - all_done_at > EXIT_GRACE_S:
                break  # every rank reported; one that does not exit is not worth waiting for
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
    return {
        "ranks": args.ranks,
        "seconds": round(time.monotonic() - started, 1),
        "timed_out": timed_out,
        "exit_codes": [proc.returncode for proc in procs],
        "ranks_done": done,
        "errors": errors,
        "per_rank": {str(rank): records for rank, records in sorted(per_rank.items())},
        "output_tails": {str(rank): tail[-15:] for rank, tail in sorted(tails.items())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench_collectives.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--ranks", type=int, default=8, help="processes, one card each (default 8)")
    parser.add_argument("--cards", default=None, help="HABANA_VISIBLE_MODULES for the ranks (default: inherited)")
    parser.add_argument("--rows", default=",".join(str(r) for r in DEFAULT_ROWS))
    parser.add_argument("--acc-mode", default="0", help="PT_HPU_LAZY_ACC_PAR_MODE of the ranks (default 0)")
    parser.add_argument("--out", default=None, help="write every result to this JSON file")
    parser.add_argument("--timeout", type=float, default=600.0, metavar="S", help="for the whole run (default 600)")
    parser.add_argument("--quick", action="store_true", help="few repeats (a smoke run)")
    parser.add_argument("--selftest", action="store_true", help="gloo ranks on the CPU: checks this script only")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--init-method", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.rows = [int(token) for token in str(args.rows).split(",") if token.strip()]
    if args.child:
        return rank_main(args)
    if args.selftest and args.ranks > 4:
        args.ranks = 2
    run = run_ranks(args)
    per_rank = {int(rank): records for rank, records in run["per_rank"].items()}
    merged = merge(per_rank)
    derived = derive(merged)
    print()
    if args.selftest:
        print("SELFTEST ONLY (gloo on the CPU): the lines below check this script, they are not results.")
    print_summary(merged, derived, args.ranks)
    complete = run["ranks_done"] == list(range(args.ranks))
    if not complete:
        print("INCOMPLETE: ranks done %s of %d, timed out %s, errors %s" %
              (run["ranks_done"], args.ranks, run["timed_out"], run["errors"]))
        for rank, tail in run["output_tails"].items():
            if int(rank) not in run["ranks_done"]:
                print("  rank %s last output:" % rank)
                for line in tail[-8:]:
                    print("     " + line)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tool": "bench_collectives", "selftest": args.selftest, "run": run, "merged": merged,
                       "derived": derived}, fh, indent=1)
        print("wrote %s" % args.out)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
