#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What one reduction of [rows, 2560] bf16 costs on the card, as an all-reduce and as an all-gather + local sum.

Collectives track (G4-1, I34, VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS). A body of 96 times (one element-wise op, then
the reduction of ``[rows, 2560]`` bf16: the 96 layer all-reduces of a Flash-Next decode step) is timed per call

* in two regimes: ``lazy`` (issued op by op, one ``mark_step`` at the end) and ``replay`` (captured once with
  ``wrap_in_hpu_graph(disable_tensor_cache=True)`` and replayed with a fresh input, as the runner serves decode);
  ``replay_async`` (the wrapper's ``asynchronous=True``) on request;
* in four forms: ``none`` (no reduction), ``all_reduce`` (today), ``gather_sum`` (the all-gather of the ``world``
  partials, a view, an fp32 sum and a cast: the flag's form) and ``gather_first`` (the all-gather and a slice: the
  collective alone, what a kernel that sums the gathered rows itself would leave).

After it the step's last collective (``--tail-rows``): a body of 4 all-gathers of the logits (``[rows, 248320 /
world]`` bf16 per rank, ``tail_logits``) against 4 all-gathers of the ``[rows, 2]`` fp32 pairs that
VLLM_HPU_VOCAB_PARALLEL_LOGITS sends instead (``tail_pairs``), each over ``tail_none``.

One process per card, the hccl backend over all ranks. Per (regime, rows): the cost of one reduction is
``(form - none) / 96`` of the slowest rank's median; ``call`` is the time until the call returns on the host,
``total`` until ``torch.hpu.synchronize()`` returns. The fits give alpha (one reduction of the smallest payload)
and the cost per MB of payload (the slope from the smallest to the largest row count).

    python3 bench_c4.py --ranks 8 --out c4_r8.json
    python3 bench_c4.py --ranks 4 --cards 0,1,2,3 --out c4_r4.json
    python3 bench_c4.py --selftest      # gloo on the CPU: checks the script only (2 ranks)

Stdlib + torch (+ habana_frameworks on the card). Prints one line per (regime, rows, form) and one per fit;
``decide_g41.py`` reads the JSON files of the 8- and 4-rank runs.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

TAG = "C4_RESULT "
HIDDEN = 2560
LAYERS = 96
FORMS = ("none", "all_reduce", "gather_sum", "gather_first")
VOCAB = 248320
TAIL_GATHERS = 4  # all-gathers per call of the tail body
TAIL_FORMS = ("tail_none", "tail_logits", "tail_pairs")


def child(args):
    import torch
    import torch.distributed as dist
    htorch = None
    if args.selftest:
        device, backend = "cpu", "gloo"
    else:
        import habana_frameworks.torch as htorch  # noqa: F811
        import habana_frameworks.torch.core  # noqa: F401
        try:
            import habana_frameworks.torch.distributed.hccl  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
        torch.hpu.set_device(args.rank)
        device, backend = "hpu", "hccl"
    dist.init_process_group(backend=backend, init_method=args.init, world_size=args.ranks, rank=args.rank,
                            timeout=timedelta(seconds=600))
    group = dist.new_group(list(range(args.ranks)), backend=backend)
    world = args.ranks

    def mark():
        if htorch is not None:
            htorch.core.mark_step()

    def sync():
        if htorch is not None:
            torch.hpu.synchronize()

    def all_reduce(x):
        dist.all_reduce(x, group=group)
        return x

    def gathered(x):
        out = torch.empty((world * x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    def gather_sum(x):
        return gathered(x).view(world, x.shape[0], x.shape[1]).sum(dim=0, dtype=torch.float32).to(x.dtype)

    def gather_first(x):
        return gathered(x).narrow(0, 0, x.shape[0])

    # (the reduction, the factor of the element-wise op): every form returns ones for an input of ones
    forms = {"none": (None, 1.0), "all_reduce": (all_reduce, 1.0 / world), "gather_sum": (gather_sum, 1.0 / world),
             "gather_first": (gather_first, 1.0)}

    class Step(torch.nn.Module):

        def __init__(self, reduce, scale):
            super().__init__()
            self.reduce = reduce
            self.scale = scale

        def forward(self, x):
            for _ in range(LAYERS):
                x = x * self.scale + 0.0
                if self.reduce is not None:
                    x = self.reduce(x)
            return x

    def timed(call, rows, repeats):
        totals, calls, first = [], [], None
        for i in range(3 + repeats):
            x = torch.full((rows, HIDDEN), 1.0, dtype=torch.bfloat16, device=device)
            mark()
            sync()
            if args.selftest:
                dist.barrier(group=group)
            else:  # a barrier on the device stream: a one-element all-reduce, executed before the clock starts
                all_reduce(torch.zeros(1, device=device))
                mark()
            sync()
            t0 = time.perf_counter()
            y = call(x)
            t1 = time.perf_counter()
            mark()
            sync()
            t2 = time.perf_counter()
            if i == 0:
                first = t2 - t0
            if i >= 3:
                totals.append(t2 - t0)
                calls.append(t1 - t0)
        value = float(y.float().mean().cpu())
        totals.sort()
        calls.sort()
        return {"p50_ms": 1e3 * totals[len(totals) // 2], "min_ms": 1e3 * totals[0],
                "call_p50_ms": 1e3 * calls[len(calls) // 2], "first_ms": 1e3 * first, "value": value}

    # The step's last collective: the all-gather of the logits ([rows, vocabulary / world] bf16 per rank) against
    # the all-gather of the (value, id) pairs ([rows, 2] fp32) that VLLM_HPU_VOCAB_PARALLEL_LOGITS sends instead.
    shard = VOCAB // world

    class Tail(torch.nn.Module):

        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        def forward(self, x):  # x: [rows, shard] bf16
            for _ in range(TAIL_GATHERS):
                x = x * 1.0 + 0.0
                if self.kind == "tail_logits":
                    x = gathered(x).narrow(0, 0, x.shape[0])
                elif self.kind == "tail_pairs":
                    pairs = gathered(x.narrow(-1, 0, 2).to(torch.float32))
                    x = x + pairs.narrow(0, 0, x.shape[0]).narrow(-1, 0, 1).to(x.dtype) * 0.0
            return x

    def timed_tail(call, rows, repeats):
        totals, calls = [], []
        for i in range(2 + repeats):
            x = torch.full((rows, shard), 1.0, dtype=torch.bfloat16, device=device)
            mark()
            sync()
            if args.selftest:
                dist.barrier(group=group)
            else:
                all_reduce(torch.zeros(1, device=device))
                mark()
            sync()
            t0 = time.perf_counter()
            y = call(x)
            t1 = time.perf_counter()
            mark()
            sync()
            t2 = time.perf_counter()
            if i >= 2:
                totals.append(t2 - t0)
                calls.append(t1 - t0)
        value = float(y.narrow(-1, 0, 4).float().mean().cpu())
        totals.sort()
        calls.sort()
        return {"p50_ms": 1e3 * totals[len(totals) // 2], "min_ms": 1e3 * totals[0],
                "call_p50_ms": 1e3 * calls[len(calls) // 2], "value": value}

    results = []
    for rows in args.rows:
        for regime in args.regimes:
            for name in args.forms:
                reduce, scale = forms[name]
                module = Step(reduce, scale)
                try:
                    if regime == "replay" and htorch is not None:  # as the runner wraps the model
                        module = htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)
                    elif regime == "replay_async" and htorch is not None:  # the wrapper's asynchronous replay
                        module = htorch.hpu.wrap_in_hpu_graph(module, asynchronous=True, disable_tensor_cache=True)
                except Exception as exc:  # noqa: BLE001 - an unknown keyword on this bridge: report it
                    rec = {"error": "wrap: %s: %s" % (type(exc).__name__, str(exc)[:300]), "rank": args.rank,
                           "rows": rows, "regime": regime, "form": name}
                    results.append(rec)
                    print(TAG + json.dumps(rec), flush=True)
                    continue
                try:
                    rec = timed(module, rows, args.repeats)
                except Exception as exc:  # noqa: BLE001 - report and go on
                    rec = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
                rec.update({"rank": args.rank, "rows": rows, "regime": regime, "form": name})
                results.append(rec)
                print(TAG + json.dumps(rec), flush=True)
    for rows in args.tail_rows:  # after the layer body, so that nothing here can cost its results
        for regime in args.regimes:
            for name in TAIL_FORMS:
                module = Tail(name)
                try:
                    if regime.startswith("replay") and htorch is not None:
                        module = htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)
                    rec = timed_tail(module, rows, max(4, args.repeats // 2))
                except Exception as exc:  # noqa: BLE001 - report and go on
                    rec = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
                rec.update({"rank": args.rank, "rows": rows, "regime": regime, "form": name})
                results.append(rec)
                print(TAG + json.dumps(rec), flush=True)
    print(TAG + json.dumps({"done": args.rank}), flush=True)
    os._exit(0)


def summarize(records, ranks, cards, regimes):
    """``(summary lines, fits)`` from the ranks' records: the slowest rank's medians, the cost of one reduction."""
    by_key = {}
    for rec in records:
        if "form" in rec:
            by_key.setdefault((rec["regime"], rec["rows"], rec["form"]), []).append(rec)
    summary = []
    for (regime, rows, form), recs in sorted(by_key.items()):
        p50 = [r["p50_ms"] for r in recs if "p50_ms" in r]
        call = [r["call_p50_ms"] for r in recs if "call_p50_ms" in r]
        base, per = ("tail_none", TAIL_GATHERS) if form.startswith("tail_") else ("none", LAYERS)
        none = [r["p50_ms"] for r in by_key.get((regime, rows, base), []) if "p50_ms" in r]
        none_call = [r["call_p50_ms"] for r in by_key.get((regime, rows, base), []) if "call_p50_ms" in r]
        line = {"regime": regime, "rows": rows, "form": form, "ranks_ok": len(p50),
                "p50_ms_slowest_rank": max(p50) if p50 else None,
                "call_p50_ms_slowest_rank": max(call) if call else None,
                "first_ms_slowest_rank": max((r["first_ms"] for r in recs if "first_ms" in r), default=None),
                "values": sorted({round(r.get("value", float("nan")), 4) for r in recs if "value" in r}),
                "errors": [r["error"] for r in recs if "error" in r][:2]}
        if p50 and none and form != base:
            line["per_reduction_us"] = 1e3 * (max(p50) - max(none)) / per
            if call and none_call:  # the host's share: time inside the call, per reduction
                line["per_reduction_call_us"] = 1e3 * (max(call) - max(none_call)) / per
        summary.append(line)
    # alpha (one reduction of the smallest payload) and the cost per MB of payload, per regime and form
    fits = []
    for regime in regimes:
        for form in FORMS[1:]:
            points = sorted((line["rows"], line["per_reduction_us"]) for line in summary
                            if line["regime"] == regime and line["form"] == form and "per_reduction_us" in line)
            if len(points) < 2:
                continue
            (r0, c0), (r1, c1) = points[0], points[-1]
            per_mb = (c1 - c0) / ((r1 - r0) * HIDDEN * 2 / 1e6)
            fits.append({"ranks": ranks, "cards": cards or "inherited", "regime": regime, "form": form,
                         "alpha_us": c0, "alpha_rows": r0, "us_per_mb": per_mb, "points": points})
    return summary, fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--rows", default="1,8,32,128")
    ap.add_argument("--regimes", default="lazy,replay")
    ap.add_argument("--forms", default=",".join(FORMS))
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--tail-rows", default="32,128",
                    help="row counts of the logits all-gather against the (value, id) all-gather; empty: none")
    ap.add_argument("--out", default="c4.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--cards", default="", help="HABANA_VISIBLE_MODULES of the ranks (default: inherited)")
    ap.add_argument("--logdir", default="", help="where the rank logs go (default: a temporary folder)")
    ap.add_argument("--timeout-s", type=float, default=1200.0)
    args = ap.parse_args()
    args.rows = [int(r) for r in str(args.rows).split(",") if r]
    args.regimes = [r for r in str(args.regimes).split(",") if r]
    args.forms = [f for f in str(args.forms).split(",") if f in FORMS]
    args.tail_rows = [int(r) for r in str(args.tail_rows).split(",") if r]
    if "none" not in args.forms:
        args.forms.insert(0, "none")
    if args.child:
        return child(args)
    if args.selftest:
        args.ranks = 2
        args.repeats = 2
    workdir = tempfile.mkdtemp(prefix="c4-")
    init = "file://" + os.path.join(workdir, "init")
    env = dict(os.environ)
    if not args.selftest:  # as the plugin runs lazy tensor parallel
        env.setdefault("PT_HPU_LAZY_MODE", "1")
        env.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        env.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        if args.cards:
            env["HABANA_VISIBLE_MODULES"] = args.cards
    procs = []
    for rank in range(args.ranks):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--rank", str(rank), "--ranks", str(args.ranks),
               "--init", init, "--rows", ",".join(map(str, args.rows)), "--regimes", ",".join(args.regimes),
               "--forms", ",".join(args.forms), "--repeats", str(args.repeats),
               "--tail-rows", ",".join(map(str, args.tail_rows))] + (["--selftest"] if args.selftest else [])
        logdir = args.logdir or workdir
        os.makedirs(logdir, exist_ok=True)
        log = open(os.path.join(logdir, "rank%d.log" % rank), "w")  # a file, not a pipe: no rank blocks on output
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env), log))
    records = []
    deadline = time.monotonic() + args.timeout_s
    finished = 0
    for rank, (proc, log) in enumerate(procs):
        try:
            proc.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            print("c4: rank %d did not end in %d s: killed" % (rank, args.timeout_s), flush=True)
        log.close()
        with open(log.name) as f:
            out = f.read()
        done = False
        for line in out.splitlines():
            if line.startswith(TAG):
                rec = json.loads(line[len(TAG):])
                done = done or "done" in rec
                records.append(rec)
            elif rank == 0 and ("Error" in line or "error" in line):
                print("rank0: " + line[:300])
        finished += int(done)
        if not done:
            print("c4: rank %d ended without its last record; the end of its log:" % rank, flush=True)
            print("\n".join("  " + ln[:300] for ln in out.splitlines()[-12:]), flush=True)
    summary, fits = summarize(records, args.ranks, args.cards, args.regimes)
    for line in summary:
        print("c4: %-12s rows %4d %-12s %s ms per body (call %s ms); per collective %s us (call %s us); "
              "values %s %s" % (
                  line["regime"], line["rows"], line["form"],
                  ("%.3f" % line["p50_ms_slowest_rank"]) if line["p50_ms_slowest_rank"] is not None else "-",
                  ("%.3f" % line["call_p50_ms_slowest_rank"]) if line["call_p50_ms_slowest_rank"] is not None else "-",
                  ("%.1f" % line["per_reduction_us"]) if "per_reduction_us" in line else "-",
                  ("%.1f" % line["per_reduction_call_us"]) if "per_reduction_call_us" in line else "-",
                  line["values"], line["errors"] or ""), flush=True)
    for fit in fits:
        print("c4-fit: ranks %d regime %-12s %-12s alpha %.1f us (%d row), %.1f us per MB of payload; points %s" % (
            fit["ranks"], fit["regime"], fit["form"], fit["alpha_us"], fit["alpha_rows"], fit["us_per_mb"],
            [(r, round(c, 1)) for r, c in fit["points"]]), flush=True)
    with open(args.out, "w") as f:
        json.dump({"ranks": args.ranks, "cards": args.cards or "inherited", "records": records, "summary": summary,
                   "fits": fits}, f, indent=1)
    return 0 if finished == args.ranks else 2


if __name__ == "__main__":
    sys.exit(main())
