#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card MME row sweep (idea sweep G1-1, target-model predictions 3 and 4): GEMM time against row count.

Per case, one chain of the decode step's own GEMMs with distinct weights (the bytes of one step, so the 48 MB
SRAM cannot hold a reused weight), captured once per row count with ``wrap_in_hpu_graph`` and replayed:

* ``hc_pair``  97 x (hyper-connection down 10240 -> 336, its first 320 columns, up 320 -> 10240), bf16, replicated
* ``router``   48 x router 2560 -> 512, bf16, replicated (the same input: the GEMMs are independent)
* ``index_qk`` 12 x indexer projection 2560 -> 640, bf16, replicated
* ``lm_head``  4 x LM head shard 2560 -> 31,040, bf16 (calibration: the fork measured 2.28 TB/s at one row)
* ``moe``      8 x the vendor ``torch.ops.hpu.mixture_of_experts`` over 64 local FP8 experts (H 2560, I 640,
  top-10), called as ``VllmMixtureOfExpertsOpFP8PerChannel.forward`` does, each call with its own 64 experts.
  Two routings: ``one`` (every row: local expert 0 and nine remote ids 64 + 48 k, which this rank skips) and
  ``spread`` (the rows walk over all 64 local experts). Equal times mean the op reads every local expert.

Per row count: ``latency`` = one replay and a wait (median of 5), ``device`` = ten replays back to back and one
wait, divided by ten (the device time once the host is ahead). Per case: the effective bandwidth of the weights
``W / device`` and the MAC rate. A fresh process per case, ``PT_HPU_LAZY_ACC_PAR_MODE=0``.

    python3 bench_mme_rows.py --out mme_rows.json --cards 0,1,2,3,4   # one case per card, a few minutes
    python3 bench_mme_rows.py --selftest                     # CPU, small shapes: checks this script only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

RESULT_TAG = "BENCH_RESULT "
H, HC_W, HC_DOWN, HC_UP_IN = 2560, 10240, 336, 320
ROUTER_N, INDEX_N, LM_N = 512, 640, 31040
MOE_I, MOE_LOCAL, MOE_TOPK = 640, 64, 10
UNITS = {"hc_pair": 97, "router": 48, "index_qk": 12, "lm_head": 4, "moe": 8}
ROWS = {
    "hc_pair": (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 256, 1024, 8192),
    "router": (1, 4, 32, 128, 192, 1024, 8192),
    "index_qk": (1, 4, 32, 128, 192, 1024, 8192),
    "lm_head": (1, 4, 32, 128, 192),
    "moe": (1, 32, 128, 192),
}
SELFTEST_ROWS = (1, 4)


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def emit(record):
    print(RESULT_TAG + json.dumps(record), flush=True)


class Env:

    def __init__(self, selftest):
        import torch
        self.torch = torch
        self.selftest = selftest
        self.htorch = None
        self.device = "cpu"
        if not selftest:
            import habana_frameworks.torch as htorch
            self.htorch = htorch
            self.device = "hpu"

    def sync(self):
        if self.htorch is not None:
            self.htorch.core.mark_step()
            self.torch.hpu.synchronize()

    def wrap(self, module):
        if self.htorch is None:
            return module
        return self.htorch.hpu.wrap_in_hpu_graph(module, disable_tensor_cache=True)


def weights(env, n, rows_out, cols_in, scale, dtype=None):
    torch = env.torch
    gen = torch.Generator().manual_seed(rows_out * 7 + cols_in)
    dtype = dtype or torch.bfloat16
    return [(scale * torch.randn(rows_out, cols_in, generator=gen)).to(dtype).to(env.device) for _ in range(n)]


def build_case(env, case, small):
    """``(module, weight_bytes, macs_per_row, input_width)`` of one case."""
    torch = env.torch
    F = torch.nn.functional
    units = 2 if small else UNITS[case]
    hc_w, hc_down, hc_in = (64, 24, 16) if small else (HC_W, HC_DOWN, HC_UP_IN)
    h = 32 if small else H

    if case == "hc_pair":
        down = weights(env, units, hc_down, hc_w, hc_w**-0.5)
        up = weights(env, units, hc_w, hc_in, hc_in**-0.5)

        class M(torch.nn.Module):

            def forward(self, x):
                for wd, wu in zip(down, up):
                    d = F.linear(x, wd)
                    x = F.linear(d[:, :hc_in], wu)
                return x

        return M(), units * 2 * (hc_down * hc_w + hc_w * hc_in), units * (hc_down * hc_w + hc_w * hc_in), hc_w
    if case in ("router", "index_qk", "lm_head"):
        n_out = {"router": ROUTER_N, "index_qk": INDEX_N, "lm_head": LM_N}[case]
        n_out = 48 if small else n_out
        ws = weights(env, units, n_out, h, h**-0.5)

        class M(torch.nn.Module):

            def forward(self, x):
                acc = None
                for w in ws:
                    y = F.linear(x, w)
                    acc = y if acc is None else acc + y
                return acc

        return M(), units * 2 * n_out * h, units * n_out * h, h
    if case == "moe":
        if env.htorch is None:
            return None
        fp8 = torch.float8_e4m3fn
        gen = torch.Generator().manual_seed(11)
        sets = []
        for _ in range(units):
            w12 = tuple((torch.randn(2 * MOE_I, H, generator=gen) * 60).clamp(-224, 224).to(fp8).to(env.device)
                        for _ in range(MOE_LOCAL))
            w3 = tuple((torch.randn(H, MOE_I, generator=gen) * 60).clamp(-224, 224).to(fp8).to(env.device)
                       for _ in range(MOE_LOCAL))
            s12 = tuple(torch.full((2 * MOE_I, ), 1e-3, dtype=torch.float32).to(env.device) for _ in range(MOE_LOCAL))
            s3 = tuple(torch.full((H, ), 1e-3, dtype=torch.float32).to(env.device) for _ in range(MOE_LOCAL))
            sets.append((w12, w3, s12, s3))

        class M(torch.nn.Module):

            def forward(self, x, ids, wts):
                out = x
                for w12, w3, s12, s3 in sets:
                    xc = out.contiguous()
                    scale = ((xc.abs().amax(dim=-1, keepdim=True) + 1e-8) * (1.0 / 240.0)).float()
                    x8 = torch.ops.hpu.cast_to_fp8_v2(xc, 1.0 / scale, False, False, fp8)[0]
                    y = torch.ops.hpu.mixture_of_experts(hidden_states=x8, expert_routing_table=ids,
                                                         router_weights=wts, w12=w12, w3=w3,
                                                         d_scale_hidden_states=scale, d_scale_w12=s12,
                                                         d_scale_w3=s3, permuted_weights=True, activation="silu",
                                                         experts_min=0, experts_max=MOE_LOCAL - 1)
                    out = out + 0.01 * y
                return out

        expert_bytes = 2 * MOE_I * H + H * MOE_I
        return M(), units * MOE_LOCAL * expert_bytes, None, H
    raise ValueError(case)


def moe_inputs(env, rows, routing):
    torch = env.torch
    if routing == "one":
        ids = torch.tensor([0] + [64 + 48 * k for k in range(1, MOE_TOPK)]).repeat(rows, 1)
    else:
        ids = (torch.arange(rows).unsqueeze(1) * MOE_TOPK + torch.arange(MOE_TOPK)) % MOE_LOCAL
    wts = torch.full((rows, MOE_TOPK), 0.1, dtype=torch.bfloat16)
    return ids.to(torch.int64).to(env.device), wts.to(env.device)


def time_module(env, module, inputs, latency_n=5, device_n=10):
    for _ in range(2):  # capture / compile, then one replay
        out = module(*inputs)
        env.sync()
    lat = []
    for _ in range(latency_n):
        t0 = time.perf_counter()
        out = module(*inputs)
        env.sync()
        lat.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(device_n):
        out = module(*inputs)
    env.sync()
    device = (time.perf_counter() - t0) / device_n
    del out
    return 1e3 * _median(lat), 1e3 * device


def child(case, rows_list, selftest):
    env = Env(selftest)
    torch = env.torch
    built = build_case(env, case, small=selftest)
    if built is None:
        emit({"case": case, "skipped": "needs the card"})
        return 0
    module, weight_bytes, macs_per_row, width = built
    wrapped = env.wrap(module)
    env.sync()
    routings = ("one", "spread") if case == "moe" else (None, )
    for rows in rows_list:
        gen = torch.Generator().manual_seed(rows)
        x = torch.randn(rows, width, generator=gen).to(torch.bfloat16).to(env.device)
        for routing in routings:
            inputs = (x, ) if routing is None else (x, *moe_inputs(env, rows, routing))
            try:
                latency_ms, device_ms = time_module(env, wrapped, inputs)
            except Exception as exc:  # noqa: BLE001 - the next row count may still work
                emit({"case": case, "rows": rows, "routing": routing, "error": "%s: %s" % (type(exc).__name__,
                                                                                          str(exc)[:300])})
                continue
            record = {"case": case, "rows": rows, "routing": routing, "units": UNITS[case] if not selftest else 2,
                      "latency_ms": round(latency_ms, 4), "device_ms": round(device_ms, 4),
                      "weight_bytes": weight_bytes,
                      "weight_tb_s": round(weight_bytes / (device_ms * 1e-3) / 1e12, 3) if device_ms > 0 else None}
            if macs_per_row:
                record["tmac_s"] = round(macs_per_row * rows / (device_ms * 1e-3) / 1e12, 2)
            emit(record)
    emit({"case": case, "done": True})
    return 0


def start_child(case, rows_list, selftest, card):
    cmd = [sys.executable, os.path.abspath(__file__), "--child", case, "--rows",
           ",".join(str(r) for r in rows_list)]
    if selftest:
        cmd.append("--selftest")
    env = dict(os.environ)
    env.setdefault("PT_HPU_LAZY_MODE", "1")
    env.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    env.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)
    if card is not None:
        env["HABANA_VISIBLE_MODULES"] = str(card)
    out = tempfile.TemporaryFile(mode="w+")
    err = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, env=env, cwd=tempfile.gettempdir(), stdout=out, stderr=err, text=True)
    return {"case": case, "card": card, "proc": proc, "out": out, "err": err, "started": time.monotonic()}


def finish_child(child_run, timeout_s):
    proc = child_run["proc"]
    timed_out = False
    try:
        proc.wait(timeout=max(1.0, timeout_s - (time.monotonic() - child_run["started"])))
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    records, tail = [], []
    child_run["out"].seek(0)
    for line in child_run["out"].read().splitlines():
        if line.startswith(RESULT_TAG):
            try:
                records.append(json.loads(line[len(RESULT_TAG):]))
            except ValueError:
                pass
        else:
            tail.append(line)
    child_run["err"].seek(0)
    err = child_run["err"].read()
    return {"case": child_run["case"], "card": child_run["card"], "exit": proc.returncode, "timed_out": timed_out,
            "seconds": round(time.monotonic() - child_run["started"], 1), "records": records,
            "stderr_tail": err[-1500:], "stdout_tail": tail[-20:]}


def describe(rec):
    if rec.get("error"):
        return "%-8s rows %5s %-6s ERROR %s" % (rec["case"], rec.get("rows"), rec.get("routing") or "", rec["error"])
    if "rows" not in rec:
        return "%-8s %s" % (rec["case"], "done" if rec.get("done") else rec.get("skipped"))
    per_unit = 1e3 * rec["device_ms"] / rec["units"]
    return ("%-8s rows %5d %-6s device %8.3f ms (%7.2f us per unit), latency %8.3f ms, weights %5.2f TB/s%s" %
            (rec["case"], rec["rows"], rec["routing"] or "", rec["device_ms"], per_unit, rec["latency_ms"],
             rec["weight_tb_s"] or 0.0, ", %.1f TMAC/s" % rec["tmac_s"] if "tmac_s" in rec else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench_mme_rows.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=",".join(UNITS))
    ap.add_argument("--rows", default=None, help="row counts (default: per case)")
    ap.add_argument("--child-timeout", type=float, default=600.0)
    ap.add_argument("--cards", default="", help="cards for the cases, one child per card at a time (default: one)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        rows = [int(r) for r in args.rows.split(",")] if args.rows else list(ROWS[args.child])
        return child(args.child, rows, args.selftest)
    cases = [c for c in args.cases.split(",") if c in UNITS]
    cards = [c.strip() for c in args.cards.split(",") if c.strip()] if args.cards else [None]
    runs, pending = [], []
    for index, case in enumerate(cases):
        rows = [int(r) for r in args.rows.split(",")] if args.rows else list(
            SELFTEST_ROWS if args.selftest else ROWS[case])
        pending.append(start_child(case, rows, args.selftest, cards[index % len(cards)]))
        if len(pending) == len(cards) or index == len(cases) - 1:
            for child_run in pending:  # one child per card at a time
                runs.append(finish_child(child_run, args.child_timeout))
            pending = []
    for run in runs:
        print("-- %s on card %s: exit %s, %.0f s%s" % (run["case"], run["card"], run["exit"], run["seconds"],
                                                      ", TIMED OUT" if run["timed_out"] else ""), flush=True)
        for rec in run["records"]:
            print("   " + describe(rec), flush=True)
        if run["exit"] not in (0, None) or not run["records"]:
            print("   stderr: " + run["stderr_tail"][-600:].replace("\n", "\n   "), flush=True)
    if args.selftest:
        print("SELFTEST ONLY (CPU, small shapes): the lines above check this script, they are not results.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tool": "bench_mme_rows", "selftest": args.selftest, "runs": runs}, fh, indent=1)
        print("wrote %s" % args.out)
    ok = all(any(r.get("done") for r in run["records"]) for run in runs)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
