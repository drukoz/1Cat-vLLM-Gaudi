#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card bench of the QSA indexer top-k: the vendor sort (``torch.topk`` of the masked row, today) against the
one-node kernel ``qsa_index_topk`` (``VLLM_HPU_QSA_TOPK_FORM=4``), plus a probe of the vendor sort's ties.

    bash tools/flash_next/kernel_probes/with_kernels.sh python tools/flash_next/hc_kernels/bench_qsa_topk.py \
        --out "$FARM_OUT"
    HABANA_PROFILE=profile_api_light ... bench_qsa_topk.py --trace --rows 16 --out "$FARM_OUT/trace"
    PROBE_DEVICE=cpu PYTHONPATH=<tree> python tools/flash_next/hc_kernels/bench_qsa_topk.py --quick   # dry run

Per row count (default 4, 16, 32, 128; N = 32,768 compressed blocks, k = 512) and per form:
  * ``sorted``: ``torch.topk(masked, k)`` (form 0, what the served graph runs);
  * ``kernel``: ``qsa_index_topk(masked, k)`` (the set, ascending ids);
  * ``kernel_sorted``: the kernel, then its k picks ordered by value as forms 1-3 do (``torch.topk`` of the k gathered
    values): what form 4 returns in the served path;
K calls CHAINED (call i + 1's scores get 1e-30 x one id of call i, a dependency the graph compiler cannot drop, as
consecutive layers have), in one HPU graph, at two K; the slope is the time per call (``chain_add`` alone is
measured too: subtract it). The scores are relu-like sums (distinct non-negative values; the visible prefix of every
row from a seeded mix of long and short contexts) and the selected SET of every form is compared with ``sorted``'s
on the card. ``--trace``: a torch-profiler trace of three replays per form at the larger K.

Tie probe (``--ties``): crafted rows where ``torch.topk`` must choose among equal values at the cut (all zeros; 300
positives and zeros; a coarse grid): which columns it takes, against the lowest-columns rule of the kernel, and the
order it returns them in. Lines "BENCH qsa-topk ..." as it goes and "VERDICT qsa-topk: ..." at the end; everything
in <out>/bench_qsa_topk.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

DEVICE = os.environ.get("PROBE_DEVICE", "hpu")
N_DEFAULT = 32768
K_DEFAULT = 512


def say(text: str) -> None:
    print(f"BENCH qsa-topk {text}", flush=True)


def masked_scores(torch, rows: int, n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    scores = torch.relu(torch.randn(rows, n, generator=gen)) / 11.3 + torch.rand(rows, n, generator=gen) * 1e-6
    # visible prefixes: mostly long contexts (the served 32-agent mix), some short ones
    visible = torch.where(torch.rand(rows, generator=gen) < 0.75, torch.randint(n // 2, n + 1, (rows, ), generator=gen),
                          torch.randint(1, 4096, (rows, ), generator=gen))
    cols = torch.arange(n)
    return scores.masked_fill(cols[None, :] >= visible[:, None], float("-inf"))


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "qsa_topk_bench")))
    ap.add_argument("--rows", default="4,16,32,128")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--topk", type=int, default=K_DEFAULT)
    ap.add_argument("--k", default="4,12", help="calls per graph (two values: the slope is per call)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--forms", default="chain_add,sorted,kernel,kernel_sorted")
    ap.add_argument("--ties", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.trace and DEVICE == "hpu" and os.environ.get("HABANA_PROFILE") != "profile_api_light":
        ap.error("--trace needs HABANA_PROFILE=profile_api_light in the environment before torch is imported")
    a.out.mkdir(parents=True, exist_ok=True)

    import torch

    htorch = None
    if DEVICE == "hpu":
        import habana_frameworks.torch as htorch  # noqa: F811
        from flashinfer_gaudi._native import load_qsa_fused_extension, qsa_fused_extension_errors
        if not load_qsa_fused_extension():
            raise SystemExit("qsa_fused_ops extension is not loaded: " + "; ".join(qsa_fused_extension_errors()))
    else:
        from tests.unit_tests.qsa_fused import qsa_ops  # noqa: F401  (CPU kernels: the references)
    from flashinfer_gaudi.qsa_fused import qsa_index_topk

    k_lo, k_hi = (int(v) for v in a.k.split(","))
    if a.quick:
        k_lo, k_hi, a.iters, a.n = 1, 2, 1, 2048
    results = {"n": a.n, "topk": a.topk, "k": [k_lo, k_hi], "rows": {}, "ties": {}}

    def sync():
        if DEVICE == "hpu":
            torch.hpu.synchronize()

    def form_call(form: str, masked):
        if form == "chain_add":
            return masked[:, :1].to(torch.int32)
        if form == "sorted":
            return torch.topk(masked, a.topk, dim=-1)[1]
        ids = qsa_index_topk(masked, a.topk)
        if form == "kernel":
            return ids
        n = masked.shape[1]
        idx = torch.where(ids >= 0, ids, torch.full_like(ids, n - 1)).long()
        order = torch.topk(masked.gather(1, idx), a.topk, dim=-1)[1]
        return idx.gather(1, order)

    forms = a.forms.split(",")
    for rows in (int(v) for v in a.rows.split(",")):
        masked = masked_scores(torch, rows, a.n, seed=rows).to(DEVICE)
        entry = results["rows"].setdefault(str(rows), {})
        # the selected sets on the device, against sorted's
        want = form_call("sorted", masked).cpu()
        for form in ("kernel", "kernel_sorted"):
            if form not in forms:
                continue
            got = form_call(form, masked).cpu()
            same = 0
            for t in range(rows):
                w = sorted(v for v in want[t].tolist() if v >= 0 and float(masked[t, v]) != float("-inf"))
                g = sorted(v for v in got[t].tolist() if v >= 0 and float(masked[t, v]) != float("-inf"))
                same += int(w == g)
            entry.setdefault("same_set_rows", {})[form] = same
            say(f"rows {rows:3d} {form:14s}: the set of sorted on {same}/{rows} rows")
        for form in forms:
            walls = {}
            for k in (k_lo, k_hi):

                class Calls(torch.nn.Module):

                    def forward(self, x, k=k, form=form):
                        outs = []
                        cur = x
                        for _ in range(k):
                            out = form_call(form, cur)
                            outs.append(out)
                            cur = x + out.reshape(-1)[:1].to(torch.float32) * 1e-30  # the chain
                        return tuple(outs)

                module = Calls()
                if htorch is not None:
                    module = htorch.hpu.wrap_in_hpu_graph(module)
                for _ in range(2):
                    module(masked)
                    sync()
                times = []
                for _ in range(a.iters):
                    t0 = time.perf_counter()
                    module(masked)
                    sync()
                    times.append((time.perf_counter() - t0) * 1e3)
                walls[k] = statistics.median(times)
                if a.trace and k == k_hi and DEVICE == "hpu":
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.HPU]) as profiler:
                        for _ in range(3):
                            module(masked)
                        sync()
                    path = a.out / f"trace_{form}_r{rows}_k{k}.json"
                    profiler.export_chrome_trace(str(path))
                    say(f"trace {path}")
            per_call = (walls[k_hi] - walls[k_lo]) / (k_hi - k_lo) * 1e3
            entry[form] = {"wall_ms": walls, "us_per_call": per_call}
            say(f"rows {rows:3d} {form:14s}: {per_call:8.1f} us per call (replay {walls[k_lo]:.3f} ms at K={k_lo}, "
                f"{walls[k_hi]:.3f} ms at K={k_hi})")
            (a.out / "bench_qsa_topk.json").write_text(json.dumps(results, indent=1) + "\n")

    if a.ties:
        n = a.n
        probes = {}
        zeros = torch.zeros(2, n)
        probes["all_zero"] = zeros
        few = torch.zeros(2, n)
        few[:, torch.randperm(n, generator=torch.Generator().manual_seed(1))[:300]] = 1.0
        probes["300_positive_then_zeros"] = few
        grid = (torch.rand(2, n, generator=torch.Generator().manual_seed(2)) * 16).floor() / 16
        probes["grid_of_16"] = grid
        for name, scores in probes.items():
            ids = torch.topk(scores.to(DEVICE), a.topk, dim=-1)[1].cpu()
            vals = scores.gather(1, ids)
            report = {}
            for t in range(scores.shape[0]):
                theta = float(vals[t, -1])
                tied = (scores[t] == theta).nonzero().reshape(-1)
                taken = ids[t][vals[t] == theta]
                need = taken.numel()
                lowest = tied[:need]
                highest = tied[-need:] if need else tied[:0]
                report[t] = {"cut": theta, "tied": int(tied.numel()), "taken": int(need),
                             "lowest_columns": bool(torch.equal(torch.sort(taken).values, lowest)),
                             "highest_columns": bool(torch.equal(torch.sort(taken).values, highest)),
                             "taken_first": taken[:8].tolist(),
                             "order_of_ties_ascending": bool(torch.equal(taken, torch.sort(taken).values)),
                             "values_descending": bool((vals[t][1:] <= vals[t][:-1]).all())}
                say(f"ties {name} row {t}: {report[t]}")
            probes[name] = None
            results["ties"][name] = report
        (a.out / "bench_qsa_topk.json").write_text(json.dumps(results, indent=1) + "\n")

    parts = []
    for rows, entry in results["rows"].items():
        cells = [f"{form} {entry[form]['us_per_call']:.1f}" for form in forms if form in entry]
        parts.append(f"rows {rows}: " + ", ".join(cells) + " us")
    print("VERDICT qsa-topk: " + "; ".join(parts), flush=True)
    (a.out / "bench_qsa_topk.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
