#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card microbenchmark 3 of the sparse-QSA decode path (qsa-fused, 2026-11-21): the two nodes to cut.

step-anatomy's table of build 4 (a 128-row MTP verify step, TP8): the attention's integer division and remainder
(`decode_selected_slots`: block = pos // 128, offset = pos % 128 over [rows, 2,051]) 2.44 ms per step (94 us per
execution), and the indexer's top-k (a bitonic sort of 32,768 fp32 scores per row, k = 512) 2.98 ms per step.

A. Slot mapping over [rows, 2,051] positions (rows 16, 32, 128): form 0 `//` and `%` (today), form 1 shift and
   mask, form 2 through float32 (floor(pos * 2**-7), exact below 2**24). Each checked equal to form 0.
B. Top-k 512 of [rows, 32,768] masked scores (rows 4, 16, 32, 128; the visible-block mask of select_blocks):
   topk sorted (today), topk unsorted, two stages (topk 512 of each 4,096 or 8,192 columns, then of the candidates),
   bf16 scores (not exact), and a threshold select (binary search of the 512th value on the int32 bit patterns of
   the non-negative scores, 31 counting passes, then a cumsum compaction: no sort). Each checked against the
   sorted ids of today's form (the same set per row).

Each case is one lazy graph; device ms = median of (issue + mark_step + synchronize) minus the median issue time.
Lines `QSA_BENCH3 {json}`.

    python3 qsa_bench3.py --out FILE.json          # on a card
    python3 qsa_bench3.py --selftest               # CPU, small shapes: checks this script only
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

TAG = "QSA_BENCH3 "
RESULTS = []


def emit(rec):
    RESULTS.append(rec)
    print(TAG + json.dumps(rec), flush=True)


def med(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=(os.environ["FARM_OUT"] + "/qsa_bench3.json") if os.environ.get("FARM_OUT") else None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warm", type=int, default=3)
    args = ap.parse_args()

    import torch
    ht = None
    dev = "cpu"
    if not args.selftest:
        import habana_frameworks.torch as ht  # noqa: F811
        dev = "hpu"

    def mark():
        if ht is not None:
            ht.core.mark_step()

    def sync():
        if ht is not None:
            torch.hpu.synchronize()

    def timed(name, fn, extra=None):
        rec = {"case": name}
        rec.update(extra or {})
        out = None
        try:
            issue, total, first = [], [], None
            for i in range(args.warm + args.iters):
                sync()
                t0 = time.perf_counter()
                out = fn()
                t1 = time.perf_counter()
                mark()
                sync()
                t2 = time.perf_counter()
                if i == 0:
                    first = t2 - t0
                if i >= args.warm:
                    issue.append(t1 - t0)
                    total.append(t2 - t0)
            rec.update({"ok": True, "first_ms": 1e3 * first, "issue_ms": 1e3 * med(issue),
                        "total_ms": 1e3 * med(total), "device_ms": 1e3 * (med(total) - med(issue)),
                        "min_ms": 1e3 * min(total)})
        except Exception as exc:  # noqa: BLE001
            rec.update({"ok": False, "err": repr(exc)[:600], "tb": traceback.format_exc()[-900:]})
            out = None
        return rec, out

    small = args.selftest
    W, BS, MB = (51, 16, 64) if small else (2051, 128, 1024)
    N, K = (1024, 64) if small else (32768, 512)
    g = torch.Generator().manual_seed(3)

    # ------------------------------------------------------------------ A. slot mapping
    shift = BS.bit_length() - 1
    for rows in ([4, 8] if small else [16, 32, 128]):
        sel = torch.randint(-1, MB * BS, (rows, W), generator=g, dtype=torch.int32)
        table = torch.randperm(rows * MB * 4, generator=g)[:rows * MB].reshape(rows, MB).to(torch.int32)
        sel_d, table_d = sel.to(dev), table.to(dev)
        mark()

        def form0():
            pos = sel_d.to(torch.long).clamp_min(0)
            return table_d.to(torch.long).gather(1, pos // BS) * BS + (pos % BS)

        def form1():
            pos = sel_d.to(torch.long).clamp_min(0)
            return table_d.to(torch.long).gather(1, pos >> shift) * BS + (pos & (BS - 1))

        def form2():
            pos = sel_d.to(torch.long).clamp_min(0)
            blk = torch.floor(pos.to(torch.float32) * (1.0 / BS)).to(torch.long)
            return table_d.to(torch.long).gather(1, blk) * BS + (pos - blk * BS)

        def form3():  # int32 throughout (the card's own width), shift and mask
            pos = sel_d.clamp_min(0)
            return table_d.gather(1, (pos >> shift).to(torch.long)) * BS + (pos & (BS - 1))

        ref = None
        for name, fn in (("slots_divmod", form0), ("slots_shift_mask", form1), ("slots_float_floor", form2),
                         ("slots_shift_mask_i32", form3)):
            rec, out = timed(name, fn, {"rows": rows, "W": W})
            if out is not None:
                o = out.to("cpu").to(torch.long)
                if ref is None:
                    ref = o
                rec["equal_to_divmod"] = bool(torch.equal(o, ref))
            emit(rec)

    # ------------------------------------------------------------------ B. top-k 512 of 32,768
    def topk_sorted(masked, k):
        return torch.topk(masked, k, dim=-1)[1]

    def topk_unsorted(masked, k):
        return torch.topk(masked, k, dim=-1, sorted=False)[1]

    def two_stage(masked, k, chunk):
        rows, n = masked.shape
        c = n // chunk
        v, i = torch.topk(masked.view(rows, c, chunk), k, dim=-1, sorted=False)  # [rows, c, k]
        base = (torch.arange(c, device=masked.device) * chunk).view(1, c, 1)
        cand_v, cand_i = v.reshape(rows, c * k), (i + base).reshape(rows, c * k)
        j = torch.topk(cand_v, k, dim=-1, sorted=False)[1]
        return cand_i.gather(1, j)

    def bf16_topk(masked, k):
        return torch.topk(masked.to(torch.bfloat16), k, dim=-1, sorted=False)[1]

    def threshold_select(masked, k):
        """The 512th largest by a binary search on the int32 bit patterns (scores >= 0; masked -> -1), then the
        columns above it and the first ties up to k, compacted with a cumsum (ascending column order)."""
        rows, n = masked.shape
        bits = masked.view(torch.int32)
        bits = torch.where(masked >= 0, bits, torch.full_like(bits, -1))
        lo = torch.zeros(rows, 1, dtype=torch.int32, device=masked.device)
        hi = torch.full((rows, 1), 0x7F800000, dtype=torch.int32, device=masked.device)
        for _ in range(31):
            mid = lo + ((hi - lo + 1) >> 1)
            ge = (bits >= mid).sum(dim=1, keepdim=True, dtype=torch.int32) >= k
            lo = torch.where(ge, mid, lo)
            hi = torch.where(ge, hi, mid - 1)
        greater = bits > lo
        equal = bits == lo
        need = k - greater.sum(dim=1, keepdim=True, dtype=torch.int32)
        take = greater | (equal & (torch.cumsum(equal.to(torch.int32), dim=1) <= need))
        slot = torch.cumsum(take.to(torch.int32), dim=1) - 1
        slot = torch.where(take, slot, torch.full_like(slot, k))
        cols = torch.arange(n, dtype=torch.int32, device=masked.device).expand(rows, n)
        out = torch.full((rows, k + 1), -1, dtype=torch.int32, device=masked.device)
        out = out.scatter(1, slot.to(torch.long), cols)
        return out[:, :k]

    for rows in ([4, 8] if small else [4, 16, 32, 128]):
        scores = torch.rand(rows, N, generator=g) * 10.0
        visible = torch.randint(N // 2, N + 1, (rows, ), generator=g)
        visible[0] = N
        scores_d, visible_d = scores.to(dev), visible.to(dev)
        mark()
        cols_d = torch.arange(N, device=dev)

        def masked_scores():
            return scores_d.masked_fill(cols_d[None, :] >= visible_d[:, None], float("-inf"))

        ref = None
        cases = [("topk_sorted", lambda: topk_sorted(masked_scores(), K)),
                 ("topk_unsorted", lambda: topk_unsorted(masked_scores(), K)),
                 ("topk_two_stage_4096", lambda: two_stage(masked_scores(), K, 4096 if not small else 256)),
                 ("topk_two_stage_8192", lambda: two_stage(masked_scores(), K, 8192 if not small else 512)),
                 ("topk_bf16", lambda: bf16_topk(masked_scores(), K)),
                 ("threshold_select", lambda: threshold_select(masked_scores(), K))]
        for name, fn in cases:
            rec, out = timed(name, fn, {"rows": rows, "N": N, "k": K})
            if out is not None:
                o = torch.sort(out.to("cpu").to(torch.long), dim=1)[0]
                if ref is None:
                    ref = o
                rec["same_set_rows"] = int((o == ref).all(dim=1).sum())
                rec["rows_total"] = rows
            emit(rec)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(RESULTS, f, indent=1)
    bad = [r for r in RESULTS if not r.get("ok")]
    print(TAG + "done: %d cases, %d failed" % (len(RESULTS), len(bad)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
