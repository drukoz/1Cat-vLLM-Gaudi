#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-card bench: the host time of a decode step's small host-to-device copies, one by one against packed.

A Flash-Next decode step copies 11 to 14 small host tensors to the card (token ids, positions, slot mapping, block
table, query start locations, PLE n-gram context, recurrent state indices, logits indices; ``async_h2d_copy``).
On the Mac the runner's whole input preparation takes about 0.3 ms at 32 rows, on the card step-anatomy measured
about 2 ms: the copies' host cost on the card is the suspect. This bench times, on one card in lazy mode:

* ``separate``: the decode step's copies as the runner makes them (shapes and dtypes of a 32-row TP8 step: about
  ten tensors from [32] to [32, 1024]), each ``tensor.to("hpu", non_blocking=True)``, then one ``mark_step``;
* ``packed_int32``: the same values in one int32 host buffer, one copy, the device tensors as slices of it;
* ``packed_views_cloned``: as ``packed_int32`` with every slice cloned on the device (what a custom op that refuses
  views would need, card rule 1).

Per case: host microseconds per step for the copies (issue), for the ``mark_step``, and for a ``synchronize`` that
ends the step, median of ``--repeats`` steps after ``--warm`` warm steps. Writes ``h2d_copies.json`` to ``--out``.
``--selftest`` runs on the CPU (no card) to check the script.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def shapes(rows: int, blocks: int):
    """(name, shape, dtype) of a decode step's copies (TP8, ``rows`` requests, ``blocks`` block-table columns)."""
    import torch
    return [
        ("token_ids", (rows, 1), torch.int32),
        ("positions", (rows, 1), torch.int32),
        ("slot_mapping", (rows, 1), torch.int64),
        ("block_table", (rows, blocks), torch.int32),
        ("query_start_loc", (rows + 1, ), torch.int32),
        ("ngram_context", (rows, 2), torch.int32),
        ("state_load", (5, rows), torch.int32),
        ("state_store", (5, rows), torch.int32),
        ("logits_indices", (rows, ), torch.int32),
        ("seq_lens", (rows, ), torch.int32),
        ("context_lens", (rows, ), torch.int32),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("FARM_OUT", "."))
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=1024)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    import torch
    if args.selftest:
        device = "cpu"

        def mark_step():
            return None

        def synchronize():
            return None
    else:
        import habana_frameworks.torch as htorch  # noqa: F401
        import habana_frameworks.torch.core as htcore
        device = "hpu"
        mark_step = htcore.mark_step
        synchronize = torch.hpu.synchronize
    spec = shapes(args.rows, args.blocks)
    gen = torch.Generator().manual_seed(1)
    host = {name: torch.randint(0, 1 << 20, shape, generator=gen, dtype=torch.int64).to(dtype)
            for name, shape, dtype in spec}
    sizes = [(name, int(torch.tensor(shape).prod())) for name, shape, _ in spec]
    total = sum(n for _, n in sizes)
    packed_host = torch.empty(total, dtype=torch.int32)

    def separate():
        out = {name: tensor.to(device, non_blocking=True) for name, tensor in host.items()}
        return out

    def packed(clone: bool):
        offset = 0
        for name, n in sizes:  # the host builds one buffer (int64 values narrowed: ids and slots fit int32)
            packed_host[offset:offset + n] = host[name].reshape(-1).to(torch.int32)
            offset += n
        device_buffer = packed_host.to(device, non_blocking=True)
        out, offset = {}, 0
        for (name, shape, dtype), (_, n) in zip(spec, sizes):
            view = device_buffer.narrow(0, offset, n).view(shape)
            out[name] = view.clone() if clone else view
            offset += n
        return out

    cases = {"separate": separate, "packed_int32": lambda: packed(False), "packed_views_cloned": lambda: packed(True)}
    results = {"rows": args.rows, "blocks": args.blocks, "copies_separate": len(spec), "device": device, "cases": {}}
    for name, body in cases.items():
        issue, mark, sync = [], [], []
        for index in range(args.warm + args.repeats):
            t0 = time.perf_counter()
            out = body()
            t1 = time.perf_counter()
            # one consumer op so the copies belong to a graph that runs
            acc = sum(t.sum().to(torch.float32) for t in out.values())
            mark_step()
            t2 = time.perf_counter()
            synchronize()
            _ = float(acc)
            t3 = time.perf_counter()
            if index >= args.warm:
                issue.append(1e6 * (t1 - t0))
                mark.append(1e6 * (t2 - t1))
                sync.append(1e6 * (t3 - t2))
        results["cases"][name] = {"issue_us": statistics.median(issue), "mark_us": statistics.median(mark),
                                  "sync_us": statistics.median(sync)}
        print("%-22s issue %8.1f us  mark %8.1f us  sync+read %8.1f us" % (
            name, results["cases"][name]["issue_us"], results["cases"][name]["mark_us"],
            results["cases"][name]["sync_us"]), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "h2d_copies.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
