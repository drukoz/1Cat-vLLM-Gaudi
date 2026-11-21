#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""probe_index_narrowing.py: are the ids an op returns on a card still right after `.to(torch.int32)` issued in the
lazy graph that made them?

Card job 00038 (2026-11-20, probe_moe_route.py): `torch.topk(...).indices.to(torch.int32)`, issued in one lazy graph
and read on the host, came back as every second id of the true table (an 8-byte read of 4-byte ids, the second half
out of range). The same ids read as int64, or narrowed in a second graph, were right. The served model has that
pattern in places a six-token prompt and one stream hide:

  * `vllm_gaudi/models/qwen4_exp/qsa_sparse_ops.py` `select_blocks`: top-k of the block scores, `.to(torch.int32)`,
    then `expand_selection` in the same graph (the sparse-QSA indexer of every decode step and prefill tile);
  * `vllm_gaudi/v1/spec_decode/hpu_flash_next_mtp.py`: `logits.argmax(dim=-1)`, then `.int()` as the next draft
    step's input ids;
  * `vllm_gaudi/v1/worker/hpu_model_runner.py` `select_compute_logits_and_greedy` (VLLM_HPU_FUSED_GREEDY_LOGITS):
    `logits.to(torch.float32).argmax(dim=-1).to(torch.int32)`.

One card, about half a minute, no kernel library and no model:

    python probe_index_narrowing.py [--out DIR]
    PROBE_DEVICE=cpu python probe_index_narrowing.py     # a dry run of this script (every case RIGHT on a CPU)

Every case is one function run twice on the same host-made inputs: on a CPU (the reference) and on the card, where
the inputs are on the device before the case starts, so the case's ops are one lazy graph, run by the read of its
result. Parts:

  served   `select_blocks` and `select_blocks` + `expand_selection` of the plugin's own file, at 2048 blocks (more
           than block_topk = 512, so the top-k selects) for 1, 4 and 32 rows, and at 64 blocks (a short prompt);
  pattern  top-k, argmax, max, sort and searchsorted ids: read as int64 (the control), narrowed in the graph, and
           the narrowed ids as the next op's input (index_select, embedding);
  fix      what could replace the narrowing without a second graph, on plain scores and on 00038's construct;
  boundary where the narrowing breaks. Job 00038 had it wrong at [16, 512] softmax probabilities with k = 10 and at
           [16, 16] with k = 2; the sparse-QSA selection at [8, 32768] plain fp32 scores with k = 512 came out
           right on a card (qsa-fused, jobs 00062 / 00063). So it depends on the shape or on what makes top-k's
           input: top-k of host-made fp32 scores, of an in-graph softmax of fp32 logits and of an in-graph softmax
           (dtype=float32) of bf16 logits, each narrowed in the graph, over rows 1, 2, 16, 32 x columns 16, 512,
           2048, 32768 x k 2, 10, 64, 512; and at four shapes the other constructs (the whole routing of 00038
           with and without its in-place division of the values, a multiply as the producer, bf16 scores, the
           QSA selection). Rows that tie at the cut on the CPU are left out of a case's verdict.

Lines `PROBE narrow <part> <case>: RIGHT | WRONG (...) | ERROR (...)`, a BOUNDARY matrix (one letter per row
count: R right, W wrong, E error), then four VERDICT lines.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

DEVICE = os.environ.get("PROBE_DEVICE", "hpu")
HERE = Path(__file__).resolve()
SPARSE_OPS = HERE.parents[3] / "vllm_gaudi" / "models" / "qwen4_exp" / "qsa_sparse_ops.py"
BLOCK_TOPK, RATIO = 512, 4  # the real model: indexer_budget 2048 tokens, 4 tokens per block
T0 = time.monotonic()


def say(text):
    print(f"PROBE narrow [{time.monotonic() - T0:5.1f}s] {text}", flush=True)


def backend():
    import torch

    if DEVICE == "cpu":

        def flush():
            return None

        return torch, flush
    import habana_frameworks.torch.core as htcore

    def flush():
        htcore.mark_step()
        torch.hpu.synchronize()

    return torch, flush


def sparse_ops():
    """The plugin's qsa_sparse_ops.py by path: it imports only math and torch, and nothing of vLLM is loaded.

    Looked up in the tree of this file, then in the working folder (a farm job runs in its tree)."""
    relative = Path("vllm_gaudi") / "models" / "qwen4_exp" / "qsa_sparse_ops.py"
    path = next((c for c in (SPARSE_OPS, Path.cwd() / relative) if c.is_file()), SPARSE_OPS)
    spec = importlib.util.spec_from_file_location("probe_qsa_sparse_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(torch, got, want, sets, clear=None):
    """('RIGHT' | 'WRONG', detail) for a result of a case against the CPU's.

    ``clear``: a bool mask of the rows that count (the others tie at the cut: either set is right)."""
    if tuple(got.shape) != tuple(want.shape):
        return "WRONG", f"shape {tuple(got.shape)}, the CPU's {tuple(want.shape)}"
    if clear is not None:
        if not bool(clear.any()):
            return "RIGHT", "every row ties at the cut: nothing to compare"
        tied = int((~clear).sum())
        state, detail = describe(torch, got[clear], want[clear], sets)
        return state, detail + (f"; {tied} rows that tie at the cut left out" if tied else "")
    if got.is_floating_point() or want.is_floating_point():
        same = torch.isclose(got.float(), want.float(), rtol=1e-2, atol=5e-2)
        rows = same.reshape(same.shape[0], -1).all(dim=1)
        if bool(rows.all()):
            return "RIGHT", f"{rows.numel()} of {rows.numel()} rows"
        return "WRONG", f"{int(rows.sum())} of {rows.numel()} rows are the CPU's values"
    g, w = got.long(), want.long()
    g2, w2 = g.reshape(g.shape[0], -1), w.reshape(w.shape[0], -1)
    same = (torch.sort(g2, dim=1).values == torch.sort(w2, dim=1).values) if sets else (g2 == w2)
    rows = same.all(dim=1)
    if bool(rows.all()):
        return "RIGHT", f"{rows.numel()} of {rows.numel()} rows" + (" (as sets)" if sets else "")
    flat_g, flat_w = g.flatten(), w.flatten()
    half = flat_w.numel() // 2
    second = bool(half > 1 and (flat_g[:half] == flat_w[::2][:half]).all())
    low, high = int(flat_w.min()), int(flat_w.max())
    in_range = bool(((flat_g >= low) & (flat_g <= high)).all())
    detail = (f"{int(rows.sum())} of {rows.numel()} rows are the CPU's" + (" (as sets)" if sets else "") +
              f"; the first half is every second id of the CPU's table: {second}; all ids inside the CPU's range "
              f"[{low}, {high}]: {in_range}; first got {flat_g[:6].tolist()}, first want {flat_w[:6].tolist()}")
    return "WRONG", detail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("FARM_OUT", "/tmp/probe_index_narrowing")))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    torch, flush = backend()
    import torch.nn.functional as F

    ops = sparse_ops()
    say(f"device {DEVICE}; torch {torch.__version__}; PT_HPU_LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')} "
        f"PT_HPU_LAZY_ACC_PAR_MODE={os.environ.get('PT_HPU_LAZY_ACC_PAR_MODE')}")
    results = {}

    def case(part, name, fn, inputs, sets=False, clear=None, quiet=False):
        """Run fn(*inputs) on the CPU and on DEVICE (one lazy graph) and compare; returns the state."""
        key = f"{part} {name}"
        try:
            want = fn(*inputs)
        except Exception as error:  # noqa: BLE001 - a probe reports every failure
            results[key] = {"state": "ERROR", "detail": f"on the CPU: {error!r}"}
            say(f"{key}: ERROR on the CPU: {error!r}")
            return "ERROR"
        try:
            device_inputs = [t.to(DEVICE) for t in inputs]
            flush()
            got = fn(*device_inputs).cpu()
            state, detail = describe(torch, got, want, sets, clear)
        except Exception as error:  # noqa: BLE001
            state, detail = "ERROR", " ".join(repr(error).split())[:300]
            if os.environ.get("PROBE_TRACEBACK"):
                traceback.print_exc()
        try:
            flush()
        except Exception as error:  # noqa: BLE001
            detail += f"; the flush after it raised {error!r}"[:200]
        results[key] = {"state": state, "detail": detail}
        if not quiet or state != "RIGHT":
            say(f"{key}: {state} ({detail})")
        return state

    g = torch.Generator().manual_seed(20261120)

    # ---- served: the plugin's own selection, as the decode indexer issues it ----
    def blocks_fn(scores, positions):
        return ops.select_blocks(scores, ops.visible_blocks(positions, RATIO), BLOCK_TOPK)

    def tokens_fn(scores, positions):
        return ops.expand_selection(blocks_fn(scores, positions), positions, RATIO)

    for rows, blocks in ((1, 2048), (4, 2048), (32, 2048), (4, 64)):
        scores = torch.rand(rows, blocks, generator=g) * 8.0  # distinct fp32 scores: no ties at the cut
        # each row sees a different number of complete blocks; the last ones see all of them
        positions = torch.tensor([blocks * RATIO - 1 - 37 * ((rows - 1 - r) % 5) for r in range(rows)],
                                 dtype=torch.int64)
        case("served", f"select_blocks rows {rows} blocks {blocks}", blocks_fn, (scores, positions), sets=True)
        case("served", f"select_blocks+expand_selection rows {rows} blocks {blocks}", tokens_fn, (scores, positions),
             sets=True)

    # ---- pattern: which ops' ids move when they are narrowed in the graph that makes them ----
    scores = torch.rand(4, 2048, generator=g) * 8.0
    logits = (torch.randn(32, 4096, generator=g) * 4.0).to(torch.bfloat16)
    logits[torch.arange(32), torch.randperm(4096, generator=g)[:32]] = 64.0  # one clear winner per row
    table = torch.randn(4096, 16, generator=g)
    edges = torch.arange(0, 4096, 128, dtype=torch.int64)
    marks = torch.randint(0, 4096, (64, ), generator=g, dtype=torch.int64)

    def topk_ids(s):
        return torch.topk(s, BLOCK_TOPK, dim=-1).indices

    case("pattern", "topk ids read as int64 (control)", topk_ids, (scores, ), sets=True)
    case("pattern", "topk ids .to(int32)", lambda s: topk_ids(s).to(torch.int32), (scores, ), sets=True)
    case("pattern", "topk ids .to(int32) then index_select (what the next op reads)",
         lambda s, t: t.index_select(0, topk_ids(s).to(torch.int32).clamp(0, 4095).reshape(-1)[:256]),
         (scores, table))
    case("pattern", "argmax read as int64 (control)", lambda x: x.argmax(dim=-1), (logits, ))
    case("pattern", "argmax .to(int32)", lambda x: x.argmax(dim=-1).to(torch.int32), (logits, ))
    case("pattern", "argmax .int() then embedding (the MTP drafter's next input)",
         lambda x, t: F.embedding(x.argmax(dim=-1).int().clamp(0, 4095).view(-1, 1), t), (logits, table))
    case("pattern", "fp32 argmax .to(int32).unsqueeze(-1) (the runner's in-graph greedy)",
         lambda x: x.to(torch.float32).argmax(dim=-1).to(torch.int32).unsqueeze(-1), (logits, ))
    case("pattern", "one-row argmax .to(int32)", lambda x: x[:1].contiguous().argmax(dim=-1).to(torch.int32),
         (logits, ))
    case("pattern", "max(dim).indices .to(int32)", lambda x: x.float().max(dim=-1).indices.to(torch.int32), (logits, ))
    case("pattern", "sort ids .to(int32)",
         lambda s: torch.sort(s, dim=-1, descending=True).indices.to(torch.int32), (scores, ))
    case("pattern", "searchsorted read as int64 (control)", lambda e, m: torch.searchsorted(e, m, right=True),
         (edges, marks))
    case("pattern", "searchsorted .to(int32)", lambda e, m: torch.searchsorted(e, m, right=True).to(torch.int32),
         (edges, marks))

    # ---- fix: replacements for the narrowing, each one graph ----
    def masked_first(s):
        vals, idx = torch.topk(s, BLOCK_TOPK, dim=-1)
        return idx.masked_fill(vals == float("-inf"), -1).to(torch.int32)

    def two_graphs(s):
        idx = topk_ids(s)
        flush()
        return idx.to(torch.int32)

    fixes = {
        "topk: int64 masked_fill, then .to(int32)": (masked_first, (scores, ), True),
        "topk: ids * 1, then .to(int32)": (lambda s: (topk_ids(s) * 1).to(torch.int32), (scores, ), True),
        "topk: ids + 0, then .to(int32)": (lambda s: (topk_ids(s) + 0).to(torch.int32), (scores, ), True),
        "topk: ids.clone(), then .to(int32)": (lambda s: topk_ids(s).clone().to(torch.int32), (scores, ), True),
        "topk: through float32": (lambda s: topk_ids(s).to(torch.float32).to(torch.int32), (scores, ), True),
        "topk: flattened, .to(int32), reshaped":
        (lambda s: topk_ids(s).reshape(-1).to(torch.int32).reshape(s.shape[0], BLOCK_TOPK), (scores, ), True),
        "topk: kept int64, masked_fill and cat in int64 (no narrowing)":
        (lambda s: torch.cat((topk_ids(s).masked_fill(s[:, :BLOCK_TOPK] < 0, -1), topk_ids(s)[:, :3]), dim=1),
         (scores, ), False),
        "topk: mark_step between the top-k and the narrowing (two graphs; 00038's control)":
        (two_graphs, (scores, ), True),
        "argmax: keepdim, then .to(int32)": (lambda x: x.argmax(dim=-1, keepdim=True).to(torch.int32), (logits, ),
                                              False),
        "argmax: ids * 1, then .to(int32)": (lambda x: (x.argmax(dim=-1) * 1).to(torch.int32), (logits, ), False),
        "argmax: through float32": (lambda x: x.argmax(dim=-1).to(torch.float32).to(torch.int32), (logits, ), False),
        "argmax: kept int64 into embedding (no narrowing)":
        (lambda x, t: F.embedding(x.argmax(dim=-1).clamp(0, 4095).view(-1, 1), t), (logits, table), False),
    }
    for name, (fn, inputs, sets) in fixes.items():
        case("fix", name, fn, inputs, sets=sets)

    # ---- 00038's construct: the routing of the MoE torch path, its ids narrowed in the graph ----
    def clear_rows(probabilities, k):
        """The rows whose k-th and (k+1)-th largest values differ (all rows when k is every column)."""
        if k >= probabilities.shape[1]:
            return torch.ones(probabilities.shape[0], dtype=torch.bool)
        ordered = torch.sort(probabilities.float(), dim=1, descending=True).values
        return ordered[:, k - 1] != ordered[:, k]

    def softmax32(logits):
        return F.softmax(logits, dim=1, dtype=torch.float32)

    def route(logits, k, divide=True):
        """torch_route of probe_moe_route.py (HPUUnquantizedFusedMoEMethod.apply_monolithic), then the narrowing."""
        weights, ids = torch.topk(softmax32(logits), k, dim=-1)
        if divide:
            weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.to(logits.dtype)
        return ids.to(torch.int64).to(torch.int32)

    def router_logits(rows, columns, dtype):
        return (torch.randn(rows, columns, generator=g) * 3.0).to(dtype)

    bf16 = torch.bfloat16
    logits_38 = router_logits(16, 512, bf16)
    clear_38 = clear_rows(softmax32(logits_38), 10)
    fixes_38 = {
        "00038's routing, ids read as int64 (control)":
        lambda x: torch.topk(softmax32(x), 10, dim=-1).indices,
        "00038's routing, ids .to(int32) (what failed)": lambda x: route(x, 10),
        "00038's routing: ids * 1, then .to(int32)":
        lambda x: (torch.topk(softmax32(x), 10, dim=-1).indices * 1).to(torch.int32),
        "00038's routing: ids.clone(), then .to(int32)":
        lambda x: torch.topk(softmax32(x), 10, dim=-1).indices.clone().to(torch.int32),
        "00038's routing: through float32":
        lambda x: torch.topk(softmax32(x), 10, dim=-1).indices.to(torch.float32).to(torch.int32),
        "00038's routing: ids .to(int32) as the next op's input (embedding)":
        lambda x, t: F.embedding(route(x, 10).clamp(0, 4095), t),
    }
    for name, fn in fixes_38.items():
        inputs = (logits_38, table) if "embedding" in name else (logits_38, )
        case("fix", name, fn, inputs, sets="embedding" not in name, clear=clear_38)

    # ---- boundary: where the narrowing breaks ----
    grid_rows, grid_columns, grid_k = (1, 2, 16, 32), (16, 512, 2048, 32768), (2, 10, 64, 512)
    producers = {
        "host fp32 scores": (lambda x, k: torch.topk(x, k, dim=-1).indices.to(torch.int32), torch.float32, False),
        "softmax of fp32 logits": (lambda x, k: torch.topk(F.softmax(x, dim=1), k, dim=-1).indices.to(torch.int32),
                                   torch.float32, True),
        "softmax(dtype=float32) of bf16 logits":
        (lambda x, k: torch.topk(softmax32(x), k, dim=-1).indices.to(torch.int32), bf16, True),
    }
    matrix = {}
    for producer, (fn, dtype, is_softmax) in producers.items():
        for columns in grid_columns:
            for k in grid_k:
                if k > columns:
                    continue
                letters = ""
                for rows in grid_rows:
                    x = (router_logits(rows, columns, dtype)
                         if is_softmax else torch.rand(rows, columns, generator=g) * 8.0)
                    clear = clear_rows(softmax32(x) if is_softmax else x, k)
                    state = case("boundary", f"{producer}, rows {rows} columns {columns} k {k}",
                                 lambda t, fn=fn, k=k: fn(t, k), (x, ), sets=True, clear=clear, quiet=True)
                    letters += state[0]
                matrix[f"{producer} | columns {columns:5d} k {k:3d}"] = letters
    others = {
        "00038's routing with the in-place division": (lambda x, k: route(x, k), bf16),
        "00038's routing without the division": (lambda x, k: route(x, k, divide=False), bf16),
        "a multiply as the producer": (lambda x, k: torch.topk(x * 1.0, k, dim=-1).indices.to(torch.int32),
                                       torch.float32),
        "bf16 scores": (lambda x, k: torch.topk(x, k, dim=-1).indices.to(torch.int32), bf16),
        "the QSA selection (masked_fill, top-k, .to(int32), masked_fill)":
        (lambda x, k: (lambda pair: pair[1].to(torch.int32).masked_fill(pair[0] == float("-inf"), -1))(torch.topk(
            x.masked_fill(torch.arange(x.shape[1], device=x.device)[None, :] >= x.shape[1] - 3, float("-inf")), k,
            dim=-1)), torch.float32),
    }
    for name, (fn, dtype) in others.items():
        letters = ""
        for rows, columns, k in ((16, 512, 10), (16, 16, 2), (4, 2048, 512), (8, 32768, 512)):
            x = router_logits(rows, columns, dtype)
            reference = softmax32(x) if "routing" in name else x.float()
            if "QSA" in name:
                reference = reference.masked_fill(torch.arange(columns)[None, :] >= columns - 3, float("-inf"))
            state = case("boundary", f"{name}, rows {rows} columns {columns} k {k}", lambda t, fn=fn, k=k: fn(t, k),
                         (x, ), sets=True, clear=clear_rows(reference, k), quiet=True)
            letters += state[0]
        matrix[f"{name} | [16,512] k10, [16,16] k2, [4,2048] k512, [8,32768] k512"] = letters
    say(f"BOUNDARY one letter per row count {grid_rows} (R right, W wrong, E error); the last five lines: one "
        f"letter per shape")
    for key, letters in matrix.items():
        say(f"BOUNDARY {key}: {letters}")

    # ---- verdicts ----
    def states(prefix):
        return {key: value["state"] for key, value in results.items() if key.startswith(prefix)}

    served = states("served ")
    wrong = [key[len("served "):] for key, state in served.items() if state != "RIGHT"]
    line1 = ("select_blocks on the card: " +
             ("RIGHT in every case" if not wrong else f"NOT the CPU's selection in {len(wrong)} of {len(served)} cases "
              f"({'; '.join(wrong)})"))
    pattern = states("pattern ")
    line2 = "ids narrowed in the graph that makes them: " + "; ".join(f"{key[len('pattern '):]} {state}"
                                                                      for key, state in pattern.items())
    fix = states("fix ")
    right = [key[len("fix "):] for key, state in fix.items() if state == "RIGHT"]
    bad = [f"{key[len('fix '):]} ({state})" for key, state in fix.items() if state != "RIGHT"]
    line3 = (f"replacements RIGHT on this device: {'; '.join(right) or 'none'} || not right: "
             f"{'; '.join(bad) or 'none'}")
    boundary = states("boundary ")
    wrong = sorted(key[len("boundary "):] for key, state in boundary.items() if state != "RIGHT")
    line4 = (f"{len(wrong)} of {len(boundary)} boundary cases are not the CPU's" +
             (": none" if not wrong else ": " + "; ".join(wrong[:12]) + (" ..." if len(wrong) > 12 else "")) +
             " (the BOUNDARY lines above hold the whole matrix)")
    for name, line in (("select_blocks", line1), ("narrowing", line2), ("fixes", line3), ("boundary", line4)):
        print(f"VERDICT {name}: {line}", flush=True)
    (args.out / "index_narrowing.json").write_text(
        json.dumps({"device": DEVICE, "results": results, "matrix": matrix,
                    "verdicts": [line1, line2, line3, line4]}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
