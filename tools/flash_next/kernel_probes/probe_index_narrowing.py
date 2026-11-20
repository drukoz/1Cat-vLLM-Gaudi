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
  fix      what could replace the narrowing without a second graph.

Lines `PROBE narrow <part> <case>: RIGHT | WRONG (...) | ERROR (...)`, then three VERDICT lines.
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
    """The plugin's qsa_sparse_ops.py by path: it imports only math and torch, and nothing of vLLM is loaded."""
    spec = importlib.util.spec_from_file_location("probe_qsa_sparse_ops", SPARSE_OPS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(torch, got, want, sets):
    """('RIGHT' | 'WRONG', detail) for a result of a case against the CPU's."""
    if tuple(got.shape) != tuple(want.shape):
        return "WRONG", f"shape {tuple(got.shape)}, the CPU's {tuple(want.shape)}"
    if got.is_floating_point() or want.is_floating_point():
        same = torch.isclose(got.float(), want.float(), rtol=1e-2, atol=5e-2)
        rows = same.reshape(same.shape[0], -1).all(dim=1)
        if bool(rows.all()):
            return "RIGHT", f"{rows.numel()} of {rows.numel()} rows"
        return "WRONG", f"{int(rows.sum())} of {rows.numel()} rows are the CPU's values"
    g, w = got.long(), want.long()
    g2, w2 = g.reshape(g.shape[0], -1), w.reshape(w.shape[0], -1)
    if sets:
        rows = (torch.sort(g2, dim=1).values == torch.sort(w2, dim=1).values).all(dim=1)
    else:
        rows = (g2 == w2).all(dim=1)
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

    def case(part, name, fn, inputs, sets=False):
        """Run fn(*inputs) on the CPU and on DEVICE (one lazy graph) and compare."""
        key = f"{part} {name}"
        try:
            want = fn(*inputs)
        except Exception as error:  # noqa: BLE001 - a probe reports every failure
            results[key] = {"state": "ERROR", "detail": f"on the CPU: {error!r}"}
            say(f"{key}: ERROR on the CPU: {error!r}")
            return
        try:
            device_inputs = [t.to(DEVICE) for t in inputs]
            flush()
            got = fn(*device_inputs).cpu()
            state, detail = describe(torch, got, want, sets)
        except Exception as error:  # noqa: BLE001
            state, detail = "ERROR", " ".join(repr(error).split())[:300]
            if os.environ.get("PROBE_TRACEBACK"):
                traceback.print_exc()
        try:
            flush()
        except Exception as error:  # noqa: BLE001
            detail += f"; the flush after it raised {error!r}"[:200]
        results[key] = {"state": state, "detail": detail}
        say(f"{key}: {state} ({detail})")

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
    for name, line in (("select_blocks", line1), ("narrowing", line2), ("fixes", line3)):
        print(f"VERDICT {name}: {line}", flush=True)
    (args.out / "index_narrowing.json").write_text(
        json.dumps({"device": DEVICE, "results": results, "verdicts": [line1, line2, line3]}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
