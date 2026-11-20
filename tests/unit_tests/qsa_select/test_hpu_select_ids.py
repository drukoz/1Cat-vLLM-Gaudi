# SPDX-License-Identifier: Apache-2.0
"""Card check of the sparse-QSA block selection at a long context (card rule 9).

kernel-fixes' job 00038 found that ``torch.topk``'s int64 ids, narrowed with ``.to(torch.int32)`` in the same
lazy graph, are read back wrong on the host (every second id). ``select_blocks`` did exactly that. This module
answers, on one card, at the real model's geometry (``select_case``):

1. Does an **in-graph consumer** of the narrowed ids see the wrong ids too, or only the host read
   (``test_the_cast_and_the_candidates``, variant ``cast``)? Report only: it is the question, not a requirement.
2. Which way of narrowing is right on the card (the other variants of the same test). Report only.
3. Is the plugin's code right as configured: ``select_blocks`` itself, and one decode step of
   ``HpuQsaIndexer._select_decode`` in mode 0 (the default path), mode 1 (request-major keys) and mode 1 sharded
   by request (each TP rank's share), on plain decode rows and on the verification lattice? These assert.

The order is the order of urgency: the plugin's ``select_blocks``, then the cast and the candidates (1 and 2), then
the decode steps.

Every result is one ``QSAIDS`` line (run with ``-s``) and one entry of ``$FARM_OUT/select_ids.json``. The whole
module skips without a Gaudi card. It needs no kernel library: every op is a stock bridge op.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

# QSA_SELECT_IDS_DEVICE=cpu runs the same file on a CPU (a dry run of the check itself, before it costs card time).
DEVICE = os.environ.get("QSA_SELECT_IDS_DEVICE", "hpu")
if DEVICE == "hpu":
    pytest.importorskip("habana_frameworks.torch")

from tests.unit_tests.qsa_select import select_case as sc  # noqa: E402

OUT = Path(os.environ.get("FARM_OUT", "/tmp")) / "select_ids.json"
T0 = time.time()


def record(name: str, entry: dict) -> None:
    """One line on the log and one entry in the JSON file (rewritten each time: a later crash keeps the earlier)."""
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[name] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print("QSAIDS [%5.1fs] %s: %s" % (time.time() - T0, name, json.dumps(entry)), flush=True)


def verdict(case, got: dict, want) -> dict:
    cmp = sc.compare(case, got, want)
    return {"rows": cmp["rows"], "host_read_wrong_rows": len(cmp["selected_wrong_rows"]),
            "in_graph_consumer_wrong_rows": len(cmp["consumer_wrong_rows"]), "ok": cmp["ok"],
            "detail": cmp["detail"]}


# ------------------------------------------------------------------ the narrowing forms, one graph each
def narrow(name: str, idx: torch.Tensor, vals: torch.Tensor, n: int) -> torch.Tensor:
    if name == "cast":  # the line select_blocks had before 2026-11-20 (the pattern of card rule 9)
        return idx.to(torch.int32)
    if name == "gather_arange":  # the ids only ever index: a gather from an int32 arange
        return torch.arange(n, dtype=torch.int32, device=idx.device).index_select(0, idx.reshape(-1)).view(idx.shape)
    if name == "gather_2d":
        cols = torch.arange(n, dtype=torch.int32, device=idx.device)[None, :].expand(idx.shape[0], n)
        return cols.gather(1, idx)
    if name == "int64":  # no narrowing: the consumers take the int64 ids
        return idx
    if name == "via_fp32":
        return idx.to(torch.float32).to(torch.int32)
    if name == "where_then_cast":
        return torch.where(vals > float("-inf"), idx, torch.full_like(idx, -1)).to(torch.int32)
    if name == "clone_then_cast":
        return idx.clone().to(torch.int32)
    raise ValueError(name)


def variant_selection(name: str, scores, visible, positions):
    """``select_blocks`` + ``expand_selection`` with the narrowing form ``name`` (same graph as the top-k)."""
    from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import expand_selection
    n = int(scores.shape[1])
    cols = torch.arange(n, device=scores.device)
    masked = scores.masked_fill(cols[None, :] >= visible[:, None], float("-inf"))
    vals, idx = torch.topk(masked, sc.TOPK, dim=-1)
    picked = narrow(name, idx, vals, n).masked_fill(vals == float("-inf"), -1)
    return expand_selection(picked, positions, sc.RATIO)


VARIANTS = ("cast", "gather_arange", "via_fp32", "int64", "where_then_cast", "clone_then_cast", "gather_2d")


@pytest.fixture(scope="module")
def decode8():
    case = sc.build_case("decode8")
    return case, sc.oracle(case)


def test_oracle_and_cpu_agree(decode8):
    """The case and its oracle are sound before any card result is read: the plugin on the CPU gives the oracle."""
    case, want = decode8
    cmp = sc.compare(case, sc.decode_selection(case, "cpu", 0), want)
    record("cpu/mode0/decode8", {"ok": cmp["ok"], "rows": cmp["rows"]})
    assert cmp["ok"], cmp


def test_select_blocks_on_the_card(decode8):
    """The plugin's ``select_blocks`` (as configured) on host-made scores: host read and in-graph consumer."""
    from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import TOPK_IDS_MODE, expand_selection, select_blocks
    case, want = decode8
    scores, visible = sc.host_scores(case)
    dev = sc.on_device(case, DEVICE)
    selected = expand_selection(select_blocks(scores.to(DEVICE), visible.to(DEVICE), sc.TOPK), dev.positions, sc.RATIO)
    probe = sc.consumer(selected, dev.md.block_table, dev.slot_values)
    entry = verdict(case, {"consumer": probe.cpu(), "selected": selected.cpu()}, want)
    entry["topk_ids_mode"] = TOPK_IDS_MODE
    record("plugin/select_blocks", entry)
    assert entry["ok"], entry


def test_the_cast_and_the_candidates(decode8):
    """Each narrowing form, read on the host and through an in-graph consumer (two graphs per form). Report only."""
    case, want = decode8
    scores, visible = sc.host_scores(case)
    results = {}
    for name in VARIANTS:
        entry = {}
        try:
            dev = sc.on_device(case, DEVICE)
            selected = variant_selection(name, scores.to(DEVICE), visible.to(DEVICE), dev.positions)
            host = sc.compare(case, {"selected": selected.cpu()}, want)
            entry["host_read_wrong_rows"] = len(host["selected_wrong_rows"])
            entry["host_read_detail"] = host["detail"]
        except Exception as exc:  # noqa: BLE001 - a probe records every failure
            entry["host_read_raised"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        try:
            dev = sc.on_device(case, DEVICE)
            selected = variant_selection(name, scores.to(DEVICE), visible.to(DEVICE), dev.positions)
            probe = sc.consumer(selected, dev.md.block_table, dev.slot_values)  # no host read before this one
            graph = sc.compare(case, {"consumer": probe.cpu()}, want)
            entry["in_graph_consumer_wrong_rows"] = len(graph["consumer_wrong_rows"])
        except Exception as exc:  # noqa: BLE001
            entry["in_graph_consumer_raised"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        entry["rows"] = len(sc.real_rows(case))
        results[name] = entry
        record(f"variant/{name}", entry)
    # Job 00038's own read: the narrowed ids are the graph's output (nothing consumes them in the graph), against
    # the int64 ids of the same top-k computed again in a graph of its own.
    for name in ("cast", "gather_arange"):
        try:
            s, v = scores.to(DEVICE), visible.to(DEVICE)
            cols = torch.arange(sc.N_KEYS, device=s.device)
            vals, idx = torch.topk(s.masked_fill(cols[None, :] >= v[:, None], float("-inf")), sc.TOPK, dim=-1)
            direct = narrow(name, idx, vals, sc.N_KEYS).cpu()
            wide_vals, wide = torch.topk(s.masked_fill(cols[None, :] >= v[:, None], float("-inf")), sc.TOPK, dim=-1)
            wide, real = wide.cpu(), wide_vals.cpu() > float("-inf")  # the order among masked columns is not compared
            differ = (direct.to(torch.int64) != wide.to(torch.int64)) & real
            record(f"variant/{name}_direct_read", {
                "elements": int(real.sum()), "differ": int(differ.sum()), "rows_wrong": int(differ.any(1).sum()),
                "rows": int(wide.shape[0]), "in_range": bool(((direct >= 0) & (direct < sc.N_KEYS)).all()),
                "got": direct[0, :6].tolist(), "want": wide[0, :6].tolist()})
        except Exception as exc:  # noqa: BLE001
            record(f"variant/{name}_direct_read", {"raised": f"{type(exc).__name__}: {str(exc)[:300]}"})
    # The second-graph form of job 00038: the ids are read first (as int64), then narrowed in a new graph.
    try:
        dev = sc.on_device(case, DEVICE)
        s, v = scores.to(DEVICE), visible.to(DEVICE)
        masked = s.masked_fill(torch.arange(sc.N_KEYS, device=s.device)[None, :] >= v[:, None], float("-inf"))
        vals, idx = torch.topk(masked, sc.TOPK, dim=-1)
        wide = idx.cpu()
        later = idx.to(torch.int32).cpu()
        record("variant/cast_in_a_second_graph", {"elements": int(wide.numel()),
                                                  "differ": int((later.to(torch.int64) != wide).sum())})
    except Exception as exc:  # noqa: BLE001
        record("variant/cast_in_a_second_graph", {"raised": f"{type(exc).__name__}: {str(exc)[:300]}"})
    cast = results["cast"]
    record("rule9/answer", {
        "cast_host_read_wrong_rows": cast.get("host_read_wrong_rows"),
        "cast_in_graph_consumer_wrong_rows": cast.get("in_graph_consumer_wrong_rows"),
        "rows": cast["rows"],
        "forms_right_both_ways": [n for n, e in results.items()
                                  if e.get("host_read_wrong_rows") == 0 and e.get("in_graph_consumer_wrong_rows") == 0],
        "forms_right_in_graph": [n for n, e in results.items() if e.get("in_graph_consumer_wrong_rows") == 0],
    })


@pytest.mark.parametrize("name,mode", [("decode8", 0), ("decode8", 1), ("lattice4", 0), ("lattice4", 1)])
def test_decode_selection_on_the_card(name, mode):
    """One decode step of ``HpuQsaIndexer._select_decode`` on the card: mode 0 (default path) and mode 1."""
    case = sc.build_case(name)
    entry = verdict(case, sc.decode_selection(case, DEVICE, mode), None)
    record(f"plugin/mode{mode}/{name}", entry)
    assert entry["ok"], entry


@pytest.mark.parametrize("name,world", [("decode8", 4), ("lattice4", 4), ("decode8", 8)])
def test_sharded_decode_selection_on_the_card(name, world):
    """Mode 1 sharded by request: every TP rank's share, computed on this card as that rank."""
    case = sc.build_case(name)
    want = sc.oracle(case)
    total = {"rows": 0, "host_read_wrong_rows": 0, "in_graph_consumer_wrong_rows": 0, "detail": None}
    for rank in range(world):
        one = verdict(case, sc.decode_selection(case, DEVICE, 1, 1, (rank, world)), want)
        for key in ("rows", "host_read_wrong_rows", "in_graph_consumer_wrong_rows"):
            total[key] += one[key]
        total["detail"] = total["detail"] or one["detail"]
    total["ok"] = total["host_read_wrong_rows"] == 0 and total["in_graph_consumer_wrong_rows"] == 0
    record(f"plugin/mode1-shard/{name}/world{world}", total)
    assert total["rows"] == len(sc.real_rows(case))
    assert total["ok"], total
