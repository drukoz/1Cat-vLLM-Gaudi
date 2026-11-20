# SPDX-License-Identifier: Apache-2.0
"""CPU checks of tools/flash_next/replay/bench_replay.py: the body, the merge and the derived costs."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("bench_replay", ROOT / "tools/flash_next/replay/bench_replay.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_body_has_two_reductions_per_layer_and_cuts_where_asked():
    body = bench.Body(torch, "cpu", layers=3, world=8)
    calls = {"reduce": 0, "cut": 0, "cut_coll": 0}

    def reduce(t):
        calls["reduce"] += 1
        return t * 8  # what an all-reduce of 8 equal ranks gives

    def cut():
        calls["cut"] += 1

    def cut_coll():
        calls["cut_coll"] += 1

    x = torch.randn(4, bench.HIDDEN).to(torch.bfloat16)
    out = body.forward(x, reduce, cut, cut_coll)
    assert out.shape == x.shape and torch.isfinite(out.float()).all()
    assert calls == {"reduce": 6, "cut": 3, "cut_coll": 6}


def test_body_without_collectives_issues_no_reduction():
    body = bench.Body(torch, "cpu", layers=2, world=8)
    x = torch.randn(1, bench.HIDDEN).to(torch.bfloat16)
    assert torch.equal(body.forward(x, None), body.forward(x, None))


def _entry(case, rows, collectives, total, call=0.1, layers=48):
    return {"case": case, "rows": rows, "collectives": collectives, "layers": layers,
            "total_p50_max_rank_ms": total, "call_p50_max_rank_ms": call}


def test_derive_gives_one_all_reduce_and_one_cut():
    merged = [
        _entry("replay", 1, False, 2.0),
        _entry("replay", 1, True, 2.0 + 96 * 0.05),
        _entry("replay_layer", 1, False, 2.0 + 47 * 0.01),
        _entry("replay_layer", 1, True, 2.0 + 96 * 0.05 + 47 * 0.02),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    assert lines[("replay", 1)]["all_reduce_ms_each"] == pytest.approx(0.05)
    assert lines[("replay_layer", 1)]["cut_ms_each_with_collectives"] == pytest.approx(0.02)
    assert lines[("replay_layer", 1)]["cut_ms_each_without_collectives"] == pytest.approx(0.01)
    assert lines[("replay", 1)]["host_call_share"] == pytest.approx(0.1 / (2.0 + 96 * 0.05))


def test_derive_counts_a_cut_before_every_collective_for_coll_regimes():
    merged = [
        _entry("lazy_layer", 32, True, 10.0),
        _entry("lazy_coll", 32, True, 10.0 + 96 * 0.03),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    assert lines[("lazy_coll", 32)]["cut_ms_each_with_collectives"] == pytest.approx(0.03)


def test_merge_takes_the_slowest_rank():
    record = {"case": "replay", "rows": 1, "collectives": True, "layers": 48,
              "call_ms": {"p50": 0.2}, "total_ms": {"p50": 5.0}}
    slow = dict(record, call_ms={"p50": 0.3}, total_ms={"p50": 7.0})
    merged = bench.merge({0: [record], 1: [slow]})
    assert merged[0]["total_p50_max_rank_ms"] == 7.0
    assert merged[0]["call_p50_max_rank_ms"] == 0.3
    assert merged[0]["total_p50_median_rank_ms"] == 6.0


def test_parse_args_lists():
    args = bench.parse_args(["--rows", "1,32", "--regimes", "replay,compile_layer", "--processes", "lazy"])
    assert args.rows == [1, 32] and args.regimes == ["replay", "compile_layer"] and args.processes == ["lazy"]
