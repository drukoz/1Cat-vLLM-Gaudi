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


def test_derive_compares_raw_graph_async_and_metadata_with_the_wrapper():
    merged = [
        _entry("replay", 32, True, 6.0, call=0.5),
        _entry("graph_raw", 32, True, 5.7, call=0.3),
        _entry("replay_meta", 32, True, 6.9, call=1.1),
        _entry("replay_async", 32, True, 6.0, call=0.05),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    assert lines[("graph_raw", 32)]["vs"] == "replay"
    assert lines[("graph_raw", 32)]["vs_total_ms"] == pytest.approx(-0.3)
    assert lines[("graph_raw", 32)]["vs_call_ms"] == pytest.approx(-0.2)
    assert lines[("replay_meta", 32)]["vs_total_ms"] == pytest.approx(0.9)
    assert lines[("replay_async", 32)]["vs_call_ms"] == pytest.approx(-0.45)
    assert "vs" not in lines[("replay", 32)]


def test_derive_compares_fused_pair_and_async_tail_with_their_references():
    merged = [
        _entry("replay_par2", 1, True, 12.0, call=10.0),
        _entry("replay_fused2", 1, True, 7.0, call=5.5),
        _entry("replay_tail", 1, True, 14.0, call=14.0),
        _entry("replay_async_tail", 1, True, 13.2, call=13.2),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    assert lines[("replay_fused2", 1)]["vs"] == "replay_par2"
    assert lines[("replay_fused2", 1)]["vs_total_ms"] == pytest.approx(-5.0)
    assert lines[("replay_async_tail", 1)]["vs"] == "replay_tail"
    assert lines[("replay_async_tail", 1)]["vs_total_ms"] == pytest.approx(-0.8)


def test_derive_divides_by_the_recorded_number_of_calls():
    merged = [
        dict(_entry("replay_fused2", 1, False, 2.0), all_reduces=0),
        dict(_entry("replay_fused2", 1, True, 2.0 + 48 * 0.1), all_reduces=48),
    ]
    line = bench.derive(merged)[0]
    assert line["all_reduce_ms_each"] == pytest.approx(0.1)


def test_pair_layer_modes_give_the_same_values_on_cpu():
    body = bench.Body(torch, "cpu", layers=2, world=1)
    x = torch.randn(3, bench.HIDDEN).to(torch.bfloat16)
    calls = []

    def reduce(t):
        calls.append(t.shape[0])
        return t  # world 1: the all-reduce is the identity

    import contextlib
    par2 = body.forward_pair(x, reduce, "par2")
    n_par2 = len(calls)
    fused2 = body.forward_pair(x, reduce, "fused2")
    n_fused2 = len(calls) - n_par2
    coal = body.forward_pair(x, reduce, "coalesce", contextlib.nullcontext)
    assert torch.equal(par2, fused2) and torch.equal(par2, coal)
    assert n_par2 == 4 and n_fused2 == 2  # two calls per layer against one over the stacked pair
    assert calls[n_par2] == 6  # the stacked call carries both outputs' rows


def test_meta_inputs_look_like_trimmed_metadata():
    rank = bench.Rank.__new__(bench.Rank)
    rank.torch, rank.device = torch, "cpu"
    metas = rank.meta_inputs(4, 2)
    assert len(metas) == 2
    meta = metas[1]
    tensors = [getattr(meta, f) for f in meta._fields if f.startswith("t")]
    assert len(tensors) == bench.META_TENSORS
    assert all(t.dtype == torch.int32 for t in tensors)
    assert tensors[0].shape == (4, bench.META_COLS) and tensors[1].shape == (4, 1)
    assert meta.is_prompt is False and meta.block_size == 128


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
