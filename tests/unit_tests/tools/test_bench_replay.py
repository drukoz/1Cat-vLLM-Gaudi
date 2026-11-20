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


def test_segment_cut_marks_every_collective_position_without_a_collective():
    body = bench.Body(torch, "cpu", layers=3, world=8)
    calls = {"seg": 0}

    def cut_seg():
        calls["seg"] += 1

    x = torch.randn(2, bench.HIDDEN).to(torch.bfloat16)
    out = body.forward(x, None, cut_seg=cut_seg)
    assert calls["seg"] == 6  # two per layer, where the all-reduces would be
    assert torch.equal(out, body.forward(x, None))  # the cut changes no value


def test_derive_splits_a_collective_into_its_launch_and_the_collective_alone():
    # 96 collective positions: the uncut body 2.0 ms, the launches alone add 96 x 0.08, the collectives 96 x 0.07.
    merged = [
        _entry("replay", 1, False, 2.0),
        _entry("replay", 1, True, 2.0 + 96 * 0.15),
        _entry("replay_seg", 1, False, 2.0 + 96 * 0.08, call=9.0),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    seg = lines[("replay_seg", 1)]
    assert seg["launch_ms_each"] == pytest.approx(0.08)
    assert seg["collective_alone_ms_each"] == pytest.approx(0.07)
    assert seg["host_call_share"] == pytest.approx(9.0 / (2.0 + 96 * 0.08))
    assert lines[("replay", 1)]["all_reduce_ms_each"] == pytest.approx(0.15)


def test_derive_uses_the_cached_uncut_body_for_cached_regimes():
    merged = [
        _entry("replay_cache", 32, False, 1.0),
        _entry("replay_cache", 32, True, 1.0 + 96 * 0.10),
        _entry("replay_seg_cache", 32, False, 1.0 + 96 * 0.03),
        _entry("replay_layer_cache", 32, False, 1.0 + 47 * 0.02),
        _entry("replay", 32, False, 2.0),
        _entry("replay", 32, True, 2.0 + 96 * 0.15),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    assert lines[("replay_seg_cache", 32)]["launch_ms_each"] == pytest.approx(0.03)
    assert lines[("replay_seg_cache", 32)]["collective_alone_ms_each"] == pytest.approx(0.07)
    assert lines[("replay_layer_cache", 32)]["cut_ms_each_without_collectives"] == pytest.approx(0.02)
    assert lines[("replay_cache", 32)]["vs"] == "replay"
    assert lines[("replay_cache", 32)]["vs_total_ms"] == pytest.approx(1.0 + 96 * 0.10 - (2.0 + 96 * 0.15))
    assert lines[("replay_cache", 32)]["vs_total_without_ms"] == pytest.approx(-1.0)


def test_parse_env_sets_and_derive_against_the_baseline():
    sets = bench.parse_env_sets("weak:HABANA_WEAK_ORDER=1; "
                                "skip:ENABLE_EXPERIMENTAL_FLAGS=true,RUNTIME_SKIP_RECIPE_VALIDATION=1;")
    assert sets == [("weak", {"HABANA_WEAK_ORDER": "1"}),
                    ("skip", {"ENABLE_EXPERIMENTAL_FLAGS": "true", "RUNTIME_SKIP_RECIPE_VALIDATION": "1"})]
    assert bench.parse_env_sets("") == []
    with pytest.raises(ValueError):
        bench.parse_env_sets("bad:NOVALUE")
    baseline = [_entry("replay", 1, False, 2.0), _entry("replay", 1, True, 16.0, call=15.0)]
    env_run = [_entry("replay@weak", 1, False, 2.1), _entry("replay@weak", 1, True, 10.1, call=9.0)]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(env_run, reference=baseline)}
    line = lines[("replay@weak", 1)]
    assert line["vs"] == "replay"
    assert line["vs_total_ms"] == pytest.approx(-5.9)
    assert line["vs_call_ms"] == pytest.approx(-6.0)
    assert line["vs_total_without_ms"] == pytest.approx(0.1)
    assert line["all_reduce_ms_each"] == pytest.approx(8.0 / 96)


def test_cpu_reference_tells_a_reduction_from_none():
    rank = bench.Rank.__new__(bench.Rank)
    rank.torch, rank.device, rank.world = torch, "cpu", 4
    rank.body = bench.Body(torch, "cpu", layers=2, world=4)
    x = torch.randn(1, bench.HIDDEN).to(torch.bfloat16)
    reduced = rank.body.forward(x, lambda t: t * 4).float()
    record = rank.cpu_reference(x, reduced)
    assert record["replay_vs_cpu_reduced_max_abs"] == 0.0
    assert record["replay_vs_cpu_unreduced_max_abs"] > 0.0


def test_parse_args_environment_sets_and_hold_options():
    args = bench.parse_args(["--rows", "1,32", "--env-sets", "a:K=1", "--env-regimes", "replay,replay_seg",
                             "--hold-file", "/tmp/h", "--hold-calls", "7", "--collectives", "off"])
    assert args.env_sets == [("a", {"K": "1"})] and args.env_rows == [1, 32]
    assert args.env_regimes == ["replay", "replay_seg"]
    assert args.hold_calls == 7 and args.hold_regime == "replay" and args.collectives == "off"


def test_split_regime_reads_the_cache_and_busy_suffixes():
    assert bench.split_regime("replay") == ("replay", "", "")
    assert bench.split_regime("replay_seg_cache") == ("replay_seg", "_cache", "")
    assert bench.split_regime("replay_layer_busy") == ("replay_layer", "", "_busy")
    assert bench.split_regime("replay_seg_cache_busy") == ("replay_seg", "_cache", "_busy")


def test_delay_adds_a_dependency_and_keeps_the_value():
    body = bench.Body(torch, "cpu", layers=1, world=1)
    body.make_busy("cpu", n=16, k=3)
    x = torch.randn(2, bench.HIDDEN).to(torch.bfloat16)
    assert torch.equal(body.delay(x), x)
    assert body.busy_w.shape == (16, 16) and body.busy_w.dtype == torch.bfloat16


def test_derive_gives_the_device_floor_behind_a_long_operation():
    # The long operation takes 30 ms; behind it 96 launches cost the device 96 x 0.03 and 96 collectives 96 x 0.05.
    merged = [
        _entry("replay_busy", 1, False, 30.0, call=0.3),
        _entry("replay_busy", 1, True, 30.0 + 96 * 0.08, call=17.0),
        _entry("replay_seg_busy", 1, False, 30.0 + 96 * 0.03, call=8.0),
        _entry("replay_layer_busy", 1, False, 30.0 + 47 * 0.03, call=4.0),
    ]
    lines = {(line["case"], line["rows"]): line for line in bench.derive(merged)}
    assert lines[("replay_busy", 1)]["all_reduce_ms_each"] == pytest.approx(0.08)
    assert lines[("replay_busy", 1)]["busy_ms"] == pytest.approx(30.0)
    assert lines[("replay_busy", 1)]["call_with_ms"] == pytest.approx(17.0)
    assert lines[("replay_seg_busy", 1)]["launch_ms_each"] == pytest.approx(0.03)
    assert lines[("replay_seg_busy", 1)]["collective_alone_ms_each"] == pytest.approx(0.05)
    assert lines[("replay_seg_busy", 1)]["call_without_ms"] == pytest.approx(8.0)
    assert lines[("replay_layer_busy", 1)]["cut_ms_each_without_collectives"] == pytest.approx(0.03)


def test_gather_sum_reduce_gives_the_all_reduce_value():
    class FakeDist:

        @staticmethod
        def all_gather_into_tensor(out, tensor, group=None):
            out.copy_(torch.cat([tensor * (k + 1) for k in range(4)], 0))  # rank k holds (k + 1) x tensor

    rank = bench.Rank.__new__(bench.Rank)
    rank.torch, rank.dist, rank.world, rank.group = torch, FakeDist, 4, object()
    x = torch.randn(3, 8)
    assert torch.allclose(rank.gather_sum_reduce(x), x * 10)
    rank.group = None  # one rank alone: the identity
    assert rank.gather_sum_reduce(x) is x


def test_work_adds_device_work_to_every_sub_block_and_keeps_the_value():
    body = bench.Body(torch, "cpu", layers=2, world=1)
    x = torch.randn(2, bench.HIDDEN).to(torch.bfloat16)
    plain = body.forward(x, None)
    body.make_work("cpu", n=8, k=2)
    assert torch.equal(body.forward(x, None), plain)  # off until a *_work regime turns it on
    body.work_on = True
    products = bench.count_ops(torch, lambda: body.forward(x, None)) - bench.count_ops(torch, lambda: plain + 0)
    assert torch.equal(body.forward(x, None), plain)
    body.work_on = False
    assert products > bench.count_ops(torch, lambda: body.forward(x, None))  # more ops with the work on
    assert bench.split_regime("replay_seg_work") == ("replay_seg", "", "_work")
    assert bench.split_regime("replay_cache_work") == ("replay", "_cache", "_work")


def test_derive_uses_the_read_of_the_output_when_it_was_measured():
    merged = [
        dict(_entry("replay", 1, False, 1.0), done_p50_max_rank_ms=2.0),
        dict(_entry("replay", 1, True, 3.0), done_p50_max_rank_ms=2.0 + 96 * 0.2),
    ]
    line = bench.derive(merged)[0]
    assert line["all_reduce_ms_each"] == pytest.approx(0.2)  # from done, not from total
    assert line["total_with_ms"] == pytest.approx(2.0 + 96 * 0.2)


def test_merge_carries_the_slowest_ranks_done_time():
    record = {"case": "replay", "rows": 1, "collectives": True, "layers": 48,
              "call_ms": {"p50": 0.2}, "total_ms": {"p50": 0.8}, "done_ms": {"p50": 13.0}}
    ahead = dict(record, done_ms={"p50": 13.5})
    merged = bench.merge({0: [record], 1: [ahead]})
    assert merged[0]["done_p50_max_rank_ms"] == 13.5 and merged[0]["total_p50_max_rank_ms"] == 0.8


def test_read_back_takes_a_tensor_or_the_first_of_a_tuple():
    rank = bench.Rank.__new__(bench.Rank)
    rank.torch = torch
    rank.read_back(torch.zeros(2, 3))  # a CPU tensor: nothing to read
    rank.read_back((torch.zeros(2), torch.zeros(1)))
    rank.read_back(None)
