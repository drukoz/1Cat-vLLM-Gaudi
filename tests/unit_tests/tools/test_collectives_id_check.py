# SPDX-License-Identifier: Apache-2.0
"""CPU checks of tools/flash_next/collectives/id_check.py: its data, its verdicts and its self-test."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/flash_next/collectives/id_check.py"
SPEC = importlib.util.spec_from_file_location("collectives_id_check", SCRIPT)
id_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(id_check)


def test_logits_have_known_largest_values_in_every_shard_and_padding_zeros():
    world, width, rows = 4, 2048, 7
    count = width - id_check.PADDING
    full = id_check.make_logits(torch, rows, world, width, count, seed=3, tied=False).float()
    shards = [full[:, r * width:(r + 1) * width] for r in range(world)]
    known = id_check.KNOWN
    for shard in shards:
        assert torch.equal(shard[:, count:], torch.zeros(rows, width - count))
        top = torch.topk(shard[:, :count], known + 1, dim=-1).values
        assert (top[:, :known] >= min(id_check.LOCAL)).all()  # the planted values lead every shard
        assert (top[:, :-1] > top[:, 1:]).all()  # distinct, and above the next: every top-k up to 64 has one order
    whole = torch.cat([s[:, :count] for s in shards], dim=-1)
    assert torch.equal(torch.topk(whole, known, dim=-1).values, torch.tensor(id_check.PLANTED).expand(rows, known))


def test_tied_logits_have_two_equal_largest_values():
    world, width, rows = 2, 2048, 9
    count = width - id_check.PADDING
    full = id_check.make_logits(torch, rows, world, width, count, seed=5, tied=True).float()
    whole = torch.cat([full[:, r * width:r * width + count] for r in range(world)], dim=-1)
    assert ((whole == id_check.PLANTED[0]).sum(dim=-1) == 2).all()
    for r in range(world):  # a shard without a global value has two equal local ones
        shard = full[:, r * width:r * width + count]
        assert ((shard == id_check.LOCAL[0]).sum(dim=-1) == 2).all()


def test_candidates_are_distinct_values_with_distinct_ids():
    for tied in (False, True):
        packed = id_check.make_candidates(torch, 5, 20, seed=1, tied=tied)
        assert packed.shape == (5, 2, 160)
        values, ids = packed[:, 0], packed[:, 1]
        for row in range(5):
            assert len(set(ids[row].tolist())) == 160
            assert len(set(values[row].tolist())) == (159 if tied else 160)
            assert int((values[row] == values[row].max()).sum()) == (2 if tied else 1)


def test_summary_names_the_rule_9_signature_and_fails_only_on_serving_cases(capsys):
    base = {"regime": "lazy", "tied": False, "rank": 0}
    records = [
        dict(base, case="vp_argmax", rows=1, ok=True),
        dict(base, case="vp_argmax", rows=32, ok=True),
        dict(base, case="topk_to_int32", rows=1, ok=True),
        dict(base, case="topk_to_int32", rows=32, ok=False, every_second_id=True, got_head=[3, 9], want_head=[3, 5]),
        dict(base, case="argmax_int", rows=2, skipped="time budget"),
    ]
    lines, serving_ok = id_check.summarize(records)
    assert serving_ok
    by_case = {line["case"]: line for line in lines}
    assert by_case["topk_to_int32"]["wrong_rows"] == [32] and by_case["topk_to_int32"]["equal_rows"] == [1]
    assert any("rule 9" in note for note in by_case["topk_to_int32"]["notes"])
    assert by_case["argmax_int"]["skipped_rows"] == [2]
    assert "WRONG at rows [32]" in capsys.readouterr().out
    _, serving_ok = id_check.summarize(records + [dict(base, case="vp_argmax", rows=128, ok=False)])
    assert not serving_ok


def test_self_test_passes_and_its_detector_sees_an_emulated_rule_9_result(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), VLLM_PLUGINS="")
    out = tmp_path / "idcheck.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--selftest", "--rows", "1,2,5", "--out", str(out),
                           "--logdir", str(tmp_path / "logs")], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "serving cases (vp_argmax, fp32_ids_row) on 1 rank(s): PASS" in proc.stdout
    summary = {(line["case"], line["tied"]): line for line in json.loads(out.read_text())["summary"]}
    assert summary[("vp_argmax", True)]["equal_rows"] == [1, 2, 5]  # the lowest id of a tie, as on the CPU
    assert summary[("fp32_ids_row", False)]["wrong_rows"] == []
    for case in ("vp_topk", "topk_gather_fp32", "stage2_topk_gather", "first_maximum", "topk_int32_embed"):
        for k in (20, 64):  # the top-k cases run at a sampled request's top-k and at the candidates per rank
            for tied in (False, True):
                assert summary[("%s@%d" % (case, k), tied)]["equal_rows"] == [1, 2, 5], (case, k, tied)
    emulated = summary[("selftest_rule9", False)]
    assert emulated["equal_rows"] == [1] and emulated["wrong_rows"] == [2, 5]
    assert any("rule 9" in note for note in emulated["notes"])
