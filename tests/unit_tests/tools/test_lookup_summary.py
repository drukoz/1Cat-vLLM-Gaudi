# SPDX-License-Identifier: Apache-2.0
"""tools/flash_next/lookup_summary.py: the arms, the per-source lines of a server log, and the table of two runs."""

from __future__ import annotations

import json

import pytest

from tools.flash_next import lookup_summary as ls


def test_arm_names():
    mtp = {"method": "mtp", "num_speculative_tokens": 3}
    assert ls.arm_name(None, None, None) == "no speculation"
    assert ls.arm_name({"method": "ngram", "num_speculative_tokens": 8}, None, None) == "n-gram k=8"
    assert ls.arm_name(mtp, "1", 0) == "MTP k=3" and ls.arm_name(mtp, "1", 1) == "hybrid k=3"
    assert ls.arm_name(mtp, "1", 2) == "lookup only k=3"
    assert ls.arm_name(mtp, "1", None) == "hybrid k=3"  # the server's default mode
    assert ls.arm_name(mtp, None, 1) == "MTP k=3" and ls.arm_name(mtp, "0", 2) == "MTP k=3"  # no hook installed
    # an n-gram server's request with its lookup off drafts nothing: the same-server no-speculation reference
    ngram = {"method": "ngram", "num_speculative_tokens": 5}
    assert ls.arm_name(ngram, None, 0) == "no speculation (n-gram server, lookup off)"
    assert ls.arm_name(ngram, None, 1) == "n-gram k=5" and ls.arm_name(ngram, "1", None) == "n-gram k=5"
    assert ls.is_reference("no speculation") and ls.is_reference("no speculation (n-gram server, lookup off)")
    assert not ls.is_reference("MTP k=3") and not ls.is_reference("n-gram k=5")


def _line(worker, lookup, mtp, none, steps):
    return ("(Worker_TP%d_EP%d pid=1) INFO 11-20 21:00:00 [flash_next_lookup.py:160] Flash-Next lookup hybrid: "
            "lookup rows %d, accepted %d over %d verified rows (1.00 per row); mtp rows %d, accepted %d over %d "
            "verified rows (0.50 per row); none rows %d, accepted %d over %d verified rows (0.00 per row) "
            "(after %d steps, a new set of requests)" % ((worker, worker) + lookup + mtp + none + (steps, )))


def test_boundary_counters_once_per_step_count():
    log = "\n".join([_line(0, (0, 0, 0), (40, 20, 38), (0, 0, 0), 40), _line(1, (0, 0, 0), (40, 20, 38), (0, 0, 0), 40),
                     "INFO Flash-Next lookup hybrid: lookup rows 9, accepted 9 over 9 verified rows (1.00 per row)",
                     _line(0, (30, 80, 29), (50, 25, 47), (0, 0, 0), 80)])
    lines = ls.boundary_counters(log)
    assert [entry["steps"] for entry in lines] == [40, 80]
    assert lines[1]["sources"]["lookup"] == {"rows": 30, "accepted": 80, "verified": 29}


def _round(kind, streams, mode, accept, gap, ids, sha):
    return {"kind": kind, "streams": streams, "tokens": 8, "mode": mode, "ok": streams, "errors": 0,
            "spec": None if accept is None else {"acceptance_length": accept, "position_rates": [0.9, 0.8, 0.7]},
            "ms_per_step": gap, "gap_ms_p90": gap, "steady_ms_per_token": gap / (accept or 1.0),
            "ms_per_token_p50": gap / (accept or 1.0) + 1.0, "aggregate_tok_s": 100.0, "ttft_p50_s": 0.5,
            "prompt_tokens": [800, 810, 820], "prompt_sha": sha, "token_ids": ids}


def _write(run_dir, probe, spec, lookup=None, log=""):
    run_dir.mkdir()
    (run_dir / "probe.json").write_text(json.dumps(probe))
    env = {"VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP": lookup} if lookup else {}
    (run_dir / "summary.json").write_text(json.dumps({"speculative_config": json.dumps(spec) if spec else None,
                                                       "tp": 4, "env": env, "healthy_after_s": 570.0}))
    (run_dir / "server0.log").write_text(log)


def test_table_of_a_reference_and_a_hybrid_run(tmp_path, capsys):
    answer = list(range(50, 58))
    shas = ["s0", "s1"]
    reference = {"label": "nospec", "warm": [{"kind": "edit", "streams": 2, "ok": 2}],
                 "rounds": [_round("edit", 2, None, None, 60.0, [answer, answer], shas)]}
    other = answer[:5] + [99, 98, 97]
    hybrid = {"label": "mtp3lookup", "warm": [{"kind": "edit", "streams": 2, "ok": 2},
                                              {"kind": "prose", "streams": 2, "ok": 0}],
              "rounds": [_round("edit", 2, 0, 2.0, 80.0, [answer, answer], shas),
                         _round("edit", 2, 1, 3.2, 80.0, [answer, other], shas),
                         {"kind": "prose", "streams": 2, "tokens": 8, "mode": 1, "ok": 0, "errors": 2,
                          "warm_failed": ["HTTP 500"]}]}
    # executed on the server: warm (mode 0), round 0 (mode 0), round 1 (hybrid); the prose warm round failed
    # before any decode step; the flush request gives the last line
    log = "\n".join([_line(0, (0, 0, 0), (10, 5, 9), (0, 0, 0), 10), _line(0, (0, 0, 0), (40, 20, 38), (0, 0, 0), 40),
                     _line(0, (30, 80, 29), (50, 25, 47), (0, 0, 0), 80)])
    _write(tmp_path / "a", reference, None)
    _write(tmp_path / "b", hybrid, {"method": "mtp", "num_speculative_tokens": 3}, lookup="1", log=log)
    out = tmp_path / "summary.json"
    assert ls.main([str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "missing"), "--out", str(out)]) == 0
    rows = json.loads(out.read_text())["rows"]
    by_arm = {r["arm"]: r for r in rows if r["kind"] == "edit"}
    assert set(by_arm) == {"no speculation", "MTP k=3", "hybrid k=3"}
    assert by_arm["MTP k=3"]["speedup_vs_no_speculation"] == pytest.approx(60.0 / 40.0)
    assert by_arm["hybrid k=3"]["speedup_vs_no_speculation"] == pytest.approx(60.0 / 25.0)
    assert by_arm["hybrid k=3"]["rows_per_step_mtp_shapes"] == 8
    assert by_arm["MTP k=3"]["lossless"] == {"compared": 2, "identical": 2, "first_difference_min": None,
                                             "first_difference_p50": None}
    assert by_arm["hybrid k=3"]["lossless"]["identical"] == 1
    assert by_arm["hybrid k=3"]["lossless"]["first_difference_min"] == 5
    # the per-source rows of each measured round: the line after it minus the line before it
    assert by_arm["MTP k=3"]["sources"]["mtp"] == {"rows": 30, "accepted": 15, "verified": 29}
    assert by_arm["hybrid k=3"]["sources"]["lookup"] == {"rows": 30, "accepted": 80, "verified": 29}
    assert by_arm["hybrid k=3"]["sources"]["mtp"] == {"rows": 10, "accepted": 5, "verified": 9}
    assert rows[0]["arm"] == "no speculation" and rows[-1]["failed"] and rows[-1]["kind"] == "prose"
    printed = capsys.readouterr().out
    assert "| edit | 2 | hybrid k=3 | 3.20 | 0.90 0.80 0.70 | 80.0 | 25.00 |" in printed
    assert "2.40x" in printed and "1/2 (first difference at token 5)" in printed
    assert "sources: lookup rows 30, 2.76 accepted per row; mtp rows 10, 0.56 accepted per row" in printed
    assert "FAILED: HTTP 500" in printed
    assert ls.main([str(tmp_path / "missing")]) == 1


def test_sources_are_left_out_when_the_log_does_not_line_up(tmp_path):
    probe = {"label": "x", "warm": [], "rounds": [_round("edit", 1, 1, 2.0, 50.0, [[1, 2]], ["s"]),
                                                   _round("edit", 1, 2, 2.0, 50.0, [[1, 2]], ["s"])]}
    assert ls.executed_rounds(probe) == [("round", 0), ("round", 1)]
    one = _line(0, (1, 1, 1), (0, 0, 0), (0, 0, 0), 5)
    # one line for two rounds: the last round has no counters (no request came after it)
    assert set(ls.source_deltas(probe, one)) == {0}
    three = "\n".join([one, _line(0, (2, 2, 2), (0, 0, 0), (0, 0, 0), 9), _line(0, (3, 3, 3), (0, 0, 0), (0, 0, 0), 12)])
    assert ls.source_deltas(probe, three) == {}  # more lines than rounds: not this probe's server log
    assert ls.source_deltas(probe, "") == {}


def test_the_ngram_servers_lookup_off_rounds_are_the_reference(tmp_path):
    answer, shas = list(range(50, 58)), ["s0", "s1"]
    other = answer[:3] + [99, 98, 97, 96, 95]
    # the folded n-gram arm: n-gram rounds on other prompts, then the same rounds with the lookup off (mode 0)
    ngram = {"label": "ngram5-nospec", "warm": [], "rounds": [
        _round("edit", 2, None, 4.0, 90.0, [answer, answer], ["s2", "s3"]),
        _round("edit", 2, 0, None, 60.0, [answer, answer], shas)]}
    hybrid = {"label": "hybrid3", "warm": [], "rounds": [_round("edit", 2, 1, 3.0, 75.0, [answer, other], shas)]}
    _write(tmp_path / "n", ngram, {"method": "ngram", "num_speculative_tokens": 5})
    _write(tmp_path / "h", hybrid, {"method": "mtp", "num_speculative_tokens": 3}, lookup="1")
    rows = ls.rows_of([ls.load_run(str(tmp_path / d)) for d in ("n", "h")])
    by_arm = {r["arm"]: r for r in rows}
    off = by_arm["no speculation (n-gram server, lookup off)"]
    assert rows[0] is off and off["lossless"] is None and off["speedup_vs_no_speculation"] == pytest.approx(1.0)
    assert by_arm["n-gram k=5"]["speedup_vs_no_speculation"] == pytest.approx(60.0 / (90.0 / 4.0))
    assert by_arm["n-gram k=5"]["lossless"] is None  # its prompts have no reference stream
    assert by_arm["hybrid k=3"]["speedup_vs_no_speculation"] == pytest.approx(60.0 / 25.0)
    assert by_arm["hybrid k=3"]["lossless"]["identical"] == 1
    assert by_arm["hybrid k=3"]["lossless"]["first_difference_min"] == 3
    # a server without speculation, when there is one, is the reference of the speedups
    real = {"label": "nospec", "warm": [], "rounds": [_round("edit", 2, None, None, 50.0, [answer, answer], shas)]}
    _write(tmp_path / "r", real, None)
    rows = ls.rows_of([ls.load_run(str(tmp_path / d)) for d in ("n", "h", "r")])
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["hybrid k=3"]["speedup_vs_no_speculation"] == pytest.approx(50.0 / 25.0)
    assert by_arm["no speculation (n-gram server, lookup off)"]["speedup_vs_no_speculation"] == pytest.approx(50 / 60)
