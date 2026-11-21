# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/mtp_k_choice.py (choose k from measured step cost and acceptance, I43)."""

from __future__ import annotations

import json

import pytest

from tools.flash_next import mtp_k_choice as kc


def test_the_closed_form_is_the_target_models():
    # target-model.md section 6: E = (1 - a^(k+1)) / (1 - a)
    assert kc.closed_form_acceptance_length(0.7, 3) == pytest.approx(1 + 0.7 + 0.49 + 0.343)
    assert kc.closed_form_acceptance_length(0.0, 5) == 1.0
    assert kc.closed_form_acceptance_length(1.0, 4) == 5.0
    with pytest.raises(ValueError):
        kc.closed_form_acceptance_length(1.5, 2)
    # its table row "Ft, Fd = 5, 1; a = 0.7": the best k is 3 at 8.5 ms (bytes term 6.5 ms at T = 4 there)


def test_position_rates_are_extended_by_the_last_conditional_rate():
    rates, measured = kc.extend_rates([0.8, 0.6, 0.45], 5)
    assert measured == 3
    assert rates[:3] == [0.8, 0.6, 0.45]
    assert rates[3] == pytest.approx(0.45 * 0.75) and rates[4] == pytest.approx(0.45 * 0.75**2)
    assert kc.extend_rates([0.8, 0.6, 0.45], 2) == ([0.8, 0.6], 2)
    assert kc.extend_rates([0.5], 3)[0] == pytest.approx([0.5, 0.25, 0.125])
    with pytest.raises(ValueError):
        kc.extend_rates([], 3)


def test_acceptance_length_counts_the_bonus_token():
    assert kc.acceptance_length([0.8, 0.6, 0.45], 3) == pytest.approx(2.85)
    assert kc.acceptance_length([0.0, 0.0, 0.0], 3) == 1.0


def test_the_step_cost_line():
    assert kc.fit_step_cost({3: 60.0}) == (60.0, None)
    base, per_k = kc.fit_step_cost({3: 60.0, 4: 64.0, 5: 68.0})
    assert (base, per_k) == (pytest.approx(48.0), pytest.approx(4.0))
    with pytest.raises(ValueError):
        kc.fit_step_cost({})


def test_choose_k_predicts_between_and_beyond_the_measured_k():
    result = kc.choose_k({3: 60.0, 4: 64.0}, {3: [0.8, 0.6, 0.45], 4: [0.8, 0.6, 0.45, 0.34]}, k_max=6,
                         baseline_ms=90.0)
    by_k = {row["k"]: row for row in result["rows"]}
    assert by_k[0]["ms_per_token"] == 90.0
    assert by_k[3]["step_measured"] and not by_k[5]["step_measured"]
    assert by_k[5]["step_ms"] == pytest.approx(68.0)
    assert by_k[4]["E"] == pytest.approx(1 + 0.8 + 0.6 + 0.45 + 0.34)
    assert by_k[4]["ms_per_token"] == pytest.approx(64.0 / by_k[4]["E"])
    assert result["best"]["k"] == min(by_k, key=lambda k: by_k[k]["ms_per_token"])
    text = kc.format_table(result)
    assert "best k = %d" % result["best"]["k"] in text and "pred." in text and "meas." in text


def test_one_measured_k_predicts_no_other():
    result = kc.choose_k({3: 60.0}, {3: [0.8, 0.6, 0.45]}, k_max=5)
    assert [row["k"] for row in result["rows"]] == [3]
    assert "no slope" in kc.format_table(result)


def test_the_cli_reads_spec_scaling_results(tmp_path, capsys):
    def scaling(k, gap, tpc, rates):
        return {"levels": [{"streams": 1, "ok": 1, "steady_ms_per_token_p50": 99.0, "tokens_per_chunk_p50": 1.0,
                            "ms_per_token_p50": 99.0, "spec": {"position_rates": rates}},
                           {"streams": 32, "ok": 32, "steady_ms_per_token_p50": gap / tpc, "tokens_per_chunk_p50": tpc,
                            "ms_per_token_p50": gap / tpc, "spec": {"position_rates": rates}}]}

    k3 = tmp_path / "k3.json"
    k3.write_text(json.dumps(scaling(3, 60.0, 2.85, [0.8, 0.6, 0.45])), encoding="utf-8")
    k4 = tmp_path / "k4.json"
    k4.write_text(json.dumps(scaling(4, 64.0, 3.19, [0.8, 0.6, 0.45, 0.34])), encoding="utf-8")
    assert kc.read_run(k3, 32) == (pytest.approx(60.0), [0.8, 0.6, 0.45], pytest.approx(60.0 / 2.85))
    with pytest.raises(ValueError):
        kc.read_run(k3, 8)
    assert kc.main(["--run", f"3={k3}", "--run", f"4={k4}", "--streams", "32", "--k-max", "5"]) == 0
    printed = capsys.readouterr().out
    assert "best k =" in printed and "step cost fit: 48.00 ms + k x 4.00 ms" in printed
