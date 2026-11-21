# SPDX-License-Identifier: Apache-2.0
"""The rehearsal oracle (``VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE``) through the real engine on CPU.

``tools/flash_next/rental/lookup_tiny.py`` rehearses the lookup proposers on a card with the tiny model, whose
drafts are only ever accepted when they come from the product's rehearsal oracle (recorded streams). This test
runs the same thing on CPU first: the oracle file is written from the engine's own baseline streams and named in
the environment, the n-gram server and the lookup + MTP hybrid draft from it through the product code
(``propose_ngram_draft_token_ids``, ``HybridLookup.merge``), and the streams must stay the baseline's.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit_tests.models.flash_next.test_lookup_engine import K, _compare, _plan, _run_child

SCHEDULE = [3, 1, 0, 2, 3, 3, 3]  # lookup_tiny.py's RAGGED schedule
SCENARIOS = ["solo-solo-7", "solo-solo-33", "solo-solo-100", "near-256", "batch", "turn-2"]
MIN_MATCHED = 150


@pytest.fixture(scope="module")
def oracle_run(tiny_checkpoint, tmp_path_factory) -> dict:
    plan = _plan(Path(tiny_checkpoint))
    oracle = {"schedule": SCHEDULE, "tamper_every": 4}
    ngram = {"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 8, "prompt_lookup_min": 3}
    plan["configs"] = [
        {"name": "oracle-ngram3", "speculative_config": ngram, "prefix_caching": True, "oracle": None,
         "env_oracle": oracle, "scenarios": SCENARIOS},
        {"name": "oracle-hybrid", "speculative_config": {"method": "mtp", "num_speculative_tokens": K},
         "prefix_caching": True, "oracle": None, "env": {"VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP": "1"},
         "env_oracle": oracle, "scenarios": SCENARIOS},
    ]
    record = _run_child(plan, tmp_path_factory.mktemp("lookup_oracle_engine"))
    for name, config in record["configs"].items():
        assert "error" not in config, f"configuration {name} failed:\n{config['error']}"
    return record


def _matched(record: dict, config: str) -> int:
    import tests.unit_tests.models.flash_next.test_lookup_engine as engine_test

    floor = engine_test.MIN_MATCHED
    engine_test.MIN_MATCHED = MIN_MATCHED  # fewer scenarios than the full plan
    try:
        return _compare(record, config)
    finally:
        engine_test.MIN_MATCHED = floor


def test_ngram_server_drafts_from_the_oracle_file(oracle_run):
    record = oracle_run["configs"]["oracle-ngram3"]
    assert record["drafter"] == "NgramProposer" and record["prefix_caching"]
    compared = _matched(oracle_run, "oracle-ngram3")
    accepted = {len(s["sampled"]) - 1 for s in record["steps"] if s["sampled"]}
    assert set(range(K + 1)) <= accepted, f"not every accepted count 0..{K} ran: {sorted(accepted)}"
    assert record["oracle"]["drafts"] > 20 and record["oracle"]["tampered"] > 0
    assert any(not s["draft"] for s in record["steps"]), "no empty draft ran (schedule length 0)"
    widened = [w for w in record["widths"] if w["width"] > max(w["scheduled"])]
    assert widened, "no step was widened to the previous accepted count"
    print(f"oracle-ngram3: {compared} tokens compared, oracle drafts {record['oracle']['drafts']} "
          f"({record['oracle']['tampered']} tampered), accepted counts {sorted(accepted)}, {len(widened)} widened "
          f"steps, {record['boundary_commits']} block-completion commits")


def test_hybrid_drafts_from_the_oracle_file(oracle_run):
    record = oracle_run["configs"]["oracle-hybrid"]
    assert record["drafter"] == "HpuFlashNextMtpProposer"
    compared = _matched(oracle_run, "oracle-hybrid")
    sources = [s for row in record["hybrid_rows"] for s in row["sources"] if s]
    assert {"lookup", "mtp"} <= set(sources), f"the hybrid never mixed the two sources: {set(sources)}"
    assert all(len(d) == K for row in record["hybrid_rows"] for d in row["drafts"]), "a hybrid row is not k long"
    drafted, accepted, verified = record["hybrid_counts"]["lookup"]
    assert accepted > verified, f"the oracle's lookup drafts were hardly accepted: {accepted} over {verified} rows"
    assert record["oracle"]["tampered"] > 0
    print(f"oracle-hybrid: {compared} tokens compared, lookup rows {drafted} accepted {accepted} over {verified}, "
          f"MTP rows {record['hybrid_counts']['mtp']}, oracle drafts {record['oracle']['drafts']} "
          f"({record['oracle']['tampered']} tampered)")
