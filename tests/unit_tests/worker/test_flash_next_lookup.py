# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the lookup + MTP hybrid proposer (``vllm_gaudi/v1/spec_decode/flash_next_lookup.py``)."""

import numpy as np
import pytest

from vllm_gaudi.v1.spec_decode.flash_next_lookup import (PAD_DRAFT, REQUEST_KEY, HybridLookup, SpecLookupMode,
                                                         drafts_off_rows, find_lookup_draft, lookup_mode,
                                                         merge_drafts, pad_draft, request_mode)

K = 3


def _row(tokens, width=64):
    row = np.zeros(width, dtype=np.int32)
    row[:len(tokens)] = tokens
    return row


def test_lookup_takes_the_earliest_occurrence_of_the_longest_match():
    # ... 5 6 7 8 ... 5 6 9 ... 5 6 -> vLLM's search continues from the earliest "5 6": 7 8 2.
    tokens = np.array([1, 5, 6, 7, 8, 2, 5, 6, 9, 3, 4, 5, 6], dtype=np.int32)
    assert find_lookup_draft(tokens, 2, 4, K, 4096) == [7, 8, 2]


def test_lookup_prefers_the_longest_matching_ngram():
    # The suffix "4 5 6" occurs once (followed by 7 8 1); "5 6" alone also occurs later (followed by 2).
    tokens = np.array([4, 5, 6, 7, 8, 1, 5, 6, 2, 9, 4, 5, 6], dtype=np.int32)
    assert find_lookup_draft(tokens, 2, 4, K, 4096) == [7, 8, 1]


def test_lookup_is_vllms_ngram_search():
    from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens

    rng = np.random.default_rng(0)
    for _ in range(200):
        tokens = rng.integers(0, 6, size=int(rng.integers(1, 40))).astype(np.int32)
        want = [int(t) for t in _find_longest_matched_ngram_and_propose_tokens(tokens, 2, 4, 4096, K)]
        assert find_lookup_draft(tokens, 2, 4, K, 4096) == want


def test_lookup_without_a_match_is_empty():
    assert find_lookup_draft(np.arange(20, dtype=np.int32), 2, 4, K, 4096) == []


def test_lookup_near_the_end_of_the_context_is_short():
    # "7 8" occurs right before the end: one token follows it.
    tokens = np.array([1, 2, 7, 8, 7, 8], dtype=np.int32)
    assert find_lookup_draft(tokens, 2, 4, K, 4096) == [7, 8]


def test_pad_draft():
    assert pad_draft([4], K) == [4, PAD_DRAFT, PAD_DRAFT]
    assert pad_draft([4, 5, 6], K) == [4, 5, 6]
    with pytest.raises(ValueError):
        pad_draft([], K)
    with pytest.raises(ValueError):
        pad_draft([1, 2, 3, 4], K)


def test_merge_keeps_mtp_rows_without_a_lookup_and_pads_lookup_rows():
    mtp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    drafts, sources = merge_drafts(mtp, [[], [10, 11], [12, 13, 14, 15]], K, [True, True, False])
    assert drafts == [[1, 2, 3], [10, 11, PAD_DRAFT], [7, 8, 9]]
    assert sources == ["mtp", "lookup", "mtp"]
    assert all(len(d) == K for d in drafts), "every row keeps the MTP width (the drafter's row layout)"


def test_merge_refuses_inconsistent_rows():
    with pytest.raises(ValueError):
        merge_drafts([[1, 2, 3]], [[], []], K, [True])
    with pytest.raises(ValueError):
        merge_drafts([[1, 2]], [[]], K, [True])


def test_hybrid_merge_replaces_decode_rows_with_a_match_and_counts_acceptance():
    hybrid = HybridLookup(K, 2, 4, 4096)
    # Request "a" repeats a span: its last two tokens 5 6 occurred before, followed by 7 8 9.
    a = [1, 5, 6, 7, 8, 9, 2, 5, 6]
    b = list(range(20, 29))
    token_ids = np.stack([_row(a), _row(b)])
    num_tokens = np.array([len(a), len(b)], dtype=np.int32)
    mtp = [[90, 91, 92], [93, 94, 95]]
    drafts = hybrid.merge(mtp, ["a", "b"], token_ids, num_tokens, [[6], [28]], num_decodes=2)
    assert drafts == [[7, 8, 9], [93, 94, 95]]
    assert hybrid.last_source == {"a": "lookup", "b": "mtp"}
    # Next step: "a" accepted all three lookup drafts (4 sampled tokens), "b" one MTP draft (2 sampled).
    a2 = a + [7, 8, 9, 2]
    b2 = b + [93, 30]
    token_ids = np.stack([_row(a2), _row(b2)])
    num_tokens = np.array([len(a2), len(b2)], dtype=np.int32)
    hybrid.merge(mtp, ["a", "b"], token_ids, num_tokens, [[7, 8, 9, 2], [93, 30]], num_decodes=2)
    assert hybrid.counts["lookup"][1:] == [3, 1] and hybrid.counts["mtp"][1:] == [1, 1]
    assert "lookup rows" in hybrid.summary()


def test_hybrid_merge_leaves_unaligned_prefill_rows_and_unsampled_rows_to_mtp():
    hybrid = HybridLookup(K, 2, 4, 4096)
    rep = [1, 5, 6, 7, 8, 9, 2, 5, 6]
    token_ids = np.stack([_row(rep), _row(rep), _row(rep)])
    num_tokens = np.array([len(rep)] * 3, dtype=np.int32)
    mtp = [[90, 91, 92], [93, 94, 95]]
    # Three requests, two draft rows (one prefill batch produced no draft row): only the decode row is replaced.
    drafts = hybrid.merge(mtp, ["d", "p1", "p2"], token_ids, num_tokens, [[6], [6], [6]], num_decodes=1)
    assert drafts == [[7, 8, 9], [93, 94, 95]]
    # A request that sampled nothing (a prefill chunk) keeps MTP's row and is not counted.
    hybrid2 = HybridLookup(K, 2, 4, 4096)
    drafts = hybrid2.merge([[90, 91, 92]], ["d"], token_ids[:1], num_tokens[:1], [[]], num_decodes=1)
    assert drafts == [[90, 91, 92]] and hybrid2.last_source == {}


def test_mode_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP", raising=False)
    assert lookup_mode() == SpecLookupMode.OFF
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP", "1")
    assert lookup_mode() == SpecLookupMode.HYBRID
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP", "7")
    assert lookup_mode() == SpecLookupMode.OFF


def test_from_env_reads_the_ngram_sizes(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_LOOKUP_MIN_NGRAM", "3")
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_LOOKUP_MAX_NGRAM", "6")
    hybrid = HybridLookup.from_env(K, 4096)
    assert (hybrid.min_ngram, hybrid.max_ngram, hybrid.k) == (3, 6, K)
    with pytest.raises(ValueError):
        HybridLookup(K, 5, 4, 4096)


def test_merge_modes_per_row():
    mtp = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    lookup = [[20], [21, 22], [], [23]]
    modes = [SpecLookupMode.HYBRID, SpecLookupMode.OFF, SpecLookupMode.LOOKUP_ONLY, SpecLookupMode.LOOKUP_ONLY]
    drafts, sources = merge_drafts(mtp, lookup, K, [True] * 4, modes)
    assert drafts == [[20, PAD_DRAFT, PAD_DRAFT], [4, 5, 6], [PAD_DRAFT] * K, [23, PAD_DRAFT, PAD_DRAFT]]
    assert sources == ["lookup", "mtp", "none", "lookup"]


def test_request_mode_overrides_the_default():
    assert request_mode(None, SpecLookupMode.HYBRID) == SpecLookupMode.HYBRID
    assert request_mode({REQUEST_KEY: 0}, SpecLookupMode.HYBRID) == SpecLookupMode.OFF
    assert request_mode({REQUEST_KEY: "2"}, SpecLookupMode.HYBRID) == SpecLookupMode.LOOKUP_ONLY
    assert request_mode({REQUEST_KEY: 9}, SpecLookupMode.HYBRID) == SpecLookupMode.HYBRID
    assert request_mode({"other": 1}, SpecLookupMode.LOOKUP_ONLY) == SpecLookupMode.LOOKUP_ONLY


def test_drafts_off_rows_are_the_requests_that_turned_lookup_off():
    # the n-gram server's per-request switch: mode 0 drafts nothing; any other mode, a bad value or none drafts
    args = [None, {REQUEST_KEY: 0}, {REQUEST_KEY: "0"}, {REQUEST_KEY: 1}, {REQUEST_KEY: 2}, {REQUEST_KEY: "x"},
            {"other": 0}, {}]
    assert drafts_off_rows(args) == [1, 2]
    assert drafts_off_rows([]) == []


def test_hybrid_merge_honours_per_request_modes():
    hybrid = HybridLookup(K, 2, 4, 4096)
    rep = [1, 5, 6, 7, 8, 9, 2, 5, 6]
    fresh = list(range(30, 39))
    token_ids = np.stack([_row(rep), _row(rep), _row(fresh)])
    num_tokens = np.array([len(rep), len(rep), len(fresh)], dtype=np.int32)
    mtp = [[90, 91, 92], [93, 94, 95], [96, 97, 98]]
    drafts = hybrid.merge(mtp, ["a", "b", "c"], token_ids, num_tokens, [[6], [6], [38]], 3,
                          [{REQUEST_KEY: 0}, None, {REQUEST_KEY: 2}])
    assert drafts == [[90, 91, 92], [7, 8, 9], [PAD_DRAFT] * K]
    assert hybrid.last_source == {"a": "mtp", "b": "lookup", "c": "none"}


def test_a_new_set_of_requests_logs_the_counters_of_the_load_that_ended(caplog):
    hybrid = HybridLookup(K, 2, 4, 4096)
    rep = [1, 5, 6, 7, 8, 9, 2, 5, 6]
    token_ids, num_tokens = np.stack([_row(rep)]), np.array([len(rep)], dtype=np.int32)
    import logging

    from vllm_gaudi.v1.spec_decode import flash_next_lookup

    # the module logs through the "vllm" logger (what a server prints); it does not propagate to the root logger
    assert flash_next_lookup.logger.name == "vllm"
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("INFO", logger="vllm"):
            hybrid.merge([[90, 91, 92]], ["a"], token_ids, num_tokens, [[6]], 1)
            hybrid.merge([[90, 91, 92]], ["a"], token_ids, num_tokens, [[6, 7, 8]], 1)  # the lookup row accepted 2
            assert not [r for r in caplog.records if "a new set of requests" in r.getMessage()]
            hybrid.merge([[90, 91, 92]], ["b"], token_ids, num_tokens, [[6]], 1)  # "a" is gone: one line
    finally:
        vllm_logger.removeHandler(caplog.handler)
    lines = [r.getMessage() for r in caplog.records if "a new set of requests" in r.getMessage()]
    assert len(lines) == 1 and "lookup rows 2, accepted 2 over 1 verified rows" in lines[0]
    assert "after 2 steps" in lines[0]


def test_oracle_drafts_the_recorded_continuation():
    from vllm_gaudi.v1.spec_decode.flash_next_lookup import LookupOracle

    stream = list(range(100, 120))
    oracle = LookupOracle([stream, [7, 8, 9]], schedule=[3, 1, 0, 2], tamper_every=0)
    # context length 5: schedule[5 % 4] = 1 token; length 6: none; length 7: 2; length 8: 3
    assert oracle.draft(np.array(stream[:5], dtype=np.int32), K) == [105]
    assert oracle.draft(np.array(stream[:6], dtype=np.int32), K) == []
    assert oracle.draft(np.array(stream[:7], dtype=np.int32), K) == [107, 108]
    assert oracle.draft(np.array(stream[:8], dtype=np.int32), K) == [108, 109, 110]
    assert oracle.draft(np.array(stream[:8], dtype=np.int32), 2) == [108, 109]  # never more than k
    # the end of a stream shortens the draft; a whole stream, an unknown context and an empty one draft nothing
    assert oracle.draft(np.array(stream[:19], dtype=np.int32), K) == [119]
    assert oracle.draft(np.array(stream, dtype=np.int32), K) == []
    assert oracle.draft(np.array(stream[:7] + [1], dtype=np.int32), K) == []
    assert oracle.draft(np.array([], dtype=np.int32), K) == []
    assert oracle.drafts == 5


def test_oracle_tampers_the_last_draft_token_on_schedule():
    from vllm_gaudi.v1.spec_decode.flash_next_lookup import LookupOracle

    stream = list(range(100, 140))
    oracle = LookupOracle([stream], schedule=[3], tamper_every=4)
    assert oracle.draft(np.array(stream[:7], dtype=np.int32), K) == [107, 108, 109]
    tampered = oracle.draft(np.array(stream[:8], dtype=np.int32), K)
    assert tampered[:2] == [108, 109] and tampered[2] != 110 and tampered[2] == 110 ^ 1
    assert oracle.tampered == 1
    with pytest.raises(ValueError):
        LookupOracle([stream], schedule=[])


def test_oracle_file_and_the_hybrid(monkeypatch, tmp_path):
    from vllm_gaudi.v1.spec_decode.flash_next_lookup import ORACLE_ENV, lookup_oracle

    monkeypatch.delenv(ORACLE_ENV, raising=False)
    assert lookup_oracle() is None and HybridLookup.from_env(K, 4096).oracle is None
    stream = list(range(200, 230))
    path = tmp_path / "oracle.json"
    path.write_text('{"streams": [%s], "schedule": [2], "tamper_every": 0}' % stream)
    monkeypatch.setenv(ORACLE_ENV, str(path))
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP", "1")
    hybrid = HybridLookup.from_env(K, 4096)
    assert hybrid.oracle is not None and hybrid.oracle.schedule == [2]
    token_ids = np.stack([_row(stream[:9]), _row(list(range(9)))])
    num_tokens = np.array([9, 9], dtype=np.int32)
    drafts = hybrid.merge([[90, 91, 92], [93, 94, 95]], ["a", "b"], token_ids, num_tokens, [[208], [8]], 2)
    # the recorded stream's next two tokens padded to k; the unknown context keeps MTP's draft (no n-gram search)
    assert drafts == [[209, 210, PAD_DRAFT], [93, 94, 95]]
