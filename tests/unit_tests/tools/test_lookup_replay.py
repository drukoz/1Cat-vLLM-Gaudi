# SPDX-License-Identifier: Apache-2.0
"""tools/flash_next/lookup_replay.py: the drafters against vLLM's own n-gram search, the replay arithmetic, and a
run over a probe record."""

from __future__ import annotations

import json
import random

import pytest

from tools.flash_next import lookup_replay as lr


def _draft(tokens, min_n, max_n, k, policy):
    drafter = lr.IndexDrafter(min_n, max_n, k, policy)
    drafter.extend(tokens)
    return drafter.propose()[0]


def test_earliest_is_vllms_ngram_search():
    """The index drafter's ``earliest`` policy drafts what vLLM's NgramProposer kernel drafts: the longest suffix
    n-gram at its earliest occurrence (random streams over a small vocabulary, so n-grams repeat and overlap)."""
    np = pytest.importorskip("numpy")
    ngram = pytest.importorskip("vllm.v1.spec_decode.ngram_proposer")
    rng = random.Random(0)
    checked = 0
    for _ in range(400):
        tokens = [rng.randrange(4) for _ in range(rng.randrange(1, 60))]
        min_n, max_n, k = rng.choice([(1, 3), (2, 4), (3, 8), (3, 3)]) + (rng.choice([1, 3, 8]), )
        want = [int(t) for t in ngram._find_longest_matched_ngram_and_propose_tokens(
            np.asarray(tokens, dtype=np.int32), min_n, max_n, 1 << 30, k)]
        assert _draft(tokens, min_n, max_n, k, "earliest") == want, (tokens, min_n, max_n, k)
        checked += bool(want)
    assert checked > 200


def test_recent_takes_the_latest_earlier_occurrence():
    # "5 6" occurs three times before the suffix: after the first comes 7, after the latest comes 9
    tokens = [5, 6, 7, 1, 5, 6, 8, 2, 5, 6, 9, 3, 5, 6]
    assert _draft(tokens, 2, 4, 2, "earliest") == [7, 1]
    assert _draft(tokens, 2, 4, 2, "recent") == [9, 3]
    # a suffix that never occurred before drafts nothing; one earlier occurrence serves both policies
    assert _draft([1, 2, 3, 4], 2, 4, 3, "recent") == []
    assert _draft([1, 2, 9, 1, 2], 2, 4, 3, "recent") == [9, 1, 2] == _draft([1, 2, 9, 1, 2], 2, 4, 3, "earliest")
    with pytest.raises(ValueError):
        lr.IndexDrafter(2, 4, 3, "newest")
    with pytest.raises(ValueError):
        lr.IndexDrafter(5, 4, 3)


def test_replay_of_a_pure_copy():
    """The output copies the prompt: after the first n-gram matched, every step accepts k drafts."""
    prompt = list(range(100, 140))
    output = list(range(100, 140))
    steps = lr.replay(prompt, output, 3, 8, 4, "earliest")
    assert sum(accepted + 1 for _drafted, accepted in steps) == len(output)
    assert steps[:3] == [(0, 0), (0, 0), (0, 0)]  # no 3-gram of the output is in the context yet
    assert all(step == (4, 4) for step in steps[3:-1])
    summary = lr.summarize(steps, 4)
    assert summary["tokens"] == 40 and summary["steps"] == len(steps)
    assert summary["position_rates_on_hits"][0] == 1.0 and summary["accepted_histogram"][0] == 3
    # the last draft is cut at the end of the output: a step never commits more than the output has
    assert steps[-1][0] <= 4 and steps[-1][1] == steps[-1][0]


def test_replay_stops_at_an_edit_and_follow_reads_on():
    prompt = list(range(100, 130))
    edited = list(range(100, 112)) + [999] + list(range(112, 130))  # one inserted token
    steps = lr.replay(prompt, edited, 3, 8, 8, "earliest")
    assert sum(a + 1 for _d, a in steps) == len(edited)
    assert any(d > a for d, a in steps), "the inserted token rejected no draft"
    summary = lr.summarize(steps, 8)
    assert summary["position_rates"] == sorted(summary["position_rates"], reverse=True)  # at least j is monotone
    assert summary["tokens_per_step"] == pytest.approx(len(edited) / len(steps))
    # follow keeps the copy position after a fully accepted draft, so it never does worse here
    follow = lr.replay(prompt, edited, 3, 8, 8, "follow")
    assert sum(a + 1 for _d, a in follow) == len(edited) and len(follow) <= len(steps)


def test_rows_per_verification_step():
    # half of the steps draft nothing (the placeholder: 2 tokens wide), half draft 4 tokens (5 wide)
    steps = [(0, 0), (4, 4)] * 50
    assert lr.expected_ngram_width(steps, 4, 1) == pytest.approx(3.5)
    assert lr.expected_ngram_width(steps, 4, 2) == pytest.approx(2 * 0.25 + 5 * 0.75)
    assert lr.expected_ngram_width(steps, 4, 32) == pytest.approx(5.0, abs=1e-6)
    assert lr.expected_ngram_width([], 4, 32) is None
    summary = lr.summarize(steps, 4)
    assert summary["rows_per_step_mtp_shapes"] == {"1": 5, "8": 40, "32": 160}
    assert summary["rows_per_step_ngram"]["1"] == pytest.approx(3.5)
    assert summary["rows_per_step_ngram"]["32"] == pytest.approx(160.0, abs=1e-4)
    assert summary["drafted_histogram"] == [50, 0, 0, 0, 50]


def test_run_over_a_probe_record(tmp_path, capsys):
    prompt = list(range(100, 160))
    answer = list(range(100, 160))
    probe = {"label": "nospec", "prompts": {"aaa": prompt, "bbb": [1, 2, 3]},
             "rounds": [{"kind": "edit", "prompt_sha": ["aaa"], "token_ids": [answer[:20]]},
                        {"kind": "editf", "prompt_sha": ["aaa"], "token_ids": [answer]},
                        {"kind": "edit", "prompt_sha": ["aaa"], "token_ids": [answer[:30]]},  # the longer one counts
                        {"kind": "prose", "prompt_sha": ["bbb"], "token_ids": [[7, 8, 9, 10, 11, 12]]},
                        {"kind": "prose", "prompt_sha": ["ccc"], "token_ids": [[1]]}]}  # no prompt ids: left out
    streams = lr.streams_of([probe], ["edit", "editf", "prose"])
    assert [len(o) for _p, o in streams["edit"]] == [30] and len(streams["prose"]) == 1
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    out = tmp_path / "replay.json"
    assert lr.main([str(path), "--depths", "3,8", "--policies", "earliest,follow", "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert set(record["kinds"]) == {"edit", "editf", "prose"}
    edit8 = record["kinds"]["editf"]["drafters"]["earliest-k8"]
    assert edit8["tokens_per_step"] > 5 and edit8["k"] == 8 and len(edit8["position_rates"]) == 8
    prose = record["kinds"]["prose"]["drafters"]["earliest-k3"]
    assert prose["tokens_per_step"] == 1.0 and prose["hit_share"] == 0.0
    printed = capsys.readouterr().out
    assert "editf: 1 streams, 60 output tokens" in printed and "earliest-k8" in printed
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"rounds": [{"kind": "edit", "prompt_sha": ["x"], "texts": ["hello"]}]}))
    assert lr.main([str(empty)]) == 1
