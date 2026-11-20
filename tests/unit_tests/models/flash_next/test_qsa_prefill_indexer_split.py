# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_PREFILL_INDEXER_SPLIT=1 (idea G6-4): the prompt indexer scores only this rank's rows.

The indexer projection is replicated and the compressed key cache holds the same keys on every rank, so today
each of the 8 TP ranks scores all query rows of a prompt chunk against all keys (fp32 ``[512, H, N]`` tiles and
a top-k over ``N``, N = context / 4) and gets the same answer. With the flag, rank r scores rows
``[r*T/tp, (r+1)*T/tp)`` and one all-gather collects the block ids (as fp32: exact below 2^24). The selection
must be the one the unsplit indexer makes, on every rank, at contexts that need the top-k and at bucket padding.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.qsa_indexer_ranks import select_rank  # noqa: E402

from harness import run_ranks  # noqa: E402

BUDGET = 64  # 16 blocks of 4: contexts of a few hundred tokens already need the top-k
CASES = [
    (0, 256, 256),  # cold first chunk
    (1024, 256, 256),  # a later chunk: context keys from the cache, top-k over 320 blocks
    (2048, 512, 300),  # bucket padding after the real tokens
    (4096, 1024, 1024),  # two score tiles of 512 rows unsplit, 1024 / tp rows per rank split
]


@pytest.fixture(scope="module", params=[2, 4])
def ranks(request) -> tuple[int, list]:
    return request.param, run_ranks(request.param, select_rank, CASES, BUDGET, timeout_s=600)


def test_split_selection_equals_the_unsplit_one_on_every_rank(ranks):
    tp, results = ranks
    for case_index, case in enumerate(CASES):
        reference = results[0][case_index]["0"]["selected"]
        for rank in range(tp):
            assert results[rank][case_index]["0"]["selected"] == reference, (tp, rank, case)
            assert results[rank][case_index]["1"]["selected"] == reference, (tp, rank, case)


def test_split_adds_one_all_gather_per_row_and_unsplit_none(ranks):
    tp, results = ranks
    for rank_results in results:
        for case_result in rank_results:
            assert case_result["0"]["collectives"] == []
            assert case_result["1"]["collectives"] == ["all_gather"]
            assert case_result["1"]["dtype"] == "torch.int32"


def test_the_selection_uses_the_top_k():
    # a guard on the cases: at these contexts the budget is smaller than the visible blocks
    assert any((ctx + seq) // 4 > BUDGET // 4 for ctx, seq, _ in CASES)
