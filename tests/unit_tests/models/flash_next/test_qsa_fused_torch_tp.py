# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_INDEXER_SHARD=1 on 1/2/4/8 gloo CPU ranks: the request-sharded indexer equals the replicated one.

Every rank builds the same standalone ``HpuQsaIndexer`` (seeded weights and caches,
``qsa_fused_ranks.shard_select_rank``) and selects one decode step in mode 0, in mode 1 replicated and
in mode 1 sharded by request (I18). Proven per TP size and case:

* the sharded selection is the same on every rank and equal to the replicated one and to mode 0 on
  every real row (each request is scored whole by one rank, so no candidate merge is involved);
* a sharded step issues exactly one TP collective per ``select``, an all-gather of the int32
  ``[bs / tp * T_cols, topk]`` block selections; a bucket whose ``bs`` is not a multiple of ``tp``
  (``decode6`` at TP4 and TP8) and the replicated modes issue none;
* the cache writes are replicated: every rank's compressed key cache is the same after the step.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.qsa_fused_ranks import CASES, shard_select_rank  # noqa: E402

from harness import run_ranks  # noqa: E402

SEED = 20261120
TOPK = 512
_RUNS: dict = {}


def ranks(tp: int, case: str) -> list:
    key = (tp, case)
    if key not in _RUNS:
        _RUNS[key] = run_ranks(tp, shard_select_rank, tp, case, SEED, timeout_s=600)
    return _RUNS[key]


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_sharded_equals_replicated(tp, case):
    results = ranks(tp, case)
    specs, t_cols = CASES[case]
    bs = len(specs)
    rows = results[0]["real_rows"]
    ref = results[0]["mode0"][rows]
    for r in results:
        assert np.array_equal(r["replicated"][rows], ref), f"rank {r['rank']}: mode 1 replicated != mode 0"
        assert np.array_equal(r["sharded"][rows], ref), f"rank {r['rank']}: mode 1 sharded != mode 0"
        assert np.array_equal(r["sharded"], results[0]["sharded"]), f"rank {r['rank']} holds another selection"
        assert np.array_equal(r["sharded_cache"], results[0]["sharded_cache"]), "the cache writes differ by rank"
        assert r["mode0_collectives"] == [] and r["replicated_collectives"] == []
        shards = tp > 1 and bs % tp == 0
        if shards:
            assert r["sharded_collectives"] == [("all_gather", (bs // tp * t_cols, TOPK), "torch.int32")], (
                r["sharded_collectives"])
        else:
            assert r["sharded_collectives"] == [], r["sharded_collectives"]
