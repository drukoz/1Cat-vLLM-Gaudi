# SPDX-License-Identifier: Apache-2.0
"""The MoE glue at TP8 with expert parallelism, on eight gloo CPU ranks (``spikes/cpu-tp/harness.py``).

The serving layout of the real model: eight ranks, every rank routes identically (the merged gate
weight is replicated), computes only its own experts (rank r of the tiny model holds experts 2r and
2r + 1) and its own shard of the shared expert, and one all-reduce follows the block.  Each rank builds
the 8-head tiny checkpoint sharded through vLLM's real loader on the plugin's MoE path
(``moe_glue_ranks.moe_rank``), once with ``VLLM_HPU_MOE_GLUE=0`` and once with 1; the fused glue ops run
the torch oracle, so the two runs must agree bit for bit on every rank.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.moe_glue_ranks import DECODE_ROWS, PREFILL_TOKENS, moe_rank  # noqa: E402

from harness import run_ranks  # noqa: E402

TP = 8
OFF_LINE = "MoE glue (VLLM_HPU_MOE_GLUE=1) is off for "
_RUN_CACHE: dict[tuple, list] = {}


def _ranks(checkpoint: Path, tp: int, expert_parallel: bool, flag: str) -> list:
    key = (tp, expert_parallel, flag)
    if key not in _RUN_CACHE:
        _RUN_CACHE[key] = run_ranks(tp, moe_rank, str(checkpoint), tp, expert_parallel, flag, timeout_s=900)
    return _RUN_CACHE[key]


@pytest.fixture(scope="module")
def plain(tiny_tp8_checkpoint) -> list:
    return _ranks(tiny_tp8_checkpoint, TP, True, "0")


@pytest.fixture(scope="module")
def fused(tiny_tp8_checkpoint) -> list:
    return _ranks(tiny_tp8_checkpoint, TP, True, "1")


def test_layout_is_tp8_with_expert_parallelism(plain):
    """Rank r holds experts 2r and 2r + 1 whole; the shared expert is sharded; every rank routes all 16."""
    for rank, result in enumerate(plain):
        assert (result["ep_size"], result["ep_rank"], result["local_experts"]) == (TP, rank, 2)
        assert result["experts_min_max"] == [(2 * rank, 2 * rank + 1)]
        assert result["shared_gate_up_shape"] == (2 * 64 // TP, 128)  # gate and up columns of this rank's shard
        low, high = result["routed_ids_min_max"]
        assert low < 2 * rank or high > 2 * rank + 1 or rank in (0, TP - 1)  # the table holds global ids
        assert result["ids_dtype"] == "torch.int64"
    assert {result["routed_ids_min_max"] for result in plain} == {plain[0]["routed_ids_min_max"]}
    assert plain[0]["routed_ids_min_max"][1] >= 8  # experts of other ranks are in every rank's table


def test_flag_off_has_no_glue(plain):
    for result in plain:
        assert result["glue_ok"] == [None] * 4 and result["glue_ops"] == [] and result["lines"] == []
        assert result["merged_gate_shape"] is None


def test_every_rank_runs_every_block_fused(fused):
    for rank, result in enumerate(fused):
        assert result["lines"] == [], f"rank {rank}: {result['lines']}"
        assert result["glue_ok"] == [True] * 4
        assert result["glue_ops"] == ["flashinfer_gaudi.moe_combine.default", "flashinfer_gaudi.moe_route.default",
                                      "flashinfer_gaudi.moe_silu_mul.default"]
        assert result["merged_gate_shape"] == (16 + 1, 128)  # replicated: the same routing on every rank
        assert result["ids_dtype"] == "torch.int64" and result["experts_min_max"] == [(2 * rank, 2 * rank + 1)]


def test_flag_on_equals_flag_off_bit_for_bit_on_every_rank(plain, fused):
    assert plain[0]["hidden"].shape == (len(PREFILL_TOKENS), 128)
    assert plain[0]["block_out"].shape == (4, DECODE_ROWS, 128)
    for rank in range(TP):
        np.testing.assert_array_equal(fused[rank]["hidden"], plain[rank]["hidden"], err_msg=f"rank {rank} hidden")
        np.testing.assert_array_equal(fused[rank]["block_out"], plain[rank]["block_out"], err_msg=f"rank {rank} blocks")
        np.testing.assert_array_equal(fused[rank]["block_out"], fused[0]["block_out"], err_msg=f"rank {rank} != rank 0")
        assert fused[rank]["vendor_calls"] == plain[rank]["vendor_calls"] == 4 * len(PREFILL_TOKENS)
    assert float(np.abs(plain[0]["block_out"]).mean()) > 1e-4  # the block output is not trivially zero


def test_collectives_are_the_same_with_and_without_the_glue(plain, fused):
    """One all-reduce after every MoE block, as before: the glue does not add, drop or reorder a collective."""
    assert plain[0]["collectives"], "no collectives recorded"
    for rank in range(TP):
        assert fused[rank]["collectives"] == plain[rank]["collectives"] == plain[0]["collectives"]


def test_tp8_without_expert_parallelism_falls_back_or_matches(tiny_tp8_checkpoint):
    """TP8 without EP shards the routed experts' weights: whatever the blocks decide, the bits are the torch path's."""
    plain = _ranks(tiny_tp8_checkpoint, TP, False, "0")
    fused = _ranks(tiny_tp8_checkpoint, TP, False, "1")
    for rank in range(TP):
        np.testing.assert_array_equal(fused[rank]["hidden"], plain[rank]["hidden"], err_msg=f"rank {rank} hidden")
        np.testing.assert_array_equal(fused[rank]["block_out"], plain[rank]["block_out"], err_msg=f"rank {rank} blocks")
        assert fused[rank]["collectives"] == plain[rank]["collectives"]
        for line in fused[rank]["lines"]:
            assert line.startswith(OFF_LINE) and line.endswith("; using the torch path")
        assert all(ok is True for ok in fused[rank]["glue_ok"]) or len(fused[rank]["lines"]) == 4
