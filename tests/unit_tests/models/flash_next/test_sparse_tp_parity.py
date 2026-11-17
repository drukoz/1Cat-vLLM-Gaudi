# SPDX-License-Identifier: Apache-2.0
"""TP1/TP2/TP4/TP8 equivalence of the sparse QSA prefill path on gloo CPU ranks (M2-T8).

Builds the 8-head tiny Flash-Next checkpoint (``tiny_tp8_checkpoint``, legal at TP8) and prefills
its 2600-token reference prompt through the model built sharded on 1/2/4/8 gloo ranks in
``QsaMode.SPARSE`` (rank function ``tp_ranks.sparse_prefill_rank``), proving:

* the TP1 last-token logits match the checkpoint's own (sparse) HF ``reference.json`` at the top-5
  ids, so the sparse ranks run the same selection M2-T3 proved on world 1,
* every rank at TP2/4/8 holds the same last-token hidden state, and rank 0 matches the TP1 run
  (the indexer is replicated, so the selection is identical on every rank and the attention shards
  split the query heads; ``o_proj`` is a ``RowParallelLinear``, so the hidden state is full again),
* every rank issues exactly the same TP collectives, in the same order, at every TP size.

fp32 throughout (the T7d tolerance regime; bf16 TP runs are the hardware stage).
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.tp_ranks import sparse_prefill_rank  # noqa: E402  (adds spikes/cpu-tp)

from harness import run_ranks  # noqa: E402

TP_SIZES = [2, 4, 8]
LONG_PROMPT = 2600

_RUN_CACHE: dict[int, list] = {}


def _ranks(tp: int, checkpoint: Path) -> list:
    """Run ``sparse_prefill_rank`` (fp32) on ``tp`` gloo ranks once per TP size and cache the results."""
    if tp not in _RUN_CACHE:
        _RUN_CACHE[tp] = run_ranks(tp, sparse_prefill_rank, str(checkpoint), tp, True, timeout_s=900)
    return _RUN_CACHE[tp]


@pytest.fixture(scope="module")
def tp1_baseline(tiny_tp8_checkpoint) -> list:
    """The fp32 sparse TP1 run: the reference every TP2/4/8 run is compared against."""
    return _ranks(1, tiny_tp8_checkpoint)


def test_tp1_matches_reference(tiny_tp8_checkpoint, tp1_baseline):
    """TP1's sparse 2600-token prefill matches the 8-head checkpoint's HF reference and keeps its top-1."""
    reference = json.loads((tiny_tp8_checkpoint / "reference.json").read_text(encoding="utf-8"))
    prompt = reference["prompts"][-1]
    assert len(prompt["prompt_ids"]) == LONG_PROMPT
    got = tp1_baseline[0]["top5_logits"]
    assert len(got) == 5, f"expected the top-5 logits of the 2600-token prompt, got {got}"
    np.testing.assert_allclose(got,
                               prompt["last_prefill_top5"]["logits"],
                               atol=1e-5,
                               err_msg="2600-token prompt top-5 logits")
    assert int(np.argmax(got)) == 0, "reference top-1 id is not ours"
    # the M2-T3 selection contract on the last query (row 2599 >= width): >= 2048 valid entries.
    assert tp1_baseline[0]["selected_rows"] >= 2048, "the sparse selection is not the M2-T3 selection"


@pytest.mark.parametrize("tp", TP_SIZES)
def test_tp_matches_tp1(tp, tiny_tp8_checkpoint, tp1_baseline):
    """At TP2/4/8 every rank's last-token hidden state agrees within 1e-6, and with TP1 within 1e-5."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    assert results[0]["hidden"].shape == (1, 128)
    for rank, res in enumerate(results):
        np.testing.assert_allclose(res["hidden"],
                                   results[0]["hidden"],
                                   atol=1e-6,
                                   err_msg=f"rank {rank} != rank 0 at TP{tp}")
        assert res["selected_rows"] == results[0]["selected_rows"], f"rank {rank} selection differs at TP{tp}"
    np.testing.assert_allclose(results[0]["hidden"],
                               tp1_baseline[0]["hidden"],
                               atol=1e-5,
                               err_msg=f"TP{tp} rank 0 != TP1")


@pytest.mark.parametrize("tp", TP_SIZES)
def test_collectives_identical_across_ranks(tp, tiny_tp8_checkpoint):
    """At TP2/4/8 every rank issues the same non-empty list of collectives, in the same order."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    assert results[0]["collectives"], f"no collectives recorded at TP{tp}"
    for rank, res in enumerate(results):
        assert res["collectives"] == results[0]["collectives"], f"rank {rank} collectives differ at TP{tp}"
