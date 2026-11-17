# SPDX-License-Identifier: Apache-2.0
"""Whole-model TP2/TP4/TP8 prefill equivalence on gloo CPU ranks (M0-T7d).

Builds the 8-head tiny Flash-Next checkpoint (``tiny_tp8_config.json``: the T1 fixture with
``num_attention_heads: 8``, legal at TP8) and runs the five reference prompts through the model
built sharded on 1/2/4/8 gloo ranks (``spikes/cpu-tp/harness.py``, rank function
``tp_ranks.prefill_rank``), proving:

* the TP1 last-token logits match the checkpoint's own HF ``reference.json`` at the top-5 ids,
* every rank at TP2/4/8 holds the same last-token hidden state, and rank 0 matches the TP1 run
  (``lm_head`` is vocab-sharded at TP>1, so the comparison across TP sizes is on the hidden state),
* every rank issues exactly the same TP collectives, in the same order, at every TP size.

fp32 throughout (the T7a/T7b tolerance regime; bf16 TP runs are the hardware stage).
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import dense_reference_prompts  # noqa: E402
from tests.unit_tests.models.flash_next.tp_ranks import prefill_rank  # noqa: E402  (adds spikes/cpu-tp to sys.path)

from harness import run_ranks  # noqa: E402

TP_SIZES = [2, 4, 8]

_RUN_CACHE: dict[int, list] = {}


def _ranks(tp: int, checkpoint: Path) -> list:
    """Run ``prefill_rank`` (fp32) on ``tp`` gloo ranks once per TP size and cache the results."""
    if tp not in _RUN_CACHE:
        _RUN_CACHE[tp] = run_ranks(tp, prefill_rank, str(checkpoint), tp, True, timeout_s=300)
    return _RUN_CACHE[tp]


@pytest.fixture(scope="module")
def tp1_baseline(tiny_tp8_checkpoint) -> list:
    """The fp32 TP1 run: the reference every TP2/4/8 run is compared against."""
    return _ranks(1, tiny_tp8_checkpoint)


def test_tp1_matches_reference(tiny_tp8_checkpoint, tp1_baseline):
    """TP1's last-prefill top-5 logits match the 8-head checkpoint's HF reference and keep its top-1."""
    reference = json.loads((tiny_tp8_checkpoint / "reference.json").read_text(encoding="utf-8"))
    assert len(tp1_baseline[0]["top5_logits"]) == len(dense_reference_prompts(reference))
    for i, prompt in enumerate(dense_reference_prompts(reference)):
        expected = prompt["last_prefill_top5"]["logits"]
        got = tp1_baseline[0]["top5_logits"][i]
        np.testing.assert_allclose(got, expected, atol=1e-5, err_msg=f"prompt {i} top-5 logits")
        assert int(np.argmax(got)) == 0, f"prompt {i}: reference top-1 id is not ours"


@pytest.mark.parametrize("tp", TP_SIZES)
def test_tp_matches_tp1(tp, tiny_tp8_checkpoint, tp1_baseline):
    """At TP2/4/8 every rank's last-token hidden states agree within 1e-6, and with TP1 within 1e-5."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    reference = json.loads((tiny_tp8_checkpoint / "reference.json").read_text(encoding="utf-8"))
    assert results[0]["hidden"].shape == (len(dense_reference_prompts(reference)), 128)
    for rank, res in enumerate(results):
        np.testing.assert_allclose(res["hidden"],
                                   results[0]["hidden"],
                                   atol=1e-6,
                                   err_msg=f"rank {rank} != rank 0 at TP{tp}")
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
