# SPDX-License-Identifier: Apache-2.0
"""GDN core-decode TP parity on 1/2/4/8 gloo CPU ranks (T3c of track RUK-2812).

Builds the 8-head tiny Flash-Next checkpoint (``tiny_tp8_config.json``, 16 key heads and 48 value
heads of dimension 16, so each rank holds ``48 / tp`` value heads), takes its **first**
``HPUGatedDeltaNetAttention`` module and runs it, called directly, through a 12-row prefill into
slot 1 and 6 decode steps on 1/2/4/8 gloo ranks (``spikes/cpu-tp/harness.py``, rank function
``gdn_core_ranks.gdn_decode_rank``) in ``GdnCoreDecodeMode.OFF`` and ``FUSED``; the last decode
step loads slot 1 and stores slot 2 (prefix caching on). This proves:

* mode OFF calls no op; mode FUSED calls ``gdn_core_decode`` exactly ``DECODE_STEPS`` times on
  every rank and its one-shot precondition check passes (``_gdn_core_ok is True``),
* every rank's decode rows equal rank 0's, the FUSED rows are within 1e-6 of OFF's on every rank,
  the slot 2 SSM state within 1e-6 and the slot 2 conv state bit-exact,
* the load != store step leaves slot 1 untouched in both modes and writes slot 2's conv state,
* rank 0's rows match the TP1 OFF run within 1e-6 and the slot 2 SSM state concatenated over the
  ranks (value-head axis) matches TP1 OFF within 1e-6.

Probe measurements on branch head ``bbe249e`` (this exact design, fp32, seed 0): FUSED vs OFF rows
<= 3.0e-8 and slot 2 SSM exactly equal on every rank, rank rows equal rank 0's exactly, TP8 rank 0
rows vs TP1 OFF <= 3.7e-8, concatenated slot 2 SSM vs TP1 OFF <= 9.4e-10.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeMode  # noqa: E402
from tests.unit_tests.models.flash_next.gdn_core_ranks import (  # noqa: E402  (adds spikes/cpu-tp to sys.path)
    DECODE_STEPS, OP_OF_MODE, gdn_decode_rank)

from harness import run_ranks  # noqa: E402

# RECURRENCE joins when the mode-2 layer wiring (RUK-2834) merges
MODES = (GdnCoreDecodeMode.OFF, GdnCoreDecodeMode.FUSED)
TP_SIZES = (1, 2, 4, 8)
ROW_TOL = 1e-6
STATE_TOL = 1e-6

_RUN_CACHE: dict[int, list] = {}


def _ranks(tp: int, checkpoint: Path) -> list:
    """Run ``gdn_decode_rank`` on ``tp`` gloo ranks once per TP size and cache the results."""
    if tp not in _RUN_CACHE:
        _RUN_CACHE[tp] = run_ranks(tp, gdn_decode_rank, str(checkpoint), tp, tuple(int(m) for m in MODES),
                                   timeout_s=300)
    return _RUN_CACHE[tp]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_op_called_once_per_step(tp, tiny_tp8_checkpoint):
    """Mode OFF calls no op on every rank; every other mode calls its op DECODE_STEPS times and passes its check."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    for rank, res in enumerate(results):
        for mode in MODES:
            calls = res[int(mode)]["calls"]
            if mode == GdnCoreDecodeMode.OFF:
                for name, count in calls.items():
                    assert count == 0, f"tp={tp} rank={rank} mode={int(mode)}: op {name} called {count} times"
            else:
                op = OP_OF_MODE[int(mode)]
                assert calls[op] == DECODE_STEPS, (f"tp={tp} rank={rank} mode={int(mode)}: op {op} called "
                                                   f"{calls[op]} times, expected {DECODE_STEPS}")
                assert res[int(mode)]["ok"] is True, f"tp={tp} rank={rank} mode={int(mode)}: _gdn_core_ok is not True"


@pytest.mark.parametrize("tp", TP_SIZES)
def test_ranks_agree(tp, tiny_tp8_checkpoint):
    """For every mode, every rank's decode rows equal rank 0's within ROW_TOL."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    for mode in MODES:
        for rank, res in enumerate(results):
            np.testing.assert_allclose(res[int(mode)]["rows"],
                                       results[0][int(mode)]["rows"],
                                       atol=ROW_TOL,
                                       rtol=0,
                                       err_msg=f"tp={tp} rank={rank} mode={int(mode)}: rows != rank 0")


@pytest.mark.parametrize("tp", TP_SIZES)
def test_op_matches_off(tp, tiny_tp8_checkpoint):
    """On every rank each non-OFF mode matches OFF: rows within ROW_TOL, SSM within STATE_TOL, conv bit-exact."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    for rank, res in enumerate(results):
        off = res[int(GdnCoreDecodeMode.OFF)]
        for mode in MODES:
            if mode == GdnCoreDecodeMode.OFF:
                continue
            np.testing.assert_allclose(res[int(mode)]["rows"],
                                       off["rows"],
                                       atol=ROW_TOL,
                                       rtol=0,
                                       err_msg=f"tp={tp} rank={rank} mode={int(mode)}: rows != OFF")
            np.testing.assert_allclose(res[int(mode)]["ssm2"],
                                       off["ssm2"],
                                       atol=STATE_TOL,
                                       rtol=0,
                                       err_msg=f"tp={tp} rank={rank} mode={int(mode)}: slot 2 SSM != OFF")
            np.testing.assert_array_equal(res[int(mode)]["conv2"],
                                          off["conv2"],
                                          err_msg=f"tp={tp} rank={rank} mode={int(mode)}: slot 2 conv != OFF")


@pytest.mark.parametrize("tp", TP_SIZES)
def test_move_step_keeps_load_slot(tp, tiny_tp8_checkpoint):
    """On every rank and mode the load 1 / store 2 step leaves slot 1 untouched and writes slot 2's conv."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    for rank, res in enumerate(results):
        for mode in MODES:
            assert res[int(mode)]["slot1_unchanged"], (f"tp={tp} rank={rank} mode={int(mode)}: slot 1 state "
                                                       "changed in the load != store step")
            assert res[int(mode)]["conv2"].any(), f"tp={tp} rank={rank} mode={int(mode)}: slot 2 conv is all zeros"


@pytest.mark.parametrize("tp", (2, 4, 8))
def test_matches_tp1(tp, tiny_tp8_checkpoint):
    """For every mode, rank 0's rows match TP1 OFF within ROW_TOL and the concatenated SSM within STATE_TOL."""
    results = _ranks(tp, tiny_tp8_checkpoint)
    tp1_off = _ranks(1, tiny_tp8_checkpoint)[0][int(GdnCoreDecodeMode.OFF)]
    for mode in MODES:
        np.testing.assert_allclose(results[0][int(mode)]["rows"],
                                   tp1_off["rows"],
                                   atol=ROW_TOL,
                                   rtol=0,
                                   err_msg=f"tp={tp} rank=0 mode={int(mode)}: rows != TP1 OFF")
        full_ssm = np.concatenate([res[int(mode)]["ssm2"] for res in results], axis=0)
        assert full_ssm.shape == tp1_off["ssm2"].shape == (48, 16, 16), f"tp={tp} mode={int(mode)}: SSM shape"
        np.testing.assert_allclose(full_ssm,
                                   tp1_off["ssm2"],
                                   atol=STATE_TOL,
                                   rtol=0,
                                   err_msg=f"tp={tp} mode={int(mode)}: concatenated slot 2 SSM != TP1 OFF")
