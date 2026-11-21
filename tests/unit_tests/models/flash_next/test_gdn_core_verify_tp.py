# SPDX-License-Identifier: Apache-2.0
"""GDN core-verify TP parity on 1/2/4/8 gloo CPU ranks (TR of track RUK-2943).

Builds the 8-head tiny Flash-Next checkpoint (``tiny_tp8_config.json``, 16 key heads and 48 value
heads of dimension 16, so each rank holds ``48 / tp`` value heads), takes its **first**
``HPUGatedDeltaNetAttention`` module and runs it, called directly, through a 12-row prefill into
slot 1, verification steps A and B on the lattice ``[1 .. 1 + k]`` (``num_accepted = 1`` and ``k``)
and one plain step (slot ``k``, ``num_accepted = k - 1``) on 1/2/4/8 gloo ranks
(``spikes/cpu-tp/harness.py``, rank function ``gdn_core_ranks.gdn_verify_rank``) in
``GdnCoreVerifyMode.OFF`` (torch MTP path) and ``FUSED``, for every ``k`` of
``flash_next_spec_sizes.SPEC_SIZES`` (3, 4, 5; ``k = 3``: lattice ``[1, 2, 3, 4]``, rows 12..15 and
16..19, plain step on slot 3 with ``num_accepted = 2``, row 20); the conv pool is rebound to
``3 + k`` rows because an MTP conv pool has that many. One set of ranks per TP size runs all the
sizes. For each size this proves:

* mode OFF calls ``gdn_core_verify`` no time; mode FUSED calls it exactly 3 times with
  ``T = [1 + k, 1 + k, 1]`` on every rank and its one-shot precondition check passes
  (``_gdn_core_verify_ok is True``),
* every rank's rows equal rank 0's, the FUSED rows are within 1e-6 of OFF's on every rank, the
  lattice slots' SSM states within 1e-6 and their conv states bit-exact,
* the steps write the states: on every rank and mode each of the ``1 + k`` SSM slots and the slot 1
  conv state (where the lattice's conv row lives) is non-zero,
* rank 0's rows match the TP1 OFF run within 1e-6 and the lattice slots' SSM states concatenated over
  the ranks (value-head axis, shape ``(1 + k, 48, 16, 16)``) match TP1 OFF within 1e-6.

Probe measurements at ``2fae161`` (``spikes/gdn-core-verify/tp_probe.py``, fp32, seed 0): FUSED vs
OFF rows <= 6.0e-8, SSM <= 1.4e-9, conv exactly equal, on every rank; every rank's rows equal rank
0's exactly; rank 0 rows vs TP1 OFF <= 1.5e-7; the SSM concatenated over ranks on the value-head
axis vs TP1 OFF <= 1.4e-9.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from flashinfer_gaudi.gdn_core_verify import GdnCoreVerifyMode  # noqa: E402
from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES  # noqa: E402
from tests.unit_tests.models.flash_next.gdn_core_ranks import (  # noqa: E402  (adds spikes/cpu-tp to sys.path)
    gdn_verify_rank)

from harness import run_ranks  # noqa: E402

MODES = (GdnCoreVerifyMode.OFF, GdnCoreVerifyMode.FUSED)
TP_SIZES = (1, 2, 4, 8)
ROW_TOL = 1e-6
STATE_TOL = 1e-6

_RUN_CACHE: dict[int, list] = {}


def _ranks(tp: int, checkpoint: Path, k: int) -> list:
    """The results of ``gdn_verify_rank`` for ``k`` on ``tp`` gloo ranks; one run per TP size covers every size
    and is cached."""
    if tp not in _RUN_CACHE:
        _RUN_CACHE[tp] = run_ranks(tp, gdn_verify_rank, str(checkpoint), tp, tuple(int(m) for m in MODES),
                                   tuple(SPEC_SIZES), timeout_s=600)
    return [res[k] for res in _RUN_CACHE[tp]]


@pytest.mark.parametrize("tp", TP_SIZES)
def test_verify_op_called_per_step(tp, tiny_tp8_checkpoint, spec_k):
    """Mode OFF calls gdn_core_verify no time; FUSED calls it 3 times with T = [1 + k, 1 + k, 1] and passes its
    check."""
    results = _ranks(tp, tiny_tp8_checkpoint, spec_k)
    want = [1 + spec_k, 1 + spec_k, 1]
    for rank, res in enumerate(results):
        assert res[int(GdnCoreVerifyMode.OFF)]["calls"] == 0, (f"tp={tp} rank={rank}: OFF called gdn_core_verify "
                                                               f"{res[int(GdnCoreVerifyMode.OFF)]['calls']} times")
        fu = res[int(GdnCoreVerifyMode.FUSED)]
        assert fu["calls"] == 3, f"tp={tp} rank={rank}: FUSED called gdn_core_verify {fu['calls']} times"
        assert fu["ts"] == want, f"tp={tp} rank={rank}: FUSED call T values {fu['ts']} != {want}"
        assert fu["ok"] is True, f"tp={tp} rank={rank}: _gdn_core_verify_ok is not True"


@pytest.mark.parametrize("tp", TP_SIZES)
def test_verify_ranks_agree(tp, tiny_tp8_checkpoint, spec_k):
    """For every mode, every rank's rows equal rank 0's within ROW_TOL."""
    results = _ranks(tp, tiny_tp8_checkpoint, spec_k)
    for mode in MODES:
        for rank, res in enumerate(results):
            assert res[int(mode)]["rows"].shape == (2 * (1 + spec_k) + 1, 128)
            np.testing.assert_allclose(res[int(mode)]["rows"],
                                       results[0][int(mode)]["rows"],
                                       atol=ROW_TOL,
                                       rtol=0,
                                       err_msg=f"tp={tp} rank={rank} mode={int(mode)}: rows != rank 0")


@pytest.mark.parametrize("tp", TP_SIZES)
def test_verify_matches_off(tp, tiny_tp8_checkpoint, spec_k):
    """On every rank FUSED matches OFF: rows within ROW_TOL, SSM within STATE_TOL, conv bit-exact."""
    results = _ranks(tp, tiny_tp8_checkpoint, spec_k)
    for rank, res in enumerate(results):
        off = res[int(GdnCoreVerifyMode.OFF)]
        fu = res[int(GdnCoreVerifyMode.FUSED)]
        np.testing.assert_allclose(fu["rows"],
                                   off["rows"],
                                   atol=ROW_TOL,
                                   rtol=0,
                                   err_msg=f"tp={tp} rank={rank}: FUSED rows != OFF")
        np.testing.assert_allclose(fu["ssm"],
                                   off["ssm"],
                                   atol=STATE_TOL,
                                   rtol=0,
                                   err_msg=f"tp={tp} rank={rank}: FUSED SSM != OFF")
        np.testing.assert_array_equal(fu["conv"], off["conv"], err_msg=f"tp={tp} rank={rank}: FUSED conv != OFF")


@pytest.mark.parametrize("tp", TP_SIZES)
def test_verify_states_written(tp, tiny_tp8_checkpoint, spec_k):
    """On every rank and mode each of the 1 + k SSM slots and the slot 1 conv state (the lattice's row) is non-zero."""
    results = _ranks(tp, tiny_tp8_checkpoint, spec_k)
    for rank, res in enumerate(results):
        for mode in MODES:
            assert res[int(mode)]["ssm"].shape[0] == 1 + spec_k
            assert res[int(mode)]["conv"].shape[:2] == (1 + spec_k, 3 + spec_k)
            for slot in range(1 + spec_k):
                assert res[int(mode)]["ssm"][slot].any(), (f"tp={tp} rank={rank} mode={int(mode)}: SSM slot "
                                                           f"{slot + 1} is all zeros")
            assert res[int(mode)]["conv"][0].any(), (f"tp={tp} rank={rank} mode={int(mode)}: slot 1 conv is "
                                                     "all zeros")


@pytest.mark.parametrize("tp", (2, 4, 8))
def test_verify_matches_tp1(tp, tiny_tp8_checkpoint, spec_k):
    """For every mode, rank 0's rows match TP1 OFF within ROW_TOL and the concatenated SSM within STATE_TOL."""
    results = _ranks(tp, tiny_tp8_checkpoint, spec_k)
    tp1_off = _ranks(1, tiny_tp8_checkpoint, spec_k)[0][int(GdnCoreVerifyMode.OFF)]
    for mode in MODES:
        np.testing.assert_allclose(results[0][int(mode)]["rows"],
                                   tp1_off["rows"],
                                   atol=ROW_TOL,
                                   rtol=0,
                                   err_msg=f"tp={tp} rank=0 mode={int(mode)}: rows != TP1 OFF")
        full_ssm = np.concatenate([res[int(mode)]["ssm"] for res in results], axis=1)
        assert full_ssm.shape == (1 + spec_k, 48, 16, 16), f"tp={tp} mode={int(mode)}: SSM shape {full_ssm.shape}"
        np.testing.assert_allclose(full_ssm,
                                   tp1_off["ssm"],
                                   atol=STATE_TOL,
                                   rtol=0,
                                   err_msg=f"tp={tp} mode={int(mode)}: concatenated SSM != TP1 OFF")
