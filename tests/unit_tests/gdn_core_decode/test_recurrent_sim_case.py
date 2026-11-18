# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the shared recurrence-only kernel test cases (``recurrent_sim_case.py``).

Every case of ``RECURRENT_HPU_CASES`` has expected outputs that pass ``compare_recurrent``; the
raw-file inputs and outputs round-trip byte-exactly; six teeth on ``rec_tp8_mixed`` prove
``compare_recurrent`` has teeth; ``make_recurrent_inputs`` is deterministic; every case respects
the kernel's slot rules.  Contract: ``docs/design/flash-next/gdn-core-decode.md``,
"Recurrence-only op".
"""

import pytest
import torch

from flashinfer_gaudi.gdn_core_decode import check_gdn_core_decode_indices
from tests.unit_tests.gdn_core_decode.recurrent_sim_case import (RECURRENT_ARG_ORDER, RECURRENT_CASES,
                                                                 RECURRENT_HPU_CASES, compare_recurrent,
                                                                 expected_recurrent, make_recurrent_inputs,
                                                                 read_recurrent_outputs, recurrent_scalars,
                                                                 write_recurrent_inputs, write_recurrent_outputs)

CASE_IDS = [case.name for case in RECURRENT_HPU_CASES]
TOOTH = RECURRENT_CASES[0]  # rec_tp8_mixed: loads (0, 1, -1), stores (0, 2, -1): slot 1 read-only, slot 3 unnamed


@pytest.mark.parametrize("case", RECURRENT_HPU_CASES, ids=CASE_IDS)
def test_expected_passes_compare(case):
    inp = make_recurrent_inputs(case)
    assert compare_recurrent(case, inp, expected_recurrent(case, inp)) == []


@pytest.mark.parametrize("case", RECURRENT_HPU_CASES, ids=CASE_IDS)
def test_raw_round_trip(case, tmp_path):
    inp = make_recurrent_inputs(case)
    write_recurrent_inputs(case, inp, tmp_path)
    for name in RECURRENT_ARG_ORDER:
        path = tmp_path / f"{name}.bin"
        assert path.stat().st_size == inp[name].numel() * inp[name].element_size()
    assert (tmp_path / "params.txt").read_text() == \
        f"{case.S} {case.H} {case.HV} {len(case.load)} {recurrent_scalars(case)[0]!r}\n"
    exp = expected_recurrent(case, inp)
    write_recurrent_outputs(exp, tmp_path)
    got = read_recurrent_outputs(case, tmp_path)
    for key in ("out", "ssm_state"):
        assert torch.equal(got[key], exp[key])


def test_tooth_out_row0():
    inp = make_recurrent_inputs(TOOTH)
    got = expected_recurrent(TOOTH, inp)
    got["out"][0, 0, 0] += 1e-2
    assert any("out row 0" in m for m in compare_recurrent(TOOTH, inp, got))


def test_tooth_padded_row2():
    inp = make_recurrent_inputs(TOOTH)
    got = expected_recurrent(TOOTH, inp)
    got["out"][2] = 1
    assert any("padded row 2" in m for m in compare_recurrent(TOOTH, inp, got))


def test_tooth_untouched_slot3():
    inp = make_recurrent_inputs(TOOTH)
    got = expected_recurrent(TOOTH, inp)
    got["ssm_state"][3] += 1e-3
    assert any("untouched slot 3" in m for m in compare_recurrent(TOOTH, inp, got))


def test_tooth_untouched_slot1():
    inp = make_recurrent_inputs(TOOTH)
    got = expected_recurrent(TOOTH, inp)
    got["ssm_state"][1] += 1e-3
    assert any("untouched slot 1" in m for m in compare_recurrent(TOOTH, inp, got))


def test_tooth_ssm_slot2():
    inp = make_recurrent_inputs(TOOTH)
    got = expected_recurrent(TOOTH, inp)
    got["ssm_state"][2] += 1e-4
    assert any("ssm_state slot 2" in m for m in compare_recurrent(TOOTH, inp, got))


def test_tooth_ssm_slot0_in_place():
    inp = make_recurrent_inputs(TOOTH)
    got = expected_recurrent(TOOTH, inp)
    got["ssm_state"][0] += 1e-4
    assert any("ssm_state slot 0" in m for m in compare_recurrent(TOOTH, inp, got))


def test_make_inputs_deterministic():
    for case in RECURRENT_HPU_CASES:
        x = make_recurrent_inputs(case)
        y = make_recurrent_inputs(case)
        for name in RECURRENT_ARG_ORDER:
            assert torch.equal(x[name], y[name])


def test_sim_cases_have_in_place_and_moved_rows():
    # The kernels write state in place and tpc-clang may reorder loads and stores of one tensor,
    # so the simulator must see both an in-place and a moved (or padding) row.
    for case in RECURRENT_CASES:
        valid = [(ls, ss) for ls, ss in zip(case.load, case.store) if 0 <= ls < case.S and 0 <= ss < case.S]
        assert any(ls == ss for ls, ss in valid), case.name
        assert any(ls != ss for ls, ss in zip(case.load, case.store)), case.name


def test_cases_respect_slot_rules():
    for case in RECURRENT_HPU_CASES:
        check_gdn_core_decode_indices(torch.tensor(case.load), torch.tensor(case.store), case.S, len(case.load))
