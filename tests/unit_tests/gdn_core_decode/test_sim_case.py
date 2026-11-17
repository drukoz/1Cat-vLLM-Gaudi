# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the shared GDN core decode kernel test cases (``sim_case.py``).

Every case of ``HPU_CASES`` has expected outputs that pass ``compare``; the raw-file inputs and
outputs round-trip byte-exactly; six teeth on ``tp8_mixed`` prove ``compare`` has teeth;
``make_inputs`` is deterministic.  Contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

import pytest
import torch

from tests.unit_tests.gdn_core_decode.sim_case import (ARG_ORDER, CASES, HPU_CASES, compare, expected, make_inputs,
                                                       read_outputs, write_inputs, write_outputs)

CASE_IDS = [case.name for case in HPU_CASES]
TOOTH = CASES[0]  # tp8_mixed: loads (0, 1, -1), stores (0, 2, -1): slot 1 read-only, slot 3 unnamed


@pytest.mark.parametrize("case", HPU_CASES, ids=CASE_IDS)
def test_expected_passes_compare(case):
    inp = make_inputs(case)
    assert compare(case, inp, expected(case, inp)) == []


@pytest.mark.parametrize("case", HPU_CASES, ids=CASE_IDS)
def test_raw_round_trip(case, tmp_path):
    inp = make_inputs(case)
    write_inputs(case, inp, tmp_path)
    for name in ARG_ORDER:
        path = tmp_path / f"{name}.bin"
        assert path.stat().st_size == inp[name].numel() * inp[name].element_size()
    exp = expected(case, inp)
    write_outputs(exp, tmp_path)
    got = read_outputs(case, tmp_path)
    for key in ("out", "conv_state", "ssm_state"):
        assert torch.equal(got[key], exp[key])


def test_tooth_out_row0():
    inp = make_inputs(TOOTH)
    got = expected(TOOTH, inp)
    got["out"][0, 0, 0] += 1e-2
    assert any("out row 0" in m for m in compare(TOOTH, inp, got))


def test_tooth_padded_row2():
    inp = make_inputs(TOOTH)
    got = expected(TOOTH, inp)
    got["out"][2] = 1
    assert any("padded row 2" in m for m in compare(TOOTH, inp, got))


def test_tooth_untouched_slot3():
    inp = make_inputs(TOOTH)
    got = expected(TOOTH, inp)
    got["ssm_state"][3] += 1e-3
    assert any("untouched slot 3" in m for m in compare(TOOTH, inp, got))


def test_tooth_untouched_slot1():
    inp = make_inputs(TOOTH)
    got = expected(TOOTH, inp)
    got["ssm_state"][1] += 1e-3
    assert any("untouched slot 1" in m for m in compare(TOOTH, inp, got))


def test_tooth_conv_state():
    inp = make_inputs(TOOTH)
    got = expected(TOOTH, inp)
    got["conv_state"][2, 0, 0] += 1
    assert any("conv_state" in m for m in compare(TOOTH, inp, got))


def test_tooth_ssm_slot2():
    inp = make_inputs(TOOTH)
    got = expected(TOOTH, inp)
    got["ssm_state"][2] += 1e-4
    assert any("ssm_state slot 2" in m for m in compare(TOOTH, inp, got))


def test_make_inputs_deterministic():
    for case in HPU_CASES:
        x = make_inputs(case)
        y = make_inputs(case)
        for name in ARG_ORDER:
            assert torch.equal(x[name], y[name])
