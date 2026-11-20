# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the shared fused hyper-connection test cases (``hc_case.py``).

Every case has expected outputs that pass ``compare``; the raw files round-trip byte-exactly;
``compare`` has teeth (two bf16 steps, a wrong copy, a missing output, a wrong shape, a kernel
that skips the oracle's rounding point); the cases cover what the contract lists; ``make_inputs``
is deterministic.  No vLLM, no toolchain.  Contract: ``docs/design/flash-next/hc-fused.md``.
"""

import pytest
import torch

from tests.unit_tests.hc_fused.hc_case import (ATOL, BF16, GUIDS, HPU_CASES, INPUT_ORDER, KINDS, OUTPUT_ORDER,
                                               ROUNDING_CASES, SIM_CASES, bf16_steps, case_by_name, compare, expected,
                                               input_shapes, kernel_params, make_inputs, output_shapes, read_outputs,
                                               rounding_discriminators, scalars, summary, write_inputs, write_outputs)
from tests.unit_tests.hc_fused.hc_reference import hc_combine_norm_unrounded, hc_norm_reference

SMALL = [case for case in SIM_CASES if case.rows <= 32]
SMALL_IDS = [case.name for case in SMALL]


def _step(t: torch.Tensor, steps: int) -> torch.Tensor:
    """`t` moved `steps` bf16 values away from zero (positive finite values only move up)."""
    return (t.view(torch.int16) + steps).view(BF16)


@pytest.mark.parametrize("case", SMALL, ids=SMALL_IDS)
def test_expected_passes_compare(case):
    inputs = make_inputs(case)
    exp = expected(case, inputs)
    assert compare(case, inputs, exp) == []
    assert all(torch.isfinite(t.float()).all() for t in exp.values())
    assert "fail=0" in summary(case, inputs, exp)


@pytest.mark.parametrize("case", SMALL, ids=SMALL_IDS)
def test_shapes_and_dtypes(case):
    inputs = make_inputs(case)
    assert tuple(inputs) == INPUT_ORDER[case.kind]
    assert {name: tuple(t.shape) for name, t in inputs.items()} == input_shapes(case)
    assert all(t.dtype == BF16 and t.is_contiguous() for t in inputs.values())
    exp = expected(case, inputs)
    assert tuple(exp) == OUTPUT_ORDER[case.kind]
    assert {name: tuple(t.shape) for name, t in exp.items()} == output_shapes(case)
    assert all(t.dtype == BF16 for t in exp.values())


@pytest.mark.parametrize("case", SMALL[::5], ids=SMALL_IDS[::5])
def test_raw_round_trip(case, tmp_path):
    inputs = make_inputs(case)
    write_inputs(case, inputs, tmp_path)
    for name in INPUT_ORDER[case.kind]:
        assert (tmp_path / f"{name}.bin").stat().st_size == 2 * inputs[name].numel()
    manifest = (tmp_path / "case.txt").read_text().splitlines()
    assert manifest[0] == f"guid {GUIDS[case.kind]}"
    assert sum(line.startswith("in ") for line in manifest) == len(INPUT_ORDER[case.kind])
    assert sum(line.startswith("out ") for line in manifest) == len(OUTPUT_ORDER[case.kind])
    exp = expected(case, inputs)
    write_outputs(case, exp, tmp_path)
    got = read_outputs(case, tmp_path)
    for name in OUTPUT_ORDER[case.kind]:
        assert torch.equal(got[name].view(torch.int16), exp[name].view(torch.int16))


def test_manifest_dims_are_fastest_first(tmp_path):
    case = case_by_name("combine_norm_real_r2_shared")
    write_inputs(case, make_inputs(case), tmp_path)
    lines = (tmp_path / "case.txt").read_text().splitlines()
    assert lines[:4] == ["guid hc_fused_combine_norm_bf16_gaudi2", "eps 1e-06", "hc_count 4", "lora_rank 0"]
    assert "in residual bf16 10240 2" in lines
    assert "in block_output bf16 2560 2" in lines
    assert "in injection bf16 4 2" in lines
    assert "in norm_weight bf16 2560" in lines
    assert lines[-2:] == ["out residual_out bf16 10240 2", "out xn bf16 10240 2"]


def test_make_inputs_is_deterministic():
    case = case_by_name("combine_norm_tiny_stress")
    a, b = make_inputs(case), make_inputs(case)
    assert all(torch.equal(a[name].view(torch.int16), b[name].view(torch.int16)) for name in a)


def test_bf16_steps():
    t = torch.tensor([1.0, -1.0, 0.0, 3.5], dtype=BF16)
    assert bf16_steps(t, t).tolist() == [0, 0, 0, 0]
    assert bf16_steps(_step(t, 1), t).tolist() == [1, 1, 1, 1]
    assert int(bf16_steps(torch.tensor([0.0], dtype=BF16), torch.tensor([-0.0], dtype=BF16))) == 0
    assert int(bf16_steps(torch.tensor([1.0], dtype=BF16), torch.tensor([1.0078125], dtype=BF16))) == 1


# ---- teeth ---------------------------------------------------------------------------------------
def test_tooth_one_step_passes_two_steps_fail():
    case = case_by_name("norm_tiny_r1")
    inputs = make_inputs(case)
    got = expected(case, inputs)
    big = got["xn"].float().abs() > 0.5  # far above ATOL, so only the step rule can pass them
    index = big.flatten().nonzero()[0]
    one = {"xn": got["xn"].clone()}
    one["xn"].view(-1)[index] = _step(got["xn"].view(-1)[index], 1)
    assert compare(case, inputs, one) == []
    two = {"xn": got["xn"].clone()}
    two["xn"].view(-1)[index] = _step(got["xn"].view(-1)[index], 2)
    assert any("more than 1 bf16 step" in m for m in compare(case, inputs, two))


def test_tooth_too_many_one_step_elements():
    case = case_by_name("gate_mix_real_r1")
    inputs = make_inputs(case)
    got = expected(case, inputs)
    got["block_input"][0, :100] = _step(got["block_input"][0, :100], 1)  # 3.9 % of the output
    assert any("bit-exact" in m for m in compare(case, inputs, got))


def test_tooth_gdn_tolerance_is_an_outer_bound():
    """An element may pass the cancellation rule and still be outside the documented GDN bf16 tolerance."""
    case = case_by_name("combine_real_stress")
    inputs = make_inputs(case)
    exp = expected(case, inputs)
    residual = inputs["residual"].float()
    index = int(residual.abs().flatten().argmax())
    scale = float(residual.flatten()[index].abs())
    assert scale > 1e12  # a 1e15-scale group: rule 3 alone would allow an error of about 1e-6 * scale
    got = {"residual_out": exp["residual_out"].clone()}
    wrong = exp["residual_out"].float().view(-1)[index] * 1.05
    got["residual_out"].view(-1)[index] = wrong.to(BF16)
    messages = compare(case, inputs, got)
    assert any("outside the GDN bf16 tolerance" in m for m in messages)


def test_tooth_cancellation_rule_is_bounded():
    """A cancelling element may be many bf16 steps off, but only by a millionth of its operands."""
    case = case_by_name("combine_tiny_r3")
    inputs = make_inputs(case)
    exp = expected(case, inputs)
    residual = inputs["residual"].float()
    index = int(residual.abs().flatten().argmax())
    scale = float(residual.flatten()[index].abs())
    assert scale > 2.0
    got = {"residual_out": exp["residual_out"].clone()}
    got["residual_out"].view(-1)[index] = (exp["residual_out"].float().view(-1)[index] + 0.25 * scale).to(BF16)
    assert any("more than 1 bf16 step" in m for m in compare(case, inputs, got))


def test_tooth_propagation_rule_needs_a_moved_combine_element():
    """xn two steps off fails when its residual_out element is the oracle's; it passes when that one moved a step."""
    case = case_by_name("combine_norm_real_r1")
    inputs = make_inputs(case)
    exp = expected(case, inputs)
    # An element whose xn moves two steps when its combine result moves one: small mantissa in out, large in xn.
    out_bits = exp["residual_out"].view(torch.int16).to(torch.int32) & 0x7F
    xn_bits = exp["xn"].view(torch.int16).to(torch.int32) & 0x7F
    candidates = ((out_bits < 16) & (xn_bits > 100) & (exp["xn"].float().abs() > 0.5)).flatten().nonzero()
    assert candidates.numel() > 0
    index = int(candidates[0])

    two_steps = {name: tensor.clone() for name, tensor in exp.items()}
    two_steps["xn"].view(-1)[index] = _step(exp["xn"].view(-1)[index], 2)
    messages = compare(case, inputs, two_steps)
    assert any("xn: 1 of" in m and "more than 1 bf16 step" in m for m in messages)

    moved = {name: tensor.clone() for name, tensor in exp.items()}
    moved["residual_out"].view(-1)[index] = _step(exp["residual_out"].view(-1)[index], 1)
    own = hc_norm_reference(moved["residual_out"], inputs["norm_weight"], case.eps, case.hc)
    assert int(bf16_steps(own, exp["xn"]).view(-1)[index]) == 2  # the case this rule exists for
    moved["xn"] = own
    assert compare(case, inputs, moved) == []

    # The oracle's xn next to a moved combine element is not the norm of the kernel's own residual_out.
    stale = {name: tensor.clone() for name, tensor in exp.items()}
    stale["residual_out"].view(-1)[index] = _step(exp["residual_out"].view(-1)[index], 1)
    assert any("not the norm of its own residual_out" in m for m in compare(case, inputs, stale))


def test_tooth_injection_copy_must_be_the_same_bits():
    case = case_by_name("silu_split_tiny_r3")
    inputs = make_inputs(case)
    got = expected(case, inputs)
    got["injection_out"][1, 2] = _step(got["injection_out"][1, 2], 1)
    assert any("not the same bits" in m for m in compare(case, inputs, got))


def test_tooth_edge_cases_must_be_exact():
    for name in ("gate_mix_tiny_edge", "combine_tiny_edge"):
        case = case_by_name(name)
        assert case.exact
        inputs = make_inputs(case)
        got = expected(case, inputs)
        key = OUTPUT_ORDER[case.kind][0]
        got[key][0, 5] = _step(got[key][0, 5], 1)
        assert any("not exact" in m for m in compare(case, inputs, got))


def test_tooth_missing_output_and_wrong_shape():
    case = case_by_name("combine_norm_tiny_r1")
    inputs = make_inputs(case)
    got = expected(case, inputs)
    assert any("xn: missing" in m for m in compare(case, inputs, {"residual_out": got["residual_out"]}))
    got["xn"] = got["xn"].float()
    assert any("xn: must be" in m for m in compare(case, inputs, got))


def test_tooth_unwritten_output_is_caught():
    """sim_host fills an output with 0xC3C3 before the run; an element the kernel skipped must fail."""
    case = case_by_name("norm_tiny_r3_shared")
    inputs = make_inputs(case)
    got = expected(case, inputs)
    got["xn"].view(torch.int16)[2, 300:428] = torch.tensor(0xC3C3 - 0x10000, dtype=torch.int16)
    assert compare(case, inputs, got) != []


# ---- the oracle's rounding point -----------------------------------------------------------------
@pytest.mark.parametrize("case", ROUNDING_CASES, ids=[case.name for case in ROUNDING_CASES])
def test_rounding_cases_tell_rounded_from_unrounded(case):
    inputs = make_inputs(case)
    mask = rounding_discriminators(case, inputs)
    assert int(mask.sum()) >= 100
    assert float(mask.float().mean()) > 0.25
    # The combine itself loses the block contribution on the even columns and has none on the odd ones.
    exp = expected(case, inputs)
    assert torch.equal(exp["residual_out"].view(torch.int16), inputs["residual"].view(torch.int16))
    assert float(inputs["block_output"].float().abs().max()) > 0


@pytest.mark.parametrize("case", ROUNDING_CASES, ids=[case.name for case in ROUNDING_CASES])
def test_tooth_norm_of_the_unrounded_combine_fails(case):
    """A kernel that normalizes the fp32 combine without rounding it to bf16 first is rejected."""
    inputs = make_inputs(case)
    got = expected(case, inputs)
    got["xn"] = hc_combine_norm_unrounded(inputs["residual"], inputs["block_output"], inputs["injection"],
                                          inputs["norm_weight"], case.eps, case.hc)
    messages = compare(case, inputs, got)
    assert any("does not round the combine before the norm" in m for m in messages)


def test_rounding_pin_also_holds_on_an_ordinary_case():
    """On random inputs, too, the unrounded norm differs from the oracle on far more than 1 % of the elements."""
    case = case_by_name("combine_norm_real_r2_shared")
    inputs = make_inputs(case)
    got = expected(case, inputs)
    got["xn"] = hc_combine_norm_unrounded(inputs["residual"], inputs["block_output"], inputs["injection"],
                                          inputs["norm_weight"], case.eps, case.hc)
    assert any("bit-exact" in m for m in compare(case, inputs, got))


# ---- coverage of the contract --------------------------------------------------------------------
def test_cases_cover_the_contract():
    names = [case.name for case in HPU_CASES]
    assert len(names) == len(set(names))
    for kind in KINDS:
        cases = [case for case in SIM_CASES if case.kind == kind]
        assert cases, kind
        if kind == "silu":
            continue  # the final mixer's op: tiny, real, stress
        tiny_rows = {case.rows for case in cases if case.hidden == 128 and case.flavor == "random" and case.hc == 4}
        real_rows = {case.rows for case in cases if case.hidden == 2560 and case.flavor == "random"}
        assert {1, 3} <= tiny_rows, kind
        assert {1, 2, 16, 32, 2048} <= real_rows, kind
        assert any(case.flavor == "stress" for case in cases), kind
    for kind in ("norm", "combine_norm"):
        cases = [case for case in SIM_CASES if case.kind == kind]
        assert any(case.shared_weight for case in cases) and any(not case.shared_weight for case in cases)
    assert {case.kind for case in SIM_CASES if case.flavor == "edge"} == {"gate_mix", "combine"}
    assert len(ROUNDING_CASES) == 2
    assert any(case.hc != 4 for case in SIM_CASES)
    decode_buckets = {1, 2, 4, 8, 12, 16, 20, 32}
    for kind in ("norm", "silu_split", "gate_mix", "combine", "combine_norm"):
        rows = {case.rows for case in HPU_CASES if case.kind == kind and case.hidden == 2560}
        assert decode_buckets <= rows and 8192 in rows, kind


def test_stress_inputs_hold_the_values_they_claim():
    gate = make_inputs(case_by_name("gate_mix_real_stress"))["gate"].float()
    assert float(gate.max()) >= 1e30 and float(gate.min()) <= -1e30 and bool((gate.abs() == 100).any())
    x = make_inputs(case_by_name("norm_tiny_stress"))["x"].float().reshape(3, 4, 128)
    mean_square = x.square().mean(dim=-1)
    assert bool((mean_square == 0).any())  # rsqrt(eps) alone
    assert bool(((mean_square > 0) & (mean_square < 1e-7)).any())  # eps dominates
    assert bool((mean_square > 1e20).any())
    assert ATOL == 1e-6


def test_scalars_and_kernel_params():
    assert scalars(case_by_name("norm_tiny_r1")) == (1e-6, 4)
    assert kernel_params(case_by_name("norm_tiny_r1")) == (1e-6, 4, 0)
    assert scalars(case_by_name("silu_split_real_r1")) == (4, 320)
    assert kernel_params(case_by_name("silu_split_real_r1")) == (0.0, 4, 320)
    assert kernel_params(case_by_name("gate_mix_hc3_h256_r2")) == (0.0, 3, 0)
    assert kernel_params(case_by_name("combine_norm_tiny_r1")) == (1e-6, 4, 0)
