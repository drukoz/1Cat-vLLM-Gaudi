# SPDX-License-Identifier: Apache-2.0
"""The card's bf16 scalar division and what it means for ``moe_row_quant`` (CPU; no vLLM, no toolchain).

The first card run of the MoE glue kernels (Gaudi2, 2026-11-20) failed
``test_row_quant_is_the_plugins_dynamic_quant_on_the_card`` with

    MOEHPU row_quant_real_r32 against dynamic_quant on the card: 22 of 32 scales and 3378 of 81920 fp8
    values differ (max abs diff 1.600e+01)

although the kernel returned the CPU oracle's bits in the same run.  The card does not compute
``bf16_tensor / 240.0`` the way CPU torch does (``hpu_arithmetic.py``).  These tests pin that rule with the
card's own numbers, and pin what follows from it: the kernel replaces the torch path *on a card*, so its
oracle is the torch code with the card's division, while the op's CPU stand-in keeps the CPU's.
"""

import pytest
import torch

from tests.unit_tests.hc_fused import hpu_arithmetic, moe_ops, moe_reference
from tests.unit_tests.hc_fused.moe_case import BF16, SIM_CASES, case_by_name, every_finite_bf16, expected, make_inputs

ROW_QUANT = [case for case in SIM_CASES if case.kind == "row_quant"]
FP8_MAX = moe_reference.GAUDI2_FP8_MAX
# The card log of 2026-11-20 for row_quant_real_r32: differing scales (of 32), differing fp8 values (of
# 81920), largest difference.  The same line in both card runs of that day.
CARD_LOG = (22, 3378, 16.0)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()])


def _cast(data, scale_inv, *_):
    return moe_reference.cast_to_fp8_reference(data, scale_inv, FP8_MAX)


def _torch_path_on_cpu(x):
    """The plugin's torch code (the line-for-line copy) with CPU torch's arithmetic."""
    return moe_reference.moe_row_quant_torch(x, FP8_MAX, _cast)


def _torch_path_on_card(x):
    """The same lines with the card's scalar division."""
    with hpu_arithmetic.card_arithmetic():
        return moe_reference.moe_row_quant_torch(x, FP8_MAX, _cast)


def _differences(first, second) -> tuple[int, int, float]:
    """What the card test prints: differing scales, differing fp8 values, the largest difference."""
    (first_fp8, first_scale), (second_fp8, second_scale) = first, second
    have, want = first_fp8.float(), second_fp8.float()
    return int((first_scale != second_scale).sum()), int((have != want).sum()), float((have - want).abs().max())


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
def test_the_reciprocal_of_240_in_bf16():
    reciprocal = torch.tensor(1.0 / 240.0, dtype=BF16)
    assert float(reciprocal) == float.fromhex("0x1.12p-8")
    assert 1.0034 < float(reciprocal) * 240.0 < 1.0035  # 0.34 % above 1 / 240


def test_a_bf16_tensor_divided_by_a_python_number_is_one_product_with_the_rounded_reciprocal():
    x = every_finite_bf16()
    with hpu_arithmetic.card_arithmetic():
        card = x / 240.0
    assert card.dtype == BF16
    # one multiplication in fp32 (exact: two 8-bit mantissas), one rounding to bf16
    product = (x.float() * float.fromhex("0x1.12p-8")).to(BF16)
    assert torch.equal(_bits(card), _bits(product))
    assert torch.equal(_bits(card), _bits(moe_reference.hpu_div_by_scalar(x, 240.0)))
    # CPU torch divides, and the card's result is that quotient or one bf16 step away from zero
    cpu = x / 240.0
    assert torch.equal(_bits(cpu), _bits((x.float() / 240.0).to(BF16)))
    steps = (_bits(card).to(torch.int32) & 0x7FFF) - (_bits(cpu).to(torch.int32) & 0x7FFF)
    assert set(steps.unique().tolist()) == {0, 1}
    assert 0.6 < float((steps == 1).float().mean()) < 0.8


def test_the_emulation_changes_nothing_else():
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(64, 32, generator=g) * 5.0).to(BF16)
    b = (torch.rand(64, 32, generator=g) + 0.5).to(BF16)
    plain = (a / b, 1.0 / b, a * 240.0, a + 1e-8, a.float() / 240.0, a.float().double() / 240.0)
    with hpu_arithmetic.card_arithmetic():
        inside = (a / b, 1.0 / b, a * 240.0, a + 1e-8, a.float() / 240.0, a.float().double() / 240.0)
    for first, second in zip(plain, inside):
        assert first.dtype == second.dtype and torch.equal(first, second)
    assert not torch.equal(_bits(a / 240.0), _bits(moe_reference.hpu_div_by_scalar(a, 240.0)))  # outside: CPU torch
    assert torch.equal(_bits(a / 240.0), _bits((a.float() / 240.0).to(BF16)))


# ---------------------------------------------------------------------------
# The card log
# ---------------------------------------------------------------------------
def test_the_card_log_of_the_row_quantization_is_reproduced_on_cpu():
    """dynamic_quant's formula with the card's division against the kernel of that build (CPU arithmetic)."""
    x = make_inputs(case_by_name("row_quant_real_r32"))["x"]
    assert _differences(_torch_path_on_cpu(x), _torch_path_on_card(x)) == CARD_LOG


@pytest.mark.parametrize("constant,counts", [(float.fromhex("0x1.10p-8"), (19, 3336)),
                                             (float.fromhex("0x1.14p-8"), (32, 10988))])
def test_a_neighbouring_constant_does_not_give_the_card_log(constant, counts):
    """The counts pin the constant: one bf16 step below or above bf16(1 / 240) gives other numbers."""
    x = make_inputs(case_by_name("row_quant_real_r32"))["x"]
    shifted = (torch.abs(x)).max(dim=-1).values + 1e-8
    scale = (shifted * torch.tensor(constant, dtype=BF16)).unsqueeze(-1)
    other = (_cast(x, 1.0 / scale)[0], scale.float())
    assert _differences(_torch_path_on_cpu(x), other)[:2] == counts != CARD_LOG[:2]


def test_the_cards_scale_is_the_cpu_scale_or_one_step_above_for_every_row_maximum():
    x = make_inputs(case_by_name("row_quant_every_amax"))["x"]
    cpu, card = _torch_path_on_cpu(x)[1].to(BF16), _torch_path_on_card(x)[1].to(BF16)
    steps = _bits(card).to(torch.int32) - _bits(cpu).to(torch.int32)
    assert {int(step): int((steps == step).sum()) for step in steps.unique()} == {0: 8018, 1: 24621}


# ---------------------------------------------------------------------------
# What the kernel and the stand-in are
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ROW_QUANT, ids=[case.name for case in ROW_QUANT])
def test_the_kernel_oracle_is_the_torch_path_with_the_cards_division(case):
    """moe_row_quant replaces dynamic_quant on a card: its oracle has the card's scales, bit for bit."""
    inputs = make_inputs(case)
    want_fp8, want_scale = _torch_path_on_card(inputs["x"])
    exp = expected(case, inputs)
    differs = int((exp["scale"] != want_scale).sum())
    assert differs == 0, f"{differs} of {case.rows} scales of the oracle are not the card's"
    assert torch.equal(_bits(exp["x_fp8"]), _bits(want_fp8))


@pytest.mark.parametrize("case", [case for case in ROW_QUANT if case.rows <= 2048],
                         ids=[case.name for case in ROW_QUANT if case.rows <= 2048])
def test_the_cpu_stand_in_keeps_the_cpu_arithmetic(case):
    """On a CPU the op stands in for the torch path on a CPU: flag 1 equals flag 0 bit for bit there."""
    x = make_inputs(case)["x"]
    want_fp8, want_scale = _torch_path_on_cpu(x)
    got_fp8, got_scale = moe_ops.moe_row_quant(x, FP8_MAX)
    assert torch.equal(got_scale, want_scale) and torch.equal(_bits(got_fp8), _bits(want_fp8))
