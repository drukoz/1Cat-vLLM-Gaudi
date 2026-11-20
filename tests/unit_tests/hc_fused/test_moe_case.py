# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MoE glue case module: the comparison has teeth, the raw files round-trip.

No vLLM, no toolchain.  ``moe_case.compare`` is what decides whether a kernel passes on the simulator
and on a card, so every rule of it is exercised here against the oracle and against tampered results.
"""

import pytest
import torch

from tests.unit_tests.hc_fused import moe_case as mc
from tests.unit_tests.hc_fused.moe_case import BF16, SIM_CASES, case_by_name, compare, expected, make_inputs

FAST = [case for case in SIM_CASES if case.rows <= 2048]


def _kernel_answer(case, inputs):
    """What a correct kernel returns: the oracle, with the kernel's own choice among equal logits."""
    answer = dict(expected(case, inputs))
    if case.kind == "route":
        answer["ids"] = mc.route_kernel_ids(case, inputs)
        answer["weights"] = mc.route_expected_weights(case, inputs, answer["ids"])
    return answer


def _step(tensor: torch.Tensor, index, steps: int = 1) -> torch.Tensor:
    """A copy of the bf16 tensor with one element moved `steps` bf16 values up."""
    moved = tensor.clone()
    bits = moved.view(torch.int16)
    bits[index] = bits[index] + steps if int(bits[index]) >= 0 else bits[index] - steps
    return moved


@pytest.mark.parametrize("case", FAST, ids=[case.name for case in FAST])
def test_a_correct_kernel_passes(case):
    inputs = make_inputs(case)
    answer = _kernel_answer(case, inputs)
    assert compare(case, inputs, answer) == []
    assert case.name in mc.summary(case, inputs, answer)


@pytest.mark.parametrize("case", FAST, ids=[case.name for case in FAST])
def test_inputs_are_deterministic_and_have_the_declared_shapes(case):
    first, second = make_inputs(case), make_inputs(case)
    assert {name: tuple(t.shape) for name, t in first.items()} == mc.input_shapes(case)
    for name in first:
        assert first[name].dtype == BF16 and torch.equal(first[name].view(torch.int16), second[name].view(torch.int16))
    exp = expected(case, first)
    assert {name: tuple(t.shape) for name, t in exp.items()} == mc.output_shapes(case)
    assert all(exp[name].dtype == mc.OUTPUT_DTYPE[name] for name in exp)


def test_the_cases_cover_the_geometries_and_the_adversarial_inputs():
    names = {case.name for case in SIM_CASES}
    for kind in mc.KINDS:
        rows = {case.rows for case in SIM_CASES if case.kind == kind and case.flavor == "random"}
        assert {1, 2, 16, 32, 2048} <= rows, (kind, rows)
    route = [case for case in SIM_CASES if case.kind == "route"]
    assert any(case.experts == 512 and case.top_k == 10 for case in route)
    assert any(case.experts == 16 and case.top_k == 2 for case in route)
    assert {"ties", "equal", "huge", "negative", "spread", "small"} <= {case.flavor for case in route}
    assert {"row_quant_real_stress", "row_quant_real_edge", "row_quant_every_amax", "silu_mul_every_gate",
            "combine_every_gate", "silu_mul_real_edge", "combine_real_edge"} <= names
    # the adversarial routing cases really tie at the cut, and the stress rows of the quantization are zero rows
    for name in ("route_real_ties", "route_real_equal", "route_real_huge", "route_real_negative"):
        case = case_by_name(name)
        assert bool(mc.route_tie_rows(case, make_inputs(case)).all()), name
    stress = make_inputs(case_by_name("row_quant_real_stress"))["x"]
    assert bool((stress[0] == 0).all()) and bool((stress[8] == 0).all())
    every = make_inputs(case_by_name("row_quant_every_amax"))["x"]
    assert every.abs().max(dim=1).values.view(torch.int16).unique().numel() == 32639  # every finite positive bf16


def test_torch_topk_itself_passes_where_no_row_ties_at_the_cut():
    """On rows without a tie at the cut the kernel's rule and torch.topk select the same set."""
    case = case_by_name("route_real_r2")
    inputs = make_inputs(case)
    assert not bool(mc.route_tie_rows(case, inputs).any())
    exp = expected(case, inputs)
    kernel_ids = mc.route_kernel_ids(case, inputs)
    assert torch.equal(torch.sort(exp["ids"], dim=1).values, torch.sort(kernel_ids, dim=1).values)


def test_route_rejects_wrong_selections():
    case = case_by_name("route_real_r32")
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    logits = inputs["gates"][:, :case.experts].float()

    wrong = {name: tensor.clone() for name, tensor in good.items()}
    wrong["ids"][3, 9] = int(torch.argmin(logits[3]))  # an expert that is not among the largest
    assert any("largest probabilities" in message for message in compare(case, inputs, wrong))

    twice = {name: tensor.clone() for name, tensor in good.items()}
    twice["ids"][5, 1] = twice["ids"][5, 0]
    assert any("twice" in message for message in compare(case, inputs, twice))

    outside = {name: tensor.clone() for name, tensor in good.items()}
    outside["ids"][0, 0] = case.experts  # the shared-gate column
    assert any("outside" in message for message in compare(case, inputs, outside))

    swapped = {name: tensor.clone() for name, tensor in good.items()}
    swapped["ids"][7, 0], swapped["ids"][7, 1] = good["ids"][7, 1], good["ids"][7, 0]
    swapped["weights"][7, 0], swapped["weights"][7, 1] = good["weights"][7, 1], good["weights"][7, 0]
    assert any("descending order" in message for message in compare(case, inputs, swapped))


def test_route_pins_the_lowest_id_among_equal_logits():
    """A valid top-K that takes another expert of a tie at the cut is not the kernel's answer."""
    case = case_by_name("route_real_equal")
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    assert torch.equal(good["ids"][0], torch.arange(case.top_k, dtype=torch.int32))
    other = {name: tensor.clone() for name, tensor in good.items()}
    other["ids"][0, 9] = 400  # the same logit, the same probability, a higher id
    messages = compare(case, inputs, other)
    assert len(messages) == 1 and "lowest expert id first among equals" in messages[0]


def test_route_weight_tolerance_is_one_step():
    case = case_by_name("route_prefill_r2048")
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    one = dict(good, weights=_step(good["weights"], (100, 4), 1))
    assert compare(case, inputs, one) == []
    two = dict(good, weights=_step(good["weights"], (100, 4), 2))
    assert any("outside the tolerance" in message for message in compare(case, inputs, two))
    many = good["weights"].clone()
    many.view(torch.int16)[:300, 0] += 1  # 300 of 20480 one step away: more than 1 %
    assert any("differ from the oracle" in message for message in compare(case, inputs, dict(good, weights=many)))


def test_route_weights_below_the_normal_range_count_as_equal():
    case = case_by_name("route_real_spread")
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    tiny = good["weights"].float().abs() < mc.FLT_MIN
    flushed = torch.where(tiny, torch.zeros_like(good["weights"]), good["weights"])
    assert compare(case, inputs, dict(good, weights=flushed)) == []


@pytest.mark.parametrize("name,output", [("silu_mul_real_tp8_r32", "act"), ("combine_real_r32", "out")])
def test_bf16_outputs_allow_one_step_and_no_more(name, output):
    case = case_by_name(name)
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    index = (3, 17)
    assert abs(float(good[output][index])) > 1e-3
    assert compare(case, inputs, {output: _step(good[output], index, 1)}) == []
    messages = compare(case, inputs, {output: _step(good[output], index, 3)})
    assert len(messages) == 1 and "outside the tolerance" in messages[0]


def test_silu_mul_accepts_the_product_of_a_neighbouring_silu():
    """A rounded silu one bf16 step away moves the product by up to two steps: that passes, a third does not."""
    case = case_by_name("silu_mul_real_tp8_r32")
    inputs = make_inputs(case)
    gate_up = inputs["gate_up"]
    d = case.hidden
    silu = torch.nn.functional.silu(gate_up[:, :d])
    below, above = mc.bf16_neighbours(silu)
    assert torch.equal(mc.bf16_steps(below, silu), torch.ones_like(silu, dtype=torch.int32))
    assert torch.equal(mc.bf16_steps(above, silu), torch.ones_like(silu, dtype=torch.int32))
    want = expected(case, inputs)["act"]
    product = above * gate_up[:, d:]
    two_steps = (mc.bf16_steps(product, want) == 2).nonzero()[:5]  # a few elements the moved silu moves twice
    assert len(two_steps) == 5
    moved = want.clone()
    for row, column in two_steps.tolist():
        moved[row, column] = product[row, column]
    assert compare(case, inputs, {"act": moved}) == []
    # a third step is not the product of a neighbouring silu (one direction is the third step, the other the first)
    row, column = two_steps[0].tolist()
    further = [any("outside the tolerance" in message
                   for message in compare(case, inputs, {"act": _step(moved, (row, column), steps)}))
               for steps in (1, -1)]
    assert sorted(further) == [False, True]
    # a whole row moved that way is allowed element by element but is more than 1 % of the output
    moved[2] = product[2]
    messages = compare(case, inputs, {"act": moved})
    assert len(messages) == 1 and "differ from the oracle" in messages[0]


def test_combine_accepts_a_row_computed_with_a_neighbouring_gate_weight():
    case = case_by_name("combine_real_r2")
    inputs = make_inputs(case)
    weight = torch.sigmoid(inputs["gates"][:, mc.gate_column(case):])
    _, above = mc.bf16_neighbours(weight)
    row = (above * inputs["shared"] + inputs["routed"])[1]
    moved = expected(case, inputs)["out"].clone()
    moved[1] = row
    assert int((moved != expected(case, inputs)["out"]).sum()) > 26  # more than 1 % of one row
    messages = compare(case, inputs, {"out": moved})
    assert messages and all("differ from the oracle" in message for message in messages)  # allowed per element...
    assert not any("outside the tolerance" in message for message in messages)  # ...but counted


def test_edge_cases_must_be_exact():
    for name, output in (("silu_mul_real_edge", "act"), ("combine_real_edge", "out")):
        case = case_by_name(name)
        inputs = make_inputs(case)
        good = _kernel_answer(case, inputs)
        messages = compare(case, inputs, {output: _step(good[output], (1, 5), 1)})
        assert len(messages) == 1 and "not exact" in messages[0]


def test_edge_cases_hold_exact_ties():
    """The edge inputs really sit on rounding boundaries: rounding the other way changes many elements."""
    case = case_by_name("combine_real_edge")
    inputs = make_inputs(case)
    exact = inputs["shared"].double() + inputs["routed"].double()  # the gate weight is 1
    rounded = exact.float().to(BF16)
    below, above = mc.bf16_neighbours(rounded)
    distance = (exact - rounded.double()).abs()
    to_below, to_above = (exact - below.double()).abs(), (exact - above.double()).abs()
    ties = (distance > 0) & ((distance == to_below) | (distance == to_above))
    assert int(ties.sum()) > 1000
    case = case_by_name("row_quant_real_edge")
    x = make_inputs(case)["x"].float()
    cast = x.to(torch.float8_e4m3fn).float()
    assert float(((cast - x).abs() > 0).float().mean()) > 0.9  # almost every element is between two fp8 values
    assert bool((x.abs().max(dim=1).values == 240.0).all())


def test_row_quant_is_exact_or_fails():
    case = case_by_name("row_quant_real_r16")
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    scale = good["scale"].clone()
    scale[3, 0] = torch.nextafter(scale[3, 0], torch.tensor(1.0))
    assert any("scale" in message for message in compare(case, inputs, dict(good, scale=scale)))
    data = good["x_fp8"].clone()
    data.view(torch.uint8)[2, 7] ^= 1
    messages = compare(case, inputs, dict(good, x_fp8=data))
    assert len(messages) == 1 and "x_fp8: 1 of" in messages[0]
    # the sign of a zero is not a difference
    zero = good["x_fp8"].clone()
    index = (zero.float() == 0).nonzero()
    if index.numel():
        row, column = index[0].tolist()
        zero.view(torch.uint8)[row, column] ^= 0x80
        assert compare(case, inputs, dict(good, x_fp8=zero)) == []


def test_zero_rows_quantize_to_zero_with_the_epsilon_scale():
    case = case_by_name("row_quant_real_stress")
    inputs = make_inputs(case)
    exp = expected(case, inputs)
    assert bool((exp["x_fp8"][0].float() == 0).all())
    epsilon_scale = (torch.tensor(1e-8, dtype=BF16) / 240.0).float()
    assert exp["scale"][0, 0] == epsilon_scale and float(epsilon_scale) > 0


def test_missing_and_misshaped_outputs():
    case = case_by_name("route_tiny_r3")
    inputs = make_inputs(case)
    good = _kernel_answer(case, inputs)
    assert any("missing" in message for message in compare(case, inputs, {"ids": good["ids"]}))
    assert any("must be" in message for message in compare(case, inputs, dict(good, ids=good["ids"].long())))
    assert any("must be" in message for message in compare(case, inputs, dict(good, weights=good["weights"][:, :1])))


@pytest.mark.parametrize("name", ["route_tiny_r3", "silu_mul_tiny_tp8_r3", "combine_tiny_r3", "row_quant_tiny_r3"])
def test_raw_files_round_trip(name, tmp_path):
    case = case_by_name(name)
    inputs = make_inputs(case)
    mc.write_inputs(case, inputs, tmp_path)
    manifest = (tmp_path / "case.txt").read_text().splitlines()
    assert manifest[0] == f"guid {mc.GUIDS[case.kind]}"
    assert manifest[1:5] == [f"experts {mc.kernel_params(case)[0]}", f"top_k {mc.kernel_params(case)[1]}",
                             f"gate_column {mc.kernel_params(case)[2]}", f"fp8_max {mc.kernel_params(case)[3]!r}"]
    for input_name in mc.INPUT_ORDER[case.kind]:
        assert (tmp_path / f"{input_name}.bin").stat().st_size == 2 * inputs[input_name].numel()
    answer = _kernel_answer(case, inputs)
    mc.write_outputs(case, answer, tmp_path)
    back = mc.read_outputs(case, tmp_path)
    assert compare(case, inputs, back) == []
    for output_name, tensor in answer.items():
        assert back[output_name].dtype == tensor.dtype
        assert (tmp_path / f"{output_name}.out.bin").stat().st_size == tensor.numel() * tensor.element_size()
    first = mc.OUTPUT_ORDER[case.kind][0]
    (tmp_path / f"{first}.out.bin").write_bytes(b"\0")
    with pytest.raises(ValueError, match=f"{first}.out.bin"):
        mc.read_outputs(case, tmp_path)


def test_hpu_cases_add_the_decode_buckets_and_the_largest_prefill():
    extra = [case for case in mc.HPU_CASES if case not in SIM_CASES]
    assert {case.rows for case in extra} == {4, 8, 12, 20, 8192}
    assert {case.kind for case in extra} == set(mc.KINDS)
    assert len({case.name for case in mc.HPU_CASES}) == len(mc.HPU_CASES)
