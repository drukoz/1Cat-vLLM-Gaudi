# SPDX-License-Identifier: Apache-2.0
"""What a CPU can say about the open card failure of ``moe_route`` (first card run, Gaudi2, 2026-11-20).

``test_moe_hpu_op.py::test_hpu_op_matches_the_torch_path_on_hpu[route_real_r16]`` printed, three times and with
the bridge's parallel graph accumulation on and off,

    MOEHPU route_real_r16 ids: 0 of 16 rows select the card's torch.topk set (15 rows do not tie at the cut)

while the kernel alone returned exactly its rule for the same input in the same runs.  The cause needs a card.
These tests pin, for exactly that input, what arithmetic rules out and which gross faults would print that
line, so that the card probe has to separate only what is left:

* **Killed here.**  The card's fp32 softmax against the kernel's arithmetic: no rounding of a monotonic softmax
  moves a set (bf16, fp16, no max subtraction, 0.5 % of relative error), and moving all 15 rows needs an error
  of 64 %.  The tie and the ordering rule of the card's ``torch.topk``: the comparison sorts the ids and one row
  ties at the cut, so no rule moves more than that row.
* **Left for the card.**  The three things the test's reference does and the production routing does not:
  it routes on a view ``gates[:, :experts]`` of the tensor the kernel reads, it narrows the int64 ids to int32
  on the device, and it shares one graph with the kernel.  ``test_gross_faults_and_the_line_they_would_print``
  holds the candidates against the card's line: a narrowing that misreads, a softmax or top-k over the wrong
  dimension and a view read without its stride print 0 (the view only together with another tie rule at row 0).

The kernel's own arithmetic on this input is the simulator case ``route_real_r16``
(``tests/standalone/hc_fused/test_moe_simulation.py``).  Contract: ``docs/design/flash-next/hc-fused.md``,
"First card run".
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.unit_tests.hc_fused import moe_case

CASE = moe_case.case_by_name("route_real_r16")
T, E, K = CASE.rows, CASE.experts, CASE.top_k


@pytest.fixture(scope="module")
def route():
    inputs = moe_case.make_inputs(CASE)
    gates = inputs["gates"]
    logits = gates[:, :E]
    probabilities = F.softmax(logits, dim=1, dtype=torch.float32)
    return {
        "inputs": inputs,
        "gates": gates,
        "logits": logits,
        "p": probabilities,
        "cpu_ids": torch.topk(probabilities, K, dim=-1).indices,
        "rule": moe_case.route_kernel_ids(CASE, inputs).to(torch.int64),
        "clear": ~moe_case.route_tie_rows(CASE, inputs),
    }


def _same_rows(ids: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row: the two selections are the same set of experts (what the card test compares)."""
    return (torch.sort(ids.to(torch.int64), dim=1).values == torch.sort(want.to(torch.int64), dim=1).values).all(dim=1)


def _gaps(route) -> torch.Tensor:
    ordered = torch.sort(route["logits"].float(), dim=1, descending=True).values
    return ordered[:, K - 1] - ordered[:, K]


def test_one_row_ties_at_the_cut_and_cpu_torch_takes_the_kernels_expert_there(route):
    assert (~route["clear"]).nonzero().flatten().tolist() == [0]
    row = route["logits"][0].float()
    cut = torch.sort(row, descending=True).values[K - 1]
    assert (row == cut).nonzero().flatten().tolist() == [132, 450]  # two experts for the tenth slot
    # the kernel takes the lowest id, and so does CPU torch on this row: on a CPU the test's line is 16 of 16
    assert 132 in route["rule"][0].tolist() and 450 not in route["rule"][0].tolist()
    assert int(_same_rows(route["rule"], route["cpu_ids"]).sum()) == T


def test_the_cut_is_wide_on_the_other_fifteen_rows(route):
    """The tenth and eleventh logits are 0.0156 to 1.5 apart: what a softmax error would have to be to swap them."""
    gaps = _gaps(route)[route["clear"]]
    assert float(gaps.min()) == 0.015625 and float(gaps.max()) == 1.5
    # p10 / p11 = exp(gap).  Two probabilities swap when their relative errors add up to that ratio:
    # (1 + e) / (1 - e) = exp(gap), so e = tanh(gap / 2).
    needed = [math.tanh(float(gap) / 2) for gap in gaps]
    assert min(needed) > 0.0078  # one row needs 0.78 %, two bf16 steps of the probability
    assert max(needed) > 0.63  # all fifteen need 64 %


@pytest.mark.parametrize("arithmetic", ["bf16", "fp16", "no_max_subtraction", "half_a_percent"])
def test_no_softmax_arithmetic_moves_a_set(route, arithmetic):
    logits = route["logits"]
    if arithmetic == "bf16":  # the softmax computed in the dtype of the logits, cast afterwards
        candidates = [F.softmax(logits, dim=1).float()]
    elif arithmetic == "fp16":
        candidates = [F.softmax(logits.float().half(), dim=1).float()]
    elif arithmetic == "no_max_subtraction":
        exp = torch.exp(logits.float())
        assert bool(torch.isfinite(exp).all())  # the largest logit of the case is 39
        candidates = [exp / exp.sum(dim=1, keepdim=True)]
    else:  # every probability off by up to half a percent, 200 draws
        g = torch.Generator().manual_seed(1)
        candidates = [route["p"] * (1 + 0.005 * (2 * torch.rand(route["p"].shape, generator=g) - 1))
                      for _ in range(200)]
    for probabilities in candidates:
        same = _same_rows(torch.topk(probabilities, K, dim=-1).indices, route["cpu_ids"])
        assert bool(same[route["clear"]].all())
        ordered = torch.sort(probabilities, dim=1, descending=True).values
        ties = ordered[:, K - 1] == ordered[:, K]
        assert int(ties[route["clear"]].sum()) == 0  # and it makes no new tie at the cut


def test_no_tie_or_ordering_rule_of_topk_moves_more_than_the_one_row(route):
    probabilities = route["p"]
    # the opposite tie rule: the highest index among equal probabilities (a stable sort of the reversed row)
    ordered = torch.sort(probabilities.flip(1), dim=1, descending=True, stable=True)
    highest = E - 1 - ordered.indices[:, :K]
    assert 450 in highest[0].tolist() and 132 not in highest[0].tolist()
    same = _same_rows(highest, route["rule"])
    assert same.tolist() == [False] + [True] * (T - 1)  # the card test's line would be 15 of 16, not 0
    # any order of the slots: the comparison sorts
    g = torch.Generator().manual_seed(2)
    shuffled = torch.gather(route["cpu_ids"], 1, torch.argsort(torch.rand(T, K, generator=g), dim=1))
    assert int(_same_rows(shuffled, route["rule"]).sum()) == T


def _fault_ids(route, fault: str) -> torch.Tensor:
    gates, logits, p, cpu_ids = route["gates"], route["logits"], route["p"], route["cpu_ids"]
    if fault == "view_without_its_stride":  # gates[:, :E] read as the first T * E elements of gates
        block = gates.reshape(-1)[:T * E].reshape(T, E)
        return torch.topk(F.softmax(block, dim=1, dtype=torch.float32), K).indices
    if fault == "softmax_over_every_column":  # the view ignored: the shared-gate logit takes part
        return torch.topk(F.softmax(gates, dim=1, dtype=torch.float32), K).indices
    if fault == "softmax_over_the_tokens":
        return torch.topk(F.softmax(logits, dim=0, dtype=torch.float32), K).indices
    if fault == "topk_over_the_tokens":
        return torch.topk(p, K, dim=0).indices.t().reshape(-1)[:T * K].reshape(T, K)
    if fault == "smallest":
        return torch.topk(p, K, largest=False).indices
    if fault == "ids_read_as_8_bytes":  # every second id of the right table
        every_second = cpu_ids.reshape(-1)[::2]
        return torch.cat([every_second, torch.full((T * K - every_second.numel(), ), -1)]).reshape(T, K)
    if fault == "ids_written_as_8_bytes":  # the right ids with a zero word after each
        return torch.stack([cpu_ids.reshape(-1), torch.zeros(T * K, dtype=torch.int64)], dim=1).reshape(-1)[:T * K] \
            .reshape(T, K)
    if fault == "ids_in_slot_major_order":
        return cpu_ids.t().reshape(T, K)
    raise KeyError(fault)


@pytest.mark.parametrize("fault, rows_the_line_would_count", [
    ("view_without_its_stride", 1),  # row 0 starts where it should; 0 when top-k also takes expert 450 there
    ("softmax_over_every_column", 15),
    ("softmax_over_the_tokens", 0),
    ("topk_over_the_tokens", 0),
    ("smallest", 0),
    ("ids_read_as_8_bytes", 0),
    ("ids_written_as_8_bytes", 0),
    ("ids_in_slot_major_order", 0),
])
def test_gross_faults_and_the_line_they_would_print(route, fault, rows_the_line_would_count):
    """"N of 16 rows select the card's torch.topk set" for a reference that goes wrong in one gross way.

    The card printed 0.  A softmax that also reads the shared-gate column is ruled out by its 15; a view read
    without its stride prints 1, or 0 when the card's top-k takes the other tied expert in row 0.
    """
    same = _same_rows(_fault_ids(route, fault), route["rule"])
    assert int(same.sum()) == rows_the_line_would_count
    if fault == "view_without_its_stride":
        assert same.tolist() == [True] + [False] * (T - 1)
