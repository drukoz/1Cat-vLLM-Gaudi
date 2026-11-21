# SPDX-License-Identifier: Apache-2.0
"""Card test of the fused QSA decode ops (``torch.ops.flashinfer_gaudi.qsa_index_prepare`` / ``qsa_index_emit``).

Runs both ops (HPU kernels from the ``qsa_fused_ops`` extension, TPC kernels from ``libqsa_fused_gaudi2.so``) on
every case of ``qsa_case.HPU_CASES`` and checks them with ``qsa_case.compare`` against the torch references on the
CPU: the simulator cases plus the target's 32 x 4 verification shape. The prepare op writes its ring and cache
inputs in place on the card; the test compares them after the call. The whole module skips without a Gaudi card or
without the built extension. ``docs/design/flash-next/qsa-fused.md`` says what a pass means.
"""

import pytest
import torch

pytest.importorskip("habana_frameworks.torch")

from flashinfer_gaudi._native import load_qsa_fused_extension, qsa_fused_extension_errors  # noqa: E402

if not load_qsa_fused_extension():
    pytest.skip("qsa_fused_ops extension is not loaded: " + "; ".join(qsa_fused_extension_errors()),
                allow_module_level=True)

from flashinfer_gaudi.qsa_fused import (qsa_fused_missing_ops, qsa_index_emit, qsa_index_prepare,  # noqa: E402
                                        qsa_index_select, qsa_index_threshold, qsa_index_topk, qsa_topk_missing_op)
from tests.unit_tests.qsa_fused.qsa_case import EPS, HPU_CASES, case_by_name, compare, make_inputs, summary  # noqa: E402


def _run(case, inputs: dict) -> dict:
    hpu = {name: t.to("hpu") for name, t in inputs.items()}
    if case.kernel == "prepare":
        q = qsa_index_prepare(hpu["key_ring"], hpu["key_cache"], hpu["qk"], hpu["rows"], hpu["q_norm_weight"],
                              hpu["k_norm_weight"], hpu["cos_sin"], EPS, case.heads, case.ring_rows, case.ratio)
        return {"key_ring": hpu["key_ring"].cpu(), "key_cache": hpu["key_cache"].cpu(), "q_idx": q.cpu()}
    if case.kernel == "threshold":
        return {"meta": qsa_index_threshold(hpu["scores"], hpu["positions"], case.ratio, case.topk).cpu()}
    if case.kernel == "select":
        return {"blocks": qsa_index_select(hpu["meta"], case.topk).cpu()}
    if case.kernel == "topk":
        return {"ids": qsa_index_topk(hpu["masked"], case.topk).cpu()}
    slots, valid = qsa_index_emit(hpu["blocks"], hpu["positions"], hpu["block_table"], case.ratio, case.block_size)
    return {"slots": slots.cpu(), "valid": valid.cpu()}


def test_every_op_has_an_hpu_kernel():
    assert qsa_fused_missing_ops("hpu") == ()
    assert not qsa_topk_missing_op("hpu")


@pytest.mark.parametrize("case", HPU_CASES, ids=[case.name for case in HPU_CASES])
def test_hpu_op_matches_reference(case):
    inputs = make_inputs(case)
    got = _run(case, inputs)
    print("QSAHPU " + summary(case, inputs, got))
    messages = compare(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []


def test_hpu_ops_run_twice_with_the_same_result():
    """A second call (recipe already compiled) returns the same bits; the prepare op from the same inputs."""
    for name in ("prepare_real_verify32x4", "emit_real_verify128", "topk_real32"):
        case = case_by_name(name)
        inputs = make_inputs(case)
        first, second = _run(case, inputs), _run(case, inputs)
        for key in first:
            assert torch.equal(first[key].view(torch.int16) if first[key].dtype == torch.bfloat16 else first[key],
                               second[key].view(torch.int16) if second[key].dtype == torch.bfloat16 else second[key])


def test_index_select_takes_the_int32_slots():
    """The attention gathers K/V rows with the emit op's int32 slots (card rule 2): the same rows as int64."""
    case = case_by_name("emit_real_decode32")
    inputs = make_inputs(case)
    slots = _run(case, inputs)["slots"].to("hpu")
    cache = torch.randn(int(slots.max()) + 1, 256).to(torch.bfloat16).to("hpu")
    a = cache.index_select(0, slots.reshape(-1)).cpu()
    b = cache.index_select(0, slots.reshape(-1).long()).cpu()
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
