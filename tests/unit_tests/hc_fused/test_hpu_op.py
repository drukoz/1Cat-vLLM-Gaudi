# SPDX-License-Identifier: Apache-2.0
"""HPU tests for the six fused hyper-connection ops (``torch.ops.flashinfer_gaudi.hc_*``).

Runs each op (HPU kernels from the ``hc_fused_ops`` extension, TPC kernels from
``libhc_fused_gaudi2.so``) on every case of ``hc_case.HPU_CASES`` and checks the outputs against
the torch oracle with ``compare``: the simulator cases plus every decode batch bucket
(1, 2, 4, 8, 12, 16, 20, 32 rows) and the largest prefill (8192 rows) at the real geometry.
The whole module skips without a Gaudi card (no ``habana_frameworks``), without the built
extension, or in a lazy process whose accumulation thread is on (``PT_HPU_LAZY_ACC_PAR_MODE`` not 0:
the loader refuses the kernels there, and the skip line says so).  This file is the rental gate of
the track; ``docs/design/flash-next/hc-fused.md`` ("Rental gate") says what a pass means.
"""

import pytest
import torch

pytest.importorskip("habana_frameworks.torch")

from tests.unit_tests.hc_fused import hc_ops, hc_stress  # noqa: E402
from tests.unit_tests.hc_fused.hc_case import (BF16, HPU_CASES, INPUT_ORDER, OP_NAME, OUTPUT_ORDER,  # noqa: E402
                                               case_by_name, compare, expected, make_inputs, scalars, summary)
from tests.unit_tests.hc_fused.hpu_loader import hc_fused_extension_errors, load_hc_fused_extension  # noqa: E402

if not load_hc_fused_extension():
    pytest.skip("the fused hyper-connection kernels did not load: " + "; ".join(hc_fused_extension_errors()),
                allow_module_level=True)

# docs/design/flash-next/gdn-core-decode.md, "Measured tolerances": bf16 against the composed HPU torch path.
HPU_TORCH_ATOL, HPU_TORCH_RTOL = 5e-2, 1e-2


def _run(case, hpu_inputs):
    op = getattr(hc_ops, OP_NAME[case.kind])
    out = op(*[hpu_inputs[name] for name in INPUT_ORDER[case.kind]], *scalars(case))
    out = out if isinstance(out, tuple) else (out, )
    assert len(out) == len(OUTPUT_ORDER[case.kind])
    return dict(zip(OUTPUT_ORDER[case.kind], out))


def test_every_op_has_an_hpu_kernel():
    assert hc_ops.hc_fused_missing_ops("hpu") == ()


@pytest.mark.parametrize("case", HPU_CASES, ids=[case.name for case in HPU_CASES])
def test_hpu_op_matches_reference(case):
    inputs = make_inputs(case)
    hpu = {name: tensor.to("hpu") for name, tensor in inputs.items()}
    got = {name: tensor.cpu() for name, tensor in _run(case, hpu).items()}
    print(summary(case, inputs, got))
    messages = compare(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []
    for name, tensor in inputs.items():  # no op writes an input
        assert torch.equal(hpu[name].cpu().view(torch.int16), tensor.view(torch.int16)), name


@pytest.mark.parametrize("name", [
    "norm_real_r16", "silu_split_real_r16", "gate_mix_real_r16", "combine_real_r16", "combine_norm_real_r16",
    "combine_norm_real_r2_shared"
])
def test_hpu_op_matches_the_torch_path_on_hpu(name):
    """Against today's production path: the same torch functions run on the card, not on the CPU."""
    from tests.unit_tests.hc_fused import hc_reference
    case = case_by_name(name)
    inputs = make_inputs(case)
    hpu = {key: tensor.to("hpu") for key, tensor in inputs.items()}
    reference = getattr(hc_reference, f"{OP_NAME[case.kind]}_reference")
    want = reference(*[hpu[key] for key in INPUT_ORDER[case.kind]], *scalars(case))
    want = want if isinstance(want, tuple) else (want, )
    got = _run(case, hpu)
    for key, torch_out in zip(OUTPUT_ORDER[case.kind], want):
        fused, torch_out = got[key].cpu().float(), torch_out.cpu().float()
        worst = float((fused - torch_out).abs().max())
        print(f"HCHPU {case.name} {key}: max abs diff against the HPU torch path {worst:.3e}")
        assert torch.allclose(fused, torch_out, atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL), (key, worst)


def test_hpu_ops_run_twice_with_the_same_result():
    """A second call (recipe already compiled) returns the same bits as the first."""
    case = case_by_name("combine_norm_real_r2_shared")
    inputs = make_inputs(case)
    hpu = {name: tensor.to("hpu") for name, tensor in inputs.items()}
    first = {name: tensor.cpu() for name, tensor in _run(case, hpu).items()}
    second = {name: tensor.cpu() for name, tensor in _run(case, hpu).items()}
    for name in first:
        assert first[name].dtype == BF16
        assert torch.equal(first[name].view(torch.int16), second[name].view(torch.int16))


@pytest.mark.parametrize("name", hc_stress.STRESS_CASES)
def test_hpu_op_behind_a_queue_of_lazy_ops(name):
    """The op is called right after 96 lazy ops were queued, eight rounds, as a layer calls it in a model.

    The cases above call an op with one clone queued in front of it. With the bridge's accumulation thread
    on they failed at random, because the op was built before the producer of its input (2026-11-20). The
    loader requires that thread off; this is the check that the op then gets its inputs under a queue too.
    The body is ``hc_stress.run_behind_a_queue``, which ``test_hc_ops.py`` runs on CPU.
    """
    messages = hc_stress.run_behind_a_queue(case_by_name(name), "hpu")
    if messages:
        print("\n".join(messages))
    assert messages == []


def test_hpu_chain_of_one_hyper_connection():
    """combine_norm -> GEMM -> silu_split -> GEMM -> gate_mix on the card against the torch chain on the card.

    The GEMMs are torch linears on both sides; the fused ops feed each other on the device, as the
    layer wires them, and the injection output of silu_split feeds a second combine_norm.  The
    difference against the CPU chain is printed, not asserted: it also contains the difference
    between the card's bf16 GEMM and the CPU's, which is not this track's.
    """
    from tests.unit_tests.hc_fused import hc_reference as ref
    hc, hidden, lora, tokens = 4, 2560, 320, 8
    g = torch.Generator().manual_seed(1234)
    width = hc * hidden
    residual = torch.randn(tokens, width, generator=g).to(BF16)
    block = torch.randn(tokens, hidden, generator=g).to(BF16)
    injection = (torch.randn(tokens, hc, generator=g) * 2).to(BF16)
    weight = (0.25 * torch.randn(width, generator=g)).to(BF16)
    down_weight = (torch.randn(lora + hc + 12, width, generator=g) * width**-0.5).to(BF16)
    up_weight = (torch.randn(width, lora, generator=g) * lora**-0.5).to(BF16)

    def chain(device, fused):
        res, blk, inj, w = (t.to(device) for t in (residual, block, injection, weight))
        down_w, up_w = down_weight.to(device), up_weight.to(device)
        if fused:
            out, xn = hc_ops.hc_combine_norm(res, blk, inj, w, 1e-6, hc)
            lora_act, inj_out = hc_ops.hc_silu_split(torch.nn.functional.linear(xn, down_w), hc, lora)
            block_input = hc_ops.hc_gate_mix(xn, torch.nn.functional.linear(lora_act, up_w), hc)
            out2, _ = hc_ops.hc_combine_norm(out, block_input, inj_out, w, 1e-6, hc)
        else:
            out, xn = ref.hc_combine_norm_torch(res, blk, inj, w, 1e-6, hc)
            down = torch.nn.functional.linear(xn, down_w)
            lora_act = ref.hc_silu_torch(down[:, :lora], hc)
            inj_out = down[:, lora:lora + hc]
            block_input = ref.hc_gate_mix_torch(xn, torch.nn.functional.linear(lora_act, up_w), hc)
            out2, _ = ref.hc_combine_norm_torch(out, block_input, inj_out, w, 1e-6, hc)
        return [t.cpu().float() for t in (out, block_input, inj_out, out2)]

    names = ("residual_out", "block_input", "injection", "next_residual_out")
    fused_hpu = chain("hpu", True)
    for name, a, b in zip(names, fused_hpu, chain("cpu", False)):
        print(f"HCHPU chain {name} against the CPU oracle chain: max abs diff {float((a - b).abs().max()):.3e}")
    for name, a, b in zip(names, fused_hpu, chain("hpu", False)):
        worst = float((a - b).abs().max())
        print(f"HCHPU chain {name} against the HPU torch chain: max abs diff {worst:.3e}")
        assert torch.allclose(a, b, atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL), (name, worst)
