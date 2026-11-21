# SPDX-License-Identifier: Apache-2.0
"""HPU tests of the interleaved hyper-connection ops ``hc_gate_mix_il`` and ``hc_combine_norm_il``
(docs/design/flash-next/hc-kernels.md): the card gate of ``VLLM_HPU_HC_FUSED_IL=1``.

Each op runs on every ``hc_case.HPU_CASES`` case of its kind (the simulator cases, every decode batch bucket and
the largest prefill at the real geometry) and must pass the HC contract against the torch oracle
(``hc_case.compare``). The same inputs also run through the shipped op on the card: ``residual_out`` must be the
shipped op's bit for bit, the other outputs in at least ``MIN_BIT_EQUAL`` of their elements. Then a chain of one
hyper-connection on the interleaved ops against the same chain on the shipped ops, and the ops inside an HPU graph
(replayed twice) against eager calls. Unlike ``test_hpu_op.py``, the module FAILS (it does not skip) when the
interleaved ops or their TPC kernels are not served once the fused extension is loaded: that is what it checks.
"""

import pytest
import torch

pytest.importorskip("habana_frameworks.torch")

from tests.unit_tests.hc_fused import hc_ops  # noqa: E402
from tests.unit_tests.hc_fused.hc_case import (BF16, HPU_CASES, INPUT_ORDER, OP_NAME, OUTPUT_ORDER,  # noqa: E402
                                               case_by_name, compare, make_inputs, scalars, summary)
from tests.unit_tests.hc_fused.hpu_loader import hc_fused_extension_errors, load_hc_fused_extension  # noqa: E402

if not load_hc_fused_extension():
    pytest.skip("the fused hyper-connection kernels did not load: " + "; ".join(hc_fused_extension_errors()),
                allow_module_level=True)

IL_OP = {"gate_mix": "hc_gate_mix_il", "combine_norm": "hc_combine_norm_il"}
CASES = tuple(case for case in HPU_CASES if case.kind in IL_OP)
MIN_BIT_EQUAL = 0.999


def _run(case, hpu_inputs, op_name):
    op = getattr(hc_ops, op_name)
    out = op(*[hpu_inputs[name] for name in INPUT_ORDER[case.kind]], *scalars(case))
    out = out if isinstance(out, tuple) else (out, )
    assert len(out) == len(OUTPUT_ORDER[case.kind])
    return dict(zip(OUTPUT_ORDER[case.kind], out))


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.view(torch.int16) == b.view(torch.int16)).float().mean())


def test_interleaved_ops_and_kernels_are_served():
    from flashinfer_gaudi import _native
    assert hc_ops.hc_fused_il_missing_ops("hpu") == ()
    assert _native.hc_fused_il_missing_kernels() == ()


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_il_op_matches_reference_and_the_shipped_op(case):
    inputs = make_inputs(case)
    hpu = {name: tensor.to("hpu") for name, tensor in inputs.items()}
    got = {name: tensor.cpu() for name, tensor in _run(case, hpu, IL_OP[case.kind]).items()}
    print(summary(case, inputs, got))
    messages = compare(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []
    shipped = {name: tensor.cpu() for name, tensor in _run(case, hpu, OP_NAME[case.kind]).items()}
    for name in got:
        equal = _bit_equal(got[name], shipped[name])
        print(f"HCILHPU {case.name} {name}: {100 * equal:.4f} % bit-equal to the shipped op on the card")
        assert equal == 1.0 if (name == "residual_out" or case.exact) else equal >= MIN_BIT_EQUAL, name
    for name, tensor in inputs.items():  # no op writes an input
        assert torch.equal(hpu[name].cpu().view(torch.int16), tensor.view(torch.int16)), name


@pytest.mark.parametrize("tokens", [4, 32, 96, 128])
def test_il_chain_of_one_hyper_connection_against_the_shipped_chain(tokens):
    """combine_norm -> GEMM -> silu_split -> GEMM -> gate_mix -> combine_norm, on the card, interleaved against
    shipped: the same GEMMs, the ops feed each other on the device as the layer wires them."""
    hc, hidden, lora = 4, 2560, 320
    g = torch.Generator().manual_seed(4321 + tokens)
    width = hc * hidden
    residual = torch.randn(tokens, width, generator=g).to(BF16)
    block = torch.randn(tokens, hidden, generator=g).to(BF16)
    injection = (torch.randn(tokens, hc, generator=g) * 2).to(BF16)
    weight = (0.25 * torch.randn(width, generator=g)).to(BF16)
    down_weight = (torch.randn(lora + hc + 12, width, generator=g) * width**-0.5).to(BF16)
    up_weight = (torch.randn(width, lora, generator=g) * lora**-0.5).to(BF16)

    def chain(il: bool):
        combine_norm = hc_ops.hc_combine_norm_il if il else hc_ops.hc_combine_norm
        gate_mix = hc_ops.hc_gate_mix_il if il else hc_ops.hc_gate_mix
        res, blk, inj, w = (t.to("hpu") for t in (residual, block, injection, weight))
        down_w, up_w = down_weight.to("hpu"), up_weight.to("hpu")
        out, xn = combine_norm(res, blk, inj, w, 1e-6, hc)
        lora_act, inj_out = hc_ops.hc_silu_split(torch.nn.functional.linear(xn, down_w), hc, lora)
        block_input = gate_mix(xn, torch.nn.functional.linear(lora_act, up_w), hc)
        out2, xn2 = combine_norm(out, block_input, inj_out, w, 1e-6, hc)
        return [t.cpu() for t in (out, xn, block_input, out2, xn2)]

    il, shipped = chain(True), chain(False)
    names = ("residual_out", "xn", "block_input", "residual_out_2", "xn_2")
    for name, a, b in zip(names, il, shipped):
        equal = _bit_equal(a, b)
        worst = float((a.float() - b.float()).abs().max())
        print(f"HCILCHAIN rows {tokens} {name}: {100 * equal:.4f} % bit-equal, max abs diff {worst:.3e}")
        if name == "residual_out":
            assert equal == 1.0
        else:
            assert equal >= 0.99, name  # differences propagate through the GEMMs in the second half
            assert torch.allclose(a.float(), b.float(), atol=5e-2, rtol=1e-2), name


@pytest.mark.parametrize("name", ["gate_mix_real_r32", "combine_norm_real_r32", "combine_norm_real_r2_shared"])
def test_il_ops_in_an_hpu_graph(name):
    """The ops captured in an HPU graph and replayed twice give the bits of the eager calls."""
    import habana_frameworks.torch as htorch
    case = case_by_name(name)
    inputs = make_inputs(case)
    hpu = {key: tensor.to("hpu") for key, tensor in inputs.items()}
    eager = {key: tensor.cpu() for key, tensor in _run(case, hpu, IL_OP[case.kind]).items()}

    class Calls(torch.nn.Module):

        def forward(self, *tensors):
            out = getattr(hc_ops, IL_OP[case.kind])(*tensors, *scalars(case))
            return out if isinstance(out, tuple) else (out, )

    module = htorch.hpu.wrap_in_hpu_graph(Calls())
    args = [hpu[key] for key in INPUT_ORDER[case.kind]]
    for _ in range(3):
        replay = dict(zip(OUTPUT_ORDER[case.kind], (t.cpu() for t in module(*args))))
        for key in eager:
            assert torch.equal(replay[key].view(torch.int16), eager[key].view(torch.int16)), key
