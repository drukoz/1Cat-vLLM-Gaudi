# SPDX-License-Identifier: Apache-2.0
"""HPU tests for the four fused MoE glue ops (``torch.ops.flashinfer_gaudi.moe_*``).

Runs each op (HPU kernels from the ``hc_fused_ops`` extension, TPC kernels from
``libhc_fused_gaudi2.so``) on every case of ``moe_case.HPU_CASES`` and checks the outputs against the
torch oracle with ``compare``: the simulator cases plus the decode batch buckets (4, 8, 12, 20 rows) and
the largest prefill (8192 rows) at the real geometry.  Then the things only a card can show: the routing
op's outputs go into ``torch.ops.hpu.mixture_of_experts`` as the block wires them, the row quantization
against the plugin's ``dynamic_quant`` on the card, the fused ops against the torch path on the card, and
one whole block chain.  Lines starting with ``MOEHPU`` are measurements to read after a run.

The whole module skips without a Gaudi card (no ``habana_frameworks``) or without the built kernels.
This file is the rental gate of the MoE glue (``gates.moe_glue``); ``docs/design/flash-next/hc-fused.md``
("MoE glue kernels", "Rental gate") says what a pass means.

``MOE_GLUE_OP_TEST_DEVICE=cpu`` is a dry run of this file without a card (``test_moe_hpu_op_dry_run.py``):
the ops are stand-ins for the kernels (the kernel's own rule for equal logits, the kernel's oracle for the
row quantization), the vendor ops run the CPU stand-ins of ``tests/unit_tests/models/flash_next/moe_block.py``,
and the plugin's ``dynamic_quant`` runs with the one arithmetic rule of the card that is known to differ
from CPU torch (``hpu_arithmetic.py``).  It proves that this file runs, not that a kernel is right.
"""

import os

import pytest
import torch
import torch.nn.functional as F

DEVICE = os.environ.get("MOE_GLUE_OP_TEST_DEVICE", "hpu")
DRY_RUN = DEVICE != "hpu"

if DRY_RUN:
    from tests.unit_tests.models.flash_next import moe_block

    moe_block.install_vendor_ops()
else:
    pytest.importorskip("habana_frameworks.torch")

from flashinfer_gaudi import _native  # noqa: E402
from tests.unit_tests.hc_fused import hpu_arithmetic, moe_case, moe_ops, moe_reference  # noqa: E402
from tests.unit_tests.hc_fused.moe_case import (BF16, HPU_CASES, INPUT_ORDER, OP_NAME, OUTPUT_ORDER,  # noqa: E402
                                                case_by_name, compare, make_inputs, scalars, summary)

if not DRY_RUN and not _native.load_moe_glue_extension():
    pytest.skip("the MoE glue kernels are not usable: " + "; ".join(_native.moe_glue_extension_errors()),
                allow_module_level=True)

# docs/design/flash-next/gdn-core-decode.md, "Measured tolerances": bf16 against the composed HPU torch path.
HPU_TORCH_ATOL, HPU_TORCH_RTOL = 5e-2, 1e-2
EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, SHARED = 16, 2, 256, 64, 32  # a small block for the chains


def _route(gates, experts, top_k):
    """The routing op.  In a dry run the stand-in is given the kernel's rule for equal logits."""
    if DRY_RUN:
        case = moe_case.MoeCase("dry", "route", gates.shape[0], experts=experts, top_k=top_k, columns=gates.shape[1])
        ids = moe_case.route_kernel_ids(case, {"gates": gates})
        return ids, moe_case.route_expected_weights(case, {"gates": gates}, ids)
    return moe_ops.moe_route(gates, experts, top_k)


def _row_quant(x, fp8_max):
    """The quantization op.  In a dry run the stand-in is the kernel's oracle, not the op's CPU kernel."""
    if DRY_RUN:
        return moe_reference.moe_row_quant_reference(x, fp8_max)
    return moe_ops.moe_row_quant(x, fp8_max)


def _dynamic_quant_on_the_card(plugin_ops, x):
    """The plugin's dynamic_quant.  A dry run has no card: the same function with the card's scalar division."""
    if DRY_RUN:
        with hpu_arithmetic.card_arithmetic():
            return plugin_ops.dynamic_quant(x)
    return plugin_ops.dynamic_quant(x)


def _run(case, device_inputs):
    op = {"route": _route, "row_quant": _row_quant}.get(case.kind) or getattr(moe_ops, OP_NAME[case.kind])
    out = op(*[device_inputs[name] for name in INPUT_ORDER[case.kind]], *scalars(case))
    out = out if isinstance(out, tuple) else (out, )
    assert len(out) == len(OUTPUT_ORDER[case.kind])
    return dict(zip(OUTPUT_ORDER[case.kind], out))


def _bits(tensor):
    return tensor.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()])


def _block_weights(seed=4321):
    """Weights of one small MoE block: router and shared-gate rows, the shared expert, the routed experts."""
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(BF16)

    return {
        "gate": randn(EXPERTS, HIDDEN, scale=HIDDEN**-0.5 * 3),
        "shared_gate": randn(1, HIDDEN, scale=HIDDEN**-0.5),
        "gate_up": randn(2 * SHARED, HIDDEN, scale=HIDDEN**-0.5),
        "down": randn(HIDDEN, SHARED, scale=SHARED**-0.5),
        "w12": [randn(2 * INTERMEDIATE, HIDDEN, scale=HIDDEN**-0.5) for _ in range(EXPERTS)],
        "w3": [randn(HIDDEN, INTERMEDIATE, scale=INTERMEDIATE**-0.5) for _ in range(EXPERTS)],
    }


def _vendor(x, ids, weights, w12, w3):
    """torch.ops.hpu.mixture_of_experts as the plugin calls it for bf16 experts (vllm_gaudi/extension/ops.py)."""
    return torch.ops.hpu.mixture_of_experts(hidden_states=x,
                                            expert_routing_table=ids,
                                            router_weights=weights,
                                            w12=tuple(w12),
                                            w3=tuple(w3),
                                            permuted_weights=True,
                                            activation="silu",
                                            experts_min=0,
                                            experts_max=len(w12) - 1)


def _hidden(rows, seed=77):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, HIDDEN, generator=g) * 1.5).to(BF16)


def test_every_op_has_a_kernel():
    assert moe_ops.moe_glue_missing_ops(DEVICE) == ()
    if not DRY_RUN:
        assert _native.moe_glue_missing_kernels() == ()  # the library serves the four GUIDs


@pytest.mark.parametrize("case", HPU_CASES, ids=[case.name for case in HPU_CASES])
def test_hpu_op_matches_reference(case):
    inputs = make_inputs(case)
    device_inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}
    got = {name: tensor.cpu() for name, tensor in _run(case, device_inputs).items()}
    print(summary(case, inputs, got))
    messages = compare(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []
    for name, tensor in inputs.items():  # no op writes an input
        assert torch.equal(_bits(device_inputs[name].cpu()), _bits(tensor)), name


def test_routing_outputs_go_into_the_vendor_op_unchanged():
    """ids (int32, widened to int64 as the layer does) and bf16 weights of moe_route feed mixture_of_experts.

    The reference is the same vendor op fed by the torch routing on the card.  Where both routings give the
    same ids and weights, the vendor op must return the same bits; otherwise the outputs are compared within
    the bf16 tolerance (a row that ties at the cut may route to another expert of equal weight).
    """
    weights = _block_weights()
    x = _hidden(32).to(DEVICE)
    gate = weights["gate"].to(DEVICE)
    w12 = [w.to(DEVICE) for w in weights["w12"]]
    w3 = [w.to(DEVICE) for w in weights["w3"]]
    logits = F.linear(x, gate)

    ids, route_weights = _route(logits, EXPERTS, TOP_K)
    assert ids.dtype == torch.int32 and route_weights.dtype == BF16
    fused = _vendor(x, ids.to(torch.int64), route_weights, w12, w3).cpu()

    class Layer:
        top_k = TOP_K

    torch_ids, torch_weights = moe_reference.moe_route_torch(x, logits, Layer)
    plain = _vendor(x, torch_ids, torch_weights, w12, w3).cpu()

    same_rows = ((ids.cpu().long() == torch_ids.cpu()).all(dim=1)
                 & (_bits(route_weights.cpu()) == _bits(torch_weights.cpu())).all(dim=1))
    worst = float((fused.float() - plain.float()).abs().max())
    print(f"MOEHPU vendor op fed by moe_route against fed by the torch routing on the card: {int(same_rows.sum())} of "
          f"{same_rows.numel()} rows have the same ids and weights; max abs diff of the outputs {worst:.3e}")
    assert torch.isfinite(fused.float()).all() and float(fused.float().abs().mean()) > 0
    if bool(same_rows.any()):
        assert torch.equal(_bits(fused[same_rows]), _bits(plain[same_rows]))
    assert torch.allclose(fused.float(), plain.float(), atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL), worst


def test_vendor_op_does_not_depend_on_the_slot_order():
    """The kernel orders equal logits by expert id, torch.topk as it likes: the weighted sum must not care."""
    weights = _block_weights()
    x = _hidden(16).to(DEVICE)
    w12 = [w.to(DEVICE) for w in weights["w12"]]
    w3 = [w.to(DEVICE) for w in weights["w3"]]
    ids, route_weights = _route(F.linear(x, weights["gate"].to(DEVICE)), EXPERTS, 4)
    ids = ids.to(torch.int64)
    forward = _vendor(x, ids, route_weights, w12, w3).cpu()
    backward = _vendor(x, ids.flip(1).contiguous(), route_weights.flip(1).contiguous(), w12, w3).cpu()
    identical = torch.equal(_bits(forward), _bits(backward))
    worst = float((forward.float() - backward.float()).abs().max())
    print(f"MOEHPU vendor op with the routing slots reversed: bit-identical {identical}, max abs diff {worst:.3e}")
    assert torch.allclose(forward.float(), backward.float(), atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL), worst


def test_tie_rule_of_the_torch_path_on_the_card_is_reported():
    """Which experts the card's softmax + torch.topk takes among equal logits (the kernel: the lowest ids)."""
    equal = torch.zeros(1, 512, dtype=BF16)
    cut = torch.full((1, 512), -4.0)
    cut[0, 100:109] = torch.arange(9, 0, -1).float()  # nine clear winners
    cut[0, [7, 44, 250, 300, 511]] = 0.5  # five candidates for the tenth slot
    for name, logits in (("all-equal logits", equal), ("five equal logits at the cut", cut.to(BF16))):
        device_logits = logits.to(DEVICE)
        probabilities = F.softmax(device_logits, dim=1, dtype=torch.float32)
        card_ids = torch.topk(probabilities, 10, dim=-1).indices.cpu()[0].tolist()
        kernel_ids = _route(device_logits, 512, 10)[0].cpu()[0].tolist()
        print(f"MOEHPU {name}: torch.topk on the card selects {card_ids}; moe_route selects {kernel_ids}; "
              f"same set {sorted(card_ids) == sorted(kernel_ids)}")
        assert kernel_ids == sorted(kernel_ids, key=lambda index: (-float(logits[0, index]), index))
    assert _route(equal.to(DEVICE), 512, 10)[0].cpu()[0].tolist() == list(range(10))


def test_row_quant_is_the_plugins_dynamic_quant_on_the_card(monkeypatch):
    """moe_row_quant against the plugin's dynamic_quant (abs, max, add, div, reciprocal, cast) on the card."""
    import vllm_gaudi.extension.ops as plugin_ops

    # The branch of dynamic_quant the kernel replaces is the default one: whatever the process inherited,
    # the flags that select another branch are off for this comparison (a block checks the same flags).
    for flag in ("VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT", "VLLM_HPU_FP8_JIT_DYNAMIC_QUANT", "VLLM_HPU_DYNAMIC_QUANT_CGUID",
                 "VLLM_HPU_CGUID_DYNAMIC_QUANT"):
        monkeypatch.setenv(flag, "0")
        assert not getattr(plugin_ops.gaudi_envs, flag), flag
    if DRY_RUN:  # a dry run stands for a Gaudi2 card: its fp8 range, in the plugin and in the cast stand-in
        monkeypatch.setattr(plugin_ops, "FP8_MAX", moe_reference.GAUDI2_FP8_MAX)
    fp8_max = float(plugin_ops.FP8_MAX)
    assert fp8_max == moe_reference.GAUDI2_FP8_MAX, "FP8_MAX of the plugin on this device is not Gaudi2's 240"
    for name in ("row_quant_real_r32", "row_quant_real_stress", "row_quant_real_edge"):
        x = make_inputs(case_by_name(name))["x"].to(DEVICE)
        want_fp8, want_scale = _dynamic_quant_on_the_card(plugin_ops, x)
        got_fp8, got_scale = _row_quant(x, fp8_max)
        assert got_fp8.dtype == want_fp8.dtype == torch.float8_e4m3fn and got_scale.dtype == want_scale.dtype
        scale_differs = int((got_scale.cpu() != want_scale.cpu()).sum())
        have, want = got_fp8.cpu().float(), want_fp8.cpu().float()
        differs = int((have != want).sum())
        worst = float((have - want).abs().max())
        print(f"MOEHPU {name} against dynamic_quant on the card: {scale_differs} of {want_scale.numel()} scales and "
              f"{differs} of {want.numel()} fp8 values differ (max abs diff {worst:.3e})")
        assert scale_differs == 0
        assert differs == 0


@pytest.mark.parametrize("name", ["silu_mul_real_tp8_r16", "silu_mul_real_tp1_r32", "combine_real_r16",
                                  "combine_tiny_r3", "route_real_r16"])
def test_hpu_op_matches_the_torch_path_on_hpu(name):
    """Against today's production path: the same torch code run on the card, not on the CPU."""
    case = case_by_name(name)
    inputs = make_inputs(case)
    device_inputs = {key: tensor.to(DEVICE) for key, tensor in inputs.items()}
    reference = getattr(moe_reference, f"{OP_NAME[case.kind]}_reference")
    want = reference(*[device_inputs[key] for key in INPUT_ORDER[case.kind]], *scalars(case))
    want = want if isinstance(want, tuple) else (want, )
    got = _run(case, device_inputs)
    for key, torch_out in zip(OUTPUT_ORDER[case.kind], want):
        fused, torch_out = got[key].cpu(), torch_out.cpu()
        if key == "ids":  # the same experts wherever no row ties at the cut
            clear = ~moe_case.route_tie_rows(case, inputs)
            same = (torch.sort(fused, dim=1).values == torch.sort(torch_out, dim=1).values).all(dim=1)
            print(f"MOEHPU {case.name} ids: {int(same.sum())} of {case.rows} rows select the card's torch.topk set "
                  f"({int(clear.sum())} rows do not tie at the cut)")
            assert bool(same[clear].all())
            continue
        if key == "weights":  # the weights of the same experts, whatever the order among equal logits
            fused, torch_out = torch.sort(fused.float(), dim=1).values, torch.sort(torch_out.float(), dim=1).values
        worst = float((fused.float() - torch_out.float()).abs().max())
        print(f"MOEHPU {case.name} {key}: max abs diff against the HPU torch path {worst:.3e}")
        assert torch.allclose(fused.float(), torch_out.float(), atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL), (key, worst)


def test_hpu_ops_run_twice_with_the_same_result():
    """A second call (recipe already compiled) returns the same bits as the first."""
    for name in ("route_real_r2", "row_quant_real_r2", "combine_real_r2", "silu_mul_real_tp8_r2"):
        case = case_by_name(name)
        device_inputs = {key: tensor.to(DEVICE) for key, tensor in make_inputs(case).items()}
        first = {key: tensor.cpu() for key, tensor in _run(case, device_inputs).items()}
        second = {key: tensor.cpu() for key, tensor in _run(case, device_inputs).items()}
        for key in first:
            assert torch.equal(_bits(first[key]), _bits(second[key])), (name, key)


def test_hpu_chain_of_one_moe_block():
    """One block as the layer wires it, on the card, against the torch chain on the card.

    merged gate GEMM -> moe_route -> (ids widened) -> gate_up GEMM -> moe_silu_mul -> down GEMM ->
    mixture_of_experts -> moe_combine.  The GEMMs and the vendor op are the same ops on both sides.
    """
    weights = _block_weights(seed=99)
    on_device = {key: ([w.to(DEVICE) for w in value] if isinstance(value, list) else value.to(DEVICE))
                 for key, value in weights.items()}
    merged = torch.cat([on_device["gate"], on_device["shared_gate"]], dim=0).contiguous()

    class Layer:
        top_k = TOP_K

    for rows in (1, 32):
        x = _hidden(rows, seed=rows).to(DEVICE)

        gates = F.linear(x, merged)
        ids, route_weights = _route(gates, EXPERTS, TOP_K)
        act = moe_ops.moe_silu_mul(F.linear(x, on_device["gate_up"]))
        shared = F.linear(act, on_device["down"])
        routed = _vendor(x, ids.to(torch.int64), route_weights, on_device["w12"], on_device["w3"])
        fused = moe_ops.moe_combine(shared, gates, routed, EXPERTS).cpu()

        logits = F.linear(x, on_device["gate"])
        torch_ids, torch_weights = moe_reference.moe_route_torch(x, logits, Layer)
        torch_act = moe_reference.moe_silu_mul_torch(F.linear(x, on_device["gate_up"]))
        torch_shared = F.linear(torch_act, on_device["down"])
        torch_routed = _vendor(x, torch_ids, torch_weights, on_device["w12"], on_device["w3"])
        plain = moe_reference.moe_combine_torch(torch_shared, F.linear(x, on_device["shared_gate"]), torch_routed).cpu()

        assert fused.dtype == BF16 and tuple(fused.shape) == (rows, HIDDEN)
        same = torch.equal(_bits(fused), _bits(plain))
        worst = float((fused.float() - plain.float()).abs().max())
        print(f"MOEHPU chain of one block at {rows} rows against the HPU torch chain: bit-identical {same}, "
              f"max abs diff {worst:.3e}")
        assert torch.allclose(fused.float(), plain.float(), atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL), worst
