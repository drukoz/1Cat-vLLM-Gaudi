# SPDX-License-Identifier: Apache-2.0
"""The torch reference copied into ``moe_reference.py`` is the torch code the MoE block runs (needs vLLM).

The TPC simulator container has neither ``vllm`` nor ``vllm_gaudi``, so the simulator tests compare the
MoE glue kernels with copies of four pieces of torch code.  This test ties the copies to the originals:
every line of a copy is a line of the plugin's (or the pinned engine's) source, in the same order, and
the copies return the bits the plugin's own functions return on CPU.
"""

import os

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import inspect  # noqa: E402
import textwrap  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("vllm")

from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP  # noqa: E402

import vllm_gaudi.extension.ops as plugin_ops  # noqa: E402
import vllm_gaudi.ops.hpu_fp8 as plugin_fp8  # noqa: E402
import vllm_gaudi.ops.hpu_fused_moe as plugin_moe  # noqa: E402
import vllm_gaudi.ops.hpu_silu_and_mul as plugin_silu  # noqa: E402
from tests.unit_tests.hc_fused import moe_reference  # noqa: E402

BF16 = torch.bfloat16


def _body_lines(function) -> list[str]:
    """The statements of a copied function: no signature, no docstring, no return of the copy's own tuple."""
    source = textwrap.dedent(inspect.getsource(function))
    body = source.split('"""', 2)[2]
    return [line.strip() for line in body.splitlines() if line.strip()]


def _code_lines(obj) -> list[str]:
    """The stripped non-comment lines of a plugin function."""
    lines = [line.strip() for line in inspect.getsource(obj).splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _assert_in_order(lines: list[str], source: list[str], where: str) -> None:
    position = 0
    for line in lines:
        assert line in source[position:], f"{line!r} is not in {where} (after line {position})"
        position = source.index(line, position) + 1


def test_copied_functions_are_listed():
    assert moe_reference.COPIED_FUNCTIONS == ("moe_route_torch", "moe_row_quant_torch", "moe_silu_mul_torch",
                                              "moe_combine_torch")


@pytest.mark.parametrize("method", [plugin_moe.HPUUnquantizedFusedMoEMethod.apply_monolithic,
                                    plugin_fp8.HPUFp8MoEMethod.apply_monolithic],
                         ids=["unquantized", "fp8"])
def test_route_copy_has_the_plugin_lines(method):
    lines = _body_lines(moe_reference.moe_route_torch)
    assert lines[-1] == "return topk_ids, topk_weights"
    _assert_in_order(lines[:-1], _code_lines(method), method.__qualname__)


def test_row_quant_copy_has_the_plugin_lines():
    lines = [line.replace("cast_to_fp8_v2(", "torch.ops.hpu.cast_to_fp8_v2(")
             for line in _body_lines(moe_reference.moe_row_quant_torch)]
    _assert_in_order(lines, _code_lines(plugin_ops.dynamic_quant), "dynamic_quant")


def test_silu_mul_copy_has_the_plugin_lines():
    _assert_in_order(_body_lines(moe_reference.moe_silu_mul_torch), _code_lines(plugin_silu.HPUSiluAndMul.forward_oot),
                     "HPUSiluAndMul.forward_oot")


def test_combine_copy_has_the_engine_and_plugin_lines():
    lines = _body_lines(moe_reference.moe_combine_torch)
    assert lines == ["out = F.sigmoid(expert_gate) * out", "shared_output = out",
                     "combined = (shared_output + fused_output) if shared_output is not None else fused_output",
                     "return combined"]
    assert "out = F.sigmoid(self.expert_gate(x)[0]) * out" in _code_lines(Qwen2MoeMLP.forward)
    assert lines[2] in _code_lines(plugin_moe.patched_fused_moe_forward)


# ---------------------------------------------------------------------------
# The same bits
# ---------------------------------------------------------------------------
def _logits(rows, experts, seed, scale=2.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, experts, generator=g) * scale).to(BF16)


@pytest.mark.parametrize("method_class", [plugin_moe.HPUUnquantizedFusedMoEMethod, plugin_fp8.HPUFp8MoEMethod],
                         ids=["unquantized", "fp8"])
@pytest.mark.parametrize("rows,experts,top_k", [(1, 512, 10), (32, 512, 10), (3, 16, 2), (64, 16, 2)])
def test_route_copy_returns_the_plugin_bits(method_class, rows, experts, top_k):
    """The plugin's apply_monolithic hands the vendor op exactly what the copy returns (ties included)."""
    seen = {}

    def moe_op(x, topk_ids, topk_weights, permuted_weights=True, activation="silu"):
        seen["ids"], seen["weights"] = topk_ids, topk_weights
        return x

    layer = SimpleNamespace(use_grouped_topk=False, custom_routing_function=None, top_k=top_k, activation="silu",
                            moe_config=SimpleNamespace(dp_size=1, is_sequence_parallel=False), moe_op=moe_op)
    method = SimpleNamespace(model_type="qwen4_exp", has_moe_quant_config=False, use_dispatch_fn=False)
    x = torch.randn(rows, 128).to(BF16)
    for seed in range(3):
        router_logits = _logits(rows, experts, seed, scale=(0.25, 2.0, 30.0)[seed])  # coarse logits tie often
        method_class.apply_monolithic(method, layer, x, router_logits)
        ids, weights = moe_reference.moe_route_torch(x, router_logits, layer)
        assert seen["ids"].dtype == torch.int64 and torch.equal(seen["ids"], ids)
        assert seen["weights"].dtype == BF16 and torch.equal(seen["weights"].view(torch.int16),
                                                             weights.view(torch.int16))
        ref_ids, ref_weights = moe_reference.moe_route_reference(router_logits, experts, top_k)
        assert ref_ids.dtype == torch.int32 and torch.equal(ref_ids.long(), ids)
        assert torch.equal(ref_weights.view(torch.int16), weights.view(torch.int16))


def test_dynamic_quant_takes_the_copied_branch_by_default_and_returns_its_bits(monkeypatch):
    """With the default flags dynamic_quant is the copied formula around the vendor cast.

    On CPU, so with CPU torch's division: ``moe_row_quant_cpu_reference``, what the op's CPU stand-in returns.
    """
    assert plugin_ops.gaudi_envs.VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT is False
    assert plugin_ops.gaudi_envs.VLLM_HPU_FP8_JIT_DYNAMIC_QUANT is False
    assert plugin_ops.gaudi_envs.VLLM_HPU_DYNAMIC_QUANT_CGUID is False
    assert not plugin_ops.envs.VLLM_HPU_CGUID_DYNAMIC_QUANT
    calls = []

    def cast(data, scale_inv, *rest):
        calls.append((scale_inv.dtype, tuple(scale_inv.shape), rest))
        return moe_reference.cast_to_fp8_reference(data, scale_inv, plugin_ops.FP8_MAX)

    monkeypatch.setattr(torch.ops.hpu, "cast_to_fp8_v2", cast, raising=False)
    g = torch.Generator().manual_seed(7)
    for rows, hidden in ((1, 128), (5, 2560)):
        data = (torch.randn(rows, hidden, generator=g) * torch.exp(torch.randn(rows, 1, generator=g))).to(BF16)
        data[0] = 0  # a zero row
        got_fp8, got_scale = plugin_ops.dynamic_quant(data)
        want_fp8, want_scale = moe_reference.moe_row_quant_cpu_reference(data, plugin_ops.FP8_MAX)
        assert got_scale.dtype == torch.float32 and torch.equal(got_scale, want_scale)
        assert got_fp8.dtype == torch.float8_e4m3fn
        assert torch.equal(got_fp8.view(torch.uint8), want_fp8.view(torch.uint8))
    # the reciprocal the vendor cast gets is a bf16 [rows, 1] tensor: every step of the scale is rounded to bf16
    assert calls[-1][0] == BF16 and calls[-1][1] == (5, 1) and calls[-1][2] == (False, False, torch.float8_e4m3fn)


def test_dynamic_quant_with_the_cards_division_is_the_kernel_oracle(monkeypatch):
    """What moe_row_quant replaces on a card: the plugin's own dynamic_quant with the card's scalar division.

    A Gaudi2 card computes ``bf16_tensor / FP8_MAX`` as a product with ``bf16(1 / FP8_MAX)``
    (``hpu_arithmetic.py``, first card run 2026-11-20), so the scales of the plugin's function on a card are
    not CPU torch's.  The kernel's oracle is the card's.
    """
    from tests.unit_tests.hc_fused import hpu_arithmetic
    from tests.unit_tests.hc_fused.moe_case import case_by_name, make_inputs

    fp8_max = moe_reference.GAUDI2_FP8_MAX
    monkeypatch.setattr(plugin_ops, "FP8_MAX", fp8_max)
    monkeypatch.setattr(torch.ops.hpu, "cast_to_fp8_v2",
                        lambda data, scale_inv, *rest: moe_reference.cast_to_fp8_reference(data, scale_inv, fp8_max),
                        raising=False)
    for name in ("row_quant_real_r32", "row_quant_real_stress", "row_quant_real_edge"):
        x = make_inputs(case_by_name(name))["x"]
        with hpu_arithmetic.card_arithmetic():
            card_fp8, card_scale = plugin_ops.dynamic_quant(x)
        want_fp8, want_scale = moe_reference.moe_row_quant_reference(x, fp8_max)
        differs = int((card_scale != want_scale).sum())
        assert differs == 0, f"{name}: {differs} of {want_scale.numel()} scales of the oracle are not the card's"
        assert torch.equal(card_fp8.view(torch.uint8), want_fp8.view(torch.uint8)), name


def test_gaudi2_fp8_max_is_the_e4m3fnuz_maximum():
    """FP8_MAX of the plugin on Gaudi2, the only value the row quantization kernel takes."""
    assert torch.finfo(torch.float8_e4m3fnuz).max == moe_reference.GAUDI2_FP8_MAX == 240.0
    source = inspect.getsource(plugin_ops)
    assert "FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max" in source


def test_silu_mul_copy_returns_the_plugin_bits():
    g = torch.Generator().manual_seed(11)
    for rows, width in ((1, 160), (32, 160), (3, 16), (2, 1280)):
        x = (torch.randn(rows, width, generator=g) * 4.0).to(BF16)
        ours = moe_reference.moe_silu_mul_torch(x)
        theirs = plugin_silu.HPUSiluAndMul.forward_oot(x)
        assert ours.dtype == theirs.dtype == BF16 and torch.equal(ours.view(torch.int16), theirs.view(torch.int16))


def test_combine_copy_returns_the_engine_bits():
    """Qwen2MoeMLP.forward's gate line and the fast path's add, on the real classes' code."""
    g = torch.Generator().manual_seed(13)
    rows, hidden = 5, 256
    out = (torch.randn(rows, hidden, generator=g) * 3.0).to(BF16)
    logit = (torch.randn(rows, 1, generator=g) * 3.0).to(BF16)
    fused = (torch.randn(rows, hidden, generator=g) * 3.0).to(BF16)
    x = torch.zeros(rows, 4, dtype=BF16)
    mlp = SimpleNamespace(gate_up_proj=lambda value: (value, None), act_fn=lambda value: value,
                          down_proj=lambda value: (out, None), expert_gate=lambda value: (logit, None))
    shared_output = Qwen2MoeMLP.forward(mlp, x)
    combined = shared_output + fused
    ours = moe_reference.moe_combine_torch(out, logit, fused)
    assert ours.dtype == BF16 and torch.equal(ours.view(torch.int16), combined.view(torch.int16))
    gates = torch.cat([torch.zeros(rows, 3, dtype=BF16), logit], dim=1)
    assert torch.equal(moe_reference.moe_combine_reference(out, gates, fused, 3).view(torch.int16),
                       combined.view(torch.int16))
