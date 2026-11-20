# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MoE glue helpers in ``vllm_gaudi/ops/hpu_fused_moe.py`` and ``hpu_fp8.py`` (no model).

The fused path of a whole MoE block is tested on the tiny model in
``tests/unit_tests/models/flash_next/test_moe_glue_layer.py``.  This file holds what needs no model: the
one log line per block, the per-call input check, the load-time merged weight on a bare runner, the
wrapper that reads the flag when a runner is built, and a guard that the FP8 branch keeps calling the
vendor op with the keywords of the plugin's own op wrapper.

Runs on CPU via ``tests/unit_tests/hpu_cpu_stubs``.
"""

import ast
import inspect
import os
import textwrap
from types import SimpleNamespace

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

hpu_cpu_stubs.install()
os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # noqa: E402

import vllm_gaudi.extension.ops as plugin_ops  # noqa: E402
import vllm_gaudi.ops.hpu_fp8 as fp8  # noqa: E402
import vllm_gaudi.ops.hpu_fused_moe as m  # noqa: E402

BF16 = torch.bfloat16
CPU = torch.device("cpu")


@pytest.fixture
def lines(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(m.logger, "warning", lambda message, *args: seen.append(message % args))
    return seen


class BareRunner(torch.nn.Module):
    """The attributes of a MoERunner the load-time step reads."""

    def __init__(self, experts=16, hidden=128, shared_gate=True):
        super().__init__()
        self.layer_name = "model.layers.3.mlp.experts"
        self.gate = SimpleNamespace(weight=torch.randn(experts, hidden).to(BF16))
        expert_gate = SimpleNamespace(weight=torch.randn(1, hidden).to(BF16)) if shared_gate else None
        self._shared_experts = SimpleNamespace(_layer=SimpleNamespace(expert_gate=expert_gate))
        self.routed_experts = SimpleNamespace()


def test_one_line_per_block_in_the_documented_form(lines):
    state = m._MoeGlueState(BareRunner())
    state.log("hidden states are torch.float32, not bfloat16")
    state.log("another reason on a later call")
    assert lines == ["MoE glue (VLLM_HPU_MOE_GLUE=1) is off for model.layers.3.mlp.experts: hidden states are "
                     "torch.float32, not bfloat16; using the torch path"]
    assert state.ok is None  # a call the kernels do not take does not turn the block off
    state.off("a fused op raised RuntimeError: boom")
    assert state.ok is False and len(lines) == 1


@pytest.mark.parametrize("tensor,columns,text", [
    (torch.zeros(3, 128, dtype=BF16), 128, None),
    (torch.zeros(3, 160, dtype=BF16), None, None),
    (torch.zeros(3, 128), 128, "torch.float32, not bfloat16"),
    (torch.zeros(3, 64, dtype=BF16), 128, "(3, 64), not [tokens, 128]"),
    (torch.zeros(1, 3, 128, dtype=BF16), 128, "not [tokens, 128]"),
    (torch.zeros(0, 128, dtype=BF16), 128, "not [tokens, 128]"),
    (torch.zeros(128, 3, dtype=BF16).t(), 128, "not contiguous"),
    (None, 128, "not a tensor"),
])
def test_matrix_reason(tensor, columns, text):
    reason = m._moe_glue_matrix_reason(tensor, "hidden states", columns, CPU)
    assert (reason is None) if text is None else (reason is not None and text in reason), reason


def test_matrix_reason_reads_no_data():
    """Shape, dtype, layout and device only: it works on a tensor without data."""
    meta = torch.empty(3, 128, dtype=BF16, device="meta")
    assert "meta" in m._moe_glue_matrix_reason(meta, "hidden states", 128, CPU)
    assert m._moe_glue_matrix_reason(meta, "hidden states", 128, torch.device("meta")) is None


def test_prepare_does_nothing_without_a_state():
    """Flag 0: the routed experts carry no state and the load-time step touches nothing."""

    class Untouchable:

        def __getattr__(self, name):
            if name == "_hpu_moe_glue":
                raise AttributeError(name)
            raise AssertionError(f"the flag-off load path read {name}")

    m._moe_glue_prepare(Untouchable())


def test_prepare_builds_the_merged_weight_as_a_buffer_that_is_not_saved():
    runner = BareRunner()
    m._moe_glue_attach(runner)
    assert runner.routed_experts._hpu_moe_glue is runner._hpu_moe_glue
    m._moe_glue_prepare(runner.routed_experts)
    merged = runner._hpu_moe_glue_gate_weight
    assert tuple(merged.shape) == (17, 128) and merged.dtype == BF16 and merged.is_contiguous()
    assert torch.equal(merged[:16], runner.gate.weight)
    assert torch.equal(merged[16:], runner._shared_experts._layer.expert_gate.weight)
    assert "_hpu_moe_glue_gate_weight" in dict(runner.named_buffers()) and runner.state_dict() == {}
    assert runner._hpu_moe_glue.prepare_error is None
    # a second load (a weight reload) builds it again from the weights as they are then
    runner.gate.weight = runner.gate.weight * 2
    m._moe_glue_prepare(runner.routed_experts)
    assert torch.equal(runner._hpu_moe_glue_gate_weight[:16], runner.gate.weight)


def test_prepare_never_raises_and_keeps_the_reason():
    runner = BareRunner(shared_gate=False)
    m._moe_glue_attach(runner)
    m._moe_glue_prepare(runner.routed_experts)
    assert not hasattr(runner, "_hpu_moe_glue_gate_weight")
    assert "AttributeError" in runner._hpu_moe_glue.prepare_error


@pytest.mark.parametrize("value,attached", [(None, False), ("0", False), ("1", True), ("2", False)])
def test_the_flag_is_read_when_a_runner_is_built(monkeypatch, value, attached):
    """MoERunner.__init__ is wrapped: exactly 1 attaches the glue state, anything else attaches nothing."""
    assert MoERunner.__init__ is m._patched_default_moe_runner_init
    built = []
    monkeypatch.setattr(m, "_orig_default_moe_runner_init", lambda self, *args, **kwargs: built.append((args, kwargs)))
    if value is None:
        monkeypatch.delenv("VLLM_HPU_MOE_GLUE", raising=False)
    else:
        monkeypatch.setenv("VLLM_HPU_MOE_GLUE", value)
    runner = BareRunner()
    m._patched_default_moe_runner_init(runner, "name", moe_config=None)
    assert built == [(("name", ), {"moe_config": None})]  # the engine's constructor got its arguments unchanged
    assert hasattr(runner, "_hpu_moe_glue") is attached
    assert hasattr(runner.routed_experts, "_hpu_moe_glue") is attached
    # a later change of the environment does not reach a built block
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE", "1")
    assert hasattr(runner, "_hpu_moe_glue") is attached


def test_the_two_expert_methods_are_the_only_ones_with_a_fused_glue():
    assert m.HPUUnquantizedFusedMoEMethod.__dict__["_hpu_moe_glue_kind"] == "bf16"
    assert fp8.HPUFp8MoEMethod.__dict__["_hpu_moe_glue_kind"] == "fp8"

    class Subclass(fp8.HPUFp8MoEMethod):
        pass

    assert Subclass.__dict__.get("_hpu_moe_glue_kind") is None  # a subclass with its own apply_monolithic is not assumed


def _vendor_call_keywords(function, which: int) -> list[str]:
    """The keyword names of the `which`-th ``torch.ops.hpu.mixture_of_experts(...)`` call in a function's source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.ops.hpu.mixture_of_experts"]
    calls.sort(key=lambda node: node.lineno)
    return [keyword.arg for keyword in calls[which].keywords]


def test_fp8_glue_calls_the_vendor_op_with_the_op_wrappers_keywords():
    """``_moe_glue_fp8_experts`` repeats the dynamic-scale call of the plugin's FP8 op wrapper; they must not drift."""
    wrapper = _vendor_call_keywords(plugin_ops.VllmMixtureOfExpertsOpFP8PerChannel.forward, 0)
    glue = _vendor_call_keywords(fp8._moe_glue_fp8_experts, 0)
    assert glue == wrapper
    assert wrapper[:3] == ["hidden_states", "expert_routing_table", "router_weights"]
    assert "d_scale_hidden_states" in wrapper and wrapper[-1] is None  # **kwargs last (chunk size, is_gated)
    source = inspect.getsource(plugin_ops.VllmMixtureOfExpertsOpFP8PerChannel.forward)
    assert "x_fp8, x_scale = dynamic_quant(x)" in source  # the line the fused row quantization replaces
    assert "x_fp8, x_scale = moe_glue.row_quant(x)" in inspect.getsource(fp8._moe_glue_fp8_experts)
