# SPDX-License-Identifier: Apache-2.0
"""The MoE block of the tiny Flash-Next model with ``VLLM_HPU_MOE_GLUE``: the flag, the fused path, every fallback.

The plugin's HPU fast path runs on CPU with stand-ins for the vendor ops (``moe_block.py``).  The fused
glue ops run their CPU stand-in, which is the torch oracle (``tests/unit_tests/hc_fused/moe_ops.py``), so a
block built with the flag at 1 must return the bits of the block built with 0.  This proves the wiring:
what each op is given, what the vendor op gets from it, the merged gate weight, and that every reason to
leave the fused path costs one log line and nothing else.  The kernels themselves are proven on the TPC
simulator (``tests/standalone/hc_fused/test_moe_simulation.py``).

One ``cpu_model`` may be open at a time, so each test builds, runs and closes one model after the other.
Run this file in its own pytest process: importing it switches every model of the process to the plugin's
MoE path (``moe_block.use_plugin_fast_path``).
"""

from pathlib import Path

from tests.unit_tests.models.flash_next import moe_block

moe_block.use_plugin_fast_path()

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.envs as gaudi_envs  # noqa: E402
import vllm_gaudi.extension.ops as plugin_ops  # noqa: E402
import vllm_gaudi.ops.hpu_fused_moe as fused_moe  # noqa: E402
from tests.unit_tests.hc_fused import moe_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from tests.unit_tests.models.flash_next.cpu_model import cpu_model  # noqa: E402
from tests.unit_tests.models.flash_next.moe_block import OpRecorder, moe_blocks, to_fp8_experts  # noqa: E402

BF16 = torch.bfloat16
FLAG = "VLLM_HPU_MOE_GLUE"
OFF_LINE = "MoE glue (VLLM_HPU_MOE_GLUE=1) is off for "
GLUE_OPS = "flashinfer_gaudi.moe_"
# One MoE block forward on the plugin's fast path with the flag at 0, recorded before the glue existed
# (tiny fixture, CPU torch, the vendor ops as one entry each): (ops, non-view ops, sha256 of the op names)[:16].
FLAG_OFF_OPS = {"bf16": (28, 16, "72e208cbedd5b0b5"), "fp8": (37, 24, "5c4f995594120e29")}
# The same forward with the flag at 1. No custom op is given a view (a card refuses one): the monolithic method
# returns the vendor op's output without its output.view(...) (one view less than before), and the FP8 block
# copies the input of moe_row_quant, which is a view on every call (one aten.clone more).
FLAG_ON_OPS = {"bf16": (16, 8), "fp8": (18, 10)}
ROWS = (1, 2, 32, 300)
# Ops that read a tensor's data on the host or whose output shape depends on data.
HOST_READS = ("aten._local_scalar_dense", "aten.item", "aten.nonzero", "aten.masked_select", "aten.unique",
              "aten.equal", "aten.is_nonzero", "aten.where.default")


def _inputs(rows: int, hidden: int, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + rows)
    return (torch.randn(rows, hidden, generator=g) * torch.exp(torch.randn(rows, 1, generator=g))).to(BF16)


def _hidden_size(model) -> int:
    return model.language_model.model.config.hidden_size


def _run_blocks(tiny_checkpoint, mode: str, rows=ROWS, mutate=None):
    """Build the tiny model with the environment as it is now and run every MoE block on fixed inputs.

    Returns ``(outputs[block][rows], recorders[rows] of block 0, glue states, blocks' runner names)``.
    ``mutate(model)`` runs after the build (and after the FP8 conversion) and before any forward.
    """
    with cpu_model(tiny_checkpoint) as (_, model):
        blocks = moe_blocks(model)
        if mode == "fp8":
            for block in blocks:
                to_fp8_experts(block)
        if mutate is not None:
            mutate(model)
        hidden = _hidden_size(model)
        outputs, recorders = [], {}
        for index, block in enumerate(blocks):
            per_rows = {}
            for count in rows:
                x = _inputs(count, hidden, seed=3 + index)
                with torch.no_grad(), OpRecorder() as recorder:
                    out = block(x)
                per_rows[count] = out.clone()
                if index == 0:
                    recorders[count] = recorder
            outputs.append(per_rows)
        states = [getattr(block.experts, "_hpu_moe_glue", None) for block in blocks]
        names = [block.experts.layer_name for block in blocks]
        buffers = [dict(block.experts.named_buffers(recurse=False)) for block in blocks]
    return outputs, recorders, states, names, buffers


@pytest.fixture
def lines(monkeypatch):
    """Every warning the MoE module logs."""
    seen: list[str] = []
    monkeypatch.setattr(fused_moe.logger, "warning", lambda message, *args: seen.append(message % args))
    return seen


# --------------------------------------------------------------------------------------------------------------
# 1. the flag


def test_flag_default_and_value(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert gaudi_envs.VLLM_HPU_MOE_GLUE == 0
    monkeypatch.setenv(FLAG, "1")
    assert gaudi_envs.VLLM_HPU_MOE_GLUE == 1


def test_flag_is_declared_after_the_hyper_connection_flag():
    names = list(gaudi_envs.environment_variables)
    assert names.index(FLAG) == names.index("VLLM_HPU_HC_FUSED") + 1
    source = Path(gaudi_envs.__file__).read_text(encoding="utf-8")
    assert "    VLLM_HPU_MOE_GLUE: int = 0" in source


# --------------------------------------------------------------------------------------------------------------
# 2. flag off: nothing changes


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
@pytest.mark.parametrize("value", [None, "0", "2"])
def test_flag_off_op_sequence_is_unchanged(tiny_checkpoint, monkeypatch, lines, mode, value):
    """Flag off (unset, 0 or any value but 1): the block dispatches exactly the ops it dispatched before the glue."""
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    _, recorders, states, _, buffers = _run_blocks(tiny_checkpoint, mode, rows=(1, 32))
    for count, recorder in recorders.items():
        assert recorder.summary() == FLAG_OFF_OPS[mode], (
            f"the flag-off op sequence of the {mode} block at {count} rows changed: (ops, non-view ops, digest) is "
            f"{recorder.summary()}, was {FLAG_OFF_OPS[mode]}")
        assert recorder.count(GLUE_OPS) == 0
    assert states == [None] * 4  # no state object, no merged weight
    assert all("_hpu_moe_glue_gate_weight" not in names for names in buffers)
    assert lines == []


# --------------------------------------------------------------------------------------------------------------
# 3. flag on: the same bits with fewer ops


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_flag_on_block_output_is_bit_identical(tiny_checkpoint, monkeypatch, lines, mode):
    monkeypatch.setenv(FLAG, "0")
    torch_out, torch_recorders, _, _, _ = _run_blocks(tiny_checkpoint, mode)
    monkeypatch.setenv(FLAG, "1")
    fused_out, fused_recorders, states, names, _ = _run_blocks(tiny_checkpoint, mode)

    assert lines == []
    assert [state is not None and state.ok is True and not state.logged for state in states] == [True] * 4
    assert [state.layer_name for state in states] == names and len(set(names)) == 4
    for block in range(4):
        for count in ROWS:
            fused, plain = fused_out[block][count], torch_out[block][count]
            assert fused.dtype == plain.dtype == BF16 and fused.shape == plain.shape
            assert torch.equal(fused.view(torch.int16), plain.view(torch.int16)), f"block {block}, {count} rows"
    # the output is not trivially zero: the routed experts and the shared expert both contribute
    assert float(torch_out[0][32].float().abs().mean()) > 1e-4
    for count in ROWS:
        ops, non_view, _ = fused_recorders[count].summary()
        assert (ops, non_view) == FLAG_ON_OPS[mode], fused_recorders[count].names
        assert (torch_recorders[count].summary()[0], torch_recorders[count].summary()[1]) == FLAG_OFF_OPS[mode][:2]


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_op_count_before_and_after(tiny_checkpoint, monkeypatch, mode):
    """What the flag removes: the block's element-wise ops; what stays: three GEMMs and the vendor op."""
    monkeypatch.setenv(FLAG, "1")
    _, recorders, _, _, _ = _run_blocks(tiny_checkpoint, mode, rows=(1, 32))
    for count, recorder in recorders.items():
        names = recorder.names
        print(f"\nMoE block ({mode}, {count} rows): {FLAG_OFF_OPS[mode][0]} ops ({FLAG_OFF_OPS[mode][1]} non-view) "
              f"with the flag at 0, {recorder.summary()[0]} ({recorder.summary()[1]} non-view) at 1: {names}")
        expected_glue = {"flashinfer_gaudi.moe_route.default": 1, "flashinfer_gaudi.moe_silu_mul.default": 1,
                         "flashinfer_gaudi.moe_combine.default": 1,
                         "flashinfer_gaudi.moe_row_quant.default": int(mode == "fp8")}
        assert {name: names.count(name) for name in expected_glue} == expected_glue
        assert names.count("hpu.mixture_of_experts") == 1 and names.count("hpu.cast_to_fp8_v2") == 0
        # three GEMMs: the merged gate (router rows and the shared-gate row), gate_up and down of the shared expert
        assert names.count("aten.linear.default") + names.count("aten.mm.default") == 3
        # the one cast left is the int32 -> int64 widening of the ids for the vendor op
        assert names.count("aten._to_copy.default") == 1
        for gone in ("aten._softmax.default", "aten.topk.default", "aten.sum.dim_IntList", "aten.div_.Tensor",
                     "aten.silu.default", "aten.sigmoid.default", "aten.mul.Tensor", "aten.add.Tensor",
                     "aten.abs.default", "aten.max.dim", "aten.div.Tensor", "aten.reciprocal.default"):
            assert gone not in names, gone
        assert not [name for name in names if name.startswith(HOST_READS)]  # no host read, no data-dependent shape
    assert recorders[1].names == recorders[32].names  # the same graph for every row count


def test_merged_gate_weight_is_built_at_load(tiny_checkpoint, monkeypatch):
    """One non-persistent buffer per block: the router rows, then the shared-gate row.  Not a parameter."""
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (_, model):
        parameters = sum(p.numel() for p in model.parameters())
        for block in moe_blocks(model):
            runner = block.experts
            merged = runner._hpu_moe_glue_gate_weight  # there before any forward
            assert merged.dtype == BF16 and tuple(merged.shape) == (16 + 1, 128) and merged.is_contiguous()
            assert torch.equal(merged[:16], block.gate.weight)
            assert torch.equal(merged[16:], block.shared_expert_gate.weight)
            assert "_hpu_moe_glue_gate_weight" not in runner.state_dict()
            assert "_hpu_moe_glue_gate_weight" not in dict(runner.named_parameters())
            assert runner._hpu_moe_glue.ok is None  # the layer check runs on the first call
            with torch.no_grad():
                block(_inputs(1, 128))
            assert runner._hpu_moe_glue.ok is True
            assert runner._hpu_moe_glue.experts == 16 and runner._hpu_moe_glue.top_k == 2
            assert runner._hpu_moe_glue_gate_weight is merged  # built once
        state_keys = [key for key in model.state_dict() if "moe_glue" in key]
    monkeypatch.setenv(FLAG, "0")
    with cpu_model(tiny_checkpoint) as (_, model):
        assert sum(p.numel() for p in model.parameters()) == parameters
    assert state_keys == []


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_what_the_ops_and_the_vendor_op_are_given(tiny_checkpoint, monkeypatch, lines, mode):
    """Every op gets what its HPU validation takes, and the vendor op gets the routing op's outputs."""
    monkeypatch.setenv(FLAG, "1")
    from flashinfer_gaudi import moe_glue

    seen: dict[str, list] = {name: [] for name in moe_glue.OP_NAMES}
    real = {name: getattr(moe_glue, name) for name in moe_glue.OP_NAMES}

    def matrix(tensor, columns=None):
        assert tensor.dtype == BF16 and tensor.dim() == 2 and tensor.is_contiguous() and tensor.shape[0] >= 1
        assert columns is None or tensor.shape[1] == columns

    def route(gates, experts, top_k):
        matrix(gates, 17)
        assert (experts, top_k) == (16, 2) and type(experts) is int and type(top_k) is int
        out = real["moe_route"](gates, experts, top_k)
        seen["moe_route"].append((gates, out))
        return out

    def silu_mul(gate_up):
        matrix(gate_up, 128)
        out = real["moe_silu_mul"](gate_up)
        seen["moe_silu_mul"].append((gate_up, out))
        return out

    def combine(shared, gates, routed, gate_column):
        matrix(shared, 128), matrix(gates, 17), matrix(routed, 128)
        assert gate_column == 16 and type(gate_column) is int
        out = real["moe_combine"](shared, gates, routed, gate_column)
        seen["moe_combine"].append((shared, gates, routed, out))
        return out

    def row_quant(x, fp8_max):
        matrix(x, 128)
        assert fp8_max == float(plugin_ops.FP8_MAX) and type(fp8_max) is float
        out = real["moe_row_quant"](x, fp8_max)
        seen["moe_row_quant"].append((x, out))
        return out

    for name, function in (("moe_route", route), ("moe_silu_mul", silu_mul), ("moe_combine", combine),
                           ("moe_row_quant", row_quant)):
        monkeypatch.setattr(moe_glue, name, function)

    with cpu_model(tiny_checkpoint) as (_, model):
        block = moe_blocks(model)[0]
        if mode == "fp8":
            to_fp8_experts(block)
        x = _inputs(5, _hidden_size(model))
        moe_block.VENDOR_CALLS.clear()
        with torch.no_grad():
            out = block(x)
        merged = block.experts._hpu_moe_glue_gate_weight
        gates = torch.nn.functional.linear(x, merged)
        gate_up = block.shared_expert.gate_up_proj(x)[0]

    assert lines == []
    assert [len(calls) for calls in seen.values()] == [1, 1, 1, int(mode == "fp8")]
    (route_gates, (ids, weights)), = seen["moe_route"]
    assert torch.equal(route_gates, gates)  # one GEMM gives the router logits and the shared-gate logit
    (silu_input, act), = seen["moe_silu_mul"]
    assert torch.equal(silu_input, gate_up)
    (shared, combine_gates, routed, combined), = seen["moe_combine"]
    assert combine_gates is route_gates and torch.equal(out, combined)
    # the vendor op: the op's weights as they are, its int32 ids widened to int64, and (FP8) its values and scale
    (call, ) = moe_block.VENDOR_CALLS
    assert ids.dtype == torch.int32 and call["expert_routing_table"].dtype == torch.int64
    assert torch.equal(call["expert_routing_table"], ids.long())
    assert call["router_weights"].dtype == BF16 and torch.equal(call["router_weights"], weights)
    assert torch.equal(routed, moe_block.mixture_of_experts(**call))
    if mode == "fp8":
        (quant_input, (x_fp8, scale)), = seen["moe_row_quant"]
        assert torch.equal(quant_input, x)
        assert call["hidden_states"] is x_fp8 and call["d_scale_hidden_states"] is scale
    else:
        assert torch.equal(call["hidden_states"], x)


def test_fp8_glue_gives_the_vendor_op_the_arguments_of_the_torch_path(tiny_checkpoint, monkeypatch):
    """The FP8 branch calls the vendor op itself: every argument must be the one the plugin's op wrapper passes."""
    calls = {}
    for value in ("0", "1"):
        monkeypatch.setenv(FLAG, value)
        with cpu_model(tiny_checkpoint) as (_, model):
            block = moe_blocks(model)[0]
            to_fp8_experts(block)
            moe_block.VENDOR_CALLS.clear()
            with torch.no_grad():
                block(_inputs(7, _hidden_size(model)))
            (calls[value], ) = moe_block.VENDOR_CALLS
    plain, fused = calls["0"], calls["1"]
    assert list(plain) == list(fused)  # the same keywords in the same order
    for key, value in plain.items():
        other = fused[key]
        if isinstance(value, torch.Tensor):
            assert value.dtype == other.dtype and torch.equal(value.float(), other.float()), key
        elif isinstance(value, (tuple, list)):
            assert len(value) == len(other), key
            assert all(torch.equal(a.float(), b.float()) for a, b in zip(value, other)), key
        else:
            assert value == other, key


# --------------------------------------------------------------------------------------------------------------
# 4. every way out of the fused path: one line, the torch result


def _fallback(tiny_checkpoint, monkeypatch, lines, mode="bf16", mutate=None, rows=(1, 32)):
    """(flag-off outputs, flag-on outputs, states, names) of a build in which `mutate` makes the glue unusable."""
    monkeypatch.setenv(FLAG, "0")
    torch_out, _, _, _, _ = _run_blocks(tiny_checkpoint, mode, rows=rows, mutate=mutate)
    assert lines == []
    monkeypatch.setenv(FLAG, "1")
    fused_out, recorders, states, names, _ = _run_blocks(tiny_checkpoint, mode, rows=rows, mutate=mutate)
    for block in range(4):
        for count in rows:
            assert torch.equal(fused_out[block][count].view(torch.int16), torch_out[block][count].view(torch.int16))
    return recorders, states, names


def _assert_one_line_per_layer(lines, names, reason):
    assert len(lines) == 4, lines
    for line, name in zip(lines, names):
        assert line.startswith(OFF_LINE + name + ": ") and line.endswith("; using the torch path"), line
        assert reason in line, line


def test_no_kernel_for_the_device_falls_back(tiny_checkpoint, monkeypatch, lines):
    """A process without the extension: no op has a kernel, every block logs once and runs torch."""
    from flashinfer_gaudi import moe_glue

    monkeypatch.setattr(moe_glue, "moe_glue_missing_ops", lambda device: moe_glue.OP_NAMES)
    recorders, states, names = _fallback(tiny_checkpoint, monkeypatch, lines)
    _assert_one_line_per_layer(lines, names, "no cpu kernel for moe_route, moe_silu_mul, moe_combine")
    assert all(state.ok is False and state.logged for state in states)
    assert recorders[1].summary() == FLAG_OFF_OPS["bf16"]  # the torch path, op for op


def test_a_kernel_missing_from_the_library_falls_back(tiny_checkpoint, monkeypatch, lines):
    """The library was built without one kernel (it did not compile): the loader names what is missing."""
    from flashinfer_gaudi import _native

    monkeypatch.setattr(_native, "moe_glue_extension_loaded", lambda: False)
    monkeypatch.setattr(_native, "moe_glue_missing_kernels", lambda: ("moe_route", ))
    monkeypatch.setattr(fused_moe, "_moe_glue_library_checked_device", lambda device: True)
    recorders, states, names = _fallback(tiny_checkpoint, monkeypatch, lines)
    _assert_one_line_per_layer(lines, names, "libhc_fused_gaudi2.so has no kernel for moe_route")
    assert recorders[32].summary() == FLAG_OFF_OPS["bf16"]


def test_a_library_that_lacks_only_the_quantization_kernel_still_serves_bf16_blocks(tiny_checkpoint, monkeypatch,
                                                                                    lines):
    from flashinfer_gaudi import _native

    monkeypatch.setattr(_native, "moe_glue_extension_loaded", lambda: False)
    monkeypatch.setattr(_native, "moe_glue_missing_kernels", lambda: ("moe_row_quant", ))
    monkeypatch.setattr(fused_moe, "_moe_glue_library_checked_device", lambda device: True)
    monkeypatch.setenv(FLAG, "1")
    _, _, states, _, _ = _run_blocks(tiny_checkpoint, "bf16", rows=(1, ))
    assert lines == [] and all(state.ok is True for state in states)
    _, _, states, names, _ = _run_blocks(tiny_checkpoint, "fp8", rows=(1, ))
    _assert_one_line_per_layer(lines, names, "libhc_fused_gaudi2.so has no kernel for moe_row_quant")


def test_kernels_that_were_never_asked_for_at_startup_fall_back(tiny_checkpoint, monkeypatch, lines):
    """On a Gaudi device the ops may have HPU kernels (VLLM_HPU_HC_FUSED loaded the shared extension) although
    nobody asked the library which MoE kernels it serves: the block does not guess."""
    from flashinfer_gaudi import _native

    assert _native.moe_glue_extension_loaded() is None  # this CPU process never called the loader
    monkeypatch.setattr(fused_moe, "_moe_glue_library_checked_device", lambda device: True)
    recorders, states, names = _fallback(tiny_checkpoint, monkeypatch, lines)
    _assert_one_line_per_layer(lines, names, "load_moe_glue_extension was not called")
    assert recorders[1].summary() == FLAG_OFF_OPS["bf16"]


def test_an_fp8_layer_needs_the_quantization_kernel_and_a_bf16_layer_does_not(tiny_checkpoint, monkeypatch, lines):
    from flashinfer_gaudi import moe_glue

    monkeypatch.setattr(moe_glue, "moe_glue_missing_ops", lambda device: ("moe_row_quant", ))
    monkeypatch.setenv(FLAG, "1")
    _, recorders, states, _, _ = _run_blocks(tiny_checkpoint, "bf16", rows=(1, ))
    assert lines == [] and all(state.ok is True for state in states)
    _, recorders, states, names, _ = _run_blocks(tiny_checkpoint, "fp8", rows=(1, ))
    _assert_one_line_per_layer(lines, names, "no cpu kernel for moe_row_quant")
    assert recorders[1].summary() == FLAG_OFF_OPS["fp8"]


def test_an_op_that_raises_falls_back_and_stays_off(tiny_checkpoint, monkeypatch, lines):
    from flashinfer_gaudi import moe_glue

    calls = []

    def boom(*args):
        calls.append(1)
        raise RuntimeError("synStatus 26 [Generic failure]")

    monkeypatch.setattr(moe_glue, "moe_combine", boom)
    recorders, states, names = _fallback(tiny_checkpoint, monkeypatch, lines)
    _assert_one_line_per_layer(lines, names, "a fused op raised RuntimeError: synStatus 26 [Generic failure]")
    assert len(calls) == 4  # once per block: the second call of a block does not try again
    assert recorders[32].summary() == FLAG_OFF_OPS["bf16"]


def test_an_input_the_kernels_do_not_take_falls_back(tiny_checkpoint, monkeypatch, lines):
    """fp32 hidden states (the fp32 debug build of the model): the call is outside what the kernels accept."""
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (_, model):
        block = moe_blocks(model)[0]
        state = block.experts._hpu_moe_glue
        x = _inputs(3, _hidden_size(model))
        with torch.no_grad():
            fused = block(x)
        assert state.ok is True and lines == []
        reason = fused_moe._moe_glue_call_reason(state, block.experts, x.float(), x.float())
        assert reason == "hidden states are torch.float32, not bfloat16"
        assert fused_moe._moe_glue_call_reason(state, block.experts, x[:, :64].contiguous(), x) is not None
        assert fused_moe._moe_glue_call_reason(state, block.experts, x.unsqueeze(0), x) is not None
        strided = x.t().contiguous().t()  # not contiguous
        assert fused_moe._moe_glue_call_reason(state, block.experts, strided, x) is not None
        assert fused_moe._moe_glue_call_reason(state, block.experts, x, x) is None


LAYER_MUTATIONS = {
    "grouped_topk": (lambda block: setattr(block.experts.routed_experts, "use_grouped_topk", True), "grouped top-k"),
    "routed_scaling": (lambda block: setattr(block.experts, "routed_scaling_factor", 2.0), "routed_scaling_factor"),
    "no_shared_gate": (lambda block: setattr(block.shared_expert, "expert_gate", None), "shared expert gate"),
    "gate_bias": (lambda block: setattr(block.gate, "bias", torch.nn.Parameter(torch.zeros(16, dtype=BF16))), "bias"),
}


@pytest.mark.parametrize("name", list(LAYER_MUTATIONS))
def test_layer_checks_name_the_reason(tiny_checkpoint, monkeypatch, name):
    """The layer check runs on the built block: each precondition has its own reason."""
    mutate, text = LAYER_MUTATIONS[name]
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (_, model):
        block = moe_blocks(model)[0]
        assert fused_moe._moe_glue_layer_reason(block.experts) is None
        mutate(block)
        reason = fused_moe._moe_glue_layer_reason(block.experts)
        assert reason is not None and text in reason, reason


@pytest.mark.parametrize("flag,text", [("VLLM_HPU_FP8_JIT_DYNAMIC_QUANT", "VLLM_HPU_FP8_JIT_DYNAMIC_QUANT"),
                                       ("VLLM_HPU_CGUID_DYNAMIC_QUANT", "VLLM_HPU_CGUID_DYNAMIC_QUANT"),
                                       ("VLLM_HPU_DYNAMIC_QUANT_CGUID", "VLLM_HPU_DYNAMIC_QUANT_CGUID"),
                                       ("VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT", "VLLM_HPU_TP2_NATIVE_DYNAMIC_QUANT")])
def test_fp8_layer_is_off_when_dynamic_quant_takes_another_branch(tiny_checkpoint, monkeypatch, flag, text):
    """The fused quantization replaces the default branch of dynamic_quant only."""
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (_, model):
        block = moe_blocks(model)[0]
        to_fp8_experts(block)
        assert fused_moe._moe_glue_layer_reason(block.experts) is None
        monkeypatch.setenv(flag, "1")
        reason = fused_moe._moe_glue_layer_reason(block.experts)
        assert reason is not None and text in reason, reason


def test_fp8_layer_checks(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (_, model):
        block = moe_blocks(model)[0]
        to_fp8_experts(block)
        runner = block.experts
        op = runner.routed_experts.moe_op
        assert fused_moe._moe_glue_layer_reason(runner) is None
        op.w13_input_scale = torch.ones(1)
        assert "static input scale" in fused_moe._moe_glue_layer_reason(runner)
        op.w13_input_scale = None
        # on a Gaudi device the kernel takes one fp8 range only
        monkeypatch.setattr(plugin_ops, "FP8_MAX", 448.0)
        assert fused_moe._moe_glue_layer_reason(runner, device_type="hpu") is not None
        assert "240" in fused_moe._moe_glue_layer_reason(runner, device_type="hpu")
        monkeypatch.setattr(plugin_ops, "FP8_MAX", 240.0)
        monkeypatch.setattr(fused_moe, "_moe_glue_library_checked_device", lambda device: False)
        reason = fused_moe._moe_glue_layer_reason(runner, device_type="hpu")
        assert reason is not None and "no hpu kernel" in reason  # a CPU process has no HPU kernel


def test_explicit_sigmoid_silu_keeps_the_torch_path(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (_, model):
        runner = moe_blocks(model)[0].experts
        monkeypatch.setenv("VLLM_HPU_EXPLICIT_SIGMOID_SILU", "1")
        assert "VLLM_HPU_EXPLICIT_SIGMOID_SILU" in fused_moe._moe_glue_layer_reason(runner)


def test_fp32_model_falls_back_with_one_line_per_layer(tiny_checkpoint, monkeypatch, lines):
    """The fp32 debug build: the merged weight is fp32, the kernels are bf16 only."""
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint, fp32=True) as (_, model):
        blocks = moe_blocks(model)
        x = _inputs(2, _hidden_size(model)).float()
        with torch.no_grad():
            for block in blocks:
                block(x)
                block(x)
        names = [block.experts.layer_name for block in blocks]
    _assert_one_line_per_layer(lines, names, "bfloat16")


@pytest.mark.parametrize("merge", ["0", "1"], ids=["separate_gate_up", "merged_gate_up"])
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_no_glue_op_is_ever_given_a_view(tiny_checkpoint, monkeypatch, lines, mode, merge):
    """Every tensor a MoE glue op gets is a tensor of its own, not a view.

    The lazy bridge hands a view to a custom op as a graph input without data, also with the accumulation
    thread off (card probe probe_view_inputs.py, 2026-11-20: 7 of 7 view cases raised at mode 0). The FP8 block
    gives the row quantization ``x.view(-1, H)``; the op must get a copy of it.
    """
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE_MERGE_GATE_UP", merge)
    from flashinfer_gaudi import moe_glue

    views: list[tuple[str, int]] = []
    calls: list[str] = []
    for name in moe_glue.OP_NAMES:
        real = getattr(moe_glue, name)

        def wrapped(*args, _name=name, _real=real):
            calls.append(_name)
            for index, arg in enumerate(args):
                if isinstance(arg, torch.Tensor) and (arg._is_view() or not arg.is_contiguous()):
                    views.append((_name, index))
            return _real(*args)

        monkeypatch.setattr(moe_glue, name, wrapped)
    outputs, _, states, _, _ = _run_blocks(tiny_checkpoint, mode, rows=(1, 32))
    assert lines == []
    assert [state.ok for state in states] == [True] * 4
    assert set(calls) == {"moe_route", "moe_silu_mul", "moe_combine"} | ({"moe_row_quant"} if mode == "fp8" else set())
    assert views == [], f"glue ops were given views: {sorted(set(views))}"
