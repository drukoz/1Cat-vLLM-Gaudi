# SPDX-License-Identifier: Apache-2.0
"""``VLLM_HPU_MOE_GLUE_MERGE_GATE_UP``: the shared expert's gate_up GEMM merged into the MoE glue's gate GEMM.

With the glue on (``VLLM_HPU_MOE_GLUE=1``) a block computes the router logits and the shared-gate logit with one
GEMM of a merged weight.  With this flag at 1 that weight also has the shared expert's gate_up rows of this rank,
so the same GEMM gives the gate_up output and ``moe_silu_mul(gates, experts + 1)`` reads it there: one GEMM per
block less.  On the plugin's fast path on CPU (stand-ins for the vendor ops, the torch oracle as the glue ops'
CPU kernels) the block must return the bits of the torch path; a block that cannot merge keeps the separate GEMM
with one line, and the flag does nothing without the glue.

Run this file in its own pytest process (it switches every model of the process to the plugin's MoE path).
"""

from tests.unit_tests.models.flash_next import moe_block

moe_block.use_plugin_fast_path()

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.envs as gaudi_envs  # noqa: E402
import vllm_gaudi.ops.hpu_fused_moe as fused_moe  # noqa: E402
from tests.unit_tests.hc_fused import moe_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from tests.unit_tests.models.flash_next.cpu_model import cpu_model  # noqa: E402
from tests.unit_tests.models.flash_next.moe_block import moe_blocks  # noqa: E402
from tests.unit_tests.models.flash_next.test_moe_glue_layer import (FLAG, FLAG_OFF_OPS, FLAG_ON_OPS, ROWS,  # noqa: E402
                                                                    _inputs, _run_blocks)

BF16 = torch.bfloat16
MERGE = "VLLM_HPU_MOE_GLUE_MERGE_GATE_UP"
MERGE_LINE = "MoE glue (VLLM_HPU_MOE_GLUE_MERGE_GATE_UP=1) keeps the separate gate_up GEMM for "
BUFFER = "_hpu_moe_glue_gate_up_weight"
EXPERTS, HIDDEN, GATE_UP = 16, 128, 128  # the tiny checkpoint at TP1: 16 experts, hidden 128, shared expert 64 wide


@pytest.fixture
def lines(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(fused_moe.logger, "warning", lambda message, *args: seen.append(message % args))
    return seen


def test_flag_default_value_and_place(monkeypatch):
    monkeypatch.delenv(MERGE, raising=False)
    assert gaudi_envs.VLLM_HPU_MOE_GLUE_MERGE_GATE_UP == 0
    monkeypatch.setenv(MERGE, "1")
    assert gaudi_envs.VLLM_HPU_MOE_GLUE_MERGE_GATE_UP == 1
    names = list(gaudi_envs.environment_variables)
    assert names.index(MERGE) == names.index(FLAG) + 1


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_merged_block_output_is_bit_identical_with_one_gemm_less(tiny_checkpoint, monkeypatch, lines, mode):
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.delenv(MERGE, raising=False)
    torch_out, _, _, _, _ = _run_blocks(tiny_checkpoint, mode)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(MERGE, "1")
    merged_out, recorders, states, _, buffers = _run_blocks(tiny_checkpoint, mode)

    assert lines == []
    assert [state.ok is True and state.gate_up_first_column == EXPERTS + 1 for state in states] == [True] * 4
    assert all(BUFFER in names for names in buffers)
    for block in range(4):
        for count in ROWS:
            fused, plain = merged_out[block][count], torch_out[block][count]
            assert torch.equal(fused.view(torch.int16), plain.view(torch.int16)), f"block {block}, {count} rows"
    for count, recorder in recorders.items():
        ops, non_view, _ = recorder.summary()
        # the gate_up GEMM is gone: on CPU a linear is a transposed-weight view and a matmul (2 ops, 1 not a view)
        assert (ops, non_view) == (FLAG_ON_OPS[mode][0] - 2, FLAG_ON_OPS[mode][1] - 1), recorder.names
        names = recorder.names
        # two GEMMs: the merged gate (router, shared gate, gate_up) and the shared expert's down projection
        assert names.count("aten.linear.default") + names.count("aten.mm.default") == 2
        assert names.count("flashinfer_gaudi.moe_silu_mul.default") == 1
    assert recorders[1].names == recorders[32].names


def test_merged_weight_is_built_at_load_and_the_ops_get_one_gemm_output(tiny_checkpoint, monkeypatch, lines):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(MERGE, "1")
    from flashinfer_gaudi import moe_glue

    seen = []
    real_silu, real_route, real_combine = moe_glue.moe_silu_mul, moe_glue.moe_route, moe_glue.moe_combine

    def silu_mul(gate_up, first_column=0):
        seen.append(("silu_mul", gate_up, first_column))
        return real_silu(gate_up, first_column)

    def route(gates, experts, top_k):
        seen.append(("route", gates, experts))
        return real_route(gates, experts, top_k)

    def combine(shared, gates, routed, gate_column):
        seen.append(("combine", gates, gate_column))
        return real_combine(shared, gates, routed, gate_column)

    monkeypatch.setattr(moe_glue, "moe_silu_mul", silu_mul)
    monkeypatch.setattr(moe_glue, "moe_route", route)
    monkeypatch.setattr(moe_glue, "moe_combine", combine)
    with cpu_model(tiny_checkpoint) as (_, model):
        block = moe_blocks(model)[0]
        runner = block.experts
        merged = getattr(runner, BUFFER)  # there before any forward
        assert merged.dtype == BF16 and tuple(merged.shape) == (EXPERTS + 1 + GATE_UP, HIDDEN)
        assert merged.is_contiguous()
        assert torch.equal(merged[:EXPERTS], block.gate.weight)
        assert torch.equal(merged[EXPERTS:EXPERTS + 1], block.shared_expert_gate.weight)
        assert torch.equal(merged[EXPERTS + 1:], block.shared_expert.gate_up_proj.weight)
        assert BUFFER not in runner.state_dict() and BUFFER not in dict(runner.named_parameters())
        x = _inputs(5, HIDDEN)
        with torch.no_grad():
            block(x)
        gates = torch.nn.functional.linear(x, merged)
    assert lines == []
    (kind_s, silu_input, first), (kind_r, route_gates, experts), (kind_c, combine_gates, column) = seen
    assert (kind_s, kind_r, kind_c) == ("silu_mul", "route", "combine")
    assert first == EXPERTS + 1 and type(first) is int and experts == EXPERTS and column == EXPERTS
    assert silu_input is route_gates and combine_gates is route_gates  # one GEMM output for the three ops
    assert torch.equal(silu_input, gates) and tuple(silu_input.shape) == (5, EXPERTS + 1 + GATE_UP)


def test_without_the_glue_the_merge_flag_does_nothing(tiny_checkpoint, monkeypatch, lines):
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.setenv(MERGE, "1")
    _, recorders, states, _, buffers = _run_blocks(tiny_checkpoint, "fp8", rows=(1, 32))
    assert states == [None] * 4 and all(BUFFER not in names for names in buffers)
    for recorder in recorders.values():
        assert recorder.summary() == FLAG_OFF_OPS["fp8"]
    assert lines == []


@pytest.mark.parametrize("damage", ["no_buffer", "odd_rows", "float32"])
def test_a_block_that_cannot_merge_keeps_the_separate_gemm(tiny_checkpoint, monkeypatch, lines, damage):
    """One line per block, and the rest of the glue as without the flag: the same bits, the same ops."""
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.delenv(MERGE, raising=False)
    torch_out, _, _, _, _ = _run_blocks(tiny_checkpoint, "bf16", rows=(1, 32))

    def mutate(model):
        for block in moe_blocks(model):
            runner = block.experts
            if damage == "no_buffer":
                delattr(runner, BUFFER)
            elif damage == "odd_rows":
                setattr(runner, BUFFER, getattr(runner, BUFFER)[:-1].contiguous())
            else:
                setattr(runner, BUFFER, getattr(runner, BUFFER).float())

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(MERGE, "1")
    out, recorders, states, names, _ = _run_blocks(tiny_checkpoint, "bf16", rows=(1, 32), mutate=mutate)
    assert len(lines) == 4 and all(line.startswith(MERGE_LINE) for line in lines)
    assert sorted(line[len(MERGE_LINE):].split(":")[0] for line in lines) == sorted(names)
    assert [state.ok is True and state.gate_up_first_column == 0 for state in states] == [True] * 4
    for block in range(4):
        for count in (1, 32):
            assert torch.equal(out[block][count].view(torch.int16), torch_out[block][count].view(torch.int16))
    for recorder in recorders.values():
        assert recorder.summary()[:2] == FLAG_ON_OPS["bf16"]
