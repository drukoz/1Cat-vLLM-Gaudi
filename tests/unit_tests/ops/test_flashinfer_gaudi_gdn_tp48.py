# SPDX-License-Identifier: Apache-2.0
"""CPU parity and gating tests for the FlashInfer-Gaudi GDN TP4/TP8 tactics."""

from __future__ import annotations

import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from flashinfer_gaudi import clear_backend_policy_override
from flashinfer_gaudi._reference import qwen38_fused_decode_step_direct, recurrent_decode_from_qkv
from vllm_gaudi.ops.flashinfer_gaudi_adapter import maybe_run_gdn_fused_decode_step, maybe_run_gdn_prefill
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_chunk_gated_delta_rule


@pytest.fixture(autouse=True)
def _reset_backend_policy():
    clear_backend_policy_override()
    yield
    clear_backend_policy_override()


def _fused_decode_inputs(value_heads, seed):
    key_heads = value_heads // 3
    dim = 128
    batch = 2
    width = (value_heads + 2 * key_heads) * dim
    generator = torch.Generator().manual_seed(seed)
    packed_qkv = torch.randn(batch, width, dtype=torch.bfloat16, generator=generator) * 0.1
    a = torch.randn(batch, value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    b = torch.randn(batch, value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    A_log = torch.randn(value_heads, dtype=torch.float32, generator=generator) * 0.1 - 2.0
    dt_bias = torch.randn(value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_state = torch.randn(batch, 3, width, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_weight = torch.randn(width, 4, dtype=torch.bfloat16, generator=generator) * 0.05
    ssm_state = torch.randn(batch, value_heads, dim, dim, dtype=torch.float32, generator=generator) * 0.01
    return {
        "key_heads": key_heads,
        "dim": dim,
        "batch": batch,
        "packed_qkv": packed_qkv,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "conv_state": conv_state,
        "conv_weight": conv_weight,
        "ssm_state": ssm_state,
    }


def _reference_fused_decode(inputs):
    """Gating + causal conv as in qwen38_fused_decode_step_direct, then the general recurrent reference."""
    key_heads = inputs["key_heads"]
    value_heads = inputs["ssm_state"].shape[1]
    dim = inputs["dim"]
    batch = inputs["batch"]
    packed_qkv = inputs["packed_qkv"]
    conv_state = inputs["conv_state"].clone()
    reference_state = inputs["ssm_state"].clone()

    gate_input = inputs["a"].to(torch.float32) + inputs["dt_bias"].to(torch.float32)
    softplus = torch.where(gate_input <= 20.0, torch.log1p(torch.exp(gate_input)), gate_input)
    log_decay = -torch.exp(inputs["A_log"].to(torch.float32)) * softplus
    beta = torch.sigmoid(inputs["b"].to(torch.float32)).to(inputs["b"].dtype)

    weights = inputs["conv_weight"].to(torch.float32)
    convolved = conv_state[:, 0, :] * weights[:, 0]
    convolved = convolved + conv_state[:, 1, :] * weights[:, 1]
    convolved = convolved + conv_state[:, 2, :] * weights[:, 2]
    convolved = convolved + packed_qkv * weights[:, 3]
    convolved = F.silu(convolved.to(torch.bfloat16))

    qk_width = 2 * key_heads * dim
    q = convolved[:, :key_heads * dim].reshape(batch, 1, key_heads, dim)
    k = convolved[:, key_heads * dim:qk_width].reshape(batch, 1, key_heads, dim)
    v = convolved[:, qk_width:].reshape(batch, 1, value_heads, dim)
    output, state = recurrent_decode_from_qkv(
        q,
        k,
        v,
        log_decay.reshape(batch, 1, value_heads),
        beta.reshape(batch, 1, value_heads),
        reference_state,
        scale=dim**-0.5,
        use_qk_l2norm=True,
        direct_state=True,
    )
    return output.reshape(batch, value_heads, dim), state


@pytest.mark.parametrize("value_heads", [12, 6])
def test_tp48_fused_decode_matches_general_reference(value_heads):
    inputs = _fused_decode_inputs(value_heads, 20261117)
    batch, dim = inputs["batch"], inputs["dim"]
    expected_output, expected_state = _reference_fused_decode(inputs)

    conv_state = inputs["conv_state"].clone()
    ssm_state = inputs["ssm_state"].clone()
    output, _, next_state = qwen38_fused_decode_step_direct(
        inputs["packed_qkv"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        conv_state,
        inputs["conv_weight"],
        None,
        ssm_state,
        dim**-0.5,
    )

    assert output.shape == (batch, value_heads, dim)
    torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(next_state, expected_state, atol=1e-4, rtol=1e-4)


def _fused_decode_gate_inputs(packed_width, value_heads, batch=1):
    packed = torch.empty(batch, packed_width, dtype=torch.bfloat16)
    a = torch.empty(batch, value_heads, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    A_log = torch.zeros(value_heads, dtype=torch.float32)
    dt_bias = torch.zeros(value_heads, dtype=torch.bfloat16)
    conv_state = torch.empty(batch, 3, packed_width, dtype=torch.bfloat16)
    conv_weight = torch.zeros(packed_width, 4, dtype=torch.bfloat16)
    ssm_state = torch.empty(batch + 2, value_heads, 128, 128, dtype=torch.float32)
    load_indices = torch.arange(1, batch + 1, dtype=torch.int32)
    canned = (torch.empty(batch, value_heads, 128, dtype=torch.bfloat16), conv_state, ssm_state.narrow(0, 1, batch))
    return packed, a, b, A_log, dt_bias, conv_state, conv_weight, ssm_state, load_indices, canned, batch


@pytest.mark.parametrize(
    "packed_width,value_heads,flag",
    [(2560, 12, "VLLM_HPU_FLASHINFER_GDN_TP4"), (1280, 6, "VLLM_HPU_FLASHINFER_GDN_TP8")],
)
def test_tp48_fused_decode_gate_requires_matching_opt_in(packed_width, value_heads, flag):
    (packed, a, b, A_log, dt_bias, conv_state, conv_weight, ssm_state, load_indices, canned,
     batch) = _fused_decode_gate_inputs(packed_width, value_heads)
    base_env = {"VLLM_HPU_FLASHINFER_GDN": "1", "VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE": "1"}

    def call_gate():
        return maybe_run_gdn_fused_decode_step(
            packed,
            a,
            b,
            A_log,
            dt_bias,
            conv_state,
            conv_weight,
            None,
            ssm_state,
            load_indices,
            direct_conv_state=True,
            direct_gdn_state=True,
            direct_state_group_count=1,
            direct_state_group_offset=0,
            scale=128**-0.5,
        )

    with mock.patch.dict(os.environ, base_env), mock.patch(
            "vllm_gaudi.ops.flashinfer_gaudi_adapter.qwen38_fused_decode_step_direct",
            return_value=canned) as run:
        assert call_gate() is None
        run.assert_not_called()

    with mock.patch.dict(os.environ, {**base_env, flag: "1"}), mock.patch(
            "vllm_gaudi.ops.flashinfer_gaudi_adapter.qwen38_fused_decode_step_direct",
            return_value=canned) as run:
        result = call_gate()

    assert result is not None
    assert result[0].shape == (1, batch, value_heads, 128)
    assert run.call_count == 1


@pytest.mark.parametrize("packed_width,value_heads", [(5120, 24)])
def test_tp2_fused_decode_gate_still_requires_tp2_opt_in(packed_width, value_heads):
    (packed, a, b, A_log, dt_bias, conv_state, conv_weight, ssm_state, load_indices, canned,
     _) = _fused_decode_gate_inputs(packed_width, value_heads)
    base_env = {"VLLM_HPU_FLASHINFER_GDN": "1", "VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE": "1"}

    def call_gate():
        return maybe_run_gdn_fused_decode_step(
            packed,
            a,
            b,
            A_log,
            dt_bias,
            conv_state,
            conv_weight,
            None,
            ssm_state,
            load_indices,
            direct_conv_state=True,
            direct_gdn_state=True,
            direct_state_group_count=1,
            direct_state_group_offset=0,
            scale=128**-0.5,
        )

    with mock.patch.dict(os.environ, base_env), mock.patch(
            "vllm_gaudi.ops.flashinfer_gaudi_adapter.qwen38_fused_decode_step_direct",
            return_value=canned) as run:
        assert call_gate() is None
        run.assert_not_called()

    with mock.patch.dict(os.environ, {**base_env, "VLLM_HPU_FLASHINFER_GDN_TP2": "1"}), mock.patch(
            "vllm_gaudi.ops.flashinfer_gaudi_adapter.qwen38_fused_decode_step_direct",
            return_value=canned) as run:
        assert call_gate() is not None
        assert run.call_count == 1


@pytest.mark.parametrize(
    "key_heads,value_heads,flag",
    [(4, 12, "VLLM_HPU_FLASHINFER_GDN_TP4"), (2, 6, "VLLM_HPU_FLASHINFER_GDN_TP8")],
)
def test_tp48_prefill_gate_and_parity(key_heads, value_heads, flag):
    torch.manual_seed(20261117)
    tokens = 256
    q = (torch.randn(1, tokens, key_heads, 128, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    k = (torch.randn(1, tokens, key_heads, 128, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    v = (torch.randn(1, tokens, value_heads, 128, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    log_decay = (-torch.rand(1, tokens, value_heads, dtype=torch.float32) * 0.5).contiguous()
    beta = torch.rand(1, tokens, value_heads, dtype=torch.bfloat16)
    kwargs = dict(
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        chunk_size=128,
        prefill_num_seqs=1,
        prefill_seq_len=tokens,
    )
    base_env = {"VLLM_HPU_FLASHINFER_GDN": "1", "VLLM_HPU_FLASHINFER_GDN_PREFILL": "1"}

    with mock.patch.dict(os.environ, base_env):
        assert maybe_run_gdn_prefill(q, k, v, log_decay, beta, None, **kwargs) is None

    with mock.patch.dict(os.environ, {**base_env, flag: "1"}):
        result = maybe_run_gdn_prefill(q, k, v, log_decay, beta, None, **kwargs)
    assert result is not None
    tactic_output, tactic_state = result

    reference_output, reference_state = hpu_chunk_gated_delta_rule(
        q,
        k,
        v,
        log_decay,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        chunk_size=128,
        prefill_num_seqs=1,
        prefill_seq_len=tokens,
    )
    torch.testing.assert_close(tactic_output.float(), reference_output.float(), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(tactic_state.float(), reference_state.float(), atol=2e-3, rtol=2e-2)
