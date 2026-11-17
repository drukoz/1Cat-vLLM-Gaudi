# SPDX-License-Identifier: Apache-2.0
"""CPU parity tests for the torch PLE gate and dilated short convolution (Flash-Next T4a).

The oracle is Hugging Face's ``Qwen4ExpTextPLELayer`` built from the program fixture
``tests/unit_tests/models/flash_next/tiny_config.json`` (``hc_count 4``, ``hidden_size 128`` so ``C = 512``,
``ple_conv_kernel_size 4``, ``ngram_size 3`` so dilation 3 and ``state_len 9``). ``ple_embedding`` (the n-gram
lookup, out of scope for T4a) is monkeypatched to a fixed embedding tensor so both sides see the same embeddings.
"""

import math
import pathlib
import shutil
import types

import pytest
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig as HFQwen4ExpConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

from vllm_gaudi.models.qwen4_exp.ple_ops import (
    ple_forward_torch,
    ple_gate_torch,
    ple_short_conv_torch,
)

TINY_CONFIG = pathlib.Path(__file__).resolve().parent / "tiny_config.json"
TOL = dict(atol=1e-5, rtol=1e-5)
LENGTHS = [1, 5, 9, 10, 23]


class FixedPLEEmbedding(torch.nn.Module):
    """Stand-in for ``ple_embedding`` returning a fixed embedding tensor."""

    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids, past_key_values):
        return self.embeddings


@pytest.fixture(scope="module")
def ple(tmp_path_factory):
    """A seeded fp32 ``Qwen4ExpTextPLELayer`` from the tiny fixture, plus its PLE geometry."""
    config_dir = tmp_path_factory.mktemp("flash_next_ple_config")
    shutil.copy(TINY_CONFIG, config_dir / "config.json")
    # Build the native transformers config class directly: creating a vLLM
    # EngineConfig for qwen4_exp (done by other test modules) re-registers the
    # model type to vLLM's config classes, which lack the HF-only `seed`.
    text_config = HFQwen4ExpConfig.from_json_file(str(config_dir / "config.json")).text_config
    torch.manual_seed(0)
    layer = Qwen4ExpTextPLELayer(text_config, layer_idx=1, ple_layer_index=0).to(torch.float32).eval()
    return types.SimpleNamespace(
        config=text_config,
        layer=layer,
        hc=text_config.hc_count,
        hidden_size=text_config.hidden_size,
        channels=text_config.hc_count * text_config.hidden_size,
        kernel_size=text_config.ple_conv_kernel_size,
        dilation=text_config.ngram_size,
        state_len=(text_config.ple_conv_kernel_size - 1) * text_config.ngram_size,
        gate_args=lambda key, value, hidden: dict(
            key=key,
            value=value,
            hidden=hidden,
            norm_key_w=layer.norm_key.weight,
            norm_query_w=layer.norm_query.weight,
            norm_conv_w=layer.norm_conv.weight,
            eps=text_config.rms_norm_eps,
        ),
    )


def make_inputs(ple, num_tokens, seed=0):
    """Random PLE embeddings and stream activations, plus HF's key/value projections of the embeddings."""
    torch.manual_seed(seed)
    embeddings = torch.randn(1, num_tokens, ple.config.ple_embed_dim)
    hidden = torch.randn(1, num_tokens, ple.channels)
    key = ple.layer.key_proj(embeddings)
    value = ple.layer.value_proj(embeddings)
    return embeddings, hidden, key, value


def hf_gate_intermediates(ple, embeddings, hidden):
    """HF's own intermediate tensors, with the same expressions as ``Qwen4ExpTextPLELayer.forward``."""
    key_normed = ple.layer.norm_key(ple.layer.key_proj(embeddings)).unflatten(-1, (ple.hc, ple.hidden_size))
    value = ple.layer.value_proj(embeddings)
    query_normed = ple.layer.norm_query(hidden).unflatten(-1, (ple.hc, ple.hidden_size))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(ple.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
    gated = gated_value.flatten(-2)
    return gated, ple.layer.norm_conv(gated)


def hf_layer_output(ple, embeddings, hidden):
    """HF's layer output with ``ple_embedding`` replaced by the fixed embeddings and no cache."""
    original = ple.layer.ple_embedding
    ple.layer.ple_embedding = FixedPLEEmbedding(embeddings)
    try:
        return ple.layer.forward(hidden, input_ids=None, past_key_values=None)
    finally:
        ple.layer.ple_embedding = original


@torch.no_grad()
def test_gate_matches_hf(ple):
    embeddings, hidden, key, value = make_inputs(ple, 5)
    expected_gated, expected_normed = hf_gate_intermediates(ple, embeddings, hidden)

    gated, normed = ple_gate_torch(**ple.gate_args(key[0], value[0], hidden[0]))

    assert gated.shape == (5, ple.channels)
    assert normed.shape == (5, ple.channels)
    torch.testing.assert_close(gated, expected_gated[0], **TOL)
    torch.testing.assert_close(normed, expected_normed[0], **TOL)


@pytest.mark.parametrize("num_tokens", LENGTHS)
@torch.no_grad()
def test_full_sequence_matches_hf(ple, num_tokens):
    embeddings, hidden, key, value = make_inputs(ple, num_tokens)
    zero_state = torch.zeros(ple.state_len, ple.channels)

    output, new_state = ple_forward_torch(**ple.gate_args(key[0], value[0], hidden[0]),
                                          conv_weight=ple.layer.conv1d.weight,
                                          state=zero_state,
                                          dilation=ple.dilation)
    expected = hf_layer_output(ple, embeddings, hidden)

    assert output.shape == (num_tokens, ple.channels)
    torch.testing.assert_close(output, expected[0], **TOL)
    assert new_state.shape == (ple.state_len, ple.channels)


@torch.no_grad()
def test_decode_matches_prefill(ple):
    num_tokens = 23
    _, hidden, key, value = make_inputs(ple, num_tokens)
    zero_state = torch.zeros(ple.state_len, ple.channels)
    conv_weight = ple.layer.conv1d.weight

    gated, normed = ple_gate_torch(**ple.gate_args(key[0], value[0], hidden[0]))
    prefill_conv, prefill_state = ple_short_conv_torch(normed, conv_weight, zero_state, ple.dilation)

    state = zero_state
    decode_rows = []
    for token in range(num_tokens):
        token_out, state = ple_short_conv_torch(normed[token:token + 1], conv_weight, state, ple.dilation)
        decode_rows.append(token_out)
    decode_conv = torch.cat(decode_rows, dim=0)

    torch.testing.assert_close(decode_conv, prefill_conv, **TOL)
    torch.testing.assert_close(state, prefill_state, **TOL)

    full_output, full_state = ple_forward_torch(**ple.gate_args(key[0], value[0], hidden[0]),
                                                conv_weight=conv_weight,
                                                state=zero_state,
                                                dilation=ple.dilation)
    torch.testing.assert_close(gated + decode_conv, full_output, **TOL)
    torch.testing.assert_close(state, full_state, **TOL)


@torch.no_grad()
def test_state_shape_and_carry(ple):
    _, hidden, key, value = make_inputs(ple, 23, seed=1)
    torch.manual_seed(2)
    state = torch.randn(ple.state_len, ple.channels)

    _, normed = ple_gate_torch(**ple.gate_args(key[0], value[0], hidden[0]))
    _, new_state = ple_short_conv_torch(normed, ple.layer.conv1d.weight, state, ple.dilation)

    assert new_state.shape == (9, 512)
    torch.testing.assert_close(new_state, torch.cat([state, normed], dim=0)[-9:], **TOL)
