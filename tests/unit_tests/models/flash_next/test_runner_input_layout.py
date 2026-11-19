# SPDX-License-Identifier: Apache-2.0
"""The HPU runner's flat token layout for Flash-Next (RUK-3102).

The engine's ``Qwen4ExpModel.forward`` embeds the ids and widens them to the
HyperConnection streams with ``hidden_states.repeat(1, hc_count)``, which only works on
flat ``[num_tokens]`` ids (embeddings ``[T, H]`` -> ``[T, hc_count * H]``).
``HpuModelAdapter.forward`` flattens the runner's ``[bs, seq]`` batch only when the
``flatten_input`` feature is on (``vllm_gaudi/v1/worker/hpu_model_runner.py:1677-1679``),
so the feature must list ``qwen4_exp``. The runner cannot be imported on CPU (no
``habana_frameworks``), so the forward tests reproduce the adapter's flatten rule in a
few lines instead of calling it.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.extension.config import Config  # noqa: E402
from vllm_gaudi.extension.features import get_features  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, eos_token_id, ple_kwargs, prefill_metadata, reset_cpu_states)

WARMUP_PROMPT_BUCKET = 512  # the first warmup prompt bucket (1, 512, 0), hpu_model_runner.py:7740


def flatten_input_for(model_type: str) -> bool:
    """The resolved ``flatten_input`` feature for ``model_type`` (as the runner reads it)."""
    values, flags = get_features()
    return Config(values, flags, model_type=model_type).get("flatten_input")


def test_flatten_input_on_for_flash_next():
    """``flatten_input`` is on for ``qwen4_exp`` (the Flash-Next runner layout) and still on for ``qwen3_5``."""
    assert flatten_input_for("qwen4_exp") is True
    assert flatten_input_for("qwen3_5") is True


def test_warmup_prompt_bucket_mix_input_shape(tiny_checkpoint):
    """The warmup bucket (1, 512) fed through the runner's flatten rule gives layer 0's ``mix``
    a ``[T, hc_count * hidden_size]`` input and a ``[T, hidden_size]`` output."""
    T = WARMUP_PROMPT_BUCKET
    token_ids = (torch.arange(T) + 10).view(1, T)
    positions = torch.arange(T).view(1, T)
    # hpu_model_runner.py:1677-1678: the adapter flattens the ids only when flatten_input is on.
    flatten = flatten_input_for("qwen4_exp")
    ids = token_ids.view(-1) if flatten else token_ids
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        text_config = vllm_config.model_config.hf_text_config
        hidden_size = int(text_config.hidden_size)
        expected_mix = (T, int(text_config.hc_count) * hidden_size)
        hyper_connection = model.language_model.model.layers[0].attn_hyper_connection
        recorded = []
        original_mix = hyper_connection.mix

        def record_mix(hidden_states):
            recorded.append(tuple(hidden_states.shape))
            return original_mix(hidden_states)

        hyper_connection.mix = record_mix
        eos = eos_token_id(vllm_config)
        reset_cpu_states(model)
        with torch.no_grad(), set_forward_context(prefill_metadata(T, 1), vllm_config):
            hidden = model(input_ids=ids, positions=positions, **ple_kwargs(T, eos))
        assert recorded == [expected_mix]
        assert tuple(hidden.shape) == (T, hidden_size)


def test_two_d_input_ids_rejected(tiny_checkpoint):
    """The HPU model forward rejects the unflattened ``[1, T]`` ids with a ValueError naming
    ``flatten_input`` (the card's warmup crash was ``[1, 2048]`` reaching layer 0's ``mix``)."""
    T = WARMUP_PROMPT_BUCKET
    token_ids = (torch.arange(T) + 10).view(1, T)
    positions = torch.arange(T).view(1, T)
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        reset_cpu_states(model)
        with pytest.raises(ValueError, match="flatten_input"), torch.no_grad(), set_forward_context(
                prefill_metadata(T, 1), vllm_config):
            model(input_ids=token_ids, positions=positions, **ple_kwargs(T, eos))
