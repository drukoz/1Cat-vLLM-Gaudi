# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the HPU Flash-Next QSA dense attention (M0).

Builds ``HpuQwen4ExpQSAAttention`` from the tiny fixture with a world-1 gloo
group and a vLLM config context (no HPU required) and checks:

1. checkpoint parameter names/shapes and ``Attention``-child registration,
2. a dense forward that matches the HF ``Qwen4ExpTextAttention`` oracle on a
   10-token prompt (the indexer selects the full causal prefix),
3. ``QsaMode.SPARSE`` raises ``NotImplementedError`` (M2),
4. ``QsaMode`` is an integer enum with the documented values.
"""

import enum
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest
import torch
import torch.nn.functional as F

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel, init_distributed_environment, initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import (  # noqa: E402
    HpuQwen4ExpQSAAttention, QsaMode,
)
import vllm_gaudi.envs as envs  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "tiny_config.json"
PREFIX = "model.layers.3.self_attn"
LAYER_ID = 3

from pathlib import Path  # noqa: E402

EXPECTED_PARAMS = [
    "indexer.index_qk_proj.weight",
    "indexer.k_layernorm.weight",
    "indexer.q_layernorm.weight",
    "k_norm.weight",
    "o_proj.weight",
    "q_norm.weight",
    "qkv_proj.weight",
]


def _engine_config(model_dir):
    args = EngineArgs(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        max_model_len=512,
        max_num_seqs=4,
        enforce_eager=True,
        language_model_only=True,
        load_format="dummy",
    )
    return args.create_engine_config()


@pytest.fixture()
def qsa_layer(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "config.json")
    vllm_config = _engine_config(tmp_path)
    init_file = tempfile.mkstemp(dir="/tmp")[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{init_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            layer = HpuQwen4ExpQSAAttention(
                vllm_config=vllm_config,
                config=vllm_config.model_config.hf_text_config,
                layer_id=LAYER_ID,
                prefix=PREFIX,
            )
        yield vllm_config, layer
    destroy_model_parallel()
    destroy_distributed_environment()


def test_parameter_names_and_registration(qsa_layer):
    vllm_config, layer = qsa_layer

    assert sorted(dict(layer.named_parameters())) == EXPECTED_PARAMS
    params = dict(layer.named_parameters())
    assert params["qkv_proj.weight"].shape == (256 + 64 + 64, 128)
    assert params["o_proj.weight"].shape == (128, 128)
    assert params["q_norm.weight"].shape == (32, )
    assert params["k_norm.weight"].shape == (32, )
    assert params["indexer.index_qk_proj.weight"].shape == ((4 + 1) * 16, 128)
    assert params["indexer.q_layernorm.weight"].shape == (16, )
    assert params["indexer.k_layernorm.weight"].shape == (16, )

    module_names = [name for name, _ in layer.named_modules()]
    assert not [n for n in module_names if n.endswith(("raw_key_cache", "compressed_key_cache"))]
    buffer_names = [name for name, _ in layer.named_buffers()]
    assert not [n for n in buffer_names if "topk" in n or "key_cache" in n]

    static_context = vllm_config.compilation_config.static_forward_context
    layer_names = [name for name in static_context if name.startswith(PREFIX)]
    assert layer_names == [f"{PREFIX}.attn"]
    assert static_context[f"{PREFIX}.attn"] is layer.attn
    assert layer.qsa_mode is QsaMode.DENSE


def test_dense_forward_matches_hf(qsa_layer):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextAttention,
        Qwen4ExpTextRotaryEmbedding,
    )

    _, layer = qsa_layer

    hf_config = Qwen4ExpTextConfig(**json.loads(FIXTURE.read_text())["text_config"])
    hf_config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    hf_attn = Qwen4ExpTextAttention(hf_config, layer_idx=LAYER_ID).to(torch.float32).eval()
    hf_rotary = Qwen4ExpTextRotaryEmbedding(hf_config).to(torch.float32).eval()

    # Run the vLLM layer in fp32 against the fp32 HF weights, rebuilding the
    # RoPE cache so it is not quantized to bf16.
    layer.float()
    layer.rotary_emb.cos_sin_cache = layer.rotary_emb._compute_cos_sin_cache()

    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, hf_attn.q_proj.weight.data, "q")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, hf_attn.k_proj.weight.data, "k")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, hf_attn.v_proj.weight.data, "v")
    layer.o_proj.weight_loader(layer.o_proj.weight, hf_attn.o_proj.weight.data)
    layer.q_norm.weight.data.copy_(hf_attn.q_norm.weight.data)
    layer.k_norm.weight.data.copy_(hf_attn.k_norm.weight.data)
    layer.indexer.index_qk_proj.weight_loader(layer.indexer.index_qk_proj.weight,
                                              hf_attn.indexer.index_qk_proj.weight.data)
    layer.indexer.q_layernorm.weight.data.copy_(hf_attn.indexer.q_layernorm.weight.data)
    layer.indexer.k_layernorm.weight.data.copy_(hf_attn.indexer.k_layernorm.weight.data)
    for linear in (layer.qkv_proj, layer.o_proj, layer.indexer.index_qk_proj):
        linear.quant_method.process_weights_after_loading(linear)

    # Stand in for the HPU/CPU paged-attention backend: causal full-prefix
    # attention, which is exactly what dense mode computes.
    class SdpaAttention(torch.nn.Module):

        def __init__(self, num_heads, num_kv_heads, head_dim, scaling):
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.scaling = scaling

        def forward(self, query, key, value):
            num_tokens = query.shape[0]
            qh = query.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            kh = key.view(num_tokens, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            vh = value.view(num_tokens, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
            kh = kh.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            vh = vh.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=self.scaling)
            return out.squeeze(0).transpose(0, 1).reshape(num_tokens, -1)

    layer.attn = SdpaAttention(layer.num_heads, layer.num_kv_heads, layer.head_dim, layer.scaling)

    torch.manual_seed(1234)
    hidden_states = torch.randn(1, 10, 128, dtype=torch.float32)
    positions = torch.arange(10, dtype=torch.int64)
    causal_mask = torch.tril(torch.ones(10, 10, dtype=torch.bool)).view(1, 1, 10, 10)

    with torch.no_grad():
        cos, sin = hf_rotary(hidden_states, positions.unsqueeze(0))
        hf_out, _ = hf_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
            past_key_values=None,
        )
        v_out = layer(positions, hidden_states)

    assert v_out.shape == hf_out.shape == (1, 10, 128)
    torch.testing.assert_close(v_out, hf_out, atol=1e-4, rtol=1e-4)


def test_sparse_mode_raises(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_QSA_MODE", "1")
    with pytest.raises(NotImplementedError, match="M2"):
        HpuQwen4ExpQSAAttention(
            vllm_config=None,
            config=None,
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )


def test_qsa_mode_is_integer_enum(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_FLASH_NEXT_QSA_MODE", raising=False)
    assert issubclass(QsaMode, enum.IntEnum)
    assert int(QsaMode.DENSE) == 0
    assert int(QsaMode.SPARSE) == 1
    assert QsaMode(0) is QsaMode.DENSE
    assert QsaMode(1) is QsaMode.SPARSE
    assert envs.VLLM_HPU_FLASH_NEXT_QSA_MODE == 0
