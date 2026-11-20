# SPDX-License-Identifier: Apache-2.0
"""Whole-model CPU tests for the Flash-Next HPU class (M0-T7a).

Builds ``HpuQwen4ExpForConditionalGeneration`` from the tiny checkpoint through vLLM's real
loader (``cpu_model.py``) and proves the two seams, the loaded weights and the kv-cache
specs. No forward pass here: the PLE forward needs T4b and is T7b's.

One ``cpu_model`` may be open at a time (world-1 gloo), so ``model_bf16`` is
function-scoped and no test nests a second build; a build takes a few seconds.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import vllm.models.qwen4_exp.nvidia.model as upstream_model  # noqa: E402
import vllm_gaudi.models.qwen4_exp as fn  # noqa: E402
from vllm.config import get_layers_from_vllm_config  # noqa: E402
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase  # noqa: E402
from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402
from vllm_gaudi.models.qwen4_exp import HpuQwen4ExpForConditionalGeneration  # noqa: E402
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention  # noqa: E402
from vllm_gaudi.v1.core.recurrent_state_pool import (  # noqa: E402
    num_recurrent_groups, recurrent_positions)

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    SdpaAttention, cpu_model, use_sdpa_attention)

COMMITTED_REFERENCE = Path(__file__).resolve().parent / "reference.json"
WEIGHTS_FILE = "model.safetensors"
PARAM_COUNT = 67_033_008

# (Checkpoint name, model name) pairs; the loader maps the HF names onto the vLLM ones.
CHECKPOINT_TO_MODEL = [
    ("model.language_model.embed_tokens.weight", "language_model.model.embed_tokens.weight"),  # (248320, 128)
    ("lm_head.weight", "language_model.lm_head.weight"),
    ("model.language_model.layers.0.linear_attn.out_proj.weight",  # checkpoint (128, 768)
     "language_model.model.layers.0.linear_attn.out_proj.weight"),
    ("model.language_model.layers.3.self_attn.o_proj.weight",  # checkpoint (128, 128)
     "language_model.model.layers.3.self_attn.o_proj.weight"),
    ("model.language_model.layers.1.ple.conv1d.weight",  # checkpoint (512, 1, 4)
     "language_model.model.layers.1.ple.conv1d.weight"),
]

GDN_LAYERS = tuple(f"language_model.model.layers.{i}.linear_attn" for i in range(3))
PLE_LAYER = "language_model.model.layers.1.ple"
QSA_LAYER = "language_model.model.layers.3.self_attn.attn"


@pytest.fixture
def model_bf16(tiny_checkpoint):
    """The tiny model built in bf16 (one world-1 gloo group per test)."""
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        yield vllm_config, model


def test_checkpoint_reference_is_committed_one(tiny_checkpoint):
    """The session checkpoint's reference.json is the committed HF oracle."""
    generated = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    committed = json.loads(COMMITTED_REFERENCE.read_text(encoding="utf-8"))
    # "build" is provenance the generator adds (fixture and generator hashes, GDN head dim), used to
    # decide whether a checkpoint folder can be reused; it changes with every edit of the generator.
    # The oracle is everything else.
    build = generated.pop("build", None)
    committed.pop("build", None)
    assert build is None or build["gdn_head_dim"] == 16  # the session checkpoint is the committed fixture's
    assert generated == committed


def test_seams_bound():
    """apply_flash_next_patches() rebinds the upstream HyperConnection and QSA names."""
    fn.apply_flash_next_patches()
    assert upstream_model.GatedResidual is HpuGatedResidual
    assert upstream_model.Qwen4ExpQSAAttention is HpuQwen4ExpQSAAttention


def test_instantiates_hpu_class(model_bf16):
    """get_model builds the HPU class with the HPU HyperConnections and QSA attention."""
    _, model = model_bf16
    assert type(model) is HpuQwen4ExpForConditionalGeneration
    assert sum(p.numel() for p in model.parameters()) == PARAM_COUNT

    layers = model.language_model.model.layers
    assert len(layers) == 4
    for layer in layers:
        assert isinstance(layer.attn_hyper_connection, HpuGatedResidual)
        assert isinstance(layer.mlp_hyper_connection, HpuGatedResidual)
    assert isinstance(model.language_model.model.hyper_connection_mixer, HpuGatedResidual)

    self_attn = layers[3].self_attn
    assert isinstance(self_attn, HpuQwen4ExpQSAAttention)
    assert isinstance(self_attn.attn, SdpaAttention)
    assert use_sdpa_attention(model) == 1  # idempotent: exactly one QSA layer in the tiny model


def test_weights_loaded(model_bf16, tiny_checkpoint):
    """The loader copied the checkpoint tensors into the model, and no weight is NaN/Inf."""
    _, model = model_bf16
    params = dict(model.named_parameters())
    with safe_open(tiny_checkpoint / WEIGHTS_FILE, framework="pt") as handle:
        for checkpoint_name, model_name in CHECKPOINT_TO_MODEL:
            param = params[model_name]
            assert torch.equal(handle.get_tensor(checkpoint_name).to(param.dtype), param), model_name
    not_finite = [name for name, param in params.items()
                  if param.is_floating_point() and not torch.isfinite(param).all()]
    assert not_finite == []


def test_hc_padding_rows_are_zero(model_bf16):
    """The merged down+inject weight's padding rows, which the checkpoint never fills, are exactly zero."""
    _, model = model_bf16
    padded = [m for m in model.modules() if isinstance(m, HpuGatedResidual) and m.pad_size]
    assert padded, "the tiny model has no padded HyperConnection weight"
    for module in padded:
        pad = module.input_mix_weight_down_block_inject.weight[module.lora_rank + module.hc_count:]
        assert pad.shape[0] == module.pad_size
        assert bool((pad == 0).all())


def test_kv_cache_specs(model_bf16):
    """The five cache layers of the tiny model report exactly the measured specs."""
    vllm_config, _ = model_bf16
    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    assert sorted(layers) == sorted((*GDN_LAYERS, PLE_LAYER, QSA_LAYER))
    specs = {name: layer.get_kv_cache_spec(vllm_config) for name, layer in layers.items()}

    for name in GDN_LAYERS:
        spec = specs[name]
        assert type(spec).__name__ == "MambaSpec", name
        assert spec.mamba_type.name == "GDN_ATTN", name
        assert tuple(spec.shapes) == ((3, 1280), (48, 16, 16)), name
        assert tuple(spec.dtypes) == (torch.bfloat16, torch.float32), name
        assert spec.page_size_bytes == 56832, name

    spec = specs[PLE_LAYER]
    assert type(spec).__name__ == "MambaSpec"
    assert spec.mamba_type.name == "SHORT_CONV"
    assert tuple(spec.shapes) == ((9, 512),)
    assert tuple(spec.dtypes) == (torch.bfloat16,)
    assert spec.page_size_bytes == 9216

    spec = specs[QSA_LAYER]
    assert type(spec).__name__ == "FullAttentionSpec"
    assert spec.num_kv_heads == 2
    assert spec.head_size == 32
    assert spec.block_size == 128
    assert spec.dtype == torch.bfloat16
    assert spec.page_size_bytes == 32768


def test_recurrent_pool_grouping(model_bf16):
    """The pool contract groups the tiny model like the runner shares its state tensors (M1-T1)."""
    vllm_config, _ = model_bf16
    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    specs = {name: layer.get_kv_cache_spec(vllm_config) for name, layer in layers.items()}

    positions = recurrent_positions(specs)
    assert {spec.mamba_type.name: pos for spec, pos in positions.items()} == {"GDN_ATTN": 1, "SHORT_CONV": 1}
    assert num_recurrent_groups(specs) == 4


def test_fp32_conversion(tiny_checkpoint):
    """cpu_model(fp32=True) leaves every floating parameter and every bound state in fp32."""
    with cpu_model(tiny_checkpoint, fp32=True) as (_, model):
        not_fp32 = [(name, param.dtype) for name, param in model.named_parameters()
                    if param.is_floating_point() and param.dtype != torch.float32]
        assert not_fp32 == []

        mamba_layers = [m for m in model.modules() if isinstance(m, MambaBase)]
        assert len(mamba_layers) == 4  # 3 GDN layers + the PLE layer
        states = [state for mamba in mamba_layers for state in mamba.kv_cache]
        assert len(states) == 7  # 3 * 2 GDN states + 1 PLE state
        assert all(state.dtype == torch.float32 for state in states)
