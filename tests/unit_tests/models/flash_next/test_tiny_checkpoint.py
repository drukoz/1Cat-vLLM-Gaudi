# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/make_tiny_checkpoint.py (tiny Flash-Next checkpoint and HF CPU reference).

Every test builds into its own ``tmp_path`` with ``--skip-tokenizer``; the committed ``reference.json`` is the
oracle the reproducibility test compares against. Runs on CPU in the shared S0 environment from the worktree root.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tools.flash_next.make_tiny_checkpoint import DEFAULT_FIXTURE, WEIGHTS_FILE, main

# Weight names the checkpoint must carry (the names the vLLM loader expects), layer 1 = GDN layer with PLE,
# layer 3 = QSA layer, plus the global tensors. Checked against the Hugging Face model in transformers 5.23.0.
CONTEXT_WEIGHT_NAMES = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.1.linear_attn.dt_bias",
    "model.language_model.layers.1.linear_attn.A_log",
    "model.language_model.layers.1.linear_attn.conv1d.weight",
    "model.language_model.layers.1.linear_attn.norm.weight",
    "model.language_model.layers.1.linear_attn.out_proj.weight",
    "model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.1.linear_attn.in_proj_z.weight",
    "model.language_model.layers.1.linear_attn.in_proj_b.weight",
    "model.language_model.layers.1.linear_attn.in_proj_a.weight",
    "model.language_model.layers.1.mlp.gate.weight",
    "model.language_model.layers.1.mlp.experts.gate_up_proj",
    "model.language_model.layers.1.mlp.experts.down_proj",
    "model.language_model.layers.1.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.1.mlp.shared_expert.up_proj.weight",
    "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
    "model.language_model.layers.1.mlp.shared_expert_gate.weight",
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight",
    "model.language_model.layers.1.ple.key_proj.weight",
    "model.language_model.layers.1.ple.value_proj.weight",
    "model.language_model.layers.1.ple.norm_key.weight",
    "model.language_model.layers.1.ple.norm_query.weight",
    "model.language_model.layers.1.ple.norm_conv.weight",
    "model.language_model.layers.1.ple.conv1d.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.self_attn.k_proj.weight",
    "model.language_model.layers.3.self_attn.v_proj.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.layers.3.self_attn.q_norm.weight",
    "model.language_model.layers.3.self_attn.k_norm.weight",
    "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
    "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
    "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
    "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    "lm_head.weight",
]
CONTEXT_WEIGHT_NAMES += [
    f"model.language_model.layers.1.{connection}.{name}"
    for connection in ("attn_hyper_connection", "mlp_hyper_connection")
    for name in ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight",
                 "block_inject_weight.weight")
]

# Integer index tables of the PLE n-gram embedding: the only tensors of the checkpoint that are not weights.
PLE_INDEX_TABLES = [
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
]

COMMITTED_REFERENCE = Path(__file__).resolve().parent / "reference.json"
PROMPT_LENGTHS = [1, 7, 33, 100]
LOGIT_TOLERANCE = 1e-4


def build(tmp_path: Path, seed: int) -> Path:
    assert main(["--out", str(tmp_path), "--seed", str(seed), "--skip-tokenizer"]) == 0
    return tmp_path


def load_reference(directory: Path) -> dict:
    return json.loads((directory / "reference.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checkpoint_seed0(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("tiny_seed0_a"), 0)


@pytest.fixture(scope="module")
def checkpoint_seed0_again(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("tiny_seed0_b"), 0)


@pytest.fixture(scope="module")
def checkpoint_seed1(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("tiny_seed1"), 1)


def test_config_round_trip(checkpoint_seed0: Path) -> None:
    """config.json keeps the fixture bytes: save-time config rewriting must not drop the mrope keys."""
    assert (checkpoint_seed0 / "config.json").read_bytes() == DEFAULT_FIXTURE.read_bytes()


def test_weight_names_and_dtype(checkpoint_seed0: Path) -> None:
    """The checkpoint carries the model's weight names in bfloat16, with the PLE index tables as int64."""
    with safe_open(checkpoint_seed0 / WEIGHTS_FILE, framework="pt") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}

    missing = [name for name in CONTEXT_WEIGHT_NAMES if name not in tensors]
    assert not missing, f"weights missing from {WEIGHTS_FILE}: {missing}"
    assert any(name.startswith("model.visual.") for name in tensors), "no vision tower tensors in the checkpoint"

    not_bfloat16 = {name: tensor.dtype for name, tensor in tensors.items() if tensor.dtype != torch.bfloat16}
    assert not_bfloat16 == {name: torch.int64 for name in PLE_INDEX_TABLES}
    assert tensors["lm_head.weight"].shape == (248320, 128)


def test_deterministic(checkpoint_seed0: Path, checkpoint_seed0_again: Path, checkpoint_seed1: Path) -> None:
    """The same seed gives byte-identical weights, a different seed does not."""
    sha_seed0 = load_reference(checkpoint_seed0)["weights_sha256"]
    sha_seed0_again = load_reference(checkpoint_seed0_again)["weights_sha256"]
    sha_seed1 = load_reference(checkpoint_seed1)["weights_sha256"]
    assert sha_seed0 == sha_seed0_again
    assert sha_seed0 != sha_seed1


def test_reference_reproducible(checkpoint_seed0: Path) -> None:
    """--reference-only on the built checkpoint reproduces the committed reference.json."""
    committed = json.loads(COMMITTED_REFERENCE.read_text(encoding="utf-8"))
    assert committed["schema"] == 1
    assert [len(prompt["prompt_ids"]) for prompt in committed["prompts"]] == PROMPT_LENGTHS
    assert all(len(prompt["generated_ids"]) == committed["max_new_tokens"] for prompt in committed["prompts"])

    assert main(["--out", str(checkpoint_seed0), "--seed", "0", "--reference-only"]) == 0
    reproduced = load_reference(checkpoint_seed0)

    assert reproduced["generation"] == committed["generation"]
    assert len(reproduced["prompts"]) == len(committed["prompts"])
    for got, expected in zip(reproduced["prompts"], committed["prompts"]):
        assert got["prompt_ids"] == expected["prompt_ids"]
        assert got["generated_ids"] == expected["generated_ids"]
        assert got["last_prefill_top5"]["ids"] == expected["last_prefill_top5"]["ids"]
        for value, want in zip(got["last_prefill_top5"]["logits"], expected["last_prefill_top5"]["logits"]):
            assert abs(value - want) <= LOGIT_TOLERANCE, f"{value} vs {want}"
