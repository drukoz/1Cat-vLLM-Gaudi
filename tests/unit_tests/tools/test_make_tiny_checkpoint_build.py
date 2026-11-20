# SPDX-License-Identifier: Apache-2.0
"""Tests for the build options of tools/flash_next/make_tiny_checkpoint.py the rental harness depends on:
the GDN head-dim override, the build settings recorded in reference.json and the loud tokenizer failure.

One fixture builds one real checkpoint (CPU, one thread, ``--skip-tokenizer``, about 15 s); the other tests
never build weights.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tools.flash_next import make_tiny_checkpoint as maker

REPO = Path(__file__).resolve().parents[3]
TP8_FIXTURE = REPO / "tests" / "unit_tests" / "models" / "flash_next" / "tiny_tp8_config.json"
GDN_LAYER = "model.language_model.layers.0.linear_attn."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_bytes_without_override_are_the_fixture_bytes():
    assert maker.config_bytes(TP8_FIXTURE, None) == TP8_FIXTURE.read_bytes()


def test_config_bytes_with_override_change_only_the_two_gdn_head_dims():
    fixture = json.loads(TP8_FIXTURE.read_text(encoding="utf-8"))
    changed = json.loads(maker.config_bytes(TP8_FIXTURE, 128).decode("utf-8"))
    assert changed["text_config"]["linear_key_head_dim"] == 128
    assert changed["text_config"]["linear_value_head_dim"] == 128
    changed["text_config"]["linear_key_head_dim"] = fixture["text_config"]["linear_key_head_dim"]
    changed["text_config"]["linear_value_head_dim"] = fixture["text_config"]["linear_value_head_dim"]
    assert changed == fixture
    assert list(changed["text_config"]) == list(fixture["text_config"])  # the key order of the fixture is kept


@pytest.fixture(scope="module")
def checkpoint_128(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny_tp8_gdn128")
    fixture_before = TP8_FIXTURE.read_bytes()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as the rental harness builds it: the tiny tensors are slower on many threads
    try:
        code = maker.main(["--out", str(out), "--fixture", str(TP8_FIXTURE), "--gdn-head-dim", "128",
                           "--skip-tokenizer"])
    finally:
        torch.set_num_threads(threads)
    assert code == 0
    assert TP8_FIXTURE.read_bytes() == fixture_before  # the committed fixture is never rewritten
    return out


def test_override_builds_gdn_layers_with_head_dim_128(checkpoint_128: Path):
    config = json.loads((checkpoint_128 / "config.json").read_text(encoding="utf-8"))["text_config"]
    assert (config["linear_key_head_dim"], config["linear_value_head_dim"]) == (128, 128)
    key_dim = config["linear_num_key_heads"] * 128
    value_dim = config["linear_num_value_heads"] * 128
    with safe_open(checkpoint_128 / maker.WEIGHTS_FILE, framework="pt") as handle:
        qkv = handle.get_slice(GDN_LAYER + "in_proj_qkv.weight").get_shape()
        out_proj = handle.get_slice(GDN_LAYER + "out_proj.weight").get_shape()
    assert qkv == [2 * key_dim + value_dim, config["hidden_size"]]
    assert out_proj == [config["hidden_size"], value_dim]


def test_reference_records_the_build_settings(checkpoint_128: Path):
    reference = json.loads((checkpoint_128 / "reference.json").read_text(encoding="utf-8"))
    assert reference["build"] == {
        "schema": maker.SCHEMA_VERSION,
        "seed": 0,
        "gdn_head_dim": 128,
        "prompt_lengths": list(maker.PROMPT_LENGTHS),
        "max_new_tokens": maker.MAX_NEW_TOKENS,
        "fixture_sha256": sha256(TP8_FIXTURE),
        "generator_sha256": sha256(Path(maker.__file__)),
    }
    assert reference["schema"] == maker.SCHEMA_VERSION
    assert reference["config_sha256"] == sha256(checkpoint_128 / "config.json")
    assert [len(p["prompt_ids"]) for p in reference["prompts"]] == [1, 7, 33, 100, 700, 2600]
    assert all(len(p["generated_ids"]) == 16 for p in reference["prompts"])


def test_build_settings_take_the_head_dim_from_the_checkpoint(tmp_path):
    # --reference-only has no --gdn-head-dim: the reference records what the checkpoint's config.json says.
    maker.write_config(tmp_path, TP8_FIXTURE, 128)
    assert maker.build_settings(tmp_path, TP8_FIXTURE, 0)["gdn_head_dim"] == 128
    maker.write_config(tmp_path, TP8_FIXTURE)
    assert maker.build_settings(tmp_path, TP8_FIXTURE, 3)["gdn_head_dim"] == 16
    assert maker.build_settings(tmp_path, TP8_FIXTURE, 3)["seed"] == 3


def test_the_orchestrator_reuses_this_checkpoint_and_no_other(checkpoint_128: Path, tmp_path):
    # The contract between the generator's `build` block and the rental harness's reuse check.
    from tools.flash_next.rental import main as orchestrator

    assert orchestrator.tiny_checkpoint_stale_reason(checkpoint_128) is None
    assert orchestrator.tiny_model_len(checkpoint_128) == 4096  # covers the 2600-token prompt plus 16 tokens
    reference = json.loads((checkpoint_128 / "reference.json").read_text(encoding="utf-8"))
    reference["build"]["gdn_head_dim"] = 16  # what a plain build of the fixture records
    (tmp_path / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    assert "gdn_head_dim" in orchestrator.tiny_checkpoint_stale_reason(tmp_path)


def test_reference_only_refuses_a_head_dim(tmp_path, capsys):
    (tmp_path / maker.WEIGHTS_FILE).write_bytes(b"")
    with pytest.raises(SystemExit) as info:
        maker.main(["--out", str(tmp_path), "--reference-only", "--gdn-head-dim", "128"])
    assert info.value.code == 2
    assert "--gdn-head-dim cannot be combined with --reference-only" in capsys.readouterr().err


def test_tokenizer_download_failure_is_loud_and_leaves_no_reference(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    def offline(**kwargs):
        raise ConnectionError("no route to huggingface.co")

    def no_build(*args, **kwargs):
        raise AssertionError("the weights must not be built after the tokenizer fetch failed")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    monkeypatch.setattr(maker, "build_checkpoint", no_build)
    out = tmp_path / "ckpt"
    code = maker.main(["--out", str(out), "--fixture", str(TP8_FIXTURE)])
    assert code == 1
    err = capsys.readouterr().err
    assert "tokenizer" in err and maker.HF_REPO_ID in err
    assert "ConnectionError: no route to huggingface.co" in err
    assert "--tokenizer-from" in err
    assert not (out / "reference.json").exists()


def test_tokenizer_download_that_brings_no_tokenizer_file_is_a_failure(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(kwargs["local_dir"]))
    monkeypatch.setattr(maker, "build_checkpoint", lambda *args, **kwargs: pytest.fail("must not build"))
    out = tmp_path / "ckpt"
    assert maker.main(["--out", str(out), "--fixture", str(TP8_FIXTURE)]) == 1
    assert "no tokenizer file" in capsys.readouterr().err
    assert not (out / "reference.json").exists()
