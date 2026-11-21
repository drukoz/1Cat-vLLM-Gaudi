# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Flash-Next MTP drafter's pure step helpers (M3-T4)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import (is_flash_next_mtp, next_step_hidden,
                                                             ring_slots_from_lattice, split_draft_output)

PROPOSER_SRC = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "v1" / "spec_decode" / "hpu_flash_next_mtp.py"


def _config(method, architectures):
    hf_config = SimpleNamespace(architectures=architectures)
    return SimpleNamespace(method=method, draft_model_config=SimpleNamespace(hf_config=hf_config))


def test_is_flash_next_mtp_true_for_qwen4exp_mtp_draft():
    assert is_flash_next_mtp(_config("mtp", ["Qwen4ExpMTP"]))


def test_is_flash_next_mtp_false_for_eagle_and_other_draft_archs():
    assert not is_flash_next_mtp(_config("eagle", ["Qwen4ExpMTP"]))
    assert not is_flash_next_mtp(_config("eagle3", ["Eagle3LlamaForCausalLM"]))
    assert not is_flash_next_mtp(_config("mtp", ["DeepSeekMTPModel"]))
    assert not is_flash_next_mtp(_config("mtp", []))
    assert not is_flash_next_mtp(_config("mtp", None))
    assert not is_flash_next_mtp(SimpleNamespace(method="mtp"))


def test_split_draft_output_returns_the_pair_unchanged():
    sample = torch.zeros(5, 8)
    multi = torch.arange(5 * 24, dtype=torch.float32).view(5, 24)
    s, m = split_draft_output((sample, multi))
    assert s is sample
    assert m is multi
    assert m.shape[-1] % s.shape[-1] == 0


def test_split_draft_output_type_errors():
    sample = torch.zeros(5, 8)
    with pytest.raises(TypeError):
        split_draft_output(sample)
    with pytest.raises(TypeError):
        split_draft_output((sample, sample, sample))
    with pytest.raises(TypeError):
        split_draft_output([sample, torch.zeros(5, 16)])
    with pytest.raises(TypeError):
        split_draft_output((sample, torch.zeros(5, 9)))


def test_ring_slots_from_lattice_cuts_the_lattice_to_column_zero():
    lattice = torch.arange(2 * 3 * 4, dtype=torch.int32).view(2, 3, 4)
    ring = ring_slots_from_lattice(lattice)
    assert ring.shape == (2, 3)
    assert ring.tolist() == lattice[:, :, 0].tolist()


def test_ring_slots_from_lattice_passes_through_2d_and_rejects_1d():
    slots = torch.arange(2 * 3, dtype=torch.int32).view(2, 3)
    assert ring_slots_from_lattice(slots) is slots
    with pytest.raises(ValueError):
        ring_slots_from_lattice(torch.arange(4, dtype=torch.int32))
    with pytest.raises(ValueError):
        ring_slots_from_lattice(torch.zeros(2, 3, 4, 5, dtype=torch.int32))


def test_next_step_hidden_selects_the_last_token_rows_as_step_input():
    # virtual batch 6 rows of a hc*H = 16 multi stream; batch is the 3 real requests
    multi = torch.arange(6 * 16, dtype=torch.float32).view(6, 16)
    last_token_indices = torch.tensor([5, 1, 4])
    out = next_step_hidden(multi, last_token_indices)
    assert out.shape == (3, 1, 16)
    assert torch.equal(out.reshape(3, 16), multi[last_token_indices])


def test_next_step_hidden_accepts_the_model_shaped_output():
    multi = torch.arange(6 * 1 * 16, dtype=torch.float32).view(6, 1, 16)
    out = next_step_hidden(multi, torch.tensor([0, 2]))
    assert out.shape == (2, 1, 16)
    assert torch.equal(out.reshape(2, 16), multi.view(-1, 16)[[0, 2]])


def test_proposer_source_uses_the_helpers():
    src = PROPOSER_SRC.read_text()
    assert "split_draft_output(" in src
    assert "spec_step_idx=token_index + 1" in src
    assert "block_table=" in src
    # the kernel table uses the defragmenter-resolved rows, like block_list and the target's table
    assert "build_block_table(block_tables_list, batch_size" in src
    assert "load_indices_tensor=ring_slots" in src
    # the draft must feed its own multi' forward, never the target's hidden states
    assert "hidden_states = target_hidden_states" not in src


# ---- the engine-start fixes (found by tests/unit_tests/hpu_cpu_engine.py; test_engine_mtp.py runs the engine)


def _steps():
    from vllm_gaudi.v1.spec_decode import flash_next_mtp_steps
    return flash_next_mtp_steps


def test_the_flash_next_target_is_looked_up_under_a_name_upstream_lists_with_image_token_id():
    # upstream load_model copies target.config.image_token_id for the listed names and reads
    # target.config.image_token_index for any other: Qwen4ExpConfig has only image_token_id
    steps = _steps()
    assert steps.flash_next_target_model_name("Qwen4ExpForConditionalGeneration") == \
        "Qwen3_5ForConditionalGeneration"
    assert steps.flash_next_target_model_name("HpuQwen4ExpForConditionalGeneration") == \
        "Qwen3_5ForConditionalGeneration"
    for other in ("Qwen3_5ForConditionalGeneration", "LlamaForCausalLM", "HpuQwen4ExpForCausalLM"):
        assert steps.flash_next_target_model_name(other) == other


def test_upstream_load_model_still_lists_the_name_with_image_token_id():
    import inspect

    from vllm.v1.spec_decode import llm_base_proposer

    src = inspect.getsource(llm_base_proposer)
    listed = src.index('"Qwen3_5ForConditionalGeneration"')
    assert "image_token_index = target_model.config.image_token_id" in src[listed:listed + 800]


def test_the_proposer_answers_with_the_lookup_name():
    src = PROPOSER_SRC.read_text()
    assert "def get_model_name(self, model)" in src
    assert "flash_next_target_model_name(super().get_model_name(model))" in src


def _model_config(rope_parameters, text_rope_parameters=None):
    hf_config = SimpleNamespace(rope_parameters=rope_parameters)
    text = SimpleNamespace(rope_parameters=text_rope_parameters) if text_rope_parameters is not None else hf_config
    return SimpleNamespace(hf_config=hf_config, hf_text_config=text)


def test_the_draft_loses_the_m_rope_keys_the_language_model_only_target_lost():
    steps = _steps()
    target = _model_config({"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    draft_rope = {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                  "mrope_section": [2, 1, 1], "mrope_interleaved": True}
    draft = _model_config(draft_rope)
    assert steps.config_uses_mrope(draft) and not steps.config_uses_mrope(target)
    assert steps.align_draft_rope_with_target(target, draft) is True
    assert draft_rope == {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25}
    assert not steps.config_uses_mrope(draft)
    # a second call finds nothing to change
    assert steps.align_draft_rope_with_target(target, draft) is False


def test_the_text_config_of_the_draft_is_cleaned_too():
    steps = _steps()
    target = _model_config({"rope_type": "default"})
    outer = {"rope_type": "default", "mrope_section": [2, 1, 1]}
    inner = {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}
    draft = _model_config(outer, inner)
    assert steps.align_draft_rope_with_target(target, draft) is True
    assert "mrope_section" not in outer and "mrope_section" not in inner and "mrope_interleaved" not in inner


def test_a_target_that_keeps_m_rope_is_refused():
    steps = _steps()
    target = _model_config({"rope_type": "default", "mrope_section": [2, 1, 1]})
    draft = _model_config({"rope_type": "default", "mrope_section": [2, 1, 1]})
    with pytest.raises(NotImplementedError, match="--language-model-only"):
        steps.align_draft_rope_with_target(target, draft)


def test_a_config_without_rope_parameters_is_left_alone():
    steps = _steps()
    target = SimpleNamespace(hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace())
    draft = SimpleNamespace(hf_config=SimpleNamespace(rope_parameters=None), hf_text_config=None)
    assert steps.align_draft_rope_with_target(target, draft) is False


def test_flat_token_rows_gives_one_row_per_token():
    steps = _steps()
    prefill = torch.arange(1 * 7 * 16, dtype=torch.float32).view(1, 7, 16)  # one bucket row of a prefill
    decode = torch.arange(5 * 1 * 16, dtype=torch.float32).view(5, 1, 16)  # [vbs, 1, width] after a decode
    flat = torch.arange(3 * 16, dtype=torch.float32).view(3, 16)
    assert steps.flat_token_rows(prefill).shape == (7, 16)
    assert torch.equal(steps.flat_token_rows(prefill), prefill[0])
    assert steps.flat_token_rows(decode).shape == (5, 16)
    assert steps.flat_token_rows(flat) is not None and steps.flat_token_rows(flat).shape == (3, 16)


def test_the_proposer_flattens_the_target_hidden_and_aligns_the_rope_before_the_base_init():
    src = PROPOSER_SRC.read_text()
    assert "target_hidden_states = flat_token_rows(target_hidden_states)" in src
    init = src[src.index("def __init__(self, vllm_config"):src.index("def _init_draft_rank_log")]
    assert init.index("align_draft_rope_with_target(") < init.index("super().__init__(")
