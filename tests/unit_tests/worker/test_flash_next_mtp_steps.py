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
