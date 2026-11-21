# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_SLOT_SHIFT: the sparse QSA decode's slot mapping without integer division (qsa_hpu_attention).

Forms 1 (shift and mask) and 2 (through float32) must give exactly the slots of form 0 (``//`` and ``%``) for every
attended position, including the -1 padding, at the served geometry (block size 128, 1,024 blocks per row: positions
to 131,071) and at the small block sizes the tests use; a block size that is not a power of two keeps form 0.
"""
import pytest
import torch

from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import decode_selected_slots
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import expand_selection


def _case(rows, width, blocks, block_size, seed):
    g = torch.Generator().manual_seed(seed)
    selected = torch.randint(-1, blocks * block_size, (rows, width), generator=g, dtype=torch.int32)
    selected[0, :3] = torch.tensor([-1, 0, blocks * block_size - 1], dtype=torch.int32)
    table = torch.randperm(rows * blocks * 2, generator=g)[:rows * blocks].reshape(rows, blocks).to(torch.int32)
    return selected, table


@pytest.mark.parametrize("form", [1, 2])
@pytest.mark.parametrize("block_size,blocks", [(128, 1024), (16, 64), (1, 32)])
def test_forms_equal_divmod(form, block_size, blocks):
    selected, table = _case(32, 2051, blocks, block_size, seed=block_size + form)
    want = decode_selected_slots(selected, table, block_size, form=0)
    got = decode_selected_slots(selected, table, block_size, form=form)
    assert got.dtype == want.dtype == torch.int64
    assert torch.equal(got, want)


@pytest.mark.parametrize("form", [1, 2])
def test_not_power_of_two_keeps_divmod(form):
    selected, table = _case(4, 51, 40, 24, seed=5)
    assert torch.equal(decode_selected_slots(selected, table, 24, form=form),
                       decode_selected_slots(selected, table, 24, form=0))


@pytest.mark.parametrize("form", [1, 2])
def test_served_selection_layout(form):
    """The decode's own input: expand_selection of a top-k block selection (ratio 4, tail of 3, -1 padding)."""
    g = torch.Generator().manual_seed(11)
    rows, topk, ratio, block_size, blocks = 16, 512, 4, 128, 1024
    positions = torch.randint(0, blocks * block_size, (rows, ), generator=g)
    positions[0] = 2  # a row with fewer complete blocks than top-k: -1 entries
    visible = (positions + 1) // ratio
    picked = torch.full((rows, topk), -1, dtype=torch.int32)
    for r in range(rows):
        n = min(topk, int(visible[r]))
        if n:
            picked[r, :n] = torch.randperm(int(visible[r]), generator=g)[:n].to(torch.int32)
    selected = expand_selection(picked, positions, ratio)
    table = torch.randperm(rows * blocks * 2, generator=g)[:rows * blocks].reshape(rows, blocks).to(torch.int32)
    assert torch.equal(decode_selected_slots(selected, table, block_size, form=form),
                       decode_selected_slots(selected, table, block_size, form=0))


def test_flag_default_is_divmod(monkeypatch):
    import vllm_gaudi.envs as envs
    monkeypatch.delenv("VLLM_HPU_QSA_SLOT_SHIFT", raising=False)
    assert envs.VLLM_HPU_QSA_SLOT_SHIFT == 0
    selected, table = _case(4, 51, 16, 16, seed=7)
    monkeypatch.setenv("VLLM_HPU_QSA_SLOT_SHIFT", "1")
    assert torch.equal(decode_selected_slots(selected, table, 16), decode_selected_slots(selected, table, 16, form=0))
