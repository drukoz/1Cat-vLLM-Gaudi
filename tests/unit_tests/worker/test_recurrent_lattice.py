# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the align-mode verification lattice builder (M3-T10a)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_lattice import align_lattice, align_lattices  # noqa: E402


def test_context_zero_loads_null_and_stores_block_zero():
    block_row = [10, 11, 12, 13, 14]
    load, store = align_lattice(block_row, context_len=0, num_scheduled=1, block_size=128, k=2)
    # No computed token: the committed state is the null block; token 0 starts block 0.
    assert load[0] == 0
    assert store[0] == 10
    # n_pos = ceil(1 / 128) = 1: the k scratch blocks follow the one position block.
    assert load[1:] == [11, 12]
    assert store[1:] == [11, 12]
    assert len(load) == len(store) == 3


def test_mid_block_context_loads_and_stores_the_same_block():
    block_row = [10, 11, 12, 13]
    load, store = align_lattice(block_row, context_len=127, num_scheduled=1, block_size=128, k=1)
    # The block of token 126 is the block of token 127: a partial block, safe to overwrite.
    assert load[0] == 10
    assert store[0] == 10
    assert load[1:] == [11]
    assert store[1:] == [11]


def test_boundary_context_loads_previous_block_and_stores_next_block():
    block_row = [10, 11, 12, 13, 14]
    load, store = align_lattice(block_row, context_len=128, num_scheduled=1, block_size=128, k=2)
    # context_len % block_size == 0: block 0 is complete (a checkpoint) and must not be written;
    # token 0's state goes to block 1. The committed state is read from block 0.
    assert load[0] == 10
    assert store[0] == 11
    # n_pos = ceil(129 / 128) = 2: scratch blocks start after the two position blocks.
    assert load[1:] == [12, 13]
    assert store[1:] == [12, 13]


def test_scratch_slots_follow_the_scheduled_tokens():
    block_row = list(range(20))
    load, store = align_lattice(block_row, context_len=100, num_scheduled=30, block_size=128, k=3)
    # n_pos = ceil(130 / 128) = 2 even though token 129 lands in block 1: the manager keeps the
    # k scratch blocks right after the position blocks of all scheduled tokens.
    assert load[1:] == [2, 3, 4]
    assert store[1:] == [2, 3, 4]


def test_zero_k_lattice_is_a_single_slot():
    load, store = align_lattice([7, 8], context_len=5, num_scheduled=1, block_size=128, k=0)
    assert load == [7]
    assert store == [7]


def test_custom_null_block():
    load, store = align_lattice([5, 6, 7], context_len=0, num_scheduled=1, block_size=128, k=1, null_block=97)
    assert load == [97, 6]
    assert store == [5, 6]


def test_short_block_row_is_rejected():
    # n_pos + k = 2 + 2 = 4 blocks required.
    with pytest.raises(ValueError, match="needs at least 4"):
        align_lattice([10, 11, 12], context_len=128, num_scheduled=1, block_size=128, k=2)
    # k = 0 on the boundary needs store_pos + 1 = 2 blocks.
    with pytest.raises(ValueError, match="needs at least 2"):
        align_lattice([10], context_len=128, num_scheduled=0, block_size=128, k=0)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="block_size"):
        align_lattice([1], context_len=0, num_scheduled=1, block_size=0, k=1)
    with pytest.raises(ValueError, match="context_len"):
        align_lattice([1], context_len=-1, num_scheduled=1, block_size=128, k=1)
    with pytest.raises(ValueError, match="num_scheduled"):
        align_lattice([1], context_len=0, num_scheduled=-1, block_size=128, k=1)
    with pytest.raises(ValueError, match="k"):
        align_lattice([1], context_len=0, num_scheduled=1, block_size=128, k=-1)


def test_align_lattices_pads_with_pad_block_id():
    table = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int64)
    load, store = align_lattices(
        table,
        req_indices=[1, 0],
        context_lens=[128, 0],
        num_scheduled=[1, 1],
        block_size=128,
        k=2,
        padded_batch_size=4,
        pad_block_id=97,
    )
    assert load.shape == store.shape == (4, 3)
    assert load.dtype == torch.int32 and store.dtype == torch.int32
    # Row 0 uses block-table row 1: boundary context loads 20, stores 21, scratch 22, 23.
    assert load[0].tolist() == [20, 22, 23]
    assert store[0].tolist() == [21, 22, 23]
    # Row 1 uses block-table row 0: fresh request, null load, store 10, scratch 11, 12.
    assert load[1].tolist() == [0, 11, 12]
    assert store[1].tolist() == [10, 11, 12]
    # Padded rows are the pad block id everywhere.
    assert torch.all(load[2:] == 97)
    assert torch.all(store[2:] == 97)


def test_align_lattices_rejects_mismatched_arguments():
    table = torch.tensor([[10, 11]], dtype=torch.int64)
    with pytest.raises(ValueError, match="equal length"):
        align_lattices(table, req_indices=[0], context_lens=[0, 128], num_scheduled=[1, 1], block_size=128, k=1,
                       padded_batch_size=2, pad_block_id=97)
    with pytest.raises(ValueError, match="padded_batch_size"):
        align_lattices(table, req_indices=[0, 0], context_lens=[0, 0], num_scheduled=[1, 1], block_size=128, k=1,
                       padded_batch_size=1, pad_block_id=97)


if __name__ == "__main__":
    pytest.main([__file__])
