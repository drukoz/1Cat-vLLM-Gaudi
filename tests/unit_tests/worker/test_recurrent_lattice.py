# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the align-mode verification lattice builder (M3-T10a, T10c)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_lattice import align_lattice, align_lattices, align_running_block  # noqa: E402

from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup, size_id  # noqa: E402

# Row of the CPU trace of the design (bs = 4, k = 3): slots 8..23, one per block index.
R = list(range(8, 24))


@pytest.fixture(params=SPEC_SIZES, ids=size_id)
def spec_k(request) -> int:
    """Each speculative size k the Flash-Next tests run at, after flash_next_spec_sizes' startup check."""
    require_startup(request.param, prefix_caching=True)
    return request.param


def test_align_running_block_matches_the_cpu_trace():
    # Verification steps (sched = 1 + k = 4): the block of the last scheduled token.
    assert align_running_block(33, 4, 4) == 9
    assert align_running_block(36, 4, 4) == 9
    assert align_running_block(37, 4, 4) == 10
    assert align_running_block(40, 4, 4) == 10
    # Plain decode steps (sched = 1).
    assert align_running_block(33, 1, 4) == 8
    assert align_running_block(36, 1, 4) == 9
    # Gaudi-scale rows (bs = 512).
    assert align_running_block(510, 4, 512) == 1
    assert align_running_block(1018, 4, 512) == 1
    assert align_running_block(1022, 4, 512) == 2
    assert align_running_block(0, 4, 4) == 0


def test_align_running_block_rejects_bad_arguments():
    with pytest.raises(ValueError, match="block_size"):
        align_running_block(33, 4, 0)
    with pytest.raises(ValueError, match="block_size"):
        align_running_block(33, 4, -1)
    with pytest.raises(ValueError, match="context_len"):
        align_running_block(-1, 4, 4)
    with pytest.raises(ValueError, match="num_scheduled"):
        align_running_block(33, 0, 4)
    with pytest.raises(ValueError, match="num_scheduled"):
        align_running_block(33, -1, 4)


def test_align_lattice_is_the_running_block_window():
    # load == store == row[curr : curr + 1 + k]: the trace's lattice of a running block 9, then 10.
    assert align_lattice(R, 33, 4, 4, 3) == ([17, 18, 19, 20], [17, 18, 19, 20])
    assert align_lattice(R, 37, 4, 4, 3) == ([18, 19, 20, 21], [18, 19, 20, 21])
    load, store = align_lattice(R, 36, 4, 4, 3)
    assert load == store == [17, 18, 19, 20]


def test_lattice_is_one_slot_per_token_at_every_size(spec_k):
    """A verification step of 1 + k tokens: load == store == the 1 + k distinct slots from the running block on."""
    t = 1 + spec_k
    row = list(range(8, 40))
    for bs, ctx in ((4, 33), (4, 36), (4, 37), (8, 33), (8, 39), (512, 510), (512, 1018), (512, 1023 - spec_k)):
        curr = (ctx + t - 1) // bs  # the block of the last of the 1 + k scheduled tokens
        assert align_running_block(ctx, t, bs) == curr
        load, store = align_lattice(row, ctx, t, bs, spec_k)
        assert load == store == row[curr:curr + t], f"bs {bs} ctx {ctx}: {load}"
        assert len(set(load)) == t
        # The window ends on the last scratch slot: a row one entry shorter is refused.
        with pytest.raises(ValueError, match=f"needs at least {curr + t} "):
            align_lattice(row[:curr + t - 1], ctx, t, bs, spec_k)
    # The last token of the step is the first of a new block: the window has moved by one block.
    assert align_lattice(row, 1024 - spec_k, t, 512, spec_k)[0] == row[2:2 + t]


def test_lattices_are_one_column_per_token_at_every_size(spec_k):
    t = 1 + spec_k
    table = torch.arange(100, 124, dtype=torch.int64).view(2, 12)
    load, store = align_lattices(table, req_indices=[1, 0], context_lens=[5, 0], num_scheduled=[t, t], block_size=4,
                                 k=spec_k, padded_batch_size=3, pad_block_id=97)
    assert load.shape == store.shape == (3, t)
    first, second = (5 + spec_k) // 4, spec_k // 4  # the blocks of the last scheduled tokens (positions 5 + k and k)
    assert load[0].tolist() == store[0].tolist() == table[1, first:first + t].tolist()
    assert load[1].tolist() == store[1].tolist() == table[0, second:second + t].tolist()
    assert torch.all(load[2] == 97) and torch.all(store[2] == 97)


def test_align_lattice_fresh_request_starts_at_block_zero():
    # No computed token: curr = cdiv(1, 128) - 1 = 0, no null load slot any more.
    load, store = align_lattice([10, 11, 12, 13], 0, 1, 128, 2)
    assert load == store == [10, 11, 12]


def test_zero_k_lattice_is_a_single_slot():
    assert align_lattice([7, 8], 5, 1, 128, 0) == ([7], [7])


def test_align_lattice_on_the_real_manager_rows():
    # RUK-2847: index-stable rows of the real manager, nulls only strictly below the running block.
    assert align_lattice([0, 2, 3, 4, 17], 1018, 4, 512, 3) == ([2, 3, 4, 17], [2, 3, 4, 17])
    assert align_lattice([0, 2, 3, 4, 17, 13], 1022, 4, 512, 3) == ([3, 4, 17, 13], [3, 4, 17, 13])
    assert align_lattice([0, 0, 3, 4, 17, 13], 1026, 4, 512, 3) == ([3, 4, 17, 13], [3, 4, 17, 13])


def test_short_block_row_is_rejected():
    # curr = 2, so the window needs 2 + 1 + 3 = 6 entries; the row has 5.
    with pytest.raises(ValueError, match="needs at least 6"):
        align_lattice([0, 2, 3, 4, 17], 1022, 4, 512, 3)
    # curr = 1 with k = 2 needs 4 entries.
    with pytest.raises(ValueError, match="needs at least 4"):
        align_lattice([10, 11, 12], 128, 1, 128, 2)
    # k = 0 on a boundary still needs the running block itself.
    with pytest.raises(ValueError, match="needs at least 2"):
        align_lattice([10], 128, 1, 128, 0)


def test_null_block_inside_the_window_is_rejected():
    # The manager nulled block row index 1, and that entry is inside the window.
    with pytest.raises(ValueError, match="null block"):
        align_lattice([1, 0, 3, 4, 5], 510, 4, 512, 3)
    # A custom null id is named with the index of the entry.
    with pytest.raises(ValueError, match="index 3"):
        align_lattice([5, 6, 7, 97, 8], 0, 1, 128, 4, null_block=97)


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
    # Row 0 uses block-table row 1: running block 1, so the window is [21, 22, 23].
    assert load[0].tolist() == [21, 22, 23]
    assert store[0].tolist() == [21, 22, 23]
    # Row 1 uses block-table row 0: fresh request, running block 0.
    assert load[1].tolist() == [10, 11, 12]
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
