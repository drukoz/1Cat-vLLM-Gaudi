# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the boundary commits of accepted verification tokens (M3-T10b)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_commit import boundary_commits, commit_conv, commit_ssm  # noqa: E402

BS = 128
K = 3


def _store(rows):
    return torch.tensor(rows, dtype=torch.int32)


def test_no_boundary_in_the_verified_window():
    # ctx=100, a=4: the boundary token is position 127 (t=27), far outside the window.
    pairs = boundary_commits([100], [4], BS, _store([[5, 1, 2, 3]]), [[7]])
    assert pairs == []


def test_boundary_completed_by_accepted_token():
    # ctx=125: token t=2 sits at position 127 and completes block 0 (slot 7).
    pairs = boundary_commits([125], [4], BS, _store([[5, 1, 2, 3]]), [[7, 8]])
    assert pairs == [(0, 2, 2, 7)]


def test_boundary_token_rejected_with_the_draft_tail():
    # Same step, accepted = 2 (only tokens 0 and 1): t=2 was rejected, no commit.
    pairs = boundary_commits([125], [2], BS, _store([[5, 1, 2, 3]]), [[7, 8]])
    assert pairs == []


def test_token_zero_boundary_is_the_store_lattices_job():
    # ctx=127: token 0 completes the block, and align_lattice already stores token 0 into it.
    pairs = boundary_commits([127], [4], BS, _store([[5, 1, 2, 3]]), [[7, 8]])
    assert pairs == []


def test_second_block_boundary_and_two_requests():
    # Request 0: ctx=383 (block 2 ends at 383), t=0 only -> no pair; request 1: ctx=253, t=2
    # at position 255 completes block 1 (block_rows[1][1] = 17).
    store = _store([[5, 1, 2, 3], [6, 1, 2, 3]])
    pairs = boundary_commits([383, 253], [1, 4], BS, store, [[7, 8, 9], [16, 17, 18]])
    assert pairs == [(1, 2, 2, 17)]


def test_aligned_context_boundary_uses_scratch_slot_of_token_t():
    # ctx=128 (aligned): t=0 stores into block 1 via store[0]=11 (no pair); a step reaching
    # the next boundary at t=... would use store[i, t]; ctx=126 with a=3 -> t=1 at position
    # 127 completes block 0 (slot 10), src = store[0, 1] = the scratch slot.
    pairs = boundary_commits([126], [3], BS, _store([[10, 11, 12, 13]]), [[10, 11]])
    assert pairs == [(0, 1, 11, 10)]


def test_negative_src_slot_is_skipped():
    # Padding rows carry the pad block id (-1 for compact groups): no commit for them.
    pairs = boundary_commits([125, 125], [4, 4], BS, _store([[5, 1, 2, 3], [5, 1, -1, 3]]), [[7], [7]])
    assert pairs == [(0, 2, 2, 7)]


def test_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        boundary_commits([100], [4, 4], BS, _store([[5, 1, 2, 3]]), [[7]])
    with pytest.raises(ValueError):
        boundary_commits([100], [4], 0, _store([[5, 1, 2, 3]]), [[7]])
    with pytest.raises(ValueError):
        # accepted=5 needs lattice width 5 (t up to 4); boundary t=4 lands on a short lattice.
        boundary_commits([123], [5], BS, _store([[5, 1, 2, 3]]), [[7]])
    with pytest.raises(ValueError):
        boundary_commits([125], [4], BS, _store([[5, 1, 2, 3]]), [[]])


def test_commit_ssm_copies_the_exact_rows():
    state = torch.zeros(10, 4)
    state[2] = 1.0
    state[3] = 5.0
    commit_ssm(state, [(0, 2, 2, 7), (1, 1, 3, 8)])
    assert torch.equal(state[7], torch.ones(4))
    assert torch.equal(state[8], torch.full((4, ), 5.0))
    assert torch.equal(state[2], torch.ones(4)) and torch.equal(state[3], torch.full((4, ), 5.0))
    assert state[0].sum() == 0.0 and state[9].sum() == 0.0  # untouched slots stay zero


def test_commit_conv_window_rows_land_at_the_head():
    state = torch.zeros(10, 6, 5)  # [slots, window_len + k, dim] with window_len=3
    state[1, :, 0] = torch.arange(6)
    commit_conv(state, [(0, 2, 1, 8)], window_len=3)
    assert torch.equal(state[8, 0:3, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert state[8, 3:, 0].sum() == 0.0  # rows beyond the window stay untouched
    assert torch.equal(state[1, :, 0], torch.arange(6))  # the src slot is preserved


def test_commit_conv_supports_the_transposed_layout():
    state = torch.zeros(10, 2, 6)  # [slots, dim, window_len + k] with dim < window_len
    state[1, 0, :] = torch.arange(6)
    commit_conv(state, [(0, 2, 1, 8)], window_len=3)
    assert torch.equal(state[8, 0, 0:3], torch.tensor([2.0, 3.0, 4.0]))
    assert state[8, 0, 3:].sum() == 0.0


def test_commit_conv_t_per_pair_override():
    state = torch.zeros(10, 6, 5)
    state[1, :, 0] = torch.arange(6)
    state[2, :, 0] = torch.arange(6) + 10
    commit_conv(state, [(0, 0, 1, 8), (0, 0, 2, 9)], window_len=3, t_per_pair=[1, 2])
    assert torch.equal(state[8, 0:3, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(state[9, 0:3, 0], torch.tensor([12.0, 13.0, 14.0]))


def test_commit_conv_rejects_bad_shapes():
    with pytest.raises(ValueError):
        commit_conv(torch.zeros(10, 2, 2), [(0, 1, 1, 8)], window_len=3)
    with pytest.raises(ValueError):
        commit_conv(torch.zeros(10, 6), [(0, 1, 1, 8)], window_len=3)
    with pytest.raises(ValueError):
        commit_conv(torch.zeros(10, 6, 5), [(0, 1, 1, 8)], window_len=0)
    with pytest.raises(ValueError):
        commit_conv(torch.zeros(10, 6, 5), [(0, 1, 1, 8)], window_len=3, t_per_pair=[])


def test_empty_pair_lists_are_no_ops():
    state = torch.ones(10, 4)
    commit_ssm(state, [])
    commit_conv(state.unsqueeze(-1), [], window_len=2)
    assert torch.equal(state, torch.ones(10, 4))


def test_commit_conv_sequential_pairs_read_before_write() -> None:
    """Consecutive pairs on one slot (src == dst): later pairs must read the pre-write rows."""
    state = torch.arange(24, dtype=torch.float32).view(2, 6, 2)
    original = state.clone()
    commit_conv(state, [(0, 1, 1, 1), (0, 3, 1, 1)], window_len=3)
    torch.testing.assert_close(state[1, 0:3], original[1, 3:6])
