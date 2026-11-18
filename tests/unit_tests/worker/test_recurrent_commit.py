# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the boundary commits and pre-step copies of align-mode MTP (M3-T10b, T10c)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_commit import (align_precopy_plan, boundary_commits, commit_conv, commit_ssm,
                                                   precopy_slot_pairs, self_commit_requests)  # noqa: E402

BS = 128
K = 3
# Row of the CPU trace of the design (bs = 4, k = 3): slots 8..23, one per block index.
R = list(range(8, 24))


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


def test_token_zero_is_committed_like_any_other_token():
    # ctx=127: token 0 completes the block. With load == store the lattice no longer special-cases
    # token 0, so the boundary commit copies lattice column 0 into the completed block.
    pairs = boundary_commits([127], [4], BS, _store([[5, 1, 2, 3]]), [[7, 8]])
    assert pairs == [(0, 0, 5, 7)]


def test_token_zero_boundary_and_the_self_commit_of_the_running_block():
    # The design trace, bs=4, store lattice [17, 18, 19, 20] of the running block 9 (slot 17).
    store = _store([[17, 18, 19, 20]])
    # ctx=35: token 0 completes block 8 (row[8] = 16) from lattice column 0.
    assert boundary_commits([35], [4], 4, store, [R]) == [(0, 0, 17, 16)]
    # ctx=36: token 3 completes block 9, whose slot IS lattice column 0: the step commits into
    # its own running block, so its next accepted count must be 1.
    self_pair = boundary_commits([36], [4], 4, store, [R])
    assert self_pair == [(0, 3, 20, 17)]
    assert self_commit_requests(self_pair, store) == {0}
    # ctx=33, accepted=3: token 2 completes block 8, which is not the running block.
    other = boundary_commits([33], [3], 4, store, [R])
    assert other == [(0, 2, 19, 16)]
    assert self_commit_requests(other, store) == set()


def test_second_block_boundary_and_two_requests():
    # Request 0: ctx=383 (block 2 ends at 383), t=0 completes it into block_rows[0][2] = 9;
    # request 1: ctx=253, t=2 at position 255 completes block 1 (block_rows[1][1] = 17).
    store = _store([[5, 1, 2, 3], [6, 1, 2, 3]])
    pairs = boundary_commits([383, 253], [1, 4], BS, store, [[7, 8, 9], [16, 17, 18]])
    assert pairs == [(0, 0, 5, 9), (1, 2, 2, 17)]


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


def test_align_precopy_plan_moves_a_moved_window():
    # Post-prefill step, prev unknown: prev = (33 - 1) // 4 = 8, running block 9, bias = 1 - 1 = 0.
    assert align_precopy_plan([None], [33], [4], [1], 4) == ([(0, 0, 8, 9)], [9], [1])
    # prev == curr: the running state is already in the running block, accepted count survives.
    assert align_precopy_plan([9], [36], [4], [3], 4) == ([], [9], [3])
    # Crossed into block 10 with 4 accepted tokens: bias = 3, so rows [3, 3 + W) of block 9 move.
    assert align_precopy_plan([9], [37], [4], [4], 4) == ([(0, 3, 9, 10)], [10], [1])
    # Gaudi-scale row (bs = 512): prev 1 -> running block 2 with bias = 4 - 1 = 3.
    assert align_precopy_plan([1], [1022], [4], [4], 512) == ([(0, 3, 1, 2)], [2], [1])
    # Prefill step: no state to move, the running block is still recorded for the next step.
    assert align_precopy_plan([None], [0], [4], [1], 4) == ([], [0], [1])


def test_align_precopy_plan_normalize_rewinds_a_multi_token_window():
    # Same running block, 3 accepted tokens: the state sits at row offset 2 of block 9, so a
    # plain-decode step (1 scheduled token) rewinds it to the head.
    assert align_precopy_plan([9], [38], [1], [3], 4, normalize=True) == ([(0, 2, 9, 9)], [9], [1])
    # accepted == 1: the state is already at the head, nothing to rewind.
    assert align_precopy_plan([9], [38], [1], [1], 4, normalize=True) == ([], [9], [1])


def test_align_precopy_plan_keeps_requests_in_order():
    plan, prev_blocks, accepted = align_precopy_plan([None, 9], [33, 37], [4, 4], [1, 4], 4)
    assert plan == [(0, 0, 8, 9), (1, 3, 9, 10)]
    assert prev_blocks == [9, 10] and accepted == [1, 1]


def test_align_precopy_plan_rejects_bad_arguments():
    with pytest.raises(ValueError, match="equal length"):
        align_precopy_plan([9], [33], [4, 4], [1], 4)
    with pytest.raises(ValueError, match="equal length"):
        align_precopy_plan([9, 9], [33, 37], [4], [1, 4], 4)
    with pytest.raises(ValueError, match="accepted"):
        align_precopy_plan([9], [33], [4], [0], 4)
    # prev = 11 is past the running block 10 of a ctx=37 step: the row lost its index stability.
    with pytest.raises(ValueError, match="past the running block"):
        align_precopy_plan([11], [37], [4], [1], 4)


def test_precopy_slot_pairs_maps_block_indices_to_slots():
    # Blocks 9 -> 10 of the trace row are slots 17 -> 18; the ssm state of token bias sits in
    # block 9 + 3 = 12 (slot 20).
    assert precopy_slot_pairs([(0, 3, 9, 10)], [R]) == ([(0, 3, 17, 18)], [(0, 3, 20, 18)], [3])
    # Gaudi-scale row: conv 2 -> 3, ssm from block 1 + 3 = 4 (slot 17) into 3.
    assert precopy_slot_pairs([(0, 3, 1, 2)], [[0, 2, 3, 4, 17, 13]]) == ([(0, 3, 2, 3)], [(0, 3, 17, 3)], [3])


def test_precopy_slot_pairs_rejects_bad_references():
    # The manager nulled the source block: the copy would read the null slot.
    with pytest.raises(ValueError, match="null block"):
        precopy_slot_pairs([(0, 0, 1, 2)], [[0, 0, 3, 4, 17, 13]])
    with pytest.raises(ValueError, match="outside block_rows"):
        precopy_slot_pairs([(0, 0, 1, 9)], [[0, 2, 3]])
    # src + bias = 4 is outside a row of 4 entries.
    with pytest.raises(ValueError, match="outside block_rows"):
        precopy_slot_pairs([(0, 3, 1, 2)], [[0, 2, 3, 4]])
    with pytest.raises(ValueError, match="outside block_rows"):
        precopy_slot_pairs([(3, 0, 1, 2)], [[0, 2, 3]])


def test_precopy_pairs_feed_the_commit_copies():
    # bias = 1: rows [1, 1 + W) of the source slot land at the head of the destination slot.
    state = torch.zeros(24, 5, 2)
    state[17, :, 0] = torch.arange(5)
    conv_pairs, ssm_pairs, ts = precopy_slot_pairs([(0, 1, 9, 10)], [R])
    commit_conv(state, conv_pairs, window_len=3, t_per_pair=ts)
    assert torch.equal(state[18, 0:3, 0], torch.tensor([1.0, 2.0, 3.0]))
    commit_ssm(torch.zeros(24, 2), ssm_pairs)


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
