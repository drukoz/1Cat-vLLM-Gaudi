# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for apply_align_precopy, the runner-side pre-step align copy (M3-T10c)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import torch  # noqa: E402

from vllm_gaudi.v1.worker.recurrent_commit import apply_align_precopy  # noqa: E402

BS = 4
# Row of the CPU trace of the design (bs = 4): slots 8..23, one per block index.
R = list(range(8, 24))


def _make_state():
    torch.manual_seed(0)
    ssm = torch.randn(24, 2, 2)
    conv = torch.randn(24, 6, 2)
    registry = {0: [(conv, "conv", 3), (ssm, "ssm", 0)]}
    rows = {0: [list(R)]}
    return ssm, conv, registry, rows


def test_first_step_moves_the_running_state_into_the_running_block():
    # No prior running block: the default prev = (33 - 1) // 4 = 8, curr = cdiv(37, 4) - 1 = 9.
    ssm, conv, registry, rows = _make_state()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {}, {}
    plan = apply_align_precopy(registry, rows, ["r0"], running, accepted, [33], [4], BS, False)
    assert plan == [(0, 0, 8, 9)]
    assert torch.equal(ssm[17], old_ssm[16])
    assert torch.equal(conv[17, 0:3], old_conv[16, 0:3])
    assert running == {"r0": 9}
    assert accepted == {"r0": 1}


def test_window_move_reads_the_ssm_state_at_the_accepted_bias():
    # prev = 9, curr = cdiv(41, 4) - 1 = 10, accepted = 4: the state after the 4th accepted
    # token sits in block 9 + 3 = 12 (ssm); the conv window moves from row 3 of block 9.
    ssm, conv, registry, rows = _make_state()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {"r0": 9}, {"r0": 4}
    plan = apply_align_precopy(registry, rows, ["r0"], running, accepted, [37], [4], BS, False)
    assert plan == [(0, 3, 9, 10)]
    assert torch.equal(ssm[18], old_ssm[20])
    assert torch.equal(conv[18, 0:3], old_conv[17, 3:6])
    assert running == {"r0": 10}
    assert accepted == {"r0": 1}


def test_window_that_stays_put_copies_nothing():
    # prev = curr = 9 and normalize off: no plan, no copies, the accepted count survives.
    ssm, conv, registry, rows = _make_state()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {"r0": 9}, {"r0": 3}
    plan = apply_align_precopy(registry, rows, ["r0"], running, accepted, [36], [4], BS, False)
    assert plan == []
    assert torch.equal(ssm, old_ssm)
    assert torch.equal(conv, old_conv)
    assert running == {"r0": 9}
    assert accepted == {"r0": 3}


def test_normalize_gathers_a_multi_token_window_in_place():
    # prev == curr = 9 with 3 accepted tokens left in the window: normalize moves the state
    # after the last accepted token (block 9 + bias 2, slot 19) into the running slot 17.
    ssm, conv, registry, rows = _make_state()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {"r0": 9}, {"r0": 3}
    plan = apply_align_precopy(registry, rows, ["r0"], running, accepted, [38], [1], BS, True)
    assert plan == [(0, 2, 9, 9)]
    assert torch.equal(ssm[17], old_ssm[19])
    assert torch.equal(conv[17, 0:3], old_conv[17, 2:5])
    assert running == {"r0": 9}
    assert accepted == {"r0": 1}


def test_two_requests_only_the_moving_one_changes():
    torch.manual_seed(0)
    ssm = torch.randn(48, 2, 2)
    conv = torch.randn(48, 6, 2)
    registry = {0: [(conv, "conv", 3), (ssm, "ssm", 0)]}
    rows = {0: [list(range(8, 24)), list(range(30, 46))]}
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {"r1": 9}, {"r1": 3}
    plan = apply_align_precopy(registry, rows, ["r0", "r1"], running, accepted, [33, 36], [4, 4], BS, False)
    # r0 moves 8 -> 9; r1 has prev == curr = 9 and stays.
    assert plan == [(0, 0, 8, 9)]
    assert torch.equal(ssm[17], old_ssm[16])
    assert torch.equal(conv[17, 0:3], old_conv[16, 0:3])
    moved = set(torch.nonzero(ssm != old_ssm, as_tuple=False)[:, 0].tolist())
    moved |= set(torch.nonzero(conv != old_conv, as_tuple=False)[:, 0].tolist())
    assert moved == {17}
    assert running == {"r0": 9, "r1": 9}
    assert accepted == {"r0": 1, "r1": 3}


def test_backward_move_copies_the_running_state_into_the_lower_block():
    # Verification step ctx 33 + 4 scheduled left the state in block 9; 1 token accepted, a plain
    # decode (ctx 34, 1 scheduled) runs in block 8: the copy moves the state back 9 -> 8 (slots 17 -> 16).
    ssm, conv, registry, rows = _make_state()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {"r0": 9}, {"r0": 1}
    plan = apply_align_precopy(registry, rows, ["r0"], running, accepted, [34], [1], BS, True)
    assert plan == [(0, 0, 9, 8)]
    assert torch.equal(ssm[16], old_ssm[17])
    assert torch.equal(conv[16, 0:3], old_conv[17, 0:3])
    assert running == {"r0": 8}
    assert accepted == {"r0": 1}
    # The source slot is not clobbered by the backward copy.
    assert torch.equal(ssm[17], old_ssm[17])


def test_backward_move_reads_the_state_at_the_accepted_bias():
    # prev = 9, 2 accepted tokens, ctx 35 + 1 scheduled: curr = cdiv(36, 4) - 1 = 8 with bias 1, so the
    # ssm state is read from block 9 + 1 = 10 (slot 18) and the conv window from row 1 of block 9 (slot 17).
    ssm, conv, registry, rows = _make_state()
    old_ssm, old_conv = ssm.clone(), conv.clone()
    running, accepted = {"r0": 9}, {"r0": 2}
    plan = apply_align_precopy(registry, rows, ["r0"], running, accepted, [35], [1], BS, True)
    assert plan == [(0, 1, 9, 8)]
    assert torch.equal(ssm[16], old_ssm[18])
    assert torch.equal(conv[16, 0:3], old_conv[17, 1:4])
    assert running == {"r0": 8}
    assert accepted == {"r0": 1}
