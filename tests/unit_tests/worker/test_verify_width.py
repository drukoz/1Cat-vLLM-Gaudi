# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the verification batch width with ragged drafts (``vllm_gaudi/v1/worker/verify_width.py``)."""

import pytest
import torch

from vllm_gaudi.v1.worker.verify_width import pad_query_rows, verification_width


def test_widest_row_without_recurrent_state():
    assert verification_width([1, 4, 2], [4, 4, 4], recurrent_spec=False) == 4


def test_plain_step_stays_one_token_wide():
    # A plain decode step loads the committed checkpoint through a scalar offset (any accepted count).
    assert verification_width([1, 1], [4, 2], recurrent_spec=True) == 1


def test_narrow_step_after_a_full_acceptance_is_widened():
    # The previous step accepted k = 3 drafts (num_accepted 4); this step drafts one token: the lattice must keep
    # column 3, where the state after the fourth committed token was stored.
    assert verification_width([2], [4], recurrent_spec=True) == 4
    assert verification_width([2, 3], [1, 4], recurrent_spec=True) == 4


def test_wide_step_keeps_its_width():
    assert verification_width([4, 2], [2, 3], recurrent_spec=True) == 4


def test_mtp_batch_is_never_widened():
    k = 3
    for accepted in range(1, k + 2):
        assert verification_width([1 + k] * 3, [accepted] * 3, recurrent_spec=True) == 1 + k


def test_no_accepted_counts_known():
    assert verification_width([3, 2], [], recurrent_spec=True) == 3


@pytest.mark.parametrize("scheduled,accepted", [([], []), ([0, 2], [1, 1]), ([2], [0])])
def test_invalid_inputs_raise(scheduled, accepted):
    with pytest.raises(ValueError):
        verification_width(scheduled, accepted, recurrent_spec=True)


def test_pad_query_rows_pads_to_the_batch_width():
    rows = torch.split(torch.tensor([7, 8, 9, 5]), [3, 1, 0])
    out = pad_query_rows(rows, 4)
    assert out.tolist() == [[7, 8, 9, 0], [5, 0, 0, 0], [0, 0, 0, 0]]


def test_pad_query_rows_matches_pad_sequence_at_the_widest_row():
    rows = torch.split(torch.arange(1, 6), [2, 3])
    assert pad_query_rows(rows, 3).tolist() == [[1, 2, 0], [3, 4, 5]]


def test_pad_query_rows_refuses_a_row_wider_than_the_batch():
    with pytest.raises(ValueError):
        pad_query_rows(torch.split(torch.arange(4), [4]), 3)
