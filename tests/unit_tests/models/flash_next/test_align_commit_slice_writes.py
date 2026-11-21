# SPDX-License-Identifier: Apache-2.0
"""The align-mode state copies write whole state rows on the pool itself, never through a slice (card job 00133).

With prefix caching and speculative decoding, a request whose accepted tokens complete a recurrent block gets its
conv window and ssm row copied into the block's slot (``recurrent_commit.commit_conv`` / ``commit_ssm``, called by
the boundary commit after sampling and by the pre-step copy before the forward). ``commit_conv`` wrote the window
with ``state[dst].narrow(...).copy_(window)``: a write through a slice of the GDN conv pool, which the lazy bridge
turns into ``hpu::slice_insert``. Card job 00133 (spec-lookup, TP4, n-gram k = 5, fused verify, prefix caching):
the first request crossed the 896-token block during decode and ran; on the next request an HPU graph captured
after that commit found the conv pool without data, ``ValidateSyncInputTensors tensor_data is empty`` at the
first input of ``gdn_core_verify``, and the server died. ``commit_ssm`` already wrote whole rows with
``index_copy_`` on the pool and its pool was not the one that failed. ``SliceWriteProbe`` models the card rule on
the CPU; these tests hold both copies to it and to the values of the old slice copy.
"""
import pytest
import torch

from tests.unit_tests.models.flash_next.slice_write_probe import SliceWriteProbe
from vllm_gaudi.v1.worker.recurrent_commit import commit_conv, commit_ssm


def _old_commit_conv(state: torch.Tensor, pairs, window_len: int, ts, row_dim: int) -> torch.Tensor:
    """The slice copy as it was (two-phase: every window read before any write), on a plain copy."""
    out = state.clone()
    windows = [out[src].narrow(row_dim - 1, t, window_len).clone() for (_i, _t, src, _dst), t in zip(pairs, ts)]
    for (_i, _t, _src, dst), window in zip(pairs, windows):
        out[dst].narrow(row_dim - 1, 0, window_len).copy_(window)
    return out


@pytest.mark.parametrize("layout", ["rows_on_dim1", "rows_on_dim2"])
@pytest.mark.parametrize("pairs,ts", [
    ([(0, 2, 3, 5)], [2]),  # one commit, window of token 2 into another slot
    ([(0, 0, 4, 4)], [3]),  # src == dst: the store lattice's column-0 slot commits into its own head
    ([(0, 1, 1, 2), (1, 2, 2, 6)], [1, 2]),  # a dst of one pair is the src of the next: read before write
    ([(0, 1, 1, 6), (1, 3, 3, 6)], [1, 3]),  # two pairs on one dst: the later one wins, as with the slice copy
])
def test_commit_conv_writes_whole_rows_and_keeps_the_values(layout, pairs, ts):
    torch.manual_seed(0)
    window_len, k = 3, 3
    # [slots, S, dim] (the GDN / PLE layout) and [slots, dim, S]; commit_conv finds the row axis by its length,
    # so the second layout keeps dim below window_len
    shape = (8, window_len + k, 4) if layout == "rows_on_dim1" else (8, 2, window_len + k)
    row_dim = 1 if layout == "rows_on_dim1" else 2
    base = torch.randn(shape)
    want = _old_commit_conv(base, pairs, window_len, ts, row_dim)
    state = SliceWriteProbe.wrap(base.clone())
    commit_conv(state, pairs, window_len, t_per_pair=ts)
    assert SliceWriteProbe.violations == []
    assert torch.equal(torch.Tensor(state), want)


def test_commit_ssm_writes_whole_rows():
    torch.manual_seed(1)
    base = torch.randn(8, 2, 3, 3)
    state = SliceWriteProbe.wrap(base.clone())
    commit_ssm(state, [(0, 1, 2, 5), (1, 0, 3, 1)])
    assert SliceWriteProbe.violations == []
    want = base.clone()
    want[5], want[1] = base[2], base[3]
    assert torch.equal(torch.Tensor(state), want)


def test_no_pair_writes_nothing():
    state = SliceWriteProbe.wrap(torch.zeros(4, 6, 2))
    commit_conv(state, [], 3)
    commit_ssm(state, [])
    assert SliceWriteProbe.violations == [] and torch.count_nonzero(torch.Tensor(state)) == 0
