# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``check_gdn_core_decode_indices`` and the ``VLLM_HPU_GDN_CORE_DECODE_DEBUG`` flag.

The check (contract: ``docs/design/flash-next/gdn-core-decode.md``, "Debug index check
(version 2)") rejects the two slot patterns the fused kernels cannot handle over the real
rows (index ``< num_rows``) whose two slots are both in ``[0, num_slots)``: a store slot
shared by two rows, and a store slot that is another row's load slot.  Padding rows beyond
``num_rows`` and out-of-range rows are ignored.
"""

import pytest
import torch

import vllm_gaudi.envs
from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeDebug, check_gdn_core_decode_indices

S = 8


def _idx(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype)


@pytest.mark.parametrize(
    "load,store,num_rows",
    [
        ([0, 1, 2], [0, 1, 2], 3),  # load == store on every row: the normal case
        ([1, 3, 5, -1], [2, 3, 6, -1], 4),  # row 1 loads and stores its own slot
        ([1, 1], [2, 3], 2),  # repeated load slots are fine
        ([0, 1, 7, 7, 7], [0, 1, 7, 7, 7], 2),  # padding rows share one dummy slot
        ([-1, -1], [-1, -1], 2),  # out-of-range rows are ignored
    ],
)
def test_passes(load, store, num_rows):
    check_gdn_core_decode_indices(_idx(load), _idx(store), S, num_rows)


def test_raises_duplicate_store():
    with pytest.raises(ValueError, match="both store slot 3"):
        check_gdn_core_decode_indices(_idx([1, 2]), _idx([3, 3]), S, 2)


def test_raises_reordered_cache():
    with pytest.raises(ValueError, match="row 0 stores slot 2, which row 1 loads"):
        check_gdn_core_decode_indices(_idx([1, 2]), _idx([2, 4]), S, 2)


def test_raises_dummy_slot_inside_num_rows():
    # Same pattern as the padding row of test_passes; num_rows = 4 makes the rows real,
    # which is what turns the passing case into a failure.
    with pytest.raises(ValueError, match="both store slot 7"):
        check_gdn_core_decode_indices(_idx([0, 1, 7, 7]), _idx([0, 1, 7, 7]), S, 4)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_accepts_index_dtypes(dtype):
    check_gdn_core_decode_indices(_idx([1, 2], dtype), _idx([3, 4], dtype), S, 2)


def test_accepts_2d_indices():
    check_gdn_core_decode_indices(_idx([[0, 1, 2]]), _idx([[0, 1, 2]]), S, 3)


def test_flag_default_and_value(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_GDN_CORE_DECODE_DEBUG", raising=False)
    assert vllm_gaudi.envs.VLLM_HPU_GDN_CORE_DECODE_DEBUG == 0
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE_DEBUG", "1")
    assert GdnCoreDecodeDebug(vllm_gaudi.envs.VLLM_HPU_GDN_CORE_DECODE_DEBUG) is GdnCoreDecodeDebug.CHECK
