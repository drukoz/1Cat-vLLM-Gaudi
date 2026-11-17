# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Flash-Next per-request block_table builder (M2)."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.v1.worker.flash_next_inputs import (build_block_table, model_requires_block_table)

PAD = 64


def test_two_rows_of_different_lengths_plus_padding_rows():
    block_tables = [[7, 3, 9], [12], [PAD], [PAD]]

    table = build_block_table(block_tables, padded_batch_size=4, max_blocks=5, pad_block_id=PAD)

    assert table.dtype == torch.int32
    assert table.shape == (4, 5)
    assert table.tolist() == [
        [7, 3, 9, PAD, PAD],
        [12, PAD, PAD, PAD, PAD],
        [PAD, PAD, PAD, PAD, PAD],
        [PAD, PAD, PAD, PAD, PAD],
    ]


def test_row_longer_than_max_blocks_raises():
    with pytest.raises(ValueError):
        build_block_table([[1, 2, 3]], padded_batch_size=2, max_blocks=2, pad_block_id=PAD)


def test_truncates_rows_beyond_padded_batch_size():
    table = build_block_table([[1], [2], [3]], padded_batch_size=2, max_blocks=2, pad_block_id=PAD)

    assert table.shape == (2, 2)
    assert table.tolist() == [[1, PAD], [2, PAD]]


def test_no_rows_is_all_padding():
    table = build_block_table([], padded_batch_size=3, max_blocks=2, pad_block_id=PAD)

    assert table.dtype == torch.int32
    assert table.shape == (3, 2)
    assert bool((table == PAD).all())


def test_requires_flag_probe():
    plain = SimpleNamespace(requires_block_table=True)
    adapter = SimpleNamespace(model=plain)
    assert model_requires_block_table(adapter)
    assert model_requires_block_table(plain)
    assert not model_requires_block_table(SimpleNamespace())
    assert not model_requires_block_table(SimpleNamespace(model=SimpleNamespace()))
    assert not model_requires_block_table(None)
