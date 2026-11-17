# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Flash-Next PLE forward-kwargs helper."""
from types import SimpleNamespace

import numpy as np
import torch

from vllm_gaudi.v1.worker.flash_next_inputs import (build_ple_inputs, dummy_ple_inputs, model_requires_raw_input_tokens)

# Values from tests/unit_tests/models/flash_next/tiny_config.json.
EOS = 248044
NGRAM = 3


def _token_ids(rows: list[list[int]]) -> np.ndarray:
    width = max(len(row) for row in rows)
    array = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        array[i, :len(row)] = row
    return array


def test_fresh_prefill_rows_are_eos():
    token_ids = _token_ids([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]])
    computed = np.zeros(3, dtype=np.int32)

    query_start_loc, ngram_context = build_ple_inputs(
        token_ids,
        computed,
        batch_indices=[0, 1, 2],
        padded_batch_size=3,
        num_tokens=5,
        eos_token_id=EOS,
        ngram_size=NGRAM,
    )

    assert query_start_loc.dtype == torch.int32
    assert ngram_context.dtype == torch.int32
    assert query_start_loc.tolist() == [0, 5, 10, 15]
    assert ngram_context.shape == (3, 2)
    for row in range(3):
        assert ngram_context[row].tolist() == [EOS, EOS]


def test_chunk_continuation_takes_previous_tokens():
    row = [100 + i for i in range(12)]
    token_ids = _token_ids([row])
    computed = np.array([7], dtype=np.int32)

    _, ngram_context = build_ple_inputs(
        token_ids,
        computed,
        batch_indices=[0],
        padded_batch_size=1,
        num_tokens=5,
        eos_token_id=EOS,
        ngram_size=NGRAM,
    )

    # The two tokens immediately before position 7: indices 5 and 6.
    assert ngram_context[0].tolist() == [row[5], row[6]]


def test_one_computed_token_is_half_eos():
    row = [500, 501, 502]
    token_ids = _token_ids([row])
    computed = np.array([1], dtype=np.int32)

    _, ngram_context = build_ple_inputs(
        token_ids,
        computed,
        batch_indices=[0],
        padded_batch_size=1,
        num_tokens=1,
        eos_token_id=EOS,
        ngram_size=NGRAM,
    )

    assert ngram_context[0].tolist() == [EOS, row[0]]


def test_decode_rows_and_padding():
    # Two decode rows with different histories.
    token_ids = _token_ids([[10, 11, 12, 13], [20, 21, 22, 23]])
    computed = np.array([4, 1], dtype=np.int32)

    query_start_loc, ngram_context = build_ple_inputs(
        token_ids,
        computed,
        batch_indices=[0, 1],
        padded_batch_size=4,
        num_tokens=1,
        eos_token_id=EOS,
        ngram_size=NGRAM,
    )

    assert query_start_loc.dtype == torch.int32
    assert ngram_context.dtype == torch.int32
    assert query_start_loc.shape == (5, )
    assert ngram_context.shape == (4, 2)
    assert query_start_loc.tolist() == [0, 1, 2, 3, 4]
    assert ngram_context[0].tolist() == [token_ids[0][2], token_ids[0][3]]
    assert ngram_context[1].tolist() == [EOS, token_ids[1][0]]
    assert ngram_context[2].tolist() == [EOS, EOS]
    assert ngram_context[3].tolist() == [EOS, EOS]


def test_ngram_size_4_gives_three_context_tokens():
    row = [70, 71, 72, 73, 74]
    token_ids = _token_ids([row])
    computed = np.array([4], dtype=np.int32)

    _, ngram_context = build_ple_inputs(
        token_ids,
        computed,
        batch_indices=[0],
        padded_batch_size=1,
        num_tokens=1,
        eos_token_id=EOS,
        ngram_size=4,
    )

    assert ngram_context.shape == (1, 3)
    assert ngram_context[0].tolist() == [row[1], row[2], row[3]]


def test_dummy_inputs():
    query_start_loc, ngram_context = dummy_ple_inputs(
        padded_batch_size=3,
        num_tokens=5,
        eos_token_id=EOS,
        ngram_size=NGRAM,
    )

    assert query_start_loc.dtype == torch.int32
    assert ngram_context.dtype == torch.int32
    assert query_start_loc.tolist() == [0, 5, 10, 15]
    assert ngram_context.shape == (3, 2)
    assert bool((ngram_context == EOS).all())

    # The flag probe: True on a model that declares it, on the adapter
    # wrapper (flag on the inner model) and on the plain model; False for
    # models that do not declare it.
    plain = SimpleNamespace(requires_raw_input_tokens=True)
    adapter = SimpleNamespace(model=plain)
    assert model_requires_raw_input_tokens(adapter)
    assert model_requires_raw_input_tokens(plain)
    assert not model_requires_raw_input_tokens(SimpleNamespace())
    assert not model_requires_raw_input_tokens(SimpleNamespace(model=SimpleNamespace()))
