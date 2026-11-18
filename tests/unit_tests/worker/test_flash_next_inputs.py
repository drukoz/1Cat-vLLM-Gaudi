# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Flash-Next PLE forward-kwargs helper."""
from types import SimpleNamespace

import numpy as np
import torch

from vllm_gaudi.v1.worker.flash_next_inputs import (_to_numpy, advance_ple_context, build_ple_inputs, dummy_ple_inputs,
                                                    model_requires_raw_input_tokens)

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


def _build_ple_inputs_loop(token_ids_cpu, num_computed_tokens_cpu, batch_indices, padded_batch_size, num_tokens,
                           eos_token_id, ngram_size):
    """Verbatim copy of the original per-row loop builder, used as the oracle."""
    context_len = max(int(ngram_size) - 1, 0)
    query_start_loc = torch.arange(padded_batch_size + 1, dtype=torch.int32) * int(num_tokens)
    ngram_context = torch.full((padded_batch_size, context_len), int(eos_token_id), dtype=torch.int32)
    token_ids_np = _to_numpy(token_ids_cpu)
    computed_np = _to_numpy(num_computed_tokens_cpu)
    for row, b in enumerate(batch_indices):
        if row >= padded_batch_size:
            break
        start = int(computed_np[b])
        row_ids = token_ids_np[b]
        for k in range(context_len):
            pos = start - context_len + k
            if pos >= 0:
                ngram_context[row, k] = int(row_ids[pos])
    return query_start_loc, ngram_context


def test_build_ple_inputs_matches_loop():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    num_tokens = 64
    for _ in range(200):
        ngram_size = int(rng.choice([1, 2, 3, 9]))
        padded_batch_size = int(rng.integers(1, 9))
        n_indices = int(rng.integers(0, padded_batch_size + 2 + 1))
        batch_indices = [int(v) for v in rng.permutation(10)[:n_indices]]
        token_ids = rng.integers(0, 1000, size=(10, 64)).astype(np.int64)
        computed = rng.integers(0, 41, size=(10,)).astype(np.int32)

        exp_q, exp_ctx = _build_ple_inputs_loop(token_ids, computed, batch_indices, padded_batch_size, num_tokens, EOS,
                                                ngram_size)
        act_q, act_ctx = build_ple_inputs(token_ids, computed, batch_indices, padded_batch_size, num_tokens, EOS,
                                          ngram_size)

        assert act_q.dtype == torch.int32 and act_ctx.dtype == torch.int32
        assert torch.equal(exp_q, act_q)
        assert torch.equal(exp_ctx, act_ctx)


def test_advance_matches_rebuild():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_reqs = 6
    width = 64
    for ngram_size in (1, 2, 3, 9):
        for T in (1, 2, 4):
            histories = rng.integers(1, 1000, size=(n_reqs, width)).astype(np.int32)
            num_computed = np.zeros(n_reqs, dtype=np.int32)
            indices = list(range(n_reqs))
            ctx = build_ple_inputs(histories, num_computed, indices, n_reqs, T, EOS, ngram_size)[1]
            for _ in range(5):
                num_committed = rng.integers(1, T + 1, size=(n_reqs, )).astype(np.int32)
                num_committed[n_reqs - 1] = 0  # one request is always "not run".
                input_rows = np.stack(
                    [histories[r, num_computed[r]:num_computed[r] + T] for r in range(n_reqs)]).astype(np.int32)

                new_ctx = advance_ple_context(ctx,
                                              torch.from_numpy(input_rows),
                                              torch.from_numpy(num_committed.copy()))

                num_computed = num_computed + num_committed
                exp_ctx = build_ple_inputs(histories, num_computed, indices, n_reqs, T, EOS, ngram_size)[1]

                assert new_ctx.dtype == torch.int32
                assert new_ctx.shape == (n_reqs, max(ngram_size - 1, 0))
                assert torch.equal(new_ctx, exp_ctx)
                ctx = new_ctx


def test_advance_left_fill():
    ngram_size = 9
    context_len = ngram_size - 1
    T = 3
    row = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    histories = np.zeros((1, 16), dtype=np.int32)
    histories[0, :len(row)] = row

    num_computed = np.array([2], dtype=np.int32)
    ctx = build_ple_inputs(histories, num_computed, [0], 1, T, EOS, ngram_size)[1]
    assert ctx[0].tolist() == [EOS] * (context_len - 2) + [100, 101]

    input_rows = np.array([row[2:2 + T]], dtype=np.int32)
    num_committed = np.array([1], dtype=np.int32)
    new_ctx = advance_ple_context(ctx, torch.from_numpy(input_rows), torch.from_numpy(num_committed))

    exp_ctx = build_ple_inputs(histories, np.array([3], dtype=np.int32), [0], 1, T, EOS, ngram_size)[1]
    assert torch.equal(new_ctx, exp_ctx)
    assert new_ctx[0].tolist() == [EOS] * (context_len - 3) + [100, 101, 102]


def test_advance_no_host_sync():
    compiled = torch.compile(advance_ple_context, fullgraph=True, backend="eager")
    context_len = 8
    T = 4
    n_reqs = 3
    ctx = torch.full((n_reqs, context_len), 7, dtype=torch.int32)
    input_rows = torch.arange(n_reqs * T, dtype=torch.int32).reshape(n_reqs, T)
    num_committed = torch.tensor([1, 2, 4], dtype=torch.int32)

    out = compiled(ctx, input_rows, num_committed)

    assert out.shape == (n_reqs, context_len)
    assert out.dtype == torch.int32
