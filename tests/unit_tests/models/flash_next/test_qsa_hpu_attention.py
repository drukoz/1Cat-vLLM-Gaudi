# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the HPU Flash-Next sparse QSA attention helpers (M2-T5c).

Pure-torch checks (fp32, seeded) of the three module-level helpers in
``vllm_gaudi/models/qwen4_exp/qsa_hpu_attention.py``: the prefill selection
bias layout, the decode slot mapping through a non-identity block table, and
the gathered masked decode SDPA against dense attention over the selected
positions. The HPU wrapper is checked by ``py_compile`` / CPU import only.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.envs as envs  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import (  # noqa: E402
    decode_selected_slots,
    prefill_selection_bias,
    rows_without_selection,
    sparse_decode_attention,
)


def test_prefill_bias_layout():
    torch.manual_seed(0)
    bs, n_ctx_cols, T = 2, 256, 8
    context_lens = torch.tensor([256, 128], dtype=torch.int64)
    # Each query row selects one context position (below ctx), one chunk
    # position (>= ctx, within the T-token chunk) and one -1 pad.
    selected = torch.full((bs, T, 3), -1, dtype=torch.int32)
    chunk_lo = context_lens.tolist()
    for b in range(bs):
        base = chunk_lo[b]
        for t in range(T):
            selected[b, t, 0] = (b * 50 + t)  # context position, < ctx
            selected[b, t, 1] = base + t  # chunk position, in [base, base + T)

    bias = prefill_selection_bias(selected, context_lens, n_ctx_cols, T, torch.float32)
    assert bias.shape == (bs, 1, T, n_ctx_cols + T)
    assert bias.dtype == torch.float32

    neg_inf = float("-inf")
    for b in range(bs):
        ctx = int(context_lens[b])
        for t in range(T):
            expected = set()
            for p in selected[b, t].tolist():
                if p < 0:
                    continue
                expected.add(p if p < ctx else n_ctx_cols + (p - ctx))
            row = bias[b, 0, t]
            zeros = set(torch.nonzero(row == 0).flatten().tolist())
            assert zeros == expected, (b, t, sorted(zeros), sorted(expected))
            others = row[~torch.isin(torch.arange(row.shape[0]), torch.tensor(sorted(expected)))]
            assert torch.all(others == neg_inf)


def test_prefill_bias_tiled_columns():
    # A long-context prefill splits the queries into tiles of height `tile`,
    # but every tile attends to the whole chunk, so its bias width must be the
    # full chunk (n_ctx_cols + seq_len), not n_ctx_cols + tile. The tile's
    # selected chunk positions reach the far end of the chunk (columns beyond
    # n_ctx_cols + tile); the old code sized the buffer by the tile height and
    # scattered out of range.
    torch.manual_seed(0)
    bs, n_ctx_cols, seq_len, tile = 2, 256, 8, 4
    context_lens = torch.tensor([256, 128], dtype=torch.int64)
    # Second tile (query rows [tile:seq_len]); select chunk positions that
    # span the whole chunk, i.e. columns up to n_ctx_cols + seq_len - 1.
    selected = torch.full((bs, tile, 2), -1, dtype=torch.int32)
    for b in range(bs):
        base = int(context_lens[b])
        for t in range(tile):
            selected[b, t, 0] = base + (seq_len - 1 - t)  # far end of the chunk
            selected[b, t, 1] = base + t  # near end of the chunk

    bias = prefill_selection_bias(selected, context_lens, n_ctx_cols, seq_len, torch.float32)
    # width is the full chunk, not the tile height
    assert bias.shape == (bs, 1, tile, n_ctx_cols + seq_len)
    assert bias.dtype == torch.float32

    neg_inf = float("-inf")
    for b in range(bs):
        ctx = int(context_lens[b])
        for t in range(tile):
            expected = set()
            for p in selected[b, t].tolist():
                if p < 0:
                    continue
                col = p if p < ctx else n_ctx_cols + (p - ctx)
                assert col < n_ctx_cols + seq_len  # lands inside the full-chunk buffer
                expected.add(col)
            row = bias[b, 0, t]
            zeros = set(torch.nonzero(row == 0).flatten().tolist())
            assert zeros == expected, (b, t, sorted(zeros), sorted(expected))
            others = row[~torch.isin(torch.arange(row.shape[0]), torch.tensor(sorted(expected)))]
            assert torch.all(others == neg_inf)


def test_rows_without_selection():
    selected = torch.tensor([[3, -1, -1], [-1, -1, -1], [-1, -1, 0]], dtype=torch.int32)
    assert rows_without_selection(selected).tolist() == [[False], [True], [False]]


def test_prefill_bias_row_without_selection_is_not_all_inf():
    """A padding row of a bucket selects nothing (position -1): its bias row must not be all -inf, or the
    softmax of that row is NaN. It gets an all-zero row; the rows that select something are unchanged."""
    bs, n_ctx_cols, T = 1, 8, 4
    context_lens = torch.tensor([8], dtype=torch.int64)
    selected = torch.full((bs, T, 3), -1, dtype=torch.int32)
    selected[0, 0, :2] = torch.tensor([2, 8])  # a context position and the first chunk position
    selected[0, 1, 0] = 9
    bias = prefill_selection_bias(selected, context_lens, n_ctx_cols, T, torch.float32)
    assert bias.shape == (bs, 1, T, n_ctx_cols + T)
    assert torch.nonzero(bias[0, 0, 0] == 0).flatten().tolist() == [2, 8]
    assert torch.nonzero(bias[0, 0, 1] == 0).flatten().tolist() == [9]
    for row in (2, 3):  # the rows with no selection
        assert bool((bias[0, 0, row] == 0).all())
    assert bool(torch.isfinite(torch.softmax(torch.randn(bs, 1, T, n_ctx_cols + T) + bias, dim=-1)).all())


def test_decode_selected_slots():
    block_table = torch.tensor([[5, 3, 0, 7]], dtype=torch.int32)
    selected = torch.tensor([[0, 5, 9, 14, -1]], dtype=torch.int32)
    slots = decode_selected_slots(selected, block_table, block_size=4)
    assert slots.dtype == torch.int64
    assert slots.tolist() == [[20, 13, 1, 30, 20]]


def _dense_over(q, k_ref, v_ref, positions, num_heads, num_kv_heads, head_dim, scale):
    """Reference dense attention over ``positions`` from the [20, KV, D] copy."""
    k = k_ref[positions].repeat_interleave(num_heads // num_kv_heads, dim=1)  # [N, H, D]
    v = v_ref[positions].repeat_interleave(num_heads // num_kv_heads, dim=1)
    qh = q.view(num_heads, head_dim)
    scores = torch.einsum("hd,nhd->hn", qh, k) * scale  # [H, N]
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.einsum("hn,nhd->hd", probs, v)
    return out.reshape(num_heads * head_dim)


def test_sparse_decode_matches_masked_dense():
    torch.manual_seed(1234)
    S, KV, H, D, block_size = 64, 2, 4, 8, 8
    scale = D**-0.5
    n_tokens = 20

    key_cache = torch.randn(S, KV, D)
    value_cache = torch.randn(S, KV, D)
    k_ref = torch.randn(n_tokens, KV, D)
    v_ref = torch.randn(n_tokens, KV, D)

    block_table = torch.randperm(S // block_size, dtype=torch.int32).unsqueeze(0)  # [1, 8] permutation
    pos = torch.arange(n_tokens)
    write_slots = (block_table[0][pos // block_size].long() * block_size + (pos % block_size))
    key_cache[write_slots] = k_ref
    value_cache[write_slots] = v_ref

    q = torch.randn(1, H * D)

    # Case 1: positions {0..3, 8..11, 17,18,19} padded to W=16 with -1.
    sel_positions = [0, 1, 2, 3, 8, 9, 10, 11, 17, 18, 19]
    selected = torch.full((1, 16), -1, dtype=torch.int32)
    for i, p in enumerate(sel_positions):
        selected[0, i] = p
    slots = decode_selected_slots(selected, block_table, block_size)
    valid = selected >= 0
    out = sparse_decode_attention(q, key_cache, value_cache, slots, valid, H, KV, D, scale)
    expected = _dense_over(q, k_ref, v_ref, sel_positions, H, KV, D, scale)
    assert out.shape == (1, H * D)
    torch.testing.assert_close(out[0], expected, atol=1e-5, rtol=1e-5)

    # Case 2: a fully-valid window with one column masked out -> dense over the
    # remaining positions (confirms the -inf mask removes that gather).
    full_positions = list(range(16))
    selected_full = torch.tensor([full_positions], dtype=torch.int32)
    slots_full = decode_selected_slots(selected_full, block_table, block_size)
    valid_mask = torch.ones(1, 16, dtype=torch.bool)
    valid_mask[0, 5] = False
    out_masked = sparse_decode_attention(q, key_cache, value_cache, slots_full, valid_mask, H, KV, D, scale)
    expected_masked = _dense_over(q, k_ref, v_ref, [p for p in full_positions if p != 5], H, KV, D, scale)
    torch.testing.assert_close(out_masked[0], expected_masked, atol=1e-5, rtol=1e-5)


def test_env_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE", raising=False)
    assert envs.VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE == 2048
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE", "512")
    assert envs.VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE == 512
