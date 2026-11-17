# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load/store slot split for GDN state writes under prefix caching.

Under prefix caching the runner gives each GDN layer two slot tensors: the
load slot (the block of the last computed token, read side) and the store
slot (the block of the last scheduled token, write side).  These CPU tests
check that every state write lands in the store slot while the incoming
state is read from the load slot, and that a load index resolving to the
null block (id 0) reads a zero initial state.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn, hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_recurrent_gated_delta_rule

DIM = 4
WIDTH = 4
STATE_LEN = WIDTH - 1
NUM_SLOTS = 6


def _conv_inputs(seed: int = 0, seq_lens: tuple[int, ...] = (5, )):
    """Small deterministic prefill-conv inputs: x (dim, total), pool (6, 3, 4)."""
    gen = torch.Generator().manual_seed(seed)
    total = sum(seq_lens)
    x = torch.randn(DIM, total, generator=gen)
    weight = torch.randn(DIM, WIDTH, generator=gen)
    bias = torch.randn(DIM, generator=gen)
    pool = torch.randn(NUM_SLOTS, STATE_LEN, DIM, generator=gen)
    qsl = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64)
    return x, weight, bias, pool, qsl


# ---------------------------------------------------------------------------
# a. Prefill conv: read from initial_state_idx (load), write to cache_indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", [(6, ), (5, 4)], ids=["batched", "variable-length"])
def test_prefill_conv_reads_load_slot_writes_store_slot(seq_lens):
    x, weight, bias, pool, qsl = _conv_inputs(seed=3, seq_lens=seq_lens)
    num_seqs = len(seq_lens)
    store = torch.tensor([3, 4][:num_seqs], dtype=torch.int64)
    load = torch.tensor([2, 1][:num_seqs], dtype=torch.int64)
    has_initial = torch.ones(num_seqs, dtype=torch.bool)

    pool_split = pool.clone()
    out_split = hpu_causal_conv1d_fn(
        x,
        weight,
        bias,
        pool_split,
        qsl,
        cache_indices=store,
        has_initial_state=has_initial,
        activation="silu",
        initial_state_idx=load,
    )
    # The same run with cache_indices == initial_state_idx reads the same
    # state from the load slot, so the outputs must match exactly.
    pool_plain = pool.clone()
    out_plain = hpu_causal_conv1d_fn(
        x,
        weight,
        bias,
        pool_plain,
        qsl,
        cache_indices=load,
        has_initial_state=has_initial,
        activation="silu",
    )
    torch.testing.assert_close(out_split, out_plain, rtol=0, atol=0)
    # The load slots are untouched and the store slots hold the new state.
    for idx in range(num_seqs):
        torch.testing.assert_close(pool_split[load[idx]], pool[load[idx]], rtol=0, atol=0)
        torch.testing.assert_close(pool_split[store[idx]], pool_plain[load[idx]], rtol=0, atol=0)


@pytest.mark.parametrize("seq_lens", [(6, ), (5, 4)], ids=["batched", "variable-length"])
def test_prefill_conv_null_block_reads_zeros(seq_lens):
    x, weight, bias, pool, qsl = _conv_inputs(seed=5, seq_lens=seq_lens)
    num_seqs = len(seq_lens)
    store = torch.tensor([3, 4][:num_seqs], dtype=torch.int64)
    load_null = torch.tensor([0, 1][:num_seqs], dtype=torch.int64)
    load_real = torch.tensor([2, 1][:num_seqs], dtype=torch.int64)
    has_initial = torch.ones(num_seqs, dtype=torch.bool)

    pool_null = pool.clone()
    out_null = hpu_causal_conv1d_fn(
        x,
        weight,
        bias,
        pool_null,
        qsl,
        cache_indices=store,
        has_initial_state=has_initial,
        activation="silu",
        initial_state_idx=load_null,
    )
    # has_initial_state = 0 for the null-block sequence must produce the
    # same output: the null block 0 provides a zero initial state.
    pool_zero = pool.clone()
    out_zero = hpu_causal_conv1d_fn(
        x,
        weight,
        bias,
        pool_zero,
        qsl,
        cache_indices=store,
        has_initial_state=torch.tensor([False] + [True] * (num_seqs - 1)),
        activation="silu",
        initial_state_idx=load_real,
    )
    torch.testing.assert_close(out_null, out_zero, rtol=0, atol=0)
    # The null block itself is never written.
    torch.testing.assert_close(pool_null[0], pool[0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# b. Decode conv: store_cache_indices writes the store slot
# ---------------------------------------------------------------------------


def test_decode_conv_update_writes_store_slot():
    torch.manual_seed(11)
    x = torch.randn(1, DIM)
    weight = torch.randn(DIM, WIDTH)
    bias = torch.randn(DIM)
    pool = torch.randn(NUM_SLOTS, STATE_LEN, DIM)
    qsl = torch.arange(2, dtype=torch.int32)

    plain = pool.clone()
    out_plain = hpu_causal_conv1d_update(
        x,
        plain,
        weight,
        bias,
        activation="silu",
        conv_state_indices=torch.tensor([2], dtype=torch.int32),
        query_start_loc=qsl,
    )
    split = pool.clone()
    out_split = hpu_causal_conv1d_update(
        x,
        split,
        weight,
        bias,
        activation="silu",
        conv_state_indices=torch.tensor([2], dtype=torch.int32),
        store_cache_indices=torch.tensor([3], dtype=torch.int32),
        query_start_loc=qsl,
    )
    torch.testing.assert_close(out_split, out_plain, rtol=0, atol=0)
    # Slot 3 (store) is updated, slot 2 (load) keeps its incoming state.
    torch.testing.assert_close(split[3], plain[2], rtol=0, atol=0)
    torch.testing.assert_close(split[2], pool[2], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# c. Decode SSM: ssm_store_indices writes the store slot
# ---------------------------------------------------------------------------


def _gdn_decode_inputs(num_slots: int):
    """Single-token decode inputs built like the smallest test in
    tests/unit_tests/ops/test_hpu_gdn_pytorch.py."""
    device = "cpu"
    gen = torch.Generator(device=device).manual_seed(42)
    H, K, HV, V = 4, 16, 4, 16
    q = torch.randn(1, 1, H, K, device=device, generator=gen)
    k = torch.randn(1, 1, H, K, device=device, generator=gen)
    v = torch.randn(1, 1, HV, V, device=device, generator=gen)
    g = -torch.abs(torch.randn(1, 1, HV, device=device, generator=gen)) * 0.1
    beta = torch.sigmoid(torch.randn(1, 1, HV, device=device, generator=gen))
    pool = torch.randn(num_slots, HV, V, K, device=device, generator=gen)
    return q, k, v, g, beta, pool


def test_decode_ssm_writes_store_slot():
    q, k, v, g, beta, pool = _gdn_decode_inputs(NUM_SLOTS)
    cu_seqlens = torch.arange(2, dtype=torch.long)

    plain = pool.clone()
    out_plain, _ = hpu_fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=plain,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=torch.tensor([2], dtype=torch.int64),
    )
    split = pool.clone()
    out_split, _ = hpu_fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=split,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=torch.tensor([2], dtype=torch.int64),
        ssm_store_indices=torch.tensor([3], dtype=torch.int64),
    )
    torch.testing.assert_close(out_split, out_plain, rtol=0, atol=0)
    # final_state[3] is updated, final_state[2] keeps its incoming state.
    torch.testing.assert_close(split[3], plain[2], rtol=0, atol=0)
    torch.testing.assert_close(split[2], pool[2], rtol=0, atol=0)
