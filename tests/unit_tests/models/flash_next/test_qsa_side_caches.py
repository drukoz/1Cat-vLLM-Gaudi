# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Flash-Next sparse QSA side-cache sizing helpers (M2-T5b).

Covers the six pure functions in ``vllm_gaudi/models/qwen4_exp/qsa_side_caches.py``:
tiny numbers (compress ratio 4, index head dim 16), the real Flash-Next
numbers (index head dim 128, 1 KV head, head 256: budget fraction 0.94118,
ring 4 rows without speculation, 8 rows with 3 speculative tokens), the
non-divisible ``ValueError``, and ``allocate_side_caches`` returning zero
bf16 tensors of the right shapes on CPU.
"""

import math

import pytest
import torch

from vllm_gaudi.models.qwen4_exp.qsa_side_caches import (
    allocate_side_caches,
    attention_budget_fraction,
    compressed_cache_shape,
    ring_bytes,
    ring_rows,
    ring_shape,
)


def test_ring_rows_tiny_and_real():
    assert ring_rows(4, 0) == 4
    assert ring_rows(4, 3) == 8
    # the formula, verified independently for a few (ratio, spec) pairs
    for ratio in (2, 4, 8):
        for spec in (0, 1, 3, 5, 8):
            assert ring_rows(ratio, spec) == ratio * math.ceil((ratio + spec) / ratio)


def test_compressed_cache_shape_tiny_and_real():
    assert compressed_cache_shape(64, 4, 16) == (16, 16)
    # real numbers: (num_blocks + 1) * block_size slots -> rows // 4
    assert compressed_cache_shape(1024 * 16, 4, 128) == (4096, 128)


def test_compressed_cache_shape_not_divisible_raises():
    with pytest.raises(ValueError):
        compressed_cache_shape(66, 4, 16)


def test_ring_shape_tiny():
    assert ring_shape(12, 4, 16) == (12, 4, 16)
    assert ring_shape(12, ring_rows(4, 3), 16) == (12, 8, 16)


def test_attention_budget_fraction_real_numbers():
    # 1 KV head, head 256, bf16 KV, ratio 4, index dim 128: 1024 / 1088
    frac = attention_budget_fraction(1, 256, 2, 4, 128)
    assert frac == pytest.approx(0.94118, abs=1e-5)


def test_attention_budget_fraction_tiny():
    # kv = 2 * 2 * 16 * 2 = 128; comp = 16 * 2 / 4 = 8; 128 / 136
    frac = attention_budget_fraction(2, 16, 2, 4, 16)
    assert frac == pytest.approx(128 / 136, abs=1e-9)


def test_ring_bytes_tiny_and_real():
    assert ring_bytes(2, 12, 4, 16) == 2 * 12 * 4 * 16 * 2
    # 12 QSA layers, 1024 GDN rows, R=4, dim 128, bf16
    assert ring_bytes(12, 1024, 4, 128) == 12 * 1024 * 4 * 128 * 2


def test_allocate_side_caches_cpu():
    comp, ring = allocate_side_caches(
        key_cache_rows=256,
        gdn_state_rows=40,
        compress_ratio=4,
        index_head_dim=16,
        num_speculative_tokens=0,
        device="cpu",
    )
    assert comp.shape == (64, 16)
    assert ring.shape == (40, 4, 16)
    assert comp.dtype == torch.bfloat16 and ring.dtype == torch.bfloat16
    assert torch.count_nonzero(comp) == 0 and torch.count_nonzero(ring) == 0


def test_allocate_side_caches_speculative_ring():
    comp, ring = allocate_side_caches(
        key_cache_rows=512,
        gdn_state_rows=8,
        compress_ratio=4,
        index_head_dim=128,
        num_speculative_tokens=3,
        device="cpu",
    )
    assert comp.shape == (128, 128)
    assert ring.shape == (8, 8, 128)
