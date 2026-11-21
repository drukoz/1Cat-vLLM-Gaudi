# SPDX-License-Identifier: Apache-2.0
"""The QSA key ring is written as a flat table, never through a view.

On a Gaudi2 card (rental 2026-11-20) the first sparse decode step failed to compile:
``ring.view(-1, D).index_copy_(...)`` on the ``[slots, rows, D]`` ring became an
``hpu::strided_insert`` and the graph compiler stopped with "Empty tensor optional ... Strided
Params Has Value: 0". The state writes that do run on a card (GDN, the compressed key cache) are
``index_copy_`` on the tensor itself, so the ring the HPU layer binds is the flat
``[slots * rows, D]`` table. CPU torch accepts both forms; these tests pin the form.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.qwen4_exp.qsa import HpuQsaIndexer
from vllm_gaudi.models.qwen4_exp.qsa_side_caches import allocate_side_caches
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import ring_block_rows, ring_write

S, R, D, RATIO = 7, 8, 16, 4


def _inputs(n=5, seed=3):
    g = torch.Generator().manual_seed(seed)
    slot = torch.randint(0, S, (n,), generator=g)
    positions = torch.randint(3, 400, (n,), generator=g)
    raw_k = torch.randn(n, D, generator=g).to(torch.bfloat16)
    return slot, positions, raw_k


def test_flat_ring_helpers_equal_the_three_axis_form():
    ring3 = torch.randn(S, R, D, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    flat = ring3.reshape(-1, D).clone()
    slot, positions, raw_k = _inputs()
    assert torch.equal(ring_block_rows(flat, slot, positions, raw_k, RATIO, rows=R),
                       ring_block_rows(ring3, slot, positions, raw_k, RATIO))
    ring_write(ring3, slot, positions, raw_k)
    ring_write(flat, slot, positions, raw_k, rows=R)
    assert torch.equal(flat.view(S, R, D), ring3)


def test_a_flat_ring_needs_its_row_count():
    flat = torch.zeros(S * R, D, dtype=torch.bfloat16)
    slot, positions, raw_k = _inputs()
    with pytest.raises(ValueError):
        ring_write(flat, slot, positions, raw_k)
    with pytest.raises(ValueError):
        ring_block_rows(flat, slot, positions, raw_k, RATIO)


def test_the_flat_write_is_not_through_a_view(monkeypatch):
    seen = []
    real = torch.Tensor.index_copy_

    def index_copy_(self, *args, **kwargs):
        seen.append(self._is_view())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_copy_", index_copy_)
    flat = torch.zeros(S * R, D, dtype=torch.bfloat16)
    slot, positions, raw_k = _inputs()
    ring_write(flat, slot, positions, raw_k, rows=R)
    assert seen == [False]


def test_allocate_side_caches_can_give_the_flat_ring():
    _, ring3 = allocate_side_caches(64, S, RATIO, D, 3, "cpu")
    _, flat = allocate_side_caches(64, S, RATIO, D, 3, "cpu", flat_ring=True)
    assert ring3.shape == (S, R, D)
    assert flat.shape == (S * R, D) and not flat._is_view()


def test_the_indexer_keeps_a_flat_ring_and_shows_the_three_axis_view():
    indexer = SimpleNamespace(_key_ring_flat=None, _key_ring_rows=0)
    assert HpuQsaIndexer.key_ring.fget(indexer) is None

    flat = torch.zeros(S * R, D, dtype=torch.bfloat16)
    HpuQsaIndexer.bind_key_ring(indexer, flat, R)
    assert indexer._key_ring_flat is flat and indexer._key_ring_rows == R
    view = HpuQsaIndexer.key_ring.fget(indexer)
    assert view.shape == (S, R, D)
    flat[R * 2 + 5] = 1.0  # a write to the flat table is what a reader of the ring sees
    assert bool((view[2, 5] == 1.0).all())

    # the CPU tests assign a three-axis ring: the indexer works on its flat view, same storage
    ring3 = torch.zeros(S, R, D, dtype=torch.bfloat16)
    HpuQsaIndexer.key_ring.fset(indexer, ring3)
    assert indexer._key_ring_flat.shape == (S * R, D) and indexer._key_ring_rows == R
    indexer._key_ring_flat[R * 3 + 1] = 2.0
    assert bool((ring3[3, 1] == 2.0).all())

    HpuQsaIndexer.key_ring.fset(indexer, None)
    assert indexer._key_ring_flat is None and HpuQsaIndexer.key_ring.fget(indexer) is None
    with pytest.raises(ValueError):
        HpuQsaIndexer.bind_key_ring(indexer, torch.zeros(S * R + 1, D), R)
