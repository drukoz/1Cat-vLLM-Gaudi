# SPDX-License-Identifier: Apache-2.0
"""Greedy verification from the arg max of every row (VLLM_HPU_VOCAB_PARALLEL_LOGITS): same tokens as from logits."""

from __future__ import annotations

import torch

from vllm_gaudi.v1.sample.hpu_greedy_verify import greedy_rejection_sample_from_tokens
from vllm_gaudi.v1.sample.hpu_rejection_sampler import (dflash2_greedy_rejection_sample_packed,
                                                        rejection_sample_pytorch)


def _step(requests: int, k: int, vocab: int, seed: int, accept_every: int):
    """A verification step of ``requests`` x (1 + k) rows: logits, and drafts that match the target's arg max on
    a prefix of every request (request ``r`` gets ``r % (k + 1)`` matching drafts, then a mismatch)."""
    gen = torch.Generator().manual_seed(seed)
    rows = requests * (1 + k)
    logits = torch.randn(rows, vocab, generator=gen).to(torch.bfloat16)
    argmax = logits.float().argmax(-1).to(torch.int32)
    drafts = []
    target_indices = []
    bonus_indices = []
    for r in range(requests):
        base = r * (1 + k)
        matching = (r * accept_every) % (k + 1)
        for j in range(k):
            token = int(argmax[base + j])
            drafts.append(token if j < matching else (token + 1) % vocab)
            target_indices.append(base + j)
        bonus_indices.append(base + k)
    return (logits, argmax, torch.tensor(drafts, dtype=torch.int32),
            torch.tensor(target_indices, dtype=torch.int64), torch.tensor(bonus_indices, dtype=torch.int64),
            torch.arange(1, requests + 1, dtype=torch.int32) * k)


def test_from_tokens_equals_the_packed_logits_path():
    for seed, (requests, k) in enumerate([(1, 1), (1, 3), (4, 3), (32, 3), (7, 5)]):
        logits, argmax, drafts, target, bonus, cu = _step(requests, k, 997, seed, accept_every=1)
        from_logits = dflash2_greedy_rejection_sample_packed(logits, drafts, target, bonus, cu)
        from_tokens = greedy_rejection_sample_from_tokens(argmax, drafts, target, bonus, cu)
        torch.testing.assert_close(from_tokens, from_logits, rtol=0, atol=0)
        # and the HPU rejection sampler's own greedy rule (patched into upstream's rejection_sample)
        reference = rejection_sample_pytorch(drafts, argmax.index_select(0, target),
                                             argmax.index_select(0, bonus).view(-1, 1), [k] * requests, cu)
        torch.testing.assert_close(from_tokens, reference, rtol=0, atol=0)


def test_from_tokens_accepts_a_full_prefix_and_appends_the_bonus():
    logits, argmax, drafts, target, bonus, cu = _step(2, 3, 50, 7, accept_every=3)  # request 1: all 3 accepted
    out = greedy_rejection_sample_from_tokens(argmax, drafts, target, bonus, cu)
    assert out.shape == (2, 4)
    assert out[1].tolist() == argmax[4:8].tolist()  # three accepted drafts and the bonus row's token
    assert out[0, 0].item() == argmax[0].item() and out[0, 1:].lt(0).all()  # first draft rejected: one token
