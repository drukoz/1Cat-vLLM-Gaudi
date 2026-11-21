# SPDX-License-Identifier: Apache-2.0
"""Sampling over candidates (``vllm_gaudi/v1/sample/candidate_sampling.py``) against vLLM's sampler.

vLLM's sampled path sorts the whole vocabulary (``apply_top_k_top_p_pytorch``), masks, scatters back, takes a
softmax and draws with exponential noise over the vocabulary (``random_sample``). The candidate path takes the
``C`` largest logits per row first and does the rest on ``[rows, C]``. These tests hold it to vLLM's result:

* the processed distribution is vLLM's: same kept tokens, same probabilities (rows with their own temperature,
  top-k and top-p; logits with ties, as bf16 logits have);
* the draw follows that distribution, and a row's draw depends only on its own noise (a seeded request);
* a greedy row of a mixed batch gets the full arg max, as in vLLM;
* what lies outside its conditions is refused by ``candidate_refusal`` (the request keeps the general path);
* a tie group at the k-th value that reaches past ``C`` is cut at ``C`` (documented difference).

Pure functions on the CPU: no engine, no card.
"""

import math

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_gaudi.v1.sample.candidate_sampling import (CANDIDATES, candidate_parameters, candidate_probs,
                                                     candidate_refusal, sample_candidates, take_candidates)

VOCAB = 5000


def vllm_probs(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor):
    """vLLM's distribution of an all-random batch: Sampler.sample up to the softmax of forward_native."""
    scaled = logits.to(torch.float32).div(temperature.unsqueeze(dim=1))
    masked = apply_top_k_top_p_pytorch(scaled.clone(), top_k.clone(), top_p.clone(), allow_cpu_sync=False)
    return masked.softmax(dim=-1, dtype=torch.float32)


def on_vocabulary(probs: torch.Tensor, ids: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.zeros(probs.shape[0], vocab).scatter_add_(1, ids, probs)


def rows_of_parameters():
    temperature = torch.tensor([1.0, 1.0, 0.7, 1.3, 1.0, 0.2, 1.0, 2.0])
    top_k = torch.tensor([20, 1, 5, 20, CANDIDATES, 40, 20, 3], dtype=torch.int32)
    top_p = torch.tensor([0.95, 0.95, 0.5, 1.0, 0.9, 0.99, 0.05, 0.8])
    return temperature, top_k, top_p


@pytest.mark.parametrize("ties", [False, True])
def test_the_distribution_is_vllms(ties):
    torch.manual_seed(3 if ties else 2)
    temperature, top_k, top_p = rows_of_parameters()
    rows = temperature.shape[0]
    logits = 3.0 * torch.randn(rows, VOCAB)
    if ties:  # bf16 logits: many equal values near the top
        logits = (logits * 0.5).to(torch.bfloat16).to(torch.float32)
    want = vllm_probs(logits, temperature, top_k, top_p)
    values, ids = take_candidates(logits)
    assert values.shape == (rows, CANDIDATES) and ids.dtype == torch.int64
    probs = candidate_probs(values, temperature, (top_k.to(torch.int64) - 1).view(-1, 1), top_p)
    got = on_vocabulary(probs, ids, VOCAB)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(rows), atol=1e-5)
    if not ties:
        assert torch.equal(got > 0, want > 0), "the kept tokens differ"
        assert torch.allclose(got, want, atol=2e-6, rtol=1e-5)
        return
    # With equal logits the top-p cut can fall inside a group of tied tokens. Which of them stay is then decided
    # by the order of equal values (in vLLM by its sort, here by top-k): the same number of tokens with the same
    # logits and probabilities stays, and a token kept by only one of the two is tied with one kept by the other.
    differing_rows = 0
    for row in range(rows):
        mine, theirs = got[row] > 0, want[row] > 0
        assert int(mine.sum()) == int(theirs.sum())
        assert torch.equal(logits[row][mine].sort().values, logits[row][theirs].sort().values)
        assert torch.allclose(got[row][mine].sort().values, want[row][theirs].sort().values, atol=2e-6, rtol=1e-5)
        both = mine & theirs
        assert torch.allclose(got[row][both], want[row][both], atol=2e-6, rtol=1e-5)
        differing_rows += int(not torch.equal(mine, theirs))
    assert differing_rows > 0, "the fixture no longer cuts a tie group: pick another seed"


def test_top_p_keeps_the_largest_token_even_when_it_alone_is_below_one_minus_p():
    values = torch.tensor([[2.0, 1.9, 1.8, 0.0] + [-50.0] * (CANDIDATES - 4)])
    probs = candidate_probs(values, torch.tensor([1.0]), torch.tensor([[CANDIDATES - 1]]), torch.tensor([1e-6]))
    assert probs[0, 0] == 1.0 and float(probs[0, 1:].sum()) == 0.0


def test_the_draw_follows_the_distribution_and_only_its_own_noise():
    torch.manual_seed(5)
    logits = torch.randn(1, VOCAB) * 2.0
    temperature, top_k, top_p = torch.tensor([1.0]), torch.tensor([20], dtype=torch.int32), torch.tensor([0.95])
    want = vllm_probs(logits, temperature, top_k, top_p)[0]
    draws = 40000
    values, ids = take_candidates(logits.expand(draws, -1))
    params, k_index = candidate_parameters([1.0] * draws, [20] * draws, [0.95] * draws, draws,
                                           generator=torch.Generator().manual_seed(11))
    tokens = sample_candidates(values, ids, params, k_index)
    assert tokens.shape == (draws, 1) and tokens.dtype == torch.int64
    counts = torch.bincount(tokens.view(-1), minlength=VOCAB).to(torch.float32)
    assert float(counts[want == 0].sum()) == 0.0, "a token outside the kept set was drawn"
    kept = want > 0
    sigma = torch.sqrt(draws * want[kept] * (1 - want[kept]))
    assert torch.all((counts[kept] - draws * want[kept]).abs() <= 5 * sigma + 1), "frequencies off the distribution"

    # Two rows with the same logits and the same noise draw the same token, whatever their neighbours do.
    generators = {1: torch.Generator().manual_seed(99), 3: torch.Generator().manual_seed(99)}
    values, ids = take_candidates(logits.expand(4, -1))
    params, k_index = candidate_parameters([1.0] * 4, [20] * 4, [0.95] * 4, 4,
                                           generator=torch.Generator().manual_seed(1), generators=generators)
    tokens = sample_candidates(values, ids, params, k_index).view(-1)
    assert tokens[1] == tokens[3]
    again = {1: torch.Generator().manual_seed(99)}
    params, k_index = candidate_parameters([1.0, 1.0], [20, 20], [0.95, 0.95], 2,
                                           generator=torch.Generator().manual_seed(77), generators=again)
    assert sample_candidates(values[:2], ids[:2], params, k_index).view(-1)[1] == tokens[1]


def test_greedy_rows_of_a_mixed_batch_get_the_full_arg_max_and_padding_rows_are_harmless():
    torch.manual_seed(7)
    logits = torch.randn(3, VOCAB).to(torch.bfloat16).to(torch.float32)
    logits[0, 4000] = logits[0, 17] = float(logits[0].max()) + 1.0  # a tie at the maximum of the greedy row
    values, ids = take_candidates(logits)
    params, k_index = candidate_parameters([0.0, 1.0], [0, 20], [1.0, 0.95], 3,
                                           generator=torch.Generator().manual_seed(4))
    assert params.shape == (3, 3 + CANDIDATES) and params.dtype == torch.float32
    assert k_index.shape == (3, 1) and k_index.dtype == torch.int64
    assert params[0, 2] == 1.0 and params[1, 2] == 0.0 and params[2, 2] == 0.0  # the greedy flags
    tokens = sample_candidates(values, ids, params, k_index, greedy_ids=logits.argmax(dim=-1, keepdim=True))
    assert int(tokens[0]) == int(logits[0].argmax()) == 17  # the first maximum, as vLLM's greedy sample
    assert int(tokens[1]) in set(logits[1].topk(20).indices.tolist())
    assert 0 <= int(tokens[2]) < VOCAB


def test_requests_outside_the_conditions_keep_the_general_path():
    ok = dict(greedy=False, top_k=20, logprobs=None, logprob_token_ids=None, logit_bias=None, allowed_token_ids=None,
              bad_words_token_ids=None, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
              min_p=0.0, thinking_token_budget=None, extra_args=None, trace_decode_token_ids=None, stop_ids=0)
    assert candidate_refusal(**ok) is None
    assert candidate_refusal(**dict(ok, greedy=True, top_k=0)) is None  # a greedy row needs no top-k
    for change in (dict(top_k=0), dict(top_k=-1), dict(top_k=CANDIDATES + 1), dict(logprobs=1), dict(min_p=0.1),
                   dict(logit_bias={5: 1.0}), dict(repetition_penalty=1.1), dict(presence_penalty=0.5),
                   dict(frequency_penalty=0.5), dict(allowed_token_ids=[1]), dict(bad_words_token_ids=[[1]]),
                   dict(extra_args={"x": 1}), dict(thinking_token_budget=10), dict(stop_ids=9),
                   dict(logprob_token_ids=[3]), dict(trace_decode_token_ids=[1])):
        assert isinstance(candidate_refusal(**dict(ok, **change)), str), change
    assert candidate_refusal(**dict(ok, top_k=CANDIDATES)) is None


def test_a_tie_group_past_the_candidates_is_cut_at_the_candidates():
    """vLLM keeps every token tied at the k-th value; with more than C of them the candidate path keeps C."""
    logits = torch.full((1, VOCAB), -5.0)
    logits[0, :200] = 1.0  # 200 tokens tied at the top: top_k 20 keeps all 200 in vLLM
    want = vllm_probs(logits, torch.tensor([1.0]), torch.tensor([20], dtype=torch.int32), torch.tensor([1.0]))
    assert int((want > 0).sum()) == 200
    values, ids = take_candidates(logits)
    probs = candidate_probs(values, torch.tensor([1.0]), torch.tensor([[19]]), torch.tensor([1.0]))
    assert int((probs > 0).sum()) == CANDIDATES
    assert math.isclose(float(probs[0, 0]), 1.0 / CANDIDATES, rel_tol=1e-5)
