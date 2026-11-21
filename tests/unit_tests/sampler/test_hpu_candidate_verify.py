# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_SPEC_SAMPLED_VERIFY=2: sampled verification of an MTP step over the top candidates of every row.

The candidate distribution equals vLLM's (temperature, top-k, top-p, softmax over the vocabulary) on its support;
the tokens are distributed as speculative sampling demands (chi-square, as test_hpu_spec_sampled_verify.py does
for the exact path); greedy rows of a mixed batch follow the greedy rule; every rank draws the same noise.
"""
from __future__ import annotations

import math

import pytest
import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_gaudi.v1.sample import hpu_candidate_verify as cv

PH = -1  # the placeholder id


def _chi_square(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, int]:
    total = counts.sum().item()
    expected = probs * total
    big = expected >= 5
    stat = (((counts[big] - expected[big])**2) / expected[big]).sum().item()
    cells = int(big.sum().item())
    rest_e, rest_o = expected[~big].sum().item(), counts[~big].sum().item()
    if rest_e > 0:
        stat += (rest_o - rest_e)**2 / max(rest_e, 1e-9)
        cells += 1
    return stat, cells - 1


def _chi_square_bound(dof: int) -> float:
    z = 4.5  # p about 3e-6: a correct sampler fails it almost never
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof)))**3


def _vllm_probs(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor):
    """vLLM's processed distribution over the whole vocabulary (the exact path's reference)."""
    scaled = logits.to(torch.float32) / temperature.unsqueeze(1)
    return apply_top_k_top_p_pytorch(scaled.clone(), top_k.to(torch.long), top_p).softmax(dim=-1)


def _scatter(probs_c: torch.Tensor, ids: torch.Tensor, vocab: int) -> torch.Tensor:
    full = torch.zeros((probs_c.shape[0], vocab), dtype=torch.float32)
    return full.scatter(1, ids.to(torch.long), probs_c)


def test_candidate_ids_are_the_token_ids_of_the_largest_logits():
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(6, 3000).to(dtype)
        values, ids = cv.candidates(logits, 64)
        assert values.dtype == torch.float32 and ids.dtype == torch.int32
        top_values, top_index = torch.topk(logits, 64, dim=-1)
        assert torch.equal(ids, top_index.to(torch.int32))
        assert torch.equal(values, top_values.to(torch.float32))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_candidate_probs_equal_vllms_distribution(seed):
    torch.manual_seed(seed)
    vocab, rows = 2000, 12
    logits = torch.randn(rows, vocab) * 3.0
    temperature = torch.tensor([1.0, 0.7, 1.3, 1.0] * 3)
    top_k = torch.tensor([20, 20, 20, 1, 5, 64, 64, 2, 20, 33, 7, 64], dtype=torch.float32)
    top_p = torch.tensor([0.95, 0.95, 0.95, 1.0, 0.5, 1.0, 0.8, 0.95, 0.3, 0.99, 1.0, 0.9])
    values, ids = cv.candidates(logits, 64)
    probs = cv.candidate_probs(values, temperature, top_k, top_p)
    reference = _vllm_probs(logits, temperature, top_k, top_p)
    assert torch.allclose(_scatter(probs, ids, vocab), reference, atol=2e-6), \
        (_scatter(probs, ids, vocab) - reference).abs().max()
    assert torch.allclose(probs.sum(dim=1), torch.ones(rows), atol=1e-5)


def _layout(batch: int, k: int, n_draft: list[int]):
    """The HPU's padded layout: request b's rows b * (k + 1) + i (i < k: verification, i = k: bonus)."""
    target = []
    for b in range(batch):
        target += [b * (k + 1) + i if i < n_draft[b] else -1 for i in range(k)]
    bonus = [b * (k + 1) + n_draft[b] for b in range(batch)]  # the bonus row follows the request's real drafts
    cumulative = torch.tensor(n_draft, dtype=torch.int32).cumsum(0).to(torch.int32)
    return torch.tensor(target, dtype=torch.int32), torch.tensor(bonus, dtype=torch.int32), cumulative


def _run(logits, draft, n_draft, k, temperatures, top_ks, top_ps, seed):
    batch = len(n_draft)
    target, bonus, cumulative = _layout(batch, k, n_draft)
    generator = torch.Generator().manual_seed(seed)
    inputs = cv.host_inputs(temperatures, top_ks, top_ps, k, generator)
    return cv.candidate_rejection_sample(logits, target, bonus, draft, cumulative, inputs, PH)


def _requests(p_rows: list[torch.Tensor], drafts: list[int], batch: int):
    """``batch`` identical requests with len(drafts) drafts against the target rows ``p_rows`` (log-probabilities
    as logits) and a bonus row (the last of ``p_rows``)."""
    k = len(drafts)
    logits = torch.stack([p.log() for p in p_rows]).repeat(batch, 1)  # [batch * (k + 1), vocab]
    draft = torch.tensor(drafts, dtype=torch.int32).repeat(batch)
    return logits, draft, [k] * batch


@pytest.mark.parametrize("seed", [0, 1])
def test_tokens_are_distributed_as_speculative_sampling_demands(seed):
    vocab, k, batch = 40, 2, 60000
    torch.manual_seed(300 + seed)
    rows = [torch.softmax(torch.randn(vocab) * 1.5, dim=0) for _ in range(k + 1)]
    top_k, top_p = 20, 0.95
    targets = [_vllm_probs(p.log().unsqueeze(0), torch.ones(1), torch.tensor([top_k]), torch.tensor([top_p]))[0]
               for p in rows]
    drafts = [int(t.argmax()) for t in targets[:k]]  # argmax drafts, as MTP proposes
    logits, draft, n_draft = _requests(rows, drafts, batch)
    out = _run(logits, draft, n_draft, k, [1.0] * batch, [top_k] * batch, [top_p] * batch, seed)
    assert out.shape == (batch, k + 1) and out.dtype == torch.int32
    # the first token is distributed as the target; the draft is accepted with probability p(draft)
    stat, dof = _chi_square(torch.bincount(out[:, 0].long(), minlength=vocab).double(), targets[0].double())
    assert stat < _chi_square_bound(dof), (stat, dof)
    accepted_first = out[:, 0] == drafts[0]
    assert abs(accepted_first.double().mean().item() - targets[0][drafts[0]].item()) < 5 * math.sqrt(0.25 / batch)
    # after an accepted draft the next token is distributed as the next target; after a rejection nothing follows
    second = out[accepted_first, 1]
    stat, dof = _chi_square(torch.bincount(second.long(), minlength=vocab).double(), targets[1].double())
    assert stat < _chi_square_bound(dof), (stat, dof)
    assert (out[~accepted_first, 1] == PH).all() and (out[~accepted_first, 2] == PH).all()
    # with both drafts accepted the bonus token is drawn from the bonus row's distribution
    both = accepted_first & (out[:, 1] == drafts[1])
    stat, dof = _chi_square(torch.bincount(out[both, 2].long(), minlength=vocab).double(), targets[2].double())
    assert stat < _chi_square_bound(dof), (stat, dof)
    assert (out[~both, 2] == PH).all()


def test_a_draft_outside_the_kept_set_is_always_replaced():
    vocab = 50
    torch.manual_seed(7)
    p = torch.softmax(torch.randn(vocab) * 2.0, dim=0)
    order = p.argsort(descending=True)
    outside = int(order[30])  # below top_k = 20
    logits, draft, n_draft = _requests([p, p], [outside], 2000)
    out = _run(logits, draft, n_draft, 1, [1.0] * 2000, [20] * 2000, [1.0] * 2000, 0)
    assert (out[:, 0] != outside).all() and (out[:, 1] == PH).all()
    assert set(out[:, 0].tolist()) <= set(order[:20].tolist())


def test_greedy_rows_of_a_mixed_batch_follow_the_greedy_rule():
    vocab, k = 30, 2
    torch.manual_seed(11)
    rows = [torch.softmax(torch.randn(vocab) * 2.0, dim=0) for _ in range(k + 1)]
    argmaxes = [int(p.argmax()) for p in rows]
    wrong = (argmaxes[1] + 1) % vocab
    logits = torch.stack([p.log() for p in rows]).repeat(2, 1)
    # request 0 greedy: its first draft is the arg max (accepted), its second is not (replaced by the arg max)
    draft = torch.tensor([argmaxes[0], wrong, argmaxes[0], argmaxes[1]], dtype=torch.int32)
    out = _run(logits, draft, [k, k], k, [0.0, 1.0], [-1, 20], [1.0, 0.95], 3)
    assert out[0].tolist() == [argmaxes[0], argmaxes[1], PH]
    # all drafts of a greedy request accepted: the bonus is the bonus row's arg max
    draft = torch.tensor([argmaxes[0], argmaxes[1], argmaxes[0], argmaxes[1]], dtype=torch.int32)
    out = _run(logits, draft, [k, k], k, [0.0, 1.0], [-1, 20], [1.0, 0.95], 4)
    assert out[0].tolist() == argmaxes


def test_padded_rows_and_requests_without_drafts():
    vocab, k = 25, 3
    torch.manual_seed(5)
    p = torch.softmax(torch.randn(vocab) * 4.0, dim=0)
    top = int(p.argmax())
    logits = p.log().repeat(2 * (k + 1), 1)
    # request 0: one draft (the arg max, greedy: accepted) then two padded rows; request 1: no draft at all
    draft = torch.tensor([top, 0, 0, 0, 0, 0], dtype=torch.int32)
    out = _run(logits, draft, [1, 0], k, [0.0, 0.0], [-1, -1], [1.0, 1.0], 6)
    assert out[0].tolist() == [top, top, PH, PH]  # the accepted draft, then the bonus
    assert out[1].tolist() == [top, PH, PH, PH]  # no draft: the bonus only


def test_every_rank_draws_the_same_noise():
    one = cv.host_inputs([1.0, 0.0], [20, -1], [0.95, 1.0], 3, torch.Generator().manual_seed(cv.NOISE_SEED))
    two = cv.host_inputs([1.0, 0.0], [20, -1], [0.95, 1.0], 3, torch.Generator().manual_seed(cv.NOISE_SEED))
    assert torch.equal(one, two) and one.shape == (2 * 4, cv.PARAM_COLUMNS + cv.CANDIDATES)
    # verification rows 0..5 (request r // 3), bonus rows 6, 7; the greedy request runs as top_k 1
    assert one[:3, :3].tolist() == [[1.0, 20.0, 0.949999988079071]] * 3
    assert one[3:6, :3].tolist() == [[1.0, 1.0, 1.0]] * 3 and one[7, :3].tolist() == [1.0, 1.0, 1.0]
    assert (one[:, 3] > 0).all() and (one[:, 3] <= 1).all() and (one[:, cv.PARAM_COLUMNS:] >= 0).all()


def test_refusal_names_what_the_candidates_cannot_serve():
    assert cv.refusal([1.0, 0.0], [20, -1], [0.95, 1.0], seeded=False, logprobs=False) is None
    assert cv.refusal([1.0], [20], [0.95], seeded=True, logprobs=False) == "a seeded request"
    assert cv.refusal([1.0], [20], [0.95], seeded=False, logprobs=True) == "logprobs"
    assert "top_k 0" in cv.refusal([1.0], [0], [0.95], seeded=False, logprobs=False)
    assert "top_k -1" in cv.refusal([0.7], [-1], [1.0], seeded=False, logprobs=False)
    assert "top_k 65" in cv.refusal([1.0], [65], [1.0], seeded=False, logprobs=False)


def test_a_lookup_padding_draft_is_never_accepted():
    # Prompt lookup pads its drafts with -1 (spec-lookup's 2dd6bc75 for the exact path): no candidate id is -1, so
    # such a position is never accepted and its token comes from the whole target row (nothing to exclude).
    vocab, batch = 30, 40000
    torch.manual_seed(13)
    p = torch.softmax(torch.randn(vocab) * 1.5, dim=0)
    target = _vllm_probs(p.log().unsqueeze(0), torch.ones(1), torch.tensor([20]), torch.tensor([0.95]))[0]
    logits, _, n_draft = _requests([p, p], [0], batch)
    draft = torch.full((batch, ), -1, dtype=torch.int32)
    out = _run(logits, draft, n_draft, 1, [1.0] * batch, [20] * batch, [0.95] * batch, 5)
    assert (out[:, 0] >= 0).all() and (out[:, 1] == PH).all()  # always rejected, never a bonus after it
    stat, dof = _chi_square(torch.bincount(out[:, 0].long(), minlength=vocab).double(), target.double())
    assert stat < _chi_square_bound(dof), (stat, dof)
