# SPDX-License-Identifier: Apache-2.0
"""Sampled (temperature > 0) speculative verification on HPU (idea-sweep I07): exact in distribution.

``sampled_rejection_sample`` is speculative sampling of argmax drafts (MTP gives no draft probabilities): accept
draft ``d`` with the target's probability ``p(d)``, at the first rejection draw from ``p`` without ``d``, with
every draft accepted append the bonus token. Its output must be distributed as plain sampling from the target:
the first token as ``p_0``, and the second, given that the first was the draft ``d_0``, as ``p_1``. These tests
check that on many independent rows with a chi-square bound, check the greedy rows of a mixed batch against the
greedy sampler, the torch ``expand_batch_to_tokens`` against upstream's definition, the HPU batch's greedy
temperature (-1.0) in ``apply_sampling_constraints``, and the request guard of a server without the flag.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.v1.sample import hpu_rejection_sampler as hrs

PLACEHOLDER = hrs.PLACEHOLDER_TOKEN_ID


def _metadata(temperature, all_greedy=False, all_random=True, generators=None, top_k=None, top_p=None):
    return SimpleNamespace(all_greedy=all_greedy, all_random=all_random, temperature=temperature,
                           generators=generators or {}, top_k=top_k, top_p=top_p)


def _layout(batch: int, k: int, n_draft: list[int]):
    cumulative = torch.tensor(list(torch.tensor(n_draft).cumsum(0)), dtype=torch.int32)
    return n_draft, cumulative


def _chi_square(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, int]:
    """Pearson's statistic over the cells with expected count >= 5 (the rest pooled), and its degrees of freedom."""
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
    # Wilson-Hilferty upper quantile at z = 4.5 (p about 3e-6): a correct sampler fails it almost never.
    z = 4.5
    return dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof)))**3


def test_expand_batch_to_tokens_is_upstreams_definition():
    x = torch.tensor([7.0, -1.0, 3.0])
    cumulative = torch.tensor([2, 5, 6], dtype=torch.int32)
    out = hrs.expand_batch_to_tokens(x, cumulative, 6)
    assert out.tolist() == [7.0, 7.0, -1.0, -1.0, -1.0, 3.0]
    out = hrs.expand_batch_to_tokens(x, cumulative, 6, replace_from=-1.0, replace_to=1.0)
    assert out.tolist() == [7.0, 7.0, 1.0, 1.0, 1.0, 3.0]
    # the HPU's padded layout: 3 requests x 3 rows, 2 + 3 + 1 real (cu = [2, 5, 6]), padded rows take any value
    assert hrs.expand_batch_to_tokens(torch.tensor([1, 2, 3]), cumulative, 6).tolist() == [1, 1, 2, 2, 2, 3]
    assert hrs.expand_batch_to_tokens(torch.tensor([4, 5]), torch.tensor([0, 3], dtype=torch.int32), 3).tolist() \
        == [5, 5, 5]


def test_apply_sampling_constraints_divides_a_greedy_row_by_one():
    logits = torch.tensor([[1.0, 4.0, 2.0], [1.0, 4.0, 2.0], [3.0, 0.0, 9.0]])
    # request 0 greedy (the HPU batch stores -1.0), request 1 at temperature 2 (two draft rows)
    md = _metadata(temperature=torch.tensor([-1.0, 2.0]))
    out = hrs.apply_sampling_constraints(logits.clone(), torch.tensor([1, 3], dtype=torch.int32), md)
    assert out.tolist() == [[1.0, 4.0, 2.0], [0.5, 2.0, 1.0], [1.5, 0.0, 4.5]]
    greedy = _metadata(temperature=None, all_greedy=True)
    assert hrs.apply_sampling_constraints(logits, torch.tensor([3], dtype=torch.int32), greedy) is logits


def test_apply_sampling_constraints_applies_top_k():
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    md = _metadata(temperature=torch.tensor([1.0]), top_k=torch.tensor([2]))
    out = hrs.apply_sampling_constraints(logits.clone(), torch.tensor([1], dtype=torch.int32), md)
    assert out[0, 1].item() == 4.0 and out[0, 3].item() == 3.0
    assert out[0, 0].item() == -math.inf and out[0, 2].item() == -math.inf


def _first_two_distribution(p0: torch.Tensor, p1: torch.Tensor, d0: int, d1: int, rows: int, seed: int):
    """Run ``rows`` identical requests with two drafts (d0, d1) against target distributions p0, p1."""
    torch.manual_seed(seed)
    vocab = p0.numel()
    k = 2
    logits = torch.stack([p0.log(), p1.log()]).repeat(rows, 1)  # [rows * 2, vocab], log-probabilities
    draft = torch.tensor([d0, d1], dtype=torch.int32).repeat(rows)
    bonus = torch.full((rows, 1), vocab - 1, dtype=torch.int32)  # a fixed bonus: only its position matters here
    n_draft, cumulative = _layout(rows, k, [k] * rows)
    out = hrs.sampled_rejection_sample(draft, n_draft, cumulative, logits, bonus,
                                       _metadata(temperature=torch.ones(rows)))
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_the_first_token_is_distributed_as_the_target(seed):
    vocab = 12
    torch.manual_seed(100 + seed)
    p0 = torch.softmax(torch.randn(vocab) * 1.5, dim=0)
    p1 = torch.softmax(torch.randn(vocab) * 1.5, dim=0)
    d0, d1 = int(p0.argmax()), int(p1.argmax())  # argmax drafts, as MTP proposes
    rows = 60000
    out = _first_two_distribution(p0, p1, d0, d1, rows, seed)
    first = torch.bincount(out[:, 0].long(), minlength=vocab).double()
    stat, dof = _chi_square(first, p0.double())
    assert stat < _chi_square_bound(dof), (stat, dof)
    # the draft is accepted with probability p0(d0)
    assert abs((out[:, 0] == d0).double().mean().item() - p0[d0].item()) < 5 * math.sqrt(0.25 / rows)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_second_token_after_an_accepted_draft_is_distributed_as_the_target(seed):
    vocab = 10
    torch.manual_seed(200 + seed)
    p0 = torch.softmax(torch.randn(vocab) * 1.5, dim=0)
    p1 = torch.softmax(torch.randn(vocab) * 1.5, dim=0)
    d0, d1 = int(p0.argmax()), int(p1.argmax())
    out = _first_two_distribution(p0, p1, d0, d1, 80000, seed)
    accepted_first = out[:, 0] == d0
    second = out[accepted_first, 1]
    assert (second != PLACEHOLDER).all()  # after an accepted draft the next position always holds a token
    counts = torch.bincount(second.long(), minlength=vocab).double()
    stat, dof = _chi_square(counts, p1.double())
    assert stat < _chi_square_bound(dof), (stat, dof)
    # after a rejected first draft nothing follows the replacement token
    assert (out[~accepted_first, 1] == PLACEHOLDER).all() and (out[~accepted_first, 2] == PLACEHOLDER).all()
    # the bonus follows exactly when both drafts are accepted
    both = accepted_first & (out[:, 1] == d1)
    assert (out[both, 2] == vocab - 1).all() and (out[~both, 2] == PLACEHOLDER).all()


def test_a_non_argmax_draft_is_still_exact():
    # Any proposal is exact under the rule (accept with p(d)); a draft that is not the argmax is accepted less.
    vocab = 8
    torch.manual_seed(7)
    p0 = torch.softmax(torch.randn(vocab), dim=0)
    p1 = torch.softmax(torch.randn(vocab), dim=0)
    d0 = int(p0.argmin())
    out = _first_two_distribution(p0, p1, d0, 0, 60000, 3)
    counts = torch.bincount(out[:, 0].long(), minlength=vocab).double()
    stat, dof = _chi_square(counts, p0.double())
    assert stat < _chi_square_bound(dof), (stat, dof)


def test_greedy_rows_of_a_mixed_batch_follow_the_greedy_rule():
    vocab = 6
    logits = torch.tensor([
        [0.0, 5.0, 1.0, 0.0, 0.0, 0.0],  # request 0 (greedy), row 0: argmax 1
        [0.0, 0.0, 0.0, 7.0, 0.0, 0.0],  # request 0, row 1: argmax 3
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # request 1 (sampled), row 0: p(0) ~ 1
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],  # request 1, row 1: p(5) ~ 1
    ])
    draft = torch.tensor([1, 2, 0, 5], dtype=torch.int32)  # request 0: d0 right, d1 wrong; request 1: both likely
    bonus = torch.tensor([[4], [4]], dtype=torch.int32)
    md = _metadata(temperature=torch.tensor([-1.0, 1.0]), all_random=False)
    torch.manual_seed(0)
    out = hrs.sampled_rejection_sample(draft, [2, 2], torch.tensor([2, 4], dtype=torch.int32), logits, bonus, md)
    greedy = hrs.rejection_sample_pytorch(draft[:2].view(1, 2), torch.tensor([1, 3]), bonus[:1], [2],
                                          torch.tensor([2], dtype=torch.int32))
    assert out[0].tolist() == greedy[0].tolist() == [1, 3, PLACEHOLDER]
    assert out[1].tolist() in ([0, 5, 4], )  # p(d) = 1 - 6e-4 per row: accepted at seed 0


def test_padded_rows_and_requests_without_drafts():
    vocab = 5
    logits = torch.full((6, vocab), -30.0)
    logits[0, 2] = 10.0  # request 0 row 0: p(2) ~ 1
    draft = torch.tensor([2, 9999, 9999, 1, 1, 1], dtype=torch.int32)  # padded rows carry any id
    bonus = torch.tensor([[3], [4]], dtype=torch.int32)
    md = _metadata(temperature=torch.tensor([1.0, 1.0]))
    out = hrs.sampled_rejection_sample(draft, [1, 0], torch.tensor([1, 1], dtype=torch.int32), logits, bonus, md)
    assert out.tolist() == [[2, 3, PLACEHOLDER, PLACEHOLDER], [4, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER]]


def test_seeded_requests_repeat_their_draws():
    vocab = 16
    torch.manual_seed(5)
    logits = torch.randn(8, vocab)
    draft = logits.argmax(dim=-1).to(torch.int32)
    bonus = torch.zeros((4, 1), dtype=torch.int32)
    outs = []
    for _ in range(2):
        generators = {1: torch.Generator().manual_seed(1234)}
        torch.manual_seed(torch.seed() % 1000)  # the unseeded rows may differ between the two calls
        md = _metadata(temperature=torch.ones(4), generators=generators)
        outs.append(hrs.sampled_rejection_sample(draft, [2] * 4, torch.tensor([2, 4, 6, 8], dtype=torch.int32),
                                                 logits.clone(), bonus, md))
    assert outs[0][1].tolist() == outs[1][1].tolist()


def test_rejection_sample_refuses_sampled_requests_without_the_flag(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_SPEC_SAMPLED_VERIFY", "0")
    md = _metadata(temperature=torch.ones(1))
    with pytest.raises(ValueError, match="VLLM_HPU_SPEC_SAMPLED_VERIFY=1"):
        hrs.rejection_sample(torch.tensor([1], dtype=torch.int32), [1], 1, torch.tensor([1], dtype=torch.int32), None,
                             torch.zeros(1, 4), torch.tensor([[0]], dtype=torch.int32), md)
    monkeypatch.setenv("VLLM_HPU_SPEC_SAMPLED_VERIFY", "1")
    out = hrs.rejection_sample(torch.tensor([1], dtype=torch.int32), [1], 1, torch.tensor([1], dtype=torch.int32),
                               None, torch.zeros(1, 4), torch.tensor([[0]], dtype=torch.int32), md)
    assert out.shape == (1, 2)


def test_the_upstream_module_runs_the_hpu_functions():
    from vllm.v1.sample import rejection_sampler

    assert rejection_sampler.rejection_sample is hrs.rejection_sample
    assert rejection_sampler.apply_sampling_constraints is hrs.apply_sampling_constraints
    assert rejection_sampler.expand_batch_to_tokens is hrs.expand_batch_to_tokens
