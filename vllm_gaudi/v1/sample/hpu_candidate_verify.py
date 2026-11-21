# SPDX-License-Identifier: Apache-2.0
"""Sampled verification of an MTP step over candidates (``VLLM_HPU_SPEC_SAMPLED_VERIFY=2``, idea-sweep I07).

A request without a temperature gets the checkpoint's default sampling (temperature 1.0, top_k 20, top_p 0.95).
The exact path (``=1``, ``hpu_rejection_sampler.sampled_rejection_sample``) works on the whole vocabulary of every
verification row: vLLM's top-k / top-p (a sort of ``[rows, 248,320]``), a softmax, the exponential noise and two
arg maxes over the vocabulary, plus the general sampler's own sort for the bonus rows. With ``0 < top_k <= C`` the
target distribution lives on the row's ``top_k`` largest logits, so everything here runs on the ``C`` largest
logits of a row (``torch.topk``, largest first):

* ``candidate_probs``: vLLM's distribution after temperature, top-k and top-p on the candidates (engine-step's
  ``candidate_sampling.candidate_probs``, with the k-th value found by an fp32 compare of column numbers instead of
  a gather with an int64 index);
* a draft is accepted with probability ``p(draft)`` (the column whose id equals the draft; zero when the draft is
  no candidate, where vLLM's probability is zero too), against a host-made uniform in ``(0, 1]``;
* at the first rejection the token comes from ``p`` with the draft's column zeroed (the exponential race
  ``argmax p / q`` over the candidates, ``q ~ Exp(1)`` host-made); with every draft accepted the bonus token is
  drawn from the bonus row's candidates the same way.

That is upstream's ``NO_DRAFT_PROBS`` speculative sampling (exact in distribution), restricted to the support of
the target distribution. Two differences of no weight, as in engine-step's module: a tie group at the k-th value
that reaches past column ``C`` is cut at ``C``, and sums are taken in another order (probabilities agree to
rounding). A greedy request of a mixed batch runs as top_k 1 (its arg max; an exact tie of the two largest logits
may break the other way than ``argmax``).

Ids (card rules 2 and 9, BRIEF 23:23: top-k indices read back to the host as data were wrong on the card, in-graph
consumers were right): the candidates' ids are read out of an fp32 id table with ``topk``'s indices (an in-graph
gather) and cast fp32 -> int32, as collectives' ``vocab_parallel.top_k`` does; no int64 id is narrowed and no
``topk`` index leaves the graph. The noise comes from the host, from a generator seeded alike on every rank, so
every TP rank draws the same tokens without a device random op.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

CANDIDATES = 64  # candidates per row: the largest top_k a request may ask for on this path
GREEDY_TEMPERATURE = 1e-5  # vLLM's _SAMPLING_EPS: below it a request is greedy
NOISE_SEED = 20261120  # the host noise generator's seed, the same on every rank
PARAM_COLUMNS = 4  # temperature, top_k, top_p, acceptance uniform; the row's exponential noise follows


def candidates(logits: torch.Tensor, count: int = CANDIDATES) -> tuple[torch.Tensor, torch.Tensor]:
    """``(values fp32 [rows, C] largest first, ids int32 [rows, C])``: the ``C`` largest logits of every row (all of
    them when the vocabulary is smaller than ``count``)."""
    values, index = torch.topk(logits, min(count, logits.shape[-1]), dim=-1)
    table = torch.arange(logits.shape[-1], dtype=torch.float32, device=logits.device).expand(logits.shape[0], -1)
    return values.to(torch.float32), torch.gather(table, -1, index).to(torch.int32)


def candidate_probs(values: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor,
                    top_p: torch.Tensor) -> torch.Tensor:
    """vLLM's processed distribution on the candidates: fp32 ``[rows, C]``, zero outside the kept set.

    ``values`` fp32 ``[rows, C]`` sorted largest first; ``temperature`` (above zero), ``top_k`` (1 .. C, as fp32)
    and ``top_p`` (in ``(0, 1]``) fp32 ``[rows]``. vLLM keeps every token at or above the k-th largest logit, then
    drops the smallest tokens whose summed probability stays at or below ``1 - top_p``, never the largest.
    """
    scaled = values / temperature.unsqueeze(1)
    columns = torch.arange(values.shape[1], dtype=torch.float32, device=values.device).unsqueeze(0)
    kth = torch.where(columns == (top_k - 1).unsqueeze(1), scaled, torch.zeros_like(scaled)).sum(dim=1, keepdim=True)
    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = scaled.softmax(dim=-1)
    # The summed probability of a column and of every smaller one (vLLM's ascending cumulative sum).
    tail = probs.sum(dim=-1, keepdim=True) - probs.cumsum(dim=-1) + probs
    rest = scaled[:, 1:].masked_fill(tail[:, 1:] <= (1 - top_p).unsqueeze(1), float("-inf"))
    return torch.cat((scaled[:, :1], rest), dim=1).softmax(dim=-1)


def refusal(temperatures: Sequence[float], top_ks: Sequence[int], top_ps: Sequence[float], *, seeded: bool,
            logprobs: bool, candidates_per_row: int = CANDIDATES) -> Optional[str]:
    """Why a step cannot take this path (it then takes the exact one, ``=1``), or None."""
    if seeded:
        return "a seeded request"
    if logprobs:
        return "logprobs"
    for temperature, top_k, top_p in zip(temperatures, top_ks, top_ps):
        if temperature < GREEDY_TEMPERATURE:
            continue
        if not 0 < int(top_k) <= candidates_per_row:
            return "top_k %s outside 1..%d" % (top_k, candidates_per_row)
        if not 0.0 < float(top_p) <= 1.0:
            return "top_p %s" % (top_p, )
    return None


def host_inputs(temperatures: Sequence[float], top_ks: Sequence[int], top_ps: Sequence[float], max_draft: int,
                generator: torch.Generator, candidates_per_row: int = CANDIDATES) -> torch.Tensor:
    """The host side of one step: fp32 ``[batch * (k + 1), 4 + C]`` on the CPU.

    Rows ``0 .. batch * k - 1`` are the verification rows (request ``r // k``), rows ``batch * k ..`` the bonus rows
    (request ``r - batch * k``). Columns: temperature, top_k, top_p, the acceptance uniform in ``(0, 1]``, then the
    row's exponential noise. A greedy request runs as temperature 1, top_k 1, top_p 1 (its arg max).
    """
    batch = len(temperatures)
    request_params = torch.empty((batch, 3), dtype=torch.float32)
    for i, (temperature, top_k, top_p) in enumerate(zip(temperatures, top_ks, top_ps)):
        if temperature < GREEDY_TEMPERATURE:
            request_params[i] = torch.tensor([1.0, 1.0, 1.0])
        else:
            request_params[i] = torch.tensor([float(temperature), float(top_k), float(top_p)])
    rows = batch * (max_draft + 1)
    out = torch.empty((rows, PARAM_COLUMNS + candidates_per_row), dtype=torch.float32)
    out[:batch * max_draft, :3] = request_params.repeat_interleave(max_draft, dim=0)
    out[batch * max_draft:, :3] = request_params
    out[:, 3] = 1.0 - torch.rand((rows, ), generator=generator)  # (0, 1]: p >= u accepts with probability p
    # Exp(1) noise, never exactly 0: a zero would make 0 / 0 (NaN) for a column outside the kept set.
    out[:, PARAM_COLUMNS:].exponential_(generator=generator).clamp_(min=torch.finfo(torch.float32).tiny)
    return out


def assemble(draft: torch.Tensor, n_draft: torch.Tensor, accepted: torch.Tensor, replacement: torch.Tensor,
             bonus: torch.Tensor, placeholder: int) -> torch.Tensor:
    """``[batch, k + 1]`` int32: the accepted drafts, the replacement at the first rejection, or the bonus token
    after the last draft when every draft was accepted; ``placeholder`` after the last token. ``draft``,
    ``accepted``, ``replacement`` ``[batch, k]``; ``n_draft`` int32 ``[batch]`` real drafts; ``bonus`` ``[batch, 1]``."""
    batch, max_draft = draft.shape
    device = draft.device
    pos = torch.arange(max_draft, device=device, dtype=torch.int32).unsqueeze(0)
    valid = pos < n_draft.unsqueeze(1)
    rejected = (~accepted) & valid
    any_rejected = rejected.any(dim=1, keepdim=True)
    first = torch.argmax(rejected.to(torch.int32), dim=1, keepdim=True).to(torch.int32)
    placeholders = torch.full_like(draft, placeholder)
    kept = valid & ((pos < first) | ~any_rejected)
    tokens = torch.where(kept, draft, placeholders)
    tokens = torch.where(any_rejected & (pos == first), replacement, tokens)
    out = torch.cat((tokens, placeholders[:, :1]), dim=1)
    out_pos = torch.arange(max_draft + 1, device=device, dtype=torch.int32).unsqueeze(0)
    bonus_here = (~any_rejected) & (out_pos == n_draft.unsqueeze(1))
    return torch.where(bonus_here, bonus.reshape(batch, 1).to(torch.int32), out)


def candidate_rejection_sample(logits: torch.Tensor, target_logits_indices: torch.Tensor,
                               bonus_logits_indices: torch.Tensor, draft_token_ids: torch.Tensor,
                               cu_num_draft_tokens: torch.Tensor, inputs: torch.Tensor, placeholder: int,
                               candidates_per_row: int = CANDIDATES) -> torch.Tensor:
    """One verification step over candidates: ``[batch, k + 1]`` int32 tokens (see ``assemble``).

    ``logits`` ``[rows, vocab]`` of the step (the HPU's padded layout: ``target_logits_indices`` ``[batch * k]``
    with -1 for a request's missing drafts, ``bonus_logits_indices`` ``[batch]``); ``draft_token_ids``
    ``[batch * k]``; ``cu_num_draft_tokens`` ``[batch]``; ``inputs`` from ``host_inputs`` on the device.
    """
    batch = bonus_logits_indices.shape[0]
    num_targets = target_logits_indices.shape[0]
    max_draft = num_targets // batch
    values, ids = candidates(logits, candidates_per_row)
    rows = torch.cat((target_logits_indices.to(torch.int32).clamp(min=0), bonus_logits_indices.to(torch.int32)))
    values = torch.index_select(values, 0, rows)
    ids = torch.index_select(ids, 0, rows)
    probs = candidate_probs(values, inputs[:, 0], inputs[:, 1], inputs[:, 2])
    noise = inputs[:, PARAM_COLUMNS:PARAM_COLUMNS + values.shape[1]]

    draft = draft_token_ids.reshape(-1).to(torch.int32)
    match = ids[:num_targets] == draft.unsqueeze(1)
    target_probs = probs[:num_targets]
    p_draft = torch.where(match, target_probs, torch.zeros_like(target_probs)).sum(dim=1)
    accepted = p_draft >= inputs[:num_targets, 3]
    # At a rejection: the target distribution with the draft removed (an exponential race over the candidates).
    remaining = torch.where(match, torch.zeros_like(target_probs), target_probs)
    choice = (remaining / noise[:num_targets]).argmax(dim=1, keepdim=True)
    replacement = torch.gather(ids[:num_targets], 1, choice)
    bonus_choice = (probs[num_targets:] / noise[num_targets:]).argmax(dim=1, keepdim=True)
    bonus = torch.gather(ids[num_targets:], 1, bonus_choice)

    cumulative = cu_num_draft_tokens.to(device=logits.device, dtype=torch.int32)
    n_draft = cumulative - torch.cat((torch.zeros(1, dtype=torch.int32, device=logits.device), cumulative[:-1]))
    return assemble(draft.view(batch, max_draft), n_draft, accepted.view(batch, max_draft),
                    replacement.view(batch, max_draft), bonus, placeholder)
