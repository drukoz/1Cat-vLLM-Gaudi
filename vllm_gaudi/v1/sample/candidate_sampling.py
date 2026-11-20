# SPDX-License-Identifier: Apache-2.0
"""Sampling over candidates (``VLLM_HPU_CANDIDATE_SAMPLING=1``): the sampled step without a pass over the vocabulary.

A request without a temperature gets the checkpoint's default sampling (temperature 1.0, top_k 20, top_p 0.95).
vLLM's sampler then, on every step and for every row, sorts the whole vocabulary with indices, masks, scatters
back to vocabulary order, takes a softmax and draws exponential noise over the vocabulary
(``vllm/v1/sample/ops/topk_topp_sampler.py``: ``apply_top_k_top_p_pytorch`` and ``random_sample``; no Triton on
a card, no CPU sync allowed). Here the ``C`` largest logits of a row are taken first (``torch.topk``, sorted,
with their token ids) and everything else runs on ``[rows, C]``:

* temperature: the candidates' logits divided by the row's temperature;
* top-k: every candidate below the row's k-th largest value gets ``-inf`` (ties at the k-th value stay, as in
  vLLM);
* top-p as vLLM cuts it: the smallest tokens whose summed probability stays at or below ``1 - top_p`` are
  dropped, the largest token never;
* a softmax, then the exponential race ``argmax(probs / noise)``, which draws column ``i`` with probability
  ``probs[i]`` (the same draw vLLM makes over the vocabulary);
* the token id by a gather from the candidates' ids.

It is vLLM's distribution whenever every token at or above the row's k-th largest logit is a candidate: that is
``0 < top_k <= C``, and no tie group at the k-th value that reaches past column ``C`` (such a group is cut at
``C`` candidates; vLLM keeps all of it). Two differences of no weight remain. When the top-p cut falls inside a
group of tokens with equal logits, which of them stay is decided by the order of equal values (in vLLM by its
sort, here by top-k): the same number of tokens with the same probabilities stays. And the sums are taken in
another order than vLLM's, so probabilities agree to rounding (about 1e-6) and a token whose summed probability
sits within rounding of the cut may fall on the other side. The random stream is not vLLM's: a seeded request
is reproducible, but not equal to its output on the general path. Requests outside the conditions keep the
general path (``candidate_refusal``).

The noise comes from the host with the step's inputs: ``[rows, C]`` fp32 (8 KB at 32 rows), one copy, a seeded
request's row from its own generator. So the device needs no random op, and what a row draws depends on its own
noise only.

Shapes are fixed per batch bucket and every op is one the card runs elsewhere (top-k, gather with a host-made
int64 index, masked fill, softmax, cumulative sum, a concatenation of slices, arg max). Integer ids (card rule
9): the candidates' ids stay int64 as ``topk`` makes them, the index of the k-th column is made on the host as
int64, and nothing is narrowed in the graph; the runner reads the sampled ids back as int64.

Pure functions: this module imports only ``torch``.
"""

from typing import Optional, Sequence

import torch

CANDIDATES = 64  # candidates per row: the largest top_k a request may ask for on this path
PARAM_COLUMNS = 3  # temperature, top_p, greedy flag; the row's noise follows
STOP_ID_LIMIT = 8  # fused_greedy.STOP_ID_WIDTH: stop ids of a min_tokens request the logits region masks
GREEDY_TEMPERATURE = 1e-5  # vLLM's _SAMPLING_EPS: below it a request is greedy


def take_candidates(logits: torch.Tensor, candidates: int = CANDIDATES) -> tuple[torch.Tensor, torch.Tensor]:
    """``(values fp32 [rows, C], ids int64 [rows, C])``: the ``C`` largest logits of every row, largest first,
    and their token ids. The one call a vocabulary-parallel top-k replaces."""
    values, ids = logits.to(torch.float32).topk(candidates, dim=-1)
    return values, ids


def candidate_probs(values: torch.Tensor, temperature: torch.Tensor, top_k_index: torch.Tensor,
                    top_p: torch.Tensor) -> torch.Tensor:
    """vLLM's processed distribution on the candidates: fp32 ``[rows, C]``, zero outside the kept set.

    ``values`` fp32 ``[rows, C]`` sorted largest first; ``temperature`` fp32 ``[rows]`` (above zero);
    ``top_k_index`` int64 ``[rows, 1]`` = ``top_k - 1``, the column of the k-th largest value; ``top_p`` fp32
    ``[rows]`` in ``(0, 1]``.
    """
    scaled = values / temperature.unsqueeze(dim=1)
    kth = scaled.gather(1, top_k_index)
    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = scaled.softmax(dim=-1)
    # The summed probability of a column and of every smaller one. vLLM drops a token when that sum is at or
    # below 1 - top_p, and never the largest token: column 0 is left out of the cut.
    tail = probs.sum(dim=-1, keepdim=True) - probs.cumsum(dim=-1) + probs
    rest = scaled[:, 1:].masked_fill(tail[:, 1:] <= (1 - top_p).unsqueeze(dim=1), float("-inf"))
    return torch.cat((scaled[:, :1], rest), dim=1).softmax(dim=-1)


def sample_candidates(values: torch.Tensor,
                      ids: torch.Tensor,
                      params: torch.Tensor,
                      top_k_index: torch.Tensor,
                      greedy_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The sampled token ids, int64 ``[rows, 1]``.

    ``values`` / ``ids`` from ``take_candidates``; ``params`` / ``top_k_index`` from ``candidate_parameters``.
    ``greedy_ids`` (int64 ``[rows, 1]``, the arg max over the whole row) is given for a batch that has greedy
    rows: those rows return it, as vLLM's sampler does for a mixed batch.
    """
    probs = candidate_probs(values, params[:, 0], top_k_index, params[:, 1])
    choice = (probs / params[:, PARAM_COLUMNS:]).argmax(dim=-1, keepdim=True)
    tokens = ids.gather(1, choice)
    if greedy_ids is not None:
        tokens = torch.where(params[:, 2:3] != 0, greedy_ids, tokens)
    return tokens


def candidate_parameters(temperatures: Sequence[float],
                         top_ks: Sequence[int],
                         top_ps: Sequence[float],
                         rows: int,
                         candidates: int = CANDIDATES,
                         generator: Optional[torch.Generator] = None,
                         generators: Optional[dict[int, torch.Generator]] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The host side of one step: ``(params fp32 [rows, 3 + C], top_k_index int64 [rows, 1])`` on the CPU.

    Row ``r`` of ``params`` is temperature, top_p, greedy flag, then the row's exponential noise. A request with
    a temperature below ``GREEDY_TEMPERATURE`` is greedy: flag 1, and harmless values elsewhere. Rows past the
    requests (the padding of the batch bucket) are harmless too. ``generators[r]`` draws row ``r``'s noise (a
    seeded request); the other rows draw from ``generator`` (the process's default generator when ``None``).
    """
    count = len(temperatures)
    if not (count == len(top_ks) == len(top_ps)) or count > rows:
        raise ValueError(f"{count} temperatures, {len(top_ks)} top_k, {len(top_ps)} top_p for {rows} rows")
    params = torch.empty((rows, PARAM_COLUMNS + candidates), dtype=torch.float32)
    params[:, 0] = 1.0
    params[:, 1] = 1.0
    params[:, 2] = 0.0
    top_k_index = torch.zeros((rows, 1), dtype=torch.int64)
    for row in range(count):
        if temperatures[row] < GREEDY_TEMPERATURE:
            params[row, 2] = 1.0
            continue
        if not 0 < top_ks[row] <= candidates:
            raise ValueError(f"row {row}: top_k {top_ks[row]} is outside 1..{candidates}")
        params[row, 0] = float(temperatures[row])
        params[row, 1] = float(top_ps[row])
        top_k_index[row, 0] = int(top_ks[row]) - 1
    noise = params[:, PARAM_COLUMNS:]
    noise.exponential_(generator=generator)
    for row, own in (generators or {}).items():
        noise[row].exponential_(generator=own)
    return params, top_k_index


def candidate_refusal(*,
                      greedy: bool,
                      top_k: int,
                      logprobs,
                      logprob_token_ids,
                      logit_bias,
                      allowed_token_ids,
                      bad_words_token_ids,
                      presence_penalty: float,
                      frequency_penalty: float,
                      repetition_penalty: float,
                      min_p: float,
                      thinking_token_budget,
                      extra_args,
                      trace_decode_token_ids,
                      stop_ids: int,
                      candidates: int = CANDIDATES) -> Optional[str]:
    """``None`` when a request may sample over candidates, else why it keeps the general sampler path.

    ``stop_ids`` is the number of stop ids of a request with ``min_tokens`` (0 without ``min_tokens``): the
    logits region masks up to ``STOP_ID_LIMIT`` of them before the candidates are taken.
    """
    if not greedy and not 0 < top_k <= candidates:
        return f"top_k {top_k} is outside 1..{candidates} (the candidates would not hold the kept tokens)"
    if logprobs is not None or logprob_token_ids:
        return "logprobs need the whole distribution"
    if logit_bias or allowed_token_ids or bad_words_token_ids:
        return "logit_bias, allowed_token_ids and bad_words change logits outside the candidates"
    if presence_penalty != 0.0 or frequency_penalty != 0.0 or repetition_penalty != 1.0:
        return "penalties change logits outside the candidates"
    if min_p:
        return "min_p is not applied on the candidates"
    if thinking_token_budget is not None or extra_args or trace_decode_token_ids is not None:
        return "a thinking budget, extra_args or a decode trace use the general sampler"
    if stop_ids > STOP_ID_LIMIT:
        return f"min_tokens with {stop_ids} stop ids (the logits region masks {STOP_ID_LIMIT})"
    return None


__all__ = [
    "CANDIDATES",
    "GREEDY_TEMPERATURE",
    "PARAM_COLUMNS",
    "candidate_parameters",
    "candidate_probs",
    "candidate_refusal",
    "sample_candidates",
    "take_candidates",
]
