# SPDX-License-Identifier: Apache-2.0
"""min_tokens, logit_bias and min_p on the HPU's general sampler path (``VLLM_HPU_SAMPLER_REQUEST_PARAMS=1``).

vLLM applies these three through logits processors (``MinTokensLogitsProcessor``, ``LogitBiasLogitsProcessor``,
``MinPLogitsProcessor``) whose per-row state the input batch feeds with its adds, removes and moves
(``update_state``). The HPU input batch never does (``HpuInputBatch.refresh_sampling_metadata``), and it samples
prefills and decodes in separate calls whose rows are not the batch's rows, so the processors apply nothing: a
request's ``min_tokens``, ``logit_bias`` and ``min_p`` were silently ignored (engine-step, 2026-11-20).

Here the runner builds the three from each row's own request at every sampling call, in the call's row order,
and applies them to the logits before vLLM's sampler, with fixed shapes and numeric ops only (gather, where,
scatter, scatter_add of host-made int64 indices; card rules 2, 3 and 9):

* ``min_tokens``: until a request has ``min_tokens`` output tokens its stop ids get ``-inf`` (as vLLM's processor;
  ``fused_greedy.mask_stop_tokens``);
* ``logit_bias``: each ``token -> bias`` is added to the row's logit (vLLM adds it to the raw logit);
* ``min_p``: tokens whose probability at the row's temperature is below ``min_p`` times the row's largest one get
  ``-inf``. vLLM applies min_p after the temperature and before top-k / top-p; masking before the temperature is
  the same mask, and the sampler's temperature, top-k and top-p then run as before. Greedy rows are left alone.

One difference from vLLM: its default ``raw_logprobs`` are computed from the logits before any processor; here a
request with logprobs gets them after its own bias and masks.

Pure functions: this module imports only ``torch``.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from vllm_gaudi.v1.sample.fused_greedy import mask_stop_tokens, min_tokens_stop_ids

GREEDY_TEMPERATURE = 1e-5  # vLLM's _SAMPLING_EPS


def _bucket(count: int) -> int:
    """The next power of two at or above ``count`` (at least 1): few shapes per server."""
    width = 1
    while width < count:
        width *= 2
    return width


@dataclass
class RowParams:
    """What one sampling row needs: its request's parameters and output length."""
    min_tokens: int = 0
    num_output_tokens: int = 0
    stop_token_ids: Sequence[int] = ()
    logit_bias: Optional[dict] = None
    min_p: float = 0.0
    temperature: float = 1.0


@dataclass
class RequestAdjustments:
    """Host tensors of one sampling call; ``None`` fields have nothing to apply."""
    stops: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # int64 [rows, W], int32 [rows]
    bias: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # int64 [rows, B], fp32 [rows, B]
    min_p: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # fp32 [rows], fp32 [rows] (1 / temperature)

    def __bool__(self) -> bool:
        return self.stops is not None or self.bias is not None or self.min_p is not None

    def to(self, device) -> "RequestAdjustments":
        move = (lambda pair: None if pair is None else tuple(t.to(device, non_blocking=True) for t in pair))
        return RequestAdjustments(move(self.stops), move(self.bias), move(self.min_p))


def request_adjustments(rows: Sequence[RowParams]) -> RequestAdjustments:
    """The host tensors for ``rows`` (one entry per logits row, in row order)."""
    count = len(rows)
    stops = [list(r.stop_token_ids) if r.min_tokens and r.num_output_tokens < r.min_tokens and r.stop_token_ids
             else None for r in rows]
    adjust = RequestAdjustments()
    widest = max((len(s) for s in stops if s), default=0)
    if widest:
        adjust.stops = min_tokens_stop_ids(stops, count, width=_bucket(widest))
    biased = [dict(r.logit_bias) if r.logit_bias else {} for r in rows]
    entries = max((len(b) for b in biased), default=0)
    if entries:
        width = _bucket(entries)
        ids = torch.zeros((count, width), dtype=torch.int64)
        values = torch.zeros((count, width), dtype=torch.float32)
        for row, bias in enumerate(biased):
            for column, (token, value) in enumerate(sorted(bias.items())):
                ids[row, column] = int(token)
                values[row, column] = float(value)
        adjust.bias = (ids, values)
    if any(r.min_p and r.temperature >= GREEDY_TEMPERATURE for r in rows):
        min_p = torch.tensor([float(r.min_p) if r.temperature >= GREEDY_TEMPERATURE else 0.0 for r in rows],
                             dtype=torch.float32)
        inverse = torch.tensor([1.0 / r.temperature if r.temperature >= GREEDY_TEMPERATURE else 1.0 for r in rows],
                               dtype=torch.float32)
        adjust.min_p = (min_p, inverse)
    return adjust


def apply_request_adjustments(logits: torch.Tensor, adjust: RequestAdjustments) -> torch.Tensor:
    """``logits`` ``[rows, vocab]`` with the call's bias, ``min_tokens`` mask and ``min_p`` cut (fp32 out)."""
    logits = logits.to(torch.float32)
    if adjust.bias is not None:
        ids, values = adjust.bias
        logits = logits.scatter_add(1, ids, values)  # padding entries add 0 to column 0
    if adjust.stops is not None:
        logits = mask_stop_tokens(logits, *adjust.stops)
    if adjust.min_p is not None:
        min_p, inverse = adjust.min_p
        probs = (logits * inverse.view(-1, 1)).softmax(dim=-1)
        threshold = probs.amax(dim=-1, keepdim=True) * min_p.view(-1, 1)
        logits = torch.where(probs < threshold, torch.full_like(logits, float("-inf")), logits)
    return logits


__all__ = ["RequestAdjustments", "RowParams", "apply_request_adjustments", "request_adjustments"]
