# SPDX-License-Identifier: Apache-2.0
"""Token choices from a vocabulary-sharded LM head without gathering the logits.

Under tensor parallelism the LM head is sharded over the vocabulary: rank ``r`` computes the logits of its
own shard, and ``compute_logits`` all-gathers the shards so that every rank holds the full ``[rows, vocab]``
tensor (497 KB per row for Flash-Next in bf16, on each of the 8 ranks). A consumer that only needs the arg
max, or the ``k`` largest logits, does not need that tensor: the largest element of a set is the largest of
the largest elements of its parts, and each of the ``k`` largest elements of the vocabulary is among the
``k`` largest of its own shard. So every rank reduces its shard first and only the reduced rows are gathered:
``[rows, 2]`` fp32 per rank for the arg max, ``[rows, 2k]`` for the top ``k``.

``VLLM_HPU_VOCAB_PARALLEL_LOGITS`` (default 0) selects it; with 0, and for a model these functions do not
take, :func:`top_tokens` and :func:`top_k` gather the full logits exactly as the callers did before.

Exactness (``docs/design/flash-next/collectives.md`` has the argument and the tests that check it):

* :func:`top_tokens` returns, for every row, the lowest token id among the largest logits: the value
  ``logits.float().argmax(-1)`` has on the CPU. The values compared are the bf16 logits converted to fp32
  (exact), so no rounding differs from the gathered path.
* :func:`top_k` returns the ``k`` largest logit values exactly. The ids are the tokens that hold them; when
  several tokens tie at the ``k``-th value, which of the tied tokens are returned is not defined (it is not
  defined for ``torch.topk`` over the gathered logits either).

Device rules: ids travel as fp32 (exact below 2**24), are never int64 on the device, and no op here indexes
with a tensor.
"""

from __future__ import annotations

import torch
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather

from vllm_gaudi import envs

# Ids are carried in fp32 lanes next to the values: one gather instead of two.
MAX_EXACT_FP32_ID = 1 << 24
_NO_TOKEN = float(MAX_EXACT_FP32_ID)


class VocabParallelMode:
    """Values of ``VLLM_HPU_VOCAB_PARALLEL_LOGITS``."""
    OFF = 0  # every consumer gets the all-gathered logits
    GREEDY = 1  # arg-max consumers (plain greedy, MTP drafts, greedy verification) gather [rows, 2] per rank
    CANDIDATES = 2  # GREEDY, and top-k sampling gathers the per-rank top candidates


def mode() -> int:
    """The flag as one of :class:`VocabParallelMode`; any other value is OFF."""
    value = int(envs.VLLM_HPU_VOCAB_PARALLEL_LOGITS)
    return value if value in (VocabParallelMode.GREEDY, VocabParallelMode.CANDIDATES) else VocabParallelMode.OFF


def head_owner(model):
    """The module that owns ``lm_head`` and ``logits_processor``: ``model`` or its ``language_model``/``model``."""
    seen = 0
    while model is not None and seen < 4:
        if hasattr(model, "lm_head") and hasattr(model, "logits_processor"):
            return model
        model = getattr(model, "language_model", None) or getattr(model, "model", None)
        seen += 1
    return None


def unsupported_reason(model) -> str | None:
    """Why the vocabulary-parallel reduction cannot serve ``model``, or ``None``."""
    owner = head_owner(model)
    if owner is None:
        return "the model has no lm_head / logits_processor pair"
    head, processor = owner.lm_head, owner.logits_processor
    indices = getattr(head, "shard_indices", None)
    if indices is None or not hasattr(head, "tp_size"):
        return f"lm_head is {type(head).__name__}, not a vocabulary-parallel head"
    if getattr(processor, "logits_as_input", False):
        return "the logits processor takes logits as its input"
    scale = float(getattr(processor, "scale", 1.0))
    if scale <= 0.0:
        return f"the logit scale {scale} is not positive (the arg max would not be the largest raw logit)"
    if indices.added_vocab_end_index > indices.added_vocab_start_index or int(head.num_added_embeddings) != 0:
        return "the head has added (LoRA) vocabulary rows"
    if int(head.org_vocab_size) > MAX_EXACT_FP32_ID:
        return f"the vocabulary ({head.org_vocab_size}) does not fit fp32 ids"
    return None


def _shard_logits(owner, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``(fp32 logits of this rank's real vocabulary rows [rows, n], first global id, n)``.

    The head's padding rows (zero weights at the end of the last shard) are cut off: their logit is 0 and
    would beat every negative logit.
    """
    head = owner.lm_head
    logits = owner.logits_processor(head, hidden_states, skip_gather=True)
    indices = head.shard_indices
    start = int(indices.org_vocab_start_index)
    count = int(indices.org_vocab_end_index) - start
    if count < logits.shape[-1]:
        logits = logits.narrow(-1, 0, count)
    return logits.to(torch.float32), start, count


def _lowest_id_of_max(values: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(max over the last dim [rows], the lowest of the ``ids`` that hold it [rows])``, both fp32.

    Written with max / where / min so that the choice among equal values does not depend on how the
    device's arg max breaks ties. A row whose values are all NaN gets ``_NO_TOKEN``; the caller clamps it.
    """
    best = values.amax(dim=-1, keepdim=True)
    candidates = torch.where(values == best, ids, torch.full_like(values, _NO_TOKEN))
    return best.squeeze(-1), candidates.amin(dim=-1)


def vocab_parallel_argmax(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """The greedy token of every row of ``hidden_states`` ``[rows, hidden]``, as int32 ``[rows]``.

    Each rank: the largest logit of its vocabulary shard and the lowest global id that holds it. One
    all-gather of ``[rows, 2]`` fp32 per rank, then the same reduction over the ranks.
    """
    owner = head_owner(model)
    logits, start, count = _shard_logits(owner, hidden_states)
    rows = logits.shape[0]
    if count > 0:
        ids = torch.arange(start, start + count, dtype=torch.float32, device=logits.device).expand(rows, count)
        value, token = _lowest_id_of_max(logits, ids)
    else:  # a shard with no real row: it never wins
        value = torch.full((rows, ), float("-inf"), dtype=torch.float32, device=logits.device)
        token = torch.full((rows, ), _NO_TOKEN, dtype=torch.float32, device=logits.device)
    world = get_tensor_model_parallel_world_size() if int(owner.lm_head.tp_size) > 1 else 1
    if world > 1:
        pairs = tensor_model_parallel_all_gather(torch.stack((value, token), dim=-1), dim=-1)  # [rows, 2 * world]
        pairs = pairs.reshape(rows, world, 2)
        _, token = _lowest_id_of_max(pairs.select(-1, 0), pairs.select(-1, 1))
    last = float(int(owner.lm_head.org_vocab_size) - 1)
    return token.clamp(min=0.0, max=last).to(torch.int32)


def vocab_parallel_topk(model, hidden_states: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest logits of every row: ``(values fp32 [rows, k] descending, ids int32 [rows, k])``.

    Each rank: ``torch.topk`` of its shard. One all-gather of ``[rows, 2k]`` fp32 per rank (values, then
    ids), then ``torch.topk`` over the ``world * k`` candidates.
    """
    owner = head_owner(model)
    logits, start, count = _shard_logits(owner, hidden_states)
    rows = logits.shape[0]
    if k < 1 or k > int(owner.lm_head.org_vocab_size):
        raise ValueError(f"k must be in [1, vocabulary size], got {k}")
    local_k = min(k, count)
    if local_k > 0:
        values, index = torch.topk(logits, local_k, dim=-1)
        ids = index.to(torch.float32) + float(start)
    else:
        values = logits.new_empty((rows, 0))
        ids = logits.new_empty((rows, 0))
    if local_k < k:  # a shard with fewer real rows than k: pad with entries that never win
        pad = (rows, k - local_k)
        values = torch.cat((values, torch.full(pad, float("-inf"), dtype=torch.float32, device=logits.device)), -1)
        ids = torch.cat((ids, torch.zeros(pad, dtype=torch.float32, device=logits.device)), -1)
    world = get_tensor_model_parallel_world_size() if int(owner.lm_head.tp_size) > 1 else 1
    if world > 1:
        gathered = tensor_model_parallel_all_gather(torch.cat((values, ids), dim=-1), dim=-1)  # [rows, world * 2k]
        gathered = gathered.reshape(rows, world, 2, k)
        all_values = gathered.select(2, 0).reshape(rows, world * k)
        all_ids = gathered.select(2, 1).reshape(rows, world * k)
        values, chosen = torch.topk(all_values, k, dim=-1)
        ids = torch.gather(all_ids, -1, chosen)
    return values, ids.to(torch.int32)


def _gathered_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    return model.compute_logits(hidden_states)


def enabled(model) -> bool:
    """``VLLM_HPU_VOCAB_PARALLEL_LOGITS`` is 1 or 2 and :func:`unsupported_reason` accepts ``model``."""
    return mode() != VocabParallelMode.OFF and unsupported_reason(model) is None


def top_tokens(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """The greedy token of every row, int32 ``[rows]``: the entry point of the runner's greedy region.

    ``VLLM_HPU_VOCAB_PARALLEL_LOGITS`` 1 or 2 and a model :func:`unsupported_reason` accepts: the
    vocabulary-parallel arg max. Otherwise ``compute_logits`` (the all-gather) and the arg max of its fp32
    conversion, which is what the runner's fused greedy region did before this function existed.
    """
    if enabled(model):
        return vocab_parallel_argmax(model, hidden_states)
    return _gathered_logits(model, hidden_states).to(torch.float32).argmax(dim=-1).to(torch.int32)


def top_k(model, hidden_states: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest logits of every row, ``(values fp32 [rows, k], ids int32 [rows, k])``.

    ``VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`` and a supported model: the vocabulary-parallel top-k. Otherwise
    ``torch.topk`` over the all-gathered logits.
    """
    if mode() == VocabParallelMode.CANDIDATES and enabled(model):
        return vocab_parallel_topk(model, hidden_states, k)
    values, ids = torch.topk(_gathered_logits(model, hidden_states).to(torch.float32), k, dim=-1)
    return values, ids.to(torch.int32)
