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
  ``logits.float().argmax(-1)`` has on the CPU. The values compared are the head's logits in their own dtype (bf16),
  as the gathered path's all-gather materializes them; only the reduced values are converted to fp32 (exact), so no
  rounding differs from the gathered path. A conversion of the whole shard right after the head's matmul would not
  do: in the matmul's lazy graph the graph compiler fuses the cast into the GEMM, which then hands out its fp32
  accumulator (engine-step's card job 00254: 99.997 % of a shard's logits off their bf16 values, the arg max on 2
  of 100 random rows), and a near-tie that is a tie in bf16 resolves another way (00230's tiny gate and texts).
* :func:`top_k` returns the ``k`` largest logit values exactly. The ids are the tokens that hold them; when
  several tokens tie at the ``k``-th value, which of the tied tokens are returned is not defined (it is not
  defined for ``torch.topk`` over the gathered logits either).

* :func:`top_k` is the head's side of a sampled step that draws from candidates (the sampler stage is
  ``vllm_gaudi/v1/sample/candidate_sampling.py``, the engine-step track's): with a top-k of at most ``k`` on
  every sampled row the tokens top-k keeps are all among the returned ones, unless more than ``k`` of them lie in
  one rank's shard. :func:`first_maximum` picks a greedy row's token from the same candidates.

Device rules: ids travel as fp32 (exact below 2**24) and no id is int64 on the device where this module can
avoid it. Where an op hands out int64 indices (``torch.topk``, the sampler's arg max) they are used as they are,
as the index of a ``torch.gather`` into an fp32 id table: an integer cast of device-made int64 ids in the graph
that made them is card rule 9's failing pattern (``topk`` ids narrowed to int32 came back as every second id).
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
    """``(logits of this rank's real vocabulary rows [rows, n] in the head's dtype, first global id, n)``.

    The head's padding rows (zero weights at the end of the last shard) are cut off: their logit is 0 and
    would beat every negative logit. The logits keep the head's dtype: the reductions run on the values the gathered
    path compares, and only their results are converted (see the module's exactness note: a cast of the whole shard
    here is fused into the matmul on the card and compares its fp32 accumulators).
    """
    head = owner.lm_head
    logits = owner.logits_processor(head, hidden_states, skip_gather=True)
    indices = head.shard_indices
    start = int(indices.org_vocab_start_index)
    count = int(indices.org_vocab_end_index) - start
    if count < logits.shape[-1]:
        logits = logits.narrow(-1, 0, count)
    return logits, start, count


def _lowest_id_of_max(values: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(max over the last dim [rows] in the dtype of ``values``, the lowest of the fp32 ``ids`` that hold it [rows])``.

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
        value, token = _lowest_id_of_max(logits, _id_table(start, count, rows, logits.device))
        value = value.to(torch.float32)  # the shard's largest logit as the head made it (exact in fp32)
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


def _id_table(start: int, count: int, rows: int, device) -> torch.Tensor:
    """The global ids ``start .. start + count - 1`` as fp32 ``[rows, count]`` (a broadcast, no copy)."""
    return torch.arange(start, start + count, dtype=torch.float32, device=device).expand(rows, count)


def vocab_parallel_candidates(model, hidden_states: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank's ``k`` largest logits of every row: ``(values fp32, global ids fp32)``, both ``[rows, world * k]``.

    Each rank: ``torch.topk`` of its shard, the ids read out of an fp32 id table with the indices as ``topk``
    made them. One all-gather of ``[rows, 2k]`` fp32 per rank (values, then ids). Rank ``r``'s candidates are the
    columns ``r * k .. (r + 1) * k - 1``, descending. A shard with fewer than ``k`` real rows fills up with
    ``-inf`` values (id 0), which no consumer picks.

    The ``j`` largest logits of the vocabulary are among these for every ``j <= k``; so is every token whose
    logit is at least the ``j``-th largest value, unless more than ``k`` such tokens lie in one rank's shard.
    """
    owner = head_owner(model)
    logits, start, count = _shard_logits(owner, hidden_states)
    rows = logits.shape[0]
    if k < 1 or k > int(owner.lm_head.org_vocab_size):
        raise ValueError(f"k must be in [1, vocabulary size], got {k}")
    local_k = min(k, count)
    if local_k > 0:
        values, index = torch.topk(logits, local_k, dim=-1)
        values = values.to(torch.float32)  # the head's values, converted after the selection (exact)
        ids = torch.gather(_id_table(start, count, rows, logits.device), -1, index)
    else:
        values = torch.empty((rows, 0), dtype=torch.float32, device=logits.device)
        ids = torch.empty((rows, 0), dtype=torch.float32, device=logits.device)
    if local_k < k:  # a shard with fewer real rows than k: pad with entries that never win
        pad = (rows, k - local_k)
        values = torch.cat((values, torch.full(pad, float("-inf"), dtype=torch.float32, device=logits.device)), -1)
        ids = torch.cat((ids, torch.zeros(pad, dtype=torch.float32, device=logits.device)), -1)
    world = get_tensor_model_parallel_world_size() if int(owner.lm_head.tp_size) > 1 else 1
    if world > 1:
        gathered = tensor_model_parallel_all_gather(torch.cat((values, ids), dim=-1), dim=-1)  # [rows, world * 2k]
        gathered = gathered.reshape(rows, world, 2, k)
        values = gathered.select(2, 0).reshape(rows, world * k)
        ids = gathered.select(2, 1).reshape(rows, world * k)
    return values, ids


def vocab_parallel_topk(model, hidden_states: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` largest logits of every row: ``(values fp32 [rows, k] descending, ids int32 [rows, k])``.

    :func:`vocab_parallel_candidates` (each rank's top ``k``, one all-gather of ``[rows, 2k]`` fp32 per rank),
    then ``torch.topk`` over the ``world * k`` candidates.
    """
    all_values, all_ids = vocab_parallel_candidates(model, hidden_states, k)
    if all_values.shape[-1] == k:  # one rank: its top k is the answer, already descending
        return all_values, all_ids.to(torch.int32)
    values, chosen = torch.topk(all_values, k, dim=-1)
    return values, torch.gather(all_ids, -1, chosen).to(torch.int32)


def first_maximum(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The lowest of the ``ids`` that hold the largest value of each row: ``[rows, 1]`` in the dtype of ``ids``.

    ``values`` ``[rows, C]`` and ``ids`` ``[rows, C]`` are candidates with their global token ids, as
    :func:`top_k` or :func:`vocab_parallel_candidates` return them. The result is a row's greedy token, the one
    :func:`top_tokens` gives, as long as every token that ties at the maximum is a candidate (always, unless more
    than ``C`` tokens of one shard tie there). A row without a maximum (NaN) gets its first candidate.

    fp32 and int32 ids go through the ops of :func:`vocab_parallel_argmax` (ids as fp32, max / where / min, an
    fp32 to int32 cast: exact below 2**24). int64 ids are compared and reduced as they are, with no cast.
    """
    best = values.amax(dim=-1, keepdim=True)
    is_best = values == best
    first = ids.narrow(-1, 0, 1)
    if ids.dtype == torch.int64:  # as an op made them (topk over gathered logits): never cast
        sentinel = torch.full_like(ids, torch.iinfo(torch.int64).max)
        token = torch.where(is_best, ids, sentinel).amin(dim=-1, keepdim=True)
        return torch.where(is_best.any(dim=-1, keepdim=True), token, first)
    as_fp32 = ids.to(torch.float32)
    token = torch.where(is_best, as_fp32, torch.full_like(as_fp32, _NO_TOKEN)).amin(dim=-1, keepdim=True)
    token = torch.where(token < _NO_TOKEN, token, as_fp32.narrow(-1, 0, 1))
    return token.to(ids.dtype)


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
    """The ``k`` largest logits of every row: ``(values fp32 [rows, k] largest first, global ids int32 [rows, k])``.

    The one call of a consumer that samples from candidates. ``VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`` and a model
    :func:`unsupported_reason` accepts: :func:`vocab_parallel_topk` (each rank's top ``k``, ``[rows, 2k]`` fp32
    gathered per rank, no logits all-gather). Otherwise ``compute_logits`` (the all-gather) and ``torch.topk``.
    Either way the ids come out of an fp32 id table and are int32: no int64 id is cast.
    """
    if mode() == VocabParallelMode.CANDIDATES and enabled(model):
        return vocab_parallel_topk(model, hidden_states, k)
    logits = _gathered_logits(model, hidden_states).to(torch.float32)
    values, index = torch.topk(logits, k, dim=-1)
    return values, torch.gather(_id_table(0, logits.shape[-1], logits.shape[0], logits.device), -1, index).to(torch.int32)
