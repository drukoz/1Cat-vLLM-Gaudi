# SPDX-License-Identifier: Apache-2.0
"""Batched prefill across requests for Flash-Next (``VLLM_HPU_PREFILL_BATCH``), the rules of v1.

Today every prompt chunk of a hybrid model is its own forward (``HPUModelRunner._can_merge_prefill_contents``
refuses any merge for a model with recurrent layers), so a step that carries the prompt chunks of N requests runs
N forwards one after the other, each with the full per-forward cost (collectives, launches, per-layer fixed work).
With ``VLLM_HPU_PREFILL_BATCH=N`` (N >= 2) the runner puts up to N chunks of one step into one forward when they
have the same padded shape:

* the same query bucket and the same context bucket, so every row is padded exactly as its own one-row forward
  would be (a merged forward does the padded work of the forwards it replaces, never more);
* the same logits class: all rows complete their prompt (they are sampled, and the MTP drafter proposes from
  them) or none does;
* no row reads a block another row of the batch writes in this step (a shared prefix the scheduler gave to a
  later request of the same step: in one forward its recurrent state would be read before it is written);
* no request asks for prompt logprobs (the teacher-forced logprob probes keep their one-row forwards);
* the batch size is a prompt bs bucket (no padded rows) and the batch stays inside the prompt HPU-graph limit
  when its one-row forwards are inside it (``bs x (query + context)`` tokens, as ``_use_graphs`` counts).

Only consecutive requests of the input batch are merged: the drafter and the asynchronous scheduling paths pair a
prompt batch with a contiguous run of input-batch rows. The model side already takes ``[bs, T_bucket]`` prompt
batches of padded rows (the GDN chunk rule's row path, the prompt conv's per-row lengths, the QSA indexer and
selection-bias attention per row, the PLE short conv per row). Pure functions here; the runner wires them in.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import vllm_gaudi.envs as gaudi_envs

# Model types whose prompt path is known to take several padded rows (the Flash-Next stack). Other hybrids keep
# the one-request prompt forwards: their mixers (Mamba2's checkpoint maps, Granite's conv) assume one row.
PREFILL_BATCH_MODEL_TYPES = ("qwen4_exp", )


def resolve_prefill_batch_rows() -> int:
    """``VLLM_HPU_PREFILL_BATCH``: 0 (default) off; N >= 2 merges up to N prompt chunks of one step per forward."""
    value = int(gaudi_envs.VLLM_HPU_PREFILL_BATCH)
    if value < 0:
        raise ValueError(f"VLLM_HPU_PREFILL_BATCH must be >= 0, got {value}")
    if value == 1:
        return 0  # one row per forward is today's behaviour
    return value


def prefill_checkpoint_on() -> bool:
    """prefill's ``VLLM_HPU_PREFILL_CHECKPOINT`` (one forward per turn) is set: its prompt path reads the Mamba2-style
    checkpoint maps, which a prompt batch of several rows does not get yet (read from the environment: the flag is
    defined on the prefill branch)."""
    return os.environ.get("VLLM_HPU_PREFILL_CHECKPOINT", "0").strip() not in ("", "0")


def prefill_batch_refusal(model_type: str, num_mamba_like_layers: int, num_gdn: int, use_merged_prefill: bool,
                          has_kv_transfer: bool, data_parallel_size: int,
                          prefill_checkpoint: bool = False) -> Optional[str]:
    """Why this server cannot merge prompt chunks (None: it can)."""
    if model_type not in PREFILL_BATCH_MODEL_TYPES:
        return f"model type {model_type!r} is not one of {PREFILL_BATCH_MODEL_TYPES}"
    if num_mamba_like_layers <= 0 or num_gdn <= 0:
        return "the model has no GDN layers"
    if use_merged_prefill:
        return "merged prefill (one flat row) is on"
    if has_kv_transfer:
        return "a KV transfer connector is configured"
    if data_parallel_size > 1:
        return "data parallel size > 1 (dummy prefill padding across ranks)"
    if prefill_checkpoint:
        # its checkpoint blocks are filled from the per-request maps, which only one-row batches get today: a
        # merged row would leave its checkpoint block stale and the next turn would resume from it
        return "VLLM_HPU_PREFILL_CHECKPOINT is on and its checkpoint maps are one-row only"
    return None


@dataclass(frozen=True)
class ChunkShape:
    """What decides whether two prompt chunks can share a forward: their one-row bucket and logits class."""
    query_bucket: int
    context_bucket: int
    has_logits: bool


def group_sizes(count: int, sizes: Iterable[int]) -> list[int]:
    """Split ``count`` equal chunks into forwards whose row counts are bs buckets (largest first, no padded rows).

    ``sizes`` are the available batch sizes (1 must be among them, or is assumed). 7 with {1, 2, 4, 8} gives
    [4, 2, 1]; 8 gives [8]; 12 with {1, 2, 4} gives [4, 4, 4].
    """
    available = sorted({int(s) for s in sizes if int(s) >= 1} | {1}, reverse=True)
    out = []
    left = int(count)
    while left > 0:
        size = next(s for s in available if s <= left)
        out.append(size)
        left -= size
    return out


def written_blocks(blocks: Sequence[int], context_len: int, block_size: int) -> list[int]:
    """The blocks a chunk writes: from the block holding its first token (position ``context_len``) on."""
    return list(blocks[context_len // block_size:])


def read_blocks(blocks: Sequence[int], context_len: int, block_size: int) -> list[int]:
    """The blocks a chunk reads as its context: every block before its first token's block, and that block too
    when the context ends inside it."""
    return list(blocks[:math.ceil(context_len / block_size)])


def reads_what_another_writes(rows: Sequence[tuple[Sequence[int], int]], block_size: int) -> bool:
    """True when a row reads a block that another row of the same forward writes.

    ``rows`` are ``(blocks, context_len)`` of the attention group per request, as ``_extract_prefill_batch_contents``
    has them. A request whose prefix hit covers blocks an earlier request of the same step computes would, in one
    forward, read that request's recurrent state slot before the forward writes it (one-row forwards run in order).
    Real blocks only: ids below 1 are vLLM's null block or padding.
    """
    writes = [set(b for b in written_blocks(blocks, ctx, block_size) if b > 0) for blocks, ctx in rows]
    for i, (blocks, ctx) in enumerate(rows):
        reads = set(b for b in read_blocks(blocks, ctx, block_size) if b > 0)
        if not reads:
            continue
        for j, other in enumerate(writes):
            if j != i and reads & other:
                return True
    return False


def within_graph_limit(rows: int, query_bucket: int, context_bucket_tokens: int, limit: int) -> bool:
    """``_use_graphs``'s test for a prompt forward of ``rows`` padded rows: ``rows x (query + context) <= limit``."""
    return rows * (query_bucket + context_bucket_tokens) <= limit
