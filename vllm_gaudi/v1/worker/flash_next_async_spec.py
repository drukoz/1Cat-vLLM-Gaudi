# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of Flash-Next MTP under async scheduling (track async-sched): a verification step's inputs built on the
device from the previous verification's acceptance.

Under async scheduling the host prepares step N+1 before step N's tokens reach it. A verification step's inputs
that depend on step N's acceptance (the accepted drafts ``a`` of each request) are then made on the device from
step N's own device inputs and from the acceptance that mtp-draft-graph's chain leaves on the device
(``runner._mtp_chain_out``; a stand-in built from the host's read when the chain is off):

* input ids: ``next_ids`` = [the bonus token of N, the k drafts for N+1];
* positions: ``positions_N[r, 0] + a_r + 1 + j``;
* the KV slot mapping: ``block_table[r, pos // bs] * bs + pos % bs`` (the one-row-per-request block table of the
  sparse decode metadata, ``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1``);
* the GDN lattice's ``num_accepted_tokens``: ``a_r + 1``;
* the PLE n-gram context: ``advance_ple_context(context_N, ids_N, a + 1)``.

Padded rows get the host's padding values (positions and ids 0, the pad block's slots in row-major order,
``num_accepted_tokens`` 1, an all-eos context). Stage M1 (``VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS``): ``1`` makes
them and compares them with the host-made ones every step, ``2`` uses them. Only a verification step whose decode
rows are the previous step's (same requests, same order, same padded batch and k) is advanced; any other step keeps
the host's inputs. Tensor ops only (int32, ``index_select``), no host read outside the check.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from vllm_gaudi.v1.worker.flash_next_inputs import advance_ple_context

TENSORS = ("token_ids", "positions", "slot_mapping", "num_accepted", "ngram_context")


@dataclass
class ChainOut:
    """The acceptance of a verification step that the next one reads (mtp-draft-graph's ``runner._mtp_chain_out``
    has these fields): int32 device tensors over the step's padded decode batch, in input-batch row order."""
    accepted: torch.Tensor  # [B] accepted drafts per row, 0..k (0 on padded rows)
    next_ids: torch.Tensor  # [B, 1 + k] the bonus token of the step, then the k drafts for the next one
    num_decodes: int
    k: int
    sampled: Optional[torch.Tensor] = None  # [B, 1 + k] the step's tokens, -1 after the bonus
    last_row: Optional[torch.Tensor] = None
    stand_in: bool = False  # made from the host's read (stand_in_chain), not by the chain


def use_scheduled_drafts(chain: ChainOut, req_ids: Sequence[str], scheduled_spec_tokens: dict) -> bool:
    """The draft columns of ``next_ids`` from the drafts the scheduler scheduled for this step, when it has them: a
    synchronous engine verifies those (the same values as the drafter's, or e.g. the engine rehearsal's oracle
    drafts). False, and nothing changed, when a row has none or has the async scheduler's -1 placeholders: then the
    device drafts stay (the asynchronous engine). The chain's own tensor is not written: a copy is."""
    rows = [scheduled_spec_tokens.get(req_id) for req_id in req_ids[:chain.num_decodes]]
    if not rows or any(r is None or len(r) != chain.k or min(r) < 0 for r in rows):
        return False
    if not chain.stand_in:
        chain.next_ids = chain.next_ids.clone()
    chain.next_ids[:chain.num_decodes, 1:] = torch.tensor(rows, dtype=torch.int32).to(chain.next_ids.device)
    return True


@dataclass
class VerifyInputs:
    """A verification step's acceptance-dependent inputs, in the runner's layouts."""
    token_ids: torch.Tensor  # [B * (1 + k), 1] int32
    positions: torch.Tensor  # [B * (1 + k), 1]
    slot_mapping: torch.Tensor  # [B * (1 + k), 1]
    num_accepted: torch.Tensor  # [B] int32
    ngram_context: Optional[torch.Tensor] = None  # [B, c] int32


@dataclass
class _Recorded:
    step: int
    req_ids: tuple
    num_decodes: int
    padded: int
    k: int
    inputs: VerifyInputs


@dataclass
class AdvanceStats:
    verify_steps: int = 0
    advanced: int = 0
    checked: int = 0
    mismatched_steps: int = 0
    mismatches: dict = field(default_factory=lambda: {name: 0 for name in TENSORS})


def stand_in_chain(accepted_tokens: Sequence[Sequence[int]], drafts: torch.Tensor, padded: int,
                   k: int) -> ChainOut:
    """The chain's output from the host's read of a verification step: ``accepted_tokens[r]`` is request r's
    committed tokens (its accepted drafts, then the bonus token), ``drafts`` the ``[rows >= num_decodes, k]`` device
    drafts for the next step in input-batch order."""
    num_decodes = len(accepted_tokens)
    accepted = torch.zeros(padded, dtype=torch.int32)
    bonus = torch.zeros(padded, dtype=torch.int32)
    for r, tokens in enumerate(accepted_tokens):
        accepted[r] = max(len(tokens) - 1, 0)
        bonus[r] = int(tokens[-1]) if len(tokens) else 0
    device = drafts.device
    next_ids = torch.zeros((padded, 1 + k), dtype=torch.int32, device=device)
    next_ids[:, 0] = bonus.to(device)
    next_ids[:num_decodes, 1:] = drafts[:num_decodes].to(torch.int32)
    return ChainOut(accepted=accepted.to(device), next_ids=next_ids, num_decodes=num_decodes, k=k, stand_in=True)


class VerifyInputAdvance:
    """Keeps a verification step's device inputs and its acceptance, and makes the next step's from them."""

    def __init__(self) -> None:
        self._recorded: Optional[_Recorded] = None
        self._chain: Optional[ChainOut] = None
        self._chain_step = -1
        self.stats = AdvanceStats()

    def reset(self) -> None:
        self._recorded = None
        self._chain = None

    def record(self, step: int, req_ids: Sequence[str], num_decodes: int, padded: int, k: int,
               inputs: VerifyInputs) -> None:
        """This verification step's inputs as the model gets them (the next step advances from them)."""
        self._recorded = _Recorded(step, tuple(req_ids[:num_decodes]), num_decodes, padded, k, inputs)

    def set_chain(self, step: int, chain: Optional[ChainOut]) -> None:
        """This verification step's acceptance (None: the step left none)."""
        self._chain, self._chain_step = chain, step

    def pending_chain(self, step: int) -> Optional[ChainOut]:
        """The acceptance the previous step left, if it is step ``step - 1``'s."""
        return self._chain if self._chain_step == step - 1 else None

    def advance(self, step: int, req_ids: Sequence[str], num_decodes: int, padded: int, k: int,
                block_table: torch.Tensor, block_size: int, pad_slot_base: int, eos_token_id: int,
                with_context: bool, table_row_stride: int = 1) -> Optional[VerifyInputs]:
        """Step ``step``'s inputs from step ``step - 1``'s, or None when this step does not continue that one.
        ``block_table`` is the int32 kernel block table on the device whose row ``r * table_row_stride`` is request
        ``r``'s (the sparse decode metadata's table repeats each request's row over its ``1 + k`` verification rows:
        stride ``1 + k``)."""
        self.stats.verify_steps += 1
        rec, chain = self._recorded, self._chain
        if rec is None or chain is None or rec.step != step - 1 or self._chain_step != step - 1:
            return None
        if (num_decodes != rec.num_decodes or padded != rec.padded or k != rec.k or chain.k != k
                or chain.num_decodes != num_decodes or tuple(req_ids[:num_decodes]) != rec.req_ids):
            return None
        if with_context and rec.inputs.ngram_context is None:
            return None
        width = 1 + k
        prev = rec.inputs
        device = prev.positions.device
        rows = torch.arange(padded, dtype=torch.int32, device=device)
        real = rows < num_decodes  # [B]
        cols = torch.arange(width, dtype=torch.int32, device=device)
        accepted = chain.accepted[:padded].to(torch.int32)

        base = prev.positions.reshape(padded, width)[:, 0].to(torch.int32) + accepted + 1  # [B]
        positions = torch.where(real[:, None], base[:, None] + cols[None, :], torch.zeros_like(base)[:, None])

        table = block_table.to(torch.int32)
        table_width = table.shape[1]
        flat_index = (rows * table_row_stride)[:, None] * table_width + torch.div(positions, block_size,
                                                                                  rounding_mode="floor")
        # padded rows read entry 0 (their slots are the pad block's below; the table may have fewer rows)
        flat_index = torch.where(real[:, None], flat_index, torch.zeros_like(flat_index))
        blocks = table.reshape(-1).index_select(0, flat_index.reshape(-1)).reshape(padded, width)
        real_slots = blocks * block_size + torch.remainder(positions, block_size)
        pad_index = (rows - num_decodes)[:, None] * width + cols[None, :]
        pad_slots = pad_slot_base + torch.remainder(pad_index, block_size)
        slots = torch.where(real[:, None], real_slots, pad_slots)

        ids = torch.where(real[:, None], chain.next_ids[:padded].to(torch.int32),
                          torch.zeros((padded, width), dtype=torch.int32, device=device))
        num_accepted = torch.where(real, accepted + 1, torch.ones_like(accepted))

        context = None
        if with_context:
            moved = advance_ple_context(prev.ngram_context.to(torch.int32), prev.token_ids.reshape(padded, width),
                                        accepted + 1)
            context = torch.where(real[:, None], moved, torch.full_like(moved, int(eos_token_id)))

        self.stats.advanced += 1
        return VerifyInputs(token_ids=ids.reshape(-1, 1),
                            positions=positions.reshape(-1, 1).to(prev.positions.dtype),
                            slot_mapping=slots.reshape(-1, 1),
                            num_accepted=num_accepted,
                            ngram_context=context)

    def check(self, made: VerifyInputs, host: VerifyInputs) -> dict:
        """Compare the device-made inputs with the host-made ones (a host read: the check mode only). Returns
        ``{name: mismatching entries}`` for the tensors that differ."""
        self.stats.checked += 1
        bad = {}
        for name in TENSORS:
            mine, theirs = getattr(made, name), getattr(host, name)
            if mine is None or theirs is None:
                continue
            a = mine.detach().to("cpu").reshape(-1).to(torch.int64)
            b = theirs.detach().to("cpu").reshape(-1).to(torch.int64)
            n = int((a != b).sum()) if a.shape == b.shape else max(a.numel(), b.numel())
            if n:
                bad[name] = n
                self.stats.mismatches[name] += 1
        if bad:
            self.stats.mismatched_steps += 1
        return bad


# ---- M2 (VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3): a chained step issued before the previous one's tokens reach the host ----


def trim_verified(row: Sequence[int], vocab_size: int) -> list:
    """A verification step's row of tokens as the rejection sampler's ``parse_output`` keeps them: every entry that
    is not the placeholder -1 and is below ``vocab_size``."""
    return [int(t) for t in row if t != -1 and t < vocab_size]


@dataclass
class PendingBooks:
    """The host's books of a step issued before its tokens reached the host, applied later in step order."""
    step: int
    req_ids: tuple  # the decode rows, in input-batch order
    cpu: torch.Tensor  # [B, 1 + k] the step's tokens, copied to the host asynchronously
    event: object  # the copy's event
    seeds: dict  # request -> its true num_computed_tokens at the step's start, when the host knew it (else absent)


@dataclass
class EarlyStats:
    issued: int = 0  # steps issued before the previous step's tokens reached the host
    settled_late: int = 0  # books applied without waiting
    waited: int = 0  # books applied after waiting for the device
    sync_steps: int = 0  # chained steps that kept the synchronous order
    reasons: dict = field(default_factory=dict)


class SpecEarlyBooks:
    """The queue of pending books and each request's true ``num_computed_tokens`` through the last applied step."""

    MAX_PENDING = 2

    def __init__(self) -> None:
        self.pending: collections.deque = collections.deque()
        self.true_computed: dict = {}
        self.stats = EarlyStats()

    def push(self, books: PendingBooks) -> None:
        for req_id, value in books.seeds.items():
            self.true_computed.setdefault(req_id, value)
        self.pending.append(books)

    def apply(self, books: PendingBooks, vocab_size: int) -> list:
        """The step's token lists per decode row; advances the true counts."""
        rows = books.cpu.tolist()
        tokens = [trim_verified(rows[r], vocab_size) for r in range(len(books.req_ids))]
        for req_id, row in zip(books.req_ids, tokens):
            if req_id in self.true_computed:
                self.true_computed[req_id] += len(row)
        return tokens
