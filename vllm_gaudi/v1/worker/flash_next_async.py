# SPDX-License-Identifier: Apache-2.0
"""What the HPU runner keeps across steps under async scheduling (``VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=1``).

Under async scheduling the engine schedules step N+1 while step N runs, and the worker prepares step N+1 before
step N's sampled tokens reach the host: a decode row's input token comes from the previous step's samples on the
device (``HPUModelRunner._prepare_input_ids``), and the host's copy of the token history lags behind. Two things in
the runner read that history and are made right here:

* :class:`PleContextCarry`: the Flash-Next PLE layer hashes each token with the two tokens before it (the n-gram
  context, ``build_ple_inputs``). On the host the context of a decode row that also decoded in the previous step
  holds that step's input token, which only the device has. The carry keeps the next context of every decode row
  on the device: ``next = (context[1:], this step's input token)``, the same shift as ``advance_ple_context``.
  Rows that did not decode in the previous step (a first decode after the prompt, a request back in the batch)
  keep the host's context, which is complete for them.
* :class:`AsyncOutputLedger`: ``CachedRequestState.output_token_ids`` gets one placeholder (-1) per sampled token
  at the end of a step, as vLLM's GPU runner does, so its length is right at once (min_tokens), and the placeholder
  is replaced by the token when that step's host copy is complete: two steps later at the latest (no wait in
  practice), or before a sampler pass that reads the values (penalties, bad words).

Everything here works on device tensors that already exist and on host lists; it adds no host read of the device
except the ledger's event wait, which never blocks on the step in flight unless a sampler pass needs the values.
Imports only ``torch``, so it is unit-testable without an HPU.
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import torch

from vllm_gaudi.v1.worker.flash_next_inputs import advance_ple_context


@dataclasses.dataclass
class _Carry:
    context: torch.Tensor  # [rows, width] int32 on the device: row i = the next context of rows[i]
    rows: dict[str, int]  # request id -> row of ``context``


class PleContextCarry:
    """The PLE n-gram context of the decode rows, carried on the device from one engine step to the next.

    Call :meth:`begin_step` once per engine step before the step's batches are built (the runner does it at the
    start of ``sample_tokens``), and :meth:`decode_context` with each decode batch. A step without a decode batch
    leaves nothing to carry, so a row is carried only from the step right before.
    """

    def __init__(self) -> None:
        self._previous: Optional[_Carry] = None
        self._current: Optional[_Carry] = None

    def begin_step(self) -> None:
        self._previous, self._current = self._current, None

    def reset(self) -> None:
        self._previous = self._current = None

    @property
    def carried_requests(self) -> tuple[str, ...]:
        """The requests whose context the next decode batch takes from the device (for tests and logs)."""
        source = self._current if self._current is not None else self._previous
        return tuple(source.rows) if source is not None else ()

    def decode_context(self, context: torch.Tensor, req_ids: Sequence[str], input_ids: torch.Tensor,
                       num_tokens_per_req: Optional[Sequence[int]] = None) -> torch.Tensor:
        """This decode batch's n-gram context with the carried rows taken from the device; records the next one.

        ``context`` is the host-built context (``build_ple_inputs``) already on the device, int32
        ``[padded_rows, width]``; ``req_ids`` the requests of rows ``0 .. len(req_ids) - 1``; ``input_ids`` this
        batch's input token ids on the device as the model gets them (``[padded_rows * T, 1]`` or
        ``[padded_rows, T]``, the decode rows' tokens already written from the previous step's samples);
        ``num_tokens_per_req`` the scheduled tokens per request (all ``T`` when ``None``). Returns a new device
        tensor of ``context``'s shape and dtype: rows of requests that decoded in the previous step get the
        context that step left them, every other row keeps ``context``'s.
        """
        rows, width = int(context.shape[0]), int(context.shape[1])
        num_reqs = len(req_ids)
        previous = self._previous
        if previous is not None and width > 0 and num_reqs > 0:
            carried, sources = [], []
            for row, req_id in enumerate(req_ids):
                source = previous.rows.get(req_id)
                if source is not None:
                    carried.append(row)
                    sources.append(source)
            if carried:
                count = len(carried)
                if carried == sources == list(range(count)):
                    # The common case: the same requests in the same rows. One concatenation, no index.
                    context = torch.cat([previous.context[:count], context[count:]], dim=0)
                else:
                    base = int(previous.context.shape[0])
                    index = list(range(base, base + rows))
                    for row, source in zip(carried, sources):
                        index[row] = source
                    index_tensor = torch.tensor(index, dtype=torch.int32).to(context.device, non_blocking=True)
                    context = torch.cat([previous.context, context], dim=0).index_select(0, index_tensor)
        if width > 0 and num_reqs > 0:
            self._current = _Carry(self._next_context(context, input_ids, num_reqs, rows, num_tokens_per_req),
                                   {req_id: row for row, req_id in enumerate(req_ids)})
        return context

    @staticmethod
    def _next_context(context: torch.Tensor, input_ids: torch.Tensor, num_reqs: int, rows: int,
                      num_tokens_per_req: Optional[Sequence[int]]) -> torch.Tensor:
        tokens = input_ids.reshape(rows, -1)
        if tokens.dtype != context.dtype:
            tokens = tokens.to(context.dtype)
        width = int(tokens.shape[1])
        if width == 1:
            return torch.cat([context[:num_reqs, 1:], tokens[:num_reqs]], dim=1)
        # Several input tokens per row (not a plain decode step): every scheduled token is committed.
        committed = list(num_tokens_per_req[:num_reqs]) if num_tokens_per_req is not None else [width] * num_reqs
        committed_tensor = torch.tensor(committed, dtype=torch.int32).to(context.device, non_blocking=True)
        return advance_ple_context(context[:num_reqs], tokens[:num_reqs], committed_tensor)


@dataclasses.dataclass
class _Pending:
    requests: Mapping[str, Any]  # the runner's request states (the same dict object for the runner's life)
    sampled_cpu: Any  # the step's sampled token ids on the host, [rows, 1] (filled when ``event`` is done)
    event: Any  # the copy's ready event (``synchronize()``)
    entries: list[tuple[str, list[int], int, int]]  # (request id, its output_token_ids, placeholder index, row)


class AsyncOutputLedger:
    """One placeholder per sampled token in each request's ``output_token_ids``, settled from the step's host copy.

    The runner calls :meth:`record` at the end of every async step (after it made the step's host copy),
    :meth:`settle` with ``keep=1`` at the start of the next step (every step but the one in flight: those copies
    are complete, so the wait returns at once), and :meth:`settle` with ``keep=0`` before a sampler pass that reads
    the token values. A placeholder is replaced only while its list is still the request's and still holds -1 at
    that index, so a request whose list was replaced (resumed with the scheduler's tokens) or that was freed in
    between is left alone.
    """

    PLACEHOLDER = -1

    def __init__(self) -> None:
        self._pending: collections.deque[_Pending] = collections.deque()

    def __len__(self) -> int:
        return len(self._pending)

    def record(self, requests: Mapping[str, Any], req_ids: Sequence[Optional[str]], invalid_rows, sampled_cpu,
               event) -> None:
        """Append a placeholder for every row of ``req_ids`` not in ``invalid_rows`` (row = index in the step's
        sampled tensor) and keep the step's host copy to settle them from."""
        invalid = set(invalid_rows or ())
        entries = []
        for row, req_id in enumerate(req_ids):
            if req_id is None or row in invalid:
                continue
            state = requests.get(req_id)
            if state is None:
                continue
            out = state.output_token_ids
            entries.append((req_id, out, len(out), row))
            out.append(self.PLACEHOLDER)
        if entries:
            self._pending.append(_Pending(requests, sampled_cpu, event, entries))

    def settle(self, keep: int = 0) -> int:
        """Replace the placeholders of every recorded step but the ``keep`` newest; returns how many were set."""
        settled = 0
        while len(self._pending) > keep:
            pending = self._pending.popleft()
            if pending.event is not None:
                pending.event.synchronize()
            values = _row_tokens(pending.sampled_cpu)
            for req_id, out, index, row in pending.entries:
                state = pending.requests.get(req_id)
                if (state is not None and state.output_token_ids is out and index < len(out)
                        and out[index] == self.PLACEHOLDER and row < len(values)):
                    out[index] = values[row]
                    settled += 1
        return settled

    def clear(self) -> None:
        self._pending.clear()


def _row_tokens(sampled_cpu) -> list[int]:
    """The first token of every row of a host copy (``[rows, 1]``, ``[rows]`` or a list of lists)."""
    if isinstance(sampled_cpu, torch.Tensor):
        flat = sampled_cpu.reshape(sampled_cpu.shape[0], -1)[:, 0] if sampled_cpu.dim() > 1 else sampled_cpu
        return [int(v) for v in flat.tolist()]
    return [int(row[0]) if isinstance(row, (list, tuple)) else int(row) for row in sampled_cpu]


def needs_output_token_values(input_batch) -> bool:
    """Whether a sampler pass over ``input_batch`` reads the output token values (penalties or bad words)."""
    return (not getattr(input_batch, "no_penalties", True)) or bool(getattr(input_batch, "bad_words_token_ids", None))


@dataclasses.dataclass
class _Drafts:
    drafts: Any  # the drafter's output: [rows, k] tensor on the device, or one host list per row
    rows: dict[str, int]  # request id -> row of ``drafts``
    computed: dict[str, tuple[int, int]]  # request id -> (the scheduler's optimistic count, the true count)


class SpecDraftHandoff:
    """Speculative decoding under async scheduling (``VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2``): the drafts stay where
    the drafter left them.

    The engine schedules step N+1 before it has step N's output: step N+1's draft slots come as -1 placeholders
    (vLLM's ``AsyncScheduler``), and its ``num_computed_tokens`` takes every draft of step N as accepted. The runner
    keeps its synchronous speculative step (it reads the verification result on the host before the drafter, as
    without async scheduling), so at the end of step N it knows each request's accepted count, and the drafter's
    output never has to reach the host. :meth:`record` keeps both after the drafter; at the start of step N+1
    :meth:`correct_computed` gives every request the true count where the scheduler's still assumes the rejected
    drafts, and :meth:`scatter_drafts` writes step N's drafts into step N+1's draft slots on the device, before the
    verification metadata reads them from the input ids. What is saved: the host read of the drafts and the wait
    for the draft passes it forces, and the engine's turn between two steps (it runs during the step).
    """

    def __init__(self) -> None:
        self._previous: Optional[_Drafts] = None
        self._current: Optional[_Drafts] = None
        self.corrections = 0  # requests whose count was corrected (tests and logs)
        self.scattered = 0  # draft slots written from the device

    def begin_step(self) -> None:
        self._previous, self._current = self._current, None

    def record(self, drafts, req_ids: Sequence[Optional[str]], num_computed: Sequence[int],
               num_scheduled: Mapping[str, int], scheduled_spec_tokens: Mapping[str, Sequence[int]],
               sampled: Sequence[Sequence[int]]) -> None:
        """After the drafter of a step: ``drafts`` row ``i`` belongs to ``req_ids[i]``, which ran with
        ``num_computed[i]`` computed tokens and ``num_scheduled[req]`` scheduled ones (``len(scheduled_spec_tokens
        [req])`` of them drafts) and committed ``len(sampled[i])`` tokens (the accepted drafts and the bonus)."""
        rows, computed = {}, {}
        for row, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            rows[req_id] = row
            spec = len(scheduled_spec_tokens.get(req_id, ()) or ())
            committed = len(sampled[row]) if row < len(sampled) else 0
            if spec > 0 and committed > 0:
                start = int(num_computed[row])
                optimistic, true = start + int(num_scheduled[req_id]), start + committed
                if optimistic != true:
                    computed[req_id] = (optimistic, true)
        self._current = _Drafts(drafts, rows, computed) if (drafts is not None or computed) else None

    def correct_computed(self, req_ids: Sequence[Optional[str]], num_computed_cpu, requests: Mapping[str, Any],
                         scheduled_spec_tokens: Mapping[str, Sequence[int]]) -> int:
        """At the start of a step: the true ``num_computed_tokens`` (in ``num_computed_cpu[row]`` and the request
        state) of every verification row whose count from the scheduler still takes the last step's rejected
        drafts as accepted. A count that is neither (a preemption, a resumed request) is left as it is."""
        previous = self._previous
        if previous is None or not previous.computed:
            return 0
        done = 0
        for row, req_id in enumerate(req_ids):
            entry = previous.computed.get(req_id) if req_id is not None else None
            if entry is None or not scheduled_spec_tokens.get(req_id):
                continue
            optimistic, true = entry
            if int(num_computed_cpu[row]) == optimistic:
                num_computed_cpu[row] = true
                state = requests.get(req_id)
                if state is not None:
                    state.num_computed_tokens = true
                done += 1
        self.corrections += done
        return done

    def scatter_drafts(self, token_ids: torch.Tensor, req_ids: Sequence[str], num_tokens: int,
                       scheduled_spec_tokens: Mapping[str, Sequence[int]]) -> torch.Tensor:
        """Write the last step's drafts into the draft slots of a verification batch.

        ``token_ids`` is the batch's input ids on the device, ``[padded_rows * num_tokens, 1]`` (row ``i``'s first
        slot is its input token, the next ``len(scheduled_spec_tokens[req_ids[i]])`` are its drafts, -1 from the
        scheduler); returns it with every draft slot of a request the last step drafted for set to that draft.
        """
        previous = self._previous
        if previous is None or previous.drafts is None or num_tokens < 2:
            return token_ids
        drafts = previous.drafts
        width = int(drafts.shape[1]) if isinstance(drafts, torch.Tensor) else max((len(d) for d in drafts), default=0)
        targets, sources, values = [], [], []
        for row, req_id in enumerate(req_ids):
            source = previous.rows.get(req_id)
            count = min(len(scheduled_spec_tokens.get(req_id, ()) or ()), num_tokens - 1, width)
            if source is None or count <= 0:
                continue
            targets.extend(range(row * num_tokens + 1, row * num_tokens + 1 + count))
            if isinstance(drafts, torch.Tensor):
                sources.extend(range(source * width, source * width + count))
            else:
                own = list(drafts[source])[:count]
                values.extend(own + [-1] * (count - len(own)))
        if not targets:
            return token_ids
        target_index = torch.tensor(targets, dtype=torch.int64).to(token_ids.device, non_blocking=True)
        if isinstance(drafts, torch.Tensor):
            source_index = torch.tensor(sources, dtype=torch.int64).to(drafts.device, non_blocking=True)
            # The drafts were made in an earlier graph: narrowing them here is the safe form of card rule 9.
            picked = drafts.reshape(-1).index_select(0, source_index).to(token_ids.dtype)
        else:
            picked = torch.tensor(values, dtype=token_ids.dtype).to(token_ids.device, non_blocking=True)
        self.scattered += len(targets)
        # Out of place (a new tensor of the same shape): no write through a view of the batch's input ids.
        return token_ids.reshape(-1).index_copy(0, target_index, picked).view(token_ids.shape)
