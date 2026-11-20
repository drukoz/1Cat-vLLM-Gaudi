# SPDX-License-Identifier: Apache-2.0
"""Prompt lookup beside the Flash-Next MTP drafter: the hybrid proposer (``VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP``).

The agents this server is built for often return a whole edited file: long spans of their output copy text that is
already in the prompt (or earlier in the output). A prompt-lookup draft accepts such spans in long runs, where the
learned MTP drafter decays with depth. With ``VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP=1`` and the MTP method, every step
keeps the MTP drafter's ``k`` drafts and then, per decode request, replaces them by the n-gram continuation when the
request's last ``n`` tokens (``n`` from ``VLLM_HPU_FLASH_NEXT_LOOKUP_MAX_NGRAM`` down to
``VLLM_HPU_FLASH_NEXT_LOOKUP_MIN_NGRAM``) occur earlier in its prompt or output; the continuation after the
earliest occurrence of the longest such n-gram, up to ``k`` tokens, is the draft (vLLM's ``NgramProposer`` search,
the same numba kernel).

Shapes stay the MTP shapes: a lookup draft shorter than ``k`` is padded with ``-1`` (the runner's empty-draft
token: never accepted, its id clamped to 0 in the model), so every verification row is ``1 + k`` wide, the MTP
drafter's ``prepare_inputs`` (one row of ``max drafts + 1`` per request) stays aligned with the target's rows, and
the HPU graphs are the MTP graphs.

The MTP drafter's own state is unchanged by a replaced row: it runs every step for every request (this module
only rewrites the drafts it returned), so its attention KV, compressed keys and key ring at the verified positions
are written by the next step's first draft pass from the target's verification rows, exactly as when its own
drafts were verified (rows past the accepted ones are rewritten at their position later, as after any rejection).

Pure host code, importable on CPU; the runner calls :meth:`HybridLookup.merge` from ``propose_draft_token_ids``.
"""

from __future__ import annotations

import enum
import json
import os
from collections.abc import Sequence

import numpy as np

from vllm_gaudi.extension.logger import logger as init_logger

# The "vllm" logger, as the runner and the platform use it: a logger named after this module is outside vLLM's
# configured hierarchy, and a server then prints nothing below WARNING from it (seen in card job 00021's log,
# which has no INFO line of the spec_decode modules).
logger = init_logger()

PAD_DRAFT = -1  # the runner's empty-draft token (``propose_ngram_draft_token_ids``)
LOG_EVERY_STEPS = 200


class SpecLookupMode(enum.IntEnum):
    OFF = 0  # MTP drafts only (the flag's default: the hook is not installed)
    HYBRID = 1  # per request and step: the lookup continuation when the last n tokens match, else MTP
    LOOKUP_ONLY = 2  # the lookup continuation, else the empty draft: what lookup alone accepts, in MTP's shapes


# Per-request override of the server's mode: ``vllm_xargs={"spec_lookup": 0 | 1 | 2}`` (SamplingParams.extra_args).
REQUEST_KEY = "spec_lookup"


def request_mode(extra_args: dict | None, default: SpecLookupMode) -> SpecLookupMode:
    """The mode of one request: its ``extra_args[REQUEST_KEY]`` when that is a known mode, else ``default``."""
    if extra_args and REQUEST_KEY in extra_args:
        try:
            return SpecLookupMode(int(extra_args[REQUEST_KEY]))
        except (TypeError, ValueError):
            return default
    return default


def lookup_mode() -> SpecLookupMode:
    from vllm_gaudi import envs as gaudi_envs

    try:
        return SpecLookupMode(int(gaudi_envs.VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP))
    except ValueError:
        return SpecLookupMode.OFF


def find_lookup_draft(tokens: np.ndarray, min_ngram: int, max_ngram: int, k: int, max_model_len: int) -> list[int]:
    """The n-gram continuation of ``tokens`` (1-D int32: prompt + output so far): vLLM's prompt-lookup search
    (the longest suffix n-gram of ``min_ngram..max_ngram`` tokens that occurs earlier, at its earliest occurrence),
    up to ``k`` tokens; ``[]`` when nothing matches."""
    from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens

    out = _find_longest_matched_ngram_and_propose_tokens(np.ascontiguousarray(tokens, dtype=np.int32), int(min_ngram),
                                                         int(max_ngram), int(max_model_len), int(k))
    return [int(t) for t in out]


def pad_draft(draft: Sequence[int], k: int) -> list[int]:
    """``draft`` (1..k tokens) padded with ``PAD_DRAFT`` to ``k`` tokens."""
    if not 1 <= len(draft) <= k:
        raise ValueError(f"a lookup draft has 1..{k} tokens, got {len(draft)}")
    return [int(t) for t in draft] + [PAD_DRAFT] * (k - len(draft))


def merge_drafts(mtp_drafts: list[list[int]], lookup_drafts: list[list[int]], k: int, eligible: Sequence[bool],
                 modes: Sequence[SpecLookupMode] | None = None) -> tuple[list[list[int]], list[str]]:
    """Per row, by its mode (``HYBRID`` when ``modes`` is None): ``HYBRID`` takes the lookup draft (padded to ``k``)
    where it is non-empty and the row is ``eligible``, else the MTP draft; ``LOOKUP_ONLY`` the lookup draft, else
    ``k`` empty-draft tokens; ``OFF`` the MTP draft. Returns ``(drafts, sources)`` with sources ``"lookup"``,
    ``"mtp"`` or ``"none"``. ValueError: unequal lengths, an MTP draft that is not ``k`` long."""
    modes = [SpecLookupMode.HYBRID] * len(mtp_drafts) if modes is None else list(modes)
    if not len(mtp_drafts) == len(lookup_drafts) == len(eligible) == len(modes):
        raise ValueError(f"rows differ: {len(mtp_drafts)} MTP, {len(lookup_drafts)} lookup, {len(eligible)} flags, "
                         f"{len(modes)} modes")
    drafts, sources = [], []
    for mtp, lookup, ok, mode in zip(mtp_drafts, lookup_drafts, eligible, modes):
        if len(mtp) != k:
            raise ValueError(f"an MTP draft has {len(mtp)} tokens, expected k = {k}")
        if ok and lookup and mode != SpecLookupMode.OFF:
            drafts.append(pad_draft(lookup[:k], k))
            sources.append("lookup")
        elif ok and mode == SpecLookupMode.LOOKUP_ONLY:
            drafts.append([PAD_DRAFT] * k)
            sources.append("none")
        else:
            drafts.append([int(t) for t in mtp])
            sources.append("mtp")
    return drafts, sources


ORACLE_ENV = "VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE"


class LookupOracle:
    """Card rehearsal hook (``VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE=FILE``): lookup drafts from recorded streams.

    A random tiny model never repeats its context, so on the tiny model neither the n-gram search nor the MTP
    layer gets a draft accepted, and a rehearsal then runs rejections only: the commits of several tokens, the
    widened steps after them and the block-completion commits stay untested until the real model. ``FILE`` holds
    token streams recorded on the same server without speculation: ``{"streams": [[prompt + output ids], ...],
    "schedule": [L, ...], "tamper_every": N}``. For a context that is a proper prefix of a recorded stream the
    draft is the stream's next ``L`` tokens (``L = schedule[len(context) % len(schedule)]``, at most ``k``; ``0``
    is no draft); when ``N > 0`` and ``len(context) % N == 0`` the draft's last token is replaced by another
    token id, so that step rejects at a known depth. Any other context gets no draft. It replaces the n-gram
    search of the ``ngram`` method and of the lookup beside MTP. Never set it on a production server.
    """

    def __init__(self, streams: Sequence[Sequence[int]], schedule: Sequence[int], tamper_every: int = 0) -> None:
        if not schedule or min(int(n) for n in schedule) < 0:
            raise ValueError(f"the oracle's schedule is a non-empty list of draft lengths, got {list(schedule)}")
        self.streams = [np.asarray(stream, dtype=np.int32) for stream in streams]
        self.schedule = [int(n) for n in schedule]
        self.tamper_every = int(tamper_every)
        self.drafts = 0
        self.tampered = 0

    @classmethod
    def from_file(cls, path: str) -> "LookupOracle":
        with open(path, encoding="utf-8") as f:
            plan = json.load(f)
        return cls(plan["streams"], plan.get("schedule") or [1 << 30], int(plan.get("tamper_every") or 0))

    def draft(self, tokens: np.ndarray, k: int) -> list[int]:
        """The draft for the context ``tokens`` (1-D ids: prompt + output so far): up to ``k`` tokens, or ``[]``."""
        n = int(len(tokens))
        length = min(int(k), self.schedule[n % len(self.schedule)])
        if n == 0 or length <= 0:
            return []
        context = np.asarray(tokens, dtype=np.int32)
        for stream in self.streams:
            if len(stream) > n and stream[n - 1] == context[-1] and np.array_equal(stream[:n], context):
                draft = [int(t) for t in stream[n:n + length]]
                if self.tamper_every > 0 and n % self.tamper_every == 0:
                    draft[-1] ^= 1  # another token id: the target's choice here is the stream's own token
                    self.tampered += 1
                self.drafts += 1
                return draft
        return []


def lookup_oracle() -> LookupOracle | None:
    """The rehearsal oracle named by ``VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE``, or None (the default: no file)."""
    path = os.environ.get(ORACLE_ENV, "").strip()
    if not path:
        return None
    oracle = LookupOracle.from_file(path)
    logger.warning("Flash-Next lookup: REHEARSAL ORACLE %s (%d recorded streams, schedule %s, tamper every %d): "
                   "drafts come from the file, not from the n-gram search", path, len(oracle.streams),
                   oracle.schedule, oracle.tamper_every)
    return oracle


class HybridLookup:
    """The runner-side state of the hybrid: the n-gram sizes, the source of every request's last draft and the
    acceptance counters (logged every ``LOG_EVERY_STEPS`` steps)."""

    def __init__(self, k: int, min_ngram: int, max_ngram: int, max_model_len: int,
                 default_mode: SpecLookupMode = SpecLookupMode.HYBRID, oracle: LookupOracle | None = None) -> None:
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError(f"lookup n-gram sizes need 1 <= min <= max, got {min_ngram}, {max_ngram}")
        self.k, self.min_ngram, self.max_ngram, self.max_model_len = int(k), int(min_ngram), int(max_ngram), int(
            max_model_len)
        self.default_mode = SpecLookupMode(default_mode)
        self.oracle = oracle  # rehearsals only (LookupOracle): replaces the n-gram search
        self.last_source: dict[str, str] = {}
        self.live: frozenset[str] = frozenset()  # the requests of the previous call (a disjoint set: a new load)
        self.steps = 0
        # per source: [rows drafted, drafts accepted, rows verified]
        self.counts = {"lookup": [0, 0, 0], "mtp": [0, 0, 0], "none": [0, 0, 0]}

    @classmethod
    def from_env(cls, k: int, max_model_len: int) -> "HybridLookup":
        from vllm_gaudi import envs as gaudi_envs

        return cls(k, int(gaudi_envs.VLLM_HPU_FLASH_NEXT_LOOKUP_MIN_NGRAM),
                   int(gaudi_envs.VLLM_HPU_FLASH_NEXT_LOOKUP_MAX_NGRAM), max_model_len, lookup_mode(),
                   lookup_oracle())

    def observe(self, req_ids: Sequence[str], sampled_token_ids: Sequence[Sequence[int]]) -> None:
        """Credit the previous step's drafts: a request with ``n`` sampled tokens accepted ``n - 1`` drafts."""
        for req_id, sampled in zip(req_ids, sampled_token_ids):
            source = self.last_source.pop(req_id, None)
            if source is None or not sampled:
                continue
            self.counts[source][1] += len(sampled) - 1
            self.counts[source][2] += 1

    def merge(self, mtp_drafts: list[list[int]], req_ids: Sequence[str], token_ids_cpu: np.ndarray,
              num_tokens_no_spec: np.ndarray, sampled_token_ids: Sequence[Sequence[int]], num_decodes: int,
              extra_args: Sequence[dict | None] | None = None) -> list[list[int]]:
        """The hybrid drafts for the rows of ``mtp_drafts`` (one per entry of ``req_ids``, input-batch order).

        Rows past ``num_decodes`` (requests that just finished a prefill) keep MTP's drafts unless every request
        has a row (the prefill drafts are aligned only then); a request that sampled nothing this step (a prefill
        chunk) gets no lookup either. ``extra_args[i]`` (the request's ``SamplingParams.extra_args``) may pick the
        row's mode (``REQUEST_KEY``); otherwise the server's default mode applies."""
        self.observe(req_ids, sampled_token_ids)
        live = frozenset(req_ids)
        if self.live and not (live & self.live):
            # None of the previous step's requests is left: the counters of the load that just ended (a probe reads
            # one line per round of requests; the last round's line comes with the next request).
            logger.info("Flash-Next lookup hybrid: %s (after %d steps, a new set of requests)", self.summary(),
                        self.steps)
        self.live = live
        rows = len(mtp_drafts)
        aligned = rows == len(req_ids)
        lookup, eligible = [], []
        for i in range(rows):
            ok = (i < num_decodes or aligned) and i < len(sampled_token_ids) and bool(sampled_token_ids[i])
            draft: list[int] = []
            if ok and request_mode(extra_args[i] if extra_args is not None and i < len(extra_args) else None,
                                   self.default_mode) != SpecLookupMode.OFF:
                n = int(num_tokens_no_spec[i])
                if self.oracle is not None:
                    draft = self.oracle.draft(token_ids_cpu[i, :n], self.k)
                else:
                    draft = find_lookup_draft(token_ids_cpu[i, :n], self.min_ngram, self.max_ngram, self.k,
                                              self.max_model_len)
            lookup.append(draft)
            eligible.append(ok)
        modes = [request_mode(extra_args[i] if extra_args is not None and i < len(extra_args) else None,
                              self.default_mode) for i in range(rows)]
        drafts, sources = merge_drafts(mtp_drafts, lookup, self.k, eligible, modes)
        for i, source in enumerate(sources):
            if i < len(req_ids) and eligible[i]:
                self.last_source[req_ids[i]] = source
                self.counts[source][0] += 1
        self.steps += 1
        if self.steps % LOG_EVERY_STEPS == 0:
            logger.info("Flash-Next lookup hybrid: %s", self.summary())
            live = set(req_ids)
            self.last_source = {r: s for r, s in self.last_source.items() if r in live}
        return drafts

    def summary(self) -> str:
        parts = []
        for source, (drafted, accepted, verified) in self.counts.items():
            mean = accepted / verified if verified else 0.0
            parts.append(f"{source} rows {drafted}, accepted {accepted} over {verified} verified rows "
                         f"({mean:.2f} per row)")
        return "; ".join(parts)

    def forget(self, req_ids: Sequence[str]) -> None:
        for req_id in req_ids:
            self.last_source.pop(req_id, None)
