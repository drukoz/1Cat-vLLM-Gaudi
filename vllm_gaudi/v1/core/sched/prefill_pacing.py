# SPDX-License-Identifier: Apache-2.0
"""Prefill pacing and runtime step-budget knobs for the scheduler (Flash-Next, prefill track).

Why. The HPU runner executes a step's prompt chunks one forward after another and then the decode
batch (``HPUModelRunner.sample_tokens``), and the scheduler gives running prompt chunks their tokens
before anything else. Every decoding stream therefore waits for all the prompt work of a step: with
``--max-num-batched-tokens 8192 --long-prefill-token-threshold 2048`` a stream got one token per four
2,048-token prompt forwards for as long as any prompt was being prefilled.

What. ``install_prefill_pacing()`` wraps ``Scheduler.schedule`` (the base class, so the HPU async
scheduler is covered through ``super()``). While at least one request is decoding, the wrapper

* gives prompt work a share of wall time: after a step that carried prompt tokens it defers all prompt
  work, through the scheduler's own ``throttle_prefills`` path, until the decode-only steps since then
  have lasted ``extra * (100 - share) / share`` seconds, where ``extra`` is how much longer the prompt
  step took than a decode-only step (measured between ``schedule`` calls);
* caps the prompt tokens of one step (the slice), which bounds the longest stall of a stream to one
  slice.

With no request decoding nothing is paced: a cold start prefills at the server's full step budget.
The wrapper also applies a step budget and a per-request prompt cap read from a JSON file, so a
settings sweep changes them on a running server instead of restarting it for every pair. The same
file carries one knob for the workers, ``prompt_graph_max_tokens`` (``prompt_graph_knob``: the model
runner's prompt HPU-graph threshold), so one server measures prompt forwards with and without graph
replay.

Flags (``vllm_gaudi/envs.py``, all off by default): ``VLLM_HPU_PREFILL_PACING`` (``PrefillPacingMode``),
``VLLM_HPU_PREFILL_PACING_SHARE``, ``VLLM_HPU_PREFILL_PACING_SLICE_TOKENS``, ``VLLM_HPU_SCHED_KNOBS_FILE``.
No HPU import: the policy (``PrefillPacer``) is plain Python and runs under the CPU tests.
"""

from __future__ import annotations

import enum
import functools
import json
import os
import time
from dataclasses import dataclass, fields, replace
from typing import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_MODE = "VLLM_HPU_PREFILL_PACING"
ENV_SHARE = "VLLM_HPU_PREFILL_PACING_SHARE"
ENV_SLICE = "VLLM_HPU_PREFILL_PACING_SLICE_TOKENS"
ENV_KNOBS_FILE = "VLLM_HPU_SCHED_KNOBS_FILE"

DEFAULT_SHARE_PERCENT = 25
DEFAULT_SLICE_TOKENS = 2048
KNOBS_POLL_S = 0.25
MAX_ACCOUNTED_STEP_S = 120.0  # a longer gap between two schedule calls is an idle engine, not a step
DECODE_EMA_WEIGHT = 0.2
STATE_ATTR = "_hpu_prefill_pacing_state"


class PrefillPacingMode(enum.IntEnum):
    """``VLLM_HPU_PREFILL_PACING`` (next free: 2)."""

    OFF = 0
    TIME_SHARE = 1


class StepKind(enum.IntEnum):
    """What one scheduled step carried."""

    NONE = 0  # nothing was scheduled
    DECODE = 1  # decode rows only
    PROMPT = 2  # prompt tokens, with or without decode rows


@dataclass(frozen=True)
class Knobs:
    """The settings one ``schedule`` call runs with.

    ``max_num_scheduled_tokens`` 0 and ``long_prefill_token_threshold`` -1 keep the server's own values
    (``--max-num-batched-tokens``, ``--long-prefill-token-threshold``); a threshold of 0 removes the
    per-request cap. ``share_percent`` is the wall-time share of prompt work while streams decode
    (100 = no deferral) and ``slice_tokens`` the prompt tokens of one step while streams decode
    (0 = no cap); both act only in ``PrefillPacingMode.TIME_SHARE``. ``prompt_graph_max_tokens`` is
    not a scheduler setting: the workers read it (``prompt_graph_knob``); -1 keeps the server's
    ``VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS``, 0 is upstream's threshold, a positive value the threshold.
    """

    mode: int = int(PrefillPacingMode.OFF)
    share_percent: int = DEFAULT_SHARE_PERCENT
    slice_tokens: int = DEFAULT_SLICE_TOKENS
    max_num_scheduled_tokens: int = 0
    long_prefill_token_threshold: int = -1
    prompt_graph_max_tokens: int = -1

    def __post_init__(self) -> None:
        PrefillPacingMode(self.mode)  # raises on an unknown mode
        if not 1 <= self.share_percent <= 100:
            raise ValueError(f"share_percent must be 1..100, got {self.share_percent}")
        if self.slice_tokens < 0:
            raise ValueError(f"slice_tokens must be >= 0, got {self.slice_tokens}")
        if self.max_num_scheduled_tokens < 0:
            raise ValueError(f"max_num_scheduled_tokens must be >= 0, got {self.max_num_scheduled_tokens}")
        if self.long_prefill_token_threshold < -1:
            raise ValueError(f"long_prefill_token_threshold must be >= -1, got {self.long_prefill_token_threshold}")
        if self.prompt_graph_max_tokens < -1:
            raise ValueError(f"prompt_graph_max_tokens must be >= -1, got {self.prompt_graph_max_tokens}")

    @property
    def paced(self) -> bool:
        return self.mode == PrefillPacingMode.TIME_SHARE

    @property
    def is_noop(self) -> bool:
        return not self.paced and self.max_num_scheduled_tokens == 0 and self.long_prefill_token_threshold == -1

    def merged(self, overrides: dict) -> "Knobs":
        """These knobs with the given fields replaced; unknown keys and non-integers raise ``ValueError``."""
        names = {f.name for f in fields(self)}
        unknown = sorted(set(overrides) - names)
        if unknown:
            raise ValueError(f"unknown scheduler knobs {unknown}; known: {sorted(names)}")
        clean = {}
        for name, value in overrides.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"scheduler knob {name} must be an integer, got {value!r}")
            clean[name] = value
        return replace(self, **clean)


def knobs_from_env(environ=None) -> Knobs:
    """The start-up knobs from the three pacing flags (the runtime file, if any, is merged on top)."""
    environ = os.environ if environ is None else environ
    return Knobs(mode=int(environ.get(ENV_MODE, "0")),
                 share_percent=int(environ.get(ENV_SHARE, str(DEFAULT_SHARE_PERCENT))),
                 slice_tokens=int(environ.get(ENV_SLICE, str(DEFAULT_SLICE_TOKENS))))


class KnobsFile:
    """A JSON object of ``Knobs`` fields, re-read when the file changes (checked every ``KNOBS_POLL_S``).

    A missing file means "no overrides". A file that does not parse or fails validation keeps the last
    good knobs and is reported once per change, so a half-written file never stops the scheduler.
    """

    def __init__(self, path: str, base: Knobs, clock: Callable[[], float] = time.monotonic,
                 poll_s: float = KNOBS_POLL_S) -> None:
        self.path = path
        self.base = base
        self.current = base
        self._clock = clock
        self._poll_s = poll_s
        self._next_poll = 0.0
        self._stamp: tuple | None = None

    def get(self) -> Knobs:
        now = self._clock()
        if now < self._next_poll:
            return self.current
        self._next_poll = now + self._poll_s
        try:
            stat = os.stat(self.path)
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        if stamp == self._stamp:
            return self.current
        self._stamp = stamp
        if stamp is None:
            self.current = self.base
            logger.info("Scheduler knobs file %s is gone: back to %s", self.path, self.current)
            return self.current
        try:
            with open(self.path, encoding="utf-8") as handle:
                overrides = json.load(handle)
            if not isinstance(overrides, dict):
                raise ValueError("the file must hold one JSON object")
            self.current = self.base.merged(overrides)
            logger.info("Scheduler knobs from %s: %s", self.path, self.current)
        except (OSError, ValueError) as exc:
            logger.warning("Scheduler knobs file %s ignored (%s); keeping %s", self.path, exc, self.current)
        return self.current


def prompt_graph_knob(server_value: int, environ=None,
                      clock: Callable[[], float] = time.monotonic) -> Callable[[], int] | None:
    """For the model runner: the prompt HPU-graph threshold under the runtime knobs file.

    ``None`` without ``VLLM_HPU_SCHED_KNOBS_FILE`` (the runner then keeps ``server_value``, its
    ``VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS``, and reads nothing). Otherwise a function for the start of every
    engine step that returns the file's ``prompt_graph_max_tokens``, or ``server_value`` while the file
    has none or says -1. The file is checked on every call (one ``stat``): every worker of a server must
    take a change at the same step, since a rank that replays a graph next to ranks that trace the same
    forward has never run. So change the knob only while the server is idle; it is a measurement knob.
    """
    environ = os.environ if environ is None else environ
    path = environ.get(ENV_KNOBS_FILE, "")
    if not path:
        return None
    source = KnobsFile(path, Knobs(), clock, poll_s=0.0)

    def current() -> int:
        value = source.get().prompt_graph_max_tokens
        return server_value if value < 0 else value

    return current


class PrefillPacer:
    """The time-share policy: when prompt work has to wait for decode-only steps.

    Call ``should_defer`` at the start of every ``schedule`` call and ``record`` with what the step
    carried. Time is measured between ``should_defer`` calls, so a step's duration is the model
    execution plus the engine's own work for it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_time: float | None = None
        self._last_kind = StepKind.NONE
        self._last_had_decoding = False
        self.decode_step_s: float | None = None  # moving average of a decode-only step
        self.debt_s = 0.0  # decode-only time owed before the next prompt step
        self.paid_s = 0.0  # decode-only time since the last prompt step

    def _account(self, share_percent: int) -> None:
        now = self._clock()
        last, self._last_time = self._last_time, now
        if last is None:
            return
        elapsed = now - last
        if elapsed > MAX_ACCOUNTED_STEP_S:  # the engine was idle: nothing waited
            self.debt_s = self.paid_s = 0.0
            return
        if self._last_kind == StepKind.DECODE:
            self.paid_s += elapsed
            self.decode_step_s = elapsed if self.decode_step_s is None else (
                (1.0 - DECODE_EMA_WEIGHT) * self.decode_step_s + DECODE_EMA_WEIGHT * elapsed)
        elif self._last_kind == StepKind.PROMPT:
            self.paid_s = 0.0
            if self._last_had_decoding:
                extra = max(elapsed - (self.decode_step_s or 0.0), 0.0)
                self.debt_s = extra * (100 - share_percent) / share_percent
            else:
                self.debt_s = 0.0

    def should_defer(self, knobs: Knobs, num_decoding: int, has_prompt_work: bool) -> bool:
        """True when this step must carry decode rows only."""
        self._account(knobs.share_percent)
        if not knobs.paced or knobs.share_percent >= 100 or num_decoding == 0 or not has_prompt_work:
            return False
        return self.paid_s < self.debt_s

    def record(self, kind: StepKind, had_decoding: bool) -> None:
        self._last_kind = kind
        self._last_had_decoding = had_decoding


def effective_slice(slice_tokens: int, block_size: int) -> int:
    """The slice as whole scheduler blocks, at least one block (0 stays 0: no cap).

    In align mode a prompt chunk ends on a scheduler block (512 tokens at TP8, 896 at TP4), where the
    recurrent state is stored; the runner takes chunks that start on a block.
    """
    if slice_tokens <= 0:
        return 0
    if block_size <= 0:
        return slice_tokens
    return max(slice_tokens // block_size, 1) * block_size


@dataclass
class _State:
    knobs_source: Callable[[], Knobs]
    pacer: PrefillPacer
    server_budget: int
    server_threshold: int
    block_size: int
    steps: int = 0
    deferred_steps: int = 0
    prompt_steps: int = 0


def _make_state(scheduler, clock: Callable[[], float] = time.monotonic) -> _State:
    base = knobs_from_env()
    path = os.environ.get(ENV_KNOBS_FILE, "")
    if getattr(scheduler.scheduler_config, "async_scheduling", False) and base.paced:
        logger.warning("%s=%d ignored: prefill pacing measures steps between schedule calls and needs "
                       "async scheduling off", ENV_MODE, base.mode)
        base = replace(base, mode=int(PrefillPacingMode.OFF))
    source = KnobsFile(path, base, clock).get if path else (lambda: base)
    block_size = int(getattr(scheduler.cache_config, "block_size", 0) or 0)
    state = _State(knobs_source=source,
                   pacer=PrefillPacer(clock),
                   server_budget=int(scheduler.max_num_scheduled_tokens),
                   server_threshold=int(scheduler.scheduler_config.long_prefill_token_threshold),
                   block_size=block_size)
    logger.info("Prefill pacing: %s (knobs file: %s, step budget %d, prompt cap %d, scheduler block %d)", base,
                path or "none", state.server_budget, state.server_threshold, block_size)
    return state


def paced_schedule(scheduler, base_schedule, throttle_prefills: bool = False, clock=time.monotonic):
    """One ``schedule`` call under the current knobs; ``base_schedule`` is the unwrapped method."""
    state = getattr(scheduler, STATE_ATTR, None)
    if state is None:
        state = _make_state(scheduler, clock)
        setattr(scheduler, STATE_ATTR, state)
    knobs = state.knobs_source()
    if knobs.is_noop:
        return base_schedule(scheduler, throttle_prefills)

    running = list(scheduler.running)
    prompt_before = {r.request_id for r in running if r.is_prefill_chunk}
    running_before = {r.request_id for r in running}
    num_decoding = len(running) - len(prompt_before)
    has_prompt_work = bool(prompt_before) or bool(scheduler.waiting) or bool(scheduler.skipped_waiting)
    defer = state.pacer.should_defer(knobs, num_decoding, has_prompt_work)

    budget = state.server_budget
    if knobs.max_num_scheduled_tokens > 0:
        budget = min(budget, knobs.max_num_scheduled_tokens)
    threshold = state.server_threshold if knobs.long_prefill_token_threshold < 0 else knobs.long_prefill_token_threshold
    slice_tokens = effective_slice(knobs.slice_tokens, state.block_size)
    if knobs.paced and num_decoding > 0 and slice_tokens > 0 and not defer:
        # Every decoding request keeps its rows (one token plus its draft tokens); prompt work gets the slice.
        decode_rows = num_decoding * (1 + int(getattr(scheduler, "num_spec_tokens", 0) or 0))
        budget = min(budget, decode_rows + slice_tokens)
        threshold = slice_tokens if threshold <= 0 else min(threshold, slice_tokens)

    config = scheduler.scheduler_config
    saved_budget, saved_threshold = scheduler.max_num_scheduled_tokens, config.long_prefill_token_threshold
    scheduler.max_num_scheduled_tokens = budget
    config.long_prefill_token_threshold = threshold
    try:
        if defer:
            # The base scheduler stops deferring when the waiting queue is "capacity bound"; that rule
            # belongs to data-parallel balancing and would let a queue of cold prompts starve the streams.
            scheduler.prefill_capacity_bound = False
        output = base_schedule(scheduler, defer or throttle_prefills)
    finally:
        scheduler.max_num_scheduled_tokens = saved_budget
        config.long_prefill_token_threshold = saved_threshold

    scheduled = output.num_scheduled_tokens
    if not scheduled:
        kind = StepKind.NONE
    elif any(rid in prompt_before or rid not in running_before for rid in scheduled):
        kind = StepKind.PROMPT
    else:
        kind = StepKind.DECODE
    state.pacer.record(kind, had_decoding=num_decoding > 0)
    state.steps += 1
    state.deferred_steps += int(defer)
    state.prompt_steps += int(kind == StepKind.PROMPT)
    return output


def pacing_requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return int(environ.get(ENV_MODE, "0")) != PrefillPacingMode.OFF or bool(environ.get(ENV_KNOBS_FILE, ""))


def install_prefill_pacing(scheduler_cls=None, force: bool = False) -> bool:
    """Wrap ``scheduler_cls.schedule`` (default: vLLM's base ``Scheduler``) when a pacing flag asks for it.

    Returns True when the wrapper is in place. Idempotent; without ``force`` and with both flags unset
    the class is left untouched, so a default server runs upstream's scheduler byte for byte.
    """
    if scheduler_cls is None:
        from vllm.v1.core.sched.scheduler import Scheduler as scheduler_cls
    if getattr(scheduler_cls.schedule, "_hpu_prefill_pacing", False):
        return True
    if not force and not pacing_requested():
        return False
    base_schedule = scheduler_cls.schedule

    @functools.wraps(base_schedule)
    def schedule(self, throttle_prefills: bool = False):
        return paced_schedule(self, base_schedule, throttle_prefills)

    schedule._hpu_prefill_pacing = True
    schedule._hpu_base_schedule = base_schedule
    scheduler_cls.schedule = schedule
    logger.info("Prefill pacing wrapper installed on %s.schedule", scheduler_cls.__name__)
    return True


def uninstall_prefill_pacing(scheduler_cls=None) -> None:
    """Put the unwrapped ``schedule`` back (tests)."""
    if scheduler_cls is None:
        from vllm.v1.core.sched.scheduler import Scheduler as scheduler_cls
    base = getattr(scheduler_cls.schedule, "_hpu_base_schedule", None)
    if base is not None:
        scheduler_cls.schedule = base


__all__ = [
    "ENV_KNOBS_FILE",
    "ENV_MODE",
    "ENV_SHARE",
    "ENV_SLICE",
    "Knobs",
    "KnobsFile",
    "PrefillPacer",
    "PrefillPacingMode",
    "StepKind",
    "effective_slice",
    "install_prefill_pacing",
    "knobs_from_env",
    "paced_schedule",
    "pacing_requested",
    "prompt_graph_knob",
    "uninstall_prefill_pacing",
]
