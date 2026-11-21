# SPDX-License-Identifier: Apache-2.0
"""What a speculative step is made of (VLLM_HPU_MTP_TIMING=1): the verification part and the draft, pass by pass.

The first real-model MTP run (card job 00021, TP8, k = 3) accepted 2.78 tokens per step at one stream, and the
step cost 84 ms against 42 ms without speculation: the 42 ms of overhead, not the acceptance, limits MTP. This
timer splits a runner step (``HPUModelRunner.sample_tokens``) of a speculative server into

* ``verify``: from the start of the step to the call of the drafter (input preparation, the target forward over
  the ``1 + k`` rows of every request, the sampler, the host read of the sampled tokens);
* ``draft``: the drafter call up to the moment the device has finished it. The timer waits for the device there
  (``mark_step()`` then ``synchronize()``, as the step-anatomy timer does), which moves a wait the step has
  anyway (the engine reads the drafts back right after the step) and gives the device time of the draft to the
  draft. Inside the draft, laps the proposer sets: ``pass0`` (issue of draft pass 0), ``pass0_wait`` (the host
  read of the positions after pass 0, a device wait: absent with VLLM_HPU_MTP_DRAFT_LOOP=1), then per later
  pass ``passN_meta`` (the host builds the pass's attention metadata) and ``passN`` (its issue), and
  ``final_wait`` (the device finishing the passes issued behind ``pass0_wait``).

With k drafted tokens ``Ft = verify`` and ``Fd = draft / k`` are the two step costs the choice of k needs
(``tools/flash_next/mtp_k_choice.py``). Records are kept per step kind and row count and written every
``EVERY`` steps of a key: one log line (the driver rank) and one JSON line in
``mtp_step_timing_r<rank>_<pid>.jsonl`` in the working directory. Wall-clock means in milliseconds; the first
steps of a key hold the graph compiles of that shape, so read the later records.

The timer costs a few ``perf_counter`` calls and one device wait per step. Off (the default) nothing is created.
"""

from __future__ import annotations

import json
import os
import time

EVERY = 25
SPAN_VERIFY = "verify"
SPAN_DRAFT = "draft"
LAP_FINAL_WAIT = "final_wait"


class MtpStepTimer:
    """Per-step spans of a speculative runner step; see the module docstring."""

    def __init__(self, rank: int = 0, log=None, clock=time.perf_counter, device_wait=None, every: int = EVERY,
                 directory: str | None = None):
        self.rank = int(rank)
        self.log = log
        self.clock = clock
        self.device_wait = device_wait
        self.every = int(every)
        self.path = os.path.join(directory or os.getcwd(), "mtp_step_timing_r%d_%d.jsonl" % (self.rank, os.getpid()))
        self.records: dict[tuple, dict] = {}
        self._step_start = None
        self._draft_start = None
        self._last = None
        self._laps: list[tuple[str, float]] = []
        self._key = None

    # ---- the runner's calls ----

    def step_start(self) -> None:
        self._step_start = self.clock()
        self._draft_start = None
        self._key = None

    def draft_start(self, kind: str, rows: int, requests: int, k: int) -> None:
        """Right before the drafter call. ``rows`` is the token row count of the step's target forward."""
        if self._step_start is None:
            return
        now = self.clock()
        self._key = (str(kind), int(rows), int(requests), int(k))
        self._draft_start = now
        self._last = now
        self._laps = []

    def lap(self, name: str) -> None:
        """A point inside the draft (the proposer's calls); the time since the previous point goes to ``name``."""
        if self._draft_start is None:
            return
        now = self.clock()
        self._laps.append((name, now - self._last))
        self._last = now

    def draft_end(self) -> None:
        """Right after the drafter call: wait for the device, then close the step's record."""
        if self._draft_start is None or self._key is None:
            return
        if self.device_wait is not None:
            self.device_wait()
        self.lap(LAP_FINAL_WAIT)
        now = self.clock()
        spans = {SPAN_VERIFY: self._draft_start - self._step_start, SPAN_DRAFT: now - self._draft_start}
        spans.update(self._merge(self._laps))
        self._add(self._key, spans)
        self._step_start = None
        self._draft_start = None

    # ---- records ----

    @staticmethod
    def _merge(laps) -> dict:
        merged: dict[str, float] = {}
        for name, seconds in laps:
            merged[name] = merged.get(name, 0.0) + seconds
        return merged

    def _add(self, key: tuple, spans: dict) -> None:
        record = self.records.setdefault(key, {"steps": 0, "sums": {}, "written": 0})
        record["steps"] += 1
        for name, seconds in spans.items():
            record["sums"][name] = record["sums"].get(name, 0.0) + seconds
        if record["steps"] % self.every == 0:
            self._write(key, record)

    def summary(self, key: tuple, record: dict) -> dict:
        """The means of the steps since the last written record of ``key``, in milliseconds."""
        steps = record["steps"] - record["written"]
        kind, rows, requests, k = key
        means = {name: 1000.0 * total / steps for name, total in record["sums"].items()} if steps else {}
        out = {"kind": kind, "rows": rows, "requests": requests, "k": k, "steps": steps, "total_steps": record["steps"],
               "rank": self.rank, "ms": {name: round(value, 3) for name, value in means.items()}}
        if steps and k > 0 and SPAN_DRAFT in means:
            out["Ft_ms"] = round(means[SPAN_VERIFY], 3)
            out["Fd_ms"] = round(means[SPAN_DRAFT] / k, 3)
        return out

    def _write(self, key: tuple, record: dict) -> None:
        out = self.summary(key, record)
        record["written"] = record["steps"]
        record["sums"] = {}
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(out) + "\n")
        except OSError:
            pass
        if self.log is not None:
            laps = ", ".join("%s %.1f" % (name, value) for name, value in out["ms"].items()
                             if name not in (SPAN_VERIFY, SPAN_DRAFT))
            self.log("MTP step timing: %s rows=%d requests=%d k=%d steps=%d: verify %.1f ms, draft %.1f ms (%s)",
                     out["kind"], out["rows"], out["requests"], out["k"], out["steps"],
                     out["ms"].get(SPAN_VERIFY, 0.0), out["ms"].get(SPAN_DRAFT, 0.0), laps)


def _hpu_device_wait():
    """``mark_step()`` then ``synchronize()`` of the bridge, or None where the bridge has neither (a CPU stand-in)."""
    import habana_frameworks.torch as htorch
    import torch

    mark_step = getattr(getattr(htorch, "core", None), "mark_step", None)
    synchronize = getattr(getattr(torch, "hpu", None), "synchronize", None)
    if not callable(mark_step) or not callable(synchronize):
        return None

    def wait():
        mark_step()
        synchronize()

    return wait


def from_env(rank: int = 0, is_driver: bool = True, log=None, device_type: str = "hpu") -> MtpStepTimer | None:
    """The runner's timer, or None when VLLM_HPU_MTP_TIMING is not 1. Every rank times (the JSON files), the
    driver rank also logs."""
    from vllm_gaudi import envs as gaudi_envs

    if int(gaudi_envs.VLLM_HPU_MTP_TIMING) != 1:
        return None
    device_wait = None
    if device_type == "hpu":
        try:
            device_wait = _hpu_device_wait()
        except Exception:  # noqa: BLE001 - a CPU stand-in has no bridge: the timer then waits for nothing
            device_wait = None
    return MtpStepTimer(rank=rank, log=log if is_driver else None, device_wait=device_wait)
