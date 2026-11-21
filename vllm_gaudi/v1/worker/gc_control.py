# SPDX-License-Identifier: Apache-2.0
"""A worker's Python garbage collector while steps run (``VLLM_HPU_WORKER_GC_FREEZE``, switchable at run time).

Every collective of a TP8 step is a barrier across the eight ranks, so one rank's stall stops all eight; a
generation-2 collection of a worker's heap (millions of objects after the model is loaded) is such a stall, at a
random step of a random rank. With the flag at ``1``:

* at the first step (after warm-up) the heap is collected once and frozen (``gc.collect(); gc.freeze()``): the
  objects that exist then (the model, its graphs, the caches) are never scanned again;
* automatic generation-2 collections are off while steps run (the third threshold set out of reach); generations 0
  and 1 still collect the young objects every step makes;
* a daemon thread runs one full collection when no step has run for ``IDLE_S`` seconds (the engine is idle), so
  cyclic garbage of finished requests is still freed.

Back at ``0`` (a runtime switch): the heap is unfrozen and the saved thresholds come back. The objects frozen at the
first step stay alive anyway (the model), and non-cyclic garbage is freed by reference counting either way; what the
flag defers is the collection of cyclic garbage, to the idle moments.
"""

from __future__ import annotations

import gc
import threading
import time

GEN2_NEVER = 1_000_000_000  # third threshold: generation-1 collections before a generation-2 one
IDLE_S = 0.5


class GcControl:
    """The worker's collector mode; ``apply`` at every step start (cheap when the mode did not change)."""

    def __init__(self, clock=time.monotonic, idle_s: float = IDLE_S, log=None):
        self.clock = clock
        self.idle_s = float(idle_s)
        self.log = log
        self.mode = 0
        self.saved_threshold = None
        self.frozen = 0
        self.idle_collections = 0
        self.busy = False
        self.last_step_end = clock()
        self.collected_since_step = True
        self._stop = threading.Event()
        self._thread = None

    def apply(self, mode) -> None:
        try:
            mode = 1 if int(mode) == 1 else 0
        except (TypeError, ValueError):
            mode = 0
        if mode == self.mode:
            return
        if mode == 1:
            gc.collect()
            gc.freeze()
            self.frozen = gc.get_freeze_count()
            self.saved_threshold = gc.get_threshold()
            gc.set_threshold(self.saved_threshold[0], self.saved_threshold[1], GEN2_NEVER)
            self._stop.clear()
            self._thread = threading.Thread(target=self._idle_loop, name="hpu-gc-idle", daemon=True)
            self._thread.start()
            if self.log is not None:
                self.log("Worker GC freeze (VLLM_HPU_WORKER_GC_FREEZE=1): %d objects frozen, generation-2 "
                         "collections only when the worker is idle (%.1f s without a step)", self.frozen, self.idle_s)
        else:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._thread = None
            gc.unfreeze()
            if self.saved_threshold is not None:
                gc.set_threshold(*self.saved_threshold)
            self.saved_threshold = None
            if self.log is not None:
                self.log("Worker GC freeze off: heap unfrozen, thresholds %s, %d idle collections while on",
                         gc.get_threshold(), self.idle_collections)
        self.mode = mode

    def step_begin(self) -> None:
        self.busy = True

    def step_end(self) -> None:
        self.busy = False
        self.last_step_end = self.clock()
        self.collected_since_step = False

    def idle_collect(self) -> bool:
        """One full collection if the worker is idle and has stepped since the last one; True when it ran."""
        if self.mode != 1 or self.busy or self.collected_since_step:
            return False
        if self.clock() - self.last_step_end < self.idle_s:
            return False
        gc.collect()
        self.collected_since_step = True
        self.idle_collections += 1
        return True

    def _idle_loop(self) -> None:
        while not self._stop.wait(self.idle_s / 2):
            try:
                self.idle_collect()
            except Exception:  # noqa: BLE001 - a collector thread never takes the worker down
                pass


class GcPauses:
    """Collector pauses between ``reset`` calls, from ``gc.callbacks`` (installed by ``install``): milliseconds per
    generation and the number of generation-2 collections. Collections in other threads count too."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.ms = 0.0
        self.gen2 = 0
        self.collections = 0
        self._start = None
        self._installed = False

    def install(self) -> None:
        if not self._installed:
            gc.callbacks.append(self._callback)
            self._installed = True

    def reset(self) -> None:
        self.ms = 0.0
        self.gen2 = 0
        self.collections = 0

    def _callback(self, phase, info) -> None:
        if phase == "start":
            self._start = self.clock()
        elif phase == "stop" and self._start is not None:
            self.ms += 1000.0 * (self.clock() - self._start)
            self.collections += 1
            if int(info.get("generation", -1)) == 2:
                self.gen2 += 1
            self._start = None


__all__ = ["GcControl", "GcPauses", "GEN2_NEVER", "IDLE_S"]
