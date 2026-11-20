# SPDX-License-Identifier: Apache-2.0
"""Per-step host timer of the HPU worker, the model runner and the engine core (``VLLM_HPU_STEP_TIMING``).

Where does the time of one engine step go, as the host sees it? The timer cuts a step into spans with one clock
read per boundary (``lap``), keeps the steps of one kind and row count together, and every N steps writes one
log line (p50 / p90 per span) and one JSON line with the same numbers and the raw per-step values.

Modes (``VLLM_HPU_STEP_TIMING``, an integer; any other value is 0):

* ``0`` off (default): no timer object exists, the instrumented code paths test one ``None``.
* ``1`` spans: the host spans of every step, the collectives per step and the host time spent issuing them.
* ``2`` spans and the debug counters of host synchronisation points: ``.item()``, ``.cpu()``, ``.tolist()``,
  ``.numpy()``, a tensor used as a Python scalar, host-to-device and device-to-host copies (``.to``, ``.copy_``),
  ``mark_step()`` and ``synchronize()`` calls, each with the source line that made it. The counters patch
  ``torch.Tensor`` methods while this mode is on.
* ``3`` spans with device attribution: the host waits for the device (``mark_step()`` then ``synchronize()``)
  before and after every collective and after the forward and the sampler, so the device time of every segment
  between two collectives and of every collective is measured on its own. A step is slower in this mode (one
  extra launch and wait per segment); the segments, not the step total, are the result.
* ``4`` spans and the dispatched-op count of the forward, the logits and the sampler (a ``TorchDispatchMode``
  around each). The Python dispatch makes the host spans of this mode slower than they are.

A step of the worker is one ``execute_model`` call and the ``sample_tokens`` call that follows it:

``gap`` (the engine's turn between two steps) | ``execute_model`` | ``rpc_gap`` (the engine's turn between the two
calls) | ``setup`` | ``prepare`` | ``launch_inputs`` | ``forward`` | ``logits`` | ``launch_model`` | ``sample`` |
``launch_sample`` | ``wait`` (the first host read of device results) | ``draft`` | ``post``

``step`` is the sum from ``execute_model`` to ``post``; ``period`` adds ``gap``. The engine core's step has
``schedule``, ``execute_model_send``, ``execute_model_wait``, ``sample_tokens_rpc``, ``update_from_output`` and
``other``. Files: ``<dir>/step_timing_<role>_r<rank>_<pid>.jsonl`` with ``<dir>`` from
``VLLM_HPU_STEP_TIMING_DIR`` (default: the working directory); N from ``VLLM_HPU_STEP_TIMING_EVERY``.

A process whose timer exists rereads ``<dir>/step_timing.mode`` (one integer) whenever a step starts after an idle
gap of two seconds, so one server can serve several modes without a restart: write the file while no request runs.

While a torch profiler records (``/start_profile``), every timed step of a process is also a profiler range named
``step_timing:step``, and its kind and row count a zero-length range ``step_timing:<kind>:<rows>`` inside it, so a
device trace can be cut into steps exactly (``tools/flash_next/anatomy/trace_device.py``).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import sys
import threading
import time

MODE_OFF = 0
MODE_SPANS = 1
MODE_COUNTERS = 2
MODE_SYNC = 3
MODE_OPS = 4
MODES = (MODE_OFF, MODE_SPANS, MODE_COUNTERS, MODE_SYNC, MODE_OPS)
MODE_FILE = "step_timing.mode"
SCHEMA = 1
IDLE_S = 2.0
MAX_SITES = 24
MAX_OP_NAMES = 80

WORKER_SPANS = ("execute_model", "rpc_gap", "setup", "prepare", "launch_inputs", "forward", "logits", "launch_model",
                "device_forward", "sample", "launch_sample", "device_sample", "wait", "draft", "post", "seg_compute",
                "seg_collective")
ENGINE_SPANS = ("schedule", "execute_model_send", "execute_model_wait", "sample_tokens_send", "sample_tokens_wait",
                "sample_tokens_rpc", "update_from_output", "other")

TIMER: StepTimer | None = None  # the process timer; None when the flag is 0
_NULL_CONTEXT = contextlib.nullcontext()
_MISSING = object()


def percentile(values, fraction):
    """Nearest-rank percentile of a non-empty list (as tools/flash_next/step_anatomy.py)."""
    ordered = sorted(values)
    rank = min(max(math.ceil(fraction * len(ordered)), 1), len(ordered))
    return ordered[rank - 1]


def stats(values) -> dict | None:
    """``{"p50", "p90", "mean", "min", "max", "n"}`` of the numbers in ``values`` (``None`` entries are skipped)."""
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return {
        "p50": percentile(kept, 0.50),
        "p90": percentile(kept, 0.90),
        "mean": sum(kept) / len(kept),
        "min": min(kept),
        "max": max(kept),
        "n": len(kept),
    }


def parse_mode(text) -> int:
    """The mode an environment value or a mode file asks for; anything that is not a known mode is 0."""
    try:
        mode = int(str(text).strip())
    except (TypeError, ValueError):
        return MODE_OFF
    return mode if mode in MODES else MODE_OFF


class _Window:
    """The steps of one kind and row count since the last flush."""

    __slots__ = ("kind", "rows", "bucket_rows", "steps", "t_first", "t_last", "series", "segments", "sites",
                 "op_names", "mode")

    def __init__(self, kind, rows, mode, wall):
        self.kind = kind
        self.rows = rows
        self.bucket_rows = None
        self.steps = 0
        self.t_first = wall
        self.t_last = wall
        self.series = {}  # name -> one value per step (ms for spans, a number for counters, None when absent)
        self.segments = []  # mode 3: per step [(name, compute ms, collective ms), ...]
        self.sites = {}  # mode 2: (label, "file:line") -> count
        self.op_names = {}  # mode 4: op name -> count
        self.mode = mode

    def put(self, name, value):
        column = self.series.get(name)
        if column is None:
            column = self.series[name] = [None] * self.steps
        column.append(value)

    def close_step(self):
        self.steps += 1
        for column in self.series.values():
            if len(column) < self.steps:
                column.append(None)


class _OpCountContext:
    """Counts the ops dispatched inside the ``with`` block into the open step (mode 4)."""

    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.mode = None

    def __enter__(self):
        timer = self.timer
        try:
            from torch.utils._python_dispatch import TorchDispatchMode

            names = timer._op_names
            box = [0]

            class Counter(TorchDispatchMode):

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    box[0] += 1
                    schema = getattr(func, "_schema", None)
                    key = schema.name if schema is not None else str(func)
                    names[key] = names.get(key, 0) + 1
                    return func(*args, **(kwargs or {}))

            self.box = box
            self.mode = Counter()
            self.mode.__enter__()
        except Exception as exc:  # noqa: BLE001 - counting must never break a step
            self.mode = None
            timer._warn_once("ops", "op counting is not available: %s: %s" % (type(exc).__name__, exc))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.mode is not None:
            self.mode.__exit__(exc_type, exc, tb)
            self.timer.count("ops_" + self.name, self.box[0])
        return False


class StepTimer:
    """One process's step timer. Not thread safe: ``begin_step`` to ``end_step`` run on the thread that executes
    the steps; only ``add_async`` may be called from another thread."""

    def __init__(self,
                 role: str,
                 rank: int = 0,
                 mode: int = MODE_SPANS,
                 every: int = 100,
                 out_dir: str | None = None,
                 log=None,
                 clock=time.perf_counter_ns,
                 wall=time.time,
                 idle_s: float = IDLE_S):
        self.role = role
        self.rank = int(rank)
        self.pid = os.getpid()
        self.mode = mode if mode in MODES else MODE_SPANS
        self.every = max(1, int(every))
        self.out_dir = out_dir or "."
        self.active = False  # a step is open: the instrumented code asks this before it does any work
        self._log = log
        self._clock = clock
        self._wall = wall
        self._idle_ns = int(idle_s * 1e9)
        self._open = False
        self._t_begin = 0
        self._t_last = 0
        self._t_end = None
        self._gap = None
        self._spans = {}
        self._nested = {}
        self._counts = {}
        self._tags = None
        self._bucket_rows = None
        self._segments = []
        self._sites = {}
        self._op_names = {}
        self._windows = {}
        self._runtime = None  # (htorch, torch) of the worker, for modes 2 and 3
        self._patches = []  # mode 2: (object, attribute, saved value)
        self._async = {}
        self._async_lock = threading.Lock()
        self._warned = set()
        self._write_failed = False
        self.phase = "forward"  # the host span a collective's preceding accumulation belongs to
        self._marker = None  # the open profiler range of the step, while a torch profiler records
        self._profiler_on = _MISSING  # torch.autograd._profiler_enabled, looked up at the first step
        self.path = os.path.join(self.out_dir, "step_timing_%s_r%d_%d.jsonl" % (role, self.rank, self.pid))
        if self.mode == MODE_COUNTERS:
            self._install_counters()

    # ----------------------------------------------------------------------------------------- setup
    def bind_runtime(self, htorch, torch) -> None:
        """The worker's bridge module and torch: ``mark_step`` and ``synchronize`` for mode 3, the patch targets of
        mode 2. Attributes are looked up at every call, so a later patch of either is honoured."""
        self._runtime = (htorch, torch)
        if self.mode == MODE_COUNTERS:
            self._uninstall_counters()
            self._install_counters()

    def set_mode(self, mode: int) -> None:
        mode = mode if mode in MODES else MODE_OFF
        if mode == self.mode:
            return
        self.flush()
        if self.mode == MODE_COUNTERS:
            self._uninstall_counters()
        previous, self.mode = self.mode, mode
        if mode == MODE_COUNTERS:
            self._install_counters()
        self._say("step timing: mode %d -> %d (%s rank %d)" % (previous, mode, self.role, self.rank))

    def _say(self, message: str) -> None:
        if self._log is not None:
            try:
                self._log(message)
            except Exception:  # noqa: BLE001 - a log line is never worth a step
                pass

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._say("step timing: " + message)

    # --------------------------------------------------------------------------------- step lifecycle
    def begin_step(self) -> None:
        """Open a step. The time since the last step ended is its ``gap``, unless the process was idle."""
        now = self._clock()
        if self._t_end is None or now - self._t_end > self._idle_ns:
            self._on_idle()
            now = self._clock()
            self._gap = None
        else:
            self._gap = now - self._t_end
        self._close_marker()  # a step that never ended (an exception) leaves no range open
        if self.mode == MODE_OFF:
            self._open = self.active = False
            return
        self._open = self.active = True
        self._t_begin = self._t_last = now
        self._spans = {}
        self._nested = {}
        self._counts = {}
        self._tags = None
        self._bucket_rows = None
        self._segments = []
        self._op_names = {}
        self.phase = "forward"
        self._open_marker()

    # ------------------------------------------------------------------------------- profiler markers
    def _profiling(self) -> bool:
        probe = self._profiler_on
        if probe is _MISSING:
            try:
                import torch
                probe = torch.autograd._profiler_enabled
            except Exception:  # noqa: BLE001 - no torch, no markers
                probe = None
            self._profiler_on = probe
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001
            return False

    def _open_marker(self) -> None:
        if self._marker is not None or not self._profiling():
            return
        try:
            import torch
            marker = torch.profiler.record_function("step_timing:step")
            marker.__enter__()
            self._marker = marker
        except Exception:  # noqa: BLE001 - a marker is never worth a step
            self._marker = None

    def _close_marker(self) -> None:
        marker, self._marker = self._marker, None
        if marker is not None:
            try:
                marker.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    def lap(self, name: str) -> None:
        """The time since the last boundary of the open step belongs to span ``name``."""
        if not self._open:
            return
        now = self._clock()
        spans = self._spans
        spans[name] = spans.get(name, 0) + now - self._t_last
        self._t_last = now

    def add(self, name: str, ns: int) -> None:
        """Time that lies inside a span (reported next to the spans, not added to the step)."""
        if self._open:
            self._nested[name] = self._nested.get(name, 0) + ns

    def add_async(self, name: str, ns: int) -> None:
        """As ``add``, from another thread (the async output thread): credited to the step that ends next."""
        with self._async_lock:
            self._async[name] = self._async.get(name, 0) + ns

    def count(self, name: str, n: int = 1) -> None:
        if self._open:
            self._counts[name] = self._counts.get(name, 0) + n

    def tag(self, kind: str, rows: int, tokens: int | None = None) -> None:
        """What the open step executes: ``prefill``, ``decode``, ``verify`` or ``mixed``, on ``rows`` requests."""
        if self._open:
            self._tags = (kind, int(rows))
            if tokens is not None:
                self._counts["tokens"] = int(tokens)
            if self._marker is not None:
                try:
                    import torch
                    with torch.profiler.record_function("step_timing:%s:%d" % (kind, int(rows))):
                        pass
                except Exception:  # noqa: BLE001
                    pass

    def bucket(self, rows: int) -> None:
        if self._open:
            self._bucket_rows = int(rows)

    def discard_step(self) -> None:
        """Close the open step without recording it (nothing was executed)."""
        if self._open:
            self._open = self.active = False
            self._t_end = self._clock()
            self._close_marker()

    def end_step(self, final_span: str = "post") -> None:
        """Close the open step; what is left since the last boundary belongs to ``final_span``. A step that was
        never tagged (nothing was executed) is dropped."""
        if not self._open:
            return
        now = self._clock()
        self._close_marker()
        spans = self._spans
        spans[final_span] = spans.get(final_span, 0) + now - self._t_last
        total = now - self._t_begin
        self._open = self.active = False
        self._t_end = now
        if self._tags is None:
            return
        kind, rows = self._tags
        key = (kind, rows)
        window = self._windows.get(key)
        wall = self._wall()
        if window is None:
            window = self._windows[key] = _Window(kind, rows, self.mode, wall)
        window.t_last = wall
        if self._bucket_rows is not None:
            window.bucket_rows = self._bucket_rows
        gap = self._gap
        window.put("step", total / 1e6)
        window.put("gap", None if gap is None else gap / 1e6)
        window.put("period", None if gap is None else (total + gap) / 1e6)
        for name, ns in spans.items():
            window.put(name, ns / 1e6)
        for name, ns in self._nested.items():
            window.put("in:" + name, ns / 1e6)
        if self._async:
            with self._async_lock:
                pending, self._async = self._async, {}
            for name, ns in pending.items():
                window.put("in:" + name, ns / 1e6)
        for name, value in self._counts.items():
            window.put("n:" + name, value)
        if self._segments:
            window.segments.append(self._segments)
        for site, n in self._sites.items():
            window.sites[site] = window.sites.get(site, 0) + n
        self._sites = {}
        for name, n in self._op_names.items():
            window.op_names[name] = window.op_names.get(name, 0) + n
        window.close_step()
        if window.steps >= self.every:
            self._flush_window(key)

    # ---------------------------------------------------------------------------------- collectives
    def collective_begin(self) -> int:
        """Before a collective of the open step. Mode 3 first runs what is accumulated so far on its own and
        records it as the compute segment in front of this collective."""
        if self.mode == MODE_SYNC and self._runtime is not None:
            self.lap(self.phase)
            self._device_wait()
            now = self._clock()
            compute = now - self._t_last
            self._spans["seg_compute"] = self._spans.get("seg_compute", 0) + compute
            self._t_last = now
            return compute
        return self._clock()

    def collective_end(self, token: int, name: str, tensor=None) -> None:
        """After a collective of the open step; ``token`` is what ``collective_begin`` returned."""
        if not self._open:
            return
        counts = self._counts
        counts[name] = counts.get(name, 0) + 1
        if tensor is not None:
            try:
                size = tensor.numel() * tensor.element_size()
                counts[name + "_bytes"] = counts.get(name + "_bytes", 0) + size
            except Exception:  # noqa: BLE001 - sizes are a courtesy
                pass
        if self.mode == MODE_SYNC and self._runtime is not None:
            self._device_wait()
            now = self._clock()
            collective = now - self._t_last
            self._spans["seg_collective"] = self._spans.get("seg_collective", 0) + collective
            self._t_last = now
            self._segments.append((name, token / 1e6, collective / 1e6))
            return
        self._nested["collective_host"] = self._nested.get("collective_host", 0) + self._clock() - token

    def device_wait(self, span: str) -> None:
        """Mode 3 only: wait for everything launched so far and give the wait to ``span``."""
        if self._open and self.mode == MODE_SYNC and self._runtime is not None:
            self._device_wait()
            self.lap(span)

    def _device_wait(self) -> None:
        htorch, torch = self._runtime
        htorch.core.mark_step()
        torch.hpu.synchronize()

    def ops(self, name: str):
        """A context manager that counts the ops dispatched inside it (mode 4); otherwise it does nothing."""
        if self._open and self.mode == MODE_OPS:
            return _OpCountContext(self, name)
        return _NULL_CONTEXT

    # ------------------------------------------------------------------------- mode 2: sync counters
    def _sync_point(self, label: str, depth: int = 2) -> None:
        counts = self._counts
        counts[label] = counts.get(label, 0) + 1
        try:
            frame = sys._getframe(depth)
            site = (label, "%s:%d" % (os.path.basename(frame.f_code.co_filename), frame.f_lineno))
        except Exception:  # noqa: BLE001 - the count stands without its source line
            site = (label, "?")
        self._sites[site] = self._sites.get(site, 0) + 1

    @staticmethod
    def _on_device(tensor) -> bool:
        return tensor.device.type != "cpu"

    def _install_counters(self) -> None:
        if self._patches:
            return
        try:
            import torch
        except Exception:  # noqa: BLE001 - no torch, no counters
            return
        timer = self
        tensor_class = torch.Tensor

        def patch(owner, name, wrapper_of):
            saved = owner.__dict__.get(name, _MISSING) if isinstance(owner, type) else getattr(owner, name, _MISSING)
            original = getattr(owner, name, None)
            if original is None:
                return
            try:
                setattr(owner, name, wrapper_of(original))
            except Exception:  # noqa: BLE001 - an attribute that cannot be patched is not counted
                return
            self._patches.append((owner, name, saved))

        def to_host(label):

            def wrapper_of(original):

                def wrapper(self, *args, **kwargs):
                    if timer.active and timer._on_device(self):
                        timer._sync_point(label)
                    return original(self, *args, **kwargs)

                return wrapper

            return wrapper_of

        def to_wrapper_of(original):

            def wrapper(self, *args, **kwargs):
                out = original(self, *args, **kwargs)
                if timer.active:
                    source, target = timer._on_device(self), timer._on_device(out)
                    if source != target:
                        timer._sync_point("h2d" if target else "d2h_to")
                return out

            return wrapper

        def copy_wrapper_of(original):

            def wrapper(self, source, *args, **kwargs):
                if timer.active and isinstance(source, tensor_class):
                    target_side, source_side = timer._on_device(self), timer._on_device(source)
                    if target_side != source_side:
                        timer._sync_point("h2d" if target_side else "d2h_copy")
                return original(self, source, *args, **kwargs)

            return wrapper

        def counted(label):

            def wrapper_of(original):

                def wrapper(*args, **kwargs):
                    if timer.active:
                        timer._sync_point(label)
                    return original(*args, **kwargs)

                return wrapper

            return wrapper_of

        for name, label in (("item", "item"), ("cpu", "cpu"), ("tolist", "tolist"), ("numpy", "numpy"),
                            ("__bool__", "scalar"), ("__int__", "scalar"), ("__float__", "scalar"), ("__index__",
                                                                                                    "scalar")):
            patch(tensor_class, name, to_host(label))
        patch(tensor_class, "to", to_wrapper_of)
        patch(tensor_class, "copy_", copy_wrapper_of)
        if self._runtime is not None:
            htorch, runtime_torch = self._runtime
            core = getattr(htorch, "core", None)
            if core is not None:
                patch(core, "mark_step", counted("mark_step"))
            hpu = getattr(runtime_torch, "hpu", None)
            if hpu is not None:
                patch(hpu, "synchronize", counted("synchronize"))

    def _uninstall_counters(self) -> None:
        for owner, name, saved in reversed(self._patches):
            try:
                if saved is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, saved)
            except Exception:  # noqa: BLE001 - leave what cannot be restored
                pass
        self._patches = []

    # ---------------------------------------------------------------------------------------- output
    def _on_idle(self) -> None:
        """Between two bursts of steps: write what is collected and reread the mode file."""
        self.flush()
        try:
            with open(os.path.join(self.out_dir, MODE_FILE), encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return
        self.set_mode(parse_mode(text))

    def flush(self) -> None:
        for key in list(self._windows):
            self._flush_window(key)

    def close(self) -> None:
        self.flush()
        if self._patches:
            self._uninstall_counters()

    def _flush_window(self, key) -> None:
        window = self._windows.pop(key, None)
        if window is None or window.steps == 0:
            return
        record = self.window_record(window)
        self._say(self.format_line(record))
        if self._write_failed:
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            self._write_failed = True
            self._say("step timing: cannot write %s (%s); the log lines continue" % (self.path, exc))

    def window_record(self, window: _Window) -> dict:
        """The JSON record of one window: the statistics per series and the raw values."""
        spans, nested, counters, raw = {}, {}, {}, {}
        for name, column in window.series.items():
            raw[name] = [None if v is None else round(v, 4) for v in column]
            summary = stats(column)
            if summary is None:
                continue
            if name.startswith("n:"):
                counters[name[2:]] = summary
            elif name.startswith("in:"):
                nested[name[3:]] = summary
            else:
                spans[name] = summary
        record = {
            "schema": SCHEMA,
            "role": self.role,
            "rank": self.rank,
            "pid": self.pid,
            "mode": window.mode,
            "kind": window.kind,
            "rows": window.rows,
            "bucket_rows": window.bucket_rows,
            "steps": window.steps,
            "t_first": round(window.t_first, 6),
            "t_last": round(window.t_last, 6),
            "spans_ms": spans,
            "nested_ms": nested,
            "counters": counters,
            "raw": raw,
        }
        if window.segments:
            record["segments"] = self._segment_record(window.segments)
        if window.sites:
            top = sorted(window.sites.items(), key=lambda item: -item[1])[:MAX_SITES]
            record["sync_sites"] = [{
                "what": label,
                "where": where,
                "per_step": n / window.steps
            } for (label, where), n in top]
        if window.op_names:
            top = sorted(window.op_names.items(), key=lambda item: -item[1])[:MAX_OP_NAMES]
            record["op_names_per_step"] = {name: n / window.steps for name, n in top}
        return record

    @staticmethod
    def _segment_record(per_step: list) -> dict:
        """Mode 3: per collective position, the median compute time in front of it and its own median time."""
        length = max(len(step) for step in per_step)
        names, compute, collective = [], [], []
        for index in range(length):
            rows = [step[index] for step in per_step if len(step) > index]
            names.append(rows[-1][0])
            compute.append(round(percentile([row[1] for row in rows], 0.50), 4))
            collective.append(round(percentile([row[2] for row in rows], 0.50), 4))
        return {"names": names, "compute_p50_ms": compute, "collective_p50_ms": collective, "steps": len(per_step)}

    def format_line(self, record: dict) -> str:
        """The log line of one window: p50/p90 in milliseconds per span, in the order of the step."""
        spans = record["spans_ms"]

        def pair(name):
            summary = spans.get(name)
            return None if summary is None else "%s %.2f/%.2f" % (name, summary["p50"], summary["p90"])

        order = WORKER_SPANS if self.role == "worker" else ENGINE_SPANS
        head = [pair(name) for name in ("period", "step", "gap")]
        body = [pair(name) for name in order]
        body += [pair(name) for name in sorted(spans) if name not in order and name not in ("period", "step", "gap")]
        extras = []
        for name, summary in sorted(record["nested_ms"].items()):
            extras.append("%s %.2f/%.2f" % (name, summary["p50"], summary["p90"]))
        for name, summary in sorted(record["counters"].items()):
            extras.append("%s %g" % (name, summary["p50"]))
        bucket = "" if record["bucket_rows"] is None else " bucket=%d" % record["bucket_rows"]
        return "step timing: %s rank %d %s rows=%d%s steps=%d mode=%d | %s | %s%s (ms, p50/p90)" % (
            self.role, self.rank, record["kind"], record["rows"], bucket, record["steps"], record["mode"], " ".join(
                part for part in head if part), " ".join(part for part in body if part),
            " | " + " ".join(extras) if extras else "")


def ops(timer: StepTimer | None, name: str):
    """``timer.ops(name)``, or a context manager that does nothing when there is no timer."""
    return _NULL_CONTEXT if timer is None else timer.ops(name)


def env_settings(environ=None) -> tuple[int, int, str]:
    """``(mode, every, directory)`` from the three variables; the plugin's ``envs`` module documents them."""
    environ = os.environ if environ is None else environ
    mode = parse_mode(environ.get("VLLM_HPU_STEP_TIMING", "0"))
    try:
        every = max(1, int(environ.get("VLLM_HPU_STEP_TIMING_EVERY", "100")))
    except ValueError:
        every = 100
    return mode, every, environ.get("VLLM_HPU_STEP_TIMING_DIR", "") or "."


def init_timer(role: str, rank: int = 0, log=None) -> StepTimer | None:
    """Create this process's timer from the environment. Returns ``None``, and leaves ``TIMER`` ``None``, when
    ``VLLM_HPU_STEP_TIMING`` is 0: nothing is timed, patched or written."""
    global TIMER
    mode, every, out_dir = env_settings()
    if TIMER is not None:
        TIMER.close()
        TIMER = None
    if mode == MODE_OFF:
        return None
    TIMER = StepTimer(role, rank=rank, mode=mode, every=every, out_dir=out_dir, log=log)
    atexit.register(TIMER.close)
    if log is not None:
        log("step timing: on, mode %d, every %d steps, %s" % (mode, every, TIMER.path))
    return TIMER


# ------------------------------------------------------------------------------------- engine core
def _engine_kind(scheduler_output) -> tuple[str, int, int] | None:
    """``(kind, rows, tokens)`` of a scheduler output as the engine sees it, or ``None`` for an empty one."""
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    rows = len(scheduled)
    if tokens <= 0 or rows == 0:
        return None
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        kind = "verify"
    elif tokens == rows:
        kind = "decode"
    else:
        kind = "prefill"
    return kind, rows, tokens


def instrument_engine_core(engine_core, timer: StepTimer) -> None:
    """Wrap the four calls an engine step makes, on this engine core object only."""
    scheduler, executor = engine_core.scheduler, engine_core.model_executor

    def timed(owner, name, span, before=None):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            timer.lap("other")
            if before is not None:
                before(*args, **kwargs)
            out = original(*args, **kwargs)
            timer.lap(span)
            return out

        setattr(owner, name, wrapper)

    def rpc(name):
        original = getattr(executor, name)

        def wrapper(*args, **kwargs):
            timer.lap("other")
            if name == "execute_model" and args:
                described = _engine_kind(args[0])
                if described is not None:
                    timer.tag(described[0], described[1], described[2])
            out = original(*args, **kwargs)
            if kwargs.get("non_block") or (len(args) > 1 and args[1]):
                timer.lap(name + "_send")
                result = getattr(out, "result", None)
                if result is not None:

                    def waited(*result_args, **result_kwargs):
                        timer.lap("other")
                        value = result(*result_args, **result_kwargs)
                        timer.lap(name + "_wait")
                        return value

                    with contextlib.suppress(Exception):
                        out.result = waited
            else:
                timer.lap(name + "_rpc")
            return out

        setattr(executor, name, wrapper)

    timed(scheduler, "schedule", "schedule")
    timed(scheduler, "update_from_output", "update_from_output")
    rpc("execute_model")
    rpc("sample_tokens")


def install_engine_step_timing(log=None, make_timer=None) -> bool:
    """Time ``EngineCore.step`` and ``step_with_batch_queue`` in the process that runs the engine core.

    Called from the plugin's general-plugin hook, which runs at the start of ``EngineCore.__init__``. Does nothing
    when the flag is 0 or the engine core module is not loaded in this process. Returns True when installed.
    ``make_timer(mode, every, out_dir)`` replaces the timer's constructor (tests)."""
    mode, _every, _out_dir = env_settings()
    if mode == MODE_OFF:
        return False
    module = sys.modules.get("vllm.v1.engine.core")
    engine_class = getattr(module, "EngineCore", None)
    if engine_class is None:
        return False
    if getattr(engine_class, "_hpu_step_timing_installed", False):
        return True

    def wrap(method_name):
        original = getattr(engine_class, method_name, None)
        if original is None:
            return

        def step(self, *args, **kwargs):
            timer = self.__dict__.get("_hpu_step_timer", _MISSING)
            if timer is _MISSING:
                timer = None
                try:
                    if make_timer is not None:
                        timer = make_timer(mode, _every, _out_dir)
                    else:
                        timer = StepTimer("engine", rank=0, mode=mode, every=_every, out_dir=_out_dir, log=log)
                    atexit.register(timer.close)
                    instrument_engine_core(self, timer)
                except Exception as exc:  # noqa: BLE001 - the engine must run without its timer
                    timer = None
                    if log is not None:
                        log("step timing: the engine core is not timed (%s: %s)" % (type(exc).__name__, exc))
                self.__dict__["_hpu_step_timer"] = timer
            if timer is None:
                return original(self, *args, **kwargs)
            timer.begin_step()
            try:
                return original(self, *args, **kwargs)
            finally:
                timer.end_step("other")

        setattr(engine_class, method_name, step)

    wrap("step")
    wrap("step_with_batch_queue")
    engine_class._hpu_step_timing_installed = True
    return True
