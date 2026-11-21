# SPDX-License-Identifier: Apache-2.0
"""Where the HPU worker processes run on the host (``VLLM_HPU_WORKER_CPU_PIN``).

A tensor-parallel decode step ends when the slowest rank has issued its recipe launches and collectives: every
other rank waits for it in its own collectives. The workers are ordinary processes, so by default the kernel
moves their threads over every CPU the server may use, across NUMA nodes and onto CPUs another worker is using.
On the card the same replayed step costs one rank 19 ms of host time and another 32 ms, and the slow rank stays
slow for a whole round and changes between rounds (docs/design/flash-next/native-replay.md).

With the flag set, each worker keeps to its own share of the CPUs the server process may use
(``os.sched_getaffinity``, so a container's CPU set is respected): the sorted CPUs are cut into equal shares, one
per visible module (``HABANA_VISIBLE_MODULES``, ascending; rank ``r`` drives the ``r``-th of them), and a worker
takes the share at its module's position (``HLS_MODULE_ID``, written by ``torch.hpu.set_device``); without those
two variables, the share at its rank of ``world``. A container whose CPU set was cut by module (the card farm:
the CPUs next to each module) then gives every worker the CPUs next to its card.

Mode 1 moves every thread the worker has at that moment to the share; threads it starts later inherit it. Mode 2
does the same and, once the model is loaded and warmed up, gives the thread that runs the model (the calling
thread) the first CPU of the share for itself and moves the other threads to the rest. A thread the calling
thread starts after that inherits its single CPU (a compile pool on first use when the warmup was skipped), so a
small daemon thread looks for such threads and moves them to the rest of the share too.

Mode 3 is for a host whose hyperthreads the VM hides: the rented node pairs vCPUs 2k and 2k+1 on one host core
(design document, section 7.12), and two busy threads on such a pair slow each other down by up to 2x. Nothing is
pinned while the model loads, compiles and warms up (a pin there made the server start about 7 minutes slower in
job 00189); once warmed up, the model thread gets one CPU of the worker's share whose sibling
(``VLLM_HPU_WORKER_CPU_SIBLINGS``: ``xor1`` = 2k with 2k+1, the default; ``none``: only its own CPU) is kept free,
the worker's other threads keep to the rest of the share, the parent process's threads (the engine core) keep off
both CPUs, and the daemon thread keeps later threads of both processes off them.
Nothing here changes what a worker computes.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

PIN_OFF = 0
PIN_SHARE = 1
PIN_SHARE_AND_OWN_CPU = 2
PIN_SHARE_AND_OWN_CORE = 3


def share_position(rank: int, world: int, module=None, visible=None) -> tuple[int, int]:
    """(which share, how many shares) for this worker.

    ``module`` and ``visible`` are the strings of ``HLS_MODULE_ID`` and ``HABANA_VISIBLE_MODULES``: when both are
    set and the module is one of the visible ones, the share is the module's position among the visible modules
    in ascending order, out of as many shares as there are visible modules. Otherwise the rank's, out of ``world``.
    """
    try:
        modules = sorted({int(m) for m in str(visible).split(",") if m.strip()}) if visible else []
        own = int(module) if module not in (None, "") else None
    except ValueError:
        modules, own = [], None
    if own is not None and own in modules:
        return modules.index(own), len(modules)
    return rank, world


def worker_cpu_share(allowed, index: int, count: int) -> list[int]:
    """The ``index``-th of ``count`` equal shares of the sorted CPUs ``allowed``.

    The CPUs left over by the division stay with nobody (the engine, the API server and the kernel use them).
    Empty when the index is not one of ``count`` or there are fewer CPUs than shares: then nothing is pinned.
    """
    cpus = sorted(set(allowed))
    if count < 1 or not 0 <= index < count or len(cpus) < count:
        return []
    size = len(cpus) // count
    return cpus[index * size:(index + 1) * size]


def format_cpus(cpus) -> str:
    """``[0, 1, 2, 5, 7, 8]`` -> ``"0-2,5,7-8"``."""
    cpus = sorted(set(cpus))
    parts, start = [], None
    for i, cpu in enumerate(cpus):
        if start is None:
            start = cpu
        if i + 1 == len(cpus) or cpus[i + 1] != cpu + 1:
            parts.append(str(start) if start == cpu else "%d-%d" % (start, cpu))
            start = None
    return ",".join(parts)


def _thread_ids() -> list[int]:
    """The ids of every thread of this process (Linux: /proc/self/task)."""
    try:
        return sorted(int(task.name) for task in Path("/proc/self/task").iterdir())
    except OSError:
        return []


def pin_worker(index: int, count: int, mode: int, log=None, *, label: str = "", getaffinity=None,
               setaffinity=None, thread_ids=None) -> list[int]:
    """Move every thread of this worker process to share ``index`` of ``count``; returns the share ([]: nothing
    done).

    The three keyword functions are the operating system's (``os.sched_getaffinity``, ``os.sched_setaffinity``,
    the thread list); tests pass their own. On a platform without them nothing is done.
    """
    if mode == PIN_OFF:
        return []
    getaffinity = getaffinity or getattr(os, "sched_getaffinity", None)
    setaffinity = setaffinity or getattr(os, "sched_setaffinity", None)
    thread_ids = thread_ids or _thread_ids
    if getaffinity is None or setaffinity is None:
        if log is not None:
            log.warning("VLLM_HPU_WORKER_CPU_PIN=%d: this platform has no sched_setaffinity, nothing is pinned", mode)
        return []
    allowed = sorted(getaffinity(0))
    share = worker_cpu_share(allowed, index, count)
    if not share:
        if log is not None:
            log.warning("VLLM_HPU_WORKER_CPU_PIN=%d: %sshare %d of %d gets no CPUs of the %d this server may use, "
                        "nothing is pinned", mode, label, index, count, len(allowed))
        return []
    moved = 0
    for tid in thread_ids() or [0]:
        try:
            setaffinity(tid, share)
            moved += 1
        except OSError:  # a thread that ended meanwhile, or one the kernel will not move: it stays where it is
            pass
    if log is not None:
        log.info("VLLM_HPU_WORKER_CPU_PIN=%d: %sshare %d of %d, CPUs %s (%d of the %d this server may use; "
                 "%d threads moved)", mode, label, index, count, format_cpus(share), len(share), len(allowed), moved)
    return share


def pin_from_env(rank: int, world: int, mode: int, log=None, *, environ=None, **os_functions) -> list[int]:
    """``pin_worker`` for the worker that runs here, after ``torch.hpu.set_device``: the share at its module's
    position among the visible modules (``share_position``); returns the share ([]: nothing done)."""
    if mode == PIN_OFF:
        return []
    environ = os.environ if environ is None else environ
    module, visible = environ.get("HLS_MODULE_ID"), environ.get("HABANA_VISIBLE_MODULES")
    index, count = share_position(rank, world, module, visible)
    label = "rank %d module %s: " % (rank, module if module not in (None, "") else "?")
    if mode == PIN_SHARE_AND_OWN_CORE:
        # mode 3: the share is only computed here; the worker keeps every CPU through load and warmup
        getaffinity = os_functions.get("getaffinity") or getattr(os, "sched_getaffinity", None)
        if getaffinity is None:
            return []
        allowed = sorted(getaffinity(0))
        share = worker_cpu_share(allowed, index, count)
        if log is not None:
            log.info("VLLM_HPU_WORKER_CPU_PIN=3: %sshare %d of %d, CPUs %s, placed after warmup (nothing pinned "
                     "before)", label, index, count, format_cpus(share) if share else "none")
        return share
    return pin_worker(index, count, mode, log, label=label, **os_functions)


def rehome_new_threads(own: int, rest, me: int, *, getaffinity, setaffinity, thread_ids) -> int:
    """Move every thread but ``me`` that is confined to the calling thread's CPU ``own`` (it inherited it) to
    ``rest``; returns how many were moved."""
    moved = 0
    for tid in thread_ids():
        if tid == me:
            continue
        try:
            if set(getaffinity(tid)) == {own}:
                setaffinity(tid, rest)
                moved += 1
        except OSError:
            pass
    return moved


def _rehome_forever(own: int, rest, me: int, log=None) -> None:
    """The daemon thread of mode 2: every second for two minutes, then every five seconds."""
    os.sched_setaffinity(0, rest)  # this thread inherited the calling thread's CPU too
    started, total = time.monotonic(), 0
    while True:
        time.sleep(1.0 if time.monotonic() - started < 120 else 5.0)
        moved = rehome_new_threads(own, rest, me, getaffinity=os.sched_getaffinity,
                                   setaffinity=os.sched_setaffinity, thread_ids=_thread_ids)
        if moved and log is not None:
            total += moved
            log.info("VLLM_HPU_WORKER_CPU_PIN=2: %d new threads moved off the calling thread's CPU %d (%d so far)",
                     moved, own, total)


def sibling_cpu(cpu: int, rule: str) -> int | None:
    """The CPU that shares ``cpu``'s host core under ``rule`` (``xor1``: 2k and 2k+1); None for ``none``."""
    return cpu ^ 1 if rule == "xor1" else None


def own_core(share, rule: str, allowed=None) -> tuple[int, int | None, list[int]] | None:
    """Mode 3: (the model thread's CPU, its sibling to keep free or None, the CPUs for the other threads).

    The share's second CPU first (the first takes the share's interrupts first), then the others; a CPU qualifies
    when its sibling is in the share (kept free there) or outside the CPUs the server may use. None: the share is
    too small."""
    share = list(share)
    allowed = set(share if allowed is None else allowed)
    for cpu in share[1:] + share[:1]:
        sib = sibling_cpu(cpu, rule)
        if sib is not None and sib != cpu and sib not in share and sib in allowed:
            continue  # its sibling belongs to another worker's share: not free
        rest = [c for c in share if c not in (cpu, sib)]
        if rest:
            return cpu, (sib if sib in share else None), rest
    return None


def trim_threads(reserved, thread_ids, *, getaffinity, setaffinity) -> int:
    """Keep every thread of ``thread_ids`` off the CPUs ``reserved`` (when it may run elsewhere too); returns how
    many were moved."""
    moved = 0
    reserved = set(reserved)
    for tid in thread_ids:
        try:
            cur = set(getaffinity(tid))
            if cur & reserved and cur - reserved:
                setaffinity(tid, sorted(cur - reserved))
                moved += 1
        except OSError:
            pass
    return moved


def _process_thread_ids(pid: int) -> list[int]:
    try:
        return sorted(int(task.name) for task in Path("/proc/%d/task" % pid).iterdir())
    except OSError:
        return []


def _keep_core_free_forever(own: int, reserved, rest, me: int, parent: int, log=None) -> None:
    """The daemon thread of mode 3: the worker's later threads off ``own`` and its sibling, the parent's off both;
    every second for two minutes, then every five seconds."""
    os.sched_setaffinity(0, rest)
    started, total = time.monotonic(), 0
    while True:
        time.sleep(1.0 if time.monotonic() - started < 120 else 5.0)
        moved = rehome_new_threads(own, rest, me, getaffinity=os.sched_getaffinity, setaffinity=os.sched_setaffinity,
                                   thread_ids=_thread_ids)
        moved += trim_threads(reserved, [t for t in _thread_ids() if t != me], getaffinity=os.sched_getaffinity,
                              setaffinity=os.sched_setaffinity)
        moved += trim_threads(reserved, _process_thread_ids(parent), getaffinity=os.sched_getaffinity,
                              setaffinity=os.sched_setaffinity)
        if moved and log is not None:
            total += moved
            log.info("VLLM_HPU_WORKER_CPU_PIN=3: %d threads moved off CPUs %s (%d so far)", moved,
                     format_cpus(reserved), total)


def give_calling_thread_its_own_core(share, mode: int, log=None, *, rule: str = "xor1", allowed=None,
                                     getaffinity=None, setaffinity=None, thread_ids=None, parent_thread_ids=None,
                                     calling_thread=None, rehome=True) -> int | None:
    """Mode 3, after load and warmup: the calling thread alone on a CPU whose host sibling nobody of this worker or
    its parent uses (``own_core``). Returns the calling thread's CPU (None: nothing done)."""
    if mode != PIN_SHARE_AND_OWN_CORE or len(share) < 2:
        return None
    injected = setaffinity is not None
    getaffinity = getaffinity or getattr(os, "sched_getaffinity", None)
    setaffinity = setaffinity or getattr(os, "sched_setaffinity", None)
    if getaffinity is None or setaffinity is None:
        return None
    thread_ids = thread_ids or _thread_ids
    parent = os.getppid()
    parent_thread_ids = parent_thread_ids or (lambda: _process_thread_ids(parent))
    if allowed is None:
        try:  # the parent (the engine core) still has the server's whole CPU set; this thread has only its share
            allowed = getaffinity(parent) if not injected else share
        except OSError:
            allowed = share
    picked = own_core(share, rule, allowed)
    if picked is None:
        return None
    own, sib, rest = picked
    reserved = [own] + ([sib] if sib is not None else [])
    me = calling_thread if calling_thread is not None else getattr(os, "gettid", os.getpid)()
    moved = 0
    for tid in thread_ids():
        if tid == me:
            continue
        try:
            setaffinity(tid, rest)
            moved += 1
        except OSError:
            pass
    try:
        setaffinity(0, [own])
    except OSError as exc:
        if log is not None:
            log.warning("VLLM_HPU_WORKER_CPU_PIN=3: the calling thread keeps the whole share (%s)", exc)
        return None
    trimmed = trim_threads(reserved, parent_thread_ids(), getaffinity=getaffinity, setaffinity=setaffinity)
    if log is not None:
        log.info("VLLM_HPU_WORKER_CPU_PIN=3: the calling thread has CPU %d for itself (sibling %s kept free, rule "
                 "%s), %d other threads keep to CPUs %s, %d parent threads moved off", own,
                 sib if sib is not None else "-", rule, moved, format_cpus(rest), trimmed)
    if rehome and not injected:
        threading.Thread(target=_keep_core_free_forever, args=(own, reserved, rest, me, parent, log),
                         name="hpu-worker-cpu-rehome", daemon=True).start()
    return own


def give_calling_thread_its_own_cpu(share, mode: int, log=None, *, setaffinity=None, thread_ids=None,
                                    calling_thread=None, rehome=True) -> int | None:
    """Mode 2, after load and warmup: the calling thread alone on the share's first CPU, every other thread of
    the process on the rest. Returns the calling thread's CPU (None: nothing done). ``rehome`` starts the daemon
    thread that keeps later threads off that CPU (tests pass False)."""
    if mode != PIN_SHARE_AND_OWN_CPU or len(share) < 2:
        return None
    injected = setaffinity is not None
    setaffinity = setaffinity or getattr(os, "sched_setaffinity", None)
    thread_ids = thread_ids or _thread_ids
    if setaffinity is None:
        return None
    me = calling_thread if calling_thread is not None else getattr(os, "gettid", os.getpid)()
    own, rest = share[0], list(share[1:])
    moved = 0
    for tid in thread_ids():
        if tid == me:
            continue
        try:
            setaffinity(tid, rest)
            moved += 1
        except OSError:
            pass
    try:
        setaffinity(0, [own])
    except OSError as exc:
        if log is not None:
            log.warning("VLLM_HPU_WORKER_CPU_PIN=2: the calling thread keeps the whole share (%s)", exc)
        return None
    if log is not None:
        log.info("VLLM_HPU_WORKER_CPU_PIN=2: the calling thread has CPU %d for itself, %d other threads keep to "
                 "CPUs %s", own, moved, format_cpus(rest))
    if rehome and not injected:
        threading.Thread(target=_rehome_forever, args=(own, rest, me, log), name="hpu-worker-cpu-rehome",
                         daemon=True).start()
    return own
