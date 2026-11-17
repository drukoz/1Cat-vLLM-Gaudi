# SPDX-License-Identifier: Apache-2.0
"""Eight-rank CPU tensor-parallel harness (gloo) for the Gaudi plugin.

Runs a callable on 1/2/4/8 spawned CPU processes wired through vLLM's own
distributed init (`init_distributed_environment` + `initialize_model_parallel`
with the `gloo` backend), so TP8 layer checks can be proven on this Mac before
renting Gaudi hardware time. Public API:

- `run_ranks(world_size, fn, *args, timeout_s=120) -> list`
- `record_collectives()`  context manager recording TP collective calls
"""

import os
import shutil
import tempfile
import time
import traceback
from contextlib import contextmanager
from queue import Empty

import torch.multiprocessing as mp

# The four device-communicator entry points the TP layers use.
_COLLECTIVE_METHODS = ("all_reduce", "all_gather", "reduce_scatter", "gather")


def _rank_main(world_size, rank, init_method, fn, args, queue):
    """Child entry point: bring up vLLM TP state, run fn(rank, *args), report."""
    try:
        import torch

        # Ranks are small CPU workers; capping intra-op threads stops N ranks
        # from oversubscribing the machine during import and forward.
        torch.set_num_threads(1)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )

        # initialize_model_parallel() and the TP layers read the current
        # vLLM config; a plain VllmConfig() (no model) builds fine at the pin.
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(world_size, rank, init_method, rank, "gloo")
            initialize_model_parallel(world_size)
            result = fn(rank, *args)
            queue.put((rank, "ok", result))
            destroy_model_parallel()
            destroy_distributed_environment()
    except BaseException:  # noqa: BLE001 - forward any traceback to the parent
        queue.put((rank, "error", traceback.format_exc()))


def run_ranks(world_size, fn, *args, timeout_s: float = 120.0):
    """Run `fn(rank, *args)` on `world_size` gloo ranks; return results in rank order.

    Each rank is a `torch.multiprocessing` (spawn) process that initializes
    vLLM's distributed + model-parallel state on the `gloo` backend, calls the
    function, and sends its return value back through a queue.

    Raises `RuntimeError` (carrying the failing rank's traceback text) if any
    rank raises, and `TimeoutError` (after terminating every rank) if any rank
    does not answer within `timeout_s`. A mismatched collective can hang gloo,
    which is exactly what the timeout protects against.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")

    # Unix domain sockets (torch FileStore, torch's shm manager, and vLLM's
    # ZMQ message-queue broadcaster) are limited to ~104 path bytes on macOS,
    # so all rendezvous files must live in a short directory, not the (long)
    # sandbox TMPDIR. Children inherit these settings through the environment.
    short_dir = "/tmp"
    if len(tempfile.gettempdir()) > 60:
        os.environ["TMPDIR"] = short_dir
    os.environ.setdefault("VLLM_RPC_BASE_PATH", short_dir)
    # This is a CPU harness: `VLLM_PLUGINS=""` keeps vLLM from auto-loading the
    # hpu platform plugin (which drags in the habana stack and slows every
    # spawned rank's import of vllm). Children inherit all of these settings.
    os.environ.setdefault("VLLM_PLUGINS", "")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    tmpdir = tempfile.mkdtemp(prefix="cpu-tp-", dir=short_dir)
    init_method = "file://" + os.path.join(tmpdir, "store")

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_rank_main,
            args=(world_size, rank, init_method, fn, args, queue),
            daemon=False,
        )
        p.start()
        procs.append(p)

    def _terminate_all():
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=10)

    try:
        results = {}
        deadline = time.monotonic() + timeout_s
        while len(results) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_all()
                raise TimeoutError(f"ranks did not all answer within {timeout_s}s "
                                   f"(got {len(results)}/{world_size})")
            try:
                rank, kind, payload = queue.get(timeout=remaining)
            except Empty:
                _terminate_all()
                raise TimeoutError(f"ranks did not all answer within {timeout_s}s "
                                   f"(got {len(results)}/{world_size})")
            if kind == "error":
                _terminate_all()
                raise RuntimeError(f"rank {rank} raised:\n{payload}")
            results[rank] = payload

        for p in procs:
            p.join(timeout=30)
        return [results[r] for r in range(world_size)]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextmanager
def record_collectives():
    """Record `(method, input_shape, input_dtype)` for TP collectives issued inside.

    Wraps the four `DeviceCommunicatorBase` methods (`all_reduce`,
    `all_gather`, `reduce_scatter`, `gather`) on the live tensor-parallel group's
    device communicator, appending one tuple per call to the list this context
    manager yields, and restores the original methods on exit.
    """
    from vllm.distributed.parallel_state import get_tp_group

    comm = get_tp_group().device_communicator
    cls = type(comm)
    recorded = []
    originals = {}

    def _make(name, original):

        def wrapper(self, input_, *a, **kw):
            recorded.append((name, tuple(input_.shape), str(input_.dtype)))
            return original(self, input_, *a, **kw)

        return wrapper

    for name in _COLLECTIVE_METHODS:
        original = getattr(cls, name, None)
        if original is None:
            continue
        originals[name] = original
        setattr(cls, name, _make(name, original))
    try:
        yield recorded
    finally:
        for name, original in originals.items():
            setattr(cls, name, original)
