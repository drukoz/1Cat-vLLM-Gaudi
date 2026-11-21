# SPDX-License-Identifier: Apache-2.0
"""Persistent host and device buffers for a decode step's inputs (``VLLM_HPU_PERSISTENT_INPUTS=1``).

``async_h2d_copy`` makes a fresh device tensor for every input of every step and copies into it from an unpinned
host tensor: per call a device allocation, and in the bridge a staging allocation from its pinned pool plus a memcpy
(native-replay, from the bridge source). The forward's inputs then live at new addresses every step, and the HPU graph
replay re-patches them; native-replay's static launch plan (lean5) only hits when every input of a segment has the
address it had two launches before.

Here every input (by name, shape and dtype) gets a pinned host tensor and a device tensor once. Each step the host
fills its tensor with CPU ops and issues one ``copy_(non_blocking=True)`` into the device tensor. The runner calls
``begin_step`` once per engine step. Within a step every upload of an input gets its own pair (the second upload of
the same name and shape in one step, e.g. two prefill chunks of one bucket, does not rewrite the first one's host
buffer or device tensor while the first copy or forward may still read them). With synchronous stepping the next
step reuses the pairs (step N's device work has finished when the runner reads its sampled tokens); under async
scheduling two step parities alternate, so a copy never rewrites what the step before may still read, and the
address seen two launches back is the same one (lean5's ring of two hits either way).

Contract (async-sched, 11:05 UTC): an uploaded tensor is rewritten by the same input's upload in the next step
(synchronous) or two steps later (async scheduling). Code that keeps a step's input tensor and reads it in a later
step (a device-side advance of positions or ids, a carried PLE context, a chained draft) must keep its own copy.

Nothing else changes: the same values, dtypes and shapes as ``async_h2d_copy``.
"""

from __future__ import annotations

import numpy as np
import torch

MAX_KEYS = 4096  # (name, shape, dtype, parity, call) pairs kept; beyond it everything is dropped and made again


class PersistentInputs:
    """``upload(name, source, dtype)`` returns the step's device tensor for one input."""

    def __init__(self, device, pin_memory: bool, buffers: int = 1):
        if buffers not in (1, 2):
            raise ValueError(f"buffers must be 1 or 2, got {buffers}")
        self.device = device
        self.pin_memory = bool(pin_memory)
        self.buffers = buffers
        self._parity = 0
        self._calls: dict = {}
        self._host: dict = {}
        self._dev: dict = {}

    def begin_step(self) -> None:
        """A new engine step: its uploads take their pairs from the start again (the other parity under async
        scheduling)."""
        self._calls.clear()
        self._parity = (self._parity + 1) % self.buffers

    @staticmethod
    def _as_host(source, dtype) -> torch.Tensor:
        if isinstance(source, torch.Tensor):
            if source.device.type != "cpu":
                raise ValueError("an input to upload must be on the CPU")
            return source if dtype is None or source.dtype == dtype else source.to(dtype)
        if isinstance(source, np.ndarray):
            host = torch.from_numpy(source)
            return host if dtype is None or host.dtype == dtype else host.to(dtype)
        if dtype is None:
            raise ValueError("dtype must be specified when source is not a tensor")
        return torch.tensor(source, dtype=dtype, device="cpu")

    def upload(self, name: str, source, dtype=None) -> torch.Tensor:
        host = self._as_host(source, dtype)
        key = (name, tuple(host.shape), host.dtype)
        call = self._calls.get(key, 0)
        self._calls[key] = call + 1
        pair = key + (self._parity, call)
        staging = self._host.get(pair)
        if staging is None:
            if len(self._host) >= MAX_KEYS:
                self._host.clear()
                self._dev.clear()
            staging = torch.empty(host.shape, dtype=host.dtype, device="cpu", pin_memory=self.pin_memory)
            self._host[pair] = staging
            self._dev[pair] = torch.empty(host.shape, dtype=host.dtype, device=self.device)
        staging.copy_(host)
        dev = self._dev[pair]
        dev.copy_(staging, non_blocking=True)
        return dev

    def __len__(self) -> int:
        return len(self._dev)


__all__ = ["PersistentInputs"]
