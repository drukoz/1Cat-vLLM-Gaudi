# SPDX-License-Identifier: Apache-2.0
"""A CPU model of one card rule: no in-place write through a slice of a persistent tensor inside an HPU graph.

Card job 00021 (2026-11-20, the first prompt of a speculative server): the PLE layer took a slice of its conv
state (``view[:, :state_len, :]``, the state a speculative config widens) and wrote it with ``index_copy_``. The
lazy bridge turns an in-place write through a slice view into ``hpu::slice_insert`` on the base tensor, and the
HPU graph could not bind that node's output: ``RuntimeError: Neither storage attached to input tensor, not its
view`` on every rank. PyTorch on CPU runs the same code without a word, so the tests wrap a state tensor in
``SliceWriteProbe`` and assert that no write reached it through a slice:

* an in-place method (``index_copy_``, ``copy_``, ``zero_``, ...) whose target is a view with fewer elements than
  its base (a slice, a narrow, a select; a transpose or a reshape of the whole tensor is not one);
* an indexed assignment ``state[index] = value`` whose index holds an integer or a slice that does not cover its
  whole dimension (``state[rows, 0:3, :] = value`` on a wider state; ``state[rows, 0:3, :]`` on a state of
  exactly 3 rows is the non-speculative form, which PyTorch hands to ``index_put_`` on the tensor itself and
  which runs on the card).

Reading through a slice is fine, and so is slicing a copy (``state.index_select(0, rows)[:, :3, :]``).
"""

from __future__ import annotations

import torch

# In-place methods that end with an underscore but write nothing.
_NOT_A_WRITE = {"requires_grad_", "retain_grad", "share_memory_", "_is_view"}


def _covers_dimension(index: slice, size: int) -> bool:
    start, stop, step = index.indices(size)
    return start == 0 and stop == size and step == 1


def _index_is_partial(index, shape) -> bool:
    """Whether ``tensor[index] = value`` writes through a slice of the tensor (see the module docstring)."""
    items = index if isinstance(index, tuple) else (index, )
    dim = 0
    for item in items:
        if item is None:
            continue
        if item is Ellipsis:
            dim = len(shape) - sum(1 for later in items[items.index(item) + 1:] if later is not None)
            continue
        if isinstance(item, slice):
            if dim < len(shape) and not _covers_dimension(item, shape[dim]):
                return True
        elif isinstance(item, int):
            return True
        dim += 1
    return False


class SliceWriteProbe(torch.Tensor):
    """A state tensor that records every in-place write that reaches it through a slice of itself."""

    violations: list[str] = []

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "SliceWriteProbe":
        """``tensor`` as a probe (same storage); the record starts empty."""
        cls.violations = []
        return tensor.as_subclass(cls)

    @staticmethod
    def _is_slice_view(tensor: torch.Tensor) -> bool:
        base = tensor._base
        return base is not None and tensor.numel() < base.numel()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", "")
        target = args[0] if args else None
        if isinstance(target, cls):
            if name == "__setitem__":
                if cls._is_slice_view(target) or _index_is_partial(args[1], tuple(target.shape)):
                    cls.violations.append(f"indexed assignment through a slice, shape {tuple(target.shape)}")
            elif name.endswith("_") and not name.startswith("__") and name not in _NOT_A_WRITE:
                if cls._is_slice_view(target):
                    cls.violations.append(f"{name} on a slice view, shape {tuple(target.shape)} of "
                                          f"{tuple(target._base.shape)}")
        return super().__torch_function__(func, types, args, kwargs or {})
