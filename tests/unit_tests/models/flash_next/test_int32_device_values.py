# SPDX-License-Identifier: Apache-2.0
"""Whole-model guard: no int64 device value outside the int32 range (RUK-3101 B).

On the card (``PT_HPU_LAZY_MODE=1``) the lazy bridge stores every int64 device tensor as int32
and raises when a value falls outside ``[-2**31, 2**31 - 1]``. This is the failure that stopped
every worker on the Gaudi card in lazy mode during rental run 1, while building the model:

    File "vllm/models/qwen4_exp/nvidia/ple_layer.py", line 301, in __init__
        torch.tensor(multipliers, dtype=torch.long),
    RuntimeError: Error when trying to cast Long to Int, Input values range [8052911324071, 23703573157769]
    exceeds Int range

The CPU has no such bridge, so :class:`Int32RangeGuard` watches every aten op with a
``TorchDispatchMode`` and records the same failure here: any int64 result (not on ``meta``,
non-empty) whose ``[min, max]`` leaves the int32 range, plus every int64 buffer/parameter of the
built model holding an out-of-range value. The tests build the tiny TP8 model through the real
loader and run one prefill and one decode, and fail on this branch exactly where the card did;
they pass once the RUK-3101 fix (tasks A1/A2) is merged.

View-op rule: while the loader reads the checkpoint, view ops (``aten.view.dtype``,
``aten.detach``, ``aten.view``) on the checkpoint tensor are host-side reads of the file and must
not count, so an op is skipped when ``func.is_view`` is True -- except ``aten.lift_fresh.default``:
that is how ``torch.tensor(data)`` creates a tensor, and it reports ``is_view`` True. Copies
(``aten.copy_``, ``aten._to_copy``) and compute ops always count.
"""

import os
import traceback

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

from vllm.forward_context import set_forward_context  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, ple_kwargs, prefill_metadata)

INT32_MIN = -2**31
INT32_MAX = 2**31 - 1

PREFILL_TOKENS = 16  # the prefill length of the forward test; ids[16] is the decoded token
FIRST_TOKEN = 1000  # the first token id of the test sequence


def _plugin_frames(depth: int = 2) -> str:
    """The innermost ``depth`` stack frames inside vllm or vllm_gaudi, as ``file:lineno <- ...``."""
    frames = [f for f in traceback.extract_stack() if "/vllm/" in f.filename or "/vllm_gaudi/" in f.filename]
    return " <- ".join(f"{f.filename}:{f.lineno}" for f in reversed(frames[-depth:]))


def _format_op_records(records) -> str:
    return "\n".join(f"  {op}: int64 range [{lo}, {hi}] at {where}" for op, lo, hi, where in records)


class Int32RangeGuard(TorchDispatchMode):
    """Records every aten op whose int64 result holds a value outside the int32 range.

    Each record is ``(str(func), lo, hi, where)``: the op, the min and max of its int64 result,
    and the innermost two stack frames inside vllm or vllm_gaudi. View ops are skipped (host-side
    checkpoint reads) except ``aten.lift_fresh.default``; copies and compute ops always count
    (module docstring, view-op rule). The min/max of a result are read with the check flagged on,
    so the observing ``min``/``max`` dispatches record nothing and cannot recurse.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, int, int, str]] = []
        self._checking = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not self._checking:
            self._checking = True
            try:
                self._record(func, out)
            finally:
                self._checking = False
        return out

    def _record(self, func, out) -> None:
        if getattr(func, "is_view", False) and func is not torch.ops.aten.lift_fresh.default:
            return
        for tensor in tree_flatten(out)[0]:
            if (isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int64 and tensor.numel() > 0
                    and tensor.device.type != "meta"):
                lo, hi = int(tensor.min()), int(tensor.max())
                if lo < INT32_MIN or hi > INT32_MAX:
                    self.records.append((str(func), lo, hi, _plugin_frames()))


def _out_of_range_tensors(model) -> list[tuple[str, int, int]]:
    """Every int64 buffer and parameter of the model holding a value outside the int32 range.

    Returns ``(name, lo, hi)`` triples: a buffer like this on the card is an int32 device tensor
    and its values are truncated at build time, which is what the lazy bridge refuses outright.
    """
    out = []
    for name, tensor in (*model.named_buffers(), *model.named_parameters()):
        if tensor.dtype == torch.int64 and tensor.numel() > 0:
            lo, hi = int(tensor.min()), int(tensor.max())
            if lo < INT32_MIN or hi > INT32_MAX:
                out.append((name, lo, hi))
    return out


def test_build_keeps_int64_in_int32_range(tiny_tp8_checkpoint):
    """The tiny TP8 build through the real loader creates no int64 value outside the int32 range."""
    guard = Int32RangeGuard()
    with guard, cpu_model(tiny_tp8_checkpoint) as (vllm_config, model):
        records = list(guard.records)
        buffers = _out_of_range_tensors(model)
    failure = ("the build created int64 values outside the int32 range; the lazy bridge rejects these\n"
               + _format_op_records(records) + "\n"
               + "\n".join(f"  buffer {name}: int64 range [{lo}, {hi}]" for name, lo, hi in buffers))
    assert not records and not buffers, failure


def test_forward_keeps_int64_in_int32_range(tiny_tp8_checkpoint):
    """One 16-token prefill and one decode of the tiny TP8 model create no int64 value out of range."""
    ids = torch.arange(FIRST_TOKEN, FIRST_TOKEN + PREFILL_TOKENS + 1)
    n = PREFILL_TOKENS
    guard = Int32RangeGuard()
    with cpu_model(tiny_tp8_checkpoint) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        with guard, torch.no_grad(), set_forward_context(prefill_metadata(n, 1), vllm_config):
            model(input_ids=ids[:n], positions=torch.arange(n), **ple_kwargs(n, eos))
        with guard, torch.no_grad(), set_forward_context(decode_metadata(1, n), vllm_config):
            model(input_ids=ids[n:n + 1],
                  positions=torch.tensor([n]),
                  **ple_kwargs(1, eos, context=ids[n - 2:n].tolist()))
        records = list(guard.records)
    failure = ("the forward created int64 values outside the int32 range; the lazy bridge rejects these\n"
               + _format_op_records(records))
    assert not records, failure
