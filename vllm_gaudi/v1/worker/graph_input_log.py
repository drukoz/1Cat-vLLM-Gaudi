# SPDX-License-Identifier: Apache-2.0
"""Which inputs of a replayed HPU graph are copied before each replay (``VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS=1``).

``wrap_in_hpu_graph`` (habana_frameworks/torch/hpu/graphs.py, cached-graph path, ``disable_tensor_cache=True``)
binds an input directly only when its own lazy tensor is an input value of the captured graph. Every other input
(a view or slice made outside the graph, for one) is copied by the wrapper into the captured graph's own input:
``copy_to(cached.graph_inputs, graph_inputs)``, a lazy device-to-device copy flushed as its own recipe right before
the replay, one launch per step and replay (native-replay, from the bridge source; step-anatomy's 00205 census: 13
copy-only graphs per 32-stream MTP step). ``copy_to`` is a module global the wrapper looks up at call time, so this
wraps it: at the outer call it logs, for every tensor of the copied structure, its path (argument name, dict key,
dataclass field), shape, dtype, contiguity and storage offset, the first time and at the 100th and 1000th time the
same set comes back. Nothing else changes; the log is what to fix (pass the base tensor, slice inside the forward).
"""

from __future__ import annotations

import collections
import dataclasses

_installed = False


def _tensors(obj, path, out, depth=0):
    import torch

    if depth > 6:
        return
    if isinstance(obj, torch.Tensor):
        out.append((path, tuple(obj.shape), str(obj.dtype).replace("torch.", ""), obj.is_contiguous(),
                    obj.storage_offset()))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _tensors(value, "%s.%s" % (path, key), out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _tensors(value, "%s[%d]" % (path, i), out, depth + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _tensors(getattr(obj, field.name, None), "%s.%s" % (path, field.name), out, depth + 1)
    elif hasattr(obj, "_fields"):  # a namedtuple
        for name in obj._fields:
            _tensors(getattr(obj, name, None), "%s.%s" % (path, name), out, depth + 1)


def install(log) -> bool:
    """Wrap ``habana_frameworks.torch.hpu.graphs.copy_to`` once; False where the module or the function is missing."""
    global _installed
    if _installed:
        return True
    try:
        import habana_frameworks.torch.hpu.graphs as graphs
    except Exception:  # noqa: BLE001 - no bridge (a CPU run): nothing to log
        return False
    original = getattr(graphs, "copy_to", None)
    if original is None:
        return False
    seen: collections.Counter = collections.Counter()
    depth = [0]

    def logged_copy_to(dst, src, *args, **kwargs):
        if depth[0] == 0:
            found: list = []
            _tensors(src, "input", found)
            key = tuple(found)
            seen[key] += 1
            if seen[key] in (1, 100, 1000):
                log("HPU graph replay copies %d unmatched input tensor(s) (x%d): %s", len(found), seen[key],
                    "; ".join("%s %s %s%s offset %d" % (p, list(s), d, "" if c else " non-contiguous", o)
                              for p, s, d, c, o in found))
        depth[0] += 1
        try:
            return original(dst, src, *args, **kwargs)
        finally:
            depth[0] -= 1

    graphs.copy_to = logged_copy_to
    _installed = True
    return True


__all__ = ["install"]
