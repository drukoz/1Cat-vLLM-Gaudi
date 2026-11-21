# SPDX-License-Identifier: Apache-2.0
"""engine-step: log which inputs of a replayed HPU graph the graph wrapper copies before each replay.

Python only, nothing else changes. Put this folder first on PYTHONPATH of the server (vllm serve) with
HPU_GRAPH_INPUT_LOG=1; the lines go to stderr (the server log) as "HPU_GRAPH_UNMATCHED ...".

Why (native-replay, from the bridge source 5176b1b2): wrap_in_hpu_graph's cached path binds an input directly only
when its own lazy tensor is an input value of the captured graph; every other input is copied by the module-level
function habana_frameworks/torch/hpu/graphs.py copy_to(cached.graph_inputs, graph_inputs), a lazy device-to-device
copy flushed as its own recipe before the replay (step-anatomy's 00205 census: 13 copy-only graphs per 32-stream MTP
step). copy_to is looked up as a module global at call time, so wrapping it is enough. For each set of copied
tensors (path in the forward's arguments, shape, dtype, contiguity, storage offset) a line is printed the 1st,
100th and 1000th time the same set comes back, with the process id (one line per worker rank).

The module is wrapped only when the worker itself imports habana_frameworks.torch.hpu.graphs (an import hook), so
processes that never use the card (the API server) are not touched. Another sitecustomize further down the path is
still run (chained).
"""
import os
import sys


def _chain_other_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        folder = os.path.abspath(entry or ".")
        if folder == here:
            continue
        candidate = os.path.join(folder, "sitecustomize.py")
        if os.path.isfile(candidate):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_chained_sitecustomize", candidate)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as error:  # noqa: BLE001 - never stop a server start here
                print("HPU_GRAPH_UNMATCHED chained sitecustomize %s failed: %r" % (candidate, error),
                      file=sys.stderr, flush=True)
            return


def _tensors(obj, path, out, depth=0):
    import dataclasses
    try:
        import torch
    except Exception:  # noqa: BLE001
        return
    if depth > 6:
        return
    if isinstance(obj, torch.Tensor):
        out.append((path, tuple(obj.shape), str(obj.dtype).replace("torch.", ""), obj.is_contiguous(),
                    obj.storage_offset()))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _tensors(value, "%s.%s" % (path, key), out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        if hasattr(obj, "_fields"):
            for name in obj._fields:
                _tensors(getattr(obj, name, None), "%s.%s" % (path, name), out, depth + 1)
        else:
            for i, value in enumerate(obj):
                _tensors(value, "%s[%d]" % (path, i), out, depth + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _tensors(getattr(obj, field.name, None), "%s.%s" % (path, field.name), out, depth + 1)


def _wrap(graphs):
    original = getattr(graphs, "copy_to", None)
    if original is None or getattr(original, "_engine_step_logged", False):
        return
    import collections
    seen = collections.Counter()
    depth = [0]

    def logged_copy_to(dst, src, *args, **kwargs):
        if depth[0] == 0:
            try:
                found = []
                _tensors(src, "in", found)
                key = tuple(found)
                seen[key] += 1
                if seen[key] in (1, 100, 1000):
                    print("HPU_GRAPH_UNMATCHED pid %d x%d: %d tensor(s): %s" % (
                        os.getpid(), seen[key], len(found),
                        "; ".join("%s %s %s%s off %d" % (p, list(s), d, "" if c else " noncontig", o)
                                  for p, s, d, c, o in found)), file=sys.stderr, flush=True)
            except Exception as error:  # noqa: BLE001 - the log never breaks a replay
                print("HPU_GRAPH_UNMATCHED log failed: %r" % (error, ), file=sys.stderr, flush=True)
        depth[0] += 1
        try:
            return original(dst, src, *args, **kwargs)
        finally:
            depth[0] -= 1

    logged_copy_to._engine_step_logged = True
    graphs.copy_to = logged_copy_to
    print("HPU_GRAPH_UNMATCHED pid %d: graphs.copy_to wrapped" % os.getpid(), file=sys.stderr, flush=True)


class _Hook:
    """A meta-path finder that wraps habana_frameworks.torch.hpu.graphs right after it is executed."""

    TARGET = "habana_frameworks.torch.hpu.graphs"

    def find_spec(self, name, path=None, target=None):
        if name != self.TARGET:
            return None
        import importlib.machinery
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(name, path, target) if hasattr(finder, "find_spec") else None
            if spec is not None and spec.loader is not None:
                loader = spec.loader
                exec_module = loader.exec_module

                def exec_and_wrap(module, _exec=exec_module):
                    _exec(module)
                    try:
                        _wrap(module)
                    except Exception as error:  # noqa: BLE001
                        print("HPU_GRAPH_UNMATCHED wrap failed: %r" % (error, ), file=sys.stderr, flush=True)

                loader.exec_module = exec_and_wrap
                return spec
        return None


if os.environ.get("HPU_GRAPH_INPUT_LOG") == "1":
    if _Hook.TARGET in sys.modules:
        _wrap(sys.modules[_Hook.TARGET])
    else:
        sys.meta_path.insert(0, _Hook())
_chain_other_sitecustomize()
