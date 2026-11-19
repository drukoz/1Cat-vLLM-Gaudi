# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the Gaudi runtime, so HPU-only plugin modules import on a CPU-only machine.

Importing this module (or loading it as a pytest plugin with ``-p tests.unit_tests.hpu_cpu_stubs``)
does two things, and only when the real runtime is missing:

* every ``habana_frameworks`` import returns a ``MagicMock`` module;
* ``torch.hpu`` is a ``MagicMock`` (``hpu_model_runner`` reads ``torch.hpu.Stream`` at import);
* ``VLLM_PLUGINS`` defaults to ``""``, so vLLM does not load the installed plugin through its
  entry point (that path compiles with ``hpu_backend``, which CPU torch does not have).

With both, ``vllm_gaudi.v1.worker.hpu_worker``, ``hpu_model_runner``, ``extension.defragmentation``,
``ops.hpu_fused_moe``, ``platform`` and the NIXL connector import on CPU. Tensors still cannot be
put on ``device="hpu"``: a test that runs HPU code patches that device to ``"cpu"`` itself.

Install it before the first ``vllm`` or ``vllm_gaudi`` import. On a machine with the real runtime
it does nothing.
"""

import importlib.machinery
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import torch


class _HabanaStubFinder:
    """Answers every ``habana_frameworks`` import with a ``MagicMock`` module."""

    class _Loader:

        def create_module(self, spec):
            return MagicMock(name=spec.name)

        def exec_module(self, module):
            pass

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "habana_frameworks":
            return importlib.machinery.ModuleSpec(fullname, self._Loader(), is_package=True)
        return None


def install() -> bool:
    """Install the stand-ins if the Gaudi runtime is missing. Returns True when they are in place."""
    if any(isinstance(finder, _HabanaStubFinder) for finder in sys.meta_path):
        return True
    if importlib.util.find_spec("habana_frameworks") is not None:
        return False
    os.environ.setdefault("VLLM_PLUGINS", "")
    sys.meta_path.insert(0, _HabanaStubFinder())
    if not hasattr(torch, "hpu"):
        torch.hpu = MagicMock(name="torch.hpu")
    return True


STUBBED = install()
