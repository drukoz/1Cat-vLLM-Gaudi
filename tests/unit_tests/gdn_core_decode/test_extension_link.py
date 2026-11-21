# SPDX-License-Identifier: Apache-2.0
"""The HPU custom-op extensions link only the Habana backend library, never a bridge-mode plugin (RUK-3103)."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest
import setuptools

REPO = Path(__file__).resolve().parents[3]
SETUP_FILES = (
    "csrc/gdn_fused/pytorch/setup.py",
    "csrc/flashinfer_gaudi/pytorch/setup.py",
    "csrc/flashqla_hpu/setup.py",
)


def _stub_habana(monkeypatch, is_torch_fork):
    """Install a minimal ``habana_frameworks`` package; ``is_torch_fork=None`` leaves the attribute out."""
    habana = types.ModuleType("habana_frameworks")
    htorch = types.ModuleType("habana_frameworks.torch")
    utils = types.ModuleType("habana_frameworks.torch.utils")
    lib_utils = types.ModuleType("habana_frameworks.torch.utils.lib_utils")
    if is_torch_fork is not None:
        htorch.is_torch_fork = is_torch_fork
    lib_utils.get_include_dir = lambda: "/stub/include"
    lib_utils.get_lib_dir = lambda: "/stub/lib"
    habana.torch = htorch
    htorch.utils = utils
    utils.lib_utils = lib_utils
    for module in (habana, htorch, utils, lib_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def _habana_libraries(monkeypatch, setup_file):
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(REPO / setup_file), run_name="setup_under_test")
    (extension, ) = captured["ext_modules"]
    return [name for name in extension.libraries if "habana" in name]


@pytest.mark.parametrize("setup_file", SETUP_FILES)
@pytest.mark.parametrize(("is_torch_fork", "expected"), [
    (True, "habana_pytorch_backend"),
    (False, "habana_pytorch_backend.upstream"),
    (None, "habana_pytorch_backend.upstream"),
])
def test_links_backend_library_only(monkeypatch, setup_file, is_torch_fork, expected):
    _stub_habana(monkeypatch, is_torch_fork)
    assert _habana_libraries(monkeypatch, setup_file) == [expected]
