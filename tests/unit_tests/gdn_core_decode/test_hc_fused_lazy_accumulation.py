# SPDX-License-Identifier: Apache-2.0
"""CPU tests: the fused hyper-connection kernels and the lazy bridge's accumulation thread.

On a Gaudi2 node (2026-11-20, bridge 1.24.1) the card test of the kernels failed 6 and 7 of 90 tests with
``PT_HPU_LAZY_ACC_PAR_MODE=1`` (the bridge's default; a different set each run) and passed 90 of 90 with it at
0: at 1 the bridge builds queued ops on a second thread, and a custom op is built at once on the calling
thread, so an input whose producer is still queued becomes an empty graph input.

Two rules are tested here. The plugin sets the variable to 0, unless the user set it, before the bridge is
imported, in every process that asks for the kernels (``vllm_gaudi.register``). The loader refuses the
kernels in a lazy process that does not have it at 0, so the layers run their torch path instead of failing
at random.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flashinfer_gaudi import _native

from .test_hc_fused_loader import EXTENSION, KERNEL_DB, _package

ACC = "PT_HPU_LAZY_ACC_PAR_MODE"
ALLOW = "FLASHINFER_GAUDI_ALLOW_LAZY_ACC_THREAD"


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for name, value in (("_HC_FUSED_LOADED", None), ("_HC_FUSED_ERRORS", []), ("_MOE_GLUE_LOADED", None),
                        ("_MOE_GLUE_ERRORS", []), ("_MOE_GLUE_MISSING", ()), ("_GDN_FUSED_LOADED", None),
                        ("_GDN_FUSED_ERRORS", []), ("_LOAD_ERRORS", []), ("_LOADED_PATHS", []),
                        ("_COMBINED_DB_ACTIVE", None), ("_FLASHINFER_DB_STATE", _native.KernelDbState.MISSING)):
        monkeypatch.setattr(_native, name, value)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    for name in ("HC_FUSED_GAUDI_KERNEL_LIBRARY", "HC_FUSED_GAUDI_NATIVE_LIBRARY", "GDN_FUSED_GAUDI_KERNEL_LIBRARY",
                 "GDN_FUSED_GAUDI_NATIVE_LIBRARY", "FLASHINFER_GAUDI_COMBINED_MEMBERS", "PT_HPU_RECIPE_CACHE_CONFIG",
                 "FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "PT_HPU_LAZY_MODE", "VLLM_HPU_HC_FUSED", "VLLM_HPU_MOE_GLUE",
                 ACC, ALLOW):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    """The schema import and the extension load, in order; neither runs for real."""
    log: list[str] = []
    real_import = _native.importlib.import_module

    def fake_import(name):
        log.append(f"import {name}")
        return real_import(name) if name == _native.HC_FUSED_SCHEMA_MODULE else name

    monkeypatch.setattr(_native.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch.ops, "load_library", lambda path: log.append(f"load {path}"))
    return log


def _bridge(monkeypatch, lazy):
    """Put a stand-in for the imported Habana bridge into sys.modules: `lazy` is what is_lazy() answers."""
    bridge = ModuleType("habana_frameworks.torch")
    bridge.utils = SimpleNamespace(internal=SimpleNamespace(is_lazy=lambda: lazy))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", bridge)
    return bridge


# ---- what the loader knows about the bridge ----------------------------------------------------
def test_without_the_bridge_nothing_is_lazy(monkeypatch):
    monkeypatch.delitem(sys.modules, "habana_frameworks.torch", raising=False)

    assert _native.lazy_bridge_loaded() is False
    assert _native.lazy_accumulation_reason() is None


@pytest.mark.parametrize("lazy", [True, False])
def test_the_bridge_is_asked_for_its_mode(monkeypatch, lazy):
    _bridge(monkeypatch, lazy)

    assert _native.lazy_bridge_loaded() is lazy


@pytest.mark.parametrize("value,lazy", [(None, True), ("1", True), ("0", False)])
def test_a_bridge_that_cannot_be_asked_goes_by_the_mode_variable(monkeypatch, value, lazy):
    """Unset counts as lazy here, as flashinfer_gaudi._bridge reads it: a refusal is the safe side."""
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", ModuleType("habana_frameworks.torch"))
    if value is not None:
        monkeypatch.setenv("PT_HPU_LAZY_MODE", value)

    assert _native.lazy_bridge_loaded() is lazy


@pytest.mark.parametrize("value,shown", [(None, "unset (the bridge's default is 1)"), ("1", "'1'"), ("", "''"),
                                         ("false", "'false'")])
def test_the_reason_names_the_variable_and_the_failure(monkeypatch, value, shown):
    _bridge(monkeypatch, True)
    if value is not None:
        monkeypatch.setenv(ACC, value)

    reason = _native.lazy_accumulation_reason()

    assert reason.startswith(f"{ACC} is {shown}: ")
    assert "Empty tensor optional" in reason and "tensor_data is empty" in reason
    assert f"{ACC}=0" in reason


@pytest.mark.parametrize("value", ["0", " 0 "])
def test_mode_0_is_safe(monkeypatch, value):
    _bridge(monkeypatch, True)
    monkeypatch.setenv(ACC, value)

    assert _native.lazy_accumulation_reason() is None


def test_an_eager_bridge_is_not_concerned(monkeypatch):
    _bridge(monkeypatch, False)

    assert _native.lazy_accumulation_reason() is None


# ---- the loader --------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [None, "1"])
def test_a_lazy_process_with_the_accumulation_thread_gets_no_kernels(monkeypatch, tmp_path, events, value):
    _package(monkeypatch, tmp_path)
    _bridge(monkeypatch, True)
    if value is not None:
        monkeypatch.setenv(ACC, value)

    assert _native.load_hc_fused_extension() is False

    assert _native.hc_fused_extension_errors() == (_native.lazy_accumulation_reason(), )
    assert events == []  # no schema, no extension: the ops have no HPU kernel and every layer runs torch
    assert os.environ["GC_KERNEL_PATH"] == ""  # and nothing was registered for the graph compiler
    assert os.environ.get(ACC) == value  # the loader does not set it: the bridge may have read it already


def test_a_lazy_process_at_mode_0_loads_the_kernels(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)
    _bridge(monkeypatch, True)
    monkeypatch.setenv(ACC, "0")

    assert _native.load_hc_fused_extension() is True

    assert _native.hc_fused_extension_errors() == ()
    assert events == [f"import {_native.HC_FUSED_SCHEMA_MODULE}", f"load {lib / EXTENSION}"]
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep)[0] == str(lib / KERNEL_DB)


def test_an_eager_process_loads_the_kernels_as_before(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)
    _bridge(monkeypatch, False)

    assert _native.load_hc_fused_extension() is True
    assert events == [f"import {_native.HC_FUSED_SCHEMA_MODULE}", f"load {lib / EXTENSION}"]


def test_a_missing_build_is_still_what_is_reported(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, members=False)
    _bridge(monkeypatch, True)

    assert _native.load_hc_fused_extension() is False

    errors = _native.hc_fused_extension_errors()
    assert len(errors) == 2 and all("file does not exist" in error for error in errors)


def test_the_diagnostic_switch_loads_them_at_mode_1(monkeypatch, tmp_path, events):
    """For the probes and the card test's control runs, which reproduce the race on purpose."""
    lib = _package(monkeypatch, tmp_path)
    _bridge(monkeypatch, True)
    monkeypatch.setenv(ACC, "1")
    monkeypatch.setenv(ALLOW, "1")

    assert _native.lazy_accumulation_reason() is None
    assert _native.load_hc_fused_extension() is True
    assert events[-1] == f"load {lib / EXTENSION}"


def test_the_moe_glue_kernels_are_refused_with_the_same_words(monkeypatch, tmp_path, events):
    """They are in the same extension, so they are exposed to the same race."""
    _package(monkeypatch, tmp_path)
    _bridge(monkeypatch, True)

    assert _native.load_moe_glue_extension() is False

    assert _native.moe_glue_extension_errors() == (_native.lazy_accumulation_reason(), )
    assert events == []


def test_the_gdn_loader_is_not_changed(monkeypatch, tmp_path):
    """Fused GDN is on by default: refusing it here would change servers that never asked for these kernels."""
    lib = _package(monkeypatch, tmp_path, members=False)
    (lib / "libgdn_fused_gaudi2.so").write_bytes(b"")
    (lib / "gdn_fused_ops.so").write_bytes(b"")
    loaded: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    _bridge(monkeypatch, True)

    assert _native.load_gdn_fused_extension() is True
    assert loaded == [str(lib / "gdn_fused_ops.so")]


# ---- the plugin's default ----------------------------------------------------------------------
@pytest.fixture
def plugin(monkeypatch):
    """A freshly loaded vllm_gaudi with a stand-in platform that records the variable when it is imported and
    when set_torch_compile runs: both happen after the default must be in place."""
    seen: dict[str, str | None] = {}

    class HpuPlatform:

        @staticmethod
        def set_torch_compile():
            seen["set_torch_compile"] = os.environ.get(ACC)

        @staticmethod
        def adjust_cuda_hooks():
            pass

    platform = ModuleType("vllm_gaudi.platform")
    platform.HpuPlatform = HpuPlatform
    patches = ModuleType("vllm_gaudi.patches")
    patches.apply = lambda: None
    saved = {name: sys.modules.get(name) for name in ("vllm_gaudi", "vllm_gaudi.platform", "vllm_gaudi.patches")}
    sys.modules.pop("vllm_gaudi", None)
    sys.modules["vllm_gaudi.platform"] = platform
    sys.modules["vllm_gaudi.patches"] = patches
    monkeypatch.setattr(sys, "argv", ["vllm"])
    module = importlib.import_module("vllm_gaudi")
    module.patches = patches
    yield module, seen
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def test_the_default_is_set_before_the_platform_and_the_bridge_are_imported(plugin, monkeypatch):
    module, seen = plugin
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")

    assert module.register() == "vllm_gaudi.platform.HpuPlatform"

    assert seen == {"set_torch_compile": "0"}
    assert os.environ[ACC] == "0"  # and every worker spawned from here inherits it


def test_the_moe_glue_flag_sets_it_too(plugin, monkeypatch):
    module, seen = plugin
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE", "1")

    module.register()

    assert seen == {"set_torch_compile": "0"}


@pytest.mark.parametrize("flags", [{}, {"VLLM_HPU_HC_FUSED": "0"}, {"VLLM_HPU_HC_FUSED": "2"},
                                   {"VLLM_HPU_HC_FUSED": "yes"}, {"VLLM_HPU_HC_FUSED": "0", "VLLM_HPU_MOE_GLUE": "0"}])
def test_without_the_kernels_nothing_is_set(plugin, monkeypatch, flags):
    """VLLM_HPU_HC_FUSED=0 (the default) must not change a process."""
    module, seen = plugin
    for name, value in flags.items():
        monkeypatch.setenv(name, value)

    module.register()

    assert seen == {"set_torch_compile": None}
    assert ACC not in os.environ


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_a_value_the_user_set_is_kept(plugin, monkeypatch, value):
    module, seen = plugin
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv(ACC, value)

    module.register()

    assert seen == {"set_torch_compile": value}
    assert os.environ[ACC] == value


def test_an_eager_process_is_left_alone(plugin, monkeypatch):
    module, seen = plugin
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")

    module.register()

    assert seen == {"set_torch_compile": None}
    assert ACC not in os.environ


def test_the_helper_says_what_it_did(plugin, monkeypatch):
    module, _ = plugin
    environ = {"VLLM_HPU_HC_FUSED": "1"}

    assert module._default_serial_lazy_accumulation(environ) == "0"
    assert environ[ACC] == "0"
    assert module._default_serial_lazy_accumulation(environ) is None  # already there: nothing to do
    assert module._default_serial_lazy_accumulation({"VLLM_HPU_HC_FUSED": "1", ACC: "1"}) is None
    assert module._default_serial_lazy_accumulation({}) is None
