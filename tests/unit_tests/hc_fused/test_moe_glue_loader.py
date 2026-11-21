# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the loader side of VLLM_HPU_MOE_GLUE and for the startup call.

The MoE glue kernels live in the fused hyper-connection kernel library and their ops in its extension,
so ``load_moe_glue_extension`` is ``load_hc_fused_extension`` plus one question to the library: which of
the four MoE kernels does it serve (each is optional at build time).  No card: the library files are
fakes, ``torch.ops.load_library`` is recorded and the GUID list is given.  The loader's rules for
``GC_KERNEL_PATH`` and the combined database are those of the hyper-connection loader and are tested in
``tests/unit_tests/gdn_core_decode/test_hc_fused_loader.py``.
"""

from __future__ import annotations

import sys

import pytest
import torch

import flashinfer_gaudi
import vllm_gaudi
import vllm_gaudi.envs
from flashinfer_gaudi import _native

KERNEL_DB = "libhc_fused_gaudi2.so"
EXTENSION = "hc_fused_ops.cpython-310-x86_64-linux-gnu.so"
HC_GUIDS = ("hc_fused_norm_bf16_gaudi2", "hc_fused_silu_bf16_gaudi2", "hc_fused_silu_split_bf16_gaudi2",
            "hc_fused_gate_mix_bf16_gaudi2", "hc_fused_combine_bf16_gaudi2", "hc_fused_combine_norm_bf16_gaudi2")
MOE_GUIDS = ("moe_glue_route_bf16_gaudi2", "moe_glue_silu_mul_bf16_gaudi2", "moe_glue_combine_bf16_gaudi2",
             "moe_glue_row_quant_bf16_gaudi2", "moe_glue_act_quant_bf16_gaudi2")
STATUS_KEYS = {"state", "gc_kernel_path", "combined_active", "combined_kernel_db", "gdn_fused_loaded",
               "hc_fused_loaded", "load_errors"}


@pytest.fixture(autouse=True)
def clean_loader_state(monkeypatch):
    monkeypatch.setattr(_native, "_HC_FUSED_LOADED", None)
    monkeypatch.setattr(_native, "_HC_FUSED_ERRORS", [])
    monkeypatch.setattr(_native, "_MOE_GLUE_LOADED", None)
    monkeypatch.setattr(_native, "_MOE_GLUE_ERRORS", [])
    monkeypatch.setattr(_native, "_MOE_GLUE_MISSING", ())
    monkeypatch.setattr(_native, "_LOAD_ERRORS", [])
    monkeypatch.setattr(_native, "_LOADED_PATHS", [])
    monkeypatch.setattr(_native, "_COMBINED_DB_ACTIVE", None)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    for name in ("HC_FUSED_GAUDI_KERNEL_LIBRARY", "HC_FUSED_GAUDI_NATIVE_LIBRARY", "GDN_FUSED_GAUDI_KERNEL_LIBRARY",
                 "FLASHINFER_GAUDI_COMBINED_MEMBERS", "PT_HPU_RECIPE_CACHE_CONFIG",
                 "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    """Records the schema import and the extension load, in order."""
    log: list[str] = []
    real_import = _native.importlib.import_module

    def fake_import(name):
        log.append(f"import {name}")
        return real_import(name)

    monkeypatch.setattr(_native.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch.ops, "load_library", lambda path: log.append(f"load {path}"))
    return log


@pytest.fixture
def package(monkeypatch, tmp_path):
    """A fake installed package: flashinfer_gaudi/lib with the kernel library and the extension."""
    package_dir = tmp_path / "flashinfer_gaudi"
    lib = package_dir / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    (lib / KERNEL_DB).write_bytes(b"")
    (lib / EXTENSION).write_bytes(b"")
    return lib


def _serve(monkeypatch, guids):
    asked: list[str] = []

    def kernel_library_guids(path):
        asked.append(str(path))
        if isinstance(guids, Exception):
            raise guids
        return tuple(guids)

    monkeypatch.setattr(_native, "kernel_library_guids", kernel_library_guids)
    return asked


def test_the_loader_and_the_op_module_name_the_same_ops():
    from flashinfer_gaudi import moe_glue

    assert _native.MOE_GLUE_OPS == moe_glue.OP_NAMES == tuple(moe_glue.GUIDS)
    assert tuple(moe_glue.GUIDS.values()) == MOE_GUIDS
    assert _native.MOE_GLUE_SCHEMA_MODULE == moe_glue.__name__


def test_never_called_reports_nothing():
    """With the flag at 0 the loader is never called: nothing is known and the status line is what it was."""
    assert _native.moe_glue_extension_loaded() is None
    assert _native.moe_glue_extension_errors() == ()
    assert _native.moe_glue_missing_kernels() == ()
    assert set(_native.kernel_database_status()) == STATUS_KEYS


def test_the_kernels_are_in_the_hyper_connection_library(monkeypatch, package, events):
    asked = _serve(monkeypatch, HC_GUIDS + MOE_GUIDS)

    assert _native.load_moe_glue_extension() is True
    # one library, one extension: the hyper-connection loader's steps, nothing of its own to register
    assert events == [f"import {_native.HC_FUSED_SCHEMA_MODULE}", f"load {package / EXTENSION}"]
    assert asked == [str(package / KERNEL_DB)]
    assert _native.hc_fused_extension_loaded() is True
    assert _native.moe_glue_extension_loaded() is True
    assert _native.moe_glue_missing_kernels() == () and _native.moe_glue_extension_errors() == ()
    status = _native.kernel_database_status()
    assert status["moe_glue_loaded"] is True and status["hc_fused_loaded"] is True
    assert set(status) == STATUS_KEYS | {"moe_glue_loaded"}
    assert _native.load_moe_glue_extension() is True and len(events) == 2  # once per process


def test_both_schema_families_are_defined_before_the_shared_extension_is_loaded(monkeypatch, package, events):
    """The extension implements the HPU kernels of both families, whichever flag asked for it."""
    _serve(monkeypatch, HC_GUIDS + MOE_GUIDS)
    assert _native.load_hc_fused_extension() is True  # VLLM_HPU_HC_FUSED=1 alone
    assert events == [f"import {_native.HC_FUSED_SCHEMA_MODULE}", f"load {package / EXTENSION}"]
    assert _native.MOE_GLUE_SCHEMA_MODULE == "flashinfer_gaudi.moe_glue"
    assert _native.MOE_GLUE_SCHEMA_MODULE in sys.modules
    names = set(torch._C._dispatch_get_all_op_names())
    for op in ("hc_norm", "hc_combine_norm", "moe_route", "moe_silu_mul", "moe_combine", "moe_row_quant"):
        assert f"flashinfer_gaudi::{op}" in names
    # the MoE loader was not called: nothing is reported for it
    assert _native.moe_glue_extension_loaded() is None
    assert "moe_glue_loaded" not in _native.kernel_database_status()


def test_missing_files_are_named(monkeypatch, tmp_path, events):
    package_dir = tmp_path / "flashinfer_gaudi"
    (package_dir / "lib").mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    asked = _serve(monkeypatch, HC_GUIDS + MOE_GUIDS)

    assert _native.load_moe_glue_extension() is False
    errors = _native.moe_glue_extension_errors()
    assert len(errors) == 2 and all("file does not exist" in error for error in errors)
    assert errors == _native.hc_fused_extension_errors()  # the library's own errors, not copies with other text
    assert events == [] and asked == []
    assert _native.kernel_database_status()["moe_glue_loaded"] is False
    assert _native.moe_glue_missing_kernels() == ("moe_route", "moe_silu_mul", "moe_combine", "moe_row_quant", "moe_act_quant")


def test_a_kernel_the_library_was_built_without_is_named(monkeypatch, package, events):
    """A MoE kernel that did not compile is left out of the library; the hyper-connection kernels are not."""
    _serve(monkeypatch, HC_GUIDS + MOE_GUIDS[1:])

    assert _native.load_moe_glue_extension() is False
    assert _native.hc_fused_extension_loaded() is True and _native.hc_fused_extension_errors() == ()
    assert _native.moe_glue_missing_kernels() == ("moe_route", )
    (error, ) = _native.moe_glue_extension_errors()
    assert str(package / KERNEL_DB) in error and "moe_route (moe_glue_route_bf16_gaudi2)" in error
    assert error in _native._LOAD_ERRORS  # the shared list of the status line
    assert _native.kernel_database_status()["moe_glue_loaded"] is False


def test_a_library_that_cannot_be_asked_serves_nothing(monkeypatch, package, events):
    _serve(monkeypatch, OSError("invalid ELF header"))

    assert _native.load_moe_glue_extension() is False
    assert _native.moe_glue_missing_kernels() == ("moe_route", "moe_silu_mul", "moe_combine", "moe_row_quant", "moe_act_quant")
    (error, ) = _native.moe_glue_extension_errors()
    assert "OSError: invalid ELF header" in error
    assert _native.hc_fused_extension_loaded() is True  # the hyper-connection ops are not affected


def test_the_library_named_by_the_override_is_the_one_asked(monkeypatch, tmp_path, events):
    package_dir = tmp_path / "source" / "flashinfer_gaudi"
    (package_dir / "lib").mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    installed = tmp_path / "installed"
    installed.mkdir()
    for name in (KERNEL_DB, EXTENSION):
        (installed / name).write_bytes(b"")
    monkeypatch.setenv("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(installed / KERNEL_DB))
    monkeypatch.setenv("HC_FUSED_GAUDI_NATIVE_LIBRARY", str(installed / EXTENSION))
    asked = _serve(monkeypatch, HC_GUIDS + MOE_GUIDS)

    assert _native.load_moe_glue_extension() is True
    assert asked == [str(installed / KERNEL_DB)]


def test_reset_for_tests_forgets_the_attempt(monkeypatch, package, events):
    _serve(monkeypatch, HC_GUIDS)
    assert _native.load_moe_glue_extension() is False
    _native._reset_native_state_for_tests()
    assert _native.moe_glue_extension_loaded() is None
    assert _native.moe_glue_extension_errors() == () and _native.moe_glue_missing_kernels() == ()


def test_an_unexpected_error_is_recorded_not_raised(monkeypatch, package, events):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(_native, "load_hc_fused_extension", boom)
    assert _native.load_moe_glue_extension() is False
    (error, ) = _native.moe_glue_extension_errors()
    assert "RuntimeError: boom" in error
    assert _native.load_moe_glue_extension() is False


# --------------------------------------------------------------------------------------------------------------
# startup (vllm_gaudi.register_ops)

ERRORS = ("a.so: file does not exist", "b.so: file does not exist")


def test_warns_when_kernel_missing_on_hpu():
    calls: list[str] = []

    message = vllm_gaudi._warn_moe_glue_without_kernel(1, False, True, ERRORS, warn=calls.append)

    assert calls == [message]
    assert message.startswith("MoE glue (VLLM_HPU_MOE_GLUE=1) has no HPU kernel")
    assert "a.so: file does not exist; b.so: file does not exist" in message
    assert "torch path" in message and "tools/build_flashinfer_gaudi.py --hc-fused-only" in message
    assert "set VLLM_HPU_MOE_GLUE=0" in message
    assert "(no error recorded)" in vllm_gaudi._warn_moe_glue_without_kernel(1, False, True, (), warn=calls.append)


@pytest.mark.parametrize("flag,loaded,on_hpu", [(1, True, True), (1, False, False), (0, False, True)])
def test_logs_nothing_when_not_applicable(flag, loaded, on_hpu):
    calls: list[str] = []
    assert vllm_gaudi._warn_moe_glue_without_kernel(flag, loaded, on_hpu, ERRORS, warn=calls.append) is None
    assert calls == []


@pytest.fixture
def loaders(monkeypatch):
    """The loaders register_ops may call, recorded in call order; none of them runs."""
    calls: list[str] = []
    state = {"moe_loaded": False}
    monkeypatch.setattr(flashinfer_gaudi, "load_native_extensions", lambda: calls.append("native") or ())
    monkeypatch.setattr(_native, "load_gdn_fused_extension", lambda: calls.append("gdn") or True)
    monkeypatch.setattr(_native, "load_hc_fused_extension", lambda: calls.append("hc") or True)
    monkeypatch.setattr(_native, "load_moe_glue_extension", lambda: calls.append("moe") or state["moe_loaded"])
    monkeypatch.setattr(_native, "moe_glue_extension_errors", lambda: () if state["moe_loaded"] else ERRORS)
    monkeypatch.setattr(_native, "kernel_database_status", lambda: calls.append("status") or {})
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "0")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "0")
    monkeypatch.delenv("VLLM_HPU_HC_FUSED", raising=False)
    monkeypatch.delenv("VLLM_HPU_MOE_GLUE", raising=False)
    return calls, state


def _register_ops() -> None:
    try:
        vllm_gaudi.register_ops()
    except Exception:  # noqa: BLE001 - without habana_frameworks the imports after the loaders fail
        pass


@pytest.mark.parametrize("value", [None, "0", "2"])
def test_register_ops_does_not_touch_the_loader_without_the_flag(loaders, monkeypatch, value):
    calls, _ = loaders
    if value is not None:
        monkeypatch.setenv("VLLM_HPU_MOE_GLUE", value)

    _register_ops()

    assert calls == ["native", "status"]


def test_register_ops_loads_the_kernels_with_the_flag_before_the_status_line(loaders, monkeypatch):
    calls, _ = loaders
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE", "1")

    _register_ops()

    assert calls == ["native", "moe", "status"]


def test_register_ops_loads_the_hyper_connection_kernels_first(loaders, monkeypatch):
    calls, _ = loaders
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE", "1")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "1")

    _register_ops()

    assert calls == ["native", "gdn", "hc", "moe", "status"]


def test_load_if_requested_warns_once_when_the_kernels_are_not_usable(loaders, monkeypatch):
    calls, state = loaders
    warnings: list[str] = []
    monkeypatch.setattr(vllm_gaudi, "_hpu_visible", lambda: True)

    assert vllm_gaudi._load_moe_glue_if_requested(1, warn=warnings.append) is False
    assert calls == ["moe"]
    assert len(warnings) == 1 and "has no HPU kernel" in warnings[0] and ERRORS[0] in warnings[0]
    state["moe_loaded"] = True
    assert vllm_gaudi._load_moe_glue_if_requested(1, warn=warnings.append) is True and len(warnings) == 1


@pytest.mark.parametrize("flag", [0, 2, -1])
def test_load_if_requested_does_nothing_without_the_flag(loaders, flag):
    calls, _ = loaders
    warnings: list[str] = []

    assert vllm_gaudi._load_moe_glue_if_requested(flag, warn=warnings.append) is None
    assert calls == [] and warnings == []
