# SPDX-License-Identifier: Apache-2.0
"""CPU tests for load_gdn_fused_extension: ordering, GC_KERNEL_PATH and failures."""

from __future__ import annotations

import os

import pytest
import torch

from flashinfer_gaudi import _native


@pytest.fixture(autouse=True)
def clean_loader_state(monkeypatch):
    monkeypatch.setattr(_native, "_GDN_FUSED_LOADED", None)
    monkeypatch.setattr(_native, "_LOAD_ERRORS", [])
    monkeypatch.setattr(_native, "_GDN_FUSED_ERRORS", [])
    monkeypatch.setattr(_native, "_LOADED_PATHS", [])
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    monkeypatch.delenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", raising=False)
    monkeypatch.delenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", raising=False)
    monkeypatch.delenv("PT_HPU_RECIPE_CACHE_CONFIG", raising=False)


def _make_libraries(tmp_path, monkeypatch):
    kernel_db = tmp_path / "libgdn_fused_gaudi2.so"
    extension = tmp_path / "gdn_fused_ops.so"
    kernel_db.write_bytes(b"")
    extension.write_bytes(b"")
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel_db))
    monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(extension))
    return kernel_db, extension


def _record_imports(monkeypatch, events):
    def fake_import(name):
        events.append(f"import {name}")
        return name

    monkeypatch.setattr(_native.importlib, "import_module", fake_import)


def test_missing_files_return_false(tmp_path, monkeypatch):
    kernel_db = tmp_path / "libgdn_fused_gaudi2.so"
    extension = tmp_path / "gdn_fused_ops.so"
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel_db))
    monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(extension))

    assert _native.load_gdn_fused_extension() is False
    assert any(str(kernel_db) in error and "file does not exist" in error for error in _native._LOAD_ERRORS)
    assert any(str(extension) in error and "file does not exist" in error for error in _native._LOAD_ERRORS)


def test_gdn_fused_errors_are_gdn_only(tmp_path, monkeypatch):
    kernel_db = tmp_path / "libgdn_fused_gaudi2.so"
    extension = tmp_path / "gdn_fused_ops.so"
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel_db))
    monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(extension))

    assert _native.load_gdn_fused_extension() is False
    errors = _native.gdn_fused_extension_errors()
    assert len(errors) == 2
    assert all("file does not exist" in error for error in errors)
    assert all(error in _native._LOAD_ERRORS for error in errors)


def test_schema_imported_before_library(tmp_path, monkeypatch):
    _, extension = _make_libraries(tmp_path, monkeypatch)
    events: list[str] = []
    _record_imports(monkeypatch, events)

    def fake_load(path):
        events.append(f"load {path}")

    monkeypatch.setattr(torch.ops, "load_library", fake_load)

    assert _native.load_gdn_fused_extension() is True
    assert events == ["import flashinfer_gaudi.gdn_core_decode", f"load {extension}"]


def test_kernel_path_prepended(tmp_path, monkeypatch):
    kernel_db, _ = _make_libraries(tmp_path, monkeypatch)
    _record_imports(monkeypatch, [])
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)

    assert _native.load_gdn_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].startswith(str(kernel_db))
    assert _native.load_gdn_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep).count(str(kernel_db)) == 1


def test_load_failure_recorded(tmp_path, monkeypatch):
    _, extension = _make_libraries(tmp_path, monkeypatch)
    _record_imports(monkeypatch, [])

    def boom(path):
        raise OSError("bad elf")

    monkeypatch.setattr(torch.ops, "load_library", boom)

    assert _native.load_gdn_fused_extension() is False
    assert any(str(extension) in error and "bad elf" in error for error in _native._LOAD_ERRORS)


_DLOPEN = "libhabana_pytorch2_plugin.so: cannot open shared object file: No such file or directory"


def _raise_wrapped(path):
    try:
        raise OSError(_DLOPEN)
    except OSError as exc:
        raise OSError(f"Could not load this library: {path}") from exc


def test_gdn_load_error_keeps_dlopen_cause(tmp_path, monkeypatch):
    _, extension = _make_libraries(tmp_path, monkeypatch)
    _record_imports(monkeypatch, [])
    monkeypatch.setattr(torch.ops, "load_library", _raise_wrapped)

    assert _native.load_gdn_fused_extension() is False
    assert _native.gdn_fused_extension_errors() == (
        f"{extension}: OSError: Could not load this library: {extension}; caused by OSError: {_DLOPEN}", )


def test_gdn_real_load_library_error_names_cause(tmp_path, monkeypatch):
    _, extension = _make_libraries(tmp_path, monkeypatch)  # an empty file is not a loadable library
    _record_imports(monkeypatch, [])

    assert _native.load_gdn_fused_extension() is False
    (error, ) = _native.gdn_fused_extension_errors()
    assert error.startswith(f"{extension}: OSError: Could not load this library: {extension}; caused by OSError: ")


def test_recipe_cache_blocks_the_fused_kernel_database(tmp_path, monkeypatch):
    kernel_db, _ = _make_libraries(tmp_path, monkeypatch)
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    calls: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: calls.append(path))

    assert _native.load_gdn_fused_extension() is False
    assert os.environ["GC_KERNEL_PATH"] == ""
    assert calls == []
    (error, ) = _native.gdn_fused_extension_errors()
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in error
    assert "GC_KERNEL_PATH" in error
    assert "libgdn_fused_gaudi2.so" in error
    assert error in _native._LOAD_ERRORS
    assert kernel_db.is_file()


def test_recipe_cache_empty_value_does_not_block(tmp_path, monkeypatch):
    kernel_db, _ = _make_libraries(tmp_path, monkeypatch)
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "")
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)

    assert _native.load_gdn_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].startswith(str(kernel_db))


def test_blocked_reason_names_the_remedy(monkeypatch):
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    reason = _native.kernel_database_blocked_reason("x.so")
    assert reason is not None
    assert "Unset PT_HPU_RECIPE_CACHE_CONFIG" in reason
    assert "libflashinfer_gaudi_combined_gaudi2.so" in reason
    monkeypatch.delenv("PT_HPU_RECIPE_CACHE_CONFIG")
    assert _native.kernel_database_blocked_reason("x.so") is None


def test_native_extensions_error_keeps_dlopen_cause(tmp_path, monkeypatch):
    library = tmp_path / "flashinfer_gaudi_ops.so"
    library.write_bytes(b"")
    monkeypatch.setenv("FLASHINFER_GAUDI_NATIVE_LIBRARY", str(library))
    monkeypatch.setattr(_native, "_LOAD_ATTEMPTED", False)
    monkeypatch.setattr(_native, "_SILU_AND_MUL_OP", None)
    monkeypatch.setattr(_native, "_ADD_RMSNORM_QUANT_OP", None)
    monkeypatch.setattr(torch.ops, "load_library", _raise_wrapped)

    _native.load_native_extensions()
    expected = f"{library}: OSError: Could not load this library: {library}; caused by OSError: {_DLOPEN}"
    assert expected in _native._LOAD_ERRORS
