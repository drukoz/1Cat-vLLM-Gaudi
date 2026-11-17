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
    monkeypatch.setattr(_native, "_LOADED_PATHS", [])
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    monkeypatch.delenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", raising=False)
    monkeypatch.delenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", raising=False)


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
