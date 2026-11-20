# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the staged ``load_hc_fused_extension``: ordering, GC_KERNEL_PATH and failures.

No card: the library files are empty fakes and ``torch.ops.load_library`` is recorded, not run.
Modelled on ``tests/unit_tests/gdn_core_decode/test_native_loader.py``.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer_gaudi import _native
from tests.unit_tests.hc_fused import hpu_loader


@pytest.fixture(autouse=True)
def clean_loader_state(monkeypatch):
    monkeypatch.setattr(hpu_loader, "_LOADED", None)
    monkeypatch.setattr(hpu_loader, "_ERRORS", [])
    monkeypatch.setattr(_native, "_LOAD_ERRORS", [])
    monkeypatch.setattr(_native, "_COMBINED_DB_ACTIVE", None)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    for name in (hpu_loader.HC_FUSED_KERNEL_LIBRARY_ENV, hpu_loader.HC_FUSED_NATIVE_LIBRARY_ENV,
                 hpu_loader.COMBINED_MEMBERS_ENV, "GDN_FUSED_GAUDI_KERNEL_LIBRARY", "PT_HPU_RECIPE_CACHE_CONFIG",
                 "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"):
        monkeypatch.delenv(name, raising=False)
    # A loader already merged into flashinfer_gaudi/_native.py would take over; these tests are for the staged one.
    monkeypatch.delattr(_native, "load_hc_fused_extension", raising=False)
    monkeypatch.delattr(_native, "hc_fused_extension_errors", raising=False)


@pytest.fixture
def events(monkeypatch):
    """Records the schema import and the extension load, in order."""
    log = []
    real_import = hpu_loader.importlib.import_module

    def fake_import(name):
        if name == hpu_loader.HC_FUSED_SCHEMA_MODULE:
            log.append(f"import {name}")
        return real_import(name)

    monkeypatch.setattr(hpu_loader.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch.ops, "load_library", lambda path: log.append(f"load {path}"))
    return log


def _package(monkeypatch, tmp_path, *, combined: bytes | None = None, members: bool = True):
    """A fake installed package: flashinfer_gaudi/lib with our two libraries and, optionally, a combined database."""
    package_dir = tmp_path / "flashinfer_gaudi"
    lib = package_dir / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    if members:
        (lib / hpu_loader.HC_FUSED_KERNEL_DB).write_bytes(b"")
        (lib / "hc_fused_ops.cpython-310-x86_64-linux-gnu.so").write_bytes(b"")
    if combined is not None:
        (lib / _native.COMBINED_KERNEL_DB).write_bytes(combined)
    return lib


def test_missing_files_are_named(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, members=False)
    assert hpu_loader.load_hc_fused_extension() is False
    errors = hpu_loader.hc_fused_extension_errors()
    assert len(errors) == 2
    assert str(lib / "libhc_fused_gaudi2.so") in errors[0] and "does not exist" in errors[0]
    assert "hc_fused_ops" in errors[1]
    assert events == []
    assert _native.os.environ["GC_KERNEL_PATH"] == ""


def test_colon_list_without_a_combined_database(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)
    assert hpu_loader.load_hc_fused_extension() is True
    assert hpu_loader.hc_fused_extension_errors() == ()
    entries = _native.os.environ["GC_KERNEL_PATH"].split(":")
    assert entries[0] == str(lib / "libhc_fused_gaudi2.so")
    # Kernel database first, then the schemas, then the extension.
    assert events == [
        f"import {hpu_loader.HC_FUSED_SCHEMA_MODULE}", f"load {lib / 'hc_fused_ops.cpython-310-x86_64-linux-gnu.so'}"
    ]


def test_loads_once(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path)
    assert hpu_loader.load_hc_fused_extension() is True
    assert hpu_loader.load_hc_fused_extension() is True
    assert len([event for event in events if event.startswith("load ")]) == 1


def test_recipe_cache_blocks_the_colon_list(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path)
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/cache,false,8192")
    assert hpu_loader.load_hc_fused_extension() is False
    (error, ) = hpu_loader.hc_fused_extension_errors()
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in error and "libhc_fused_gaudi2.so" in error
    assert events == []  # no HPU kernel is registered for a GUID no kernel database serves
    assert _native.os.environ["GC_KERNEL_PATH"] == ""


def test_combined_database_that_serves_the_library_is_the_only_entry(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=b"\x7fELF...libgdn_fused_gaudi2.so\0libhc_fused_gaudi2.so\0...")
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/cache,false,8192")  # allowed: one regular file
    assert hpu_loader.load_hc_fused_extension() is True
    assert _native.os.environ["GC_KERNEL_PATH"] == str(lib / _native.COMBINED_KERNEL_DB)
    assert [event.split(" ")[0] for event in events] == ["import", "load"]


def test_old_combined_database_falls_back_to_the_colon_list(monkeypatch, tmp_path, events):
    """An adapter built before this track does not name our library: its GUIDs would be missing."""
    old_adapter = b"\x7fELF...libgdn_fused_gaudi2.so\0libflashinfer_gaudi_kernels.so\0"
    lib = _package(monkeypatch, tmp_path, combined=old_adapter)
    assert hpu_loader.load_hc_fused_extension() is True
    entries = _native.os.environ["GC_KERNEL_PATH"].split(":")
    assert entries[0] == str(lib / "libhc_fused_gaudi2.so")


def test_old_combined_database_with_a_recipe_cache_turns_the_fused_path_off(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, combined=b"\x7fELF...libgdn_fused_gaudi2.so\0")
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/cache,false,8192")
    assert hpu_loader.load_hc_fused_extension() is False
    (error, ) = hpu_loader.hc_fused_extension_errors()
    assert "does not serve libhc_fused_gaudi2.so" in error and "--combined-only" in error
    assert events == []


def test_combined_database_does_not_serve_a_library_from_another_folder(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=b"libhc_fused_gaudi2.so")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "libhc_fused_gaudi2.so").write_bytes(b"")
    monkeypatch.setenv(hpu_loader.HC_FUSED_KERNEL_LIBRARY_ENV, str(other / "libhc_fused_gaudi2.so"))
    combined = lib / _native.COMBINED_KERNEL_DB
    assert hpu_loader.combined_db_serves(combined, str(other / "libhc_fused_gaudi2.so")) is False
    assert hpu_loader.load_hc_fused_extension() is True
    assert _native.os.environ["GC_KERNEL_PATH"].split(":")[0] == str(other / "libhc_fused_gaudi2.so")


def test_members_override_decides_what_the_combined_database_serves(monkeypatch, tmp_path):
    lib = _package(monkeypatch, tmp_path, combined=b"")
    ours = str(lib / "libhc_fused_gaudi2.so")
    monkeypatch.setenv(hpu_loader.COMBINED_MEMBERS_ENV, f"/x/libgdn_fused_gaudi2.so:{ours}")
    assert hpu_loader.combined_db_serves(lib / _native.COMBINED_KERNEL_DB, ours) is True
    monkeypatch.setenv(hpu_loader.COMBINED_MEMBERS_ENV, "/x/libgdn_fused_gaudi2.so")
    assert hpu_loader.combined_db_serves(lib / _native.COMBINED_KERNEL_DB, ours) is False


def test_extension_load_failure_is_recorded(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path)

    def broken(path):
        raise OSError("undefined symbol: _ZN6habana9custom_op20registerUserCustomOpE")

    monkeypatch.setattr(torch.ops, "load_library", broken)
    assert hpu_loader.load_hc_fused_extension() is False
    (error, ) = hpu_loader.hc_fused_extension_errors()
    assert "hc_fused_ops" in error and "undefined symbol" in error


def test_forwards_to_the_native_loader_once_it_exists(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, members=False)
    monkeypatch.setattr(_native, "load_hc_fused_extension", lambda: True, raising=False)
    monkeypatch.setattr(_native, "hc_fused_extension_errors", lambda: ("from native", ), raising=False)
    assert hpu_loader.load_hc_fused_extension() is True
    assert hpu_loader.hc_fused_extension_errors() == ("from native", )
    assert events == []
