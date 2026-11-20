# SPDX-License-Identifier: Apache-2.0
"""CPU tests for load_gdn_fused_extension: ordering, GC_KERNEL_PATH and failures."""

from __future__ import annotations

import json
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
    monkeypatch.setattr(_native, "_COMBINED_DB_ACTIVE", None)
    monkeypatch.setattr(_native, "_FLASHINFER_DB_STATE", _native.KernelDbState.MISSING)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    monkeypatch.delenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", raising=False)
    monkeypatch.delenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", raising=False)
    monkeypatch.delenv("PT_HPU_RECIPE_CACHE_CONFIG", raising=False)
    monkeypatch.delenv("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", raising=False)


def _make_libraries(tmp_path, monkeypatch):
    kernel_db = tmp_path / "libgdn_fused_gaudi2.so"
    extension = tmp_path / "gdn_fused_ops.so"
    kernel_db.write_bytes(b"")
    extension.write_bytes(b"")
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel_db))
    monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(extension))
    return kernel_db, extension


def _make_combined(monkeypatch, tmp_path):
    """Point the loader at a fake package whose lib/ holds the combined database and the fused members."""
    package_dir = tmp_path / "flashinfer_gaudi"
    lib = package_dir / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    (lib / _native.COMBINED_KERNEL_DB).touch()
    return _make_libraries(lib, monkeypatch)


def _make_installed_libraries(monkeypatch, tmp_path, combined=True, kernel_name="libgdn_fused_gaudi2.so"):
    """The layout of the rental's kernel gate (G1b): the imported package is a source tree with an empty lib/,
    and the installed libraries are named by GDN_FUSED_GAUDI_KERNEL_LIBRARY / GDN_FUSED_GAUDI_NATIVE_LIBRARY."""
    package_dir = tmp_path / "source" / "flashinfer_gaudi"
    (package_dir / "lib").mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    installed = tmp_path / "installed" / "lib"
    installed.mkdir(parents=True)
    if combined:
        (installed / _native.COMBINED_KERNEL_DB).touch()
    kernel_db = installed / kernel_name
    extension = installed / "gdn_fused_ops.so"
    kernel_db.write_bytes(b"")
    extension.write_bytes(b"")
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel_db))
    monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(extension))
    return installed, kernel_db, extension


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


def test_combined_database_is_the_only_kernel_path(tmp_path, monkeypatch):
    kernel_db, extension = _make_combined(monkeypatch, tmp_path)
    combined = tmp_path / "flashinfer_gaudi" / "lib" / _native.COMBINED_KERNEL_DB
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    calls: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: calls.append(path))

    assert _native.load_gdn_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"] == str(combined)
    assert os.pathsep not in os.environ["GC_KERNEL_PATH"]
    assert calls == [str(extension)]
    assert _native.gdn_fused_extension_errors() == ()
    assert kernel_db.is_file()


def test_combined_database_disabled_by_the_switch(tmp_path, monkeypatch):
    _make_combined(monkeypatch, tmp_path)
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    calls: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: calls.append(path))

    assert _native.load_gdn_fused_extension() is False
    assert os.environ["GC_KERNEL_PATH"] == ""
    assert calls == []
    (error, ) = _native.gdn_fused_extension_errors()
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in error
    assert "libgdn_fused_gaudi2.so" in error


def test_combined_database_yields_to_a_foreign_gc_kernel_path(tmp_path, monkeypatch):
    kernel_db, _ = _make_combined(monkeypatch, tmp_path)
    _record_imports(monkeypatch, [])
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)
    monkeypatch.setenv("GC_KERNEL_PATH", "/opt/other.so")

    assert _native.load_gdn_fused_extension() is True
    (error, ) = _native._LOAD_ERRORS
    assert "/opt/other.so" in error
    assert "FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0" in error
    assert os.environ["GC_KERNEL_PATH"].startswith(str(kernel_db))


def test_combined_database_replaces_the_stock_default(tmp_path, monkeypatch):
    _make_combined(monkeypatch, tmp_path)
    combined = tmp_path / "flashinfer_gaudi" / "lib" / _native.COMBINED_KERNEL_DB
    _record_imports(monkeypatch, [])
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)
    monkeypatch.setenv("GC_KERNEL_PATH", "/usr/lib/habanalabs/libtpc_kernels.so")

    assert _native.load_gdn_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"] == str(combined)
    assert _native._LOAD_ERRORS == []


def test_combined_database_found_next_to_the_env_kernel_library(tmp_path, monkeypatch):
    installed, kernel_db, extension = _make_installed_libraries(monkeypatch, tmp_path)
    combined = installed / _native.COMBINED_KERNEL_DB
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    calls: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: calls.append(path))

    assert _native._combined_kernel_db_path() == combined
    # the import-time step of every process, then the fused loader
    assert _native._configure_kernel_database_path() is _native.KernelDbState.COMBINED
    assert os.environ["GC_KERNEL_PATH"] == str(combined)
    assert _native.load_gdn_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"] == str(combined)
    assert os.pathsep not in os.environ["GC_KERNEL_PATH"]
    assert calls == [str(extension)]
    assert _native.gdn_fused_extension_errors() == ()
    assert _native._LOAD_ERRORS == []
    assert kernel_db.is_file()


def test_package_combined_database_wins_over_the_env_directory(tmp_path, monkeypatch):
    installed, _, _ = _make_installed_libraries(monkeypatch, tmp_path)
    assert _native._combined_kernel_db_path() == installed / _native.COMBINED_KERNEL_DB  # the package lib/ is empty
    package_combined = tmp_path / "source" / "flashinfer_gaudi" / "lib" / _native.COMBINED_KERNEL_DB
    package_combined.touch()

    assert _native._combined_kernel_db_path() == package_combined


def test_env_directory_combined_database_disabled_by_the_switch(tmp_path, monkeypatch):
    installed, _, _ = _make_installed_libraries(monkeypatch, tmp_path)
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    calls: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: calls.append(path))
    assert _native._combined_kernel_db_path() == installed / _native.COMBINED_KERNEL_DB  # found without the switch

    monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")

    assert _native._combined_kernel_db_path() is None
    assert _native._configure_kernel_database_path() is _native.KernelDbState.MISSING
    assert _native.load_gdn_fused_extension() is False
    assert os.environ["GC_KERNEL_PATH"] == ""
    assert calls == []
    (error, ) = _native.gdn_fused_extension_errors()
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in error
    assert "libgdn_fused_gaudi2.so" in error


def test_env_directory_combined_database_needs_the_library_it_serves(tmp_path, monkeypatch):
    # The combined database serves the libgdn_fused_gaudi2.so next to itself, so a kernel library with another
    # name gets no combined database from its folder: it would be a different file, or none at all.
    installed, _, _ = _make_installed_libraries(monkeypatch, tmp_path)
    assert _native._combined_kernel_db_path() == installed / _native.COMBINED_KERNEL_DB
    other = installed / "libgdn_fused_gaudi2.dev.so"
    other.write_bytes(b"")
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(other))
    assert _native._combined_kernel_db_path() is None

    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(installed / "missing" / "libgdn_fused_gaudi2.so"))
    assert _native._combined_kernel_db_path() is None


def test_env_directory_combined_database_yields_to_the_package_kernel_database(tmp_path, monkeypatch):
    # A package with its own FlashInfer database and no combined file keeps the per-library behaviour: the other
    # folder's combined file serves the libraries next to itself, so it would leave this package's database out.
    installed, kernel_db, _ = _make_installed_libraries(monkeypatch, tmp_path)
    assert _native._combined_kernel_db_path() == installed / _native.COMBINED_KERNEL_DB
    package_db = tmp_path / "source" / "flashinfer_gaudi" / "lib" / "libflashinfer_gaudi_kernels.so"
    package_db.touch()
    _record_imports(monkeypatch, [])
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)

    assert _native._combined_kernel_db_path() is None
    assert _native._configure_kernel_database_path() is _native.KernelDbState.REGISTERED
    assert _native.load_gdn_fused_extension() is True
    configured = os.environ["GC_KERNEL_PATH"].split(os.pathsep)
    assert configured[:2] == [str(kernel_db), str(package_db)]
    assert str(installed / _native.COMBINED_KERNEL_DB) not in configured


@pytest.mark.parametrize(
    "case, package_database, state",
    [
        ("absent", True, "BLOCKED"),  # no combined file anywhere
        ("package_has_its_own_database", True, "BLOCKED"),  # a combined file next to the env library, not used
        ("switched_off", False, "MISSING"),  # FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0
        ("other_library_name", False, "MISSING"),  # the env library is not the one the combined file serves
        ("foreign_gc_kernel_path", False, "MISSING"),  # the combined file is found and yields to the foreign value
    ],
)
def test_recipe_cache_without_a_usable_combined_database_makes_no_colon_list(tmp_path, monkeypatch, case,
                                                                              package_database, state):
    kernel_name = "libgdn_fused_gaudi2.dev.so" if case == "other_library_name" else "libgdn_fused_gaudi2.so"
    installed, _, _ = _make_installed_libraries(monkeypatch,
                                                 tmp_path,
                                                 combined=case != "absent",
                                                 kernel_name=kernel_name)
    if package_database:  # the per-library path then has a FlashInfer database it would prepend
        (tmp_path / "source" / "flashinfer_gaudi" / "lib" / "libflashinfer_gaudi_kernels.so").touch()
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    if case == "switched_off":
        monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")
    before = "/opt/other.so" if case == "foreign_gc_kernel_path" else ""
    monkeypatch.setenv("GC_KERNEL_PATH", before)
    calls: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", lambda path: calls.append(path))

    assert _native._configure_kernel_database_path().name == state
    assert _native.load_gdn_fused_extension() is False
    assert os.environ["GC_KERNEL_PATH"] == before
    assert os.pathsep not in os.environ["GC_KERNEL_PATH"]
    assert calls == []
    if case == "foreign_gc_kernel_path":
        # the loader did look next to the env library: it names that combined file and leaves the foreign value
        assert any(
            str(installed / _native.COMBINED_KERNEL_DB) in error and "/opt/other.so" in error
            for error in _native._LOAD_ERRORS)
    status = _native.kernel_database_status()
    assert status["state"] == state
    assert status["gc_kernel_path"] == before
    assert status["combined_active"] is False
    assert status["combined_kernel_db"] is None
    assert status["gdn_fused_loaded"] is False
    assert any("PT_HPU_RECIPE_CACHE_CONFIG" in error for error in status["load_errors"])


def test_kernel_database_status_reports_what_the_loader_decided(tmp_path, monkeypatch):
    installed, _, extension = _make_installed_libraries(monkeypatch, tmp_path)
    combined = installed / _native.COMBINED_KERNEL_DB
    _record_imports(monkeypatch, [])
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/tmp/rc,false,1024")
    monkeypatch.setattr(torch.ops, "load_library", lambda path: None)

    assert _native.kernel_database_status() == {
        "state": "MISSING",
        "gc_kernel_path": "",
        "combined_active": False,
        "combined_kernel_db": None,
        "gdn_fused_loaded": None,
        "hc_fused_loaded": None,
        "load_errors": [],
    }

    _native._configure_kernel_database_path()
    assert _native.load_gdn_fused_extension() is True
    status = _native.kernel_database_status()
    assert status == {
        "state": "COMBINED",
        "gc_kernel_path": str(combined),
        "combined_active": True,
        "combined_kernel_db": str(combined),
        "gdn_fused_loaded": True,
        "hc_fused_loaded": None,  # the hyper-connection loader runs only with VLLM_HPU_HC_FUSED=1
        "load_errors": [],
    }
    assert json.loads(json.dumps(status)) == status  # plain data: it goes into a log line and a JSON file

    # Another component replaces GC_KERNEL_PATH afterwards: the combined database is no longer what SynapseAI reads.
    monkeypatch.setenv("GC_KERNEL_PATH", "/opt/other.so")
    status = _native.kernel_database_status()
    assert status["gc_kernel_path"] == "/opt/other.so"
    assert status["combined_active"] is False
    assert status["combined_kernel_db"] == str(combined)


def test_kernel_database_status_lists_the_load_errors(tmp_path, monkeypatch):
    kernel_db = tmp_path / "libgdn_fused_gaudi2.so"
    extension = tmp_path / "gdn_fused_ops.so"
    monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel_db))
    monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(extension))

    assert _native.load_gdn_fused_extension() is False
    status = _native.kernel_database_status()
    assert status["gdn_fused_loaded"] is False
    assert status["load_errors"] == [f"{kernel_db}: file does not exist", f"{extension}: file does not exist"]
    assert status["load_errors"] is not _native._LOAD_ERRORS  # a copy: the caller cannot change the loader's list


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
