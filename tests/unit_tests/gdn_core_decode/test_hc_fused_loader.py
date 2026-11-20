# SPDX-License-Identifier: Apache-2.0
"""CPU tests for load_hc_fused_extension: ordering, GC_KERNEL_PATH, the combined database and failures.

No card: the library files are fakes and ``torch.ops.load_library`` is recorded, not run. The loader
follows the rules of ``load_gdn_fused_extension`` (``test_native_loader.py``), with one more check:
the combined kernel database is only relied on when it serves ``libhc_fused_gaudi2.so``.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from flashinfer_gaudi import _native

KERNEL_DB = "libhc_fused_gaudi2.so"
EXTENSION = "hc_fused_ops.cpython-310-x86_64-linux-gnu.so"
SERVING_ADAPTER = b"\x7fELF...libgdn_fused_gaudi2.so\0libhc_fused_gaudi2.so\0libflashinfer_gaudi_kernels.so\0"
OLD_ADAPTER = b"\x7fELF...libgdn_fused_gaudi2.so\0libflashinfer_gaudi_kernels.so\0"
RECIPE_CACHE = "/tmp/rc,false,1024"


@pytest.fixture(autouse=True)
def clean_loader_state(monkeypatch):
    monkeypatch.setattr(_native, "_HC_FUSED_LOADED", None)
    monkeypatch.setattr(_native, "_HC_FUSED_ERRORS", [])
    monkeypatch.setattr(_native, "_GDN_FUSED_LOADED", None)
    monkeypatch.setattr(_native, "_GDN_FUSED_ERRORS", [])
    monkeypatch.setattr(_native, "_LOAD_ERRORS", [])
    monkeypatch.setattr(_native, "_LOADED_PATHS", [])
    monkeypatch.setattr(_native, "_COMBINED_DB_ACTIVE", None)
    monkeypatch.setattr(_native, "_FLASHINFER_DB_STATE", _native.KernelDbState.MISSING)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    for name in ("HC_FUSED_GAUDI_KERNEL_LIBRARY", "HC_FUSED_GAUDI_NATIVE_LIBRARY", "GDN_FUSED_GAUDI_KERNEL_LIBRARY",
                 "GDN_FUSED_GAUDI_NATIVE_LIBRARY", "FLASHINFER_GAUDI_COMBINED_MEMBERS", "PT_HPU_RECIPE_CACHE_CONFIG",
                 "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    """Records the schema import and the extension load, in order."""
    log: list[str] = []
    real_import = _native.importlib.import_module

    def fake_import(name):
        log.append(f"import {name}")
        return real_import(name) if name == _native.HC_FUSED_SCHEMA_MODULE else name

    monkeypatch.setattr(_native.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch.ops, "load_library", lambda path: log.append(f"load {path}"))
    return log


def _package(monkeypatch, tmp_path, *, combined: bytes | None = None, members: bool = True):
    """A fake installed package: flashinfer_gaudi/lib with the two libraries and, optionally, a combined database."""
    package_dir = tmp_path / "flashinfer_gaudi"
    lib = package_dir / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    if members:
        (lib / KERNEL_DB).write_bytes(b"")
        (lib / EXTENSION).write_bytes(b"")
    if combined is not None:
        (lib / _native.COMBINED_KERNEL_DB).write_bytes(combined)
    return lib


def _installed(monkeypatch, tmp_path, *, combined: bytes | None = SERVING_ADAPTER, gdn: bool = True):
    """The layout of the rental's kernel gate: the imported package is a source tree with an empty lib/, and the
    installed libraries are named by the *_GAUDI_KERNEL_LIBRARY / *_GAUDI_NATIVE_LIBRARY variables."""
    package_dir = tmp_path / "source" / "flashinfer_gaudi"
    (package_dir / "lib").mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    installed = tmp_path / "installed" / "lib"
    installed.mkdir(parents=True)
    if combined is not None:
        (installed / _native.COMBINED_KERNEL_DB).write_bytes(combined)
    for name in (KERNEL_DB, EXTENSION, "libgdn_fused_gaudi2.so", "gdn_fused_ops.so"):
        (installed / name).write_bytes(b"")
    monkeypatch.setenv("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(installed / KERNEL_DB))
    monkeypatch.setenv("HC_FUSED_GAUDI_NATIVE_LIBRARY", str(installed / EXTENSION))
    if gdn:
        monkeypatch.setenv("GDN_FUSED_GAUDI_KERNEL_LIBRARY", str(installed / "libgdn_fused_gaudi2.so"))
        monkeypatch.setenv("GDN_FUSED_GAUDI_NATIVE_LIBRARY", str(installed / "gdn_fused_ops.so"))
    return installed


def test_names_and_overrides_follow_the_gdn_ones():
    assert _native.HC_FUSED_KERNEL_DB == KERNEL_DB
    assert _native.HC_FUSED_KERNEL_LIBRARY_ENV == "HC_FUSED_GAUDI_KERNEL_LIBRARY"
    assert _native.HC_FUSED_NATIVE_LIBRARY_ENV == "HC_FUSED_GAUDI_NATIVE_LIBRARY"
    assert _native.HC_FUSED_SCHEMA_MODULE == "flashinfer_gaudi.hc_fused"


def test_never_called_reports_nothing():
    assert _native.hc_fused_extension_loaded() is None
    assert _native.hc_fused_extension_errors() == ()
    assert _native.kernel_database_status()["hc_fused_loaded"] is None


def test_missing_files_are_named(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, members=False)

    assert _native.load_hc_fused_extension() is False
    errors = _native.hc_fused_extension_errors()
    assert len(errors) == 2
    assert str(lib / KERNEL_DB) in errors[0] and "file does not exist" in errors[0]
    assert "hc_fused_ops" in errors[1] and "file does not exist" in errors[1]
    assert all(error in _native._LOAD_ERRORS for error in errors)  # the shared list of the status line
    assert _native.gdn_fused_extension_errors() == ()  # and not the GDN loader's
    assert events == []  # no schema is defined and nothing is loaded
    assert os.environ["GC_KERNEL_PATH"] == ""
    assert _native.hc_fused_extension_loaded() is False


def test_colon_list_without_a_combined_database(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)

    assert _native.load_hc_fused_extension() is True
    assert _native.hc_fused_extension_errors() == ()
    entries = os.environ["GC_KERNEL_PATH"].split(os.pathsep)
    assert entries[0] == str(lib / KERNEL_DB)
    # The kernel database first, then the schemas, then the extension.
    assert events == [f"import {_native.HC_FUSED_SCHEMA_MODULE}", f"load {lib / EXTENSION}"]
    assert str(lib / EXTENSION) in _native._LOADED_PATHS
    assert _native.hc_fused_extension_loaded() is True


def test_loads_once(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)

    assert _native.load_hc_fused_extension() is True
    assert _native.load_hc_fused_extension() is True
    assert len([event for event in events if event.startswith("load ")]) == 1
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep).count(str(lib / KERNEL_DB)) == 1


def test_a_failure_is_not_retried(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, members=False)
    assert _native.load_hc_fused_extension() is False
    (lib / KERNEL_DB).write_bytes(b"")
    (lib / EXTENSION).write_bytes(b"")

    assert _native.load_hc_fused_extension() is False  # one attempt per process, as for the GDN extension
    assert len(_native.hc_fused_extension_errors()) == 2


def test_env_overrides_name_both_files(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, members=False)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / KERNEL_DB).write_bytes(b"")
    (other / "my_ops.so").write_bytes(b"")
    monkeypatch.setenv("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(other / KERNEL_DB))
    monkeypatch.setenv("HC_FUSED_GAUDI_NATIVE_LIBRARY", str(other / "my_ops.so"))

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep)[0] == str(other / KERNEL_DB)
    assert events[-1] == f"load {other / 'my_ops.so'}"


def test_recipe_cache_blocks_the_colon_list(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", RECIPE_CACHE)

    assert _native.load_hc_fused_extension() is False
    (error, ) = _native.hc_fused_extension_errors()
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in error and "GC_KERNEL_PATH" in error and KERNEL_DB in error
    assert error in _native._LOAD_ERRORS
    assert events == []  # no HPU kernel is registered for a GUID no kernel database serves
    assert os.environ["GC_KERNEL_PATH"] == ""
    assert (lib / KERNEL_DB).is_file()


def test_recipe_cache_empty_value_does_not_block(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "")

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].startswith(str(lib / KERNEL_DB))


def test_combined_database_that_serves_the_library_is_the_only_entry(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=SERVING_ADAPTER)
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", RECIPE_CACHE)  # allowed: one regular file

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"] == str(lib / _native.COMBINED_KERNEL_DB)
    assert os.pathsep not in os.environ["GC_KERNEL_PATH"]
    assert [event.split(" ")[0] for event in events] == ["import", "load"]
    assert _native.hc_fused_extension_errors() == ()
    status = _native.kernel_database_status()
    assert status["combined_active"] is True and status["hc_fused_loaded"] is True


def test_combined_database_replaces_the_stock_default(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=SERVING_ADAPTER)
    monkeypatch.setenv("GC_KERNEL_PATH", "/usr/lib/habanalabs/libtpc_kernels.so")

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"] == str(lib / _native.COMBINED_KERNEL_DB)
    assert _native._LOAD_ERRORS == []


def test_combined_database_disabled_by_the_switch(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, combined=SERVING_ADAPTER)
    monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", RECIPE_CACHE)

    assert _native.load_hc_fused_extension() is False
    assert os.environ["GC_KERNEL_PATH"] == ""
    assert events == []
    (error, ) = _native.hc_fused_extension_errors()
    assert "PT_HPU_RECIPE_CACHE_CONFIG" in error and KERNEL_DB in error


def test_combined_database_switched_off_without_a_recipe_cache_is_a_colon_list(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=SERVING_ADAPTER)
    monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep)[0] == str(lib / KERNEL_DB)
    assert str(lib / _native.COMBINED_KERNEL_DB) not in os.environ["GC_KERNEL_PATH"]


def test_combined_database_yields_to_a_foreign_gc_kernel_path(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=SERVING_ADAPTER)
    monkeypatch.setenv("GC_KERNEL_PATH", "/opt/other.so")

    assert _native.load_hc_fused_extension() is True  # as the GDN loader: the library joins the foreign list
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep) == [str(lib / KERNEL_DB), "/opt/other.so"]
    (error, ) = _native._LOAD_ERRORS
    assert "/opt/other.so" in error and "FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0" in error


def test_old_combined_database_falls_back_to_the_colon_list(monkeypatch, tmp_path, events):
    """An adapter built before the hyper-connection library was a member does not name it: its GUIDs would
    be missing from the single file, and the first graph with a fused op would fail to compile."""
    lib = _package(monkeypatch, tmp_path, combined=OLD_ADAPTER)

    assert _native.load_hc_fused_extension() is True
    entries = os.environ["GC_KERNEL_PATH"].split(os.pathsep)
    assert entries[0] == str(lib / KERNEL_DB)


def test_old_combined_database_that_is_active_keeps_serving_the_other_members(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=OLD_ADAPTER)
    combined = lib / _native.COMBINED_KERNEL_DB
    assert _native._configure_kernel_database_path() is _native.KernelDbState.COMBINED  # every process, at import

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep) == [str(lib / KERNEL_DB), str(combined)]


def test_old_combined_database_with_a_recipe_cache_turns_the_fused_path_off(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=OLD_ADAPTER)
    combined = lib / _native.COMBINED_KERNEL_DB
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", RECIPE_CACHE)
    assert _native._configure_kernel_database_path() is _native.KernelDbState.COMBINED

    assert _native.load_hc_fused_extension() is False
    (error, ) = _native.hc_fused_extension_errors()
    assert "does not serve libhc_fused_gaudi2.so" in error and "--combined-only" in error
    assert events == []
    assert os.environ["GC_KERNEL_PATH"] == str(combined)  # never a colon list next to a recipe cache


def test_combined_database_does_not_serve_a_library_from_another_folder(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path, combined=SERVING_ADAPTER)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / KERNEL_DB).write_bytes(b"")
    monkeypatch.setenv("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(other / KERNEL_DB))
    combined = lib / _native.COMBINED_KERNEL_DB

    assert _native.combined_db_serves_hc_fused(combined, str(other / KERNEL_DB)) is False
    assert _native.combined_db_serves_hc_fused(combined, str(lib / KERNEL_DB)) is True
    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep)[0] == str(other / KERNEL_DB)


def test_members_override_decides_what_the_combined_database_serves(monkeypatch, tmp_path):
    lib = _package(monkeypatch, tmp_path, combined=b"")
    ours = str(lib / KERNEL_DB)
    monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_MEMBERS", f"/x/libgdn_fused_gaudi2.so:{ours}")
    assert _native.combined_db_serves_hc_fused(lib / _native.COMBINED_KERNEL_DB, ours) is True
    monkeypatch.setenv("FLASHINFER_GAUDI_COMBINED_MEMBERS", "/x/libgdn_fused_gaudi2.so")
    assert _native.combined_db_serves_hc_fused(lib / _native.COMBINED_KERNEL_DB, ours) is False


def test_an_unreadable_combined_database_serves_nothing(monkeypatch, tmp_path):
    lib = _package(monkeypatch, tmp_path)
    assert _native.combined_db_serves_hc_fused(lib / _native.COMBINED_KERNEL_DB, str(lib / KERNEL_DB)) is False


def test_extension_load_failure_keeps_the_dlopen_cause(monkeypatch, tmp_path, events):
    lib = _package(monkeypatch, tmp_path)
    cause = "undefined symbol: _ZN6habana9custom_op20registerUserCustomOpE"

    def broken(path):
        try:
            raise OSError(cause)
        except OSError as exc:
            raise OSError(f"Could not load this library: {path}") from exc

    monkeypatch.setattr(torch.ops, "load_library", broken)

    assert _native.load_hc_fused_extension() is False
    extension = lib / EXTENSION
    assert _native.hc_fused_extension_errors() == (
        f"{extension}: OSError: Could not load this library: {extension}; caused by OSError: {cause}", )
    assert str(extension) not in _native._LOADED_PATHS


def test_gate_layout_runs_through_the_combined_database_next_to_the_installed_libraries(monkeypatch, tmp_path, events):
    installed = _installed(monkeypatch, tmp_path)
    combined = installed / _native.COMBINED_KERNEL_DB
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", RECIPE_CACHE)

    assert _native._configure_kernel_database_path() is _native.KernelDbState.COMBINED
    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"] == str(combined)
    assert events[-1] == f"load {installed / EXTENSION}"


def test_gate_layout_without_a_combined_database_is_a_colon_list(monkeypatch, tmp_path, events):
    installed = _installed(monkeypatch, tmp_path, combined=None)

    assert _native.load_hc_fused_extension() is True
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep)[0] == str(installed / KERNEL_DB)


def test_both_fused_extensions_share_one_kernel_path(monkeypatch, tmp_path, events):
    installed = _installed(monkeypatch, tmp_path, combined=None)

    assert _native.load_gdn_fused_extension() is True
    assert _native.load_hc_fused_extension() is True
    entries = os.environ["GC_KERNEL_PATH"].split(os.pathsep)
    assert entries[:2] == [str(installed / KERNEL_DB), str(installed / "libgdn_fused_gaudi2.so")]
    status = _native.kernel_database_status()
    assert status["gdn_fused_loaded"] is True and status["hc_fused_loaded"] is True


def test_the_hc_loader_leaves_the_gdn_loader_alone(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, members=False)

    assert _native.load_hc_fused_extension() is False
    assert _native._GDN_FUSED_LOADED is None
    status = _native.kernel_database_status()
    assert status["gdn_fused_loaded"] is None and status["hc_fused_loaded"] is False
    assert json.loads(json.dumps(status)) == status


def test_reset_for_tests_forgets_the_attempt(monkeypatch, tmp_path, events):
    _package(monkeypatch, tmp_path, members=False)
    assert _native.load_hc_fused_extension() is False

    _native._reset_native_state_for_tests()

    assert _native.hc_fused_extension_loaded() is None
    assert _native.hc_fused_extension_errors() == ()


def test_an_unexpected_error_inside_the_loader_is_recorded_not_raised(monkeypatch, tmp_path, events):
    """register_ops calls the loader at server start: whatever goes wrong, the answer is False and a reason."""
    _package(monkeypatch, tmp_path)

    def broken():
        raise PermissionError("lib folder is not readable")

    monkeypatch.setattr(_native, "_combined_kernel_db_path", broken)

    assert _native.load_hc_fused_extension() is False
    (error, ) = _native.hc_fused_extension_errors()
    assert "PermissionError: lib folder is not readable" in error
    assert events == []
    assert _native.load_hc_fused_extension() is False  # and it stays False
