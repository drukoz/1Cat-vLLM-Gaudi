# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``hpu_loader.py``: it is the package loader, not a second one.

The loader's own tests (ordering, ``GC_KERNEL_PATH``, recipe cache, combined database, failures)
are ``tests/unit_tests/gdn_core_decode/test_hc_fused_loader.py``; this file only checks that the
names the card test imports are ``flashinfer_gaudi._native``'s.
"""

from __future__ import annotations

import pytest

from flashinfer_gaudi import _native
from tests.unit_tests.hc_fused import hpu_loader


@pytest.fixture(autouse=True)
def clean_loader_state(monkeypatch):
    monkeypatch.setattr(_native, "_HC_FUSED_LOADED", None)
    monkeypatch.setattr(_native, "_HC_FUSED_ERRORS", [])
    monkeypatch.setattr(_native, "_LOAD_ERRORS", [])
    monkeypatch.setattr(_native, "_COMBINED_DB_ACTIVE", None)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    for name in ("HC_FUSED_GAUDI_KERNEL_LIBRARY", "HC_FUSED_GAUDI_NATIVE_LIBRARY", "GDN_FUSED_GAUDI_KERNEL_LIBRARY",
                 "PT_HPU_RECIPE_CACHE_CONFIG", "FLASHINFER_GAUDI_COMBINED_KERNEL_DB",
                 "FLASHINFER_GAUDI_COMBINED_MEMBERS"):
        monkeypatch.delenv(name, raising=False)


def test_the_names_are_the_package_loaders():
    assert hpu_loader.load_hc_fused_extension is _native.load_hc_fused_extension
    assert hpu_loader.hc_fused_extension_errors is _native.hc_fused_extension_errors
    assert hpu_loader.combined_db_serves is _native.combined_db_serves_hc_fused
    assert hpu_loader.HC_FUSED_KERNEL_DB == _native.HC_FUSED_KERNEL_DB == "libhc_fused_gaudi2.so"
    assert hpu_loader.HC_FUSED_KERNEL_LIBRARY_ENV == "HC_FUSED_GAUDI_KERNEL_LIBRARY"
    assert hpu_loader.HC_FUSED_NATIVE_LIBRARY_ENV == "HC_FUSED_GAUDI_NATIVE_LIBRARY"
    assert hpu_loader.HC_FUSED_SCHEMA_MODULE == "flashinfer_gaudi.hc_fused"


def test_a_missing_build_is_reported_through_it(monkeypatch, tmp_path):
    """What test_hpu_op.py prints when it skips: the two missing files."""
    package_dir = tmp_path / "flashinfer_gaudi"
    (package_dir / "lib").mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))

    assert hpu_loader.load_hc_fused_extension() is False
    errors = hpu_loader.hc_fused_extension_errors()
    assert len(errors) == 2 and all("file does not exist" in error for error in errors)
    assert "libhc_fused_gaudi2.so" in errors[0] and "hc_fused_ops" in errors[1]
