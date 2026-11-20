# SPDX-License-Identifier: Apache-2.0
"""Entry points of ``libhc_fused_gaudi2.so``, called the way the graph compiler calls them.

``lib_check.cpp`` opens the CMake-built library with ``dlopen`` and checks ``GetKernelGuids`` (two
call protocol), ``InstantiateTpcKernel`` (the index space, the access patterns, the scalar
arguments, the ELF handshake and every rejection), ``GetSupportedDataLayouts`` (one all-'x'
layout for each of our GUIDs), ``GetShapeInference`` (GUID named through ``pGuid``, through
``guid``, or not at all) and ``GetSuggestedManipulation``.  Needs the TPC toolchain to build the
library; skips without it.  Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit_tests.hc_fused import sim_build
from tests.unit_tests.hc_fused.hc_case import GUIDS, KINDS

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))

ENTRY_POINTS = ("GetKernelGuids", "InstantiateTpcKernel", "GetShapeInference", "GetSupportedDataLayouts",
                "GetSuggestedManipulation")


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


@pytest.fixture(scope="module")
def check_lines(build: Path) -> list[str]:
    proc = sim_build.run([build / "lib_check", build / "lib" / sim_build.LIBRARY], timeout=600)
    print(proc.stdout)
    return proc.stdout.splitlines()


def test_every_entry_point_check_passes(check_lines: list[str]) -> None:
    failed = [line for line in check_lines if line.startswith("FAIL")]
    passed = [line for line in check_lines if line.startswith("PASS")]
    assert failed == []
    assert len(passed) >= 80
    assert check_lines[-1] == "OK: 0 failure(s)"


def test_guid_list_is_the_six_kernels_in_order(check_lines: list[str]) -> None:
    guids = [line.split(" ")[2] for line in check_lines if line.startswith("GUID ")]
    assert guids == [GUIDS[kind] for kind in KINDS]
    assert all(len(guid) < 64 for guid in guids)


def test_library_exports_the_entry_points(build: Path) -> None:
    symbols = sim_build.run(["nm", "-D", "--defined-only", build / "lib" / sim_build.LIBRARY]).stdout
    exported = {line.split()[-1] for line in symbols.splitlines() if " T " in line}
    assert set(ENTRY_POINTS) <= exported
    assert "GetLibVersion" not in exported  # the stock library's version is the only one the adapter reports


def test_library_needs_no_habana_runtime(build: Path) -> None:
    """The kernel database links nothing from Habana: the graph compiler only dlopens it."""
    needed = sim_build.run(["readelf", "-d", build / "lib" / sim_build.LIBRARY]).stdout
    libraries = [line.split("[")[1].rstrip("]") for line in needed.splitlines() if "(NEEDED)" in line]
    assert all(not name.startswith(("libhabana", "libSynapse", "libtpc", "libhl")) for name in libraries), libraries


def test_library_serves_its_guids_through_the_combined_kernel_database(build: Path, tmp_path: Path) -> None:
    """As a member of csrc/kernel_db_adapter (the single GC_KERNEL_PATH file of a server with a recipe cache).

    The adapter is built from its own CMake project with its fake members; ``libhc_fused_gaudi2.so``
    is given as the only custom member and the adapter's full fake member stands in for Intel's
    stock library.  Skips when the adapter itself does not build (it belongs to another track).
    """
    adapter_source = sim_build.REPO_ROOT / "csrc" / "kernel_db_adapter"
    adapter_build = tmp_path / "adapter"
    try:
        sim_build.run([
            "cmake", "-S", adapter_source, "-B", adapter_build, f"-DTPC_INCLUDE_DIR={sim_build.TOOLS_INCLUDE}",
            f"-DHABANA_INCLUDE_DIR={sim_build.host_include()}"
        ],
                      timeout=600)
        sim_build.run(
            ["cmake", "--build", adapter_build, "--target", "flashinfer_gaudi_combined_gaudi2", "fake_member_a"],
            timeout=1800)
    except RuntimeError as exc:
        pytest.skip(f"csrc/kernel_db_adapter does not build here: {str(exc)[-400:]}")
    adapter = adapter_build / "libflashinfer_gaudi_combined_gaudi2.so"
    stock = adapter_build / "libfake_member_a.so"
    assert adapter.is_file() and stock.is_file()
    env = dict(os.environ,
               FLASHINFER_GAUDI_COMBINED_MEMBERS=str(build / "lib" / sim_build.LIBRARY),
               FLASHINFER_GAUDI_STOCK_KERNEL_DB=str(stock))
    proc = subprocess.run([str(build / "lib_check"), str(adapter), "--combined"],
                          capture_output=True,
                          text=True,
                          timeout=600,
                          env=env)
    print(proc.stdout)
    print(proc.stderr)
    lines = proc.stdout.splitlines()
    assert [line for line in lines if line.startswith("FAIL")] == []
    assert proc.returncode == 0
    assert len([line for line in lines if line.startswith("PASS")]) >= 20
    assert lines[-1] == "OK: 0 failure(s)"


def test_library_is_served_as_a_sibling_of_the_installed_combined_kernel_database(build: Path, tmp_path: Path) -> None:
    """The installed layout: ``libhc_fused_gaudi2.so`` next to the combined file, no members override.

    The adapter finds the library by its name in the sibling list (``csrc/kernel_db_adapter/adapter.cpp``),
    and the Python loader's test for "this adapter serves the library" holds for the real build: the
    name is in the adapter's bytes.  Skips when the adapter itself does not build.
    """
    adapter_source = sim_build.REPO_ROOT / "csrc" / "kernel_db_adapter"
    adapter_build = tmp_path / "adapter"
    try:
        sim_build.run([
            "cmake", "-S", adapter_source, "-B", adapter_build, f"-DTPC_INCLUDE_DIR={sim_build.TOOLS_INCLUDE}",
            f"-DHABANA_INCLUDE_DIR={sim_build.host_include()}"
        ],
                      timeout=600)
        sim_build.run(
            ["cmake", "--build", adapter_build, "--target", "flashinfer_gaudi_combined_gaudi2", "fake_member_a"],
            timeout=1800)
    except RuntimeError as exc:
        pytest.skip(f"csrc/kernel_db_adapter does not build here: {str(exc)[-400:]}")
    installed = tmp_path / "installed"
    installed.mkdir()
    adapter = installed / "libflashinfer_gaudi_combined_gaudi2.so"
    shutil.copy2(adapter_build / "libflashinfer_gaudi_combined_gaudi2.so", adapter)
    shutil.copy2(build / "lib" / sim_build.LIBRARY, installed / sim_build.LIBRARY)
    assert sim_build.LIBRARY.encode() in adapter.read_bytes()
    env = {name: value for name, value in os.environ.items() if name != "FLASHINFER_GAUDI_COMBINED_MEMBERS"}
    env["FLASHINFER_GAUDI_STOCK_KERNEL_DB"] = str(adapter_build / "libfake_member_a.so")
    proc = subprocess.run([str(build / "lib_check"), str(adapter), "--combined"],
                          capture_output=True,
                          text=True,
                          timeout=600,
                          env=env)
    print(proc.stdout)
    print(proc.stderr)
    lines = proc.stdout.splitlines()
    assert [line for line in lines if line.startswith("FAIL")] == []
    assert proc.returncode == 0
    assert lines[-1] == "OK: 0 failure(s)"
    assert "FLASHINFER_GAUDI_COMBINED_MEMBERS" not in proc.stderr  # the siblings, not an override


def test_build_script_accepts_the_library_and_refuses_a_broken_one(build: Path, tmp_path: Path) -> None:
    """``tools/build_flashinfer_gaudi.py`` checks the built library the way the combined database uses it."""
    from tools.build_flashinfer_gaudi import check_kernel_library

    assert check_kernel_library(build / "lib" / sim_build.LIBRARY) == len(KINDS)
    broken = tmp_path / sim_build.LIBRARY
    broken.write_text("not a shared object\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not load"):
        check_kernel_library(broken)
