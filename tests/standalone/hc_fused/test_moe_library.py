# SPDX-License-Identifier: Apache-2.0
"""Entry points of ``libhc_fused_gaudi2.so`` for its MoE glue GUIDs, and the build isolation of those kernels.

``moe_lib_check.cpp`` opens the CMake-built library with ``dlopen`` and checks, for the four MoE GUIDs,
``GetKernelGuids`` (they follow the six hyper-connection GUIDs), ``InstantiateTpcKernel`` (index space,
access patterns, scalar arguments, the ELF handshake and every rejection), ``GetSupportedDataLayouts``
and ``GetShapeInference`` (GUID named through ``pGuid``, through ``guid``, or not at all), directly and
through the real combined kernel database.

The build tests are the guard ``tests/standalone/gdn_core_decode/test_kernel_build.py`` is for the GDN
library: every kernel of ``csrc/hc_fused`` is compiled with one flag list and no local-memory override
(``-vlm``, ``-slm``, ``-spill-vlm``), and a MoE kernel that does not compile (a tpc-clang wrapper fails on
it), or ``-DHC_FUSED_MOE_GLUE=OFF``, costs only its own GUID: the hyper-connection kernels stay.

Needs the TPC toolchain; skips without it.  Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from tests.unit_tests.hc_fused import hc_case, moe_case, sim_build

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))

GUID_HOST_SOURCE = sim_build.REPO_ROOT / "tests" / "standalone" / "gdn_core_decode" / "guid_list_host.cpp"
HC_GUIDS = [hc_case.GUIDS[kind] for kind in hc_case.KINDS]
MOE_GUIDS = [moe_case.GUIDS[kind] for kind in moe_case.KINDS]
IL_GUIDS = list(sim_build.IL_GUIDS)  # the interleaved hyper-connection kernels, listed after the MoE glue GUIDs
GLUE_NODE_NOT_FOUND = 1
MEMORY_OVERRIDE = re.compile(r"-+(vlm|slm|spill-vlm)(=.*)?")


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


@pytest.fixture(scope="module")
def check_lines(build: Path) -> list[str]:
    proc = sim_build.run([build / "moe_lib_check", build / "lib" / sim_build.LIBRARY], timeout=600)
    print(proc.stdout)
    return proc.stdout.splitlines()


def test_every_entry_point_check_passes(check_lines: list[str]) -> None:
    failed = [line for line in check_lines if line.startswith("FAIL")]
    passed = [line for line in check_lines if line.startswith("PASS")]
    assert failed == []
    assert len(passed) >= 60
    assert check_lines[-1] == "OK: 0 failure(s)"


def test_library_serves_the_moe_guids_through_the_combined_kernel_database(build: Path, tmp_path: Path) -> None:
    """As a member of csrc/kernel_db_adapter, the single GC_KERNEL_PATH file of a server with a recipe cache."""
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
    env = dict(os.environ,
               FLASHINFER_GAUDI_COMBINED_MEMBERS=str(build / "lib" / sim_build.LIBRARY),
               FLASHINFER_GAUDI_STOCK_KERNEL_DB=str(adapter_build / "libfake_member_a.so"))
    proc = subprocess.run([str(build / "moe_lib_check"), str(adapter), "--combined"],
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


# ---------------------------------------------------------------------------
# The build: flags, and what a kernel that does not compile costs
# ---------------------------------------------------------------------------
def _run(cmd: list, *, timeout: float = 1800) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(part) for part in cmd], capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (f"command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}\n"
                                  f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}")
    return proc


def _cmake_build(build_dir: Path, *options: str) -> tuple[str, str]:
    """Configure and build the library; returns (configure output, verbose build output)."""
    configure = _run([
        "cmake", "-S", sim_build.CSRC, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release",
        f"-DTPC_INCLUDE_DIR={sim_build.TOOLS_INCLUDE}", f"-DHABANA_INCLUDE_DIR={sim_build.host_include()}", *options
    ])
    built = _run(["cmake", "--build", build_dir, "--target", "hc_fused_gaudi2", "--verbose"])
    assert (build_dir / sim_build.LIBRARY).is_file()
    return configure.stdout + configure.stderr, built.stdout + built.stderr


def _kernel_compiles(build_output: str) -> dict[str, list[str]]:
    """kernel name -> the arguments of its tpc-clang command in the verbose build output."""
    compiles = {}
    for line in build_output.splitlines():
        words = line.split()
        if not words or Path(words[0]).name not in ("tpc-clang", "tpc-clang-wrapper"):
            continue
        for kernel in sim_build.KERNELS + sim_build.MOE_KERNELS + sim_build.IL_KERNELS:
            if any(word.endswith(f"/kernels/{kernel}.c") for word in words):
                compiles[kernel] = words[1:]
    return compiles


@pytest.fixture(scope="module")
def guid_host(tmp_path_factory: pytest.TempPathFactory) -> Path:
    exe = tmp_path_factory.mktemp("guid_list_host") / "guid_list_host"
    _run(["g++", "-std=c++17", "-O1", "-Wall", GUID_HOST_SOURCE, f"-I{sim_build.TOOLS_INCLUDE}", "-ldl", "-o", exe])
    return exe


def _served(guid_host: Path, build_dir: Path) -> tuple[list[str], dict[str, int]]:
    """(the GUIDs the built library lists, GUID -> InstantiateTpcKernel return code with no tensors)."""
    out = _run([guid_host, build_dir / sim_build.LIBRARY, *HC_GUIDS, *MOE_GUIDS, *IL_GUIDS]).stdout
    guids = [line.split()[1] for line in out.splitlines() if line.startswith("guid ")]
    codes = {line.split()[1]: int(line.split()[2]) for line in out.splitlines() if line.startswith("instantiate ")}
    return guids, codes


@pytest.fixture(scope="module")
def default_build(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    build_dir = tmp_path_factory.mktemp("hc_fused_default")
    configure, built = _cmake_build(build_dir)
    return build_dir, configure, built


def test_build_compiles_every_kernel_with_the_default_flags(default_build) -> None:
    """No -vlm, -slm or -spill-vlm on any kernel of the library, and one flag list for all ten."""
    _, _, built = default_build
    compiles = _kernel_compiles(built)
    assert sorted(compiles) == sorted(sim_build.KERNELS + sim_build.MOE_KERNELS + sim_build.IL_KERNELS), built[-3000:]
    for kernel, arguments in compiles.items():
        assert not [a for a in arguments if MEMORY_OVERRIDE.fullmatch(a)], (kernel, arguments)
        flags = [a for a in arguments if not a.startswith("-I") and not a.endswith((".c", ".o")) and
                 a not in ("-c", "-o")]
        assert flags == list(sim_build.TPC_FLAGS), (kernel, flags)


def test_build_serves_the_ten_guids(default_build, guid_host: Path) -> None:
    build_dir, configure, _ = default_build
    assert "MoE glue kernels in the library: [" + ";".join(sim_build.MOE_KERNELS) + "]" in configure
    guids, codes = _served(guid_host, build_dir)
    assert guids == HC_GUIDS + MOE_GUIDS + IL_GUIDS
    assert all(code != GLUE_NODE_NOT_FOUND for code in codes.values()), codes


_PACKAGE_MODULES: dict = {}


def _package_module(name: str):
    """``flashinfer_gaudi/<name>.py`` loaded by path: the package needs the plugin, which this container lacks."""
    import importlib.util

    if name not in _PACKAGE_MODULES:
        path = sim_build.REPO_ROOT / "flashinfer_gaudi" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"flashinfer_gaudi_standalone_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PACKAGE_MODULES[name] = module
    return _PACKAGE_MODULES[name]


def test_the_python_loader_reads_the_guid_list_of_the_real_library(default_build, tmp_path: Path) -> None:
    """``flashinfer_gaudi._native.kernel_library_guids`` (ctypes, the library's own GetKernelGuids) on the real build.

    This is how a server with VLLM_HPU_MOE_GLUE=1 learns which MoE kernels its library serves.
    """
    _native, moe_glue = _package_module("_native"), _package_module("moe_glue")
    build_dir, _, _ = default_build
    assert _native.kernel_library_guids(build_dir / sim_build.LIBRARY) == tuple(HC_GUIDS + MOE_GUIDS + IL_GUIDS)
    assert [moe_glue.GUIDS[name] for name in moe_glue.OP_NAMES] == MOE_GUIDS
    assert _native.MOE_GLUE_OPS == moe_glue.OP_NAMES
    broken = tmp_path / sim_build.LIBRARY
    broken.write_text("not a shared object\n", encoding="utf-8")
    with pytest.raises(OSError):
        _native.kernel_library_guids(broken)


def test_the_python_loader_names_a_kernel_the_library_was_built_without(tmp_path: Path) -> None:
    _native, moe_glue = _package_module("_native"), _package_module("moe_glue")
    build_dir = tmp_path / "build"
    _cmake_build(build_dir, "-DHC_FUSED_MOE_GLUE=OFF")
    served = _native.kernel_library_guids(build_dir / sim_build.LIBRARY)
    assert served == tuple(HC_GUIDS + IL_GUIDS)
    assert [name for name in moe_glue.OP_NAMES if moe_glue.GUIDS[name] not in served] == list(moe_glue.OP_NAMES)


def test_a_moe_kernel_that_does_not_compile_costs_only_its_own_guid(tmp_path: Path, guid_host: Path) -> None:
    """The routing kernel fails to compile: the library keeps the six hyper-connection kernels and the other three."""
    import shutil

    real = shutil.which("tpc-clang")
    wrapper = tmp_path / "tpc-clang-wrapper"
    wrapper.write_text("#!/bin/sh\n"
                       'case "$*" in *moe_glue_route_gaudi2.c*) echo "moe_glue_route_gaudi2.c:1:1: error: '
                       'vector local memory exceeded (test)" >&2; exit 1;; esac\n'
                       f'exec "{real}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    build_dir = tmp_path / "build"
    configure, built = _cmake_build(build_dir, f"-DTPC_COMPILER={wrapper}")
    warning = " ".join(configure.split())  # CMake wraps the lines of a warning
    assert "moe_glue_route_gaudi2.c does not compile with the default TPC flags" in warning
    assert "vector local memory exceeded (test)" in warning
    kept = [kernel for kernel in sim_build.MOE_KERNELS if kernel != "moe_glue_route_gaudi2"]
    assert "MoE glue kernels in the library: [" + ";".join(kept) + "]" in configure
    assert sorted(_kernel_compiles(built)) == sorted(sim_build.KERNELS + tuple(kept) + sim_build.IL_KERNELS)
    guids, codes = _served(guid_host, build_dir)
    assert guids == HC_GUIDS + MOE_GUIDS[1:] + IL_GUIDS
    assert codes[moe_case.GUIDS["route"]] == GLUE_NODE_NOT_FOUND
    assert all(codes[guid] != GLUE_NODE_NOT_FOUND for guid in HC_GUIDS + MOE_GUIDS[1:]), codes


def test_the_switch_leaves_every_moe_kernel_out(tmp_path: Path, guid_host: Path) -> None:
    build_dir = tmp_path / "build"
    configure, built = _cmake_build(build_dir, "-DHC_FUSED_MOE_GLUE=OFF")
    assert "MoE glue kernels in the library: []" in configure
    assert sorted(_kernel_compiles(built)) == sorted(sim_build.KERNELS + sim_build.IL_KERNELS)
    guids, codes = _served(guid_host, build_dir)
    assert guids == HC_GUIDS + IL_GUIDS
    assert all(codes[guid] == GLUE_NODE_NOT_FOUND for guid in MOE_GUIDS), codes
    assert all(codes[guid] != GLUE_NODE_NOT_FOUND for guid in HC_GUIDS + IL_GUIDS), codes


def test_the_interleaved_switch_and_a_failing_interleaved_kernel_cost_only_their_own_guids(tmp_path: Path,
                                                                                         guid_host: Path) -> None:
    """-DHC_FUSED_IL=OFF leaves both interleaved kernels out; an interleaved kernel that does not compile costs only
    its own GUID (docs/design/flash-next/hc-kernels.md). The shipped kernels and the MoE glue kernels stay."""
    configure, built = _cmake_build(tmp_path / "off", "-DHC_FUSED_IL=OFF")
    assert "interleaved kernels in the library: []" in configure
    assert sorted(_kernel_compiles(built)) == sorted(sim_build.KERNELS + sim_build.MOE_KERNELS)
    guids, codes = _served(guid_host, tmp_path / "off")
    assert guids == HC_GUIDS + MOE_GUIDS
    assert all(codes[guid] == GLUE_NODE_NOT_FOUND for guid in IL_GUIDS), codes

    import shutil
    real = shutil.which("tpc-clang")
    wrapper = tmp_path / "tpc-clang-wrapper"
    wrapper.write_text("#!/bin/sh\n"
                       'case "$*" in *hc_fused_gate_mix_il_gaudi2.c*) echo "hc_fused_gate_mix_il_gaudi2.c:1:1: error: '
                       'too many vector spills (test)" >&2; exit 1;; esac\n'
                       f'exec "{real}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    configure, built = _cmake_build(tmp_path / "one", f"-DTPC_COMPILER={wrapper}")
    warning = " ".join(configure.split())
    assert "hc_fused_gate_mix_il_gaudi2.c does not compile with the default TPC flags" in warning
    assert "interleaved kernels in the library: [hc_fused_combine_norm_il_gaudi2]" in configure
    guids, codes = _served(guid_host, tmp_path / "one")
    assert guids == HC_GUIDS + MOE_GUIDS + IL_GUIDS[1:]
    assert codes[IL_GUIDS[0]] == GLUE_NODE_NOT_FOUND
    assert all(codes[guid] != GLUE_NODE_NOT_FOUND for guid in HC_GUIDS + MOE_GUIDS + IL_GUIDS[1:]), codes
