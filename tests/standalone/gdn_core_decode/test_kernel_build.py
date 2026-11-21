# SPDX-License-Identifier: Apache-2.0
"""Build ``csrc/gdn_fused`` with CMake, as ``tools/build_flashinfer_gaudi.py`` does, and check what comes out.

RUK-3105: the verify kernel was compiled with ``-vlm 32``, which lifts tpc-clang's 16 KiB limit on the vector
local memory of a kernel that calls lookup-table functions.  The simulator passed, the op segfaulted on a card.
These tests keep that from coming back and keep a broken verify kernel from taking the working kernels with it:

- the real build compiles every kernel with one flag list and no local-memory override (``-vlm``, ``-slm``,
  ``-spill-vlm``), the library serves the three GUIDs and the verify ELF keeps its ``.llvmbc`` section;
- when the verify kernel does not compile (a tpc-clang wrapper fails on it), or with ``-DGDN_FUSED_VERIFY=OFF``,
  the library still builds and serves the decode and recurrent GUIDs, without the verify GUID.

Skips on machines without the TPC toolchain; runs in the tpcsim container (``docs/design/flash-next/tpcsim.md``).
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "csrc" / "gdn_fused"
TOOLS_INCLUDE = Path("/usr/lib/habanatools/include")
GUID_HOST_SOURCE = Path(__file__).resolve().parent / "guid_list_host.cpp"
DECODE_GUID = "gdn_fused_core_decode_f32state_gaudi2"
RECURRENT_GUID = "gdn_fused_recurrent_decode_f32state_gaudi2"
VERIFY_GUID = "gdn_fused_core_verify_f32state_gaudi2"
KERNELS = ("gdn_core_decode_gaudi2", "gdn_recurrent_decode_gaudi2", "gdn_core_verify_gaudi2",
           "gdn_core_decode_interleaved_gaudi2", "gdn_core_verify_interleaved_gaudi2")
INTERLEAVED = ("gdn_core_decode_interleaved_gaudi2", "gdn_core_verify_interleaved_gaudi2")
TPC_FLAGS = ["-Wall", "-Werror", "-O2", "-march=gaudi2"]  # the decode kernel's flags
GLUE_NODE_NOT_FOUND = 1


def _missing() -> list[str]:
    missing = [f"{tool} is not on PATH" for tool in ("tpc-clang", "cmake", "objcopy", "g++") if not shutil.which(tool)]
    if not (TOOLS_INCLUDE / "gc_interface.h").is_file():
        missing.append(f"{TOOLS_INCLUDE / 'gc_interface.h'} is missing")
    return missing


_MISSING = _missing()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC toolchain not installed: " + "; ".join(_MISSING))


def _run(cmd: list, *, timeout: float = 1800) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(part) for part in cmd], capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (f"command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}\n"
                                  f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}")
    return proc


def _cmake_build(build: Path, *options: str) -> tuple[str, str]:
    """Configure and build the library; returns (configure output, verbose build output)."""
    configure = _run([
        "cmake", "-S", SOURCE_DIR, "-B", build, "-DCMAKE_BUILD_TYPE=Release", f"-DHABANA_INCLUDE_DIR={TOOLS_INCLUDE}",
        *options
    ])
    built = _run(["cmake", "--build", build, "--target", "gdn_fused_gaudi2", "--verbose"])
    assert (build / "libgdn_fused_gaudi2.so").is_file()
    return configure.stdout + configure.stderr, built.stdout + built.stderr


def _kernel_compiles(build_output: str) -> dict[str, list[str]]:
    """kernel name -> the arguments of its tpc-clang command in the verbose build output."""
    compiles = {}
    for line in build_output.splitlines():
        words = line.split()
        if not words or Path(words[0]).name != "tpc-clang":
            continue
        for kernel in KERNELS:
            if any(word.endswith(f"/kernels/{kernel}.c") for word in words):
                compiles[kernel] = words[1:]
    return compiles


@pytest.fixture(scope="module")
def guid_host(tmp_path_factory: pytest.TempPathFactory) -> Path:
    exe = tmp_path_factory.mktemp("guid_list_host") / "guid_list_host"
    _run(["g++", "-std=c++17", "-O1", "-Wall", GUID_HOST_SOURCE, f"-I{TOOLS_INCLUDE}", "-ldl", "-o", exe])
    return exe


def _served(guid_host: Path, build: Path) -> tuple[list[str], dict[str, int]]:
    """(the GUIDs the built library lists, GUID -> InstantiateTpcKernel return code with no tensors)."""
    out = _run([guid_host, build / "libgdn_fused_gaudi2.so", DECODE_GUID, RECURRENT_GUID, VERIFY_GUID]).stdout
    guids = [line.split()[1] for line in out.splitlines() if line.startswith("guid ")]
    codes = {line.split()[1]: int(line.split()[2]) for line in out.splitlines() if line.startswith("instantiate ")}
    return guids, codes


@pytest.fixture(scope="module")
def default_build(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    build = tmp_path_factory.mktemp("gdn_fused_default")
    configure, built = _cmake_build(build)
    return build, configure, built


def test_build_compiles_every_kernel_with_the_default_flags(default_build) -> None:
    """No -vlm, -slm or -spill-vlm on any kernel, and the verify kernel's flags are the decode kernel's."""
    _, _, built = default_build
    compiles = _kernel_compiles(built)
    assert sorted(compiles) == sorted(KERNELS), f"tpc-clang commands found for {sorted(compiles)}:\n{built[-3000:]}"
    flags = {}
    for kernel, arguments in compiles.items():
        assert not [a for a in arguments if re.fullmatch(r"-+(vlm|slm|spill-vlm)(=.*)?", a)], (kernel, arguments)
        flags[kernel] = [a for a in arguments if not a.startswith("-I") and not a.endswith((".c", ".o")) and
                         a not in ("-c", "-o")]
    assert flags["gdn_core_decode_gaudi2"] == TPC_FLAGS
    assert flags["gdn_core_verify_gaudi2"] == flags["gdn_core_decode_gaudi2"]
    assert flags["gdn_recurrent_decode_gaudi2"] == flags["gdn_core_decode_gaudi2"]
    for kernel in INTERLEAVED:
        assert flags[kernel] == flags["gdn_core_decode_gaudi2"], kernel


def _interleaved_symbols(build: Path) -> list[str]:
    """The interleaved kernels whose embedded ELF the built library defines."""
    symbols = _run(["nm", build / "libgdn_fused_gaudi2.so"]).stdout
    return [kernel for kernel in INTERLEAVED
            if re.search(rf"\s[A-Za-z]\s_binary_{kernel}_o_start$", symbols, re.MULTILINE)
            and not re.search(rf"\sw\s_binary_{kernel}_o_start$", symbols, re.MULTILINE)]


def test_build_carries_the_interleaved_kernels(default_build) -> None:
    """VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 needs both interleaved ELFs in the library, defined, not weak."""
    build, configure, _ = default_build
    assert ("interleaved kernels in the library: gdn_core_decode_interleaved_gaudi2;"
            "gdn_core_verify_interleaved_gaudi2") in configure
    assert _interleaved_symbols(build) == list(INTERLEAVED)


def test_interleave_switched_off_keeps_the_three_guids(tmp_path: Path, guid_host: Path) -> None:
    build = tmp_path / "build"
    configure, built = _cmake_build(build, "-DGDN_FUSED_INTERLEAVE=OFF")
    assert re.search(r"interleaved kernels in the library: *$", configure, re.MULTILINE), configure[-2000:]
    assert not [kernel for kernel in INTERLEAVED if kernel in _kernel_compiles(built)]
    assert _interleaved_symbols(build) == []
    guids, codes = _served(guid_host, build)
    assert guids == [DECODE_GUID, RECURRENT_GUID, VERIFY_GUID]
    assert all(code != GLUE_NODE_NOT_FOUND for code in codes.values()), codes


def test_build_serves_the_three_guids(default_build, guid_host: Path) -> None:
    build, configure, _ = default_build
    assert "verify kernel in the library: ON" in configure
    guids, codes = _served(guid_host, build)
    assert guids == [DECODE_GUID, RECURRENT_GUID, VERIFY_GUID]
    assert all(code != GLUE_NODE_NOT_FOUND for code in codes.values()), codes


def test_build_does_not_strip_the_verify_bitcode(default_build) -> None:
    build, _, _ = default_build
    sections = _run(["readelf", "-S", "-W", build / "gdn_core_verify_gaudi2.o"]).stdout
    assert ".llvmbc" in sections


def _assert_decode_and_recurrent_only(guid_host: Path, build: Path, built: str) -> None:
    # the interleaved decode kernel stays; the interleaved verify kernel is built only with the verify kernel
    assert sorted(_kernel_compiles(built)) == ["gdn_core_decode_gaudi2", "gdn_core_decode_interleaved_gaudi2",
                                               "gdn_recurrent_decode_gaudi2"]
    guids, codes = _served(guid_host, build)
    assert guids == [DECODE_GUID, RECURRENT_GUID]
    assert codes[VERIFY_GUID] == GLUE_NODE_NOT_FOUND
    assert codes[DECODE_GUID] != GLUE_NODE_NOT_FOUND and codes[RECURRENT_GUID] != GLUE_NODE_NOT_FOUND


def test_verify_compile_failure_keeps_decode_and_recurrent(tmp_path: Path, guid_host: Path) -> None:
    """A verify kernel that does not compile costs only the verify GUID, not the library."""
    wrapper = tmp_path / "tpc-clang"  # the real compiler for every source but the verify kernel
    wrapper.write_text("#!/bin/sh\n"
                       'case "$*" in *gdn_core_verify_gaudi2.c*|*gdn_core_verify_interleaved_gaudi2.c*)\n'
                       '    echo "fatal error: error in backend: too much vector memory is used for statically '
                       'allocated data: 24576 is allocated, but only 16384 is available" >&2\n'
                       "    exit 1;;\n"
                       "esac\n"
                       f'exec "{shutil.which("tpc-clang")}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    build = tmp_path / "build"
    configure, built = _cmake_build(build, f"-DTPC_COMPILER={wrapper}")
    assert "does not compile with the default TPC flags" in configure
    assert "too much vector memory" in configure
    assert "verify kernel in the library: OFF" in configure
    _assert_decode_and_recurrent_only(guid_host, build, built)


def test_verify_switched_off_keeps_decode_and_recurrent(tmp_path: Path, guid_host: Path) -> None:
    build = tmp_path / "build"
    configure, built = _cmake_build(build, "-DGDN_FUSED_VERIFY=OFF")
    assert "verify kernel in the library: OFF" in configure
    _assert_decode_and_recurrent_only(guid_host, build, built)
