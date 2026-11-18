# SPDX-License-Identifier: Apache-2.0
"""Run the fused GDN core-decode kernel and the recurrence-only kernel on the Gaudi2 TPC simulator.

The module fixture compiles ``csrc/gdn_fused/kernels/gdn_core_decode_gaudi2.c`` with
``tpc-clang`` and embeds its ELF with ``objcopy`` exactly as ``csrc/gdn_fused/CMakeLists.txt``
does, then links ``sim_host.cpp`` with the real host glue
(``csrc/gdn_fused/host/gdn_core_decode_gaudi2.cpp``) and the simulator library.  Each
``sim_case.CASES`` case is written to a raw-file folder, run through the simulator and compared
with the torch reference by ``sim_case.compare``.

The second fixture does the same for the recurrence-only kernel
``csrc/gdn_fused/kernels/gdn_recurrent_decode_gaudi2.c`` (host program
``recurrent_sim_host.cpp``, glue ``csrc/gdn_fused/host/gdn_recurrent_decode_gaudi2.cpp``), and
each ``recurrent_sim_case.RECURRENT_CASES`` case is compared by
``recurrent_sim_case.compare_recurrent``.

The third fixture does the same for the fused GDN verification kernel (phase 2, T = 1 + k)
``csrc/gdn_fused/kernels/gdn_core_verify_gaudi2.c`` (host program ``verify_sim_host.cpp``, glue
``csrc/gdn_fused/host/gdn_core_verify_gaudi2.cpp``), and each ``verify_sim_case.VERIFY_CASES``
case is compared by ``verify_sim_case.compare_verify``.

Skips on machines without the TPC toolchain (this Mac); runs for real in the tpcsim container
(``tools/flash_next/tpcsim.sh``, guide ``docs/design/flash-next/tpcsim.md``) and in the rental's
stage 1 container.  Contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit_tests.gdn_core_decode.recurrent_sim_case import (RECURRENT_CASES, compare_recurrent,
                                                                  make_recurrent_inputs, read_recurrent_outputs,
                                                                  write_recurrent_inputs)
from tests.unit_tests.gdn_core_decode.sim_case import CASES, compare, make_inputs, read_outputs, write_inputs
from tests.unit_tests.gdn_core_decode.verify_sim_case import (VERIFY_CASES, compare_verify, make_verify_inputs,
                                                              read_verify_outputs, write_verify_inputs)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_INCLUDE = Path("/usr/lib/habanatools/include")
DRIVER_INCLUDE = Path("/usr/include/habanalabs")
KERNEL_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_decode_gaudi2.c"
GLUE_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "host" / "gdn_core_decode_gaudi2.cpp"
HOST_SOURCE = Path(__file__).resolve().parent / "sim_host.cpp"
KERNEL_OBJECT = "gdn_core_decode_gaudi2.o"
RECURRENT_KERNEL_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_recurrent_decode_gaudi2.c"
RECURRENT_GLUE_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "host" / "gdn_recurrent_decode_gaudi2.cpp"
RECURRENT_HOST_SOURCE = Path(__file__).resolve().parent / "recurrent_sim_host.cpp"
RECURRENT_KERNEL_OBJECT = "gdn_recurrent_decode_gaudi2.o"
VERIFY_KERNEL_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_verify_gaudi2.c"
VERIFY_GLUE_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "host" / "gdn_core_verify_gaudi2.cpp"
VERIFY_HOST_SOURCE = Path(__file__).resolve().parent / "verify_sim_host.cpp"
VERIFY_KERNEL_OBJECT = "gdn_core_verify_gaudi2.o"


def _missing_sdk_parts() -> list[str]:
    missing = []
    if not shutil.which("tpc-clang"):
        missing.append("tpc-clang is not on PATH")
    if not (TOOLS_INCLUDE / "tpc_test_core_api.h").is_file():
        missing.append(f"{TOOLS_INCLUDE / 'tpc_test_core_api.h'} is missing")
    if not (TOOLS_INCLUDE / "gc_interface.h").is_file() and not (DRIVER_INCLUDE / "gc_interface.h").is_file():
        missing.append(f"gc_interface.h is missing from {TOOLS_INCLUDE} and {DRIVER_INCLUDE}")
    return missing


_MISSING = _missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))


def _run(cmd: list, *, cwd: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([str(part) for part in cmd], check=True, capture_output=True, text=True, cwd=cwd,
                              timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed (exit {exc.returncode}): {' '.join(map(str, cmd))}\n"
                           f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}") from exc


@pytest.fixture(scope="module")
def sim_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build folder with the compiled kernel .o and the linked sim_host binary (built once)."""
    build = tmp_path_factory.mktemp("gdn_core_decode_sim")
    _run([
        "tpc-clang", "-Wall", "-Werror", "-O2", "-march=gaudi2", f"-I{TOOLS_INCLUDE}", KERNEL_SOURCE, "-c", "-o",
        build / KERNEL_OBJECT
    ],
         timeout=1800)
    _run([
        "objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64", KERNEL_OBJECT,
        "gdn_core_decode_gaudi2_embedded.o"
    ],
         cwd=build,
         timeout=600)
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", HOST_SOURCE, GLUE_SOURCE,
        build / "gdn_core_decode_gaudi2_embedded.o", f"-I{REPO_ROOT / 'csrc' / 'gdn_fused'}",
        f"-I{REPO_ROOT / 'csrc' / 'gdn_fused' / 'host'}", f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}",
        f"-I{DRIVER_INCLUDE}", "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext",
        "-o", build / "sim_host"
    ],
         timeout=1800)
    return build


@pytest.fixture(scope="module")
def recurrent_sim_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build folder with the compiled recurrence kernel .o and the linked recurrent_sim_host binary (built once)."""
    build = tmp_path_factory.mktemp("gdn_recurrent_decode_sim")
    _run([
        "tpc-clang", "-Wall", "-Werror", "-O2", "-march=gaudi2", f"-I{TOOLS_INCLUDE}", RECURRENT_KERNEL_SOURCE, "-c",
        "-o", build / RECURRENT_KERNEL_OBJECT
    ],
         timeout=1800)
    _run([
        "objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64", RECURRENT_KERNEL_OBJECT,
        "gdn_recurrent_decode_gaudi2_embedded.o"
    ],
         cwd=build,
         timeout=600)
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", RECURRENT_HOST_SOURCE, RECURRENT_GLUE_SOURCE,
        build / "gdn_recurrent_decode_gaudi2_embedded.o", f"-I{REPO_ROOT / 'csrc' / 'gdn_fused'}",
        f"-I{REPO_ROOT / 'csrc' / 'gdn_fused' / 'host'}", f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}",
        f"-I{DRIVER_INCLUDE}", "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext",
        "-o", build / "recurrent_sim_host"
    ],
         timeout=1800)
    return build


@pytest.fixture(scope="module")
def verify_sim_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build folder with the compiled verification kernel .o and the linked verify_sim_host binary (built once).

    The kernel is compiled with ``-vlm 32`` exactly as the ``gdn_core_verify_gaudi2.o`` rule in
    ``csrc/gdn_fused/CMakeLists.txt`` does (its per-node ``__local__`` arrays exceed the default VLM).
    With ``GDN_VERIFY_SIM_STRIP_BITCODE=1`` in the environment the compiled object is stripped of its
    ``.llvmbc`` section before embedding, mirroring ``cmake -DGDN_FUSED_VERIFY_STRIP_BITCODE=ON``
    (RUK-3105 H2 diagnostic).
    """
    build = tmp_path_factory.mktemp("gdn_core_verify_sim")
    _run([
        "tpc-clang", "-Wall", "-Werror", "-O2", "-march=gaudi2", "-vlm", "32", f"-I{TOOLS_INCLUDE}",
        VERIFY_KERNEL_SOURCE, "-c", "-o", build / VERIFY_KERNEL_OBJECT
    ],
         timeout=1800)
    if os.environ.get("GDN_VERIFY_SIM_STRIP_BITCODE") == "1":
        _run(["objcopy", "--remove-section=.llvmbc", VERIFY_KERNEL_OBJECT], cwd=build, timeout=600)
    _run([
        "objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64", VERIFY_KERNEL_OBJECT,
        "gdn_core_verify_gaudi2_embedded.o"
    ],
         cwd=build,
         timeout=600)
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", VERIFY_HOST_SOURCE, VERIFY_GLUE_SOURCE,
        build / "gdn_core_verify_gaudi2_embedded.o", f"-I{REPO_ROOT / 'csrc' / 'gdn_fused'}",
        f"-I{REPO_ROOT / 'csrc' / 'gdn_fused' / 'host'}", f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}",
        f"-I{DRIVER_INCLUDE}", "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext",
        "-o", build / "verify_sim_host"
    ],
         timeout=1800)
    return build


def test_kernel_disassembles_for_gaudi2(sim_binary: Path) -> None:
    listing = _run(["tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d", sim_binary / KERNEL_OBJECT]).stdout
    assert "TPC architecture: gaudi2" in listing


def test_recurrent_kernel_disassembles_for_gaudi2(recurrent_sim_binary: Path) -> None:
    listing = _run([
        "tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d", recurrent_sim_binary / RECURRENT_KERNEL_OBJECT
    ]).stdout
    assert "TPC architecture: gaudi2" in listing


def test_verify_kernel_disassembles_for_gaudi2(verify_sim_binary: Path) -> None:
    listing = _run(
        ["tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d", verify_sim_binary / VERIFY_KERNEL_OBJECT]).stdout
    assert "TPC architecture: gaudi2" in listing


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_simulation_matches_reference(sim_binary: Path, case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    inputs = make_inputs(case)
    write_inputs(case, inputs, case_dir)
    proc = _run([sim_binary / "sim_host", case_dir], timeout=900)
    print(proc.stdout)
    got = read_outputs(case, case_dir)
    assert compare(case, inputs, got) == []


@pytest.mark.parametrize("case", RECURRENT_CASES, ids=[case.name for case in RECURRENT_CASES])
def test_recurrent_simulation_matches_reference(recurrent_sim_binary: Path, case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    inputs = make_recurrent_inputs(case)
    write_recurrent_inputs(case, inputs, case_dir)
    proc = _run([recurrent_sim_binary / "recurrent_sim_host", case_dir], timeout=900)
    print(proc.stdout)
    got = read_recurrent_outputs(case, case_dir)
    assert compare_recurrent(case, inputs, got) == []


@pytest.mark.parametrize("case", VERIFY_CASES, ids=[case.name for case in VERIFY_CASES])
def test_verify_simulation_matches_reference(verify_sim_binary: Path, case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    inputs = make_verify_inputs(case)
    write_verify_inputs(case, inputs, case_dir)
    proc = _run([verify_sim_binary / "verify_sim_host", case_dir], timeout=1800)
    print(proc.stdout)
    got = read_verify_outputs(case, case_dir)
    assert compare_verify(case, inputs, got) == []
