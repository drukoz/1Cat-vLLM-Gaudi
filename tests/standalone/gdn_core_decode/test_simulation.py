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
case (T = 1 to 6) is compared by ``verify_sim_case.compare_verify``.  One node more than the op
takes (T = 7) is rejected by the glue, and the kernel itself, reached through a glue built to let
7 nodes pass, writes zeros and touches no state.

All three kernels are compiled with the same flags, ``TPC_FLAGS``, and no local-memory override:
a kernel that calls lookup-table functions has 16 KiB of vector local memory, tpc-clang enforces
that only without ``-vlm``, and the simulator does not (RUK-3105: the verify kernel built with
``-vlm 32`` passed here and segfaulted on a card).  ``test_vector_local_memory`` reads the
compiler's own count (``-reg-mem-count``) for each kernel.

Skips on machines without the TPC toolchain (this Mac); runs for real in the tpcsim container
(``tools/flash_next/tpcsim.sh``, guide ``docs/design/flash-next/tpcsim.md``) and in the rental's
stage 1 container.  Contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tests.unit_tests.gdn_core_decode.recurrent_sim_case import (RECURRENT_CASES, compare_recurrent,
                                                                  make_recurrent_inputs, read_recurrent_outputs,
                                                                  write_recurrent_inputs)
from tests.unit_tests.gdn_core_decode.sim_case import CASES, compare, make_inputs, read_outputs, write_inputs
from tests.unit_tests.gdn_core_decode.verify_sim_case import (MAX_NODES, QUERY_LEN, VERIFY_CASES,
                                                              VERIFY_TOO_MANY_NODES_CASE, compare_verify,
                                                              make_verify_inputs, read_verify_outputs,
                                                              write_verify_inputs)

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

# The flags of every kernel, as GDN_TPC_FLAGS in csrc/gdn_fused/CMakeLists.txt.  No -vlm, -slm or -spill-vlm.
TPC_FLAGS = ("-Wall", "-Werror", "-O2", "-march=gaudi2")
# Vector local memory of a kernel that calls lookup-table functions, in bytes (tpc-clang's own limit for
# __local__ vectors plus vector spills), and the most each kernel may use so that 3 KiB (12 vectors) stay free.
# The verify kernel at 6 nodes uses 12544 bytes: 36 __local__ vectors and 13 spilled ones (tpc-clang 2.7.33).
VLM_LIMIT = 16384
VLM_BUDGET = VLM_LIMIT - 3072


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
    _run(["tpc-clang", *TPC_FLAGS, f"-I{TOOLS_INCLUDE}", KERNEL_SOURCE, "-c", "-o", build / KERNEL_OBJECT],
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
        "tpc-clang", *TPC_FLAGS, f"-I{TOOLS_INCLUDE}", RECURRENT_KERNEL_SOURCE, "-c", "-o",
        build / RECURRENT_KERNEL_OBJECT
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

    The kernel is compiled with ``TPC_FLAGS``, the flags of the decode kernel, exactly as the
    ``gdn_core_verify_gaudi2.o`` rule in ``csrc/gdn_fused/CMakeLists.txt`` does: with the default vector
    local memory limit this compile fails when the kernel's ``__local__`` arrays plus spills pass 16 KiB.
    With ``GDN_VERIFY_SIM_STRIP_BITCODE=1`` in the environment the compiled object is stripped of its
    ``.llvmbc`` section before embedding, mirroring ``cmake -DGDN_FUSED_VERIFY_STRIP_BITCODE=ON``
    (a card diagnostic, never the shipped build).
    """
    build = tmp_path_factory.mktemp("gdn_core_verify_sim")
    _run([
        "tpc-clang", *TPC_FLAGS, f"-I{TOOLS_INCLUDE}", VERIFY_KERNEL_SOURCE, "-c", "-o",
        build / VERIFY_KERNEL_OBJECT
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
    _link_verify_host(build, build / "verify_sim_host")
    return build


def _link_verify_host(build: Path, output: Path, first_include: Path | None = None) -> None:
    """Link ``verify_sim_host.cpp`` and the real host glue with the kernel ELF embedded in ``build``.

    ``first_include`` is searched before ``csrc/gdn_fused``: a folder with another ``gdn_core_decode_params.h``
    gives the glue another ``kGdnCoreVerifyMaxNodes`` without touching the sources.
    """
    includes = [f"-I{first_include}"] if first_include is not None else []
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", VERIFY_HOST_SOURCE, VERIFY_GLUE_SOURCE,
        build / "gdn_core_verify_gaudi2_embedded.o", *includes, f"-I{REPO_ROOT / 'csrc' / 'gdn_fused'}",
        f"-I{REPO_ROOT / 'csrc' / 'gdn_fused' / 'host'}", f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}",
        f"-I{DRIVER_INCLUDE}", "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext",
        "-o", output
    ],
         timeout=1800)


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


def test_no_kernel_is_compiled_with_a_local_memory_override() -> None:
    """The flags of this file's fixtures: one list for the three kernels, without -vlm, -slm or -spill-vlm."""
    assert not [flag for flag in TPC_FLAGS if flag.lstrip("-") in ("vlm", "slm", "spill-vlm")]
    source = Path(__file__).read_text()
    compiles = re.findall('"tpc-' + r'clang",([^\]]*)\]', source)  # every tpc-clang argument list of this file
    assert len(compiles) == 4, compiles  # the three fixtures and test_vector_local_memory
    for arguments in compiles:
        assert "*TPC_FLAGS" in arguments, arguments
        assert not re.search(r'"-(vlm|slm|spill-vlm)"', arguments), arguments


@pytest.mark.parametrize("source", [KERNEL_SOURCE, RECURRENT_KERNEL_SOURCE, VERIFY_KERNEL_SOURCE],
                         ids=["decode", "recurrent", "verify"])
def test_vector_local_memory(source: Path, tmp_path: Path) -> None:
    """tpc-clang's own count of vector local memory (statics plus spills), compiled with the default flags.

    The compile itself fails past ``VLM_LIMIT`` ("too much vector memory is used for statically allocated
    data", "Too many vector spills"); the assertion keeps 3 KiB of that free.
    """
    proc = _run(
        ["tpc-clang", *TPC_FLAGS, "-reg-mem-count", f"-I{TOOLS_INCLUDE}", source, "-c", "-o", tmp_path / "kernel.o"],
        timeout=1800)
    match = re.search(r"Total VLM used: (\d+) bytes", proc.stdout + proc.stderr)
    assert match, f"no 'Total VLM used' line in the -reg-mem-count output:\n{proc.stdout}\n{proc.stderr}"
    used = int(match.group(1))
    print(f"{source.name}: Total VLM used: {used} bytes of {VLM_LIMIT}")
    assert used <= VLM_BUDGET, f"{source.name} uses {used} bytes of vector local memory, budget {VLM_BUDGET}"


def test_verify_object_keeps_its_bitcode(verify_sim_binary: Path) -> None:
    """The verify kernel ELF is embedded as tpc-clang wrote it, with its .llvmbc section (never stripped)."""
    if os.environ.get("GDN_VERIFY_SIM_STRIP_BITCODE") == "1":
        pytest.skip("GDN_VERIFY_SIM_STRIP_BITCODE=1 strips the section on purpose")
    sections = _run(["readelf", "-S", "-W", verify_sim_binary / VERIFY_KERNEL_OBJECT]).stdout
    assert ".llvmbc" in sections


def test_verify_glue_rejects_too_many_nodes(verify_sim_binary: Path, tmp_path: Path) -> None:
    """T = MAX_NODES + 1: the host glue answers GLUE_INCOMPATIBLE_INPUT_SIZE and the kernel never runs."""
    case = VERIFY_TOO_MANY_NODES_CASE
    assert case.T == MAX_NODES + 1
    case_dir = tmp_path / case.name
    write_verify_inputs(case, make_verify_inputs(case), case_dir)
    proc = subprocess.run([str(verify_sim_binary / "verify_sim_host"), str(case_dir)],
                          capture_output=True,
                          text=True,
                          timeout=900)
    assert proc.returncode == 6, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "GLUE_INCOMPATIBLE_INPUT_SIZE" in proc.stderr
    assert "instructions executed" not in proc.stdout
    assert not (case_dir / "out.bin").exists()


def test_verify_kernel_pads_a_node_count_above_its_arrays(verify_sim_binary: Path, tmp_path: Path) -> None:
    """The kernel's own guard, behind the glue's: T = MAX_NODES + 1 through a glue built to accept it.

    The glue is compiled against a copy of ``gdn_core_decode_params.h`` that says one node more, so the same
    kernel ELF is simulated with 7 nodes.  Its per-node arrays hold 6: it must treat the row as padding, write
    zeros for every node and leave both state pools as they were.
    """
    case = VERIFY_TOO_MANY_NODES_CASE
    assert case.T == MAX_NODES + 1
    params = (REPO_ROOT / "csrc" / "gdn_fused" / "gdn_core_decode_params.h").read_text()
    stated = f"kGdnCoreVerifyMaxNodes = {MAX_NODES};"
    assert params.count(stated) == 1
    patched = tmp_path / "patched"
    patched.mkdir()
    (patched / "gdn_core_decode_params.h").write_text(
        params.replace(stated, f"kGdnCoreVerifyMaxNodes = {MAX_NODES + 1};"))
    host = tmp_path / "verify_sim_host_wide"
    _link_verify_host(verify_sim_binary, host, first_include=patched)

    inputs = make_verify_inputs(case)
    assert int(inputs["row_indices"][0, QUERY_LEN]) == case.T  # a real row: only the node count makes it padding
    case_dir = tmp_path / case.name
    write_verify_inputs(case, inputs, case_dir)
    proc = _run([host, case_dir], timeout=900)
    assert "instructions executed" in proc.stdout
    got = read_verify_outputs(case, case_dir)
    assert tuple(got["out"].shape) == (case.B * case.T, case.HV, 128)
    assert not got["out"].any()
    assert (got["conv_state"].view(torch.int16) == inputs["conv_state"].view(torch.int16)).all()
    assert (got["ssm_state"].view(torch.int32) == inputs["ssm_state"].view(torch.int32)).all()


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
