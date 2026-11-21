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
# The interleaved builds (VLLM_HPU_GDN_KERNEL_INTERLEAVE=1, idea I27): the same kernels with four value rows per
# iteration of the recurrence. A host linked with both ELFs serves either, chosen by the glue from the variable.
INTERLEAVED_DECODE_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_decode_interleaved_gaudi2.c"
INTERLEAVED_VERIFY_SOURCE = REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_verify_interleaved_gaudi2.c"
INTERLEAVE_ENV = "VLLM_HPU_GDN_KERNEL_INTERLEAVE"
# The variant builds of the interleaved verify kernel (VLLM_HPU_GDN_VERIFY_VARIANT with the variable above;
# docs/design/flash-next/hc-kernels.md, "GDN verify"): 1 request-major index space {requests, key heads}, 2 lean
# broadcasts in the recurrence, 3 both, 6 the fold recurrence. Each must give the interleaved kernel's bytes.
VARIANT_VERIFY_SOURCES = {
    1: REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_verify_interleaved_rmajor_gaudi2.c",
    2: REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_verify_interleaved_lean_gaudi2.c",
    3: REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_verify_interleaved_rmajor_lean_gaudi2.c",
    6: REPO_ROOT / "csrc" / "gdn_fused" / "kernels" / "gdn_core_verify_interleaved_fold_gaudi2.c",
}
VARIANT_ENV = "VLLM_HPU_GDN_VERIFY_VARIANT"
# Variants 4 and 5 use the interleaved and the lean ELF with today's index space and the graph compiler's
# preferred split of the key-head dimension (glue only; the simulator runs the whole index space on one TPC); 7 the
# fold ELF with that split.
VARIANTS = (1, 2, 3, 4, 5, 6, 7)
REQUEST_MAJOR_VARIANTS = (1, 3)
LEAN_VARIANTS = (2, 3, 5)
# Instructions of the lean variants over the interleaved kernel, at most (measured 2026-11-21: 0.887 to 0.925).
LEAN_VERIFY_RATIO = 0.94
FOLD_VARIANTS = (6, 7)
# Instructions of the fold variants over the interleaved kernel, at most (measured 2026-11-21: 0.609 to 0.630).
FOLD_VERIFY_RATIO = 0.70
# Instructions of the interleaved kernel over today's, at most (measured 2026-11-20, tpc-clang 2.7.33: decode
# 0.344 on every case, verify 0.369 to 0.378).
INTERLEAVED_DECODE_RATIO = 0.40
INTERLEAVED_VERIFY_RATIO = 0.42

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


def _embed(build: Path, source: Path) -> Path:
    """Compile one kernel with TPC_FLAGS and embed its ELF as CMakeLists.txt does; the embedded object."""
    obj = build / (source.stem + ".o")
    _run(["tpc-clang", *TPC_FLAGS, f"-I{TOOLS_INCLUDE}", source, "-c", "-o", obj], timeout=1800)
    embedded = source.stem + "_embedded.o"
    _run(["objcopy", "-I", "binary", "-O", "elf64-x86-64", "-B", "i386:x86-64", obj.name, embedded], cwd=build,
         timeout=600)
    return build / embedded


@pytest.fixture(scope="module")
def interleaved_sim_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """sim_host linked with today's decode ELF and the interleaved one (the glue picks by INTERLEAVE_ENV)."""
    build = tmp_path_factory.mktemp("gdn_core_decode_interleaved_sim")
    objects = [_embed(build, KERNEL_SOURCE), _embed(build, INTERLEAVED_DECODE_SOURCE)]
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", HOST_SOURCE, GLUE_SOURCE, *objects,
        f"-I{REPO_ROOT / 'csrc' / 'gdn_fused'}", f"-I{REPO_ROOT / 'csrc' / 'gdn_fused' / 'host'}",
        f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}", f"-I{DRIVER_INCLUDE}", "-L/usr/lib/habanatools",
        "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext", "-o", build / "sim_host"
    ],
         timeout=1800)
    return build


@pytest.fixture(scope="module")
def interleaved_verify_sim_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """verify_sim_host linked with today's verify ELF, the interleaved one and its four variant builds."""
    build = tmp_path_factory.mktemp("gdn_core_verify_interleaved_sim")
    variants = [_embed(build, source) for source in VARIANT_VERIFY_SOURCES.values()]
    for source in (VERIFY_KERNEL_SOURCE, INTERLEAVED_VERIFY_SOURCE):
        _embed(build, source)
    extra = build / "gdn_core_verify_interleaved_gaudi2_embedded.o"
    _run([
        "g++", "-std=c++17", "-O2", "-Wall", VERIFY_HOST_SOURCE, VERIFY_GLUE_SOURCE,
        build / "gdn_core_verify_gaudi2_embedded.o", extra, *variants, f"-I{REPO_ROOT / 'csrc' / 'gdn_fused'}",
        f"-I{REPO_ROOT / 'csrc' / 'gdn_fused' / 'host'}", f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}",
        f"-I{DRIVER_INCLUDE}", "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext",
        "-o", build / "verify_sim_host"
    ],
         timeout=1800)
    return build


def _run_both(binary: Path, write, case_dir: Path, files) -> tuple[int, int]:
    """Run one case with today's kernel and with the interleaved one; (instructions today, interleaved).

    The outputs of both runs must be the same bytes, file by file."""
    counts, outputs = [], []
    for interleave in (None, "1"):
        run_dir = case_dir / ("interleaved" if interleave else "today")
        write(run_dir)
        env = {key: value for key, value in os.environ.items() if key != INTERLEAVE_ENV}
        if interleave:
            env[INTERLEAVE_ENV] = interleave
        proc = subprocess.run([str(binary), str(run_dir)], capture_output=True, text=True, timeout=3600, env=env)
        assert proc.returncode == 0, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        match = re.search(r"instructions executed: (\d+)", proc.stdout)
        assert match, proc.stdout
        counts.append(int(match.group(1)))
        outputs.append({name: (run_dir / name).read_bytes() for name in files})
    for name in files:
        assert outputs[0][name] == outputs[1][name], f"{name}: the interleaved kernel's bytes differ"
    return counts[0], counts[1]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_interleaved_decode_kernel_is_bit_identical_and_shorter(interleaved_sim_binary: Path, case,
                                                                 tmp_path: Path) -> None:
    from tests.unit_tests.gdn_core_decode.sim_case import OUTPUT_FILES

    inputs = make_inputs(case)
    today, interleaved = _run_both(interleaved_sim_binary / "sim_host",
                                   lambda d: write_inputs(case, inputs, d), tmp_path / case.name,
                                   list(OUTPUT_FILES.values()))
    print(f"{case.name}: instructions today {today}, interleaved {interleaved} ({interleaved / today:.3f})")
    assert compare(case, inputs, read_outputs(case, tmp_path / case.name / "interleaved")) == []
    assert interleaved <= INTERLEAVED_DECODE_RATIO * today


@pytest.mark.parametrize("case", VERIFY_CASES, ids=[case.name for case in VERIFY_CASES])
def test_interleaved_verify_kernel_is_bit_identical_and_shorter(interleaved_verify_sim_binary: Path, case,
                                                                 tmp_path: Path) -> None:
    inputs = make_verify_inputs(case)
    case_dir = tmp_path / case.name
    files = ["out.bin", "conv_state.out.bin", "ssm_state.out.bin"]
    today, interleaved = _run_both(interleaved_verify_sim_binary / "verify_sim_host",
                                   lambda d: write_verify_inputs(case, inputs, d), case_dir, files)
    print(f"{case.name}: instructions today {today}, interleaved {interleaved} ({interleaved / today:.3f})")
    assert compare_verify(case, inputs, read_verify_outputs(case, case_dir / "interleaved")) == []
    assert interleaved <= INTERLEAVED_VERIFY_RATIO * today


def _run_verify(binary: Path, case, inputs, run_dir: Path, env_values: dict) -> tuple[int, str, dict]:
    """(instructions, the index-space line, output bytes by file) of one verify run with the given variables."""
    write_verify_inputs(case, inputs, run_dir)
    env = {key: value for key, value in os.environ.items() if key not in (INTERLEAVE_ENV, VARIANT_ENV)}
    env.update(env_values)
    proc = subprocess.run([str(binary), str(run_dir)], capture_output=True, text=True, timeout=3600, env=env)
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    count = int(re.search(r"instructions executed: (\d+)", proc.stdout).group(1))
    geometry = re.search(r"index space: \{([^}]*)\}", proc.stdout).group(1)
    files = ("out.bin", "conv_state.out.bin", "ssm_state.out.bin")
    return count, geometry, {name: (run_dir / name).read_bytes() for name in files}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", VERIFY_CASES, ids=[case.name for case in VERIFY_CASES])
def test_verify_variant_is_bit_identical_to_the_interleaved_kernel(interleaved_verify_sim_binary: Path, case,
                                                                   variant: int, tmp_path: Path) -> None:
    """VLLM_HPU_GDN_VERIFY_VARIANT 1 (request-major index space), 2 (lean broadcasts), 3 (both), 4 (preferred split
    of the key heads), 5 (4 with the lean ELF), 6 (the fold recurrence), 7 (6 with the split of 4): every output and
    state byte is the interleaved kernel's; 1 and 4 run the same instructions, 2, 3 and 5 fewer, 6 and 7 fewer
    still."""
    binary = interleaved_verify_sim_binary / "verify_sim_host"
    inputs = make_verify_inputs(case)
    base = _run_verify(binary, case, inputs, tmp_path / "interleaved", {INTERLEAVE_ENV: "1"})
    got = _run_verify(binary, case, inputs, tmp_path / f"variant{variant}", {
        INTERLEAVE_ENV: "1",
        VARIANT_ENV: str(variant)
    })
    print(f"{case.name} variant {variant}: index space {base[1]} -> {got[1]}, instructions {base[0]} -> {got[0]} "
          f"({got[0] / base[0]:.3f})")
    first, second = (int(v) for v in base[1].split(","))
    want = [second, first] if variant in REQUEST_MAJOR_VARIANTS else [first, second]
    assert [int(v) for v in got[1].split(",")] == want
    for name in base[2]:
        assert base[2][name] == got[2][name], f"{name}: variant {variant}'s bytes differ"
    if variant in LEAN_VARIANTS:
        assert got[0] <= LEAN_VERIFY_RATIO * base[0]
    elif variant in FOLD_VARIANTS:
        assert got[0] <= FOLD_VERIFY_RATIO * base[0]
    else:
        assert abs(got[0] - base[0]) <= 0.02 * base[0]


def test_a_variant_needs_the_interleave_variable_and_an_exact_value(interleaved_verify_sim_binary: Path,
                                                                    tmp_path: Path) -> None:
    """The glue takes a variant only with the interleave variable at "1" and the variant at exactly "1" to "7"."""
    binary = interleaved_verify_sim_binary / "verify_sim_host"
    case = VERIFY_CASES[0]
    inputs = make_verify_inputs(case)
    runs = {}
    for label, values in (("none", {}), ("variant_only", {VARIANT_ENV: "3"}),
                          ("interleaved", {INTERLEAVE_ENV: "1"}), ("variant_8", {INTERLEAVE_ENV: "1", VARIANT_ENV: "8"}),
                          ("variant_x", {INTERLEAVE_ENV: "1", VARIANT_ENV: "1 "}),
                          ("variant_3", {INTERLEAVE_ENV: "1", VARIANT_ENV: "3"})):
        runs[label] = _run_verify(binary, case, inputs, tmp_path / label, values)
    assert runs["none"][0] == runs["variant_only"][0] and runs["none"][1] == runs["variant_only"][1]
    assert runs["interleaved"][0] == runs["variant_8"][0] == runs["variant_x"][0]
    assert runs["variant_3"][0] < runs["interleaved"][0] and runs["variant_3"][1] != runs["interleaved"][1]


def test_without_the_variable_a_host_with_both_elfs_runs_todays_kernel(interleaved_sim_binary: Path,
                                                                        sim_binary: Path, tmp_path: Path) -> None:
    """The glue picks the interleaved ELF only for exactly "1": unset, "0" and "2" run today's kernel."""
    case = CASES[0]
    inputs = make_inputs(case)
    counts = {}
    for label, binary, value in (("today", sim_binary, None), ("unset", interleaved_sim_binary, None),
                                 ("0", interleaved_sim_binary, "0"), ("2", interleaved_sim_binary, "2"),
                                 ("1", interleaved_sim_binary, "1")):
        run_dir = tmp_path / label
        write_inputs(case, inputs, run_dir)
        env = {key: v for key, v in os.environ.items() if key != INTERLEAVE_ENV}
        if value is not None:
            env[INTERLEAVE_ENV] = value
        proc = subprocess.run([str(binary / "sim_host"), str(run_dir)], capture_output=True, text=True, timeout=900,
                              env=env)
        assert proc.returncode == 0, proc.stderr
        counts[label] = int(re.search(r"instructions executed: (\d+)", proc.stdout).group(1))
    assert counts["unset"] == counts["0"] == counts["2"] == counts["today"]
    assert counts["1"] < counts["today"]


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
    assert len(compiles) == 5, compiles  # the three fixtures, _embed and test_vector_local_memory
    for arguments in compiles:
        assert "*TPC_FLAGS" in arguments, arguments
        assert not re.search(r'"-(vlm|slm|spill-vlm)"', arguments), arguments


@pytest.mark.parametrize("source", [
    KERNEL_SOURCE, RECURRENT_KERNEL_SOURCE, VERIFY_KERNEL_SOURCE, INTERLEAVED_DECODE_SOURCE, INTERLEAVED_VERIFY_SOURCE
],
                         ids=["decode", "recurrent", "verify", "decode_interleaved", "verify_interleaved"])
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
