# SPDX-License-Identifier: Apache-2.0
"""Run the GDN prefill kernels (``csrc/flashqla_hpu``) on the Gaudi2 TPC simulator.

The fixture builds the real kernel library with CMake (``csrc/flashqla_hpu/CMakeLists.txt``: every kernel
with ``tpc-clang -Wall -Werror -march=gaudi2 -O2``, no ``-vlm``; ``objcopy`` embedding; the host glue), then
links ``sim_host.cpp`` against ``libflashqla_hpu_perf_lib.so`` and the simulator library
(``sim_build.py``).  Each ``kernel_cases.SIM_CASES`` case is written to a raw-file folder, run through the
library's exported ``InstantiateTpcKernel`` and the simulator, and compared with the torch reference by
``kernel_cases.compare``.  The cases cover the geometry of one TP8 rank (2 key / 6 value heads, packed
q|k|v width 1280) at one 128-token chunk and at a 2048-token step, TP4 (4 / 12, 2560), and the layouts the
library already had (TP2 8 / 24, TP1 16 / 48).

Every run has the simulator's index-space partition shuffle on, so a kernel is run on partial index spaces
in a shuffled order.  The index-space mapping test is on for the kernels whose glue declares exact access
patterns (``MAPPING_TESTED``).

Skips on machines without the TPC toolchain (this Mac); runs for real in the tpcsim container::

    tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest \
        tests/standalone/gdn_prefill/test_simulation.py'
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.standalone.gdn_prefill import sim_build
from tests.standalone.gdn_prefill.kernel_cases import (REFUSED_CASES, SIM_CASES, Case, compare, make_inputs,
                                                       read_outputs, summary, write_inputs)

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))

# Kinds whose glue access patterns pass the simulator's index-space mapping test.
MAPPING_TESTED = ("compact_qk", "expanded_qk", "f32_qk", "compact_kkt", "rmsnorm_gated", "inverse16", "kkt_form",
                  "pair_transform")


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


def test_cmake_compiles_every_kernel_with_the_default_flags(build: Path) -> None:
    """Every kernel source and every compile-time variant went through tpc-clang with the flags and no -vlm."""
    log = (build / "build.log").read_text()
    commands = [line for line in log.splitlines() if "tpc-clang" in line]
    for kernel in sim_build.KERNELS:
        mine = [line for line in commands if f"{kernel}.c" in line and f"-o {kernel}.o" in line]
        assert mine, f"the build log has no tpc-clang command for {kernel}"
        for flag in sim_build.TPC_FLAGS:
            assert f" {flag}" in mine[0]
    for variant, (source, define) in sim_build.VARIANTS.items():
        mine = [line for line in commands if f"{source}.c" in line and f"-o {variant}.o" in line]
        assert mine, f"the build log has no tpc-clang command for the variant {variant}"
        assert define in mine[0]
    assert all("-vlm" not in line and "-slm" not in line for line in commands)


@pytest.mark.parametrize("kernel", sim_build.KERNELS + tuple(sim_build.VARIANTS))
def test_vector_local_memory(kernel: str, tmp_path: Path) -> None:
    """tpc-clang's own count of vector local memory leaves 3 KiB of the 16 KiB free for every kernel."""
    source, extra = sim_build.VARIANTS.get(kernel, (kernel, None))
    command = ["tpc-clang", *sim_build.TPC_FLAGS, "-reg-mem-count", f"-I{sim_build.TOOLS_INCLUDE}"]
    if extra:
        command.append(extra)
    proc = sim_build.run(command + [sim_build.CSRC / f"{source}.c", "-c", "-o", tmp_path / f"{kernel}.o"],
                         timeout=1800)
    match = re.search(r"Total VLM used: (\d+) bytes", proc.stdout + proc.stderr)
    assert match, proc.stdout + proc.stderr
    print(f"{kernel}: VLM {match.group(1)} bytes")
    assert int(match.group(1)) <= sim_build.VLM_LIMIT - 3072


@pytest.mark.parametrize("case", SIM_CASES, ids=[case.name for case in SIM_CASES])
def test_simulation_matches_reference(build: Path, case: Case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    inputs = make_inputs(case)
    write_inputs(case, inputs, case_dir)
    options = [] if case.kind in MAPPING_TESTED else ["--no-mapping-test"]
    proc = sim_build.run([build / "sim_host", case_dir, *options], timeout=3600)
    print(proc.stdout)
    assert "shuffle: on" in proc.stdout
    assert ("mapping test: on" in proc.stdout) == (case.kind in MAPPING_TESTED)
    got = read_outputs(case, inputs, case_dir)
    print(summary(case, inputs, got))
    assert compare(case, inputs, got) == []


@pytest.mark.parametrize("case", REFUSED_CASES, ids=[case.name for case in REFUSED_CASES])
def test_glue_refuses_a_shape_of_no_supported_layout(build: Path, case: Case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    write_inputs(case, make_inputs(case), case_dir)
    refused = sim_build.glue_code("GLUE_UNSUPPORTED_LAYER_CONFIGURATION")
    proc = sim_build.run([build / "sim_host", case_dir, "--expect-glue", refused], timeout=600, check=False)
    print(proc.stdout, proc.stderr)
    assert proc.returncode == 0, f"the glue did not answer {refused}: {proc.stdout}"


def test_inverse16_glue_takes_the_kernels_one_output(build: Path, tmp_path: Path) -> None:
    """The inverse kernel has one output (``main(tensor lower, tensor inverse)``, schema ``-> Tensor``)."""
    case = Case("inverse16_one_output", "inverse16", key_heads=2, outer=1, chunk=32)
    case_dir = tmp_path / case.name
    inputs = make_inputs(case)
    write_inputs(case, inputs, case_dir)
    proc = sim_build.run([build / "sim_host", case_dir], timeout=600)
    print(proc.stdout)
    assert compare(case, inputs, read_outputs(case, inputs, case_dir)) == []
