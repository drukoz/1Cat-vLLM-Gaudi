# SPDX-License-Identifier: Apache-2.0
"""Run the fused hyper-connection kernels (``csrc/hc_fused``) on the Gaudi2 TPC simulator.

The fixture builds the real kernel library with CMake (``csrc/hc_fused/CMakeLists.txt``:
``tpc-clang -Wall -Werror -O2 -march=gaudi2``, no ``-vlm``; ``objcopy`` embedding; host glue), then
links ``sim_host.cpp`` against ``libhc_fused_gaudi2.so`` and the simulator library
(``tests/unit_tests/hc_fused/sim_build.py``).  Each ``hc_case.SIM_CASES`` case is written to a
raw-file folder, run through the library's exported entry points and the simulator, and compared
with the torch oracle by ``hc_case.compare``.  Every run has the simulator's index-space mapping
test on and its index-space partition shuffle on, so the access patterns of the glue code are
checked against what the kernel really reads and writes, and the kernel is run on partial index
spaces in a shuffled order.

Skips on machines without the TPC toolchain (this Mac); runs for real in the tpcsim container
(``docs/design/flash-next/tpcsim.md``) and on the rental.  Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit_tests.hc_fused import sim_build
from tests.unit_tests.hc_fused.hc_case import SIM_CASES, compare, make_inputs, read_outputs, summary, write_inputs

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


def test_cmake_compiles_kernels_with_the_default_flags(build: Path) -> None:
    """The CMake project names every kernel, has no -vlm, and really ran tpc-clang with the default flags."""
    lines = (sim_build.CSRC / "CMakeLists.txt").read_text().splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    assert "-vlm" not in code
    assert " ".join(sim_build.TPC_FLAGS) in code
    log = (build / "build.log").read_text()
    for kernel in sim_build.KERNELS:
        assert kernel in code
        commands = [line for line in log.splitlines() if "tpc-clang" in line and f"{kernel}.c" in line]
        assert commands, f"the build log has no tpc-clang command for {kernel}"
        for command in commands:
            assert " ".join(sim_build.TPC_FLAGS) in command
            assert "-vlm" not in command


@pytest.mark.parametrize("kernel", sim_build.KERNELS)
def test_kernel_compiles_with_default_flags_for_gaudi2(kernel: str, tmp_path: Path) -> None:
    """Each kernel compiles on its own with exactly the GDN decode kernel's flags and no -vlm."""
    obj = tmp_path / f"{kernel}.o"
    proc = sim_build.run([
        "tpc-clang", *sim_build.TPC_FLAGS, f"-I{sim_build.TOOLS_INCLUDE}", sim_build.CSRC / "kernels" / f"{kernel}.c",
        "-c", "-o", obj
    ],
                         timeout=1800)
    print(proc.stderr)
    listing = sim_build.run(["tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d", obj]).stdout
    assert "TPC architecture: gaudi2" in listing


def test_kernels_keep_no_vector_arrays_in_local_memory() -> None:
    """No kernel source declares local-memory data: with special functions only 16 KiB of it exist."""
    for path in sorted((sim_build.CSRC / "kernels").iterdir()):
        assert "__local__" not in path.read_text(), path.name


@pytest.mark.parametrize("case", SIM_CASES, ids=[case.name for case in SIM_CASES])
def test_simulation_matches_reference(build: Path, case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    inputs = make_inputs(case)
    write_inputs(case, inputs, case_dir)
    proc = sim_build.run([build / "sim_host", case_dir], timeout=3600)
    print(proc.stdout)
    assert "mapping test: on, shuffle: on" in proc.stdout
    got = read_outputs(case, case_dir)
    print(summary(case, inputs, got))
    assert compare(case, inputs, got) == []
