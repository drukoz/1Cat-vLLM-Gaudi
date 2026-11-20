# SPDX-License-Identifier: Apache-2.0
"""Run the fused MoE glue kernels (the ``moe_glue_*`` kernels of ``csrc/hc_fused``) on the Gaudi2 TPC simulator.

The fixture is the one of ``test_hc_simulation.py``: the real CMake project builds
``libhc_fused_gaudi2.so`` (which carries the hyper-connection and the MoE kernels), and ``sim_host.cpp``
runs each ``moe_case.SIM_CASES`` case through the library's exported entry points and the simulator, with
the simulator's index-space mapping test and partition shuffle on.  ``moe_case.compare`` checks the
result against the torch oracle (``tests/unit_tests/hc_fused/moe_reference.py``).

Skips on machines without the TPC toolchain (this Mac); runs for real in the tpcsim container
(``docs/design/flash-next/tpcsim.md``) and on the rental.  Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit_tests.hc_fused import sim_build
from tests.unit_tests.hc_fused.moe_case import SIM_CASES, compare, make_inputs, read_outputs, summary, write_inputs

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))

VLM_LIMIT = 16384  # bytes of vector local memory of a kernel that calls lookup-table functions
VLM_HEADROOM = 3072  # what every kernel must leave free (the rule of the GDN kernels)


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


def test_cmake_built_every_moe_kernel_with_the_default_flags(build: Path) -> None:
    """Every MoE kernel compiled at configure time, and the build ran tpc-clang on it with the default flags."""
    cache = (build / "lib" / "CMakeCache.txt").read_text()
    built = re.search(r"^HC_FUSED_MOE_GLUE_BUILT:INTERNAL=(.*)$", cache, re.MULTILINE)
    assert built and built.group(1).split(";") == list(sim_build.MOE_KERNELS)
    log = (build / "build.log").read_text()
    for kernel in sim_build.MOE_KERNELS:
        commands = [line for line in log.splitlines() if "tpc-clang" in line and f"{kernel}.c" in line]
        assert commands, f"the build log has no tpc-clang command for {kernel}"
        for command in commands:
            assert " ".join(sim_build.TPC_FLAGS) in command
            assert not re.search(r"-+(vlm|slm|spill-vlm)\b", command)


@pytest.mark.parametrize("kernel", sim_build.MOE_KERNELS)
def test_kernel_compiles_alone_and_leaves_local_memory_free(kernel: str, tmp_path: Path) -> None:
    """Each kernel compiles with exactly the default flags, for gaudi2, and uses no vector local memory to speak of."""
    obj = tmp_path / f"{kernel}.o"
    proc = sim_build.run([
        "tpc-clang", *sim_build.TPC_FLAGS, "-reg-mem-count", f"-I{sim_build.TOOLS_INCLUDE}",
        sim_build.CSRC / "kernels" / f"{kernel}.c", "-c", "-o", obj
    ],
                         timeout=1800)
    report = proc.stdout + proc.stderr
    used = re.search(r"Total VLM used: (\d+) bytes", report)
    assert used, report
    print(f"MOEVLM {kernel}: {used.group(0)}")
    assert int(used.group(1)) <= VLM_LIMIT - VLM_HEADROOM
    listing = sim_build.run(["tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d", obj]).stdout
    assert "TPC architecture: gaudi2" in listing


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
