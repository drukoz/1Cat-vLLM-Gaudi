# SPDX-License-Identifier: Apache-2.0
"""Run the fused QSA decode kernels (``csrc/qsa_fused``) on the Gaudi2 TPC simulator.

The fixture builds the real kernel library with CMake (``tpc-clang -Wall -Werror -O2 -march=gaudi2``, no
``-vlm``; ``objcopy`` embedding; host glue) and links ``sim_host.cpp`` against it and the simulator library
(``tests/unit_tests/qsa_fused/sim_build.py``). Each ``qsa_case.HPU_CASES`` case (the card cases) is written to a raw-file
folder, run through the library's exported entry points (shape inference, the two-call ELF handshake) and the
simulator with its index-space mapping test and partition shuffle on, and compared with the torch reference
by ``qsa_case.compare``. Skips without the TPC toolchain; runs in the tpcsim container
(``docs/design/flash-next/tpcsim.md``). Contract: ``docs/design/flash-next/qsa-fused.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit_tests.qsa_fused import sim_build
from tests.unit_tests.qsa_fused.qsa_case import HPU_CASES, compare, make_inputs, read_outputs, summary, write_inputs

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


@pytest.mark.parametrize("kernel", sim_build.KERNELS)
def test_kernel_compiles_with_default_flags(kernel: str, tmp_path: Path) -> None:
    obj = tmp_path / f"{kernel}.o"
    proc = sim_build.run(["tpc-clang", *sim_build.TPC_FLAGS, "-reg-mem-count", f"-I{sim_build.TOOLS_INCLUDE}",
                          sim_build.CSRC / "kernels" / f"{kernel}.c", "-c", "-o", obj], timeout=1800)
    local = sim_build.LOCAL_MEMORY_KERNELS.get(kernel, 0)
    assert f"Total VLM used: {local} bytes" in proc.stdout + proc.stderr
    listing = sim_build.run(["tpc-llvm-objdump", "--triple=tpc", "--mcpu=gaudi2", "-d", obj]).stdout
    assert "TPC architecture: gaudi2" in listing
    if local:
        # no lookup-table special function: those leave only 16 KiB of vector local memory
        assert "lookup" not in listing


def test_kernels_keep_no_vector_arrays_in_local_memory() -> None:
    """Only the kernels listed in LOCAL_MEMORY_KERNELS declare __local__ arrays."""
    for path in sorted((sim_build.CSRC / "kernels").iterdir()):
        if path.stem in sim_build.LOCAL_MEMORY_KERNELS:
            continue
        assert "__local__" not in path.read_text(), path.name


@pytest.mark.parametrize("case", HPU_CASES, ids=[case.name for case in HPU_CASES])
def test_simulation_matches_reference(build: Path, case, tmp_path: Path) -> None:
    case_dir = tmp_path / case.name
    inputs = make_inputs(case)
    write_inputs(case, inputs, case_dir)
    proc = sim_build.run([build / "sim_host", case_dir], timeout=3600)
    print(proc.stdout)
    assert "mapping test: on, shuffle: on" in proc.stdout
    got = read_outputs(case, case_dir, inputs)
    print(summary(case, inputs, got))
    assert compare(case, inputs, got) == []
