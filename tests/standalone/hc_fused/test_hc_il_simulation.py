# SPDX-License-Identifier: Apache-2.0
"""Run the interleaved hyper-connection kernels (``hc_fused_gate_mix_il``, ``hc_fused_combine_norm_il``) on the
Gaudi2 TPC simulator: docs/design/flash-next/hc-kernels.md.

The library is the one ``test_hc_simulation.py`` builds (``tests/unit_tests/hc_fused/sim_build.py``: the real
CMake project, the shipped flags). Every ``hc_case.SIM_CASES`` case of the kinds ``gate_mix`` and
``combine_norm`` runs through the interleaved GUID with the simulator's index-space mapping test and partition
shuffle on, and must pass the oracle comparison of the HC contract (``hc_case.compare``). The same inputs also run
through the shipped GUID, and the outputs are compared bit for bit: ``residual_out`` of combine_norm must be the
shipped kernel's exactly (same rounding points, same injection-weight sigmoid), the other outputs in at least
``MIN_BIT_EQUAL`` of their elements (the interleaved sigmoid and the norm's summation order differ in the last
fp32 bit). A cost check holds the speed-up: the simulator's instructions per index-space point at the real
geometry, against the shipped kernels' counts measured when the kernels were written.

Skips on machines without the TPC toolchain (this Mac); runs in the tpcsim container (docs/design/flash-next/tpcsim.md).
"""

from __future__ import annotations

import ctypes
import math
import re
from pathlib import Path

import pytest
import torch

from tests.unit_tests.hc_fused import sim_build
from tests.unit_tests.hc_fused.hc_case import (GUIDS, SIM_CASES, HcCase, compare, make_inputs, read_outputs, summary,
                                               write_inputs)

_MISSING = sim_build.missing_sdk_parts()
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="Gaudi2 TPC simulator SDK not installed: " + "; ".join(_MISSING))

IL_GUIDS = {
    "gate_mix": "hc_fused_gate_mix_il_bf16_gaudi2",
    "combine_norm": "hc_fused_combine_norm_il_bf16_gaudi2",
}
CASES = tuple(case for case in SIM_CASES if case.kind in IL_GUIDS)
MIN_BIT_EQUAL = 0.999
# Instructions per index-space point of the shipped kernels at the real geometry (simulator, 2026-11-21): gate_mix
# 670 per (chunk, token), combine_norm 2,063 per (stream, token), i.e. 8,252 per token.
SHIPPED_PER_TOKEN = {"gate_mix": 670 * 20, "combine_norm": 2063 * 4}
MIN_SPEEDUP = {"gate_mix": 2.4, "combine_norm": 3.3}


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return sim_build.build(tmp_path_factory)


def _run(build: Path, case: HcCase, inputs, directory: Path, guid: str, checks: bool = True) -> tuple[dict, str]:
    write_inputs(case, inputs, directory)
    manifest = directory / "case.txt"
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join(f"guid {guid}" if line.startswith("guid ") else line for line in lines) + "\n")
    args = [build / "sim_host", directory] + ([] if checks else ["--no-mapping-test", "--no-shuffle"])
    proc = sim_build.run(args, timeout=3600)
    return read_outputs(case, directory), proc.stdout


def _library_guids(path: Path) -> tuple[str, ...]:
    """GetKernelGuids of the kernel library (the two-call protocol), as flashinfer_gaudi._native asks it."""

    class GuidInfo(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char * 64), ("name_hash", ctypes.c_uint64),
                    ("kernel_properties", ctypes.c_uint32)]

    library = ctypes.CDLL(str(path))
    library.GetKernelGuids.restype = ctypes.c_int
    library.GetKernelGuids.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    count = ctypes.c_uint32(0)
    assert library.GetKernelGuids(3, ctypes.byref(count), None) == 0  # 3 = DEVICE_ID_GAUDI2
    guids = (GuidInfo * max(count.value, 1))()
    assert library.GetKernelGuids(3, ctypes.byref(count), ctypes.cast(guids, ctypes.c_void_p)) == 0
    return tuple(guids[index].name.decode() for index in range(count.value))


def test_the_library_lists_the_interleaved_guids_after_the_moe_glue(build: Path) -> None:
    guids = _library_guids(build / "lib" / sim_build.LIBRARY)
    print(guids)
    assert guids[-2:] == tuple(IL_GUIDS.values())
    assert guids[:6] == tuple(GUIDS[kind] for kind in ("norm", "silu", "silu_split", "gate_mix", "combine",
                                                        "combine_norm"))


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_simulation_matches_reference_and_shipped_kernel(build: Path, case: HcCase, tmp_path: Path) -> None:
    inputs = make_inputs(case)
    got, stdout = _run(build, case, inputs, tmp_path / "il", IL_GUIDS[case.kind])
    print(stdout)
    assert "mapping test: on, shuffle: on" in stdout
    print(summary(case, inputs, got))
    assert compare(case, inputs, got) == []
    shipped, _ = _run(build, case, inputs, tmp_path / "shipped", GUIDS[case.kind], checks=False)
    for name, tensor in got.items():
        equal = float((tensor.view(torch.int16) == shipped[name].view(torch.int16)).float().mean())
        print(f"HCIL {case.name} {name}: {100 * equal:.4f} % bit-equal to the shipped kernel")
        if name == "residual_out" or case.exact:
            assert equal == 1.0, name
        else:
            assert equal >= MIN_BIT_EQUAL, name


@pytest.mark.parametrize("kind", sorted(IL_GUIDS))
def test_cost_per_token_at_the_real_geometry(build: Path, kind: str, tmp_path: Path) -> None:
    case = HcCase(name=f"cost_{kind}_r32", kind=kind, hidden=2560, rows=32, seed=932)
    _, stdout = _run(build, case, make_inputs(case), tmp_path, IL_GUIDS[kind], checks=False)
    count = int(re.search(r"instructions executed: (\d+)", stdout).group(1))
    geometry = re.search(r"index space: rank \d+ \{([^}]*)\}", stdout).group(1)
    points = math.prod(int(v) for v in geometry.split(","))
    per_token = count / case.rows
    print(f"HCIL cost {kind}: index space {{{geometry}}} = {points} points, {count} instructions, "
          f"{per_token:.0f} per token (shipped {SHIPPED_PER_TOKEN[kind]})")
    assert SHIPPED_PER_TOKEN[kind] / per_token >= MIN_SPEEDUP[kind]
