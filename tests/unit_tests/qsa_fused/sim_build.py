# SPDX-License-Identifier: Apache-2.0
"""Build helpers for the simulator-side tests of the fused QSA decode kernels (as tests/unit_tests/hc_fused/sim_build.py).

``build()`` runs the real CMake project ``csrc/qsa_fused`` and links ``tests/standalone/qsa_fused/sim_host.cpp``
against the resulting ``libqsa_fused_gaudi2.so`` and the simulator library. Done once per process.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit_tests.hc_fused.sim_build import TOOLS_INCLUDE, host_include, missing_sdk_parts, run

REPO_ROOT = Path(__file__).resolve().parents[3]
CSRC = REPO_ROOT / "csrc" / "qsa_fused"
STANDALONE = REPO_ROOT / "tests" / "standalone" / "qsa_fused"
KERNELS = ("qsa_fused_prepare_gaudi2", "qsa_fused_emit_gaudi2", "qsa_fused_threshold_gaudi2", "qsa_fused_select_gaudi2")
TPC_FLAGS = ("-Wall", "-Werror", "-O2", "-march=gaudi2")
LIBRARY = "libqsa_fused_gaudi2.so"

_BUILD: Path | None = None

__all__ = ["CSRC", "KERNELS", "LIBRARY", "TPC_FLAGS", "TOOLS_INCLUDE", "build", "missing_sdk_parts", "run"]


def build(tmp_path_factory) -> Path:
    """Folder with ``lib/libqsa_fused_gaudi2.so`` (CMake) and ``sim_host``. Built once per process."""
    global _BUILD
    if _BUILD is not None:
        return _BUILD
    root = tmp_path_factory.mktemp("qsa_fused_build")
    lib = root / "lib"
    run(["cmake", "-S", CSRC, "-B", lib, f"-DTPC_INCLUDE_DIR={TOOLS_INCLUDE}", f"-DHABANA_INCLUDE_DIR={host_include()}"],
        timeout=600)
    proc = run(["cmake", "--build", lib, "--target", "qsa_fused_gaudi2", "--", "VERBOSE=1"], timeout=3600)
    (root / "build.log").write_text(proc.stdout + proc.stderr)
    assert (lib / LIBRARY).is_file()
    common = ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", f"-I{CSRC}", f"-I{TOOLS_INCLUDE}", f"-I{host_include()}"]
    run(common + [
        STANDALONE / "sim_host.cpp", f"-I{REPO_ROOT / 'tools'}", f"-L{lib}", f"-Wl,-rpath,{lib}", "-lqsa_fused_gaudi2",
        "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext", "-o", root / "sim_host"
    ],
        timeout=1800)
    _BUILD = root
    return root
