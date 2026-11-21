# SPDX-License-Identifier: Apache-2.0
"""Build helpers for the simulator-side tests of the fused hyper-connection kernels.

``build()`` runs the real CMake project ``csrc/hc_fused`` (the kernels with ``tpc-clang -Wall -Werror
-O2 -march=gaudi2``, ``objcopy`` embedding, host glue) and links the two host programs of
``tests/standalone/hc_fused`` against the resulting ``libhc_fused_gaudi2.so``.  The build is done
once per process and shared by ``test_hc_simulation.py`` and ``test_hc_library.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_INCLUDE = Path("/usr/lib/habanatools/include")
DRIVER_INCLUDE = Path("/usr/include/habanalabs")
CSRC = REPO_ROOT / "csrc" / "hc_fused"
STANDALONE = REPO_ROOT / "tests" / "standalone" / "hc_fused"
KERNELS = ("hc_fused_norm_gaudi2", "hc_fused_silu_gaudi2", "hc_fused_silu_split_gaudi2", "hc_fused_gate_mix_gaudi2",
           "hc_fused_combine_gaudi2", "hc_fused_combine_norm_gaudi2")
# The MoE glue kernels of the same library (each optional at build time, csrc/hc_fused/CMakeLists.txt).
MOE_KERNELS = ("moe_glue_route_gaudi2", "moe_glue_silu_mul_gaudi2", "moe_glue_combine_gaudi2",
               "moe_glue_row_quant_gaudi2", "moe_glue_act_quant_gaudi2")
# The flags of the proven GDN decode kernel.  No -vlm: a kernel with lookup-table special functions has
# 16 KiB of vector local memory, and the graph compiler may recompile the embedded bitcode without the flag.
TPC_FLAGS = ("-Wall", "-Werror", "-O2", "-march=gaudi2")
LIBRARY = "libhc_fused_gaudi2.so"

_BUILD: Path | None = None


def missing_sdk_parts() -> list[str]:
    """Why the simulator tests cannot run on this machine ([] when they can)."""
    missing = []
    if not shutil.which("tpc-clang"):
        missing.append("tpc-clang is not on PATH")
    if not (TOOLS_INCLUDE / "tpc_test_core_api.h").is_file():
        missing.append(f"{TOOLS_INCLUDE / 'tpc_test_core_api.h'} is missing")
    if not (TOOLS_INCLUDE / "gc_interface.h").is_file() and not (DRIVER_INCLUDE / "gc_interface.h").is_file():
        missing.append(f"gc_interface.h is missing from {TOOLS_INCLUDE} and {DRIVER_INCLUDE}")
    return missing


def run(cmd: list, *, cwd: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([str(part) for part in cmd], check=True, capture_output=True, text=True, cwd=cwd,
                              timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed (exit {exc.returncode}): {' '.join(map(str, cmd))}\n"
                           f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}") from exc


def host_include() -> Path:
    """Where the Gaudi host API headers are: the driver package on a card machine, the tools package elsewhere."""
    return DRIVER_INCLUDE if (DRIVER_INCLUDE / "tpc_kernel_lib_interface.h").is_file() else TOOLS_INCLUDE


def build(tmp_path_factory) -> Path:
    """Folder with ``lib/libhc_fused_gaudi2.so`` (CMake), ``sim_host``, ``lib_check`` and ``moe_lib_check``.

    Built once per process."""
    global _BUILD
    if _BUILD is not None:
        return _BUILD
    root = tmp_path_factory.mktemp("hc_fused_build")
    lib = root / "lib"
    run([
        "cmake", "-S", CSRC, "-B", lib, f"-DTPC_INCLUDE_DIR={TOOLS_INCLUDE}", f"-DHABANA_INCLUDE_DIR={host_include()}"
    ],
        timeout=600)
    proc = run(["cmake", "--build", lib, "--target", "hc_fused_gaudi2", "--", "VERBOSE=1"], timeout=3600)
    (root / "build.log").write_text(proc.stdout + proc.stderr)
    assert (lib / LIBRARY).is_file()
    common = ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", f"-I{CSRC}", f"-I{TOOLS_INCLUDE}", f"-I{host_include()}"]
    run(common + [
        STANDALONE / "sim_host.cpp", f"-I{REPO_ROOT / 'tools'}", f"-L{lib}", f"-Wl,-rpath,{lib}", "-lhc_fused_gaudi2",
        "-L/usr/lib/habanatools", "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext", "-o", root / "sim_host"
    ],
        timeout=1800)
    run(common + [STANDALONE / "lib_check.cpp", "-ldl", "-o", root / "lib_check"], timeout=1800)
    run(common + [STANDALONE / "moe_lib_check.cpp", "-ldl", "-o", root / "moe_lib_check"], timeout=1800)
    _BUILD = root
    return root
