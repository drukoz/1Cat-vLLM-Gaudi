# SPDX-License-Identifier: Apache-2.0
"""Build helpers for the simulator tests of the GDN prefill kernels (``csrc/flashqla_hpu``).

``build()`` runs the real CMake project (every kernel with ``tpc-clang -Wall -Werror -march=gaudi2 -O2``,
``objcopy`` embedding, the host glue ``flashqla_pair_transform_perf_lib.cpp``) and links ``sim_host.cpp``
of this folder against the resulting ``libflashqla_hpu_perf_lib.so`` and the simulator library.  Built once
per process.  No vLLM import: this runs in the tpcsim container (``docs/design/flash-next/tpcsim.md``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_INCLUDE = Path("/usr/lib/habanatools/include")
DRIVER_INCLUDE = Path("/usr/include/habanalabs")
CSRC = REPO_ROOT / "csrc" / "flashqla_hpu"
HOST_SOURCE = Path(__file__).resolve().parent / "sim_host.cpp"
LIBRARY = "libflashqla_hpu_perf_lib.so"
# The kernel sources of the library, and the compile-time variants CMake builds from them
# (object name -> (source, extra flags)).
KERNELS = ("flashqla_pair_transform_bf16_gaudi2", "flashqla_recurrent_prefill_f32_gaudi2",
           "flashqla_unit_lower_inverse16_f32_gaudi2", "flashqla_kkt_form_bf16_gaudi2",
           "qwen38_compact_kkt_bf16_gaudi2", "qwen38_post_conv_qk_bf16_gaudi2",
           "qwen38_post_conv_qk_compact_bf16_gaudi2", "qwen38_post_conv_qk_expanded_bf16_gaudi2",
           "qwen38_conv_qkv_prep_bf16_gaudi2", "qwen38_rmsnorm_gated_bf16_gaudi2")
VARIANTS = {
    "qwen38_post_conv_qk_compact_tp2_bf16_gaudi2": ("qwen38_post_conv_qk_compact_bf16_gaudi2", "-DQWEN38_QK_HEADS=8"),
    "qwen38_post_conv_qk_compact_tp4_bf16_gaudi2": ("qwen38_post_conv_qk_compact_bf16_gaudi2", "-DQWEN38_QK_HEADS=4"),
    "qwen38_post_conv_qk_compact_tp8_bf16_gaudi2": ("qwen38_post_conv_qk_compact_bf16_gaudi2", "-DQWEN38_QK_HEADS=2"),
}
TPC_FLAGS = ("-Wall", "-Werror", "-march=gaudi2", "-O2")  # as csrc/flashqla_hpu/CMakeLists.txt; no -vlm
# Vector local memory of a kernel that calls lookup-table functions (tpc-clang's own limit), in bytes.
VLM_LIMIT = 16384
# tpc_lib_api::GlueCodeReturn values the tests name (gc_interface.h of habanalabs-tools 1.24.1).
GLUE_SUCCESS = 0

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


def run(cmd: list, *, cwd: Path | None = None, timeout: float | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(part) for part in cmd], capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}\n"
                           f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def host_include() -> Path:
    """Where the Gaudi host API headers are: the driver package on a card machine, the tools package elsewhere."""
    return DRIVER_INCLUDE if (DRIVER_INCLUDE / "tpc_kernel_lib_interface.h").is_file() else TOOLS_INCLUDE


def glue_code(name: str) -> int:
    """The numeric value of one ``tpc_lib_api::GlueCodeReturn`` enumerator, read from the SDK header."""
    for folder in (host_include(), TOOLS_INCLUDE):
        header = folder / "tpc_kernel_lib_interface.h"
        if not header.is_file():
            continue
        match = re.search(rf"\b{name}\s*=\s*(-?\d+)", header.read_text(errors="replace"))
        if match:
            return int(match.group(1))
    raise RuntimeError(f"{name} is not in tpc_kernel_lib_interface.h")


def build(tmp_path_factory) -> Path:
    """Folder with ``lib/libflashqla_hpu_perf_lib.so`` (CMake), ``build.log`` and ``sim_host``."""
    global _BUILD
    if _BUILD is not None:
        return _BUILD
    root = tmp_path_factory.mktemp("gdn_prefill_build")
    lib = root / "lib"
    run(["cmake", "-S", CSRC, "-B", lib, "-DCMAKE_BUILD_TYPE=Release"], timeout=600)
    proc = run(["cmake", "--build", lib, "--target", "flashqla_hpu_perf_lib", "--", "VERBOSE=1"], timeout=3600)
    (root / "build.log").write_text(proc.stdout + proc.stderr)
    assert (lib / LIBRARY).is_file()
    run([
        "g++", "-std=c++17", "-O2", "-Wall", "-Werror", HOST_SOURCE, f"-I{REPO_ROOT / 'tools'}", f"-I{TOOLS_INCLUDE}",
        f"-I{host_include()}", f"-L{lib}", f"-Wl,-rpath,{lib}", "-lflashqla_hpu_perf_lib", "-L/usr/lib/habanatools",
        "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext", "-o", root / "sim_host"
    ],
        timeout=1800)
    _BUILD = root
    return root
