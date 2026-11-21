# SPDX-License-Identifier: Apache-2.0
"""Build the in-tree FlashInfer-Gaudi kernel database and PyTorch op."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_kernel_database(source_root: Path, build_root: Path, output_dir: Path) -> Path:
    cmake_build = build_root / "kernel_db"
    _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=source_root,
    )
    _run(
        ["cmake", "--build", str(cmake_build), "--target", "flashinfer_gaudi_kernels", "--parallel"],
        cwd=source_root,
    )
    built = cmake_build / "libflashinfer_gaudi_kernels.so"
    if not built.is_file():
        raise FileNotFoundError(f"CMake did not produce {built}.")
    destination = output_dir / built.name
    shutil.copy2(built, destination)
    return destination


def build_pytorch_extension(source_root: Path, build_root: Path, output_dir: Path) -> Path:
    pytorch_root = source_root / "pytorch"
    extension_build = build_root / "pytorch"
    _run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(output_dir),
            "--build-temp",
            str(extension_build),
        ],
        cwd=pytorch_root,
    )
    matches = sorted(output_dir.glob("flashinfer_gaudi_ops*.so"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one flashinfer_gaudi_ops extension in {output_dir}, found {matches}.")
    return matches[0]


def build_gdn_fused(repository_root: Path, build_root: Path, output_dir: Path) -> Path:
    source_root = repository_root / "csrc" / "gdn_fused"
    cmake_build = build_root / "gdn_fused_kernel_db"
    _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=source_root,
    )
    _run(
        ["cmake", "--build", str(cmake_build), "--target", "gdn_fused_gaudi2", "--parallel"],
        cwd=source_root,
    )
    built = cmake_build / "libgdn_fused_gaudi2.so"
    if not built.is_file():
        raise FileNotFoundError(f"CMake did not produce {built}.")
    shutil.copy2(built, output_dir / built.name)

    pytorch_root = source_root / "pytorch"
    extension_build = build_root / "gdn_fused_pytorch"
    _run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(output_dir),
            "--build-temp",
            str(extension_build),
        ],
        cwd=pytorch_root,
    )
    matches = sorted(output_dir.glob("gdn_fused_ops*.so"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one gdn_fused_ops extension in {output_dir}, found {matches}.")
    return matches[0]


def _remove_gdn_fused_outputs(output_dir: Path) -> None:
    """Delete both fused GDN libraries from output_dir, so a failed build leaves neither half behind."""
    for path in [output_dir / "libgdn_fused_gaudi2.so", *sorted(output_dir.glob("gdn_fused_ops*.so"))]:
        if path.is_file():
            path.unlink()


# What the combined kernel database (csrc/kernel_db_adapter/adapter.cpp) needs from a member library.
KERNEL_LIBRARY_ENTRY_POINTS = ("GetKernelGuids", "InstantiateTpcKernel", "GetShapeInference")
DEVICE_ID_GAUDI2 = 3  # tpc_lib_api::DeviceId, tpc_kernel_lib_interface.h
GLUE_SUCCESS = 0


def check_kernel_library(path: Path) -> int:
    """Raise RuntimeError unless `path` is a kernel library the combined kernel database can use; its GUID count.

    The combined database loads every member library that sits next to it, and one member it cannot
    use (it does not load, lacks a required entry point, or its GetKernelGuids fails) makes the whole
    database serve nothing. A new library is therefore checked here, the way the adapter uses it,
    before it is left in the folder.
    """
    try:
        library = ctypes.CDLL(str(path))
    except OSError as exc:
        raise RuntimeError(f"{path} does not load: {exc}") from exc
    for name in KERNEL_LIBRARY_ENTRY_POINTS:
        if not hasattr(library, name):
            raise RuntimeError(f"{path} does not export the required entry point {name}")
    count = ctypes.c_uint32(0)
    library.GetKernelGuids.restype = ctypes.c_int
    library.GetKernelGuids.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    status = library.GetKernelGuids(DEVICE_ID_GAUDI2, ctypes.byref(count), None)
    if status != GLUE_SUCCESS or count.value == 0:
        raise RuntimeError(f"{path}: GetKernelGuids for Gaudi2 returned {status} with {count.value} kernels")
    return count.value


# The MoE glue kernels of csrc/hc_fused (VLLM_HPU_MOE_GLUE): part of libhc_fused_gaudi2.so, each optional.
MOE_GLUE_KERNELS = ("moe_glue_route_gaudi2", "moe_glue_silu_mul_gaudi2", "moe_glue_combine_gaudi2",
                    "moe_glue_row_quant_gaudi2")


def report_moe_glue_kernels(cmake_build: Path) -> tuple[str, ...]:
    """Print which MoE glue kernels CMake built into libhc_fused_gaudi2.so; the kernels that are missing.

    csrc/hc_fused/CMakeLists.txt tries the compile of each MoE glue kernel at configure time and leaves one
    that does not compile out of the library (HC_FUSED_MOE_GLUE_BUILT in CMakeCache.txt). That is not a
    failed build: the hyper-connection kernels are complete, and a server with VLLM_HPU_MOE_GLUE=1 then
    runs the torch path of its MoE blocks and says so.
    """
    built: tuple[str, ...] = ()
    cache = cmake_build / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("HC_FUSED_MOE_GLUE_BUILT:"):
                built = tuple(name for name in line.split("=", 1)[1].split(";") if name)
    missing = tuple(name for name in MOE_GLUE_KERNELS if name not in built)
    if built:
        print("MoE glue kernels in libhc_fused_gaudi2.so: " + ", ".join(built))
    if missing:
        print("MoE glue kernels missing from libhc_fused_gaudi2.so: " + ", ".join(missing) +
              " (they did not compile or were switched off; with VLLM_HPU_MOE_GLUE=1 the MoE blocks run the "
              "torch path)")
    return missing


def build_hc_fused(repository_root: Path, build_root: Path, output_dir: Path) -> Path:
    """Build the fused hyper-connection kernel database and its HPU op extension into output_dir.

    The same two steps as build_gdn_fused (CMake for libhc_fused_gaudi2.so, setuptools for hc_fused_ops*.so),
    then check_kernel_library on the installed kernel database. The library also carries the MoE glue
    kernels; report_moe_glue_kernels says which of them were built.
    """
    source_root = repository_root / "csrc" / "hc_fused"
    cmake_build = build_root / "hc_fused_kernel_db"
    _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=source_root,
    )
    _run(
        ["cmake", "--build", str(cmake_build), "--target", "hc_fused_gaudi2", "--parallel"],
        cwd=source_root,
    )
    built = cmake_build / "libhc_fused_gaudi2.so"
    if not built.is_file():
        raise FileNotFoundError(f"CMake did not produce {built}.")
    kernel_db = output_dir / built.name
    shutil.copy2(built, kernel_db)

    pytorch_root = source_root / "pytorch"
    extension_build = build_root / "hc_fused_pytorch"
    _run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-lib",
            str(output_dir),
            "--build-temp",
            str(extension_build),
        ],
        cwd=pytorch_root,
    )
    matches = sorted(output_dir.glob("hc_fused_ops*.so"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one hc_fused_ops extension in {output_dir}, found {matches}.")
    check_kernel_library(kernel_db)
    report_moe_glue_kernels(cmake_build)
    return matches[0]


def _remove_hc_fused_outputs(output_dir: Path) -> None:
    """Delete both fused hyper-connection libraries from output_dir, and nothing else."""
    for path in [output_dir / "libhc_fused_gaudi2.so", *sorted(output_dir.glob("hc_fused_ops*.so"))]:
        if path.is_file():
            path.unlink()


def build_qsa_fused(repository_root: Path, build_root: Path, output_dir: Path) -> Path:
    """Build the fused QSA decode kernel database and its HPU op extension into output_dir (as build_hc_fused)."""
    source_root = repository_root / "csrc" / "qsa_fused"
    cmake_build = build_root / "qsa_fused_kernel_db"
    _run(["cmake", "-S", str(source_root), "-B", str(cmake_build), "-DCMAKE_BUILD_TYPE=Release"], cwd=source_root)
    _run(["cmake", "--build", str(cmake_build), "--target", "qsa_fused_gaudi2", "--parallel"], cwd=source_root)
    built = cmake_build / "libqsa_fused_gaudi2.so"
    if not built.is_file():
        raise FileNotFoundError(f"CMake did not produce {built}.")
    kernel_db = output_dir / built.name
    shutil.copy2(built, kernel_db)
    pytorch_root = source_root / "pytorch"
    extension_build = build_root / "qsa_fused_pytorch"
    _run([sys.executable, "setup.py", "build_ext", "--build-lib", str(output_dir), "--build-temp", str(extension_build)],
         cwd=pytorch_root)
    matches = sorted(output_dir.glob("qsa_fused_ops*.so"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one qsa_fused_ops extension in {output_dir}, found {matches}.")
    check_kernel_library(kernel_db)
    return matches[0]


def _remove_qsa_fused_outputs(output_dir: Path) -> None:
    """Delete both fused QSA libraries from output_dir, and nothing else."""
    for path in [output_dir / "libqsa_fused_gaudi2.so", *sorted(output_dir.glob("qsa_fused_ops*.so"))]:
        if path.is_file():
            path.unlink()


def build_combined_kernel_db(repository_root: Path, build_root: Path, output_dir: Path) -> Path:
    """Build the combined kernel database adapter that forwards to the member libraries."""
    source_root = repository_root / "csrc" / "kernel_db_adapter"
    cmake_build = build_root / "kernel_db_adapter"
    _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(cmake_build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKERNEL_DB_ADAPTER_TESTS=OFF",
        ],
        cwd=source_root,
    )
    _run(
        ["cmake", "--build", str(cmake_build), "--target", "flashinfer_gaudi_combined_gaudi2", "--parallel"],
        cwd=source_root,
    )
    built = cmake_build / "libflashinfer_gaudi_combined_gaudi2.so"
    if not built.is_file():
        raise FileNotFoundError(f"CMake did not produce {built}.")
    shutil.copy2(built, output_dir / built.name)
    return built


def _remove_combined_kernel_db_output(output_dir: Path) -> None:
    """Delete the combined kernel database from output_dir, so a failed build leaves it not at all."""
    path = output_dir / "libflashinfer_gaudi_combined_gaudi2.so"
    if path.is_file():
        path.unlink()


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "flashinfer_gaudi" / "lib",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=repository_root / "build" / "flashinfer_gaudi",
    )
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--extension-only", action="store_true")
    parser.add_argument("--gdn-fused-only", action="store_true")
    parser.add_argument("--hc-fused-only", action="store_true")
    parser.add_argument("--qsa-fused-only", action="store_true")
    parser.add_argument("--combined-only", action="store_true")
    args = parser.parse_args()
    if args.kernel_only and args.extension_only:
        parser.error("--kernel-only and --extension-only are mutually exclusive")
    if args.gdn_fused_only and (args.kernel_only or args.extension_only):
        parser.error("--gdn-fused-only is mutually exclusive with --kernel-only and --extension-only")
    if args.combined_only and (args.kernel_only or args.extension_only or args.gdn_fused_only):
        parser.error("--combined-only is mutually exclusive with --kernel-only, --extension-only and --gdn-fused-only")
    if args.hc_fused_only and (args.kernel_only or args.extension_only or args.gdn_fused_only or args.combined_only):
        parser.error("--hc-fused-only is mutually exclusive with --kernel-only, --extension-only, --gdn-fused-only "
                     "and --combined-only")

    source_root = repository_root / "csrc" / "flashinfer_gaudi"
    output_dir = args.output_dir.resolve()
    build_root = args.build_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    if args.qsa_fused_only and (args.kernel_only or args.extension_only or args.gdn_fused_only or args.combined_only
                                or args.hc_fused_only):
        parser.error("--qsa-fused-only is mutually exclusive with the other --*-only options")
    if not (args.gdn_fused_only or args.hc_fused_only or args.qsa_fused_only or args.combined_only):
        if not args.extension_only:
            build_kernel_database(source_root, build_root, output_dir)
        if not args.kernel_only:
            build_pytorch_extension(source_root, build_root, output_dir)
    if args.gdn_fused_only or os.environ.get("VLLM_GAUDI_BUILD_GDN_FUSED") == "1":
        try:
            build_gdn_fused(repository_root, build_root, output_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"GDN fused build failed (the FlashInfer-Gaudi libraries are unaffected): {exc}", file=sys.stderr)
            _remove_gdn_fused_outputs(output_dir)
            if args.gdn_fused_only:
                sys.exit(3)
            print("Continuing without libgdn_fused_gaudi2.so: GDN decode will run the torch path.", file=sys.stderr)
    # The fused hyper-connection kernels (VLLM_HPU_HC_FUSED=1 at run time). Their own try block and a
    # wide except: whatever goes wrong here removes only the two hyper-connection files, and the
    # install goes on with every other library as it is.
    if args.hc_fused_only or os.environ.get("VLLM_GAUDI_BUILD_HC_FUSED") == "1":
        try:
            build_hc_fused(repository_root, build_root, output_dir)
        except Exception as exc:  # noqa: BLE001 - an optional library never fails the install
            print(f"HC fused build failed (the other libraries are unaffected): {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            _remove_hc_fused_outputs(output_dir)
            if args.hc_fused_only:
                sys.exit(3)
            print(
                "Continuing without libhc_fused_gaudi2.so: the hyper-connections will run the torch path.",
                file=sys.stderr,
            )

    # The fused QSA decode kernels (VLLM_HPU_QSA_FUSED=2 at run time): the rule of the hyper-connection
    # libraries, a failure removes only the two QSA files.
    if args.qsa_fused_only or os.environ.get("VLLM_GAUDI_BUILD_QSA_FUSED") == "1":
        try:
            build_qsa_fused(repository_root, build_root, output_dir)
        except Exception as exc:  # noqa: BLE001 - an optional library never fails the install
            print(f"QSA fused build failed (the other libraries are unaffected): {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            _remove_qsa_fused_outputs(output_dir)
            if args.qsa_fused_only:
                sys.exit(3)
            print("Continuing without libqsa_fused_gaudi2.so: the QSA layers will run the torch data flow.",
                  file=sys.stderr)

    # The combined kernel database is only useful once at least one member library was built;
    # --combined-only asks for it explicitly (e.g. over a member copied into output_dir). It is built
    # last, over the folder as the steps above left it: a member that failed to build is not in it.
    members_present = any(
        (output_dir / name).is_file()
        for name in ("libgdn_fused_gaudi2.so", "libhc_fused_gaudi2.so", "libqsa_fused_gaudi2.so",
                     "libflashinfer_gaudi_kernels.so"))
    if args.combined_only or members_present:
        try:
            build_combined_kernel_db(repository_root, build_root, output_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(
                f"Combined kernel database build failed (the member libraries are unaffected): {exc}",
                file=sys.stderr,
            )
            _remove_combined_kernel_db_output(output_dir)
            if args.combined_only:
                sys.exit(3)
            print(
                "Continuing without the combined kernel database: GC_KERNEL_PATH stays a colon list.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
