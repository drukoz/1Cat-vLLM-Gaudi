# SPDX-License-Identifier: Apache-2.0
"""Build the in-tree FlashInfer-Gaudi kernel database and PyTorch op."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--combined-only", action="store_true")
    args = parser.parse_args()
    if args.kernel_only and args.extension_only:
        parser.error("--kernel-only and --extension-only are mutually exclusive")
    if args.gdn_fused_only and (args.kernel_only or args.extension_only):
        parser.error("--gdn-fused-only is mutually exclusive with --kernel-only and --extension-only")
    if args.combined_only and (args.kernel_only or args.extension_only or args.gdn_fused_only):
        parser.error("--combined-only is mutually exclusive with --kernel-only, --extension-only and --gdn-fused-only")

    source_root = repository_root / "csrc" / "flashinfer_gaudi"
    output_dir = args.output_dir.resolve()
    build_root = args.build_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    if not (args.gdn_fused_only or args.combined_only):
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

    # The combined kernel database is only useful once at least one member library was built;
    # --combined-only asks for it explicitly (e.g. over a member copied into output_dir).
    members_present = (output_dir / "libgdn_fused_gaudi2.so").is_file() or (
        output_dir / "libflashinfer_gaudi_kernels.so"
    ).is_file()
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
