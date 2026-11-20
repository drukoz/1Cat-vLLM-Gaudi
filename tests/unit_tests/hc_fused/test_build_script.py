# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the hyper-connection step of ``tools/build_flashinfer_gaudi.py``.

No toolchain: the build functions are stubs that write the files a real build leaves.  What is
pinned: the step runs only on request (``VLLM_GAUDI_BUILD_HC_FUSED=1`` or ``--hc-fused-only``),
after the fused GDN step and before the combined kernel database; a failed step removes its own two
files and nothing else, logs, and the install goes on; a built ``libhc_fused_gaudi2.so`` that the
combined database could not use is treated as a failed build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import tools.build_flashinfer_gaudi as bfg

REPO_ROOT = Path(__file__).resolve().parents[3]
OTHER_FILES = ("libflashinfer_gaudi_kernels.so", "flashinfer_gaudi_ops.cpython-310-x86_64-linux-gnu.so",
               "libgdn_fused_gaudi2.so", "gdn_fused_ops.cpython-310-x86_64-linux-gnu.so")
HC_KERNEL_DB = "libhc_fused_gaudi2.so"
HC_EXTENSION = "hc_fused_ops.cpython-310-x86_64-linux-gnu.so"
COMBINED = "libflashinfer_gaudi_combined_gaudi2.so"


@pytest.fixture
def steps(monkeypatch, tmp_path):
    """Every build step replaced by a stub that records its name and writes its output files."""
    out = tmp_path / "out"
    order: list[str] = []
    seen_by_combined: list[list[str]] = []
    state = {"hc": None}  # an exception the hc step raises after writing its files, or None

    def kernel_db(source_root, build_root, output_dir):
        order.append("kernel_db")
        (output_dir / OTHER_FILES[0]).write_bytes(b"k")

    def pytorch(source_root, build_root, output_dir):
        order.append("pytorch")
        (output_dir / OTHER_FILES[1]).write_bytes(b"p")

    def gdn(repository_root, build_root, output_dir):
        order.append("gdn")
        (output_dir / OTHER_FILES[2]).write_bytes(b"g")
        (output_dir / OTHER_FILES[3]).write_bytes(b"g")

    def hc(repository_root, build_root, output_dir):
        order.append("hc")
        assert repository_root == REPO_ROOT
        (output_dir / HC_KERNEL_DB).write_bytes(b"h")
        (output_dir / HC_EXTENSION).write_bytes(b"h")
        if state["hc"] is not None:
            raise state["hc"]
        return output_dir / HC_EXTENSION

    def combined(repository_root, build_root, output_dir):
        order.append("combined")
        seen_by_combined.append(sorted(path.name for path in output_dir.iterdir()))
        (output_dir / COMBINED).write_bytes(b"c")

    monkeypatch.setattr(bfg, "build_kernel_database", kernel_db)
    monkeypatch.setattr(bfg, "build_pytorch_extension", pytorch)
    monkeypatch.setattr(bfg, "build_gdn_fused", gdn)
    monkeypatch.setattr(bfg, "build_hc_fused", hc)
    monkeypatch.setattr(bfg, "build_combined_kernel_db", combined)
    monkeypatch.delenv("VLLM_GAUDI_BUILD_GDN_FUSED", raising=False)
    monkeypatch.delenv("VLLM_GAUDI_BUILD_HC_FUSED", raising=False)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["build_flashinfer_gaudi.py", "--output-dir", str(out), "--build-root",
                                          str(tmp_path / "build"), *args])
        bfg.main()

    return run, out, order, seen_by_combined, state


def _names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_the_step_does_not_run_by_default(steps, monkeypatch):
    run, out, order, _, _ = steps
    monkeypatch.setenv("VLLM_GAUDI_BUILD_GDN_FUSED", "1")

    run()

    assert order == ["kernel_db", "pytorch", "gdn", "combined"]
    assert HC_KERNEL_DB not in _names(out) and HC_EXTENSION not in _names(out)


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_the_value_1_asks_for_the_step(steps, monkeypatch, value):
    run, _, order, _, _ = steps
    monkeypatch.setenv("VLLM_GAUDI_BUILD_HC_FUSED", value)

    run()

    assert "hc" not in order


def test_the_step_runs_after_the_gdn_step_and_before_the_combined_database(steps, monkeypatch, capsys):
    run, out, order, seen_by_combined, _ = steps
    monkeypatch.setenv("VLLM_GAUDI_BUILD_GDN_FUSED", "1")
    monkeypatch.setenv("VLLM_GAUDI_BUILD_HC_FUSED", "1")

    run()

    assert order == ["kernel_db", "pytorch", "gdn", "hc", "combined"]
    assert HC_KERNEL_DB in seen_by_combined[0]  # the adapter is built over the folder that holds the member
    assert _names(out) == sorted((*OTHER_FILES, HC_KERNEL_DB, HC_EXTENSION, COMBINED))
    assert "failed" not in capsys.readouterr().err


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(1, ["cmake"]),
    FileNotFoundError("CMake did not produce libhc_fused_gaudi2.so."),
    RuntimeError("libhc_fused_gaudi2.so does not load: undefined symbol"),
    PermissionError("build folder"),
],
                         ids=["cmake", "missing_output", "unusable_library", "os_error"])
def test_a_failed_step_removes_its_own_files_only_and_the_install_goes_on(steps, monkeypatch, capsys, error):
    run, out, order, seen_by_combined, state = steps
    monkeypatch.setenv("VLLM_GAUDI_BUILD_GDN_FUSED", "1")
    monkeypatch.setenv("VLLM_GAUDI_BUILD_HC_FUSED", "1")
    state["hc"] = error

    run()  # no exception, no exit: pip install goes on

    assert order == ["kernel_db", "pytorch", "gdn", "hc", "combined"]
    assert _names(out) == sorted((*OTHER_FILES, COMBINED))  # every other library is still there
    for name in OTHER_FILES:
        assert (out / name).read_bytes() != b""  # and untouched
    # the combined database was built after the cleanup: it never sees the half-built member
    assert HC_KERNEL_DB not in seen_by_combined[0]
    stderr = capsys.readouterr().err
    assert "HC fused build failed (the other libraries are unaffected)" in stderr
    assert "Continuing without libhc_fused_gaudi2.so: the hyper-connections will run the torch path." in stderr
    assert "GDN fused build failed" not in stderr


def test_a_failed_step_removes_a_library_of_an_earlier_build_too(steps, monkeypatch):
    run, out, _, _, state = steps
    out.mkdir(parents=True)
    (out / "hc_fused_ops.cpython-39-x86_64-linux-gnu.so").write_bytes(b"old")
    monkeypatch.setenv("VLLM_GAUDI_BUILD_HC_FUSED", "1")
    state["hc"] = subprocess.CalledProcessError(2, ["tpc-clang"])

    run()

    assert [name for name in _names(out) if "hc_fused" in name] == []


def test_hc_fused_only_builds_the_step_and_the_combined_database(steps):
    run, out, order, _, _ = steps

    run("--hc-fused-only")

    assert order == ["hc", "combined"]
    assert _names(out) == sorted((HC_KERNEL_DB, HC_EXTENSION, COMBINED))


def test_hc_fused_only_failure_exits_3(steps, capsys):
    run, out, order, _, state = steps
    state["hc"] = subprocess.CalledProcessError(1, ["cmake"])

    with pytest.raises(SystemExit) as excinfo:
        run("--hc-fused-only")

    assert excinfo.value.code == 3
    assert order == ["hc"]
    assert _names(out) == []
    stderr = capsys.readouterr().err
    assert "HC fused build failed" in stderr and "Continuing without" not in stderr


@pytest.mark.parametrize("other", ["--kernel-only", "--extension-only", "--gdn-fused-only", "--combined-only"])
def test_hc_fused_only_excludes_the_other_single_step_switches(steps, capsys, other):
    run, _, order, _, _ = steps

    with pytest.raises(SystemExit) as excinfo:
        run("--hc-fused-only", other)

    assert excinfo.value.code == 2
    assert order == []
    assert "mutually exclusive" in capsys.readouterr().err


def test_a_failed_gdn_step_does_not_stop_the_hc_step(steps, monkeypatch, capsys):
    run, out, order, _, _ = steps

    def broken_gdn(repository_root, build_root, output_dir):
        order.append("gdn")
        raise subprocess.CalledProcessError(1, ["cmake"])

    monkeypatch.setattr(bfg, "build_gdn_fused", broken_gdn)
    monkeypatch.setenv("VLLM_GAUDI_BUILD_GDN_FUSED", "1")
    monkeypatch.setenv("VLLM_GAUDI_BUILD_HC_FUSED", "1")

    run()

    assert order == ["kernel_db", "pytorch", "gdn", "hc", "combined"]
    assert HC_KERNEL_DB in _names(out) and HC_EXTENSION in _names(out)


def test_remove_outputs_names_exactly_the_two_files(tmp_path):
    for name in (*OTHER_FILES, HC_KERNEL_DB, HC_EXTENSION, COMBINED):
        (tmp_path / name).write_bytes(b"x")

    bfg._remove_hc_fused_outputs(tmp_path)

    assert _names(tmp_path) == sorted((*OTHER_FILES, COMBINED))
    bfg._remove_hc_fused_outputs(tmp_path)  # nothing left to remove is not an error


def test_build_hc_fused_runs_cmake_then_the_extension_then_the_check(monkeypatch, tmp_path):
    """The commands of build_gdn_fused with the hyper-connection names, and the library check at the end."""
    out = tmp_path / "out"
    out.mkdir()
    build = tmp_path / "build"
    commands: list[tuple[list[str], Path]] = []
    checked: list[Path] = []

    def fake_run(command, cwd):
        commands.append((command, cwd))
        if command[:2] == ["cmake", "--build"]:
            folder = Path(command[2])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / HC_KERNEL_DB).write_bytes(b"elf")
        if "build_ext" in command:
            (Path(command[command.index("--build-lib") + 1]) / HC_EXTENSION).write_bytes(b"elf")

    monkeypatch.setattr(bfg, "_run", fake_run)
    monkeypatch.setattr(bfg, "check_kernel_library", lambda path: checked.append(path) or 6)

    extension = bfg.build_hc_fused(REPO_ROOT, build, out)

    source = REPO_ROOT / "csrc" / "hc_fused"
    assert extension == out / HC_EXTENSION
    assert commands[0] == (["cmake", "-S", str(source), "-B", str(build / "hc_fused_kernel_db"),
                            "-DCMAKE_BUILD_TYPE=Release"], source)
    assert commands[1] == (["cmake", "--build", str(build / "hc_fused_kernel_db"), "--target", "hc_fused_gaudi2",
                            "--parallel"], source)
    assert commands[2] == ([sys.executable, "setup.py", "build_ext", "--build-lib", str(out), "--build-temp",
                            str(build / "hc_fused_pytorch")], source / "pytorch")
    assert (out / HC_KERNEL_DB).read_bytes() == b"elf"
    assert checked == [out / HC_KERNEL_DB]


def test_build_hc_fused_refuses_a_missing_or_doubled_extension(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    def fake_run(command, cwd):
        if command[:2] == ["cmake", "--build"]:
            folder = Path(command[2])
            folder.mkdir(parents=True, exist_ok=True)
            (folder / HC_KERNEL_DB).write_bytes(b"elf")

    monkeypatch.setattr(bfg, "_run", fake_run)
    monkeypatch.setattr(bfg, "check_kernel_library", lambda path: 6)

    with pytest.raises(FileNotFoundError, match="hc_fused_ops"):
        bfg.build_hc_fused(REPO_ROOT, tmp_path / "build", out)


def test_build_hc_fused_refuses_a_missing_kernel_database(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(bfg, "_run", lambda command, cwd: None)

    with pytest.raises(FileNotFoundError, match="libhc_fused_gaudi2.so"):
        bfg.build_hc_fused(REPO_ROOT, tmp_path / "build", out)


def test_check_kernel_library_refuses_a_file_that_does_not_load(tmp_path):
    broken = tmp_path / HC_KERNEL_DB
    broken.write_text("not a shared object\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not load"):
        bfg.check_kernel_library(broken)


def test_the_sources_the_step_builds_are_in_the_tree():
    assert (REPO_ROOT / "csrc" / "hc_fused" / "CMakeLists.txt").is_file()
    assert (REPO_ROOT / "csrc" / "hc_fused" / "pytorch" / "setup.py").is_file()
    assert "hc_fused_gaudi2" in (REPO_ROOT / "csrc" / "hc_fused" / "CMakeLists.txt").read_text(encoding="utf-8")
