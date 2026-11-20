# SPDX-License-Identifier: Apache-2.0
"""Tests for container_entry.sh, run on the Mac against a stand-in bundle folder.

`FN_RENTAL_BUNDLE_ROOT` points the script at a temporary bundle (BUNDLE.txt, plugin/, vllm/) instead
of /bundle, and stand-in `pip`, `python`, `tpc-clang` and `dpkg-query` on PATH record what the script
runs: the install steps, the stale-build check, `pip freeze`, and the orchestrator command line.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "tools" / "flash_next" / "rental" / "container_entry.sh"

SHA_A = "a" * 40
SHA_B = "b" * 40

FAKE_PIP = r"""
echo "$*" >> "$FAKE_PIP_LOG"
if [ "$1" = "freeze" ]; then
  if [ -n "${FAKE_FREEZE_FAILS:-}" ]; then exit 1; fi
  echo "pytest==9.2.0"
  echo "vllm==0.28.1rc1.dev392+g8f816a3f6"
  exit 0
fi
case " $* " in
  *" $FAKE_BUNDLE/plugin "*)
    # the plugin install: say what the build folder held, then "build" into it
    if [ -e "$FAKE_BUNDLE/plugin/build/old.o" ]; then echo "BUILD-SAW-OLD" >> "$FAKE_PIP_LOG"; fi
    if [ -n "${FAKE_PLUGIN_INSTALL_FAILS:-}" ]; then echo "boom"; exit 1; fi
    mkdir -p "$FAKE_BUNDLE/plugin/build"
    : > "$FAKE_BUNDLE/plugin/build/new.o" ;;
esac
exit 0
"""

FAKE_PYTHON = r"""
echo "cwd=$(pwd)" >> "$FAKE_PY_LOG"
echo "$*" >> "$FAKE_PY_LOG"
echo "FAKE ORCHESTRATOR"
exit "${FAKE_PY_EXIT:-0}"
"""


def _bundle(tmp_path, sha=SHA_A):
    bundle = tmp_path / "bundle"
    (bundle / "plugin").mkdir(parents=True, exist_ok=True)
    (bundle / "vllm").mkdir(exist_ok=True)
    if sha is not None:
        (bundle / "BUNDLE.txt").write_text("plugin sha: %s\nbranch: x\nvllm commit: y\nbuild date: z\n" % sha)
    elif (bundle / "BUNDLE.txt").exists():
        (bundle / "BUNDLE.txt").unlink()
    return bundle


def _run(tmp_path, *args, sha=SHA_A, **fake):
    bundle = _bundle(tmp_path, sha)
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    fakes = {"pip": FAKE_PIP, "python": FAKE_PYTHON, "tpc-clang": 'echo "TPC C/C++ compiler version 2.7.33"\n',
             "dpkg-query": 'printf "1.24.1-482"\n'}
    for name, body in fakes.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    env["FN_RENTAL_BUNDLE_ROOT"] = str(bundle)
    env["FAKE_BUNDLE"] = str(bundle)
    env["FAKE_PIP_LOG"] = str(tmp_path / "pip.log")
    env["FAKE_PY_LOG"] = str(tmp_path / "python.log")
    env.update({k: str(v) for k, v in fake.items()})
    return subprocess.run(["/bin/bash", str(SCRIPT), "--results", str(tmp_path / "results"), "--run-id", "r1", *args],
                          capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)


def _lines(path):
    return path.read_text().splitlines() if path.exists() else []


# --------------------------------------------------------------------------- item 1: pass-through


def test_new_options_reach_the_orchestrator(tmp_path):
    proc = _run(tmp_path, "--skip-install", "--model", "/models/X", "--quick-gates", "--no-verify-probe",
                "--load-read-timeout", "900", "--warm-request-timeout", "1200", "--from", "tiny")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cwd, argv = _lines(tmp_path / "python.log")
    assert cwd == "cwd=%s" % (tmp_path / "bundle" / "plugin").resolve()
    assert argv.startswith("-m tools.flash_next.rental.main --model /models/X --results ")
    for part in ("--quick-gates", "--no-verify-probe", "--load-read-timeout 900", "--warm-request-timeout 1200",
                 "--from tiny", "--run-id r1"):
        assert part in argv, part


def test_new_options_are_not_passed_by_default(tmp_path):
    proc = _run(tmp_path, "--skip-install", "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    argv = _lines(tmp_path / "python.log")[1]
    for part in ("--quick-gates", "--no-verify-probe", "--load-read-timeout", "--warm-request-timeout"):
        assert part not in argv, part


def test_the_orchestrator_exit_code_is_the_scripts(tmp_path):
    proc = _run(tmp_path, "--skip-install", "--model", "/models/X", FAKE_PY_EXIT=7)
    assert proc.returncode == 7


@pytest.mark.parametrize("option", ["--load-read-timeout", "--warm-request-timeout", "--server-timeout", "--from",
                                    "--model"])
def test_an_option_without_its_value_is_a_usage_error(tmp_path, option):
    proc = _run(tmp_path, "--skip-install", "--model", "/models/X", option)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "%s needs a value" % option in proc.stderr
    assert not (tmp_path / "python.log").exists()


def test_an_unknown_option_is_still_an_error(tmp_path):
    proc = _run(tmp_path, "--skip-install", "--model", "/models/X", "--quick-gate")
    assert proc.returncode == 2
    assert "unknown option: --quick-gate" in proc.stderr
    assert not (tmp_path / "python.log").exists()


def test_usage_names_the_new_options():
    proc = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    for part in ("--quick-gates", "--no-verify-probe", "--load-read-timeout S", "--warm-request-timeout S"):
        assert part in proc.stdout, part


# --------------------------------------------------------------------------- item 6: the install


def test_the_install_pins_pytest_and_records_pip_freeze(tmp_path):
    proc = _run(tmp_path, "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pip = _lines(tmp_path / "pip.log")
    assert "install pytest==9.2.0" in pip  # the version the 2026-11-18 rental resolved (install-3b.log)
    assert not [line for line in pip if line.split()[-1] == "pytest"]  # never unpinned
    assert pip.index("install pytest==9.2.0") < pip.index("freeze")  # freeze comes after every install step
    freeze = (tmp_path / "results" / "r1" / "pip-freeze.txt").read_text()
    assert "pytest==9.2.0" in freeze and "vllm==0.28.1rc1.dev392+g8f816a3f6" in freeze
    assert _lines(tmp_path / "python.log")  # and the orchestrator started


def test_install_only_also_records_pip_freeze(tmp_path):
    proc = _run(tmp_path, "--install-only")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "install only: done" in proc.stdout
    assert (tmp_path / "results" / "r1" / "pip-freeze.txt").is_file()
    assert not (tmp_path / "python.log").exists()


def test_a_failing_pip_freeze_does_not_stop_the_run(tmp_path):
    proc = _run(tmp_path, "--model", "/models/X", FAKE_FREEZE_FAILS=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _lines(tmp_path / "python.log")


def test_every_phase_line_carries_a_utc_timestamp(tmp_path):
    proc = _run(tmp_path, "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    phases = [line for line in proc.stdout.splitlines() if line.startswith("phase: ")]
    names = " | ".join(phases)
    for step in ("toolchain", "install 0b", "install 1 ", "install 2 ", "install 3 ", "install 3b", "orchestrator"):
        assert "phase: %s" % step in names, step
    stamp = re.compile(r" \[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\]$")
    for line in phases:
        assert stamp.search(line), line


# --------------------------------------------------------------------------- item 9: the stale native build


def _old_build(tmp_path, recorded_sha=None):
    build = tmp_path / "bundle" / "plugin" / "build"
    build.mkdir(parents=True)
    (build / "old.o").write_text("stale object")
    if recorded_sha is not None:
        (build / ".bundle-sha").write_text(recorded_sha + "\n")
    return build


def test_a_build_folder_of_an_unrecorded_bundle_is_removed_before_the_plugin_install(tmp_path):
    # plugin/build survives `tar xzf` of a new bundle; its objects can be newer than the new sources.
    build = _old_build(tmp_path)
    proc = _run(tmp_path, "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUILD-SAW-OLD" not in _lines(tmp_path / "pip.log")
    assert not (build / "old.o").exists()
    assert (build / "new.o").exists()
    assert (build / ".bundle-sha").read_text().strip() == SHA_A
    assert "plugin/build" in proc.stdout  # the console says that it was removed


def test_a_build_folder_of_another_bundle_is_removed(tmp_path):
    build = _old_build(tmp_path, recorded_sha=SHA_B)
    proc = _run(tmp_path, "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUILD-SAW-OLD" not in _lines(tmp_path / "pip.log")
    assert not (build / "old.o").exists()
    assert (build / ".bundle-sha").read_text().strip() == SHA_A


def test_a_build_folder_of_this_bundle_is_reused(tmp_path):
    build = _old_build(tmp_path, recorded_sha=SHA_A)
    proc = _run(tmp_path, "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUILD-SAW-OLD" in _lines(tmp_path / "pip.log")  # the 16 s rebuild of the second start stays
    assert (build / "old.o").exists()
    assert (build / ".bundle-sha").read_text().strip() == SHA_A


def test_without_a_bundle_sha_the_build_folder_is_never_trusted(tmp_path):
    build = _old_build(tmp_path, recorded_sha=SHA_A)
    proc = _run(tmp_path, "--model", "/models/X", sha=None)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUILD-SAW-OLD" not in _lines(tmp_path / "pip.log")
    assert not (build / "old.o").exists()
    assert not (build / ".bundle-sha").exists()  # nothing to record


def test_a_failed_plugin_install_leaves_no_record(tmp_path):
    build = _old_build(tmp_path, recorded_sha=SHA_B)
    proc = _run(tmp_path, "--model", "/models/X", FAKE_PLUGIN_INSTALL_FAILS=1)
    assert proc.returncode == 1
    assert "install 2" in proc.stderr
    assert not (build / ".bundle-sha").exists()  # the next start removes the half-made build again
    assert not (tmp_path / "python.log").exists()


def test_skip_install_leaves_the_build_folder_alone(tmp_path):
    build = _old_build(tmp_path, recorded_sha=SHA_B)
    proc = _run(tmp_path, "--skip-install", "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (build / "old.o").exists()
    assert (build / ".bundle-sha").read_text().strip() == SHA_B
