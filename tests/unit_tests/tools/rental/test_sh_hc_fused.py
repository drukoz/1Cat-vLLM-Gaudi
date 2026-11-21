# SPDX-License-Identifier: Apache-2.0
"""The rental scripts and the fused hyper-connection kernels: `run_rental.sh --hc-fused` reaches the
orchestrator through `container_entry.sh`, the install builds the kernels with the same toolchain check
as the other native libraries, `tiny_smoke.sh --gdn hc` reaches the check, and the run sheet says all of
it, with the three one-smoke-run experiments. Off by default everywhere.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import shlex
import subprocess
from pathlib import Path

from . import test_sh_container_entry as entry_tests
from .test_sh_run_rental import _dry_run as run_rental_dry_run
from .test_sh_run_rental import _flat

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SHEET = REPO_ROOT / "docs" / "design" / "flash-next" / "run-sheets" / "rental.md"

# A pip that also says which native builds the plugin install was asked for.
FAKE_PIP_WITH_BUILD_FLAGS = r"""
case " $* " in
  *" $FAKE_BUNDLE/plugin "*)
    echo "BUILD flashinfer=${VLLM_GAUDI_BUILD_FLASHINFER-unset} gdn=${VLLM_GAUDI_BUILD_GDN_FUSED-unset}" \
         "hc=${VLLM_GAUDI_BUILD_HC_FUSED-unset}" >> "$FAKE_PIP_LOG" ;;
esac
""" + entry_tests.FAKE_PIP


def _usage(script: str) -> str:
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / script), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# --------------------------------------------------------------------------- container_entry.sh


def test_entry_passes_the_option_to_the_orchestrator(tmp_path):
    proc = entry_tests._run(tmp_path, "--skip-install", "--model", "/models/X", "--hc-fused", "--quick-gates")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    argv = entry_tests._lines(tmp_path / "python.log")[1]
    assert argv.startswith("-m tools.flash_next.rental.main --model /models/X --results ")
    assert "--hc-fused" in argv.split() and "--quick-gates" in argv.split()


def test_entry_does_not_pass_the_option_by_default(tmp_path):
    proc = entry_tests._run(tmp_path, "--skip-install", "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--hc-fused" not in entry_tests._lines(tmp_path / "python.log")[1]


def test_entry_usage_names_the_option():
    assert "--hc-fused" in _usage("container_entry.sh")


def test_the_install_builds_the_kernels_with_the_toolchain_check_of_the_other_two(tmp_path, monkeypatch):
    monkeypatch.setattr(entry_tests, "FAKE_PIP", FAKE_PIP_WITH_BUILD_FLAGS)
    proc = entry_tests._run(tmp_path, "--install-only")  # the fake PATH has a tpc-clang: NATIVE=1
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUILD flashinfer=1 gdn=1 hc=1" in entry_tests._lines(tmp_path / "pip.log")


def test_one_value_drives_the_three_build_flags():
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    for flag in ("VLLM_GAUDI_BUILD_FLASHINFER", "VLLM_GAUDI_BUILD_GDN_FUSED", "VLLM_GAUDI_BUILD_HC_FUSED"):
        assert '%s="$NATIVE"' % flag in text, flag
    assert "VLLM_GAUDI_BUILD_HC_FUSED=1" not in text
    # The servers' flag is the orchestrator's decision: the script never sets it.
    assert "VLLM_HPU_HC_FUSED" not in text


# --------------------------------------------------------------------------- run_rental.sh


def test_run_rental_passes_the_option_to_the_entry(tmp_path):
    proc = run_rental_dry_run(tmp_path, "--hc-fused")
    assert proc.returncode == 0, proc.stderr
    flat = _flat(proc)
    assert flat.index("--hc-fused") > flat.index("container_entry.sh")  # an entry argument, not a docker flag
    assert "VLLM_HPU_HC_FUSED" not in flat  # no docker -e: the orchestrator sets the flag per server


def test_run_rental_does_not_pass_the_option_by_default(tmp_path):
    proc = run_rental_dry_run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "hc-fused" not in proc.stdout and "HC_FUSED" not in proc.stdout


def test_no_warm_pass_stays_the_last_entry_argument(tmp_path):
    proc = run_rental_dry_run(tmp_path, "--no-warm-pass", "--hc-fused")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("--no-warm-pass")
    assert "--hc-fused" in proc.stdout


def test_run_rental_usage_says_what_the_option_does():
    usage = _usage("run_rental.sh")
    assert "--hc-fused" in usage
    assert "VLLM_HPU_HC_FUSED=1" in usage and "card test" in usage


# --------------------------------------------------------------------------- tiny_smoke.sh


def _smoke_dry_run(tmp_path, *extra):
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / "tiny_smoke.sh"), "--dry-run", "--results",
                           str(tmp_path / "results"), "--run-id", "t1", *extra],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return [shlex.split(ln) for ln in proc.stdout.splitlines() if ln.startswith("docker ")][2]


def test_tiny_smoke_passes_the_hc_mode_to_the_check(tmp_path):
    smoke = _smoke_dry_run(tmp_path, "--gdn", "hc")
    assert smoke[smoke.index("--gdn") + 1] == "hc"
    smoke = _smoke_dry_run(tmp_path, "--gdn", "pytorch,fused,hc")
    assert smoke[smoke.index("--gdn") + 1] == "pytorch,fused,hc"
    assert not any("VLLM_HPU_HC_FUSED" in word for word in smoke)  # the mode sets it, in the check


def test_tiny_smoke_default_modes_are_unchanged(tmp_path):
    smoke = _smoke_dry_run(tmp_path)
    assert smoke[smoke.index("--gdn") + 1] == "pytorch,fused"


def test_tiny_smoke_usage_names_the_hc_mode():
    usage = _usage("tiny_smoke.sh")
    line = [ln for ln in usage.splitlines() if ln.strip().startswith("--gdn LIST")][0]
    assert "default: pytorch,fused" in line and "hc" in line


def test_the_experiments_pass_their_env_pairs_to_the_check(tmp_path):
    smoke = _smoke_dry_run(tmp_path, "--gdn", "fused", "--env", "VLLM_HPU_COLLECTIVE_MARK_STEP=0")
    assert smoke[smoke.index("--gdn") + 1] == "fused"
    assert smoke[smoke.index("--env") + 1] == "VLLM_HPU_COLLECTIVE_MARK_STEP=0"
    smoke = _smoke_dry_run(tmp_path, "--gdn", "pytorch", "--env", "VLLM_HPU_FLASH_NEXT_QSA_MODE=0")
    assert smoke[smoke.index("--env") + 1] == "VLLM_HPU_FLASH_NEXT_QSA_MODE=0"


# --------------------------------------------------------------------------- the run sheet


def test_the_sheet_has_rows_for_the_option_and_the_mode():
    sheet = SHEET.read_text(encoding="utf-8")
    assert "| `--hc-fused` |" in sheet
    row = [line for line in sheet.splitlines() if line.startswith("| `--hc-fused` |")][0]
    assert "`VLLM_HPU_HC_FUSED=1`" in row and "`VLLM_HPU_HC_FUSED=0`" in row and "`hc_hpu_op.log`" in row
    assert "| `--gdn LIST` |" in sheet
    gdn_row = [line for line in sheet.splitlines() if line.startswith("| `--gdn LIST` |")][0]
    assert "`hc`" in gdn_row and "`pytorch,fused`" in gdn_row
    assert "HC fused: on" in sheet and "off (not requested)" in sheet and "off (test_hpu_op.py failed)" in sheet
    assert "attempt-hc-fused" in sheet


def test_the_sheet_has_the_three_one_smoke_run_experiments():
    sheet = SHEET.read_text(encoding="utf-8")
    title = "## Experiments that cost one smoke run each"
    assert title in sheet
    section = sheet[sheet.index(title):]
    section = section[:section.index("\n## ", 1)] if "\n## " in section[1:] else section
    commands = ["sudo bash tiny_smoke.sh --gdn hc",
                "sudo bash tiny_smoke.sh --gdn fused --env VLLM_HPU_COLLECTIVE_MARK_STEP=0",
                "sudo bash tiny_smoke.sh --gdn pytorch --env VLLM_HPU_FLASH_NEXT_QSA_MODE=0"]
    positions = [section.index(command) for command in commands]
    assert positions == sorted(positions)
    assert len(section.splitlines()) <= 40  # a short section
    # what each one answers
    assert "hyper-connection kernels" in section
    assert "mark_step" in section and "ms/token" in section and "fb7b2d3" in section
    assert "dense attention" in section and "2,600" in section
