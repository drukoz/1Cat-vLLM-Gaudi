# SPDX-License-Identifier: Apache-2.0
"""The rental scripts and the MoE glue kernels: `run_rental.sh --moe-glue` reaches the orchestrator
through `container_entry.sh`, nothing new is built (the kernels are in the fused hyper-connection
library, which the install already builds), `tiny_smoke.sh --gdn moe` reaches the check, and the run
sheet says all of it, with one more one-smoke-run experiment. Off by default everywhere.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import shlex
import subprocess
from pathlib import Path

from tools.flash_next.rental import main as rental_main

from . import test_sh_container_entry as entry_tests
from .test_sh_hc_fused import FAKE_PIP_WITH_BUILD_FLAGS
from .test_sh_run_rental import _dry_run as run_rental_dry_run
from .test_sh_run_rental import _flat

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SHEET = REPO_ROOT / "docs" / "design" / "flash-next" / "run-sheets" / "rental.md"
EXPERIMENTS = "## Experiments that cost one smoke run each"


def _usage(script: str) -> str:
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / script), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _experiments(sheet: str) -> str:
    section = sheet[sheet.index(EXPERIMENTS):]
    return section[:section.index("\n## ", 1)] if "\n## " in section[1:] else section


# --------------------------------------------------------------------------- container_entry.sh


def test_entry_passes_the_option_to_the_orchestrator(tmp_path):
    proc = entry_tests._run(tmp_path, "--skip-install", "--model", "/models/X", "--moe-glue", "--quick-gates")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    argv = entry_tests._lines(tmp_path / "python.log")[1]
    assert argv.startswith("-m tools.flash_next.rental.main --model /models/X --results ")
    assert "--moe-glue" in argv.split() and "--quick-gates" in argv.split()
    assert "--hc-fused" not in argv.split()  # each switch is its own


def test_entry_passes_both_switches(tmp_path):
    proc = entry_tests._run(tmp_path, "--skip-install", "--model", "/models/X", "--moe-glue", "--hc-fused")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    words = entry_tests._lines(tmp_path / "python.log")[1].split()
    assert "--moe-glue" in words and "--hc-fused" in words
    # what the entry hands over is what the orchestrator takes
    args = rental_main._parse_args(words[words.index("--model"):])
    assert args.moe_glue is True and args.hc_fused is True


def test_entry_does_not_pass_the_option_by_default(tmp_path):
    proc = entry_tests._run(tmp_path, "--skip-install", "--model", "/models/X")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--moe-glue" not in entry_tests._lines(tmp_path / "python.log")[1]
    proc = entry_tests._run(tmp_path, "--skip-install", "--model", "/models/X", "--hc-fused")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--moe-glue" not in entry_tests._lines(tmp_path / "python.log")[-1]  # not implied by the other switch


def test_entry_usage_names_the_option():
    usage = _usage("container_entry.sh")
    assert "[--moe-glue]" in usage and "[--hc-fused]" in usage
    assert "--moe-glue are handed to the orchestrator unchanged" in " ".join(usage.split())


def test_the_install_builds_nothing_new_for_the_kernels(tmp_path, monkeypatch):
    """The MoE glue kernels are in the fused hyper-connection library: the three build flags of the
    install, with or without the switch, and no build flag of their own."""
    monkeypatch.setattr(entry_tests, "FAKE_PIP", FAKE_PIP_WITH_BUILD_FLAGS)
    proc = entry_tests._run(tmp_path, "--install-only")  # the fake PATH has a tpc-clang: NATIVE=1
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUILD flashinfer=1 gdn=1 hc=1" in entry_tests._lines(tmp_path / "pip.log")
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert 'VLLM_GAUDI_BUILD_HC_FUSED="$NATIVE"' in text
    assert "VLLM_GAUDI_BUILD_MOE" not in text
    # The servers' flag is the orchestrator's decision: the script never sets it.
    assert "VLLM_HPU_MOE_GLUE" not in text


# --------------------------------------------------------------------------- run_rental.sh


def test_run_rental_passes_the_option_to_the_entry(tmp_path):
    proc = run_rental_dry_run(tmp_path, "--moe-glue")
    assert proc.returncode == 0, proc.stderr
    flat = _flat(proc)
    assert flat.index("--moe-glue") > flat.index("container_entry.sh")  # an entry argument, not a docker flag
    assert "VLLM_HPU_MOE_GLUE" not in flat  # no docker -e: the orchestrator sets the flag per server
    assert "--hc-fused" not in flat


def test_run_rental_does_not_pass_the_option_by_default(tmp_path):
    proc = run_rental_dry_run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "moe-glue" not in proc.stdout and "MOE_GLUE" not in proc.stdout
    proc = run_rental_dry_run(tmp_path, "--hc-fused")
    assert proc.returncode == 0, proc.stderr
    assert "moe-glue" not in proc.stdout and "MOE_GLUE" not in proc.stdout


def test_run_rental_passes_both_switches_in_either_order(tmp_path):
    for order in (("--hc-fused", "--moe-glue"), ("--moe-glue", "--hc-fused")):
        proc = run_rental_dry_run(tmp_path, *order, "--variants", "A")
        assert proc.returncode == 0, proc.stderr
        entry = _flat(proc)[_flat(proc).index("container_entry.sh"):].split()
        assert "--hc-fused" in entry and "--moe-glue" in entry
        assert entry[entry.index("--variants") + 1] == "A"  # the switch takes no value of its own


def test_no_warm_pass_stays_the_last_entry_argument(tmp_path):
    proc = run_rental_dry_run(tmp_path, "--no-warm-pass", "--moe-glue", "--hc-fused")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("--no-warm-pass")
    assert "--moe-glue" in proc.stdout and "--hc-fused" in proc.stdout


def test_run_rental_usage_says_what_the_option_does():
    usage = _usage("run_rental.sh")
    assert "  --moe-glue " in usage
    text = " ".join(usage[usage.index("  --moe-glue "):].split())
    text = text[:text.index("--load-read-timeout")]  # the option's own lines
    assert "MoE glue kernels" in text and "off by default" in text and "passed to the orchestrator" in text
    assert "VLLM_HPU_MOE_GLUE=1" in text and "VLLM_HPU_MOE_GLUE=0" in text and "card test" in text
    assert "runs once more without" in text  # the tiny gate's retry


# --------------------------------------------------------------------------- tiny_smoke.sh


def _smoke_dry_run(tmp_path, *extra):
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / "tiny_smoke.sh"), "--dry-run", "--results",
                           str(tmp_path / "results"), "--run-id", "t1", *extra],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return [shlex.split(ln) for ln in proc.stdout.splitlines() if ln.startswith("docker ")][2]


def test_tiny_smoke_passes_the_moe_mode_to_the_check(tmp_path):
    smoke = _smoke_dry_run(tmp_path, "--gdn", "moe")
    assert smoke[smoke.index("--gdn") + 1] == "moe"
    smoke = _smoke_dry_run(tmp_path, "--gdn", "pytorch,fused,hc,moe")
    assert smoke[smoke.index("--gdn") + 1] == "pytorch,fused,hc,moe"
    assert not any("VLLM_HPU_MOE_GLUE" in word for word in smoke)  # the mode sets it, in the check


def test_tiny_smoke_default_modes_are_unchanged(tmp_path):
    smoke = _smoke_dry_run(tmp_path)
    assert smoke[smoke.index("--gdn") + 1] == "pytorch,fused"


def test_tiny_smoke_usage_names_the_moe_mode():
    usage = _usage("tiny_smoke.sh")
    line = [ln for ln in usage.splitlines() if ln.strip().startswith("--gdn LIST")][0]
    assert "default: pytorch,fused" in line and "hc" in line
    assert "moe" in line and "VLLM_HPU_MOE_GLUE=1" in line and "--gdn moe" in line


def test_tiny_smoke_header_says_what_the_mode_is():
    text = (SCRIPTS_DIR / "tiny_smoke.sh").read_text()
    header = " ".join(line.lstrip("# ") for line in text[:text.index("set -euo pipefail")].splitlines())
    assert "moe, is the fused mode plus the MoE glue kernels (VLLM_HPU_MOE_GLUE=1)" in header
    assert "--gdn moe" in header and "--gdn pytorch,fused,hc,moe" in header
    assert "A third mode, hc, is the fused mode plus the fused hyper-connection kernels" in header  # as before


def test_the_check_takes_what_the_script_passes(tmp_path):
    """The mode names the script documents are the ones the check in the container accepts."""
    from tools.flash_next.rental import tiny_smoke

    smoke = _smoke_dry_run(tmp_path, "--gdn", "pytorch,fused,hc,moe")
    for mode in smoke[smoke.index("--gdn") + 1].split(","):
        assert mode in tiny_smoke.GDN_MODES, mode


# --------------------------------------------------------------------------- the run sheet


def test_the_sheet_has_rows_for_the_option_and_the_mode():
    sheet = SHEET.read_text(encoding="utf-8")
    assert "| `--moe-glue` |" in sheet
    row = [line for line in sheet.splitlines() if line.startswith("| `--moe-glue` |")][0]
    assert "`VLLM_HPU_MOE_GLUE=1`" in row and "`VLLM_HPU_MOE_GLUE=0`" in row and "`moe_hpu_op.log`" in row
    assert "`tests/unit_tests/hc_fused/test_moe_hpu_op.py`" in row and "`tiny_smoke.sh --gdn moe`" in row
    lines = sheet.splitlines()
    assert lines.index(row) == lines.index([line for line in lines if line.startswith("| `--hc-fused` |")][0]) + 1
    assert "| `--gdn LIST` |" in sheet
    gdn_row = [line for line in sheet.splitlines() if line.startswith("| `--gdn LIST` |")][0]
    assert "`moe`" in gdn_row and "`hc`" in gdn_row and "`pytorch,fused`" in gdn_row
    mode_row = [line for line in sheet.splitlines() if line.startswith("| `moe` (only when named: `--gdn moe`) |")]
    assert len(mode_row) == 1 and "`VLLM_HPU_MOE_GLUE=1`" in mode_row[0]
    assert "MoE glue: on" in sheet and "MoE glue: off (not requested)" in sheet
    assert "attempt-moe-glue" in sheet and "MoE glue fallback lines in <variant>" in sheet


def test_the_sheet_says_the_tiny_gate_retries_once_for_both():
    sheet = " ".join(SHEET.read_text(encoding="utf-8").split())
    assert "once more without both" in sheet
    assert "`tiny/attempt-moe-glue/`" in sheet and "`tiny/attempt-hc-fused/`" in sheet


def test_the_sheet_has_the_moe_glue_experiment():
    sheet = SHEET.read_text(encoding="utf-8")
    section = _experiments(sheet)
    earlier = ["sudo bash tiny_smoke.sh --gdn hc",
               "sudo bash tiny_smoke.sh --gdn fused --env VLLM_HPU_COLLECTIVE_MARK_STEP=0",
               "sudo bash tiny_smoke.sh --gdn pytorch --env VLLM_HPU_FLASH_NEXT_QSA_MODE=0"]
    command = "sudo bash tiny_smoke.sh --gdn moe"
    assert section.count(command) == 1  # one entry for it
    assert section.index(command) > max(section.index(text) for text in earlier)  # after the three there were
    assert len(section.splitlines()) <= 40  # still a short section
    entry = [line for line in section.splitlines() if command in line][0]
    assert entry.startswith("4. **")
    # what it checks, and what to look for
    assert "`VLLM_HPU_MOE_GLUE=1`" in entry and "tokens match" in entry
    assert "`MoE glue ... is off for`" in entry and "`has no HPU kernel`" in entry
    assert "`--moe-glue`" in entry
    # what it cannot show: the tiny checkpoint is BF16
    assert "BF16" in entry and "FP8 row-quantization kernel" in entry and "card test" in entry
