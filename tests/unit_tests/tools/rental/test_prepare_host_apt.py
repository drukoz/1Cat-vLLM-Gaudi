# SPDX-License-Identifier: Apache-2.0
"""Tests that prepare_host.sh runs `apt-get update` at most once per run (RUK-3238).

The rental node makes apt-get update very slow, so the script must call it only right before a step
installs something, skip it entirely on a rerun where every prerequisite is installed, and refresh
only the Habana repo list when step 6 adds it. Checks the script text and the `--dry-run` output.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "tools" / "flash_next" / "rental" / "prepare_host.sh"


def _text():
    return SCRIPT.read_text()


def _dry_run():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        cwd="/tmp",
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_bare_apt_update_runs_only_inside_apt_update_once():
    text = _text()
    assert text.count('run_logged "apt-get update"') == 1
    # the single call lives in the once-per-run helper, not in a step
    helper = text.split("apt_update_once() {", 1)[1].split("}", 1)[0]
    assert 'run_logged "apt-get update"' in helper


def test_helpers_defined_and_called_by_the_steps():
    text = _text()
    assert "APT_UPDATED=0" in text
    assert text.count("apt_update_once") >= 4  # the definition plus steps 2, 5 and 6
    assert text.count("APT_UPDATED=1") == 2  # apt_update_once and the driver installer


def test_step2_skips_apt_when_every_prerequisite_is_installed():
    text = _text()
    assert "install ok installed" in text  # pkg_installed asks dpkg for a fully installed package
    assert "all prerequisites installed; no apt-get update" in text
    step2 = text.split("step2_apt() {", 1)[1].split("\n}\n", 1)[0]
    assert "apt_update_once" in step2
    assert 'run_logged "apt-get update"' not in step2
    assert "${#missing[@]}" in step2


def test_step3_marks_the_update_done_after_the_installer():
    text = _text()
    step3 = text.split("step3_driver() {", 1)[1].split("\n}\n", 1)[0]
    assert "APT_UPDATED=1" in step3
    assert step3.index("APT_UPDATED=1") < step3.index("step_ok")


def test_step5_updates_once_before_docker():
    text = _text()
    step5 = text.split("step5_docker() {", 1)[1].split("\n}\n", 1)[0]
    assert step5.index("apt_update_once") < step5.index("apt-get install -y docker.io")


def test_step6_refreshes_only_the_habana_list_it_added():
    text = _text()
    step6 = text.split("step6_runtime() {", 1)[1].split("\n}\n", 1)[0]
    assert "local added_list=0" in step6
    assert 'apt_update_list "$REPO_LIST"' in step6
    assert "apt_update_once" in step6
    assert 'run_logged "apt-get update"' not in step6
    assert step6.index("added_list=1") < step6.index('apt_update_list "$REPO_LIST"')


def test_dry_run_says_when_each_update_runs():
    out = _dry_run()
    assert "Dir::Etc::sourcelist=/etc/apt/sources.list.d/habanalabs.list" in out
    assert "at most once per run" in out
    assert "only when no earlier step ran it" in out
