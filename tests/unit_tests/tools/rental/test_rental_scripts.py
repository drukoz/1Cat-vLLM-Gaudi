# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental harness shell scripts (RUK-2709 M4-H8).

`bash -n` on the three scripts, the `run_rental.sh --dry-run` docker line (Habana runtime and the
three mounts), and that `make_rental_bundle.sh --allow-dirty` builds a tarball with run_rental.sh
at the top level, the plugin tree, vLLM at the pin and BUNDLE.txt. The bundle test is skipped when
the vLLM checkout is absent. Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SCRIPTS = ["make_rental_bundle.sh", "run_rental.sh", "container_entry.sh"]
VLLM_SRC = Path("/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/vllm-src")


@pytest.mark.parametrize("name", SCRIPTS)
def test_bash_syntax(name):
    script = SCRIPTS_DIR / name
    assert script.is_file(), script
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_vllm_install_pins_version():
    # vllm/ in the bundle has no .git: without the override setuptools_scm fails the install.
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert "VLLM_VERSION_OVERRIDE=0.28.1rc1.dev392+g8f816a3f6" in text


def test_run_rental_dry_run(tmp_path):
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "run_rental.sh"), "--dry-run", "--model", "/tmp/x"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "docker run" in out
    assert "--runtime=habana" in out
    assert ":/bundle" in out
    assert ":/models" in out
    assert ":/results" in out
    assert "container_entry.sh" in out


@pytest.mark.skipif(not (VLLM_SRC / ".git").is_dir(), reason="vLLM checkout not present")
def test_make_rental_bundle(tmp_path):
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "make_rental_bundle.sh"), "--allow-dirty", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    tarballs = sorted(tmp_path.glob("flash-next-rental-*.tar.gz"))
    assert len(tarballs) == 1, proc.stdout
    with tarfile.open(tarballs[0], "r:gz") as tar:
        names = set(tar.getnames())
    assert "run_rental.sh" in names
    assert "BUNDLE.txt" in names
    assert "vllm/setup.py" in names
    assert any(name.startswith("plugin/tools/flash_next/rental/") for name in names)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
