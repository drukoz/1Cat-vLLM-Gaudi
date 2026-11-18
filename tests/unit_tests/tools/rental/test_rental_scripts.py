# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental harness shell scripts (RUK-2709 M4-H8).

`bash -n` on the three scripts, the `run_rental.sh --dry-run` docker line (Habana runtime and the
three mounts), and that `make_rental_bundle.sh --allow-dirty` builds a tarball with run_rental.sh
at the top level, the plugin tree, vLLM at the pin and BUNDLE.txt. The bundle test is skipped when
the vLLM checkout is absent. Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SCRIPTS = ["make_rental_bundle.sh", "run_rental.sh", "container_entry.sh", "prepare_host.sh"]
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


def test_container_entry_toolchain():
    # The base image lacks tpc-clang: the script installs habanalabs-tools so the native kernels
    # build, drives both build flags from one NATIVE value, and records the toolchain (RUK-2709).
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert "habanalabs-tools=1.24.1-482" in text
    assert "habanalabs-tools_1.24.1-482_amd64.deb" in text
    assert "/etc/ld.so.conf.d/habanatools.conf" in text
    assert 'VLLM_GAUDI_BUILD_FLASHINFER="$NATIVE"' in text
    assert 'VLLM_GAUDI_BUILD_GDN_FUSED="$NATIVE"' in text
    assert "VLLM_GAUDI_BUILD_FLASHINFER=1" not in text
    assert "toolchain.json" in text


def test_container_entry_installs_pytest():
    # Gate G1b runs the fused GDN card checks with `python -m pytest`; the image ships no pytest (RUK-3100).
    text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    assert 'install_step 3b "test tools" pip install pytest' in text


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


def _run_rental_env(tmp_path, avail_kb, docker_root=None):
    """PATH with a fake `df` reporting `avail_kb` free (and a fake `docker` when docker_root is set)."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    fakes = {
        "df": "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
              f"printf '/dev/vda1 96000000 95000000 {avail_kb} 99%% /\\n'\n",
    }
    if docker_root is not None:
        fakes["docker"] = (
            'case "$*" in\n'
            "  *Runtimes*) echo '{\"habana\":{\"path\":\"/usr/bin/habana-container-runtime\"}}' ;;\n"
            f"  *DockerRootDir*) echo '{docker_root}' ;;\n"
            "  run*) echo FAKE-DOCKER-RUN ;;\n"
            "esac\n"
        )
    for name, body in fakes.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    return env


def test_run_rental_download_stops_on_low_disk(tmp_path):
    env = _run_rental_env(tmp_path, 1000)
    proc = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPTS_DIR / "run_rental.sh"),
            "--dry-run",
            "--download",
            "--model",
            str(tmp_path / "models" / "X"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 1, proc.stdout
    assert "GB free" in proc.stderr
    assert "--data-disk" in proc.stderr
    assert str(tmp_path) in proc.stderr
    assert len(proc.stderr.strip().splitlines()) == 1
    assert "docker run" not in proc.stdout
    assert not (tmp_path / "models").exists()


def test_run_rental_download_disk_ok(tmp_path):
    env = _run_rental_env(tmp_path, 400000000)
    proc = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPTS_DIR / "run_rental.sh"),
            "--dry-run",
            "--download",
            "--model",
            str(tmp_path / "models" / "X"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "docker run" in proc.stdout
    assert "--download needs 250 GB: ok" in proc.stdout


def test_run_rental_no_disk_check(tmp_path):
    env = _run_rental_env(tmp_path, 1000)
    proc = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPTS_DIR / "run_rental.sh"),
            "--dry-run",
            "--download",
            "--no-disk-check",
            "--model",
            str(tmp_path / "models" / "X"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout
    assert "docker run" in proc.stdout


def test_run_rental_docker_root_stops_on_low_disk(tmp_path):
    (tmp_path / "docker-root").mkdir()
    (tmp_path / "models").mkdir()
    env = _run_rental_env(tmp_path, 1000, docker_root=str(tmp_path / "docker-root"))
    proc = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPTS_DIR / "run_rental.sh"),
            "--model",
            str(tmp_path / "models" / "X"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 1, proc.stdout
    assert "Docker's root" in proc.stderr
    assert "data-root" in proc.stderr
    assert "FAKE-DOCKER-RUN" not in proc.stdout


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
    assert "prepare_host.sh" in names
    assert "BUNDLE.txt" in names
    assert "vllm/setup.py" in names
    assert any(name.startswith("plugin/tools/flash_next/rental/") for name in names)


def test_prepare_host_dry_run(tmp_path):
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "prepare_host.sh"), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "habanalabs-installer.sh install --type base -y" in out
    assert "1.24.1" in out
    assert "habanalabs-container-runtime=1.24.1-482" in out
    assert "/etc/docker/daemon.json" in out
    assert (
        "docker pull vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/"
        "habanalabs/pytorch-installer-2.11.0:latest" in out
    )
    assert "--runtime=habana" in out
    assert not (tmp_path / "prepare-host-logs").exists()


def test_prepare_host_dry_run_version_override(tmp_path):
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "prepare_host.sh"), "--dry-run", "--version", "1.24.2", "--build", "481"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "gaudi-installer/1.24.2/" in out
    assert "habanalabs-container-runtime=1.24.2-481" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
