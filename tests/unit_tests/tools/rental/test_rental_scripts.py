# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental harness shell scripts (RUK-2709 M4-H8).

`bash -n` on the three scripts, the `run_rental.sh --dry-run` docker line (Habana runtime and the
three mounts), and that `make_rental_bundle.sh --allow-dirty` builds a tarball with run_rental.sh
at the top level, the plugin tree, vLLM at the pin and BUNDLE.txt. The bundle test is skipped when
the vLLM checkout is absent. Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SCRIPTS = ["make_rental_bundle.sh", "run_rental.sh", "container_entry.sh", "prepare_host.sh", "tiny_smoke.sh"]
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
    assert "tiny_smoke.sh" in names
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
    assert "lsblk" in out
    assert "--data-disk" in out
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


def _disk_tool_env(tmp_path, **fake):
    """PATH with fake lsblk/blkid/wipefs driven by FAKE_* variables (a blank, unmounted /dev/vdc by default)."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    fakes = {
        "lsblk": (
            'case "$*" in\n'
            '  *"-o TYPE"*) echo "${FAKE_LSBLK_TYPE-disk}" ;;\n'
            '  *"-o NAME"*) echo vdc; if [ -n "${FAKE_LSBLK_PART:-}" ]; then echo "$FAKE_LSBLK_PART"; fi ;;\n'
            '  *"-o MOUNTPOINT"*) echo "${FAKE_LSBLK_MOUNT:-}" ;;\n'
            '  *) echo "NAME SIZE TYPE FSTYPE MOUNTPOINT"; echo "vdc 26.4T disk" ;;\n'
            "esac\nexit 0\n"
        ),
        "blkid": (
            'if [ -n "${FAKE_BLKID_OUT:-}" ]; then printf "%b\\n" "$FAKE_BLKID_OUT"; fi\n'
            'exit "${FAKE_BLKID_RC:-2}"\n'
        ),
        "wipefs": 'if [ -n "${FAKE_WIPEFS_OUT:-}" ]; then printf "%b\\n" "$FAKE_WIPEFS_OUT"; fi\nexit "${FAKE_WIPEFS_RC:-0}"\n',
    }
    for name, body in fakes.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    env.update({f"FAKE_{k.upper()}": v for k, v in fake.items()})
    return env


def _prepare_host_data_disk(tmp_path, env, mount=None):
    mount = mount or tmp_path / "models"
    return subprocess.run(
        ["/bin/bash", str(SCRIPTS_DIR / "prepare_host.sh"), "--dry-run", "--data-disk", "/dev/vdc", "--mount", str(mount)],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60,
    )


def test_prepare_host_data_disk_dry_run(tmp_path):
    proc = _prepare_host_data_disk(tmp_path, _disk_tool_env(tmp_path))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "mkfs.ext4 -F -L fn-models -m 0 /dev/vdc" in out
    assert "nofail" in out
    assert str(tmp_path / "models") in out
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "prepare-host-logs").exists()


def test_prepare_host_data_disk_refuses_blkid_signature(tmp_path):
    env = _disk_tool_env(tmp_path, blkid_rc="0", blkid_out="DEVNAME=/dev/vdc\nTYPE=xfs")
    proc = _prepare_host_data_disk(tmp_path, env)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr
    assert "signature" in proc.stderr
    assert "mkfs.ext4" not in proc.stdout


def test_prepare_host_data_disk_refuses_wipefs_signature(tmp_path):
    env = _disk_tool_env(tmp_path, wipefs_out="DEVICE OFFSET TYPE\nvdc 0x1fe dos")
    proc = _prepare_host_data_disk(tmp_path, env)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr
    assert "mkfs.ext4" not in proc.stdout


def test_prepare_host_data_disk_refuses_unreadable(tmp_path):
    env = _disk_tool_env(tmp_path, blkid_rc="4")
    proc = _prepare_host_data_disk(tmp_path, env)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr


def test_prepare_host_data_disk_refuses_mounted(tmp_path):
    env = _disk_tool_env(tmp_path, lsblk_mount="/")
    proc = _prepare_host_data_disk(tmp_path, env)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stderr


def test_prepare_host_data_disk_refuses_partitions(tmp_path):
    env = _disk_tool_env(tmp_path, lsblk_part="vdc1")
    proc = _prepare_host_data_disk(tmp_path, env)
    assert proc.returncode == 1


def test_prepare_host_data_disk_refuses_nonempty_mount(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "old.safetensors").write_text("x")
    proc = _prepare_host_data_disk(tmp_path, _disk_tool_env(tmp_path))
    assert proc.returncode == 1
    assert "not empty" in proc.stderr


def test_prepare_host_data_disk_ours_skips_mkfs(tmp_path):
    env = _disk_tool_env(tmp_path, blkid_rc="0", blkid_out="TYPE=ext4\nLABEL=fn-models\nUUID=1234")
    proc = _prepare_host_data_disk(tmp_path, env)
    assert proc.returncode == 0
    assert "mkfs.ext4" not in proc.stdout
    assert "nofail" in proc.stdout


def test_prepare_host_mount_needs_data_disk(tmp_path):
    proc = subprocess.run(
        ["/bin/bash", str(SCRIPTS_DIR / "prepare_host.sh"), "--dry-run", "--mount", "/x"],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert proc.returncode == 2


# --------------------------------------------------------------------------- tiny_smoke.sh


def test_container_entry_install_only(tmp_path):
    # tiny_smoke.sh installs the stack and then runs its own check: --install-only stops before the
    # orchestrator and needs no checkpoint (so no --model).
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "container_entry.sh"), "--install-only", "--skip-install", "--results",
         str(tmp_path), "--run-id", "x"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "install only: done" in proc.stdout
    assert "orchestrator" not in proc.stdout


def test_bundle_carries_tiny_smoke_at_the_top_level():
    text = (SCRIPTS_DIR / "make_rental_bundle.sh").read_text()
    assert 'cp "$SCRIPT_DIR/tiny_smoke.sh" "$STAGE/tiny_smoke.sh"' in text
    assert "BUNDLE.txt run_rental.sh prepare_host.sh tiny_smoke.sh plugin vllm" in text


def test_tiny_smoke_dry_run(tmp_path):
    results = tmp_path / "results"
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "tiny_smoke.sh"), "--dry-run", "--results", str(results), "--run-id", "t1"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [shlex.split(ln) for ln in proc.stdout.splitlines() if ln.startswith("docker ")]
    assert len(lines) == 3, proc.stdout
    start, install, smoke = lines
    assert start[:5] == ["docker", "run", "-d", "--name", "fn-smoke"]
    for part in ("--runtime=habana", "HABANA_VISIBLE_DEVICES=all", "--net=host", "--ipc=host",
                 f"{SCRIPTS_DIR}:/bundle", f"{results}:/results"):
        assert part in start, part
    assert not any("/models" in word for word in start)  # the smoke test needs no checkpoint
    assert start[-2:] == ["sleep", "infinity"]
    assert install == ["docker", "exec", "fn-smoke", "bash",
                       "/bundle/plugin/tools/flash_next/rental/container_entry.sh", "--install-only", "--results",
                       "/results", "--run-id", "t1"]
    assert smoke == ["docker", "exec", "-w", "/bundle/plugin", "fn-smoke", "python", "-m",
                     "tools.flash_next.rental.tiny_smoke", "--out", "/results/t1", "--ckpt", "/tmp/fn-smoke-ckpt",
                     "--gdn", "pytorch,fused"]
    assert not results.exists()  # a dry run writes nothing


FAKE_SMOKE_DOCKER = r"""
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
  *Runtimes*) echo '{"habana":{"path":"/usr/bin/habana-container-runtime"}}' ;;
  "ps --format {{.Names}}") for n in $FAKE_RUNNING; do echo "$n"; done ;;
  "ps -a --format {{.Names}}") for n in $FAKE_EXISTING; do echo "$n"; done ;;
  "exec fn-smoke test -f /tmp/fn-smoke.installed") exit ${FAKE_MARKER_EXIT:-1} ;;
  inspect*) echo "$FAKE_RESULTS_MOUNT" ;;
  *container_entry.sh*) echo "install only: done"; exit ${FAKE_INSTALL_EXIT:-0} ;;
  *pkill*) ;;
  *tools.flash_next.rental.tiny_smoke*) echo "tiny smoke [pytorch]: PASSED"; exit ${FAKE_SMOKE_EXIT:-0} ;;
esac
exit 0
"""


def _smoke(tmp_path, *args, in_tree=False, **fake_env):
    """Run tiny_smoke.sh from a fake unpacked bundle with a fake `docker`; returns (proc, docker calls)."""
    bundle = tmp_path / "bundle"
    rental = bundle / "plugin" / "tools" / "flash_next" / "rental"
    rental.mkdir(parents=True)
    (rental / "container_entry.sh").write_text("#!/bin/sh\n")
    script = (rental if in_tree else bundle) / "tiny_smoke.sh"
    script.write_text((SCRIPTS_DIR / "tiny_smoke.sh").read_text())
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\n" + FAKE_SMOKE_DOCKER)
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    env["FAKE_DOCKER_LOG"] = str(log)
    env.update({k: str(v) for k, v in fake_env.items()})
    proc = subprocess.run(["bash", str(script), "--results", str(tmp_path / "results"), "--run-id", "t1", *args],
                          capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls, bundle


def _calls(calls, needle):
    return [c for c in calls if needle in c]


def test_tiny_smoke_new_container_installs_then_runs(tmp_path):
    proc, calls, bundle = _smoke(tmp_path)
    assert proc.returncode == 0, proc.stderr
    steps = [c for c in calls if c.startswith(("run ", "exec ")) and "test -f" not in c]
    assert len(steps) == 4, calls
    assert steps[0].startswith("run -d --name fn-smoke ") and f"{bundle}:/bundle" in steps[0]
    assert "container_entry.sh --install-only" in steps[1]
    assert steps[2] == "exec fn-smoke touch /tmp/fn-smoke.installed"
    assert "tools.flash_next.rental.tiny_smoke --out /results/t1" in steps[3]
    assert "tiny smoke [pytorch]: PASSED" in proc.stdout
    assert "sudo docker exec -it fn-smoke bash" in proc.stdout  # how to get in, the container stays
    assert not _calls(calls, "rm -f")
    console = tmp_path / "results" / "t1" / "console.log"
    assert "tiny smoke [pytorch]: PASSED" in console.read_text()


def test_tiny_smoke_from_inside_the_plugin_tree_finds_the_bundle(tmp_path):
    proc, calls, bundle = _smoke(tmp_path, in_tree=True)
    assert proc.returncode == 0, proc.stderr
    assert f"{bundle}:/bundle" in _calls(calls, "run -d")[0]


def test_tiny_smoke_reuses_an_installed_container(tmp_path):
    proc, calls, _ = _smoke(tmp_path, FAKE_RUNNING="fn-smoke", FAKE_EXISTING="fn-smoke", FAKE_MARKER_EXIT=0)
    assert proc.returncode == 0, proc.stderr
    assert not _calls(calls, "run -d")
    assert not _calls(calls, "container_entry.sh")
    assert len(_calls(calls, "tools.flash_next.rental.tiny_smoke")) == 1
    assert "--reinstall" in proc.stdout  # says how to pick up a code change


def test_tiny_smoke_reuse_writes_where_the_container_mounts_results(tmp_path):
    # /results was mounted when the container was created: a later call cannot move it.
    mounted = tmp_path / "first-results"
    proc, calls, _ = _smoke(tmp_path, FAKE_RUNNING="fn-smoke", FAKE_EXISTING="fn-smoke", FAKE_MARKER_EXIT=0,
                            FAKE_RESULTS_MOUNT=mounted)
    assert proc.returncode == 0, proc.stderr
    assert (mounted / "t1" / "console.log").is_file()
    assert f"results: {mounted}/t1" in proc.stdout
    assert not (tmp_path / "results").exists()


def test_tiny_smoke_reinstall_in_the_same_container(tmp_path):
    proc, calls, _ = _smoke(tmp_path, "--reinstall", FAKE_RUNNING="fn-smoke", FAKE_EXISTING="fn-smoke",
                            FAKE_MARKER_EXIT=0)
    assert proc.returncode == 0, proc.stderr
    assert not _calls(calls, "run -d")
    assert len(_calls(calls, "container_entry.sh --install-only")) == 1


def test_tiny_smoke_fresh_replaces_the_container(tmp_path):
    proc, calls, _ = _smoke(tmp_path, "--fresh", FAKE_RUNNING="fn-smoke", FAKE_EXISTING="fn-smoke",
                            FAKE_MARKER_EXIT=0)
    assert proc.returncode == 0, proc.stderr
    assert calls.index("rm -f fn-smoke") < calls.index(_calls(calls, "run -d")[0])
    assert len(_calls(calls, "container_entry.sh --install-only")) == 1


def test_tiny_smoke_removes_a_stopped_container_first(tmp_path):
    proc, calls, _ = _smoke(tmp_path, FAKE_RUNNING="", FAKE_EXISTING="fn-smoke")
    assert proc.returncode == 0, proc.stderr
    assert calls.index("rm -f fn-smoke") < calls.index(_calls(calls, "run -d")[0])


def test_tiny_smoke_refuses_while_the_rental_run_holds_the_cards(tmp_path):
    proc, calls, _ = _smoke(tmp_path, FAKE_RUNNING="fn-rental", FAKE_EXISTING="fn-rental")
    assert proc.returncode == 1
    assert "fn-rental" in proc.stderr
    assert not _calls(calls, "run -d") and not _calls(calls, "exec ")


def test_tiny_smoke_failure_keeps_the_container_and_the_exit_code(tmp_path):
    proc, calls, _ = _smoke(tmp_path, FAKE_SMOKE_EXIT=1)
    assert proc.returncode == 1
    assert not _calls(calls, "rm -f")
    assert "sudo docker exec -it fn-smoke bash" in proc.stdout


def test_tiny_smoke_interrupt_stops_the_check_inside_the_container(tmp_path):
    # Ctrl-C ends the docker client, not the process in the container: without this the tiny server
    # would keep all 8 cards and the next start would fail with "Device acquire failed".
    import signal
    import time

    bundle = tmp_path / "bundle"
    rental = bundle / "plugin" / "tools" / "flash_next" / "rental"
    rental.mkdir(parents=True)
    (rental / "container_entry.sh").write_text("#!/bin/sh\n")
    script = bundle / "tiny_smoke.sh"
    script.write_text((SCRIPTS_DIR / "tiny_smoke.sh").read_text())
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\n" + FAKE_SMOKE_DOCKER.replace(
        'echo "tiny smoke [pytorch]: PASSED"; exit ${FAKE_SMOKE_EXIT:-0}', "sleep 30"))
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env.update(PATH=f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin", FAKE_DOCKER_LOG=str(log))
    proc = subprocess.Popen(["bash", str(script), "--results", str(tmp_path / "results"), "--run-id", "t1"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path, env=env,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if log.exists() and "tools.flash_next.rental.tiny_smoke" in log.read_text():
                break
            time.sleep(0.05)
        os.killpg(proc.pid, signal.SIGINT)  # what Ctrl-C does: the whole foreground group
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode == 130
    assert "exec fn-smoke pkill -INT -f tools.flash_next.rental.tiny_smoke" in log.read_text().splitlines()


def test_tiny_smoke_install_failure_stops_before_the_check(tmp_path):
    proc, calls, _ = _smoke(tmp_path, FAKE_INSTALL_EXIT=7)
    assert proc.returncode == 7
    assert not _calls(calls, "touch /tmp/fn-smoke.installed")
    assert not _calls(calls, "tools.flash_next.rental.tiny_smoke")
    assert "install failed" in proc.stderr


def test_tiny_smoke_outside_a_bundle_is_refused(tmp_path):
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / "tiny_smoke.sh"), "--results", str(tmp_path / "r")],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 1
    assert "unpacked rental bundle" in proc.stderr


def test_tiny_smoke_unknown_option(tmp_path):
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / "tiny_smoke.sh"), "--nope"], capture_output=True, text=True,
                          cwd=tmp_path, timeout=60)
    assert proc.returncode == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
