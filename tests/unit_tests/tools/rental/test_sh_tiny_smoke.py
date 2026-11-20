# SPDX-License-Identifier: Apache-2.0
"""Tests for tiny_smoke.sh: the container flags, the Habana log folder and the signal trap.

The real path runs from a stand-in bundle with a stand-in `docker` on PATH. Its check is a state file:
`docker exec ... tiny_smoke` blocks until `pkill -INT` (or `docker stop`) removes it, as the real
client does until the check in the container has stopped its server.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"

FAKE_DOCKER = r"""
echo "$*" >> "$FAKE_DOCKER_LOG"
alive="$FAKE_STATE/check-alive"
case "$*" in
  *Runtimes*) echo '{"habana":{"path":"/usr/bin/habana-container-runtime"}}' ;;
  "ps --format {{.Names}}") for n in ${FAKE_RUNNING:-}; do echo "$n"; done ;;
  "ps -a --format {{.Names}}") for n in ${FAKE_EXISTING:-}; do echo "$n"; done ;;
  "exec fn-smoke test -f /tmp/fn-smoke.installed") exit ${FAKE_MARKER_EXIT:-1} ;;
  inspect*) echo "${FAKE_RESULTS_MOUNT:-}" ;;
  *container_entry.sh*) echo "install only: done" ;;
  *"pkill -INT"*) if [ "${FAKE_INT:-obey}" = obey ]; then rm -f "$alive"; fi ;;
  *pgrep*) if [ -f "$alive" ]; then exit 0; fi; exit 1 ;;
  "stop -t 10 fn-smoke") rm -f "$alive" ;;
  *tools.flash_next.rental.tiny_smoke*)
    if [ -n "${FAKE_CHECK_EXIT:-}" ]; then
      echo "tiny smoke [pytorch]: PASSED"
      exit "$FAKE_CHECK_EXIT"
    fi
    : > "$alive"
    echo "tiny smoke [pytorch]: starting the server"
    while [ -f "$alive" ]; do sleep 0.1; done
    echo "FAKE CHECK STOPPED"
    exit 1 ;;
esac
exit 0
"""


def _setup(tmp_path, **fake):
    bundle = tmp_path / "bundle"
    rental = bundle / "plugin" / "tools" / "flash_next" / "rental"
    rental.mkdir(parents=True)
    (rental / "container_entry.sh").write_text("#!/bin/sh\n")
    script = bundle / "tiny_smoke.sh"
    script.write_text((SCRIPTS_DIR / "tiny_smoke.sh").read_text())
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\n" + FAKE_DOCKER)
    docker.chmod(0o755)
    (tmp_path / "state").mkdir()
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    env["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")
    env["FAKE_STATE"] = str(tmp_path / "state")
    env["FN_SMOKE_STOP_WAIT_S"] = "2"
    env.update({k: str(v) for k, v in fake.items()})
    return ["/bin/bash", str(script), "--results", str(tmp_path / "results"), "--run-id", "t1"], env


def _calls(tmp_path):
    log = tmp_path / "docker.log"
    return log.read_text().splitlines() if log.exists() else []


def _dry_run(tmp_path, *extra):
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / "tiny_smoke.sh"), "--dry-run", "--results",
                           str(tmp_path / "results"), "--run-id", "t1", *extra],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 0, proc.stderr
    lines = [shlex.split(ln) for ln in proc.stdout.splitlines() if ln.startswith("docker ")]
    assert len(lines) == 3, proc.stdout
    return lines


def test_the_smoke_container_flags(tmp_path):
    start, _install, _smoke = _dry_run(tmp_path)
    assert start[:5] == ["docker", "run", "-d", "--name", "fn-smoke"]
    image = start.index("vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest")
    flags = start[:image]
    for part in ("--init", "--runtime=habana", "--cap-add=sys_nice", "--net=host", "--ipc=host"):
        assert part in flags, part
    for value in ("HABANA_VISIBLE_DEVICES=all", "OMPI_MCA_btl_vader_single_copy_mechanism=none", "PYTHONUNBUFFERED=1",
                  "PYTHONFAULTHANDLER=1"):
        assert flags[flags.index(value) - 1] == "-e", value
    assert start[image + 1:] == ["sleep", "infinity"]
    assert not (tmp_path / "results").exists()  # a dry run writes nothing


def test_the_check_gets_the_runs_habana_log_folder(tmp_path):
    # The container outlives one call, so the per-run folder travels with the docker exec, which also
    # reaches a container an older bundle created without these variables.
    _start, _install, smoke = _dry_run(tmp_path)
    name = smoke.index("fn-smoke")
    flags = smoke[:name]
    assert smoke[:2] == ["docker", "exec"]
    for value in ("HABANA_LOGS=/results/t1/habana_logs", "PYTHONFAULTHANDLER=1"):
        assert flags[flags.index(value) - 1] == "-e", value
    assert smoke[name + 1:name + 4] == ["python", "-m", "tools.flash_next.rental.tiny_smoke"]


def test_the_habana_log_folder_is_made_before_the_check(tmp_path):
    cmd, env = _setup(tmp_path, FAKE_CHECK_EXIT=0)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "results" / "t1" / "habana_logs").is_dir()
    assert "tiny smoke [pytorch]: PASSED" in (tmp_path / "results" / "t1" / "console.log").read_text()
    assert not [c for c in _calls(tmp_path) if "pkill" in c or c.startswith("stop")]


def test_the_checks_exit_code_is_kept(tmp_path):
    cmd, env = _setup(tmp_path, FAKE_CHECK_EXIT=1)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)
    assert proc.returncode == 1
    assert "sudo docker exec -it fn-smoke bash" in proc.stdout


def _interrupt(tmp_path, signum, group, **fake):
    cmd, env = _setup(tmp_path, **fake)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path, env=env,
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (tmp_path / "state" / "check-alive").exists():
            time.sleep(0.05)
        assert (tmp_path / "state" / "check-alive").exists(), "the stand-in check did not start"
        time.sleep(0.2)
        if group:
            os.killpg(proc.pid, signum)  # what Ctrl-C and a dropped terminal do: the whole foreground group
        else:
            os.kill(proc.pid, signum)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    return proc.returncode, out, err


@pytest.mark.parametrize("signum,group,code", [(signal.SIGINT, True, 130), (signal.SIGTERM, False, 143),
                                               (signal.SIGHUP, False, 129), (signal.SIGHUP, True, 129)])
def test_a_signal_stops_the_check_inside_the_container(tmp_path, signum, group, code):
    # Without this the tiny server keeps all 8 cards and the next start fails with "Device acquire failed".
    rc, out, err = _interrupt(tmp_path, signum, group)
    assert rc == code, out + err
    calls = _calls(tmp_path)
    assert "exec fn-smoke pkill -INT -f tools.flash_next.rental.tiny_smoke" in calls
    assert not [c for c in calls if c.startswith("stop")], calls  # the check stopped in time: the container stays
    assert not (tmp_path / "state" / "check-alive").exists()
    console = (tmp_path / "results" / "t1" / "console.log").read_text()
    assert "interrupted" in console
    assert "FAKE CHECK STOPPED" in console  # the client kept logging while the check went down


def test_a_check_that_does_not_stop_costs_the_container_after_a_bounded_wait(tmp_path):
    started = time.monotonic()
    rc, out, err = _interrupt(tmp_path, signal.SIGTERM, False, FAKE_INT="ignore")
    assert rc == 143, out + err
    assert time.monotonic() - started < 40
    calls = _calls(tmp_path)
    assert calls.index("exec fn-smoke pkill -INT -f tools.flash_next.rental.tiny_smoke") < calls.index(
        "stop -t 10 fn-smoke")
    assert not (tmp_path / "state" / "check-alive").exists()
    assert "new container" in out + err  # the next call starts over: said, not silent
