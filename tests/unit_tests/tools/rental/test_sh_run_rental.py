# SPDX-License-Identifier: Apache-2.0
"""Tests for run_rental.sh: option pass-through, `--from` checks, the end path and the signal trap.

The script is driven with `--dry-run`, and for the real path with a stand-in `docker` on PATH whose
one "container" is a state file: `docker run` blocks until something removes it, as the real client
does until the container is gone. Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import ast
import json
import os
import re
import signal
import subprocess
import tarfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SCRIPT = SCRIPTS_DIR / "run_rental.sh"

FAKE_DOCKER = r"""
echo "$*" >> "$FAKE_DOCKER_LOG"
state="$FAKE_STATE/container"
case "$1" in
  info)
    case "$*" in
      *Runtimes*) echo '{"habana":{"path":"/usr/bin/habana-container-runtime"}}' ;;
    esac ;;
  ps) for n in ${FAKE_RUNNING:-}; do echo "$n"; done ;;
  run)
    case " $* " in
      *" -d "*) echo "0123456789ab"; exit "${FAKE_DETACHED_EXIT:-0}" ;;
    esac
    if [ -n "${FAKE_NO_INIT:-}" ]; then
      case " $* " in
        *" --init "*)
          echo 'docker: Error response from daemon: exec: "docker-init": executable file not found in $PATH.' >&2
          exit 125 ;;
      esac
    fi
    if [ -n "${FAKE_RUN_CONFLICT:-}" ]; then
      echo 'docker: Error response from daemon: Conflict. The container name "/fn-rental" is already in use.' >&2
      exit 125
    fi
    label=""
    for a in "$@"; do
      case "$a" in fn-rental.run-id=*) label="${a#fn-rental.run-id=}" ;; esac
    done
    if [ -n "${FAKE_EXPECT_DIR:-}" ] && [ -d "$FAKE_EXPECT_DIR" ]; then : > "$FAKE_STATE/dir-was-there"; fi
    echo "$label" > "$state"
    echo "FAKE CONTAINER OUTPUT"
    if [ -n "${FAKE_RUN_EXIT:-}" ]; then
      if [ -z "${FAKE_RUN_LEAVES_CONTAINER:-}" ]; then rm -f "$state"; fi
      exit "$FAKE_RUN_EXIT"
    fi
    while [ -f "$state" ]; do sleep 0.1; done
    echo "FAKE CONTAINER STOPPED"
    exit "${FAKE_STOPPED_EXIT:-1}" ;;
  inspect)
    [ -f "$state" ] || exit 1
    case "$*" in
      *Labels*) cat "$state" ;;
      *State*) echo '{"Status":"running","ExitCode":0,"OOMKilled":false}' ;;
    esac ;;
  exec)
    case "$*" in
      *"pkill -INT -f tools.flash_next.rental.main"*)
        case "${FAKE_INT:-obey}" in
          obey) rm -f "$state"; exit 0 ;;
          ignore) exit 0 ;;
          *) exit 1 ;;
        esac ;;
    esac ;;
  stop) if [ "${FAKE_STOP:-obey}" = obey ]; then rm -f "$state"; fi ;;
  kill) rm -f "$state" ;;
  image)
    if [ -n "${FAKE_NO_DIGEST:-}" ]; then exit 1; fi
    case "$*" in
      *RepoDigests*) echo "vault.habana.ai/gaudi-docker/x@sha256:1111" ;;
      *.Id*) echo "sha256:2222" ;;
    esac ;;
esac
exit 0
"""

FAKE_DMESG = r"""
if [ -n "${FAKE_DMESG_DENIED:-}" ]; then
  echo "dmesg: read kernel buffer failed: Operation not permitted" >&2
  exit 1
fi
i=1
while [ "$i" -le 500 ]; do echo "kernel line $i"; i=$((i + 1)); done
"""


def _dry_run(tmp_path, *args):
    return subprocess.run(["bash", str(SCRIPT), "--dry-run", "--no-disk-check", "--model", "/tmp/x", *args],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)


def _flat(proc):
    # printf %q escapes `:` and `,` on bash 3.2; compare on the unescaped line.
    return proc.stdout.replace("\\", "")


def _past_run(tmp_path, run_id, manifest=True):
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    if manifest:
        (run_dir / "manifest.json").write_text("{}")
    return run_dir


def _entries(folder):
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


# --------------------------------------------------------------------------- item 1: pass-through


def test_new_orchestrator_options_reach_the_entry(tmp_path):
    proc = _dry_run(tmp_path, "--quick-gates", "--no-verify-probe", "--load-read-timeout", "900",
                    "--warm-request-timeout", "1200.5")
    assert proc.returncode == 0, proc.stderr
    flat = _flat(proc)
    entry = flat.index("container_entry.sh")
    for part in ("--quick-gates", "--no-verify-probe", "--load-read-timeout 900", "--warm-request-timeout 1200.5"):
        assert part in flat, part
        assert flat.index(part) > entry, part  # an entry argument, not a docker flag


def test_new_orchestrator_options_are_off_by_default(tmp_path):
    proc = _dry_run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    for part in ("--quick-gates", "--no-verify-probe", "--load-read-timeout", "--warm-request-timeout"):
        assert part not in proc.stdout, part


def test_no_warm_pass_stays_the_last_entry_argument(tmp_path):
    proc = _dry_run(tmp_path, "--no-warm-pass", "--quick-gates", "--load-read-timeout", "900")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("--no-warm-pass")


@pytest.mark.parametrize("option", ["--load-read-timeout", "--warm-request-timeout", "--server-timeout", "--from",
                                    "--run-id", "--model", "--variants"])
def test_an_option_without_its_value_is_a_usage_error(tmp_path, option):
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--model", "/tmp/x", option],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "%s needs a value" % option in proc.stderr
    assert "docker run" not in proc.stdout


@pytest.mark.parametrize("option", ["--load-read-timeout", "--warm-request-timeout", "--server-timeout"])
@pytest.mark.parametrize("value", ["abc", "--quick-gates", "12s", "1.2.3", ""])
def test_a_timeout_must_be_a_number_of_seconds(tmp_path, option, value):
    proc = _dry_run(tmp_path, option, value)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert option in proc.stderr
    assert "docker run" not in proc.stdout


def test_an_unknown_option_is_still_an_error(tmp_path):
    proc = _dry_run(tmp_path, "--quick-gate")
    assert proc.returncode == 2
    assert "unknown option: --quick-gate" in proc.stderr
    assert "docker run" not in proc.stdout


def test_usage_names_the_new_options():
    proc = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    for part in ("--quick-gates", "--no-verify-probe", "--load-read-timeout S", "--warm-request-timeout S",
                 "fused_gdn", "card_tests"):
        assert part in proc.stdout, part


# --------------------------------------------------------------------------- item 2: --from


def test_from_without_run_id_is_a_usage_error_that_lists_the_runs(tmp_path):
    _past_run(tmp_path, "20261118T222241Z")
    _past_run(tmp_path, "20261118T223100Z")
    _past_run(tmp_path, "smoke-20261119T010000Z", manifest=False)  # no manifest: not a run to resume
    before = _entries(tmp_path / "results")
    proc = _dry_run(tmp_path, "--from", "tiny")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--from needs --run-id" in proc.stderr
    assert "20261118T222241Z" in proc.stderr
    assert "20261118T223100Z" in proc.stderr
    assert "smoke-20261119T010000Z" not in proc.stderr
    assert "docker run" not in proc.stdout
    assert _entries(tmp_path / "results") == before  # no empty run folder appears


def test_from_without_run_id_says_when_there_is_no_run(tmp_path):
    proc = _dry_run(tmp_path, "--from", "tiny")
    assert proc.returncode == 2
    assert "--from needs --run-id" in proc.stderr
    assert "none" in proc.stderr
    assert not (tmp_path / "results").exists()


def test_from_with_an_unknown_run_id_is_an_error_and_creates_nothing(tmp_path):
    _past_run(tmp_path, "20261118T223100Z")
    proc = _dry_run(tmp_path, "--from", "tiny", "--run-id", "20261118T223100")  # the Z is missing
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "20261118T223100Z" in proc.stderr  # the existing ids are listed
    assert _entries(tmp_path / "results") == ["20261118T223100Z"]


def test_from_needs_the_manifest_of_an_earlier_start(tmp_path):
    # The orchestrator refuses to resume a folder without a manifest.json; refuse it before the container.
    _past_run(tmp_path, "20261118T223100Z")
    _past_run(tmp_path, "20261118T215518Z", manifest=False)  # that run stopped in the install steps
    proc = _dry_run(tmp_path, "--from", "preflight", "--run-id", "20261118T215518Z")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "manifest.json" in proc.stderr
    assert "  20261118T223100Z" in proc.stderr
    assert "  20261118T215518Z" not in proc.stderr
    assert "docker run" not in proc.stdout


@pytest.mark.parametrize("phase", ["preflight", "build", "fused_gdn", "tiny", "card_tests", "A", "B", "C", "report"])
def test_from_accepts_every_phase_of_the_run(tmp_path, phase):
    _past_run(tmp_path, "r1")
    proc = _dry_run(tmp_path, "--from", phase, "--run-id", "r1")
    assert proc.returncode == 0, proc.stderr
    assert "--from %s" % phase in _flat(proc)
    assert "--run-id r1" in _flat(proc)


@pytest.mark.parametrize("phase", ["tinny", "fused-gdn", "cardtests", "a", "D", "Report"])
def test_from_rejects_a_misspelt_phase_before_anything_starts(tmp_path, phase):
    _past_run(tmp_path, "r1")
    proc = _dry_run(tmp_path, "--from", phase, "--run-id", "r1")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--from %s" % phase in proc.stderr
    for name in ("preflight", "build", "fused_gdn", "tiny", "card_tests", "report"):
        assert name in proc.stderr, name  # the message lists the phases
    assert "docker run" not in proc.stdout


def test_from_checks_the_variants_of_this_run(tmp_path):
    _past_run(tmp_path, "r1")
    assert _dry_run(tmp_path, "--from", "C", "--run-id", "r1", "--variants", "A,B").returncode == 2
    assert _dry_run(tmp_path, "--from", "B", "--run-id", "r1", "--variants", "A,B").returncode == 0
    assert _dry_run(tmp_path, "--from", "P", "--run-id", "r1").returncode == 2  # P runs only with --with-profile
    assert _dry_run(tmp_path, "--from", "P", "--run-id", "r1", "--with-profile").returncode == 0


def test_an_unknown_variant_id_is_an_error(tmp_path):
    proc = _dry_run(tmp_path, "--variants", "A,X")
    assert proc.returncode == 2
    assert "X" in proc.stderr and "--variants" in proc.stderr


def _script_words(name):
    match = re.search(r'^%s="([^"]*)"$' % name, SCRIPT.read_text(), re.M)
    assert match, name
    return match.group(1).split()


def test_the_phase_list_matches_the_orchestrator():
    # One list in the script; this keeps it equal to main.py's gate phases and the variant ids.
    main_text = (SCRIPTS_DIR / "main.py").read_text()
    match = re.search(r"^GATE_PHASES = (\(.*\))$", main_text, re.M)
    assert match
    assert _script_words("GATE_PHASES") == list(ast.literal_eval(match.group(1)))
    variants = json.loads((SCRIPTS_DIR / "variants.json").read_text())
    assert _script_words("KNOWN_VARIANTS") == [v["id"] for v in variants]
    default = re.search(r'add_argument\("--variants", default="([^"]*)"\)', main_text)
    assert default
    assert _script_words("DEFAULT_VARIANTS") == default.group(1).split(",")


def test_usage_quotes_the_orchestrators_timeout_defaults():
    main_text = (SCRIPTS_DIR / "main.py").read_text()
    usage = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60).stdout
    flat = " ".join(usage.split())
    for option, constant in (("--load-read-timeout S", "LOAD_REQUEST_TIMEOUT_S"),
                             ("--warm-request-timeout S", "WARM_PASS_REQUEST_TIMEOUT_S")):
        value = re.search(r"^%s = ([0-9.]+)$" % constant, main_text, re.M)
        assert value, constant
        text = flat[flat.index(option):]
        assert "its default: %d)" % float(value.group(1)) in text[:text.index(")") + 1], option


# --------------------------------------------------------------------------- item 4: docker run flags


def test_the_run_container_flags(tmp_path):
    proc = _dry_run(tmp_path, "--run-id", "r9")
    assert proc.returncode == 0, proc.stderr
    flat = _flat(proc)
    image = flat.index("vault.habana.ai/")
    for part in ("docker run --rm --name fn-rental", "--init", "-e PYTHONFAULTHANDLER=1",
                 "-e HABANA_LOGS=/results/r9/habana_logs", "--runtime=habana", "-e HABANA_VISIBLE_DEVICES=all",
                 "-e OMPI_MCA_btl_vader_single_copy_mechanism=none", "-e PYTHONUNBUFFERED=1", "--cap-add=sys_nice",
                 "--net=host", "--ipc=host", ":/bundle", ":/models", ":/results", ":/recipe-cache"):
        assert part in flat, part
        assert flat.index(part) < image, part  # a docker flag: before the image
    assert not (tmp_path / "results" / "r9" / "habana_logs").exists()  # the folder appears only for a real run


def test_a_dry_run_writes_nothing(tmp_path):
    (tmp_path / "there").mkdir()
    for extra in ([], ["--download"], ["--download-only"]):
        proc = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--no-disk-check", "--model",
                               str(tmp_path / ("there" if not extra else "models") / "X"), *extra],
                              capture_output=True, text=True, cwd=tmp_path, timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert "docker run" in proc.stdout
        assert f"{tmp_path}/results:/results" in _flat(proc) or extra == ["--download-only"]
    assert _entries(tmp_path) == ["there"]  # no results folder, no run folder, no models folder


def test_a_dry_run_still_needs_the_model_folder_when_nothing_downloads_it(tmp_path):
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--no-disk-check", "--model",
                           str(tmp_path / "models" / "X")],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr


# --------------------------------------------------------------------------- the real path, stand-in docker


def _env(tmp_path, **fake):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("docker", FAKE_DOCKER), ("dmesg", FAKE_DMESG)):
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    env["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")
    env["FAKE_STATE"] = str(state)
    env["FN_RENTAL_STOP_WAIT_S"] = "2"
    env.update({k: str(v) for k, v in fake.items()})
    return env


def _args(tmp_path, *extra):
    return ["/bin/bash", str(SCRIPT), "--no-disk-check", "--no-recipe-cache", "--model", str(tmp_path / "models" / "X"),
            "--run-id", "r1", *extra]


def _calls(tmp_path):
    log = tmp_path / "docker.log"
    return log.read_text().splitlines() if log.exists() else []


def _run(tmp_path, *extra, **fake):
    proc = subprocess.run(_args(tmp_path, *extra), capture_output=True, text=True, cwd=tmp_path,
                          env=_env(tmp_path, **fake), timeout=120)
    return proc


def _tar_names(tmp_path):
    with tarfile.open(tmp_path / "results" / "r1.tar.gz", "r:gz") as tar:
        return set(tar.getnames())


def _closing_lines_once(out, tmp_path):
    assert out.count("PASTE THIS FILE: %s/results/r1/summary.md" % tmp_path) == 1, out
    assert out.count("SEND THIS FILE: %s/results/r1.tar.gz" % tmp_path) == 1, out


def test_a_finished_run_keeps_the_exit_code_and_writes_the_tarball(tmp_path):
    proc = _run(tmp_path, FAKE_RUN_EXIT=3, FAKE_EXPECT_DIR=tmp_path / "results" / "r1" / "habana_logs")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "FAKE CONTAINER OUTPUT" in proc.stdout
    assert "container exit code: 3" in proc.stdout
    _closing_lines_once(proc.stdout, tmp_path)
    console = (tmp_path / "results" / "r1" / "console.log").read_text()
    assert "run id: r1" in console and "FAKE CONTAINER OUTPUT" in console and "container exit code: 3" in console
    assert (tmp_path / "state" / "dir-was-there").exists()  # habana_logs was made before the container started
    names = _tar_names(tmp_path)
    assert "r1/console.log" in names
    assert "r1/habana_logs" in names
    # a finished container is not asked to stop
    assert not [c for c in _calls(tmp_path) if c.startswith(("stop", "kill", "exec"))]


def test_a_passing_run_exits_zero(tmp_path):
    proc = _run(tmp_path, FAKE_RUN_EXIT=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _closing_lines_once(proc.stdout, tmp_path)


def test_a_host_without_docker_init_runs_the_container_without_it(tmp_path):
    # --init is new on the node: a host whose Docker has no docker-init must not lose the run to it.
    proc = _run(tmp_path, FAKE_RUN_EXIT=0, FAKE_NO_INIT=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = [" %s " % c for c in _calls(tmp_path) if c.startswith("run ")]
    assert len(runs) == 2, runs
    assert " --init " in runs[0] and " --init " not in runs[1]
    assert "without --init" in proc.stdout
    assert "FAKE CONTAINER OUTPUT" in proc.stdout
    _closing_lines_once(proc.stdout, tmp_path)


def test_another_docker_error_is_not_retried(tmp_path):
    proc = _run(tmp_path, FAKE_RUN_CONFLICT=1)
    assert proc.returncode == 125, proc.stdout + proc.stderr
    assert len([c for c in _calls(tmp_path) if c.startswith("run ")]) == 1


def test_the_run_container_carries_the_run_id_label(tmp_path):
    proc = _run(tmp_path, FAKE_RUN_EXIT=0)
    assert proc.returncode == 0, proc.stderr
    run_line = [c for c in _calls(tmp_path) if c.startswith("run ")][0]
    assert "--label fn-rental.run-id=r1" in run_line
    assert "--init" in run_line and "--sig-proxy=false" in run_line


def test_evidence_is_captured_before_the_tarball(tmp_path):
    proc = _run(tmp_path, FAKE_RUN_EXIT=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run_dir = tmp_path / "results" / "r1"
    dmesg = (run_dir / "dmesg-tail.txt").read_text().splitlines()
    assert len(dmesg) == 400 and dmesg[0] == "kernel line 101" and dmesg[-1] == "kernel line 500"
    digest = (run_dir / "image-digest.txt").read_text()
    assert "vault.habana.ai/gaudi-docker/x@sha256:1111" in digest
    assert "sha256:2222" in digest
    names = _tar_names(tmp_path)
    assert "r1/dmesg-tail.txt" in names and "r1/image-digest.txt" in names
    assert not (run_dir / "container-state.json").exists()  # --rm: the finished container is gone


def test_evidence_capture_never_fails_the_run(tmp_path):
    proc = _run(tmp_path, FAKE_RUN_EXIT=0, FAKE_DMESG_DENIED=1, FAKE_NO_DIGEST=1)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run_dir = tmp_path / "results" / "r1"
    assert not (run_dir / "dmesg-tail.txt").exists()  # not permitted: skipped quietly
    assert "Operation not permitted" not in proc.stdout + proc.stderr
    assert not (run_dir / "image-digest.txt").exists()
    _closing_lines_once(proc.stdout, tmp_path)


def test_the_tarball_leaves_out_the_tiny_weights(tmp_path):
    ckpt = tmp_path / "results" / "r1" / "tiny" / "ckpt"
    ckpt.mkdir(parents=True)
    for name in ("model.safetensors", "model-00001-of-00002.safetensors", "config.json", "reference.json",
                 "tokenizer.json"):
        (ckpt / name).write_text("x")
    (tmp_path / "results" / "r1" / "tiny" / "server0.log").write_text("log")
    proc = _run(tmp_path, FAKE_RUN_EXIT=0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = _tar_names(tmp_path)
    assert not [n for n in names if n.endswith(".safetensors")], names
    for kept in ("r1/tiny/ckpt/config.json", "r1/tiny/ckpt/reference.json", "r1/tiny/ckpt/tokenizer.json",
                 "r1/tiny/server0.log"):
        assert kept in names, kept
    assert (ckpt / "model.safetensors").exists()  # left out of the tarball, not deleted


def test_another_runs_container_is_never_stopped(tmp_path):
    env_dir = tmp_path / "state"
    env_dir.mkdir()
    (env_dir / "container").write_text("the-other-run\n")  # fn-rental exists and belongs to another run id
    proc = _run(tmp_path, FAKE_RUN_CONFLICT=1)
    assert proc.returncode == 125, proc.stdout + proc.stderr
    assert not [c for c in _calls(tmp_path) if c.startswith(("stop", "kill", "exec"))], _calls(tmp_path)
    assert (env_dir / "container").exists()
    _closing_lines_once(proc.stdout, tmp_path)


def test_a_container_that_outlives_its_client_is_recorded_and_stopped(tmp_path):
    # The docker client died (exit 143) and the container is still there: its state goes into the run
    # folder and it is stopped, so no server keeps the cards after the script returns.
    proc = _run(tmp_path, FAKE_RUN_EXIT=143, FAKE_RUN_LEAVES_CONTAINER=1)
    assert proc.returncode == 143, proc.stdout + proc.stderr
    state = json.loads((tmp_path / "results" / "r1" / "container-state.json").read_text())
    assert state["OOMKilled"] is False and state["ExitCode"] == 0
    assert "exec fn-rental pkill -INT -f tools.flash_next.rental.main" in _calls(tmp_path)
    assert not (tmp_path / "state" / "container").exists()
    assert "r1/container-state.json" in _tar_names(tmp_path)
    _closing_lines_once(proc.stdout, tmp_path)


def _interrupt(tmp_path, signum, group, *extra, **fake):
    proc = subprocess.Popen(_args(tmp_path, *extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=tmp_path, env=_env(tmp_path, **fake), start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (tmp_path / "state" / "container").exists():
            time.sleep(0.05)
        assert (tmp_path / "state" / "container").exists(), "the stand-in container did not start"
        time.sleep(0.2)
        if group:
            os.killpg(proc.pid, signum)  # what Ctrl-C does: the whole foreground group
        else:
            os.kill(proc.pid, signum)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    return proc.returncode, out, err


@pytest.mark.parametrize("signum,group,code", [(signal.SIGINT, True, 130), (signal.SIGTERM, False, 143),
                                               (signal.SIGHUP, False, 129), (signal.SIGHUP, True, 129)])
def test_a_signal_stops_the_container_and_still_writes_the_tarball(tmp_path, signum, group, code):
    rc, out, err = _interrupt(tmp_path, signum, group)
    assert rc == code, out + err
    calls = _calls(tmp_path)
    assert "exec fn-rental pkill -INT -f tools.flash_next.rental.main" in calls
    assert not [c for c in calls if c.startswith(("stop", "kill"))], calls  # the orchestrator stopped in time
    assert not (tmp_path / "state" / "container").exists()
    _closing_lines_once(out, tmp_path)
    names = _tar_names(tmp_path)
    assert "r1/console.log" in names
    console = (tmp_path / "results" / "r1" / "console.log").read_text()
    assert "interrupted" in console
    assert "FAKE CONTAINER STOPPED" in console  # the client kept logging while the container went down


def test_a_container_that_ignores_the_stop_request_is_killed_after_a_bounded_wait(tmp_path):
    started = time.monotonic()
    rc, out, err = _interrupt(tmp_path, signal.SIGTERM, False, FAKE_INT="ignore", FAKE_STOP="ignore")
    assert rc == 143, out + err
    assert time.monotonic() - started < 40
    calls = _calls(tmp_path)
    order = [calls.index("exec fn-rental pkill -INT -f tools.flash_next.rental.main"),
             calls.index("stop -t 10 fn-rental"), calls.index("kill fn-rental")]
    assert order == sorted(order), calls
    assert not (tmp_path / "state" / "container").exists()
    _closing_lines_once(out, tmp_path)
    assert (tmp_path / "results" / "r1.tar.gz").is_file()


def test_no_orchestrator_yet_means_no_wait_for_it(tmp_path):
    # During the install steps there is no orchestrator to ask: go straight to docker stop.
    rc, out, err = _interrupt(tmp_path, signal.SIGINT, True, FAKE_INT="absent", FN_RENTAL_STOP_WAIT_S=600)
    assert rc == 130, out + err
    calls = _calls(tmp_path)
    assert "stop -t 10 fn-rental" in calls
    assert "kill fn-rental" not in calls
    _closing_lines_once(out, tmp_path)


def test_a_second_signal_does_not_run_the_end_path_twice(tmp_path):
    proc = subprocess.Popen(_args(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path,
                            env=_env(tmp_path, FAKE_INT="ignore", FN_RENTAL_STOP_WAIT_S=3), start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (tmp_path / "state" / "container").exists():
            time.sleep(0.05)
        time.sleep(0.2)
        os.kill(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not [c for c in _calls(tmp_path) if "pkill" in c]:
            time.sleep(0.05)  # the first signal is being handled once the orchestrator was asked to stop
        os.kill(proc.pid, signal.SIGINT)
        os.kill(proc.pid, signal.SIGHUP)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode == 143, out + err
    _closing_lines_once(out, tmp_path)
    assert _calls(tmp_path).count("stop -t 10 fn-rental") == 1


# --------------------------------------------------------------------------- the detached download


def test_download_only_dry_run(tmp_path):
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--no-disk-check", "--download-only", "--model",
                           str(tmp_path / "models" / "Qwen3.8-Flash-Next-FP8")],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == 0, proc.stderr
    flat = _flat(proc)
    assert flat.count("docker run") == 1
    assert "docker run -d --name fn-download" in flat
    assert f"{tmp_path}/models:/models" in flat
    assert "hf download" in flat and "Qwen/Qwen3.8-Flash-Next-FP8" in flat
    assert "/models/Qwen3.8-Flash-Next-FP8" in flat
    assert "--runtime=habana" not in flat  # the download holds no card: the smoke test can run next to it
    assert "container_entry.sh" not in flat
    assert not (tmp_path / "results").exists()


def test_download_only_uses_the_entry_scripts_repository_and_hub_version():
    run_text = SCRIPT.read_text()
    entry_text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    repo = re.search(r'^REPO_ID="([^"]+)"$', entry_text, re.M).group(1)
    assert 'REPO_ID="%s"' % repo in run_text
    hub = re.search(r"huggingface_hub==([0-9.]+)", entry_text).group(1)
    assert 'HF_HUB_VERSION="%s"' % hub in run_text


def test_download_only_keeps_the_free_space_check(tmp_path):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    df = bin_dir / "df"
    df.write_text("#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
                  "printf '/dev/vda1 96000000 95000000 1000 99%% /\\n'\n")
    df.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    proc = subprocess.run(["/bin/bash", str(SCRIPT), "--dry-run", "--download-only", "--model",
                           str(tmp_path / "models" / "X")],
                          capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)
    assert proc.returncode == 1, proc.stdout
    assert "GB free" in proc.stderr
    assert "docker run" not in proc.stdout


def test_download_only_starts_one_detached_container_and_returns(tmp_path):
    proc = subprocess.run(["/bin/bash", str(SCRIPT), "--no-disk-check", "--download-only", "--model",
                           str(tmp_path / "models" / "X")],
                          capture_output=True, text=True, cwd=tmp_path, env=_env(tmp_path), timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = [c for c in _calls(tmp_path) if c.startswith("run ")]
    assert len(runs) == 1 and runs[0].startswith("run -d --name fn-download ")
    assert "docker wait fn-download" in proc.stdout  # how to tell that it finished
    assert not (tmp_path / "results").exists()  # no run folder, no tarball


def test_download_only_refuses_a_second_download(tmp_path):
    proc = subprocess.run(["/bin/bash", str(SCRIPT), "--no-disk-check", "--download-only", "--model",
                           str(tmp_path / "models" / "X")],
                          capture_output=True, text=True, cwd=tmp_path, env=_env(tmp_path, FAKE_RUNNING="fn-download"),
                          timeout=60)
    assert proc.returncode == 1
    assert "fn-download" in proc.stderr
    assert not [c for c in _calls(tmp_path) if c.startswith("run ")]


def test_a_run_waits_for_the_download_container(tmp_path):
    proc = _run(tmp_path, FAKE_RUNNING="fn-download")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "fn-download" in proc.stderr and "docker wait fn-download" in proc.stderr
    assert not [c for c in _calls(tmp_path) if c.startswith("run ")]
