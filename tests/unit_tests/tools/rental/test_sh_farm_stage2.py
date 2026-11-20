# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests of the workstation wrapper farm.sh, stage 2 (farm sync, job, tiny, serve) against a
stand-in node: node-ssh runs the command here with HOME set to a node folder that holds the real
cardq.sh, farm_job.sh, farm_unpack.sh and build_track_image.sh; `docker` and `sudo` are stand-ins and
the job's `docker run` records its arguments and writes a file into the job's results folder.
Runs on CPU; on macOS the Perl stand-ins for flock(1) and setsid(1) of test_sh_cardq are used."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit_tests.tools.rental.test_sh_cardq import FLOCK_SHIM, SETSID_SHIM
from tests.unit_tests.tools.rental.test_sh_node_bringup import FAKES as NODE_FAKES
from tests.unit_tests.tools.rental.test_sh_node_bringup import PREPARE_HOST

REPO_ROOT = Path(__file__).resolve().parents[4]
FARM_SRC = REPO_ROOT / "tools" / "flash_next" / "rental" / "farm"

NODE_SSH = r"""#!/bin/bash
cd "$FAKE_NODE_HOME" || exit 255
exec env HOME="$FAKE_NODE_HOME" bash -c "$1"
"""
NODE_SCP = r"""#!/bin/bash
args=("$@"); dest="${args[$# - 1]}"; unset 'args[$# - 1]'
cp "${args[@]}" "$FAKE_NODE_HOME/${dest#:}"
"""
FAKE_SUDO = r"""#!/bin/bash
[ "${1:-}" = -n ] && shift
exec "$@"
"""
FAKE_DOCKER = r"""#!/bin/bash
echo "$*" >> "$FAKE_STATE/docker.log"
case "$1" in
  image)
    [ "$2" = inspect ] && { [ -e "$FAKE_STATE/image-$3" ] || exit 1; }
    ;;
  commit)
    for a in "$@"; do last="$a"; done
    n=$(($(cat "$FAKE_STATE/image-seq" 2>/dev/null || echo 0) + 1)); echo "$n" > "$FAKE_STATE/image-seq"
    printf '%06d\n' "$n" > "$FAKE_STATE/image-$last"
    ;;
  tag) [ -e "$FAKE_STATE/image-$3" ] || touch "$FAKE_STATE/image-$3" ;;
  images)
    # docker images REPO --format '{{.Tag}} {{.CreatedAt}}': the newest has the highest counter
    for f in "$FAKE_STATE"/image-"$2":*; do
      [ -e "$f" ] || continue
      echo "${f##*:} 2026-01-01 $(cat "$f")"
    done
    ;;
  rmi) rm -f "$FAKE_STATE/image-$2" ;;
  info) echo /var/lib/docker ;;
  run)
    case " $* " in *" -d "*) exit 0 ;; esac
    n=$(ls "$FAKE_STATE" | grep -c '^run-' || true)
    out="$FAKE_STATE/run-$n"
    : > "$out"
    results=""; run=""
    prev=""
    for a in "$@"; do
      printf '%s\n' "$a" >> "$out"
      if [ "$prev" = -v ]; then case "$a" in *:/results) results="${a%:/results}" ;; esac; fi
      if [ "$prev" = -e ]; then case "$a" in FARM_RUN_ID=*) run="${a#FARM_RUN_ID=}" ;; esac; fi
      prev="$a"
    done
    mkdir -p "$results/$run" && echo hello > "$results/$run/hello.txt"
    echo "container output: done"
    exit "${FAKE_JOB_RC:-0}"
    ;;
esac
exit 0
"""
STUB_BUNDLE = r"""#!/bin/bash
# stand-in for the track's make_rental_bundle.sh: a bundle of HEAD with BUNDLE.txt and run_rental.sh
out="$2"; sha="$(git rev-parse HEAD)"
stage="$(mktemp -d)"; mkdir -p "$stage/plugin"
printf 'plugin sha: %s\n' "$sha" > "$stage/BUNDLE.txt"; echo '#!/bin/bash' > "$stage/run_rental.sh"
mkdir -p "$out"; tar czf "$out/flash-next-rental-${sha:0:12}.tar.gz" -C "$stage" BUNDLE.txt run_rental.sh plugin
echo "bundle: $out/flash-next-rental-${sha:0:12}.tar.gz"
"""


class Rig:

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.g = tmp_path / "work"
        self.node = tmp_path / "node"
        self.state = tmp_path / "state"
        for d in (self.g / "bin", self.node / "farm" / "tools", self.state, tmp_path / "fakebin"):
            d.mkdir(parents=True)
        fakebin = tmp_path / "fakebin"
        for name, text in (("docker", FAKE_DOCKER), ("sudo", FAKE_SUDO)):
            (fakebin / name).write_text(text)
            (fakebin / name).chmod(0o755)
        for name, text in (("flock", FLOCK_SHIM), ("setsid", SETSID_SHIM)):
            if shutil.which(name) is None:
                (fakebin / name).write_text(text)
                (fakebin / name).chmod(0o755)
        for name, text in (("node-ssh", NODE_SSH), ("node-scp", NODE_SCP)):
            (self.g / "bin" / name).write_text(text)
            (self.g / "bin" / name).chmod(0o755)
        for f in ("cardq.sh", "farm_job.sh", "farm_unpack.sh", "build_track_image.sh"):
            shutil.copy(FARM_SRC / f, self.node / "farm" / f)
            (self.node / "farm" / f).chmod(0o755)
        for f in ("serve_probe.sh", "decode_scaling.py", "farm_tiny.py"):
            shutil.copy(FARM_SRC / f, self.node / "farm" / "tools" / f)
        (self.state / "image-farm-base:latest").touch()
        # the track's worktree: a git repo with one commit and the stand-in bundle script
        wt = self.g / "wt" / "kf"
        (wt / "tools" / "flash_next" / "rental").mkdir(parents=True)
        (wt / "tools" / "flash_next" / "rental" / "make_rental_bundle.sh").write_text(STUB_BUNDLE)
        git = ["git", "-C", str(wt)]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "."], check=True)
        subprocess.run(git + ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x"], check=True)
        self.sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CARDQ_", "FARM_", "BASH_ENV"))}
        self.env.update({
            "PATH": "%s:%s" % (fakebin, os.environ.get("PATH", "/usr/bin:/bin")),
            "FARM_HOME": str(self.g), "FAKE_NODE_HOME": str(self.node), "FAKE_STATE": str(self.state),
            "CARDQ_POLL_S": "0.1", "CARDQ_GRACE_S": "1", "CARDQ_MARGIN_S": "0", "CARDQ_SUDO": "",
            "CARDQ_ALLOW_ROOT": "1", "CARDQ_MODULE_MAP": "3,0,1,6,7,4,5,2",
            "FARM_RECIPE_ROOT": str(tmp_path / "shm"), "FARM_DOCKER": "docker",
        })

    def farm(self, *args, timeout=180, **env):
        return subprocess.run(["bash", str(FARM_SRC / "farm.sh"), *args], capture_output=True, text=True,
                              env=dict(self.env, **env), timeout=timeout, stdin=subprocess.DEVNULL)

    def synced(self):
        b = self.node / "farm" / "kf" / "b" / self.sha
        b.mkdir(parents=True)
        (b / "BUNDLE.txt").write_text("plugin sha: %s\n" % self.sha)
        (self.node / "farm" / "kf" / "current").symlink_to("b/" + self.sha)
        (self.state / ("image-farm-kf:%s" % self.sha)).touch()

    def runs(self):
        return [p.read_text().splitlines() for p in sorted(self.state.glob("run-*"))]


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def after(words, marker):
    return words[words.index(marker) + 1:]


def env_of(words):
    return [words[i + 1] for i, w in enumerate(words) if w == "-e"]


def test_farm_job_runs_the_command_on_the_granted_card_and_fetches_its_results(rig):
    rig.synced()
    proc = rig.farm("job", "kf", "hc-op", "any:1", "15m", "--no-sync", "--env", "VLLM_X={\"a\": 1}", "--",
                    "python", "-m", "pytest", "-q", "tests/x y.py", "-k", "it's")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (words,) = rig.runs()
    assert after(words, "farm-kf:%s" % rig.sha) == ["python", "-m", "pytest", "-q", "tests/x y.py", "-k", "it's"]
    e = env_of(words)
    assert "HABANA_VISIBLE_DEVICES=4" in e and "HABANA_VISIBLE_MODULES=7" in e  # any:1: the top module, 7
    assert 'VLLM_X={"a": 1}' in e and "FARM_PORT_BASE=8500" in e
    card = rig.g / "reports" / "kf" / "card"
    assert (card / "00001-hc-op" / "hello.txt").read_text() == "hello\n"
    log = (card / "00001-hc-op.log").read_text()
    assert "container output: done" in log and "exit code 0" in log
    assert "farm: job 00001 ended with exit code 0" in proc.stdout


def test_farm_job_returns_the_jobs_exit_code(rig):
    rig.synced()
    proc = rig.farm("job", "kf", "fails", "any:1", "5m", "--no-sync", "--", "false", FAKE_JOB_RC="3")
    assert proc.returncode == 3
    assert "ended with exit code 3" in proc.stdout


def test_farm_serve_writes_the_server_job_with_its_options(rig):
    rig.synced()
    proc = rig.farm("serve", "kf", "real4", "any:4", "45m", "--no-sync", "--env", "VLLM_HPU_HC_FUSED=0", "--mtp", "3",
                    "--levels", "1,8", "--ctx", "8:4000", "--", "--max-num-batched-tokens", "4096")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (words,) = rig.runs()
    assert after(words, "farm-kf:%s" % rig.sha) == ["bash", "/farm/serve_probe.sh", "--mtp", "3", "--levels", "1,8",
                                                     "--ctx", "8:4000", "--", "--max-num-batched-tokens", "4096"]
    e = env_of(words)
    assert "HABANA_VISIBLE_DEVICES=0,1,2,7" in e and "HABANA_VISIBLE_MODULES=0,1,2,3" in e
    assert "VLLM_HPU_HC_FUSED=0" in e and "FARM_NCARDS=4" in e and "FARM_PORT_BASE=8100" in e


def test_farm_serve_passes_the_min_tokens_options_to_the_probe(rig):
    rig.synced()
    proc = rig.farm("serve", "kf", "real4-short", "any:4", "30m", "--no-sync", "--levels", "1,32",
                    "--no-mintok-check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = rig.farm("serve", "kf", "real4-mask", "any:4", "30m", "--no-sync", "--min-tokens")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    short, mask = rig.runs()
    assert after(short, "farm-kf:%s" % rig.sha) == ["bash", "/farm/serve_probe.sh", "--levels", "1,32",
                                                     "--no-mintok-check", "--"]
    assert after(mask, "farm-kf:%s" % rig.sha) == ["bash", "/farm/serve_probe.sh", "--min-tokens", "--"]


def test_farm_harvest_copies_every_tracks_results_and_the_queue(rig):
    rig.synced()
    proc = rig.farm("job", "kf", "hc-op", "any:1", "15m", "--no-sync", "--", "true")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    other = rig.node / "farm" / "pf" / "results" / "00009-x"  # a track whose agent is gone: nobody fetched it
    other.mkdir(parents=True)
    (other / "scaling.json").write_text("{}")
    (other / "model.safetensors").write_text("weights")
    (other / "ckpt-tiny").mkdir()
    (other / "ckpt-tiny" / "config.json").write_text("{}")
    images = rig.node / "farm" / "kf" / "images" / rig.sha
    images.mkdir(parents=True, exist_ok=True)
    (images / "install.log").write_text("ok")
    dest = rig.tmp / "harvest"
    proc = rig.farm("harvest", str(dest))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "kf" / "results" / "00001-hc-op" / "hello.txt").read_text() == "hello\n"
    assert (dest / "pf" / "results" / "00009-x" / "scaling.json").is_file()
    assert (dest / "kf" / "images" / rig.sha / "install.log").is_file()
    assert "00001\tkf\thc-op" in (dest / "q" / "ledger.tsv").read_text()
    assert "container output: done" in (dest / "q" / "jobs" / "00001" / "log").read_text()
    assert not (dest / "pf" / "results" / "00009-x" / "model.safetensors").exists()  # checkpoints stay behind
    assert not (dest / "pf" / "results" / "00009-x" / "ckpt-tiny").exists()
    assert "files on the node now" in proc.stdout and str(dest) in proc.stdout


def test_farm_health_and_status_print_what_the_queue_prints(rig):
    rig.synced()
    proc = rig.farm("health")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "habanalabs kernel lines since boot" in proc.stdout
    proc = rig.farm("status")
    assert proc.returncode == 0 and "waiting (in the order they start):" in proc.stdout


def _fresh_node(rig):
    """The rig as a freshly started node: host tools as stand-ins, the farm's own scripts as wt/card-farm."""
    fakebin = rig.tmp / "fakebin"
    for name, text in NODE_FAKES.items():
        if name in ("sudo", "docker"):
            continue  # the rig's own
        (fakebin / name).write_text(text)
        (fakebin / name).chmod(0o755)
    (fakebin / "ssh-keyscan").write_text('#!/bin/bash\nfor a in "$@"; do h="$a"; done\necho "$h ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyOfTheNewNode"\n')
    (fakebin / "ssh-keyscan").chmod(0o755)
    (rig.g / "wt" / "card-farm" / "tools" / "flash_next" / "rental").mkdir(parents=True)
    (rig.g / "wt" / "card-farm" / "tools" / "flash_next" / "rental" / "farm").symlink_to(FARM_SRC)
    (rig.node / "rental").mkdir()
    (rig.node / "rental" / "prepare_host.sh").write_text(PREPARE_HOST)  # the stub bundle has none
    (rig.tmp / "os-release").write_text('ID=ubuntu\nVERSION_ID="22.04"\n')
    (rig.tmp / "meminfo").write_text("MemTotal:       970000000 kB\n")
    (rig.g / "NODE").write_text("9.9.9.9\n")  # the node of the last rental
    (rig.g / "node_known_hosts").write_text("9.9.9.9 ssh-ed25519 OLDKEY\n")
    return dict(BRINGUP_MODELS_DIR=str(rig.tmp / "models"), BRINGUP_OS_RELEASE=str(rig.tmp / "os-release"),
                BRINGUP_MEMINFO=str(rig.tmp / "meminfo"), BRINGUP_MIN_ROOT_GB="0", FARM_BRINGUP_POLL_S="0.2")


def test_farm_bringup_takes_a_fresh_node_to_a_queue_that_answers(rig):
    env = _fresh_node(rig)
    proc = rig.farm("bringup", "1.2.3.4", "--commit", "HEAD", "--from", "kf", "--no-download", **env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (rig.g / "NODE").read_text() == "1.2.3.4\n"
    assert (rig.g / "node_known_hosts").read_text().startswith("1.2.3.4 ssh-ed25519 AAAA")  # the old pin is gone
    full = subprocess.run(["git", "-C", str(rig.g / "wt" / "kf"), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert (rig.node / "rental" / "BUNDLE.txt").read_text() == "plugin sha: %s\n" % full
    for f in ("cardq.sh", "farm_job.sh", "node_bringup.sh", "build_base.sh", "tools/serve_probe.sh"):
        assert (rig.node / "farm" / f).is_file(), f
    out = proc.stdout
    for line in ("STEP 2 host: ok", "STEP 3 ram-disk: ok", "STEP 5 base-image: ok", "STEP 7 cards: ok",
                 "  0-3 | 0,1,2,7 | 0-75", "BRINGUP-DONE", "farm: the node is up."):
        assert line in out, line
    assert "waiting (in the order they start):" in out  # the queue's status at the end
    # the same command again: the address, the key and the bundle stay, the finished steps are skipped
    proc = rig.farm("bringup", "1.2.3.4", "--commit", "HEAD", "--from", "kf", "--no-download", **env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "host key kept" in proc.stdout and "is the bundle of %s already" % rig.sha in proc.stdout
    assert "is mounted already" in proc.stdout and "BRINGUP-DONE" in proc.stdout


def test_farm_bringup_stops_at_a_failed_step(rig):
    env = _fresh_node(rig)
    proc = rig.farm("bringup", "1.2.3.4", "--commit", "HEAD", "--from", "kf", "--no-download",
                    FAKE_PREPARE_RC="1", **env)
    assert proc.returncode != 0
    assert "BRINGUP-FAILED at step 2 (host)" in proc.stdout and "the bring-up failed" in proc.stderr
    proc = rig.farm("bringup", "not an address")
    assert proc.returncode != 0 and "usage: farm bringup ADDRESS" in proc.stderr


def test_farm_tiny_runs_the_farms_tiny_runner(rig):
    rig.synced()
    proc = rig.farm("tiny", "kf", "tiny-tp2", "any:2", "20m", "--no-sync", "--gdn", "hc", "--full-warmup")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (words,) = rig.runs()
    assert after(words, "farm-kf:%s" % rig.sha) == ["python", "/farm/farm_tiny.py", "--gdn", "hc", "--full-warmup"]
    assert "HABANA_VISIBLE_DEVICES=3,4" in env_of(words)  # any:2: modules 6,7


def test_farm_sync_unpacks_the_commit_and_builds_its_image_once(rig):
    proc = rig.farm("sync", "kf")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    b = rig.node / "farm" / "kf" / "b" / rig.sha
    assert (b / "BUNDLE.txt").is_file() and os.readlink(rig.node / "farm" / "kf" / "current") == "b/" + rig.sha
    assert (rig.state / ("image-farm-kf:%s" % rig.sha)).exists()
    log = (rig.state / "docker.log").read_text()
    assert "container_entry.sh --install-only" in log and "commit" in log
    assert "IMAGE-READY farm-kf:%s" % rig.sha in proc.stdout
    assert "synced at %s" % rig.sha in proc.stdout
    proc = rig.farm("sync", "kf")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "on the node already" in proc.stdout and "it was there already" in proc.stdout


def test_farm_job_syncs_first_by_default(rig):
    proc = rig.farm("job", "kf", "first", "any:1", "5m", "--", "true")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMAGE-READY" in proc.stdout
    (words,) = rig.runs()
    assert "farm-kf:%s" % rig.sha in words


@pytest.mark.parametrize("args, text", [
    (["job", "kf", "x", "any:1", "5m", "--no-sync"], "the command follows --"),
    (["job", "kf", "x", "any:1", "soon", "--", "true"], "MAX_HOLD"),
    (["serve", "kf", "x", "any:4", "5m", "--no-sync", "--frob"], "unknown option --frob"),
    (["tiny", "kf", "x", "any:4", "5m", "--no-sync", "--", "true"], "takes no command"),
    (["job", "KF", "x", "any:1", "5m", "--", "true"], "TRACK"),
    (["serve", "kf", "x"], "usage: farm serve"),
])
def test_bad_calls_are_refused_before_anything_is_queued(rig, args, text):
    rig.synced()
    proc = rig.farm(*args)
    assert proc.returncode != 0 and text in proc.stderr
    assert not (rig.node / "farm" / "q" / "jobs").exists() or not list((rig.node / "farm" / "q" / "jobs").iterdir())


def _commit(rig, text):
    wt = rig.g / "wt" / "kf"
    (wt / "note.txt").write_text(text)
    git = ["git", "-C", str(wt)]
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", text], check=True)
    return subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]


def test_old_images_go_but_never_the_image_of_a_job_that_waits(rig):
    # A track that syncs four commits while its first job still waits must not lose that job's image: the
    # job would rebuild it at its start, three minutes with the cards in its hands.
    first = rig.sha
    assert rig.farm("sync", "kf", FARM_KEEP_IMAGES="2").returncode == 0
    waiting = rig.node / "farm" / "q" / "jobs" / "00007"
    waiting.mkdir(parents=True)
    (waiting / "state").write_text("waiting\n")
    (waiting / "track").write_text("kf\n")
    (waiting / "cmd.sh").write_text("exec ~/farm/farm_job.sh --sha %s -- python -m pytest -q x.py\n" % first)
    shas = [first]
    for i in range(4):
        shas.append(_commit(rig, "step %d" % i))
        proc = rig.farm("sync", "kf", FARM_KEEP_IMAGES="2")
        assert proc.returncode == 0, proc.stdout + proc.stderr
    have = sorted(p.name.split(":")[1] for p in rig.state.glob("image-farm-kf:*") if not p.name.endswith(":latest"))
    assert have == sorted([first, shas[-2], shas[-1]])  # the two newest, and the one the waiting job runs from
    assert "kept for a waiting or running job: farm-kf:%s" % first in proc.stdout
    # a bundle goes with its image (270 MB each on the node: 45 of them in two hours filled 12 GB)
    bundles = sorted(p.name for p in (rig.node / "farm" / "kf" / "b").iterdir())
    assert bundles == sorted([first, shas[-2], shas[-1]])
    # the job has ended: the next sync lets its image go
    (waiting / "state").write_text("ended\n")
    shas.append(_commit(rig, "last"))
    assert rig.farm("sync", "kf", FARM_KEEP_IMAGES="2").returncode == 0
    have = sorted(p.name.split(":")[1] for p in rig.state.glob("image-farm-kf:*") if not p.name.endswith(":latest"))
    assert have == sorted(shas[-2:])
    assert sorted(p.name for p in (rig.node / "farm" / "kf" / "b").iterdir()) == sorted(shas[-2:])


def test_a_commit_named_in_a_tracks_own_job_script_keeps_its_image_and_bundle(rig):
    # Some jobs read their steps from a file of the track when they start (~/farm/<track>/job/arm.sh): the
    # commit that file names must stay although no cmd.sh names it.
    first = rig.sha
    assert rig.farm("sync", "kf", FARM_KEEP_IMAGES="1").returncode == 0
    job_dir = rig.node / "farm" / "kf" / "job"
    job_dir.mkdir()
    (job_dir / "arm.sh").write_text("exec ~/farm/farm_job.sh --sha %s -- bash /farm/serve_probe.sh\n" % first)
    waiting = rig.node / "farm" / "q" / "jobs" / "00009"
    waiting.mkdir(parents=True)
    (waiting / "state").write_text("waiting\n")
    (waiting / "track").write_text("kf\n")
    (waiting / "cmd.sh").write_text('bash "$HOME/farm/kf/job/arm.sh" off\n')
    shas = [first] + [_commit(rig, "s%d" % i) for i in range(1)]
    assert rig.farm("sync", "kf", FARM_KEEP_IMAGES="1").returncode == 0
    shas.append(_commit(rig, "s2"))
    proc = rig.farm("sync", "kf", FARM_KEEP_IMAGES="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    have = sorted(p.name.split(":")[1] for p in rig.state.glob("image-farm-kf:*") if not p.name.endswith(":latest"))
    assert have == sorted([first, shas[-1]])
    # the bundle of the commit that was `current` while this build ran stays one sync longer
    assert sorted(p.name for p in (rig.node / "farm" / "kf" / "b").iterdir()) == sorted([first, shas[-2], shas[-1]])
    # no job of the track waits or runs any more: the file's commit goes with the next sync
    (waiting / "state").write_text("ended\n")
    shas.append(_commit(rig, "s3"))
    assert rig.farm("sync", "kf", FARM_KEEP_IMAGES="1").returncode == 0
    have = sorted(p.name.split(":")[1] for p in rig.state.glob("image-farm-kf:*") if not p.name.endswith(":latest"))
    assert have == [shas[-1]]
    assert sorted(p.name for p in (rig.node / "farm" / "kf" / "b").iterdir()) == sorted(shas[-2:])


def test_an_image_build_runs_on_the_build_cpus_and_is_written_to_the_builds_log(rig):
    proc = rig.farm("sync", "kf", CARDQ_HOST_CPUS="152")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = [ln for ln in (rig.state / "docker.log").read_text().splitlines() if ln.startswith("run -d --name farm-img-kf")]
    assert len(run) == 1 and "--cpuset-cpus 18,37,56,75,94,113,132,151" in run[0] and "--cpus " not in run[0]
    log = (rig.node / "farm" / "q" / "builds.log").read_text().split()
    assert log[0] == "START" and log[2:4] == ["kf", rig.sha]
    assert log[4] == "END" and log[6:9] == ["kf", rig.sha, "0"] and int(log[5]) >= int(log[1])
    # a host with too few CPUs for a build set: the old limit by count
    sha2 = _commit(rig, "second")
    proc = rig.farm("sync", "kf", CARDQ_HOST_CPUS="16")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = [ln for ln in (rig.state / "docker.log").read_text().splitlines() if ln.startswith("run -d --name farm-img-kf-%s" % sha2)]
    assert len(run) == 1 and "--cpus 24" in run[0] and "--cpuset-cpus" not in run[0]


def test_image_builds_wait_for_a_free_node_slot(rig):
    import fcntl
    slots = rig.node / "farm" / "images-slots"
    slots.mkdir(parents=True)
    held = open(slots / "slot1.lock", "a")
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        proc = rig.farm("sync", "kf", FARM_IMAGE_SLOTS="1", timeout=8)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        assert "waiting for a slot" in out
    else:
        raise AssertionError("the build did not wait: %s" % (proc.stdout + proc.stderr))
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
    proc = rig.farm("sync", "kf", FARM_IMAGE_SLOTS="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMAGE-READY" in proc.stdout
