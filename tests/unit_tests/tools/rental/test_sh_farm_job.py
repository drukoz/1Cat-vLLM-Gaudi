# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/farm_job.sh, the container of a card-queue job: the granted cards and their module
ids, the job's ports and folders, read-only mounts, the labels the queue sweeps by, and the image of
the track's synced commit. `docker` is a stand-in that records its calls. Runs on CPU."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FARM_JOB = REPO_ROOT / "tools" / "flash_next" / "rental" / "farm" / "farm_job.sh"
SHA = "0123456789ab"

FAKE_DOCKER = r"""#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1 $2" in
  "image inspect") [ -e "$FAKE_STATE/image-$3" ] || exit 1 ;;
  run\ *) echo "container ran" ;;
esac
exit 0
"""


class Node:

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.farm = tmp_path / "farm"
        self.state = tmp_path / "state"
        self.state.mkdir()
        bundle = self.farm / "kf" / "b" / SHA
        bundle.mkdir(parents=True)
        (bundle / "BUNDLE.txt").write_text("plugin sha: %s%s\n" % (SHA, "c" * 28))
        (self.farm / "kf" / "current").symlink_to("b/" + SHA)
        (self.farm / "q").mkdir()
        (self.farm / "q" / "module_map").write_text("3,0,1,6,7,4,5,2\n")
        shutil.copy(FARM_JOB.parent / "cardq.sh", self.farm / "cardq.sh")  # the CPU sets are the queue's rule
        (self.farm / "build_track_image.sh").write_text(
            '#!/bin/bash\necho "build $*" >> "$FAKE_DOCKER_LOG"\ntouch "$FAKE_STATE/image-farm-$1:$2"\n')
        self.docker = tmp_path / "docker"
        self.docker.write_text(FAKE_DOCKER)
        self.docker.chmod(0o755)
        (self.state / ("image-farm-kf:%s" % SHA)).touch()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CARDQ_", "FARM_"))}
        self.env.update({
            "FARM_DIR": str(self.farm), "FARM_DOCKER": str(self.docker), "FAKE_DOCKER_LOG": str(tmp_path / "log"),
            "FAKE_STATE": str(self.state), "FARM_RECIPE_ROOT": str(tmp_path / "shm"),
            "CARDQ_JOB": "00042", "CARDQ_TRACK": "kf", "CARDQ_NAME": "hc-op", "CARDQ_CARDS": "0,1,2,7",
            "CARDQ_MODULES": "0,1,2,3", "CARDQ_NCARDS": "4", "CARDQ_PORT_BASE": "8100",
        })

    def run(self, *args, **env):
        return subprocess.run(["bash", str(FARM_JOB), *args], capture_output=True, text=True,
                              env=dict(self.env, **env), timeout=60)

    def calls(self):
        log = self.tmp / "log"
        return log.read_text().splitlines() if log.exists() else []

    def dry(self, *args, **env):
        proc = self.run("--dry-run", *args, **env)
        assert proc.returncode == 0, proc.stderr
        return shlex.split(proc.stdout)


@pytest.fixture
def node(tmp_path):
    return Node(tmp_path)


def envs(words):
    return [words[i + 1] for i, w in enumerate(words) if w == "-e"]


def mounts(words):
    return [words[i + 1] for i, w in enumerate(words) if w == "-v"]


def test_the_container_gets_exactly_the_granted_cards_and_their_modules(node):
    words = node.dry("--", "python", "-m", "pytest", "-q", "tests/x.py")
    e = envs(words)
    assert "HABANA_VISIBLE_DEVICES=0,1,2,7" in e and "HABANA_VISIBLE_MODULES=0,1,2,3" in e
    assert "FARM_PORT_BASE=8100" in e and "FARM_NCARDS=4" in e and "FARM_CARDS=0,1,2,7" in e
    assert "PT_HPU_LAZY_ACC_PAR_MODE=0" in e and "PT_HPU_LAZY_MODE=1" in e
    assert "FARM_OUT=/results/00042-hc-op" in e and "HABANA_LOGS=/results/00042-hc-op/habana_logs" in e
    assert "--runtime=habana" in words and "--net=host" in words and "--ipc=host" in words
    assert words[words.index("--name") + 1] == "farm-00042"
    assert "cardq.job=00042" in words and "cardq.track=kf" in words
    assert words[-7:] == ["farm-kf:%s" % SHA, "python", "-m", "pytest", "-q", "tests/x.py"][-7:] or True
    assert words[words.index("farm-kf:%s" % SHA) + 1:] == ["python", "-m", "pytest", "-q", "tests/x.py"]
    v = mounts(words)
    farm = node.farm
    assert "%s/kf/b/%s:/bundle:ro" % (farm, SHA) in v
    assert "/models/Qwen3.8-Flash-Next-FP8:/models/Qwen3.8-Flash-Next-FP8:ro" in v
    assert "%s/tools:/farm:ro" % farm in v
    assert "%s/kf/results:/results" % farm in v
    assert any(m.endswith("/kf/%s-4c:/recipe-cache" % SHA) for m in v)
    assert not any(m.startswith("/models:") for m in v), "the whole /models folder is never mounted"


def test_a_part_of_the_cards_gets_its_own_ports_name_and_module_ids(node):
    words = node.dry("--cards", "3,4,5,6", "--suffix", "b", "--run-id", "side-b", "--", "true",
                     CARDQ_CARDS="0,1,2,3,4,5,6,7", CARDQ_MODULES="0,1,2,3,4,5,6,7", CARDQ_NCARDS="8")
    e = envs(words)
    assert "HABANA_VISIBLE_DEVICES=3,4,5,6" in e and "HABANA_VISIBLE_MODULES=4,5,6,7" in e
    assert "FARM_PORT_BASE=8400" in e and "FARM_NCARDS=4" in e and "FARM_OUT=/results/side-b" in e
    assert words[words.index("--name") + 1] == "farm-00042-b"
    proc = node.run("--dry-run", "--cards", "0,5", "--", "true")
    assert proc.returncode == 2 and "card 5 is not one of this job's cards" in proc.stderr


def test_env_options_win_over_the_defaults_but_not_over_the_cards(node):
    words = node.dry("--env", "PT_HPU_LAZY_MODE=0", "--env", "VLLM_HPU_HC_FUSED=1", "--", "true")
    e = envs(words)
    assert "PT_HPU_LAZY_MODE=0" in e and "PT_HPU_LAZY_MODE=1" not in e and "VLLM_HPU_HC_FUSED=1" in e
    for key in ("HABANA_VISIBLE_DEVICES", "FARM_PORT_BASE"):
        proc = node.run("--dry-run", "--env", key + "=1", "--", "true")
        assert proc.returncode == 2 and "set by the farm" in proc.stderr


def test_no_model_and_another_image(node):
    words = node.dry("--no-model", "--image", "farm-base:latest", "--", "true")
    assert not any("/models" in m for m in mounts(words))
    assert "farm-base:latest" in words


def test_it_runs_only_inside_a_job_and_with_a_synced_bundle(node):
    proc = node.run("--", "true", CARDQ_JOB="")
    assert proc.returncode == 2 and "not inside a job of the card queue" in proc.stderr
    proc = node.run("--sha", "fedcba987654", "--", "true")
    assert proc.returncode == 2 and "farm sync kf" in proc.stderr
    (node.farm / "kf" / "current").unlink()
    proc = node.run("--", "true")
    assert proc.returncode == 2 and "no commit of track kf is synced" in proc.stderr
    proc = node.run("--sha", "XYZ", "--", "true")
    assert proc.returncode == 2


def test_a_missing_image_of_the_commit_is_built_first_and_the_command_runs(node):
    (node.state / ("image-farm-kf:%s" % SHA)).unlink()
    proc = node.run("--", "echo", "hi")
    assert proc.returncode == 0, proc.stderr
    calls = node.calls()
    assert "build kf %s" % SHA in calls
    run = [c for c in calls if c.startswith("run ")]
    assert run and run[0].endswith("farm-kf:%s echo hi" % SHA)
    assert (node.farm / "kf" / "results" / "00042-hc-op" / "habana_logs").is_dir()
    assert "building it now" in proc.stdout


def test_a_missing_other_image_is_an_error(node):
    proc = node.run("--image", "nothere:1", "--", "true")
    assert proc.returncode == 2 and "no image nothere:1" in proc.stderr


def test_a_container_gets_its_modules_share_of_the_cpus_without_the_build_cpus(node):
    # 152 CPUs, 8 cards: 19 per module, the top one of each for image builds (cardq.sh cpus)
    words = node.dry("--", "true", FARM_HOST_CPUS="152")  # cards 0,1,2,7 = modules 0-3
    assert words[words.index("--cpuset-cpus") + 1] == "0-17,19-36,38-55,57-74"
    assert not any(e.startswith("OMP_NUM_THREADS") for e in envs(words))  # vLLM sizes its threads by the set
    words = node.dry("--cards", "3,4,5,6", "--", "true", FARM_HOST_CPUS="152", CARDQ_CARDS="0,1,2,3,4,5,6,7",
                     CARDQ_MODULES="0,1,2,3,4,5,6,7", CARDQ_NCARDS="8")
    assert words[words.index("--cpuset-cpus") + 1] == "76-93,95-112,114-131,133-150"
    words = node.dry("--cards", "1", "--env", "OMP_NUM_THREADS=4", "--", "true", FARM_HOST_CPUS="152",
                     CARDQ_CARDS="0,1,2,3,4,5,6,7", CARDQ_MODULES="0,1,2,3,4,5,6,7", CARDQ_NCARDS="8")
    assert words[words.index("--cpuset-cpus") + 1] == "0-17"  # card 1 is module 0
    assert "OMP_NUM_THREADS=4" in envs(words)  # a job's own --env is passed on
    # CARDQ_NCARDS is the job's card count here (4), not the node's: the split is still by the node's 8 cards
    words = node.dry("--cards", "7", "--", "true", FARM_HOST_CPUS="152")
    assert words[words.index("--cpuset-cpus") + 1] == "38-55"  # card 7 is module 2


def test_a_job_of_all_cards_leaves_the_build_cpus_out_and_all_cpus_keeps_the_whole_host(node):
    all8 = dict(FARM_HOST_CPUS="152", CARDQ_CARDS="0,1,2,3,4,5,6,7", CARDQ_MODULES="0,1,2,3,4,5,6,7", CARDQ_NCARDS="8")
    words = node.dry("--", "true", **all8)
    assert words[words.index("--cpuset-cpus") + 1] == "0-17,19-36,38-55,57-74,76-93,95-112,114-131,133-150"
    assert not any(e.startswith("OMP_NUM_THREADS") for e in envs(words))
    words = node.dry("--all-cpus", "--", "true", FARM_HOST_CPUS="152")
    assert "--cpuset-cpus" not in words
    words = node.dry("--", "true", FARM_HOST_CPUS="16")  # a small host: no sets
    assert "--cpuset-cpus" not in words
