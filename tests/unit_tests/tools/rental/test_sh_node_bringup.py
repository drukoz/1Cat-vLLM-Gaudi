# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/node_bringup.sh (a freshly started rented node, from bare Ubuntu to a card queue that takes
jobs) against stand-ins for the host tools: lspci, hl-smi, sudo, docker, mount, mountpoint, nproc and uvx. The
rental bundle's prepare_host.sh and the farm's build_base.sh are stand-ins that record their call; cardq.sh
is the real one. Runs on CPU."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit_tests.tools.rental.test_sh_cardq import FLOCK_SHIM, SETSID_SHIM

REPO_ROOT = Path(__file__).resolve().parents[4]
FARM_SRC = REPO_ROOT / "tools" / "flash_next" / "rental" / "farm"
BRINGUP = FARM_SRC / "node_bringup.sh"

FAKES = {
    "lspci": '#!/bin/bash\nfor i in $(seq 1 "${FAKE_PCI_CARDS:-8}"); do echo "0$i:00.0 Processing accelerators: Habana Labs Ltd. Device 1020"; done\n',
    # hl-smi exists only after prepare_host.sh "installed the driver" (the marker file)
    "hl-smi": r'''#!/bin/bash
[ -e "$FAKE_STATE/driver" ] || exit 127
case "$*" in
  *index,module_id*) i=0; for m in ${FAKE_MODULES:-3 0 1 6 7 4 5 2}; do echo "$i, $m"; i=$((i+1)); done ;;
  *index,bus_id*) for i in 0 1 2 3 4 5 6 7; do printf '%d, 0000:%02x:00.0\n' $i $((10+i)); done ;;
  *memory*) for i in 0 1 2 3 4 5 6 7; do echo "$i, 768 MiB, 98304 MiB"; done ;;
  *index*) for i in 0 1 2 3 4 5 6 7; do echo "$i"; done ;;
esac
''',
    "sudo": '#!/bin/bash\n[ "${1:-}" = -n ] && shift\nexec "$@"\n',
    "docker": '#!/bin/bash\necho "docker $*" >>"$FAKE_STATE/calls"\n[ "$1" = info ] && [ -e "$FAKE_STATE/driver" ] && echo " Runtimes: habana runc"\nexit 0\n',
    "mount": '#!/bin/bash\necho "mount $*" >>"$FAKE_STATE/calls"\ntouch "$FAKE_STATE/mounted"\n',
    "mountpoint": '#!/bin/bash\n[ -e "$FAKE_STATE/mounted" ]\n',
    "chown": '#!/bin/bash\necho "chown $*" >>"$FAKE_STATE/calls"\n',
    "nproc": '#!/bin/bash\necho 152\n',
}
PREPARE_HOST = '#!/bin/bash\necho "prepare_host $*" >>"$FAKE_STATE/calls"\n[ "${FAKE_PREPARE_RC:-0}" = 0 ] || { echo "apt failed"; exit 1; }\ntouch "$FAKE_STATE/driver"\necho "host ready"\n'
BUILD_BASE = '#!/bin/bash\necho "build_base $*" >>"$FAKE_STATE/calls"\necho "BASE-IMAGE-DONE farm-base:abc"\n'
# uvx: "downloads" the checkpoint: one sparse file of FAKE_CKPT_GB in the --local-dir
UVX = r'''#!/bin/bash
echo "uvx $*" >>"$FAKE_STATE/calls"
[ "${FAKE_DL_RC:-0}" = 0 ] || exit 1
while [ $# -gt 0 ]; do if [ "$1" = --local-dir ]; then d="$2"; fi; shift; done
mkdir -p "$d" && dd if=/dev/zero of="$d/model-00001.bin" bs=1048576 count="${FAKE_CKPT_MB:-3}" 2>/dev/null
echo "{}" >"$d/config.json"
'''


class Node:

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.state = tmp_path / "state"
        self.models = tmp_path / "models"
        fakebin = tmp_path / "fakebin"
        for d in (self.home / "rental", self.home / "farm", self.home / ".local" / "bin", self.state, fakebin):
            d.mkdir(parents=True)
        for name, text in FAKES.items():
            (fakebin / name).write_text(text)
            (fakebin / name).chmod(0o755)
        for name, text in (("flock", FLOCK_SHIM), ("setsid", SETSID_SHIM)):
            if shutil.which(name) is None:
                (fakebin / name).write_text(text)
                (fakebin / name).chmod(0o755)
        (self.home / "rental" / "BUNDLE.txt").write_text("plugin sha: abc\n")
        (self.home / "rental" / "prepare_host.sh").write_text(PREPARE_HOST)
        (self.home / "farm" / "build_base.sh").write_text(BUILD_BASE)
        shutil.copy(FARM_SRC / "cardq.sh", self.home / "farm" / "cardq.sh")
        (self.home / ".local" / "bin" / "uvx").write_text(UVX)
        (self.home / ".local" / "bin" / "uvx").chmod(0o755)
        (tmp_path / "os-release").write_text('ID=ubuntu\nVERSION_ID="22.04"\n')
        (tmp_path / "meminfo").write_text("MemTotal:       970000000 kB\n")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("CARDQ_", "FARM_", "BRINGUP_"))}
        self.env.update({
            "PATH": "%s:%s" % (fakebin, os.environ.get("PATH", "/usr/bin:/bin")), "HOME": str(self.home),
            "FAKE_STATE": str(self.state), "BRINGUP_MODELS_DIR": str(self.models),
            "BRINGUP_OS_RELEASE": str(tmp_path / "os-release"), "BRINGUP_MEMINFO": str(tmp_path / "meminfo"),
            "BRINGUP_MIN_ROOT_GB": "0", "CARDQ_ALLOW_ROOT": "1", "CARDQ_SUDO": "",
        })

    def run(self, *args, **env):
        return subprocess.run(["bash", str(BRINGUP), *args], capture_output=True, text=True, timeout=120,
                              env=dict(self.env, **env), stdin=subprocess.DEVNULL)

    def calls(self):
        f = self.state / "calls"
        return f.read_text().splitlines() if f.exists() else []


@pytest.fixture
def node(tmp_path):
    return Node(tmp_path)


def test_a_bare_node_is_brought_up_in_order_and_a_second_run_changes_nothing(node):
    proc = node.run("--min-gb", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    order = [out.index("STEP %d %s: ok" % (n, name)) for n, name in enumerate(
        ("checks", "host", "ram-disk", "download", "base-image", "checkpoint", "cards"), start=1)]
    assert order == sorted(order) and out.rstrip().splitlines()[-1].startswith("BRINGUP-DONE")
    calls = node.calls()
    assert calls[0].startswith("prepare_host")  # the driver first: nothing else works without it
    assert "mount -t tmpfs -o size=320G tmpfs %s" % node.models in calls
    assert any(c.startswith("uvx --from huggingface_hub==1.33.0 hf download Qwen/Qwen3.8-Flash-Next-FP8 "
                            "--local-dir %s/Qwen3.8-Flash-Next-FP8" % node.models) for c in calls)
    assert "build_base %s/rental" % node.home in calls
    # the card order of this machine, and its module groups for CARD.md
    assert (node.home / "farm" / "q" / "module_map").read_text().strip() == "3,0,1,6,7,4,5,2"
    assert "  0-3 | 0,1,2,7 | 0-75" in out and "  4-7 | 3,4,5,6 | 76-151" in out
    assert "  0,1 | 1,2 | 0-37" in out and "  6,7 | 3,4 | 114-151" in out
    assert not any("vdc" in c or "mkfs" in c for c in calls)  # no disk is ever touched
    # again: every step sees its work done
    before = len(node.calls())
    proc = node.run("--min-gb", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "are there already" in proc.stdout and "is mounted already" in proc.stdout
    assert "STEP 4 download: ok" in proc.stdout and "done already" in proc.stdout
    new = node.calls()[before:]
    assert not any(c.startswith(("prepare_host", "mount", "uvx")) for c in new)


def test_a_new_machine_with_another_card_order_gets_its_own_map(node):
    q = node.home / "farm" / "q"
    q.mkdir()
    (q / "module_map").write_text("3,0,1,6,7,4,5,2\n")  # left by a copy of the old node's folder
    (q / "pci_map").write_text("0 0000:aa:00.0\n")
    proc = node.run("--min-gb", "0", "--no-download", FAKE_MODULES="0 1 2 3 4 5 6 7")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (q / "module_map").read_text().strip() == "0,1,2,3,4,5,6,7"
    assert "0000:0a:00.0" in (q / "pci_map").read_text()
    assert "  0-3 | 0,1,2,3 | 0-75" in proc.stdout
    assert "STEP 4 download: ok" in proc.stdout and "skipped (--no-download)" in proc.stdout


@pytest.mark.parametrize("env, args, step, text", [
    ({"FAKE_PCI_CARDS": "7"}, [], "1 (checks)", "7 Gaudi cards on PCI, 8 expected"),
    ({"FAKE_PREPARE_RC": "1"}, [], "2 (host)", "prepare_host.sh failed"),
    ({"FAKE_DL_RC": "1"}, [], "6 (checkpoint)", "the download failed"),
    ({}, ["--min-gb", "5"], "6 (checkpoint)", "holds 0 GB, at least 5 expected"),
])
def test_a_failed_step_stops_the_bring_up_and_names_itself(node, env, args, step, text):
    if "FAKE_DL_RC" in env:  # the download script retries six times with a pause: no pause in the test
        real_sleep = shutil.which("sleep")
        fake = Path(node.env["PATH"].split(":")[0]) / "sleep"
        fake.write_text('#!/bin/bash\n[ "$1" = 10 ] && exit 0\nexec %s "$@"\n' % real_sleep)
        fake.chmod(0o755)
    proc = node.run(*args, **env)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    last = proc.stdout.rstrip().splitlines()[-1]
    assert last.startswith("BRINGUP-FAILED at step %s" % step) and text in last


def test_too_little_memory_refuses_the_ram_disk_and_never_offers_a_disk_silently(node):
    (node.tmp / "meminfo").write_text("MemTotal:       400000000 kB\n")
    proc = node.run()
    assert proc.returncode == 1
    assert "a 320 GB RAM disk needs 520 GB" in proc.stdout and "Do not use a disk without the owner's word" in proc.stdout
    assert node.calls() == []  # nothing was installed or mounted


def test_dry_run_prints_the_commands_and_changes_nothing(node):
    proc = node.run("--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "$ sudo -n bash %s/rental/prepare_host.sh" % node.home in proc.stdout
    assert "$ sudo -n mount -t tmpfs -o size=320G tmpfs %s" % node.models in proc.stdout
    assert "$ bash %s/farm/build_base.sh %s/rental" % (node.home, node.home) in proc.stdout
    assert node.calls() == [] and not (node.home / "farm" / "q").exists()
