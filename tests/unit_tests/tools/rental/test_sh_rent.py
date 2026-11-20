# SPDX-License-Identifier: Apache-2.0
"""Tests for rent.sh, the workstation-side command that runs a whole rental session over SSH.

No test contacts a host. `ssh`, `scp` and `ssh-keygen` are stand-ins in a temp folder at the front of
PATH, and the address is from a documentation range. The stand-in ssh logs its arguments and runs the
one command string it was given in a fake node: a temp folder as the node's home, with stand-ins for
sudo, docker, lsblk, df, du, free, setsid and the three bundle scripts (prepare_host.sh, tiny_smoke.sh,
run_rental.sh). Files under <node>/ctl switch their behaviour; files under <node>/log record their
calls. Every script runs under /bin/bash (3.2 on the Mac this command is for).
"""

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(os.environ.get("FN_TEST_RENT_SH", REPO_ROOT / "tools" / "flash_next" / "rental" / "rent.sh"))
BASH = "/bin/bash"
IP = "198.51.100.7"  # TEST-NET-2: not routable, and only the stand-in ssh ever sees it
SHA = "0123456789abcdef0123456789abcdef01234567"
REMINDER = "stop the node in the provider's console when you are done"

# ----------------------------------------------------------------------------- workstation stand-ins

FAKE_SSH = r'''
import json, os, signal, subprocess, sys

node = os.environ["FAKE_NODE"]
ctl = os.path.join(node, "ctl")
argv = sys.argv[1:]
opts, i = [], 0
while i < len(argv):
    a = argv[i]
    if a in ("-i", "-o", "-p", "-l", "-F"):
        opts.append([a, argv[i + 1]])
        i += 2
    elif a.startswith("-"):
        opts.append([a, None])
        i += 1
    else:
        break
dest, cmd = argv[i], argv[i + 1:]
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps({"tool": "ssh", "argv": argv, "opts": opts, "dest": dest, "cmd": cmd}) + "\n")
host = dest.split("@")[-1]


def refuse(text):
    sys.stderr.write(text + "\n")
    sys.exit(255)


if os.path.exists(os.path.join(ctl, "down")):
    refuse("ssh: connect to host %s port 22: Connection refused" % host)
left_path = os.path.join(ctl, "fail-ssh")
if os.path.exists(left_path):
    left = int(open(left_path).read().strip() or 0)
    if left > 0:
        open(left_path, "w").write(str(left - 1))
        refuse("ssh: connect to host %s port 22: Operation timed out" % host)
known = [v.split("=", 1)[1] for o, v in opts if o == "-o" and v.startswith("UserKnownHostsFile=")]
if known and " " in known[0]:
    if not (known[0].startswith('"') and known[0].endswith('"')):  # ssh would take two file names
        refuse("command-line: line 0: garbage at end of line; UserKnownHostsFile needs quotes around a path with spaces")
    known[0] = known[0][1:-1]
known = [k.replace("%%", "%") for k in known]
key = "node-key-1"
if os.path.exists(os.path.join(ctl, "hostkey")):
    key = open(os.path.join(ctl, "hostkey")).read().strip()
if known:
    line = "%s ssh-ed25519 %s\n" % (host, key)
    if os.path.exists(known[0]):
        if open(known[0]).read() != line:
            refuse("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
                   "@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n"
                   "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
                   "Host key verification failed.")
    else:
        open(known[0], "w").write(line)
if os.path.exists(os.path.join(ctl, "denied")):
    refuse("%s: Permission denied (publickey)." % dest)

env = {"HOME": os.path.join(node, "home"), "FAKE_NODE": node, "USER": "sesterce", "LANG": "C",
       "PATH": os.path.join(node, "bin") + ":/usr/bin:/bin:/usr/sbin:/sbin"}


def clean():  # sshd starts the command in a session of its own, with default signal actions
    os.setsid()
    for s in (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP, signal.SIGTERM, signal.SIGPIPE):
        signal.signal(s, signal.SIG_DFL)


line = " ".join(cmd)  # as ssh does: the words joined by spaces, handed to the login shell
drop_path = os.path.join(ctl, "drop-follow")
if os.path.exists(drop_path):
    job, after = open(drop_path).read().split()
    if ("fn_job_follow %s " % job) in line:
        os.remove(drop_path)
        p = subprocess.Popen(["/bin/sh", "-c", line], env=env, cwd=env["HOME"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, preexec_fn=clean)
        seen = 0
        for raw in p.stdout:
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            if raw.startswith(b"L "):
                seen += 1
            if seen >= int(after):
                break
        sys.stdout.buffer.write(b"L half a li")  # a line cut by the drop
        sys.stdout.buffer.flush()
        p.kill()
        refuse("client_loop: send disconnect: Broken pipe")
p = subprocess.Popen(["/bin/sh", "-c", line], env=env, cwd=env["HOME"], stdin=subprocess.DEVNULL, preexec_fn=clean)
sys.exit(p.wait())
'''

FAKE_SCP = r'''
import json, os, shutil, sys

node = os.environ["FAKE_NODE"]
argv = sys.argv[1:]
opts, i = [], 0
while i < len(argv):
    a = argv[i]
    if a in ("-i", "-o", "-P", "-F"):
        opts.append([a, argv[i + 1]])
        i += 2
    elif a.startswith("-"):
        opts.append([a, None])
        i += 1
    else:
        break
paths = argv[i:]
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps({"tool": "scp", "argv": argv, "opts": opts, "paths": paths}) + "\n")
if os.path.exists(os.path.join(node, "ctl", "down")):
    sys.stderr.write("ssh: connect to host port 22: Connection refused\n")
    sys.exit(255)


def local(path):
    if path.startswith("/"):
        return path  # an absolute path is local, whatever it contains
    if ":" in path and "@" in path.split(":", 1)[0]:
        rest = path.split(":", 1)[1]
        return rest if rest.startswith("/") else os.path.join(node, "home", rest)
    return path


src, dst = local(paths[0]), local(paths[1])
if not os.path.isfile(src):
    sys.stderr.write("scp: %s: No such file or directory\n" % paths[0])
    sys.exit(1)
if os.path.isdir(dst):
    dst = os.path.join(dst, os.path.basename(src))
shutil.copyfile(src, dst)
'''

FAKE_KEYGEN = r"""#!/bin/bash
echo "$*" >>"$FAKE_KEYGEN_LOG"
file=""
while [ $# -gt 0 ]; do case "$1" in -f) file="$2"; shift 2 ;; *) shift ;; esac; done
[ -f "$file" ] || { echo "ssh-keygen: $file: No such file or directory" >&2; exit 255; }
echo "256 SHA256:FAKEfingerprintOfTheNodeKey $(cut -d' ' -f1 "$file") (ED25519)"
"""

# ----------------------------------------------------------------------------- node stand-ins

NODE_SUDO = r"""#!/bin/bash
echo "$*" >>"$FAKE_NODE/log/sudo"
if [ "${1:-}" != "-n" ]; then echo "sudo: a terminal is required to read the password" >&2; exit 1; fi
shift
if [ -f "$FAKE_NODE/ctl/sudo-password" ]; then echo "sudo: a password is required" >&2; exit 1; fi
exec "$@"
"""

NODE_SETSID = r'''
import os, sys

args = sys.argv[1:]
fork = False
while args and args[0].startswith("-"):
    fork = fork or args[0] in ("-f", "--fork")
    args = args[1:]
if not fork:
    sys.stderr.write("fake setsid: called without -f\n")
    sys.exit(1)
if os.fork() > 0:
    os._exit(0)
os.setsid()
os.execvp(args[0], args)
'''

NODE_DOCKER = r"""#!/bin/bash
echo "$*" >>"$FAKE_NODE/log/docker"
ctl="$FAKE_NODE/ctl"
[ -f "$ctl/host-prepared" ] || { echo "docker: command not found" >&2; exit 127; }
case "$1" in
  info) echo '{"habana":{"path":"/usr/bin/habana-container-runtime"},"runc":{"path":"runc"}}' ;;
  inspect)
    [ -f "$ctl/download-state" ] || { echo "Error: No such object: fn-download" >&2; exit 1; }
    left="$(cat "$ctl/download-polls" 2>/dev/null || echo 0)"
    if [ "$left" -gt 0 ]; then
      echo $((left - 1)) >"$ctl/download-polls"
    elif [ -f "$ctl/download-final" ]; then
      mv "$ctl/download-final" "$ctl/download-state"
      if [ "$(cat "$ctl/download-state")" = "exited 0" ]; then
        m="$(cat "$ctl/download-model")"
        mkdir -p "$m"; echo '{}' >"$m/config.json"; : >"$m/model-00001-of-00040.safetensors"
      fi
    fi
    cat "$ctl/download-state" ;;
  logs) cat "$ctl/download-log" 2>/dev/null ;;
esac
exit 0
"""

NODE_DU = r"""#!/bin/bash
seq="$FAKE_NODE/ctl/du-seq"
if [ -s "$seq" ]; then
  kb="$(head -n 1 "$seq")"
  if [ "$(wc -l <"$seq")" -gt 1 ]; then tail -n +2 "$seq" >"$seq.tmp" && mv "$seq.tmp" "$seq"; fi
  printf '%s\t%s\n' "$kb" "${@: -1}"
else
  exec /usr/bin/du "$@"
fi
"""

NODE_DF = r"""#!/bin/bash
free="$(cat "$FAKE_NODE/ctl/df-free-kb" 2>/dev/null || echo 58593750)"
echo "Filesystem     1024-blocks     Used Available Capacity Mounted on"
echo "/dev/vda1         96000000 30000000 $free      35% /"
"""

NODE_LSBLK = r"""#!/bin/bash
echo "$*" >>"$FAKE_NODE/log/lsblk"
case " $* " in
  *" -P "*) cat "$FAKE_NODE/ctl/lsblk-pairs" ;;
  *) cat "$FAKE_NODE/ctl/lsblk-table" ;;
esac
"""

NODE_LSB_RELEASE = '#!/bin/bash\necho "Ubuntu 22.04.5 LTS"\n'
NODE_NPROC = '#!/bin/bash\necho 160\n'
NODE_FREE = ('#!/bin/bash\necho "               total        used        free"\n'
             'echo "Mem:            1007          12         990"\n')
NODE_HL_SMI = '#!/bin/bash\nfor i in 0 1 2 3 4 5 6 7; do echo "$i, HL-225"; done\n'

BUNDLE_PREPARE_HOST = r"""#!/usr/bin/env bash
ctl="$FAKE_NODE/ctl"; here="$(cd "$(dirname "$0")" && pwd)"
{ echo CALL; for a in "$@"; do printf '%s\n' "$a"; done; } >>"$FAKE_NODE/log/prepare_host"
echo "host step 1 (checks): ok"
while [ -f "$ctl/host-hold" ]; do sleep 0.1; done
if [ -f "$ctl/host-reboot" ]; then
  echo "host step 4 (verify driver): FAILED" >&2
  echo "remedy: the driver loaded but hl-smi does not see all cards: reboot and rerun this script" >&2
  exit 1
fi
if [ -f "$ctl/host-fail" ]; then
  echo "note: on Ubuntu 22.04.5 / kernel 6.8, set GRUB_CMDLINE_LINUX_DEFAULT, run update-grub, and reboot (this script does not edit GRUB)." >&2
  echo "host step 5 (docker): FAILED" >&2
  echo "remedy: apt-get install -y docker.io failed" >&2
  exit 1
fi
dev=""; mnt="/models"
while [ $# -gt 0 ]; do
  case "$1" in --data-disk) dev="$2"; shift 2 ;; --mount) mnt="$2"; shift 2 ;; *) shift ;; esac
done
if [ -n "$dev" ]; then
  echo "mkfs $dev at $mnt" >>"$FAKE_NODE/log/mkfs"
  mkdir -p "$mnt"
  echo 25000000000 >"$ctl/df-free-kb"
  echo "data disk: $dev mounted at $mnt, 25000 GB free"
fi
mkdir -p "$here/prepare-host-logs"
echo "step 1 log" >"$here/prepare-host-logs/step-1.log"
ln -sf "$FAKE_NODE/stock/hl-smi" "$FAKE_NODE/bin/hl-smi"  # the driver is installed now
: >"$ctl/host-prepared"
echo "host step 9 (smoke test): ok"
echo "host preparation summary"
"""

BUNDLE_TINY_SMOKE = r"""#!/usr/bin/env bash
ctl="$FAKE_NODE/ctl"
{ echo CALL; for a in "$@"; do printf '%s\n' "$a"; done; } >>"$FAKE_NODE/log/tiny_smoke"
mode=""; id=""
while [ $# -gt 0 ]; do
  case "$1" in --gdn) mode="$2"; shift 2 ;; --run-id) id="$2"; shift 2 ;; *) shift ;; esac
done
out="$PWD/results/$id"
mkdir -p "$out/$mode" "$out/ckpt"
echo "server log of $mode" >"$out/$mode/server0.log"
echo "weights" >"$out/ckpt/model.safetensors"
echo "container fn-smoke: starting"
while [ -f "$ctl/smoke-hold" ]; do sleep 0.1; done
if [ -f "$ctl/smoke-fail-$mode" ]; then
  echo '{"passed": false}' >"$out/summary.json"
  echo "tiny smoke [$mode]: FAILED (the server exited with code 1 before it was healthy)"
  echo "    RuntimeError: synthetic $mode failure"
  echo "    log: /results/$id/$mode/server0.log"
  echo "tiny smoke: FAILED ($mode)"
  exit 1
fi
echo '{"passed": true}' >"$out/summary.json"
echo "tiny smoke [$mode]: PASSED (server up in 61 s, tokens match the reference, warmup skipped, recipe cache on)"
echo "tiny smoke: PASSED ($mode)"
"""

BUNDLE_RUN_RENTAL = r"""#!/usr/bin/env bash
# --hc-fused
ctl="$FAKE_NODE/ctl"
{ echo CALL; for a in "$@"; do printf '%s\n' "$a"; done; } >>"$FAKE_NODE/log/run_rental"
model=""; id=""; dl=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model="$2"; shift 2 ;;
    --run-id) id="$2"; shift 2 ;;
    --download-only) dl=1; shift ;;
    *) shift ;;
  esac
done
if [ "$dl" -eq 1 ]; then
  if [ -f "$ctl/download-start-fail" ]; then
    echo "error: / (filesystem mounted at /) has 61 GB free; --download needs 250 GB there" >&2
    exit 1
  fi
  echo "running 0" >"$ctl/download-state"
  cp "$ctl/download-result" "$ctl/download-final" 2>/dev/null || echo "exited 0" >"$ctl/download-final"
  cp "$ctl/download-polls-init" "$ctl/download-polls" 2>/dev/null || echo 1 >"$ctl/download-polls"
  echo "$model" >"$ctl/download-model"
  mkdir -p "$model"
  echo "checkpoint download started in container fn-download (detached: it goes on without this terminal)"
  exit 0
fi
out="$PWD/results/$id"
mkdir -p "$out"
finish() {
  { echo "# Rental run $id"; echo "| A | $2 |"; } >"$out/summary.md"
  tar czf "$PWD/results/$id.tar.gz" -C "$PWD/results" "$id"
  echo "PASTE THIS FILE: $out/summary.md (plus the last 40 lines of $out/console.log if the run stopped early)"
  echo "SEND THIS FILE: $PWD/results/$id.tar.gz"
  exit "$1"
}
trap 'echo "interrupted by SIGINT: stopping the run container, then writing the tarball" | tee -a "$out/console.log"; : >"$ctl/run-got-sigint"; finish 130 interrupted' INT
{
  echo "run id: $id"
  echo "phase: preflight"
  echo "preflight: PASSED after 3 s"
  echo "phase: A"
  echo "variant A server0: first token after 41.2 s: 'Hello'"
} | tee -a "$out/console.log"
n="$(cat "$ctl/run-lines" 2>/dev/null || echo 0)"; i=1
while [ "$i" -le "$n" ]; do echo "load line $i"; i=$((i + 1)); done
while [ -f "$ctl/run-hold" ]; do sleep 0.1; done
rc="$(cat "$ctl/run-exit" 2>/dev/null || echo 0)"
if [ "$rc" -eq 0 ]; then
  echo "variant A: PASSED after 900 s"
  echo "RUN $id: PASSED"
  echo "container exit code: 0"
  finish 0 PASSED
fi
echo "variant A: FAILED after 900 s (load: 3 non-OK rows)"
echo "RUN $id: FAILED at A"
echo "container exit code: $rc"
finish "$rc" FAILED
"""

FAKE_MAKE_BUNDLE = r"""#!/usr/bin/env bash
# stand-in for make_rental_bundle.sh in a temp checkout: the bundle of the test, named after HEAD
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
echo "$*" >>"$root/make-bundle-calls"
if [ -f "$root/fail-build" ]; then
  echo "error: vLLM commit 8f816a3 not found in /nowhere/vllm-src" >&2
  exit 1
fi
out="dist"
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2 ;; --vllm-src) shift 2 ;; *) shift ;; esac; done
sha="$(git -C "$root" rev-parse HEAD)"
stage="$(mktemp -d)"
tar xzf "$root/template.tar.gz" -C "$stage"
echo "plugin sha: $sha" >"$stage/BUNDLE.txt"
mkdir -p "$out"
tar czf "$out/flash-next-rental-${sha:0:12}.tar.gz" -C "$stage" BUNDLE.txt run_rental.sh prepare_host.sh tiny_smoke.sh plugin vllm
rm -rf "$stage"
echo "bundle: $(cd "$out" && pwd)/flash-next-rental-${sha:0:12}.tar.gz"
echo "size: 1K"
"""

GB = 1000 ** 3
ROOT_DISK = [("vda", "disk", 100 * GB, "", "", "", ""),
             ("vda1", "part", 99 * GB, "ext4", "cloudimg-rootfs", "/", "vda"),
             ("vda15", "part", GB // 10, "vfat", "UEFI", "/boot/efi", "vda")]
BIG_BLANK = ("vdc", "disk", 26400 * GB, "", "", "", "")


def _exe(path, text, python=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("#!%s\n" % sys.executable if python else "") + text.lstrip("\n"))
    path.chmod(0o755)


_STOCK = {}


def _stock():
    """The stand-in programs, written once per test process: every Rig links to them. (macOS checks
    a new executable file on its first start, about 0.13 s each; a link to a file it knows is free.)"""
    if not _STOCK:
        root = Path(tempfile.mkdtemp(prefix="fn-rent-standins-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        for name, text, python in (("ssh", FAKE_SSH, True), ("scp", FAKE_SCP, True), ("ssh-keygen", FAKE_KEYGEN, False)):
            _exe(root / "workstation" / name, text, python)
        for name, text, python in (("sudo", NODE_SUDO, False), ("setsid", NODE_SETSID, True), ("docker", NODE_DOCKER, False),
                                   ("du", NODE_DU, False), ("df", NODE_DF, False), ("lsblk", NODE_LSBLK, False),
                                   ("lsb_release", NODE_LSB_RELEASE, False), ("nproc", NODE_NPROC, False),
                                   ("free", NODE_FREE, False)):
            _exe(root / "node" / name, text, python)
        _exe(root / "later" / "hl-smi", NODE_HL_SMI)
        _STOCK["root"] = root
    return _STOCK["root"]


def _link_all(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        (target / path.name).symlink_to(path)


def _default_signals():
    # a test runner started in the background hands down an ignored SIGINT, which bash cannot trap
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class Rig:
    """A workstation (stand-in ssh, scp and ssh-keygen on PATH) and the fake node they reach."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.node = tmp_path / "node"
        self.home = self.node / "home"
        self.ctl = self.node / "ctl"
        self.nlog = self.node / "log"
        self.bin = tmp_path / "bin"
        self.session = tmp_path / "session"
        self.wshome = tmp_path / "wshome"
        self.key = tmp_path / "keys" / "id_test"
        self.calls = tmp_path / "calls.jsonl"
        self.keygen_log = tmp_path / "keygen.log"
        self.model = self.node / "data" / "models" / "Qwen3.8-Flash-Next-FP8"
        self.bundle = tmp_path / ("flash-next-rental-%s.tar.gz" % SHA[:12])
        for folder in (self.home, self.ctl, self.nlog, self.wshome, self.key.parent):
            folder.mkdir(parents=True)
        self.key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETKEYMATERIAL\n-----END OPENSSH PRIVATE KEY-----\n")
        self.calls.write_text("")
        _link_all(_stock() / "workstation", self.bin)
        _link_all(_stock() / "node", self.node / "bin")
        _link_all(_stock() / "later", self.node / "stock")  # hl-smi: prepare_host.sh "installs" it
        self.disks(*ROOT_DISK, BIG_BLANK)
        self.make_bundle(self.bundle, SHA)

    # -- the fake node ---------------------------------------------------------------------------

    def disks(self, *rows):
        pairs, table = [], ["NAME    SIZE TYPE FSTYPE LABEL MOUNTPOINT"]
        for name, kind, size, fstype, label, mount, parent in rows:
            pairs.append('NAME="%s" TYPE="%s" SIZE="%d" FSTYPE="%s" LABEL="%s" MOUNTPOINT="%s" PKNAME="%s"'
                         % (name, kind, size, fstype, label, mount, parent))
            table.append("%s %dG %s %s %s %s" % (name, size // GB, kind, fstype, label, mount))
        (self.ctl / "lsblk-pairs").write_text("\n".join(pairs) + "\n")
        (self.ctl / "lsblk-table").write_text("\n".join(table) + "\n")

    def free_gb(self, gb):
        (self.ctl / "df-free-kb").write_text("%d\n" % (gb * GB // 1024))

    def flag(self, name, text=""):
        (self.ctl / name).write_text(text)

    def unflag(self, name):
        path = self.ctl / name
        if path.exists():
            path.unlink()

    def make_bundle(self, path, sha, stage=None):
        stage = stage or self.tmp / ("bundle-stage-" + sha[:6])
        (stage / "plugin").mkdir(parents=True, exist_ok=True)
        (stage / "vllm").mkdir(exist_ok=True)
        (stage / "BUNDLE.txt").write_text("plugin sha: %s\nbranch: test\nvllm commit: 8f816a3\nbuild date: now\n" % sha)
        (stage / "plugin" / "README").write_text("plugin at %s\n" % sha)
        (stage / "vllm" / "README").write_text("vllm\n")
        for name, text in (("prepare_host.sh", BUNDLE_PREPARE_HOST), ("tiny_smoke.sh", BUNDLE_TINY_SMOKE),
                           ("run_rental.sh", BUNDLE_RUN_RENTAL)):
            _exe(stage / name, text)
        with tarfile.open(path, "w:gz") as tar:
            for name in ("BUNDLE.txt", "run_rental.sh", "prepare_host.sh", "tiny_smoke.sh", "plugin", "vllm"):
                tar.add(stage / name, arcname=name)
        return path

    def node_calls(self, name):
        """The argument lists of every call of a bundle script on the node."""
        path = self.nlog / name
        if not path.exists():
            return []
        calls = []
        for line in path.read_text().splitlines():
            if line == "CALL":
                calls.append([])
            else:
                calls[-1].append(line)
        return calls

    def runs(self):
        return [c for c in self.node_calls("run_rental") if "--download-only" not in c]

    def job_dir(self):
        return self.home / "rental" / ".jobs"

    def job_alive(self, name):
        pid_file = self.job_dir() / (name + ".pid")
        if not pid_file.exists() or (self.job_dir() / (name + ".rc")).exists():
            return False
        try:
            os.kill(int(pid_file.read_text()), 0)
        except (OSError, ValueError):
            return False
        return True

    def cleanup(self):
        for path in self.ctl.glob("*-hold"):
            path.unlink()
        jobs = self.job_dir()
        deadline = time.time() + 10
        while jobs.exists() and time.time() < deadline and any(self.job_alive(p.stem) for p in jobs.glob("*.pid")):
            time.sleep(0.05)

    # -- the workstation -------------------------------------------------------------------------

    def env(self, **extra):
        env = {"PATH": "%s:/usr/bin:/bin:/usr/sbin:/sbin" % self.bin, "HOME": str(self.wshome), "LANG": "C",
               "FAKE_NODE": str(self.node), "FAKE_CALLS": str(self.calls), "FAKE_KEYGEN_LOG": str(self.keygen_log),
               "FN_RENTAL_POLL_S": "0.1", "FN_RENTAL_FOLLOW_S": "0.05", "FN_RENTAL_HEARTBEAT_S": "1",
               "FN_RENTAL_RETRY_S": "0.1", "FN_RENTAL_RETRY_MAX_S": "0.2", "FN_RENTAL_SSH_TRIES": "3"}
        env.update({k: str(v) for k, v in extra.items()})
        assert shutil.which("ssh", path=env["PATH"]) == str(self.bin / "ssh")
        assert shutil.which("scp", path=env["PATH"]) == str(self.bin / "scp")
        return env

    def argv(self, *args, plain=False):
        if plain:
            return [BASH, str(SCRIPT), *[str(a) for a in args]]
        return [BASH, str(SCRIPT), IP, "--session", str(self.session), "--key", str(self.key),
                "--bundle", str(self.bundle), "--model", str(self.model), *[str(a) for a in args]]

    def run(self, *args, plain=False, env=None, timeout=120, cwd=None):
        proc = subprocess.run(self.argv(*args, plain=plain), capture_output=True, text=True, timeout=timeout,
                              env=self.env(**(env or {})), cwd=cwd or self.tmp, preexec_fn=_default_signals)
        proc.out = proc.stdout + proc.stderr
        return proc

    def start(self, *args, env=None):
        """The command in the background, its output in a file the test can read while it runs."""
        out = self.tmp / ("out-%d.txt" % len(list(self.tmp.glob("out-*.txt"))))
        handle = open(out, "w")
        proc = subprocess.Popen(self.argv(*args), stdout=handle, stderr=subprocess.STDOUT, text=True,
                                env=self.env(**(env or {})), cwd=self.tmp, preexec_fn=_default_signals)
        proc.out_path = out
        return proc

    def wait_for(self, proc, text, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if text in proc.out_path.read_text():
                return
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        raise AssertionError("no %r in the output:\n%s" % (text, proc.out_path.read_text()))

    def entries(self, tool=None):
        rows = [json.loads(line) for line in self.calls.read_text().splitlines() if line]
        return [r for r in rows if tool is None or r["tool"] == tool]

    def remote_text(self):
        """Every remote command sent so far, without the helper functions sent in front of each."""
        return "\n".join(_body(r["cmd"][0]) for r in self.entries("ssh"))

    def state(self):
        path = self.session / "state"
        return dict(line.split("=", 1) for line in path.read_text().splitlines()) if path.exists() else {}


def _body(command):
    """The part of a remote command after the helper functions (they end with a marker line)."""
    marker = "# --- end of the node helpers"
    return command.split(marker, 1)[1] if marker in command else command


@pytest.fixture()
def rig(tmp_path):
    rig = Rig(tmp_path)
    yield rig
    rig.cleanup()


def _stage_events(text):
    return re.findall(r"^stage (\d)/8 (\w+): (start|ok|FAILED)", text, flags=re.M)


def _ok(proc):
    assert proc.returncode == 0, proc.out
    return proc


# ----------------------------------------------------------------------------- static checks


def test_bash_n():
    proc = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_no_construct_newer_than_bash_3_2():
    text = SCRIPT.read_text()
    for pattern in (r"declare -A", r"\bmapfile\b", r"\breadarray\b", r"\$\{[A-Za-z_]+(,,|\^\^)", r"&>>", r"\|&",
                    r"\bcoproc\b", r"wait -n", r"\$\{[A-Za-z_]+@[A-Za-z]\}", r"declare -n", r"local -n"):
        assert not re.search(pattern, text), pattern
    assert text.splitlines()[0] == "#!/usr/bin/env bash"
    assert "\nset -euo pipefail\n" in text


def test_ssh_options_are_spelled_in_one_function_only():
    text = SCRIPT.read_text()
    match = re.search(r"^ssh_opts\(\) \{\n(.*?)^\}\n", text, flags=re.M | re.S)
    assert match, "no ssh_opts() function"
    body = match.group(1)
    for word in ("IdentitiesOnly=yes", "UserKnownHostsFile=", "StrictHostKeyChecking=accept-new", "ConnectTimeout=",
                 "ServerAliveInterval=", "ServerAliveCountMax="):
        assert word in body, word
    rest = text.replace(match.group(0), "")
    code = [line for line in rest.splitlines() if not line.lstrip().startswith("#")]
    for word in ("IdentitiesOnly", "UserKnownHostsFile", "StrictHostKeyChecking", "ConnectTimeout", "ServerAlive",
                 "BatchMode", "LogLevel", '-i "$SSH_KEY"'):
        assert word not in "\n".join(code), word
    assert not re.search(r"-o\s*[\"']?[A-Z][A-Za-z]*=", "\n".join(code))  # no -o Key=value anywhere else
    # every place that runs or prints ssh or scp takes the list that function built
    uses = [(line, m.group(2)) for line in code for m in re.finditer(r"(?<![\w./-])(ssh|scp) +(\S+)", line)]
    assert len(uses) >= 5, uses
    for line, following in uses:
        assert following.startswith(('"${SSH_OPTS[@]}"', "$(rq")) and '"${SSH_OPTS[@]}"' in line, line


def test_usage_errors(rig):
    proc = rig.run(plain=True)
    assert proc.returncode == 2 and "usage:" in proc.out
    proc = rig.run("--no-such-option")
    assert proc.returncode == 2 and "unknown option" in proc.out
    proc = rig.run("--rate", "ten")
    assert proc.returncode == 2
    proc = rig.run("--smoke-modes", "pytorch,turbo")
    assert proc.returncode == 2 and "turbo" in proc.out
    proc = rig.run("-oProxyCommand=evil", plain=True)
    assert proc.returncode == 2
    assert rig.entries() == []


# ----------------------------------------------------------------------------- dry run


def test_dry_run_contacts_nothing_and_prints_the_commands_in_order(rig):
    proc = _ok(rig.run("--dry-run"))
    assert rig.entries() == []  # neither ssh nor scp was started
    assert not rig.keygen_log.exists()
    assert not rig.session.exists()  # and nothing was written
    out = proc.stdout
    wanted = ["stage 1/8 connect", "lsb_release", "stage 2/8 bundle", "scp ", "tar xzf", "BUNDLE.txt",
              "stage 3/8 host", "sudo -n bash prepare_host.sh", "stage 4/8 download", "--download-only",
              "stage 5/8 smoke", "tiny_smoke.sh --gdn pytorch", "tiny_smoke.sh --gdn fused",
              "stage 6/8 wait", "fn_ckpt", "stage 7/8 run", "--quick-gates --variants A,C", "stage 8/8 fetch",
              "summary.md"]
    at = 0
    for part in wanted:
        found = out.find(part, at)
        assert found >= 0, "%r missing or out of order in:\n%s" % (part, out)
        at = found
    assert "IdentitiesOnly=yes" in out and "UserKnownHostsFile=%s/known_hosts" % rig.session in out
    assert REMINDER in out


def test_dry_run_in_a_live_session_contacts_nothing_and_writes_nothing(rig):
    _ok(rig.run("--skip-smoke", "--no-download"))
    calls, session = rig.calls.read_text(), _tree(rig.session)
    for args in (("--dry-run",), ("--dry-run", "--status"), ("--dry-run", "--stop"), ("--dry-run", "--new-run")):
        proc = _ok(rig.run(*args))
        assert "[dry-run] remote: " in proc.stdout
    assert rig.calls.read_text() == calls  # no ssh, no scp
    assert _tree(rig.session) == session  # not a byte in the log or the state
    assert rig.keygen_log.read_text().count("\n") == 1


def test_dry_run_defaults_and_bundle_build_options(rig):
    proc = _ok(rig.run(IP, "--dry-run", "--vllm-src", "/src/my vllm", "--allow-dirty", plain=True,
                       env={"FN_RENTAL_SSH_KEY": ""}))
    out = proc.stdout
    assert "make_rental_bundle.sh --out" in out and "--vllm-src '/src/my vllm' --allow-dirty" in out
    assert "sesterce@%s" % IP in out
    assert "-i %s/.ssh/sesterce_ed25519" % rig.wshome in out
    assert "--model /models/Qwen3.8-Flash-Next-FP8" in out
    assert re.search(r"UserKnownHostsFile=%s/rental-sessions/%s-\d{4}-\d\d-\d\d/known_hosts" % (rig.wshome, IP), out)
    proc = _ok(rig.run(IP, "--dry-run", plain=True, env={"FN_VLLM_SRC": "/env/vllm"}))
    build = [ln.split("   #")[0] for ln in proc.stdout.splitlines() if "make_rental_bundle.sh --out" in ln]
    assert len(build) == 1 and build[0].endswith("/dist --vllm-src /env/vllm")
    assert rig.entries() == []


def test_dry_run_run_arguments(rig):
    out = _ok(rig.run("--dry-run")).stdout
    assert "run_rental.sh --model %s --run-id " % rig.model in out
    out = _ok(rig.run("--dry-run", "--variants", "A,B,C", "--no-quick-gates", "--hc-fused", "--run-arg",
                      "--server-timeout", "--run-arg", "2400", "--run-arg", "--no-warm-pass")).stdout
    line = [ln for ln in out.splitlines() if "run_rental.sh" in ln and "--download-only" not in ln and "--run-id" in ln][0]
    assert "--quick-gates" not in line
    assert line.rstrip().endswith("--variants A,B,C --hc-fused --server-timeout 2400 --no-warm-pass")
    for owned in ("--run-id", "--model", "--download-only", "--dry-run"):
        proc = rig.run("--dry-run", "--run-arg", owned)
        assert proc.returncode == 2, owned


# ----------------------------------------------------------------------------- a whole session


def test_fresh_node_runs_the_eight_stages_in_order(rig):
    proc = _ok(rig.run())
    events = _stage_events(proc.stdout)
    names = ["connect", "bundle", "host", "download", "smoke", "wait", "run", "fetch"]
    assert events == [(str(n), name, word) for n, name in enumerate(names, 1) for word in ("start", "ok")]
    # the node saw the six manual steps, once each and in this order
    assert rig.node_calls("prepare_host") == [["--data-disk", "/dev/vdc", "--mount", str(rig.model.parent)]]
    smoke = rig.node_calls("tiny_smoke")
    assert [c[:2] for c in smoke] == [["--gdn", "pytorch"], ["--gdn", "fused"]]
    calls = rig.node_calls("run_rental")
    assert calls[0] == ["--model", str(rig.model), "--download-only"]
    run_id = rig.state()["run_id"]
    assert calls[1] == ["--model", str(rig.model), "--run-id", run_id, "--quick-gates", "--variants", "A,C"]
    assert len(calls) == 2
    assert (rig.home / "rental" / "BUNDLE.txt").read_text().startswith("plugin sha: " + SHA)
    assert "plugin sha: " + SHA in proc.stdout  # BUNDLE.txt was shown
    # the node summary of stage 1
    assert "Ubuntu 22.04.5 LTS" in proc.stdout and "160 CPUs" in proc.stdout and "1007 GB RAM" in proc.stdout
    assert "driver not installed" in proc.stdout and "vdc" in proc.stdout
    # the fetched results
    results = rig.session / "results"
    assert (results / (run_id + ".tar.gz")).is_file()
    assert (results / run_id / "summary.md").read_text().startswith("# Rental run " + run_id)
    assert (results / run_id / "console.log").is_file()
    smoke_dirs = sorted(p.name.rsplit("-", 1)[1] for p in results.glob("smoke-*") if p.is_dir())
    assert smoke_dirs == ["fused", "pytorch"]
    assert len(list(results.glob("smoke-*-pytorch/pytorch/server0.log"))) == 1
    assert not list(results.rglob("*.safetensors"))
    assert (results / "prepare-host-logs" / "step-1.log").is_file()
    assert "# Rental run " + run_id in proc.stdout  # summary.md is printed
    for path in (results / (run_id + ".tar.gz"), results / run_id / "summary.md"):
        assert str(path) in proc.stdout
    assert REMINDER in proc.stdout


def test_session_log_state_and_secrets(rig):
    proc = _ok(rig.run())
    log = (rig.session / "session.log").read_text()
    lines = log.splitlines()
    assert lines and all(re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", ln) for ln in lines)
    for line in proc.stdout.splitlines():  # everything printed is in the log
        assert line in log, line
    assert "SECRETKEYMATERIAL" not in log and "SECRETKEYMATERIAL" not in proc.out
    state = rig.state()
    assert state["clock_start"].isdigit() and abs(int(state["clock_start"]) - time.time()) < 120
    for key in ("bundle_sha", "host", "run_id", "run_state", "fetch"):
        assert key in state, state


def test_every_ssh_and_scp_call_carries_the_session_options(rig):
    _ok(rig.run())
    rows = rig.entries()
    assert len([r for r in rows if r["tool"] == "ssh"]) > 8 and len([r for r in rows if r["tool"] == "scp"]) >= 4
    for row in rows:
        opts = [tuple(o) for o in row["opts"]]
        assert ("-i", str(rig.key)) in opts, row
        for value in ("IdentitiesOnly=yes", "UserKnownHostsFile=%s/known_hosts" % rig.session,
                      "StrictHostKeyChecking=accept-new"):
            assert ("-o", value) in opts, (value, row)
        values = [v for o, v in opts if o == "-o"]
        assert any(v.startswith("ConnectTimeout=") for v in values), row
        assert any(v.startswith("ServerAliveInterval=") for v in values), row
        assert any(v.startswith("ServerAliveCountMax=") for v in values), row
        if row["tool"] == "ssh":
            assert row["dest"] == "sesterce@" + IP
            assert len(row["cmd"]) == 1, row  # the remote command is ONE string
            assert row["cmd"][0].startswith("bash -c '")
        else:
            assert any(p.startswith("sesterce@%s:" % IP) for p in row["paths"]), row


def test_user_and_key_overrides(rig):
    other = rig.tmp / "keys" / "other key"
    other.write_text("k")
    _ok(rig.run("--dry-run", "--user", "ubuntu", "--key", other))
    proc = _ok(rig.run(IP, "--session", rig.session, "--bundle", rig.bundle, "--model", rig.model, "--skip-smoke",
                       "--no-download", plain=True, env={"FN_RENTAL_SSH_USER": "envuser", "FN_RENTAL_SSH_KEY": other}))
    assert proc.returncode == 0
    for row in rig.entries("ssh"):
        assert row["dest"] == "envuser@" + IP and ["-i", str(other)] in row["opts"]
    rig.calls.write_text("")
    _ok(rig.run("--user", "flaguser", "--skip-smoke", "--no-download", env={"FN_RENTAL_SSH_USER": "envuser"}))
    assert {row["dest"] for row in rig.entries("ssh")} == {"flaguser@" + IP}  # the option beats the variable
    proc = rig.run(IP, "--key", rig.tmp / "no-such-key", "--bundle", rig.bundle, plain=True)
    assert proc.returncode == 1 and "no-such-key" in proc.out


def test_host_key_fingerprint_is_printed_once(rig):
    first = _ok(rig.run("--skip-smoke", "--no-download"))
    assert first.stdout.count("host key: 256 SHA256:FAKEfingerprintOfTheNodeKey") == 1
    assert rig.keygen_log.read_text().split() == ["-l", "-f", str(rig.session / "known_hosts")]
    again = _ok(rig.run("--skip-smoke", "--no-download"))
    assert "SHA256:FAKEfingerprint" not in again.stdout


def test_a_changed_host_key_in_a_named_session_stops_with_the_way_out(rig):
    _ok(rig.run("--skip-smoke", "--no-download"))
    rig.flag("hostkey", "node-key-2")  # the address now answers with another key: another node
    proc = rig.run("--skip-smoke", "--no-download")
    assert proc.returncode == 1
    assert "host key" in proc.out and "--session" in proc.out and REMINDER in proc.out


def test_default_session_folder_continues_and_restarts_on_a_changed_host_key(rig):
    args = (IP, "--key", rig.key, "--bundle", rig.bundle, "--model", rig.model, "--skip-smoke", "--no-download")
    first = _ok(rig.run(*args, plain=True))
    sessions = sorted(p.name for p in (rig.wshome / "rental-sessions").iterdir())
    today = time.strftime("%Y-%m-%d", time.gmtime())
    assert sessions == ["%s-%s" % (IP, today)]
    assert "host key:" in first.stdout
    second = _ok(rig.run(*args, plain=True))  # the same command continues the session
    assert sorted(p.name for p in (rig.wshome / "rental-sessions").iterdir()) == sessions
    assert "host key:" not in second.stdout and len(rig.node_calls("prepare_host")) == 1
    # a session of yesterday (UTC) that is still young is continued across midnight
    old = rig.wshome / "rental-sessions" / ("%s-2001-01-01" % IP)
    (rig.wshome / "rental-sessions" / sessions[0]).rename(old)
    _ok(rig.run(*args, plain=True))
    assert sorted(p.name for p in (rig.wshome / "rental-sessions").iterdir()) == [old.name]
    # the provider reuses the address for another node: a fresh session, no stale host key in the way
    rig.flag("hostkey", "node-key-2")
    third = _ok(rig.run(*args, plain=True))
    names = sorted(p.name for p in (rig.wshome / "rental-sessions").iterdir())
    assert names == [old.name, "%s-%s" % (IP, today)]
    assert "host key:" in third.stdout and "another node" in third.out


def test_stage_lines_show_elapsed_time_and_cost(rig):
    _ok(rig.run("--skip-smoke", "--no-download"))
    state = rig.session / "state"
    text = re.sub(r"clock_start=\d+", "clock_start=%d" % (time.time() - 5400), state.read_text())
    state.write_text(text)
    proc = _ok(rig.run("--skip-smoke", "--no-download"))
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("stage 1/8 connect: ok")][0]
    assert "1h30m" in line and "$15.0" in line and "$10/h" in line
    assert re.search(r"still billing: 1h30m\d\ds .* about \$15\.0\d at \$10/h", proc.stdout)
    proc = _ok(rig.run("--skip-smoke", "--no-download", "--rate", "20"))
    assert "$30.0" in proc.stdout and "$20/h" in proc.stdout


# ----------------------------------------------------------------------------- resume


def test_resume_skips_finished_stages_after_checking_them_on_the_node(rig):
    _ok(rig.run())
    rig.calls.write_text("")
    proc = _ok(rig.run())
    assert len(rig.node_calls("prepare_host")) == 1
    assert len(rig.node_calls("tiny_smoke")) == 2
    assert len(rig.runs()) == 1 and len(rig.node_calls("run_rental")) == 2
    assert rig.entries("scp") == []  # the node has this bundle, and the results are here
    events = _stage_events(proc.stdout)
    assert [e[1] for e in events if e[2] == "ok"] == ["connect", "bundle", "host", "download", "smoke", "wait", "run",
                                                       "fetch"]
    assert proc.stdout.count("already") >= 5
    # each skip was decided on the node, not from the state file alone
    sent = rig.remote_text()
    run_id = rig.state()["run_id"]
    for check in ("BUNDLE.txt", "fn_host_facts", "fn_ckpt", "summary.json", "results/%s.tar.gz" % run_id):
        assert check in sent, check
    assert REMINDER in proc.stdout


def test_a_replaced_node_is_not_trusted_on_local_state(rig):
    _ok(rig.run())
    first_run = rig.state()["run_id"]
    # the same address, key and session folder, but an empty machine behind it
    shutil.rmtree(rig.home / "rental")
    shutil.rmtree(rig.model.parent)
    for name in ("host-prepared", "download-state"):
        rig.unflag(name)
    (rig.node / "bin" / "hl-smi").unlink()
    rig.free_gb(60)
    time.sleep(1.1)  # run ids are UTC seconds
    proc = _ok(rig.run())
    assert len(rig.node_calls("prepare_host")) == 2
    assert len(rig.node_calls("tiny_smoke")) == 4
    assert len(rig.runs()) == 2
    assert rig.state()["run_id"] != first_run
    assert [e for e in rig.entries("scp") if e["paths"][0] == str(rig.bundle)]  # the bundle went up again
    assert "already" not in "\n".join(ln for ln in proc.stdout.splitlines() if ln.startswith("stage 3/8"))


def test_a_new_bundle_is_copied_and_the_smoke_test_reinstalls(rig):
    rig.flag("smoke-fail-fused")
    assert rig.run().returncode != 0
    rig.unflag("smoke-fail-fused")
    newer = rig.make_bundle(rig.tmp / "flash-next-rental-fedcba987654.tar.gz", "fedcba9876543210fedcba9876543210fedcba98")
    args = [a if a != str(rig.bundle) else str(newer) for a in rig.argv()[2:]]
    proc = _ok(rig.run(*args, plain=True))
    assert (rig.home / "rental" / "BUNDLE.txt").read_text().startswith("plugin sha: fedcba98")
    smoke = rig.node_calls("tiny_smoke")
    assert [c[1] for c in smoke] == ["pytorch", "fused", "pytorch", "fused"]  # a pass of the old bundle proves nothing
    assert "--reinstall" in smoke[2] and "--reinstall" not in smoke[0]
    assert len(rig.runs()) == 1 and proc.stdout.count("stage 7/8 run: ok") == 1


def test_an_ended_run_is_not_started_twice_without_new_run(rig):
    rig.flag("run-exit", "3")
    proc = rig.run()
    assert proc.returncode == 1 and "stage 7/8 run: FAILED" in proc.stdout and "exit 3" in proc.stdout
    assert "stage 8/8 fetch: ok" in proc.stdout  # the results of a failed run are fetched all the same
    run_id = rig.state()["run_id"]
    again = rig.run()
    assert again.returncode == 1 and len(rig.runs()) == 1
    assert "--new-run" in again.out and "--from" in again.out
    resumed = rig.run("--from", "A")
    assert rig.runs()[1][:6] == ["--model", str(rig.model), "--run-id", run_id, "--quick-gates", "--variants"]
    assert rig.runs()[1][-2:] == ["--from", "A"] and resumed.returncode == 1
    rig.unflag("run-exit")
    time.sleep(1.1)
    _ok(rig.run("--new-run"))
    assert len(rig.runs()) == 3 and rig.runs()[2][3] != run_id and "--from" not in rig.runs()[2]


# ----------------------------------------------------------------------------- the data disk


def test_one_blank_big_disk_is_chosen_and_named(rig):
    proc = _ok(rig.run("--skip-smoke"))
    assert rig.node_calls("prepare_host") == [["--data-disk", "/dev/vdc", "--mount", str(rig.model.parent)]]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("data disk: /dev/vdc")][0]
    assert "only blank disk" in line and "26400 GB" in line and "60 GB free" in line


@pytest.mark.parametrize("rows, reason", [
    ([BIG_BLANK, ("vdd", "disk", 500 * GB, "", "", "", "")], "2 blank disks"),
    ([("vdc", "disk", 200 * GB, "", "", "", "")], "no blank disk of at least 300 GB"),
    ([("vdc", "disk", 26400 * GB, "ext4", "data", "", "")], "no blank disk of at least 300 GB"),
    ([("vdc", "disk", 26400 * GB, "", "", "", ""), ("vdc1", "part", 26000 * GB, "", "", "", "vdc")],
     "no blank disk of at least 300 GB"),
    ([("vdc", "disk", 26400 * GB, "", "", "/mnt/scratch", "")], "no blank disk of at least 300 GB"),
    ([], "no blank disk of at least 300 GB"),
])
def test_an_unclear_data_disk_stops_before_anything_changes(rig, rows, reason):
    rig.disks(*ROOT_DISK, *rows)
    proc = rig.run()
    assert proc.returncode == 3, proc.out
    assert reason in proc.out
    assert rig.node_calls("prepare_host") == [] and not (rig.nlog / "mkfs").exists()
    assert not (rig.job_dir() / "host.run").exists()  # no job was started
    assert "NAME    SIZE TYPE FSTYPE LABEL MOUNTPOINT" in proc.out  # the lsblk table
    rerun = [ln for ln in proc.out.splitlines() if "--data-disk /dev/" in ln]
    assert rerun and all(str(SCRIPT) in ln and IP in ln and "--session" in ln for ln in rerun)
    assert "stage 3/8 host: FAILED" in proc.out and REMINDER in proc.out
    assert "stage 4/8" not in proc.out


def test_enough_free_space_needs_no_data_disk(rig):
    rig.free_gb(400)
    proc = _ok(rig.run("--skip-smoke"))
    assert rig.node_calls("prepare_host") == [[]]
    assert "no data disk is needed" in proc.stdout and not (rig.nlog / "mkfs").exists()


def test_a_complete_checkpoint_needs_neither_disk_nor_download(rig):
    rig.model.mkdir(parents=True)
    (rig.model / "config.json").write_text("{}")
    (rig.model / "model-00001-of-00040.safetensors").write_text("")
    rig.flag("du-seq", "%d\n" % (186 * GB // 1024))
    proc = _ok(rig.run("--skip-smoke"))
    assert rig.node_calls("prepare_host") == [[]]
    assert rig.node_calls("run_rental")[0][-1] != "--download-only" and len(rig.node_calls("run_rental")) == 1
    assert "checkpoint is complete" in proc.stdout


@pytest.mark.parametrize("rows, passed", [
    ([BIG_BLANK, ("vdd", "disk", 500 * GB, "", "", "", "")], "/dev/vdd"),
    ([("vdd", "disk", 26400 * GB, "ext4", "fn-models", "", "")], "/dev/vdd"),  # the disk prepare_host.sh made
])
def test_a_named_data_disk_is_used(rig, rows, passed):
    rig.disks(*ROOT_DISK, *rows)
    _ok(rig.run("--skip-smoke", "--data-disk", "vdd"))
    assert rig.node_calls("prepare_host") == [["--data-disk", passed, "--mount", str(rig.model.parent)]]


@pytest.mark.parametrize("rows, word", [
    ([("vdd", "disk", 26400 * GB, "xfs", "", "", "")], "filesystem"),
    ([("vdd", "disk", 26400 * GB, "", "", "", ""), ("vdd1", "part", 100 * GB, "", "", "", "vdd")], "partitions"),
    ([("vdd", "disk", 26400 * GB, "", "", "/mnt/x", "")], "mounted"),
    ([], "not a disk"),
])
def test_a_named_disk_with_data_is_never_formatted(rig, rows, word):
    rig.disks(*ROOT_DISK, *rows)
    proc = rig.run("--data-disk", "/dev/vdd")
    assert proc.returncode == 3, proc.out
    assert word in proc.out and "never formats" in proc.out
    assert rig.node_calls("prepare_host") == [] and not (rig.nlog / "mkfs").exists()


def test_without_a_readable_disk_listing_nothing_is_chosen_and_a_named_disk_is_left_to_prepare_host(rig):
    rig.flag("lsblk-pairs", "")  # lsblk -P printed nothing this command can read
    proc = rig.run()
    assert proc.returncode == 3 and "no blank disk of at least 300 GB" in proc.out
    assert rig.node_calls("prepare_host") == []
    proc = _ok(rig.run("--skip-smoke", "--data-disk", "/dev/vdc"))
    assert rig.node_calls("prepare_host") == [["--data-disk", "/dev/vdc", "--mount", str(rig.model.parent)]]
    assert "prepare_host.sh alone checks that it is blank" in proc.stdout


def test_a_reboot_request_stops_and_the_node_is_never_rebooted(rig):
    rig.flag("host-reboot")
    proc = rig.run()
    assert proc.returncode == 3, proc.out
    assert "does not see all cards: reboot and rerun this script" in proc.out
    assert "never reboots the node" in proc.out and "stage 3/8 host: FAILED" in proc.out
    assert "stage 4/8" not in proc.out and REMINDER in proc.out
    for row in rig.entries("ssh"):
        assert not re.search(r"\b(reboot|shutdown|poweroff|halt)\b", _body(row["cmd"][0])), row["cmd"][0]
    rig.unflag("host-reboot")
    _ok(rig.run("--skip-smoke"))  # after the operator's reboot the same command goes on
    assert len(rig.node_calls("prepare_host")) == 2


def test_a_failed_host_step_stops_with_its_remedy(rig):
    rig.flag("host-fail")
    proc = rig.run()
    assert proc.returncode == 1 and "remedy: apt-get install -y docker.io failed" in proc.out
    assert "stage 3/8 host: FAILED" in proc.out and "stage 4/8" not in proc.out
    assert "asks for a reboot" not in proc.out  # a note that names a reboot is not the remedy of this failure


def test_sudo_with_a_password_stops_at_stage_one(rig):
    rig.flag("sudo-password")
    proc = rig.run()
    assert proc.returncode == 1 and "stage 1/8 connect: FAILED" in proc.out and "password" in proc.out
    assert rig.entries("scp") == [] and REMINDER in proc.out


# ----------------------------------------------------------------------------- the download


def test_the_download_is_started_before_the_smoke_test_and_waited_for_after_it(rig):
    rig.flag("download-polls-init", "2")
    rig.flag("du-seq", "".join("%d\n" % (gb * GB // 1024) for gb in (31, 93, 186)))
    proc = _ok(rig.run(env={"FN_RENTAL_POLL_S": "1.1"}))
    out = proc.stdout
    assert out.index("stage 4/8 download: ok") < out.index("stage 5/8 smoke: start") < out.index("stage 6/8 wait: start")
    progress = [ln for ln in out.splitlines() if ln.startswith("download: ")]
    assert progress, out
    assert any(re.search(r"\d+\.\d GB on disk of about 185\.5 GB, \d+(\.\d+)? GB/s, ETA \d", ln) for ln in progress), progress
    assert "stage 6/8 wait: ok" in out


def test_a_failed_download_prints_the_container_log_and_stops_before_the_run(rig):
    rig.flag("download-result", "exited 1")
    rig.flag("download-log", "Fetching 40 files\nOSError: [Errno 28] No space left on device\n")
    proc = rig.run()
    assert proc.returncode == 1, proc.out
    assert "stage 6/8 wait: FAILED" in proc.out and "No space left on device" in proc.out
    assert "exit code 1" in proc.out
    assert rig.runs() == [] and "stage 7/8" not in proc.out and REMINDER in proc.out
    # the same command starts the download again (hf resumes in the folder)
    rig.unflag("download-result")
    _ok(rig.run())
    assert len([c for c in rig.node_calls("run_rental") if "--download-only" in c]) == 2 and len(rig.runs()) == 1


def test_a_download_that_cannot_start_stops_at_stage_four(rig):
    rig.flag("download-start-fail")
    proc = rig.run("--data-disk", "vdc")
    assert proc.returncode == 1 and "stage 4/8 download: FAILED" in proc.out
    assert "--download needs 250 GB" in proc.out and rig.node_calls("tiny_smoke") == []


def test_no_download_skips_both_download_stages(rig):
    proc = _ok(rig.run("--no-download", "--skip-smoke"))
    assert all("--download-only" not in c for c in rig.node_calls("run_rental"))
    assert "docker logs" not in rig.remote_text() and "stage 6/8 wait: ok (skipped: --no-download)" in proc.stdout
    assert re.search(r"stage 4/8 download: ok \(skipped: --no-download\)", proc.stdout)


# ----------------------------------------------------------------------------- the smoke test


def _local_smoke_log(rig, mode):
    found = list((rig.session / "results").glob("smoke-*-%s/%s/server0.log" % (mode, mode)))
    assert len(found) == 1, found
    return found[0]


def test_a_smoke_failure_stops_before_the_run_and_fetches_the_folder(rig):
    rig.flag("smoke-fail-pytorch")
    proc = rig.run()
    assert proc.returncode == 1, proc.out
    assert "stage 5/8 smoke: FAILED" in proc.out
    assert rig.runs() == [] and "stage 6/8" not in proc.out and "stage 7/8" not in proc.out
    assert [c[1] for c in rig.node_calls("tiny_smoke")] == ["pytorch"]  # fused is not tried after it
    log = _local_smoke_log(rig, "pytorch")
    assert log.read_text() == "server log of pytorch\n"
    assert not list((rig.session / "results").rglob("*.safetensors"))
    tail = proc.out[proc.out.index("stage 5/8 smoke: FAILED"):]
    assert "tiny smoke [pytorch]: FAILED (the server exited with code 1 before it was healthy)" in tail
    assert "RuntimeError: synthetic pytorch failure" in tail
    assert "server log: %s" % log in tail
    assert REMINDER in tail
    assert "--continue-after-smoke-failure" not in tail  # nothing overrides a failed pytorch mode


def test_a_fused_failure_stops_unless_overridden(rig):
    rig.flag("smoke-fail-fused")
    proc = rig.run()
    assert proc.returncode == 1 and rig.runs() == []
    assert [c[1] for c in rig.node_calls("tiny_smoke")] == ["pytorch", "fused"]
    assert "server log: %s" % _local_smoke_log(rig, "fused") in proc.out
    assert "--continue-after-smoke-failure fused" in proc.out  # the exact rerun
    over = _ok(rig.run("--continue-after-smoke-failure", "fused"))
    assert len(rig.runs()) == 1
    # pytorch passed and fused failed with this very bundle: neither is paid for again
    assert [c[1] for c in rig.node_calls("tiny_smoke")] == ["pytorch", "fused"]
    assert "failed before with this bundle" in over.stdout
    assert len([e for e in rig.entries("scp") if e["paths"][0] == str(rig.bundle)]) == 1  # the node had the bundle
    assert "PyTorch" in over.stdout and "stage 7/8 run: ok" in over.stdout


def test_a_failed_pytorch_mode_can_never_be_overridden(rig):
    for value in ("pytorch", "pytorch,fused", "all"):
        proc = rig.run("--continue-after-smoke-failure", value)
        assert proc.returncode == 2 and "pytorch" in proc.out, value
    assert rig.entries() == []
    rig.flag("smoke-fail-pytorch")
    proc = rig.run("--continue-after-smoke-failure", "fused")
    assert proc.returncode == 1 and rig.runs() == []


def test_the_hc_mode_is_passed_on_and_can_be_overridden(rig):
    rig.flag("smoke-fail-hc")
    proc = rig.run("--smoke-modes", "pytorch,fused,hc", "--hc-fused")
    assert proc.returncode == 1 and rig.runs() == []
    assert [c[1] for c in rig.node_calls("tiny_smoke")] == ["pytorch", "fused", "hc"]
    assert "stage 5/8 smoke: FAILED (the hc mode failed" in proc.out
    assert "--continue-after-smoke-failure hc" in proc.out and "torch hyper-connections" in proc.out
    over = _ok(rig.run("--smoke-modes", "pytorch,fused,hc", "--hc-fused", "--continue-after-smoke-failure", "fused,hc"))
    assert [c[1] for c in rig.node_calls("tiny_smoke")] == ["pytorch", "fused", "hc"]
    assert rig.runs()[0][-1] == "--hc-fused" and "FAILED and overridden: hc" in over.stdout
    assert "job smoke-hc: ended 1" in _ok(rig.run("--status")).stdout


def test_smoke_modes_and_skip_smoke(rig):
    proc = _ok(rig.run("--smoke-modes", "fused", "--smoke-arg", "--no-combined-kernel-db"))
    assert [c[:2] for c in rig.node_calls("tiny_smoke")] == [["--gdn", "fused"]]
    assert rig.node_calls("tiny_smoke")[0][-1] == "--no-combined-kernel-db"
    assert "stage 5/8 smoke: ok" in proc.stdout


def test_skip_smoke(rig):
    proc = _ok(rig.run("--skip-smoke"))
    assert rig.node_calls("tiny_smoke") == [] and len(rig.runs()) == 1
    assert re.search(r"stage 5/8 smoke: ok \(skipped: --skip-smoke\)", proc.stdout)


# ----------------------------------------------------------------------------- the run


def test_the_run_surfaces_phases_gates_and_the_first_token(rig):
    proc = _ok(rig.run("--skip-smoke"))
    out = proc.stdout
    for line in (">>> phase: preflight", ">>> preflight: PASSED after 3 s", ">>> phase: A",
                 ">>> variant A: PASSED after 900 s"):
        assert line in out, line
    assert re.search(r">>> RUN \S+: PASSED", out)
    token = [ln for ln in out.splitlines() if "first token after" in ln]
    assert len(token) == 1 and "FIRST TOKEN" in token[0] and "variant A server0: first token after 41.2 s: 'Hello'" in token[0]
    rig.flag("run-exit", "5")
    time.sleep(1.1)
    proc = rig.run("--skip-smoke", "--new-run")
    assert re.search(r">>> RUN \S+: FAILED at A", proc.stdout) and ">>> variant A: FAILED after 900 s" in proc.stdout


def test_run_options_reach_run_rental(rig):
    _ok(rig.run("--skip-smoke", "--variants", "A", "--no-quick-gates", "--hc-fused", "--run-arg", "--server-timeout",
                "--run-arg", "2400"))
    run = rig.runs()[0]
    assert run[4:] == ["--variants", "A", "--hc-fused", "--server-timeout", "2400"]


def test_hc_fused_needs_a_bundle_that_knows_it(rig):
    stage = rig.tmp / "old-stage"
    old = rig.make_bundle(rig.tmp / "old.tar.gz", SHA, stage=stage)
    (stage / "run_rental.sh").write_text(BUNDLE_RUN_RENTAL.replace("# --hc-fused\n", ""))
    with tarfile.open(old, "w:gz") as tar:
        for name in ("BUNDLE.txt", "run_rental.sh", "prepare_host.sh", "tiny_smoke.sh", "plugin", "vllm"):
            tar.add(stage / name, arcname=name)
    args = [a if a != str(rig.bundle) else str(old) for a in rig.argv()[2:]]
    proc = rig.run(*args, "--skip-smoke", "--hc-fused", plain=True)
    assert proc.returncode == 1 and "--hc-fused" in proc.out and "stage 7/8 run: FAILED" in proc.out
    assert rig.runs() == []


def test_paths_with_spaces_and_quotes_survive(tmp_path):
    rig = Rig(tmp_path / "work station")
    rig.model = rig.node / "data disk" / "it's models" / "Qwen3.8 Flash $HOME"
    rig.session = rig.tmp / "my sessions" / "node one"
    try:
        proc = _ok(rig.run("--run-arg", "--env", "--run-arg", "A B=c d"))
        assert rig.node_calls("prepare_host") == [["--data-disk", "/dev/vdc", "--mount", str(rig.model.parent)]]
        calls = rig.node_calls("run_rental")
        assert calls[0] == ["--model", str(rig.model), "--download-only"]
        assert calls[1][:2] == ["--model", str(rig.model)] and calls[1][-2:] == ["--env", "A B=c d"]
        assert (rig.model / "config.json").is_file()  # the fake download wrote where the path really points
        assert (rig.session / "results" / (rig.state()["run_id"] + ".tar.gz")).is_file()
        assert "stage 8/8 fetch: ok" in proc.stdout
        assert (rig.session / "known_hosts").is_file()
        for row in rig.entries():  # a file name with a space is quoted inside the option, as ssh_config wants it
            assert ["-o", 'UserKnownHostsFile="%s/known_hosts"' % rig.session] in row["opts"], row["opts"]
    finally:
        rig.cleanup()


# ----------------------------------------------------------------------------- jobs: drops, Ctrl-C, status, stop


def test_a_dropped_connection_is_retried_and_the_real_exit_code_returned(rig):
    rig.flag("run-lines", "40")
    rig.flag("run-exit", "7")
    rig.flag("drop-follow", "run 12")
    proc = rig.run("--skip-smoke")
    assert proc.returncode == 1, proc.out
    lost = [ln for ln in proc.out.splitlines() if "connection lost, retrying" in ln]
    assert len(lost) == 1 and re.search(r"outage so far: \d+ s", lost[0]) and "Broken pipe" in lost[0]
    assert "connection is back" in proc.out
    assert "stage 7/8 run: FAILED" in proc.out and "exit 7" in proc.out
    assert "half a li" not in proc.out  # the cut line is not shown; it comes again whole
    node_log = (rig.job_dir() / "run.log").read_text()
    assert (rig.session / "jobs" / "run.log").read_text() == node_log  # every line once, none lost
    assert node_log.count("load line 13\n") == 1 and proc.stdout.count("load line 13\n") == 1
    assert "stage 8/8 fetch: ok" in proc.out


def test_an_outage_at_the_start_is_retried_with_a_bounded_number_of_attempts(rig):
    rig.flag("fail-ssh", "2")
    proc = _ok(rig.run("--skip-smoke", "--no-download"))
    assert proc.out.count("connection lost, retrying") == 2
    rig.flag("down")
    rig.calls.write_text("")
    proc = rig.run("--skip-smoke", "--no-download")
    assert proc.returncode == 1 and "cannot reach" in proc.out and "Connection refused" in proc.out
    assert len(rig.entries("ssh")) == 3  # FN_RENTAL_SSH_TRIES
    assert REMINDER in proc.out


def test_a_refused_key_is_not_retried(rig):
    rig.flag("denied")
    proc = rig.run()
    assert proc.returncode == 1 and "Permission denied" in proc.out and len(rig.entries("ssh")) == 1
    assert "connection lost" not in proc.out and REMINDER in proc.out


def _detach_during_run(rig):
    rig.flag("run-hold")
    proc = rig.start("--skip-smoke")
    rig.wait_for(proc, "first token after")
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=30) == 130
    return proc.out_path.read_text()


def test_ctrl_c_detaches_and_the_same_command_attaches_again(rig):
    out = _detach_during_run(rig)
    assert "detached" in out and "nothing on the node was stopped" in out
    rerun = [ln for ln in out.splitlines() if str(SCRIPT) in ln and "--skip-smoke" in ln]
    assert rerun and IP in rerun[-1]  # how to re-attach: the same command
    assert REMINDER in out
    assert "stage 7/8 run: FAILED" not in out
    time.sleep(0.5)
    assert rig.job_alive("run")  # the job on the node goes on
    assert not (rig.ctl / "run-got-sigint").exists()
    again = rig.start("--skip-smoke")
    rig.wait_for(again, "attaching")
    rig.unflag("run-hold")
    assert again.wait(timeout=60) == 0, again.out_path.read_text()
    text = again.out_path.read_text()
    assert len(rig.runs()) == 1  # attached, not started a second time
    assert "stage 7/8 run: ok" in text and "stage 8/8 fetch: ok" in text
    assert text.count("first token after") == 0  # shown before the Ctrl-C; the log goes on where it stopped
    assert (rig.session / "jobs" / "run.log").read_text() == (rig.job_dir() / "run.log").read_text()


def test_ctrl_c_during_the_host_stage_leaves_prepare_host_running(rig):
    rig.flag("host-hold")
    proc = rig.start()
    rig.wait_for(proc, "host step 1 (checks): ok")
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=30) == 130
    assert rig.job_alive("host") and REMINDER in proc.out_path.read_text()
    again = rig.start("--skip-smoke")
    rig.wait_for(again, "attaching")
    rig.unflag("host-hold")
    assert again.wait(timeout=60) == 0, again.out_path.read_text()
    assert len(rig.node_calls("prepare_host")) == 1


def test_a_smoke_call_that_ends_while_detached_is_not_run_again(rig):
    rig.flag("smoke-hold")
    proc = rig.start()
    rig.wait_for(proc, "container fn-smoke: starting")
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=30) == 130
    assert rig.job_alive("smoke-pytorch")
    again = rig.start()  # attaches while it still runs
    rig.wait_for(again, "job smoke-pytorch: already running on the node: attaching")
    again.send_signal(signal.SIGINT)
    assert again.wait(timeout=30) == 130
    rig.unflag("smoke-hold")
    deadline = time.time() + 20
    while not (rig.job_dir() / "smoke-pytorch.rc").exists() and time.time() < deadline:
        time.sleep(0.05)
    last = _ok(rig.run())
    assert [c[1] for c in rig.node_calls("tiny_smoke")] == ["pytorch", "fused"]
    assert "ended on the node while nobody watched" in last.stdout
    assert rig.state()["smoke_pytorch"].startswith("passed:%s:smoke-" % SHA)
    assert "stage 5/8 smoke: ok" in last.stdout and len(rig.runs()) == 1


def test_a_run_that_ends_while_detached_is_collected_not_restarted(rig):
    _detach_during_run(rig)
    rig.flag("run-exit", "4")
    rig.unflag("run-hold")
    deadline = time.time() + 20
    while not (rig.job_dir() / "run.rc").exists() and time.time() < deadline:
        time.sleep(0.05)
    proc = rig.run("--skip-smoke")
    assert proc.returncode == 1, proc.out
    assert len(rig.runs()) == 1
    assert "ended on the node while nobody watched" in proc.stdout
    assert "stage 7/8 run: FAILED" in proc.stdout and "exit 4" in proc.stdout
    assert re.search(r">>> RUN \S+: FAILED at A", proc.stdout)  # the end of its log is shown
    assert "stage 8/8 fetch: ok" in proc.stdout and rig.state()["run_state"] == "ended:4"
    assert (rig.session / "results" / (rig.state()["run_id"] + ".tar.gz")).is_file()


def _tree(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in folder.rglob("*") if p.is_file())


def test_status_prints_the_jobs_and_changes_nothing(rig):
    _detach_during_run(rig)
    before_home, before_calls = _tree(rig.home), {n: rig.node_calls(n) for n in ("prepare_host", "tiny_smoke", "run_rental")}
    state_before = (rig.session / "state").read_text()
    proc = _ok(rig.run("--status"))
    out = proc.stdout
    assert re.search(r"job host: ended 0", out) and re.search(r"job run: running \d+", out)
    assert "job smoke-pytorch: none" in out
    assert "variant A server0: first token after 41.2 s" in out  # the last lines of the run's log
    assert "host preparation summary" in out
    assert "download:" in out and REMINDER in out
    assert _tree(rig.home) == before_home
    assert {n: rig.node_calls(n) for n in before_calls} == before_calls
    assert (rig.session / "state").read_text() == state_before
    assert rig.job_alive("run") and not (rig.ctl / "run-got-sigint").exists()
    assert "stage " not in out


def test_stop_interrupts_the_run_and_fetches(rig):
    _detach_during_run(rig)
    proc = _ok(rig.run("--stop"))
    assert (rig.ctl / "run-got-sigint").exists()  # run_rental.sh got SIGINT and took its own end path
    assert not rig.job_alive("run")
    assert (rig.job_dir() / "run.rc").read_text().strip() == "130"
    out = proc.stdout
    assert "interrupted by SIGINT" in out and "SEND THIS FILE:" in out
    run_id = rig.state()["run_id"]
    assert (rig.session / "results" / (run_id + ".tar.gz")).is_file()
    assert "| A | interrupted |" in out  # summary.md of the stopped run
    assert "stage 8/8 fetch: ok" in out and REMINDER in out
    assert rig.state()["run_state"] == "ended:130"
    assert len(rig.runs()) == 1


def test_stop_without_a_run_says_so(rig):
    _ok(rig.run("--skip-smoke"))
    proc = _ok(rig.run("--stop"))
    assert "no run is running" in proc.stdout and REMINDER in proc.stdout
    assert not (rig.ctl / "run-got-sigint").exists()


# ----------------------------------------------------------------------------- the rules for remote commands


def test_remote_commands_keep_the_rules(rig):
    _ok(rig.run())
    _ok(rig.run("--status"))
    _ok(rig.run("--stop"))
    sent = [_body(r["cmd"][0]) for r in rig.entries("ssh")]
    for text in sent:
        assert not re.search(r"\b(reboot|shutdown|poweroff|mkfs|wipefs|dd|parted|fdisk)\b", text), text
        for match in re.finditer(r"\brm\s+-\S*[rR]\S*\s+(\S+)", text):
            assert ".jobs" in match.group(1), text  # nothing is removed recursively outside ~/rental/.jobs
        assert not re.search(r"\bsudo\s+(?!-n\b)", text), text  # sudo never waits for a password
    helpers = rig.entries("ssh")[0]["cmd"][0].split("# --- end of the node helpers")[0]
    assert not re.search(r"\brm\s+-\S*[rR]", helpers) and not re.search(r"\bsudo\s+(?!-n\b)", helpers)
    # sudo only runs what the stages name
    allowed = ("-n true", "-n bash prepare_host.sh", "-n bash run_rental.sh", "-n bash tiny_smoke.sh", "-n docker info",
               "-n docker inspect", "-n docker logs", "-n du -sk", "-n find", "-n lsblk", "-n tar czf", "-n kill -INT")
    for line in (rig.nlog / "sudo").read_text().splitlines():
        assert line.startswith(allowed), line


def test_the_reminder_is_printed_on_every_exit_path(rig):
    outputs = {}
    outputs["usage error"] = rig.run("--no-such-option").stdout
    outputs["dry run"] = rig.run("--dry-run").stdout
    rig.flag("down")
    outputs["unreachable"] = rig.run().stdout
    rig.unflag("down")
    rig.disks(*ROOT_DISK)
    outputs["data disk"] = rig.run().stdout
    rig.disks(*ROOT_DISK, BIG_BLANK)
    rig.flag("smoke-fail-pytorch")
    outputs["smoke"] = rig.run().stdout
    rig.unflag("smoke-fail-pytorch")
    outputs["success"] = rig.run().stdout
    outputs["status"] = rig.run("--status").stdout
    outputs["stop"] = rig.run("--stop").stdout
    for name, out in outputs.items():
        tail = "\n".join(out.strip().splitlines()[-2:])  # the last thing on stdout, whatever came before
        assert "still billing" in tail or "is billing" in tail, name
        assert REMINDER in tail and "cannot release the node" in tail, name


# ----------------------------------------------------------------------------- building the bundle


def _checkout(tmp_path, rig):
    """A temp git checkout holding rent.sh and a stand-in make_rental_bundle.sh next to it."""
    repo = tmp_path / "checkout"
    tools = repo / "tools" / "flash_next" / "rental"
    tools.mkdir(parents=True)
    shutil.copy(SCRIPT, tools / "rent.sh")
    _exe(tools / "make_rental_bundle.sh", FAKE_MAKE_BUNDLE)
    shutil.copy(rig.bundle, repo / "template.tar.gz")
    (repo / ".gitignore").write_text("dist/\nmake-bundle-calls\nfail-build\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "one"]):
        subprocess.run(git + args, check=True, env=env, capture_output=True, timeout=60)
    head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, env=env, capture_output=True, text=True).stdout.strip()
    return repo, tools / "rent.sh", head


def test_the_bundle_is_built_from_the_checkout_and_a_dirty_tree_is_refused(rig, tmp_path):
    repo, script, head = _checkout(tmp_path, rig)
    env = dict(rig.env(FN_VLLM_SRC="/src/vllm"), PATH="%s:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin" % rig.bin)
    base = [BASH, str(script), IP, "--session", str(rig.session), "--key", str(rig.key), "--model", str(rig.model),
            "--skip-smoke", "--no-download"]

    def call(*extra):
        proc = subprocess.run(base + list(extra), capture_output=True, text=True, timeout=120, env=env, cwd=tmp_path,
                              preexec_fn=_default_signals)
        proc.out = proc.stdout + proc.stderr
        return proc

    (repo / "fail-build").write_text("")
    proc = call()  # the build's own message and the way out; nothing is copied
    assert proc.returncode == 1 and "stage 2/8 bundle: FAILED (make_rental_bundle.sh failed)" in proc.out
    assert "vLLM commit 8f816a3 not found" in proc.out and "--vllm-src PATH" in proc.out and "--bundle FILE" in proc.out
    assert rig.entries("scp") == [] and REMINDER in proc.out
    (repo / "fail-build").unlink()
    (repo / "make-bundle-calls").unlink()

    proc = call()
    assert proc.returncode == 0, proc.out
    assert (repo / "make-bundle-calls").read_text().split("\n")[0] == "--out %s/dist --vllm-src /src/vllm" % repo.resolve()
    assert (rig.home / "rental" / "BUNDLE.txt").read_text().strip() == "plugin sha: " + head
    assert (rig.home / ("flash-next-rental-%s.tar.gz" % head[:12])).is_file()
    assert rig.state()["bundle_sha"] == head

    # the node has this commit: nothing is built or copied again, a dirty tree only gets a warning
    (repo / "notes.txt").write_text("uncommitted\n")
    proc = call()
    assert proc.returncode == 0, proc.out
    assert len((repo / "make-bundle-calls").read_text().splitlines()) == 1
    assert "uncommitted" in proc.out

    # a new commit goes to the node with --new-run (an ended run keeps its bundle), and a dirty tree cannot be bundled
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    genv = {"PATH": env["PATH"], "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "two"], check=True, env=genv, capture_output=True)
    proc = call()
    assert proc.returncode == 0 and "the bundle it ran" in proc.out, proc.out
    assert (rig.home / "rental" / "BUNDLE.txt").read_text().strip() == "plugin sha: " + head
    proc = call("--new-run")
    assert proc.returncode == 1 and "not clean" in proc.out and "--allow-dirty" in proc.out
    assert "stage 2/8 bundle: FAILED" in proc.out
    assert len((repo / "make-bundle-calls").read_text().splitlines()) == 1 and len(rig.runs()) == 1
    time.sleep(1.1)
    proc = call("--new-run", "--allow-dirty")
    assert proc.returncode == 0, proc.out
    assert (repo / "make-bundle-calls").read_text().splitlines()[1].endswith("--vllm-src /src/vllm --allow-dirty")
    assert (rig.home / "rental" / "BUNDLE.txt").read_text().strip() != "plugin sha: " + head
    assert len(rig.runs()) == 2


# ----------------------------------------------------------------------------- the page


def test_the_page_says_what_the_command_does():
    page = (REPO_ROOT / "docs" / "design" / "flash-next" / "run-sheets" / "one-command.md").read_text(encoding="utf-8")
    script = SCRIPT.read_text()
    proc = subprocess.run([BASH, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and proc.stdout.startswith("usage: rent.sh <ip> [options]")
    assert REMINDER not in proc.stdout  # asking for help is not a stop of a session
    own = set(re.findall(r"^\s{2}(--[a-z][a-z-]*)", proc.stdout, re.M)) | {"--help"}
    assert len(own) >= 20
    node_side = {"--gdn", "--download-only", "--quick-gates", "--out"}  # options of the other scripts the page quotes
    named = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", page))
    assert named - own - node_side == set()
    for option in ("--status", "--stop", "--dry-run", "--data-disk", "--continue-after-smoke-failure", "--from",
                   "--new-run", "--session", "--rate"):
        assert option in named, option
    stages = re.findall(r"^\| (\d) (\w+) \|", page, re.M)
    assert stages == [(str(n), name) for n, name in enumerate(
        ["connect", "bundle", "host", "download", "smoke", "wait", "run", "fetch"], 1)]
    for name in ("connect", "bundle", "host", "download", "smoke", "wait", "run", "fetch"):
        assert re.search(r"stage_start \d %s\b" % name, script), name
    # the numbers and the messages the page quotes are the script's
    assert "at least 300 GB" in page and re.search(r"^DISK_MIN_GB=300\b", script, re.M)
    assert "less than 250 GB free" in page and re.search(r"^DOWNLOAD_NEED_GB=250\b", script, re.M)
    assert "--quick-gates --variants A,C" in page and 'VARIANTS="A,C"' in script
    assert "`sesterce`" in page and "~/.ssh/sesterce_ed25519" in page
    for message in ("no data disk chosen; nothing was changed on the node", "prepare_host.sh asks for a reboot",
                    "mode failed: ", "the download container fn-download ended with exit code"):
        assert message in page and message in script, message
    assert ">>> FIRST TOKEN >>> variant A server0: first token after" in page and ">>> FIRST TOKEN >>> " in script
    assert "Only the provider's console stops the billing" in page
