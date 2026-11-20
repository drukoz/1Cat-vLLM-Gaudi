# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/cardq.sh, the card queue of the rented node: the per-card locks, the card sets,
the order of waiting jobs, the maximum hold, cancel and the ledger.

The jobs are stub commands that record when they held which card. On a host without flock(1) and
setsid(1) (macOS) two Perl stand-ins with the same system calls are put on PATH; on Linux the real
tools run. `docker` is a stand-in that records its calls.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CARDQ = REPO_ROOT / "tools" / "flash_next" / "rental" / "farm" / "cardq.sh"

FLOCK_SHIM = r"""#!/usr/bin/perl
# flock [-n] [-u] FD: flock(2) on a descriptor inherited from the caller, as util-linux flock does.
use Fcntl qw(:flock);
my $op = LOCK_EX; my $nb = 0;
while (@ARGV && $ARGV[0] =~ /^-/) { my $a = shift; $nb = 1 if $a eq '-n'; $op = LOCK_UN if $a eq '-u'; }
my $fd = shift;
open(my $fh, ">>&=", $fd) or exit 1;
exit(flock($fh, $op | ($nb ? LOCK_NB : 0)) ? 0 : 1);
"""

SETSID_SHIM = r"""#!/usr/bin/perl
# setsid [-f] COMMAND...: a new session for COMMAND; forks first with -f or when it leads a group.
use POSIX qw(setsid getpid getpgrp);
my $fork = 0;
while (@ARGV && $ARGV[0] =~ /^-/) { my $a = shift; $fork = 1 if $a eq '-f'; }
if ($fork || getpid() == getpgrp()) { my $pid = fork(); exit 0 if $pid; }
setsid();
exec @ARGV or exit 127;
"""

FAKE_DOCKER = r"""#!/bin/bash
all="$*"
echo "$all" >> "$FAKE_DOCKER_LOG"
case "$all" in
  "ps -aq --filter label=cardq.job="*) cat "$FAKE_STATE/labelled-${all##*=}" 2>/dev/null ;;
  "ps -q") cat "$FAKE_STATE/running" 2>/dev/null ;;
  "inspect --format {{.HostConfig.Runtime}} "*) cat "$FAKE_STATE/runtime-${all##* }" 2>/dev/null ;;
  "inspect --format {{.Name}} "*) echo "/name-${all##* }" ;;
esac
exit 0
"""

NOW = "perl -MTime::HiRes=time -e 'printf \"%.3f\\n\", time'"


class Farm:

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.q = tmp_path / "q"
        self.events = tmp_path / "events"
        self.gates = tmp_path / "gates"
        self.gates.mkdir()
        self.state = tmp_path / "state"
        self.state.mkdir()
        home = tmp_path / "home"
        home.mkdir()
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, text in (("flock", FLOCK_SHIM), ("setsid", SETSID_SHIM)):
            if shutil.which(name) is None:
                (bin_dir / name).write_text(text)
                (bin_dir / name).chmod(0o755)
        (bin_dir / "docker").write_text(FAKE_DOCKER)
        (bin_dir / "docker").chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if k != "BASH_ENV" and not k.startswith("CARDQ_")}
        self.env.update({
            "PATH": "%s:%s" % (bin_dir, os.environ.get("PATH", "/usr/bin:/bin")),
            "HOME": str(home),
            "CARDQ_DIR": str(self.q),
            "CARDQ_POLL_S": "0.1",
            "CARDQ_GRACE_S": "1",
            "CARDQ_MARGIN_S": "0",
            "CARDQ_SUDO": "",
            "CARDQ_ALLOW_ROOT": "1",
            "EVENTS": str(self.events),
            "GATES": str(self.gates),
            "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
            "FAKE_STATE": str(self.state),
        })

    def cq(self, *args, stdin=None, timeout=120, **env):
        run_env = dict(self.env, **{k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", str(CARDQ), *args], input=stdin, capture_output=True, text=True,
                              env=run_env, timeout=timeout, stdin=None if stdin is not None else subprocess.DEVNULL)

    def submit(self, cards, script, track="alpha", name=None, hold=None, check=True):
        name = name or "job-%d" % random.randrange(10**9)
        args = ["submit", "--track", track, "--name", name, "--cards", cards]
        if hold is not None:
            args += ["--max-hold", str(hold)]
        proc = self.cq(*args, stdin=script)
        if check:
            assert proc.returncode == 0, proc.stderr
            return proc.stdout.strip()
        return proc

    def hold(self, cards, seconds, **kw):
        """A job that records the cards it got, holds them for `seconds`, and records the end."""
        script = ('echo "S $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\nsleep %s\n'
                  'echo "E $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\n' % (NOW, seconds, NOW))
        return self.submit(cards, script, **kw)

    def block(self, cards, **kw):
        """A job that records the cards it got and holds them until release(job). The test decides when it
        ends, so the steps that must happen while it runs do so on a slow or busy host too (a fixed sleep
        of two seconds was over before three submits were done when the workstation ran other test suites)."""
        script = ('echo "S $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\n'
                  'while [ ! -e "$GATES/$CARDQ_JOB" ]; do sleep 0.05; done\n'
                  'echo "E $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\n' % (NOW, NOW))
        return self.submit(cards, script, **kw)

    def release(self, *jobs):
        for job in jobs:
            (self.gates / job).touch()

    def wait(self, job, timeout=60):
        return self.cq("wait", job, "--timeout", str(timeout), timeout=timeout + 30)

    def file(self, job, name):
        path = self.q / "jobs" / job / name
        return path.read_text().strip() if path.exists() else ""

    def until(self, cond, seconds=20.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.05)
        return False

    def intervals(self):
        """{job: (cards, start, end)} from the events file."""
        out = {}
        for line in self.events.read_text().splitlines():
            kind, job, cards, stamp = line.split()
            cards = [int(c) for c in cards.split(",")]
            if kind == "S":
                out[job] = (cards, float(stamp), None)
            else:
                out[job] = (cards, out[job][1], float(stamp))
        return out

    def docker_calls(self):
        log = self.tmp / "docker.log"
        return log.read_text().splitlines() if log.exists() else []

    def close(self):
        jobs = self.q / "jobs"
        if not jobs.is_dir():
            return
        for d in jobs.iterdir():
            for name in ("pgid", "pid"):
                text = (d / name).read_text().strip() if (d / name).exists() else ""
                if text.isdigit() and int(text) > 1:
                    for target in (-int(text), int(text)):
                        try:
                            os.kill(target, signal.SIGKILL)
                        except OSError:
                            pass


@pytest.fixture
def farm(tmp_path):
    f = Farm(tmp_path)
    yield f
    f.close()


def _overlaps(intervals):
    """Pairs of jobs that held one card at the same time."""
    bad = []
    jobs = sorted(intervals.items())
    for i, (a, (ca, sa, ea)) in enumerate(jobs):
        for b, (cb, sb, eb) in jobs[i + 1:]:
            if set(ca) & set(cb) and sa < eb and sb < ea:
                bad.append((a, b))
    return bad


def test_a_job_runs_detached_with_its_cards_and_is_recorded(farm):
    job = farm.submit("0,1", 'echo "cards=$CARDQ_CARDS n=$CARDQ_NCARDS track=$CARDQ_TRACK name=$CARDQ_NAME '
                      'job=$CARDQ_JOB port=$CARDQ_PORT_BASE"\necho "it\'s \\"quoted\\" && fine"\n',
                      track="kernel-fixes", name="hc-op")
    assert job == "00001"
    proc = farm.wait(job)
    assert proc.returncode == 0, proc.stderr
    assert "job 00001 ended: exit code 0" in proc.stderr
    log = farm.cq("log", job).stdout
    assert "cards=0,1 n=2 track=kernel-fixes name=hc-op job=00001 port=8100" in log
    assert 'it\'s "quoted" && fine' in log
    assert "cardq: job 00001 (track kernel-fixes, name hc-op) on cards 0,1" in log
    assert "exit code 0" in log and log.rstrip().endswith("ok")
    ledger = farm.cq("ledger").stdout.splitlines()
    assert ledger[0].split("\t") == ["id", "track", "name", "wanted", "cards", "submitted", "start", "end",
                                     "seconds", "rc", "reason"]
    row = ledger[1].split("\t")
    assert row[:5] == ["00001", "kernel-fixes", "hc-op", "0,1", "0,1"] and row[9:] == ["0", "ok"]
    assert row[6].endswith("Z") and row[7].endswith("Z") and row[8].isdigit()
    # the supervisor is not in the caller's session
    assert farm.file(job, "state") == "ended"


def test_the_exit_code_is_the_jobs(farm):
    job = farm.submit("any:1", "exit 7")
    proc = farm.wait(job)
    assert proc.returncode == 7
    assert "exit code 7" in proc.stderr and "failed" in proc.stderr
    assert farm.cq("ledger").stdout.splitlines()[1].split("\t")[9:] == ["7", "failed"]


def test_the_command_can_follow_two_dashes_or_come_from_a_file(farm):
    job = farm.cq("submit", "--track", "t", "--name", "a", "--cards", "any:1", "--", "echo", "from-args").stdout.strip()
    assert farm.wait(job).returncode == 0
    assert "from-args" in farm.cq("log", job).stdout
    script = farm.tmp / "job.sh"
    script.write_text("echo from-file\n")
    job = farm.cq("submit", "--track", "t", "--name", "b", "--cards", "any:1", "--script", str(script)).stdout.strip()
    assert farm.wait(job).returncode == 0
    assert "from-file" in farm.cq("log", job).stdout


@pytest.mark.parametrize("args, text", [
    (["--name", "n", "--cards", "all"], "--track is required"),
    (["--track", "T!", "--name", "n", "--cards", "all"], "--track is required"),
    (["--track", "t", "--cards", "all"], "--name is required"),
    (["--track", "t", "--name", "n"], "--cards is required"),
    (["--track", "t", "--name", "n", "--cards", "8"], "--cards: '8'"),
    (["--track", "t", "--name", "n", "--cards", "0,0"], "--cards: '0,0'"),
    (["--track", "t", "--name", "n", "--cards", "any:9"], "--cards: 'any:9'"),
    (["--track", "t", "--name", "n", "--cards", "any:0"], "--cards: 'any:0'"),
    (["--track", "t", "--name", "n", "--cards", "some"], "--cards: 'some'"),
    (["--track", "t", "--name", "n", "--cards", "all", "--max-hold", "soon"], "--max-hold takes"),
    (["--track", "t", "--name", "n", "--cards", "all", "--max-hold", "4h"], "above the limit"),
])
def test_a_bad_submit_is_refused_and_queues_nothing(farm, args, text):
    proc = farm.cq("submit", *args, stdin="echo hi\n")
    assert proc.returncode == 1 and text in proc.stderr
    assert proc.stdout == ""
    assert not (farm.q / "jobs").exists() or not list((farm.q / "jobs").iterdir())


def test_an_empty_command_is_refused(farm):
    proc = farm.cq("submit", "--track", "t", "--name", "n", "--cards", "all", stdin="  \n")
    assert proc.returncode == 1 and "empty" in proc.stderr


def test_jobs_on_the_same_cards_run_one_after_the_other_and_others_side_by_side(farm):
    a = farm.block("0,1,2,3")
    b = farm.hold("3,2", 0.3)  # written descending: taken ascending
    c = farm.hold("4,5,6,7", 0.3)
    assert farm.wait(c).returncode == 0  # next to the job on 0-3, which still runs
    assert farm.file(b, "state") == "waiting"
    farm.release(a)
    for job in (a, b):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[b][0] == [2, 3]
    assert iv[b][1] >= iv[a][2], "the job on cards 2,3 started before the job on 0-3 ended"
    assert iv[c][1] < iv[a][2], "the job on cards 4-7 did not run next to the job on 0-3"
    assert _overlaps(iv) == []


def test_any_n_takes_fours_from_the_bottom_and_small_sets_from_the_top(farm):
    one = farm.block("any:1")
    assert farm.until(lambda: farm.file(one, "cards") != "")
    two = farm.block("any:2")
    assert farm.until(lambda: farm.file(two, "cards") != "")
    four = farm.block("any:4")
    assert farm.until(lambda: farm.file(four, "cards") != "")
    three = farm.hold("any:3", 0.2)
    assert farm.file(one, "cards") == "7"
    assert farm.file(two, "cards") == "4,5"  # the pair 6,7 is broken by the single job
    assert farm.file(four, "cards") == "0,1,2,3"
    assert farm.file(three, "cards") == ""  # one card is free: it waits
    farm.release(one, two, four)
    assert farm.until(lambda: farm.file(three, "cards") != "")
    assert farm.file(three, "cards") in ("0,1,2", "4,5,6")  # after the others ended: the lowest three free
    for job in (one, two, four, three):
        assert farm.wait(job).returncode == 0
    assert _overlaps(farm.intervals()) == []


def test_any_8_is_all(farm):
    job = farm.hold("any:8", 0.1)
    assert farm.wait(job).returncode == 0
    assert farm.file(job, "spec") == "all" and farm.file(job, "cards") == "0,1,2,3,4,5,6,7"


def test_many_jobs_never_share_a_card_and_all_end(farm):
    rng = random.Random(7)
    specs = ["all", "any:1", "any:1", "any:2", "any:4", "any:3", "0,1,2,3", "4,5,6,7", "1,2", "6,7", "3,4", "0", "7,0"]
    jobs = []
    for i in range(26):
        jobs.append(farm.hold(rng.choice(specs), "0.%d" % rng.randrange(1, 4), track="t%d" % (i % 5), hold="20s"))
    for job in jobs:
        proc = farm.wait(job, timeout=120)
        assert proc.returncode == 0, (job, proc.stderr)
    iv = farm.intervals()
    assert len(iv) == 26 and all(end is not None for _, _, end in iv.values())
    assert _overlaps(iv) == []
    for job in jobs:
        cards, spec = farm.file(job, "cards").split(","), farm.file(job, "spec")
        if spec == "all":
            assert len(cards) == 8
        elif spec.startswith("any:"):
            assert len(cards) == int(spec[4:])
        else:
            assert cards == spec.split(",")
    assert len(farm.cq("ledger").stdout.splitlines()) == 27


def test_an_all_card_job_is_not_starved_and_short_jobs_slip_in_front(farm):
    long_one = farm.block("0", hold="120s")  # its card is free by start + 120 s + 1 s grace at the latest
    assert farm.until(lambda: farm.file(long_one, "state") == "running")
    whole = farm.hold("all", 0.2, hold="50s")
    late = farm.hold("any:1", 0.2, hold="300s")  # would outlast the whole-node job's latest start: must wait
    quick = farm.hold("any:1", 0.2, hold="30s")  # ends before it: runs now, on a card the first job leaves free
    assert farm.wait(quick).returncode == 0
    assert farm.file(whole, "state") == "waiting" and farm.file(late, "state") == "waiting"
    farm.release(long_one)
    for job in (long_one, whole, late):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[quick][2] <= iv[whole][1], "the short job did not run ahead of the whole-node job"
    assert iv[quick][1] < iv[long_one][2], "the short job did not run next to the running job"
    assert iv[whole][1] >= iv[long_one][2]
    assert iv[late][1] >= iv[whole][2], "a later long job overtook the waiting whole-node job"
    assert _overlaps(iv) == []


def test_status_shows_holders_and_waiters(farm):
    a = farm.block("0,1,2,3", track="prefill", name="real-tp4")
    assert farm.until(lambda: farm.file(a, "state") == "running")
    b = farm.hold("all", 0.1, track="step-anatomy", name="profile-8card")
    out = farm.cq("status").stdout
    lines = out.splitlines()
    assert lines[0].startswith("cardq status at ")
    for card in "0123":
        assert any(ln.startswith("  %s  job %s  prefill/real-tp4  running" % (card, a)) for ln in lines), out
    for card in "4567":
        assert "  %s  free" % card in lines
    waiting = [ln for ln in lines if ln.startswith("  %s " % b)]
    assert waiting and "step-anatomy" in waiting[0] and "profile-8card" in waiting[0] and "wants all" in waiting[0]
    assert "starts by" in waiting[0]
    assert "(none)" not in out.split("ended (the last 8):")[0], "a waiting job is listed, and so is '(none)'"
    farm.cq("cancel", "--track", "prefill", a)
    assert farm.wait(b).returncode == 0
    out = farm.cq("status").stdout
    assert "cancelled" in out and "(none)" in out


def test_the_maximum_hold_stops_the_process_group_and_the_jobs_containers(farm):
    (farm.state / "labelled-00001").write_text("c0ffee\n")
    job = farm.submit("0", 'sleep 300 &\necho "child $!" >>"$EVENTS"\nsleep 300\n', hold="1s")
    proc = farm.wait(job, timeout=30)
    assert proc.returncode == 143, proc.stderr
    assert "timeout" in proc.stderr
    child = int(farm.events.read_text().split()[1])
    assert farm.until(lambda: _gone(child)), "the job's background child survived"
    assert "rm -f c0ffee" in farm.docker_calls()
    log = farm.cq("log", job).stdout
    assert "cardq: timeout at " in log and "SIGTERM to the job's process group" in log
    assert "removing the containers of job 00001" in log
    assert farm.cq("ledger").stdout.splitlines()[1].split("\t")[9:] == ["143", "timeout"]
    # one card only: no other container is touched
    assert not any(call == "ps -q" for call in farm.docker_calls())


def _gone(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def test_a_job_that_ignores_sigterm_is_killed_after_the_grace_time(farm):
    job = farm.submit("0", "trap '' TERM\nsleep 300\n", hold="1s")
    started = time.monotonic()
    proc = farm.wait(job, timeout=30)
    assert proc.returncode == 137, proc.stderr
    assert time.monotonic() - started < 15
    assert "did not end in 1 s: SIGKILL" in farm.cq("log", job).stdout
    nxt = farm.hold("0", 0.1)
    assert farm.wait(nxt).returncode == 0


def test_a_stopped_whole_node_job_loses_every_habana_container_and_a_finished_one_none(farm):
    (farm.state / "running").write_text("aaa\nbbb\n")
    (farm.state / "runtime-aaa").write_text("habana\n")
    (farm.state / "runtime-bbb").write_text("runc\n")
    ok = farm.hold("all", 0.1)
    assert farm.wait(ok).returncode == 0
    assert not any(call.startswith("rm -f") for call in farm.docker_calls())
    job = farm.submit("all", "sleep 300\n", hold="1s")
    assert farm.wait(job, timeout=30).returncode == 143
    calls = farm.docker_calls()
    assert "rm -f aaa" in calls and "rm -f bbb" not in calls
    assert "removing the container name-aaa" in farm.cq("log", job).stdout


def test_a_track_cancels_only_its_own_jobs(farm):
    job = farm.submit("0", "sleep 300\n", track="alpha", name="mine")
    assert farm.until(lambda: farm.file(job, "state") == "running")
    proc = farm.cq("cancel", "--track", "beta", job)
    assert proc.returncode == 4 and "belongs to track alpha" in proc.stderr
    assert proc.stderr.count("\n") == 1
    assert farm.cq("cancel", job).returncode == 1  # no track named
    time.sleep(0.5)
    assert farm.file(job, "state") == "running"
    proc = farm.cq("cancel", "--track", "alpha", job)
    assert proc.returncode == 0, proc.stderr
    assert farm.file(job, "state") == "ended" and farm.file(job, "reason") == "cancelled"
    assert farm.wait(job).returncode == 143
    assert farm.cq("ledger").stdout.splitlines()[1].split("\t")[9:] == ["143", "cancelled"]


def test_a_waiting_job_can_be_cancelled_and_never_runs(farm):
    a = farm.block("0")  # still running when the two submits and the cancel below are done
    assert farm.until(lambda: farm.file(a, "state") == "running")
    b = farm.hold("0", 0.1, name="second")
    c = farm.hold("0", 0.1, name="third")
    assert farm.cq("cancel", "--track", "alpha", b).returncode == 0
    assert farm.file(b, "state") == "cancelled"
    assert farm.wait(b).returncode == 125
    farm.release(a)
    assert farm.wait(a).returncode == 0 and farm.wait(c).returncode == 0
    assert b not in farm.intervals()
    row = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(b)][0]
    assert row[4] == "-" and row[8:] == ["0", "-", "cancelled"]


def test_a_second_job_of_the_same_name_is_refused_while_the_first_is_active(farm):
    a = farm.submit("0", "sleep 300\n", track="alpha", name="smoke")
    proc = farm.submit("1", "echo hi\n", track="alpha", name="smoke", check=False)
    assert proc.returncode == 3 and "already has the job 'smoke'" in proc.stderr and a in proc.stderr
    assert proc.stdout == ""
    other = farm.submit("1", "echo hi\n", track="beta", name="smoke")  # another track may use the name
    assert farm.wait(other).returncode == 0
    farm.cq("cancel", "--track", "alpha", a)
    again = farm.submit("1", "echo hi\n", track="alpha", name="smoke")
    assert farm.wait(again).returncode == 0


def test_a_killed_supervisor_still_ends_its_job_and_frees_the_cards(farm):
    job = farm.submit("0,1", 'echo "pid $$" >>"$EVENTS"\nsleep 300\n', hold="60s")
    assert farm.until(lambda: farm.events.exists() and farm.file(job, "pgid") != "")
    job_pid = int(farm.events.read_text().split()[1])
    os.kill(int(farm.file(job, "pid")), signal.SIGKILL)
    nxt = farm.hold("0", 0.1)
    assert farm.until(lambda: _gone(job_pid)), "the job outlived its supervisor"
    assert farm.until(lambda: farm.file(job, "state") == "lost")
    assert farm.wait(job).returncode == 126
    assert farm.wait(nxt).returncode == 0
    row = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(job)][0]
    assert row[4] == "0,1" and row[10] == "lost"


def test_a_dead_waiting_job_does_not_block_the_queue(farm):
    a = farm.block("0")
    assert farm.until(lambda: farm.file(a, "state") == "running")
    b = farm.hold("0", 0.1, name="dies-waiting")
    assert farm.until(lambda: farm.file(b, "pid") != "")
    os.kill(int(farm.file(b, "pid")), signal.SIGKILL)
    c = farm.hold("0", 0.1, name="after")
    farm.release(a)
    assert farm.wait(a).returncode == 0
    assert farm.wait(c).returncode == 0
    assert farm.file(b, "state") == "lost"
    assert farm.wait(b).returncode == 126


def test_wait_times_out_without_touching_the_job_and_log_follows(farm):
    job = farm.submit("0", "echo first\nsleep 1.5\necho last\n")
    proc = farm.cq("wait", job, "--timeout", "0")
    assert proc.returncode == 124 and "the job goes on" in proc.stderr
    follow = farm.cq("log", job, "-f", timeout=30)
    assert "first" in follow.stdout and "last" in follow.stdout and "exit code 0" in follow.stdout
    assert farm.wait(job).returncode == 0


def test_unknown_ids_and_commands(farm):
    assert farm.cq("wait", "42").returncode == 1
    assert farm.cq("log", "x").returncode == 1
    assert farm.cq("frobnicate").returncode == 2
    assert "cardq.sh submit" in farm.cq("--help").stdout
    assert farm.cq("ledger").stdout.strip() == "no job has ended yet"
    out = farm.cq("status").stdout
    assert out.count("free") == 8 and "(none)" in out


def test_root_is_refused(farm):
    if os.geteuid() != 0:
        pytest.skip("needs root to see the refusal")
    proc = farm.cq("status", CARDQ_ALLOW_ROOT="0")
    assert proc.returncode == 1 and "do not run cardq.sh as root" in proc.stderr


# ------------------------------------------------------------------ stage 2: module groups, fairness

NODE_MAP = "3,0,1,6,7,4,5,2"  # the 2026-11-20 node: the module id of card (index) 0..7


def test_any_4_and_any_2_take_whole_module_groups_on_a_scrambled_node(farm):
    farm.env["CARDQ_MODULE_MAP"] = NODE_MAP
    script = ('echo "S $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\necho "modules=$CARDQ_MODULES"\n'
              'while [ ! -e "$GATES/$CARDQ_JOB" ]; do sleep 0.05; done\n'
              'echo "E $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\n' % (NOW, NOW))
    first = farm.submit("any:4", script)
    assert farm.until(lambda: farm.file(first, "cards") != "")
    second = farm.submit("any:4", script)
    assert farm.until(lambda: farm.file(second, "cards") != "")
    assert farm.file(first, "cards") == "0,1,2,7"  # modules 3,0,1,2: the group 0-3
    assert farm.file(second, "cards") == "3,4,5,6"  # modules 6,7,4,5: the group 4-7
    farm.release(first, second)
    for job in (first, second):
        assert farm.wait(job).returncode == 0
    assert "modules=0,1,2,3" in farm.cq("log", first).stdout
    assert "modules=4,5,6,7" in farm.cq("log", second).stdout
    assert "on cards 0,1,2,7 (modules 0,1,2,3)" in farm.cq("log", first).stdout
    pair = farm.submit("any:2", script)
    assert farm.until(lambda: farm.file(pair, "cards") != "")
    one = farm.submit("any:1", script)
    assert farm.until(lambda: farm.file(one, "cards") != "")
    farm.release(pair, one)
    for job in (pair, one):
        assert farm.wait(job).returncode == 0
    assert farm.file(pair, "cards") == "3,4"  # modules 6,7, the top pair
    assert farm.file(one, "cards") == "6"  # module 5: the top module the pair left free... (7 and 6 are taken)
    assert _overlaps(farm.intervals()) == []


def test_single_cards_come_from_the_top_modules(farm):
    farm.env["CARDQ_MODULE_MAP"] = NODE_MAP
    jobs = []
    for _ in range(3):
        jobs.append(farm.block("any:1"))
        assert farm.until(lambda: farm.file(jobs[-1], "cards") != "")
    assert [farm.file(j, "cards") for j in jobs] == ["4", "3", "6"]  # modules 7, 6, 5
    four = farm.hold("any:4", 0.1)
    assert farm.until(lambda: farm.file(four, "cards") != "")
    assert farm.file(four, "cards") == "0,1,2,7"  # the whole group 0-3 is still free
    farm.release(*jobs)
    for job in jobs + [four]:
        assert farm.wait(job).returncode == 0


def test_the_module_map_comes_from_the_queue_folder_and_a_bad_one_is_refused(farm):
    farm.q.mkdir(parents=True, exist_ok=True)
    (farm.q / "module_map").write_text(NODE_MAP + "\n")
    job = farm.hold("any:4", 0.1)
    assert farm.wait(job).returncode == 0
    assert farm.file(job, "cards") == "0,1,2,7" and farm.file(job, "modules") == "0,1,2,3"
    out = farm.cq("status").stdout
    assert "card:module 0:3 1:0 2:1 3:6 4:7 5:4 6:5 7:2" in out
    proc = farm.cq("status", CARDQ_MODULE_MAP="1,1,2,3,4,5,6,7")
    assert "not the module id of each of the 8 cards" in proc.stderr
    assert "card:module 0:0 1:1" in proc.stdout


def test_an_explicit_list_that_is_no_module_group_is_run_with_a_note(farm):
    farm.env["CARDQ_MODULE_MAP"] = NODE_MAP
    proc = farm.submit("0,1,2,3", "echo hi\n", check=False)
    assert proc.returncode == 0 and "not one module group" in proc.stderr
    assert farm.wait(proc.stdout.strip()).returncode == 0
    proc = farm.submit("0,1,2,7", "echo hi\n", check=False)
    assert proc.returncode == 0 and "module group" not in proc.stderr
    assert farm.wait(proc.stdout.strip()).returncode == 0


def test_without_a_map_card_n_is_module_n(farm):
    job = farm.hold("any:4", 0.1)
    assert farm.wait(job).returncode == 0
    assert farm.file(job, "cards") == "0,1,2,3" and farm.file(job, "modules") == "0,1,2,3"


def test_a_long_all_card_job_waits_behind_shorter_ones_submitted_soon_after(farm):
    farm.env.update(CARDQ_LONG_HOLD_S="30", CARDQ_LONG_DELAY_S="600")
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    long_one = farm.hold("all", 0.3, hold="60s", name="long-all")  # more than 30 s of every card
    short_all = farm.hold("all", 0.3, hold="20s", name="short-all")
    short_one = farm.hold("any:1", 0.3, hold="20s", name="short-one")
    lines = farm.cq("status").stdout.split("waiting (in the order they start):")[1].splitlines()
    order = [ln.split()[0] for ln in lines if ln.startswith("  0")]
    assert order.index(short_all) < order.index(long_one), lines
    farm.release(busy)
    for job in (busy, long_one, short_all, short_one):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[short_all][1] < iv[long_one][1], "the long all-card job went first"
    assert iv[short_one][1] < iv[long_one][1]
    assert _overlaps(iv) == []


def test_an_all_card_job_up_to_the_limit_keeps_its_place(farm):
    farm.env.update(CARDQ_LONG_HOLD_S="30", CARDQ_LONG_DELAY_S="600")
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    first = farm.hold("all", 0.3, hold="30s", name="at-the-limit")
    second = farm.hold("all", 0.3, hold="20s", name="later")
    farm.release(busy)
    for job in (busy, first, second):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[first][1] < iv[second][1]


def test_the_coordinator_bumps_a_waiting_job_to_the_front(farm):
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    a = farm.hold("all", 0.2, name="first-all")
    b = farm.hold("all", 0.2, name="second-all")
    c = farm.hold("all", 0.2, name="bumped-all")
    proc = farm.cq("bump", "--by", "prefill", c)
    assert proc.returncode == 4 and "only the coordinator" in proc.stderr
    proc = farm.cq("bump", "--by", "coord", c, "unlocks", "partial", "cards")
    assert proc.returncode == 0, proc.stderr
    assert "bumped to the front by coord (unlocks partial cards)" in proc.stderr
    assert "[bumped: coord: unlocks partial cards]" in farm.cq("status").stdout
    proc = farm.cq("bump", "--by", "coord", busy)
    assert proc.returncode == 1 and "not waiting" in proc.stderr
    farm.release(busy)
    for job in (busy, a, b, c):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[c][1] < iv[a][1] < iv[b][1], iv
    rows = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(c)]
    assert rows[0][10] == "bump by coord: unlocks partial cards" and rows[0][9] == "-"
    assert rows[1][10] == "ok"
    assert "bumped to the front of the queue" in farm.cq("log", c).stdout


def _waiting_order(farm):
    waiting = farm.cq("status").stdout.split("waiting (in the order they start):")[1].split("ended (the last 8):")[0]
    return [ln.split()[0] for ln in waiting.splitlines() if ln.startswith("  0")]


def test_the_coordinator_places_a_waiting_job_before_or_after_another(farm):
    # A bump moves a job to the very front. Packing a wave (nine 1-card jobs ahead of the next all-card job)
    # needs a place next to a given job; the coordinator did it by editing order keys by hand.
    busy = farm.block("all")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    a = farm.hold("all", 0.2, name="a")
    b = farm.hold("all", 0.2, name="b")
    c = farm.hold("all", 0.2, name="c")
    d = farm.hold("all", 0.2, name="d")
    assert _waiting_order(farm) == [a, b, c, d]
    proc = farm.cq("bump", "--by", "coord", d, "--before", b, "one", "wave")
    assert proc.returncode == 0, proc.stderr
    assert "job %s: placed before job %s by coord (one wave)" % (d, b) in proc.stderr
    assert _waiting_order(farm) == [a, d, b, c]
    proc = farm.cq("bump", "--by", "coord", a, "--after", c)
    assert proc.returncode == 0, proc.stderr
    assert _waiting_order(farm) == [d, b, c, a]
    line = [ln for ln in farm.cq("status").stdout.splitlines() if ln.startswith("  %s " % d)][0]
    assert "[bumped: coord: one wave (before %s)]" % b in line
    # a job placed next to a placed job lands next to it
    e = farm.hold("all", 0.2, name="e")
    assert farm.cq("bump", "--by", "coord", e, "--before", d).returncode == 0
    assert _waiting_order(farm) == [e, d, b, c, a]
    for args, rc, text in (
        (["--by", "coord", a, "--before", busy], 1, "is running, not waiting"),
        (["--by", "coord", a, "--before", a], 1, "itself"),
        (["--by", "prefill", a, "--before", b], 4, "only the coordinator"),
        (["--by", "coord", a, "--before"], 1, "needs a job id"),
        (["--by", "coord", a, "--before", "777"], 1, "no job 00777"),
    ):
        proc = farm.cq("bump", *args)
        assert proc.returncode == rc and text in proc.stderr, (args, proc.returncode, proc.stderr)
    farm.release(busy)
    for job in (busy, a, b, c, d, e):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[e][1] < iv[d][1] < iv[b][1] < iv[c][1] < iv[a][1], iv
    rows = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(d)]
    assert rows[0][10] == "bump by coord: one wave (before %s)" % b
    assert "placed before job %s" % b in farm.cq("log", d).stdout


def test_a_replacement_inherits_the_bump_of_the_job_it_replaces(farm):
    # Four tracks lost a bump on 2026-11-20 when they replaced the bumped job by a smaller or corrected one.
    busy = farm.block("all")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    first = farm.hold("all", 0.2, track="beta", name="first")
    old = farm.hold("all", 0.2, track="alpha", name="old-all")
    assert farm.cq("bump", "--by", "coord", old, "decisive").returncode == 0
    proc = _replace(farm, "any:4", old)
    assert proc.returncode == 0, proc.stderr
    new = proc.stdout.strip()
    assert farm.until(lambda: farm.file(old, "state") == "cancelled")
    assert "inherits the bump of job %s" % old in farm.cq("log", new).stdout
    line = [ln for ln in farm.cq("status").stdout.splitlines() if ln.startswith("  %s " % new)][0]
    assert "[bumped: coord: decisive]" in line
    assert _waiting_order(farm) == [new, first]  # with only the old job's place it would stand behind `first`
    # the same through the coordinator's place, for a job the track cancelled itself and submitted again
    again = farm.hold("all", 0.2, track="beta", name="again")
    assert farm.cq("bump", "--by", "coord", again, "--before", new, "first of the wave").returncode == 0
    assert farm.cq("cancel", "--track", "beta", again).returncode == 0
    conv = farm.hold("any:4", 0.2, track="beta", name="converted")
    assert farm.cq("place", "--by", "coord", conv, again).returncode == 0
    assert _waiting_order(farm) == [conv, new, first]
    farm.release(busy)
    for job in (busy, first, new, conv):
        assert farm.wait(job).returncode == 0


def test_adopt_gives_a_waiting_job_a_new_supervisor_and_keeps_it(farm):
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    w = farm.hold("0", 0.2, name="waits")
    assert farm.until(lambda: farm.file(w, "pid") != "")
    old = int(farm.file(w, "pid"))
    running_pid = farm.file(busy, "pid")
    waiter = subprocess.Popen(["bash", str(CARDQ), "wait", w, "--timeout", "60"], env=farm.env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc = farm.cq("adopt")
    assert proc.returncode == 0 and "adopt: 1 waiting jobs" in proc.stderr, proc.stderr
    new = int(farm.file(w, "pid"))
    assert new != old and farm.until(lambda: _gone(old))
    assert farm.file(busy, "pid") == running_pid, "a running job's supervisor was touched"
    farm.release(busy)
    out, err = waiter.communicate(timeout=90)
    assert waiter.returncode == 0, err
    assert farm.file(w, "state") == "ended" and farm.wait(busy).returncode == 0
    assert "replaced by %d" % new in farm.cq("log", w).stdout


def test_a_long_job_queued_before_the_fairness_rule_keeps_its_place(farm):
    farm.env.update(CARDQ_LONG_HOLD_S="30", CARDQ_LONG_DELAY_S="600")
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    old_long = farm.hold("all", 0.2, hold="60s", name="queued-before-the-rule")
    (farm.q / "jobs" / old_long / "rules").unlink()
    short = farm.hold("all", 0.2, hold="20s", name="short-after")
    farm.release(busy)
    for job in (busy, old_long, short):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[old_long][1] < iv[short][1]


def _replace(farm, cards, old, seconds=0.2, track="alpha", name=None):
    script = ('echo "S $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\nsleep %s\n'
              'echo "E $CARDQ_JOB $CARDQ_CARDS $(%s)" >>"$EVENTS"\n' % (NOW, seconds, NOW))
    return farm.cq("submit", "--track", track, "--name", name or "new-%s" % cards.replace(":", ""), "--cards", cards,
                   "--replaces", old, stdin=script)


def test_a_job_on_fewer_cards_takes_the_place_of_the_job_it_replaces(farm):
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    old = farm.hold("all", 0.2, track="alpha", name="old-all")
    other = farm.hold("all", 0.2, track="beta", name="other-all")
    proc = _replace(farm, "any:4", old)
    assert proc.returncode == 0, proc.stderr
    new = proc.stdout.strip()
    assert "replaces job %s and takes its place" % old in proc.stderr
    assert farm.until(lambda: farm.file(old, "state") == "cancelled")
    farm.release(busy)
    for job in (busy, other, new):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[new][1] < iv[other][1], "the replacement lost the old job's place"
    assert old not in iv
    assert "replaced by job %s" % new in farm.cq("log", old).stdout
    assert "takes the place of job %s" % old in farm.cq("log", new).stdout


def test_replacements_hold_at_most_the_cards_of_the_old_job(farm):
    busy = farm.block("all")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    old = farm.hold("any:4", 0.1, name="old-four")
    first = _replace(farm, "any:2", old, name="half-a")
    assert first.returncode == 0, first.stderr
    assert farm.until(lambda: farm.file(old, "state") == "cancelled")
    second = _replace(farm, "any:2", old, name="half-b")  # the old job is cancelled now: still its place
    assert second.returncode == 0, second.stderr
    third = _replace(farm, "any:1", old, name="one-more")
    assert third.returncode == 1 and "do not fit" in third.stderr
    farm.release(busy)
    for job in (busy, first.stdout.strip(), second.stdout.strip()):
        assert farm.wait(job).returncode == 0


def test_only_the_tracks_own_waiting_or_recently_cancelled_job_can_be_replaced(farm):
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    theirs = farm.hold("0", 0.1, track="beta", name="theirs")
    proc = _replace(farm, "any:1", theirs, track="alpha")
    assert proc.returncode == 4 and "belongs to track beta" in proc.stderr
    proc = _replace(farm, "any:1", busy, track="alpha")
    assert proc.returncode == 1 and "is running" in proc.stderr
    farm.release(busy)
    assert farm.wait(busy).returncode == 0 and farm.wait(theirs).returncode == 0
    proc = _replace(farm, "any:1", busy, track="alpha")
    assert proc.returncode == 1 and "is ended" in proc.stderr
    proc = _replace(farm, "any:1", "777", track="alpha")
    assert proc.returncode == 1 and "no job 00777" in proc.stderr


FAKE_DMESG = r"""#!/bin/bash
cat <<'LINES'
[  999.000001] habanalabs 0000:0b:00.0: Received unexpected user error interrupt
[ 1001.500000] habanalabs 0000:0b:00.0: Received unexpected user error interrupt
[ 1002.000000] habanalabs 0000:0b:00.0: user process got signal while waiting for interrupt ID 52
[ 1003.000000] habanalabs 0000:0c:00.0 enp12s0d8: renamed from eth3
[ 1004.000000] habanalabs 0000:0e:00.0: Received unexpected user error interrupt
LINES
if [ -n "${FAKE_DMESG_MORE:-}" ]; then printf '%s\n' "$FAKE_DMESG_MORE"; fi
"""


def _fake_dmesg(farm, uptime):
    bin_dir = Path(farm.env["PATH"].split(":")[0])
    (bin_dir / "dmesg").write_text(FAKE_DMESG)
    (bin_dir / "dmesg").chmod(0o755)
    up = farm.tmp / "uptime"
    up.write_text("%s 5.0\n" % uptime)
    farm.env.update(CARDQ_UPTIME_FILE=str(up))


def test_the_kernel_lines_of_the_cards_during_a_job_go_to_its_log_and_the_ledger(farm):
    _fake_dmesg(farm, "1000.25")
    farm.env.update(CARDQ_MODULE_MAP=NODE_MAP)
    farm.q.mkdir(parents=True, exist_ok=True)
    (farm.q / "pci_map").write_text("".join("%d 0000:%02x:00.0\n" % (i, 0x0a + i) for i in range(8)))
    job = farm.hold("any:4", 0.1)  # cards 0,1,2,7 = modules 0-3; card 1 = 0000:0b:00.0 = module 0
    assert farm.wait(job).returncode == 0
    assert farm.file(job, "cards") == "0,1,2,7"
    log = farm.cq("log", job).stdout
    assert ("card 1 (module 0, 0000:0b:00.0): 2 (one of this job's cards): 1 failure analysis, "
            "1 killed while waiting; first: [ 1001.500000]") in log
    # another card's line: in the log, not in this job's reason
    assert "card 4 (module 7, 0000:0e:00.0): 1 (not a card of this job): 1 failure analysis; first:" in log
    assert "0000:0c:00.0" not in log  # routine lines are left out
    row = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(job)][0]
    assert row[10] == ("ok; habanalabs: card 1 (module 0, 0000:0b:00.0) 2 lines (1 failure analysis, "
                       "1 killed while waiting)")
    assert ": exit code 0, " in log and log.rstrip().endswith("1 killed while waiting)")
    health = [ln.split("\t") for ln in (farm.q / "health.tsv").read_text().splitlines()]
    assert health[0][4:] == ["1", "0", "0000:0b:00.0", "2", "own", "1", "1", "0", ""]
    assert health[1][4:] == ["4", "7", "0000:0e:00.0", "1", "other", "0", "1", "0", ""]


def test_a_kernel_line_of_no_known_class_is_an_error_in_the_reason(farm):
    # Two classes are consequences of what the job did (a process killed while it waited for the card; Synapse
    # starting its failure analysis). Any other habanalabs line is a card error and says so.
    _fake_dmesg(farm, "1003.5")
    farm.env.update(CARDQ_MODULE_MAP=NODE_MAP,
                    FAKE_DMESG_MORE="[ 1006.000000] habanalabs 0000:0c:00.0: Device heartbeat failed!\n"
                                    "[ 1007.000000] habanalabs 0000:0c:00.0: Going to reset device\n"
                                    "[ 1008.000000] habanalabs 0000:0c:00.0: user process got signal while "
                                    "waiting for interrupt ID 52")
    farm.q.mkdir(parents=True, exist_ok=True)
    (farm.q / "pci_map").write_text("".join("%d 0000:%02x:00.0\n" % (i, 0x0a + i) for i in range(8)))
    job = farm.hold("any:4", 0.1)  # cards 0,1,2,7; card 2 = 0000:0c:00.0 = module 1
    assert farm.wait(job).returncode == 0
    row = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(job)][0]
    assert row[10] == ("ok; habanalabs ERROR: card 2 (module 1, 0000:0c:00.0) 3 lines (2 ERROR: Device heartbeat "
                       "failed!, 1 killed while waiting)")
    health = [ln.split("\t") for ln in (farm.q / "health.tsv").read_text().splitlines()]
    mine = [h for h in health if h[6] == "0000:0c:00.0"][0]
    assert mine[7:] == ["3", "own", "1", "0", "2", "Device heartbeat failed!"]
    assert "3 (one of this job's cards): 2 ERROR, 1 killed while waiting; first:" in farm.cq("log", job).stdout


def test_health_lists_the_kernel_lines_since_boot_per_card_and_class(farm):
    _fake_dmesg(farm, "1000.25")
    farm.env.update(CARDQ_MODULE_MAP=NODE_MAP)
    farm.q.mkdir(parents=True, exist_ok=True)
    (farm.q / "pci_map").write_text("".join("%d 0000:%02x:00.0\n" % (i, 0x0a + i) for i in range(8)))
    job = farm.hold("any:4", 0.1)
    assert farm.wait(job).returncode == 0
    out = farm.cq("health").stdout
    assert "card 1 (module 0, 0000:0b:00.0): 3 lines: 2 failure analysis, 1 killed while waiting" in out
    assert "card 4 (module 7, 0000:0e:00.0): 1 lines: 1 failure analysis" in out
    assert "no line of an unknown class: no card error" in out
    assert "%s\talpha" % job in out  # the jobs that had lines, from health.tsv
    out = farm.cq("health", FAKE_DMESG_MORE="[ 1006.000000] habanalabs 0000:0c:00.0: Device heartbeat failed!").stdout
    assert "card 2 (module 1, 0000:0c:00.0): 1 lines: 1 ERROR (first: Device heartbeat failed!)" in out
    assert "CARD ERROR" in out


def test_without_kernel_lines_since_the_start_the_reason_stays_ok(farm):
    _fake_dmesg(farm, "5000.0")
    job = farm.hold("any:1", 0.1)
    assert farm.wait(job).returncode == 0
    row = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(job)][0]
    assert row[10] == "ok"


def test_the_coordinator_gives_a_converted_job_the_place_of_the_cancelled_one(farm):
    busy = farm.block("all")  # nothing starts while the places are given
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    old = farm.hold("all", 0.2, track="alpha", name="old-all")
    other = farm.hold("all", 0.2, track="beta", name="other-all")
    assert farm.cq("cancel", "--track", "alpha", old).returncode == 0
    new = farm.hold("any:4", 0.2, track="alpha", name="converted")
    assert farm.cq("place", "--by", "alpha", new, old).returncode == 4
    assert farm.cq("place", "--by", "coord", other, old).returncode == 4  # another track's job
    proc = farm.cq("place", "--by", "coord", new, old)
    assert proc.returncode == 0, proc.stderr
    farm.release(busy)
    for job in (busy, other, new):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[new][1] < iv[other][1]
    rows = [ln.split("\t") for ln in farm.cq("ledger").stdout.splitlines() if ln.startswith(new)]
    assert rows[0][10] == "place of %s by coord" % old


def test_place_never_cancels_a_waiting_job(farm):
    busy = farm.block("0")
    assert farm.until(lambda: farm.file(busy, "state") == "running")
    old = farm.hold("0", 0.1, name="still-waiting")
    new = farm.hold("0", 0.1, name="newer")
    proc = farm.cq("place", "--by", "coord", new, old)
    assert proc.returncode == 1 and "only gives the place of a job its track cancelled" in proc.stderr
    assert farm.file(old, "state") == "waiting"
    farm.release(busy)
    for job in (busy, old, new):
        assert farm.wait(job).returncode == 0
