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
    # 3 s: the two submits that follow must be done before the first job ends, also on a slow host
    a = farm.hold("0,1,2,3", 3.0)
    b = farm.hold("3,2", 0.3)  # written descending: taken ascending
    c = farm.hold("4,5,6,7", 1.0)
    for job in (a, b, c):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[b][0] == [2, 3]
    assert iv[b][1] >= iv[a][2], "the job on cards 2,3 started before the job on 0-3 ended"
    assert iv[c][1] < iv[a][2], "the job on cards 4-7 did not run next to the job on 0-3"
    assert _overlaps(iv) == []


def test_any_n_takes_fours_from_the_bottom_and_small_sets_from_the_top(farm):
    one = farm.hold("any:1", 1.5)
    assert farm.until(lambda: farm.file(one, "cards") != "")
    two = farm.hold("any:2", 1.5)
    assert farm.until(lambda: farm.file(two, "cards") != "")
    four = farm.hold("any:4", 1.5)
    assert farm.until(lambda: farm.file(four, "cards") != "")
    three = farm.hold("any:3", 0.2)
    assert farm.until(lambda: farm.file(three, "cards") != "")
    assert farm.file(one, "cards") == "7"
    assert farm.file(two, "cards") == "4,5"  # the pair 6,7 is broken by the single job
    assert farm.file(four, "cards") == "0,1,2,3"
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
    long_one = farm.hold("0", 6.0, hold="12s")  # its card is free by start + 12 s + 1 s grace at the latest
    assert farm.until(lambda: farm.file(long_one, "state") == "running")
    whole = farm.hold("all", 0.2, hold="5s")
    time.sleep(0.4)
    assert farm.file(whole, "state") == "waiting"
    late = farm.hold("any:1", 0.2, hold="30s")  # would outlast the whole-node job's latest start: must wait
    quick = farm.hold("any:1", 0.2, hold="3s")  # ends before it: runs now, on a card the first job leaves free
    for job in (long_one, whole, late, quick):
        assert farm.wait(job).returncode == 0
    iv = farm.intervals()
    assert iv[quick][2] <= iv[whole][1], "the short job did not run ahead of the whole-node job"
    assert iv[quick][1] < iv[long_one][2], "the short job did not run next to the running job"
    assert iv[whole][1] >= iv[long_one][2]
    assert iv[late][1] >= iv[whole][2], "a later long job overtook the waiting whole-node job"
    assert _overlaps(iv) == []


def test_status_shows_holders_and_waiters(farm):
    a = farm.hold("0,1,2,3", 3, track="prefill", name="real-tp4")
    assert farm.until(lambda: farm.file(a, "state") == "running")
    b = farm.hold("all", 0.1, track="step-anatomy", name="profile-8card")
    time.sleep(0.3)
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
    a = farm.hold("0", 4.0)  # still running when the two submits and the cancel below are done
    assert farm.until(lambda: farm.file(a, "state") == "running")
    b = farm.hold("0", 0.1, name="second")
    c = farm.hold("0", 0.1, name="third")
    assert farm.cq("cancel", "--track", "alpha", b).returncode == 0
    assert farm.file(b, "state") == "cancelled"
    assert farm.wait(b).returncode == 125
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
    a = farm.hold("0", 1.0)
    assert farm.until(lambda: farm.file(a, "state") == "running")
    b = farm.hold("0", 0.1, name="dies-waiting")
    assert farm.until(lambda: farm.file(b, "pid") != "")
    os.kill(int(farm.file(b, "pid")), signal.SIGKILL)
    c = farm.hold("0", 0.1, name="after")
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
