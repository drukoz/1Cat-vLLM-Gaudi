# SPDX-License-Identifier: Apache-2.0
"""CPU self-tests of the step-anatomy card microbenchmarks (tools/flash_next/anatomy).

The scripts run on a Gaudi card; ``--selftest`` runs the same code paths on CPU tensors (gloo ranks for the
collectives), so these tests check the scripts, not the numbers.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ANATOMY = ROOT / "tools" / "flash_next" / "anatomy"


def _run(script, *args, timeout=240):
    return subprocess.run([sys.executable, str(ANATOMY / script), *args],
                          capture_output=True,
                          text=True,
                          timeout=timeout,
                          cwd=str(ROOT))


def test_bench_lazy_ops_selftest(tmp_path):
    out = tmp_path / "lazy_ops.json"
    proc = _run("bench_lazy_ops.py", "--selftest", "--quick", "--acc-modes", "0", "--out", str(out))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "SELFTEST ONLY" in proc.stdout
    data = json.loads(out.read_text())
    assert data["tool"] == "bench_lazy_ops" and data["selftest"] is True
    summary = data["summaries"][0]
    assert summary["complete"] and summary["error"] is None
    assert set(summary["fits"]) == {"chain", "mix"}
    cases = {record["case"] for record in data["runs"][0]["records"]}
    assert {"env", "graph", "cuts", "mark_empty", "h2d", "d2h", "head", "done"} <= cases


def test_bench_collectives_selftest(tmp_path):
    out = tmp_path / "collectives.json"
    proc = _run("bench_collectives.py", "--selftest", "--out", str(out))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["selftest"] is True
    assert data["derived"], "no derived per-collective numbers"


REAPER_SCRIPT = r'''
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
import {module} as bench
bench.install_reaper()
child = bench.spawn(["sleep", "60"])
print(child.pid, flush=True)
time.sleep(60)
'''


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_parents_kill_their_children_on_sigterm():
    """A parent stopped by SIGTERM (``timeout``, the queue's maximum hold) takes its children with it."""
    for module in ("bench_lazy_ops", "bench_collectives", "bench_mme_rows"):
        proc = subprocess.Popen([sys.executable, "-c", REAPER_SCRIPT.replace("{module}", module), str(ANATOMY)],
                                stdout=subprocess.PIPE, text=True)
        child_pid = int(proc.stdout.readline())
        assert _pid_alive(child_pid)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 128 + signal.SIGTERM
        deadline = time.time() + 10
        while _pid_alive(child_pid) and time.time() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(child_pid), module


def test_bench_lazy_ops_budget_skips_graphs_but_keeps_the_cheap_cases(tmp_path):
    """A child out of budget skips the graph cases (reported, the fits then fail the run) and still measures the
    cuts, mark_step, copies and the LM head, which run first."""
    out = tmp_path / "lazy_ops.json"
    proc = _run("bench_lazy_ops.py", "--selftest", "--quick", "--acc-modes", "0", "--child-budget", "0",
                "--out", str(out))
    assert proc.returncode == 2, proc.stdout[-2000:]
    records = json.loads(out.read_text())["runs"][0]["records"]
    graphs = [r for r in records if r["case"] == "graph"]
    assert graphs and all(r.get("skipped") == "budget" for r in graphs)
    assert {"cuts", "mark_empty", "h2d", "d2h", "head", "done"} <= {r["case"] for r in records}
    assert "SKIPPED (predicted compile" in proc.stdout


def test_bench_lazy_ops_recipe_cache_replaces_the_uncached_mode_0_run(tmp_path):
    out = tmp_path / "lazy_ops.json"
    proc = _run("bench_lazy_ops.py", "--selftest", "--quick", "--acc-modes", "0,1", "--recipe-cache",
                str(tmp_path / "cache"), "--out", str(out))
    assert proc.returncode == 0, proc.stdout[-2000:]
    labels = [run["label"] for run in json.loads(out.read_text())["runs"]]
    assert labels == ["acc0-cache-cold", "acc0-cache-warm", "acc1"]
