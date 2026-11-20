# SPDX-License-Identifier: Apache-2.0
"""CPU self-tests of the step-anatomy card microbenchmarks (tools/flash_next/anatomy).

The scripts run on a Gaudi card; ``--selftest`` runs the same code paths on CPU tensors (gloo ranks for the
collectives), so these tests check the scripts, not the numbers.
"""

import json
import subprocess
import sys
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
