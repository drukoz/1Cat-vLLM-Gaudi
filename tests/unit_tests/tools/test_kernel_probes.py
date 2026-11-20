# SPDX-License-Identifier: Apache-2.0
"""The card probes of ``tools/flash_next/kernel_probes`` run their own logic on a CPU before they get a card.

* ``probe_moe_route.py`` (the open ``route_real_r16`` card failure of the MoE glue): its self-test injects every
  fault the card could have (a strided view read as contiguous, an 8-byte read of 4-byte ids, a wrong softmax
  or top-k, a kernel that moves in a shared graph, ...) and the verdict must name each one.
* ``probe_moe_block.py`` (one Flash-Next FP8 MoE block, torch path against the glue and the glue with the merged
  gate_up GEMM): with the tests' CPU stand-ins every variant returns the torch path's bits.
* ``probe_moe_expert_bytes.py`` (idea G2-1: does the vendor MoE op read every local expert): every routing arm
  runs and the verdict line is printed (times on a CPU mean nothing).
* ``probe_view_inputs.py`` (can a custom op take a view at accumulation mode 0): its own self-test.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBES = ROOT / "tools" / "flash_next" / "kernel_probes"


def _run(args, tmp_path, **env):
    full = dict(os.environ, PYTHONPATH=str(ROOT), VLLM_PLUGINS="", PROBE_DEVICE="cpu", **env)
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=tmp_path, env=full, capture_output=True, text=True,
                          timeout=1800)
    return proc, proc.stdout + proc.stderr


def test_route_probe_names_every_injected_fault(tmp_path):
    proc, out = _run([PROBES / "selftest_probe_moe_route.py"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    assert "12 of 12 faults are named by the verdict" in out


def test_block_probe_variants_are_the_torch_path_on_cpu(tmp_path):
    proc, out = _run([PROBES / "probe_moe_block.py", "--out", tmp_path / "block", "--iters", "1", "--blocks", "1"],
                     tmp_path)
    assert proc.returncode == 0, out[-3000:]
    assert "merged gate_up (moe_silu_mul first_column): True" in out
    parity = [line for line in out.splitlines() if line.startswith("PROBE moe-block parity")]
    assert len(parity) == 6, parity  # glue and merged at 1, 32 and 128 rows
    assert all(" max abs diff 0.000e+00" in line for line in parity), parity
    assert "VERDICT moe-block:" in out


def test_expert_bytes_probe_runs_every_arm(tmp_path):
    proc, out = _run([PROBES / "probe_moe_expert_bytes.py", "--out", tmp_path / "eb", "--iters", "1", "--calls", "1",
                      "--calls-large", "1"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    arms = [line for line in out.splitlines() if line.startswith("PROBE expert-bytes rows")]
    assert len(arms) == 15, arms  # 3 arms at 1 row, 4 at 4, 32 and 128 rows
    assert "routing one  ( 1 local experts hit)" in out and "(64 local experts hit)" in out
    assert "VERDICT expert-bytes:" in out


def test_view_probe_selftest(tmp_path):
    proc, out = _run([PROBES / "probe_view_inputs.py", "--selftest"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    assert "VERDICT: VIEWS WORK" in out
