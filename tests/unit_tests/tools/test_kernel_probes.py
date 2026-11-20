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
* ``probe_index_narrowing.py`` (ids narrowed to int32 in the graph that makes them: the sparse-QSA selection,
  argmax, replacements): on a CPU every case is RIGHT.
* ``probe_hc_copies.py`` (the owned copies in front of the fused hyper-connection ops): every copy mode returns
  the bits of mode 0.
* ``decode_ab_client.py --compare``: the 1-stream rounds are the test of two arms' texts.
"""

import json
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
    assert "IDENTITY VIEWS: 5 of 5 cases give the op the right rows" in out
    assert "OP OUTPUTS: 4 of 4 cases need no copy" in out


def test_index_narrowing_probe_is_right_on_cpu(tmp_path):
    proc, out = _run([PROBES / "probe_index_narrowing.py", "--out", tmp_path / "narrow"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    states = ("RIGHT", "WRONG", "ERROR")
    cases = [line for line in out.splitlines()
             if line.startswith("PROBE narrow") and any(f": {state} (" in line for state in states)]
    assert len(cases) == 32, cases  # 8 served, 12 pattern, 12 fix
    assert all(": RIGHT (" in line for line in cases), [line for line in cases if ": RIGHT (" not in line]
    assert "VERDICT select_blocks: select_blocks on the card: RIGHT in every case" in out
    result = json.loads((tmp_path / "narrow" / "index_narrowing.json").read_text())
    assert len(result["results"]) == 32 and len(result["verdicts"]) == 3


def test_hc_copies_probe_modes_are_bit_identical_on_cpu(tmp_path):
    proc, out = _run([PROBES / "probe_hc_copies.py", "--out", tmp_path / "hc", "--calls", "2", "--iters", "1"],
                     tmp_path)
    assert proc.returncode == 0, out[-3000:]
    rows = [line for line in out.splitlines() if line.startswith("PROBE hc-copies rows")]
    assert len(rows) == 9, rows  # copy modes 0, 1, 2 at 1, 32 and 128 rows
    assert all("bit-identical to mode 0: True" in line for line in rows), rows
    assert "VERDICT hc-copies:" in out and "RAISED" not in out


def test_decode_ab_client_compares_the_one_stream_rounds(tmp_path):
    a = [{"round": "meas N=1 rep=0", "tokens": 4, "ctx_tokens": 0, "texts": ["one two three four"]},
         {"round": "meas N=2 rep=0", "tokens": 4, "ctx_tokens": 0, "texts": ["a b c d", "e f g h"]}]
    b = [dict(a[0]), dict(a[1], texts=["a b X d", "e f g h"])]
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    client = PROBES / "decode_ab_client.py"
    proc, out = _run([client, "--compare", tmp_path / "a.json", tmp_path / "b.json"], tmp_path)
    assert proc.returncode == 0, out
    assert "COMPARE meas N=1 rep=0         1 of 1 streams returned the same text" in out
    assert "COMPARE meas N=2 rep=0         1 of 2 streams returned the same text; shortest common prefix 4" in out
    assert "1-stream rounds (the test): the same text in 1 of 1" in out
    source = client.read_text()
    assert '"min_tokens"] =' not in source and "min_tokens\": ntok" not in source  # no request carries it
