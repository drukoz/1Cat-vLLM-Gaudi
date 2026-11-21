# SPDX-License-Identifier: Apache-2.0
"""The card test of the MoE glue ops (``test_moe_hpu_op.py``) run on CPU as a dry run.

On a machine without a card the card test skips as a whole, so a mistake in it would first show during
a rental.  With ``MOE_GLUE_OP_TEST_DEVICE=cpu`` the same file runs here: the ops are their CPU stand-in
(the oracle, with the kernel's own rule for equal logits) and the vendor ops are the CPU stand-ins of
``tests/unit_tests/models/flash_next/moe_block.py``.  Every test of it must pass and none may skip: the
rental gate counts a skip as a failure.  This proves that the file runs and that its checks hold for a
correct kernel; whether the kernels are correct on a card is what the card run shows.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CARD_TEST = "tests/unit_tests/hc_fused/test_moe_hpu_op.py"


def _pytest(env_overrides: dict) -> subprocess.CompletedProcess:
    """The card test as the rental gate runs it (tools/flash_next/rental/gates.py): ``pytest -q -rs``, no ``-s``."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), VLLM_PLUGINS="", **env_overrides)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", CARD_TEST],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800, env=env)


def test_the_card_test_passes_as_a_dry_run_on_cpu():
    proc = _pytest({"MOE_GLUE_OP_TEST_DEVICE": "cpu"})
    text = proc.stdout + proc.stderr
    assert proc.returncode == 0, text[-4000:]
    summary = re.search(r"(\d+) passed", text)
    assert summary and int(summary.group(1)) >= 80, text[-2000:]
    assert " skipped" not in text.splitlines()[-1] and " failed" not in text.splitlines()[-1]
    # The measurements an operator reads after a card run are in the log although every test passed and pytest
    # captured their output: the card test prints them once more after its last test.
    for needle in ("MOEHPU vendor op fed by moe_route", "MOEHPU vendor op with the routing slots reversed",
                   "MOEHPU all-equal logits", "MOEHPU row_quant_real_stress against dynamic_quant",
                   "MOEHPU chain of one block at 32 rows", "MOECASE route_prefill_r2048",
                   "MOEHPU route_real_r16 ids: moe_route returns the kernel's rule (CPU) on 16 of 16 rows; the card's "
                   "torch path returns CPU torch.topk's set on 15 of the 15 rows that do not tie at the cut",
                   "MOEHPU row_quant_real_r32 against dynamic_quant on the card: 0 of 32 scales and 0 of 81920 fp8"):
        assert needle in text, needle
    assert re.search(r"MOEHPU measurements of this run \((\d+) lines, passed tests included\):", text)


def test_the_card_test_skips_as_a_whole_without_a_card():
    """Without the dry-run switch and without habana_frameworks the module skips (and the gate fails it)."""
    env = {key: value for key, value in os.environ.items() if key != "MOE_GLUE_OP_TEST_DEVICE"}
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", CARD_TEST],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
                          env=dict(env, PYTHONPATH=str(REPO_ROOT), VLLM_PLUGINS=""))
    text = proc.stdout + proc.stderr
    assert "1 skipped" in text and "passed" not in text.splitlines()[-1], text[-2000:]
    assert "habana_frameworks" in text
