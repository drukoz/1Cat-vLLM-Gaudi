# SPDX-License-Identifier: Apache-2.0
"""The rental run sheet must say what the scripts do.

docs/design/flash-next/run-sheets/rental.md is what the operator reads in a paid session. These tests
compare it with the scripts' own usage texts and with the constants it quotes, so an option, a phase
or a number that changes in the code fails here until the sheet follows.
Runs on CPU from the worktree root with ``PYTHONPATH=$PWD``.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "tools" / "flash_next" / "rental"
SHEET = REPO_ROOT / "docs" / "design" / "flash-next" / "run-sheets" / "rental.md"


def _sheet():
    return SHEET.read_text(encoding="utf-8")


def _usage_options(script):
    proc = subprocess.run(["bash", str(SCRIPTS_DIR / script), "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    # the option that starts a usage line: "  --name ..."
    return sorted(set(re.findall(r"^\s{2}(--[a-z][a-z-]*)", proc.stdout, re.M)))


def _constant(path, name):
    match = re.search(r"^%s = (.+?)(?:\s+#.*)?$" % name, (REPO_ROOT / path).read_text(), re.M)
    assert match, "%s not found in %s" % (name, path)
    return ast.literal_eval(match.group(1))


@pytest.mark.parametrize("script", ["run_rental.sh", "tiny_smoke.sh"])
def test_every_option_of_the_script_is_in_the_sheet(script):
    options = _usage_options(script)
    assert len(options) >= 10, options
    sheet = _sheet()
    missing = [opt for opt in options if "`%s" % opt not in sheet]
    assert not missing, "%s options the run sheet does not name: %s" % (script, missing)


def test_the_new_options_have_rows_in_the_run_table():
    sheet = _sheet()
    for option in ("`--quick-gates`", "`--no-verify-probe`", "`--load-read-timeout S`", "`--warm-request-timeout S`",
                   "`--download-only`"):
        assert "| %s |" % option in sheet, option


def test_the_from_row_names_every_phase():
    sheet = _sheet()
    row = [line for line in sheet.splitlines() if line.startswith("| `--from PHASE` |")]
    assert len(row) == 1
    phases = list(_constant("tools/flash_next/rental/main.py", "GATE_PHASES")) + ["report"]
    assert "fused_gdn" in phases and "card_tests" in phases
    for phase in phases:
        assert "`%s`" % phase in row[0], phase
    assert "`--run-id` is required" in row[0]
    # and the description of the run lists them too
    intro = sheet[sheet.index("## What the run is"):sheet.index("## Cheapest session order")]
    for phase in phases:
        assert "`%s`" % phase in intro, phase


def test_the_sheet_quotes_the_tiny_gates_numbers():
    sheet = _sheet()
    lengths = _constant("tools/flash_next/make_tiny_checkpoint.py", "PROMPT_LENGTHS")
    words = ["{:,}".format(n) for n in lengths]
    assert "six prompts of %s and %s tokens" % (", ".join(words[:-1]), words[-1]) in sheet
    assert len(lengths) == 6
    head_dim = _constant("tools/flash_next/rental/main.py", "TINY_GDN_HEAD_DIM")
    assert "head dim %d" % head_dim in sheet
    health = _constant("tools/flash_next/rental/main.py", "TINY_HEALTH_TIMEOUT_S")
    assert "fixed at {:,} s".format(int(health)) in sheet  # the tiny gate does not use --server-timeout
    assert "default {:d})".format(int(_constant("tools/flash_next/rental/main.py", "LOAD_REQUEST_TIMEOUT_S"))) in sheet
    assert "default {:d})".format(int(_constant("tools/flash_next/rental/main.py",
                                                "WARM_PASS_REQUEST_TIMEOUT_S"))) in sheet


def test_the_stale_statements_are_gone():
    sheet = _sheet()
    for stale in ("four short prompts", "20–40 min", "plan 1–2 h", "until then, run it by hand",
                  "the `fused` mode never gets it", "the tarball at the end is never written"):
        assert stale not in sheet, stale


def test_the_sheet_has_the_session_order_and_the_fallback():
    sheet = _sheet()
    order = sheet[sheet.index("## Cheapest session order"):sheet.index("## Before the rental")]
    steps = ["prepare_host.sh", "--download-only", "tiny_smoke.sh --gdn pytorch", "tiny_smoke.sh --gdn fused",
             "--quick-gates", "first token after", "first-token.json", "compile probe", "stop the node"]
    positions = [order.index(step) for step in steps]
    assert positions == sorted(positions), list(zip(steps, positions))
    assert "tmux" in order
    assert "rerun with `--no-combined-kernel-db`" in order
    assert "retries by itself" in order


def test_the_sheet_names_what_the_scripts_add_to_the_run_folder():
    sheet = _sheet()
    run_text = (SCRIPTS_DIR / "run_rental.sh").read_text()
    entry_text = (SCRIPTS_DIR / "container_entry.sh").read_text()
    for name in ("dmesg-tail.txt", "image-digest.txt", "container-state.json", "habana_logs"):
        assert name in run_text, name
        assert name in sheet, name
    assert "pip-freeze.txt" in entry_text and "pip-freeze.txt" in sheet
    assert "pytest==9.2.0" in entry_text and "pinned to 9.2.0" in sheet
    assert "tiny/ckpt/*.safetensors" in run_text and "tiny/ckpt/*.safetensors" in sheet
    assert ".bundle-sha" in entry_text and "plugin/build/.bundle-sha" in sheet
