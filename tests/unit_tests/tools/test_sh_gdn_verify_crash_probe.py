# SPDX-License-Identifier: Apache-2.0
"""Tests for a real run of tools/flash_next/gdn_verify_crash_probe.sh with stand-in commands.

`test_gdn_verify_crash_probe.py` covers `--dry-run` and `--classify`. Here the six checks really run,
against a stand-in `python` (it plays pytest: a line on stderr as the graph compiler would write it,
then the pytest summary, or a segfault), with and without a stand-in `gdb`, and with stand-in
`timeout`, `cmake`, `readelf` and `apt-get`, so the test needs no card and runs on the Mac.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "flash_next" / "gdn_verify_crash_probe.sh"

NAMES = ["control_decode_lazy", "verify_t1_lazy", "verify_chain_lazy", "verify_chain_eager", "strip_verify_t1_lazy",
         "strip_verify_chain_lazy"]

FAKE_PYTHON = r"""
case "$1" in
  -c) echo "2.11.0"; exit 0 ;;
esac
{
  echo "argv: $*"
  echo "env: PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-unset} PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-unset}" \
    "HABANA_LOGS=${HABANA_LOGS:-unset}"
} >> "$FAKE_PY_LOG"
echo "GRAPH COMPILER: compilation failed for node gdn_core_verify" >&2
case "$*" in
  *"${FAKE_CRASH_NODE:-no-such-node}"*)
    echo "Fatal Python error: Segmentation fault" >&2
    echo "" >&2
    echo "Current thread 0x00007f0000000000 (most recent call first):" >&2
    echo '  File "/bundle/plugin/tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py", line 88 in test_oracle' >&2
    kill -SEGV $$ ;;
esac
echo "."
echo "1 passed in 0.10s"
"""

# `timeout [-s SIG] [-k 10] 300 CMD...`: record the options, run CMD in place.
FAKE_TIMEOUT = r"""
echo "timeout: $1 $2 $3 $4" >> "$FAKE_PY_LOG"
while [ "$1" = "-k" ] || [ "$1" = "-s" ]; do shift 2; done
shift
exec "$@"
"""

# `gdb ... --args CMD...`: run CMD in place (no backtrace: the checks pass).
FAKE_GDB = r"""
echo "gdb: $*" >> "$FAKE_PY_LOG"
while [ $# -gt 0 ] && [ "$1" != "--args" ]; do shift; done
shift
exec "$@"
"""

FAKE_CMAKE = r"""
case "$1" in
  --build)
    mkdir -p "$2"
    : > "$2/gdn_core_verify_gaudi2.o"
    : > "$2/libgdn_fused_gaudi2.so" ;;
esac
exit 0
"""


def _run_probe(tmp_path, with_gdb, **fake):
    plugin = tmp_path / "plugin"
    (plugin / "csrc" / "gdn_fused").mkdir(parents=True)
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    fakes = {"python": FAKE_PYTHON, "timeout": FAKE_TIMEOUT, "cmake": FAKE_CMAKE, "readelf": "exit 0\n",
             "apt-get": 'echo "apt-get $*" >> "$FAKE_PY_LOG"\nexit 100\n', "tpc-clang": 'echo "tpc 2.7.33"\n',
             "sha256sum": 'echo "0000  $1"\n'}
    if with_gdb:
        fakes["gdb"] = FAKE_GDB
    for name, body in fakes.items():
        path = bin_dir / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GDN_FUSED_") and k not in (
        "BASH_ENV", "PROBE_TIMEOUT_S", "PYTHONFAULTHANDLER", "HABANA_LOGS")}
    env.update(PATH=f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin", PYTHON="python", PLUGIN_ROOT=str(plugin),
               GDN_FUSED_GAUDI_KERNEL_LIBRARY="/opt/lib/libgdn_fused_gaudi2.so",
               GDN_FUSED_GAUDI_NATIVE_LIBRARY="/opt/lib/gdn_fused_ops.so", FAKE_PY_LOG=str(tmp_path / "python.log"))
    env.update({k: str(v) for k, v in fake.items()})
    out_dir = tmp_path / "out"
    proc = subprocess.run(["/bin/bash", str(SCRIPT), str(out_dir)], capture_output=True, text=True, cwd=tmp_path,
                          env=env, timeout=120)
    log = (tmp_path / "python.log").read_text().splitlines() if (tmp_path / "python.log").exists() else []
    return proc, out_dir, log


def _summary(out_dir):
    return (out_dir / "summary.txt").read_text().splitlines()


def test_dry_run_shows_the_uncaptured_pytest_command(tmp_path):
    proc = subprocess.run(["bash", str(SCRIPT), "--dry-run", str(tmp_path / "probe")], capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("run ")]
    assert len(lines) == 6
    for line in lines:
        assert "PYTHONFAULTHANDLER=1" in line
        assert "-m pytest -q -p no:cacheprovider -rs -s" in line


def test_without_gdb_every_check_still_runs_and_keeps_its_output(tmp_path):
    proc, out_dir, log = _run_probe(tmp_path, with_gdb=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _summary(out_dir)
    assert "gdb: missing" in summary
    for name in NAMES:
        assert "%s: passed" % name in summary, summary
        text = (out_dir / ("%s.log" % name)).read_text()
        # -s: the graph compiler's stderr is in the log, next to pytest's own summary
        assert "GRAPH COMPILER: compilation failed for node gdn_core_verify" in text, name
        assert "1 passed in 0.10s" in text, name
    runs = [line for line in log if line.startswith("argv: ")]
    assert len(runs) == 6
    for line in runs:
        assert " -m pytest -q -p no:cacheprovider -rs -s tests/unit_tests/gdn_core_decode/" in line
    envs = [line for line in log if line.startswith("env: ")]
    assert len(envs) == 6
    for k, line in enumerate(envs):
        assert "PYTHONFAULTHANDLER=1" in line
        assert "HABANA_LOGS=%s/%s.habana_logs" % (out_dir, NAMES[k]) in line
    assert not [line for line in log if line.startswith("gdb: ")]
    # a check that hits the time limit is ended with SIGABRT, so the fault handler writes its stack
    assert len([line for line in log if line.startswith("timeout: -s ABRT -k 10")]) == 6


def test_with_gdb_the_checks_run_under_it_with_the_same_pytest_options(tmp_path):
    proc, out_dir, log = _run_probe(tmp_path, with_gdb=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _summary(out_dir)
    assert "gdb: present" in summary
    for name in NAMES:
        assert "%s: passed" % name in summary, summary
        assert "GRAPH COMPILER" in (out_dir / ("%s.log" % name)).read_text()
    gdb = [line for line in log if line.startswith("gdb: ")]
    assert len(gdb) == 6
    for line in gdb:
        assert "--args" in line and " -m pytest -q -p no:cacheprovider -rs -s " in line
    for line in [line for line in log if line.startswith("env: ")]:
        assert "PYTHONFAULTHANDLER=1" in line
    assert not [line for line in log if line.startswith("apt-get")]  # gdb is there: nothing to install


def test_without_gdb_a_segfault_is_classified_and_its_python_stack_reaches_the_summary(tmp_path):
    proc, out_dir, log = _run_probe(tmp_path, with_gdb=False, FAKE_CRASH_NODE="ver_tp4_t1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _summary(out_dir)
    text = "\n".join(summary)
    assert "control_decode_lazy: passed" in summary
    assert "verify_t1_lazy: crashed:exit139" in summary
    assert "strip_verify_t1_lazy: crashed:exit139" in summary
    assert "verify_chain_lazy: passed" in summary  # the checks after a crash still run
    assert len([line for line in summary if line.split(":")[0] in NAMES]) == 6
    # no gdb backtrace: the faulthandler stack stands in for it
    assert "Fatal Python error: Segmentation fault" in text
    assert "line 88 in test_oracle" in text
    crash_log = (out_dir / "verify_t1_lazy.log").read_text()
    assert "GRAPH COMPILER: compilation failed for node gdn_core_verify" in crash_log
    assert proc.stdout.splitlines() == summary  # the summary is what the probe prints


def test_classify_reads_a_summary_line_that_follows_uncaptured_output(tmp_path):
    # With -s a library can leave text without a newline right before pytest's summary line.
    log = tmp_path / "glued.log"
    log.write_text("synapse: graph compiled.1 passed in 3.21s\n")
    proc = subprocess.run(["bash", str(SCRIPT), "--classify", str(log), "0"], capture_output=True, text=True)
    assert proc.stdout.strip() == "passed"
    log.write_text("synapse: graph compiled.1 failed, 1 passed in 3.21s\n")
    proc = subprocess.run(["bash", str(SCRIPT), "--classify", str(log), "1"], capture_output=True, text=True)
    assert proc.stdout.strip() == "failed"
    log.write_text("synapse: graph compiled.1 skipped in 0.21s\n")
    proc = subprocess.run(["bash", str(SCRIPT), "--classify", str(log), "0"], capture_output=True, text=True)
    assert proc.stdout.strip() == "failed"


def test_classify_is_not_misled_by_uncaptured_output_that_mentions_failures(tmp_path):
    log = tmp_path / "noise.log"
    log.write_text("synapse: 3 failed attempts to reuse a recipe, error code 0\n.\n1 passed in 3.21s\n")
    proc = subprocess.run(["bash", str(SCRIPT), "--classify", str(log), "0"], capture_output=True, text=True)
    assert proc.stdout.strip() == "passed"
    log.write_text("1 passed in 3.21s\nsynapse: 3 failed attempts while closing the device\n")
    proc = subprocess.run(["bash", str(SCRIPT), "--classify", str(log), "0"], capture_output=True, text=True)
    assert proc.stdout.strip() == "passed"
