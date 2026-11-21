# SPDX-License-Identifier: Apache-2.0
"""Mac tests for tools/flash_next/gdn_verify_crash_probe.sh: --dry-run planning and --classify."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "flash_next" / "gdn_verify_crash_probe.sh"

NODE_DECODE = "tests/unit_tests/gdn_core_decode/test_hpu_op.py::test_hpu_op_matches_reference[tp8_mixed]"
NODE_T1 = "tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py::test_hpu_verify_op_matches_oracle[ver_tp4_t1]"
NODE_CHAIN = "tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py::test_hpu_verify_op_matches_oracle[ver_tp8_chain]"

EXPECTED = [
    ("control_decode_lazy", "1", NODE_DECODE, "<installed>"),
    ("verify_t1_lazy", "1", NODE_T1, "<installed>"),
    ("verify_chain_lazy", "1", NODE_CHAIN, "<installed>"),
    ("verify_chain_eager", "0", NODE_CHAIN, "<installed>"),
    ("strip_verify_t1_lazy", "1", NODE_T1, "build_strip/libgdn_fused_gaudi2.so"),
    ("strip_verify_chain_lazy", "1", NODE_CHAIN, "build_strip/libgdn_fused_gaudi2.so"),
]


def script_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GDN_FUSED_") and k not in ("PLUGIN_ROOT", "PYTHON", "PROBE_TIMEOUT_S")}
    if extra:
        env.update(extra)
    return env


def run_script(args, env=None):
    return subprocess.run(
        ["bash", str(SCRIPT)] + list(args),
        capture_output=True,
        text=True,
        timeout=60,
        env=script_env(env),
    )


def test_dry_run_lists_six_runs_in_order(tmp_path):
    out_dir = tmp_path / "probe"
    proc = run_script(["--dry-run", str(out_dir)])
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("run ")]
    assert len(lines) == 6
    for line, (name, lazy, node, lib) in zip(lines, EXPECTED):
        assert line.startswith("run %s: PT_HPU_LAZY_MODE=%s " % (name, lazy))
        lib_field = line.split(" ", 4)[3]
        if lib == "<installed>":
            assert lib_field == "GDN_FUSED_GAUDI_KERNEL_LIBRARY=<installed>"
        else:
            assert lib_field == "GDN_FUSED_GAUDI_KERNEL_LIBRARY=%s/%s" % (out_dir, lib)
        assert node in line
        assert "-m pytest -q -p no:cacheprovider -rs" in line
    assert "timeout -k 10 300" in lines[0]


def test_dry_run_uses_kernel_library_env_override(tmp_path):
    proc = run_script(["--dry-run", str(tmp_path / "probe")], env={"GDN_FUSED_GAUDI_KERNEL_LIBRARY": "/opt/gdn/libgdn_fused_gaudi2.so"})
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("run ")]
    assert len(lines) == 6
    for line, (_name, _lazy, _node, lib) in zip(lines, EXPECTED):
        if lib == "<installed>":
            assert "GDN_FUSED_GAUDI_KERNEL_LIBRARY=/opt/gdn/libgdn_fused_gaudi2.so" in line
        else:
            assert "build_strip/libgdn_fused_gaudi2.so" in line


def test_classify_crash_signal(tmp_path):
    log = tmp_path / "crash.log"
    log.write_text('some output\nThread 3 "python" received signal SIGSEGV, Segmentation fault.\n#0  0x0 in foo ()\n')
    proc = run_script(["--classify", str(log), "0"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "crashed:SIGSEGV"


def test_classify_timeout():
    proc = run_script(["--classify", "/dev/null", "124"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "timed_out"


def test_classify_killed():
    proc = run_script(["--classify", "/dev/null", "137"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "timed_out"


def test_classify_segv_exit_without_gdb(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("")
    proc = run_script(["--classify", str(log), "139"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "crashed:exit139"


def test_classify_passed(tmp_path):
    log = tmp_path / "pass.log"
    log.write_text(".\n1 passed in 3.21s\n")
    proc = run_script(["--classify", str(log), "0"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "passed"


def test_classify_failed(tmp_path):
    log = tmp_path / "fail.log"
    log.write_text("F\n1 failed in 2.00s\n")
    proc = run_script(["--classify", str(log), "1"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "failed"


def test_classify_skipped_is_not_passed(tmp_path):
    log = tmp_path / "skip.log"
    log.write_text("s\n1 skipped in 0.10s\n")
    proc = run_script(["--classify", str(log), "0"])
    assert proc.returncode == 0
    assert proc.stdout.strip() == "failed"


def test_no_arguments_exits_2():
    proc = run_script([])
    assert proc.returncode == 2
    assert "usage" in proc.stderr
