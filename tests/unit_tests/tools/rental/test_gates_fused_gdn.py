# SPDX-License-Identifier: Apache-2.0
"""Tests for gates.fused_gdn and gates.gdn_fallback_lines (plan M4, RUK-2709)."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from pathlib import Path

from tools.flash_next.rental import gates
from tools.flash_next.rental.schema import GateStatus

OPS_SO = "gdn_fused_ops.cpython-310-x86_64-linux-gnu.so"
COMBINED_DB = "libflashinfer_gaudi_combined_gaudi2.so"
KILL_SWITCH = "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"


def write_fake_python(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


FAKE_PY = r"""
{
  echo "ARGS: $*"
  echo "PWD: $(pwd)"
  echo "KERNEL: ${GDN_FUSED_GAUDI_KERNEL_LIBRARY-<unset>}"
  echo "NATIVE: ${GDN_FUSED_GAUDI_NATIVE_LIBRARY-<unset>}"
  echo "PYTHONPATH: ${PYTHONPATH-<unset>}"
  echo "GDNMODE: ${VLLM_HPU_GDN_CORE_DECODE-<unset>}"
  echo "GDNVERIFY: ${VLLM_HPU_GDN_CORE_VERIFY-<unset>}"
  echo "LAZY: ${PT_HPU_LAZY_MODE-<unset>}"
  echo "COMBINED: ${FLASHINFER_GAUDI_COMBINED_KERNEL_DB-<unset>}"
} >> "$FAKE_PY_LOG"
case "$1" in
  -c) if [ -n "$FAKE_LOOKUP_NOISE" ]; then echo "$FAKE_LOOKUP_NOISE"; fi
      echo "$FAKE_LIB_DIR"
      exit ${FAKE_LOOKUP_EXIT:-0} ;;
  -m) case "$*" in
        *test_hpu_op.py*) if [ "$FLASHINFER_GAUDI_COMBINED_KERNEL_DB" != "0" ]; then
                            if [ -n "$FAKE_COMBINED_OP_SLEEP" ]; then sleep "$FAKE_COMBINED_OP_SLEEP"; fi
                            if [ -n "$FAKE_COMBINED_OP_FAILS" ]; then
                              echo "FAILED tests/unit_tests/gdn_core_decode/test_hpu_op.py::test_a - synStatus=26"
                              echo "1 failed in 0.20s"
                              exit 1
                            fi
                          fi
                          if [ -n "$FAKE_OP_PRINT" ]; then
                            echo "$FAKE_OP_PRINT"
                            sleep 60
                          fi
                          if [ -n "$FAKE_HANG_PIDFILE" ]; then
                            sleep 60 &
                            echo "$!" > "$FAKE_HANG_PIDFILE"
                            sleep 60
                          fi ;;
      esac
      case "$*" in
        *test_hpu_verify_op.py*)
          if [ -n "$FAKE_VERIFY_CRASH" ]; then
            echo "Fatal Python error: Segmentation fault"
            kill -s SEGV $$
          fi
          if [ -n "$FAKE_VERIFY_PYTEST_SLEEP" ]; then sleep "$FAKE_VERIFY_PYTEST_SLEEP"; fi
          printf '%s\n' "$FAKE_VERIFY_PYTEST_OUT"
          exit ${FAKE_VERIFY_PYTEST_EXIT:-0} ;;
      esac
      printf '%s\n' "$FAKE_PYTEST_OUT"
      exit ${FAKE_PYTEST_EXIT:-0} ;;
  *replay*)
    case "$*" in
      *"--ops verify"*)
        if [ -n "$FAKE_VERIFY_REPLAY_SLEEP" ]; then sleep "$FAKE_VERIFY_REPLAY_SLEEP"; fi
        case "$*" in
          *ver_tp8_b64*) exit ${FAKE_VERIFY_REPLAY2_EXIT:-0} ;;
          *) exit ${FAKE_VERIFY_REPLAY_EXIT:-0} ;;
        esac ;;
      *) if [ "$PT_HPU_LAZY_MODE" = "0" ]; then
           if [ -n "$FAKE_EAGER_REPLAY_SLEEP" ]; then sleep "$FAKE_EAGER_REPLAY_SLEEP"; fi
           exit ${FAKE_EAGER_REPLAY_EXIT:-0}
         fi
         if [ -n "$FAKE_COMBINED_REPLAY_FAILS" ] && [ "$FLASHINFER_GAUDI_COMBINED_KERNEL_DB" != "0" ]; then
           echo "replay through the combined database: mismatch at step 3"
           exit 1
         fi
         echo "replay ok"
         if [ -n "$FAKE_REPLAY_SLEEP" ]; then sleep "$FAKE_REPLAY_SLEEP"; fi
         case "$*" in
           *--case*) exit ${FAKE_REPLAY2_EXIT:-0} ;;
           *) exit ${FAKE_REPLAY_EXIT:-0} ;;
         esac ;;
    esac ;;
  *bench*)
    if [ -n "$FAKE_BENCH_SLEEP" ]; then sleep "$FAKE_BENCH_SLEEP"; fi
    printf '%s\n' "core recurrent slope 0.51 ms/step" "core slope 0.42 ms/step" \
"core tp8 B=1 0.31 ms" "plain line"
    exit ${FAKE_BENCH_EXIT:-0} ;;
esac
exit 0
"""


def make_lib(tmp_path: Path, ops: bool = True, combined: bool = False) -> Path:
    """A fake installed flashinfer_gaudi/lib/ with the fused kernel library (and optionally the op and
    the combined kernel database)."""
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    (lib / "libgdn_fused_gaudi2.so").write_bytes(b"\0")
    if ops:
        (lib / OPS_SO).write_bytes(b"\0")
    if combined:
        (lib / COMBINED_DB).write_bytes(b"\0")
    return lib


def setup(tmp_path: Path, monkeypatch, *, lib: Path, pytest_out: str = "..... [100%]\n5 passed in 12.34s",
          pytest_exit: int = 0, verify_test: bool = True,
          verify_pytest_out: str = "... [100%]\n3 passed in 8.90s", **extra) -> tuple:
    monkeypatch.delenv(KILL_SWITCH, raising=False)  # a test that wants the operator's switch sets it afterwards
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_PY_LOG", str(log))
    monkeypatch.setenv("FAKE_LIB_DIR", str(lib))
    monkeypatch.setenv("FAKE_PYTEST_OUT", pytest_out)
    monkeypatch.setenv("FAKE_PYTEST_EXIT", str(pytest_exit))
    monkeypatch.setenv("FAKE_VERIFY_PYTEST_OUT", verify_pytest_out)
    for name, value in extra.items():
        monkeypatch.setenv(name, str(value))
    fake = write_fake_python(tmp_path / "fake_python.sh", FAKE_PY)
    plugin = tmp_path / "plugin"
    (plugin / "tests").mkdir(parents=True)
    if verify_test:  # the verify op's test file reaches the bundle only when the program branch is merged
        verify_op = plugin / gates.GDN_VERIFY_OP_TEST
        verify_op.parent.mkdir(parents=True)
        verify_op.write_text("# fake verify op test\n", encoding="utf-8")
    return fake, plugin, log


def records(log: Path) -> list:
    """The fake interpreter's invocations, in order, as {args, PWD, KERNEL, NATIVE, PYTHONPATH, GDNMODE,
    GDNVERIFY, LAZY}."""
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    out = []
    for block in text.split("ARGS: ")[1:]:
        lines = block.splitlines()
        rec = {"args": lines[0]}
        for ln in lines[1:]:
            key, _, value = ln.partition(": ")
            rec[key] = value
        out.append(rec)
    return out


def assert_pid_gone(pid: int, within_s: float = 5.0) -> None:
    """Poll os.kill(pid, 0) until it raises ProcessLookupError; fail when the pid survives within_s."""
    deadline = time.monotonic() + within_s
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise AssertionError("pid %d is still alive %s s after the step ended" % (pid, within_s))
        time.sleep(0.05)


def test_fused_gdn_all_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "1")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "1")
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    out = tmp_path / "g1b"
    status, details = gates.fused_gdn(plugin, out, python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert details["verify_mode"] == 1
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert details["seconds"] >= 0
    assert details["hpu_op"] == {"exit": 0, "passed": 5, "skipped": 0, "failed": 0}
    assert details["failed_tests"] == []
    assert details["replay"] == [{"case": "tp8_b16", "lazy_mode": 1, "exit": 0},
                                 {"case": "tp8_b64", "lazy_mode": 1, "exit": 0}]
    assert details["replay_eager"] == [{"case": "tp8_b16", "lazy_mode": 0, "exit": 0},
                                       {"case": "tp8_b64", "lazy_mode": 0, "exit": 0}]
    assert details["eager_compile_ok"] is True
    assert details["verify"]["status"] == "passed"
    assert details["verify"]["hpu_op"] == {"exit": 0, "passed": 3, "skipped": 0, "failed": 0}
    assert details["verify"]["replay"] == [{"case": "ver_tp8_b16", "lazy_mode": 1, "exit": 0},
                                           {"case": "ver_tp8_b64", "lazy_mode": 1, "exit": 0}]
    assert details["bench"]["exit"] == 0
    assert details["bench"]["lines"] == [
        "core recurrent slope 0.51 ms/step",
        "core slope 0.42 ms/step",
        "core tp8 B=1 0.31 ms",
    ]
    recs = records(log)
    assert len(recs) == 10  # the library lookup, then the nine steps, in order
    assert recs[0]["args"].startswith("-c import flashinfer_gaudi")
    assert os.path.realpath(recs[0]["PWD"]) == os.path.realpath(tempfile.gettempdir())
    steps = recs[1:]
    assert "pytest" in steps[0]["args"] and "test_hpu_op.py" in steps[0]["args"]
    assert "gdn_core_decode_replay.py" in steps[1]["args"] and "--case" not in steps[1]["args"]
    assert "gdn_core_decode_replay.py" in steps[2]["args"] and "--case tp8_b64" in steps[2]["args"]
    assert "gdn_core_decode_replay.py" in steps[3]["args"] and "--case" not in steps[3]["args"]
    assert "gdn_core_decode_replay.py" in steps[4]["args"] and "--case tp8_b64" in steps[4]["args"]
    assert "pytest" in steps[5]["args"] and "test_hpu_verify_op.py" in steps[5]["args"]
    assert "--ops verify" in steps[6]["args"] and "--case" not in steps[6]["args"]
    assert "--ops verify" in steps[7]["args"] and "--case ver_tp8_b64" in steps[7]["args"]
    assert "gdn_core_decode_bench.py" in steps[8]["args"]
    for rec in steps:
        assert os.path.realpath(rec["PWD"]) == os.path.realpath(plugin)
        assert rec["KERNEL"] == str(lib / "libgdn_fused_gaudi2.so")
        assert rec["NATIVE"] == str(lib / OPS_SO)
        assert rec["PYTHONPATH"] == str(plugin)
        assert rec["GDNMODE"] == "<unset>"  # the gate always removes it for the steps
        assert rec["GDNVERIFY"] == "<unset>"  # ... and the verify flag too
    for name in ("hpu_op.log", "hpu_verify_op.log", "replay.log", "bench.log"):
        assert (out / name).is_file()
    assert "5 passed" in (out / "hpu_op.log").read_text(encoding="utf-8")
    assert "3 passed" in (out / "hpu_verify_op.log").read_text(encoding="utf-8")


def test_fused_gdn_lookup_ignores_log_lines(tmp_path, monkeypatch):
    # On the card the lookup's import logs to stdout before it prints the folder (2026-11-18 rental, RUK-3100).
    lib = make_lib(tmp_path)
    noise = "INFO 11-18 22:17:11 [__init__.py:272] Platform plugin hpu is activated"
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_LOOKUP_NOISE=noise)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert records(log)[1]["KERNEL"] == str(lib / "libgdn_fused_gaudi2.so")


def test_fused_gdn_missing_library_skips(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, ops=False)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.SKIPPED
    assert details["mode"] == 0
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["reason"].startswith("the fused GDN library is not installed")
    assert "gdn_fused_ops" in details["reason"]
    assert len(records(log)) == 1  # only the lookup: no step ran
    assert not (tmp_path / "g1b" / "hpu_op.log").exists()


def test_fused_gdn_pytest_skip_is_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, pytest_out="s\n1 skipped in 0.10s", pytest_exit=0)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED
    assert details["mode"] == 0
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["reason"] == "test_hpu_op.py skipped"
    assert details["hpu_op"] == {"exit": 0, "passed": 0, "skipped": 1, "failed": 0}
    assert details["verify"] == {"status": "not_run", "reason": "decode checks failed", "hpu_op": None, "replay": []}
    assert "replay" not in details and "bench" not in details
    assert len(records(log)) == 2  # lookup + pytest: replay and bench did not run


def test_fused_gdn_pytest_failures_recorded(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    pytest_out = ("FAILED tests/unit_tests/gdn_core_decode/test_hpu_op.py::test_gdn_core_decode_matches_cpu - assert\n"
                  "FAILED tests/unit_tests/gdn_core_decode/test_hpu_op.py::test_gdn_recurrent_decode_matches_cpu\n"
                  "2 failed in 0.20s")
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out=pytest_out, pytest_exit=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED
    assert details["mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "not_run" and details["verify"]["reason"] == "decode checks failed"
    assert details["failed_tests"] == pytest_out.splitlines()[:2]
    assert details["hpu_op"] == {"exit": 1, "passed": 0, "skipped": 0, "failed": 2}
    assert "test_hpu_op.py" in details["reason"] and "exit 1" in details["reason"]


def test_fused_gdn_second_replay_fails(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_REPLAY2_EXIT=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED
    assert details["mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "not_run" and details["verify"]["reason"] == "decode checks failed"
    assert details["replay"] == [{"case": "tp8_b16", "lazy_mode": 1, "exit": 0},
                                 {"case": "tp8_b64", "lazy_mode": 1, "exit": 1}]
    assert "bench" not in details
    assert not any("bench.py" in rec["args"] for rec in records(log))
    assert "tp8_b64" in details["reason"] and "exit 1" in details["reason"]


def test_fused_gdn_replay_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_REPLAY_TIMEOUT_S", 1)
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_REPLAY_SLEEP=4)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.TIMED_OUT
    assert details["mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "not_run" and details["verify"]["reason"] == "decode checks failed"
    assert "replay" in details["reason"]
    assert "bench" not in details
    assert not any("bench.py" in rec["args"] for rec in records(log))


def test_fused_gdn_verify_op_missing_from_bundle(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, verify_test=False)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "not_run"
    assert details["verify"]["reason"] == "%s is not in the bundle" % gates.GDN_VERIFY_OP_TEST
    assert details["verify"]["hpu_op"] is None and details["verify"]["replay"] == []
    args = [rec["args"] for rec in records(log)]
    assert not any("test_hpu_verify_op.py" in a for a in args)  # no verify command ran
    assert not any("--ops verify" in a for a in args)
    assert any("bench.py" in a for a in args)  # the decode gate and the bench are unchanged


def test_fused_gdn_verify_pytest_skip_is_verify_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, verify_pytest_out="s\n1 skipped in 0.10s")
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED  # G1b's status is the decode kernel's
    assert details["mode"] == 1
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "failed"
    assert details["verify"]["reason"] == "test_hpu_verify_op.py skipped"
    assert details["verify"]["hpu_op"] == {"exit": 0, "passed": 0, "skipped": 1, "failed": 0}
    assert details["verify"]["replay"] == []
    args = [rec["args"] for rec in records(log)]
    assert not any("--ops verify" in a for a in args)  # the verify replays did not run
    assert any("bench.py" in a for a in args)  # the bench still ran


def test_fused_gdn_verify_pytest_failure_is_verify_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib,
                              verify_pytest_out="F\n1 failed, 2 passed in 0.20s", FAKE_VERIFY_PYTEST_EXIT=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "failed"
    assert "test_hpu_verify_op.py" in details["verify"]["reason"] and "exit 1" in details["verify"]["reason"]
    assert details["verify"]["hpu_op"] == {"exit": 1, "passed": 2, "skipped": 0, "failed": 1}
    args = [rec["args"] for rec in records(log)]
    assert not any("--ops verify" in a for a in args)
    assert any("bench.py" in a for a in args)


def test_fused_gdn_verify_crash_runs_probe(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    calls = []

    def fake_probe(plugin_root, out_dir, env, python):
        calls.append({"plugin_root": plugin_root, "out_dir": out_dir, "env": env, "python": python})
        return {"exit": 0, "summary": ["x"]}

    monkeypatch.setattr(gates, "_gdn_crash_probe", fake_probe)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_VERIFY_CRASH=1)
    out = tmp_path / "g1b"
    status, details = gates.fused_gdn(plugin, out, python=fake)
    assert status is GateStatus.PASSED  # the gate's status is the decode kernel's
    assert details["mode"] == 1
    assert details["verify_mode"] == 0
    assert details["verify"]["status"] == "failed"
    assert details["env"]["VLLM_HPU_GDN_CORE_VERIFY"] == "0"
    assert len(calls) == 1  # the probe ran exactly once
    assert details["verify"]["crash_probe"] == {"exit": 0, "summary": ["x"]}
    assert calls[0]["plugin_root"] == plugin and calls[0]["out_dir"] == out
    assert calls[0]["python"] == fake and calls[0]["env"]["PT_HPU_LAZY_MODE"] == "1"
    assert "Fatal Python error: Segmentation fault" in (out / "hpu_verify_op.log").read_text(encoding="utf-8")


def test_fused_gdn_verify_plain_failure_no_probe(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    calls = []
    monkeypatch.setattr(gates, "_gdn_crash_probe", lambda *args: calls.append(args))
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib,
                              verify_pytest_out="F\n1 failed, 2 passed in 0.20s", FAKE_VERIFY_PYTEST_EXIT=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["verify"]["status"] == "failed"
    assert calls == []  # exit 1 is not a crash: the probe did not run
    assert "crash_probe" not in details["verify"]


def test_fused_gdn_timeout_keeps_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_OP_TIMEOUT_S", 1)
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_OP_PRINT="partial-marker")
    out = tmp_path / "out"
    status, details = gates.fused_gdn(plugin, out, python=fake)
    assert status is GateStatus.TIMED_OUT
    assert details["mode"] == 0
    assert "partial-marker" in (out / "hpu_op.log").read_text(encoding="utf-8")


def test_fused_gdn_verify_second_replay_fails(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_VERIFY_REPLAY2_EXIT=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "failed"
    assert details["verify"]["replay"] == [{"case": "ver_tp8_b16", "lazy_mode": 1, "exit": 0},
                                           {"case": "ver_tp8_b64", "lazy_mode": 1, "exit": 1}]
    assert "ver_tp8_b64" in details["verify"]["reason"] and "exit 1" in details["verify"]["reason"]
    assert any("bench.py" in rec["args"] for rec in records(log))


def test_fused_gdn_verify_replay_timeout_keeps_kernel_on(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_REPLAY_TIMEOUT_S", 1)
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_VERIFY_REPLAY_SLEEP=4)
    status, details = gates.fused_gdn(plugin, tmp_path / "out", python=fake)
    assert status is GateStatus.PASSED  # a verify timeout never changes the gate's status
    assert details["mode"] == 1
    assert details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["verify"]["status"] == "timed_out"
    assert "verify ver_tp8_b16" in details["verify"]["reason"]
    assert details["verify"]["replay"] == []
    args = [rec["args"] for rec in records(log)]
    assert not any("ver_tp8_b64" in a for a in args)  # the second verify replay did not run
    assert any("bench.py" in a for a in args)  # the bench still ran


FALLBACK_LOG = "\n".join([
    "INFO 10:00:00 [config] Setting attention block size to 128 tokens",
    "WARNING 10:00:01 [gdn] Qwen3NextGatedDeltaNet has no HPU kernel: libgdn_fused_gaudi2.so did not load",
    "INFO 10:00:02 [loader] Loaded 640 shards",
    "  WARNING 10:00:03 [gdn] GDN core decode (FUSED) is off for model.layers.3: head_dim 256 not supported",
    "INFO 10:00:04 [api_server] Started server process",
    "  WARNING 10:00:05 [gdn] GDN core verify (VLLM_HPU_GDN_CORE_VERIFY=1) is off for model.layers.3:"
    " the fused verify kernel is unavailable",
])


def test_fused_gdn_bench_timeout_keeps_kernel_on(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_BENCH_TIMEOUT_S", 1)
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_BENCH_SLEEP=4)
    status, details = gates.fused_gdn(plugin, tmp_path / "out", python=fake)
    assert status == GateStatus.PASSED
    assert details["mode"] == 1 and details["verify_mode"] == 1
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert details["bench"]["timed_out"] is True


def test_fused_gdn_modes_per_step(tmp_path, monkeypatch):
    monkeypatch.delenv("PT_HPU_LAZY_MODE", raising=False)
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    recs = records(log)
    assert len(recs) == 10
    for idx in (4, 5):  # the two eager replays: replay args, no verify, eager mode
        assert "replay" in recs[idx]["args"] and "--ops verify" not in recs[idx]["args"]
        assert recs[idx]["LAZY"] == "0"
    for idx in (1, 2, 3, 6, 7, 8, 9):  # every other step runs in the servers' lazy mode
        assert recs[idx]["LAZY"] == "1"


def test_fused_gdn_eager_replay_failure_keeps_kernel_on(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_EAGER_REPLAY_EXIT=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED  # an eager failure never changes the gate's status
    assert details["mode"] == 1
    assert details["verify_mode"] == 1
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert details["replay_eager"][0]["exit"] == 1
    assert details["eager_compile_ok"] is False
    assert details["verify"]["status"] == "passed"  # the verify checks still ran
    assert any("test_hpu_verify_op.py" in rec["args"] for rec in records(log))
    assert any("bench.py" in rec["args"] for rec in records(log))


def test_fused_gdn_eager_replay_timeout_keeps_kernel_on(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_REPLAY_TIMEOUT_S", 1)
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_EAGER_REPLAY_SLEEP=4)
    status, details = gates.fused_gdn(plugin, tmp_path / "out", python=fake)
    assert status is GateStatus.PASSED  # an eager timeout never changes the gate's status
    assert details["mode"] == 1
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert [entry["timed_out"] for entry in details["replay_eager"]] == [True, True]
    assert all(entry["exit"] is None for entry in details["replay_eager"])
    assert details["eager_compile_ok"] is False
    assert any("bench.py" in rec["args"] for rec in records(log))  # the gate went on to the verify checks


def test_fused_gdn_hung_step_group_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_OP_TIMEOUT_S", 1)
    monkeypatch.setattr(gates, "STEP_KILL_WAIT_S", 5)
    lib = make_lib(tmp_path)
    pidfile = tmp_path / "hung.pid"
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_HANG_PIDFILE=str(pidfile))
    t0 = time.monotonic()
    status, details = gates.fused_gdn(plugin, tmp_path / "out", python=fake)
    elapsed = time.monotonic() - t0
    assert status is GateStatus.TIMED_OUT
    assert details["mode"] == 0
    assert elapsed < 15  # the grandchild holding the pipe cannot make the gate wait for its sleep 60
    assert_pid_gone(int(pidfile.read_text(encoding="utf-8").strip()))


def test_run_step_kills_leftovers_after_exit(tmp_path):
    t0 = time.monotonic()
    proc = gates.run_step(["/bin/sh", "-c", "sleep 60 >/dev/null 2>&1 & echo $!"], tmp_path, 10)
    assert time.monotonic() - t0 < 15  # the sleeper redirects the pipes away, so the exit is not a timeout
    assert proc.returncode == 0
    assert_pid_gone(int(proc.stdout.strip()))


def test_gdn_fallback_lines_finds_all_forms():
    lines = gates.gdn_fallback_lines(FALLBACK_LOG)
    assert lines == [
        "WARNING 10:00:01 [gdn] Qwen3NextGatedDeltaNet has no HPU kernel: libgdn_fused_gaudi2.so did not load",
        "WARNING 10:00:03 [gdn] GDN core decode (FUSED) is off for model.layers.3: head_dim 256 not supported",
        "WARNING 10:00:05 [gdn] GDN core verify (VLLM_HPU_GDN_CORE_VERIFY=1) is off for model.layers.3:"
        " the fused verify kernel is unavailable",
    ]


def test_gdn_fallback_lines_nothing():
    assert gates.gdn_fallback_lines("INFO all good\nno kernel here, decode is on\n") == []


# --------------------------------------------------------------------------- the combined kernel database (RUK-3248)


def test_fused_gdn_records_the_combined_database_state(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["combined_kernel_db"] == "not_installed"
    assert "combined_attempt" not in details

    (lib / COMBINED_DB).write_bytes(b"\0")
    log.unlink()
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b-2", python=fake)
    assert status is GateStatus.PASSED
    assert details["combined_kernel_db"] == "active"  # the loader finds it next to the kernel library
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}  # no kill switch for the servers
    recs = records(log)
    assert len(recs) == 10 and all(rec["COMBINED"] == "<unset>" for rec in recs)
    assert "combined_attempt" not in details


def test_fused_gdn_reruns_the_decode_checks_without_the_combined_database(tmp_path, monkeypatch):
    # The adapter has never run on a card: when the op test fails through it, the decode checks run once
    # more with the kill switch, and the kernel stays on when they pass.
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_COMBINED_OP_FAILS=1)
    out = tmp_path / "g1b"
    status, details = gates.fused_gdn(plugin, out, python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1 and details["verify_mode"] == 1
    assert details["combined_kernel_db"] == "disabled_by_g1b"
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1", KILL_SWITCH: "0"}  # for every later server
    assert details["hpu_op"] == {"exit": 0, "passed": 5, "skipped": 0, "failed": 0}  # the attempt that decided
    assert details["failed_tests"] == []
    assert details["replay"] == [{"case": "tp8_b16", "lazy_mode": 1, "exit": 0},
                                 {"case": "tp8_b64", "lazy_mode": 1, "exit": 0}]
    first = details["combined_attempt"]
    assert first["status"] == "failed" and first["reason"] == "test_hpu_op.py failed (exit 1)"
    assert first["hpu_op"] == {"exit": 1, "passed": 0, "skipped": 0, "failed": 1}
    assert first["failed_tests"] == ["FAILED tests/unit_tests/gdn_core_decode/test_hpu_op.py::test_a - synStatus=26"]
    assert first["replay"] is None  # the replays did not run in the first attempt
    assert first["logs"] == ["hpu_op.combined.log"]
    assert "synStatus=26" in (out / "hpu_op.combined.log").read_text(encoding="utf-8")
    assert "5 passed" in (out / "hpu_op.log").read_text(encoding="utf-8")
    assert "synStatus=26" not in (out / "hpu_op.log").read_text(encoding="utf-8")

    recs = records(log)
    assert len(recs) == 11  # the lookup, the failed op test, then the nine steps of a full pass
    assert "test_hpu_op.py" in recs[1]["args"] and recs[1]["COMBINED"] == "<unset>"
    assert "test_hpu_op.py" in recs[2]["args"]
    assert all(rec["COMBINED"] == "0" for rec in recs[2:])  # the rerun and every later step of the gate
    assert sum(1 for rec in recs if "test_hpu_op.py" in rec["args"]) == 2  # once more, not more


def test_fused_gdn_reruns_after_a_replay_failed_through_the_combined_database(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_COMBINED_REPLAY_FAILS=1)
    out = tmp_path / "g1b"
    status, details = gates.fused_gdn(plugin, out, python=fake)
    assert status is GateStatus.PASSED
    assert details["combined_kernel_db"] == "disabled_by_g1b"
    first = details["combined_attempt"]
    assert first["status"] == "failed"
    assert first["reason"] == "gdn_core_decode_replay.py (tp8_b16) failed (exit 1)"
    assert first["hpu_op"] == {"exit": 0, "passed": 5, "skipped": 0, "failed": 0}
    assert first["replay"] == [{"case": "tp8_b16", "lazy_mode": 1, "exit": 1}]
    assert first["logs"] == ["hpu_op.combined.log", "replay.combined.log"]
    assert "mismatch at step 3" in (out / "replay.combined.log").read_text(encoding="utf-8")
    assert "mismatch" not in (out / "replay.log").read_text(encoding="utf-8")
    assert details["replay"] == [{"case": "tp8_b16", "lazy_mode": 1, "exit": 0},
                                 {"case": "tp8_b64", "lazy_mode": 1, "exit": 0}]
    recs = records(log)
    assert [rec["COMBINED"] for rec in recs[:3]] == ["<unset>", "<unset>", "<unset>"]  # lookup, op, replay 1
    assert all(rec["COMBINED"] == "0" for rec in recs[3:])
    assert len(recs) == 12


def test_fused_gdn_rerun_after_a_timeout_through_the_combined_database(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_OP_TIMEOUT_S", 2)
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_COMBINED_OP_SLEEP=6)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert details["combined_kernel_db"] == "disabled_by_g1b"
    assert details["combined_attempt"]["status"] == "timed_out"
    assert "test_hpu_op.py did not finish in 2 s" in details["combined_attempt"]["reason"]


def test_fused_gdn_failing_with_and_without_the_combined_database_turns_the_kernel_off(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, pytest_out="F\n1 failed in 0.20s", pytest_exit=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED  # as before the fallback existed
    assert details["mode"] == 0 and details["verify_mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert details["reason"] == "test_hpu_op.py failed (exit 1)"
    assert details["verify"] == {"status": "not_run", "reason": "decode checks failed", "hpu_op": None, "replay": []}
    assert details["combined_kernel_db"] == "failed_with_and_without"
    assert details["combined_attempt"]["reason"] == "test_hpu_op.py failed (exit 1)"
    recs = records(log)
    assert len(recs) == 3  # the lookup and the op test twice: exactly one rerun
    assert [rec["COMBINED"] for rec in recs] == ["<unset>", "<unset>", "0"]


def test_fused_gdn_does_not_rerun_when_the_operator_switched_the_database_off(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, pytest_out="F\n1 failed in 0.20s", pytest_exit=1)
    monkeypatch.setenv(KILL_SWITCH, "0")
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED
    assert details["combined_kernel_db"] == "switched_off"
    assert "combined_attempt" not in details
    assert len(records(log)) == 2  # the database was never in the steps: nothing to retry without


def test_fused_gdn_does_not_rerun_without_the_combined_database(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_COMBINED_OP_FAILS=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED
    assert details["combined_kernel_db"] == "not_installed"
    assert "combined_attempt" not in details
    assert len(records(log)) == 2


# --------------------------------------------------------------------------- --quick-gates, --no-verify-probe


def test_fused_gdn_default_skips_nothing(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.PASSED
    assert details["skipped_steps"] == {}
    assert gates.GDN_RECORDED_ONLY_STEPS == ("replay_eager", "bench", "verify_crash_probe")


def test_fused_gdn_quick_skips_the_recorded_only_steps(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    out = tmp_path / "g1b"
    status, details = gates.fused_gdn(plugin, out, python=fake, skip_recorded_only="--quick-gates")
    assert status is GateStatus.PASSED
    assert details["mode"] == 1 and details["verify_mode"] == 1  # the deciding checks all ran
    assert details["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert details["skipped_steps"] == {"replay_eager": "--quick-gates", "bench": "--quick-gates",
                                        "verify_crash_probe": "--quick-gates"}
    assert details["replay_eager"] == [] and details["eager_compile_ok"] is None
    assert details["bench"] == {"exit": None, "skipped": "--quick-gates", "lines": []}
    assert details["verify"]["status"] == "passed"
    recs = records(log)
    assert len(recs) == 7  # the lookup, the op test, two lazy replays, the verify op test, two verify replays
    assert all(rec["LAZY"] in ("1", "<unset>") for rec in recs)  # no eager replay
    assert not any("bench.py" in rec["args"] for rec in recs)
    assert not (out / "bench.log").exists()


def test_fused_gdn_quick_skips_the_crash_probe(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)

    def no_probe(*args):
        raise AssertionError("the crash probe must not run with --quick-gates")

    monkeypatch.setattr(gates, "_gdn_crash_probe", no_probe)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_VERIFY_CRASH=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake, skip_recorded_only="--quick-gates")
    assert status is GateStatus.PASSED and "error" not in details
    assert details["verify_mode"] == 0 and details["verify"]["status"] == "failed"
    assert details["verify"]["crash_probe"] == {"exit": None, "skipped": "--quick-gates", "summary": []}


def test_fused_gdn_no_verify_probe_skips_only_the_probe(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)

    def no_probe(*args):
        raise AssertionError("the crash probe must not run with --no-verify-probe")

    monkeypatch.setattr(gates, "_gdn_crash_probe", no_probe)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_VERIFY_CRASH=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake, skip_verify_probe="--no-verify-probe")
    assert status is GateStatus.PASSED and "error" not in details
    assert details["skipped_steps"] == {"verify_crash_probe": "--no-verify-probe"}
    assert details["verify"]["crash_probe"] == {"exit": None, "skipped": "--no-verify-probe", "summary": []}
    recs = records(log)
    assert sum(1 for rec in recs if rec["LAZY"] == "0") == 2  # the eager replays still ran
    assert any("bench.py" in rec["args"] for rec in recs)  # and the bench
    assert details["bench"]["exit"] == 0


def test_fused_gdn_no_verify_probe_after_a_verify_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GDN_OP_TIMEOUT_S", 2)
    monkeypatch.setattr(gates, "_gdn_crash_probe", lambda *args: (_ for _ in ()).throw(AssertionError("probe ran")))
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_VERIFY_PYTEST_SLEEP=6)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake, skip_verify_probe="--no-verify-probe")
    assert status is GateStatus.PASSED and "error" not in details
    assert details["verify"]["status"] == "timed_out"
    assert details["verify"]["crash_probe"]["skipped"] == "--no-verify-probe"
