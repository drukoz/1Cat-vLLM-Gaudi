# SPDX-License-Identifier: Apache-2.0
"""Tests for gates.fused_gdn and gates.gdn_fallback_lines (plan M4, RUK-2709)."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from tools.flash_next.rental import gates
from tools.flash_next.rental.schema import GateStatus

OPS_SO = "gdn_fused_ops.cpython-310-x86_64-linux-gnu.so"


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
} >> "$FAKE_PY_LOG"
case "$1" in
  -c) echo "$FAKE_LIB_DIR"
      exit ${FAKE_LOOKUP_EXIT:-0} ;;
  -m) printf '%s\n' "$FAKE_PYTEST_OUT"
      exit ${FAKE_PYTEST_EXIT:-0} ;;
  *replay*)
    if [ -n "$FAKE_REPLAY_SLEEP" ]; then sleep "$FAKE_REPLAY_SLEEP"; fi
    case "$*" in
      *--case*) exit ${FAKE_REPLAY2_EXIT:-0} ;;
      *) exit ${FAKE_REPLAY_EXIT:-0} ;;
    esac ;;
  *bench*)
    printf '%s\n' "core recurrent slope 0.51 ms/step" "core slope 0.42 ms/step" \
"core tp8 B=1 0.31 ms" "plain line"
    exit ${FAKE_BENCH_EXIT:-0} ;;
esac
exit 0
"""


def make_lib(tmp_path: Path, ops: bool = True) -> Path:
    """A fake installed flashinfer_gaudi/lib/ with the fused kernel library (and optionally the op)."""
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    (lib / "libgdn_fused_gaudi2.so").write_bytes(b"\0")
    if ops:
        (lib / OPS_SO).write_bytes(b"\0")
    return lib


def setup(tmp_path: Path, monkeypatch, *, lib: Path, pytest_out: str = "..... [100%]\n5 passed in 12.34s",
          pytest_exit: int = 0, **extra) -> tuple:
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_PY_LOG", str(log))
    monkeypatch.setenv("FAKE_LIB_DIR", str(lib))
    monkeypatch.setenv("FAKE_PYTEST_OUT", pytest_out)
    monkeypatch.setenv("FAKE_PYTEST_EXIT", str(pytest_exit))
    for name, value in extra.items():
        monkeypatch.setenv(name, str(value))
    fake = write_fake_python(tmp_path / "fake_python.sh", FAKE_PY)
    plugin = tmp_path / "plugin"
    (plugin / "tests").mkdir(parents=True)
    return fake, plugin, log


def records(log: Path) -> list:
    """The fake interpreter's invocations, in order, as {args, PWD, KERNEL, NATIVE, PYTHONPATH, GDNMODE}."""
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


def test_fused_gdn_all_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "1")
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    out = tmp_path / "g1b"
    status, details = gates.fused_gdn(plugin, out, python=fake)
    assert status is GateStatus.PASSED
    assert details["mode"] == 1
    assert details["env"] == {}
    assert details["seconds"] >= 0
    assert details["hpu_op"] == {"exit": 0, "passed": 5, "skipped": 0, "failed": 0}
    assert details["failed_tests"] == []
    assert details["replay"] == [{"case": "tp8_b16", "exit": 0}, {"case": "tp8_b64", "exit": 0}]
    assert details["bench"]["exit"] == 0
    assert details["bench"]["lines"] == [
        "core recurrent slope 0.51 ms/step",
        "core slope 0.42 ms/step",
        "core tp8 B=1 0.31 ms",
    ]
    recs = records(log)
    assert len(recs) == 5  # the library lookup, then the four steps, in order
    assert recs[0]["args"].startswith("-c import flashinfer_gaudi")
    assert os.path.realpath(recs[0]["PWD"]) == os.path.realpath(tempfile.gettempdir())
    steps = recs[1:]
    assert "pytest" in steps[0]["args"] and "test_hpu_op.py" in steps[0]["args"]
    assert "gdn_core_decode_replay.py" in steps[1]["args"] and "--case" not in steps[1]["args"]
    assert "gdn_core_decode_replay.py" in steps[2]["args"] and "--case tp8_b64" in steps[2]["args"]
    assert "gdn_core_decode_bench.py" in steps[3]["args"]
    for rec in steps:
        assert os.path.realpath(rec["PWD"]) == os.path.realpath(plugin)
        assert rec["KERNEL"] == str(lib / "libgdn_fused_gaudi2.so")
        assert rec["NATIVE"] == str(lib / OPS_SO)
        assert rec["PYTHONPATH"] == str(plugin)
        assert rec["GDNMODE"] == "<unset>"  # the gate always removes it for the steps
    for name in ("hpu_op.log", "replay.log", "bench.log"):
        assert (out / name).is_file()
    assert "5 passed" in (out / "hpu_op.log").read_text(encoding="utf-8")


def test_fused_gdn_missing_library_skips(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, ops=False)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.SKIPPED
    assert details["mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0"}
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
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0"}
    assert details["reason"] == "test_hpu_op.py skipped"
    assert details["hpu_op"] == {"exit": 0, "passed": 0, "skipped": 1, "failed": 0}
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
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0"}
    assert details["failed_tests"] == pytest_out.splitlines()[:2]
    assert details["hpu_op"] == {"exit": 1, "passed": 0, "skipped": 0, "failed": 2}
    assert "test_hpu_op.py" in details["reason"] and "exit 1" in details["reason"]


def test_fused_gdn_second_replay_fails(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, FAKE_REPLAY2_EXIT=1)
    status, details = gates.fused_gdn(plugin, tmp_path / "g1b", python=fake)
    assert status is GateStatus.FAILED
    assert details["mode"] == 0
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0"}
    assert details["replay"] == [{"case": "tp8_b16", "exit": 0}, {"case": "tp8_b64", "exit": 1}]
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
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0"}
    assert "replay" in details["reason"]
    assert "bench" not in details
    assert not any("bench.py" in rec["args"] for rec in records(log))


FALLBACK_LOG = "\n".join([
    "INFO 10:00:00 [config] Setting attention block size to 128 tokens",
    "WARNING 10:00:01 [gdn] Qwen3NextGatedDeltaNet has no HPU kernel: libgdn_fused_gaudi2.so did not load",
    "INFO 10:00:02 [loader] Loaded 640 shards",
    "  WARNING 10:00:03 [gdn] GDN core decode (FUSED) is off for model.layers.3: head_dim 256 not supported",
    "INFO 10:00:04 [api_server] Started server process",
])


def test_gdn_fallback_lines_finds_both_forms():
    lines = gates.gdn_fallback_lines(FALLBACK_LOG)
    assert lines == [
        "WARNING 10:00:01 [gdn] Qwen3NextGatedDeltaNet has no HPU kernel: libgdn_fused_gaudi2.so did not load",
        "WARNING 10:00:03 [gdn] GDN core decode (FUSED) is off for model.layers.3: head_dim 256 not supported",
    ]


def test_gdn_fallback_lines_nothing():
    assert gates.gdn_fallback_lines("INFO all good\nno kernel here, decode is on\n") == []
