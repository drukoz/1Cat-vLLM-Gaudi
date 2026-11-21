# SPDX-License-Identifier: Apache-2.0
"""Tests for gates.hc_fused: the fused hyper-connection kernels' card test inside gate G1b.

The step runs tests/unit_tests/hc_fused/test_hpu_op.py with the installed libraries by path, in lazy
mode, in its own process group with its own log and timeout. Its result is recorded only: it has no
gate status of its own and nothing here can turn the fused GDN kernel off.
"""

from __future__ import annotations

import inspect

from tools.flash_next.rental import gates

from .test_gates_fused_gdn import assert_pid_gone, records, write_fake_python

HC_KERNEL = "libhc_fused_gaudi2.so"
HC_OPS = "hc_fused_ops.cpython-310-x86_64-linux-gnu.so"
GDN_KERNEL = "libgdn_fused_gaudi2.so"
COMBINED_DB = "libflashinfer_gaudi_combined_gaudi2.so"
KILL_SWITCH = "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"
PASS_OUT = "." * 90 + " [100%]\n90 passed in 41.20s"

FAKE_PY = r"""
{
  echo "ARGS: $*"
  echo "PWD: $(pwd)"
  echo "KERNEL: ${HC_FUSED_GAUDI_KERNEL_LIBRARY-<unset>}"
  echo "NATIVE: ${HC_FUSED_GAUDI_NATIVE_LIBRARY-<unset>}"
  echo "GDNKERNEL: ${GDN_FUSED_GAUDI_KERNEL_LIBRARY-<unset>}"
  echo "PYTHONPATH: ${PYTHONPATH-<unset>}"
  echo "LAZY: ${PT_HPU_LAZY_MODE-<unset>}"
  echo "COMBINED: ${FLASHINFER_GAUDI_COMBINED_KERNEL_DB-<unset>}"
  echo "HCFLAG: ${VLLM_HPU_HC_FUSED-<unset>}"
  echo "RECIPE: ${PT_HPU_RECIPE_CACHE_CONFIG-<unset>}"
  echo "ACCPAR: ${PT_HPU_LAZY_ACC_PAR_MODE-<unset>}"
} >> "$FAKE_PY_LOG"
case "$1" in
  -c) echo "INFO some import noise"
      echo "$FAKE_LIB_DIR"
      exit ${FAKE_LOOKUP_EXIT:-0} ;;
  -m) if [ -n "$FAKE_OP_PRINT" ]; then
        echo "$FAKE_OP_PRINT"
        sleep 60
      fi
      if [ -n "$FAKE_HANG_PIDFILE" ]; then
        sleep 60 &
        echo "$!" > "$FAKE_HANG_PIDFILE"
        sleep 60
      fi
      printf '%s\n' "$FAKE_PYTEST_OUT"
      exit ${FAKE_PYTEST_EXIT:-0} ;;
esac
exit 0
"""


def make_lib(tmp_path, *, kernel=True, ops=True, gdn=True, combined=False):
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    if kernel:
        (lib / HC_KERNEL).write_bytes(b"\0")
    if ops:
        (lib / HC_OPS).write_bytes(b"\0")
    if gdn:
        (lib / GDN_KERNEL).write_bytes(b"\0")
    if combined:
        (lib / COMBINED_DB).write_bytes(b"\0")
    return lib


def setup(tmp_path, monkeypatch, *, lib, pytest_out=PASS_OUT, pytest_exit=0, test_file=True, **extra):
    for name in (KILL_SWITCH, "VLLM_HPU_HC_FUSED", "PT_HPU_RECIPE_CACHE_CONFIG", "PT_HPU_LAZY_ACC_PAR_MODE"):
        monkeypatch.delenv(name, raising=False)
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
    if test_file:
        op_test = plugin / gates.HC_OP_TEST
        op_test.parent.mkdir(parents=True)
        op_test.write_text("# fake hc op test\n", encoding="utf-8")
    return fake, plugin, log


def test_names_and_limits():
    assert gates.HC_OP_TEST == "tests/unit_tests/hc_fused/test_hpu_op.py"
    assert gates.HC_OP_TIMEOUT_S == 600.0
    assert gates.HC_OP_LOG == "hc_hpu_op.log"
    assert gates.HC_FUSED_ENV == "VLLM_HPU_HC_FUSED"
    assert gates.HC_KERNEL_LIB == HC_KERNEL


def test_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0")  # the operator's or the default: not what the test runs with
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0")
    out = tmp_path / "g1b"

    result = gates.hc_fused(plugin, out, python=fake)

    assert result["status"] == "passed"
    assert result["reason"] == ""
    assert result["hpu_op"] == {"exit": 0, "passed": 90, "skipped": 0, "failed": 0, "error": 0, "timed_out": False}
    assert result["failed_tests"] == []
    assert result["log"] == "hc_hpu_op.log"
    assert result["seconds"] >= 0
    assert "90 passed" in (out / "hc_hpu_op.log").read_text(encoding="utf-8")
    assert not (out / "hpu_op.log").exists()  # its own log: G1b's GDN logs are not touched

    lookup, step = records(log)
    assert lookup["args"].startswith("-c ")
    assert step["args"] == "-m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op.py"
    assert step["PWD"] == str(plugin.resolve())
    assert step["PYTHONPATH"] == str(plugin)
    assert step["KERNEL"] == str(lib / HC_KERNEL)  # the installed libraries, by path
    assert step["NATIVE"] == str(lib / HC_OPS)
    assert step["GDNKERNEL"] == str(lib / GDN_KERNEL)  # where the loader looks for the combined database
    assert step["LAZY"] == "1"  # the bridge mode of every server
    assert step["HCFLAG"] == "<unset>"
    assert step["COMBINED"] == "<unset>"


def test_the_step_never_runs_next_to_a_recipe_cache(tmp_path, monkeypatch):
    """The gate's process has no recipe cache variable; one inherited from the container would block the
    colon-list registration and make the card test skip, which reads as a broken kernel."""
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/recipe-cache,false,8192")

    assert gates.hc_fused(plugin, tmp_path / "g1b", python=fake)["status"] == "passed"
    assert records(log)[1]["RECIPE"] == "<unset>"


def test_the_step_runs_with_the_accumulation_thread_off(tmp_path, monkeypatch):
    """A custom op is not safe next to the lazy bridge's accumulation thread (the card test of 2026-11-20
    failed 6 and 7 of 90 at PT_HPU_LAZY_ACC_PAR_MODE=1 and none at 0), and the loader refuses the kernels in a
    lazy process without the variable at 0: the step sets it, as the plugin does for a server."""
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "passed"
    assert records(log)[1]["ACCPAR"] == "0"
    assert result["lazy_acc_par_mode"] == "0"


def test_the_operators_accumulation_mode_reaches_the_step(tmp_path, monkeypatch):
    """A control run at mode 1 is the operator's to ask for; the servers of that run inherit the same value."""
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv("PT_HPU_LAZY_ACC_PAR_MODE", "1")

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert records(log)[1]["ACCPAR"] == "1"
    assert result["lazy_acc_par_mode"] == "1"


def test_a_skip_inside_the_test_file_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib,
                            pytest_out="SKIPPED [1] test_hpu_op.py:25: hc_fused_ops extension is not built: "
                            "x.so: OSError: undefined symbol\n1 skipped in 0.31s")

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed"
    assert result["reason"] == "test_hpu_op.py skipped"
    assert result["hpu_op"]["skipped"] == 1 and result["hpu_op"]["passed"] == 0


def test_failures_are_recorded(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    out_text = ("FAILED tests/unit_tests/hc_fused/test_hpu_op.py::test_hpu_op_matches_reference[norm_real_r1] - x\n"
                "FAILED tests/unit_tests/hc_fused/test_hpu_op.py::test_hpu_chain_of_one_hyper_connection - y\n"
                "2 failed, 88 passed in 50.00s")
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out=out_text, pytest_exit=1)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed"
    assert result["reason"] == "test_hpu_op.py failed"
    assert result["hpu_op"] == {"exit": 1, "passed": 88, "skipped": 0, "failed": 2, "error": 0, "timed_out": False}
    assert len(result["failed_tests"]) == 2 and result["failed_tests"][0].startswith("FAILED ")


# What `pytest -q -rs` printed on the card on 2026-11-20 (cut to two of the six failures, lines unchanged): no
# "FAILED ..." line anywhere, and a bridge banner in the captured stderr that looks like a pytest section header.
CARD_FAILURE_OUT = r"""
.F...................................................................... [ 80%]
..........F.FFF..F                                                       [100%]
=================================== FAILURES ===================================
_________________ test_hpu_op_matches_reference[norm_tiny_r1] __________________

case = HcCase(name='norm_tiny_r1', kind='norm', hidden=128, rows=1, hc=4, lora=16, pad=12, shared_weight=False, flavor='random', seed=1, eps=1e-06, exact=False)

>       got = {name: tensor.cpu() for name, tensor in _run(case, hpu).items()}

tests/unit_tests/hc_fused/test_hpu_op.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

>   got = {name: tensor.cpu() for name, tensor in _run(case, hpu).items()}
E   RuntimeError: Empty tensor optional Uniqueid: 3 Strided Params Has Value: 0 Failing Graph: IR {
E      %id_24_prim__constant Unique ID:0 = n6_prim::constant{} , value=4
E      %id_22_prim__constant Unique ID:0 = n5_prim::constant{} , value=9.9999999999999995e-07
E      %id_12_hpu__input Unique ID:2 = n3_hpu::input{} 
E      %id_20_hpu__input Unique ID:3 = n4_hpu::input{} 
E      %id_17_flashinfer_gaudi__hc_norm Unique ID:4 = n7_flashinfer_gaudi::hc_norm{id_20_hpu__input Uniqueid: 3 id_12_hpu__input Uniqueid: 2 id_22_prim__constant id_24_prim__constant } , ROOT=0
E      %id_27_prim__constant Unique ID:0 = n8_prim::constant{} , value=0
E      %id_5_hpu__input Unique ID:1 = n1_hpu::input{} 
E      %id_20_aten__clone Unique ID:3 = n9_aten::clone{id_5_hpu__input Uniqueid: 1 id_27_prim__constant } , ROOT=1
E   }

tests/unit_tests/hc_fused/test_hpu_op.py:47: RuntimeError
----------------------------- Captured stderr call -----------------------------
============================= HPU PT BRIDGE CONFIGURATION ON RANK = 0 ============= 
 PT_HPU_LAZY_MODE = 1
 PT_HPU_LAZY_ACC_PAR_MODE = 1
---------------------------: System Configuration :---------------------------
Num CPU Cores : 152
------------------------------------------------------------------------------
____________________ test_hpu_chain_of_one_hyper_connection ____________________

    def test_hpu_chain_of_one_hyper_connection():
>       fused_hpu = chain("hpu", True)

tests/unit_tests/hc_fused/test_hpu_op.py:127: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/unit_tests/hc_fused/test_hpu_op.py:124: in chain
    return [t.cpu().float() for t in (out, block_input, inj_out, out2)]
E   AssertionError: something else

tests/unit_tests/hc_fused/test_hpu_op.py:124: AssertionError
=============================== warnings summary ===============================
../../usr/local/lib/python3.10/dist-packages/torch/jit/_script.py:365: 14 warnings

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 failed, 88 passed, 14 warnings in 120.42s (0:02:00)"""


def test_failed_tests_are_named_from_the_failure_headers(tmp_path, monkeypatch):
    """`pytest -q -rs` prints no "FAILED ..." line, so the card run of 2026-11-20 recorded `failed_tests: []`
    next to `failed: 6`. The names are in the headers of the FAILURES section."""
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out=CARD_FAILURE_OUT, pytest_exit=1)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["hpu_op"]["failed"] == 2
    assert result["failed_tests"] == ["FAILED test_hpu_op_matches_reference[norm_tiny_r1]",
                                      "FAILED test_hpu_chain_of_one_hyper_connection"]


def test_failed_test_names_from_either_form_of_pytest_output():
    """The helper the GDN op tests of G1b use too."""
    summary_lines = "x\nFAILED tests/a.py::test_one - boom\n____ test_one ____\n1 failed in 1s"
    assert gates._pytest_failed_tests(summary_lines) == ["FAILED tests/a.py::test_one - boom"]

    headers_only = ("____ not_a_failure_header_before_the_section ____\n"
                    "==================== ERRORS ====================\n"
                    "____ ERROR at setup of test_two ____\n"
                    "=================== FAILURES ===================\n"
                    "____ test_one[a b] ____\n"
                    "_ _ _ _ _ _ _ _\n"
                    "____ test_one[a b] ____\n"
                    "1 failed, 1 error in 1s")
    assert gates._pytest_failed_tests(headers_only) == ["FAILED ERROR at setup of test_two", "FAILED test_one[a b]"]
    assert gates._pytest_failed_tests("90 passed in 41.20s") == []


def test_the_failing_graphs_of_the_lazy_bridge_are_classified(tmp_path, monkeypatch):
    """One of the two failures is the bridge's empty graph input with its producer in the same graph."""
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out=CARD_FAILURE_OUT, pytest_exit=1)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["empty_inputs"] == {"dumps": 1, "queued_producer": 1, "ops": ["flashinfer_gaudi::hc_norm"]}


def test_a_pass_has_no_failing_graph(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["empty_inputs"] == {"dumps": 0, "queued_producer": 0, "ops": []}


def test_a_crash_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out="Fatal Python error: Aborted", pytest_exit=134)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed" and result["reason"] == "test_hpu_op.py failed"
    assert result["hpu_op"]["exit"] == 134 and result["hpu_op"]["passed"] == 0


def test_no_test_ran_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out="no tests ran in 0.01s", pytest_exit=0)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed" and result["reason"] == "test_hpu_op.py failed"


def test_timeout_keeps_the_partial_output_and_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_OP_PRINT="HCHPU norm_real_r1 xn: started")
    monkeypatch.setattr(gates, "HC_OP_TIMEOUT_S", 1.0)
    out = tmp_path / "g1b"

    result = gates.hc_fused(plugin, out, python=fake)

    assert result["status"] == "failed"
    assert result["reason"] == "test_hpu_op.py did not finish in 1.0 s"
    assert result["hpu_op"]["timed_out"] is True and result["hpu_op"]["exit"] is None
    assert "HCHPU norm_real_r1 xn: started" in (out / "hc_hpu_op.log").read_text(encoding="utf-8")


def test_a_hung_step_is_killed_with_its_children(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    pidfile = tmp_path / "hang.pid"
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_HANG_PIDFILE=pidfile)
    monkeypatch.setattr(gates, "HC_OP_TIMEOUT_S", 1.0)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed" and result["hpu_op"]["timed_out"] is True
    assert_pid_gone(int(pidfile.read_text(encoding="utf-8").strip()))  # its own process group: nothing holds a card


def test_missing_kernel_library_is_skipped(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, kernel=False)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "skipped"
    assert result["reason"] == ("the fused hyper-connection library is not installed: libhc_fused_gaudi2.so "
                                "missing in %s" % lib)
    assert result["hpu_op"] is None
    assert len(records(log)) == 1  # the lookup only: no card test


def test_missing_op_extension_is_skipped(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, ops=False)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "skipped"
    assert "0 hc_fused_ops*.so files in %s, expected 1" % lib in result["reason"]


def test_failed_lookup_is_skipped(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_LOOKUP_EXIT=3)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "skipped"
    assert result["reason"].startswith("the fused hyper-connection library is not installed: the interpreter exited 3")


def test_test_file_missing_from_the_bundle_is_not_run(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, test_file=False)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result == {"status": "not_run", "reason": "tests/unit_tests/hc_fused/test_hpu_op.py is not in the bundle",
                      "hpu_op": None}
    assert records(log) == []


def test_a_skip_reason_runs_nothing(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    out = tmp_path / "g1b"

    result = gates.hc_fused(plugin, out, python=fake, skip="--quick-gates without --hc-fused")

    assert result == {"status": "skipped", "reason": "--quick-gates without --hc-fused", "hpu_op": None}
    assert records(log) == []
    assert not (out / "hc_hpu_op.log").exists()


def test_the_step_runs_through_the_combined_database_by_default(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["combined_kernel_db"] == "active"
    assert records(log)[1]["COMBINED"] == "<unset>"


def test_the_step_runs_without_the_combined_database_when_the_servers_will(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake, combined_db_off=True)

    assert result["status"] == "passed"
    assert result["combined_kernel_db"] == "switched_off"
    assert records(log)[1]["COMBINED"] == "0"


def test_the_operators_switch_reaches_the_step(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv(KILL_SWITCH, "0")

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["combined_kernel_db"] == "switched_off"
    assert records(log)[1]["COMBINED"] == "0"


def test_no_combined_database_installed(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    assert gates.hc_fused(plugin, tmp_path / "g1b", python=fake)["combined_kernel_db"] == "not_installed"


def test_without_the_gdn_library_the_step_still_runs(tmp_path, monkeypatch):
    """The two kernels are independent builds: a missing fused GDN library does not stop this test."""
    lib = make_lib(tmp_path, gdn=False)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "passed"
    assert records(log)[1]["GDNKERNEL"] == "<unset>"


def test_it_never_raises(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    _, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.hc_fused(plugin, tmp_path / "g1b", python=str(tmp_path / "no-such-python"))

    assert result["status"] == "not_run"
    assert "FileNotFoundError" in result["reason"]
    assert result["hpu_op"] is None


def test_it_has_no_way_to_change_the_gdn_gate():
    """A dict comes back, not a (status, details) pair, and the function takes nothing of G1b's decision:
    there is no gate status and no GDN mode it could change."""
    assert list(inspect.signature(gates.hc_fused).parameters) == ["plugin_root", "out_dir", "python", "skip",
                                                                  "combined_db_off"]
    source = inspect.getsource(gates.hc_fused)
    assert "GateStatus." not in source and "GDN_MODE" not in source and "GDN_VERIFY" not in source


HC_LOG = """\
INFO 10:00:00 [serve] starting
(Worker_TP0 pid=11) WARNING 10:00:01 [__init__.py:140] HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel: the \
hc_fused_ops extension did not load (x.so: file does not exist); every hyper-connection runs the torch path.
(Worker_TP3 pid=14) WARNING 10:00:03 [hyperconnection.py:260] HC fused (VLLM_HPU_HC_FUSED=1) is off for \
language_model.model.layers.3.attn_hyper_connection: hidden states are 3-D, the fused ops take [tokens, width]; \
using the torch path
WARNING 10:00:04 [gdn] GDN core decode (VLLM_HPU_GDN_CORE_DECODE=1) is off for model.layers.3: head dims differ
INFO 10:00:05 HC fused is a hyper-connection thing and this line says nothing is off
"""


def test_fallback_lines_catch_the_hc_lines():
    lines = gates.gdn_fallback_lines(HC_LOG)
    assert len(lines) == 3
    assert "has no HPU kernel" in lines[0] and "HC fused" in lines[0]
    assert "is off for language_model.model.layers.3.attn_hyper_connection" in lines[1]
    assert "GDN core decode" in lines[2]


def test_hc_fallback_lines_are_the_hc_ones_only():
    lines = gates.hc_fallback_lines(HC_LOG)
    assert len(lines) == 2 and all("HC fused" in line for line in lines)
    assert gates.hc_fallback_lines("INFO HC fused: on\nWARNING GDN core decode (1) is off for x: y\n") == []
