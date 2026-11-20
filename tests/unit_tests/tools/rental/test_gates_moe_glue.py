# SPDX-License-Identifier: Apache-2.0
"""Tests for gates.moe_glue: the MoE glue kernels' card test inside gate G1b.

The step runs tests/unit_tests/hc_fused/test_moe_hpu_op.py with the installed libraries by path (the
kernel library and the op extension of the fused hyper-connection kernels: the MoE glue kernels live in
the same two files), in lazy mode, in its own process group with its own log and timeout. Its result is
recorded only: it has no gate status of its own and nothing here can turn the fused GDN kernel off.
The last tests are about the server log: the MoE glue's two fallback lines are found, and a
hyper-connection line is never counted as a MoE glue line or the other way round.
"""

from __future__ import annotations

import inspect

from tools.flash_next.rental import gates

from .test_gates_fused_gdn import assert_pid_gone, records, write_fake_python
from .test_gates_hc_fused import COMBINED_DB, GDN_KERNEL, HC_KERNEL, HC_LOG, HC_OPS, KILL_SWITCH, make_lib  # noqa: F401

MOE_TEST = "tests/unit_tests/hc_fused/test_moe_hpu_op.py"
PASS_OUT = "." * 84 + " [100%]\n84 passed in 61.70s"

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
  echo "MOEFLAG: ${VLLM_HPU_MOE_GLUE-<unset>}"
  echo "RECIPE: ${PT_HPU_RECIPE_CACHE_CONFIG-<unset>}"
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


def setup(tmp_path, monkeypatch, *, lib, pytest_out=PASS_OUT, pytest_exit=0, test_file=True, hc_test_file=False,
          **extra):
    for name in (KILL_SWITCH, "VLLM_HPU_HC_FUSED", "VLLM_HPU_MOE_GLUE", "PT_HPU_RECIPE_CACHE_CONFIG"):
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
    (plugin / "tests" / "unit_tests" / "hc_fused").mkdir(parents=True)
    if test_file:
        (plugin / gates.MOE_OP_TEST).write_text("# fake moe glue op test\n", encoding="utf-8")
    if hc_test_file:
        (plugin / gates.HC_OP_TEST).write_text("# fake hc op test\n", encoding="utf-8")
    return fake, plugin, log


def test_names_and_limits():
    assert gates.MOE_GLUE_ENV == "VLLM_HPU_MOE_GLUE"
    assert gates.MOE_OP_TEST == MOE_TEST
    assert gates.MOE_OP_LOG == "moe_hpu_op.log"
    assert gates.MOE_OP_TIMEOUT_S == 900.0  # about 90 cases, two of them large
    # the hyper-connection kernels' own names stay what they were
    assert gates.HC_OP_TEST == "tests/unit_tests/hc_fused/test_hpu_op.py"
    assert gates.HC_OP_LOG == "hc_hpu_op.log" and gates.HC_OP_TIMEOUT_S == 600.0
    assert gates.HC_FUSED_ENV == "VLLM_HPU_HC_FUSED"


def test_pass(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE", "1")  # the operator's shell: not what the test runs with
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    out = tmp_path / "g1b"

    result = gates.moe_glue(plugin, out, python=fake)

    assert result["status"] == "passed"
    assert result["reason"] == ""
    assert result["hpu_op"] == {"exit": 0, "passed": 84, "skipped": 0, "failed": 0, "error": 0, "timed_out": False}
    assert result["failed_tests"] == []
    assert result["log"] == "moe_hpu_op.log"
    assert result["seconds"] >= 0
    assert "84 passed" in (out / "moe_hpu_op.log").read_text(encoding="utf-8")
    # its own log, next to the hyper-connection test's: neither that one nor G1b's GDN logs are touched
    assert not (out / "hc_hpu_op.log").exists() and not (out / "hpu_op.log").exists()

    lookup, step = records(log)
    assert lookup["args"].startswith("-c ")
    assert step["args"] == "-m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_moe_hpu_op.py"
    assert step["PWD"] == str(plugin.resolve())
    assert step["PYTHONPATH"] == str(plugin)
    assert step["KERNEL"] == str(lib / HC_KERNEL)  # the hyper-connection library holds the MoE glue kernels
    assert step["NATIVE"] == str(lib / HC_OPS)
    assert step["GDNKERNEL"] == str(lib / GDN_KERNEL)  # where the loader looks for the combined database
    assert step["LAZY"] == "1"  # the bridge mode of every server
    assert step["MOEFLAG"] == "<unset>"  # the test loads the kernels itself
    assert step["HCFLAG"] == "<unset>"
    assert step["COMBINED"] == "<unset>"


def test_the_result_has_the_keys_of_the_hyper_connection_test(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, hc_test_file=True)

    moe = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)
    hc = gates.hc_fused(plugin, tmp_path / "g1b", python=fake)

    assert sorted(moe) == sorted(hc) == ["combined_kernel_db", "failed_tests", "hpu_op", "log", "reason", "seconds",
                                         "status"]
    assert sorted(moe["hpu_op"]) == sorted(hc["hpu_op"])
    assert (moe["status"], hc["status"]) == ("passed", "passed")
    assert (moe["log"], hc["log"]) == ("moe_hpu_op.log", "hc_hpu_op.log")  # two logs in one folder


def test_the_step_never_runs_next_to_a_recipe_cache(tmp_path, monkeypatch):
    """The gate's process has no recipe cache variable; one inherited from the container would block the
    colon-list registration and make the card test skip, which reads as a broken kernel."""
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv("PT_HPU_RECIPE_CACHE_CONFIG", "/recipe-cache,false,8192")

    assert gates.moe_glue(plugin, tmp_path / "g1b", python=fake)["status"] == "passed"
    assert records(log)[1]["RECIPE"] == "<unset>"


def test_a_skip_inside_the_test_file_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib,
                            pytest_out="SKIPPED [1] test_moe_hpu_op.py:43: the MoE glue kernels are not usable: "
                            "x.so: OSError: undefined symbol\n1 skipped in 0.31s")

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed"
    assert result["reason"] == "test_moe_hpu_op.py skipped"
    assert result["hpu_op"]["skipped"] == 1 and result["hpu_op"]["passed"] == 0


def test_failures_are_recorded(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    out_text = ("FAILED tests/unit_tests/hc_fused/test_moe_hpu_op.py::test_hpu_op_matches_reference[route_r4] - x\n"
                "FAILED tests/unit_tests/hc_fused/test_moe_hpu_op.py::test_hpu_block_chain - y\n"
                "2 failed, 82 passed in 70.00s")
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out=out_text, pytest_exit=1)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed"
    assert result["reason"] == "test_moe_hpu_op.py failed"
    assert result["hpu_op"] == {"exit": 1, "passed": 82, "skipped": 0, "failed": 2, "error": 0, "timed_out": False}
    assert len(result["failed_tests"]) == 2 and result["failed_tests"][0].startswith("FAILED ")


def test_a_crash_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out="Fatal Python error: Aborted", pytest_exit=134)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed" and result["reason"] == "test_moe_hpu_op.py failed"
    assert result["hpu_op"]["exit"] == 134 and result["hpu_op"]["passed"] == 0


def test_no_test_ran_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, pytest_out="no tests ran in 0.01s", pytest_exit=0)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed" and result["reason"] == "test_moe_hpu_op.py failed"


def test_timeout_keeps_the_partial_output_and_is_a_failure(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_OP_PRINT="MOEHPU route_r4 weights: started")
    monkeypatch.setattr(gates, "MOE_OP_TIMEOUT_S", 1.0)
    out = tmp_path / "g1b"

    result = gates.moe_glue(plugin, out, python=fake)

    assert result["status"] == "failed"
    assert result["reason"] == "test_moe_hpu_op.py did not finish in 1.0 s"
    assert result["hpu_op"]["timed_out"] is True and result["hpu_op"]["exit"] is None
    assert "MOEHPU route_r4 weights: started" in (out / "moe_hpu_op.log").read_text(encoding="utf-8")


def test_each_card_test_has_its_own_limit(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, hc_test_file=True)
    limits = {}
    real_run_step = gates.run_step

    def run_step(cmd, cwd, timeout_s, env=None):
        if "pytest" in cmd:
            limits[cmd[-1]] = timeout_s
        return real_run_step(cmd, cwd, timeout_s, env)

    monkeypatch.setattr(gates, "run_step", run_step)
    assert gates.moe_glue(plugin, tmp_path / "g1b", python=fake)["status"] == "passed"
    assert gates.hc_fused(plugin, tmp_path / "g1b", python=fake)["status"] == "passed"

    assert limits == {MOE_TEST: 900.0, "tests/unit_tests/hc_fused/test_hpu_op.py": 600.0}


def test_a_hung_step_is_killed_with_its_children(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    pidfile = tmp_path / "hang.pid"
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_HANG_PIDFILE=pidfile)
    monkeypatch.setattr(gates, "MOE_OP_TIMEOUT_S", 1.0)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "failed" and result["hpu_op"]["timed_out"] is True
    assert_pid_gone(int(pidfile.read_text(encoding="utf-8").strip()))  # its own process group: nothing holds a card


def test_missing_kernel_library_is_skipped(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, kernel=False)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "skipped"
    assert result["reason"] == ("the library of the MoE glue kernels (the fused hyper-connection library) is not "
                                "installed: libhc_fused_gaudi2.so missing in %s" % lib)
    assert result["hpu_op"] is None
    assert len(records(log)) == 1  # the lookup only: no card test


def test_missing_op_extension_is_skipped(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, ops=False)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "skipped"
    assert "0 hc_fused_ops*.so files in %s, expected 1" % lib in result["reason"]


def test_failed_lookup_is_skipped(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib, FAKE_LOOKUP_EXIT=3)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "skipped"
    assert result["reason"].startswith("the library of the MoE glue kernels (the fused hyper-connection library) is "
                                       "not installed: the interpreter exited 3")


def test_test_file_missing_from_the_bundle_is_not_run(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    # the hyper-connection kernels' test file alone is not this test
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, test_file=False, hc_test_file=True)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result == {"status": "not_run",
                      "reason": "tests/unit_tests/hc_fused/test_moe_hpu_op.py is not in the bundle", "hpu_op": None}
    assert records(log) == []


def test_a_skip_reason_runs_nothing(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    out = tmp_path / "g1b"

    result = gates.moe_glue(plugin, out, python=fake, skip="--quick-gates without --moe-glue")

    assert result == {"status": "skipped", "reason": "--quick-gates without --moe-glue", "hpu_op": None}
    assert records(log) == []
    assert not (out / "moe_hpu_op.log").exists()


def test_the_step_runs_through_the_combined_database_by_default(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["combined_kernel_db"] == "active"
    assert records(log)[1]["COMBINED"] == "<unset>"


def test_the_step_runs_without_the_combined_database_when_the_servers_will(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake, combined_db_off=True)

    assert result["status"] == "passed"
    assert result["combined_kernel_db"] == "switched_off"
    assert records(log)[1]["COMBINED"] == "0"


def test_the_operators_switch_reaches_the_step(tmp_path, monkeypatch):
    lib = make_lib(tmp_path, combined=True)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)
    monkeypatch.setenv(KILL_SWITCH, "0")

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["combined_kernel_db"] == "switched_off"
    assert records(log)[1]["COMBINED"] == "0"


def test_no_combined_database_installed(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    assert gates.moe_glue(plugin, tmp_path / "g1b", python=fake)["combined_kernel_db"] == "not_installed"


def test_without_the_gdn_library_the_step_still_runs(tmp_path, monkeypatch):
    """The kernels are independent builds: a missing fused GDN library does not stop this test."""
    lib = make_lib(tmp_path, gdn=False)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=fake)

    assert result["status"] == "passed"
    assert records(log)[1]["GDNKERNEL"] == "<unset>"


def test_it_never_raises(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    _, plugin, _ = setup(tmp_path, monkeypatch, lib=lib)

    result = gates.moe_glue(plugin, tmp_path / "g1b", python=str(tmp_path / "no-such-python"))

    assert result["status"] == "not_run"
    assert "FileNotFoundError" in result["reason"]
    assert result["hpu_op"] is None


def test_it_has_no_way_to_change_the_gdn_gate():
    """A dict comes back, not a (status, details) pair, and the function takes nothing of G1b's decision:
    there is no gate status and no GDN mode it could change. The same holds for the code it shares with
    the hyper-connection kernels' card test."""
    assert list(inspect.signature(gates.moe_glue).parameters) == ["plugin_root", "out_dir", "python", "skip",
                                                                  "combined_db_off"]
    assert inspect.signature(gates.moe_glue) == inspect.signature(gates.hc_fused)
    for function in (gates.moe_glue, gates.hc_fused, gates._kernel_card_test):
        source = inspect.getsource(function)
        assert "GateStatus." not in source and "GDN_MODE" not in source and "GDN_VERIFY" not in source


def test_the_two_card_tests_are_one_piece_of_code():
    """Neither function has a body of its own to drift: both hand their names to the same helper."""
    for function in (gates.moe_glue, gates.hc_fused):
        source = inspect.getsource(function)
        assert "_kernel_card_test(" in source
        assert "run_step(" not in source and "pytest" not in source.split('"""')[-1]


def test_the_hyper_connection_test_is_not_changed_by_the_new_one(tmp_path, monkeypatch):
    lib = make_lib(tmp_path)
    fake, plugin, log = setup(tmp_path, monkeypatch, lib=lib, hc_test_file=True,
                              pytest_out="." * 90 + " [100%]\n90 passed in 41.20s")
    monkeypatch.setenv("VLLM_HPU_MOE_GLUE", "1")
    out = tmp_path / "g1b"

    result = gates.hc_fused(plugin, out, python=fake)

    assert result["status"] == "passed" and result["log"] == "hc_hpu_op.log"
    assert result["hpu_op"]["passed"] == 90
    assert not (out / "moe_hpu_op.log").exists()
    step = records(log)[1]
    assert step["args"] == "-m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op.py"
    assert step["HCFLAG"] == "<unset>" and step["MOEFLAG"] == "<unset>"  # neither flag reaches either test


# --------------------------------------------------------------------------- the fallback lines of a server log

MOE_LOG = """\
INFO 10:00:00 [serve] starting
(Worker_TP0 pid=11) WARNING 10:00:01 [__init__.py:197] MoE glue (VLLM_HPU_MOE_GLUE=1) has no HPU kernel: the \
kernels of the hc_fused_ops extension are not usable (x.so: file does not exist); every MoE block runs the torch \
path. Build them with tools/build_flashinfer_gaudi.py --hc-fused-only, or set VLLM_HPU_MOE_GLUE=0.
(Worker_TP3 pid=14) WARNING 10:00:03 [hpu_fused_moe.py:357] MoE glue (VLLM_HPU_MOE_GLUE=1) is off for \
language_model.model.layers.3.mlp: the FP8 experts have a static input scale; using the torch path
(Worker_TP3 pid=14) WARNING 10:00:03 [hyperconnection.py:260] HC fused (VLLM_HPU_HC_FUSED=1) is off for \
language_model.model.layers.3.attn_hyper_connection: hidden states are 3-D, the fused ops take [tokens, width]; \
using the torch path
WARNING 10:00:04 [gdn] GDN core decode (VLLM_HPU_GDN_CORE_DECODE=1) is off for model.layers.3: head dims differ
INFO 10:00:05 MoE glue is a mixture-of-experts thing and this line says nothing is off
"""


def test_fallback_lines_catch_the_moe_glue_lines():
    lines = gates.gdn_fallback_lines(MOE_LOG)
    assert len(lines) == 4
    assert "has no HPU kernel" in lines[0] and "MoE glue" in lines[0]
    assert "MoE glue (VLLM_HPU_MOE_GLUE=1) is off for language_model.model.layers.3.mlp" in lines[1]
    assert "HC fused" in lines[2]
    assert "GDN core decode" in lines[3]


def test_moe_fallback_lines_are_the_moe_glue_ones_only():
    lines = gates.moe_fallback_lines(MOE_LOG)
    assert len(lines) == 2 and all("MoE glue (VLLM_HPU_MOE_GLUE=1)" in line for line in lines)
    assert gates.moe_fallback_lines("INFO MoE glue: on\nWARNING GDN core decode (1) is off for x: y\n") == []
    assert gates.moe_fallback_lines(None) == []


def test_a_hyper_connection_line_is_not_a_moe_glue_line_and_the_other_way_round():
    hc_lines = gates.hc_fallback_lines(MOE_LOG)
    assert len(hc_lines) == 1 and "HC fused (VLLM_HPU_HC_FUSED=1) is off for" in hc_lines[0]
    assert not set(hc_lines) & set(gates.moe_fallback_lines(MOE_LOG))
    # the hyper-connection kernels' own log has two of their lines and none of the MoE glue's
    assert len(gates.hc_fallback_lines(HC_LOG)) == 2 and gates.moe_fallback_lines(HC_LOG) == []
    for line in gates.gdn_fallback_lines(MOE_LOG):
        assert not (gates.HC_FALLBACK_RE.search(line) and gates.MOE_FALLBACK_RE.search(line)), line


def test_a_line_that_names_both_families_belongs_to_the_one_it_starts_with():
    """The MoE glue's startup line names the hyper-connection extension (the kernels live in it), and a
    reason may quote the other family: the family is the one whose prefix opens the message."""
    moe = ("WARNING MoE glue (VLLM_HPU_MOE_GLUE=1) is off for layers.0.mlp: no hpu kernel for moe_route (HC fused "
           "(VLLM_HPU_HC_FUSED=1) has no HPU kernel); using the torch path")
    hc = ("WARNING HC fused (VLLM_HPU_HC_FUSED=1) is off for layers.0.attn_hyper_connection: the extension that "
          "MoE glue (VLLM_HPU_MOE_GLUE=1) is off for too; using the torch path")
    text = moe + "\n" + hc + "\n"
    assert gates.moe_fallback_lines(text) == [moe]
    assert gates.hc_fallback_lines(text) == [hc]
