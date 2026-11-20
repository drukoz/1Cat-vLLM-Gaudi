# SPDX-License-Identifier: Apache-2.0
"""The orchestrator and the fused hyper-connection kernels: the operator's --hc-fused, the card test
inside gate G1b, the flag every server gets, the tiny gate's fallback, and what the manifest and the
summary record. Off by default: without --hc-fused every server gets an explicit VLLM_HPU_HC_FUSED=0."""

import json

import pytest

from tools.flash_next.rental import gates, report, serve
from tools.flash_next.rental import main as rental_main
from tools.flash_next.rental.schema import GateStatus

from .test_main import make_model, read_json, run_stub
from .test_main_fallback import BROKEN_WITH_COMBINED_DB, FUSED_ENV, KILL_SWITCH, fused_node  # noqa: F401
from .test_tiny_smoke import FAKE_SERVER

HC = gates.HC_FUSED_ENV
GDN_PASSED = {"mode": 1, "verify_mode": 1, "env": {"VLLM_HPU_GDN_CORE_VERIFY": "1"}, "combined_kernel_db": "active"}
HC_PASSED = {"status": "passed", "reason": "", "hpu_op": {"exit": 0, "passed": 90, "skipped": 0, "failed": 0,
                                                          "error": 0, "timed_out": False}, "log": "hc_hpu_op.log"}
HC_FAILED = {"status": "failed", "reason": "test_hpu_op.py failed", "hpu_op": {"exit": 1, "passed": 88, "skipped": 0,
                                                                               "failed": 2, "error": 0,
                                                                               "timed_out": False}}

# A tiny server that answers wrong tokens (or dies, or hangs) with the fused hyper-connections and
# serves the reference without them.
BROKEN_WITH_HC_FUSED = '''
import os, sys, time
if os.environ.get("VLLM_HPU_HC_FUSED") == "1":
    if os.environ.get("FAKE_HANG_WITH_HC"):
        time.sleep(600)
    if os.environ.get("FAKE_DIE_WITH_HC"):
        print("(Worker_TP0 pid=5) RuntimeError: synStatus=26 [Generic failure] hc_fused_gate_mix", flush=True)
        sys.exit(4)
    os.environ["FAKE_SERVER_WRONG"] = "1"
''' + FAKE_SERVER
# ... and one that also cannot start with the combined kernel database.
BROKEN_WITH_BOTH = '''
import os, sys
if os.environ.get("FLASHINFER_GAUDI_COMBINED_KERNEL_DB") != "0":
    print("(Worker_TP0 pid=5) RuntimeError: synStatus=26 [Generic failure] the adapter returned no kernel", flush=True)
    sys.exit(4)
''' + BROKEN_WITH_HC_FUSED


@pytest.fixture
def gate_calls(monkeypatch):
    """gates.fused_gdn and gates.hc_fused replaced: what they were called with, and what they answer."""
    calls = {"order": [], "hc": []}
    answers = {"gdn": (GateStatus.PASSED, dict(GDN_PASSED, env=dict(GDN_PASSED["env"]))), "hc": dict(HC_PASSED)}

    def fused_gdn(plugin_root, out_dir, python, **kwargs):
        calls["order"].append("gdn")
        status, details = answers["gdn"]
        return status, json.loads(json.dumps(details))

    def hc_fused(plugin_root, out_dir, python, **kwargs):
        calls["order"].append("hc")
        calls["hc"].append(dict(kwargs, out_dir=str(out_dir)))
        if isinstance(answers["hc"], Exception):
            raise answers["hc"]
        if kwargs.get("skip"):
            return {"status": "skipped", "reason": kwargs["skip"], "hpu_op": None}
        return json.loads(json.dumps(answers["hc"]))

    monkeypatch.setattr(gates, "fused_gdn", fused_gdn)
    monkeypatch.setattr(gates, "hc_fused", hc_fused)
    # The MoE glue kernels' card test follows this one in G1b (test_main_moe_glue.py): never the real step here.
    monkeypatch.setattr(gates, "moe_glue", lambda *args, **kwargs: {"status": "not_run", "reason": "not in these tests",
                                                                    "hpu_op": None})
    return calls, answers


def summary_md(runner) -> str:
    runner.write_manifest()
    assert report.main([str(runner.run_dir)]) == 0
    return (runner.run_dir / "summary.md").read_text(encoding="utf-8")


def hc_line(runner) -> str:
    (line, ) = [ln for ln in summary_md(runner).splitlines() if ln.startswith("- HC fused:")]
    return line


# --------------------------------------------------------------------------- the option


def test_the_option_is_off_by_default():
    args = rental_main._parse_args(["--model", "/m", "--results", "/r"])
    assert args.hc_fused is False
    assert rental_main._parse_args(["--model", "/m", "--results", "/r", "--hc-fused"]).hc_fused is True


# --------------------------------------------------------------------------- the card test inside G1b


def test_the_console_says_what_the_card_test_did_and_what_the_servers_get(fused_node, gate_calls, capsys):
    _, answers = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    out = capsys.readouterr().out
    assert "fused_gdn: PASSED after" in out
    assert "fused_gdn: hyper-connection kernels' card test passed; HC fused: on" in out

    answers["hc"] = dict(HC_FAILED, log="hc_hpu_op.log")
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    out = capsys.readouterr().out
    assert ("fused_gdn: hyper-connection kernels' card test failed (test_hpu_op.py failed); HC fused: off "
            "(test_hpu_op.py failed)") in out
    assert "    log: %s" % (runner.run_dir / "fused_gdn" / "hc_hpu_op.log") in out

    runner = fused_node(True, FAKE_SERVER)  # not asked for
    answers["hc"] = dict(HC_PASSED)
    runner.run_gate("fused_gdn")
    assert ("fused_gdn: hyper-connection kernels' card test passed; HC fused: off (not requested)"
            in capsys.readouterr().out)


def test_the_card_test_runs_right_after_g1b_and_is_recorded_in_its_details(fused_node, gate_calls):
    calls, _ = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")

    assert runner.run_gate("fused_gdn") is True

    assert calls["order"] == ["gdn", "hc"]
    assert calls["hc"] == [{"skip": None, "combined_db_off": False, "out_dir": str(runner.run_dir / "fused_gdn")}]
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["hc_fused"]["status"] == "passed"
    assert details["hc_fused"]["hpu_op"]["passed"] == 90
    assert details["hc_fused"]["requested"] is True
    assert details["env"][HC] == "1"  # with the GDN flags of every later server
    assert details["mode"] == 1 and details["env"]["VLLM_HPU_GDN_CORE_VERIFY"] == "1"
    assert read_json(runner.run_dir / "fused_gdn" / "status.json")["status"] == int(GateStatus.PASSED)
    assert runner.hc_fused_on() is True


@pytest.mark.parametrize("hc_answer", [HC_FAILED, RuntimeError("boom"), {"status": "not_run", "reason": "x",
                                                                         "hpu_op": None}])
def test_the_card_test_never_changes_the_gate_or_the_gdn_kernel(fused_node, gate_calls, hc_answer):
    calls, answers = gate_calls
    answers["hc"] = hc_answer
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")

    assert runner.run_gate("fused_gdn") is True

    status = read_json(runner.run_dir / "fused_gdn" / "status.json")
    assert status["status"] == int(GateStatus.PASSED) and status["reason"] is None
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["mode"] == 1 and details["verify_mode"] == 1
    assert details["env"]["VLLM_HPU_GDN_CORE_VERIFY"] == "1" and "VLLM_HPU_GDN_CORE_DECODE" not in details["env"]
    assert details["env"][HC] == "0"
    assert details["hc_fused"]["status"] in ("failed", "not_run")
    assert runner.gdn_mode == gates.GDN_MODE_FUSED and runner.combined_db_active() is True
    assert runner.hc_fused_on() is False
    if isinstance(hc_answer, Exception):
        assert details["hc_fused"]["reason"] == "RuntimeError: boom"


def test_a_failed_g1b_does_not_stop_the_card_test(fused_node, gate_calls):
    """The two kernels are independent: the hyper-connection test runs, without the combined database
    (no server loads it once the fused GDN kernel is off), and its pass still counts."""
    calls, answers = gate_calls
    answers["gdn"] = (GateStatus.FAILED, {"mode": 0, "verify_mode": 0, "env": dict(gates.GDN_FALLBACK_ENV),
                                          "reason": "test_hpu_op.py failed (exit 1)",
                                          "combined_kernel_db": "failed_with_and_without"})
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")

    assert runner.run_gate("fused_gdn") is True  # recorded only, as before

    assert calls["hc"][0]["combined_db_off"] is True
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["mode"] == 0 and details["reason"] == "test_hpu_op.py failed (exit 1)"
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0", HC: "1",
                              gates.MOE_GLUE_ENV: "0"}  # the MoE glue flag is explicit too (not asked for here)
    assert read_json(runner.run_dir / "fused_gdn" / "status.json")["status"] == int(GateStatus.FAILED)
    assert runner.hc_fused_on() is True


def test_g1b_without_the_combined_database_runs_the_card_test_without_it(fused_node, gate_calls):
    calls, answers = gate_calls
    answers["gdn"] = (GateStatus.PASSED, dict(GDN_PASSED, combined_kernel_db="disabled_by_g1b",
                                              env={"VLLM_HPU_GDN_CORE_VERIFY": "1", KILL_SWITCH: "0"}))
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")

    runner.run_gate("fused_gdn")

    assert calls["hc"][0]["combined_db_off"] is True


def test_without_the_request_the_card_test_still_runs_and_the_servers_stay_off(fused_node, gate_calls):
    calls, _ = gate_calls
    runner = fused_node(True, FAKE_SERVER)

    runner.run_gate("fused_gdn")

    assert calls["hc"][0]["skip"] is None  # recorded: the kernels' first card numbers
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["hc_fused"]["status"] == "passed" and details["hc_fused"]["requested"] is False
    assert details["env"][HC] == "0"
    assert runner.hc_fused_on() is False
    assert runner.variant_server_env({})[HC] == "0"


def test_quick_gates_skips_the_card_test_unless_the_kernel_was_asked_for(fused_node, gate_calls):
    calls, _ = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--quick-gates")
    runner.run_gate("fused_gdn")
    assert calls["hc"][0]["skip"] == "--quick-gates without --hc-fused"
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["hc_fused"]["status"] == "skipped" and details["env"][HC] == "0"
    assert details["hc_fused"]["reason"] == "--quick-gates without --hc-fused"

    runner = fused_node(True, FAKE_SERVER, "--quick-gates", "--hc-fused")  # it decides a kernel: it runs
    runner.run_gate("fused_gdn")
    assert calls["hc"][1]["skip"] is None
    assert runner.hc_fused_on() is True


def test_stub_mode_runs_no_card_test(tmp_path, capsys, gate_calls):
    calls, _ = gate_calls
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "stubhc", "--variants", "A", "--hc-fused") == 0
    capsys.readouterr()
    run = tmp_path / "results" / "stubhc"
    assert calls["order"] == []
    details = read_json(run / "fused_gdn" / "details.json")
    assert details["hc_fused"] == {"status": "not_run", "reason": "stub mode", "hpu_op": None, "requested": True}
    manifest = read_json(run / "manifest.json")
    assert manifest["args"]["hc_fused"] is True
    assert manifest["hc_fused"] == {"requested": True, "status": "not_run", "reason": "stub mode", "on": False,
                                    "disabled_by": None, "outcome": "off (stub mode)"}
    assert "- HC fused: off (stub mode)" in (run / "summary.md").read_text(encoding="utf-8")
    assert read_json(run / "A" / "status.json")["servers"][0]["planned_env"][HC] == "0"


def test_a_default_stub_run_records_off_not_requested(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "plain", "--variants", "A") == 0
    capsys.readouterr()
    run = tmp_path / "results" / "plain"
    manifest = read_json(run / "manifest.json")
    assert manifest["args"]["hc_fused"] is False
    assert manifest["hc_fused"]["requested"] is False and manifest["hc_fused"]["on"] is False
    assert manifest["hc_fused"]["outcome"] == "off (not requested)"
    assert "- HC fused: off (not requested)" in (run / "summary.md").read_text(encoding="utf-8")
    assert read_json(run / "A" / "status.json")["servers"][0]["planned_env"][HC] == "0"


# --------------------------------------------------------------------------- the flag of every server


def test_servers_get_the_kernels_only_when_asked_and_the_card_test_passed(fused_node, gate_calls):
    calls, answers = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    variant = runner.variant_by_id("A")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = runner.variant_server_env(eff["env"])
    _, env = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    assert env[HC] == "1"
    assert runner.tiny_env()[HC] == "1"
    assert hc_line(runner) == "- HC fused: on"
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["hc_fused"] == {"requested": True, "status": "passed", "reason": "", "on": True,
                                    "disabled_by": None, "outcome": "on"}

    answers["hc"] = dict(HC_FAILED)
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    assert runner.variant_server_env({})[HC] == "0" and runner.tiny_env()[HC] == "0"
    assert hc_line(runner) == "- HC fused: off (test_hpu_op.py failed)"
    assert read_json(runner.run_dir / "manifest.json")["hc_fused"]["outcome"] == "off (test_hpu_op.py failed)"


def test_the_orchestrators_decision_wins_over_the_environment_and_the_variant(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv(HC, "1")  # the operator's shell, or docker -e
    runner = fused_node(True, FAKE_SERVER)  # no --hc-fused
    runner.run_gate("fused_gdn")
    variant = runner.variant_by_id("A")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = runner.variant_server_env(dict(eff["env"], **{HC: "1"}))
    _, env = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    assert env[HC] == "0"
    assert hc_line(runner) == "- HC fused: off (not requested)"


def test_before_g1b_ran_the_servers_are_off(fused_node):
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    assert runner.hc_fused_on() is False
    assert runner.variant_server_env({})[HC] == "0"
    assert hc_line(runner) == "- HC fused: off (the card test has not run)"


@pytest.mark.parametrize("hc_answer,outcome", [
    ({"status": "skipped", "reason": "the fused hyper-connection library is not installed: libhc_fused_gaudi2.so "
      "missing in /x", "hpu_op": None},
     "off (the fused hyper-connection library is not installed: libhc_fused_gaudi2.so missing in /x)"),
    ({"status": "failed", "reason": "test_hpu_op.py skipped", "hpu_op": {}}, "off (test_hpu_op.py skipped)"),
    ({"status": "failed", "reason": "test_hpu_op.py did not finish in 600.0 s", "hpu_op": {}},
     "off (test_hpu_op.py did not finish in 600.0 s)"),
    ({"status": "not_run", "reason": "tests/unit_tests/hc_fused/test_hpu_op.py is not in the bundle", "hpu_op": None},
     "off (tests/unit_tests/hc_fused/test_hpu_op.py is not in the bundle)"),
])
def test_every_way_the_card_test_does_not_pass_leaves_the_servers_off(fused_node, gate_calls, hc_answer, outcome):
    _, answers = gate_calls
    answers["hc"] = hc_answer
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    assert runner.variant_server_env({})[HC] == "0"
    assert hc_line(runner) == "- HC fused: " + outcome


# --------------------------------------------------------------------------- the recipe cache


def test_a_server_with_the_kernels_gets_the_recipe_cache_only_through_the_combined_database(fused_node, gate_calls):
    _, answers = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[HC] == "1" and serve.RECIPE_CACHE_ENV in env and KILL_SWITCH not in env

    # the fused GDN kernel is off, so no server loads the combined database: the hyper-connection library
    # is then registered in the colon list, which the bridge refuses next to a recipe cache
    answers["gdn"] = (GateStatus.FAILED, {"mode": 0, "verify_mode": 0, "env": dict(gates.GDN_FALLBACK_ENV),
                                          "reason": "replay failed"})
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[HC] == "1" and env["VLLM_HPU_GDN_CORE_DECODE"] == "0"
    assert serve.RECIPE_CACHE_ENV not in env
    assert runner.recipe_cache_note == serve.RECIPE_CACHE_OFF_REASON

    answers["hc"] = dict(HC_FAILED)  # without the kernels that server keeps its cache, as before
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[HC] == "0" and serve.RECIPE_CACHE_ENV in env


def test_recipe_cache_env_counts_the_hc_flag_as_a_kernel_database():
    off = {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0"}
    assert serve.recipe_cache_env("/rc", 64, off)[0] != {}
    assert serve.recipe_cache_env("/rc", 64, dict(off, **{HC: "0"}))[0] != {}
    assert serve.recipe_cache_env("/rc", 64, dict(off, **{HC: "1"})) == ({}, serve.RECIPE_CACHE_OFF_REASON)
    assert serve.recipe_cache_env("/rc", 64, dict(off, **{HC: "1"}), combined_db=True)[0] != {}
    assert serve.recipe_cache_env(None, 64, dict(off, **{HC: "1"})) == ({}, serve.RECIPE_CACHE_NONE)


# --------------------------------------------------------------------------- the tiny gate's fallback


def test_tiny_gate_runs_once_more_without_the_kernels(fused_node, gate_calls, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s,%s" % (KILL_SWITCH, serve.RECIPE_CACHE_ENV, HC))
    runner = fused_node(True, BROKEN_WITH_HC_FUSED, "--hc-fused")
    runner.run_gate("fused_gdn")
    assert runner.hc_fused_on() is True

    assert runner.run_gate("tiny") is True  # the second attempt passed: the run goes on, on the torch path

    tiny = runner.run_dir / "tiny"
    kept = tiny / "attempt-hc-fused"
    first = read_json(kept / "status.json")
    assert first["status"] == int(GateStatus.FAILED)
    assert "the shortest failing prompt has 1 tokens" in first["reason"]
    assert read_json(kept / "details.json")["hc_fused"] is True
    assert (kept / "server0.log").is_file()
    assert not (tiny / "attempt-combined").exists()  # the combined database was not what failed

    assert read_json(tiny / "status.json")["status"] == int(GateStatus.PASSED)
    details = read_json(tiny / "details.json")
    assert details["hc_fused"] is False and details["hc_fused_disabled_by"] == "tiny"
    assert details["attempt_hc_fused"]["dir"] == "attempt-hc-fused"
    assert details["attempt_hc_fused"]["status"] == int(GateStatus.FAILED)
    assert details["combined_kernel_db"] is True and "combined_kernel_db_disabled_by" not in details
    second_env = read_json(tiny / "server_env.json")
    assert second_env[HC] == "0"
    assert second_env[KILL_SWITCH] is None and second_env[serve.RECIPE_CACHE_ENV] is not None  # the rest stays

    assert runner.hc_fused_on() is False and runner.hc_fused_disabled_by == "tiny"
    assert runner.combined_db_disabled_by is None and runner.gdn_mode == gates.GDN_MODE_FUSED
    assert runner.variant_server_env({})[HC] == "0"  # every later server
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["hc_fused"]["on"] is False and manifest["hc_fused"]["disabled_by"] == "tiny"
    assert hc_line(runner) == "- HC fused: off (the tiny gate failed with it and passed without)"
    out = capsys.readouterr().out
    assert "tiny: FAILED with the fused hyper-connection kernels" in out and "VLLM_HPU_HC_FUSED=0" in out


@pytest.mark.parametrize("how,status", [("FAKE_DIE_WITH_HC", GateStatus.FAILED),
                                        ("FAKE_HANG_WITH_HC", GateStatus.TIMED_OUT)])
def test_tiny_gate_falls_back_after_a_crash_or_a_hang_too(fused_node, gate_calls, monkeypatch, how, status):
    monkeypatch.setenv(how, "1")
    monkeypatch.setattr(rental_main, "TINY_HEALTH_TIMEOUT_S", 2.0)
    runner = fused_node(True, BROKEN_WITH_HC_FUSED, "--hc-fused")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    first = read_json(runner.run_dir / "tiny" / "attempt-hc-fused" / "status.json")
    assert first["status"] == int(status)
    assert runner.hc_fused_disabled_by == "tiny"


def test_tiny_gate_with_working_kernels_runs_once(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", HC)
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    tiny = runner.run_dir / "tiny"
    assert not (tiny / "attempt-hc-fused").exists()
    assert read_json(tiny / "server_env.json")[HC] == "1"
    details = read_json(tiny / "details.json")
    assert details["hc_fused"] is True and "hc_fused_disabled_by" not in details
    assert runner.hc_fused_on() is True and len(runner.builds) == 1


def test_tiny_gate_without_the_kernels_keeps_its_one_fallback(fused_node, gate_calls, monkeypatch):
    """Not asked for: the gate behaves exactly as before (the combined database's own fallback)."""
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (KILL_SWITCH, HC))
    runner = fused_node(True, BROKEN_WITH_COMBINED_DB)
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-combined").is_dir() and not (tiny / "attempt-hc-fused").exists()
    assert read_json(tiny / "server_env.json")[HC] == "0"
    assert read_json(tiny / "details.json")["hc_fused"] is False
    assert len(runner.builds) == 2


def test_tiny_gate_tries_the_kernels_first_and_the_combined_database_second(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (KILL_SWITCH, HC))
    runner = fused_node(True, BROKEN_WITH_BOTH, "--hc-fused")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True  # the third attempt: no kernels, no combined database

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-hc-fused" / "status.json").is_file()
    assert (tiny / "attempt-combined" / "status.json").is_file()
    assert read_json(tiny / "attempt-hc-fused" / "details.json")["hc_fused"] is True
    assert read_json(tiny / "attempt-combined" / "details.json")["hc_fused"] is False
    env = read_json(tiny / "server_env.json")
    assert env[HC] == "0" and env[KILL_SWITCH] == "0"
    details = read_json(tiny / "details.json")
    assert details["hc_fused_disabled_by"] == "tiny" and details["combined_kernel_db_disabled_by"] == "tiny"
    assert details["attempt_hc_fused"]["dir"] == "attempt-hc-fused"
    assert details["attempt_combined"]["dir"] == "attempt-combined"
    assert runner.hc_fused_disabled_by == "tiny" and runner.combined_db_disabled_by == "tiny"
    assert len(runner.builds) == 3


def test_tiny_gate_failing_with_and_without_the_kernels_fails_the_gate(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    runner = fused_node(False, FAKE_SERVER, "--hc-fused")  # no combined database installed
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is False

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-hc-fused").is_dir()
    assert read_json(tiny / "status.json")["status"] == int(GateStatus.FAILED)
    assert len(runner.builds) == 3  # one rerun for the kernels, one on the PyTorch GDN path (the last fallback)
    # the kernels were not shown to be at fault, but they stay off: the proven path is what goes on
    assert runner.hc_fused_on() is False
    assert hc_line(runner) == "- HC fused: off (the tiny gate failed with it)"


def test_tiny_gate_does_not_rerun_when_its_server_never_started(fused_node, gate_calls, monkeypatch):
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    monkeypatch.setattr(rental_main, "ensure_tiny_checkpoint", lambda ckpt, **kwargs: "tiny checkpoint failed (exit 1)")

    assert runner.run_gate("tiny") is False

    assert not (runner.run_dir / "tiny" / "attempt-hc-fused").exists()
    assert runner.hc_fused_disabled_by is None and runner.hc_fused_on() is True


# --------------------------------------------------------------------------- resume


def test_a_resumed_run_keeps_the_card_test_and_the_tiny_gates_decision(fused_node, gate_calls):
    runner = fused_node(True, FAKE_SERVER, "--hc-fused")
    runner.run_gate("fused_gdn")
    runner.write_manifest()

    resumed = fused_node(True, FAKE_SERVER, "--hc-fused", "--from", "A")
    assert resumed.hc_fused_on() is False  # nothing restored yet
    resumed.restore_gdn()
    resumed.restore_hc_fused(read_json(runner.run_dir / "manifest.json"))
    assert resumed.hc_fused_on() is True and resumed.variant_server_env({})[HC] == "1"

    runner.hc_fused_disabled_by = "tiny"
    runner.write_manifest()
    resumed = fused_node(True, FAKE_SERVER, "--hc-fused", "--from", "A")
    resumed.restore_gdn()
    resumed.restore_hc_fused(read_json(runner.run_dir / "manifest.json"))
    assert resumed.hc_fused_disabled_by == "tiny" and resumed.variant_server_env({})[HC] == "0"

    # resumed without the option: the request is this start's
    resumed = fused_node(True, FAKE_SERVER, "--from", "A")
    resumed.restore_gdn()
    resumed.restore_hc_fused(read_json(runner.run_dir / "manifest.json"))
    assert resumed.variant_server_env({})[HC] == "0"
    assert hc_line(resumed) == "- HC fused: off (not requested)"


def test_a_resume_from_a_run_made_before_the_option_existed(fused_node):
    earlier = fused_node(True, FAKE_SERVER)
    gdn_dir = earlier.run_dir / "fused_gdn"
    gdn_dir.mkdir(parents=True, exist_ok=True)
    (gdn_dir / "details.json").write_text(json.dumps(GDN_PASSED), encoding="utf-8")  # no "hc_fused" in it
    (earlier.run_dir / "manifest.json").write_text(json.dumps({"run_id": "r"}), encoding="utf-8")  # nor here

    runner = fused_node(True, FAKE_SERVER, "--hc-fused", "--from", "A")
    runner.restore_gdn()
    runner.restore_hc_fused(read_json(runner.run_dir / "manifest.json"))
    assert runner.gdn_mode == gates.GDN_MODE_FUSED
    assert runner.hc_fused_on() is False
    assert runner.variant_server_env({})[HC] == "0"
    assert hc_line(runner) == "- HC fused: off (the card test has not run)"


# --------------------------------------------------------------------------- fallback lines in a variant


def test_the_variants_fallback_file_lists_the_hc_lines_and_the_summary_counts_them(tmp_path):
    run = tmp_path / "run"
    (run / "fused_gdn").mkdir(parents=True)
    (run / "fused_gdn" / "details.json").write_text(json.dumps({"mode": 1, "reason": None, "env": {}}),
                                                    encoding="utf-8")
    vdir = run / "A"
    vdir.mkdir()
    (vdir / "server0.log").write_text(
        "INFO ok\n"
        "(Worker_TP0 pid=3) WARNING [hyperconnection.py:250] HC fused (VLLM_HPU_HC_FUSED=1) is off for "
        "language_model.model.layers.0.attn_hyper_connection: hidden states are 3-D, the fused ops take "
        "[tokens, width]; using the torch path\n"
        "WARNING [qwen3_5.py:306] GDN core decode (VLLM_HPU_GDN_CORE_DECODE=1) is off for layers.1: conv1d has a bias; "
        "using the torch decode path\n"
        "(Worker_TP1 pid=4) WARNING [__init__.py:140] HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel: the "
        "hc_fused_ops extension did not load (x); every hyper-connection runs the torch path.\n", encoding="utf-8")
    args = rental_main._parse_args(["--mode", "stub", "--model", str(tmp_path / "m"), "--results", str(tmp_path),
                                    "--run-id", "run", "--variants", "A"])
    runner = rental_main.Runner(args, rental_main.load_variants())

    runner.write_gdn_fallback(vdir, [{}])

    lines = (vdir / "gdn_fallback.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(line.startswith("server0: ") for line in lines)
    assert sum("HC fused" in line for line in lines) == 2
    assert rental_main.gdn_summary_lines(run, ["A"]) == [
        "GDN decode: fused kernel (G1b passed)",
        "GDN fallback lines in A: 1 (the kernel did not run on every layer)",
        "HC fused fallback lines in A: 2 (the hyper-connection kernels did not run on every layer)",
    ]
