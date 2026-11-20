# SPDX-License-Identifier: Apache-2.0
"""The orchestrator and the MoE glue kernels: the operator's --moe-glue, the card test inside gate G1b
(right after the hyper-connection kernels'), the flag every server gets, the tiny gate's one retry
without the kernels (shared with --hc-fused), and what the manifest and the summary record. Off by
default: without --moe-glue every server gets an explicit VLLM_HPU_MOE_GLUE=0. The last sections ask
for both kernel families at once."""

import json

import pytest

from tools.flash_next.rental import gates, report, serve
from tools.flash_next.rental import main as rental_main
from tools.flash_next.rental.schema import GateStatus

from .test_main import make_model, read_json, run_stub
from .test_main_fallback import BROKEN_WITH_COMBINED_DB, KILL_SWITCH, fused_node  # noqa: F401
from .test_main_hc_fused import BROKEN_WITH_HC_FUSED, GDN_PASSED, HC_FAILED, HC_PASSED
from .test_tiny_smoke import FAKE_SERVER

HC = gates.HC_FUSED_ENV
MOE = gates.MOE_GLUE_ENV
MOE_PASSED = {"status": "passed", "reason": "", "hpu_op": {"exit": 0, "passed": 84, "skipped": 0, "failed": 0,
                                                           "error": 0, "timed_out": False}, "log": "moe_hpu_op.log"}
MOE_FAILED = {"status": "failed", "reason": "test_moe_hpu_op.py failed", "hpu_op": {"exit": 1, "passed": 82,
                                                                                    "skipped": 0, "failed": 2,
                                                                                    "error": 0, "timed_out": False}}
TINY_OFF = "off (the tiny gate failed with it and passed without)"

# A tiny server that answers wrong tokens (or dies, or hangs) with the MoE glue kernels and serves the
# reference without them.
BROKEN_WITH_MOE_GLUE = '''
import os, sys, time
if os.environ.get("VLLM_HPU_MOE_GLUE") == "1":
    if os.environ.get("FAKE_HANG_WITH_MOE"):
        time.sleep(600)
    if os.environ.get("FAKE_DIE_WITH_MOE"):
        print("(Worker_TP0 pid=5) RuntimeError: synStatus=26 [Generic failure] moe_glue_route", flush=True)
        sys.exit(4)
    os.environ["FAKE_SERVER_WRONG"] = "1"
''' + FAKE_SERVER
# ... one that also cannot start with the combined kernel database,
BROKEN_WITH_MOE_AND_COMBINED = '''
import os, sys
if os.environ.get("FLASHINFER_GAUDI_COMBINED_KERNEL_DB") != "0":
    print("(Worker_TP0 pid=5) RuntimeError: synStatus=26 [Generic failure] the adapter returned no kernel", flush=True)
    sys.exit(4)
''' + BROKEN_WITH_MOE_GLUE
# ... and one that answers wrong tokens as long as either kernel family is on: it serves the reference
# only when both are off, so a retry that switched off one family at a time would need two more runs.
BROKEN_WITH_EITHER = '''
import os
if os.environ.get("VLLM_HPU_MOE_GLUE") == "1" or os.environ.get("VLLM_HPU_HC_FUSED") == "1":
    os.environ["FAKE_SERVER_WRONG"] = "1"
''' + FAKE_SERVER
BROKEN_WITH_EITHER_AND_COMBINED = '''
import os, sys
if os.environ.get("FLASHINFER_GAUDI_COMBINED_KERNEL_DB") != "0":
    print("(Worker_TP0 pid=5) RuntimeError: synStatus=26 [Generic failure] the adapter returned no kernel", flush=True)
    sys.exit(4)
''' + BROKEN_WITH_EITHER


@pytest.fixture
def gate_calls(monkeypatch):
    """gates.fused_gdn, gates.hc_fused and gates.moe_glue replaced: what they were called with, and what
    they answer."""
    calls = {"order": [], "hc": [], "moe": []}
    answers = {"gdn": (GateStatus.PASSED, dict(GDN_PASSED, env=dict(GDN_PASSED["env"]))), "hc": dict(HC_PASSED),
               "moe": dict(MOE_PASSED)}

    def fused_gdn(plugin_root, out_dir, python, **kwargs):
        calls["order"].append("gdn")
        status, details = answers["gdn"]
        return status, json.loads(json.dumps(details))

    def card_test(name):
        def run(plugin_root, out_dir, python, **kwargs):
            calls["order"].append(name)
            calls[name].append(dict(kwargs, out_dir=str(out_dir)))
            if isinstance(answers[name], Exception):
                raise answers[name]
            if kwargs.get("skip"):
                return {"status": "skipped", "reason": kwargs["skip"], "hpu_op": None}
            return json.loads(json.dumps(answers[name]))

        return run

    monkeypatch.setattr(gates, "fused_gdn", fused_gdn)
    monkeypatch.setattr(gates, "hc_fused", card_test("hc"))
    monkeypatch.setattr(gates, "moe_glue", card_test("moe"))
    return calls, answers


def summary_md(runner) -> str:
    runner.write_manifest()
    assert report.main([str(runner.run_dir)]) == 0
    return (runner.run_dir / "summary.md").read_text(encoding="utf-8")


def moe_line(runner) -> str:
    (line, ) = [ln for ln in summary_md(runner).splitlines() if ln.startswith("- MoE glue:")]
    return line


def hc_line(runner) -> str:
    (line, ) = [ln for ln in summary_md(runner).splitlines() if ln.startswith("- HC fused:")]
    return line


# --------------------------------------------------------------------------- the option


def test_the_option_is_off_by_default():
    args = rental_main._parse_args(["--model", "/m", "--results", "/r"])
    assert args.moe_glue is False
    args = rental_main._parse_args(["--model", "/m", "--results", "/r", "--moe-glue"])
    assert args.moe_glue is True and args.hc_fused is False  # each switch asks for its own kernels only
    args = rental_main._parse_args(["--model", "/m", "--results", "/r", "--hc-fused"])
    assert args.moe_glue is False and args.hc_fused is True


# --------------------------------------------------------------------------- the card test inside G1b


def test_the_console_says_what_the_card_test_did_and_what_the_servers_get(fused_node, gate_calls, capsys):
    _, answers = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    out = capsys.readouterr().out
    assert "fused_gdn: PASSED after" in out
    assert "fused_gdn: MoE glue kernels' card test passed; MoE glue: on" in out
    # after the hyper-connection kernels' line, which stays what it was
    hc_at = out.index("fused_gdn: hyper-connection kernels' card test passed; HC fused: off (not requested)")
    assert hc_at < out.index("fused_gdn: MoE glue kernels' card test")

    answers["moe"] = dict(MOE_FAILED, log="moe_hpu_op.log")
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    out = capsys.readouterr().out
    assert ("fused_gdn: MoE glue kernels' card test failed (test_moe_hpu_op.py failed); MoE glue: off "
            "(test_moe_hpu_op.py failed)") in out
    assert "    log: %s" % (runner.run_dir / "fused_gdn" / "moe_hpu_op.log") in out

    runner = fused_node(True, FAKE_SERVER)  # not asked for
    answers["moe"] = dict(MOE_PASSED)
    runner.run_gate("fused_gdn")
    assert ("fused_gdn: MoE glue kernels' card test passed; MoE glue: off (not requested)"
            in capsys.readouterr().out)


def test_the_card_test_runs_right_after_the_hc_one_and_is_recorded_in_g1bs_details(fused_node, gate_calls):
    calls, _ = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")

    assert runner.run_gate("fused_gdn") is True

    assert calls["order"] == ["gdn", "hc", "moe"]
    assert calls["moe"] == [{"skip": None, "combined_db_off": False, "out_dir": str(runner.run_dir / "fused_gdn")}]
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["moe_glue"]["status"] == "passed"
    assert details["moe_glue"]["hpu_op"]["passed"] == 84
    assert details["moe_glue"]["requested"] is True
    assert details["env"][MOE] == "1"  # with the GDN flags of every later server
    assert details["env"][HC] == "0"  # the other kernels were not asked for
    assert details["hc_fused"]["status"] == "passed" and details["hc_fused"]["requested"] is False
    assert details["mode"] == 1 and details["env"]["VLLM_HPU_GDN_CORE_VERIFY"] == "1"
    assert read_json(runner.run_dir / "fused_gdn" / "status.json")["status"] == int(GateStatus.PASSED)
    assert runner.moe_glue_on() is True and runner.hc_fused_on() is False


@pytest.mark.parametrize("moe_answer", [MOE_FAILED, RuntimeError("boom"), {"status": "not_run", "reason": "x",
                                                                           "hpu_op": None}])
def test_the_card_test_never_changes_the_gate_or_the_gdn_kernel(fused_node, gate_calls, moe_answer):
    calls, answers = gate_calls
    answers["moe"] = moe_answer
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")

    assert runner.run_gate("fused_gdn") is True

    status = read_json(runner.run_dir / "fused_gdn" / "status.json")
    assert status["status"] == int(GateStatus.PASSED) and status["reason"] is None
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["mode"] == 1 and details["verify_mode"] == 1
    assert details["env"]["VLLM_HPU_GDN_CORE_VERIFY"] == "1" and "VLLM_HPU_GDN_CORE_DECODE" not in details["env"]
    assert details["env"][MOE] == "0"
    assert details["moe_glue"]["status"] in ("failed", "not_run")
    assert details["hc_fused"]["status"] == "passed"  # the other card test's record is its own
    assert runner.gdn_mode == gates.GDN_MODE_FUSED and runner.combined_db_active() is True
    assert runner.moe_glue_on() is False
    if isinstance(moe_answer, Exception):
        assert details["moe_glue"]["reason"] == "RuntimeError: boom"


def test_a_failed_g1b_does_not_stop_the_card_test(fused_node, gate_calls):
    """The kernels are independent: the MoE glue test runs, without the combined database (no server
    loads it once the fused GDN kernel is off), and its pass still counts."""
    calls, answers = gate_calls
    answers["gdn"] = (GateStatus.FAILED, {"mode": 0, "verify_mode": 0, "env": dict(gates.GDN_FALLBACK_ENV),
                                          "reason": "test_hpu_op.py failed (exit 1)",
                                          "combined_kernel_db": "failed_with_and_without"})
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")

    assert runner.run_gate("fused_gdn") is True  # recorded only, as before

    assert calls["moe"][0]["combined_db_off"] is True
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["mode"] == 0 and details["reason"] == "test_hpu_op.py failed (exit 1)"
    assert details["env"] == {"VLLM_HPU_GDN_CORE_DECODE": "0", "VLLM_HPU_GDN_CORE_VERIFY": "0", HC: "0", MOE: "1"}
    assert read_json(runner.run_dir / "fused_gdn" / "status.json")["status"] == int(GateStatus.FAILED)
    assert runner.moe_glue_on() is True


def test_g1b_without_the_combined_database_runs_the_card_test_without_it(fused_node, gate_calls):
    calls, answers = gate_calls
    answers["gdn"] = (GateStatus.PASSED, dict(GDN_PASSED, combined_kernel_db="disabled_by_g1b",
                                              env={"VLLM_HPU_GDN_CORE_VERIFY": "1", KILL_SWITCH: "0"}))
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")

    runner.run_gate("fused_gdn")

    assert calls["moe"][0]["combined_db_off"] is True


def test_without_the_request_the_card_test_still_runs_and_the_servers_stay_off(fused_node, gate_calls):
    calls, _ = gate_calls
    runner = fused_node(True, FAKE_SERVER)

    runner.run_gate("fused_gdn")

    assert calls["moe"][0]["skip"] is None  # recorded: the kernels' first card numbers
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["moe_glue"]["status"] == "passed" and details["moe_glue"]["requested"] is False
    assert details["env"][MOE] == "0"
    assert runner.moe_glue_on() is False
    assert runner.variant_server_env({})[MOE] == "0"
    assert runner.tiny_env()[MOE] == "0"


def test_quick_gates_skips_the_card_test_unless_the_kernels_were_asked_for(fused_node, gate_calls):
    calls, _ = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--quick-gates")
    runner.run_gate("fused_gdn")
    assert calls["moe"][0]["skip"] == "--quick-gates without --moe-glue"
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["moe_glue"]["status"] == "skipped" and details["env"][MOE] == "0"
    assert details["moe_glue"]["reason"] == "--quick-gates without --moe-glue"

    runner = fused_node(True, FAKE_SERVER, "--quick-gates", "--moe-glue")  # it decides a kernel: it runs
    runner.run_gate("fused_gdn")
    assert calls["moe"][1]["skip"] is None
    assert calls["hc"][1]["skip"] == "--quick-gates without --hc-fused"  # each switch covers its own test
    assert runner.moe_glue_on() is True

    runner = fused_node(True, FAKE_SERVER, "--quick-gates", "--hc-fused")
    runner.run_gate("fused_gdn")
    assert calls["hc"][2]["skip"] is None
    assert calls["moe"][2]["skip"] == "--quick-gates without --moe-glue"
    assert runner.moe_glue_on() is False and runner.hc_fused_on() is True


def test_stub_mode_runs_no_card_test(tmp_path, capsys, gate_calls):
    calls, _ = gate_calls
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "stubmoe", "--variants", "A", "--moe-glue") == 0
    capsys.readouterr()
    run = tmp_path / "results" / "stubmoe"
    assert calls["order"] == []
    details = read_json(run / "fused_gdn" / "details.json")
    assert details["moe_glue"] == {"status": "not_run", "reason": "stub mode", "hpu_op": None, "requested": True}
    manifest = read_json(run / "manifest.json")
    assert manifest["args"]["moe_glue"] is True
    assert manifest["moe_glue"] == {"requested": True, "status": "not_run", "reason": "stub mode", "on": False,
                                    "disabled_by": None, "outcome": "off (stub mode)"}
    assert "- MoE glue: off (stub mode)" in (run / "summary.md").read_text(encoding="utf-8")
    assert read_json(run / "A" / "status.json")["servers"][0]["planned_env"][MOE] == "0"


def test_a_default_stub_run_records_off_not_requested(tmp_path, capsys):
    model = make_model(tmp_path)
    assert run_stub(tmp_path, model, "plain", "--variants", "A,B") == 0
    capsys.readouterr()
    run = tmp_path / "results" / "plain"
    manifest = read_json(run / "manifest.json")
    assert manifest["args"]["moe_glue"] is False
    assert manifest["moe_glue"] == {"requested": False, "status": "not_run", "reason": "stub mode", "on": False,
                                    "disabled_by": None, "outcome": "off (not requested)"}
    lines = (run / "summary.md").read_text(encoding="utf-8").splitlines()
    at = lines.index("- HC fused: off (not requested)")
    assert lines[at + 1] == "- MoE glue: off (not requested)"  # right after the hyper-connection line
    for vid in ("A", "B"):  # neither requested: every server gets both flags, explicitly off
        for server in read_json(run / vid / "status.json")["servers"]:
            assert server["planned_env"][MOE] == "0" and server["planned_env"][HC] == "0"


# --------------------------------------------------------------------------- the flag of every server


def test_servers_get_the_kernels_only_when_asked_and_the_card_test_passed(fused_node, gate_calls):
    calls, answers = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    variant = runner.variant_by_id("A")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = runner.variant_server_env(eff["env"])
    _, env = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    assert env[MOE] == "1" and env[HC] == "0"
    assert runner.tiny_env()[MOE] == "1"
    assert moe_line(runner) == "- MoE glue: on"
    assert hc_line(runner) == "- HC fused: off (not requested)"
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["moe_glue"] == {"requested": True, "status": "passed", "reason": "", "on": True,
                                    "disabled_by": None, "outcome": "on"}

    answers["moe"] = dict(MOE_FAILED)
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    assert runner.variant_server_env({})[MOE] == "0" and runner.tiny_env()[MOE] == "0"
    assert moe_line(runner) == "- MoE glue: off (test_moe_hpu_op.py failed)"
    assert read_json(runner.run_dir / "manifest.json")["moe_glue"]["outcome"] == "off (test_moe_hpu_op.py failed)"


def test_the_orchestrators_decision_wins_over_the_environment_and_the_variant(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv(MOE, "1")  # the operator's shell, or docker -e
    runner = fused_node(True, FAKE_SERVER)  # no --moe-glue
    runner.run_gate("fused_gdn")
    variant = runner.variant_by_id("A")
    eff, _ = runner.effective_variant(variant)
    eff["env"] = runner.variant_server_env(dict(eff["env"], **{MOE: "1"}))
    _, env = serve.build_command(eff, variant["servers"][0], runner.model, runner.served_name)
    assert env[MOE] == "0"
    assert moe_line(runner) == "- MoE glue: off (not requested)"


def test_before_g1b_ran_the_servers_are_off(fused_node):
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    assert runner.moe_glue_on() is False
    assert runner.variant_server_env({})[MOE] == "0"
    assert moe_line(runner) == "- MoE glue: off (the card test has not run)"


@pytest.mark.parametrize("moe_answer,outcome", [
    ({"status": "skipped", "reason": "the library of the MoE glue kernels (the fused hyper-connection library) is "
      "not installed: libhc_fused_gaudi2.so missing in /x", "hpu_op": None},
     "off (the library of the MoE glue kernels (the fused hyper-connection library) is not installed: "
     "libhc_fused_gaudi2.so missing in /x)"),
    ({"status": "failed", "reason": "test_moe_hpu_op.py skipped", "hpu_op": {}}, "off (test_moe_hpu_op.py skipped)"),
    ({"status": "failed", "reason": "test_moe_hpu_op.py did not finish in 900.0 s", "hpu_op": {}},
     "off (test_moe_hpu_op.py did not finish in 900.0 s)"),
    ({"status": "not_run", "reason": "tests/unit_tests/hc_fused/test_moe_hpu_op.py is not in the bundle",
      "hpu_op": None}, "off (tests/unit_tests/hc_fused/test_moe_hpu_op.py is not in the bundle)"),
    ({"status": "not_run", "reason": "", "hpu_op": None}, "off (not_run)"),
])
def test_every_way_the_card_test_does_not_pass_leaves_the_servers_off(fused_node, gate_calls, moe_answer, outcome):
    _, answers = gate_calls
    answers["moe"] = moe_answer
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    assert runner.variant_server_env({})[MOE] == "0"
    assert moe_line(runner) == "- MoE glue: " + outcome


# --------------------------------------------------------------------------- the recipe cache


def test_a_server_with_the_kernels_gets_the_recipe_cache_only_through_the_combined_database(fused_node, gate_calls):
    _, answers = gate_calls
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[MOE] == "1" and serve.RECIPE_CACHE_ENV in env and KILL_SWITCH not in env

    # the fused GDN kernel is off, so no server loads the combined database: the library that holds the
    # MoE glue kernels is then registered in the colon list, which the bridge refuses next to a recipe cache
    answers["gdn"] = (GateStatus.FAILED, {"mode": 0, "verify_mode": 0, "env": dict(gates.GDN_FALLBACK_ENV),
                                          "reason": "replay failed"})
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[MOE] == "1" and env[HC] == "0" and env["VLLM_HPU_GDN_CORE_DECODE"] == "0"
    assert serve.RECIPE_CACHE_ENV not in env
    assert runner.recipe_cache_note == serve.RECIPE_CACHE_OFF_REASON

    answers["moe"] = dict(MOE_FAILED)  # without the kernels that server keeps its cache, as before
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[MOE] == "0" and serve.RECIPE_CACHE_ENV in env


# --------------------------------------------------------------------------- the tiny gate's fallback


def test_tiny_gate_runs_once_more_without_the_kernels(fused_node, gate_calls, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s,%s,%s" % (KILL_SWITCH, serve.RECIPE_CACHE_ENV, HC, MOE))
    runner = fused_node(True, BROKEN_WITH_MOE_GLUE, "--moe-glue")
    runner.run_gate("fused_gdn")
    assert runner.moe_glue_on() is True

    assert runner.run_gate("tiny") is True  # the second attempt passed: the run goes on, on the torch path

    tiny = runner.run_dir / "tiny"
    kept = tiny / "attempt-moe-glue"
    first = read_json(kept / "status.json")
    assert first["status"] == int(GateStatus.FAILED)
    assert "the shortest failing prompt has 1 tokens" in first["reason"]
    kept_details = read_json(kept / "details.json")
    assert kept_details["moe_glue"] is True and kept_details["hc_fused"] is False
    assert (kept / "server0.log").is_file()
    assert not (tiny / "attempt-hc-fused").exists()  # that folder is for a run with the hyper-connection kernels only
    assert not (tiny / "attempt-combined").exists()  # the combined database was not what failed

    assert read_json(tiny / "status.json")["status"] == int(GateStatus.PASSED)
    details = read_json(tiny / "details.json")
    assert details["moe_glue"] is False and details["moe_glue_disabled_by"] == "tiny"
    assert details["attempt_moe_glue"]["dir"] == "attempt-moe-glue"
    assert details["attempt_moe_glue"]["status"] == int(GateStatus.FAILED)
    assert details["hc_fused"] is False
    assert "hc_fused_disabled_by" not in details and "attempt_hc_fused" not in details
    assert details["combined_kernel_db"] is True and "combined_kernel_db_disabled_by" not in details
    second_env = read_json(tiny / "server_env.json")
    assert second_env[MOE] == "0" and second_env[HC] == "0"
    assert second_env[KILL_SWITCH] is None and second_env[serve.RECIPE_CACHE_ENV] is not None  # the rest stays

    assert runner.moe_glue_on() is False and runner.moe_glue_disabled_by == "tiny"
    assert runner.hc_fused_disabled_by is None
    assert runner.combined_db_disabled_by is None and runner.gdn_mode == gates.GDN_MODE_FUSED
    assert runner.variant_server_env({})[MOE] == "0"  # every later server
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["moe_glue"]["on"] is False and manifest["moe_glue"]["disabled_by"] == "tiny"
    assert manifest["hc_fused"]["disabled_by"] is None
    assert moe_line(runner) == "- MoE glue: " + TINY_OFF
    assert hc_line(runner) == "- HC fused: off (not requested)"
    out = capsys.readouterr().out
    assert "tiny: FAILED with the MoE glue kernels (" in out and "running the gate once more with %s=0" % MOE in out
    assert "attempt-moe-glue" in out and "hyper-connection kernels (" not in out
    assert len(runner.builds) == 2


@pytest.mark.parametrize("how,status", [("FAKE_DIE_WITH_MOE", GateStatus.FAILED),
                                        ("FAKE_HANG_WITH_MOE", GateStatus.TIMED_OUT)])
def test_tiny_gate_falls_back_after_a_crash_or_a_hang_too(fused_node, gate_calls, monkeypatch, how, status):
    monkeypatch.setenv(how, "1")
    monkeypatch.setattr(rental_main, "TINY_HEALTH_TIMEOUT_S", 2.0)
    runner = fused_node(True, BROKEN_WITH_MOE_GLUE, "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    first = read_json(runner.run_dir / "tiny" / "attempt-moe-glue" / "status.json")
    assert first["status"] == int(status)
    assert runner.moe_glue_disabled_by == "tiny"


def test_tiny_gate_with_working_kernels_runs_once(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (HC, MOE))
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    tiny = runner.run_dir / "tiny"
    assert not (tiny / "attempt-moe-glue").exists() and not (tiny / "attempt-hc-fused").exists()
    env = read_json(tiny / "server_env.json")
    assert env[MOE] == "1" and env[HC] == "0"
    details = read_json(tiny / "details.json")
    assert details["moe_glue"] is True and "moe_glue_disabled_by" not in details
    assert "attempt_moe_glue" not in details
    assert runner.moe_glue_on() is True and len(runner.builds) == 1


def test_tiny_gate_without_the_kernels_keeps_its_one_fallback(fused_node, gate_calls, monkeypatch):
    """Neither family asked for: the gate behaves exactly as before (the combined database's own
    fallback), and both its servers got both flags explicitly off."""
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s,%s" % (KILL_SWITCH, HC, MOE))
    runner = fused_node(True, BROKEN_WITH_COMBINED_DB)
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-combined").is_dir()
    assert not (tiny / "attempt-moe-glue").exists() and not (tiny / "attempt-hc-fused").exists()
    env = read_json(tiny / "server_env.json")
    assert env[MOE] == "0" and env[HC] == "0"
    details = read_json(tiny / "details.json")
    assert details["moe_glue"] is False and details["hc_fused"] is False
    assert read_json(tiny / "attempt-combined" / "details.json")["moe_glue"] is False
    assert len(runner.builds) == 2


def test_tiny_gate_tries_the_kernels_first_and_the_combined_database_second(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (KILL_SWITCH, MOE))
    runner = fused_node(True, BROKEN_WITH_MOE_AND_COMBINED, "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True  # the third attempt: no kernels, no combined database

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-moe-glue" / "status.json").is_file()
    assert (tiny / "attempt-combined" / "status.json").is_file()
    assert read_json(tiny / "attempt-moe-glue" / "details.json")["moe_glue"] is True
    assert read_json(tiny / "attempt-combined" / "details.json")["moe_glue"] is False
    env = read_json(tiny / "server_env.json")
    assert env[MOE] == "0" and env[KILL_SWITCH] == "0"
    details = read_json(tiny / "details.json")
    assert details["moe_glue_disabled_by"] == "tiny" and details["combined_kernel_db_disabled_by"] == "tiny"
    assert details["attempt_moe_glue"]["dir"] == "attempt-moe-glue"
    assert details["attempt_combined"]["dir"] == "attempt-combined"
    assert runner.moe_glue_disabled_by == "tiny" and runner.combined_db_disabled_by == "tiny"
    assert len(runner.builds) == 3


def test_tiny_gate_failing_with_and_without_the_kernels_fails_the_gate(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    runner = fused_node(False, FAKE_SERVER, "--moe-glue")  # no combined database installed
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is False

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-moe-glue").is_dir()
    assert read_json(tiny / "status.json")["status"] == int(GateStatus.FAILED)
    assert len(runner.builds) == 2  # exactly one rerun for the kernels
    # the kernels were not shown to be at fault, but they stay off: the proven path is what goes on
    assert runner.moe_glue_on() is False
    assert moe_line(runner) == "- MoE glue: off (the tiny gate failed with it)"


def test_tiny_gate_does_not_rerun_when_its_server_never_started(fused_node, gate_calls, monkeypatch):
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    monkeypatch.setattr(rental_main, "ensure_tiny_checkpoint", lambda ckpt, **kwargs: "tiny checkpoint failed (exit 1)")

    assert runner.run_gate("tiny") is False

    assert not (runner.run_dir / "tiny" / "attempt-moe-glue").exists()
    assert runner.moe_glue_disabled_by is None and runner.moe_glue_on() is True


def test_a_rerun_of_the_gates_decides_again(fused_node, gate_calls):
    runner = fused_node(True, BROKEN_WITH_MOE_GLUE, "--moe-glue")
    runner.run_gate("fused_gdn")
    runner.run_gate("tiny")
    assert runner.moe_glue_disabled_by == "tiny"

    runner.run_gate("fused_gdn")  # --from fused_gdn: the tiny gate that follows gets the kernels again
    assert runner.moe_glue_disabled_by is None and runner.moe_glue_on() is True

    runner.moe_glue_disabled_by = "tiny"
    runner.run_gate("tiny")  # --from tiny: the gate resets its own decision, then makes it again
    assert len(runner.builds) == 4  # two attempts each time: the second run of the gate tried the kernels again
    assert runner.moe_glue_disabled_by == "tiny"


# --------------------------------------------------------------------------- resume


def test_a_resumed_run_keeps_the_card_test_and_the_tiny_gates_decision(fused_node, gate_calls):
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")
    runner.write_manifest()

    resumed = fused_node(True, FAKE_SERVER, "--moe-glue", "--from", "A")
    assert resumed.moe_glue_on() is False  # nothing restored yet
    resumed.restore_gdn()
    resumed.restore_moe_glue(read_json(runner.run_dir / "manifest.json"))
    assert resumed.moe_glue_on() is True and resumed.variant_server_env({})[MOE] == "1"

    runner.moe_glue_disabled_by = "tiny"
    runner.write_manifest()
    resumed = fused_node(True, FAKE_SERVER, "--moe-glue", "--from", "A")
    resumed.restore_gdn()
    resumed.restore_moe_glue(read_json(runner.run_dir / "manifest.json"))
    assert resumed.moe_glue_disabled_by == "tiny" and resumed.variant_server_env({})[MOE] == "0"

    # resumed without the option: the request is this start's
    resumed = fused_node(True, FAKE_SERVER, "--from", "A")
    resumed.restore_gdn()
    resumed.restore_moe_glue(read_json(runner.run_dir / "manifest.json"))
    assert resumed.variant_server_env({})[MOE] == "0"
    assert moe_line(resumed) == "- MoE glue: off (not requested)"


def test_a_resume_from_a_run_made_before_the_option_existed(fused_node):
    earlier = fused_node(True, FAKE_SERVER)
    gdn_dir = earlier.run_dir / "fused_gdn"
    gdn_dir.mkdir(parents=True, exist_ok=True)
    # the hyper-connection kernels' record is there, the MoE glue's is not: neither in G1b's details ...
    (gdn_dir / "details.json").write_text(json.dumps(dict(GDN_PASSED, hc_fused=dict(HC_PASSED, requested=True))),
                                          encoding="utf-8")
    # ... nor in the manifest
    (earlier.run_dir / "manifest.json").write_text(json.dumps({"run_id": "r", "hc_fused": {"disabled_by": None}}),
                                                   encoding="utf-8")

    runner = fused_node(True, FAKE_SERVER, "--moe-glue", "--hc-fused", "--from", "A")
    runner.restore_gdn()
    runner.restore_hc_fused(read_json(runner.run_dir / "manifest.json"))
    runner.restore_moe_glue(read_json(runner.run_dir / "manifest.json"))
    assert runner.gdn_mode == gates.GDN_MODE_FUSED
    assert runner.hc_fused_on() is True  # as before
    assert runner.moe_glue_on() is False
    assert runner.variant_server_env({})[MOE] == "0"
    assert moe_line(runner) == "- MoE glue: off (the card test has not run)"


def test_the_run_itself_restores_the_decisions_on_a_resume(fused_node, gate_calls, capsys):
    """Through Runner.run, not the restore functions: `--from report` must not lose who took the kernels out."""
    runner = fused_node(True, FAKE_SERVER, "--moe-glue", "--hc-fused")
    runner.run_gate("fused_gdn")
    runner.moe_glue_disabled_by = "tiny"
    runner.hc_fused_disabled_by = "tiny"
    runner.write_manifest()

    resumed = fused_node(True, FAKE_SERVER, "--moe-glue", "--hc-fused", "--from", "report")
    resumed.run()
    capsys.readouterr()

    manifest = read_json(resumed.run_dir / "manifest.json")
    assert manifest["moe_glue"]["status"] == "passed" and manifest["moe_glue"]["disabled_by"] == "tiny"
    assert manifest["moe_glue"]["on"] is False and manifest["moe_glue"]["requested"] is True
    assert manifest["hc_fused"]["disabled_by"] == "tiny" and manifest["hc_fused"]["on"] is False
    md = (resumed.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "- MoE glue: off (the tiny gate failed with it)" in md
    assert "- HC fused: off (the tiny gate failed with it)" in md


# --------------------------------------------------------------------------- both kernel families at once


@pytest.mark.parametrize("failed,env_flags,lines", [
    ("hc", {HC: "0", MOE: "1"}, ["- HC fused: off (test_hpu_op.py failed)", "- MoE glue: on"]),
    ("moe", {HC: "1", MOE: "0"}, ["- HC fused: on", "- MoE glue: off (test_moe_hpu_op.py failed)"]),
])
def test_both_requested_and_one_card_test_fails(fused_node, gate_calls, failed, env_flags, lines):
    """Each card test decides its own kernels: the one that failed stays off, the other one is on."""
    calls, answers = gate_calls
    answers[failed] = dict(HC_FAILED if failed == "hc" else MOE_FAILED)
    runner = fused_node(True, FAKE_SERVER, "--hc-fused", "--moe-glue")

    assert runner.run_gate("fused_gdn") is True

    assert calls["order"] == ["gdn", "hc", "moe"]  # a failed first card test does not stop the second
    details = read_json(runner.run_dir / "fused_gdn" / "details.json")
    assert details["hc_fused"]["requested"] is True and details["moe_glue"]["requested"] is True
    assert details["hc_fused"]["status"] == ("failed" if failed == "hc" else "passed")
    assert details["moe_glue"]["status"] == ("failed" if failed == "moe" else "passed")
    assert {name: details["env"][name] for name in (HC, MOE)} == env_flags
    env = runner.variant_server_env({})
    assert {name: env[name] for name in (HC, MOE)} == env_flags
    tiny_env = runner.tiny_env()
    assert {name: tiny_env[name] for name in (HC, MOE)} == env_flags
    assert (runner.hc_fused_on(), runner.moe_glue_on()) == (failed != "hc", failed != "moe")
    md = summary_md(runner).splitlines()
    at = md.index(lines[0])
    assert md[at + 1] == lines[1]
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["hc_fused"]["on"] is (failed != "hc") and manifest["moe_glue"]["on"] is (failed != "moe")


def test_both_requested_and_both_card_tests_pass(fused_node, gate_calls):
    runner = fused_node(True, FAKE_SERVER, "--hc-fused", "--moe-glue")
    runner.run_gate("fused_gdn")
    env = runner.variant_server_env({})
    assert env[HC] == "1" and env[MOE] == "1"
    assert serve.RECIPE_CACHE_ENV in env  # through the combined kernel database
    assert hc_line(runner) == "- HC fused: on" and moe_line(runner) == "- MoE glue: on"


def test_tiny_gate_with_both_on_turns_both_off_in_one_retry(fused_node, gate_calls, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s,%s,%s" % (KILL_SWITCH, serve.RECIPE_CACHE_ENV, HC, MOE))
    runner = fused_node(True, BROKEN_WITH_EITHER, "--hc-fused", "--moe-glue")
    runner.run_gate("fused_gdn")
    assert runner.hc_fused_on() is True and runner.moe_glue_on() is True

    assert runner.run_gate("tiny") is True

    assert len(runner.builds) == 2  # one retry with both off, never one per family
    tiny = runner.run_dir / "tiny"
    kept = tiny / "attempt-moe-glue"  # the folder of an attempt that ran the MoE glue, alone or not
    assert read_json(kept / "status.json")["status"] == int(GateStatus.FAILED)
    kept_details = read_json(kept / "details.json")
    assert kept_details["hc_fused"] is True and kept_details["moe_glue"] is True
    assert (kept / "server0.log").is_file()
    assert not (tiny / "attempt-hc-fused").exists() and not (tiny / "attempt-combined").exists()

    assert read_json(tiny / "status.json")["status"] == int(GateStatus.PASSED)
    details = read_json(tiny / "details.json")
    assert details["moe_glue"] is False and details["moe_glue_disabled_by"] == "tiny"
    assert details["attempt_moe_glue"] == {"dir": "attempt-moe-glue", "status": int(GateStatus.FAILED),
                                           "reason": read_json(kept / "status.json")["reason"], "errors": []}
    assert details["hc_fused"] is False and details["hc_fused_disabled_by"] == "tiny"
    assert "attempt_hc_fused" not in details  # the one kept attempt is recorded under the MoE glue's keys
    second_env = read_json(tiny / "server_env.json")
    assert second_env[HC] == "0" and second_env[MOE] == "0"
    assert second_env[KILL_SWITCH] is None and second_env[serve.RECIPE_CACHE_ENV] is not None

    assert runner.hc_fused_disabled_by == "tiny" and runner.moe_glue_disabled_by == "tiny"
    assert runner.combined_db_disabled_by is None
    env = runner.variant_server_env({})
    assert env[HC] == "0" and env[MOE] == "0"  # every later server
    manifest = read_json(runner.run_dir / "manifest.json")
    assert manifest["hc_fused"]["disabled_by"] == "tiny" and manifest["moe_glue"]["disabled_by"] == "tiny"
    assert hc_line(runner) == "- HC fused: " + TINY_OFF
    assert moe_line(runner) == "- MoE glue: " + TINY_OFF
    out = capsys.readouterr().out
    assert "tiny: FAILED with the fused hyper-connection kernels and the MoE glue kernels (" in out
    assert "running the gate once more with %s=0 and %s=0" % (HC, MOE) in out
    assert out.count("running the gate once more") == 1


def test_tiny_gate_with_both_on_and_a_broken_combined_database_runs_three_times(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s,%s" % (KILL_SWITCH, HC, MOE))
    runner = fused_node(True, BROKEN_WITH_EITHER_AND_COMBINED, "--hc-fused", "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    assert len(runner.builds) == 3  # the kernels (both families at once), then the combined database
    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-moe-glue").is_dir() and (tiny / "attempt-combined").is_dir()
    assert not (tiny / "attempt-hc-fused").exists()
    env = read_json(tiny / "server_env.json")
    assert env[HC] == "0" and env[MOE] == "0" and env[KILL_SWITCH] == "0"
    details = read_json(tiny / "details.json")
    assert details["hc_fused_disabled_by"] == "tiny" and details["moe_glue_disabled_by"] == "tiny"
    assert details["combined_kernel_db_disabled_by"] == "tiny"
    assert details["attempt_moe_glue"]["dir"] == "attempt-moe-glue"
    assert details["attempt_combined"]["dir"] == "attempt-combined"
    combined_details = read_json(tiny / "attempt-combined" / "details.json")
    assert combined_details["hc_fused"] is False and combined_details["moe_glue"] is False


def test_both_requested_but_only_the_hc_kernels_on_keeps_the_hc_folder(fused_node, gate_calls, monkeypatch):
    """The MoE glue's card test failed, so the tiny server ran the hyper-connection kernels only: the
    attempt is kept as before, and the tiny gate is not what took the MoE glue out."""
    _, answers = gate_calls
    answers["moe"] = dict(MOE_FAILED)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s" % (HC, MOE))
    runner = fused_node(True, BROKEN_WITH_HC_FUSED, "--hc-fused", "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-hc-fused").is_dir() and not (tiny / "attempt-moe-glue").exists()
    details = read_json(tiny / "details.json")
    assert details["hc_fused_disabled_by"] == "tiny" and details["attempt_hc_fused"]["dir"] == "attempt-hc-fused"
    assert details["moe_glue"] is False
    assert "moe_glue_disabled_by" not in details and "attempt_moe_glue" not in details
    assert runner.hc_fused_disabled_by == "tiny" and runner.moe_glue_disabled_by is None
    assert hc_line(runner) == "- HC fused: " + TINY_OFF
    assert moe_line(runner) == "- MoE glue: off (test_moe_hpu_op.py failed)"
    assert len(runner.builds) == 2


def test_both_requested_but_only_the_moe_glue_on_leaves_the_hc_record_alone(fused_node, gate_calls, monkeypatch):
    _, answers = gate_calls
    answers["hc"] = dict(HC_FAILED)
    runner = fused_node(True, BROKEN_WITH_MOE_GLUE, "--hc-fused", "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True

    tiny = runner.run_dir / "tiny"
    assert (tiny / "attempt-moe-glue").is_dir() and not (tiny / "attempt-hc-fused").exists()
    details = read_json(tiny / "details.json")
    assert details["moe_glue_disabled_by"] == "tiny" and "hc_fused_disabled_by" not in details
    assert runner.moe_glue_disabled_by == "tiny" and runner.hc_fused_disabled_by is None
    assert hc_line(runner) == "- HC fused: off (test_hpu_op.py failed)"
    assert moe_line(runner) == "- MoE glue: " + TINY_OFF


def test_a_resumed_run_restores_both(fused_node, gate_calls):
    runner = fused_node(True, FAKE_SERVER, "--hc-fused", "--moe-glue")
    runner.run_gate("fused_gdn")
    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")

    resumed = fused_node(True, FAKE_SERVER, "--hc-fused", "--moe-glue", "--from", "A")
    assert (resumed.hc_fused_on(), resumed.moe_glue_on()) == (False, False)  # nothing restored yet
    resumed.restore_gdn()
    resumed.restore_hc_fused(manifest)
    resumed.restore_moe_glue(manifest)
    env = resumed.variant_server_env({})
    assert (resumed.hc_fused_on(), resumed.moe_glue_on()) == (True, True)
    assert env[HC] == "1" and env[MOE] == "1"

    # the tiny gate of the earlier start took both out: both stay out
    runner.hc_fused_disabled_by = "tiny"
    runner.moe_glue_disabled_by = "tiny"
    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")
    resumed = fused_node(True, FAKE_SERVER, "--hc-fused", "--moe-glue", "--from", "A")
    resumed.restore_gdn()
    resumed.restore_hc_fused(manifest)
    resumed.restore_moe_glue(manifest)
    env = resumed.variant_server_env({})
    assert (resumed.hc_fused_disabled_by, resumed.moe_glue_disabled_by) == ("tiny", "tiny")
    assert env[HC] == "0" and env[MOE] == "0"

    # ... and a decision about one is not a decision about the other
    runner.hc_fused_disabled_by = None
    runner.write_manifest()
    manifest = read_json(runner.run_dir / "manifest.json")
    resumed = fused_node(True, FAKE_SERVER, "--hc-fused", "--moe-glue", "--from", "A")
    resumed.restore_gdn()
    resumed.restore_hc_fused(manifest)
    resumed.restore_moe_glue(manifest)
    env = resumed.variant_server_env({})
    assert env[HC] == "1" and env[MOE] == "0"


# --------------------------------------------------------------------------- fallback lines in a variant

HC_OFF = ("(Worker_TP0 pid=3) WARNING [hyperconnection.py:250] HC fused (VLLM_HPU_HC_FUSED=1) is off for "
          "language_model.model.layers.0.attn_hyper_connection: hidden states are 3-D, the fused ops take "
          "[tokens, width]; using the torch path\n")
HC_NO_KERNEL = ("(Worker_TP1 pid=4) WARNING [__init__.py:140] HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel: the "
                "hc_fused_ops extension did not load (x); every hyper-connection runs the torch path.\n")
GDN_OFF = ("WARNING [qwen3_5.py:306] GDN core decode (VLLM_HPU_GDN_CORE_DECODE=1) is off for layers.1: conv1d has a "
           "bias; using the torch decode path\n")
MOE_OFF = ("(Worker_TP0 pid=3) WARNING [hpu_fused_moe.py:357] MoE glue (VLLM_HPU_MOE_GLUE=1) is off for "
           "language_model.model.layers.%d.mlp: the FP8 experts have a static input scale; using the torch path\n")
MOE_NO_KERNEL = ("(Worker_TP1 pid=4) WARNING [__init__.py:197] MoE glue (VLLM_HPU_MOE_GLUE=1) has no HPU kernel: the "
                 "kernels of the hc_fused_ops extension are not usable (x); every MoE block runs the torch path. "
                 "Build them with tools/build_flashinfer_gaudi.py --hc-fused-only, or set VLLM_HPU_MOE_GLUE=0.\n")


def fallback_summary(tmp_path, log_text: str):
    run = tmp_path / "run"
    (run / "fused_gdn").mkdir(parents=True)
    (run / "fused_gdn" / "details.json").write_text(json.dumps({"mode": 1, "reason": None, "env": {}}),
                                                    encoding="utf-8")
    vdir = run / "A"
    vdir.mkdir()
    (vdir / "server0.log").write_text(log_text, encoding="utf-8")
    args = rental_main._parse_args(["--mode", "stub", "--model", str(tmp_path / "m"), "--results", str(tmp_path),
                                    "--run-id", "run", "--variants", "A"])
    runner = rental_main.Runner(args, rental_main.load_variants())
    runner.write_gdn_fallback(vdir, [{}])
    lines = (vdir / "gdn_fallback.txt").read_text(encoding="utf-8").splitlines()
    return lines, rental_main.gdn_summary_lines(run, ["A"])


def test_the_variants_fallback_file_lists_the_moe_lines_and_the_summary_counts_them(tmp_path):
    lines, summary = fallback_summary(
        tmp_path, "INFO ok\n" + HC_OFF + MOE_OFF % 0 + GDN_OFF + MOE_NO_KERNEL + HC_NO_KERNEL + MOE_OFF % 1)

    assert len(lines) == 6 and all(line.startswith("server0: ") for line in lines)
    assert sum("MoE glue" in line for line in lines) == 3 and sum("HC fused" in line for line in lines) == 2
    assert summary == [
        "GDN decode: fused kernel (G1b passed)",
        "GDN fallback lines in A: 1 (the kernel did not run on every layer)",
        "HC fused fallback lines in A: 2 (the hyper-connection kernels did not run on every layer)",
        "MoE glue fallback lines in A: 3 (the MoE glue kernels did not run on every layer)",
    ]


def test_moe_lines_alone_are_not_counted_as_gdn_or_hc_lines(tmp_path):
    lines, summary = fallback_summary(tmp_path, "INFO ok\n" + MOE_NO_KERNEL + MOE_OFF % 7)

    assert len(lines) == 2
    assert summary == [
        "GDN decode: fused kernel (G1b passed)",
        "MoE glue fallback lines in A: 2 (the MoE glue kernels did not run on every layer)",
    ]


def test_hc_and_gdn_lines_alone_are_counted_as_before(tmp_path):
    _, summary = fallback_summary(tmp_path, HC_OFF + GDN_OFF + HC_NO_KERNEL)

    assert summary == [
        "GDN decode: fused kernel (G1b passed)",
        "GDN fallback lines in A: 1 (the kernel did not run on every layer)",
        "HC fused fallback lines in A: 2 (the hyper-connection kernels did not run on every layer)",
    ]


def test_the_tiny_gates_details_record_the_moe_lines_of_its_server(fused_node, gate_calls, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_LINE", (MOE_OFF % 2).strip())
    runner = fused_node(True, FAKE_SERVER, "--moe-glue")
    runner.run_gate("fused_gdn")

    assert runner.run_gate("tiny") is True  # recorded only: the tokens match on the torch path

    recorded = read_json(runner.run_dir / "tiny" / "details.json")["gdn_fallback_lines"]
    assert recorded["count"] == 1 and "MoE glue (VLLM_HPU_MOE_GLUE=1) is off for" in recorded["first"][0]
