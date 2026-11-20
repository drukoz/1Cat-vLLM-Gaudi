# SPDX-License-Identifier: Apache-2.0
"""Tests for the tiny smoke test's `moe` mode: the `fused` mode plus the MoE glue kernels
(VLLM_HPU_MOE_GLUE=1). The mode is not in the default list; `--gdn moe` or `--gdn pytorch,fused,hc,moe`
selects it. It fails when the server log says a MoE block ran the torch path, as `hc` does for a
hyper-connection and `fused` for the GDN decode line; a line of one kernel family never fails the mode
of another."""

from __future__ import annotations

import json

import pytest

from tools.flash_next.rental import gates
from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental import serve, tiny_smoke

from .test_tiny_smoke import CACHE_KEY, by_mode, fakes, run  # noqa: F401
from .test_tiny_smoke_hc import DECODE_LINE
from .test_tiny_smoke_hc import OFF_LINE as HC_OFF_LINE

HC = gates.HC_FUSED_ENV
MOE = gates.MOE_GLUE_ENV
OFF_LINE = ("(Worker_TP2 pid=9) WARNING 11-20 09:00:00 [hpu_fused_moe.py:357] MoE glue (VLLM_HPU_MOE_GLUE=1) is off "
            "for language_model.model.layers.0.mlp: no hpu kernel for moe_route, moe_combine (the hc_fused_ops "
            "extension is not loaded); using the torch path")
NO_KERNEL_LINE = ("(Worker_TP0 pid=7) WARNING 11-20 09:00:00 [__init__.py:197] MoE glue (VLLM_HPU_MOE_GLUE=1) has no "
                  "HPU kernel: the kernels of the hc_fused_ops extension are not usable (x.so: file does not exist); "
                  "every MoE block runs the torch path. Build them with tools/build_flashinfer_gaudi.py "
                  "--hc-fused-only, or set VLLM_HPU_MOE_GLUE=0.")


@pytest.fixture
def smoke(fakes, monkeypatch):  # noqa: F811
    """The fake builder and server of test_tiny_smoke.py; the server also records the two kernel flags."""
    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: False)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS",
                       "%s,%s,%s,%s" % (HC, MOE, serve.RECIPE_CACHE_ENV, serve.COMBINED_KERNEL_DB_ENV))
    monkeypatch.delenv(HC, raising=False)
    monkeypatch.delenv(MOE, raising=False)
    return fakes


def server_env(out, mode: str) -> dict:
    return json.loads((out / mode / "server_env.json").read_text(encoding="utf-8"))


def test_the_mode_is_the_fused_mode_plus_the_flag():
    assert tiny_smoke.GDN_MODES["moe"] == dict(tiny_smoke.GDN_MODES["fused"], **{MOE: "1"})
    assert tiny_smoke.GDN_MODES["moe"] == {"VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_GDN_CORE_VERIFY": "0", MOE: "1"}
    for other in ("pytorch", "fused", "hc"):
        assert MOE not in tiny_smoke.GDN_MODES[other], other
    assert HC not in tiny_smoke.GDN_MODES["moe"]  # one kernel family per mode
    assert tiny_smoke.GDN_MODES["hc"] == dict(tiny_smoke.GDN_MODES["fused"], **{HC: "1"})  # as before
    assert list(tiny_smoke.GDN_MODES) == ["pytorch", "fused", "hc", "moe"]


def test_which_checks_apply_to_the_mode():
    assert "moe" in tiny_smoke.DECODE_KERNEL_MODES and "hc" in tiny_smoke.DECODE_KERNEL_MODES  # the GDN decode line
    assert tiny_smoke.MOE_KERNEL_MODES == ("moe", )
    assert tiny_smoke.HC_KERNEL_MODES == ("hc", )


def test_the_default_modes_do_not_include_it(smoke):
    out, port = smoke
    assert tiny_smoke.DEFAULT_MODES == "pytorch,fused"
    code, summary = run(out, port)
    assert code == 0
    assert list(by_mode(summary)) == ["pytorch", "fused"]
    assert not (out / "moe").exists()
    assert server_env(out, "pytorch")[MOE] is None and server_env(out, "fused")[MOE] is None  # the plugin default


def test_moe_mode_serves_with_the_kernels_and_passes(smoke, capsys):
    out, port = smoke
    code, summary = run(out, port, "--gdn", "moe")
    assert code == 0
    assert list(by_mode(summary)) == ["moe"]
    assert by_mode(summary)["moe"]["status"] == "passed"
    env = server_env(out, "moe")
    assert env[MOE] == "1"
    assert env[HC] is None  # not this mode's kernels: the plugin default
    assert (env["VLLM_HPU_GDN_CORE_DECODE"], env["VLLM_HPU_GDN_CORE_VERIFY"]) == ("1", "0")
    assert env["PT_HPU_LAZY_MODE"] == "1" and env["VLLM_HPU_FLASH_NEXT_QSA_MODE"] == "1"
    assert "tiny smoke [moe]: PASSED" in capsys.readouterr().out


def test_the_four_modes_in_the_order_given(smoke):
    out, port = smoke
    code, summary = run(out, port, "--gdn", "pytorch,fused,hc,moe")
    assert code == 0
    modes = ["pytorch", "fused", "hc", "moe"]
    assert list(by_mode(summary)) == modes
    assert [server_env(out, mode)[MOE] for mode in modes] == [None, None, None, "1"]
    assert [server_env(out, mode)[HC] for mode in modes] == [None, None, "1", None]


@pytest.mark.parametrize("line", [OFF_LINE, NO_KERNEL_LINE], ids=["is_off_for", "has_no_hpu_kernel"])
def test_moe_mode_fails_when_a_moe_block_ran_the_torch_path(smoke, monkeypatch, capsys, line):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", line)
    code, summary = run(out, port, "--gdn", "moe")
    assert code == 1  # the tokens match (the torch path is right), but the kernels did not run
    entry = by_mode(summary)["moe"]
    assert entry["status"] == "failed"
    assert entry["reason"] == "the MoE glue kernels did not run: 1 log line(s) say a layer used the torch path"
    assert entry["fallback_lines"] == [line]
    details = json.loads((out / "moe" / "details.json").read_text(encoding="utf-8"))
    assert details["mode"] == "moe" and details["reason"] == entry["reason"]
    printed = capsys.readouterr().out
    assert "tiny smoke [moe]: FAILED" in printed and "MoE glue (VLLM_HPU_MOE_GLUE=1)" in printed


def test_moe_mode_also_fails_on_the_gdn_decode_fallback(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", DECODE_LINE)
    code, summary = run(out, port, "--gdn", "moe")
    assert code == 1
    entry = by_mode(summary)["moe"]
    assert entry["status"] == "failed" and "torch decode path" in entry["reason"]
    assert entry["fallback_lines"] == [DECODE_LINE]


def test_a_moe_line_does_not_fail_the_other_modes(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", OFF_LINE)
    code, summary = run(out, port, "--gdn", "pytorch,fused,hc")
    assert code == 0  # these modes do not ask for the MoE glue kernels; to `hc` it is not a hyper-connection line
    assert [entry["status"] for entry in summary["modes"]] == ["passed", "passed", "passed"]


def test_a_hyper_connection_line_does_not_fail_the_moe_mode(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", HC_OFF_LINE)
    code, summary = run(out, port, "--gdn", "moe,hc")
    assert code == 1
    modes = by_mode(summary)
    assert modes["moe"]["status"] == "passed" and modes["moe"]["fallback_lines"] == []
    assert modes["hc"]["status"] == "failed"  # the line's own mode still sees it
    assert "hyper-connection kernels did not run" in modes["hc"]["reason"]


def test_moe_mode_with_wrong_tokens_reports_the_tokens(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    monkeypatch.setenv("FAKE_SERVER_LINE", OFF_LINE)
    code, summary = run(out, port, "--gdn", "moe")
    assert code == 1
    assert "the shortest failing prompt has 1 tokens" in by_mode(summary)["moe"]["reason"]


def test_moe_mode_gets_the_recipe_cache_only_with_the_combined_kernel_database(smoke, tmp_path, monkeypatch):
    out, port = smoke
    cache = tmp_path / "rc"
    code, summary = run(out, port, "--gdn", "moe", "--recipe-cache", str(cache))
    assert code == 0
    assert server_env(out, "moe")[serve.RECIPE_CACHE_ENV] is None  # its kernel database is in a colon list
    assert by_mode(summary)["moe"]["recipe_cache"] == serve.RECIPE_CACHE_OFF_REASON

    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: True)
    out2 = tmp_path / "out2"
    code, summary = run(out2, port, "--gdn", "moe", "--recipe-cache", str(cache))
    assert code == 0
    assert server_env(out2, "moe")[serve.RECIPE_CACHE_ENV] == "%s/%s,false,8192" % (cache, CACHE_KEY)


def test_env_pairs_can_still_override_the_mode(smoke):
    out, port = smoke
    code, _ = run(out, port, "--gdn", "moe", "--env", "%s=0" % MOE)
    assert code == 0
    assert server_env(out, "moe")[MOE] == "0"


def test_env_pairs_can_add_the_other_kernels_to_the_mode(smoke):
    """Both kernel families in one server, as `run_rental.sh --hc-fused --moe-glue` serves them."""
    out, port = smoke
    code, _ = run(out, port, "--gdn", "moe", "--env", "%s=1" % HC)
    assert code == 0
    env = server_env(out, "moe")
    assert env[MOE] == "1" and env[HC] == "1"


def test_unknown_mode_names_the_four_modes(smoke, capsys):
    out, port = smoke
    with pytest.raises(SystemExit) as info:
        tiny_smoke.main(["--out", str(out), "--port", str(port), "--gdn", "pytorch,moeglue"])
    assert info.value.code == 2
    assert "pytorch, fused, hc, moe" in capsys.readouterr().err


def test_the_help_and_the_module_text_name_the_mode(capsys):
    with pytest.raises(SystemExit) as info:
        tiny_smoke.main(["--help"])
    assert info.value.code == 0
    usage = " ".join(capsys.readouterr().out.split())
    assert "pytorch, fused, hc, moe" in usage and "VLLM_HPU_MOE_GLUE=1" in usage
    assert "VLLM_HPU_HC_FUSED=1" in usage  # the hc mode's words stay
    doc = " ".join(tiny_smoke.__doc__.split())
    assert "`moe` is `fused` plus the MoE glue kernels (`VLLM_HPU_MOE_GLUE=1`)" in doc
    assert "MoE glue ... is off for" in doc and "MoE glue ... has no HPU kernel" in doc
    assert "--gdn pytorch,fused,hc,moe" in doc


def test_a_passed_moe_mode_records_and_prints_its_decode_time(smoke, capsys):
    out, port = smoke
    code, summary = run(out, port, "--gdn", "moe")
    assert code == 0
    value = by_mode(summary)["moe"]["decode_ms_per_token"]
    assert isinstance(value, float) and value >= 0.0
    assert "tiny smoke [moe]: decode %.1f ms/token" % value in capsys.readouterr().out


def test_a_moe_mode_that_fell_back_keeps_no_decode_time_claim(smoke, monkeypatch, capsys):
    """The tokens matched, so the time was measured, but the mode failed: the PASSED line and its decode
    line are not printed for kernels that did not run."""
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", NO_KERNEL_LINE)
    code, _ = run(out, port, "--gdn", "moe")
    assert code == 1
    printed = capsys.readouterr().out
    assert "tiny smoke [moe]: PASSED" not in printed and "ms/token" not in printed
