# SPDX-License-Identifier: Apache-2.0
"""Tests for the tiny smoke test's `hc` mode: the `fused` mode plus the fused hyper-connection kernels
(VLLM_HPU_HC_FUSED=1). The mode is not in the default list; `--gdn pytorch,fused,hc` selects it. It fails
when the server log says a hyper-connection ran the torch path, as `fused` does for the GDN decode line."""

from __future__ import annotations

import json

import pytest

from tools.flash_next.rental import gates
from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental import serve, tiny_smoke

from .test_tiny_smoke import CACHE_KEY, by_mode, fakes, run  # noqa: F401

HC = gates.HC_FUSED_ENV
OFF_LINE = ("(Worker_TP2 pid=9) WARNING 11-19 21:00:00 [hyperconnection.py:252] HC fused (VLLM_HPU_HC_FUSED=1) is off "
            "for language_model.model.layers.0.attn_hyper_connection: the hc_fused_ops extension is not loaded (no "
            "hpu kernel for hc_norm, hc_silu, hc_silu_split, hc_gate_mix, hc_combine, hc_combine_norm); using the "
            "torch path")
NO_KERNEL_LINE = ("(Worker_TP0 pid=7) WARNING 11-19 21:00:00 [__init__.py:141] HC fused (VLLM_HPU_HC_FUSED=1) has no "
                  "HPU kernel: the hc_fused_ops extension did not load (x.so: file does not exist); every "
                  "hyper-connection runs the torch path.")
DECODE_LINE = ("WARNING 11-18 22:26:38 [__init__.py:119] GDN core decode (VLLM_HPU_GDN_CORE_DECODE=1) has no HPU "
               "kernel: the gdn_fused_ops extension is not loaded")


@pytest.fixture
def smoke(fakes, monkeypatch):  # noqa: F811
    """The fake builder and server of test_tiny_smoke.py; the server also records the hyper-connection flag."""
    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: False)
    monkeypatch.setenv("FAKE_SERVER_ENV_KEYS", "%s,%s,%s" % (HC, serve.RECIPE_CACHE_ENV, serve.COMBINED_KERNEL_DB_ENV))
    monkeypatch.delenv(HC, raising=False)
    return fakes


def server_env(out, mode: str) -> dict:
    return json.loads((out / mode / "server_env.json").read_text(encoding="utf-8"))


def test_the_mode_is_the_fused_mode_plus_the_flag():
    assert tiny_smoke.GDN_MODES["hc"] == dict(tiny_smoke.GDN_MODES["fused"], **{HC: "1"})
    assert tiny_smoke.GDN_MODES["hc"] == {"VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_GDN_CORE_VERIFY": "0", HC: "1"}
    assert HC not in tiny_smoke.GDN_MODES["pytorch"] and HC not in tiny_smoke.GDN_MODES["fused"]


def test_the_default_modes_do_not_include_it(smoke):
    out, port = smoke
    assert tiny_smoke.DEFAULT_MODES == "pytorch,fused"
    code, summary = run(out, port)
    assert code == 0
    assert list(by_mode(summary)) == ["pytorch", "fused"]
    assert not (out / "hc").exists()
    assert server_env(out, "pytorch")[HC] is None and server_env(out, "fused")[HC] is None  # the plugin default


def test_hc_mode_serves_with_the_kernels_and_passes(smoke, capsys):
    out, port = smoke
    code, summary = run(out, port, "--gdn", "hc")
    assert code == 0
    assert list(by_mode(summary)) == ["hc"]
    assert by_mode(summary)["hc"]["status"] == "passed"
    env = server_env(out, "hc")
    assert env[HC] == "1"
    assert (env["VLLM_HPU_GDN_CORE_DECODE"], env["VLLM_HPU_GDN_CORE_VERIFY"]) == ("1", "0")
    assert env["PT_HPU_LAZY_MODE"] == "1" and env["VLLM_HPU_FLASH_NEXT_QSA_MODE"] == "1"
    assert "tiny smoke [hc]: PASSED" in capsys.readouterr().out


def test_the_three_modes_in_the_order_given(smoke):
    out, port = smoke
    code, summary = run(out, port, "--gdn", "pytorch,fused,hc")
    assert code == 0
    assert list(by_mode(summary)) == ["pytorch", "fused", "hc"]
    assert [server_env(out, mode)[HC] for mode in ("pytorch", "fused", "hc")] == [None, None, "1"]


@pytest.mark.parametrize("line", [OFF_LINE, NO_KERNEL_LINE], ids=["is_off_for", "has_no_hpu_kernel"])
def test_hc_mode_fails_when_a_hyper_connection_ran_the_torch_path(smoke, monkeypatch, capsys, line):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", line)
    code, summary = run(out, port, "--gdn", "hc")
    assert code == 1  # the tokens match (the torch path is right), but the kernels did not run
    entry = by_mode(summary)["hc"]
    assert entry["status"] == "failed"
    assert entry["reason"] == ("the fused hyper-connection kernels did not run: 1 log line(s) say a layer used the "
                               "torch path")
    assert entry["fallback_lines"] == [line]
    printed = capsys.readouterr().out
    assert "tiny smoke [hc]: FAILED" in printed and "HC fused (VLLM_HPU_HC_FUSED=1)" in printed


def test_hc_mode_also_fails_on_the_gdn_decode_fallback(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", DECODE_LINE)
    code, summary = run(out, port, "--gdn", "hc")
    assert code == 1
    entry = by_mode(summary)["hc"]
    assert entry["status"] == "failed" and "torch decode path" in entry["reason"]
    assert entry["fallback_lines"] == [DECODE_LINE]


def test_an_hc_line_does_not_fail_the_other_modes(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_LINE", OFF_LINE)
    code, summary = run(out, port, "--gdn", "pytorch,fused")
    assert code == 0  # these modes do not ask for the hyper-connection kernels
    assert [entry["status"] for entry in summary["modes"]] == ["passed", "passed"]


def test_hc_mode_with_wrong_tokens_reports_the_tokens(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    monkeypatch.setenv("FAKE_SERVER_LINE", OFF_LINE)
    code, summary = run(out, port, "--gdn", "hc")
    assert code == 1
    assert "the shortest failing prompt has 1 tokens" in by_mode(summary)["hc"]["reason"]


def test_hc_mode_gets_the_recipe_cache_only_with_the_combined_kernel_database(smoke, tmp_path, monkeypatch):
    out, port = smoke
    cache = tmp_path / "rc"
    code, summary = run(out, port, "--gdn", "hc", "--recipe-cache", str(cache))
    assert code == 0
    assert server_env(out, "hc")[serve.RECIPE_CACHE_ENV] is None  # its kernel database is in a colon list
    assert by_mode(summary)["hc"]["recipe_cache"] == serve.RECIPE_CACHE_OFF_REASON

    monkeypatch.setattr(orchestrator, "combined_kernel_db_installed", lambda *args, **kwargs: True)
    out2 = tmp_path / "out2"
    code, summary = run(out2, port, "--gdn", "hc", "--recipe-cache", str(cache))
    assert code == 0
    assert server_env(out2, "hc")[serve.RECIPE_CACHE_ENV] == "%s/%s,false,8192" % (cache, CACHE_KEY)


def test_env_pairs_can_still_override_the_mode(smoke):
    out, port = smoke
    code, _ = run(out, port, "--gdn", "hc", "--env", "%s=0" % HC)
    assert code == 0
    assert server_env(out, "hc")[HC] == "0"


def test_unknown_mode_names_the_three_modes(smoke, capsys):
    out, port = smoke
    with pytest.raises(SystemExit) as info:
        tiny_smoke.main(["--out", str(out), "--port", str(port), "--gdn", "pytorch,hcfused"])
    assert info.value.code == 2
    assert "pytorch, fused, hc" in capsys.readouterr().err


# --------------------------------------------------------------------------- the decode time of a passed mode


def write_reference(tmp_path) -> "Path":  # noqa: F821
    path = tmp_path / "reference.json"
    prompts = [{"prompt_ids": list(range(1, n + 1)), "generated_ids": list(range(16))} for n in (7, 1, 33)]
    path.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
    return path


def test_decode_time_is_the_difference_of_a_long_and_a_short_request(tmp_path):
    """16 tokens against 1 token of the same prompt: prefill and the HTTP round trip cancel."""
    payloads = []
    now = [0.0]
    costs = {16: [1.9, 1.6, 1.7], 1: [0.4, 0.1, 0.2]}  # seconds per request; the best of each counts

    def post(base_url, path, payload, timeout_s):
        payloads.append((base_url, path, dict(payload), timeout_s))
        now[0] += costs[payload["max_tokens"]].pop(0)
        return {"choices": [{"token_ids": list(range(payload["max_tokens"]))}]}

    value = tiny_smoke.decode_ms_per_token("http://127.0.0.1:1", write_reference(tmp_path), post=post,
                                           clock=lambda: now[0])

    assert value == pytest.approx((1.6 - 0.1) / 15 * 1000)  # 100 ms/token
    assert len(payloads) == 6
    for base_url, path, payload, timeout_s in payloads:
        assert (base_url, path) == ("http://127.0.0.1:1", "/v1/completions")
        assert payload["prompt"] == [1]  # the shortest reference prompt: its graphs are compiled already
        assert payload["temperature"] == 0 and payload["model"] == "tiny" and payload["return_token_ids"] is True
        assert timeout_s == tiny_smoke.DECODE_TIMING_TIMEOUT_S
    assert sorted(p[2]["max_tokens"] for p in payloads) == [1, 1, 1, 16, 16, 16]


def test_decode_time_is_none_when_it_cannot_be_measured(tmp_path):
    reference = write_reference(tmp_path)

    def refuses(*args):
        raise RuntimeError("HTTP 500 from /v1/completions")

    assert tiny_smoke.decode_ms_per_token("http://127.0.0.1:1", reference, post=refuses) is None
    one_token = lambda *args: {"choices": [{"token_ids": [5]}]}  # noqa: E731 - the model stopped at once
    assert tiny_smoke.decode_ms_per_token("http://127.0.0.1:1", reference, post=one_token) is None
    assert tiny_smoke.decode_ms_per_token("http://127.0.0.1:1", tmp_path / "missing.json", post=one_token) is None


def test_a_passed_mode_records_and_prints_its_decode_time(smoke, capsys):
    out, port = smoke
    code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 0
    value = by_mode(summary)["pytorch"]["decode_ms_per_token"]
    assert isinstance(value, float) and value >= 0.0
    assert "decode %.1f ms/token" % value in capsys.readouterr().out


def test_a_failed_measurement_never_fails_the_mode(smoke, monkeypatch, capsys):
    out, port = smoke
    monkeypatch.setattr(tiny_smoke, "decode_ms_per_token", lambda *args, **kwargs: None)
    code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 0
    assert by_mode(summary)["pytorch"]["decode_ms_per_token"] is None
    printed = capsys.readouterr().out
    assert "tiny smoke [pytorch]: PASSED" in printed and "ms/token" not in printed


def test_a_failed_mode_is_not_measured(smoke, monkeypatch):
    out, port = smoke
    monkeypatch.setenv("FAKE_SERVER_WRONG", "1")
    monkeypatch.setattr(tiny_smoke, "decode_ms_per_token", lambda *args, **kwargs: pytest.fail("not measured"))
    code, summary = run(out, port, "--gdn", "pytorch")
    assert code == 1
    assert by_mode(summary)["pytorch"]["decode_ms_per_token"] is None
