# SPDX-License-Identifier: Apache-2.0
"""lookup_tiny.py (the tiny-model card rehearsal of the lookup proposers) on the workstation: its scenarios, the
stream comparison, the server command, and a whole run against stand-ins for the server and its answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.flash_next.rental import lookup_tiny as lt


def _reference():
    return {"prompts": [{"prompt_ids": [1000 * n + i for i in range(n)]} for n in (1, 7, 33, 100, 700, 2600)],
            "max_new_tokens": 16}


def test_scenarios_cover_the_shapes():
    plan = lt.scenarios(_reference(), 48)
    names = [s["name"] for s in plan]
    assert names == ["solo-7", "solo-33", "solo-100", "solo-700", "near-256", "near-512", "batch", "turn-2"]
    batch = plan[names.index("batch")]["requests"]
    assert [r["max_tokens"] for r in batch] == [48, 42, 36, 30] and len({len(r["prompt_ids"]) for r in batch}) == 4
    assert len(plan[names.index("near-256")]["requests"][0]["prompt_ids"]) == 250
    turn = plan[-1]["requests"][0]
    baseline = {"near-256": {"prompt_ids": [1, 2], "tokens": [3, 4], "error": None}}
    assert lt.request_prompt(turn, baseline) == [1, 2, 3, 4] + turn["suffix_ids"]
    assert lt.request_prompt({"prompt_ids": [5]}, {}) == [5]


def test_every_oracle_schedule_fits_its_k():
    for name, config in lt.CONFIGS.items():
        k = config["spec"]["num_speculative_tokens"]
        if config["oracle"]:
            assert max(config["oracle"]["schedule"]) == k and 0 in config["oracle"]["schedule"], name
        assert config["passes"], name
    assert lt.CONFIGS["mtp3-lookup"]["env"] == {lt.LOOKUP_ENV: "1", "VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert lt.CONFIGS["ngram3-fused"]["env"] == {"VLLM_HPU_GDN_CORE_VERIFY": "1"}
    assert "env" not in lt.CONFIGS["ngram8"]  # 9 rows: the torch recurrence (the preset's VERIFY=0)
    assert lt.DEFAULT_CONFIGS.split(",") == ["ngram3", "ngram8", "mtp3-lookup", "ngram3-search"]
    assert [p[1] for p in lt.CONFIGS["mtp3-lookup"]["passes"]] == [1, 2, 0]


def test_top_gaps_reads_token_id_keys():
    choice = {"logprobs": {"top_logprobs": [{"token_id:7": -0.5, "token_id:9": -0.502, "token_id:3": -4.0}, None]}}
    rows = lt.top_gaps(choice)
    assert rows[0][0] == [7, 0.0] and rows[0][1][0] == 9 and rows[0][1][1] == pytest.approx(0.002)
    assert rows[1] == [] and lt.top_gaps({}) == []


def test_compare_tells_ties_from_breaks():
    baseline = {"a": {"tokens": [1, 2, 3, 4], "error": None}, "b": {"tokens": [1, 2, 3, 4], "error": None},
                "c": {"tokens": [1, 2, 3, 4], "error": None}, "d": {"tokens": [1, 2, 3, 4], "error": None},
                "e": {"tokens": [1, 2, 3, 4], "error": None}}
    tops = {k: {"top": [[[1, 0.0]], [[2, 0.0], [9, 0.001]], [[3, 0.0], [8, 0.5]], [[4, 0.0]]]} for k in "abc"}
    streams = {"a": {"tokens": [1, 2, 3, 4], "error": None},  # identical
               "b": {"tokens": [1, 9, 7, 7], "error": None},  # leaves at a near-tie
               "c": {"tokens": [1, 2, 8, 7], "error": None},  # leaves 0.5 nats below the best: a break
               "d": {"tokens": [1, 2, 3, 5], "error": None},  # no log-probability record
               "e": {"tokens": [], "error": "HTTP 500"}}
    out = lt.compare(baseline, tops, streams)
    assert out["requests"] == 4 and out["identical"] == 1 and out["matched"] == 4 + 1 + 2 + 3
    assert out["ties"] == [["b", 1, 9, 0.001]] and out["breaks"] == [["c", 2, 8, 0.5]]
    assert out["unclassified"] == [["d", 3, 5]] and out["errors"] == [["e", "HTTP 500"]]


def test_serve_command(tmp_path):
    (tmp_path / "reference.json").write_text(json.dumps(_reference()))
    cmd = lt.serve_command(tmp_path, 8310, 2, {"method": "ngram", "num_speculative_tokens": 8})
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2" and cmd[cmd.index("--port") + 1] == "8310"
    assert "--enable-prefix-caching" in cmd and cmd[cmd.index("--max-model-len") + 1] == "4096"
    assert json.loads(cmd[cmd.index("--speculative-config") + 1])["num_speculative_tokens"] == 8
    assert "--speculative-config" not in lt.serve_command(tmp_path, 8310, 2, None)


class _FakeServer:
    """Stands in for lt.Server: no process; the log says what a server with these settings would say."""
    started: list = []
    dead: set = set()

    def __init__(self, name, ckpt, out_dir, port, tp, spec, env, health_timeout_s):
        self.name, self.spec, self.env, self.out_dir = name, spec, env, out_dir / name
        self.startup_seconds, self.reason, self.popen = 1.0, None, None
        type(self).started.append((name, spec, dict(env)))

    def __enter__(self):
        if self.name in type(self).dead:
            self.reason = "exited with code 1"
            return None
        return "http://fake/%s" % self.name

    def __exit__(self, *exc):
        return None

    def log_text(self):
        return "WARNING REHEARSAL ORACLE x\n" if lt.ORACLE_ENV in self.env else "INFO started\n"


def _fake_world(monkeypatch, tmp_path, wrong_token_in=None):
    """A checkpoint folder with a reference, lt.Server and the HTTP calls replaced. A server answers every prompt
    with tokens made from the prompt's length (the same on every server: lossless), except the configuration
    ``wrong_token_in``, whose third token is another one."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "reference.json").write_text(json.dumps(_reference()))
    monkeypatch.setattr(lt.orchestrator, "ensure_tiny_checkpoint", lambda *a, **k: None)
    _FakeServer.started, _FakeServer.dead = [], set()
    monkeypatch.setattr(lt, "Server", _FakeServer)
    calls = []

    def post(base_url, path, body, timeout_s):
        calls.append((base_url, body))
        n = len(body["prompt"])
        tokens = [50000 + (n * 7 + i) % 911 for i in range(body["max_tokens"])]
        if wrong_token_in and base_url.endswith("/" + wrong_token_in):
            tokens[2] += 1
        choice = {"token_ids": tokens}
        if body.get("logprobs"):
            choice["logprobs"] = {"top_logprobs": [{"token_id:%d" % t: -0.1, "token_id:%d" % (t + 1): -3.0}
                                                   for t in tokens]}
        return {"choices": [choice]}

    monkeypatch.setattr(lt.gates, "_post_json", post)
    state = {"n": 0}

    def counters(base_url):
        state["n"] += 1
        return {"drafts": 10 * state["n"], "draft_tokens": 30 * state["n"], "accepted": 12 * state["n"],
                "accepted_per_pos": [6 * state["n"], 4 * state["n"], 2 * state["n"]]}

    monkeypatch.setattr(lt, "counters", counters)
    return ckpt, calls


def test_whole_run_passes_on_lossless_servers(monkeypatch, tmp_path, capsys):
    ckpt, calls = _fake_world(monkeypatch, tmp_path)
    out = tmp_path / "out"
    rc = lt.main(["--ckpt", str(ckpt), "--out", str(out), "--tp", "2", "--port", "8310"])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    record = json.loads((out / "lookup_tiny.json").read_text())
    assert record["baseline"]["status"] == "passed" and len(record["baseline"]["streams"]) == 11
    assert [c["config"] for c in record["configs"]] == lt.DEFAULT_CONFIGS.split(",")
    assert all(c["status"] == "passed" for c in record["configs"])
    hybrid = record["configs"][2]
    assert [p["pass"] for p in hybrid["passes"]] == ["hybrid", "lookup-only", "mtp"]
    assert hybrid["passes"][0]["acceptance"]["acceptance_length"] == pytest.approx(2.2)
    assert hybrid["passes"][0]["compare"]["identical"] == 11
    # the servers: no oracle for the baseline and the search configuration, the oracle file for the others
    by_name = {name: (spec, env) for name, spec, env in _FakeServer.started}
    assert by_name["baseline"] == (None, {}) and lt.ORACLE_ENV not in by_name["ngram3-search"][1]
    oracle = json.loads(open(by_name["ngram8"][1][lt.ORACLE_ENV]).read())
    assert len(oracle["streams"]) == 11 and oracle["schedule"][0] == 8 and oracle["tamper_every"] == 5
    assert by_name["mtp3-lookup"][1][lt.LOOKUP_ENV] == "1"
    # the per-request modes of the three passes, and no min_tokens anywhere
    modes = [body.get("vllm_xargs") for url, body in calls if url.endswith("/mtp3-lookup")]
    assert modes[:11] == [{"spec_lookup": 1}] * 11 and modes[11:22] == [{"spec_lookup": 2}] * 11
    assert modes[22:] == [{"spec_lookup": 0}] * 11
    assert all("min_tokens" not in body for _url, body in calls)
    assert "lookup tiny: 4 of 4 configurations passed" in printed and "[mtp3-lookup/hybrid]: PASSED" in printed


def test_a_wrong_token_away_from_a_tie_fails_its_configuration(monkeypatch, tmp_path, capsys):
    ckpt, _calls = _fake_world(monkeypatch, tmp_path, wrong_token_in="ngram8")
    rc = lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out"), "--configs", "ngram3,ngram8"])
    printed = capsys.readouterr().out
    assert rc == 1
    record = json.loads((tmp_path / "out" / "lookup_tiny.json").read_text())
    assert [c["status"] for c in record["configs"]] == ["passed", "failed"]
    assert "away from a near-tie" in record["configs"][1]["reason"]
    assert len(record["configs"][1]["passes"][0]["compare"]["breaks"]) == 11
    assert "lookup tiny: 1 of 2 configurations passed" in printed


def test_a_server_that_does_not_start_fails_only_its_configuration(monkeypatch, tmp_path, capsys):
    ckpt, _calls = _fake_world(monkeypatch, tmp_path)
    _FakeServer.dead = {"ngram3"}
    rc = lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out"), "--configs", "ngram3,ngram3-search"])
    assert rc == 1
    record = json.loads((tmp_path / "out" / "lookup_tiny.json").read_text())
    assert [c["status"] for c in record["configs"]] == ["failed", "passed"]
    assert "did not start" in record["configs"][0]["reason"]
    _FakeServer.dead = {"baseline"}
    assert lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out2")]) == 1
    assert json.loads((tmp_path / "out2" / "lookup_tiny.json").read_text())["configs"] == []
    capsys.readouterr()


def test_unknown_configuration_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        lt.main(["--configs", "ngram3,bogus", "--out", str(tmp_path)])


def test_the_lookup_off_gate(monkeypatch, tmp_path, capsys):
    ckpt, calls = _fake_world(monkeypatch, tmp_path)
    reads = {}

    def counters(base_url):
        # the n-gram server verifies no draft in the lookup-off pass (its first two reads), then oracle drafts
        n = reads[base_url] = reads.get(base_url, 0) + 1
        m = max(0, n - 2)
        return {"drafts": 10 * m, "draft_tokens": 50 * m, "accepted": 12 * m,
                "accepted_per_pos": [6 * m, 4 * m, 2 * m, 0, 0]}

    monkeypatch.setattr(lt, "counters", counters)
    off_server = {"verify": 0, "rows1": 1}

    def step_records(folder, t0=None, t1=None):
        name = Path(folder).name
        rows1 = off_server["rows1"] if name == "ngram5-off" else 1
        recs = [{"kind": "prefill", "rows": 1, "steps": 3}, {"kind": "decode", "rows": 1, "bucket_rows": rows1,
                                                             "steps": 40},
                {"kind": "decode", "rows": 4, "bucket_rows": 4, "steps": 20}]
        if name == "ngram5-off" and off_server["verify"]:
            recs.append({"kind": "verify", "rows": 1, "bucket_rows": 6, "steps": off_server["verify"]})
        return recs

    monkeypatch.setattr(lt, "step_records", step_records)
    rc = lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out"), "--configs", "ngram5-off"])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    record = json.loads((tmp_path / "out" / "lookup_tiny.json").read_text())
    entry = record["configs"][0]
    assert entry["status"] == "passed" and [p["pass"] for p in entry["passes"]] == ["off", "ngram5"]
    off = entry["passes"][0]
    assert off["steps"]["kinds"] == {"prefill": 3, "decode": 60} and off["compare"]["identical"] == 11
    assert record["baseline"]["decode_shapes"] == {"1": [1], "4": [4]}
    # the baseline and the n-gram server run with the step timer, the n-gram server with plain steps
    by_name = {name: env for name, _spec, env in _FakeServer.started}
    assert by_name["baseline"] == lt.STEP_TIMER_ENV
    assert by_name["ngram5-off"]["VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS"] == "1"
    assert by_name["ngram5-off"]["VLLM_HPU_STEP_TIMING"] == "1" and lt.ORACLE_ENV in by_name["ngram5-off"]
    modes = [body.get("vllm_xargs") for url, body in calls if url.endswith("/ngram5-off")]
    assert modes[:11] == [{"spec_lookup": 0}] * 11 and modes[11:] == [None] * 11
    assert "[ngram5-off/off]: PASSED" in printed and "steps decode 60, prefill 3" in printed

    # a verification step in the lookup-off pass fails the gate
    reads.clear()
    off_server["verify"] = 2
    assert lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out2"), "--configs", "ngram5-off"]) == 1
    entry = json.loads((tmp_path / "out2" / "lookup_tiny.json").read_text())["configs"][0]
    assert entry["status"] == "failed" and "2 verification step(s)" in entry["reason"]
    # so do decode steps whose input rows are not the baseline's (a verification-shaped graph without drafts)
    reads.clear()
    off_server.update(verify=0, rows1=6)
    assert lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out3"), "--configs", "ngram5-off"]) == 1
    entry = json.loads((tmp_path / "out3" / "lookup_tiny.json").read_text())["configs"][0]
    assert "decode input rows" in entry["reason"] and "'1': [[6], [1]]" in entry["reason"]
    # and a drafted step: the counters moved during the lookup-off pass
    reads.clear()
    reads["http://fake/ngram5-off"] = 5
    off_server.update(rows1=1)
    assert lt.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out4"), "--configs", "ngram5-off"]) == 1
    entry = json.loads((tmp_path / "out4" / "lookup_tiny.json").read_text())["configs"][0]
    assert "draft(s) verified" in entry["reason"]
    capsys.readouterr()


def test_step_records_and_their_summary(tmp_path):
    folder = tmp_path / "srv"
    folder.mkdir()
    lines = [{"kind": "decode", "rows": 2, "bucket_rows": 2, "steps": 1, "t_first": 100.0, "t_last": 100.1},
             {"kind": "verify", "rows": 2, "bucket_rows": 12, "steps": 1, "t_first": 200.0, "t_last": 200.1}]
    (folder / "step_timing_worker_r0_11.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    (folder / "step_timing_worker_r1_12.jsonl").write_text(json.dumps(lines[1]) + "\n")  # another rank: not read
    assert len(lt.step_records(folder)) == 2
    assert lt.step_records(folder, 150.0, 250.0) == [lines[1]]
    assert lt.step_summary(lt.step_records(folder)) == {"kinds": {"decode": 1, "verify": 1},
                                                        "decode_shapes": {"2": [2]}}
