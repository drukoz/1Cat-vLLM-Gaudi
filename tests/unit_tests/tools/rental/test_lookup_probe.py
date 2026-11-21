# SPDX-License-Identifier: Apache-2.0
"""lookup_probe.py and lookup_serve.sh (the spec-lookup card job), on the workstation: the request bodies, the round
arithmetic, a whole probe against a small fake server, and the server script's dry run."""

from __future__ import annotations

import http.server
import json
import subprocess
import threading
from pathlib import Path

import pytest

from tools.flash_next.rental import lookup_probe as lp

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVE = REPO_ROOT / "tools" / "flash_next" / "rental" / "lookup_serve.sh"


def test_rounds_parse():
    assert lp.parse_rounds("edit:32:384:0,prose:1:256:-, editl:8:512:2,editf:8:1400:-,edit:1:384:-:100") == [
        ("edit", 32, 384, 0, 0), ("prose", 1, 256, None, 0), ("editl", 8, 512, 2, 0), ("editf", 8, 1400, None, 0),
        ("edit", 1, 384, None, 100)]
    for bad in ("edit:32:384", "poem:1:8:-", "edit:0:8:-", "edit:1:1:-", "edit:1:8:3", "edit:x:8:-", "edit:1:8:-:x",
                "edit:1:8:-:1:2"):
        with pytest.raises(ValueError):
            lp.parse_rounds(bad)


def test_prompts_are_fixed_per_stream_and_distinct():
    small = lp.load_files(str(REPO_ROOT), lp.SMALL_FILES)
    large = lp.load_files(str(REPO_ROOT), lp.LARGE_FILES)
    assert len(small) == len(lp.SMALL_FILES) and len(large) == len(lp.LARGE_FILES), "a listed file left the tree"
    for kind, files in (("edit", small), ("editl", large), ("prose", [])):
        prompts = [lp.prompt_for(kind, i, files) for i in range(32)]
        assert len(set(prompts)) == 32
        assert prompts == [lp.prompt_for(kind, i, files) for i in range(32)]
    edit = lp.prompt_for("edit", 9, small)
    assert edit.startswith("Request 9. ") and small[9 % len(small)][1] in edit and "Return the complete file" in edit
    # Measured with the checkpoint's tokenizer and chat template on 2026-11-20: the 32 edit prompts are 664 to 859
    # tokens (2,019 to 3,696 characters), the editl prompts 3,164 to 3,601 tokens, the prose prompts 39 to 41. An
    # edit prompt must stay under 1,024 tokens (one prefill chunk on an MTP server with prefix caching): a file
    # that grew past these sizes needs the measurement again.
    assert max(len(lp.prompt_for("edit", i, small)) for i in range(32)) <= 3700
    assert max(len(lp.prompt_for("editl", i, large)) for i in range(32)) <= 14300
    with pytest.raises(ValueError):
        lp.prompt_for("edit", 0, [])
    with pytest.raises(ValueError):
        lp.prompt_for("poem", 0, [])


def test_request_body_has_no_min_tokens_and_carries_the_mode():
    body = lp.request_body("fn", "hello", 64, None)
    assert "min_tokens" not in body and body["max_tokens"] == 64 and body["ignore_eos"] is True
    assert body["temperature"] == 0 and body["stream"] is True and "vllm_xargs" not in body
    assert body["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert body["return_token_ids"] is True
    assert lp.request_body("fn", "hello", 64, 2)["vllm_xargs"] == {"spec_lookup": 2}
    whole = lp.request_body("fn", "hello", 64, None, ignore_eos=False)
    assert whole["ignore_eos"] is False and whole["max_tokens"] == 64 and "min_tokens" not in whole
    small = lp.load_files(str(REPO_ROOT), lp.SMALL_FILES)
    assert lp.prompt_for("editf", 5, small) == lp.prompt_for("edit", 5, small)  # the same prompt, answered whole


def test_request_body_is_a_valid_vllm_request():
    protocol = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
    from vllm_gaudi.v1.spec_decode.flash_next_lookup import REQUEST_KEY, SpecLookupMode, request_mode

    request = protocol.ChatCompletionRequest(**lp.request_body("fn", "hello", 64, 1))
    params = request.to_sampling_params(64, {})
    assert params.max_tokens == 64 and params.min_tokens == 0 and params.ignore_eos and params.temperature == 0
    assert params.extra_args == {REQUEST_KEY: 1}
    assert request_mode(params.extra_args, SpecLookupMode.OFF) == SpecLookupMode.HYBRID
    assert request.chat_template_kwargs == {"enable_thinking": False} and request.return_token_ids is True
    plain = protocol.ChatCompletionRequest(**lp.request_body("fn", "hello", 64, None)).to_sampling_params(64, {})
    assert not plain.extra_args


def _sse(chunks, ids=False):
    """SSE lines of a chat stream: ``chunks`` = [(text, completion tokens so far)]; with ``ids`` the chunks carry
    token ids (700, 701, ...) and the first chunk the prompt's ids, as a server asked for return_token_ids does."""
    first = {"choices": [{"delta": {"role": "assistant", "content": ""}}],
             "usage": {"prompt_tokens": 9, "completion_tokens": 0}}
    if ids:
        first["prompt_token_ids"] = list(range(9))
    lines = [b"data: " + json.dumps(first).encode() + b"\n", b"\n"]
    done = 0
    for text, tokens in chunks:
        choice = {"delta": {"content": text}}
        if ids:
            choice["token_ids"] = list(range(700 + done, 700 + tokens))
        done = tokens
        lines.append(b"data: " + json.dumps({"choices": [choice],
                                             "usage": {"prompt_tokens": 9, "completion_tokens": tokens}}).encode()
                     + b"\n")
        lines.append(b"\n")
    lines.append(b'data: {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": %d, '
                 b'"prompt_tokens_details": {"cached_tokens": 4}}}\n' % chunks[-1][1])
    lines.append(b"data: [DONE]\n")
    return lines


def test_stream_reading_and_step_sizes():
    clock = iter(range(100))
    rec = lp.consume_stream(_sse([("a", 1), ("bcd", 4), ("e", 5), ("fghi", 9)]), now=lambda: float(next(clock)))
    assert rec["text"] == ["a", "bcd", "e", "fghi"] and rec["tokens"] == [1, 4, 5, 9]
    assert rec["times"] == [0.0, 1.0, 2.0, 3.0]
    assert rec["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}
    assert lp.step_sizes(rec["tokens"]) == [1, 3, 1, 4]
    assert lp.step_sizes([1, None, 3]) == [] and lp.step_sizes([]) == []
    assert rec["ids"] == [] and rec["prompt_ids"] is None  # this server sent no ids: the usage counted the tokens


def test_stream_reading_with_token_ids():
    """With return_token_ids a chunk without text still carries its tokens (one chunk per engine step): the ids
    count the tokens, and the stream's ids and the prompt's ids are kept."""
    clock = iter(range(100))
    rec = lp.consume_stream(_sse([("a", 1), ("", 4), ("e", 5), ("fghi", 9)], ids=True),
                            now=lambda: float(next(clock)))
    assert rec["tokens"] == [1, 4, 5, 9] and rec["text"] == ["a", "", "e", "fghi"]
    assert rec["ids"] == list(range(700, 709)) and rec["prompt_ids"] == list(range(9))
    assert lp.step_sizes(rec["tokens"]) == [1, 3, 1, 4]


def test_round_summary_arithmetic():
    streams = []
    for _ in range(2):
        # 5 chunks 100 ms apart: 1 + 4 + 4 + 4 + 4 = 17 tokens
        streams.append({"status": 200, "err": None, "t0": 0.0, "times": [1.0, 1.1, 1.2, 1.3, 1.4],
                        "tokens": [1, 5, 9, 13, 17], "text": list("abcde"),
                        "usage": {"prompt_tokens": 800, "completion_tokens": 17,
                                  "prompt_tokens_details": {"cached_tokens": 512}}})
    streams.append({"status": 500, "err": "boom", "t0": 0.0, "times": [], "tokens": [], "text": [], "usage": None})
    spec = {"drafts": 8, "draft_tokens": 24, "accepted": 24, "acceptance_length": 4.0, "draft_acceptance_rate": 1.0,
            "position_rates": [1.0, 1.0, 1.0]}
    m = lp.summarize_round("edit", 3, 17, 1, streams, 2.0, spec)
    assert m["ok"] == 2 and m["errors"] == 1 and m["error_samples"] == ["boom"]
    assert m["ms_per_step"] == pytest.approx(100.0) and m["steady_ms_per_token"] == pytest.approx(25.0)
    assert m["ms_per_token_p50"] == pytest.approx(400.0 / 16)  # 0.4 s for the 16 tokens after the first chunk
    assert m["step_sizes"] == {"1": 2, "4": 8} and m["tokens_per_chunk_p50"] == pytest.approx(3.4)
    assert m["prompt_tokens"] == [800, 800, 800] and m["cached_tokens_p50"] == 512
    assert m["texts"] == ["abcde", "abcde", ""]
    line = lp.round_line("job", "MEAS", m)
    assert "tokens/step=4.00" in line and "ms/step=100.0" in line and "steady=25.00" in line
    failed = lp.summarize_round("edit", 1, 17, 1, streams[2:], 2.0, None)
    assert failed["ok"] == 0 and "FAILED" in lp.round_line("job", "MEAS", failed)


def test_mixed_round_halves():
    small = lp.load_files(str(REPO_ROOT), lp.SMALL_FILES)
    assert lp.parse_rounds("mix:32:384:1") == [("mix", 32, 384, 1, 0)]
    # stream 2j is edit prompt j, stream 2j + 1 prose prompt j: the same prompts as the edit and prose rounds
    for j in range(16):
        assert lp.prompt_for("mix", 2 * j, small) == lp.prompt_for("edit", j, small)
        assert lp.prompt_for("mix", 2 * j + 1, small) == lp.prompt_for("prose", j, small)
    assert lp.stream_kinds("mix", 4) == ["edit", "prose", "edit", "prose"]
    assert lp.stream_kinds("edit", 2) == ["edit", "edit"]
    assert "mix" in lp.FULL_WARM

    def stream(tokens):  # 5 chunks 100 ms apart
        return {"status": 200, "err": None, "t0": 0.0, "times": [1.0, 1.1, 1.2, 1.3, 1.4], "tokens": tokens,
                "text": list("abcde"), "usage": {"prompt_tokens": 800, "completion_tokens": tokens[-1]}}

    out = [stream([1, 5, 9, 13, 17]), stream([1, 2, 3, 4, 5]), stream([1, 5, 9, 13, 17]), stream([1, 2, 3, 4, 5])]
    m = lp.summarize_round("mix", 4, 17, 1, out, 2.0, {"acceptance_length": 2.5})
    assert m["by_kind"]["edit"]["streams"] == 2 and m["by_kind"]["prose"]["streams"] == 2
    assert m["by_kind"]["edit"]["tokens_per_chunk_p50"] == pytest.approx(3.4)
    assert m["by_kind"]["prose"]["tokens_per_chunk_p50"] == pytest.approx(1.0)
    assert m["by_kind"]["prose"]["ms_per_token_p50"] == pytest.approx(100.0)
    line = lp.round_line("job", "MEAS", m)
    assert "edit: tok/chunk 3.40" in line and "prose: tok/chunk 1.00" in line
    assert "by_kind" not in lp.summarize_round("edit", 2, 17, 1, out[:2], 2.0, None)


def test_text_comparison():
    a = {"label": "mtp", "rounds": [{"kind": "edit", "streams": 2, "mode": 0, "ok": 2, "texts": ["hello you", "abc"]},
                                    {"kind": "edit", "streams": 2, "mode": 1, "ok": 2, "texts": ["hello you", "abd"]},
                                    {"kind": "edit", "streams": 1, "mode": 1, "ok": 1, "texts": ["hello"]}]}
    b = {"label": "ngram", "rounds": [{"kind": "edit", "streams": 2, "mode": None, "ok": 2,
                                       "texts": ["hello you", "abc"]}]}
    lines = lp.compare_texts([a, b])
    assert len(lines) == 3
    assert "mtp/1/N2" in lines[0] and "identical 1/2, shortest common prefix 2 chars" in lines[0]
    assert "mtp/1/N1" in lines[1] and "identical 1/1" in lines[1]  # the shorter round: its common part
    assert "ngram/None/N2" in lines[2] and "identical 2/2" in lines[2]


class _FakeServer(http.server.BaseHTTPRequestHandler):
    drafts = 0
    bodies: list = []

    def log_message(self, *args):  # noqa: D401 - quiet
        pass

    def do_GET(self):  # noqa: N802
        cls = type(self)
        text = ("vllm:spec_decode_num_drafts_total{engine=\"0\"} %d\n"
                "vllm:spec_decode_num_draft_tokens_total{engine=\"0\"} %d\n"
                "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} %d\n"
                "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",position=\"0\"} %d\n"
                "vllm:spec_decode_num_accepted_tokens_per_pos_total{engine=\"0\",position=\"1\"} %d\n" %
                (cls.drafts, 2 * cls.drafts, cls.drafts, cls.drafts, 0))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(text.encode())

    def do_POST(self):  # noqa: N802
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls.bodies.append(body)
        chunks, done = [], 0
        while done < body["max_tokens"]:  # every step commits 2 tokens (one accepted draft)
            done = min(body["max_tokens"], done + 2)
            chunks.append(("x" * 2, done))
            cls.drafts += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for line in _sse(chunks, ids=bool(body.get("return_token_ids"))):
            self.wfile.write(line)


def test_probe_end_to_end_against_a_fake_server(tmp_path, capsys):
    _FakeServer.drafts, _FakeServer.bodies = 0, []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out = tmp_path / "probe.json"
        rc = lp.main(["--port", str(server.server_address[1]), "--out", str(out), "--label", "fake", "--rounds",
                      "edit:3:8:0,edit:3:8:1,prose:1:6:-,editf:2:120:-", "--tree", str(REPO_ROOT)])
    finally:
        server.shutdown()
    assert rc == 0
    record = json.loads(out.read_text())
    assert [(r["kind"], r["streams"], r["mode"], r["ok"]) for r in record["rounds"]] == [("edit", 3, 0, 3),
                                                                                       ("edit", 3, 1, 3),
                                                                                       ("prose", 1, None, 1),
                                                                                       ("editf", 2, None, 2)]
    first = record["rounds"][0]
    assert first["spec"]["drafts"] == 12 and first["spec"]["acceptance_length"] == pytest.approx(2.0)
    assert first["spec"]["position_rates"] == [1.0, 0.0] and first["step_sizes"] == {"2": 12}
    assert first["token_ids"][0] == list(range(700, 708)) and "prompt_token_ids" not in first
    # measured: 3 edit + 1 prose (editf's two are edit prompts 0 and 1); the warm rounds ran on other prompts of
    # their kind (WARM_FIRST on: 3 edit, of which editf's warm round reuses two, + 1 prose)
    assert set(first["prompt_sha"]) <= set(record["prompts"]) and len(record["prompts"]) == 8
    warm_shas = {sha for w in record["warm"] for sha in w["prompt_sha"]}
    measured_shas = {sha for r in record["rounds"] for sha in r["prompt_sha"]}
    assert len(warm_shas) == 4 and not warm_shas & measured_shas
    assert [r["first"] for r in record["rounds"]] == [0, 0, 0, 0]
    assert record["prompts"][first["prompt_sha"][0]] == list(range(9))
    # one warm round per (kind, streams, mode): the two edit rounds (modes 0 and 1) warm on the same other prompts
    assert len(record["warm"]) == 4 and "texts" not in record["warm"][0]
    assert [(w["kind"], w["mode"]) for w in record["warm"]] == [("edit", 0), ("edit", 1), ("prose", None),
                                                                ("editf", None)]
    # the warm round is as long as the round for edit and prose, 48 tokens for the whole-answer round
    assert [w["tokens"] for w in record["warm"]] == [8, 8, 6, 48]
    whole = [b for b in _FakeServer.bodies if b["max_tokens"] == 120]
    assert len(whole) == 2 and all(b["ignore_eos"] is False for b in whole)
    # the request after the last round (a server with the lookup hook then logs the last round's counters)
    assert record["flush"]["ok"] == 1 and _FakeServer.bodies[-1]["max_tokens"] == 4
    assert all(b["ignore_eos"] is True for b in _FakeServer.bodies if b["max_tokens"] in (8, 6))
    assert all("min_tokens" not in b for b in _FakeServer.bodies)
    # the prose requests (warm, measured, the flush) carry no mode
    assert [b.get("vllm_xargs") for b in _FakeServer.bodies if b["messages"][0]["content"].startswith("Stream")] == [
        None, None, None]
    printed = capsys.readouterr().out
    assert "fake MEAS edit  N=3   mode=1" in printed and "fake TEXT edit" in printed and "identical 3/3" in printed


def _dry_run(*args):
    return subprocess.run(["bash", str(SERVE), "--dry-run", *args], capture_output=True, text=True, check=False,
                          env={"PATH": "/usr/bin:/bin", "FARM_NCARDS": "4", "FARM_PORT_BASE": "8300",
                               "FARM_OUT": "/results/x", "VLLM_HPU_HC_FUSED": "0"})


def test_serve_script_dry_run():
    done = _dry_run("--label", "mtp3", "--rounds", "edit:32:384:0", "--mtp", "3", "--lookup", "1")
    assert done.returncode == 0, done.stderr
    serve = next(line for line in done.stdout.splitlines() if line.startswith("serve:"))
    assert "--tensor-parallel-size 4 --enable-expert-parallel" in serve and "--port 8300" in serve
    assert "--enable-prefix-caching" in serve and "--speculative-config" in serve and "mtp" in serve
    assert "min_tokens" not in done.stdout
    env = done.stdout.split("\n")
    assert "VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP=1" in env and "VLLM_HPU_GDN_CORE_VERIFY=0" in env
    assert "VLLM_HPU_HC_FUSED=0" in env and "PT_HPU_LAZY_ACC_PAR_MODE=0" in env  # the job's value wins
    probe = next(line for line in done.stdout.splitlines() if line.startswith("probe:"))
    assert "lookup_probe.py" in probe and "--out /results/x/probe.json" in probe and "edit:32:384:0" in probe

    ngram = _dry_run("--label", "ngram8", "--rounds", "edit:1:64:-", "--ngram", "8")
    assert ngram.returncode == 0, ngram.stderr
    assert "ngram" in ngram.stdout and "prompt_lookup_max" in ngram.stdout
    assert "VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP" not in ngram.stdout

    plain = _dry_run("--label", "nospec", "--rounds", "edit:1:64:-")
    assert plain.returncode == 0 and "--speculative-config" not in plain.stdout


def test_serve_script_refuses_bad_arguments():
    assert _dry_run("--rounds", "edit:1:64:-").returncode == 2  # no label
    assert _dry_run("--label", "x").returncode == 2  # no rounds
    assert _dry_run("--label", "x", "--rounds", "edit:1:8:-", "--mtp", "3", "--ngram", "3").returncode == 2
    assert _dry_run("--label", "x", "--rounds", "edit:1:8:-", "--bogus").returncode == 2


ARMS = SERVE.parent / "lookup_arms.sh"


def _arms(*args, **env):
    done = subprocess.run(["bash", str(ARMS), "--dry-run", *args], capture_output=True, text=True, check=False,
                          env=dict({"PATH": "/usr/bin:/bin", "FARM_OUT": "/results/x"}, **env))
    assert done.returncode == 0, done.stderr
    return [line for line in done.stdout.splitlines() if line.startswith("arm ")], done.stdout


def test_arms_script_dry_run():
    lines, out = _arms("nospec", "ngram5", "hybrid3")
    assert "lookup_arms: placement off" in out  # no switch file
    assert len(lines) == 3 and [line.split(":")[0] for line in lines] == ["arm nospec", "arm ngram5", "arm hybrid3"]
    assert "--label nospec" in lines[0] and "--speculative" not in lines[0] and "--out /results/x/nospec" in lines[0]
    assert "--ngram 5" in lines[1] and "edit:1:384:-:100\\,edit:32:384:-:0\\,mix:32:384:-:400" in lines[1]
    # the hybrid arm: every round twice, mode 1 on the round's prompts, then 0 on other prompts, with the
    # integration build's MTP flags
    assert "--mtp 3 --lookup 1" in lines[2] and "edit:1:384:1:100\\,edit:1:384:0:400" in lines[2]
    assert "mix:32:384:1:400\\,mix:32:384:0:700" in lines[2] and "VLLM_HPU_MTP_DRAFT_LOOP=1" in lines[2]
    bad = subprocess.run(["bash", str(ARMS), "--dry-run", "ngram"], capture_output=True, text=True, check=False)
    assert bad.returncode == 2 and "unknown arm" in bad.stderr


def test_arms_script_folds_the_reference_into_the_ngram_server():
    lines, _ = _arms("tiny-off", "hybrid3", "ngram5-nospec")
    assert [line.split(":")[0] for line in lines] == ["arm tiny-off", "arm hybrid3", "arm ngram5-nospec"]
    assert "lookup_tiny.py --configs ngram5-off --out /results/x/tiny-off" in lines[0]
    # n-gram on the prompts 300 on, then the lookup off on the round's own prompts (the hybrid's mode 1 prompts)
    assert "--ngram 5" in lines[2] and "VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS=1" in lines[2]
    assert "edit:1:384:-:400\\,edit:1:384:0:100\\,edit:32:384:-:300\\,edit:32:384:0:0" in lines[2]
    assert "mix:32:384:-:700\\,mix:32:384:0:400" in lines[2] and "--out /results/x/ngram5-nospec" in lines[2]
    assert "edit:1:384:1:100" in lines[1]
    # the job's own setting wins
    lines, _ = _arms("ngram5-nospec", VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS="0")
    assert "NGRAM_PLAIN_STEPS" not in lines[0]
    # LOOKUP_ARMS_SPLIT=1 (as after a failed tiny gate): two servers, the reference first
    lines, out = _arms("hybrid3", "ngram5-nospec", LOOKUP_ARMS_SPLIT="1")
    assert [line.split(":")[0] for line in lines] == ["arm hybrid3", "arm nospec", "arm ngram5"]
    assert "ngram5-nospec runs as nospec and ngram5" in out and "edit:1:384:-:100" in lines[2]


def test_arms_script_placement_switch(tmp_path):
    switch, wrap, place = tmp_path / "place", tmp_path / "place_wrap.sh", tmp_path / "place.sh"
    wrap.write_text("#!/bin/bash\n")
    place.write_text("#!/bin/bash\n")
    env = {"LOOKUP_ARMS_PLACE_SWITCH": str(switch), "LOOKUP_ARMS_PLACE_WRAP": str(wrap),
           "LOOKUP_ARMS_PLACE_SCRIPT": str(place)}
    switch.write_text("on\n")
    lines, out = _arms("tiny-off", "nospec", **env)
    assert "lookup_arms: placement ON (rule a^1" in out
    assert "place_wrap" not in lines[0] and "WORKER_CPU_PIN" not in lines[0]  # the tiny gate is never placed
    assert "VLLM_HPU_WORKER_CPU_PIN=1" in lines[1] and "PLACE_SIBLING_RULE=a\\^1" in lines[1]
    assert ("PLACE_ARGS=--log\\ /results/x/nospec/server0.log" in lines[1]
            and "PLACE_LOG=/results/x/nospec/placement.log" in lines[1])
    assert "bash %s %s -- bash " % (wrap, place) in lines[1] and lines[1].endswith("--out /results/x/nospec")
    switch.write_text("off\n")
    lines, out = _arms("nospec", **env)
    assert "lookup_arms: placement off" in out and "place_wrap" not in lines[0]
    switch.write_text("a+N/2\n")
    wrap.unlink()  # switched on, but the wrapper is missing: off
    lines, out = _arms("nospec", **env)
    assert "placement switch on but" in out and "place_wrap" not in lines[0]
