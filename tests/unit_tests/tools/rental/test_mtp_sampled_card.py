# SPDX-License-Identifier: Apache-2.0
"""``tools/flash_next/rental/mtp_sampled_card.py`` against a stand-in server: its three arms and their checks.

The stand-in answers like a tiny vLLM server: greedy requests get the reference tokens, sampled requests a stream
that depends on the seed (or HTTP 400 in the refusing mode), ``/metrics`` the speculative counters.
"""
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.flash_next.rental import mtp_sampled_card, serve, tiny_smoke

REFERENCE = {"prompts": [{"prompt_ids": list(range(n)), "generated_ids": [n + i for i in range(16)]}
                         for n in (1, 7, 33, 100, 700, 2600)]}


class Stub:
    """The stand-in server's behaviour for one arm."""

    def __init__(self, refuse_sampled=False, speculative=True, broken_greedy=False):
        self.refuse_sampled = refuse_sampled
        self.speculative = speculative
        self.broken_greedy = broken_greedy
        self.drafts = 0
        self.lock = threading.Lock()

    def handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                lines = []
                if stub.speculative:
                    lines = ['vllm:spec_decode_num_drafts_total{engine="0"} %d' % stub.drafts,
                             'vllm:spec_decode_num_draft_tokens_total{engine="0"} %d' % (3 * stub.drafts),
                             'vllm:spec_decode_num_accepted_tokens_total{engine="0"} %d' % (stub.drafts // 4),
                             'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} %d' %
                             (stub.drafts // 4)]
                self._send(200, "\n".join(lines) + "\n")

            def do_POST(self):  # noqa: N802
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                sampled = request.get("temperature", 1.0) != 0
                count = request["max_tokens"]
                if sampled and stub.refuse_sampled:
                    return self._send(400, json.dumps({"error": {"message": "greedy requests only: start the server "
                                                                            "with VLLM_HPU_SPEC_SAMPLED_VERIFY=1"}}))
                if sampled:
                    assert request["top_k"] == 20 and request["top_p"] == 0.95, request
                    rng = random.Random(request.get("seed"))
                    ids = [rng.randrange(1000) for _ in range(count)]
                else:
                    prompt = next(p for p in REFERENCE["prompts"] if p["prompt_ids"] == request["prompt"])
                    ids = list(prompt["generated_ids"][:count])
                    if stub.broken_greedy and len(request["prompt"]) == 100:
                        ids[3] += 1
                with stub.lock:
                    stub.drafts += count
                self._send(200, json.dumps({"choices": [{"token_ids": ids}]}))

        return Handler


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """Starts the stand-in for an arm; the script's server start, health wait and stop are replaced."""
    state = {}

    def start(stub):
        server = ThreadingHTTPServer(("127.0.0.1", 0), stub.handler())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        state["server"] = server
        return server.server_address[1]

    monkeypatch.setattr(serve, "start_server", lambda command, env, log_path, cwd: state.update(
        command=command, env=env) or Path(log_path).write_text("server log\n") and object())
    monkeypatch.setattr(serve, "wait_healthy", lambda port, timeout, popen: True)
    monkeypatch.setattr(serve, "stop_server", lambda popen: state["server"].shutdown())
    monkeypatch.setattr(tiny_smoke, "serve_command", lambda ckpt, port: ["vllm", "serve", str(ckpt),
                                                                           "--tensor-parallel-size", "8"])
    state["start"] = start
    state["out"] = tmp_path
    return state


def run(stand_in, arm, stub, **kwargs):
    port = stand_in["start"](stub)
    return mtp_sampled_card.run_arm(arm, 3, Path("/ckpt"), stand_in["out"] / arm, port, 2, 12, 5.0, REFERENCE,
                                    **kwargs)


def test_the_sampled_arm_passes_and_starts_a_speculative_server_with_the_flag(stand_in):
    entry = run(stand_in, "sampled", Stub())
    assert entry["status"] == "passed", entry["checks"]
    assert stand_in["env"]["VLLM_HPU_SPEC_SAMPLED_VERIFY"] == "1"
    command = stand_in["command"]
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--speculative-config" in command
    assert entry["acceptance"]["drafts"] > 0
    names = [c["check"] for c in entry["checks"]]
    assert "the same seed gives the same stream" in names
    assert "the greedy request of the mixed batch reproduces the reference" in names


def test_the_refuse_arm_wants_http_400_with_the_flag_name(stand_in):
    entry = run(stand_in, "refuse", Stub(refuse_sampled=True))
    assert entry["status"] == "passed", entry["checks"]
    assert stand_in["env"]["VLLM_HPU_SPEC_SAMPLED_VERIFY"] == "0"


def test_the_refuse_arm_fails_when_the_server_serves_the_sampled_request(stand_in):
    entry = run(stand_in, "refuse", Stub(refuse_sampled=False))
    assert entry["status"] == "failed"
    assert [c["check"] for c in entry["checks"] if not c["ok"]] == [
        "a sampled request is refused with HTTP 400 naming the flag"]


def test_the_plain_arm_has_no_speculation(stand_in):
    entry = run(stand_in, "plain", Stub(speculative=False))
    assert entry["status"] == "passed", entry["checks"]
    assert "--speculative-config" not in stand_in["command"]
    assert "VLLM_HPU_SPEC_SAMPLED_VERIFY" not in stand_in["env"] or stand_in["env"][
        "VLLM_HPU_SPEC_SAMPLED_VERIFY"] in ("0", "")
    assert entry["acceptance"] is None


def test_a_greedy_stream_that_leaves_the_reference_fails_the_arm(stand_in):
    entry = run(stand_in, "sampled", Stub(broken_greedy=True))
    assert entry["status"] == "failed"
    assert "a greedy request afterwards reproduces the reference" in [c["check"] for c in entry["checks"]
                                                                      if not c["ok"]]
