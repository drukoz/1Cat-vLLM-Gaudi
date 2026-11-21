# SPDX-License-Identifier: Apache-2.0
"""``tools/flash_next/rental/commit_turns_probe.py`` against a stand-in completions server (no streaming).

The stand-in answers each prompt with token ids derived from the prompt (so the same prompt gives the same ids),
counts drafts on ``/metrics``, reports cached tokens for a prompt that extends an earlier one, and can be told to
drift (a different answer the second time a prompt is sent) or to die after a number of requests.
"""
import json
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools.flash_next.rental import commit_turns_probe as probe


def _server(drift: bool = False, die_after: int | None = None):
    state = {"seen": {}, "requests": [], "drafts": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def _send(self, status: int, payload: bytes):
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            with lock:
                drafts = state["drafts"]
            lines = ['vllm:spec_decode_num_drafts_total{engine="0"} %d' % drafts,
                     'vllm:spec_decode_num_draft_tokens_total{engine="0"} %d' % (3 * drafts),
                     'vllm:spec_decode_num_accepted_tokens_total{engine="0"} %d' % drafts]
            self._send(200, ("\n".join(lines) + "\n").encode())

        def do_POST(self):  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = request["prompt"]
            with lock:
                state["requests"].append(request)
                if die_after is not None and len(state["requests"]) > die_after:
                    self._send(500, b'{"error": "engine dead"}')
                    return
                count = state["seen"].get(prompt, 0)
                state["seen"][prompt] = count + 1
                cached = max((len(p) for p in state["seen"] if p != prompt and prompt.startswith(p)), default=0)
                state["drafts"] += request["max_tokens"] // 2
            seed = zlib.crc32(prompt.encode()) + (count if drift else 0)
            ids = [(seed * 31 + i * 7) % 1000 for i in range(request["max_tokens"])]
            reply = {"choices": [{"text": " answer %d" % seed, "token_ids": ids}],
                     "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(ids),
                               "prompt_tokens_details": {"cached_tokens": cached // 4}}}
            self._send(200, json.dumps(reply).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


@pytest.fixture
def stand_in():
    servers = []

    def start(**kwargs):
        server, state = _server(**kwargs)
        servers.append(server)
        return server.server_address[1], state

    yield start
    for server in servers:
        server.shutdown()


def _run(tmp_path, port):
    out = tmp_path / "turns.json"
    rc = probe.main(["--port", str(port), "--out", str(out), "--block", "64", "--decode", "12", "--timeout", "30"])
    return rc, json.loads(out.read_text())


def test_rounds_in_order_with_equal_repeats_and_the_next_turn(stand_in, tmp_path, capsys):
    port, state = stand_in()
    rc, data = _run(tmp_path, port)
    assert rc == 0
    names = [r["round"] for r in data["rounds"]]
    assert names == ["first", "shapes_2", "shapes_3", "shapes_5", "shapes_8", "again", "eight", "eight_again",
                     "next_turn"]
    assert all(r["equal"] for r in data["comparisons"]["again_vs_first"])
    assert len(data["comparisons"]["eight_again_vs_eight"]) == 8
    assert all(r["equal"] for r in data["comparisons"]["eight_again_vs_eight"])
    assert data["rounds"][-1]["results"][0]["cached_tokens"] > 0  # the next turn extends round 1's prompt
    assert all(req["temperature"] == 0 and req["ignore_eos"] and "min_tokens" not in req for req in state["requests"])
    assert data["rounds"][0]["spec"]["acceptance_length"] is not None
    printed = capsys.readouterr().out
    assert "COMPARE again_vs_first" in printed and "equal 1 of 1" in printed


def test_a_drifting_repeat_is_reported_with_its_first_difference(stand_in, tmp_path, capsys):
    port, _ = stand_in(drift=True)
    rc, data = _run(tmp_path, port)
    assert rc == 0
    row = data["comparisons"]["again_vs_first"][0]
    assert row["equal"] is False and row["first_diff"] == 0
    assert "equal 0 of 1" in capsys.readouterr().out


def test_a_dead_server_stops_the_probe_with_exit_1(stand_in, tmp_path, capsys):
    port, _ = stand_in(die_after=3)
    rc, data = _run(tmp_path, port)
    assert rc == 1
    assert data["rounds"][-1]["ok"] is False and len(data["rounds"]) < 9
    assert "the server is probably dead" in capsys.readouterr().out
