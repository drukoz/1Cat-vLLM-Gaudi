# SPDX-License-Identifier: Apache-2.0
"""Tests for the deterministic agent-128k-v1 workload builder (plan M4, RUK-2922).

The small-size builds (4 agents, 2000 shared, 8000 initial tokens) run over the real repository with
the offline ``--estimate`` counter; one build runs with the full defaults (32 agents, 124000 tokens).
A fake ``/tokenize`` server on a thread proves ``tokenize_counter`` posts the contract body and reads
the ``count`` field.
"""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.flash_next.rental.workload import TEMPLATES, build_workload, estimate_counter, tokenize_counter

REPO_ROOT = Path(__file__).resolve().parents[4]

SMALL = dict(agents=4, shared_tokens=2000, initial_tokens=8000, turns=4, max_tokens=64)

EXPECTED_TEMPLATES = [
    "add a one-line docstring to every function and method that lacks one.",
    "rename the local variable names that are a single letter (except loop indices i, j, k) to descriptive names.",
    "add type hints to every function signature that lacks them.",
]


def build(**overrides) -> dict:
    kwargs = dict(SMALL)
    kwargs.update(overrides)
    return build_workload(REPO_ROOT, count_tokens=estimate_counter, **kwargs)


def as_bytes(workload: dict) -> bytes:
    return json.dumps(workload, ensure_ascii=False, sort_keys=True).encode("utf-8")


BLOCK_START = re.compile(r"File: (\S+)\n```python\n")


def block_paths(user_content: str) -> list[str]:
    return BLOCK_START.findall(user_content)


def first_file_path(user_content: str) -> str:
    return block_paths(user_content)[0]


@pytest.fixture(scope="module")
def workload() -> dict:
    return build()


def test_templates_match_the_contract():
    assert TEMPLATES == EXPECTED_TEMPLATES


def test_output_shape(workload):
    assert set(workload) == {
        "version", "shared_prefix_tokens", "target_initial_tokens", "turns", "max_tokens", "agents"
    }
    assert workload["version"] == "agent-128k-v1"
    assert workload["turns"] == SMALL["turns"]
    assert workload["max_tokens"] == SMALL["max_tokens"]
    assert workload["target_initial_tokens"] == SMALL["initial_tokens"]
    assert len(workload["agents"]) == SMALL["agents"]
    for i, entry in enumerate(workload["agents"]):
        assert set(entry) == {"agent", "messages", "followups"}
        assert entry["agent"] == i
        assert [m["role"] for m in entry["messages"]] == ["system", "user"]


def test_two_builds_are_byte_identical():
    assert as_bytes(build()) == as_bytes(build())


def test_shared_system_message_identical_and_in_window(workload):
    sys_msgs = {json.dumps(entry["messages"][0], sort_keys=True) for entry in workload["agents"]}
    assert len(sys_msgs) == 1
    sys_msg = workload["agents"][0]["messages"][0]
    count = estimate_counter([sys_msg])
    assert SMALL["shared_tokens"] - 64 <= count <= SMALL["shared_tokens"]
    assert workload["shared_prefix_tokens"] == count
    assert sys_msg["content"].startswith(
        "You are a coding agent working in the repository below. Follow the repository guide exactly.\n\n")
    assert "=== AGENTS.md ===\n" in sys_msg["content"]


def test_first_prompts_in_window(workload):
    for entry in workload["agents"]:
        messages = entry["messages"]
        count = estimate_counter(messages)
        assert SMALL["initial_tokens"] - 512 <= count <= SMALL["initial_tokens"], entry["agent"]
        user = messages[1]["content"]
        assert user.startswith("File: ")
        assert user.endswith(
            f"Task: in File: {first_file_path(user)}, {TEMPLATES[entry['agent'] % 3]}"
            " Return the complete file in one fenced code block.")


def test_agents_start_at_different_files(workload):
    starts = [first_file_path(entry["messages"][1]["content"]) for entry in workload["agents"]]
    assert len(set(starts)) == len(starts)


def test_followups(workload):
    for entry in workload["agents"]:
        paths = block_paths(entry["messages"][1]["content"])
        assert len(entry["followups"]) == SMALL["turns"] - 1
        for t, followup in enumerate(entry["followups"], start=1):
            assert followup == (f"Now, in File: {paths[min(t, len(paths) - 1)]}, {TEMPLATES[(entry['agent'] + t) % 3]}"
                                " Return the complete file in one fenced code block.")


def test_budget_check_raises():
    with pytest.raises(ValueError):
        build(initial_tokens=130000)


def test_tokenize_counter_uses_the_pinned_fields():
    captured = {}

    class Handler(BaseHTTPRequestHandler):

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            captured["path"] = self.path
            captured["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps({"count": 77, "token_ids": [1], "tokens": ["a"]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        counter = tokenize_counter(f"http://127.0.0.1:{server.server_address[1]}", "flash-next")
        messages = [{"role": "user", "content": "hello"}]
        assert counter(messages) == 77
        assert captured["path"] == "/tokenize"
        assert captured["body"] == {
            "model": "flash-next",
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        }
    finally:
        server.shutdown()
        server.server_close()


def test_full_size_build_with_estimate_under_60s():
    start = time.monotonic()
    workload = build_workload(REPO_ROOT,
                              agents=32,
                              shared_tokens=16384,
                              initial_tokens=124000,
                              turns=4,
                              max_tokens=1024,
                              count_tokens=estimate_counter)
    elapsed = time.monotonic() - start
    assert elapsed < 60.0, f"{elapsed:.1f} s"
    assert len(workload["agents"]) == 32
    for entry in workload["agents"]:
        count = estimate_counter(entry["messages"])
        assert 124000 - 512 <= count <= 124000, entry["agent"]
