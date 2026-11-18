# SPDX-License-Identifier: Apache-2.0
"""Tests for tools.flash_next.rental.gates (plan M4, RUK-2709)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tools.flash_next.rental import gates
from tools.flash_next.rental.schema import GateStatus

_HAS_CHECK_FP8 = importlib.util.find_spec("tools.flash_next.check_fp8_layout") is not None

HL8 = ["sh", "-c", "for i in 0 1 2 3 4 5 6 7; do echo \"$i, Gaudi 2\"; done"]
HL7 = ["sh", "-c", "for i in 0 1 2 3 4 5 6; do echo \"$i, Gaudi 2\"; done"]


def make_checkpoint(root: Path,
                    *,
                    quant_method: str = "fp8",
                    weight_block_size=None,
                    index: bool = True,
                    size: int = 4096) -> Path:
    """A fake block-FP8 checkpoint folder: config.json, index, one .safetensors file."""
    if weight_block_size is None:
        weight_block_size = [128, 128]
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3NextForCausalLM"],
        "hidden_size": 4096,
        "quantization_config": {
            "quant_method": quant_method,
            "weight_block_size": list(weight_block_size),
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if index:
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {},
            "weight_map": {}
        }),
                                                            encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\0" * size)
    return root


@pytest.fixture
def fake_api():
    """Start canned-answer HTTP fakes; yields (start(responder) -> (url, received), teardown)."""
    servers = []

    def start(responder):
        received = []

        class Handler(BaseHTTPRequestHandler):

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                received.append((self.path, req))
                body = json.dumps(responder(self.path, req)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return "http://127.0.0.1:%d" % server.server_address[1], received

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- capacity


def test_capacity_pass():
    status, details = gates.capacity({"max_concurrency": 40.4, "max_concurrency_len": 131072}, 32)
    assert status is GateStatus.PASSED
    assert details["max_concurrency"] == 40.4
    assert details["agents"] == 32


def test_capacity_too_few():
    status, _ = gates.capacity({"max_concurrency": 10.0, "max_concurrency_len": 131072}, 32)
    assert status is GateStatus.FAILED


def test_capacity_short_len():
    status, _ = gates.capacity({"max_concurrency": 40.0, "max_concurrency_len": 131071}, 32)
    assert status is GateStatus.FAILED


def test_capacity_missing_startup_line():
    status, details = gates.capacity({"max_concurrency": None, "max_concurrency_len": None}, 32)
    assert status is GateStatus.FAILED
    assert details["reason"] == "startup line missing"


def test_capacity_not_a_dict():
    status, details = gates.capacity(None, 32)
    assert status is GateStatus.FAILED
    assert "error" in details


# --------------------------------------------------------------------------- preflight


def test_preflight_pass_with_lowered_limits(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=8,
                                      hlsmi_cmd=HL8,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.PASSED
    assert details["cards"] == 8
    assert details["config_ok"] is True
    assert details["quant_method"] == "fp8"
    assert details["weight_block_size"] == [128, 128]
    assert details["index_ok"] is True
    assert details["checkpoint_bytes"] == 4096
    assert details["free_bytes"] > 1
    assert set(details["checks"].values()) <= {"passed", "skipped"}


def test_preflight_skips_cards_in_stub_mode(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      hlsmi_cmd=["definitely-not-a-command-xyz"],
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.PASSED
    assert details["cards"] is None
    assert details["checks"]["cards"] == "skipped"


def test_preflight_seven_cards_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=8,
                                      hlsmi_cmd=HL7,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert details["cards"] == 7
    assert details["checks"]["cards"] == "failed"


def test_preflight_non_fp8_config_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model", quant_method="gptq")
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert details["checks"]["config"] == "failed"


def test_preflight_wrong_block_size_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model", weight_block_size=[64, 64])
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert details["checks"]["config"] == "failed"


def test_preflight_missing_index_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model", index=False)
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert details["index_ok"] is False
    assert "index" in details["reason"]


def test_preflight_too_few_bytes_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model", size=4096)
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=4097,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert details["checks"]["checkpoint"] == "failed"
    assert "incomplete download" in details["reason"]


def test_preflight_disk_too_small_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=2**62)
    assert status is GateStatus.FAILED
    assert details["checks"]["disk"] == "failed"


def test_preflight_missing_hlsmi_fails(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [],
                                      tmp_path / "results",
                                      expected_cards=8,
                                      hlsmi_cmd=["definitely-not-a-command-xyz"],
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert "error" in details


def test_preflight_layout_rows_ok(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    rows = [{"module": "mlp.experts.down", "dim": "n", "size": 12288, "per_rank": 1536, "block": 128, "ok": True}]
    monkeypatch.setattr(gates, "check_layout", lambda config, tp, ep: [dict(r) for r in rows])
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [(8, False)],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.PASSED
    assert details["checks"]["layout"] == "passed"
    assert details["layout_rows"][0]["tp"] == 8


def test_preflight_layout_rows_fail(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    rows = [{"module": "mlp.experts.down", "dim": "n", "size": 12288, "per_rank": 192, "block": 128, "ok": False}]
    monkeypatch.setattr(gates, "check_layout", lambda config, tp, ep: [dict(r) for r in rows])
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [(8, False)],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED
    assert details["checks"]["layout"] == "failed"
    assert details["layout_failed"][0]["module"] == "mlp.experts.down"


@pytest.mark.skipif(not _HAS_CHECK_FP8, reason="check_fp8_layout.py (M4-H0) is not merged yet")
def test_preflight_real_check_layout_rejects_toy_config(tmp_path):
    (tmp_path / "results").mkdir()
    model = make_checkpoint(tmp_path / "model")
    status, details = gates.preflight(model, [(8, False)],
                                      tmp_path / "results",
                                      expected_cards=0,
                                      min_checkpoint_bytes=1,
                                      min_free_bytes=1)
    assert status is GateStatus.FAILED  # the toy config has no MoE fields
    assert details["checks"]["layout"] == "failed"
    assert details["checks"]["disk"] == "passed"  # later checks still run
    # Flash-Next's real expert shape: TP8 without EP splits 640 into 80 (not a 128 multiple); with EP it passes
    cfg = json.loads((model / "config.json").read_text(encoding="utf-8"))
    cfg.update(hidden_size=2560, moe_intermediate_size=640, num_experts=512)
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert gates.preflight(model, [(8, False)], tmp_path / "results", expected_cards=0, min_checkpoint_bytes=1,
                           min_free_bytes=1)[0] is GateStatus.FAILED
    assert gates.preflight(model, [(8, True)], tmp_path / "results", expected_cards=0, min_checkpoint_bytes=1,
                           min_free_bytes=1)[0] is GateStatus.PASSED


# --------------------------------------------------------------------------- build_check


def write_fake_python(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


FAKE_PY_OK = """
if [ -n "$GATES_FAKE_PY_LOG" ]; then pwd >> "$GATES_FAKE_PY_LOG"; fi
case "$2" in
  *flashinfer_gaudi*) echo '{}' ;;
  *vllm_gaudi*) : ;;
  *) echo 0.28.1rc1.dev392+g8f816a3f6 ;;
esac
exit 0
"""


def test_build_check_pass_runs_outside_repo_root(tmp_path, monkeypatch):
    log = tmp_path / "cwd.log"
    monkeypatch.setenv("GATES_FAKE_PY_LOG", str(log))
    fake = write_fake_python(tmp_path / "fake_python.sh", FAKE_PY_OK)
    status, details = gates.build_check(python=fake)
    assert status is GateStatus.PASSED
    assert details["vllm_version"] == "0.28.1rc1.dev392+g8f816a3f6"
    assert details["vllm_gaudi_ok"] is True
    assert details["flashinfer_gaudi_ok"] is True
    assert details["flashinfer_capabilities"] == "{}"
    lines = log.read_text(encoding="utf-8").split()
    temp = os.path.realpath(tempfile.gettempdir())
    assert len(lines) == 3
    assert all(os.path.realpath(line) == temp for line in lines)


def test_build_check_failing_interpreter(tmp_path):
    fake = write_fake_python(tmp_path / "fake_python.sh", "echo 'boom' >&2\nexit 1\n")
    status, details = gates.build_check(python=fake)
    assert status is GateStatus.FAILED
    assert "exit 1" in details["reason"]
    assert details["vllm_gaudi_ok"] is False


def test_build_check_wrong_version(tmp_path):
    body = FAKE_PY_OK.replace("0.28.1rc1.dev392+g8f816a3f6", "9.9.9")
    fake = write_fake_python(tmp_path / "fake_python.sh", body)
    status, details = gates.build_check(python=fake)
    assert status is GateStatus.FAILED
    assert details["vllm_version"] == "9.9.9"
    assert "9.9.9" in details["reason"]


FAKE_PY_NOISY = """
echo 'INFO 11-18 22:17:11 [__init__.py:272] Platform plugin hpu is activated'
echo 'WARNING 11-18 22:17:11 [platform.py:678] Pin memory is not supported on HPU.'
case "$2" in
  *flashinfer_gaudi*) echo '{"device": "gaudi2"}' ;;
  *vllm_gaudi*) : ;;
  *) echo 0.28.1rc1.dev392+g8f816a3f6 ;;
esac
exit 0
"""


def test_build_check_ignores_log_lines_before_the_answer(tmp_path):
    # On the card `import vllm` logs to stdout before the probe prints (2026-11-18 rental, RUK-3100).
    fake = write_fake_python(tmp_path / "fake_python.sh", FAKE_PY_NOISY)
    status, details = gates.build_check(python=fake)
    assert status is GateStatus.PASSED
    assert details["vllm_version"] == "0.28.1rc1.dev392+g8f816a3f6"
    assert details["flashinfer_capabilities"] == '{"device": "gaudi2"}'


def test_build_check_missing_interpreter(tmp_path):
    status, details = gates.build_check(python=str(tmp_path / "no-such-interpreter"))
    assert status is GateStatus.FAILED
    assert "error" in details


# --------------------------------------------------------------------------- tiny_compare


def write_reference(path: Path) -> dict:
    prompts = []
    expected = {}
    for i, n in enumerate([1, 7]):
        prompt_ids = list(range(1, n + 1))
        generated_ids = list(range(100 + 16 * i, 100 + 16 * i + 16))
        prompts.append({"prompt_ids": prompt_ids, "generated_ids": generated_ids})
        expected[tuple(prompt_ids)] = generated_ids
    path.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
    return expected


def test_tiny_compare_pass(tmp_path, fake_api):
    reference = tmp_path / "reference.json"
    expected = write_reference(reference)

    def responder(path, req):
        return {"choices": [{"token_ids": list(expected[tuple(req["prompt"])])}]}

    url, received = fake_api(responder)
    status, details = gates.tiny_compare(url, reference)
    assert status is GateStatus.PASSED
    assert details["n_prompts"] == 2
    assert all(p["full_match"] and p["match"] == 16 for p in details["prompts"])
    path, req = received[0]
    assert path == "/v1/completions"
    assert req["model"] == "tiny" and req["max_tokens"] == 16 and req["return_token_ids"] is True


def test_tiny_compare_mismatch(tmp_path, fake_api):
    reference = tmp_path / "reference.json"
    expected = write_reference(reference)

    def responder(path, req):
        ids = list(expected[tuple(req["prompt"])])
        if len(req["prompt"]) > 1:  # the 7-token prompt differs at the last position
            ids[-1] += 1
        return {"choices": [{"token_ids": ids}]}

    url, _ = fake_api(responder)
    status, details = gates.tiny_compare(url, reference)
    assert status is GateStatus.FAILED
    assert details["prompts"][0]["full_match"] is True
    assert details["prompts"][1]["match"] == 15
    assert details["prompts"][1]["first_diff"] == 15


def test_tiny_compare_server_down(tmp_path):
    reference = tmp_path / "reference.json"
    write_reference(reference)
    status, details = gates.tiny_compare("http://127.0.0.1:1", reference)
    assert status is GateStatus.FAILED
    assert "error" in details


# --------------------------------------------------------------------------- coherence

GOOD_ANSWER = "The sky is blue because the atmosphere scatters short blue light more strongly than red light."
JSON_ANSWER = '{"name": "Ada", "age": 33, "city": "Lisbon"}'
LOOP_ANSWER = "the sea the sea the sea the sea the sea the sea the sea the sea the sea the sea"


def coherence_responder(overrides=None, chat_answer="7319-ALPHA"):
    answers = {p: GOOD_ANSWER for p in gates.COHERENCE_PROMPTS}
    answers[gates.COHERENCE_PROMPTS[gates.COHERENCE_JSON_PROMPT_INDEX]] = JSON_ANSWER
    answers.update(overrides or {})

    def responder(path, req):
        if path == "/v1/completions":
            return {"choices": [{"text": answers.get(req["prompt"], GOOD_ANSWER)}]}
        return {"choices": [{"message": {"content": chat_answer}}]}

    return responder


NEEDLE_MESSAGES = [
    {"role": "system", "content": "You are a deployment agent operating a Gaudi rental node."},
    {"role": "user", "content": "config preamble\n" + ("padding words for context length " * 500) + "\ntail line"},
]


def test_coherence_pass_needle_hit(fake_api):
    url, received = fake_api(coherence_responder())
    status, details = gates.coherence(url, "Qwen3.8-Flash-Next", NEEDLE_MESSAGES)
    assert status is GateStatus.PASSED
    assert details["needle"]["hit"] is True
    assert len(details["answers"]) == 8
    completions = [req for path, req in received if path == "/v1/completions"]
    assert len(completions) == 8
    assert completions[0]["max_tokens"] == 64 and completions[0]["temperature"] == 0
    chat = [req for path, req in received if path == "/v1/chat/completions"][0]
    assert chat["max_tokens"] == 32 and chat["temperature"] == 0
    assert chat["chat_template_kwargs"] == {"enable_thinking": False}
    content = chat["messages"][-1]["content"]
    original = NEEDLE_MESSAGES[1]["content"]
    assert gates.NEEDLE_SENTENCE in content
    assert content.index(gates.NEEDLE_SENTENCE) >= int(len(original) * 0.30)
    assert content[content.index(gates.NEEDLE_SENTENCE) - 1] == "\n"
    assert content.endswith(gates.NEEDLE_QUESTION)


def test_coherence_needle_miss(fake_api):
    url, _ = fake_api(coherence_responder(chat_answer="7318-BETA"))
    status, details = gates.coherence(url, "Qwen3.8-Flash-Next", NEEDLE_MESSAGES)
    assert status is GateStatus.FAILED
    assert details["needle"]["hit"] is False
    assert gates.NEEDLE_CODE not in details["needle"]["answer"]


def test_coherence_empty_answer_fails(fake_api):
    url, received = fake_api(coherence_responder({gates.COHERENCE_PROMPTS[0]: "   "}))
    status, details = gates.coherence(url, "Qwen3.8-Flash-Next", NEEDLE_MESSAGES)
    assert status is GateStatus.FAILED
    assert details["answers"][0]["empty"] is True
    assert not any(path == "/v1/chat/completions" for path, _ in received)


def test_coherence_repetition_loop_fails(fake_api):
    url, received = fake_api(coherence_responder({gates.COHERENCE_PROMPTS[1]: LOOP_ANSWER}))
    status, details = gates.coherence(url, "Qwen3.8-Flash-Next", NEEDLE_MESSAGES)
    assert status is GateStatus.FAILED
    assert details["answers"][1]["loop"] is True
    assert not any(path == "/v1/chat/completions" for path, _ in received)


def test_coherence_prompt7_without_json_fails(fake_api):
    url, _ = fake_api(coherence_responder({gates.COHERENCE_PROMPTS[6]: "name Ada, age 33, city Lisbon"}))
    status, details = gates.coherence(url, "Qwen3.8-Flash-Next", NEEDLE_MESSAGES)
    assert status is GateStatus.FAILED
    assert details["answers"][6]["json_ok"] is False


def test_coherence_needle_without_user_message(fake_api):
    url, _ = fake_api(coherence_responder())
    status, details = gates.coherence(url, "Qwen3.8-Flash-Next", [{"role": "system", "content": "only system"}])
    assert status is GateStatus.FAILED
    assert "no user message" in details["reason"]


def test_coherence_server_down():
    status, details = gates.coherence("http://127.0.0.1:1", "Qwen3.8-Flash-Next", NEEDLE_MESSAGES)
    assert status is GateStatus.FAILED
    assert "error" in details
