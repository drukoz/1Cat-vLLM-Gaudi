# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tools/flash_next/replay/text_check.py against a stub OpenAI-compatible server."""
import http.server
import importlib.util
import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("text_check", ROOT / "tools/flash_next/replay/text_check.py")
tc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tc)


class _Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "stub-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        assert body["temperature"] == 0.0 and body["ignore_eos"] is True
        text = "echo: " + body["messages"][0]["content"][:20]
        self._send({"choices": [{"message": {"content": text}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": body["max_tokens"]}})


def test_run_and_compare(tmp_path, monkeypatch, capsys):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "a.json"
        monkeypatch.setattr("sys.argv", ["text_check.py", "--port", str(server.server_address[1]), "--out", str(out),
                                         "--streams", "2", "--prompts", "3", "--tokens", "16"])
        assert tc.main() == 0
        report = json.loads(out.read_text())
        assert report["model"] == "stub-model" and len(report["results"]) == 3
        assert all(r["completion_tokens"] == 16 for r in report["results"])
    finally:
        server.shutdown()
    b = tmp_path / "b.json"
    changed = json.loads(out.read_text())
    changed["results"][1]["text"] = changed["results"][1]["text"][:8] + "!!!!!!!!"
    b.write_text(json.dumps(changed))
    assert tc.compare(str(out), str(out)) == 0
    assert tc.compare(str(out), str(b)) == 1
    assert "1 of 3 prompts differ" in capsys.readouterr().out


def test_garbage():
    assert tc.garbage("abc")["longest_run"] == 1
    assert tc.garbage("a!!!!!b")["longest_run"] == 5
    assert tc.garbage("��ab")["replacement_share"] == 0.5


def test_churn_round(tmp_path, monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "churn.json"
        monkeypatch.setattr("sys.argv", ["text_check.py", "--port", str(server.server_address[1]), "--out", str(out),
                                         "--churn", "--prompts", "2", "--tokens", "16", "--long-tokens", "200"])
        assert tc.main() == 0
        report = json.loads(out.read_text())
        assert report["churn"] is True and len(report["results"]) == 2
        assert report["results"][0]["text"].startswith("echo: Write a long")
    finally:
        server.shutdown()
    assert len(tc.long_prompt(1000).split()) > 600
