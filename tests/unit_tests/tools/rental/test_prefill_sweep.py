# SPDX-License-Identifier: Apache-2.0
"""``tools/flash_next/rental/prefill_sweep.py`` against a fake server with a block prefix cache.

The fake server prefills at a fixed rate, one request after the other, and stops every decoding stream
while a prefill runs (what the HPU runner does today), unless the knobs file asks for pacing. So the
tests can check the numbers the tool reports: prefill rate, time to first token, cached tokens of warm,
second-turn and shared-prefix requests, and the stall of the canary streams.
"""

import asyncio
import json
import socket
import threading
import time

import pytest
from aiohttp import web

from tools.flash_next.rental import prefill_sweep as ps

BLOCK = 512
PREFILL_RATE = 40000.0  # prompt tokens per second
DECODE_RATE = 200.0  # tokens per second per stream


class FakeServer:
    """Token-id completions with a prefix cache of full ``BLOCK``-token blocks."""

    def __init__(self, knobs_path=None) -> None:
        self.cached: set[tuple] = set()
        self.prefilling = 0
        self.knobs_path = knobs_path
        self.knobs_seen: list[dict] = []
        self.prompt_tokens = 0.0
        self.hits = 0.0
        self.queries = 0.0
        self.requests: list[dict] = []
        self.lock = None

    def _knobs(self) -> dict:
        if not self.knobs_path:
            return {}
        try:
            with open(self.knobs_path, encoding="utf-8") as handle:
                knobs = json.load(handle)
        except (OSError, ValueError):
            return {}
        if not self.knobs_seen or self.knobs_seen[-1] != knobs:
            self.knobs_seen.append(knobs)
        return knobs

    def _hit(self, ids: list[int]) -> int:
        blocks = 0
        while (blocks + 1) * BLOCK <= len(ids) - 1 and tuple(ids[:(blocks + 1) * BLOCK]) in self.cached:
            blocks += 1
        return blocks * BLOCK

    def _store(self, ids: list[int]) -> None:
        for end in range(BLOCK, len(ids) + 1, BLOCK):
            self.cached.add(tuple(ids[:end]))

    async def tokenize(self, request: web.Request) -> web.Response:
        body = await request.json()
        tokens = [2000 + (hash(word) % 50000) for word in str(body["prompt"]).split()]
        return web.json_response({"count": len(tokens), "max_model_len": 131072, "tokens": tokens})

    async def metrics(self, request: web.Request) -> web.Response:
        return web.Response(text="\n".join([
            'vllm:prompt_tokens_total{model_name="m"} %g' % self.prompt_tokens,
            'vllm:prefix_cache_hits_total{model_name="m"} %g' % self.hits,
            'vllm:prefix_cache_queries_total{model_name="m"} %g' % self.queries,
        ]) + "\n")

    async def completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        ids = list(body["prompt"])
        assert all(isinstance(t, int) for t in ids) and body["stream"] and body["add_special_tokens"] is False
        if self.lock is None:
            self.lock = asyncio.Lock()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        paced = self._knobs().get("mode") == 1
        async with self.lock:  # one prefill after the other; the hit is looked up when its turn comes
            cached = self._hit(ids)
            self.requests.append({"tokens": len(ids), "cached": cached, "max_tokens": body["max_tokens"]})
            self.prompt_tokens += len(ids)
            self.queries += len(ids)
            self.hits += cached
            self.prefilling += 0 if paced else 1
            try:
                await asyncio.sleep((len(ids) - cached) / PREFILL_RATE)
                self._store(ids)
            finally:
                self.prefilling -= 0 if paced else 1
        produced = []
        try:
            for n in range(1, int(body["max_tokens"]) + 1):
                while self.prefilling:  # a decoding stream waits for every prefill
                    await asyncio.sleep(0.002)
                token = 3000 + n
                produced.append(token)
                chunk = {"choices": [{"index": 0, "text": "t"}],
                         "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}}
                if body.get("return_token_ids"):
                    chunk["choices"][0]["token_ids"] = [token]
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                await asyncio.sleep(1.0 / DECODE_RATE)
            self._store(ids + produced)
            final = {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(produced),
                                              "prompt_tokens_details": {"cached_tokens": cached}}}
            await resp.write(b"data: " + json.dumps(final).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):  # a canary the tool stopped
            pass
        return resp


@pytest.fixture()
def server(tmp_path):
    knobs_path = tmp_path / "knobs.json"
    fake = FakeServer(str(knobs_path))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    loop = asyncio.new_event_loop()
    started = threading.Event()

    holder = {}

    async def serve() -> None:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/tokenize", fake.tokenize)
        app.router.add_get("/metrics", fake.metrics)
        app.router.add_post("/v1/completions", fake.completions)
        holder["runner"] = web.AppRunner(app, shutdown_timeout=0.5)
        await holder["runner"].setup()
        await web.SockSite(holder["runner"], sock).start()
        started.set()

    def main() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve())
        loop.run_forever()

    thread = threading.Thread(target=main, daemon=True)
    thread.start()
    assert started.wait(10.0)
    fake.url = f"http://127.0.0.1:{port}"
    fake.knobs = knobs_path
    yield fake
    asyncio.run_coroutine_threadsafe(holder["runner"].cleanup(), loop).result(10.0)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5.0)


def run_tool(server, tmp_path, *extra):
    out = tmp_path / "out"
    argv = ["--url", server.url, "--model", "m", "--out", str(out), "--synthetic-ids", "--corpus-tokens", "50000",
            "--settle-seconds", "0.25", "--canary-warm-tokens", "4", "--nonce", "test", *extra]
    code = ps.main(argv)
    report = json.loads((out / "prefill_sweep.json").read_text())
    return code, report, out


def by_case(report):
    return {row["case"]: row for row in report["cases"]}


def test_baseline_plan_reports_rates_hits_and_the_stall_of_decoding_streams(server, tmp_path):
    code, report, out = run_tool(server, tmp_path, "--plan", "baseline", "--cold-tokens", "8192", "--warm-tokens",
                                 "64,512", "--concurrent", "3", "--concurrent-tokens", "4096", "--turn2-tokens",
                                 "2048", "--turn2-generate", "600", "--shared-prefix", "2048", "--shared-tokens",
                                 "4096", "--shared-together", "2", "--canaries", "2")
    assert code == 0
    cases = by_case(report)
    cold = cases["cold 8192"]
    request = cold["requests"][0]
    assert (request["prompt_tokens"], request["cached_tokens"], request["uncached_tokens"]) == (8192, 0, 8192)
    # 8192 tokens at 40,000 per second is 0.20 s; the first token follows one decode period later
    assert 0.20 <= request["ttft_s"] <= 0.40
    assert 20000 <= cold["aggregate"]["prefill_tokens_per_s"] <= 41000
    # the two canary streams decode at 200 tokens per second and stop while the prompt is prefilled
    streams = cold["canaries"]
    assert streams["before"]["streams"] == 2
    assert 120 <= streams["before"]["tokens_per_s_p50"] <= 210
    assert streams["during"]["stall_ms"] >= 180  # the 0.2 s prefill
    assert streams["during"]["tokens_per_s_p50"] < 0.5 * streams["before"]["tokens_per_s_p50"]

    # a later turn: the first 8192 tokens are 16 full blocks, the 64 or 512 new tokens are prefilled
    warm = cases["warm 8192+64"]["requests"][0]
    assert (warm["cached_tokens"], warm["uncached_tokens"]) == (8192, 64)
    warm = cases["warm 8192+512"]["requests"][0]
    assert (warm["cached_tokens"], warm["uncached_tokens"]) == (8192, 512)

    together = cases["concurrent 3x4096"]
    assert together["aggregate"]["requests"] == 3 and together["aggregate"]["uncached_tokens"] == 3 * 4096
    assert [r["cached_tokens"] for r in together["requests"]] == [0, 0, 0]
    # one prefill after the other: the last first token comes after 12,288 tokens' worth of time
    assert 0.30 <= together["aggregate"]["seconds_to_last_first_token"] <= 0.60
    assert 20000 <= together["aggregate"]["prefill_tokens_per_s"] <= 41000

    # second turn: prompt 2048 + 600 generated = 2648 tokens are cached as 5 full blocks
    turn2 = cases["turn2 2048+600+40"]["requests"][0]
    assert turn2["prompt_tokens"] == 2688 and turn2["cached_tokens"] == 2560 and turn2["uncached_tokens"] == 128

    assert cases["shared 2048 first"]["requests"][0]["cached_tokens"] == 0
    assert cases["shared 2048 second"]["requests"][0]["cached_tokens"] == 2048
    assert cases["shared 2048 third"]["requests"][0]["cached_tokens"] == 2048
    group = cases["shared 2048 x2 together"]["requests"]
    # the fake server prefills one at a time, so the second of the pair finds the first one's blocks
    assert sorted(r["cached_tokens"] for r in group) == [0, 2048]

    assert cold["metrics_delta"]["prompt_tokens"] >= 8192
    text = (out / "prefill_sweep.md").read_text()
    assert "| server | cold 8192 | 1 | 8192 | 0 |" in text
    assert len((out / "cases.jsonl").read_text().splitlines()) == len(report["cases"])


def test_cold_prompts_stay_cold_across_runs_with_different_nonces(server, tmp_path):
    for nonce in ("one", "two"):
        out = tmp_path / nonce
        assert ps.main(["--url", server.url, "--model", "m", "--out", str(out), "--synthetic-ids", "--corpus-tokens",
                        "50000", "--plan", "ladder", "--ladder", "1024,2048", "--canaries", "0", "--nonce",
                        nonce]) == 0
    assert [r["cached"] for r in server.requests] == [0, 0, 0, 0]
    assert [r["tokens"] for r in server.requests] == [1024, 2048, 1024, 2048]


def test_ladder_reports_the_rate_of_each_added_range(server, tmp_path):
    code, report, out = run_tool(server, tmp_path, "--plan", "ladder", "--ladder", "2048,4096,8192", "--canaries", "0")
    assert code == 0
    marginals = ps.depth_marginals(report["cases"])
    assert [(m["from"], m["to"]) for m in marginals] == [(0, 2048), (2048, 4096), (4096, 8192)]
    assert all(15000 <= m["tokens_per_s"] <= 60000 for m in marginals[1:])
    assert "Rate by context depth" in (out / "prefill_sweep.md").read_text()


def test_sweep_and_pacing_plans_drive_the_knobs_file(server, tmp_path):
    code, report, _ = run_tool(server, tmp_path, "--plan", "sweep,pacing", "--knobs-file", str(server.knobs),
                               "--settings", "8192:2048,16384:off", "--pacing", "100:0,25:512", "--sweep-tokens",
                               "4096", "--sweep-concurrent", "2", "--canaries", "1")
    assert code == 0
    # what the server found in the file when its requests arrived (the reset to {} between the two plans
    # and at the end is followed by no request)
    assert server.knobs_seen == [
        {"long_prefill_token_threshold": 2048, "max_num_scheduled_tokens": 8192},
        {"long_prefill_token_threshold": 0, "max_num_scheduled_tokens": 16384},
        {"mode": 0},
        {"mode": 1, "share_percent": 25, "slice_tokens": 512},
    ]
    assert json.loads(server.knobs.read_text()) == {}  # the server is left at its own settings
    labels = [(row["setting"], row["case"]) for row in report["cases"]]
    assert labels == [("budget 8192 cap 2048", "cold 4096"), ("budget 8192 cap 2048", "concurrent 2x4096"),
                      ("budget 16384 cap off", "cold 4096"), ("budget 16384 cap off", "concurrent 2x4096"),
                      ("pacing off", "cold 4096"), ("share 25% slice 512", "cold 4096")]
    off, paced = report["cases"][4], report["cases"][5]
    # the fake server keeps its streams decoding when the knobs ask for pacing
    assert off["canaries"]["during"]["stall_ms"] >= 80
    assert paced["canaries"]["during"]["stall_ms"] < off["canaries"]["during"]["stall_ms"]
    assert paced["canaries"]["during"]["tokens_per_s_p50"] > 100


def test_a_plan_that_changes_settings_needs_the_knobs_file(server, tmp_path):
    with pytest.raises(SystemExit, match="needs --knobs-file"):
        run_tool(server, tmp_path, "--plan", "sweep", "--canaries", "0")


def test_unknown_plan_is_refused():
    with pytest.raises(SystemExit):
        ps.parse_args(["--url", "u", "--model", "m", "--out", "o", "--plan", "nope"])


def test_window_stats_counts_tokens_and_stalls_inside_the_window():
    # one token every 10 ms from t=0 to t=1, nothing until t=3, then three tokens per chunk every 30 ms
    arrivals = [[0.01 * i, i] for i in range(1, 101)]
    arrivals += [[3.0 + 0.03 * i, 100 + 3 * (i + 1)] for i in range(34)]
    before = ps.window_stats(arrivals, 0.0, 1.0)
    assert before["tokens"] == 100 and before["tokens_per_s"] == pytest.approx(100.0)
    assert before["itl_p50_ms"] == pytest.approx(10.0) and before["stall_ms"] == pytest.approx(10.0, abs=0.1)
    during = ps.window_stats(arrivals, 1.0, 3.0)
    assert during["tokens"] == 3 and during["stall_ms"] == pytest.approx(2000.0)
    assert during["itl_p50_ms"] == pytest.approx(2000.0 / 3)  # the chunk at t=3 carried three tokens
    after = ps.window_stats(arrivals, 3.0, 4.0)
    assert after["tokens_per_s"] == pytest.approx(99.0, abs=3.1)
    assert after["itl_p50_ms"] == pytest.approx(10.0)  # 30 ms per chunk of three
    empty = ps.window_stats(arrivals, 1.2, 2.5)
    assert empty["tokens"] == 0 and empty["tokens_per_s"] == 0.0 and empty["stall_ms"] == pytest.approx(1300.0)
    assert empty["itl_p50_ms"] is None
    assert ps.window_stats(arrivals, 2.0, 2.0)["tokens_per_s"] is None


def test_corpus_prompts_have_exact_lengths_and_distinct_first_blocks():
    corpus = ps.Corpus(ps.synthetic_ids(20000), "n")
    first, second = corpus.cold(5000), corpus.cold(5000)
    assert len(first) == len(second) == 5000
    assert first[:ps.SALT_TOKENS] != second[:ps.SALT_TOKENS]
    assert ps.Corpus(ps.synthetic_ids(20000), "n").cold(5000) == first  # deterministic per nonce
    assert ps.Corpus(ps.synthetic_ids(20000), "other").cold(5000)[:ps.SALT_TOKENS] != first[:ps.SALT_TOKENS]
    assert len(corpus.take(45000, "wraps")) == 45000
    assert len(corpus.cold(7)) == 7
    low, high = ps.SYNTHETIC_ID_RANGE
    assert all(low <= t < high for t in first)
    with pytest.raises(ValueError):
        ps.Corpus([1, 2, 3], "n")


def test_tokenize_corpus_uses_the_servers_tokenizer(server):
    ids = ps.tokenize_corpus(server.url, "m", ["alpha beta gamma " * 3000, "delta " * 100], want=5000)
    assert len(ids) >= 5000 and all(isinstance(t, int) for t in ids)


def test_request_timeout_is_reported_as_an_error_row(tmp_path):
    record = ps.new_record("x", 10)

    async def go():
        import aiohttp
        async with aiohttp.ClientSession() as session:
            return await ps.stream_completion(session, "http://127.0.0.1:9", {"prompt": [1]}, record)

    asyncio.run(go())
    assert record["status"] == 2 and record["error"]
    row = ps.request_row(record)
    assert row["ttft_s"] is None and row["prefill_tokens_per_s"] is None
    assert time.time() - record["t_send"] < 30
