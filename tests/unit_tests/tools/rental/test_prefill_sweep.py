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

from tests.unit_tests.tools.rental.prefill_fake_server import FakeServer
from tools.flash_next.rental import prefill_sweep as ps


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


@pytest.fixture()
def slow_prefill(monkeypatch):
    """The fake server at 8,000 prompt tokens per second: the times the tests bound are then long against
    the delays of a busy workstation's event loop."""
    from tests.unit_tests.tools.rental import prefill_fake_server

    monkeypatch.setattr(prefill_fake_server, "PREFILL_RATE", 8000.0)
    return 8000.0


def test_baseline_plan_reports_rates_hits_and_the_stall_of_decoding_streams(server, tmp_path, slow_prefill):
    code, report, out = run_tool(server, tmp_path, "--plan", "baseline", "--cold-tokens", "8192", "--warm-tokens",
                                 "64,512", "--concurrent", "3", "--concurrent-tokens", "4096", "--turn2-tokens",
                                 "2048", "--turn2-generate", "600", "--shared-prefix", "2048", "--shared-tokens",
                                 "4096", "--shared-together", "2", "--canaries", "2")
    assert code == 0
    cases = by_case(report)
    cold = cases["cold 8192"]
    request = cold["requests"][0]
    assert (request["prompt_tokens"], request["cached_tokens"], request["uncached_tokens"]) == (8192, 0, 8192)
    # 8192 tokens at 8,000 per second is 1.02 s; the first token follows one decode period later
    assert 1.02 <= request["ttft_s"] <= 2.0
    assert 4000 <= cold["aggregate"]["prefill_tokens_per_s"] <= 8100
    # the two canary streams decode at 200 tokens per second and stop while the prompt is prefilled
    streams = cold["canaries"]
    assert streams["before"]["streams"] == 2
    assert 60 <= streams["before"]["tokens_per_s_p50"] <= 210
    assert streams["during"]["stall_ms"] >= 900  # the 1.02 s prefill
    assert streams["during"]["tokens_per_s_p50"] < 0.5 * streams["before"]["tokens_per_s_p50"]

    # a later turn: the first 8192 tokens are 16 full blocks, the 64 or 512 new tokens are prefilled
    warm = cases["warm 8192+64"]["requests"][0]
    assert (warm["cached_tokens"], warm["uncached_tokens"]) == (8192, 64)
    warm = cases["warm 8192+512"]["requests"][0]
    assert (warm["cached_tokens"], warm["uncached_tokens"]) == (8192, 512)

    together = cases["concurrent 3x4096"]
    assert together["aggregate"]["requests"] == 3 and together["aggregate"]["uncached_tokens"] == 3 * 4096
    assert [r["cached_tokens"] for r in together["requests"]] == [0, 0, 0]
    # one prefill after the other: the last first token comes after 12,288 tokens' worth of time (1.54 s)
    assert 1.53 <= together["aggregate"]["seconds_to_last_first_token"] <= 3.0
    assert 4000 <= together["aggregate"]["prefill_tokens_per_s"] <= 8100

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


def test_ladder_reports_the_rate_of_each_added_range(server, tmp_path, slow_prefill):
    code, report, out = run_tool(server, tmp_path, "--plan", "ladder", "--ladder", "2048,4096,8192", "--canaries", "0")
    assert code == 0
    marginals = ps.depth_marginals(report["cases"])
    assert [(m["from"], m["to"]) for m in marginals] == [(0, 2048), (2048, 4096), (4096, 8192)]
    # each added range takes 0.26 s and 0.51 s at 8,000 tokens per second
    assert all(0.4 * slow_prefill <= m["tokens_per_s"] <= 2.5 * slow_prefill for m in marginals[1:])
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
    assert paced["canaries"]["during"]["tokens_per_s_p50"] > 40


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


def test_case_max_tokens_and_return_ids_record_the_greedy_output(server, tmp_path):
    code, report, _ = run_tool(server, tmp_path, "--plan", "ladder", "--ladder", "2048", "--canaries", "0",
                               "--case-max-tokens", "4", "--return-ids")
    assert code == 0
    request = by_case(report)["cold 2048"]["requests"][0]
    assert request["output_tokens"] == 4
    assert len(request["token_ids"]) == 4


def test_sweep_concurrent_requests_take_their_own_size(server, tmp_path):
    code, report, _ = run_tool(server, tmp_path, "--plan", "sweep", "--knobs-file", str(server.knobs), "--settings",
                               "8192:8192", "--sweep-tokens", "4096", "--sweep-concurrent", "2",
                               "--sweep-concurrent-tokens", "1024", "--canaries", "0")
    assert code == 0
    together = by_case(report)["concurrent 2x1024"]
    assert together["aggregate"]["uncached_tokens"] == 2 * 1024
    assert by_case(report)["cold 4096"]["requests"][0]["uncached_tokens"] == 4096


def test_requests_ask_for_an_exact_length_without_min_tokens(server, tmp_path):
    # min_tokens adds a logits processor that masks the stop tokens of its rows at every decode step:
    # no request of the tool sends it, neither a case nor a canary stream
    code, _, _ = run_tool(server, tmp_path, "--plan", "baseline", "--cold-tokens", "2048", "--warm-tokens", "",
                          "--concurrent", "2", "--concurrent-tokens", "1024", "--turn2-tokens", "0",
                          "--shared-prefix", "0", "--canaries", "1", "--case-max-tokens", "3")
    assert code == 0
    assert len(server.requests) == 5  # a canary and the cold case, a canary and the two concurrent ones
    for request in server.requests:
        assert "min_tokens" not in request["fields"]
        assert request["ignore_eos"] is True
    assert sorted(r["max_tokens"] for r in server.requests) == [3, 3, 3, 60000, 60000]
    assert "min_tokens" not in ps.completion_payload("m", [1, 2, 3], 8, return_ids=True)


def test_run_knobs_lie_under_every_case_and_under_the_plans_own_knobs(server, tmp_path):
    code, report, out = run_tool(server, tmp_path, "--plan", "baseline,pacing", "--knobs-file", str(server.knobs),
                                 "--knobs", '{"prompt_graph_max_tokens": 140000}', "--label", "graphs",
                                 "--cold-tokens", "2048", "--warm-tokens", "", "--concurrent", "",
                                 "--turn2-tokens", "0", "--shared-prefix", "0", "--pacing", "100:0,25:512",
                                 "--sweep-tokens", "2048", "--canaries", "1")
    assert code == 0
    assert server.knobs_seen == [
        {"prompt_graph_max_tokens": 140000},
        {"mode": 0, "prompt_graph_max_tokens": 140000},
        {"mode": 1, "share_percent": 25, "slice_tokens": 512, "prompt_graph_max_tokens": 140000},
    ]
    # the run leaves the server in its own arm, not at the server's start-up settings
    assert json.loads(server.knobs.read_text()) == {"prompt_graph_max_tokens": 140000}
    assert [(row["setting"], row["case"]) for row in report["cases"]] == [
        ("graphs", "cold 2048"), ("graphs, pacing off", "cold 2048"), ("graphs, share 25% slice 512", "cold 2048")]
    assert report["cases"][0]["knobs"] == {"prompt_graph_max_tokens": 140000}
    assert "| graphs, pacing off | cold 2048 |" in (out / "prefill_sweep.md").read_text()


def test_run_knobs_need_the_knobs_file_and_a_json_object():
    base = ["--url", "u", "--model", "m", "--out", "o"]
    with pytest.raises(SystemExit):
        ps.parse_args(base + ["--knobs", '{"prompt_graph_max_tokens": 0}'])
    with pytest.raises(SystemExit):
        ps.parse_args(base + ["--knobs-file", "f", "--knobs", "[1]"])
    args = ps.parse_args(base + ["--knobs-file", "f", "--knobs", '{"long_prefill_token_threshold": 8192}'])
    assert args.knobs == {"long_prefill_token_threshold": 8192} and args.label == "server"


def test_streams_report_their_median_inter_token_latency(server, tmp_path):
    code, report, out = run_tool(server, tmp_path, "--plan", "baseline", "--cold-tokens", "8192", "--warm-tokens", "",
                                 "--concurrent", "", "--turn2-tokens", "0", "--shared-prefix", "0", "--canaries", "2")
    assert code == 0
    streams = report["cases"][0]["canaries"]
    # 200 tokens per second per stream before the prefill: 5 ms between tokens (the event loop adds a little)
    assert 4.0 <= streams["before"]["itl_p50_ms"] <= 60.0
    # the fake server stops its streams for the whole prefill: that window has a stall of the prefill's
    # length, and the wait across it is an inter-token latency of the window in which it ends
    assert streams["during"]["stall_ms"] >= 180
    assert streams["after"]["itl_p50_ms"] is not None
    assert set(streams["during"]) >= {"itl_p50_ms", "itl_p99_ms", "stall_ms", "tokens_per_s_p50"}
    assert "ITL p50 ms before" in (out / "prefill_sweep.md").read_text()


def test_an_interrupted_run_still_writes_its_report(server, tmp_path):
    # prefill_job.sh stops a plan line with SIGINT when the job's time is up; the queue sends SIGTERM
    import signal
    import subprocess
    import sys

    for number, name in ((signal.SIGINT, "int"), (signal.SIGTERM, "term")):
        out = tmp_path / name
        proc = subprocess.Popen([sys.executable, "-m", "tools.flash_next.rental.prefill_sweep", "--url", server.url,
                                 "--model", "m", "--out", str(out), "--synthetic-ids", "--corpus-tokens", "50000",
                                 "--plan", "ladder", "--ladder", "2048,400000", "--canaries", "0", "--nonce", name],
                                cwd=str(ps.REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 60.0
        while time.time() < deadline and not (out / "cases.jsonl").exists():
            time.sleep(0.05)
        while time.time() < deadline and not (out / "cases.jsonl").read_text().strip():
            time.sleep(0.05)  # the 2,048-token case is done; the 400,000-token one takes 10 s
        proc.send_signal(number)
        output, _ = proc.communicate(timeout=30.0)
        assert proc.returncode != 0, output
        report = json.loads((out / "prefill_sweep.json").read_text())
        assert [row["case"] for row in report["cases"]] == ["cold 2048"], output
        assert (out / "prefill_sweep.md").exists()
