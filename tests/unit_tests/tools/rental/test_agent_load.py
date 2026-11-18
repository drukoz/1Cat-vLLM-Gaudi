# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental harness agent load client and its stub server
(plan M4, RUK-2709). The stub runs inside the test's event loop."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from aiohttp import web

from tools.flash_next.rental.agent_load import main as agent_load_main
from tools.flash_next.rental.agent_load import run_load
from tools.flash_next.rental.schema import REQUEST_KEYS, RequestStatus
from tools.flash_next.rental.stub_server import make_app


def make_workload(agents: int = 3, turns: int = 3, max_tokens: int = 8) -> dict:
    agent_list = []
    for i in range(agents):
        system = "agent %d system prompt " % i + "s" * 400
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "question %d for agent %d" % (0, i)},
        ]
        followups = ["followup %d for agent %d" % (t, i) for t in range(turns - 1)]
        agent_list.append({"agent": i, "messages": messages, "followups": followups})
    return {
        "version": "agent-128k-v1",
        "shared_prefix_tokens": 128,
        "target_initial_tokens": 128,
        "turns": turns,
        "max_tokens": max_tokens,
        "agents": agent_list,
    }


async def start_stub(tokens_per_second: float = 400.0, fail_after=None):
    app = make_app("flash-next", tokens_per_second, fail_after)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, app, "http://127.0.0.1:%d" % port


def read_rows(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_closed_loop_ok(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner, app, url = await start_stub()
        state = app["stub"]
        out = tmp_path / "requests.jsonl"
        workload = make_workload(agents=3, turns=3, max_tokens=8)
        try:
            errors = await run_load(workload, [url], "flash-next", out, stagger_s=0.3)
        finally:
            await runner.cleanup()
        assert errors == 0
        rows = read_rows(out)
        assert len(rows) == 9
        for row in rows:
            assert list(row.keys()) == list(REQUEST_KEYS)
            assert row["status"] == int(RequestStatus.OK)
            assert row["error"] is None
            assert row["server"] == 0
            assert row["prompt_tokens"] > 0
            assert row["output_tokens"] == 8
            assert row["t_send"] is not None and row["t_first"] is not None and row["t_last"] is not None
            assert row["t_send"] <= row["t_first"] <= row["t_last"]
            assert row["itl_p50_ms"] is not None and row["itl_p50_ms"] > 0
        by_agent = {}
        for row in rows:
            by_agent.setdefault(row["agent"], []).append(row)
        assert sorted(by_agent) == [0, 1, 2]
        first_sends = []
        for agent, agent_rows in by_agent.items():
            agent_rows.sort(key=lambda r: r["turn"])
            assert [r["turn"] for r in agent_rows] == [0, 1, 2]
            assert agent_rows[0]["cached_tokens"] == 0
            assert all(r["cached_tokens"] > 0 for r in agent_rows[1:])
            first_sends.append((agent_rows[0]["t_send"], agent))
        # agents start in stagger order
        assert [a for _, a in sorted(first_sends)] == [0, 1, 2]
        # the stub saw each agent's growing message list, assistant replies included
        for agent in (0, 1, 2):
            seen = [m for m in state.received if m[0]["content"].startswith("agent %d " % agent)]
            assert [len(m) for m in seen] == [2, 4, 6]
            assert seen[1][2]["role"] == "assistant" and "tok" in seen[1][2]["content"]
            assert seen[2][3]["role"] == "user" and seen[2][3]["content"].startswith("followup 0")
            assert seen[2][5]["role"] == "user" and seen[2][5]["content"].startswith("followup 1")

    asyncio.run(scenario())


def test_two_servers_alternate(tmp_path: Path) -> None:
    async def scenario() -> None:
        r1, _, u1 = await start_stub()
        r2, _, u2 = await start_stub()
        out = tmp_path / "requests.jsonl"
        workload = make_workload(agents=4, turns=1, max_tokens=8)
        try:
            errors = await run_load(workload, [u1, u2], "flash-next", out, stagger_s=0.0)
        finally:
            await r1.cleanup()
            await r2.cleanup()
        assert errors == 0
        rows = read_rows(out)
        assert len(rows) == 4
        assert {row["agent"]: row["server"] for row in rows} == {0: 0, 1: 1, 2: 0, 3: 1}

    asyncio.run(scenario())


def test_fail_after_http_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner, _, url = await start_stub(fail_after=4)
        out = tmp_path / "requests.jsonl"
        workload = make_workload(agents=3, turns=3, max_tokens=8)
        try:
            errors = await run_load(workload, [url], "flash-next", out, stagger_s=0.3)
        finally:
            await runner.cleanup()
        rows = read_rows(out)
        assert len(rows) <= 9
        bad = [r for r in rows if r["status"] != int(RequestStatus.OK)]
        assert len(bad) >= 1
        assert all(r["status"] == int(RequestStatus.HTTP_ERROR) for r in bad)
        assert all(r["error"] and r["error"].startswith("500 ") for r in bad)
        assert errors == len(bad)
        # a failed agent stops: no later turns after its HTTP_ERROR row
        by_agent = {}
        for row in rows:
            by_agent.setdefault(row["agent"], []).append(row)
        for agent_rows in by_agent.values():
            fails = [r["turn"] for r in agent_rows if r["status"] == int(RequestStatus.HTTP_ERROR)]
            if fails:
                last_fail = max(fails)
                assert not [r for r in agent_rows if r["turn"] > last_fail]
                assert len([r for r in agent_rows if r["status"] == int(RequestStatus.HTTP_ERROR)]) == 1

    asyncio.run(scenario())


def test_run_timeout(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner, _, url = await start_stub(tokens_per_second=2)
        out = tmp_path / "requests.jsonl"
        workload = make_workload(agents=2, turns=2, max_tokens=8)
        start = time.monotonic()
        try:
            errors = await run_load(workload, [url], "flash-next", out, stagger_s=0.0, timeout_s=0.5)
        finally:
            await runner.cleanup()
        elapsed = time.monotonic() - start
        assert elapsed < 3.0
        rows = read_rows(out)
        assert len(rows) == 2
        assert all(r["status"] == int(RequestStatus.TIMEOUT) for r in rows)
        assert all(r["turn"] == 0 for r in rows)
        assert errors == 2

    asyncio.run(scenario())


def test_short_output(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = web.Application()

        async def chat(request: web.Request) -> web.StreamResponse:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for _ in range(3):
                chunk = {"choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}]}
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                await asyncio.sleep(0.01)
            usage = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
            await resp.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp

        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = "http://127.0.0.1:%d" % runner.addresses[0][1]
        out = tmp_path / "requests.jsonl"
        workload = make_workload(agents=1, turns=2, max_tokens=8)
        try:
            errors = await run_load(workload, [url], "flash-next", out, stagger_s=0.0)
        finally:
            await runner.cleanup()
        rows = read_rows(out)
        assert len(rows) == 1
        assert rows[0]["status"] == int(RequestStatus.SHORT_OUTPUT)
        assert rows[0]["output_tokens"] == 3
        assert rows[0]["agent"] == 0 and rows[0]["turn"] == 0
        assert errors == 1

    asyncio.run(scenario())


def test_rows_flush_as_they_finish(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner, _, url = await start_stub(tokens_per_second=20)
        out = tmp_path / "requests.jsonl"
        workload = make_workload(agents=1, turns=2, max_tokens=20)
        task = asyncio.ensure_future(run_load(workload, [url], "flash-next", out, stagger_s=0.0))
        try:
            deadline = time.monotonic() + 5.0
            lines = 0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.1)
                if out.exists():
                    with open(out, encoding="utf-8") as fh:
                        lines = len([ln for ln in fh if ln.strip()])
                if lines >= 1:
                    assert not task.done(), "the first row landed only after the run ended"
                    break
            assert lines >= 1
            errors = await task
            assert errors == 0
            assert len(read_rows(out)) == 2
        finally:
            if not task.done():
                task.cancel()
            await runner.cleanup()

    asyncio.run(scenario())


def _stub_subprocess(port: int, extra: list) -> subprocess.Popen:
    root = Path(__file__).resolve().parents[4]
    cmd = [sys.executable, "-m", "tools.flash_next.rental.stub_server", "--port",
           str(port), "--tokens-per-second", "400"] + extra
    proc = subprocess.Popen(cmd, cwd=str(root))
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=0.5) as resp:
                if resp.status == 200:
                    return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("stub server did not start")


def _run_cli(workload_path: Path, port: int, out: Path) -> int:
    return agent_load_main([
        "--workload",
        str(workload_path),
        "--servers",
        "http://127.0.0.1:%d" % port,
        "--model",
        "flash-next",
        "--out",
        str(out),
        "--stagger-seconds",
        "0",
    ])


def test_cli_exit_codes(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(make_workload(agents=2, turns=2, max_tokens=8)), encoding="utf-8")

    proc = _stub_subprocess(8931, [])
    try:
        rc = _run_cli(workload_path, 8931, tmp_path / "ok.jsonl")
        assert rc == 0
        rows = read_rows(tmp_path / "ok.jsonl")
        assert len(rows) == 4 and all(r["status"] == 0 for r in rows)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    proc = _stub_subprocess(8932, ["--fail-after", "1"])
    try:
        rc = _run_cli(workload_path, 8932, tmp_path / "bad.jsonl")
        assert rc == 1
        rows = read_rows(tmp_path / "bad.jsonl")
        assert any(r["status"] == int(RequestStatus.HTTP_ERROR) for r in rows)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
