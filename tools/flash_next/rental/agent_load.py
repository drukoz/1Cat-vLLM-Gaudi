# SPDX-License-Identifier: Apache-2.0
"""Closed-loop 32-agent streaming client for the Flash-Next rental harness
(plan M4, RUK-2709).

One asyncio task per agent plays that agent's conversation turn after turn
against a vLLM chat-completions server with streaming, and appends one
`requests.jsonl` row per request as soon as the request ends, so a killed
run keeps its rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Optional

import aiohttp

from tools.flash_next.rental.schema import RequestStatus

PROGRESS_INTERVAL_S = 30.0


def make_row(agent: int, turn: int, server: int) -> dict:
    return {
        "agent": agent,
        "turn": turn,
        "server": server,
        "prompt_tokens": 0,
        "cached_tokens": None,
        "output_tokens": 0,
        "t_send": None,
        "t_first": None,
        "t_last": None,
        "itl_p50_ms": None,
        "status": int(RequestStatus.OK),
        "error": None,
    }


class _Writer:
    """Appends rows to the JSONL file, flushing every line as it lands."""

    def __init__(self, out_path: Path) -> None:
        self._path = Path(out_path)
        self._fh = None

    def open(self) -> None:
        self._fh = open(self._path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write(self, row: dict) -> None:
        assert self._fh is not None, "writer is closed"
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()


def _finalize(row: dict, stream: dict, expected_tokens: int) -> None:
    """Fill usage-derived columns and the ITL median from collected chunks."""
    usage = stream.get("usage")
    if usage:
        row["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
        row["output_tokens"] = int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        if details is not None and details.get("cached_tokens") is not None:
            row["cached_tokens"] = int(details["cached_tokens"])
    if not row["output_tokens"]:
        row["output_tokens"] = len(stream["times"])
    times = stream["times"]
    if len(times) >= 2:
        gaps = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
        row["itl_p50_ms"] = float(statistics.median(gaps))
    if row["status"] == int(RequestStatus.OK) and row["output_tokens"] < expected_tokens:
        row["status"] = int(RequestStatus.SHORT_OUTPUT)


async def _stream_one(session: aiohttp.ClientSession, url: str, payload: dict, row: dict, stream: dict) -> str:
    """POST one streaming chat request, record timings into `row`/`stream`
    and return the streamed text. Raises aiohttp/os errors on failure."""
    row["t_send"] = time.time()
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = (await resp.text())[:300]
            row["status"] = int(RequestStatus.HTTP_ERROR)
            row["error"] = "%d %s" % (resp.status, body)
            return ""
        buffer = b""
        async for raw in resp.content.iter_any():
            buffer += raw
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]":
                    continue
                try:
                    obj = json.loads(data.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if obj.get("usage"):
                    stream["usage"] = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    now = time.time()
                    stream["texts"].append(piece)
                    stream["times"].append(now)
                    if row["t_first"] is None:
                        row["t_first"] = now
                    row["t_last"] = now
    return "".join(stream["texts"])


async def _run_agent(agent_obj: dict,
                     server_index: int,
                     server_url: str,
                     model: str,
                     turns: int,
                     max_tokens: int,
                     session: aiohttp.ClientSession,
                     writer: _Writer,
                     stagger_delay: float,
                     counter: dict) -> None:
    agent_id = int(agent_obj["agent"])
    messages = [dict(m) for m in agent_obj["messages"]]
    followups = list(agent_obj.get("followups", []))
    url = server_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stagger_delay > 0:
        await asyncio.sleep(stagger_delay)
    counter["running"] += 1
    try:
        for turn in range(turns):
            payload["messages"] = messages
            row = make_row(agent_id, turn, server_index)
            stream = {"texts": [], "times": [], "usage": None}
            counter["in_flight"] = counter.get("in_flight", 0) + 1
            try:
                text = await _stream_one(session, url, payload, row, stream)
            except asyncio.CancelledError:
                _finalize(row, stream, max_tokens)
                row["status"] = int(RequestStatus.TIMEOUT)
                row["error"] = "run timeout"
                _emit(writer, counter, row)
                raise
            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
                _finalize(row, stream, max_tokens)
                row["status"] = int(RequestStatus.TIMEOUT)
                row["error"] = "request timeout after request_timeout_s"
                _emit(writer, counter, row)
                return
            except (aiohttp.ClientError, OSError) as exc:
                row["status"] = int(RequestStatus.HTTP_ERROR)
                row["error"] = "%s: %s" % (type(exc).__name__, exc)
                _emit(writer, counter, row)
                return
            finally:
                counter["in_flight"] -= 1
            if row["status"] == int(RequestStatus.HTTP_ERROR):
                _emit(writer, counter, row)
                return
            _finalize(row, stream, max_tokens)
            _emit(writer, counter, row)
            if row["status"] != int(RequestStatus.OK):
                return
            messages.append({"role": "assistant", "content": text})
            if turn < turns - 1:
                messages.append({"role": "user", "content": followups[turn]})
    finally:
        counter["running"] -= 1


def _emit(writer: _Writer, counter: dict, row: dict) -> None:
    writer.write(row)
    counter["rows"] += 1
    if row["status"] != int(RequestStatus.OK):
        counter["errors"] += 1


async def _progress(counter: dict, total_rows: int, start: float, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=PROGRESS_INTERVAL_S)
            return
        except asyncio.TimeoutError:
            pass
        print("t=%ds done=%d/%d running=%d errors=%d" %
              (int(time.time() - start), counter["rows"], total_rows, counter["running"], counter["errors"]),
              flush=True)


async def run_load(workload: dict,
                   servers: list[str],
                   model: str,
                   out_path: Path,
                   stagger_s: float = 60.0,
                   timeout_s: float = 5400.0,
                   request_timeout_s: float = 1800.0) -> int:
    """Play every agent's conversation; return the number of non-OK rows."""
    agents = workload["agents"]
    turns = int(workload["turns"])
    max_tokens = int(workload["max_tokens"])
    n = len(agents)
    total_rows = n * turns

    writer = _Writer(out_path)
    writer.open()
    counter = {"rows": 0, "running": 0, "errors": 0, "in_flight": 0}
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=request_timeout_s)
    start = time.time()
    stop = asyncio.Event()
    reporter = asyncio.ensure_future(_progress(counter, total_rows, start, stop))
    tasks = []
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for i, agent_obj in enumerate(agents):
                server_index = i % len(servers)
                delay = (i * stagger_s / n) if n > 0 else 0.0
                tasks.append(
                    asyncio.ensure_future(
                        _run_agent(agent_obj, server_index, servers[server_index], model, turns, max_tokens, session,
                                   writer, delay, counter)))
            runner = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(asyncio.shield(runner), timeout=timeout_s)
            except asyncio.TimeoutError:
                pass
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        stop.set()
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)
        writer.close()
    return counter["errors"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Closed-loop agent load for the Flash-Next rental harness")
    parser.add_argument("--workload", required=True, help="workload.json built by workload.py")
    parser.add_argument("--servers", required=True, help="comma-separated base URLs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, help="requests.jsonl to append rows to")
    parser.add_argument("--stagger-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=5400.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)

    with open(args.workload, encoding="utf-8") as fh:
        workload = json.load(fh)
    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    if not servers:
        print("no servers given", flush=True)
        return 1

    errors = asyncio.run(
        run_load(workload, servers, args.model, Path(args.out), args.stagger_seconds, args.timeout_seconds,
                 args.request_timeout_seconds))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
