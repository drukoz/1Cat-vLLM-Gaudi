# SPDX-License-Identifier: Apache-2.0
"""Fake OpenAI server for the Flash-Next rental harness (plan M4, RUK-2709).

A single aiohttp process that answers the endpoints `agent_load.py` and
`monitor.py` use, so the whole harness runs on a Mac without a Gaudi. It is
deliberately cheap: content chunks are the word "tok ", the token count is
the same character estimate as `workload.py --estimate`, and prompt cache
hits are synthesised from a simple key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Optional

from aiohttp import web

from tools.flash_next.rental.schema import MAX_MODEL_LEN


def estimate_tokens(messages: list[dict]) -> int:
    """Chars of all message contents // 4 + 8 * messages (as in workload.py)."""
    total = sum(len(str(m.get("content", ""))) for m in messages)
    return total // 4 + 8 * len(messages)


def _cache_key(messages: list[dict]) -> str:
    """First 200 chars of the system message plus the user message."""
    system = ""
    user = ""
    for m in messages:
        if m.get("role") == "system" and not system:
            system = str(m.get("content", ""))
        elif m.get("role") == "user" and not user:
            user = str(m.get("content", ""))
    return (system + user)[:200]


class StubState:
    def __init__(self, model: str, tokens_per_second: float, fail_after: Optional[int]) -> None:
        self.model = model
        self.tokens_per_second = max(float(tokens_per_second), 1e-6)
        self.fail_after = fail_after
        self.chat_requests = 0
        self.in_flight = 0
        self.seen_keys: set[str] = set()
        self.prefix_queries = 0.0
        self.prefix_hits = 0.0
        self.generation_tokens = 0.0
        self.prompt_tokens = 0.0
        self.received: list[list[dict]] = []


def make_app(model: str, tokens_per_second: float, fail_after: Optional[int]) -> web.Application:
    state = StubState(model, tokens_per_second, fail_after)
    app = web.Application()
    app["stub"] = state
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_post("/tokenize", handle_tokenize)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/v1/completions", handle_completions)
    return app


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok", status=200)


async def handle_models(request: web.Request) -> web.Response:
    state = request.app["stub"]
    return web.json_response({"object": "list", "data": [{"id": state.model, "object": "model"}]})


async def handle_tokenize(request: web.Request) -> web.Response:
    body = await request.json()
    messages = body.get("messages")
    if messages is None:
        messages = [{"role": "user", "content": str(body.get("prompt", ""))}]
    count = estimate_tokens(messages)
    return web.json_response({"count": count, "max_model_len": MAX_MODEL_LEN, "tokens": []})


async def handle_chat(request: web.Request) -> web.Response:
    state = request.app["stub"]
    body = await request.json()
    messages = body.get("messages", [])
    state.received.append(messages)
    state.chat_requests += 1
    if state.fail_after is not None and state.chat_requests > state.fail_after:
        return web.Response(status=500, text="stub failure after %d requests" % state.fail_after)

    prompt_tokens = estimate_tokens(messages)
    key = _cache_key(messages)
    if key in state.seen_keys:
        cached = max(prompt_tokens - 100, 0)
    else:
        cached = 0
        state.seen_keys.add(key)
    state.prefix_queries += prompt_tokens
    state.prefix_hits += cached
    state.prompt_tokens += prompt_tokens

    max_tokens = int(body.get("max_tokens") or 16)
    stream = bool(body.get("stream"))

    if not stream:
        text = "tok " * max_tokens
        state.generation_tokens += max_tokens
        return web.json_response({
            "id": "stub",
            "object": "chat.completion",
            "model": state.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
                "total_tokens": prompt_tokens + max_tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        })

    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
    )
    await resp.prepare(request)
    state.in_flight += 1
    try:
        await _stream_chat(resp, state, max_tokens, prompt_tokens, cached)
    finally:
        state.in_flight -= 1
    return resp


async def _stream_chat(resp: web.StreamResponse, state: StubState, max_tokens: int, prompt_tokens: int,
                       cached: int) -> None:
    period = 1.0 / state.tokens_per_second
    await asyncio.sleep(prompt_tokens / 50000.0)
    for i in range(max_tokens):
        chunk = {
            "id": "stub",
            "object": "chat.completion.chunk",
            "model": state.model,
            "choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}],
        }
        await resp.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
        state.generation_tokens += 1
        if i < max_tokens - 1:
            await asyncio.sleep(period)
    usage_chunk = {
        "id": "stub",
        "object": "chat.completion.chunk",
        "model": state.model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    await resp.write(b"data: " + json.dumps(usage_chunk).encode("utf-8") + b"\n\n")
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()


async def handle_completions(request: web.Request) -> web.Response:
    await request.read()
    return web.json_response({"choices": [{"text": 'stub answer 7319-ALPHA {"name": "a"}', "token_ids": [1, 2, 3]}]})


async def handle_metrics(request: web.Request) -> web.Response:
    state = request.app["stub"]
    m = state.model
    lines = [
        "# TYPE vllm:num_requests_running gauge",
        'vllm:num_requests_running{model_name="%s"} %d' % (m, state.in_flight),
        "# TYPE vllm:num_requests_waiting gauge",
        'vllm:num_requests_waiting{model_name="%s"} 0' % m,
        "# TYPE vllm:kv_cache_usage_perc gauge",
        'vllm:kv_cache_usage_perc{model_name="%s"} 0.5' % m,
        "# TYPE vllm:prefix_cache_queries_total counter",
        'vllm:prefix_cache_queries_total{model_name="%s"} %g' % (m, state.prefix_queries),
        "# TYPE vllm:prefix_cache_hits_total counter",
        'vllm:prefix_cache_hits_total{model_name="%s"} %g' % (m, state.prefix_hits),
        "# TYPE vllm:num_preemptions_total counter",
        'vllm:num_preemptions_total{model_name="%s"} 0' % m,
        "# TYPE vllm:generation_tokens_total counter",
        'vllm:generation_tokens_total{model_name="%s"} %g' % (m, state.generation_tokens),
        "# TYPE vllm:prompt_tokens_total counter",
        'vllm:prompt_tokens_total{model_name="%s"} %g' % (m, state.prompt_tokens),
    ]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stub OpenAI server for the rental harness")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tokens-per-second", type=float, default=400.0)
    parser.add_argument("--model", default="flash-next")
    parser.add_argument("--fail-after", type=int, default=None)
    args = parser.parse_args(argv)

    app = make_app(args.model, args.tokens_per_second, args.fail_after)
    print("stub_server listening on port %d (model=%s, %.1f tok/s)" % (args.port, args.model, args.tokens_per_second),
          flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
