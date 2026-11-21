# SPDX-License-Identifier: Apache-2.0
"""A fake completions server with a block prefix cache, for the prefill_sweep and prefill_job tests.

It prefills at a fixed rate, one request after the other, and stops every decoding stream while a
prefill runs (what the HPU runner does today), unless the knobs file asks for pacing. Run as a module
it stands in for ``vllm serve MODEL --port N ...`` (every other argument is ignored) with a ``/health``
endpoint, so ``prefill_job.sh`` runs end to end on a workstation.
"""

import argparse
import asyncio
import json
import os
import time

from aiohttp import web

BLOCK = 512
PREFILL_RATE = 40000.0  # prompt tokens per second
DECODE_RATE = 200.0  # tokens per second per stream


class FakeServer:
    """Token-id completions with a prefix cache of full ``BLOCK``-token blocks."""

    def __init__(self, knobs_path=None) -> None:
        self.cached: set[tuple] = set()
        self.prefilling = 0
        self.profile_calls: list[tuple[str, float]] = []
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

    async def profile(self, request: web.Request) -> web.Response:
        """``/start_profile`` and ``/stop_profile``: recorded (path, time)."""
        self.profile_calls.append((request.path, time.time()))
        return web.Response(text="")

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
            self.requests.append({"tokens": len(ids), "cached": cached, "max_tokens": body["max_tokens"],
                                  "fields": sorted(body), "ignore_eos": body.get("ignore_eos")})
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


def make_app(fake: FakeServer) -> web.Application:
    async def health(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/tokenize", fake.tokenize)
    app.router.add_get("/metrics", fake.metrics)
    app.router.add_post("/v1/completions", fake.completions)
    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="fake `vllm serve` for prefill_job.sh")
    parser.add_argument("--port", type=int, required=True)
    args, _ = parser.parse_known_args(argv)
    global PREFILL_RATE
    PREFILL_RATE = float(os.environ.get("PREFILL_FAKE_RATE", PREFILL_RATE))  # prompt tokens per second
    fake = FakeServer(os.environ.get("VLLM_HPU_SCHED_KNOBS_FILE") or None)
    print("fake server on port %d, knobs file %s" % (args.port, fake.knobs_path), flush=True)
    web.run_app(make_app(fake), host="127.0.0.1", port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
