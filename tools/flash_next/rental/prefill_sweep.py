# SPDX-License-Identifier: Apache-2.0
"""Prefill settings experiment for a running Flash-Next server: minutes, not hours.

``python -m tools.flash_next.rental.prefill_sweep --url http://127.0.0.1:8000 --model flash-next --out DIR
[--plan baseline,ladder,sweep,pacing] [--knobs-file PATH] ...``

Measures, per case, the time to first token and the prefill rate (uncached prompt tokens per second
until the first token) of cold requests, alone and several at once, and what happens meanwhile to
streams that are already decoding (the canaries): their tokens per second and their longest stall
before, during and after the prefill. The cases:

* ``cold``: one request of N prompt tokens that shares no cached block with anything sent before;
* ``concurrent``: C such requests sent together (aggregate rate = their uncached tokens over the time
  from the first send to the last first token);
* ``warm``: the previous cold prompt plus D new tokens, the shape of a later conversation turn (a
  prefix hit up to the last full block, then a small prefill at a long context);
* ``turn2``: a prompt, its generated tokens (asked for as ids) and 40 new tokens sent back, which
  shows whether blocks produced while decoding are reused;
* ``shared``: requests with a common P-token prefix and different tails, one after the other and
  together, which shows whether a shared system block is computed once.

Plans: ``baseline`` runs all of them at the server's own settings; ``ladder`` sends cold requests of
growing length (the rate by context depth: a rate that falls with depth is work that grows with the
context); ``sweep`` repeats ``cold`` and ``concurrent`` for every ``--settings`` pair of step budget and
per-request prompt cap; ``pacing`` repeats ``cold`` under every ``--pacing`` pair of share and slice;
``agents`` is the workload itself in small: ``--agents`` conversations of ``--agent-context`` tokens, each
sending its next turn (the history, what it generated and ``--agent-new-tokens`` new tokens) as soon as
its last one ended, for ``--agent-seconds``, once per ``--pacing`` pair when that option is given. It
reports what an agent sees while the others' turns arrive: output tokens per second in all and per
stream while it decodes, the inter-token latency and the time to first token of a turn.
``sweep`` and ``pacing`` change the settings of the RUNNING server through ``--knobs-file``, the file the
server was started with as ``VLLM_HPU_SCHED_KNOBS_FILE`` (``vllm_gaudi/v1/core/sched/prefill_pacing.py``),
so the server starts once, with ``--max-num-batched-tokens`` at the largest budget of the sweep.
``--knobs JSON`` puts knobs of its own under every case of the run (for example
``{"prompt_graph_max_tokens": 140000}``, the workers' prompt HPU-graph threshold, or a prompt cap), so one
server measures several arms, one run after the other; ``--label`` names the arm in the report.

Every request asks for an exact number of tokens with ``max_tokens`` and ``ignore_eos`` and never sends
``min_tokens``: that field adds a logits processor that masks the stop tokens of its rows at every step
(an ``index_put_`` per step), which would be part of every inter-token time measured here.

Prompts are token ids sent to ``/v1/completions`` (exact lengths, no chat template): the ids of the
repository's own documents and sources, tokenized once through the server's ``/tokenize``
(``--synthetic-ids`` skips that). Every cold prompt starts with ids no earlier prompt started with, so
a rerun against the same server stays cold. Each row is appended to ``<out>/cases.jsonl`` as soon as
its case ends; ``<out>/prefill_sweep.json`` and ``prefill_sweep.md`` are written at the end.

Python 3.10, standard library plus ``aiohttp`` (in the vLLM image).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import aiohttp

from tools.flash_next.rental.monitor import parse_prometheus

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_GLOBS = ("docs/design/flash-next/**/*.md", "vllm_gaudi/**/*.py", "tools/flash_next/**/*.py", "README.md")
TOKENIZE_CHARS = 60000
TOKENIZE_TIMEOUT_S = 120.0
SALT_TOKENS = 16
KNOBS_SETTLE_S = 0.8  # the engine polls the knobs file every 0.25 s
SYNTHETIC_ID_RANGE = (1000, 60000)  # ordinary vocabulary ids, far from the special tokens at the top
RECORDED_TOKEN_IDS = 64  # output token ids kept per request row (--return-ids)

PLANS = ("baseline", "ladder", "sweep", "pacing", "agents", "capture")
DEFAULT_SETTINGS = "8192:2048,8192:8192,16384:4096,16384:16384,32768:8192,32768:32768,65536:16384,65536:0"
DEFAULT_PACING = "100:0,50:2048,25:2048,25:512,10:2048"
METRIC_NAMES = {
    "prompt_tokens": "vllm:prompt_tokens",
    "generation_tokens": "vllm:generation_tokens",
    "prefix_hits": "vllm:prefix_cache_hits",
    "prefix_queries": "vllm:prefix_cache_queries",
    "preemptions": "vllm:num_preemptions",
}


# ----- token ids -----


class Corpus:
    """A long list of ordinary token ids that prompts are cut from."""

    def __init__(self, ids: list[int], nonce: str) -> None:
        if len(ids) < 4096:
            raise ValueError(f"the corpus has only {len(ids)} token ids")
        self.ids = ids
        self.nonce = nonce
        self._serial = 0

    def _digest(self, label: str) -> bytes:
        return hashlib.sha256(f"{self.nonce}/{label}".encode()).digest()

    def salt(self, label: str) -> list[int]:
        """``SALT_TOKENS`` corpus ids picked by ``label`` and this run's nonce: a first block nobody sent before."""
        raw = self._digest(label) + self._digest(label + "+")
        return [self.ids[int.from_bytes(raw[2 * i:2 * i + 4], "big") % len(self.ids)] for i in range(SALT_TOKENS)]

    def take(self, count: int, label: str) -> list[int]:
        """``count`` ids starting at an offset picked by ``label`` (wrapping around the corpus)."""
        start = int.from_bytes(self._digest(label)[:8], "big") % len(self.ids)
        out: list[int] = []
        while len(out) < count:
            out.extend(self.ids[start:start + count - len(out)])
            start = 0
        return out

    def cold(self, count: int, label: str | None = None) -> list[int]:
        """A prompt of exactly ``count`` ids that shares no leading block with any other."""
        if label is None:
            self._serial += 1
            label = f"cold-{self._serial}"
        if count <= SALT_TOKENS:
            return self.salt(label)[:count]
        return self.salt(label) + self.take(count - SALT_TOKENS, label)


def synthetic_ids(count: int, seed: int = 20261120) -> list[int]:
    """Deterministic pseudo-random ordinary ids (a 64-bit LCG), for runs without ``/tokenize``."""
    low, high = SYNTHETIC_ID_RANGE
    state = seed & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(low + (state >> 33) % (high - low))
    return out


def corpus_texts(root: Path) -> list[str]:
    texts = []
    seen = set()
    for pattern in CORPUS_GLOBS:
        for path in sorted(root.glob(pattern), key=lambda p: p.as_posix()):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return texts


def tokenize_corpus(url: str, model: str, texts: list[str], want: int) -> list[int]:
    """Token ids of ``texts`` from the server's ``/tokenize``, ``TOKENIZE_CHARS`` characters per call,
    stopping once ``want`` ids are there."""
    ids: list[int] = []
    endpoint = url.rstrip("/") + "/tokenize"
    for text in texts:
        for start in range(0, len(text), TOKENIZE_CHARS):
            body = json.dumps({"model": model, "prompt": text[start:start + TOKENIZE_CHARS],
                               "add_special_tokens": False}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=TOKENIZE_TIMEOUT_S) as resp:
                ids.extend(int(t) for t in json.loads(resp.read().decode("utf-8"))["tokens"])
            if len(ids) >= want:
                return ids
    return ids


# ----- one streamed request -----


def new_record(name: str, prompt_tokens: int) -> dict:
    return {"name": name, "sent_tokens": prompt_tokens, "prompt_tokens": None, "cached_tokens": None,
            "output_tokens": 0, "t_send": None, "t_first": None, "t_last": None, "status": 0, "error": None,
            "arrivals": [], "token_ids": []}


async def stream_completion(session: aiohttp.ClientSession, url: str, payload: dict, record: dict) -> dict:
    """POST one streaming ``/v1/completions`` request and fill ``record``.

    ``arrivals`` gets one ``[time, cumulative output tokens]`` pair per chunk that carried tokens. The
    count comes from the chunk's running ``usage.completion_tokens`` (``continuous_usage_stats``), which
    stays right when one chunk carries several tokens (speculative decoding); a server that sends no
    running usage counts one token per text chunk. A cancelled request keeps what it received.
    """
    record["t_send"] = time.time()
    try:
        async with session.post(url.rstrip("/") + "/v1/completions", json=payload) as resp:
            if resp.status != 200:
                record["status"] = 1
                record["error"] = "%d %s" % (resp.status, (await resp.text())[:300])
                return record
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
                    _consume_chunk(obj, record)
    except asyncio.CancelledError:
        record["status"] = record["status"] or 3  # stopped by the harness (a canary)
        raise
    except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
        record["status"] = 2
        record["error"] = "%s: %s" % (type(exc).__name__, exc)
    wanted = int(payload.get("max_tokens") or 0)
    if record["status"] == 0 and payload.get("ignore_eos") and record["output_tokens"] < wanted:
        # ignore_eos gives exactly max_tokens: a stream that ended before is a server that dropped it (00014: the
        # engine died and 28 streams ended without a token, which the setup row then counted as served)
        record["status"] = 4
        record["error"] = "the stream ended after %d of %d tokens" % (record["output_tokens"], wanted)
    return record


def _consume_chunk(obj: dict, record: dict) -> None:
    now = time.time()
    usage = obj.get("usage")
    choices = obj.get("choices") or []
    produced = None
    if usage:
        record["prompt_tokens"] = usage.get("prompt_tokens", record["prompt_tokens"])
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            record["cached_tokens"] = int(details["cached_tokens"])
        if usage.get("completion_tokens") is not None:
            produced = int(usage["completion_tokens"])
    carried = False
    for choice in choices:
        ids = choice.get("token_ids")
        if ids:
            record["token_ids"].extend(int(t) for t in ids)
            carried = True
        if choice.get("text"):
            carried = True
    if produced is None:
        if not carried:
            return
        produced = record["output_tokens"] + 1
    if produced > record["output_tokens"]:
        record["output_tokens"] = produced
        record["arrivals"].append([now, produced])
        if record["t_first"] is None:
            record["t_first"] = now
        record["t_last"] = now


def completion_payload(model: str, prompt_ids: list[int], max_tokens: int, return_ids: bool = False,
                       cache_salt: Optional[str] = None) -> dict:
    """Exactly ``max_tokens`` greedy tokens: ``ignore_eos`` gives the length, no ``min_tokens`` (module text)."""
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "add_special_tokens": False,
    }
    if return_ids:
        payload["return_token_ids"] = True
    if cache_salt:
        payload["cache_salt"] = cache_salt  # vLLM: a salted request shares no prefix-cache block with others
    return payload


# ----- statistics -----


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def window_stats(arrivals: list[list[float]], start: float, end: float) -> dict:
    """Decode statistics of one stream inside the window ``(start, end]``.

    ``tokens_per_s`` is the tokens that arrived in the window over its length. An inter-token latency is
    the time between two arrivals divided by the tokens the later one carried, counted in the window the
    later arrival falls in. ``stall_ms`` is the longest time inside the window without an arrival (the
    waits that reach into the window from before it and out of it at its end are clipped to the window).
    """
    empty = {"seconds": max(end - start, 0.0), "tokens": 0, "tokens_per_s": None, "itl_p50_ms": None,
             "itl_p99_ms": None, "stall_ms": None}
    if end <= start:
        return empty
    tokens = 0
    gaps: list[float] = []
    longest = 0.0
    previous_t: Optional[float] = None
    previous_n = 0
    last_inside = start
    for t, n in arrivals:
        if start < t <= end:
            tokens += n - previous_n
            longest = max(longest, t - max(previous_t if previous_t is not None else start, start))
            if previous_t is not None:
                gaps.extend([(t - previous_t) / max(n - previous_n, 1)] * max(n - previous_n, 1))
            last_inside = t
        elif t > end:
            break
        previous_t, previous_n = t, n
    longest = max(longest, end - last_inside)
    return {
        "seconds": end - start,
        "tokens": tokens,
        "tokens_per_s": tokens / (end - start),
        "itl_p50_ms": None if not gaps else statistics.median(gaps) * 1000.0,
        "itl_p99_ms": None if not gaps else percentile(gaps, 0.99) * 1000.0,
        "stall_ms": longest * 1000.0,
    }


def summarize_canaries(canaries: list[dict], t_ready: float, t_send: float, t_first_all: float, t_stop: float) -> dict:
    """Per window (``before``, ``during``, ``after`` the prefill): the streams' median tokens per second,
    their slowest stream, the inter-token latency (the median stream's median, the worst stream's 99th
    percentile) and the longest stall of any stream."""
    out = {}
    for name, (start, end) in (("before", (t_ready, t_send)), ("during", (t_send, t_first_all)),
                               ("after", (t_first_all, t_stop))):
        rows = [window_stats(c["arrivals"], start, end) for c in canaries]
        rates = [r["tokens_per_s"] for r in rows if r["tokens_per_s"] is not None]
        stalls = [r["stall_ms"] for r in rows if r["stall_ms"] is not None]
        p50s = [r["itl_p50_ms"] for r in rows if r["itl_p50_ms"] is not None]
        p99s = [r["itl_p99_ms"] for r in rows if r["itl_p99_ms"] is not None]
        out[name] = {
            "seconds": max(end - start, 0.0),
            "streams": len(rows),
            "tokens_per_s_p50": statistics.median(rates) if rates else None,
            "tokens_per_s_min": min(rates) if rates else None,
            "itl_p50_ms": statistics.median(p50s) if p50s else None,
            "itl_p99_ms": max(p99s) if p99s else None,
            "stall_ms": max(stalls) if stalls else None,
        }
    return out


def request_row(record: dict) -> dict:
    row = {k: record[k] for k in ("name", "sent_tokens", "prompt_tokens", "cached_tokens", "output_tokens",
                                  "t_send", "t_first", "t_last", "status", "error")}
    prompt = record["prompt_tokens"] if record["prompt_tokens"] is not None else record["sent_tokens"]
    cached = record["cached_tokens"] or 0
    row["uncached_tokens"] = prompt - cached
    row["ttft_s"] = None if record["t_first"] is None else record["t_first"] - record["t_send"]
    row["prefill_tokens_per_s"] = (None if not row["ttft_s"] else row["uncached_tokens"] / row["ttft_s"])
    if record["token_ids"]:
        row["token_ids"] = record["token_ids"][:RECORDED_TOKEN_IDS]  # greedy output, to compare two servers
    return row


# ----- the runner -----


class Runner:

    def __init__(self, args, corpus: Corpus, out_dir: Path) -> None:
        self.args = args
        self.url = args.url
        self.model = args.model
        self.corpus = corpus
        self.out_dir = out_dir
        self.rows: list[dict] = []
        self.base_label = args.label
        self.base_knobs: dict = dict(args.knobs or {})
        self.setting = {"label": self.base_label, "knobs": None}
        self._session: aiohttp.ClientSession | None = None
        self._cases_file = None

    async def __aenter__(self) -> "Runner":
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.args.request_timeout)
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout)
        self._cases_file = open(self.out_dir / "cases.jsonl", "a", encoding="utf-8")
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()
        if self._cases_file is not None:
            self._cases_file.close()

    def metrics(self) -> dict:
        try:
            with urllib.request.urlopen(self.url.rstrip("/") + "/metrics", timeout=10.0) as resp:
                sums = parse_prometheus(resp.read().decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            return {}
        return {key: sums.get(name) for key, name in METRIC_NAMES.items()}

    async def set_knobs(self, label: str, knobs: dict | None) -> None:
        """Write the knobs for the next cases (``None``: leave the server alone). The run's own knobs
        (``--knobs``) lie under them, and its label (``--label``) goes in front of ``label``."""
        if knobs is not None:
            knobs = {**self.base_knobs, **knobs}
        if label == "server":
            label = self.base_label
        elif self.base_label != "server":
            label = f"{self.base_label}, {label}"
        self.setting = {"label": label, "knobs": knobs}
        if knobs is None:
            return
        if not self.args.knobs_file:
            raise SystemExit("prefill_sweep: this plan changes scheduler settings and needs --knobs-file "
                             "(the server's VLLM_HPU_SCHED_KNOBS_FILE)")
        path = Path(self.args.knobs_file)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(knobs, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        await asyncio.sleep(KNOBS_SETTLE_S)

    async def _start_canaries(self) -> tuple[list[dict], list[asyncio.Task], float]:
        count = self.args.canaries
        records, tasks = [], []
        for i in range(count):
            prompt = self.corpus.cold(self.args.canary_prompt_tokens)
            record = new_record(f"canary-{i}", len(prompt))
            payload = completion_payload(self.model, prompt, self.args.canary_max_tokens)
            records.append(record)
            tasks.append(asyncio.ensure_future(stream_completion(self._session, self.url, payload, record)))
        deadline = time.time() + self.args.request_timeout
        while records and time.time() < deadline:
            if all(r["output_tokens"] >= self.args.canary_warm_tokens or r["status"] in (1, 2) for r in records):
                break
            await asyncio.sleep(0.05)
        return records, tasks, time.time()

    async def _stop_canaries(self, tasks: list[asyncio.Task]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_case(self, case: str, prompts: list[tuple[str, list[int]]], max_tokens: int = 0,
                       return_ids: bool = False, canaries: bool = True, cache_salt: Optional[str] = None,
                       extra: Optional[dict] = None) -> dict:
        """Send ``prompts`` together and wait for them; with canary streams decoding around them.

        ``max_tokens`` 0 means ``--case-max-tokens``; ``--return-ids`` asks every case for its token ids.
        """
        max_tokens = max_tokens or self.args.case_max_tokens
        return_ids = return_ids or self.args.return_ids
        before = self.metrics()
        canary_records, canary_tasks, t_ready = ([], [], time.time())
        if canaries and self.args.canaries > 0:
            canary_records, canary_tasks, t_ready = await self._start_canaries()
            await asyncio.sleep(self.args.settle_seconds)
        records = [new_record(name, len(ids)) for name, ids in prompts]
        t_send = time.time()
        await asyncio.gather(*[
            stream_completion(self._session, self.url,
                              completion_payload(self.model, ids, max_tokens, return_ids, cache_salt=cache_salt),
                              record) for (_, ids), record in zip(prompts, records)
        ])
        firsts = [r["t_first"] for r in records if r["t_first"] is not None]
        t_first_all = max(firsts) if firsts else time.time()
        if canary_tasks:
            await asyncio.sleep(self.args.settle_seconds)
        t_stop = time.time()
        await self._stop_canaries(canary_tasks)
        after = self.metrics()

        rows = [request_row(r) for r in records]
        uncached = sum(r["uncached_tokens"] for r in rows)
        span = t_first_all - t_send
        result = {
            "setting": self.setting["label"],
            "knobs": self.setting["knobs"],
            "case": case,
            "requests": rows,
            "aggregate": {
                "requests": len(rows),
                "uncached_tokens": uncached,
                "seconds_to_last_first_token": span,
                "prefill_tokens_per_s": ((uncached / span) if span > 0 and firsts
                                         and all(r["status"] == 0 for r in rows) else None),
                "errors": sum(1 for r in rows if r["status"] != 0),
            },
            "canaries": (summarize_canaries(canary_records, t_ready, t_send, t_first_all, t_stop)
                         if canary_records else None),
            "metrics_delta": {k: (None if before.get(k) is None or after.get(k) is None else after[k] - before[k])
                              for k in METRIC_NAMES},
        }
        result.update(extra or {})
        self.rows.append(result)
        self._cases_file.write(json.dumps(result) + "\n")
        self._cases_file.flush()
        print(format_line(result), flush=True)
        result["_records"] = records  # for a follow-up case (token ids); not written
        return result

    # ----- cases -----

    async def cold(self, tokens: int, tag: str = "") -> tuple[dict, list[int]]:
        prompt = self.corpus.cold(tokens)
        result = await self.run_case(f"cold {tokens}{tag}", [("cold", prompt)])
        return result, prompt

    async def concurrent(self, count: int, tokens: int) -> dict:
        prompts = [(f"cold-{i}", self.corpus.cold(tokens)) for i in range(count)]
        return await self.run_case(f"concurrent {count}x{tokens}", prompts)

    async def warm(self, base_prompt: list[int], extra: int) -> dict:
        prompt = base_prompt + self.corpus.take(extra, f"warm-{len(base_prompt)}-{extra}-{len(self.rows)}")
        name = f"warm {len(base_prompt)}+{extra}"
        salted = self.args.warm_salted
        result = await self.run_case(name, [("warm", prompt)], return_ids=salted)
        if salted:
            # the same turn again with a cache salt: computed from token 0 without any prefix hit, so its greedy
            # tokens are what the turn must give; a prefix hit that read a wrong recurrent state gives others
            first = result["_records"][0]["token_ids"]
            again = await self.run_case(f"{name} salted", [("salted", prompt)], return_ids=True,
                                        cache_salt=f"fresh-{self.corpus.nonce}-{len(self.rows)}",
                                        extra={"same_tokens_as": name})
            second = again["_records"][0]["token_ids"]
            equal = bool(first) and first == second
            again["tokens_equal"] = equal
            self.rows[-1]["tokens_equal"] = equal
            common = next((i for i, (x, y) in enumerate(zip(first, second)) if x != y), min(len(first), len(second)))
            print("SALTED %s: tokens %s (%d of %d equal from the start; cached %s then %s)" % (
                name, "SAME" if equal else "DIFF", common, len(first),
                result["requests"][0].get("cached_tokens"), again["requests"][0].get("cached_tokens")), flush=True)
        return result

    async def turn2(self, tokens: int, generated: int, extra: int = 40) -> dict:
        prompt = self.corpus.cold(tokens)
        first = await self.run_case(f"turn1 {tokens} gen {generated}", [("turn1", prompt)], max_tokens=generated,
                                    return_ids=True, canaries=False)
        produced = first["_records"][0]["token_ids"]
        if len(produced) < generated:
            print(f"turn2: the server returned {len(produced)} of {generated} token ids (return_token_ids); "
                  "the second turn is sent with what came back", flush=True)
        again = prompt + produced + self.corpus.take(extra, f"turn2-{tokens}-{len(self.rows)}")
        name = f"turn2 {tokens}+{len(produced)}+{extra}"
        salted = self.args.warm_salted
        result = await self.run_case(name, [("turn2", again)], return_ids=salted)
        if salted:
            # the second turn again from token 0 (a cache salt): what it must give if its prefix hit read the right
            # state; with VLLM_HPU_PREFIX_KEEP_DECODE_END the hit lies inside the first turn's output
            first_ids = result["_records"][0]["token_ids"]
            fresh = await self.run_case(f"{name} salted", [("salted", again)], return_ids=True,
                                        cache_salt=f"fresh-{self.corpus.nonce}-{len(self.rows)}",
                                        extra={"same_tokens_as": name})
            second_ids = fresh["_records"][0]["token_ids"]
            equal = bool(first_ids) and first_ids == second_ids
            fresh["tokens_equal"] = equal
            self.rows[-1]["tokens_equal"] = equal
            print("SALTED %s: tokens %s (cached %s then %s)" % (name, "SAME" if equal else "DIFF",
                                                                result["requests"][0].get("cached_tokens"),
                                                                fresh["requests"][0].get("cached_tokens")), flush=True)
        return result

    async def shared(self, prefix_tokens: int, tokens: int, together: int) -> None:
        prefix = self.corpus.cold(prefix_tokens)
        tail = tokens - prefix_tokens
        serial = len(self.rows)
        await self.run_case(f"shared {prefix_tokens} first", [("first", prefix + self.corpus.take(tail,
                                                                                             f"sh-{serial}-a"))])
        await self.run_case(f"shared {prefix_tokens} second", [("second", prefix + self.corpus.take(tail,
                                                                                               f"sh-{serial}-b"))])
        await self.run_case(f"shared {prefix_tokens} third", [("third", prefix + self.corpus.take(tail,
                                                                                             f"sh-{serial}-c"))])
        fresh = self.corpus.cold(prefix_tokens)
        prompts = [(f"together-{i}", fresh + self.corpus.take(tail, f"sh-{serial}-t{i}")) for i in range(together)]
        await self.run_case(f"shared {prefix_tokens} x{together} together", prompts)

    # ----- plans -----

    async def plan_baseline(self) -> None:
        a = self.args
        last_prompt = None
        for tokens in a.cold_tokens:
            for repeat in range(a.repeat):
                _, last_prompt = await self.cold(tokens, tag="" if a.repeat == 1 else f" pass {repeat + 1}")
            for extra in a.warm_tokens:
                await self.warm(last_prompt, extra)
        for count in a.concurrent:
            await self.concurrent(count, a.concurrent_tokens)
        if a.turn2_tokens > 0:
            await self.turn2(a.turn2_tokens, a.turn2_generate)
        if a.shared_prefix > 0:
            await self.shared(a.shared_prefix, a.shared_tokens, a.shared_together)

    async def plan_ladder(self) -> None:
        for tokens in self.args.ladder:
            await self.cold(tokens)

    async def plan_sweep(self) -> None:
        a = self.args
        for budget, threshold in a.settings:
            await self.set_knobs(f"budget {budget} cap {threshold or 'off'}",
                           {"max_num_scheduled_tokens": budget, "long_prefill_token_threshold": threshold})
            for repeat in range(a.repeat):
                await self.cold(a.sweep_tokens, tag="" if a.repeat == 1 else f" pass {repeat + 1}")
            for count in a.sweep_concurrent:
                await self.concurrent(count, a.sweep_concurrent_tokens or a.sweep_tokens)
        await self.set_knobs("server", {})

    async def plan_pacing(self) -> None:
        a = self.args
        for share, slice_tokens in a.pacing:
            knobs = ({"mode": 0} if share >= 100 else {"mode": 1, "share_percent": share,
                                                         "slice_tokens": slice_tokens})
            await self.set_knobs("pacing off" if share >= 100 else f"share {share}% slice {slice_tokens}", knobs)
            await self.cold(a.sweep_tokens)
        await self.set_knobs("server", {})

    # ----- the agent loop -----

    async def _profile(self, action: str) -> dict:
        """POST ``/start_profile`` or ``/stop_profile`` (the server needs ``--profiler-config``); the status and how
        long the call took (``/stop_profile`` writes the traces and holds the server for minutes)."""
        t0 = time.time()
        try:
            async with self._session.post(self.url.rstrip("/") + f"/{action}_profile", json={},
                                          timeout=aiohttp.ClientTimeout(total=self.args.profile_timeout)) as resp:
                body = (await resp.text())[:200]
                status = resp.status
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as error:
            body, status = f"{type(error).__name__}: {error}"[:200], 0
        out = {"status": status, "seconds": round(time.time() - t0, 3), "at": t0, "body": body}
        print("CAPTURE %s_profile: status %s after %.1f s%s" % (action, status, out["seconds"],
                                                              "" if status == 200 else f" ({body})"), flush=True)
        return out

    async def plan_capture(self) -> None:
        """Device traces of prompt forwards for step-anatomy (the server starts with ``--profiler-config`` (torch)
        and ``HABANA_PROFILE=profile_api_light``). No canary streams: a decoding stream makes the traced steps
        mixed. The timings of the traced cases are not speed numbers.

        ``--capture-cached C --capture-new N``: a cold prompt of C + 1 tokens (the cache keeps its first C tokens),
        then C + N tokens (it compiles the shapes; not traced), then C + N other tokens inside ``/start_profile`` ..
        ``/stop_profile`` with one output token: the window holds that prompt forward (one forward of N tokens at
        context C when C and C + N are recurrent block boundaries). ``--capture-cold T``: a cold prompt of T tokens;
        ``--capture-after`` seconds after it is sent a window of ``--capture-seconds`` (one or two chunks).
        """
        args = self.args
        cached, new = args.capture_cached, args.capture_new
        if cached > 0 and new > 0:
            base = self.corpus.cold(cached + 1, f"capture-base-{cached}")
            await self.run_case(f"capture prime {cached + 1}", [("prime", base)], canaries=False)
            warm = base[:cached] + self.corpus.take(new, f"capture-warm-{cached}-{new}")
            await self.run_case(f"capture warm-up {cached}+{new}", [("warm-up", warm)], canaries=False)
            traced = base[:cached] + self.corpus.take(new, f"capture-traced-{cached}-{new}")
            start = await self._profile("start")
            result = await self.run_case(f"capture traced {cached}+{new}", [("traced", traced)], canaries=False)
            stop = await self._profile("stop")
            result["trace"] = {"start_profile": start, "stop_profile": stop, "window": "one prompt forward"}
        if args.capture_cold > 0:
            prompt = self.corpus.cold(args.capture_cold, f"capture-cold-{args.capture_cold}")
            task = asyncio.ensure_future(self.run_case(f"capture cold {args.capture_cold}", [("cold", prompt)],
                                                       canaries=False))
            await asyncio.sleep(args.capture_after)
            early = task.done()
            if early:
                print("CAPTURE the cold prompt ended before the window (%.1f s): the trace holds no chunk of it"
                      % args.capture_after, flush=True)
            start = await self._profile("start")
            await asyncio.sleep(args.capture_seconds)
            stop = await self._profile("stop")
            result = await task
            result["trace"] = {"start_profile": start, "stop_profile": stop, "after_s": args.capture_after,
                               "window_s": args.capture_seconds, "prompt_done_before_window": early}

    async def plan_agents(self) -> None:
        """``--agents`` conversations in a closed loop (the module text), once per ``--agent-pacing`` pair."""
        a = self.args
        contexts = a.agent_contexts or [a.agent_context]
        prompts = [self.corpus.cold(contexts[i % len(contexts)], f"agent-{i}") for i in range(a.agents)]
        setup = await self.run_case(f"agents setup {a.agents}x{self._contexts_label()}",
                                    [(f"agent-{i}", ids) for i, ids in enumerate(prompts)],
                                    max_tokens=a.agent_output, return_ids=True, canaries=False)
        histories = [ids + record["token_ids"] for ids, record in zip(prompts, setup["_records"])]
        if a.agent_warmup_seconds > 0:
            # every shape the measured loops meet compiles (and is captured) here, not inside a measured loop
            await self.agent_loop(histories, seconds=a.agent_warmup_seconds, warmup=True)
        if not a.agent_pacing:
            await self.agent_loop(histories)
            return
        for share, slice_tokens in a.agent_pacing:
            knobs = ({"mode": 0} if share >= 100 else {"mode": 1, "share_percent": share,
                                                         "slice_tokens": slice_tokens})
            await self.set_knobs("pacing off" if share >= 100 else f"share {share}% slice {slice_tokens}", knobs)
            await self.agent_loop(histories)
        await self.set_knobs("server", {})

    def _contexts_label(self) -> str:
        contexts = self.args.agent_contexts or [self.args.agent_context]
        return str(contexts[0]) if len(set(contexts)) == 1 else "%d..%d" % (min(contexts), max(contexts))

    async def agent_loop(self, histories: list[list[int]], seconds: Optional[float] = None,
                         warmup: bool = False) -> dict:
        """Every agent sends turn after turn until ``--agent-seconds`` (or ``seconds``) have passed; ``histories``
        (one token list per agent) grow by each turn's new tokens and output, so a later loop goes on from there.
        ``warmup`` labels the loop's row "<setting> warm-up" (its compiles and graph captures stay out of the
        measured loops that follow)."""
        a = self.args
        before = self.metrics()
        t_start = time.time()
        # new turn tokens in every loop of every run: a turn an earlier loop sent would be a full prefix hit
        stamp = "%s-%d-%d" % (self.setting["label"], len(self.rows), int(t_start * 1000))
        deadline = t_start + (a.agent_seconds if seconds is None else seconds)
        records: list[dict] = []

        async def one_agent(index: int) -> None:
            turn = 0
            while time.time() < deadline:
                turn += 1
                prompt = histories[index] + self.corpus.take(a.agent_new_tokens, f"agent-{index}-{stamp}-{turn}")
                if len(prompt) + a.agent_output > a.agent_max_context:
                    return  # the conversation is full
                record = new_record(f"agent-{index}-turn-{turn}", len(prompt))
                records.append(record)
                await stream_completion(self._session, self.url,
                                        completion_payload(self.model, prompt, a.agent_output, True), record)
                if record["status"] != 0:
                    return
                histories[index] = prompt + record["token_ids"]

        await asyncio.gather(*[one_agent(i) for i in range(len(histories))])
        t_end = time.time()  # the turns in flight at the deadline have ended
        after = self.metrics()
        rows = [request_row(r) for r in records]
        result = {
            "setting": self.setting["label"] + (" warm-up" if warmup else ""),
            "knobs": self.setting["knobs"],
            "case": "agents %dx%s +%d in %d out" % (len(histories), self._contexts_label(), a.agent_new_tokens,
                                                    a.agent_output),
            "requests": rows,
            "aggregate": {
                "requests": len(rows),
                "uncached_tokens": sum(r["uncached_tokens"] for r in rows),
                "seconds_to_last_first_token": t_end - t_start,
                "prefill_tokens_per_s": sum(r["uncached_tokens"] for r in rows) / (t_end - t_start),
                "errors": sum(1 for r in rows if r["status"] != 0),
            },
            "canaries": None,
            "agents": summarize_agents(records, t_start, t_end),
            "metrics_delta": {k: (None if before.get(k) is None or after.get(k) is None else after[k] - before[k])
                              for k in METRIC_NAMES},
        }
        self.rows.append(result)
        self._cases_file.write(json.dumps(result) + "\n")
        self._cases_file.flush()
        print(format_line(result), flush=True)
        return result


def summarize_agents(records: list[dict], t_start: float, t_end: float) -> dict:
    """What the agents of one loop saw between ``t_start`` and ``t_end`` (every turn had ended by then).

    ``output_tokens_per_s`` is all output over the wall time (prefill waits included: the rate the workload
    gets); ``stream_tokens_per_s`` is a turn's output over the time from its first to its last token (the
    rate an agent sees while it decodes), median and slowest turn; the inter-token latencies are over every
    token of every turn; ``ttft_s`` is per turn (the prefill of the turn's uncached tokens and the wait for
    it); ``decoding_share`` is the part of the wall time an average agent spent between a first and a last
    token.
    """
    done = [r for r in records if r["status"] == 0 and r["t_first"] is not None]
    gaps: list[float] = []
    rates: list[float] = []
    decoding = 0.0
    for r in done:
        previous_t, previous_n = None, 0
        for t, n in r["arrivals"]:
            if previous_t is not None:
                gaps.extend([(t - previous_t) / max(n - previous_n, 1)] * max(n - previous_n, 1))
            previous_t, previous_n = t, n
        span = r["t_last"] - r["t_first"]
        decoding += span
        if span > 0 and r["output_tokens"] > 1:
            rates.append((r["output_tokens"] - 1) / span)
    ttfts = [r["t_first"] - r["t_send"] for r in done]
    wall = max(t_end - t_start, 1e-9)
    agents = len({r["name"].rsplit("-turn-", 1)[0] for r in records}) or 1
    uncached = [(r["prompt_tokens"] if r["prompt_tokens"] is not None else r["sent_tokens"]) - (r["cached_tokens"] or 0)
                for r in done]
    return {
        "seconds": wall,
        "agents": agents,
        "turns": len(done),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "output_tokens_per_s": sum(r["output_tokens"] for r in records) / wall,
        "stream_tokens_per_s_p50": statistics.median(rates) if rates else None,
        "stream_tokens_per_s_min": min(rates) if rates else None,
        "itl_p50_ms": None if not gaps else statistics.median(gaps) * 1000.0,
        "itl_p90_ms": None if not gaps else percentile(gaps, 0.90) * 1000.0,
        "itl_p99_ms": None if not gaps else percentile(gaps, 0.99) * 1000.0,
        "itl_max_ms": None if not gaps else max(gaps) * 1000.0,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p90_s": percentile(ttfts, 0.90) if ttfts else None,
        "ttft_p99_s": percentile(ttfts, 0.99) if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "uncached_tokens_per_turn": statistics.mean(uncached) if uncached else None,
        "decoding_share": decoding / (wall * agents),
    }


def format_line(result: dict) -> str:
    agg = result["aggregate"]
    rows = result["requests"]
    if result.get("agents"):
        g = result["agents"]
        text = ("%-22s %-28s %d turns in %.0f s: output %s tok/s in all, %s per stream while decoding (slowest "
                "%s); ITL ms p50 %s p99 %s max %s; TTFT s p50 %s p99 %s max %s; %s uncached tokens per turn" % (
                    result["setting"], result["case"], g["turns"], g["seconds"], _fmt(g["output_tokens_per_s"]),
                    _fmt(g["stream_tokens_per_s_p50"]), _fmt(g["stream_tokens_per_s_min"]), _fmt(g["itl_p50_ms"]),
                    _fmt(g["itl_p99_ms"]), _fmt(g["itl_max_ms"], 0), _fmt(g["ttft_p50_s"], 2),
                    _fmt(g.get("ttft_p99_s"), 2), _fmt(g["ttft_max_s"], 2), _fmt(g["uncached_tokens_per_turn"], 0)))
        if agg["errors"]:
            text += " | ERRORS %d: %s" % (agg["errors"], next(r["error"] for r in rows if r["status"] != 0))
        return text
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    text = "%-22s %-28s uncached %7d cached %7d ttft %s s rate %s tok/s" % (
        result["setting"], result["case"], agg["uncached_tokens"], cached,
        "-" if not ttfts else "%.2f..%.2f" % (min(ttfts), max(ttfts)) if len(ttfts) > 1 else "%.2f" % ttfts[0],
        "-" if agg["prefill_tokens_per_s"] is None else "%.0f" % agg["prefill_tokens_per_s"])
    canaries = result["canaries"]
    if canaries:
        text += " | streams tok/s before %s during %s, ITL p50 ms before %s during %s, stall %s ms" % (
            _fmt(canaries["before"]["tokens_per_s_p50"]), _fmt(canaries["during"]["tokens_per_s_p50"]),
            _fmt(canaries["before"]["itl_p50_ms"]), _fmt(canaries["during"]["itl_p50_ms"]),
            _fmt(canaries["during"]["stall_ms"], 0))
    if agg["errors"]:
        text += " | ERRORS %d: %s" % (agg["errors"], next(r["error"] for r in rows if r["status"] != 0))
    return text


def _fmt(value, digits: int = 1) -> str:
    return "-" if value is None else ("%." + str(digits) + "f") % value


def write_report(out_dir: Path, rows: list[dict], meta: dict) -> None:
    clean = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    (out_dir / "prefill_sweep.json").write_text(json.dumps({"meta": meta, "cases": clean}, indent=1),
                                                encoding="utf-8")
    lines = [
        "# Prefill sweep", "",
        "- server: %s, model %s" % (meta["url"], meta["model"]),
        "- plans: %s; canary streams: %d" % (", ".join(meta["plans"]), meta["canaries"]),
        "- rate = uncached prompt tokens over the time to the (last) first token; streams = the canaries, "
        "median tokens per second per stream and the longest stall of any stream while the case prefilled", "",
        "| setting | case | requests | uncached | cached | TTFT s (min..max) | rate tok/s | streams before tok/s | "
        "streams during tok/s | slowest stream during | stall ms | ITL p50 ms before | ITL p50 ms during | "
        "ITL p99 ms during | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in clean:
        if row.get("agents"):
            continue  # the agent loops have their own table
        agg, reqs, can = row["aggregate"], row["requests"], row["canaries"]
        ttfts = [r["ttft_s"] for r in reqs if r["ttft_s"] is not None]
        cached = sum(r["cached_tokens"] or 0 for r in reqs)
        during = can["during"] if can else {}
        before = can["before"] if can else {}
        lines.append("| %s | %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
            row["setting"], row["case"], agg["requests"], agg["uncached_tokens"], cached,
            "-" if not ttfts else "%.2f..%.2f" % (min(ttfts), max(ttfts)),
            _fmt(agg["prefill_tokens_per_s"], 0), _fmt(before.get("tokens_per_s_p50")),
            _fmt(during.get("tokens_per_s_p50")), _fmt(during.get("tokens_per_s_min")),
            _fmt(during.get("stall_ms"), 0), _fmt(before.get("itl_p50_ms")), _fmt(during.get("itl_p50_ms")),
            _fmt(during.get("itl_p99_ms")), agg["errors"]))
    loops = [row for row in clean if row.get("agents")]
    if loops:
        lines += ["", "## Agent loops (every agent sends its next turn as soon as its last one ended)", "",
                  "| setting | case | seconds | turns | output tok/s in all | per stream while decoding (median, slowest "
                  "turn) | ITL ms p50 | p90 | p99 | max | TTFT s p50 / p90 / p99 | max | uncached tokens per turn | "
                  "decoding share | errors |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in loops:
            g = row["agents"]
            lines.append("| %s | %s | %.0f | %d | %s | %s, %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
                row["setting"], row["case"], g["seconds"], g["turns"], _fmt(g["output_tokens_per_s"]),
                _fmt(g["stream_tokens_per_s_p50"]), _fmt(g["stream_tokens_per_s_min"]), _fmt(g["itl_p50_ms"]),
                _fmt(g["itl_p90_ms"]), _fmt(g["itl_p99_ms"]), _fmt(g["itl_max_ms"], 0),
                "%s / %s / %s" % (_fmt(g["ttft_p50_s"], 2), _fmt(g.get("ttft_p90_s"), 2), _fmt(g.get("ttft_p99_s"), 2)),
                _fmt(g["ttft_max_s"], 2), _fmt(g["uncached_tokens_per_turn"], 0), _fmt(g["decoding_share"], 2),
                row["aggregate"]["errors"]))
    marginals = depth_marginals(clean)
    if marginals:
        lines += ["", "## Rate by context depth (single cold requests of growing length)", "",
                  "| setting | from tokens | to tokens | seconds | tokens per second in this range |", "|---|---|---|---|---|"]
        lines += ["| %s | %d | %d | %.2f | %s |" % (m["setting"], m["from"], m["to"], m["seconds"],
                                                  _fmt(m["tokens_per_s"], 0)) for m in marginals]
    (out_dir / "prefill_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def depth_marginals(rows: list[dict]) -> list[dict]:
    """From the single cold requests of one setting, sorted by size: the rate of each added range.

    A request of ``b`` tokens does the work of one of ``a < b`` tokens and then the range ``a..b`` at
    that depth, so ``(b - a) / (ttft_b - ttft_a)`` is the prefill rate between those depths. When the
    same size ran several times the last run counts (the first one compiles graphs).
    """
    by_setting: dict[str, dict[int, float]] = {}
    for row in rows:
        reqs = row["requests"]
        if not row["case"].startswith("cold ") or len(reqs) != 1 or reqs[0]["ttft_s"] is None:
            continue
        by_setting.setdefault(row["setting"], {})[int(reqs[0]["uncached_tokens"])] = float(reqs[0]["ttft_s"])
    out = []
    for setting, points in by_setting.items():
        previous = (0, 0.0)
        for tokens in sorted(points):
            seconds = points[tokens] - previous[1]
            out.append({"setting": setting, "from": previous[0], "to": tokens, "seconds": seconds,
                        "tokens_per_s": (tokens - previous[0]) / seconds if seconds > 0 else None})
            previous = (tokens, points[tokens])
    return out


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _json_object(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("a JSON object is needed, got %r" % (value, ))
    return value


def _pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for part in text.split(","):
        if not part.strip():
            continue
        left, _, right = part.partition(":")
        pairs.append((int(left), 0 if right.strip().lower() in ("", "off") else int(right)))
    return pairs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.prefill_sweep",
                                description="Prefill rate, time to first token and the effect on decoding streams.")
    p.add_argument("--url", required=True, help="base URL of the running server")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--plan", default="baseline", help="comma-separated plans: " + ", ".join(PLANS))
    p.add_argument("--knobs-file", default="", help="the server's VLLM_HPU_SCHED_KNOBS_FILE (sweep and pacing plans)")
    p.add_argument("--knobs", type=_json_object, default=None,
                   help="a JSON object of knobs written to --knobs-file before the first case and kept under "
                   "the knobs of the sweep and pacing plans, for example '{\"prompt_graph_max_tokens\": 140000}'")
    p.add_argument("--label", default="server", help="the name of this run's setting in the report")
    p.add_argument("--cold-tokens", type=_int_list, default=[32000, 120000], help="baseline: cold prompt sizes")
    p.add_argument("--warm-tokens", type=_int_list, default=[64, 512, 2048],
                   help="baseline: new tokens appended to each cold prompt")
    p.add_argument("--concurrent", type=_int_list, default=[4, 8], help="baseline: cold requests sent together")
    p.add_argument("--concurrent-tokens", type=int, default=32000)
    p.add_argument("--turn2-tokens", type=int, default=32000, help="baseline: prompt of the two-turn case (0 = skip)")
    p.add_argument("--turn2-generate", type=int, default=600, help="tokens generated in turn 1 and sent back")
    p.add_argument("--shared-prefix", type=int, default=16384, help="baseline: shared prefix tokens (0 = skip)")
    p.add_argument("--shared-tokens", type=int, default=32000, help="prompt size of the shared-prefix requests")
    p.add_argument("--shared-together", type=int, default=4)
    p.add_argument("--ladder", type=_int_list, default=[8000, 16000, 32000, 64000, 120000])
    p.add_argument("--settings", type=_pairs, default=_pairs(DEFAULT_SETTINGS),
                   help="sweep: budget:cap pairs, cap 0 or off = no per-request cap")
    p.add_argument("--pacing", type=_pairs, default=_pairs(DEFAULT_PACING),
                   help="pacing: share_percent:slice_tokens pairs, share 100 = pacing off")
    p.add_argument("--sweep-tokens", type=int, default=32000, help="sweep and pacing: cold prompt size")
    p.add_argument("--sweep-concurrent", type=_int_list, default=[4])
    p.add_argument("--sweep-concurrent-tokens", type=int, default=0,
                   help="sweep: prompt size of the concurrent requests (0 = --sweep-tokens)")
    p.add_argument("--agents", type=int, default=8, help="agents: conversations in the loop")
    p.add_argument("--agent-context", type=int, default=32768, help="agents: prompt tokens of every first turn")
    p.add_argument("--agent-contexts", type=_int_list, default=[],
                   help="agents: first-turn prompt tokens per agent, cycled (e.g. 16384,49152,81920,120000); "
                   "replaces --agent-context")
    p.add_argument("--agent-warmup-seconds", type=float, default=0.0,
                   help="agents: one loop of this many seconds before the measured loops (row '<setting> warm-up')")
    p.add_argument("--agent-new-tokens", type=int, default=1024, help="agents: new prompt tokens of every later turn")
    p.add_argument("--agent-output", type=int, default=256, help="agents: tokens generated per turn")
    p.add_argument("--agent-seconds", type=float, default=60.0, help="agents: wall time of one loop")
    p.add_argument("--agent-max-context", type=int, default=131072, help="agents: a conversation stops before this")
    p.add_argument("--agent-pacing", type=_pairs, default=[],
                   help="agents: one loop per share_percent:slice_tokens pair (share 100 = pacing off); "
                   "empty = one loop at the server's settings")
    p.add_argument("--capture-cached", type=int, default=32768,
                   help="capture: cached prefix of the traced prompt forward (0 = no such capture)")
    p.add_argument("--capture-new", type=int, default=1024, help="capture: new tokens of the traced prompt forward")
    p.add_argument("--capture-cold", type=int, default=0,
                   help="capture: a cold prompt of this many tokens traced in its middle (0 = none)")
    p.add_argument("--capture-after", type=float, default=20.0, help="capture: seconds into the cold prompt")
    p.add_argument("--capture-seconds", type=float, default=2.5, help="capture: the cold prompt's window")
    p.add_argument("--profile-timeout", type=float, default=1800.0, help="capture: limit of one profile call")
    p.add_argument("--repeat", type=int, default=1,
                   help="cold passes per size; with 2 the first pass compiles the graphs and the second measures")
    p.add_argument("--canaries", type=int, default=2, help="streams already decoding during each case (0 = none)")
    p.add_argument("--canary-prompt-tokens", type=int, default=64)
    p.add_argument("--canary-max-tokens", type=int, default=60000)
    p.add_argument("--canary-warm-tokens", type=int, default=8, help="tokens a canary must have produced first")
    p.add_argument("--settle-seconds", type=float, default=3.0, help="canary-only time before and after a case")
    p.add_argument("--case-max-tokens", type=int, default=1,
                   help="tokens generated per case request (greedy); more than 1 gives an output to compare")
    p.add_argument("--warm-salted", action="store_true",
                   help="send every warm turn a second time with a cache_salt (no prefix hit) and compare the greedy "
                   "tokens: a prefix hit that read a wrong recurrent state shows as DIFF")
    p.add_argument("--return-ids", action="store_true",
                   help="ask every case for its token ids and keep the first %d in each row" % 64)
    p.add_argument("--request-timeout", type=float, default=3600.0)
    p.add_argument("--synthetic-ids", action="store_true", help="pseudo-random token ids, no /tokenize call")
    p.add_argument("--corpus-tokens", type=int, default=400000)
    p.add_argument("--nonce", default=None, help="salt of this run's prompts (default: the start time)")
    args = p.parse_args(argv)
    args.plans = [name.strip() for name in args.plan.split(",") if name.strip()]
    unknown = [name for name in args.plans if name not in PLANS]
    if unknown:
        p.error("unknown plan(s) %s; known: %s" % (unknown, ", ".join(PLANS)))
    if args.knobs is not None and not args.knobs_file:
        p.error("--knobs needs --knobs-file (the server's VLLM_HPU_SCHED_KNOBS_FILE)")
    if args.agents < 1 or args.agent_output < 2:
        p.error("--agents needs at least one agent and --agent-output at least 2 tokens")
    return args


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    nonce = args.nonce or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.synthetic_ids:
        ids = synthetic_ids(args.corpus_tokens)
    else:
        ids = tokenize_corpus(args.url, args.model, corpus_texts(REPO_ROOT), args.corpus_tokens)
        if len(ids) < 4096:
            print("prefill_sweep: /tokenize returned %d ids; using synthetic ids" % len(ids), flush=True)
            ids = synthetic_ids(args.corpus_tokens)
    corpus = Corpus(ids, nonce)
    meta = {"url": args.url, "model": args.model, "plans": args.plans, "canaries": args.canaries, "nonce": nonce,
            "corpus_tokens": len(ids), "synthetic_ids": bool(args.synthetic_ids), "started": time.time(),
            "args": {k: v for k, v in vars(args).items() if k != "plans"}}
    print("prefill_sweep: %d corpus token ids, plans %s, nonce %s" % (len(ids), args.plans, nonce), flush=True)
    async with Runner(args, corpus, out_dir) as runner:
        try:
            if args.knobs is not None:
                await runner.set_knobs("server", {})  # the run's own knobs, before its first case
            for name in args.plans:
                await getattr(runner, "plan_" + name)()
        finally:
            meta["finished"] = time.time()
            write_report(out_dir, runner.rows, meta)
        errors = sum(row["aggregate"]["errors"] for row in runner.rows)
    print("prefill_sweep: wrote %s (%d cases, %d request errors)" % (out_dir / "prefill_sweep.md", len(runner.rows),
                                                                     errors), flush=True)
    return 0 if errors == 0 else 1


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if threading.current_thread() is threading.main_thread():
        # A job that runs out of time stops this tool with SIGINT or SIGTERM: both end the run the way
        # Ctrl-C does, so the report of the cases that finished is written (run's finally). SIGINT is set
        # too because a shell that started its parent in the background leaves it ignored.
        signal.signal(signal.SIGTERM, _interrupt)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
