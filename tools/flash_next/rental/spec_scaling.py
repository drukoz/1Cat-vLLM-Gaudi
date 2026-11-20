#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode scaling of a speculative server: step time, tokens per step and acceptance per draft position.

    python tools/flash_next/rental/spec_scaling.py --port P --model fn --out scaling.json --levels 1,8,32 \
        [--tokens 256] [--sampled-levels 1,32] [--ctx N:TOKENS]

Per stream count N: a warm round (compiles the graphs of that batch), then the measured round(s); every stream
generates exactly ``--tokens`` tokens (``max_tokens`` with ``ignore_eos``; never ``min_tokens``). Around every
round the server's speculative counters are read (``spec_probe``, the ``/metrics`` page), so a round reports

* ms per step: the median gap between a stream's chunks (one chunk per engine step);
* tokens per step: tokens per chunk, and vLLM's acceptance length ``1 + accepted / drafts`` from the counters;
* acceptance per draft position;
* effective ms per token (the stream's decode span over its tokens, compile pauses included) and the steady
  figure (ms per step over tokens per step).

``--sampled-levels`` adds rounds that send no temperature: what a client that sets none gets, the checkpoint's
generation_config (temperature 1.0, top_k 20, top_p 0.95) and the general sampler; on a speculative server that
needs VLLM_HPU_SPEC_SAMPLED_VERIFY=1 (without it the server answers HTTP 400). ``--ctx N:TOKENS`` adds a greedy
round of N streams with prompts of about TOKENS tokens. The start of the text of streams 0, 1, N/2 and N-1 is
kept (``texts``) and printed (``TEXT`` lines): a round whose later streams decode garbage has normal timings.
A server without speculation gives the same lines without the acceptance.

Standard library only. ``scaling.json`` is rewritten after every round.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_probe  # noqa: E402 - the sibling module, also when the script is run by path

WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river signal mapmaker bridge clock").split()
TEXT_CHARS = 400


def prompt_for(i: int, ctx_tokens: int = 0) -> str:
    """The coordinator's probe prompts (``coord/decode_scaling.py``): the same stream index gives the same prompt."""
    head = "Stream %d. Write a long, detailed story about a %s named %s who finds a %s. Begin:" % (
        i, WORDS[i % len(WORDS)], WORDS[(i * 7 + 3) % len(WORDS)].title(), WORDS[(i * 3 + 5) % len(WORDS)])
    if ctx_tokens <= 0:
        return head
    filler, count, k = [], 0.0, 0
    while count < ctx_tokens:  # about 1.3 tokens per word; a distinct first sentence per stream: no shared prefix
        filler.append(WORDS[(i + k * 11) % len(WORDS)])
        count += 1.3
        k += 1
    return "Stream %d notes: " % i + " ".join(filler) + "\n\n" + head


def one_stream(host, port, model, prompt, ntok, out, idx, barrier, timeout_s, sampled):
    request = {"model": model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    if not sampled:  # a sampled round sends no temperature: the server's default sampling
        request["temperature"] = 0
    record = {"status": None, "t0": None, "times": [], "usage": None, "err": None, "text": ""}
    try:
        barrier.wait()
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        record["t0"] = time.perf_counter()
        connection.request("POST", "/v1/completions", body=json.dumps(request),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        record["status"] = response.status
        if response.status != 200:
            record["err"] = "HTTP %d: %s" % (response.status, response.read()[:300].decode("utf-8", "replace"))
        else:
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if event.get("usage"):
                    record["usage"] = event["usage"]
                choices = event.get("choices") or []
                if choices and choices[0].get("text"):
                    record["times"].append(time.perf_counter())
                    if len(record["text"]) < TEXT_CHARS:
                        record["text"] += choices[0]["text"]
        connection.close()
    except Exception as error:  # noqa: BLE001 - a probe records every failure
        record["err"] = repr(error)
    out[idx] = record


def run_round(args, n: int, ntok: int, ctx_tokens: int = 0, sampled: bool = False) -> dict:
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream, args=(args.host, args.port, args.model, prompt_for(i, ctx_tokens),
                                                         ntok, out, i, barrier, args.timeout, sampled), daemon=True)
               for i in range(n)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(args.timeout + 60)
    result = {"streams": n, "tokens": ntok, "ctx_tokens": ctx_tokens, "sampled": bool(sampled), "min_tokens": False,
              "wall_s": time.perf_counter() - started}
    per = []
    for record in out:
        if not record or record["err"] or record["status"] != 200 or len(record["times"]) < 2:
            continue
        tokens = (record["usage"] or {}).get("completion_tokens") or len(record["times"])
        gaps = [b - a for a, b in zip(record["times"], record["times"][1:])]
        per.append({"gap_ms": 1000.0 * statistics.median(gaps),
                    "gap_p90_ms": 1000.0 * sorted(gaps)[int(0.9 * (len(gaps) - 1))],
                    "pause_s": sum(g for g in gaps if g > 0.5), "tokens_per_chunk": tokens / len(record["times"]),
                    "ttft_s": record["times"][0] - record["t0"],
                    "ms_per_token": 1000.0 * (record["times"][-1] - record["times"][0]) / max(1, tokens - 1),
                    "tokens": tokens, "prompt_tokens": (record["usage"] or {}).get("prompt_tokens") or 0,
                    "first": record["times"][0], "last": record["times"][-1]})
    result["ok"] = len(per)
    result["errors"] = n - len(per)
    if per:
        result["ms_per_step_p50"] = statistics.median(p["gap_ms"] for p in per)
        result["gap_p90_ms_p50"] = statistics.median(p["gap_p90_ms"] for p in per)
        result["tokens_per_chunk_p50"] = statistics.median(p["tokens_per_chunk"] for p in per)
        result["steady_ms_per_token_p50"] = statistics.median(p["gap_ms"] / max(p["tokens_per_chunk"], 1e-9)
                                                              for p in per)
        result["ms_per_token_p50"] = statistics.median(p["ms_per_token"] for p in per)
        result["ms_per_token_max"] = max(p["ms_per_token"] for p in per)
        result["tok_s_per_stream_p50"] = 1000.0 / result["ms_per_token_p50"]
        result["pause_s_max"] = max(p["pause_s"] for p in per)
        result["ttft_p50_s"] = statistics.median(p["ttft_s"] for p in per)
        result["prompt_tokens_p50"] = statistics.median(p["prompt_tokens"] for p in per)
        span = max(p["last"] for p in per) - min(p["first"] for p in per)
        result["aggregate_tok_s"] = sum(p["tokens"] for p in per) / span if span > 0 else None
    picks = sorted({0, 1, n // 2, n - 1} & set(range(n)))
    result["texts"] = {str(i): (out[i] or {}).get("text", "") for i in picks}
    errors = [record["err"] for record in out if record and record["err"]][:3]
    if errors:
        result["error_samples"] = errors
    return result


def counters(base: str):
    try:
        return spec_probe.read_spec_counters(base)
    except Exception as error:  # noqa: BLE001 - a record
        print("metrics read failed: %r" % (error, ), flush=True)
        return None


def measured(args, base: str, label: str, n: int, ntok: int, ctx_tokens: int = 0, sampled: bool = False) -> dict:
    before = counters(base)
    record = run_round(args, n, ntok, ctx_tokens, sampled)
    record["spec"] = spec_probe.spec_summary(spec_probe.counters_delta(before, counters(base)))
    record["label"] = label
    return record


def report(record: dict) -> None:
    n, label = record["streams"], record["label"]
    if not record.get("ok"):
        print("%-5s N=%-3d FAILED: %s" % (label, n, record.get("error_samples")), flush=True)
        return
    spec = record.get("spec") or {}
    length = spec.get("acceptance_length")
    rates = ",".join("%.2f" % rate for rate in spec.get("position_rates") or [] if rate is not None)
    print("%-5s N=%-3d ok=%d err=%d  ms/step=%.2f  tok/step(chunk)=%.2f  acceptance length=%s (per position %s)  "
          "steady ms/tok=%.2f  effective ms/tok p50=%.2f  tok/s/stream=%.1f  aggregate=%.1f  ttft p50=%.2fs  "
          "gap p90=%.1f ms  pauses>0.5s max=%.1fs  prompt~%d tok" % (
              label, n, record["ok"], record["errors"], record["ms_per_step_p50"], record["tokens_per_chunk_p50"],
              ("%.2f" % length) if length else "-", rates or "-", record["steady_ms_per_token_p50"],
              record["ms_per_token_p50"], record["tok_s_per_stream_p50"], record["aggregate_tok_s"] or 0,
              record["ttft_p50_s"], record["gap_p90_ms_p50"], record["pause_s_max"], record["prompt_tokens_p50"]),
          flush=True)
    for index, text in record["texts"].items():
        print("TEXT  N=%-3d %s stream %-2s: %r" % (n, label, index, text[:110]), flush=True)


def level_list(text: str) -> list[int]:
    return [int(word) for word in text.split(",") if word]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="fn")
    parser.add_argument("--levels", default="1,8,32")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warm-tokens", type=int, default=48)
    parser.add_argument("--repeat-top", type=int, default=2, help="measured rounds at the largest level")
    parser.add_argument("--sampled-levels", default="", help="stream counts of the rounds with the server's sampling")
    parser.add_argument("--ctx", default="", metavar="N:TOKENS", help="one more greedy round with long prompts")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="scaling.json")
    args = parser.parse_args(argv)
    base = "http://%s:%d" % (args.host, args.port)
    results = {"levels": [], "sampled": [], "ctx": []}

    def save():
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)

    def rounds(key: str, label: str, n: int, repeats: int, ctx_tokens: int = 0, sampled: bool = False) -> bool:
        warm = measured(args, base, "warm", n, args.warm_tokens, ctx_tokens, sampled)
        length = (warm.get("spec") or {}).get("acceptance_length")
        print("warm  N=%-3d %s ok=%d err=%d wall=%.1fs acceptance=%s" % (
            n, label, warm["ok"], warm["errors"], warm["wall_s"], ("%.2f" % length) if length else "-"), flush=True)
        if not warm["ok"]:
            results[key].append(dict(warm, label=label, warm_failed=True))
            save()
            print("%-5s N=%-3d the warm round failed: %s" % (label, n, warm.get("error_samples")), flush=True)
            return False
        for rep in range(repeats):
            record = measured(args, base, label, n, args.tokens, ctx_tokens, sampled)
            record["rep"] = rep
            record["warm"] = {key2: warm.get(key2) for key2 in ("ok", "errors", "wall_s")}
            record["warm_spec"] = warm.get("spec")
            results[key].append(record)
            save()
            report(record)
            if not record.get("ok"):
                return False
        return True

    levels = level_list(args.levels)
    for n in levels:
        if not rounds("levels", "MEAS", n, args.repeat_top if n == levels[-1] else 1):
            break
    for n in level_list(args.sampled_levels):
        if not rounds("sampled", "SAMPL", n, 1, sampled=True):
            break
    if args.ctx:
        n, ctx_tokens = (int(word) for word in args.ctx.split(":"))
        rounds("ctx", "CTX", n, 1, ctx_tokens=ctx_tokens)
    save()
    # Exit 1 when no round measured anything (the server died at its first prompt, or never answered): a job
    # that runs arms (mtp_arms.sh) then knows that the arm gave no number.
    measured_any = any(record.get("ok") and not record.get("warm_failed")
                       for key in ("levels", "sampled", "ctx") for record in results[key])
    return 0 if measured_any else 1


if __name__ == "__main__":
    sys.exit(main())
