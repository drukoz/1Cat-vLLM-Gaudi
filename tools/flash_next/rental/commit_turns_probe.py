#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Requests across a recurrent block boundary, new batch shapes after it, and the same prompt again (card job 00133).

    python tools/flash_next/rental/commit_turns_probe.py --port P --model fn --out turns.json [--block 896]

With prefix caching and speculation, a request whose decode completes a recurrent block gets align-mode state
copies into that block's slot. Card job 00133 died on the request after such a copy (the GDN conv pool written
through a slice, fixed by commit 5c770d96); a replay on stale pool storage would not die but drift. Rounds, in order:

1. ``first``: one stream, a prompt of about ``0.8 x block`` tokens, ``--decode`` tokens: its decode crosses the block.
2. ``shapes``: 2, 3, 5 and 8 streams of short prompts: new decode batch sizes, so new graphs are captured after
   the copies of round 1.
3. ``again``: round 1's prompt again, one stream: tokens equal to round 1 (same server, same batch shape) unless a
   state went wrong after the copies.
4. ``eight``: 8 streams of distinct prompts of about ``0.8 x block`` tokens with ``--decode`` tokens (8 crossings at
   once), then ``eight_again``: the same 8 prompts.
5. ``next_turn``: round 1's prompt, its answer and a new sentence: a prefix-cache hit on round 1's blocks if the
   engine keeps them (``cached_tokens`` from the usage, ``--enable-prompt-tokens-details``).

Greedy, ``ignore_eos``, never ``min_tokens``. Per round: token ids and text of every stream, the speculative
counters around it (acceptance per draft position, ``spec_probe``), wall time. Comparisons: ``again`` against
``first``, ``eight_again`` against ``eight`` (exact token equality and the first difference). A server without
speculation gives the same rounds without acceptance, for a comparison across servers. ``--out`` is rewritten after
every round. Exit 1 when a round failed (a dead server: the remaining rounds are not sent).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec_probe  # noqa: E402

WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river signal mapmaker bridge clock").split()


def long_prompt(i: int, tokens: int) -> str:
    """About ``tokens`` tokens (about 1.3 tokens per word), a distinct first sentence per index."""
    words, count, k = [], 0.0, 0
    while count < tokens:
        words.append(WORDS[(i * 5 + k * 7) % len(WORDS)])
        count += 1.3
        k += 1
    return "Notes %d: " % i + " ".join(words) + "\n\nWrite a long story about the notes above. Begin:"


def short_prompt(i: int) -> str:
    return "Prompt %d. Write a story about a %s and a %s. Begin:" % (i, WORDS[i % len(WORDS)],
                                                                    WORDS[(3 * i + 1) % len(WORDS)])


def complete(host: str, port: int, model: str, prompt, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True}
    record = {"ok": False}
    try:
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        connection.close()
        if response.status != 200:
            record["error"] = "HTTP %d: %s" % (response.status, data[:300].decode("utf-8", "replace"))
            return record
        reply = json.loads(data)
        choice = (reply.get("choices") or [{}])[0]
        usage = reply.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        record.update(ok=True, text=choice.get("text", ""), token_ids=choice.get("token_ids") or [],
                      prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                      cached_tokens=details.get("cached_tokens"))
    except Exception as error:  # noqa: BLE001 - a probe records every failure
        record["error"] = repr(error)
    return record


def run_round(args, base: str, name: str, prompts: list, max_tokens: int) -> dict:
    before = _counters(base)
    out = [None] * len(prompts)

    def one(i):
        out[i] = complete(args.host, args.port, args.model, prompts[i], max_tokens, args.timeout)

    threads = [threading.Thread(target=one, args=(i, ), daemon=True) for i in range(len(prompts))]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(args.timeout + 30)
    record = {"round": name, "streams": len(prompts), "max_tokens": max_tokens,
              "wall_s": round(time.perf_counter() - started, 2), "results": out,
              "ok": all(r and r.get("ok") for r in out),
              "spec": spec_probe.spec_summary(spec_probe.counters_delta(before, _counters(base)))}
    return record


def _counters(base: str):
    try:
        return spec_probe.read_spec_counters(base)
    except Exception:  # noqa: BLE001 - a server without speculation, or a dead one
        return None


def compare(a: dict, b: dict) -> list:
    """Per stream: (equal, first difference index or None) of two rounds' token ids."""
    rows = []
    for x, y in zip(a["results"], b["results"]):
        ids_x, ids_y = (x or {}).get("token_ids") or [], (y or {}).get("token_ids") or []
        diff = next((i for i, (p, q) in enumerate(zip(ids_x, ids_y)) if p != q), None)
        if diff is None and len(ids_x) != len(ids_y):
            diff = min(len(ids_x), len(ids_y))
        rows.append({"equal": diff is None and bool(ids_x), "first_diff": diff})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="fn")
    parser.add_argument("--block", type=int, default=896, help="the recurrent block (896 at TP4, 512 at TP8)")
    parser.add_argument("--decode", type=int, default=400, help="output tokens of the crossing rounds")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="turns.json")
    args = parser.parse_args(argv)
    base = "http://%s:%d" % (args.host, args.port)
    prompt_tokens = int(0.8 * args.block)
    first_prompt = long_prompt(0, prompt_tokens)
    result = {"block": args.block, "prompt_tokens_target": prompt_tokens, "rounds": [], "comparisons": {}}

    def save():
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)

    def report(record):
        spec = record.get("spec") or {}
        rates = ",".join("%.2f" % r for r in spec.get("position_rates") or [] if r is not None)
        lengths = [r.get("prompt_tokens") for r in record["results"] if r]
        cached = [r.get("cached_tokens") for r in record["results"] if r]
        errors = [r.get("error") for r in record["results"] if r and not r.get("ok")]
        print("ROUND %-12s streams=%d ok=%s wall=%.1fs prompt_tokens=%s cached=%s acceptance=%s per-position=%s%s" % (
            record["round"], record["streams"], record["ok"], record["wall_s"], lengths[:3], cached[:3],
            ("%.2f" % spec["acceptance_length"]) if spec.get("acceptance_length") else "-", rates or "-",
            (" errors=%s" % errors[:2]) if errors else ""), flush=True)
        for i, r in enumerate(record["results"][:2]):
            if r and r.get("ok"):
                print("TEXT  %-12s stream %d: %r" % (record["round"], i, r["text"][-140:]), flush=True)

    plan = [("first", [first_prompt], args.decode)]
    plan += [("shapes_%d" % n, [short_prompt(100 * n + i) for i in range(n)], 64) for n in (2, 3, 5, 8)]
    plan += [("again", [first_prompt], args.decode)]
    eight = [long_prompt(10 + i, prompt_tokens) for i in range(8)]
    plan += [("eight", eight, args.decode), ("eight_again", eight, args.decode)]
    by_name = {}
    for name, prompts, max_tokens in plan:
        record = run_round(args, base, name, prompts, max_tokens)
        result["rounds"].append(record)
        by_name[name] = record
        report(record)
        save()
        if not record["ok"]:
            print("PROBE: round %s failed; the server is probably dead" % name, flush=True)
            return 1
    result["comparisons"]["again_vs_first"] = compare(by_name["first"], by_name["again"])
    result["comparisons"]["eight_again_vs_eight"] = compare(by_name["eight"], by_name["eight_again"])
    answer = by_name["first"]["results"][0]["text"]
    record = run_round(args, base, "next_turn", [first_prompt + answer + "\n\nNow continue the story with a twist."],
                       64)
    result["rounds"].append(record)
    report(record)
    for key, rows in result["comparisons"].items():
        print("COMPARE %-22s equal %d of %d, first differences %s" % (
            key, sum(r["equal"] for r in rows), len(rows), [r["first_diff"] for r in rows]), flush=True)
    save()
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
