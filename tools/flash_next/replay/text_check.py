#!/usr/bin/env python3
"""Long greedy outputs from a running server, saved whole, for comparing builds (native-replay, lean4).

The bridge's output-buffer reuse (``PT_HPU_HPUGRAPH_PREPARED_REUSE``) changes buffer lifetimes, so it is qualified
on long outputs: several prompts at 1 stream with 600 generated tokens each (``ignore_eos``, greedy), whose texts
must equal the arm without reuse token for token, and a 32-stream round whose texts must be coherent (no NaN
garbage). Only the standard library; talks to the OpenAI-compatible endpoint of ``vllm serve``.

    python3 text_check.py --out texts_1.json --streams 1 --prompts 6 --tokens 600
    python3 text_check.py --out texts_32.json --streams 32 --tokens 600
    python3 text_check.py --compare a/texts_1.json b/texts_1.json

The comparison prints per prompt "same" or the first differing character, and a garbage check of each text (the
share of the replacement character and of the longest run of one repeated character).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.request

PROMPTS = [
    "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.",
    "Explain step by step how a CPU executes a program, from fetching instructions to writing back results.",
    "Write a Python module that implements a priority queue with a binary heap, with docstrings and tests.",
    "Describe the history of the printing press and its effects on European society, in several sections.",
    "Solve this carefully and show all steps: a train leaves at 9:00 at 80 km/h, another at 10:30 at 120 km/h "
    "on the same track; when and where does the second catch up? Then generalise the method.",
    "List and explain twenty practical tips for writing clear technical documentation.",
    "Write a dialogue between two scientists arguing about whether time travel is possible.",
    "Translate the idea of compound interest into a short lesson for children, then a lesson for adults.",
]


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def model_name(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as resp:
        return json.loads(resp.read().decode())["data"][0]["id"]


def one(base: str, model: str, prompt: str, tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0.0, "ignore_eos": True, "min_tokens": tokens}
    t0 = time.time()
    out = _post(base + "/v1/chat/completions", body, timeout)
    choice = out["choices"][0]
    text = choice["message"]["content"] or ""
    return {"prompt": prompt, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "completion_tokens": out.get("usage", {}).get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"), "seconds": round(time.time() - t0, 2)}


def garbage(text: str) -> dict:
    """Signs of NaN-driven output: replacement characters and long runs of one character."""
    run = best = 0
    prev = None
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return {"replacement_share": round(text.count("�") / max(1, len(text)), 4), "longest_run": best}


def compare(a_path: str, b_path: str) -> int:
    a = json.load(open(a_path))["results"]
    b = json.load(open(b_path))["results"]
    differ = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x["text"] == y["text"]:
            verdict = "same"
        else:
            differ += 1
            k = next((j for j, (c, d) in enumerate(zip(x["text"], y["text"])) if c != d),
                     min(len(x["text"]), len(y["text"])))
            verdict = "DIFFERS at char %d of %d/%d: %r | %r" % (k, len(x["text"]), len(y["text"]),
                                                               x["text"][max(0, k - 20):k + 40],
                                                               y["text"][max(0, k - 20):k + 40])
        print("prompt %d: %s; garbage a %s b %s" % (i, verdict, garbage(x["text"]), garbage(y["text"])))
    print("text_check compare: %d of %d prompts differ" % (differ, min(len(a), len(b))))
    return 1 if differ else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("FARM_PORT_BASE", "8100")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--prompts", type=int, default=None, help="default: 6 at 1 stream, the stream count otherwise")
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    base = "http://%s:%d" % (args.host, args.port)
    model = model_name(base)
    n = args.prompts or (6 if args.streams == 1 else args.streams)
    prompts = [PROMPTS[i % len(PROMPTS)] + ("" if i < len(PROMPTS) else " (variant %d)" % (i // len(PROMPTS)))
               for i in range(n)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.streams)) as pool:
        results = list(pool.map(lambda p: one(base, model, p, args.tokens, args.timeout), prompts))
    report = {"model": model, "streams": args.streams, "tokens": args.tokens, "seconds": round(time.time() - t0, 1),
              "results": results}
    out = args.out or os.path.join(os.environ.get("FARM_OUT", "."), "texts_%d.json" % args.streams)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    worst = max((garbage(r["text"])["longest_run"] for r in results), default=0)
    print("text_check: %d prompts at %d streams, %d tokens each, %.0f s; tokens %s; longest run of one character %d;"
          " %s" % (n, args.streams, args.tokens, report["seconds"], sorted({r["completion_tokens"] for r in results}),
                   worst, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
