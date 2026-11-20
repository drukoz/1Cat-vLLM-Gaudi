#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engine-step card probe: decode at 1 and 32 streams, short and long (shared-prefix) contexts.

Stdlib only; prompts are token-id lists, so the long rounds share one exact prefix of --prefix tokens that the
server prefills once (a seed request) and then serves from the prefix cache: 32 streams decode at about
--prefix tokens of context each without 32 long prefills. Every stream generates exactly N tokens
(min_tokens = max_tokens, ignore_eos), temperature 0.

Per measured round: per stream the median gap between streamed chunks (steady ms per token, compile pauses
excluded), the decode span / (tokens - 1) (pauses included), the sum of gaps over 0.5 s (compile pauses), TTFT;
the round's medians and aggregate tokens per second; every stream's tokens (to compare two servers).
"""
import argparse
import http.client
import json
import random
import statistics
import sys
import threading
import time


def one_stream(host, port, model, prompt_ids, ntok, out, idx, barrier, timeout_s):
    body = json.dumps({"model": model, "prompt": prompt_ids, "max_tokens": ntok, "min_tokens": ntok,
                       "ignore_eos": True, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    rec = {"status": None, "t0": None, "times": [], "text": [], "usage": None, "err": None}
    try:
        barrier.wait()
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        rec["t0"] = time.perf_counter()
        conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["status"] = resp.status
        if resp.status != 200:
            rec["err"] = resp.read()[:500].decode("utf-8", "replace")
        else:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
                ch = obj.get("choices") or []
                if ch and ch[0].get("text"):
                    rec["times"].append(time.perf_counter())
                    rec["text"].append(ch[0]["text"])
        conn.close()
    except Exception as e:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(e)
    out[idx] = rec


def run_round(args, label, prompts, ntok):
    n = len(prompts)
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream, args=(args.host, args.port, args.model, prompts[i], ntok, out, i,
                                                         barrier, args.timeout), daemon=True) for i in range(n)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.timeout + 60)
    wall = time.perf_counter() - t_start
    per, errors, texts = [], 0, []
    for r in out:
        if not r or r["err"] or r["status"] != 200 or len(r["times"]) < 2:
            errors += 1
            texts.append(None)
            continue
        texts.append("".join(r["text"]))
        comp = (r["usage"] or {}).get("completion_tokens") or len(r["times"])
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:])]
        tpc = comp / len(r["times"])
        per.append({"steady_ms": 1000.0 * statistics.median(gaps) / max(tpc, 1e-9),
                    "gap_p90_ms": 1000.0 * sorted(gaps)[int(0.9 * (len(gaps) - 1))],
                    "pause_s": sum(g for g in gaps if g > 0.5), "ttft_s": r["times"][0] - r["t0"],
                    "span_ms": 1000.0 * (r["times"][-1] - r["times"][0]) / max(1, comp - 1), "tokens": comp,
                    "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "first": r["times"][0],
                    "last": r["times"][-1], "gaps_ms": [round(1000.0 * g, 2) for g in gaps]})
    res = {"label": label, "streams": n, "tokens": ntok, "ok": len(per), "errors": errors, "wall_s": wall,
           "texts": texts}
    if per:
        res["steady_ms_p50"] = statistics.median(p["steady_ms"] for p in per)
        res["steady_ms_min"] = min(p["steady_ms"] for p in per)
        res["steady_ms_max"] = max(p["steady_ms"] for p in per)
        res["gap_p90_ms_p50"] = statistics.median(p["gap_p90_ms"] for p in per)
        res["span_ms_p50"] = statistics.median(p["span_ms"] for p in per)
        res["pause_s_max"] = max(p["pause_s"] for p in per)
        res["ttft_p50_s"] = statistics.median(p["ttft_s"] for p in per)
        res["prompt_tokens_p50"] = statistics.median(p["prompt_tokens"] or 0 for p in per)
        all_gaps = sorted(g for p in per for g in p["gaps_ms"])
        res["all_gaps_ms_p50"] = all_gaps[len(all_gaps) // 2]
        res["all_gaps_ms_p10"] = all_gaps[len(all_gaps) // 10]
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = sum(p["tokens"] for p in per) / span_all if span_all > 0 else None
        res["per_stream"] = per
    errs = [r["err"] for r in out if r and r["err"]][:3]
    if errs:
        res["error_samples"] = errs
    print("%-14s N=%-3d ok=%d err=%d wall=%6.1fs steady ms/tok p50=%s [min %s max %s] all-gaps p50=%s p10=%s "
          "span ms/tok p50=%s pauses max=%ss ttft p50=%ss prompt~%s agg=%s tok/s" % (
              label, n, res["ok"], errors, wall, _f(res.get("steady_ms_p50")), _f(res.get("steady_ms_min")),
              _f(res.get("steady_ms_max")), _f(res.get("all_gaps_ms_p50")), _f(res.get("all_gaps_ms_p10")),
              _f(res.get("span_ms_p50")), _f(res.get("pause_s_max")), _f(res.get("ttft_p50_s")),
              res.get("prompt_tokens_p50"), _f(res.get("aggregate_tok_s"))), flush=True)
    return res


def _f(x):
    return "-" if x is None else "%.2f" % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prefix", type=int, default=65536, help="shared prefix tokens of the long rounds")
    ap.add_argument("--suffix", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="probe.json")
    args = ap.parse_args()
    rng = random.Random(20261120)
    ids = lambda n: [rng.randrange(1000, 150000) for _ in range(n)]  # noqa: E731
    prefix = ids(args.prefix)
    results = {"args": vars(args), "rounds": []}

    def run(label, prompts, ntok):
        res = run_round(args, label, prompts, ntok)
        results["rounds"].append(res)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        return res

    n = args.streams
    # Short contexts (about 64 prompt tokens): warm rounds compile the decode graphs of batch 1 and n.
    run("warm-short-1", [ids(64)], 32)
    run("warm-short-%d" % n, [ids(64) for _ in range(n)], 32)
    run("short-1", [ids(64)], args.tokens)
    run("short-%d" % n, [ids(64) for _ in range(n)], args.tokens)
    # Long contexts: one seed request prefills the shared prefix, then every stream is prefix + its own suffix.
    t = time.time()
    run("seed-prefix", [prefix + ids(args.suffix)], 2)
    print("seed prefill of %d tokens: %.1f s" % (args.prefix, time.time() - t), flush=True)
    run("warm-long-%d" % n, [prefix + ids(args.suffix) for _ in range(n)], 16)
    run("long-1", [prefix + ids(args.suffix)], args.tokens)
    run("long-%d" % n, [prefix + ids(args.suffix) for _ in range(n)], args.tokens)
    run("long-%d-again" % n, [prefix + ids(args.suffix) for _ in range(n)], args.tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
