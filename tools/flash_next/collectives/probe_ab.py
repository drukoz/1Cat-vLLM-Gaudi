#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode scaling probe: N concurrent streaming completions of equal length against one vLLM server.

Copy of the coordinator's coord/decode_scaling.py (collectives track, for its A/B jobs) with one addition: every
measured round records the generated text of each stream ("texts", in stream order), so two servers that differ
only by a collective cut can be compared token for token at temperature 0.

Per level N: a warm round (short, compiles the decode graphs of batch N), then a measured round. Every stream
generates exactly --tokens tokens (ignore_eos; no min_tokens, see one_stream), temperature 0. Reports per stream the
time to first token and the decode time per token over the stream's own decode span, and for the round the
aggregate output tokens per second over the window in which all N streams were decoding.

Stdlib only. Usage:
    python3 decode_scaling.py --port 8000 --model fn --levels 1,8,16,32 --tokens 256 --out scaling.json
    python3 decode_scaling.py ... --ctx 32:4000     # one more measured round: 32 streams with ~4000-token prompts
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time

WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river signal mapmaker bridge clock").split()


def prompt_for(i, ctx_tokens=0):
    head = "Stream %d. Write a long, detailed story about a %s named %s who finds a %s. Begin:" % (
        i, WORDS[i % len(WORDS)], WORDS[(i * 7 + 3) % len(WORDS)].title(), WORDS[(i * 3 + 5) % len(WORDS)])
    if ctx_tokens <= 0:
        return head
    # Roughly 1.3 tokens per word; a distinct first sentence so no two streams share a cached prefix block.
    filler = []
    n = 0
    k = 0
    while n < ctx_tokens:
        w = WORDS[(i + k * 11) % len(WORDS)]
        filler.append(w)
        n += 1.3
        k += 1
    return "Stream %d notes: " % i + " ".join(filler) + "\n\n" + head


def one_stream(host, port, model, prompt, ntok, out, idx, barrier, timeout_s, sampled=False):
    req = {
        # No min_tokens: a request with min_tokens gets the MinTokens logits processor, which keeps it out of the
        # runner's greedy region (and so out of VLLM_HPU_VOCAB_PARALLEL_LOGITS); ignore_eos alone gives ntok tokens.
        "model": model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    if sampled:  # what a client that sets no sampling parameter gets: the checkpoint's generation_config
        del req["temperature"]  # (Flash-Next: do_sample, temperature 1.0, top_k 20, top_p 0.95)
        req["seed"] = 1000 + idx
    body = json.dumps(req)
    rec = {"status": None, "t0": None, "times": [], "ntoks": [], "usage": None, "err": None, "text": []}
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


def run_round(args, n, ntok, ctx_tokens=0, sampled=False, first=0):
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream,
                                args=(args.host, args.port, args.model, prompt_for(first + i, ctx_tokens), ntok, out,
                                      i, barrier, args.timeout, sampled), daemon=True) for i in range(n)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.timeout + 60)
    wall = time.perf_counter() - t_start
    per = []
    errors = 0
    for r in out:
        if not r or r["err"] or r["status"] != 200 or len(r["times"]) < 2:
            errors += 1
            continue
        comp = (r["usage"] or {}).get("completion_tokens") or len(r["times"])
        prompt_toks = (r["usage"] or {}).get("prompt_tokens")
        span = r["times"][-1] - r["times"][0]
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:])]
        tpc = comp / len(r["times"])  # tokens per streamed chunk (above 1 with speculation)
        gmed = statistics.median(gaps)
        per.append({"steady_ms_per_token": 1000.0 * gmed / max(tpc, 1e-9), "gap_p50_ms": 1000.0 * gmed,
                    "gap_p90_ms": 1000.0 * sorted(gaps)[int(0.9 * (len(gaps) - 1))],
                    "pause_s": sum(g for g in gaps if g > 0.5), "tokens_per_chunk": tpc,
                    "ttft_s": r["times"][0] - r["t0"], "ms_per_token": 1000.0 * span / max(1, comp - 1),
                    "completion_tokens": comp, "prompt_tokens": prompt_toks, "chunks": len(r["times"]),
                    "first": r["times"][0], "last": r["times"][-1]})
    res = {"streams": n, "tokens": ntok, "ctx_tokens": ctx_tokens, "sampled": bool(sampled), "ok": len(per),
           "errors": errors, "wall_s": wall, "texts": ["".join(r["text"]) if r else None for r in out]}
    if per:
        ms = sorted(p["ms_per_token"] for p in per)
        tt = sorted(p["ttft_s"] for p in per)
        res["steady_ms_per_token_p50"] = statistics.median(p["steady_ms_per_token"] for p in per)
        res["gap_p90_ms_p50"] = statistics.median(p["gap_p90_ms"] for p in per)
        res["pause_s_max"] = max(p["pause_s"] for p in per)
        res["tokens_per_chunk_p50"] = statistics.median(p["tokens_per_chunk"] for p in per)
        res["ms_per_token_p50"] = statistics.median(ms)
        res["ms_per_token_min"] = ms[0]
        res["ms_per_token_max"] = ms[-1]
        res["tok_s_per_stream_p50"] = 1000.0 / res["ms_per_token_p50"]
        res["ttft_p50_s"] = statistics.median(tt)
        res["ttft_max_s"] = tt[-1]
        res["prompt_tokens_p50"] = statistics.median([p["prompt_tokens"] or 0 for p in per])
        w0 = max(p["first"] for p in per)
        w1 = min(p["last"] for p in per)
        res["all_decoding_window_s"] = max(0.0, w1 - w0)
        total = sum(p["completion_tokens"] for p in per)
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = total / span_all if span_all > 0 else None
    errs = [r["err"] for r in out if r and r["err"]][:3]
    if errs:
        res["error_samples"] = errs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--levels", default="1,8,16,32")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--warm-tokens", type=int, default=48)
    ap.add_argument("--repeat-top", type=int, default=2, help="measured rounds at the largest level")
    ap.add_argument("--repeat-all", type=int, default=0, help="measured rounds at every level (0: --repeat-top rule)")
    ap.add_argument("--ctx", default="", help="N:TOKENS, one more measured round with long prompts")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="scaling.json")
    ap.add_argument("--sampled-round", action="store_true",
                    help="one more warm and measured round at the top level with the server's default sampling "
                         "(no temperature in the request); also PROBE_SAMPLED_ROUND=1")
    args = ap.parse_args()
    if os.environ.get("PROBE_SAMPLED_ROUND", "") == "1":
        args.sampled_round = True
    levels = [int(x) for x in args.levels.split(",") if x]
    results = {"levels": [], "ctx": [], "sampled": [], "serial": []}

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    for n in levels:
        t = time.time()
        w = run_round(args, n, args.warm_tokens)
        print("warm  N=%-3d ok=%d err=%d wall=%.1fs ms/tok p50=%s" % (
            n, w["ok"], w["errors"], w["wall_s"], ("%.1f" % w["ms_per_token_p50"]) if w.get("ok") else "-"),
              flush=True)
        reps = args.repeat_all or (args.repeat_top if n == levels[-1] else 1)
        for rep in range(reps):
            m = run_round(args, n, args.tokens)
            m["rep"] = rep
            m["warm"] = {k: w.get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50")}
            results["levels"].append(m)
            save()
            if m.get("ok"):
                print("MEAS  N=%-3d rep=%d ok=%d err=%d  ms/tok p50=%.2f min=%.2f max=%.2f  tok/s/stream=%.1f  "
                      "aggregate=%.1f tok/s  ttft p50=%.2fs | steady ms/tok p50=%.2f (gap p50 x chunk), gap p90=%.1f ms, "
                      "pauses>0.5s max=%.1fs, tok/chunk=%.2f" % (
                          n, rep, m["ok"], m["errors"], m["ms_per_token_p50"], m["ms_per_token_min"],
                          m["ms_per_token_max"], m["tok_s_per_stream_p50"], m["aggregate_tok_s"] or 0,
                          m["ttft_p50_s"], m["steady_ms_per_token_p50"], m["gap_p90_ms_p50"], m["pause_s_max"],
                          m["tokens_per_chunk_p50"]), flush=True)
            else:
                print("MEAS  N=%d rep=%d FAILED: %s" % (n, rep, m.get("error_samples")), flush=True)
                break
        if not w.get("ok"):
            print("stopping: the warm round at N=%d failed: %s" % (n, w.get("error_samples")), flush=True)
            break
    serial = int(os.environ.get("PROBE_SERIAL_PROMPTS", "0") or 0)
    if serial > 0 and levels:
        # One request at a time, a different prompt each: the greedy text of a lone request is reproducible (that of
        # a request in a batch is not: it changes with the batch around it), so these texts are the token-for-token
        # test of two arms.
        for j in range(serial):
            m = run_round(args, 1, args.tokens, first=100 + j)
            m["prompt_index"] = 100 + j
            results["serial"].append(m)
            save()
            print("SERIAL prompt %d ok=%d err=%d steady ms/tok=%s text=%r" % (
                100 + j, m["ok"], m["errors"], ("%.2f" % m["steady_ms_per_token_p50"]) if m.get("ok") else "-",
                ((m.get("texts") or [""])[0] or "")[:70]), flush=True)
    if args.sampled_round and levels:
        # The general sampler path at the top level: no probe at temperature 0 runs it. Its text differs from run
        # to run and from arm to arm; only its ms per token is read.
        n = levels[-1]
        w = run_round(args, n, args.warm_tokens, sampled=True)
        m = run_round(args, n, args.tokens, sampled=True)
        m["warm"] = {k: w.get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50")}
        results["sampled"].append(m)
        save()
        if m.get("ok"):
            greedy = [x for x in results["levels"] if x["streams"] == n and x.get("ok")]
            ref = min(x["steady_ms_per_token_p50"] for x in greedy) if greedy else float("nan")
            print("SAMPLED N=%-3d server-default sampling: ms/tok p50=%.2f, steady ms/tok p50=%.2f (greedy: %.2f), "
                  "aggregate=%.1f tok/s, ok=%d err=%d" % (n, m["ms_per_token_p50"], m["steady_ms_per_token_p50"], ref,
                                                          m["aggregate_tok_s"] or 0, m["ok"], m["errors"]), flush=True)
        else:
            print("SAMPLED N=%d FAILED: %s (warm: %s)" % (n, m.get("error_samples"), w.get("error_samples")), flush=True)
    if args.ctx:
        n, ctx = (int(x) for x in args.ctx.split(":"))
        w = run_round(args, n, args.warm_tokens, ctx)
        m = run_round(args, n, args.tokens, ctx)
        m["warm"] = {k: w.get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50", "ttft_p50_s")}
        results["ctx"].append(m)
        save()
        if m.get("ok"):
            print("CTX   N=%d prompt~%s tok  ms/tok p50=%.2f max=%.2f  tok/s/stream=%.1f  aggregate=%.1f  "
                  "ttft p50=%.2fs (warm round ttft p50=%s)" % (
                      n, m.get("prompt_tokens_p50"), m["ms_per_token_p50"], m["ms_per_token_max"],
                      m["tok_s_per_stream_p50"], m["aggregate_tok_s"] or 0, m["ttft_p50_s"],
                      w.get("ttft_p50_s")), flush=True)
        else:
            print("CTX FAILED: %s" % m.get("error_samples"), flush=True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
