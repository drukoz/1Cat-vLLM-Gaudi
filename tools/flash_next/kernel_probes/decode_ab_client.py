#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode A/B client: the coordinator's decode scaling probe (coord/decode_scaling.py of 2026-11-20, same rounds,
same prompts, same scaling.json) with two changes for an A/B of a flag between two servers.

  * It never sends ``min_tokens`` (its logits mask runs ``index_put_`` on every row of every step): ``max_tokens``
    with ``ignore_eos`` gives the exact length. There is no MINTOK round.
  * It stores what every stream generated (``texts.json`` next to ``--out``), so two arms can be compared at
    temperature 0:  ``decode_ab_client.py --compare A/texts.json B/texts.json`` prints, per round, how many
    streams returned the same text and the shortest common prefix.

Per level N: a warm round (short, compiles the decode graphs of batch N), then a measured round. Every stream
generates exactly --tokens tokens, temperature 0. Reports per stream the time to first token and the decode time
per token over the stream's own decode span, the median gap between streamed chunks (the steady ms per token,
without compile pauses), and for the round the aggregate output tokens per second.

Stdlib only. The farm's serve_probe.sh runs the file named decode_scaling.py next to it with
``--port --model --out --levels --tokens --warm-tokens --repeat-top [--ctx]``: this client takes the same options
(tools/flash_next/kernel_probes/real_ab.sh puts it there).
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


def one_stream(host, port, model, prompt, ntok, out, idx, barrier, timeout_s):
    req = {"model": model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True,
           "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
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


def run_round(args, n, ntok, ctx_tokens=0):
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream,
                                args=(args.host, args.port, args.model, prompt_for(i, ctx_tokens), ntok, out, i,
                                      barrier, args.timeout), daemon=True) for i in range(n)]
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
    res = {"streams": n, "tokens": ntok, "ctx_tokens": ctx_tokens, "min_tokens": False, "ok": len(per),
           "errors": errors,
           "wall_s": wall}
    res["_texts"] = ["".join(r["text"]) if r and not r["err"] and r["status"] == 200 else None for r in out]
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


def compare(path_a, path_b):
    """Per round of two arms: streams with the same text, and the shortest common prefix in characters.

    Only the 1-stream rounds are a test: the real model's greedy output is not reproducible across batch
    compositions, nor between two 32-stream rounds of one server (job 00021, 2026-11-20: stream 0 leaves its
    1-stream text after 55 characters at 8 streams and after 23 at 32). The other rounds are printed to be read.
    """
    with open(path_a) as f:
        a = {r["round"]: r for r in json.load(f)}
    with open(path_b) as f:
        b = {r["round"]: r for r in json.load(f)}
    all_same = True
    single_same, single_rounds = 0, 0
    for name, ra in a.items():
        rb = b.get(name)
        if rb is None:
            print("COMPARE %-22s only in %s" % (name, path_a))
            continue
        same, shortest, pairs, lengths = 0, None, 0, []
        for ta, tb in zip(ra["texts"], rb["texts"]):
            if ta is None or tb is None:
                continue
            pairs += 1
            common = 0
            for ca, cb in zip(ta, tb):
                if ca != cb:
                    break
                common += 1
            same += int(ta == tb)
            lengths.append(min(len(ta), len(tb)))
            shortest = common if shortest is None else min(shortest, common)
        all_same = all_same and same == pairs
        if " N=1 " in name and pairs:
            single_rounds += 1
            single_same += int(same == pairs)
        print("COMPARE %-22s %d of %d streams returned the same text; shortest common prefix %s of about %s "
              "characters" % (name, same, pairs, shortest, (sum(lengths) // len(lengths)) if lengths else "-"))
    print("COMPARE verdict: 1-stream rounds (the test): the same text in %d of %d; all rounds: %s" % (
        single_same, single_rounds, "the same text everywhere" if all_same else
        "texts differ in some multi-stream round (expected: batched greedy output is not reproducible; read them)"
        if single_same == single_rounds else "texts differ"))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--levels", default="1,8,16,32")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--warm-tokens", type=int, default=48)
    ap.add_argument("--repeat-top", type=int, default=2, help="measured rounds at the largest level")
    ap.add_argument("--ctx", default="", help="N:TOKENS, one more measured round with long prompts")
    ap.add_argument("--compare", nargs=2, metavar=("A_TEXTS", "B_TEXTS"),
                    help="compare the texts.json of two arms and exit (no server)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="scaling.json")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    levels = [int(x) for x in args.levels.split(",") if x]
    results = {"levels": [], "ctx": [], "mintok": []}
    texts = []
    texts_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "texts.json")

    def keep(kind, m, rep=0):
        """Move the round's texts out of its record (scaling.json keeps the coordinator's fields).

        A measured round prints the start of the first, the second, a middle and the last stream's text (TEXT
        lines): a multi-stream round whose later streams decode garbage must not pass as a speed number.
        """
        round_texts = m.pop("_texts", [])
        texts.append({"round": "%s N=%d rep=%d" % (kind, m["streams"], rep), "tokens": m["tokens"],
                      "ctx_tokens": m["ctx_tokens"], "texts": round_texts})
        if kind in ("meas", "ctx") and rep == 0:
            n = len(round_texts)
            for i in sorted({0, 1, n // 2, n - 1} & set(range(n))):
                print("TEXT  N=%-3d stream %2d: %r" % (n, i, (round_texts[i] or "")[:90]), flush=True)

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        with open(texts_path, "w") as f:
            json.dump(texts, f, indent=1)

    for n in levels:
        t = time.time()
        w = run_round(args, n, args.warm_tokens)
        keep("warm", w)
        print("warm  N=%-3d ok=%d err=%d wall=%.1fs ms/tok p50=%s" % (
            n, w["ok"], w["errors"], w["wall_s"], ("%.1f" % w["ms_per_token_p50"]) if w.get("ok") else "-"),
              flush=True)
        reps = args.repeat_top if n == levels[-1] else 1
        for rep in range(reps):
            m = run_round(args, n, args.tokens)
            keep("meas", m, rep)
            m["rep"] = rep
            m["warm"] = {k: w.get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50")}
            results["levels"].append(m)
            save()
            if m.get("ok"):
                print("MEAS  N=%-3d rep=%d ok=%d err=%d  ms/tok p50=%.2f min=%.2f max=%.2f  tok/s/stream=%.1f  "
                      "aggregate=%.1f tok/s  ttft p50=%.2fs | steady ms/tok p50=%.2f (gap p50 x chunk), "
                      "gap p90=%.1f ms, pauses>0.5s max=%.1fs, tok/chunk=%.2f" % (
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
    if args.ctx:
        n, ctx = (int(x) for x in args.ctx.split(":"))
        w = run_round(args, n, args.warm_tokens, ctx)
        keep("ctx-warm", w)
        m = run_round(args, n, args.tokens, ctx)
        keep("ctx", m)
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
