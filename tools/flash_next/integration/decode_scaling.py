#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode scaling probe: N concurrent streaming completions of equal length against one vLLM server.

Per level N: a warm round (short, compiles the decode graphs of batch N), then a measured round. Every stream
generates exactly --tokens tokens (max_tokens with ignore_eos; no min_tokens, whose logits mask runs on every row
of every step; --min-tokens adds it back, and one labelled MINTOK round at the top level measures its cost), temperature 0. Reports per stream the
time to first token and the decode time per token over the stream's own decode span, and for the round the
aggregate output tokens per second over the window in which all N streams were decoding.

Stdlib only. Usage:
    python3 decode_scaling.py --port 8000 --model fn --levels 1,8,16,32 --tokens 256 --out scaling.json
    python3 decode_scaling.py ... --ctx 32:4000     # one more measured round: 32 streams with ~4000-token prompts

The integration track's copy of the coordinator's probe (coord/decode_scaling.py, 2026-11-20 19:45 UTC) with two
additions: --sampled-levels LIST runs the SAMPLED round (warm, then measured) at every listed level instead of the
top level only, and on a server with speculation every measured round carries the speculative counters of
/metrics read around it ("spec": acceptance length, rate per draft position; a SPEC line), so each level has its
own acceptance. Everything else, the output format included, is the coordinator's.
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time

def _load_spec_probe():
    """The tree's reader of the speculative counters (tools/flash_next/rental/spec_probe.py, standard library
    only), loaded by its path; None when it is missing (the probe then has no per-round acceptance)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rental", "spec_probe.py")
    try:
        spec = importlib.util.spec_from_file_location("integration_spec_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001
        return None


spec_probe = _load_spec_probe()


def read_spec(args):
    """The server's speculative counters, or None (no speculation, no reader, no answer)."""
    if spec_probe is None:
        return None
    try:
        return spec_probe.read_spec_counters("http://%s:%d" % (args.host, args.port))
    except Exception:  # noqa: BLE001 - a probe records what it can
        return None


def measured_round(args, n, ntok, **kw):
    """run_round with the speculative counters read before and after it ("spec" in the result when present)."""
    before = read_spec(args)
    m = run_round(args, n, ntok, **kw)
    after = read_spec(args)
    if before is not None and after is not None:
        m["spec"] = spec_probe.spec_summary(spec_probe.counters_delta(before, after))
        s = m["spec"] or {}
        rates = ", ".join("%.3f" % r for r in (s.get("position_rates") or []) if r is not None)
        print("SPEC  N=%-3d%s acceptance length %s, tokens per chunk %s, draft acceptance %s, per position [%s], "
              "drafts %s" % (n, (" sampled" if kw.get("sampled") else "") +
                             ((" ctx%d" % kw["ctx_tokens"]) if kw.get("ctx_tokens") else ""),
                             ("%.3f" % s["acceptance_length"]) if s.get("acceptance_length") else "-",
                             ("%.2f" % m["tokens_per_chunk_p50"]) if m.get("tokens_per_chunk_p50") else "-",
                             ("%.3f" % s["draft_acceptance_rate"]) if s.get("draft_acceptance_rate") else "-",
                             rates, s.get("drafts")), flush=True)
    return m

WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river signal mapmaker bridge clock").split()


def prompt_for(i, ctx_tokens=0, salt=""):
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
    # salt: the warm and the measured long-prompt round start differently, so neither reads the other's prefix cache.
    return "%sStream %d notes: " % (salt, i) + " ".join(filler) + "\n\n" + head


def one_stream(host, port, model, prompt, ntok, out, idx, barrier, timeout_s, mintok=False, sampled=False):
    req = {"model": model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True,
           "stream": True, "stream_options": {"include_usage": True}}
    if not sampled:  # a sampled round sends no temperature: the server's default sampling (the checkpoint's generation_config)
        req["temperature"] = 0
    if mintok:
        req["min_tokens"] = ntok
    body = json.dumps(req)
    rec = {"status": None, "t0": None, "times": [], "ntoks": [], "usage": None, "err": None, "text": ""}
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
                    if len(rec["text"]) < 400:
                        rec["text"] += ch[0]["text"]
        conn.close()
    except Exception as e:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(e)
    out[idx] = rec


def run_round(args, n, ntok, ctx_tokens=0, mintok=None, sampled=False, salt=""):
    mintok = args.min_tokens if mintok is None else mintok
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream,
                                args=(args.host, args.port, args.model, prompt_for(i, ctx_tokens, salt), ntok, out, i,
                                      barrier, args.timeout, mintok, sampled), daemon=True) for i in range(n)]
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
    res = {"streams": n, "tokens": ntok, "ctx_tokens": ctx_tokens, "min_tokens": bool(mintok), "sampled": bool(sampled), "ok": len(per),
           "errors": errors,
           "wall_s": wall}
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
    # The start of a few streams' text: a multi-stream round whose later streams decode garbage must be visible.
    res["samples"] = {str(i): (out[i] or {}).get("text", "")[:160] for i in sorted({0, 1, n // 2, n - 1}) if i < n}
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
    ap.add_argument("--ctx", default="", help="N:TOKENS, one more measured round with long prompts")
    ap.add_argument("--min-tokens", action="store_true", help="send min_tokens = max_tokens in every round")
    ap.add_argument("--mintok-check", action="store_true",
                    help="one extra round at the top level that sends min_tokens (on HPU the mask never runs: off by default)")
    ap.add_argument("--no-mintok-check", action="store_true", help="accepted for older callers; the check is off by default")
    ap.add_argument("--no-sampled-check", action="store_true",
                    help="skip the extra round at the top level with the server's default sampling (no temperature in the "
                         "request). Pass it for a server with speculation: sampled verification is not built yet")
    ap.add_argument("--sampled-levels", default="",
                    help="stream counts of the SAMPLED rounds (default: the top level of --levels only)")
    ap.add_argument("--interleave-sampled", action="store_true",
                    help="run each level's SAMPLED rounds right after its greedy rounds (levels in the given order), "
                         "so the first level's numbers are complete first")
    ap.add_argument("--sampled-rounds", type=int, default=0,
                    help="measured SAMPLED rounds per level (default: --rounds, at least 1)")
    ap.add_argument("--rounds", type=int, default=0,
                    help="measured rounds at EVERY level, greedy and sampled, after one warm round (default 0: one "
                         "per level and --repeat-top at the top level, as the coordinator's probe)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="scaling.json")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x]
    sampled_levels = [int(x) for x in args.sampled_levels.split(",") if x] or levels[-1:]
    results = {"levels": [], "ctx": [], "mintok": [], "sampled": []}

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    def sampled_level(n):
        # What a client that sets no temperature gets: the server's default sampling on every step.
        w = run_round(args, n, args.warm_tokens, sampled=True)
        for rep in range(max(1, args.sampled_rounds or args.rounds)):
            m = measured_round(args, n, args.tokens, sampled=True)
            m["rep"] = rep
            m["warm"] = {k: w.get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50")}
            results["sampled"].append(m)
            save()
            if m.get("ok"):
                base = [x for x in results["levels"] if x["streams"] == n and x.get("ok")]
                ref = min(x["steady_ms_per_token_p50"] for x in base) if base else float("nan")
                print("SAMPLED N=%-3d rep=%d server default sampling: ms/tok p50=%.2f, steady ms/tok p50=%.2f "
                      "(greedy: %.2f), tok/s/stream=%.1f, aggregate=%.1f tok/s" % (
                          n, rep, m["ms_per_token_p50"], m["steady_ms_per_token_p50"], ref,
                          m["tok_s_per_stream_p50"], m["aggregate_tok_s"] or 0), flush=True)
                for k, t in m["samples"].items():
                    print("TEXT  N=%-3d sampled stream %-2s: %r" % (n, k, t[:110]), flush=True)
            else:
                print("SAMPLED N=%d rep=%d FAILED: %s" % (n, rep, m.get("error_samples")), flush=True)
                break


    for n in levels:
        t = time.time()
        w = run_round(args, n, args.warm_tokens)
        print("warm  N=%-3d ok=%d err=%d wall=%.1fs ms/tok p50=%s" % (
            n, w["ok"], w["errors"], w["wall_s"], ("%.1f" % w["ms_per_token_p50"]) if w.get("ok") else "-"),
              flush=True)
        reps = args.rounds if args.rounds > 0 else (args.repeat_top if n == levels[-1] else 1)
        for rep in range(reps):
            m = measured_round(args, n, args.tokens)
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
        last = [x for x in results["levels"] if x["streams"] == n and x.get("samples")]
        if last:
            for k, t in last[-1]["samples"].items():
                print("TEXT  N=%-3d stream %-2s: %r" % (n, k, t[:110]), flush=True)
        if args.interleave_sampled and not args.no_sampled_check and n in sampled_levels:
            sampled_level(n)
    if levels and args.mintok_check and not args.min_tokens:
        n = levels[-1]
        w = run_round(args, n, args.warm_tokens, mintok=True)  # compiles whatever the mask adds
        m = run_round(args, n, args.tokens, mintok=True)
        results["mintok"].append(m)
        save()
        if m.get("ok"):
            base = [x for x in results["levels"] if x["streams"] == n and x.get("ok")]
            ref = min(x["steady_ms_per_token_p50"] for x in base) if base else float("nan")
            print("MINTOK N=%-3d with min_tokens: ms/tok p50=%.2f, steady ms/tok p50=%.2f (without: %.2f), "
                  "tok/s/stream=%.1f, aggregate=%.1f tok/s" % (
                      n, m["ms_per_token_p50"], m["steady_ms_per_token_p50"], ref, m["tok_s_per_stream_p50"],
                      m["aggregate_tok_s"] or 0), flush=True)
        else:
            print("MINTOK N=%d FAILED: %s" % (n, m.get("error_samples")), flush=True)
    if args.ctx:
        n, ctx = (int(x) for x in args.ctx.split(":"))
        w = run_round(args, n, args.warm_tokens, ctx, salt="Warm round. ")
        m = measured_round(args, n, args.tokens, ctx_tokens=ctx, salt="Measured round. ")
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
    if not args.interleave_sampled:
        for n in (sampled_levels if levels and not args.no_sampled_check else []):
            sampled_level(n)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
