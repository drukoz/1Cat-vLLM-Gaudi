#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode probe of the sparse-QSA A/B arms (qsa_ab_job.sh): short and long context, texts saved, a needle.

Against one running vLLM server, in this order (each result is one plain ``QSAAB`` line on standard output as it
is produced, and an entry of the JSON file):

1. **short context**: the coordinator's decode scaling rounds (the prompts and metrics of
   ``coord/decode_scaling.py``): per stream level a warm round, then a measured one (two at the top level). Every
   stream generates exactly ``--tokens`` tokens (``max_tokens`` with ``ignore_eos``; never ``min_tokens``, whose
   logits mask runs on every row of every step), temperature 0.
2. **needle**: one chat request whose user message is about ``--ctx`` tokens of this repository's design
   documents with the rental gate's needle sentence at 30 % of its length and the gate's question at the end
   (``gates.NEEDLE_*``, ``gates._insert_needle``): the long-context correctness check, and the request that
   prefills the shared prefix.
3. **long context**: ``N`` streams whose prompts are that same document and question plus a few words of their
   own, so the server serves the prefix from its cache and every stream decodes at about ``--ctx`` tokens of
   context. One trial stream first: its cached-token count says whether the prefix was reused. Without reuse
   every stream prefills the whole document once (in the warm round; the measured rounds send the same prompts
   again), and the round runs with as many streams as the time to the deadline covers.

Every round keeps each stream's text, so two servers (the two arms) can be compared at temperature 0. A stage is
skipped, with a line that says so, when the time to ``--deadline`` does not cover it. Stdlib only.
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river signal mapmaker bridge clock").split()

# The rental gate's needle (tools/flash_next/rental/gates.py); the fallbacks are its values of 2026-11-20.
NEEDLE_SENTENCE = "Note for the operator: the secret deployment code for the Gaudi rental is 7319-ALPHA."
NEEDLE_CODE = "7319-ALPHA"
NEEDLE_QUESTION = "\n\nQuestion: what is the secret deployment code for the Gaudi rental? Answer with the code only."
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.flash_next.rental import gates as _gates
    NEEDLE_SENTENCE, NEEDLE_CODE, NEEDLE_QUESTION = _gates.NEEDLE_SENTENCE, _gates.NEEDLE_CODE, _gates.NEEDLE_QUESTION
    insert_needle = _gates._insert_needle
except Exception:  # noqa: BLE001 - the probe also runs where the harness does not import

    def insert_needle(content: str) -> str:
        pos = int(len(content) * 0.30)
        idx = content.find("\n", pos)
        if idx == -1:
            return content + "\n" + NEEDLE_SENTENCE
        return content[:idx + 1] + NEEDLE_SENTENCE + "\n" + content[idx + 1:]


def say(text: str) -> None:
    print("QSAAB %s %s" % (time.strftime("%H:%M:%S", time.gmtime()), text), flush=True)


def short_prompt(i: int) -> str:
    """The coordinator's short prompt of stream ``i`` (coord/decode_scaling.py ``prompt_for``)."""
    return "Stream %d. Write a long, detailed story about a %s named %s who finds a %s. Begin:" % (
        i, WORDS[i % len(WORDS)], WORDS[(i * 7 + 3) % len(WORDS)].title(), WORDS[(i * 3 + 5) % len(WORDS)])


def post_json(args, path: str, payload: dict, timeout_s: float) -> dict:
    conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout_s)
    conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError("HTTP %s from %s: %s" % (resp.status, path, body[:400].decode("utf-8", "replace")))
    return json.loads(body)


def one_stream(args, request: dict, out: list, idx: int, barrier, timeout_s: float) -> None:
    """One streamed request: ``request`` is ``{"path", "body"}``; records chunk times, the text and the usage."""
    rec = {"status": None, "t0": None, "times": [], "text": [], "usage": None, "err": None}
    try:
        barrier.wait()
        conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout_s)
        rec["t0"] = time.perf_counter()
        conn.request("POST", request["path"], body=json.dumps(request["body"]),
                     headers={"Content-Type": "application/json"})
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
                if not ch:
                    continue
                piece = ch[0].get("text")
                if piece is None:
                    piece = (ch[0].get("delta") or {}).get("content")
                if piece:
                    rec["times"].append(time.perf_counter())
                    rec["text"].append(piece)
        conn.close()
    except Exception as e:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(e)
    out[idx] = rec


def completion(args, prompt: str, ntok: int) -> dict:
    return {"path": "/v1/completions",
            "body": {"model": args.model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True, "temperature": 0,
                     "stream": True, "stream_options": {"include_usage": True}}}


def chat(args, content: str, ntok: int) -> dict:
    return {"path": "/v1/chat/completions",
            "body": {"model": args.model, "messages": [{"role": "user", "content": content}], "max_tokens": ntok,
                     "ignore_eos": True, "temperature": 0, "stream": True,
                     "stream_options": {"include_usage": True},
                     # Special tokens as text: with ignore_eos a chat answer goes on past its end token, and a
                     # token without text would be a step without a streamed chunk (no time for it).
                     "skip_special_tokens": False,
                     "chat_template_kwargs": {"enable_thinking": False}}}


def run_round(args, label: str, requests: list, ntok: int) -> dict:
    """``len(requests)`` concurrent streams; the metrics of coord/decode_scaling.py plus texts and cached tokens."""
    n = len(requests)
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream, args=(args, requests[i], out, i, barrier, args.timeout),
                                daemon=True) for i in range(n)]
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
        usage = r["usage"] or {}
        comp = usage.get("completion_tokens") or len(r["times"])
        span = r["times"][-1] - r["times"][0]
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:])]
        tpc = comp / len(r["times"])  # tokens per streamed chunk (above 1 with speculation)
        gmed = statistics.median(gaps)
        per.append({"steady_ms_per_token": 1000.0 * gmed / max(tpc, 1e-9),
                    "gap_p90_ms": 1000.0 * sorted(gaps)[int(0.9 * (len(gaps) - 1))],
                    "pause_s": sum(g for g in gaps if g > 0.5), "tokens_per_chunk": tpc,
                    "ttft_s": r["times"][0] - r["t0"], "ms_per_token": 1000.0 * span / max(1, comp - 1),
                    "completion_tokens": comp, "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "first": r["times"][0], "last": r["times"][-1]})
    res = {"label": label, "streams": n, "tokens": ntok, "min_tokens": False, "ok": len(per), "errors": errors,
           "wall_s": wall, "texts": texts}
    if per:
        ms = sorted(p["ms_per_token"] for p in per)
        res["steady_ms_per_token_p50"] = statistics.median(p["steady_ms_per_token"] for p in per)
        res["gap_p90_ms_p50"] = statistics.median(p["gap_p90_ms"] for p in per)
        res["pause_s_max"] = max(p["pause_s"] for p in per)
        res["tokens_per_chunk_p50"] = statistics.median(p["tokens_per_chunk"] for p in per)
        res["ms_per_token_p50"] = statistics.median(ms)
        res["ms_per_token_min"] = ms[0]
        res["ms_per_token_max"] = ms[-1]
        res["tok_s_per_stream_p50"] = 1000.0 / res["ms_per_token_p50"]
        res["ttft_p50_s"] = statistics.median(p["ttft_s"] for p in per)
        res["ttft_max_s"] = max(p["ttft_s"] for p in per)
        res["prompt_tokens_p50"] = statistics.median([p["prompt_tokens"] or 0 for p in per])
        res["cached_tokens_p50"] = statistics.median([p["cached_tokens"] or 0 for p in per])
        total = sum(p["completion_tokens"] for p in per)
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = total / span_all if span_all > 0 else None
    errs = [r["err"] for r in out if r and r["err"]][:3]
    if errs:
        res["error_samples"] = errs
    return res


def line(res: dict) -> str:
    if not res.get("ok"):
        return "%s N=%d FAILED: %s" % (res["label"], res["streams"], res.get("error_samples"))
    return ("%s N=%d ok=%d err=%d: ms/token p50=%.2f (min %.2f max %.2f), steady ms/token p50=%.2f, "
            "tok/s per stream=%.1f, total tok/s=%.1f, ttft p50=%.2fs, pauses>0.5s max=%.1fs, prompt tokens p50=%s "
            "(cached %s), wall %.1fs" % (
                res["label"], res["streams"], res["ok"], res["errors"], res["ms_per_token_p50"],
                res["ms_per_token_min"], res["ms_per_token_max"], res["steady_ms_per_token_p50"],
                res["tok_s_per_stream_p50"], res["aggregate_tok_s"] or 0, res["ttft_p50_s"], res["pause_s_max"],
                int(res["prompt_tokens_p50"]), int(res["cached_tokens_p50"]), res["wall_s"]))


def document(args) -> tuple[str, int]:
    """About ``--ctx`` tokens of the repository's design documents with the needle sentence at 30 %: (text, tokens)."""
    root = Path(args.docs)
    files = sorted(root.rglob("*.md")) if root.is_dir() else []
    parts, size = [], 0
    want_chars = args.ctx * 8  # more than enough text for the cut below
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if NEEDLE_CODE in text:
            continue
        parts.append("# File %s\n\n%s" % (f.name, text))
        size += len(text)
        if size >= want_chars:
            break
    corpus = "\n\n".join(parts)
    if len(corpus) < 2000:  # no documents here: filler words
        corpus = "\n".join(" ".join(WORDS[(i + k * 11) % len(WORDS)] for k in range(12)) for i in range(args.ctx))
    while len(corpus) < want_chars:  # a small tree: the same text again (the needle stays unique)
        corpus = corpus + "\n\n" + corpus
    sample = corpus[:200000]
    count = post_json(args, "/tokenize", {"model": args.model, "prompt": sample}, 600.0)["count"]
    chars = int(args.ctx * len(sample) / max(1, count))
    cut = corpus.rfind("\n", 0, chars)
    text = insert_needle(corpus[:cut if cut > 0 else chars])
    tokens = post_json(args, "/tokenize", {"model": args.model, "prompt": text}, 600.0)["count"]
    return text, tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--levels", default="1,16,32")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warm-tokens", type=int, default=48)
    ap.add_argument("--repeat-top", type=int, default=2, help="measured rounds at the largest level")
    ap.add_argument("--ctx", type=int, default=32000, help="tokens of the long-context document (0: no long rounds)")
    ap.add_argument("--long-streams", type=int, default=0, help="streams of the long round (default: the top level)")
    ap.add_argument("--docs", default="docs/design/flash-next", help="folder of the documents of the long prompt")
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch seconds by which the probe must have ended")
    ap.add_argument("--timeout", type=float, default=1500.0)
    ap.add_argument("--out", default="qsa_ab.json")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x]
    results = {"args": {k: v for k, v in vars(args).items()}, "levels": [], "needle": None, "long": [], "skipped": []}

    def save():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, args.out)

    def left() -> float:
        return float("inf") if args.deadline <= 0 else args.deadline - time.time()

    def skip(stage: str, need_s: float) -> bool:
        if left() >= need_s:
            return False
        say("SKIPPED %s: %.0f s to the deadline, %.0f s wanted" % (stage, left(), need_s))
        results["skipped"].append({"stage": stage, "left_s": left(), "need_s": need_s})
        save()
        return True

    # ---- 1. short context
    for n in levels:
        if skip("short N=%d" % n, 150):
            break
        warm = run_round(args, "warm-short", [completion(args, short_prompt(i), args.warm_tokens) for i in range(n)],
                         args.warm_tokens)
        say("warm  " + line(warm))
        if not warm.get("ok"):
            results["levels"].append(warm)
            save()
            break
        for rep in range(args.repeat_top if n == levels[-1] else 1):
            res = run_round(args, "short", [completion(args, short_prompt(i), args.tokens) for i in range(n)],
                            args.tokens)
            res["rep"] = rep
            res["warm"] = {k: warm.get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50")}
            results["levels"].append(res)
            save()
            say("MEAS  " + line(res))
    # ---- 2. the needle, which also prefills the shared prefix
    if args.ctx <= 0 or skip("needle and long context", 420):
        save()
        return 0
    try:
        doc, doc_tokens = document(args)
    except Exception as e:  # noqa: BLE001
        say("NEEDLE FAILED: the document could not be built: %r" % (e, ))
        results["needle"] = {"error": repr(e)}
        save()
        return 0
    say("long prompt: %d characters, %d tokens, needle sentence at 30 %% of its length" % (len(doc), doc_tokens))
    t = time.time()
    needle = {"doc_chars": len(doc), "doc_tokens": doc_tokens}
    try:
        resp = post_json(args, "/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": doc + NEEDLE_QUESTION}], "max_tokens": 32,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, args.timeout)
        answer = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        needle.update({"answer": answer[:500], "hit": NEEDLE_CODE in answer, "seconds": time.time() - t,
                       "usage": resp.get("usage")})
    except Exception as e:  # noqa: BLE001
        needle.update({"error": repr(e), "hit": False, "seconds": time.time() - t})
    results["needle"] = needle
    save()
    say("NEEDLE at %d prompt tokens: hit=%s, answer=%r, %.1f s (prefill of the whole prompt, graphs compiled)" % (
        ((needle.get("usage") or {}).get("prompt_tokens") or doc_tokens), needle.get("hit"),
        needle.get("answer", needle.get("error")), needle["seconds"]))
    if "error" in needle:
        return 0

    def long_requests(n: int, ntok: int) -> list:
        # The same prompt for stream i in every round: a round after the first finds its own prompts in the cache
        # even on a server that reuses a prefix only for a prompt it has seen whole.
        return [chat(args, doc + NEEDLE_QUESTION + "\nRequest %d." % i, ntok) for i in range(n)]

    def hits(res: dict) -> int:
        return sum(1 for text in res.get("texts", []) if text and NEEDLE_CODE in text)

    # ---- 3. long context: a trial stream says whether the prefix is served from the cache
    if skip("long trial", 240):
        return 0
    trial = run_round(args, "long-trial", long_requests(1, 16), 16)
    trial["needle_hits"] = hits(trial)
    results["long"].append(trial)
    save()
    say("TRIAL " + line(trial))
    if not trial.get("ok"):
        return 0
    one_doc_s = trial["ttft_p50_s"]  # without reuse: the warm prefill of one whole document
    reuse = (trial.get("cached_tokens_p50") or 0) >= 0.5 * doc_tokens
    n_long = args.long_streams or levels[-1]
    if not reuse:  # every stream prefills its document once: as many streams as the time covers
        while n_long > 1 and 180 + 1.3 * n_long * one_doc_s > left() - 120:
            n_long //= 2
    say("prefix reuse: %s (cached tokens %s of %d; the trial's first token after %.1f s, the first prefill took "
        "%.1f s): the long round runs %d streams" % (reuse, trial.get("cached_tokens_p50"), doc_tokens, one_doc_s,
                                                     needle["seconds"], n_long))
    results["prefix_reuse"] = reuse
    per_round = 60
    if not skip("long N=1", per_round):
        res = run_round(args, "long", long_requests(1, args.tokens), args.tokens)
        res["needle_hits"] = hits(res)
        results["long"].append(res)
        save()
        say("MEAS  " + line(res) + ", needle in %d of %d texts" % (res["needle_hits"], res["streams"]))
    if n_long > 1 and not skip("long N=%d warm" % n_long, per_round + 60):
        warm = run_round(args, "warm-long", long_requests(n_long, 16), 16)
        warm["needle_hits"] = hits(warm)
        results["long"].append(warm)
        save()
        say("warm  " + line(warm))
        for rep in range(args.repeat_top):
            if skip("long N=%d rep %d" % (n_long, rep), per_round):
                break
            res = run_round(args, "long", long_requests(n_long, args.tokens), args.tokens)
            res["rep"] = rep
            res["needle_hits"] = hits(res)
            results["long"].append(res)
            save()
            say("MEAS  " + line(res) + ", needle in %d of %d texts" % (res["needle_hits"], res["streams"]))
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
