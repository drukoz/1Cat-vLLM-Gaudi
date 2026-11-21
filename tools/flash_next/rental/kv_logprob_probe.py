#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced accuracy probe of a K/V storage form (build 5's FP8 K/V decision, coordinator 2026-11-21 08:12 UTC).

    python3 kv_logprob_probe.py --port P --out FILE.json [--lengths 16000,64000,120000] [--last 2048]
                                [--passes 2] [--reference BF16.json] [--docs DIR] [--model fn]

For each length L, one completion request whose prompt is L tokens of this tree's documents and Python sources
(sorted, never repeated) with the rental gate's needle sentence at 30 % and its question at the end, sent as token
ids (the server's own /tokenize), greedy, 24 new tokens, ``prompt_logprobs=1``. It keeps, for the last ``--last``
prompt positions, the logprob of the prompt's own next token (teacher forcing), the top-1 token id and its logprob,
and the generated answer (the needle check at that length).

``--passes 2`` (default) sends each prompt twice and keeps the second: the server's prompt-logprob gather narrows
top-k ids to int32 and reads them on the host, which card rule 9 makes wrong on the FIRST run of a recipe; the
second run of the same shapes replays it. Both passes' top-1 ids are compared (``first_pass_top1_agree``).

With ``--reference`` (a file this script wrote on the same build with bf16 K/V) it compares, per length, over the
same positions: mean, p99 and max |delta| of the teacher-token logprobs, the top-1 agreement, and each run's mean
teacher logprob (``exp(-mean)`` = the perplexity over those positions); the prompts must be the same token ids
(checked by hash). Lines: ``KVLP ...``; the JSON holds every number.
"""
import argparse
import hashlib
import http.client
import json
import math
import sys
import time
from pathlib import Path

NEEDLE_SENTENCE = "Note for the operator: the secret deployment code for the Gaudi rental is 7319-ALPHA."
NEEDLE_CODE = "7319-ALPHA"
NEEDLE_QUESTION = "\n\nQuestion: what is the secret deployment code for the Gaudi rental? Answer with the code only."
try:  # the rental gate's values when this runs from the tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gates as _gates  # noqa: E402

    NEEDLE_SENTENCE, NEEDLE_CODE, NEEDLE_QUESTION = _gates.NEEDLE_SENTENCE, _gates.NEEDLE_CODE, _gates.NEEDLE_QUESTION
except Exception:  # noqa: BLE001
    pass


def say(text):
    print("KVLP " + text, flush=True)


def post(args, path, payload, timeout_s):
    conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout_s)
    conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError("HTTP %s from %s: %s" % (resp.status, path, body[:600].decode("utf-8", "replace")))
    return json.loads(body)


def corpus(root: Path, want_chars: int) -> str:
    """This tree's Markdown documents, then its Python sources, sorted, without the needle code, never repeated."""
    files = sorted(root.rglob("*.md")) + sorted(p for p in root.rglob("*.py") if "/tests/" not in str(p))
    parts, size = [], 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if NEEDLE_CODE in text or len(text) < 200:
            continue
        parts.append("# File %s\n\n%s" % (f.relative_to(root), text))
        size += len(text)
        if size >= want_chars:
            break
    return "\n\n".join(parts)


def prompt_ids(args, length: int) -> list:
    """``length`` token ids: the corpus cut to ``length`` minus the question, the needle at 30 %, the question."""
    text = corpus(Path(args.docs), length * 6)
    q_ids = post(args, "/tokenize", {"model": args.model, "prompt": NEEDLE_QUESTION, "add_special_tokens": False},
                 600.0)["tokens"]
    n_ids = post(args, "/tokenize", {"model": args.model, "prompt": "\n" + NEEDLE_SENTENCE + "\n",
                                     "add_special_tokens": False}, 600.0)["tokens"]
    body = post(args, "/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": False}, 1200.0)["tokens"]
    room = length - len(q_ids) - len(n_ids)
    if len(body) < room:
        raise RuntimeError("the corpus has %d tokens, %d wanted" % (len(body), room))
    body = body[:room]
    at = int(0.3 * room)
    return body[:at] + n_ids + body[at:] + q_ids


def one_request(args, ids: list) -> dict:
    t0 = time.perf_counter()
    res = post(args, "/v1/completions", {"model": args.model, "prompt": ids, "max_tokens": 24, "temperature": 0,
                                         "prompt_logprobs": 1, "skip_special_tokens": False}, args.timeout)
    wall = time.perf_counter() - t0
    ch = res["choices"][0]
    plp = ch.get("prompt_logprobs") or []
    n = len(ids)
    lo = max(1, n - args.last)
    teacher, top1, top1_lp, ranks, missing = [], [], [], [], 0
    for i in range(lo, n):
        entry = plp[i] if i < len(plp) else None
        if not entry:
            missing += 1
            teacher.append(None), top1.append(None), top1_lp.append(None), ranks.append(None)
            continue
        tid = str(ids[i])
        t = entry.get(tid) or entry.get(int(ids[i]))
        ranked = [(int(k), float(v["logprob"])) for k, v in entry.items() if v.get("rank") == 1]
        best = ranked[0] if ranked else max(((int(k), float(v["logprob"])) for k, v in entry.items()),
                                            key=lambda kv: kv[1])
        teacher.append(float(t["logprob"]) if t else None)
        ranks.append(int(t["rank"]) if t and t.get("rank") is not None else None)
        top1.append(best[0] if best else None)
        top1_lp.append(best[1] if best else None)
    return {"wall_s": round(wall, 2), "text": ch.get("text", ""), "needle_hit": NEEDLE_CODE in (ch.get("text") or ""),
            "first_position": lo, "teacher_logprob": teacher, "top1": top1, "top1_logprob": top1_lp,
            "teacher_rank": ranks, "missing": missing, "usage": res.get("usage")}


def compare(mine: dict, ref: dict) -> dict:
    a, b = mine["teacher_logprob"], ref["teacher_logprob"]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    diffs = sorted(abs(x - y) for x, y in pairs)
    t1 = [(x, y) for x, y in zip(mine["top1"], ref["top1"]) if x is not None and y is not None]
    out = {"positions": len(pairs)}
    if diffs:
        out.update(mean_abs_delta=sum(diffs) / len(diffs), p99_abs_delta=diffs[min(len(diffs) - 1,
                                                                                   int(0.99 * len(diffs)))],
                   max_abs_delta=diffs[-1], mean_logprob=sum(x for x, _ in pairs) / len(pairs),
                   ref_mean_logprob=sum(y for _, y in pairs) / len(pairs))
        out["ppl"] = math.exp(-out["mean_logprob"])
        out["ref_ppl"] = math.exp(-out["ref_mean_logprob"])
    if t1:
        out["top1_agree"] = sum(1 for x, y in t1 if x == y) / len(t1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="16000,64000,120000")
    ap.add_argument("--last", type=int, default=2048)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--docs", default=None, help="the build's source tree (default: this script's tree)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.docs is None:
        here = Path(__file__).resolve().parents[3]
        if not (here / "vllm_gaudi").is_dir():
            ap.error("--docs: this script is not inside the tree; give the build's source tree")
        args.docs = str(here)
    ref = json.load(open(args.reference)) if args.reference else None
    result = {"label": args.label, "args": vars(args), "lengths": {}}
    for length in [int(v) for v in args.lengths.split(",") if v]:
        rec = {"length": length}
        try:
            ids = prompt_ids(args, length)
            rec["prompt_sha"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
            rec["prompt_tokens"] = len(ids)
            runs = [one_request(args, ids) for _ in range(max(1, args.passes))]
            keep = runs[-1]
            rec.update(keep)
            rec["walls_s"] = [r["wall_s"] for r in runs]
            if len(runs) > 1:
                p = [(x, y) for x, y in zip(runs[0]["top1"], keep["top1"]) if x is not None and y is not None]
                rec["first_pass_top1_agree"] = (sum(1 for x, y in p if x == y) / len(p)) if p else None
            valid = [v for v in keep["teacher_logprob"] if v is not None]
            rec["mean_teacher_logprob"] = sum(valid) / len(valid) if valid else None
            line = "%s L=%d tokens=%d sha=%s walls=%s needle=%s answer=%r mean_lp=%s missing=%d" % (
                args.label, length, len(ids), rec["prompt_sha"], rec["walls_s"], keep["needle_hit"],
                keep["text"][:60], "%.4f" % rec["mean_teacher_logprob"] if valid else None, keep["missing"])
            if ref is not None:
                r = ref.get("lengths", {}).get(str(length))
                if not r or r.get("prompt_sha") != rec["prompt_sha"]:
                    rec["compare"] = {"error": "no reference with the same prompt (sha %s against %s)" % (
                        rec["prompt_sha"], r.get("prompt_sha") if r else None)}
                else:
                    rec["compare"] = compare(keep, r)
                c = rec["compare"]
                if "error" in c:
                    line += " COMPARE %s" % c["error"]
                else:
                    line += (" COMPARE positions=%d mean|d|=%.4f p99|d|=%.4f max|d|=%.4f top1_agree=%.4f ppl=%.4f "
                             "ref_ppl=%.4f" % (c["positions"], c.get("mean_abs_delta", float("nan")),
                                               c.get("p99_abs_delta", float("nan")), c.get("max_abs_delta",
                                                                                         float("nan")),
                                               c.get("top1_agree", float("nan")), c.get("ppl", float("nan")),
                                               c.get("ref_ppl", float("nan"))))
            say(line)
        except Exception as e:  # noqa: BLE001 - a probe records every failure
            rec["error"] = repr(e)[:800]
            say("%s L=%d FAILED: %s" % (args.label, length, rec["error"]))
        result["lengths"][str(length)] = rec
        Path(args.out).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
