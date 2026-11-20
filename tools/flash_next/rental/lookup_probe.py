#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Speculation probe of the spec-lookup track: edit-and-return and fresh-prose chat requests, with the server's
speculative counters around every round.

Each round sends N concurrent streaming chat completions (temperature 0, thinking off, exactly TOKENS output tokens:
``max_tokens`` with ``ignore_eos``; never ``min_tokens``, whose logits processor runs ``index_put_`` every step and is
the one argmax-changing processor the HPU rejection sampler special-cases) and reads ``/metrics`` before and after.

Prompt kinds (stream ``i`` of a round always gets the same prompt, so the rounds of one job, and the jobs of one
commit, decode the same contexts and their texts can be compared token for token: greedy speculation is lossless):

  edit   an editing instruction plus one small Python file of this tree (600 to 800 tokens); the answer is the whole
         edited file, so long spans copy the prompt (the coding-agent case). The prompt stays under 1,024 tokens:
         with prefix caching on, the scheduler ends a prefill chunk at the last cacheable 512-token block boundary
         (one block earlier on an MTP server), and the MTP drafter never sees a chunk that completes no prompt
         (VLLM_HPU_MTP_CHUNK_KV=0); under 1,024 tokens an MTP server prefills the prompt in one chunk.
  editl  the same with a file of 3,000 to 3,600 tokens (longer copy spans; on an MTP server most of the prompt is in
         chunks the drafter did not see).
  editf  the edit prompts again, answered to the model's own end (no ignore_eos; TOKENS is only the cap): whole
         answers, for replaying lookup drafters over what the model really wrote (tools/flash_next/lookup_replay.py).
  prose  a story request with a distinct opening per stream (nothing to copy).

A round may carry a per-request lookup mode for a server with VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP (vllm_xargs
{"spec_lookup": m}: 0 MTP drafts, 1 hybrid, 2 lookup only).

Per round: tokens per verification step per request (vLLM's acceptance length ``1 + accepted / drafts`` from the
counters) and the rate of every draft position; ms per engine step (the median gap between a stream's chunks: one
chunk per step); effective ms per token over the decode span (compile pauses included) and steady
(``gap / acceptance length``); the sizes of the client-side steps (tokens per chunk, from the usage of every chunk).

Usage:
  lookup_probe.py --port P --out OUT.json --label L --rounds edit:32:384:0,edit:32:384:1,prose:32:256:-
      round = KIND:STREAMS:TOKENS:MODE (MODE '-' sends no vllm_xargs). Before the first round of each
      (KIND, STREAMS) a warm round runs: as long as the round for edit and prose, 48 tokens for editl and editf
      (--warm-tokens N sets one length for all, 0 skips it).
  lookup_probe.py --compare A.json B.json ...   compare the texts of several probe files
  lookup_probe.py --list-prompts                the prompts' files and character counts
Standard library only (Python 3.10).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import statistics
import sys
import threading
import time

try:  # beside this file on the node (/bundle/plugin/tools/flash_next/rental) and in the repository
    from spec_probe import counters_delta, parse_spec_counters, spec_summary
except ImportError:  # imported as a package module (the unit tests)
    from tools.flash_next.rental.spec_probe import counters_delta, parse_spec_counters, spec_summary

TEMPLATES = [
    "Add a one-line docstring to every function and method that lacks one. Return the complete file.",
    "Rename the local variable names that are a single letter (except loop indices i, j, k) to descriptive names. "
    "Return the complete file.",
    "Add type hints to every function signature that lacks them. Return the complete file.",
]
WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter orchard "
         "river signal mapmaker bridge clock").split()
# Files of this tree by size with the checkpoint's tokenizer (2026-11-20): 600 to 800 tokens, and 3,000 to 3,600.
SMALL_FILES = [
    "vllm_gaudi/ops/hpu_attention.py",
    "vllm_gaudi/ops/hpu_lora.py",
    "vllm_gaudi/ops/hpu_conv.py",
    "vllm_gaudi/compilation/ordered_boundary.py",
    "vllm_gaudi/extension/runtime.py",
    "vllm_gaudi/ops/hpu_mm_encoder_attention.py",
    "vllm_gaudi/ops/gqa_compact.py",
    "vllm_gaudi/models/kimi_k25.py",
]
LARGE_FILES = [
    "vllm_gaudi/models/starcoder2.py",
    "vllm_gaudi/extension/bucketing/vision.py",
    "vllm_gaudi/models/llama4.py",
    "vllm_gaudi/utils.py",
]
KINDS = ("edit", "editl", "editf", "prose")
FULL_WARM = ("edit", "prose")  # kinds whose warm round is as long as the round (the others: SHORT_WARM tokens)
SHORT_WARM = 48
TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_files(tree: str, paths: list[str]) -> list[tuple[str, str]]:
    """``[(relative path, text)]`` of the listed files that exist under ``tree``."""
    files = []
    for rel in paths:
        path = os.path.join(tree, rel)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                files.append((rel, f.read()))
    return files


def prompt_for(kind: str, i: int, files: list[tuple[str, str]]) -> str:
    """The prompt of stream ``i`` of a round of ``kind`` (``files``: the kind's files; unused for prose)."""
    if kind in ("edit", "editl", "editf"):
        if not files:
            raise ValueError(f"no file for the {kind} prompts")
        rel, text = files[i % len(files)]
        body = text if text.endswith("\n") else text + "\n"
        template = TEMPLATES[(i // len(files)) % len(TEMPLATES)]
        return "Request %d. %s\n\nFile: %s\n\n```python\n%s```\n" % (i, template, rel, body)
    if kind == "prose":
        return ("Stream %d. Write a long, detailed story about a %s named %s who finds a %s. Do not use lists. Begin."
                % (i, WORDS[i % len(WORDS)], WORDS[(i * 7 + 3) % len(WORDS)].title(), WORDS[(i * 3 + 5) % len(WORDS)]))
    raise ValueError(f"unknown prompt kind {kind!r} (one of {', '.join(KINDS)})")


def parse_rounds(text: str) -> list[tuple[str, int, int, int | None]]:
    """``KIND:STREAMS:TOKENS:MODE,...`` as ``[(kind, streams, tokens, mode or None)]``. ValueError on a bad round."""
    rounds = []
    for spec in text.split(","):
        parts = spec.strip().split(":")
        if len(parts) != 4:
            raise ValueError(f"a round is KIND:STREAMS:TOKENS:MODE, got {spec!r}")
        kind, n, ntok, mode = parts
        if kind not in KINDS:
            raise ValueError(f"unknown prompt kind {kind!r} in {spec!r} (one of {', '.join(KINDS)})")
        if not (n.isdigit() and ntok.isdigit() and int(n) >= 1 and int(ntok) >= 2):
            raise ValueError(f"STREAMS is at least 1 and TOKENS at least 2 in {spec!r}")
        if mode != "-" and mode not in ("0", "1", "2"):
            raise ValueError(f"MODE is -, 0, 1 or 2 in {spec!r}")
        rounds.append((kind, int(n), int(ntok), None if mode == "-" else int(mode)))
    return rounds


def request_body(model: str, prompt: str, ntok: int, mode: int | None, ignore_eos: bool = True) -> dict:
    """The chat completion request of one stream. No ``min_tokens`` (see the module docstring); ``ignore_eos``
    False lets the answer end where the model ends it (``ntok`` is then only the cap)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": ntok,
            "ignore_eos": bool(ignore_eos), "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if mode is not None:
        body["vllm_xargs"] = {"spec_lookup": mode}
    return body


def read_counters(host: str, port: int) -> dict | None:
    """The speculative counters of the server's /metrics page (None: no counters, or the page did not answer)."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        conn.request("GET", "/metrics")
        text = conn.getresponse().read().decode("utf-8", "replace")
        conn.close()
    except (OSError, http.client.HTTPException):
        return None
    return parse_spec_counters(text)


def consume_stream(lines, now=time.perf_counter) -> dict:
    """Read the ``data:`` lines of one streamed chat completion: ``{"times", "tokens", "text", "usage"}`` with one
    entry of ``times`` and ``tokens`` (the completion tokens so far, from the chunk's usage; None without it) per
    chunk that carried text."""
    rec = {"times": [], "tokens": [], "text": [], "usage": None}
    for line in lines:
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
        usage = obj.get("usage")
        if usage:
            rec["usage"] = usage
        choices = obj.get("choices") or []
        if choices and (choices[0].get("delta") or {}).get("content"):
            rec["times"].append(now())
            rec["tokens"].append((usage or {}).get("completion_tokens"))
            rec["text"].append(choices[0]["delta"]["content"])
    return rec


def step_sizes(tokens: list[int | None]) -> list[int]:
    """Tokens per chunk from the cumulative completion tokens of each chunk ([] when a chunk had no usage)."""
    if not tokens or any(t is None for t in tokens):
        return []
    return [b - a for a, b in zip([0] + tokens[:-1], tokens)]


def one_stream(args, body: dict, out: list, idx: int, barrier: threading.Barrier) -> None:
    rec = {"status": None, "t0": None, "times": [], "tokens": [], "text": [], "usage": None, "err": None}
    try:
        barrier.wait()
        conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
        rec["t0"] = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["status"] = resp.status
        if resp.status != 200:
            rec["err"] = resp.read()[:500].decode("utf-8", "replace")
        else:
            rec.update(consume_stream(iter(resp.readline, b"")))
        conn.close()
    except Exception as e:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(e)
    out[idx] = rec


def summarize_round(kind: str, n: int, ntok: int, mode: int | None, out: list, wall: float, spec: dict | None) -> dict:
    """The record of one round from its streams' raw records and the counters' summary."""
    per, errors, sizes = [], 0, []
    for r in out:
        if not r or r["err"] or r["status"] != 200 or len(r["times"]) < 2:
            errors += 1
            continue
        usage = r["usage"] or {}
        comp = usage.get("completion_tokens") or len(r["times"])
        span = r["times"][-1] - r["times"][0]
        gaps = sorted(1000.0 * (b - a) for a, b in zip(r["times"], r["times"][1:]))
        first_tokens = r["tokens"][0] or 1
        sizes.extend(step_sizes(r["tokens"]))
        per.append({"ttft_s": r["times"][0] - r["t0"],
                    "ms_per_token": 1000.0 * span / max(1, comp - first_tokens),
                    "completion_tokens": comp, "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "chunks": len(r["times"]), "gap_ms_p50": statistics.median(gaps),
                    "gap_ms_p90": gaps[min(len(gaps) - 1, int(0.9 * len(gaps)))], "gap_ms_max": gaps[-1],
                    "tokens_per_chunk": comp / len(r["times"]), "first": r["times"][0], "last": r["times"][-1]})
    res = {"kind": kind, "streams": n, "tokens": ntok, "mode": mode, "ok": len(per), "errors": errors,
           "wall_s": wall, "spec": spec,
           "texts": ["".join(r["text"]) if r else None for r in out],
           "error_samples": [r["err"] for r in out if r and r["err"]][:3]}
    if per:
        res["ms_per_token_p50"] = statistics.median(p["ms_per_token"] for p in per)
        res["ms_per_token_max"] = max(p["ms_per_token"] for p in per)
        res["gap_ms_p50"] = statistics.median(p["gap_ms_p50"] for p in per)
        res["gap_ms_p90"] = statistics.median(p["gap_ms_p90"] for p in per)
        res["gap_ms_max"] = max(p["gap_ms_max"] for p in per)
        res["tokens_per_chunk_p50"] = statistics.median(p["tokens_per_chunk"] for p in per)
        res["ttft_p50_s"] = statistics.median(p["ttft_s"] for p in per)
        prompts = [p["prompt_tokens"] for p in per if p["prompt_tokens"] is not None]
        res["prompt_tokens"] = [min(prompts), statistics.median(prompts), max(prompts)] if prompts else None
        cached = [p["cached_tokens"] for p in per if p["cached_tokens"] is not None]
        res["cached_tokens_p50"] = statistics.median(cached) if cached else None
        res["completion_tokens"] = sum(p["completion_tokens"] for p in per)
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = res["completion_tokens"] / span_all if span_all > 0 else None
        res["all_decoding_window_s"] = max(0.0, min(p["last"] for p in per) - max(p["first"] for p in per))
        if sizes:
            res["step_sizes"] = {str(s): sizes.count(s) for s in sorted(set(sizes))}
        al = (spec or {}).get("acceptance_length")
        # steady figures: the median gap between chunks is one engine step; a step commits `al` tokens per request
        res["ms_per_step"] = res["gap_ms_p50"]
        res["steady_ms_per_token"] = res["gap_ms_p50"] / (al or res["tokens_per_chunk_p50"])
    return res


def run_round(args, kind: str, n: int, ntok: int, mode: int | None, files: dict) -> dict:
    prompts = [prompt_for(kind, i, files.get(kind, [])) for i in range(n)]
    before = read_counters(args.host, args.port)
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream, args=(args, request_body(args.model, prompts[i], ntok, mode,
                                                                           ignore_eos=kind != "editf"), out,
                                                         i, barrier), daemon=True) for i in range(n)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.timeout + 60)
    wall = time.perf_counter() - t_start
    spec = spec_summary(counters_delta(before, read_counters(args.host, args.port)))
    res = summarize_round(kind, n, ntok, mode, out, wall, spec)
    res["prompt_sha"] = [hashlib.sha256(p.encode("utf-8")).hexdigest()[:12] for p in prompts]
    return res


def round_line(label: str, tag: str, m: dict) -> str:
    """The one-line summary of a measured round for the job log."""
    if not m.get("ok"):
        return "%s %s %-5s N=%-3d mode=%s FAILED %s" % (label, tag, m["kind"], m["streams"], m["mode"],
                                                         m.get("error_samples"))
    s = m.get("spec") or {}
    al = s.get("acceptance_length")
    rates = ",".join("%.2f" % r for r in s.get("position_rates") or [] if r is not None)
    prompt = m.get("prompt_tokens") or [0, 0, 0]
    return ("%s %s %-5s N=%-3d mode=%s ok=%d err=%d  tokens/step=%s (pos %s)  ms/step=%.1f (p90 %.1f max %.0f)  "
            "ms/token steady=%.2f span p50=%.2f max=%.2f  agg=%.1f tok/s  tok/chunk=%.2f  ttft p50=%.2fs  "
            "prompt %d..%d cached p50 %s  wall=%.1fs" % (
                label, tag, m["kind"], m["streams"], m["mode"], m["ok"], m["errors"], ("%.2f" % al) if al else "-",
                rates or "-", m["ms_per_step"], m["gap_ms_p90"], m["gap_ms_max"], m["steady_ms_per_token"],
                m["ms_per_token_p50"], m["ms_per_token_max"], m.get("aggregate_tok_s") or 0.0,
                m["tokens_per_chunk_p50"], m["ttft_p50_s"], prompt[0], prompt[2], m.get("cached_tokens_p50"),
                m["wall_s"]))


def compare_texts(runs: list[dict]) -> list[str]:
    """Lines comparing the stream texts of probe records: per (kind, streams) every round against the first one
    of that shape (identical streams, the shortest common prefix in characters). Rounds of fewer streams compare
    their streams with the same stream indices of the first round of the kind that has at least as many."""
    by_kind: dict[str, list] = {}
    for run in runs:
        for r in run.get("rounds", []):
            if r.get("texts") and r.get("ok"):
                by_kind.setdefault(r["kind"], []).append(("%s/%s/N%d" % (run.get("label"), r.get("mode"),
                                                                         r["streams"]), r))
    lines = []
    for kind, entries in sorted(by_kind.items()):
        ref_name, ref = max(entries, key=lambda e: e[1]["streams"])
        for name, r in entries:
            if r is ref:
                continue
            same = total = 0
            shortest = None
            for i, text in enumerate(r["texts"]):
                if text is None or i >= len(ref["texts"]) or ref["texts"][i] is None:
                    continue
                want = ref["texts"][i]
                limit = min(len(want), len(text))  # rounds may differ in length: compare the common part
                common = next((j for j in range(limit) if want[j] != text[j]), limit)
                total += 1
                same += int(common == limit)
                shortest = common if shortest is None else min(shortest, common)
            lines.append("%-5s %-28s vs %-28s identical %d/%d, shortest common prefix %s chars" % (
                kind, ref_name, name, same, total, shortest))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--rounds", default="")
    ap.add_argument("--warm-tokens", type=int, default=-1,
                    help="tokens of the warm round before the first round of each (kind, streams): -1 (default) the "
                    "round's own length for edit and prose and 48 for editl and editf, 0 no warm round")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--tree", default=TREE, help="the repository tree whose files the edit prompts hold")
    ap.add_argument("--out", default="lookup_probe.json")
    ap.add_argument("--compare", nargs="+", metavar="PROBE.json", help="compare the texts of probe files and exit")
    ap.add_argument("--list-prompts", action="store_true")
    args = ap.parse_args(argv)
    if args.compare:
        runs = []
        for path in args.compare:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        print("\n".join(compare_texts(runs)))
        return 0
    files = {"edit": load_files(args.tree, SMALL_FILES), "editl": load_files(args.tree, LARGE_FILES)}
    files["editf"] = files["edit"]
    if args.list_prompts:
        for kind in KINDS:
            for i in range(len(files.get(kind) or [0])):
                print(kind, i, (files[kind][i][0] if kind in files else "-"), len(prompt_for(kind, i, files.get(
                    kind, []))), "chars")
        return 0
    try:
        rounds = parse_rounds(args.rounds)
    except ValueError as e:
        print(f"lookup_probe: {e}", file=sys.stderr)
        return 2
    for kind in {r[0] for r in rounds}:
        if kind in files and not files[kind]:
            print(f"lookup_probe: none of the {kind} files exists under {args.tree}", file=sys.stderr)
            return 2
    results = {"label": args.label, "tree": args.tree, "files": {k: [f for f, _ in v] for k, v in files.items()},
               "rounds": [], "warm": []}
    warmed = set()

    def save():
        with open(args.out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        os.replace(args.out + ".tmp", args.out)

    failed = 0
    for kind, n, ntok, mode in rounds:
        if args.warm_tokens != 0 and (kind, n) not in warmed:
            if args.warm_tokens >= 0:
                warm_tokens = args.warm_tokens
            else:
                warm_tokens = ntok if kind in FULL_WARM else min(ntok, SHORT_WARM)
            w = run_round(args, kind, n, warm_tokens, mode, files)
            warmed.add((kind, n))
            print(round_line(args.label, "warm", w), flush=True)
            w.pop("texts", None)
            results["warm"].append(w)
            save()
            if not w["ok"]:
                # the server refuses or fails this shape: its measured rounds would fail the same way
                results["rounds"].append({"kind": kind, "streams": n, "tokens": ntok, "mode": mode, "ok": 0,
                                          "errors": n, "warm_failed": w["error_samples"]})
                failed += 1
                save()
                continue
        m = run_round(args, kind, n, ntok, mode, files)
        results["rounds"].append(m)
        save()
        print(round_line(args.label, "MEAS", m), flush=True)
        failed += int(not m["ok"])
    for line in compare_texts([results]):
        print(args.label, "TEXT", line, flush=True)
    save()
    return 3 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
