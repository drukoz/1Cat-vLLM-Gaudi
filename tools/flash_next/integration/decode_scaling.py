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

--switch-file PATH with --switch-arm NAME=LINES (repeatable; LINES are NAME=VALUE pairs separated by ';') runs an A/B
on ONE server (track mtp-draft-graph): the server reads PATH as its VLLM_HPU_RUNTIME_SWITCH_FILE. Per level, one warm
round per arm, then the measured rounds with the arms in turn (A, B, A, B, ...), the file replaced with a rename
while no request runs. Every greedy round carries "arm" and "switch_lines" (the workers' "Runtime switches from"
lines that came with it in --server-log: 0 means the round may not be that arm, a SWITCH line says so).
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


_SYNC = {"count": 0}


def sync_point(args):
    """--barrier-dir: every round (warm or measured) waits until --barrier-n probes have reached the same round
    (files <dir>/<tag>.<round>.<name>), so two servers of one job decode at the same time; at most
    --barrier-timeout seconds (a dead partner must not stop this probe)."""
    if not getattr(args, "barrier_dir", ""):
        return
    import glob
    key = "%s.%03d" % (args.barrier_tag or "probe", _SYNC["count"])
    _SYNC["count"] += 1
    os.makedirs(args.barrier_dir, exist_ok=True)
    open(os.path.join(args.barrier_dir, "%s.%s" % (key, args.barrier_name)), "w").close()
    t0 = time.time()
    while len(glob.glob(os.path.join(args.barrier_dir, key + ".*"))) < args.barrier_n:
        if time.time() - t0 > args.barrier_timeout:
            print("BARRIER %s: timeout after %.0f s, going on alone" % (key, time.time() - t0), flush=True)
            return
        time.sleep(0.2)
    waited = time.time() - t0
    if waited > 5:
        print("BARRIER %s: waited %.1f s" % (key, waited), flush=True)


def run_round(args, n, ntok, ctx_tokens=0, mintok=None, sampled=False, salt=""):
    sync_point(args)
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
    ap.add_argument("--barrier-dir", default="", help="align every round with other probes (see sync_point)")
    ap.add_argument("--barrier-n", type=int, default=2)
    ap.add_argument("--barrier-name", default="probe")
    ap.add_argument("--barrier-tag", default="")
    ap.add_argument("--barrier-timeout", type=float, default=900.0)
    ap.add_argument("--rounds", type=int, default=0,
                    help="measured rounds at EVERY level, greedy and sampled, after one warm round (default 0: one "
                         "per level and --repeat-top at the top level, as the coordinator's probe)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="scaling.json")
    ap.add_argument("--switch-file", default="", help="the server's VLLM_HPU_RUNTIME_SWITCH_FILE (with --switch-arm)")
    ap.add_argument("--switch-arm", action="append", default=[], metavar="NAME=LINES",
                    help="an arm of the A/B on one server: NAME=FLAG=VALUE[;FLAG=VALUE...] (repeatable, in order)")
    ap.add_argument("--server-log", default="", help="the server's log, to check that the workers took each switch")
    ap.add_argument("--switch-settle", type=float, default=2.0, help="seconds without requests around a switch")
    ap.add_argument("--require-log", action="append", default=[], metavar="RE",
                    help="(repeatable, with --server-log) a line of the server's log must match the regular expression "
                         "RE after the first warm round, else the probe stops with exit code 8 (a flag that silently "
                         "fell back, e.g. the dense MoE form)")
    args = ap.parse_args()
    arms = []
    for spec in args.switch_arm:
        name, sep, lines = spec.partition("=")
        if not sep or not name or "=" not in lines:
            raise SystemExit("--switch-arm takes NAME=FLAG=VALUE[;FLAG=VALUE...], got %r" % spec)
        arms.append((name, lines))
    if arms and not args.switch_file:
        raise SystemExit("--switch-arm needs --switch-file (the server's VLLM_HPU_RUNTIME_SWITCH_FILE)")
    levels = [int(x) for x in args.levels.split(",") if x]
    sampled_levels = [int(x) for x in args.sampled_levels.split(",") if x] or levels[-1:]
    results = {"levels": [], "ctx": [], "mintok": [], "sampled": []}

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    required = {"checked": not args.require_log, "missing": []}

    def require_logs():
        """After the first warm round: every --require-log pattern must match a line of --server-log. False: stop."""
        if required["checked"]:
            return True
        required["checked"] = True
        import re
        try:
            with open(args.server_log, errors="replace") as f:
                text = f.read()
        except OSError as e:
            text = ""
            print("REQUIRE-LOG: cannot read the server log %r: %r" % (args.server_log, e), flush=True)
        for pattern in args.require_log:
            m = re.search(pattern, text)
            if m:
                line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end()) if text.find("\n", m.end()) >= 0 else None]
                print("REQUIRE-LOG ok (%s): %s" % (pattern, line.strip()[:300]), flush=True)
            else:
                required["missing"].append(pattern)
                print("REQUIRE-LOG MISSING: no line of %s matches %r after the first warm round: stopping (exit 8)" % (
                    args.server_log, pattern), flush=True)
        results["require_log"] = {"patterns": args.require_log, "missing": required["missing"]}
        save()
        return not required["missing"]

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

    current = {"arm": None}

    def switch_lines():
        """How many "Runtime switches from" lines the server's log holds (-1: no log to read)."""
        if not args.server_log:
            return -1
        try:
            with open(args.server_log, errors="replace") as f:
                return sum(1 for line in f if "Runtime switches from" in line)
        except OSError:
            return -1

    def switch_to(name, lines):
        """Replace the switch file with the arm's flags (a rename) while no request runs; the log count before
        (None when the arm is in force already: no switch, nothing to check)."""
        if current["arm"] == name:
            return None
        before = switch_lines()
        time.sleep(args.switch_settle)
        staging = args.switch_file + ".new"
        with open(staging, "w") as f:
            f.write("".join(pair.strip() + "\n" for pair in lines.split(";") if pair.strip()))
        os.replace(staging, args.switch_file)
        time.sleep(args.switch_settle)
        current["arm"] = name
        return before

    def switched(name, before):
        """The log lines the switch to ``name`` gave (after the round's first step); a SWITCH line if none."""
        if before is None:
            return None
        after = switch_lines()
        seen = after - before if before >= 0 and after >= 0 else None
        if seen == 0:
            print("SWITCH arm %s: the workers logged no switch: this round may not be that arm" % name, flush=True)
        return seen

    def level_with_arms(n):
        """One warm round per arm, then the measured rounds with the arms in turn. False: stop the probe."""
        reps = args.rounds if args.rounds > 0 else (args.repeat_top if n == levels[-1] else 1)
        warm = {}
        for name, lines in arms:
            before = switch_to(name, lines)
            w = run_round(args, n, args.warm_tokens)
            w["switch_lines"] = switched(name, before)
            warm[name] = w
            print("warm  N=%-3d arm=%s ok=%d err=%d wall=%.1fs ms/tok p50=%s" % (
                n, name, w["ok"], w["errors"], w["wall_s"], ("%.1f" % w["ms_per_token_p50"]) if w.get("ok") else "-"),
                  flush=True)
            if not w.get("ok"):
                print("stopping: the warm round at N=%d arm %s failed: %s" % (n, name, w.get("error_samples")),
                      flush=True)
                return False
            if not require_logs():
                return False
        for rep in range(reps):
            for name, lines in arms:
                before = switch_to(name, lines)
                m = measured_round(args, n, args.tokens)
                m["rep"] = rep
                m["arm"] = name
                m["switch_lines"] = switched(name, before)
                m["warm"] = {k: warm[name].get(k) for k in ("ok", "errors", "wall_s", "ms_per_token_p50")}
                results["levels"].append(m)
                save()
                if not m.get("ok"):
                    print("MEAS  N=%d arm=%s rep=%d FAILED: %s" % (n, name, rep, m.get("error_samples")), flush=True)
                    return False
                print("MEAS  N=%-3d arm=%s rep=%d ok=%d err=%d  ms/tok p50=%.2f  steady ms/tok p50=%.2f, gap p90=%.1f ms, "
                      "pauses>0.5s max=%.1fs, tok/chunk=%.2f, aggregate=%.1f tok/s" % (
                          n, name, rep, m["ok"], m["errors"], m["ms_per_token_p50"], m["steady_ms_per_token_p50"],
                          m["gap_p90_ms_p50"], m["pause_s_max"], m["tokens_per_chunk_p50"], m["aggregate_tok_s"] or 0),
                      flush=True)
                for k, t in m["samples"].items():
                    print("TEXT  N=%-3d arm=%s rep=%d stream %-2s: %r" % (n, name, rep, k, t[:110]), flush=True)
        return True

    for n in levels:
        if arms:
            if not level_with_arms(n):
                break
            continue
        t = time.time()
        w = run_round(args, n, args.warm_tokens)
        print("warm  N=%-3d ok=%d err=%d wall=%.1fs ms/tok p50=%s" % (
            n, w["ok"], w["errors"], w["wall_s"], ("%.1f" % w["ms_per_token_p50"]) if w.get("ok") else "-"),
              flush=True)
        if w.get("ok") and not require_logs():
            break
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
    if required["missing"]:
        save()
        return 8
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
    if not args.interleave_sampled:
        for n in (sampled_levels if levels and not args.no_sampled_check else []):
            sampled_level(n)
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
    save()
    return 8 if required["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
