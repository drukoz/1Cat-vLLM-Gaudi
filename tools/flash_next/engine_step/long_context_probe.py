#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engine-step card probe: an A/B of the per-step flags on ONE server, at a short and at a long context.

The server starts with ``VLLM_HPU_RUNTIME_SWITCH_FILE`` pointing at ``--switch-file``; before a round the probe
replaces that file with the flags of the round's arm, so the same server and cards decode the same prompts with
a path off and on, in turn (base, arm, base, arm: the second pair bounds the drift). Arms:

    base  VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0  VLLM_HPU_FUSED_GREEDY_LOGITS=0
    both  1 1          i01  1 0          i05  0 1

Rounds (``N`` = ``--streams``, every stream exactly ``--tokens`` tokens: ``max_tokens`` with ``ignore_eos``, no
``min_tokens``, temperature 0):

* ``text1``  one stream, a prompt in words: the only text that is reproducible on the real model (two rounds of
  the same 32 prompts on one server already differ, coordinator 19:35 UTC on job 00021), so the arms' texts
  are compared here and must be equal.
* ``text``   N prompts in words, each asking for its own run of numbers: what the streams answer is printed
  and scored (does a stream go on with its own numbers?), so a round whose later streams decode garbage is
  seen (card rule 9).
* ``same``   N streams with the ``text1`` prompt: how many different texts come back, and how many equal the
  one-stream text. More than one text means a row's result depends on its place in the batch or on the rows
  decoding next to it (each row joins at another step).
* ``short``  N prompts of 64 random token ids.
* ``long``   N prompts that share one exact prefix of ``--prefix`` token ids (a seed request prefills it once,
  the prefix cache then serves it) plus 64 ids of their own: N streams decode at about ``--prefix`` tokens of
  context each without N long prefills. The prefix length is a multiple of the server's prefill chunk, so the
  whole prefix is reused; ``cached`` in the output says what the server reused.
* ``mintok`` one more round with ``min_tokens`` (the first arm only): the cost of the mask in the in-graph
  greedy region.

Per round: per stream the median gap between streamed chunks (steady ms per token: a decode step, compile pauses
excluded), the span per token (pauses included), pauses over 0.5 s, TTFT; the round's medians and total tokens
per second; every stream's text. Rounds of one kind send the same prompts, so two rounds' texts must be equal
stream by stream; ``same text`` counts the streams that are (base against base gives the noise of the server
itself). The result file is rewritten after every round. Stdlib only.
"""
import argparse
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time

ARMS = {
    "base": {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": 0, "VLLM_HPU_FUSED_GREEDY_LOGITS": 0},
    "both": {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": 1, "VLLM_HPU_FUSED_GREEDY_LOGITS": 1},
    "i01": {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": 1, "VLLM_HPU_FUSED_GREEDY_LOGITS": 0},
    "i05": {"VLLM_HPU_FLASH_NEXT_DECODE_METADATA": 0, "VLLM_HPU_FUSED_GREEDY_LOGITS": 1},
}
SWITCH_LINE = "Runtime switches from"
REFUSAL_LINE = "is off for this server"


def one_stream(host, port, model, prompt, ntok, out, idx, barrier, timeout_s, min_tokens):
    request = {"model": model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
    if min_tokens:
        request["min_tokens"] = ntok
    body = json.dumps(request)
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


def run_round(args, label, prompts, ntok, min_tokens=False):
    n = len(prompts)
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream, args=(args.host, args.port, args.model, prompts[i], ntok, out, i,
                                                         barrier, args.timeout, min_tokens), daemon=True)
               for i in range(n)]
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
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:])]
        tpc = comp / len(r["times"])
        per.append({"steady_ms": 1000.0 * statistics.median(gaps) / max(tpc, 1e-9),
                    "gap_p90_ms": 1000.0 * sorted(gaps)[int(0.9 * (len(gaps) - 1))],
                    "pause_s": sum(g for g in gaps if g > 0.5), "ttft_s": r["times"][0] - r["t0"],
                    "span_ms": 1000.0 * (r["times"][-1] - r["times"][0]) / max(1, comp - 1), "tokens": comp,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "first": r["times"][0], "last": r["times"][-1],
                    "gaps_ms": [round(1000.0 * g, 2) for g in gaps]})
    res = {"label": label, "streams": n, "tokens": ntok, "min_tokens": bool(min_tokens), "ok": len(per),
           "errors": errors, "wall_s": wall, "texts": texts}
    if per:
        res["steady_ms_p50"] = statistics.median(p["steady_ms"] for p in per)
        res["steady_ms_min"] = min(p["steady_ms"] for p in per)
        res["steady_ms_max"] = max(p["steady_ms"] for p in per)
        res["gap_p90_ms_p50"] = statistics.median(p["gap_p90_ms"] for p in per)
        res["span_ms_p50"] = statistics.median(p["span_ms"] for p in per)
        res["pause_s_max"] = max(p["pause_s"] for p in per)
        res["ttft_p50_s"] = statistics.median(p["ttft_s"] for p in per)
        res["prompt_tokens_p50"] = statistics.median(p["prompt_tokens"] or 0 for p in per)
        res["cached_tokens_p50"] = statistics.median(p["cached_tokens"] or 0 for p in per)
        all_gaps = sorted(g for p in per for g in p["gaps_ms"])
        res["all_gaps_ms_p50"] = all_gaps[len(all_gaps) // 2]
        res["all_gaps_ms_p10"] = all_gaps[len(all_gaps) // 10]
        res["all_gaps_ms_p90"] = all_gaps[(9 * len(all_gaps)) // 10]
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = sum(p["tokens"] for p in per) / span_all if span_all > 0 else None
        res["per_stream"] = per
    errs = [r["err"] for r in out if r and r["err"]][:3]
    if errs:
        res["error_samples"] = errs
    print("%-24s N=%-3d ok=%d err=%d wall=%6.1fs steady ms/tok p50=%s [min %s max %s] gaps p10=%s p50=%s p90=%s "
          "span ms/tok p50=%s pauses max=%ss ttft p50=%ss prompt~%s cached~%s total=%s tok/s" % (
              label, n, res["ok"], errors, wall, _f(res.get("steady_ms_p50")), _f(res.get("steady_ms_min")),
              _f(res.get("steady_ms_max")), _f(res.get("all_gaps_ms_p10")), _f(res.get("all_gaps_ms_p50")),
              _f(res.get("all_gaps_ms_p90")), _f(res.get("span_ms_p50")), _f(res.get("pause_s_max")),
              _f(res.get("ttft_p50_s")), res.get("prompt_tokens_p50"), res.get("cached_tokens_p50"),
              _f(res.get("aggregate_tok_s"))), flush=True)
    return res


def _f(x):
    return "-" if x is None else "%.2f" % x


def same_texts(a, b):
    """(streams with equal text, streams compared, mean share of the text that is a common prefix)."""
    equal, compared, share = 0, 0, 0.0
    for x, y in zip(a or [], b or []):
        if x is None or y is None:
            continue
        compared += 1
        if x == y:
            equal += 1
            share += 1.0
            continue
        common = 0
        for cx, cy in zip(x, y):
            if cx != cy:
                break
            common += 1
        share += common / max(len(x), len(y), 1)
    return equal, compared, (share / compared if compared else None)


def log_count(path, needle):
    if not path:
        return None
    try:
        with open(path, errors="replace") as handle:
            return sum(1 for line in handle if needle in line)
    except OSError:
        return None


class Probe:

    def __init__(self, args):
        self.args = args
        self.results = {"args": vars(args), "arms": ARMS, "rounds": [], "switches": [], "notes": []}
        self.arm = None
        self.pending = None
        self.started = time.time()
        self.switch_ok = True

    def note(self, text):
        print("NOTE " + text, flush=True)
        self.results["notes"].append(text)
        self.save()

    def save(self):
        with open(self.args.out + ".tmp", "w") as handle:
            json.dump(self.results, handle, indent=1)
        os.replace(self.args.out + ".tmp", self.args.out)

    def elapsed(self):
        return time.time() - self.started

    def over_budget(self, what):
        if self.elapsed() <= self.args.budget:
            return False
        self.note("budget of %d s used up after %.0f s: %s left out" % (self.args.budget, self.elapsed(), what))
        return True

    def set_arm(self, arm):
        """Replace the switch file with the arm's flags (a rename) while no request runs."""
        if arm == self.arm:
            return
        flags = ARMS[arm]
        if self.args.switch_file:
            time.sleep(self.args.settle)  # the last round's requests are done on every worker
            before = log_count(self.args.server_log, SWITCH_LINE)
            staging = self.args.switch_file + ".new"
            with open(staging, "w") as handle:
                handle.write("# arm %s\n" % arm + "".join("%s=%d\n" % kv for kv in sorted(flags.items())))
            os.replace(staging, self.args.switch_file)
            time.sleep(self.args.settle)
            self.pending = (arm, before)
        elif arm != "base":
            raise SystemExit("arm %s needs --switch-file" % arm)
        self.arm = arm
        print("ARM %s: %s" % (arm, " ".join("%s=%d" % kv for kv in sorted(flags.items()))), flush=True)

    def check_switch(self):
        """After the first round of an arm: did the workers log the switch, did the server refuse a flag?"""
        if not self.pending or not self.args.server_log:
            return
        arm, before = self.pending
        self.pending = None
        after = log_count(self.args.server_log, SWITCH_LINE)
        refused = log_count(self.args.server_log, REFUSAL_LINE)
        seen = None if after is None or before is None else after - before
        self.results["switches"].append({"arm": arm, "worker_lines": seen, "refusals_in_log": refused})
        print("SWITCH to %s: %s worker line(s) in the server log, %s refusal line(s) so far" % (arm, seen, refused),
              flush=True)
        if seen is not None and seen <= 0:
            self.switch_ok = False
            self.note("the server did not log the switch to arm %s: its rounds are NOT that arm" % arm)
        if refused:
            self.note("the server log has %d '%s' line(s): an arm with that flag ran the old path" %
                      (refused, REFUSAL_LINE))

    def run(self, kind, arm, prompts, ntok, tag="", min_tokens=False):
        self.set_arm(arm)
        label = "%s-%d%s:%s" % (kind, len(prompts), tag, arm)
        res = run_round(self.args, label, prompts, ntok, min_tokens)
        res.update(kind=kind, arm=arm, tag=tag, at_s=round(self.elapsed(), 1))
        self.results["rounds"].append(res)
        self.check_switch()
        self.save()
        return res

    def compare(self, kind, first, second):
        """Texts of two rounds of one kind (same prompts)."""
        equal, compared, share = same_texts(first.get("texts"), second.get("texts"))
        line = {"kind": kind, "a": first["label"], "b": second["label"], "equal": equal, "compared": compared,
                "common_prefix_share": share}
        self.results.setdefault("text_compare", []).append(line)
        print("TEXTS %-26s vs %-26s same text in %d of %d streams (common prefix %s)" % (
            first["label"], second["label"], equal, compared, "-" if share is None else "%.3f" % share), flush=True)
        self.save()


def summary(probe):
    """Per kind and arm: the measured rounds' steady ms per token, and each arm against base."""
    rows = {}
    for res in probe.results["rounds"]:
        if res.get("tag") != "" or res.get("steady_ms_p50") is None or res.get("min_tokens"):
            continue
        if res["kind"] not in ("short", "long"):
            continue
        rows.setdefault((res["kind"], res["arm"]), []).append(res)
    print("\nSUMMARY steady ms per token (median gap between chunks), measured rounds only")
    table = {}
    for (kind, arm), rounds in sorted(rows.items()):
        values = [r["steady_ms_p50"] for r in rounds]
        gaps = [r["all_gaps_ms_p50"] for r in rounds]
        table[(kind, arm)] = statistics.mean(values)
        print("  %-6s %-5s rounds %d: %s (all-gaps p50 %s), total tok/s %s" % (
            kind, arm, len(values), " ".join("%.2f" % v for v in values), " ".join("%.2f" % v for v in gaps),
            " ".join(_f(r.get("aggregate_tok_s")) for r in rounds)))
    out = []
    for (kind, arm), value in sorted(table.items()):
        base = table.get((kind, "base"))
        if arm == "base" or base is None:
            continue
        line = {"kind": kind, "arm": arm, "base_ms": base, "arm_ms": value, "delta_ms": value - base,
                "delta_pct": 100.0 * (value - base) / base}
        out.append(line)
        print("  %-6s %-5s against base: %.2f -> %.2f ms per token (%+.2f ms, %+.1f %%)" % (
            kind, arm, base, value, value - base, line["delta_pct"]))
    probe.results["summary"] = out
    probe.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prefix", type=int, default=120832, help="shared prefix tokens of the long rounds "
                    "(a multiple of the server's prefill chunk, 2048, so the prefix cache serves all of it)")
    ap.add_argument("--suffix", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=1500.0)
    ap.add_argument("--out", default="probe.json")
    ap.add_argument("--switch-file", default="", help="the server's VLLM_HPU_RUNTIME_SWITCH_FILE")
    ap.add_argument("--server-log", default="", help="to check that the workers logged each switch")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds without requests around a switch")
    ap.add_argument("--arms", default="both,i01,i05", help="arms measured against base; the first one twice")
    ap.add_argument("--budget", type=float, default=1200.0, help="seconds after which optional rounds are left out")
    ap.add_argument("--no-long", action="store_true")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        if arm not in ARMS or arm == "base":
            raise SystemExit("unknown arm %r (one of both, i01, i05)" % arm)
    main_arm, other_arms = arms[0], arms[1:]
    rng = random.Random(20261120)
    ids = lambda count: [rng.randrange(1000, 150000) for _ in range(count)]  # noqa: E731
    n, ntok = args.streams, args.tokens
    probe = Probe(args)
    measured = {}  # (kind, arm) -> the measured rounds

    def measure(kind, arm, prompts, warm=None):
        if warm:
            probe.run(kind, arm, prompts, warm, tag="-warm")
        res = probe.run(kind, arm, prompts, ntok)
        measured.setdefault((kind, arm), []).append(res)
        return res

    # What the streams answer, in words: garbage in the later streams of a batch is seen here.
    starts = [10 * i + 1 for i in range(n)]
    text_prompts = ["The numbers from %d to %d, separated by commas: %d, %d, %d," % (a, a + 29, a, a + 1, a + 2)
                    for a in starts]
    one_prompt = "Here is a short story about a lighthouse keeper and a storm.\n\nThe keeper"
    short_prompts = [ids(64) for _ in range(n)]
    prefix = ids(args.prefix)
    long_prompts = [prefix + ids(args.suffix) for _ in range(n)]
    shown = sorted({0, min(1, n - 1), n // 2, n - 1})

    def words(arm):
        """The arm's rounds in words: one stream, N streams with their own numbers, N streams with one prompt."""
        one = probe.run("text1", arm, [one_prompt], 48)
        print("TEXT %-5s 1 stream: %r" % (arm, (one["texts"][0] or "")[:160]), flush=True)
        many = probe.run("text", arm, text_prompts, 24)
        for i in shown:
            print("TEXT %-5s stream %2d of %d: %r" % (arm, i, n, (many["texts"][i] or "")[:70]), flush=True)
        on_task = sum(1 for a, text in zip(starts, many["texts"])
                      if text is not None and text.replace(" ", "").startswith("%d,%d," % (a + 3, a + 4)))
        many["on_task"] = on_task
        print("TASK %-5s %d of %d streams go on with their own numbers" % (arm, on_task, n), flush=True)
        same = probe.run("same", arm, [one_prompt] * n, 48)
        texts = [t for t in same["texts"] if t is not None]
        same["distinct_texts"] = len(set(texts))
        same["equal_to_one_stream"] = sum(1 for t in texts if t == one["texts"][0])
        print("SAME %-5s %d streams with one prompt gave %d different text(s); %d equal the one-stream text" % (
            arm, len(texts), same["distinct_texts"], same["equal_to_one_stream"]), flush=True)
        probe.save()
        return one

    # --- short context: base, first arm, base, first arm; the first round of each arm warms its graphs
    probe.run("short", "base", [short_prompts[0]], 24, tag="-warm1")
    one_base = words("base")
    measure("short", "base", short_prompts, warm=24)
    probe.run("short", main_arm, [short_prompts[0]], 24, tag="-warm1")
    one_arm = words(main_arm)
    probe.compare("text1", one_base, one_arm)
    measure("short", main_arm, short_prompts, warm=24)
    measure("short", "base", short_prompts)
    measure("short", main_arm, short_prompts)
    probe.compare("short", measured[("short", "base")][0], measured[("short", "base")][1])
    probe.compare("short", measured[("short", "base")][0], measured[("short", main_arm)][0])
    probe.compare("short", measured[("short", "base")][1], measured[("short", main_arm)][1])

    # --- long context over the shared prefix
    long_ok = not args.no_long
    if long_ok:
        started = time.time()
        seed = probe.run("seed", "base", [long_prompts[0]], 2, tag="-prefix")
        print("seed prefill of %d tokens: %.1f s" % (args.prefix, time.time() - started), flush=True)
        check = probe.run("long", "base", [long_prompts[1]], 8, tag="-cachecheck")
        cached = check.get("cached_tokens_p50") or 0
        if seed["errors"] or check["errors"]:
            long_ok = False
            probe.note("the long prompt failed (%s): no long rounds" % (seed.get("error_samples") or
                                                                         check.get("error_samples")))
        elif cached < 0.9 * args.prefix and (check.get("ttft_p50_s") or 0) > 20.0:
            long_ok = False
            probe.note("the prefix cache served %s of %d prefix tokens (TTFT %.1f s): %d streams would prefill "
                       "the prefix each, no long rounds" % (cached, args.prefix, check["ttft_p50_s"], n))
    if long_ok:
        measure("long", "base", long_prompts, warm=16)
        measure("long", main_arm, long_prompts, warm=16)
        measure("long", "base", long_prompts)
        measure("long", main_arm, long_prompts)
        probe.compare("long", measured[("long", "base")][0], measured[("long", "base")][1])
        probe.compare("long", measured[("long", "base")][0], measured[("long", main_arm)][0])
        probe.compare("long", measured[("long", "base")][1], measured[("long", main_arm)][1])

    # --- the other arms (which flag carries the effect), then the min_tokens mask; left out when time is short
    for arm in other_arms:
        if long_ok and not probe.over_budget("long rounds of arm " + arm):
            measure("long", arm, long_prompts, warm=16)
            probe.compare("long", measured[("long", "base")][-1], measured[("long", arm)][0])
        if not probe.over_budget("short rounds of arm " + arm):
            measure("short", arm, short_prompts, warm=24)
            probe.compare("short", measured[("short", "base")][-1], measured[("short", arm)][0])
    if not probe.over_budget("the min_tokens round"):
        kind, prompts = ("long", long_prompts) if long_ok else ("short", short_prompts)
        mintok = probe.run(kind, main_arm, prompts, ntok, tag="-mintok", min_tokens=True)
        last = measured[(kind, main_arm)][-1]
        if mintok.get("steady_ms_p50") is not None and last.get("steady_ms_p50") is not None:
            print("MINTOK %s arm %s: %.2f ms per token with min_tokens, %.2f without" % (
                kind, main_arm, mintok["steady_ms_p50"], last["steady_ms_p50"]), flush=True)
    probe.set_arm("base")
    summary(probe)
    bad = sum(1 for r in probe.results["rounds"] if r["errors"])
    if bad or not probe.switch_ok:
        print("PROBE: %d round(s) with errors, switches ok: %s" % (bad, probe.switch_ok), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
