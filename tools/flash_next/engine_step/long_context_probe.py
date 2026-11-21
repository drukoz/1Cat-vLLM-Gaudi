#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engine-step card probe: an A/B of the per-step flags on ONE server, at a short and at a long context.

The server starts with ``VLLM_HPU_RUNTIME_SWITCH_FILE`` pointing at ``--switch-file``; before a round the probe
replaces that file with the flags of the round's arm, so the same server and cards decode the same prompts with
a path off and on, in turn (base, arm, base, arm: the second pair bounds the drift). Arms:

    base  VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0  VLLM_HPU_FUSED_GREEDY_LOGITS=0  VLLM_HPU_CANDIDATE_SAMPLING=0
          VLLM_HPU_VOCAB_PARALLEL_LOGITS=0
    both  1 1 0 0      i01  1 0 0 0      i05  0 1 0 0      cand  0 0 1 0      candvp  0 0 1 2

``candvp`` is sampling over candidates with the vocabulary-parallel head (``vocab_parallel.top_k``: each rank's
top 64 and one ``[rows, 128]`` fp32 all-gather per rank instead of the logits all-gather); ``--candvp`` adds its
rounds after the ``cand`` ones (candvp, cand, candvp in turn).

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
* ``sampled`` the short prompts WITHOUT a temperature in the request, as the owner's agents send them: the
  server then samples with the checkpoint's defaults (temperature 1.0, top_k 20, top_p 0.95). Arms ``base``
  (vLLM's sampler: a full sort of the vocabulary per row and step) and ``cand`` (sampling over candidates), in
  turn; the ``text`` prompts sampled too, scored by the same task; a seeded request sent twice under ``cand``
  must repeat itself. Last of all a seeded request under ``base``: whether vLLM's sampler takes a seed on a
  card at all is not known, and nothing is lost if that request ends the server.

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

FLAGS = ("VLLM_HPU_FLASH_NEXT_DECODE_METADATA", "VLLM_HPU_FUSED_GREEDY_LOGITS", "VLLM_HPU_CANDIDATE_SAMPLING",
         "VLLM_HPU_VOCAB_PARALLEL_LOGITS")
ARMS = {name: dict(zip(FLAGS, values)) for name, values in (
    ("base", (0, 0, 0, 0)), ("both", (1, 1, 0, 0)), ("i01", (1, 0, 0, 0)), ("i05", (0, 1, 0, 0)),
    ("cand", (0, 0, 1, 0)), ("candvp", (0, 0, 1, 2)), ("cand2", (0, 0, 2, 0)), ("cand2vp", (0, 0, 2, 2)), ("gc", (0, 0, 0, 0)))}


def _load_spec_probe():
    """tools/flash_next/rental/spec_probe.py (standard library only), loaded by its path; None when missing."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rental", "spec_probe.py")
    try:
        spec = importlib.util.spec_from_file_location("engine_step_spec_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - a probe records what it can
        return None


SPEC_PROBE = _load_spec_probe()
GC_FLAG = "VLLM_HPU_WORKER_GC_FREEZE"
SWITCH_LINE = "Runtime switches from"
REFUSAL_LINE = "is off for this server"


def one_stream(host, port, model, prompt, ntok, out, idx, barrier, timeout_s, min_tokens, sampling):
    request = {"model": model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    if sampling is None:
        request["temperature"] = 0
    else:  # no temperature: the server's default sampling; {"seed": N} makes the request seeded
        request.update(sampling)
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
                if ch:  # one chunk per decode step; its text may be empty (special tokens, held-back bytes)
                    rec["times"].append(time.perf_counter())
                    rec["text"].append(ch[0].get("text") or "")
        conn.close()
    except Exception as e:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(e)
    out[idx] = rec


def run_round(args, label, prompts, ntok, min_tokens=False, sampling=None):
    n = len(prompts)
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [threading.Thread(target=one_stream, args=(args.host, args.port, args.model, prompts[i], ntok, out, i,
                                                         barrier, args.timeout, min_tokens, sampling), daemon=True)
               for i in range(n)]
    t_start = time.perf_counter()
    wall_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.timeout + 60)
    wall = time.perf_counter() - t_start
    per, errors, texts, untimed = [], 0, [], 0
    for r in out:
        # An error is a failed request (an exception, an HTTP status, no chunk at all). A stream that came back
        # in fewer than two chunks (a 2-token request) has no gap to time: counted as untimed, not as an error.
        if not r or r["err"] or r["status"] != 200 or not r["times"]:
            errors += 1
            texts.append(None)
            continue
        texts.append("".join(r["text"]))
        if len(r["times"]) < 2:
            untimed += 1
            continue
        usage = r["usage"] or {}
        comp = usage.get("completion_tokens") or len(r["times"])
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:])]
        tpc = comp / len(r["times"])
        per.append({"steady_ms": 1000.0 * statistics.median(gaps) / max(tpc, 1e-9),
                    "step_ms": 1000.0 * statistics.median(gaps), "tokens_per_chunk": tpc,
                    "gap_p90_ms": 1000.0 * sorted(gaps)[int(0.9 * (len(gaps) - 1))],
                    "pause_s": sum(g for g in gaps if g > 0.5), "ttft_s": r["times"][0] - r["t0"],
                    "span_ms": 1000.0 * (r["times"][-1] - r["times"][0]) / max(1, comp - 1), "tokens": comp,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "first": r["times"][0], "last": r["times"][-1],
                    "gaps_ms": [round(1000.0 * g, 2) for g in gaps]})
    res = {"label": label, "streams": n, "tokens": ntok, "min_tokens": bool(min_tokens), "ok": len(per),
           "errors": errors, "untimed": untimed, "wall_s": wall, "texts": texts,
           "t_start": round(wall_start, 3), "t_end": round(wall_start + wall, 3)}
    if per:
        res["steady_ms_p50"] = statistics.median(p["steady_ms"] for p in per)
        res["step_ms_p50"] = statistics.median(p["step_ms"] for p in per)
        res["tokens_per_chunk_p50"] = statistics.median(p["tokens_per_chunk"] for p in per)
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
        flags = dict(ARMS[arm])
        if getattr(self.args, "gc_arms", False):  # only then: a server without the flag refuses the whole file
            flags[GC_FLAG] = 1 if arm == "gc" else 0
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

    def run(self, kind, arm, prompts, ntok, tag="", min_tokens=False, sampling=None):
        self.set_arm(arm)
        label = "%s-%d%s:%s" % (kind, len(prompts), tag, arm)
        before = self.spec_counters()
        res = run_round(self.args, label, prompts, ntok, min_tokens, sampling)
        after = self.spec_counters()
        res.update(kind=kind, arm=arm, tag=tag, at_s=round(self.elapsed(), 1))
        if before is not None and after is not None:
            spec = SPEC_PROBE.spec_summary(SPEC_PROBE.counters_delta(before, after)) or {}
            res["spec"] = spec
            if spec.get("drafts"):
                rates = ", ".join("%.3f" % r for r in (spec.get("position_rates") or []) if r is not None)
                print("SPEC  %-24s step ms p50=%s tokens per chunk %s acceptance length %s, per position [%s], "
                      "drafts %s" % (label, _f(res.get("step_ms_p50")), _f(res.get("tokens_per_chunk_p50")),
                                     _f(spec.get("acceptance_length")), rates, spec.get("drafts")), flush=True)
        self.results["rounds"].append(res)
        self.check_switch()
        self.save()
        return res

    def spec_counters(self):
        """The server's speculative counters (vllm:spec_decode_*), or None (no speculation, no reader)."""
        if SPEC_PROBE is None or not self.args.spec_arms:
            return None
        try:
            return SPEC_PROBE.read_spec_counters("http://%s:%d" % (self.args.host, self.args.port))
        except Exception:  # noqa: BLE001
            return None

    def compare(self, kind, first, second):
        """Texts of two rounds of one kind (same prompts)."""
        equal, compared, share = same_texts(first.get("texts"), second.get("texts"))
        line = {"kind": kind, "a": first["label"], "b": second["label"], "equal": equal, "compared": compared,
                "common_prefix_share": share}
        self.results.setdefault("text_compare", []).append(line)
        print("TEXTS %-26s vs %-26s same text in %d of %d streams (common prefix %s)" % (
            first["label"], second["label"], equal, compared, "-" if share is None else "%.3f" % share), flush=True)
        self.save()


def _request(args, path, payload=None, timeout=120.0):
    """(status, parsed JSON or text) of one plain request; (None, the exception's text) when it failed."""
    try:
        conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout)
        if payload is None:
            conn.request("GET", path)
        else:
            conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        conn.close()
        try:
            return resp.status, json.loads(body)
        except ValueError:
            return resp.status, body[:500]
    except Exception as e:  # noqa: BLE001 - a probe records every failure
        return None, repr(e)


def _token_id(key):
    """The id of a ``return_tokens_as_token_ids`` key (``token_id:123``), or None."""
    if isinstance(key, str) and key.startswith("token_id:"):
        try:
            return int(key.split(":", 1)[1])
        except ValueError:
            return None
    return None


def logprobs_check(args, prompt):
    """logprobs=5 and prompt_logprobs=2 on a greedy request: HTTP 200, every top-logprob id an id of the
    vocabulary (int, 0 <= id < 2**31; a garbled id is out of that range or the server dies on it), the generated
    token is the first of its position's top logprobs, and /health answers after both."""
    result = {"ok": False}
    status, body = _request(args, "/v1/completions", {
        "model": args.model, "prompt": prompt, "max_tokens": 8, "temperature": 0, "logprobs": 5,
        "return_tokens_as_token_ids": True})
    entry = {"status": status}
    if status == 200 and isinstance(body, dict):
        lp = ((body.get("choices") or [{}])[0].get("logprobs") or {})
        tokens, tops = lp.get("tokens") or [], lp.get("top_logprobs") or []
        ids = [[_token_id(k) for k in (top or {})] for top in tops]
        entry.update(positions=len(tops), top_ids=ids[:3], tokens=tokens[:3])
        entry["ids_valid"] = bool(ids) and all(i is not None and 0 <= i < 2**31 for row in ids for i in row)
        firsts = []
        for token, top in zip(tokens, tops):
            best = max(top.items(), key=lambda kv: kv[1])[0] if top else None
            firsts.append(token == best or (top and top.get(token) == top.get(best)))
        entry["sampled_is_top"] = bool(firsts) and all(firsts)
    else:
        entry["error"] = str(body)[:300]
    result["logprobs"] = entry
    status, body = _request(args, "/v1/completions", {
        "model": args.model, "prompt": prompt, "max_tokens": 2, "temperature": 0, "prompt_logprobs": 2})
    entry = {"status": status}
    if status == 200 and isinstance(body, dict):
        plp = (body.get("choices") or [{}])[0].get("prompt_logprobs") or []
        keys = [int(k) for pos in plp if pos for k in pos if str(k).lstrip("-").isdigit()]
        entry.update(positions=len(plp), ids=len(keys), ids_valid=bool(keys) and all(0 <= k < 2**31 for k in keys))
    else:
        entry["error"] = str(body)[:300]
    result["prompt_logprobs"] = entry
    health, _ = _request(args, "/health", timeout=30.0)
    result["health_after"] = health
    result["ok"] = bool(health == 200 and result["logprobs"].get("ids_valid") and
                        result["logprobs"].get("sampled_is_top") and result["prompt_logprobs"].get("ids_valid"))
    print("LOGPROBS %s: logprobs %s, prompt_logprobs %s, /health after: %s" % (
        "ok" if result["ok"] else "FAILED", json.dumps(result["logprobs"])[:400],
        json.dumps(result["prompt_logprobs"])[:200], health), flush=True)
    return result


def summary(probe):
    """Per kind and arm: the measured rounds' steady ms per token, and each arm against base."""
    rows = {}
    for res in probe.results["rounds"]:
        if res.get("tag") != "" or res.get("steady_ms_p50") is None or res.get("min_tokens"):
            continue
        if res["kind"] not in ("short", "long", "sampled", "sampled1", "gc32", "gc1"):
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
        specs = [r.get("spec") or {} for r in rounds]
        if any(sp.get("drafts") for sp in specs):
            print("  %-6s %-5s step ms %s, tokens per chunk %s, acceptance length %s, per position %s" % (
                kind, arm, " ".join(_f(r.get("step_ms_p50")) for r in rounds),
                " ".join(_f(r.get("tokens_per_chunk_p50")) for r in rounds),
                " ".join(_f(sp.get("acceptance_length")) for sp in specs),
                " ".join("[%s]" % ", ".join("%.3f" % x for x in (sp.get("position_rates") or []) if x is not None)
                         for sp in specs)))
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
        cand = table.get((kind, "cand"))
        if arm == "candvp" and cand is not None:
            out.append({"kind": kind, "arm": arm, "against": "cand", "base_ms": cand, "arm_ms": value,
                        "delta_ms": value - cand, "delta_pct": 100.0 * (value - cand) / cand})
            print("  %-6s %-5s against cand: %.2f -> %.2f ms per token (%+.2f ms, %+.1f %%)" % (
                kind, arm, cand, value, value - cand, 100.0 * (value - cand) / cand))
    for kind in sorted({k for k, _ in table}):  # a speculative server: the candidate paths against each other
        for arm, against in (("cand2", "cand"), ("cand2vp", "cand"), ("cand2vp", "cand2")):
            a, b = table.get((kind, against)), table.get((kind, arm))
            if a is None or b is None:
                continue
            out.append({"kind": kind, "arm": arm, "against": against, "base_ms": a, "arm_ms": b,
                        "delta_ms": b - a, "delta_pct": 100.0 * (b - a) / a})
            print("  %-7s %-7s against %-5s: %.2f -> %.2f ms per token (%+.2f ms, %+.1f %%)" % (
                kind, arm, against, a, b, b - a, 100.0 * (b - a) / a))
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
    ap.add_argument("--no-sampled", action="store_true", help="leave out the rounds without a temperature")
    ap.add_argument("--candvp", action="store_true", help="add the candvp arm (candidates on the vocabulary-"
                    "parallel head) to the sampled rounds")
    ap.add_argument("--spec-arms", default="", help="a speculative server (SERVE_SPEC_K): the sampled rounds' arms "
                    "instead of base / cand / candvp, each with a 32-stream and a 1-stream round, in turn and twice, "
                    "with the speculative counters per round (e.g. cand,cand2,cand2vp: VLLM_HPU_CANDIDATE_SAMPLING=1, "
                    "which leaves speculative steps to VLLM_HPU_SPEC_SAMPLED_VERIFY's path, 2, the candidate region, "
                    "and 2 with VLLM_HPU_VOCAB_PARALLEL_LOGITS=2)")
    ap.add_argument("--gc-arms", action="store_true", help="after the other rounds: greedy rounds at N streams and "
                    "1 stream with VLLM_HPU_WORKER_GC_FREEZE off (base) and on (gc), in turn and twice; every arm's "
                    "switch file then carries the flag (the server must know it)")
    ap.add_argument("--logprobs-check", action="store_true", help="last: a request with logprobs=5 and one with "
                    "prompt_logprobs=2, the ids checked, then /health (a server that dies there loses nothing)")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        if arm not in ("both", "i01", "i05"):
            raise SystemExit("unknown arm %r (one of both, i01, i05)" % arm)
    main_arm, other_arms = arms[0], arms[1:]
    rng = random.Random(20261120)
    ids = lambda count: [rng.randrange(1000, 150000) for _ in range(count)]  # noqa: E731
    n, ntok = args.streams, args.tokens
    probe = Probe(args)
    measured = {}  # (kind, arm) -> the measured rounds

    def measure(kind, arm, prompts, warm=None, sampling=None):
        if warm:
            probe.run(kind, arm, prompts, warm, tag="-warm", sampling=sampling)
        res = probe.run(kind, arm, prompts, ntok, sampling=sampling)
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

    # --- sampled steps (no temperature in the request): vLLM's sampler against sampling over candidates
    def sampled_words(arm):
        many = probe.run("text", arm, text_prompts, 24, tag="-sampled", sampling={})
        for i in shown:
            print("TEXT %-5s sampled stream %2d of %d: %r" % (arm, i, n, (many["texts"][i] or "")[:70]), flush=True)
        on_task = sum(1 for a, text in zip(starts, many["texts"])
                      if text is not None and text.replace(" ", "").startswith("%d,%d," % (a + 3, a + 4)))
        many["on_task"] = on_task
        print("TASK %-5s sampled: %d of %d streams go on with their own numbers" % (arm, on_task, n), flush=True)
        probe.save()

    spec_arms = [a for a in args.spec_arms.split(",") if a]
    for arm in spec_arms:
        if arm not in ARMS:
            raise SystemExit("unknown spec arm %r" % arm)
    if spec_arms and not args.no_sampled:
        # A speculative server: the sampled verification paths in turn, twice, 32 streams and 1 stream each.
        for arm in spec_arms:
            measure("sampled", arm, short_prompts, warm=24, sampling={})
            sampled_words(arm)
            measure("sampled1", arm, [one_prompt], warm=24, sampling={})
        for arm in spec_arms:
            measure("sampled", arm, short_prompts, sampling={})
            measure("sampled1", arm, [one_prompt], sampling={})
        last_arm = spec_arms[-1]
        seeded = [probe.run("text1", last_arm, [one_prompt], 48, tag="-seeded%d" % i, sampling={"seed": 20261120})
                  for i in (1, 2)]
        probe.compare("seeded", seeded[0], seeded[1])
        print("SEED %s: %r" % (last_arm, (seeded[0]["texts"][0] or "")[:120]), flush=True)
    # vLLM's sampler first (it runs on a card today): what a sampled step of real traffic costs.
    if not args.no_sampled and not spec_arms:
        measure("sampled", "base", short_prompts, warm=24, sampling={})
        sampled_words("base")

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

    # --- sampling over candidates: its first run on a card comes after the long rounds, so a failure of the
    # new region costs none of them. cand, base, cand: with the base round above that is two of each, in turn.
    if not args.no_sampled and not spec_arms:
        measure("sampled", "cand", short_prompts, warm=24, sampling={})
        sampled_words("cand")
        measure("sampled", "base", short_prompts, sampling={})
        measure("sampled", "cand", short_prompts, sampling={})
        seeded = [probe.run("text1", "cand", [one_prompt], 48, tag="-seeded%d" % i, sampling={"seed": 20261120})
                  for i in (1, 2)]
        probe.compare("seeded", seeded[0], seeded[1])
        print("SEED cand: %r" % ((seeded[0]["texts"][0] or "")[:120], ), flush=True)
        if args.candvp:
            measure("sampled", "candvp", short_prompts, warm=24, sampling={})
            sampled_words("candvp")
            measure("sampled", "cand", short_prompts, sampling={})
            measure("sampled", "candvp", short_prompts, sampling={})

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
    if args.gc_arms:
        # The worker's collector frozen (gc) against as is (base), greedy, N streams and 1 stream, in turn, twice.
        for first in (True, False):
            for arm in ("base", "gc"):
                measure("gc32", arm, short_prompts, warm=24 if first else None)
                measure("gc1", arm, [one_prompt], warm=24 if first else None)
    probe.set_arm("base")
    summary(probe)
    if not args.no_sampled and not spec_arms:
        # Last: a seeded request through vLLM's sampler (a generator on the device). It may end the server.
        seeded = [probe.run("text1", "base", [one_prompt], 48, tag="-seeded%d" % i, sampling={"seed": 20261120})
                  for i in (1, 2)]
        probe.compare("seeded", seeded[0], seeded[1])
        if any(r["errors"] for r in seeded):
            probe.note("a seeded request through vLLM's sampler failed: %s" % (
                seeded[0].get("error_samples") or seeded[1].get("error_samples")))
            return 0 if probe.switch_ok else 1
    if args.logprobs_check:
        probe.set_arm("base")
        check = logprobs_check(args, one_prompt)
        probe.results["logprobs_check"] = check
        probe.save()
        if not check.get("ok"):
            return 1
    bad = sum(1 for r in probe.results["rounds"] if r["errors"])
    if bad or not probe.switch_ok:
        print("PROBE: %d round(s) with errors, switches ok: %s" % (bad, probe.switch_ok), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
