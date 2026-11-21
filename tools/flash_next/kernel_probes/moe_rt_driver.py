#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""moe_rt_driver.py: the probe of moe_rt_ab.sh (kernel-fixes). One live server, the MoE dense range switched at
run time.

The integration kit's headline_job.sh starts the server (``VLLM_HPU_MOE_DENSE=1``, ``VLLM_HPU_RUNTIME_SWITCH_FILE``)
and calls this script in place of decode_scaling.py (the same ``--port`` / ``--out`` arguments; the others are
ignored). Per setting of ``MOE_RT_SETTINGS`` (default ``off,1,17``: ``off`` = the vendor op with the job's chunk
setting at every size, ``N`` = the dense form from N to 512 tokens and the vendor op outside):

1. the switch: the switch file gets ``VLLM_HPU_MOE_DENSE_MIN_TOKENS`` / ``_MAX_TOKENS`` by a rename while no request
   runs, then one short request makes every worker take it at the same step (the runner drops the captured HPU
   graphs of the model and the drafter; the next steps capture again). The server log must then hold the drop line
   naming the model, or the probe stops with exit code 9 (a setting that did not reach the graphs gives no verdict);
2. decode rounds at ``MOE_RT_LEVELS`` (default 1,8,32): one warm round, then ``MOE_RT_ROUNDS`` (2) rounds of each
   prompt set (story, code), greedy, ``MOE_RT_TOKENS`` (128) tokens per stream. Per round, the median over the
   streams of: ms per step (a stream's median gap between chunks; one chunk per engine step), tokens per step
   ((tokens - 1) / (chunks - 1), the first token comes from the prefill) and ms per token;
3. with ``MOE_RT_TRACE_STREAMS=N`` and the setting in ``MOE_RT_TRACE_SETTINGS`` (default ``off,1``): one device-trace
   capture of step-anatomy's anatomy_probe.py (``MOE_RT_ANATOMY_PROBE``) at N streams; the launches per step come
   from that trace (trace_components.py on the workstation).

With ``MOE_RT_ACCURACY=1``, after the rounds: the prompt logprobs (``prompt_logprobs=1``) of ``MOE_RT_ACC_PROMPTS``
(10) prompts of 300 to 480 tokens (Python standard-library sources cut by the server's /tokenize), one request at a
time, under the settings of ``MOE_RT_ACC_ORDER`` (default ``1,off,1,off``): a prefill of 17 to 512 tokens runs the
dense form under ``1`` and the vendor op under ``off``. Reported: the mean and max absolute difference of the prompt
tokens' logprobs and the top-1 agreement, dense against vendor and each against its own repeat.

Writes ``--out`` (the kit's scaling.json: every measured round under ``levels``) and ``moe_rt.json`` next to it.
"""

import argparse
import http.client
import importlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_DENSE = 512
DROP_LINE = "captured HPU graphs dropped: model"
STACKED_LINE = "keep their stacked experts"
ACC_MODULES = ("argparse", "textwrap", "difflib", "heapq", "fractions", "statistics", "string", "bisect", "functools",
               "calendar", "shlex", "json.decoder", "csv", "fnmatch")


WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river bridge clock market violin garden letter forest mountain engine signal mirror island "
         "library candle meadow canyon glacier desert falcon").split()


def one_stream(args, prompt: str, ntok: int, out: list, idx: int, barrier) -> None:
    """One greedy streamed completion: the arrival time of every chunk with text (one chunk per engine step)."""
    req = {"model": args.model, "prompt": prompt, "max_tokens": ntok, "ignore_eos": True, "temperature": 0,
           "stream": True, "stream_options": {"include_usage": True}}
    rec = {"status": None, "t0": None, "times": [], "usage": None, "err": None, "text": ""}
    try:
        barrier.wait()
        conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
        rec["t0"] = time.perf_counter()
        conn.request("POST", "/v1/completions", body=json.dumps(req), headers={"Content-Type": "application/json"})
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
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    rec["times"].append(time.perf_counter())
                    if len(rec["text"]) < 400:
                        rec["text"] += choices[0]["text"]
        conn.close()
    except Exception as error:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(error)
    out[idx] = rec


def setting_env(setting: str) -> dict:
    """The switch file's flags for a setting: ``off`` (a minimum above the maximum: the vendor op at every size) or
    the dense form's minimum token count."""
    low = MAX_DENSE + 1 if setting == "off" else int(setting)
    return {"VLLM_HPU_MOE_DENSE_MIN_TOKENS": str(low), "VLLM_HPU_MOE_DENSE_MAX_TOKENS": str(MAX_DENSE)}


def write_switch(path: str, setting: str) -> None:
    """Replace the switch file with a rename (every worker sees one change)."""
    staging = path + ".new"
    with open(staging, "w", encoding="utf-8") as handle:
        handle.write("# moe_rt_driver: setting %s\n" % setting)
        for name, value in sorted(setting_env(setting).items()):
            handle.write("%s=%s\n" % (name, value))
    os.replace(staging, path)


def count_lines(path: str, needle: str) -> int:
    try:
        with open(path, errors="replace") as handle:
            return sum(1 for line in handle if needle in line)
    except OSError:
        return 0


def post(args, path: str, body: dict, timeout: float = 600.0) -> dict:
    conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError("%s: HTTP %d %s" % (path, resp.status, data[:300].decode("utf-8", "replace")))
        return json.loads(data)
    finally:
        conn.close()


def switch(args, setting: str, log_path: str) -> int:
    """Switch the server to a setting; returns the number of workers that logged the graph drop."""
    before = count_lines(log_path, DROP_LINE)
    write_switch(args.switch_file, setting)
    post(args, "/v1/completions", {"model": args.model, "prompt": "Switch %s." % setting, "max_tokens": 2,
                                   "temperature": 0})
    seen = 0
    deadline = time.time() + 60
    while time.time() < deadline:
        seen = count_lines(log_path, DROP_LINE) - before
        if seen >= args.workers:
            break
        time.sleep(0.5)
    return seen


def prompt_for(prompt_set: str, i: int) -> str:
    """Stream i's prompt: ``story`` (the integration kit's decode prompt) or ``code``."""
    w = WORDS
    if prompt_set == "code":
        return ("Stream %d. Write a Python module that reads a log of %s and %s records, groups them by %s, and "
                "prints a summary table. Explain each function before you write it.\n\n" %
                (i, w[(i * 5 + 1) % len(w)], w[(i * 3 + 2) % len(w)], w[(i * 7 + 4) % len(w)]))
    return "Stream %d. Write a long, detailed story about a %s named %s who finds a %s. Begin:" % (
        i, w[i % len(w)], w[(i * 7 + 3) % len(w)].title(), w[(i * 3 + 5) % len(w)])


def run_round(args, n: int, ntok: int, prompt_set: str) -> dict:
    """One round of n greedy streams; the per-step numbers, median over the streams."""
    out = [None] * n
    barrier = threading.Barrier(n)
    threads = [
        threading.Thread(target=one_stream, args=(args, prompt_for(prompt_set, i), ntok, out, i, barrier), daemon=True)
        for i in range(n)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(args.timeout + 60)
    wall = time.perf_counter() - start
    per, errors = [], []
    for rec in out:
        if not rec or rec["err"] or rec["status"] != 200 or len(rec["times"]) < 3:
            errors.append((rec or {}).get("err") or "no record or fewer than 3 chunks")
            continue
        comp = (rec["usage"] or {}).get("completion_tokens") or len(rec["times"])
        times = rec["times"]
        gaps = [b - a for a, b in zip(times, times[1:])]
        span = times[-1] - times[0]
        per.append({
            "step_ms": 1000.0 * statistics.median(gaps),
            "step_ms_mean": 1000.0 * span / (len(times) - 1),
            "tokens_per_step": (comp - 1) / (len(times) - 1),
            "ms_per_token": 1000.0 * span / max(1, comp - 1),
            "steady_ms_per_token": 1000.0 * statistics.median(gaps) / max(comp / len(times), 1e-9),
            "tokens_per_chunk": comp / len(times),
            "ttft_s": times[0] - rec["t0"],
            "completion_tokens": comp,
            "pause_s": sum(g for g in gaps if g > 0.5),
            "first": times[0],
            "last": times[-1],
        })
    res = {"streams": n, "tokens": ntok, "prompt_set": prompt_set, "ok": len(per), "errors": len(errors),
           "wall_s": wall}
    if per:
        for key in ("step_ms", "step_ms_mean", "tokens_per_step", "ms_per_token", "steady_ms_per_token",
                    "tokens_per_chunk", "ttft_s"):
            res[key + "_p50"] = statistics.median(p[key] for p in per)
        res["ms_per_token_max"] = max(p["ms_per_token"] for p in per)
        res["pause_s_max"] = max(p["pause_s"] for p in per)
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = sum(p["completion_tokens"] for p in per) / span_all if span_all > 0 else None
    res["texts"] = {str(i): (out[i] or {}).get("text", "") for i in sorted({0, n - 1})}
    if errors:
        res["error_samples"] = [str(e)[:300] for e in errors[:3]]
    return res


def first_difference(a: str, b: str) -> int:
    """The index of the first word where two texts differ (-1: none within the shorter one)."""
    for index, (x, y) in enumerate(zip(a.split(), b.split())):
        if x != y:
            return index
    return -1


def capture(args, setting: str, out_dir: str, trace_dir: str) -> dict:
    """One anatomy_probe capture round at MOE_RT_TRACE_STREAMS streams; the trace files it added."""
    before = set(os.listdir(trace_dir)) if os.path.isdir(trace_dir) else set()
    target = os.path.join(out_dir, "trace_probe_%s.json" % setting)
    cmd = [sys.executable, args.anatomy_probe, "--port", str(args.port), "--model", args.model, "--phases", "",
           "--trace", str(args.trace_streams), "--trace-tokens", "64", "--trace-after", "12", "--trace-seconds",
           str(args.trace_seconds), "--measured-timeout", "600", "--out", target]
    start = time.time()
    rc = subprocess.call(cmd)
    time.sleep(5)
    after = set(os.listdir(trace_dir)) if os.path.isdir(trace_dir) else set()
    files = sorted(after - before)
    print("moe_rt: setting %s: capture at %d streams ended with exit code %d after %.0f s, %d new trace files" %
          (setting, args.trace_streams, rc, time.time() - start, len(files)),
          flush=True)
    return {"setting": setting, "streams": args.trace_streams, "rc": rc, "files": files, "probe_out": target,
            "seconds": time.time() - start}


def tokenize(args, text: str):
    try:
        data = post(args, "/tokenize", {"model": args.model, "prompt": text}, timeout=60)
        return data.get("tokens")
    except (OSError, RuntimeError, ValueError):
        return None


def accuracy_prompts(args, count: int, low: int = 300, high: int = 480) -> list:
    """Up to ``count`` prompts of ``low`` to ``high`` tokens: standard-library sources, cut at a line."""
    prompts = []
    for name in ACC_MODULES:
        if len(prompts) >= count:
            break
        try:
            with open(importlib.import_module(name).__file__, encoding="utf-8") as handle:
                lines = handle.read().splitlines(keepends=True)
        except (ImportError, OSError, TypeError, AttributeError):
            continue
        lo, hi, best = 1, min(len(lines), 400), None
        while lo <= hi:  # the most lines with at most `high` tokens
            mid = (lo + hi) // 2
            tokens = tokenize(args, "".join(lines[:mid]))
            if tokens is None:
                return prompts
            if len(tokens) <= high:
                best, lo = (mid, len(tokens)), mid + 1
            else:
                hi = mid - 1
        if best and best[1] >= low:
            prompts.append({"name": name, "text": "".join(lines[:best[0]]), "tokens": best[1]})
    return prompts


def prompt_logprobs(args, prompt: dict) -> dict:
    """The prompt tokens' logprobs and the top-1 token per position (positions 1 onwards)."""
    data = post(args, "/v1/completions", {"model": args.model, "prompt": prompt["text"], "max_tokens": 1,
                                          "temperature": 0, "prompt_logprobs": 1})
    entries = data["choices"][0].get("prompt_logprobs") or []
    ids = prompt.get("ids") or []
    actual, top1 = [], []
    for position, entry in enumerate(entries):
        if position == 0 or not entry:
            continue
        if position < len(ids) and str(ids[position]) in entry:
            actual.append(entry[str(ids[position])]["logprob"])
        else:
            actual.append(None)
        ranked = [(value.get("rank") or 10**9, key) for key, value in entry.items()]
        top1.append(min(ranked)[1] if ranked else None)
    return {"actual": actual, "top1": top1, "positions": len(entries)}


def compare(a: list, b: list) -> dict:
    """Two runs of the same prompts: |delta| of the prompt tokens' logprobs and the top-1 agreement."""
    deltas, agree, total = [], 0, 0
    for x, y in zip(a, b):
        for u, v in zip(x["actual"], y["actual"]):
            if u is not None and v is not None:
                deltas.append(abs(u - v))
        for u, v in zip(x["top1"], y["top1"]):
            total += 1
            agree += int(u == v)
    if not deltas:
        return {"positions": 0}
    deltas.sort()
    return {"positions": len(deltas), "mean_abs": sum(deltas) / len(deltas), "max_abs": deltas[-1],
            "p99_abs": deltas[min(len(deltas) - 1, int(0.99 * len(deltas)))],
            "top1_agree": agree / total if total else None, "top1_positions": total}


def run_accuracy(args, log_path: str, results: dict, save) -> bool:
    prompts = accuracy_prompts(args, args.acc_prompts)
    print("moe_rt: accuracy: %d prompts (%s tokens)" % (len(prompts), ", ".join(str(p["tokens"]) for p in prompts)),
          flush=True)
    if not prompts:
        print("moe_rt: accuracy: no prompts (/tokenize failed?)", flush=True)
        return False
    for prompt in prompts:
        prompt["ids"] = tokenize(args, prompt["text"])
    runs = []
    for index, setting in enumerate(args.acc_order):
        seen = switch(args, setting, log_path)
        if seen < 1:
            print("moe_rt: FAIL: accuracy switch to %s: no worker dropped its graphs" % setting, flush=True)
            return False
        start = time.time()
        rows = [prompt_logprobs(args, prompt) for prompt in prompts]
        runs.append({"setting": setting, "index": index, "rows": rows})
        print("moe_rt: accuracy run %d (%s): %d prompts in %.1f s" % (index, setting, len(rows), time.time() - start),
              flush=True)
    pairs = []
    for i, x in enumerate(runs):
        for y in runs[i + 1:]:
            stats = compare(x["rows"], y["rows"])
            stats.update({"a": "%s#%d" % (x["setting"], x["index"]), "b": "%s#%d" % (y["setting"], y["index"])})
            pairs.append(stats)
            if stats.get("positions"):
                print("ACC  %-8s vs %-8s positions=%d mean|d|=%.5f max|d|=%.4f p99|d|=%.4f top1 agree=%.4f" %
                      (stats["a"], stats["b"], stats["positions"], stats["mean_abs"], stats["max_abs"], stats["p99_abs"],
                       stats["top1_agree"]),
                      flush=True)
            else:
                print("ACC  %s vs %s: no comparable positions" % (stats["a"], stats["b"]), flush=True)
    results["accuracy"] = {"prompts": [{k: p[k] for k in ("name", "tokens")} for p in prompts], "pairs": pairs,
                           "runs": runs}
    save()
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--out", default="scaling.json")
    ap.add_argument("--timeout", type=float, default=1500.0)
    args, _ = ap.parse_known_args()
    env = os.environ.get
    args.switch_file = env("MOE_RT_SWITCH_FILE") or env("VLLM_HPU_RUNTIME_SWITCH_FILE") or ""
    args.settings = [s for s in env("MOE_RT_SETTINGS", "off,1,17").split(",") if s]
    args.levels = [int(x) for x in env("MOE_RT_LEVELS", "1,8,32").split(",") if x]
    args.rounds = int(env("MOE_RT_ROUNDS", "2"))
    args.prompt_sets = [s for s in env("MOE_RT_PROMPTS", "story,code").split(",") if s]
    args.tokens = int(env("MOE_RT_TOKENS", "128"))
    args.warm_tokens = int(env("MOE_RT_WARM", "48"))
    args.workers = int(env("MOE_RT_WORKERS", "8"))
    args.trace_streams = int(env("MOE_RT_TRACE_STREAMS", "0") or 0)
    args.trace_settings = [s for s in env("MOE_RT_TRACE_SETTINGS", "off,1").split(",") if s]
    args.trace_seconds = float(env("MOE_RT_TRACE_SECONDS", "0.4"))
    args.trace_dir = env("MOE_RT_TRACE_DIR", "")
    args.anatomy_probe = env("MOE_RT_ANATOMY_PROBE", "")
    args.accuracy = env("MOE_RT_ACCURACY", "0") == "1"
    args.acc_prompts = int(env("MOE_RT_ACC_PROMPTS", "10"))
    args.acc_order = [s for s in env("MOE_RT_ACC_ORDER", "1,off,1,off").split(",") if s]
    if not args.switch_file:
        print("moe_rt: FAIL: no switch file (MOE_RT_SWITCH_FILE / VLLM_HPU_RUNTIME_SWITCH_FILE)", flush=True)
        return 9
    out_dir = os.path.dirname(os.path.abspath(args.out))
    log_path = os.path.join(out_dir, "server0.log")
    results = {"levels": [], "ctx": [], "mintok": [], "sampled": [], "settings": args.settings, "captures": [],
               "switches": []}

    def save():
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)
        with open(os.path.join(out_dir, "moe_rt.json"), "w") as handle:
            json.dump(results, handle, indent=1)

    if count_lines(log_path, STACKED_LINE) < 1:
        print("moe_rt: FAIL: %r is not in %s: the server did not keep the stacked experts" % (STACKED_LINE, log_path),
              flush=True)
        return 9
    failed = False
    for setting in args.settings:
        t0 = time.time()
        seen = switch(args, setting, log_path)
        results["switches"].append({"setting": setting, "workers": seen, "at": time.time()})
        print("=== moe_rt: setting %s (%s): %d workers dropped their graphs (%.1f s) %s" %
              (setting, ", ".join("%s=%s" % kv for kv in sorted(setting_env(setting).items())), seen, time.time() - t0,
               time.strftime("%H:%M:%S", time.gmtime())),
              flush=True)
        if seen < 1:
            print("moe_rt: FAIL: the switch to %s reached no worker's graphs; no verdict" % setting, flush=True)
            save()
            return 9
        for n in args.levels:
            warm = run_round(args, n, args.warm_tokens, args.prompt_sets[0])
            print("warm  %-4s N=%-3d ok=%d err=%d wall=%.1fs" % (setting, n, warm["ok"], warm["errors"], warm["wall_s"]),
                  flush=True)
            for rep in range(args.rounds):
                for prompt_set in args.prompt_sets:
                    m = run_round(args, n, args.tokens, prompt_set)
                    m.update({"setting": setting, "rep": rep})
                    results["levels"].append(m)
                    save()
                    if m["ok"] and not m["errors"]:
                        print("MEAS  %-4s N=%-3d %-5s rep=%d  ms/step p50=%.2f (mean %.2f)  tokens/step=%.3f  "
                              "ms/token=%.2f (max %.2f)  pauses>0.5s max=%.1fs  text0=%r" %
                              (setting, n, prompt_set, rep, m["step_ms_p50"], m["step_ms_mean_p50"],
                               m["tokens_per_step_p50"], m["ms_per_token_p50"], m["ms_per_token_max"],
                               m["pause_s_max"], m["texts"]["0"][:70]),
                              flush=True)
                    else:
                        failed = True
                        print("MEAS  %-4s N=%d %s rep=%d FAILED: %s" % (setting, n, prompt_set, rep,
                                                                         m.get("error_samples")),
                              flush=True)
        if args.trace_streams and setting in args.trace_settings and args.anatomy_probe:
            results["captures"].append(capture(args, setting, out_dir, args.trace_dir))
            save()
    # Text differences between settings (stream 0, same level, prompt set and repeat 0).
    by_key = {}
    for m in results["levels"]:
        if m.get("rep") == 0 and m.get("ok"):
            by_key.setdefault((m["streams"], m["prompt_set"]), {})[m["setting"]] = m["texts"].get("0", "")
    results["text_diff"] = []
    for (n, prompt_set), texts in sorted(by_key.items()):
        base = texts.get(args.settings[0])
        for setting, text in texts.items():
            if setting != args.settings[0] and base is not None:
                index = first_difference(base, text)
                results["text_diff"].append({"streams": n, "prompt_set": prompt_set, "a": args.settings[0],
                                             "b": setting, "first_word_diff": index})
                print("TEXT  N=%-3d %-5s %s vs %s: first different word %s" %
                      (n, prompt_set, args.settings[0], setting, index if index >= 0 else "none (in the first 400 chars)"),
                      flush=True)
    save()
    if args.accuracy and not run_accuracy(args, log_path, results, save):
        failed = True
    save()
    print("moe_rt: done, %d measured rounds%s" % (len(results["levels"]), ", SOME FAILED" if failed else ""), flush=True)
    return 5 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
