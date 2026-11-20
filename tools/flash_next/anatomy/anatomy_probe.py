#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode probe of the step anatomy: equal-length streaming completions against one vLLM server, in phases.

A phase is ``MODE:LEVELS[:TOKENS]``. ``MODE`` is ``warm`` or a step-timer mode (``VLLM_HPU_STEP_TIMING``, the
server must have been started with a nonzero mode): the probe writes it to ``<timing-dir>/step_timing.mode``,
waits ``--idle`` seconds (the server rereads the file when a step starts after two idle seconds), then runs one
round per level. ``LEVELS`` is a comma list of stream counts. A ``warm`` round generates as many tokens as the
measured rounds, so every decode bucket those rounds touch is compiled before the first measured round (the
server runs with ``VLLM_SKIP_WARMUP=true`` and compiles a bucket the first time a step needs it).

Every stream generates exactly ``TOKENS`` tokens (``min_tokens = max_tokens``, ``ignore_eos``), temperature 0.
Per round: ms per token over each stream's own decode span (p50, min, max over the streams), time to first token,
the aggregate tokens per second, and a hash of every stream's text (greedy text parity between two servers).

``--trace LEVELS`` then takes one torch-profiler capture per level in the middle of a decode round: the round
starts, the probe waits until every stream has ``--trace-after`` tokens, calls ``/start_profile``, sleeps
``--trace-seconds``, calls ``/stop_profile`` (the server needs ``--profiler-config`` with a torch profiler).

After the last phase the probe waits ``--idle`` seconds and sends one short request, so the timer writes the
records it still holds. Standard library only::

    python3 anatomy_probe.py --port 8100 --timing-dir /results/a/timing \\
        --phases "warm:1,8,32:128;1:1,8,32:128;4:1,32:16" --trace 32,1 --out probe.json
    python3 anatomy_probe.py --selftest      # a stand-in server on this machine: checks this script only
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

WORDS = ("lighthouse keeper harbour storm lantern rope gull tide anchor compass ledger window ember winter "
         "orchard river signal mapmaker bridge clock").split()
MODE_FILE = "step_timing.mode"


def prompt_for(i: int) -> str:
    """The prompt of stream ``i``: about 30 tokens, a distinct first sentence per stream."""
    return "Stream %d. Write a long, detailed story about a %s named %s who finds a %s. Begin:" % (
        i, WORDS[i % len(WORDS)], WORDS[(i * 7 + 3) % len(WORDS)].title(), WORDS[(i * 3 + 5) % len(WORDS)])


def parse_phases(text: str, default_tokens: int) -> list:
    """``"warm:1,8,32:128;1:1,32"`` -> ``[{"mode": "warm", "levels": [1, 8, 32], "tokens": 128}, ...]``."""
    phases = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) not in (2, 3):
            raise ValueError("a phase is MODE:LEVELS[:TOKENS], got %r" % part)
        mode = fields[0].strip()
        if mode != "warm" and not mode.isdigit():
            raise ValueError("MODE is 'warm' or a timer mode number, got %r" % mode)
        levels = [int(x) for x in fields[1].split(",") if x.strip()]
        if not levels or min(levels) < 1:
            raise ValueError("LEVELS are positive stream counts, got %r" % fields[1])
        tokens = int(fields[2]) if len(fields) == 3 else default_tokens
        if tokens < 2:
            raise ValueError("TOKENS must be at least 2, got %d" % tokens)
        phases.append({"mode": mode, "levels": levels, "tokens": tokens})
    return phases


def one_stream(host, port, model, prompt, ntok, rec, start_barrier, timeout_s):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": ntok, "min_tokens": ntok, "ignore_eos": True,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    })
    try:
        start_barrier.wait()
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
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    rec["times"].append(time.perf_counter())
                    rec["text"].append(choices[0]["text"])
        conn.close()
    except Exception as exc:  # noqa: BLE001 - a probe records every failure
        rec["err"] = repr(exc)
    rec["done"] = True


def new_record():
    return {"status": None, "t0": None, "times": [], "text": [], "usage": None, "err": None, "done": False}


def start_round(args, n, ntok):
    """Start ``n`` streams; returns ``(records, threads, t_start)``."""
    records = [new_record() for _ in range(n)]
    barrier = threading.Barrier(n)
    threads = [
        threading.Thread(target=one_stream,
                         args=(args.host, args.port, args.model, prompt_for(i), ntok, records[i], barrier,
                               args.timeout),
                         daemon=True) for i in range(n)
    ]
    t_start = time.perf_counter()
    for thread in threads:
        thread.start()
    return records, threads, t_start


def finish_round(args, n, ntok, records, threads, t_start) -> dict:
    for thread in threads:
        thread.join(args.timeout + 60)
    wall = time.perf_counter() - t_start
    per, errors = [], 0
    for rec in records:
        if rec["err"] or rec["status"] != 200 or len(rec["times"]) < 2:
            errors += 1
            continue
        comp = (rec["usage"] or {}).get("completion_tokens") or len(rec["times"])
        span = rec["times"][-1] - rec["times"][0]
        per.append({"ttft_s": rec["times"][0] - rec["t0"], "ms_per_token": 1000.0 * span / max(1, comp - 1),
                    "completion_tokens": comp, "prompt_tokens": (rec["usage"] or {}).get("prompt_tokens"),
                    "first": rec["times"][0], "last": rec["times"][-1]})
    texts = ["".join(rec["text"]) for rec in records]
    res = {"streams": n, "tokens": ntok, "ok": len(per), "errors": errors, "wall_s": round(wall, 3),
           "text_sha1": [hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for t in texts],
           "text0": texts[0][:400] if texts else ""}
    if per:
        ms = sorted(p["ms_per_token"] for p in per)
        res["ms_per_token_p50"] = statistics.median(ms)
        res["ms_per_token_min"] = ms[0]
        res["ms_per_token_max"] = ms[-1]
        res["ttft_p50_s"] = statistics.median(p["ttft_s"] for p in per)
        res["prompt_tokens_p50"] = statistics.median(p["prompt_tokens"] or 0 for p in per)
        span_all = max(p["last"] for p in per) - min(p["first"] for p in per)
        res["aggregate_tok_s"] = sum(p["completion_tokens"] for p in per) / span_all if span_all > 0 else None
        res["all_decoding_window_s"] = max(0.0, min(p["last"] for p in per) - max(p["first"] for p in per))
    errs = [rec["err"] for rec in records if rec["err"]][:3]
    if errs:
        res["error_samples"] = errs
    return res


def run_round(args, n, ntok) -> dict:
    records, threads, t_start = start_round(args, n, ntok)
    return finish_round(args, n, ntok, records, threads, t_start)


def post(args, path, timeout=None) -> tuple:
    """POST ``path`` with an empty body; ``(status, seconds, error text or None)``."""
    started = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(args.host, args.port, timeout=timeout or args.timeout)
        conn.request("POST", path, body=b"", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()[:300].decode("utf-8", "replace")
        conn.close()
        return resp.status, round(time.perf_counter() - started, 3), None if resp.status == 200 else body
    except Exception as exc:  # noqa: BLE001 - recorded
        return None, round(time.perf_counter() - started, 3), repr(exc)


def write_mode(args, mode: int) -> None:
    if not args.timing_dir:
        return
    os.makedirs(args.timing_dir, exist_ok=True)
    path = os.path.join(args.timing_dir, MODE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("%d\n" % mode)
    os.replace(tmp, path)


def trace_round(args, n) -> dict:
    """One profiler capture in the middle of an ``n``-stream decode round."""
    ntok = args.trace_tokens
    records, threads, t_start = start_round(args, n, ntok)
    deadline = time.perf_counter() + args.timeout
    while time.perf_counter() < deadline:
        if all(len(rec["times"]) >= args.trace_after or rec["done"] for rec in records):
            break
        time.sleep(0.01)
    start = post(args, "/start_profile", timeout=900)
    t_started = time.perf_counter()
    tokens_before = sum(len(rec["times"]) for rec in records)
    time.sleep(args.trace_seconds)
    tokens_at_stop = sum(len(rec["times"]) for rec in records)
    stop = post(args, "/stop_profile", timeout=900)
    result = finish_round(args, n, ntok, records, threads, t_start)
    result.update({"trace": True, "start_profile": start, "stop_profile": stop,
                   "tokens_in_window": tokens_at_stop - tokens_before,
                   "window_s": round(time.perf_counter() - t_started, 3)})
    return result


def describe(res: dict) -> str:
    if not res.get("ok"):
        return "FAILED: %s" % res.get("error_samples")
    return ("ok=%d err=%d ms/tok p50=%.2f min=%.2f max=%.2f aggregate=%.1f tok/s ttft p50=%.2fs wall=%.1fs" %
            (res["ok"], res["errors"], res["ms_per_token_p50"], res["ms_per_token_min"], res["ms_per_token_max"],
             res["aggregate_tok_s"] or 0.0, res["ttft_p50_s"], res["wall_s"]))


def run(args) -> int:
    phases = parse_phases(args.phases, args.tokens)
    results = {"phases": [], "traces": [], "args": {k: v for k, v in vars(args).items() if k != "selftest"}}
    failed = False

    def save():
        if args.out:
            tmp = args.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1)
            os.replace(tmp, args.out)

    for phase in phases:
        mode = phase["mode"]
        if mode != "warm":
            write_mode(args, int(mode))
        elif args.timing_dir:
            write_mode(args, args.warm_mode)
        entry = {"mode": mode, "tokens": phase["tokens"], "rounds": [], "t_start": time.time()}
        results["phases"].append(entry)
        for n in phase["levels"]:
            # An idle gap before every round: the timer then writes the steps of each round as their own records
            # (it keeps the steps of one kind and row count together until a step starts after two idle seconds).
            time.sleep(args.idle)
            t_round = time.time()
            res = run_round(args, n, phase["tokens"])
            res["mode"] = mode
            res["t_start"], res["t_end"] = t_round, time.time()
            entry["rounds"].append(res)
            save()
            print("phase %-4s N=%-3d tokens=%-4d %s" % (mode, n, phase["tokens"], describe(res)), flush=True)
            if not res.get("ok"):
                failed = True
                break
        entry["t_end"] = time.time()
        if failed:
            break
    if args.trace and not failed:
        write_mode(args, args.trace_mode)
        time.sleep(args.idle)
        for n in [int(x) for x in args.trace.split(",") if x.strip()]:
            time.sleep(args.idle)
            t_round = time.time()
            res = trace_round(args, n)
            res["t_start"], res["t_end"] = t_round, time.time()
            results["traces"].append(res)
            save()
            print("trace N=%-3d start %s stop %s, %d tokens in the window (%.2f s); round %s" %
                  (n, res["start_profile"], res["stop_profile"], res["tokens_in_window"], res["window_s"],
                   describe(res)),
                  flush=True)
            if res["start_profile"][0] != 200:
                break
    # The timer writes what it holds when a step starts after an idle gap.
    write_mode(args, args.final_mode)
    time.sleep(args.idle)
    results["flush_round"] = run_round(args, 1, 4)
    save()
    print("flush round: %s" % describe(results["flush_round"]), flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------
# Self-test: a stand-in OpenAI server on this machine (no vLLM, no card).
# ---------------------------------------------------------------------------------------------
def _stub_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = {"start_profile": 0, "stop_profile": 0, "completions": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            if self.path in ("/start_profile", "/stop_profile"):
                calls[self.path[1:]] += 1
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            calls["completions"] += 1
            req = json.loads(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(data: bytes):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            for i in range(req["max_tokens"]):
                time.sleep(0.002)
                chunk(b"data: " + json.dumps({"choices": [{"text": " w%d" % i}]}).encode() + b"\n\n")
            usage = {"completion_tokens": req["max_tokens"], "prompt_tokens": 30}
            chunk(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
            chunk(b"data: [DONE]\n\n")
            chunk(b"")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


def selftest(args) -> int:
    import tempfile
    server, calls = _stub_server()
    with tempfile.TemporaryDirectory() as tmp:
        args.port = server.server_address[1]
        args.timing_dir = os.path.join(tmp, "timing")
        args.out = os.path.join(tmp, "probe.json")
        args.phases = "warm:1,3:8;1:1,3:8;4:2:4"
        args.trace = "3"
        args.trace_tokens = 40
        args.trace_after = 3
        args.trace_seconds = 0.02
        args.idle = 0.05
        rc = run(args)
        data = json.load(open(args.out, encoding="utf-8"))
        mode_text = open(os.path.join(args.timing_dir, MODE_FILE), encoding="utf-8").read().strip()
    server.shutdown()
    assert rc == 0, rc
    assert [p["mode"] for p in data["phases"]] == ["warm", "1", "4"], data["phases"]
    assert data["phases"][1]["rounds"][1]["ok"] == 3
    assert data["phases"][0]["rounds"][0]["text_sha1"] == data["phases"][1]["rounds"][0]["text_sha1"]
    assert data["traces"][0]["start_profile"][0] == 200 and calls["stop_profile"] == 1
    assert mode_text == str(args.final_mode)
    print("SELFTEST OK (stand-in server): phases, mode file, trace calls and the flush round work")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="anatomy_probe.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="fn")
    ap.add_argument("--phases", default="warm:1,8,32;1:1,8,32")
    ap.add_argument("--tokens", type=int, default=128, help="tokens per stream of a phase without its own")
    ap.add_argument("--timing-dir", default="", help="VLLM_HPU_STEP_TIMING_DIR of the server (the mode file)")
    ap.add_argument("--warm-mode", type=int, default=1, help="timer mode written for warm phases")
    ap.add_argument("--final-mode", type=int, default=1, help="timer mode written before the flush round")
    ap.add_argument("--idle", type=float, default=3.0, help="seconds between phases (the timer needs 2)")
    ap.add_argument("--trace", default="", help="stream counts of one profiler capture each")
    ap.add_argument("--trace-mode", type=int, default=1)
    ap.add_argument("--trace-tokens", type=int, default=96)
    ap.add_argument("--trace-after", type=int, default=12, help="tokens every stream has before the capture")
    ap.add_argument("--trace-seconds", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="probe.json")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
