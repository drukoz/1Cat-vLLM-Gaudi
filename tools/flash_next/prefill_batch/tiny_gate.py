# SPDX-License-Identifier: Apache-2.0
"""Token gate of batched prefill (``VLLM_HPU_PREFILL_BATCH``) on the tiny model, on a card-farm job's cards.

    farm job prefill-batch NAME all 25m -- python3 tools/flash_next/prefill_batch/tiny_gate.py [--mtp 2]

One tiny server per speculative setting (``--mtp 0,2``: without speculation, then MTP k = 2 with build 5's MTP
flags and async mode 2), tensor parallel size = the job's cards, started with ``VLLM_HPU_PREFILL_BATCH=8`` and a
runtime switch file (``VLLM_HPU_RUNTIME_SWITCH_FILE``) that is ``VLLM_HPU_PREFILL_BATCH=0`` for the arm ``off``
(every prompt chunk its own forward, today's path) and ``=8`` for the arm ``on``. Both arms send the same requests,
each arm with its own ``cache_salt`` (no prefix-cache block is shared across arms):

* reference: every prompt of the checkpoint's reference.json, four salted copies at once (equal shapes, so the arm
  ``on`` merges them); each copy must reproduce the HF CPU reference (a near tie at the first token is recorded,
  as the tiny gate does);
* burst: eight 300-token prompts and two 1,500-token prompts at once (the long ones are chunked), then each short
  one's next turn (its prompt, its tokens and 60 new ones: a prefix-cache hit) at once;
* arrival: four requests decoding 48 tokens, and four 200-token prompts sent while they decode.

Passes when every reference copy matches (or is a recorded near tie), every stream of ``on`` equals ``off`` (a first
difference only where ``off``'s best two logprobs are within ``NEAR_TIE``), the server log shows merged prompt
forwards in ``on`` and none in ``off``. Results: ``OUT/<server>/{server0.log, streams.json}``, ``OUT/summary.json``.
Exit code: the number of servers that failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from pathlib import Path

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from tools.flash_next.rental import gates, serve, tiny_smoke  # noqa: E402
from tools.flash_next.rental.mtp_sampled_card import post, token_ids, with_tp  # noqa: E402

ROWS = 8
NEAR_TIE = 0.0045  # logprob units: the tiny model's flat logits put its best two this close (test_engine_mtp)
MERGED_RE = re.compile(r"VLLM_HPU_PREFILL_BATCH: (\d+) merged prompt forwards so far")
COMMON_ENV = {"VLLM_HPU_PLE_HOST_NGRAM_IDS": "1", "PT_HPU_LAZY_ACC_PAR_MODE": "0",
              "VLLM_HPU_COLLECTIVE_MARK_STEP": "0", "VLLM_SKIP_WARMUP": "true",
              "VLLM_HPU_PREFIX_ALIGN_SPLIT": "1", "VLLM_HPU_PREFIX_KEEP_PROMPT_END": "1",
              "VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS": "64000"}
MTP_ENV = {"VLLM_HPU_GDN_CORE_VERIFY": "1", "VLLM_HPU_MTP_DRAFT_LOOP": "1", "VLLM_HPU_MTP_CHUNK_KV": "1",
           "VLLM_HPU_FLASH_NEXT_ASYNC_SCHED": "2"}


def write_switch(path: Path, rows: int) -> None:
    staging = path.with_suffix(".tmp")
    staging.write_text("VLLM_HPU_PREFILL_BATCH=%d\n" % rows, encoding="utf-8")
    os.replace(staging, path)  # a rename: every worker sees the change at the same step


def request(base_url: str, prompt_ids: list[int], max_tokens: int, salt: str) -> dict:
    payload = {"model": "tiny", "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0,
               "ignore_eos": True, "return_token_ids": True, "logprobs": 2, "cache_salt": salt}
    status, answer = post(base_url, payload)
    choice = ((answer.get("choices") or [{}])[0]) if isinstance(answer, dict) else {}
    top = (choice.get("logprobs") or {}).get("top_logprobs") or []
    return {"status": status, "ids": token_ids(answer), "top": [sorted(t.values(), reverse=True)[:2] for t in top],
            "error": None if status == 200 else str(answer)[:300]}


def at_once(base_url: str, jobs: list[tuple[str, list[int], int]], salt: str, delay_s: float = 0.0) -> dict:
    """Send every (name, prompt, max_tokens) at once (one thread each); returns {name: request record}."""
    if delay_s:
        time.sleep(delay_s)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = {name: pool.submit(request, base_url, prompt, max_tokens, salt) for name, prompt, max_tokens in jobs}
        return {name: future.result() for name, future in futures.items()}


def run_arm(base_url: str, arm: str, reference: dict, seed: int) -> dict:
    """One arm's requests (the same ones in both arms; ``seed`` makes the random prompts)."""
    rng = random.Random(seed)

    def ids(n):
        return [rng.randrange(1000, 60000) for _ in range(n)]

    salt = "pb-%s-%d" % (arm, seed)
    streams: dict = {}
    for index, prompt in enumerate(reference["prompts"]):
        want = prompt["generated_ids"]
        jobs = [("ref%d.%d" % (index, copy), prompt["prompt_ids"], len(want)) for copy in range(4)]
        # distinct salts per copy: four cold prompts of one shape, which the arm "on" can merge
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {name: pool.submit(request, base_url, prompt_ids, max_tokens, "%s-%s" % (salt, name))
                       for name, prompt_ids, max_tokens in jobs}
            streams.update({name: future.result() for name, future in futures.items()})
    short = [ids(300) for _ in range(ROWS)]
    long = [ids(1500) for _ in range(2)]
    jobs = [("b%d" % i, p, 12) for i, p in enumerate(short)] + [("L%d" % i, p, 12) for i, p in enumerate(long)]
    streams.update(at_once(base_url, jobs, salt))
    turns = []
    for i, p in enumerate(short):
        got = streams["b%d" % i]["ids"] or []
        turns.append(("b%dt2" % i, p + got + ids(60), 12))
    streams.update(at_once(base_url, turns, salt))
    decoding = [("d%d" % i, ids(120), 48) for i in range(4)]
    arriving = [("c%d" % i, ids(200), 12) for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(at_once, base_url, decoding, salt)
        second = pool.submit(at_once, base_url, arriving, salt, 0.5)
        streams.update(first.result())
        streams.update(second.result())
    return streams


def compare_reference(reference: dict, streams: dict) -> list[str]:
    problems = []
    for index, prompt in enumerate(reference["prompts"]):
        want = prompt["generated_ids"]
        for copy in range(4):
            got = streams.get("ref%d.%d" % (index, copy), {}).get("ids") or []
            first = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), None)
            if first is None and len(got) == len(want):
                continue
            if first is not None and gates._tiny_near_tie(prompt, got, want, first) is not None:
                continue
            problems.append("reference prompt %d (%d tokens) copy %d: first difference at %s" %
                            (index, len(prompt["prompt_ids"]), copy, first))
    return problems


def compare_arms(off: dict, on: dict) -> tuple[list[str], int]:
    """(problems, near ties) of ``on`` against ``off``, stream by stream."""
    problems, ties = [], 0
    for name, a in off.items():
        b = on.get(name) or {}
        if a.get("status") != 200 or b.get("status") != 200:
            problems.append("%s: status off %s on %s (%s)" % (name, a.get("status"), b.get("status"),
                                                              a.get("error") or b.get("error")))
            continue
        x, y = a.get("ids") or [], b.get("ids") or []
        first = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), None)
        if first is None and len(x) == len(y):
            continue
        top = a.get("top") or []
        margin = (top[first][0] - top[first][1]) if first is not None and first < len(top) and len(
            top[first]) == 2 else None
        if margin is not None and margin <= NEAR_TIE:
            ties += 1
            continue
        problems.append("%s: on leaves off at token %s (off's top-2 margin %s)" % (name, first, margin))
    return problems, ties


def merged_lines(log_path: Path) -> int:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    counts = [int(m.group(1)) for m in MERGED_RE.finditer(text)]
    return max(counts) if counts else 0


def run_server(k: int, ckpt: Path, out: Path, port: int, tp: int, health_timeout: float, extra_env: dict,
               reference: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    switch = out / "switches"
    write_switch(switch, 0)  # the arm "off" first, from the first step on
    overrides = dict(tiny_smoke.GDN_MODES["hc"])
    overrides.update(COMMON_ENV)
    overrides.update({"VLLM_HPU_PREFILL_BATCH": str(ROWS), "VLLM_HPU_RUNTIME_SWITCH_FILE": str(switch)})
    if k:
        overrides.update(MTP_ENV)
    overrides.update(extra_env)
    command = with_tp(tiny_smoke.serve_command(ckpt, port), tp)
    command[command.index("--max-num-seqs") + 1] = str(2 * ROWS)
    command += ["--max-num-batched-tokens", "4096", "--long-prefill-token-threshold", "1024"]
    if k:
        command += tiny_smoke.spec_serve_args(k)
    log_path = out / "server0.log"
    entry = {"server": out.name, "k": k, "tp": tp, "status": "failed", "problems": [], "near_ties": 0,
             "merged_off": None, "merged_on": None, "startup_seconds": None, "log": str(log_path)}
    print("tiny_gate [%s]: starting the server (log %s)" % (out.name, log_path), flush=True)
    popen = serve.start_server(command, serve.server_environment(overrides), log_path, out)
    base_url = "http://127.0.0.1:%d" % port
    try:
        started = time.monotonic()
        healthy = serve.wait_healthy(port, health_timeout, popen)
        entry["startup_seconds"] = round(time.monotonic() - started, 1)
        if not healthy:
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else None
            entry["problems"].append("not healthy: %s" %
                                     serve.unhealthy_reason(popen, entry["startup_seconds"], health_timeout)[1])
            entry["errors"] = serve.error_lines(log_text)
            return entry
        streams = {}
        for arm, rows in (("off", 0), ("on", ROWS)):
            write_switch(switch, rows)
            time.sleep(1.0)
            before = merged_lines(log_path)
            print("tiny_gate [%s]: arm %s" % (out.name, arm), flush=True)
            streams[arm] = run_arm(base_url, arm, reference, seed=20261121)
            entry["merged_" + arm] = merged_lines(log_path) - before
        (out / "streams.json").write_text(json.dumps(streams), encoding="utf-8")
        problems = compare_reference(reference, streams["off"])
        problems += ["on: " + p for p in compare_reference(reference, streams["on"])]
        arm_problems, ties = compare_arms(streams["off"], streams["on"])
        problems += arm_problems
        if entry["merged_on"] is None or merged_lines(log_path) < 1:
            problems.append("no merged prompt forward in the arm on (the flag did nothing)")
        if entry["merged_off"]:
            problems.append("merged prompt forwards in the arm off: %s" % entry["merged_off"])
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if problems else None
        entry.update(problems=problems, near_ties=ties, status="passed" if not problems else "failed",
                     errors=serve.error_lines(log_text) if problems else [])
        return entry
    finally:
        serve.stop_server(popen)  # also on an exception: the workers must not keep the cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.environ.get("FARM_OUT") or "prefill-batch-tiny")
    parser.add_argument("--ckpt", default="/tmp/fn-smoke-ckpt")
    parser.add_argument("--mtp", default="0,2", help="speculative sizes, one server each (0: none)")
    parser.add_argument("--tp", type=int, default=int(os.environ.get("FARM_NCARDS") or 8))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FARM_PORT_BASE") or 8000) + 30)
    parser.add_argument("--health-timeout", type=float, default=900.0)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE added to every server")
    args = parser.parse_args(argv)
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    reason = tiny_smoke.build_checkpoint(ckpt)
    if reason:
        print("tiny_gate: no tiny checkpoint: %s" % reason, flush=True)
        return 1
    reference = json.loads((ckpt / "reference.json").read_text(encoding="utf-8"))
    extra = dict(kv.split("=", 1) for kv in args.env)
    results = []
    for k in [int(v) for v in args.mtp.split(",") if v != ""]:
        name = "mtp%d" % k if k else "plain"
        entry = run_server(k, ckpt, out / name, args.port, args.tp, args.health_timeout, extra, reference)
        results.append(entry)
        print("tiny_gate [%s]: %s; merged prompt forwards off %s, on %s; near ties %s" %
              (name, entry["status"].upper(), entry["merged_off"], entry["merged_on"], entry["near_ties"]),
              flush=True)
        for line in entry["problems"][:12]:
            print("   problem: %s" % line, flush=True)
        (out / "summary.json").write_text(json.dumps({"servers": results}, indent=1) + "\n", encoding="utf-8")
    failed = sum(1 for entry in results if entry["status"] != "passed")
    print("tiny_gate: %d of %d servers failed" % (failed, len(results)), flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
