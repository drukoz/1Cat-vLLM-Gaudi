# SPDX-License-Identifier: Apache-2.0
"""Sampled requests against the tiny model's speculative server on a card-farm job's cards (idea-sweep I07).

The owner's agents sample: a request without a temperature gets the checkpoint's generation_config (temperature
1.0, top_k 20, top_p 0.95). Every card probe before this one was greedy. Run it from the track's tree:

    farm job mtp-card-ready NAME any:2 15m -- python tools/flash_next/rental/mtp_sampled_card.py [--k 3] [--arms ...]

Arms, each a tiny server in mode hc at the job's tensor parallel size (FARM_NCARDS, or --tp):

* ``plain``: no speculation. Sampled requests must be served: sampling itself runs on the card.
* ``refuse``: MTP k with VLLM_HPU_SPEC_SAMPLED_VERIFY=0. A sampled request must come back as HTTP 400 naming the
  flag, and the greedy request after it must still reproduce the reference: the engine survived the refusal.
* ``sampled``: MTP k with VLLM_HPU_SPEC_SAMPLED_VERIFY=1. Sampled requests are served; a batch that mixes greedy
  and sampled requests serves the greedy one as the reference; the counters of /metrics show the drafts and what
  was accepted; a greedy request afterwards still reproduces the reference (the recurrent state of a request with
  accepted drafts did not leak); last, a seed gives the same stream twice.

The tiny checkpoint has random weights, so its acceptance says nothing about the real model: with top_k 20 a
draft is accepted now and then, which is what puts accepted counts above 0 through the verification path.

Results: ``OUT/<arm>/server0.log`` and ``OUT/summary.json``. Exit code: the number of arms that failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from tools.flash_next.rental import gates, serve, spec_probe, tiny_smoke  # noqa: E402

ARMS = ("plain", "refuse", "sampled")
SAMPLING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}  # the checkpoint's generation_config
SAMPLED_FLAG = "VLLM_HPU_SPEC_SAMPLED_VERIFY"
TP_FLAG = "--tensor-parallel-size"
COMMON_ENV = {"VLLM_HPU_PLE_HOST_NGRAM_IDS": "1", "PT_HPU_LAZY_ACC_PAR_MODE": "0",
              "VLLM_HPU_COLLECTIVE_MARK_STEP": "0", "VLLM_SKIP_WARMUP": "true"}
HTTP_TIMEOUT_S = 600.0


def post(base_url: str, payload: dict, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[int, dict | str]:
    """``(status, parsed JSON or the body's first 600 characters)`` of a completion request; 0 on no answer."""
    request = urllib.request.Request(base_url + "/v1/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - the job's own server
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")[:600]
    except (OSError, ValueError) as error:
        return 0, repr(error)[:600]


def completion(prompt_ids: list[int], max_tokens: int, sampled: bool, seed: int | None = None) -> dict:
    payload = {"model": "tiny", "prompt": prompt_ids, "max_tokens": max_tokens, "return_token_ids": True}
    if sampled:
        payload.update(SAMPLING)
        if seed is not None:
            payload["seed"] = seed
    else:
        payload["temperature"] = 0
    return payload


def token_ids(answer) -> list[int] | None:
    if not isinstance(answer, dict):
        return None
    return ((answer.get("choices") or [{}])[0]).get("token_ids")


def with_tp(command: list, tp: int) -> list:
    command = [str(word) for word in command]
    if TP_FLAG in command:
        command[command.index(TP_FLAG) + 1] = str(tp)
    else:
        command += [TP_FLAG, str(tp)]
    return command


def check(checks: list, name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print("  %s %s%s" % ("ok  " if ok else "FAIL", name, (": " + str(detail)[:300]) if detail else ""), flush=True)
    return bool(ok)


def greedy_reference(base_url: str, prompt: dict, checks: list, name: str) -> None:
    status, answer = post(base_url, completion(prompt["prompt_ids"], gates.TINY_MAX_TOKENS, sampled=False))
    got = token_ids(answer)
    want = prompt["generated_ids"]
    first = None if got is None else next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    check(checks, name, status == 200 and got is not None and first is None and len(got) == len(want),
          "status %s, first difference %s of %d tokens" % (status, first, len(want)))


def run_arm(arm: str, k: int, ckpt: Path, out: Path, port: int, tp: int, max_tokens: int, health_timeout: float,
            reference: dict) -> dict:
    speculative = arm != "plain"
    overrides = dict(tiny_smoke.GDN_MODES["hc"])
    overrides.update(COMMON_ENV)
    if speculative:
        overrides[gates.GDN_VERIFY_ENV] = "1"
        overrides[SAMPLED_FLAG] = "1" if arm == "sampled" else "0"
    command = with_tp(tiny_smoke.serve_command(ckpt, port), tp)
    if speculative:
        command += tiny_smoke.spec_serve_args(k)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "server0.log"
    entry: dict = {"arm": arm, "k": k if speculative else 0, "tp": tp, "status": "failed", "checks": [],
                   "log": str(log_path), "startup_seconds": None, "acceptance": None}
    checks = entry["checks"]
    prompts = sorted(reference["prompts"], key=lambda p: len(p["prompt_ids"]))
    short, middle = prompts[1], prompts[3]  # 7 and 100 tokens in the tiny reference
    print("mtp_sampled_card [%s]: starting the server (log: %s)" % (arm, log_path), flush=True)
    popen = serve.start_server(command, serve.server_environment(overrides), log_path, out)
    base_url = "http://127.0.0.1:%d" % port
    try:
        started = time.monotonic()
        healthy = serve.wait_healthy(port, health_timeout, popen)
        entry["startup_seconds"] = round(time.monotonic() - started, 1)
        if not check(checks, "the server is healthy", healthy, "after %s s" % entry["startup_seconds"]):
            return entry
        greedy_reference(base_url, short, checks, "a greedy request reproduces the reference")
        before = spec_probe.read_spec_counters(base_url) if speculative else None
        status, answer = post(base_url, completion(middle["prompt_ids"], max_tokens, sampled=True))
        ids_a = token_ids(answer)
        if arm == "refuse":
            text = answer if isinstance(answer, str) else json.dumps(answer)
            check(checks, "a sampled request is refused with HTTP 400 naming the flag",
                  status == 400 and SAMPLED_FLAG in text, "status %s: %s" % (status, text[:200]))
        else:
            check(checks, "a sampled request is served", status == 200 and ids_a is not None
                  and len(ids_a) == max_tokens, "status %s, %s tokens%s" % (
                      status, None if ids_a is None else len(ids_a), "" if status == 200 else ": %s" % (answer, )))
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(post, base_url, completion(short["prompt_ids"], gates.TINY_MAX_TOKENS, False)),
                        pool.submit(post, base_url, completion(middle["prompt_ids"], max_tokens, True)),
                        pool.submit(post, base_url, completion(prompts[2]["prompt_ids"], max_tokens, True))]
                mixed = [job.result() for job in jobs]
            greedy_ids = token_ids(mixed[0][1])
            check(checks, "a batch of one greedy and two sampled requests is served",
                  all(status == 200 for status, _ in mixed), "statuses %s" % [status for status, _ in mixed])
            check(checks, "the greedy request of the mixed batch reproduces the reference",
                  greedy_ids == short["generated_ids"], "first difference %s" % (None if greedy_ids is None else next(
                      (i for i, (a, b) in enumerate(zip(greedy_ids, short["generated_ids"])) if a != b), None), ))
        if speculative:
            after = spec_probe.read_spec_counters(base_url)
            entry["acceptance"] = spec_probe.spec_summary(spec_probe.counters_delta(before, after))
            if arm == "sampled":
                drafts = (entry["acceptance"] or {}).get("drafts") or 0
                check(checks, "the sampled requests went through verification steps", drafts > 0,
                      json.dumps(entry["acceptance"]))
        greedy_reference(base_url, middle, checks, "a greedy request afterwards reproduces the reference")
        if arm != "refuse":
            # Seeded requests last: a per-request generator is one more thing a card has not run, and a failure
            # here must not hide the checks above.
            seeded = [post(base_url, completion(middle["prompt_ids"], max_tokens, sampled=True, seed=seed))
                      for seed in (11, 11, 12)]
            ids = [token_ids(answer) for _, answer in seeded]
            check(checks, "seeded sampled requests are served", all(status == 200 for status, _ in seeded),
                  "statuses %s" % [status for status, _ in seeded])
            check(checks, "the same seed gives the same stream", ids[0] is not None and ids[0] == ids[1],
                  "first difference %s" % (None if not ids[0] or not ids[1] else next(
                      (i for i, (a, b) in enumerate(zip(ids[0], ids[1])) if a != b), None), ))
            check(checks, "another seed gives another stream", ids[2] is not None and ids[2] != ids[0])
        status, _ = post(base_url, completion(short["prompt_ids"], 2, sampled=False), timeout_s=60)
        check(checks, "the server still answers", status == 200, "status %s" % status)
    finally:
        serve.stop_server(popen)
    entry["errors"] = serve.error_lines(log_path.read_text(encoding="utf-8", errors="replace"))[:8]
    entry["status"] = "passed" if all(c["ok"] for c in checks) else "failed"
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.environ.get("FARM_OUT") or "mtp-sampled")
    parser.add_argument("--ckpt", default="/tmp/fn-smoke-ckpt")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--tp", type=int, default=int(os.environ.get("FARM_NCARDS") or 8))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FARM_PORT_BASE") or 8000) + 20)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--health-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    arms = [arm for arm in args.arms.split(",") if arm]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        parser.error("unknown arm(s) %s; known: %s" % (unknown, ", ".join(ARMS)))
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    reason = tiny_smoke.build_checkpoint(ckpt)
    if reason:
        print("mtp_sampled_card: no tiny checkpoint: %s" % reason, flush=True)
        return len(arms)
    reference = json.loads((ckpt / "reference.json").read_text(encoding="utf-8"))
    results = []
    for arm in arms:
        print("=== mtp_sampled_card: arm %s (k = %d, tp %d)" % (arm, args.k, args.tp), flush=True)
        entry = run_arm(arm, args.k, ckpt, out / arm, args.port, args.tp, args.max_tokens, args.health_timeout,
                        reference)
        results.append(entry)
        print("=== mtp_sampled_card: arm %s %s; acceptance %s" % (arm, entry["status"].upper(), entry["acceptance"]),
              flush=True)
        for line in entry.get("errors") or []:
            print("   error line: %s" % str(line)[:300], flush=True)
        (out / "summary.json").write_text(json.dumps({"arms": results}, indent=1) + "\n", encoding="utf-8")
    failed = sum(1 for entry in results if entry["status"] != "passed")
    print("mtp_sampled_card: %d of %d arms failed" % (failed, len(results)), flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
