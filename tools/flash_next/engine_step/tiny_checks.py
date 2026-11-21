#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engine-step one-card job: the tiny model with one per-step flag at a time, and the logprobs check.

Run inside a farm container from the track's tree (``farm job engine-step <name> any:1 15m -- python3
tools/flash_next/engine_step/tiny_checks.py``). For each arm the farm's tiny smoke (``/farm/farm_tiny.py --gdn hc``)
serves the tiny checkpoint with the arm's flag and compares its greedy tokens with the CPU reference; while that
server is still up, the reference prompts go out once more with ``logprobs=5`` (``return_tokens_as_token_ids``)
and one with ``prompt_logprobs=2``:

* every request answers HTTP 200 and ``/health`` answers after them (job 00086: a ``logprobs=5`` request killed
  the server's output handler with ``OverflowError`` on garbled top-k ids);
* every top-logprob id is an id of the vocabulary;
* the generated token is the first of its position's top logprobs (greedy);
* the first position's top 5 against the reference's ``last_prefill_top5`` (set overlap, recorded only: the
  tiny model's logits are near-flat, its 4th and 5th may swap).

Arms: ``i01`` = ``VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1``, ``i05`` = ``VLLM_HPU_FUSED_GREEDY_LOGITS=1``,
``base`` = neither, ``params`` = ``VLLM_HPU_SAMPLER_REQUEST_PARAMS=1`` on the general sampler path, with
``request_params_check`` (a ``logit_bias`` of +100 must win, a stop token before ``min_tokens`` must not appear,
a ``min_p`` request must answer). Job 00093 ran both flags at once: the 700-token prompt took another token at its 5th position
(a near tie of the tiny model resolved the other way by the in-graph greedy region, INFERRED from the same token
stream in collectives' and mtp-card-ready's runs): one flag at a time says which flag carries it. The exit code is
1 when a logprobs check failed, else 0 (the token comparison of each arm is reported, not judged here).
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ARMS = {
    "base": [],
    "i01": ["VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1"],
    "i05": ["VLLM_HPU_FUSED_GREEDY_LOGITS=1"],
    "both": ["VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1", "VLLM_HPU_FUSED_GREEDY_LOGITS=1"],
    # min_tokens / logit_bias / min_p on the general sampler path (the fused greedy region off)
    "params": ["VLLM_HPU_SAMPLER_REQUEST_PARAMS=1", "VLLM_HPU_FUSED_GREEDY_LOGITS=0"],
}
BIAS_TOKEN = 4242  # a token the tiny model never picks by itself
ID_LIMIT = 1 << 24  # every id of the tiny model's vocabulary is below it; a garbled id is not
CHECKS: list = []  # one record per arm, filled while its server is up


def _token_id(key):
    if isinstance(key, str) and key.startswith("token_id:"):
        try:
            return int(key.split(":", 1)[1])
        except ValueError:
            return None
    return None


def logprobs_check(gates, base_url: str, reference_path) -> dict:
    """The logprobs requests against a running tiny server (see the module docstring)."""
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    prompts = sorted(reference["prompts"], key=lambda p: len(p["prompt_ids"]))
    record = {"prompts": [], "ok": True}
    for p in prompts:
        entry = {"prompt_tokens": len(p["prompt_ids"])}
        record["prompts"].append(entry)
        try:
            resp = gates._post_json(base_url, "/v1/completions", {
                "model": "tiny", "prompt": p["prompt_ids"], "max_tokens": 4, "temperature": 0, "logprobs": 5,
                "return_tokens_as_token_ids": True}, 120.0)
        except Exception as e:  # noqa: BLE001 - recorded, the next prompt still runs
            entry["error"] = ("%s: %s" % (type(e).__name__, e))[:300]
            record["ok"] = False
            continue
        lp = (resp.get("choices") or [{}])[0].get("logprobs") or {}
        tokens, tops = lp.get("tokens") or [], lp.get("top_logprobs") or []
        ids = [[_token_id(k) for k in (top or {})] for top in tops]
        valid = bool(ids) and all(i is not None and 0 <= i < ID_LIMIT for row in ids for i in row)
        firsts = []
        for token, top in zip(tokens, tops):
            best = max(top.items(), key=lambda kv: kv[1])[0] if top else None
            firsts.append(bool(top) and (token == best or top.get(token) == top.get(best)))
        ref_top5 = (p.get("last_prefill_top5") or {}).get("ids") or []
        overlap = len(set(i for i in ids[0] if i is not None) & set(ref_top5)) if ids else 0
        entry.update(positions=len(tops), ids_valid=valid, sampled_is_top=bool(firsts) and all(firsts),
                     first_top=ids[0][:6] if ids else None, reference_top5=ref_top5[:5], top5_overlap=overlap)
        if not (valid and entry["sampled_is_top"]):
            record["ok"] = False
    short = prompts[1] if len(prompts) > 1 else prompts[0]
    try:
        resp = gates._post_json(base_url, "/v1/completions", {
            "model": "tiny", "prompt": short["prompt_ids"], "max_tokens": 2, "temperature": 0,
            "prompt_logprobs": 2}, 120.0)
        plp = (resp.get("choices") or [{}])[0].get("prompt_logprobs") or []
        keys = [int(k) for pos in plp if pos for k in pos if str(k).lstrip("-").isdigit()]
        record["prompt_logprobs"] = {"positions": len(plp), "ids": len(keys),
                                     "ids_valid": bool(keys) and all(0 <= k < ID_LIMIT for k in keys)}
        record["ok"] = record["ok"] and record["prompt_logprobs"]["ids_valid"]
    except Exception as e:  # noqa: BLE001
        record["prompt_logprobs"] = {"error": ("%s: %s" % (type(e).__name__, e))[:300]}
        record["ok"] = False
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=30) as resp:
            record["health_after"] = resp.status
    except Exception as e:  # noqa: BLE001
        record["health_after"] = repr(e)[:200]
    record["ok"] = record["ok"] and record["health_after"] == 200
    return record


def request_params_check(gates, base_url: str, reference_path) -> dict:
    """VLLM_HPU_SAMPLER_REQUEST_PARAMS=1 on the general sampler path: a logit_bias of +100 makes a greedy request
    write only that token; a stop token the model would emit before min_tokens is not emitted; a min_p request
    answers; /health after."""
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    prompts = sorted(reference["prompts"], key=lambda p: len(p["prompt_ids"]))
    record = {"ok": True}

    def post(payload):
        return gates._post_json(base_url, "/v1/completions", dict({"model": "tiny", "return_token_ids": True},
                                                                   **payload), 120.0)

    short = next(p for p in prompts if len(p["prompt_ids"]) == 7)
    try:
        resp = post({"prompt": short["prompt_ids"], "max_tokens": 6, "temperature": 0,
                     "logit_bias": {str(BIAS_TOKEN): 100.0}})
        got = (resp.get("choices") or [{}])[0].get("token_ids") or []
        record["logit_bias"] = {"tokens": got, "ok": got == [BIAS_TOKEN] * 6}
    except Exception as e:  # noqa: BLE001
        record["logit_bias"] = {"error": ("%s: %s" % (type(e).__name__, e))[:300], "ok": False}
    longer = next(p for p in prompts if 30 <= len(p["prompt_ids"]) <= 700 and len(p["generated_ids"]) >= 8
                  and p["generated_ids"][2] not in p["generated_ids"][:2])
    stop = longer["generated_ids"][2]
    try:
        resp = post({"prompt": longer["prompt_ids"], "max_tokens": 6, "min_tokens": 6, "temperature": 0,
                     "stop_token_ids": [stop]})
        got = (resp.get("choices") or [{}])[0].get("token_ids") or []
        record["min_tokens"] = {"tokens": got, "stop": stop, "ok": stop not in got and len(got) == 6}
    except Exception as e:  # noqa: BLE001
        record["min_tokens"] = {"error": ("%s: %s" % (type(e).__name__, e))[:300], "ok": False}
    try:
        resp = post({"prompt": short["prompt_ids"], "max_tokens": 8, "temperature": 1.0, "top_k": 20,
                     "min_p": 0.5})
        got = (resp.get("choices") or [{}])[0].get("token_ids") or []
        record["min_p"] = {"tokens": got, "ok": len(got) == 8}
    except Exception as e:  # noqa: BLE001
        record["min_p"] = {"error": ("%s: %s" % (type(e).__name__, e))[:300], "ok": False}
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=30) as resp:
            record["health_after"] = resp.status
    except Exception as e:  # noqa: BLE001
        record["health_after"] = repr(e)[:200]
    record["ok"] = all(record[k]["ok"] for k in ("logit_bias", "min_tokens", "min_p")) and \
        record["health_after"] == 200
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="i01,i05")
    ap.add_argument("--out", default=os.environ.get("FARM_OUT", "/results/tiny-checks"))
    args = ap.parse_args()
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    sys.path.insert(0, "/farm")
    from tools.flash_next.rental import gates  # the same module object tiny_smoke calls
    import farm_tiny

    original = gates.tiny_compare

    current = {"arm": None}

    def compare_then_logprobs(base_url, reference_path):
        status, details = original(base_url, reference_path)
        check = logprobs_check(gates, base_url, reference_path)
        if current["arm"] == "params":
            params = request_params_check(gates, base_url, reference_path)
            check["request_params"] = params
            check["ok"] = check["ok"] and params["ok"]
            print("PARAMS %s: %s" % ("ok" if params["ok"] else "FAILED", json.dumps(params)), flush=True)
        CHECKS.append(check)
        details["logprobs_check"] = check
        return status, details

    gates.tiny_compare = compare_then_logprobs
    failed = 0
    for number, arm in enumerate([a for a in args.arms.split(",") if a]):
        if number:
            time.sleep(8)  # the last server's workers have let go of the card (CARD.md, two steps on one card)
        current["arm"] = arm
        env = [w for kv in ARMS[arm] for w in ("--env", kv)]
        out = str(Path(args.out) / ("tiny-" + arm))
        print("== tiny model, arm %s: %s" % (arm, " ".join(ARMS[arm]) or "no engine-step flag"), flush=True)
        before = len(CHECKS)
        rc = farm_tiny.main(["--gdn", "hc", "--out", out] + env)
        summary = Path(out) / "hc" / "details.json"
        if summary.exists():
            details = json.loads(summary.read_text())
            for p in details.get("prompts", []):
                if not p.get("full_match"):
                    print("TOKENS %s: %d-token prompt: %s of %s match, first difference at %s%s; got %s want %s" % (
                        arm, p.get("prompt_tokens"), p.get("match"), p.get("expected"), p.get("first_diff"),
                        " (near tie at token 0)" if p.get("near_tie") else "", (p.get("got") or [])[:6],
                        (p.get("want") or [])[:6]), flush=True)
        check = CHECKS[before] if len(CHECKS) > before else None
        if check is None:
            print("LOGPROBS %s: not run (the server did not come up)" % arm, flush=True)
            failed += 1
        else:
            print("LOGPROBS %s: %s; /health after: %s; prompt_logprobs %s" % (
                arm, "ok" if check["ok"] else "FAILED", check.get("health_after"),
                json.dumps(check.get("prompt_logprobs"))), flush=True)
            for entry in check["prompts"]:
                print("  %s" % json.dumps(entry), flush=True)
            failed += 0 if check["ok"] else 1
        print("tiny arm %s: smoke exit %d" % (arm, rc), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "logprobs_checks.json").write_text(json.dumps(CHECKS, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
