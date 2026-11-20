#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The tiny-model card rehearsal of the lookup proposers (n-gram with prefix caching, the lookup + MTP hybrid).

A first-prompt crash or a verification crash does not need the real model's weight load to show: this runs the
speculative paths on the tiny checkpoint in a few minutes per server. Inside a farm container, from the tree:

    python tools/flash_next/rental/lookup_tiny.py [--configs LIST] [--tp N] [--tokens N] [--out DIR] [--port P]

1. It serves the tiny checkpoint WITHOUT speculation (prefix caching on, the real-model job's decode flags) and
   records the greedy stream of every scenario's request (the baseline), then once more with the five best
   log-probabilities of every token (for near-ties).
2. It writes the oracle file: every request's prompt plus its baseline output. The random tiny model never
   repeats its context, so only an oracle gets drafts accepted (``flash_next_lookup.LookupOracle``).
3. Per configuration it serves with the speculative config, runs the same scenarios, reads the speculative
   counters of ``/metrics`` and compares each stream with the baseline: equal up to its first difference, which
   must be a near-tie of the baseline (the best two log-probabilities within ``TIE``; bfloat16 logits of a
   random model are nearly flat, and two correct computations in different batch shapes can pick either).

Configurations (``--configs``, default ngram3,ngram8,mtp3-lookup,ngram3-search), one server each:
  ngram3         n-gram k = 3, oracle drafts on the ragged schedule 3,1,0,2,3,3,3, a tampered draft every 4th
                 context length: every accepted count 0..3, empty drafts, steps widened after a full acceptance
  ngram8         n-gram k = 8 (9-row verification, the torch GDN recurrence), schedule 8,4,0,6,8,3,7,1,5,2
  mtp3-lookup    MTP k = 3 with VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP=1 and the oracle of ngram3, three passes over the
                 scenarios with the request's own mode (vllm_xargs spec_lookup): hybrid (lookup rows padded with
                 -1 to k, and MTP rows, in the same streams), lookup-only (lookup rows or k empty drafts), mtp
                 (the MTP drafts alone: what the other two share with plain MTP). It verifies with the fused
                 GDN verification op (VLLM_HPU_GDN_CORE_VERIFY=1), the path the MTP track runs on the card; the
                 n-gram configurations verify with the torch recurrence (their steps are ragged, and 9 rows are
                 more than the op takes)
  ngram3-fused   ngram3 with the fused GDN verification op (rows of 2 to 4 tokens)
  ngram3-search  n-gram k = 3 with vLLM's own search and prompt_lookup_min 1 (drafts on most steps, hardly any
                 accepted): the numba search in the workers, ragged widths without an oracle

A configuration passes when its server starts, every request of every pass is answered, no stream leaves the
baseline away from a near-tie, and (passes with oracle drafts) drafts were accepted. Writes ``lookup_tiny.json``,
one folder with ``server0.log`` per server, and one summary line per pass. Exit code: 0 all passed, 1 otherwise.
Standard library and the rental helpers only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[3]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from tools.flash_next.rental import gates, serve, spec_probe  # noqa: E402
from tools.flash_next.rental import main as orchestrator  # noqa: E402

TIE = 0.0045  # nats: two bfloat16 steps of a logit near 0.3 (tests/unit_tests/models/flash_next/test_lookup_engine)
TOP = 5
ORACLE_ENV = "VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE"
LOOKUP_ENV = "VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP"
# The decode flags of the real-model job (lookup_serve.sh), so the tiny server runs the same paths.
PRESET = {"PT_HPU_LAZY_MODE": "1", "PT_HPU_LAZY_ACC_PAR_MODE": "0", "VLLM_HPU_FLASH_NEXT_QSA_MODE": "1",
          "VLLM_HPU_RECURRENT_STATE_POOL": "1", "VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_HC_FUSED": "1",
          "VLLM_HPU_PLE_HOST_NGRAM_IDS": "1", "VLLM_HPU_COLLECTIVE_MARK_STEP": "0", "VLLM_SKIP_WARMUP": "true",
          "VLLM_HPU_GDN_CORE_VERIFY": "0"}
NGRAM = {"method": "ngram", "prompt_lookup_max": 8, "prompt_lookup_min": 3}
MTP3 = {"method": "mtp", "num_speculative_tokens": 3}
RAGGED = {"schedule": [3, 1, 0, 2, 3, 3, 3], "tamper_every": 4}
# name -> the server's speculative config and environment, its oracle, and its passes over the scenarios:
# (pass name, the request's spec_lookup mode or None, drafts must be accepted)
CONFIGS = {
    "ngram3": {"spec": dict(NGRAM, num_speculative_tokens=3), "oracle": RAGGED, "passes": [("ngram3", None, True)]},
    "ngram8": {"spec": dict(NGRAM, num_speculative_tokens=8),
               "oracle": {"schedule": [8, 4, 0, 6, 8, 3, 7, 1, 5, 2], "tamper_every": 5},
               "passes": [("ngram8", None, True)]},
    "mtp3-lookup": {"spec": MTP3, "env": {LOOKUP_ENV: "1", "VLLM_HPU_GDN_CORE_VERIFY": "1"}, "oracle": RAGGED,
                    "passes": [("hybrid", 1, True), ("lookup-only", 2, True), ("mtp", 0, False)]},
    "ngram3-fused": {"spec": dict(NGRAM, num_speculative_tokens=3), "env": {"VLLM_HPU_GDN_CORE_VERIFY": "1"},
                     "oracle": RAGGED, "passes": [("ngram3-fused", None, True)]},
    "ngram3-search": {"spec": dict(NGRAM, num_speculative_tokens=3, prompt_lookup_min=1), "oracle": None,
                      "passes": [("ngram3-search", None, False)]},
}


DEFAULT_CONFIGS = "ngram3,ngram8,mtp3-lookup,ngram3-search"


def scenarios(reference: dict, tokens: int) -> list[dict]:
    """The requests of the rehearsal, in batches sent together: single prompts of 7 to 700 tokens, prompts that
    end just before a 256- and a 512-token boundary, four requests of different lengths in one batch, and a
    second turn that starts with a prefix-cache hit on an earlier request (``turn_of``)."""
    by_len = {len(p["prompt_ids"]): p["prompt_ids"] for p in reference["prompts"]}
    long_prompt = by_len[700]
    near_256, near_512 = long_prompt[:250], long_prompt[:505]
    out = [{"name": f"solo-{n}", "requests": [{"id": f"solo-{n}", "prompt_ids": by_len[n], "max_tokens": tokens}]}
           for n in (7, 33, 100, 700)]
    out += [
        {"name": "near-256", "requests": [{"id": "near-256", "prompt_ids": near_256, "max_tokens": tokens}]},
        {"name": "near-512", "requests": [{"id": "near-512", "prompt_ids": near_512, "max_tokens": tokens}]},
        {"name": "batch", "requests": [{"id": f"batch-{i}", "prompt_ids": p, "max_tokens": max(8, tokens - 6 * i)}
                                       for i, p in enumerate((by_len[7], by_len[33], near_256, by_len[100]))]},
        {"name": "turn-2", "requests": [{"id": "turn-2", "turn_of": "near-256", "suffix_ids": by_len[33][:20],
                                         "max_tokens": tokens}]},
    ]
    return out


def request_prompt(request: dict, baseline: dict) -> list[int]:
    """The prompt ids of ``request``; a ``turn_of`` request continues that request's prompt and baseline output."""
    if "turn_of" in request:
        prev = baseline[request["turn_of"]]
        return list(prev["prompt_ids"]) + list(prev["tokens"]) + list(request["suffix_ids"])
    return list(request["prompt_ids"])


def top_gaps(choice: dict) -> list[list]:
    """Per generated token the best tokens with their distance to the best log-probability:
    ``[[token id, gap], ...]`` (from ``logprobs.top_logprobs`` with ``return_tokens_as_token_ids``)."""
    out = []
    for entry in ((choice.get("logprobs") or {}).get("top_logprobs") or []):
        ranked = sorted(((float(v), k) for k, v in (entry or {}).items()), reverse=True)
        row = []
        for value, key in ranked[:TOP]:
            try:
                row.append([int(str(key).split(":")[-1]), ranked[0][0] - value])
            except ValueError:
                continue
        out.append(row)
    return out


def run_scenarios(base_url: str, plan: list[dict], baseline: dict | None, logprobs: bool, timeout_s: float,
                  post=None, mode: int | None = None) -> dict:
    """Run every scenario (its requests together) and return ``{request id: {"prompt_ids", "tokens", "top",
    "error"}}``. ``baseline`` gives the prompts of ``turn_of`` requests (None while the baseline itself runs: the
    streams recorded so far are used). ``mode``: the requests' lookup mode (vllm_xargs spec_lookup), or None."""
    post = post or gates._post_json
    streams: dict[str, dict] = {}

    def one(request: dict, prompt: list[int]) -> None:
        body = {"model": "tiny", "prompt": prompt, "max_tokens": request["max_tokens"], "temperature": 0,
                "return_token_ids": True}
        if logprobs:
            body.update(logprobs=TOP, return_tokens_as_token_ids=True)
        if mode is not None:
            body["vllm_xargs"] = {"spec_lookup": mode}
        record = {"prompt_ids": prompt, "tokens": [], "top": [], "error": None}
        try:
            choice = (post(base_url, "/v1/completions", body, timeout_s).get("choices") or [{}])[0]
            record["tokens"] = [int(t) for t in (choice.get("token_ids") or [])]
            record["top"] = top_gaps(choice) if logprobs else []
            if len(record["tokens"]) != request["max_tokens"]:
                record["error"] = "%d tokens, not %d" % (len(record["tokens"]), request["max_tokens"])
        except Exception as e:  # noqa: BLE001 - one request's failure is recorded, the others still run
            record["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
        streams[request["id"]] = record

    for scenario in plan:
        threads = []
        for request in scenario["requests"]:
            source = baseline if baseline is not None else streams
            if "turn_of" in request and (request["turn_of"] not in source or source[request["turn_of"]]["error"]):
                streams[request["id"]] = {"prompt_ids": [], "tokens": [], "top": [],
                                          "error": "its first turn %s has no stream" % request["turn_of"]}
                continue
            threads.append(threading.Thread(target=one, args=(request, request_prompt(request, source)),
                                            daemon=True))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout_s + 30)
    return streams


def compare(baseline: dict, tops: dict, streams: dict) -> dict:
    """``streams`` against ``baseline``: per request the first differing token is a tie when the speculative
    token is among the baseline's recorded best tokens there within ``TIE`` (``tops``: the baseline's
    log-probability record; without one the difference is ``unclassified``), else a break."""
    out = {"matched": 0, "identical": 0, "requests": 0, "ties": [], "breaks": [], "unclassified": [], "errors": []}
    for request_id, got in streams.items():
        want = baseline.get(request_id)
        if got["error"] or want is None or want["error"]:
            out["errors"].append([request_id, got["error"] or "no baseline stream"])
            continue
        out["requests"] += 1
        first = next((i for i, (a, b) in enumerate(zip(got["tokens"], want["tokens"])) if a != b), None)
        if first is None:
            out["matched"] += len(want["tokens"])
            out["identical"] += 1
            continue
        out["matched"] += first
        top = (tops.get(request_id) or {}).get("top") or []
        if first >= len(top) or not top[first]:
            out["unclassified"].append([request_id, first, got["tokens"][first]])
            continue
        gap = {int(t): float(d) for t, d in top[first]}.get(got["tokens"][first])
        entry = [request_id, first, got["tokens"][first], None if gap is None else round(gap, 5)]
        (out["ties"] if gap is not None and gap <= TIE else out["breaks"]).append(entry)
    return out


def serve_command(ckpt: Path, port: int, tp: int, spec: dict | None) -> list[str]:
    """The tiny gate's server command at ``tp`` with prefix caching on and the speculative config."""
    cmd = [str(word) for word in orchestrator.tiny_serve_command(ckpt, port)]
    i = cmd.index("--tensor-parallel-size")
    cmd[i + 1] = str(tp)
    cmd.append("--enable-prefix-caching")
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    return cmd


class Server:
    """One tiny server: ``with Server(...) as base_url`` (None when it did not get healthy)."""

    def __init__(self, name: str, ckpt: Path, out_dir: Path, port: int, tp: int, spec: dict | None, env: dict,
                 health_timeout_s: float) -> None:
        self.name, self.out_dir, self.port, self.health_timeout_s = name, out_dir / name, port, health_timeout_s
        self.command = serve_command(ckpt, port, tp, spec)
        self.env = serve.server_environment(dict(PRESET, **env))
        self.popen = None
        self.startup_seconds = None
        self.reason = None

    def __enter__(self) -> str | None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if serve.port_in_use(self.port):
            self.reason = "port %d already in use" % self.port
            return None
        print("lookup tiny [%s]: starting the server: %s" % (self.name, " ".join(self.command)), flush=True)
        started = time.monotonic()
        self.popen = serve.start_server(self.command, self.env, self.out_dir / "server0.log", self.out_dir)
        healthy = serve.wait_healthy(self.port, self.health_timeout_s, self.popen)
        self.startup_seconds = round(time.monotonic() - started, 1)
        if not healthy:
            _status, self.reason = serve.unhealthy_reason(self.popen, self.startup_seconds, self.health_timeout_s)
            return None
        print("lookup tiny [%s]: healthy after %.0f s" % (self.name, self.startup_seconds), flush=True)
        return "http://127.0.0.1:%d" % self.port

    def __exit__(self, *exc) -> None:
        if self.popen is not None:
            serve.stop_server(self.popen)  # also on an exception: the workers must not keep the cards

    def log_text(self) -> str:
        try:
            return (self.out_dir / "server0.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def counters(base_url: str) -> dict | None:
    try:
        return spec_probe.read_spec_counters(base_url)
    except OSError:
        return None


def run_config(name: str, config: dict, ckpt: Path, out_dir: Path, port: int, tp: int, plan: list[dict],
               baseline: dict, tops: dict, args) -> dict:
    """One server of ``config`` and its passes over the scenarios; the configuration's entry."""
    entry = {"config": name, "status": "failed", "reason": None, "spec": config["spec"], "passes": []}
    env = dict(config.get("env") or {})
    if config.get("oracle"):
        oracle_path = out_dir / ("oracle-%s.json" % name)
        oracle_path.write_text(json.dumps({
            "streams": [s["prompt_ids"] + s["tokens"] for s in baseline.values() if not s["error"]],
            "schedule": config["oracle"]["schedule"], "tamper_every": config["oracle"]["tamper_every"]}),
                               encoding="utf-8")
        env[ORACLE_ENV] = str(oracle_path)
    server = Server(name, ckpt, out_dir, port, tp, config["spec"], env, args.health_timeout)
    with server as base_url:
        entry["startup_seconds"] = server.startup_seconds
        if base_url is None:
            entry["reason"] = "the server did not start: %s" % server.reason
        for pass_name, mode, wants_accepted in (config["passes"] if base_url is not None else []):
            before = counters(base_url)
            started = time.monotonic()
            streams = run_scenarios(base_url, plan, baseline, False, args.timeout, mode=mode)
            one = {"pass": pass_name, "mode": mode, "status": "failed", "reason": None,
                   "seconds": round(time.monotonic() - started, 1),
                   "acceptance": spec_probe.spec_summary(spec_probe.counters_delta(before, counters(base_url))),
                   "compare": compare(baseline, tops, streams),
                   "streams": {k: v["tokens"] for k, v in streams.items()}}
            result, accepted = one["compare"], (one["acceptance"] or {}).get("accepted") or 0
            if result["errors"]:
                one["reason"] = "%d request(s) failed: %s" % (len(result["errors"]), result["errors"][0])
            elif result["breaks"]:
                one["reason"] = "stream(s) leave the baseline away from a near-tie: %s" % result["breaks"][:3]
            elif wants_accepted and accepted <= 0:
                one["reason"] = "no draft was accepted (the oracle's drafts are the baseline's own tokens)"
            else:
                one["status"] = "passed"
            entry["passes"].append(one)
            print(pass_line(name, one), flush=True)
            if result["errors"] and server.popen is not None and server.popen.poll() is not None:
                break  # the server is gone: the later passes would only time out
    log_text = server.log_text()
    entry["errors"] = serve.error_lines(log_text)
    entry["fallback_lines"] = gates.gdn_fallback_lines(log_text)[:8]
    entry["hybrid_lines"] = [ln.strip()[-300:] for ln in log_text.splitlines() if "Flash-Next lookup hybrid" in ln][-6:]
    entry["oracle_line"] = any("REHEARSAL ORACLE" in ln for ln in log_text.splitlines())
    if entry["reason"] is None:
        failed = [one for one in entry["passes"] if one["status"] != "passed"]
        if config.get("oracle") and not entry["oracle_line"]:
            entry["reason"] = "the server did not load the oracle (no REHEARSAL ORACLE line in its log)"
        elif failed:
            entry["reason"] = "pass %s: %s" % (failed[0]["pass"], failed[0]["reason"])
        elif len(entry["passes"]) != len(config["passes"]):
            entry["reason"] = "only %d of %d passes ran" % (len(entry["passes"]), len(config["passes"]))
        else:
            entry["status"] = "passed"
    return entry


def pass_line(config: str, one: dict) -> str:
    acceptance, result = one.get("acceptance") or {}, one.get("compare") or {}
    rates = ",".join("%.2f" % r for r in acceptance.get("position_rates") or [] if r is not None)
    length = acceptance.get("acceptance_length")
    return ("lookup tiny [%s/%s]: %s%s; %s s; tokens/step %s (pos %s), drafts %s, accepted %s; streams identical "
            "%s/%s, %s tokens matched, ties %d, unclassified %d, breaks %d" % (
                config, one["pass"], one["status"].upper(), (" (%s)" % one["reason"]) if one["reason"] else "",
                one.get("seconds"), ("%.2f" % length) if length else "-", rates or "-", acceptance.get("drafts"),
                acceptance.get("accepted"), result.get("identical"), result.get("requests"), result.get("matched"),
                len(result.get("ties") or []), len(result.get("unclassified") or []), len(result.get("breaks") or [])))


def summary_line(entry: dict) -> str:
    return "lookup tiny [%s]: %s%s; start %s s; %d pass(es); fallback lines %d; oracle loaded: %s" % (
        entry["config"], entry["status"].upper(), (" (%s)" % entry["reason"]) if entry["reason"] else "",
        entry.get("startup_seconds"), len(entry["passes"]), len(entry.get("fallback_lines") or []),
        entry.get("oracle_line"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="comma-separated, of: %s" % ", ".join(CONFIGS))
    ap.add_argument("--tp", type=int, default=int(os.environ.get("FARM_NCARDS") or 8))
    ap.add_argument("--tokens", type=int, default=48, help="tokens generated per request (default 48)")
    ap.add_argument("--out", default=os.environ.get("FARM_OUT") or "/results/lookup-tiny")
    ap.add_argument("--port", type=int, default=int(os.environ.get("FARM_PORT_BASE") or 8000) + 10)
    ap.add_argument("--ckpt", default="/tmp/fn-smoke-ckpt")
    ap.add_argument("--health-timeout", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    args = ap.parse_args(argv)
    names = [n for n in args.configs.split(",") if n]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        ap.error("unknown configuration(s) %s (of: %s)" % (unknown, ", ".join(CONFIGS)))
    out_dir, ckpt = Path(args.out), Path(args.ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"tp": args.tp, "tokens": args.tokens, "tie": TIE, "configs": [], "baseline": None}

    def save() -> None:
        (out_dir / "lookup_tiny.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    reason = orchestrator.ensure_tiny_checkpoint(ckpt, orchestrator.tiny_checkpoint_command(ckpt),
                                                 say=lambda text: print("lookup tiny: %s" % text, flush=True))
    if reason:
        print("lookup tiny: no tiny checkpoint: %s" % reason, flush=True)
        record["baseline"] = {"status": "failed", "reason": reason}
        save()
        return 1
    plan = scenarios(json.loads((ckpt / "reference.json").read_text(encoding="utf-8")), args.tokens)

    base = Server("baseline", ckpt, out_dir, args.port, args.tp, None, {}, args.health_timeout)
    baseline, tops = {}, {}
    with base as base_url:
        if base_url is not None:
            baseline = run_scenarios(base_url, plan, None, False, args.timeout)
            if not any(s["error"] for s in baseline.values()):
                # the log-probabilities last: a failure here costs the tie classification, not the baseline
                tops = run_scenarios(base_url, plan, baseline, True, args.timeout)
    failed = [[k, v["error"]] for k, v in baseline.items() if v["error"]]
    drift = [k for k, v in tops.items() if not v["error"] and v["tokens"] != baseline[k]["tokens"]]
    record["baseline"] = {"status": "passed" if baseline and not failed else "failed",
                          "reason": base.reason or (("request(s) failed: %s" % failed[:2]) if failed else None),
                          "startup_seconds": base.startup_seconds, "errors": serve.error_lines(base.log_text()),
                          "streams": {k: v["tokens"] for k, v in baseline.items()},
                          "prompt_lengths": {k: len(v["prompt_ids"]) for k, v in baseline.items()},
                          "logprob_requests_failed": [[k, v["error"]] for k, v in tops.items() if v["error"]],
                          "streams_that_differ_with_logprobs": drift}
    save()
    print("lookup tiny [baseline]: %s%s; start %s s; %d streams; with log-probabilities: %d failed, %d differ" % (
        record["baseline"]["status"].upper(),
        (" (%s)" % record["baseline"]["reason"]) if record["baseline"]["reason"] else "", base.startup_seconds,
        len(baseline), len(record["baseline"]["logprob_requests_failed"]), len(drift)), flush=True)
    if record["baseline"]["status"] != "passed":
        for line in record["baseline"]["errors"]:
            print("lookup tiny [baseline]:   %s" % line[:300], flush=True)
        return 1
    # a stream that changed when log-probabilities were asked for gives no tie record for its request
    tops = {k: v for k, v in tops.items() if not v["error"] and k not in drift}

    for name in names:
        entry = run_config(name, CONFIGS[name], ckpt, out_dir, args.port, args.tp, plan, baseline, tops, args)
        record["configs"].append(entry)
        save()
        print(summary_line(entry), flush=True)
        for line in (entry.get("errors") or [])[:5]:
            print("lookup tiny [%s]:   %s" % (name, line[:300]), flush=True)
        for line in entry.get("hybrid_lines") or []:
            print("lookup tiny [%s]:   %s" % (name, line), flush=True)
    passed = sum(1 for e in record["configs"] if e["status"] == "passed")
    print("lookup tiny: %d of %d configurations passed" % (passed, len(record["configs"])), flush=True)
    return 0 if passed == len(record["configs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
