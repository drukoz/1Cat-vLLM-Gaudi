# SPDX-License-Identifier: Apache-2.0
"""The headline table of an integration card job: one row per measured round of every arm's scaling.json.

    python3 tools/flash_next/integration/headline_table.py OUT [--md FILE] [--json FILE]

OUT holds one folder per arm (nospec, mtp3, ...) as headline_job.sh leaves them. Per round:
* ms per step = the median gap between a stream's streamed chunks (one chunk per engine step), medianed over the
  streams: `steady_ms_per_token_p50 x tokens_per_chunk_p50` (the MTP track's reading of decode_scaling.py);
* tokens per step = `tokens_per_chunk_p50` (1 without speculation; with MTP the accepted drafts + 1);
* ms per token = `steady_ms_per_token_p50` (steady: no compile pause) and the span form `ms_per_token_p50`
  (the stream's whole decode span: compile pauses included);
* tok/s per stream = 1000 / steady ms per token; aggregate = all streams' tokens over the round's span;
* acceptance (speculation): the /metrics counters around the round (`spec`): acceptance length and the rate of
  every draft position.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


def _f(value, fmt="%.1f"):
    return "-" if value is None else fmt % value


def rows_of(arm: str, data: dict) -> list:
    rows = []
    for kind in ("levels", "sampled", "ctx"):
        for r in data.get(kind) or []:
            mode = {"levels": "greedy", "sampled": "sampled"}.get(kind) or "greedy, ~%s-token prompts" % (
                r.get("prompt_tokens_p50") or r.get("ctx_tokens"))
            # an A/B on one server (decode_scaling.py --switch-arm): the round's arm after the folder's
            row = {"arm": arm + ("/" + r["arm"] if r.get("arm") else ""), "mode": mode, "streams": r.get("streams"),
                   "rep": r.get("rep", 0), "ok": r.get("ok"), "errors": r.get("errors")}
            if r.get("ok"):
                steady = r.get("steady_ms_per_token_p50")
                tpc = r.get("tokens_per_chunk_p50") or 1.0
                row.update(ms_per_step=steady * tpc if steady else None, tokens_per_step=tpc, ms_per_token=steady,
                           ms_per_token_span=r.get("ms_per_token_p50"),
                           tok_s_per_stream=1000.0 / steady if steady else None,
                           aggregate_tok_s=r.get("aggregate_tok_s"), pause_s_max=r.get("pause_s_max"),
                           ttft_p50_s=r.get("ttft_p50_s"))
            spec = r.get("spec") or {}
            if spec:
                row.update(acceptance_length=spec.get("acceptance_length"),
                           position_rates=spec.get("position_rates"), drafts=spec.get("drafts"))
            row["samples"] = {k: (v or "")[:80] for k, v in (r.get("samples") or {}).items()}
            if r.get("error_samples"):
                row["error_samples"] = r["error_samples"]
            rows.append(row)
    return rows


MTP_LINE = re.compile(r"MTP step timing.*")


def mtp_timing(server_log: str) -> list:
    """The last MTP step timing lines of a server log (VLLM_HPU_MTP_TIMING=1)."""
    try:
        with open(server_log, errors="replace") as f:
            lines = [m.group(0)[:400] for m in (MTP_LINE.search(line) for line in f) if m]
    except OSError:
        return []
    return lines[-12:]


def table(rows: list) -> str:
    out = ["| server | streams | requests | ms per step | tokens per step | ms per token (steady) | ms per token (span) "
           "| tok/s per stream | aggregate tok/s | longest pause s | acceptance length | per position |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("ok"):
            out.append("| %s | %s | %s | FAILED (%s errors) | | | | | | | | |" % (
                r["arm"], r["streams"], r["mode"], r.get("errors")))
            continue
        rates = r.get("position_rates")
        out.append("| %s | %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["arm"], r["streams"], r["mode"], (" (rep %d)" % r["rep"]) if r.get("rep") else "",
            _f(r.get("ms_per_step")), _f(r.get("tokens_per_step"), "%.2f"), _f(r.get("ms_per_token")),
            _f(r.get("ms_per_token_span")), _f(r.get("tok_s_per_stream")), _f(r.get("aggregate_tok_s"), "%.0f"),
            _f(r.get("pause_s_max")), _f(r.get("acceptance_length"), "%.2f"),
            "-" if not rates else ", ".join("-" if x is None else "%.2f" % x for x in rates)))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="the job's results folder (one folder per arm)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    arms = sorted(d for d in os.listdir(args.out) if os.path.isfile(os.path.join(args.out, d, "scaling.json")))
    arms.sort(key=lambda a: (0 if a == "nospec" else 1, a))
    rows, timing = [], {}
    for arm in arms:
        try:
            with open(os.path.join(args.out, arm, "scaling.json")) as f:
                rows += rows_of(arm, json.load(f))
        except (OSError, ValueError) as error:
            print("headline_table: %s: %r" % (arm, error), file=sys.stderr)
        # the rounds of a post step on the same server (headline_job.sh: scaling_sw.json of the sw step)
        for extra in sorted(f for f in os.listdir(os.path.join(args.out, arm))
                            if f.startswith("scaling_") and f.endswith(".json")):
            try:
                with open(os.path.join(args.out, arm, extra)) as f:
                    rows += rows_of(arm + ":" + extra[len("scaling_"):-len(".json")], json.load(f))
            except (OSError, ValueError) as error:
                print("headline_table: %s/%s: %r" % (arm, extra, error), file=sys.stderr)
        timing[arm] = mtp_timing(os.path.join(args.out, arm, "server0.log"))
    md = [table(rows), ""]
    for r in rows:
        for k, t in (r.get("samples") or {}).items():
            md.append("- TEXT %s %s N=%s stream %s: %r" % (r["arm"], r["mode"], r["streams"], k, t))
    for arm, lines in timing.items():
        for line in lines:
            md.append("- %s: %s" % (arm, line))
    text = "\n".join(md)
    print(text)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "mtp_timing": timing}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
