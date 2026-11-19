# SPDX-License-Identifier: Apache-2.0
"""Per-variant summary and comparison report of a rental run (plan M4, RUK-2709).

``python -m tools.flash_next.rental.report RUN_DIR`` reads every variant folder
of a run (``status.json``, ``startup.json``, ``warm-pass.json``,
``requests.jsonl``, ``metrics.jsonl``), the gates' ``status.json`` files and
``manifest.json``, and
writes ``RUN_DIR/summary.json`` and ``RUN_DIR/summary.md``. Every file may be
missing (a stopped run): the report is still written, with nulls and the
reason. Statuses stay integers in the JSON; only the markdown turns them into
names. Python 3.10 standard library only.

``manifest.json`` (written by main.py, M4-H6) is read for these keys; unknown
keys are ignored and missing ones become null: ``run_id``, ``bundle_commit``,
``workload_sha``, ``mode``, ``created``, ``skip_warmup``, ``server_timeout_s``,
``recipe_cache`` (the RUK-3242 cache folder, size, combined-kernel-database flag
and per-variant notes), ``workload`` (the workload.json header, for ``turns``)
and ``variants`` (the variants.json entries that ran).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

from tools.flash_next.rental import serve
from tools.flash_next.rental.schema import (
    METRICS_FILE,
    REQUESTS_FILE,
    STATUS_FILE,
    STARTUP_FILE,
    GateStatus,
    RequestStatus,
)

SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"
MANIFEST_FILE = "manifest.json"
WARM_PASS_FILE = "warm-pass.json"

GATE_DIRS = ("preflight", "build", "tiny")

# rows of one monitor sample share `t` within this many seconds across servers
SAMPLE_WINDOW_S = 1.0

# a window counts as steady while the summed `running` is >= this share of agents
STEADY_RUNNING_FRACTION = 0.9


def _read_json(path: Path) -> tuple[Optional[Any], Optional[str]]:
    """Return (value, reason): value is None when the file is missing,
    unreadable or not valid JSON, and reason explains why."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "%s missing" % path.name
    except (OSError, ValueError) as exc:
        return None, "%s: %s" % (path.name, exc)


def _read_jsonl(path: Path) -> tuple[list, Optional[str]]:
    """Return (rows, reason): a missing/unreadable file gives ([], reason); a
    truncated (or corrupt) last line is dropped, not fatal."""
    lines, reason = _read_json_text_lines(path)
    if reason is not None:
        return [], reason
    rows = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i == len(lines) - 1:  # a row cut off by a killed run
                continue
            return rows, "%s: bad JSON on line %d" % (path.name, i + 1)
    return rows, None


def _read_json_text_lines(path: Path) -> tuple[list, Optional[str]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().splitlines(), None
    except FileNotFoundError:
        return [], "%s missing" % path.name
    except OSError as exc:
        return [], "%s: %s" % (path.name, exc)


def percentile(values: list, q: float) -> Optional[float]:
    """Nearest-rank percentile of a (possibly unsorted) list; None when empty."""
    if not values:
        return None
    s = sorted(values)
    idx = math.ceil(q * len(s)) - 1
    idx = min(max(idx, 0), len(s) - 1)
    return float(s[idx])


def _as_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _ttft_values(rows: list, warm: bool) -> list:
    out = []
    for row in rows:
        if row.get("status") != int(RequestStatus.OK):
            continue
        turn = row.get("turn")
        if turn is None:
            continue
        if (int(turn) >= 1) != warm:
            continue
        t_send, t_first = row.get("t_send"), row.get("t_first")
        if t_send is None or t_first is None:
            continue
        out.append(float(t_first) - float(t_send))
    return out


def _ttft_stats(rows: list, warm: bool) -> dict:
    values = _ttft_values(rows, warm)
    return {"p50": percentile(values, 0.5), "p90": percentile(values, 0.9), "max": max(values) if values else None}


def _decode_per_agent(rows: list) -> list:
    """Average decode tok/s for every agent that has at least one usable
    OK row: per row (output_tokens - 1) / (t_last - t_first), rows with a
    non-positive decode span skipped."""
    per_agent: dict = {}
    for row in rows:
        if row.get("status") != int(RequestStatus.OK):
            continue
        agent = row.get("agent")
        t_first, t_last = row.get("t_first"), row.get("t_last")
        if agent is None or t_first is None or t_last is None:
            continue
        span = float(t_last) - float(t_first)
        if span <= 0:
            continue
        tps = (int(row.get("output_tokens") or 0) - 1) / span
        per_agent.setdefault(int(agent), []).append(tps)
    return [sum(v) / len(v) for v in per_agent.values()]


def _group_samples(metrics: list) -> list:
    """Group metrics rows into samples: rows sorted by `t`, a new sample starts
    when `t` moves more than SAMPLE_WINDOW_S past the first row of the group.
    Each sample is {"t": mean t, "rows": [...]}."""
    rows = [r for r in metrics if r.get("t") is not None]
    rows.sort(key=lambda r: float(r["t"]))
    samples = []
    group = []
    for row in rows:
        if group and float(row["t"]) - float(group[0]["t"]) > SAMPLE_WINDOW_S:
            samples.append(group)
            group = []
        group.append(row)
    if group:
        samples.append(group)
    return [{"t": sum(float(r["t"]) for r in g) / len(g), "rows": g} for g in samples]


def _sample_sum(sample: dict, key: str) -> float:
    return sum(float(r[key]) for r in sample["rows"] if r.get(key) is not None)


def _steady_window(samples: list, agents: int) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """(aggregate tok/s, window seconds, reason). The steady window is the
    longest run of consecutive samples whose summed `running` is >=
    STEADY_RUNNING_FRACTION * agents; needs at least two samples in the run."""
    if not samples:
        return None, None, "no metrics samples"
    threshold = STEADY_RUNNING_FRACTION * agents
    runs: list = []
    current: list = []
    for sample in samples:
        if _sample_sum(sample, "running") >= threshold:
            current.append(sample)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)
    if not runs:
        return None, None, "never reached %.0f%% of agents running" % (STEADY_RUNNING_FRACTION * 100)
    best = max(runs, key=len)  # ties keep the earliest window
    span = best[-1]["t"] - best[0]["t"]
    if span <= 0:
        return None, None, "steady window has no time span"
    delta = _metrics_delta(best, "generation_tokens")
    if delta is None:
        return None, None, "no generation_tokens in steady window"
    return delta / span, span, None


def _metrics_delta(samples: list, key: str) -> Optional[float]:
    """Sum over servers of (last - first) non-null `key` within `samples`;
    None when no server has two values."""
    first: dict = {}
    last: dict = {}
    count: dict = {}
    for sample in samples:
        for row in sample["rows"]:
            if row.get(key) is None:
                continue
            server = row.get("server")
            first.setdefault(server, float(row[key]))
            last[server] = float(row[key])
            count[server] = count.get(server, 0) + 1
    deltas = [last[s] - first[s] for s in first if count[s] >= 2]
    return sum(deltas) if deltas else None


def _status_from_file(status_obj: Any) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """(status int, reason, seconds) from a parsed status.json."""
    if not isinstance(status_obj, dict):
        return None, "status.json missing or unreadable", None
    status = status_obj.get("status")
    reason = status_obj.get("reason")
    seconds = status_obj.get("seconds")
    return (None if status is None else int(status)), reason, _as_float(seconds)


def summarize_variant(variant_dir: Path, agents: int, turns: Optional[int] = None) -> dict:
    """Summarize one variant folder of a run.

    `agents` is the variant's agent count (sum over its servers); `turns` is
    the workload's turn count (from the manifest when the caller has it). Both
    feed `requests.expected` = agents x turns, so a short `requests.jsonl`
    reports `missing` rows instead of looking like a clean run. Every field is
    null (with a reason in `reasons`) when its source file is missing.
    """
    variant_dir = Path(variant_dir)
    reasons: dict = {}

    status_obj, _ = _read_json(variant_dir / STATUS_FILE)
    status, reason, seconds = _status_from_file(status_obj)
    if status is None:
        reasons["status"] = "status.json missing"
        reason = reason or "status.json missing"

    startup, startup_reason = _read_json(variant_dir / STARTUP_FILE)
    if startup_reason is not None:
        startup = None
        reasons["startup"] = startup_reason

    # warm-pass.json is legitimately absent (--no-warm-pass, a run before RUK-3203): no reason entry
    warm_pass, _ = _read_json(variant_dir / WARM_PASS_FILE)
    if not isinstance(warm_pass, dict):
        warm_pass = None

    rows, requests_reason = _read_jsonl(variant_dir / REQUESTS_FILE)
    if requests_reason is not None:
        reasons["requests"] = requests_reason
    metrics, metrics_reason = _read_jsonl(variant_dir / METRICS_FILE)
    if metrics_reason is not None:
        reasons["metrics"] = metrics_reason

    ttft_cold = _ttft_stats(rows, warm=False)
    ttft_warm = _ttft_stats(rows, warm=True)
    if ttft_cold["p50"] is None:
        reasons["ttft_cold_s"] = "no turn-0 OK rows with timings"
    if ttft_warm["p50"] is None:
        reasons["ttft_warm_s"] = "no turn>=1 OK rows with timings"

    decode = _decode_per_agent(rows)
    decode_stats = {"p50": percentile(decode, 0.5), "min": min(decode) if decode else None}
    if decode_stats["p50"] is None:
        reasons["decode_tps_per_agent"] = "no OK rows with a positive decode span"

    samples = _group_samples(metrics)
    steady_tps, steady_window_s, steady_reason = _steady_window(samples, agents)
    if steady_reason is not None:
        reasons["aggregate_tps_steady"] = steady_reason

    prefix_hit_rate = None
    delta_queries = _metrics_delta(samples, "prefix_queries")
    delta_hits = _metrics_delta(samples, "prefix_hits")
    if delta_queries is not None and delta_queries > 0 and delta_hits is not None:
        prefix_hit_rate = delta_hits / delta_queries
    else:
        reasons["prefix_hit_rate"] = metrics_reason or "no prefix query delta"

    preemptions = _metrics_delta(samples, "preemptions")
    if preemptions is None:
        reasons["preemptions"] = metrics_reason or "no metrics samples"

    mtp = None
    delta_drafts = _metrics_delta(samples, "spec_drafts")
    delta_accepted = _metrics_delta(samples, "spec_accepted")
    if delta_drafts is not None and delta_drafts > 0 and delta_accepted is not None:
        mtp = 1.0 + delta_accepted / delta_drafts
    else:
        reasons["mtp_acceptance_length"] = metrics_reason or "no spec drafts"

    cached_sum = sum(int(r["cached_tokens"]) for r in rows if r.get("cached_tokens") is not None)
    prompt_sum = sum(int(r["prompt_tokens"] or 0) for r in rows if r.get("prompt_tokens") is not None)
    cached_token_share = cached_sum / prompt_sum if prompt_sum > 0 else None
    if cached_token_share is None:
        reasons["cached_token_share"] = requests_reason or "no prompt tokens"

    counts = {RequestStatus(i).name: 0 for i in range(len(RequestStatus))}
    for row in rows:
        status_value = row.get("status")
        try:
            counts[RequestStatus(int(status_value)).name] += 1
        except (TypeError, ValueError):
            counts["UNKNOWN"] = counts.get("UNKNOWN", 0) + 1
    counts["rows"] = len(rows)
    agent_count = int(agents) if agents else len({r["agent"] for r in rows if r.get("agent") is not None})
    resolved_turns = int(turns) if turns else None
    if resolved_turns is None and rows:
        max_turn = max((int(r["turn"]) for r in rows if r.get("turn") is not None), default=None)
        resolved_turns = None if max_turn is None else max_turn + 1
    expected = agent_count * resolved_turns if (agent_count and resolved_turns) else None
    counts["expected"] = expected
    counts["missing"] = max(0, expected - len(rows)) if expected is not None else None

    t_sends = [float(r["t_send"]) for r in rows if r.get("t_send") is not None]
    t_lasts = [float(r["t_last"]) for r in rows if r.get("t_last") is not None]
    wall_s = max(t_lasts) - min(t_sends) if t_sends and t_lasts else None
    if wall_s is None:
        reasons["wall_s"] = requests_reason or "no request timings"

    peak_running = max((_sample_sum(s, "running") for s in samples), default=None)
    if peak_running is None:
        reasons["peak_running"] = metrics_reason or "no metrics samples"

    return {
        "status": status,
        "reason": reason,
        "seconds": seconds,
        "startup": startup,
        "warm_pass": warm_pass,
        "ttft_cold_s": ttft_cold,
        "ttft_warm_s": ttft_warm,
        "decode_tps_per_agent": decode_stats,
        "aggregate_tps_steady": steady_tps,
        "steady_window_s": steady_window_s,
        "prefix_hit_rate": prefix_hit_rate,
        "preemptions": preemptions,
        "mtp_acceptance_length": mtp,
        "cached_token_share": cached_token_share,
        "requests": counts,
        "wall_s": wall_s,
        "peak_running": peak_running,
        "reasons": reasons,
    }


def _manifest_pick(manifest: dict, *keys: str) -> Any:
    for key in keys:
        if manifest.get(key) is not None:
            return manifest[key]
    return None


def summarize_run(run_dir: Path) -> dict:
    """Read the manifest, the gates and every variant folder of a run into one
    dict: {"run": manifest subset, "gates": {...}, "variants": {id: summary}}."""
    run_dir = Path(run_dir)
    manifest, _ = _read_json(run_dir / MANIFEST_FILE)
    manifest = manifest if isinstance(manifest, dict) else {}
    workload = manifest.get("workload")
    workload = workload if isinstance(workload, dict) else {}
    manifest_variants = manifest.get("variants")
    manifest_variants = manifest_variants if isinstance(manifest_variants, list) else []
    turns = workload.get("turns")

    run = {
        "run_id": _manifest_pick(manifest, "run_id", "id", "run"),
        "bundle_commit": _manifest_pick(manifest, "bundle_commit", "commit", "plugin_commit"),
        "workload_sha": _manifest_pick(manifest, "workload_sha", "workload_sha256"),
        "mode": manifest.get("mode"),
        "created": manifest.get("created"),
        "skip_warmup": manifest.get("skip_warmup"),
        "server_timeout_s": manifest.get("server_timeout_s"),
        "recipe_cache": manifest.get("recipe_cache"),
        "workload": {
            "version": workload.get("version"),
            "shared_prefix_tokens": workload.get("shared_prefix_tokens"),
            "target_initial_tokens": workload.get("target_initial_tokens"),
            "turns": turns,
            "max_tokens": workload.get("max_tokens"),
            "agents": len(workload["agents"]) if isinstance(workload.get("agents"), list) else workload.get("agents"),
        },
        "variants": manifest_variants,
    }

    gates: dict = {}
    for gate in GATE_DIRS:
        status_obj, _ = _read_json(run_dir / gate / STATUS_FILE)
        status, reason, seconds = _status_from_file(status_obj)
        if status is None:
            status = int(GateStatus.NOT_RUN)
            reason = reason or "gate not run"
        gates[gate] = {"status": status, "reason": reason, "seconds": seconds}

    variants: dict = {}
    entries: list = []
    if manifest_variants:
        for entry in manifest_variants:
            if isinstance(entry, dict) and entry.get("id"):
                entries.append(entry)
    else:  # no manifest: every subfolder (except the gate folders) with an artifact is a variant
        for child in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name not in GATE_DIRS):
            found = any((child / name).exists() for name in (STATUS_FILE, STARTUP_FILE, REQUESTS_FILE, METRICS_FILE))
            if found:
                entries.append({"id": child.name})
    for entry in entries:
        variant_id = str(entry["id"])
        servers = entry.get("servers")
        servers = servers if isinstance(servers, list) else []
        agents = sum(int(s.get("agents") or 0) for s in servers if isinstance(s, dict))
        if isinstance(run["workload"]["agents"], int) and run["workload"]["agents"] > 0:
            agents = min(agents, run["workload"]["agents"])  # --agents below variants.json runs fewer agents
        summary = summarize_variant(run_dir / variant_id, agents, turns=turns)
        summary["title"] = entry.get("title")
        variants[variant_id] = summary

    return {"run": run, "gates": gates, "variants": variants}


def _fmt(value: Optional[float], spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _fmt_rate(value: Optional[float]) -> str:
    return "-" if value is None else "%.0f%%" % (value * 100.0)


def _status_name(value: Optional[int]) -> str:
    if value is None:
        return "MISSING"
    try:
        return GateStatus(int(value)).name
    except ValueError:
        return "INVALID(%s)" % value


def _max_concurrency(summary: dict) -> Optional[float]:
    """The limiting server's max concurrency: the minimum over the variant's
    startup rows (every server must hold its agents on its own)."""
    startup = summary.get("startup")
    if not isinstance(startup, list):
        return None
    values = [
        float(s["max_concurrency"]) for s in startup
        if isinstance(s, dict) and s.get("max_concurrency") is not None
    ]
    return min(values) if values else None


def _recipe_cache_text(recipe: Any) -> str:
    """The `Recipe cache:` line body (RUK-3242): `none`, the folder for every
    variant, the ids that got the folder and the ids that did not with the reason."""
    if not isinstance(recipe, dict) or not recipe.get("dir"):
        return "none"
    rc_dir = recipe["dir"]
    notes = recipe.get("variants")
    notes = notes if isinstance(notes, dict) else {}
    on_ids = [vid for vid, note in notes.items() if note == rc_dir]
    off_ids = [vid for vid, note in notes.items() if note != rc_dir]
    reason = serve.RECIPE_CACHE_OFF_REASON.removeprefix("off: ")
    if not off_ids:
        return "%s (all variants)" % rc_dir
    if on_ids:
        return "%s for %s; off for %s: %s" % (rc_dir, ", ".join(on_ids), ", ".join(off_ids), reason)
    return "off for all variants: %s" % reason


def render_markdown(summary: dict) -> str:
    """The human-readable report: header, gate table, one comparison table,
    then one section per variant. Statuses appear as names here, never ints."""
    run = summary.get("run", {})
    lines: list = []
    lines.append("# Flash-Next rental summary")
    lines.append("")
    lines.append("- Run: %s" % (run.get("run_id") or "n/a"))
    lines.append("- Bundle commit: %s" % (run.get("bundle_commit") or "n/a"))
    lines.append("- Workload sha: %s" % (run.get("workload_sha") or "n/a"))
    lines.append("- Warmup: %s" % ("skipped (graphs compiled on first use)" if run.get("skip_warmup") else "full"))
    lines.append("- Recipe cache: %s" % _recipe_cache_text(run.get("recipe_cache")))
    server_timeout = run.get("server_timeout_s")
    lines.append("- Server health timeout: %s" % ("%s s" % server_timeout if server_timeout is not None else "n/a"))
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("| gate | status | reason | seconds |")
    lines.append("|---|---|---|---|")
    for gate, gate_summary in summary.get("gates", {}).items():
        lines.append("| %s | %s | %s | %s |" % (
            gate,
            _status_name(gate_summary.get("status")),
            gate_summary.get("reason") or "-",
            _fmt(gate_summary.get("seconds"), ".1f"),
        ))
    lines.append("")

    lines.append("## Variants")
    lines.append("")
    lines.append("TTFT cold is turn 1 of the measured load, which runs after the warm pass.")
    lines.append("")
    lines.append("| id | title | status | max conc | TTFT cold p50/p90 (s) | TTFT warm p50/p90 (s) | "
                 "decode tok/s p50/min | agg tok/s | hit rate | MTP accept | preemptions | errors | wall (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for variant_id, v in summary.get("variants", {}).items():
        requests = v.get("requests", {})
        errors = sum(counts for name, counts in requests.items() if name not in ("rows", "expected", "missing"))
        errors -= requests.get("OK", 0)
        missing = requests.get("missing")
        if missing is not None:
            errors += missing
        cold, warm, dec = v.get("ttft_cold_s", {}), v.get("ttft_warm_s", {}), v.get("decode_tps_per_agent", {})
        lines.append("| %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s | %s | %s | %s | %d | %s |" % (
            variant_id,
            v.get("title") or "-",
            _status_name(v.get("status")),
            _fmt(_max_concurrency(v), ".1f"),
            _fmt(cold.get("p50"), ".2f"),
            _fmt(cold.get("p90"), ".2f"),
            _fmt(warm.get("p50"), ".2f"),
            _fmt(warm.get("p90"), ".2f"),
            _fmt(dec.get("p50"), ".1f"),
            _fmt(dec.get("min"), ".1f"),
            _fmt(v.get("aggregate_tps_steady"), ".1f"),
            _fmt_rate(v.get("prefix_hit_rate")),
            _fmt(v.get("mtp_acceptance_length"), ".2f"),
            _fmt(v.get("preemptions"), ".0f"),
            errors,
            _fmt(v.get("wall_s"), ".2f"),
        ))
    lines.append("")

    for variant_id, v in summary.get("variants", {}).items():
        lines.append("## Variant %s" % variant_id)
        lines.append("")
        lines.append("- title: %s" % (v.get("title") or "-"))
        lines.append("- status: %s" % _status_name(v.get("status")))
        lines.append("- reason: %s" % (v.get("reason") or "-"))
        lines.append("- seconds: %s" % _fmt(v.get("seconds"), ".1f"))
        lines.append("- startup: %s" % _format_startup(v.get("startup")))
        lines.append("- Warm pass: %s" % _format_warm_pass(v.get("warm_pass")))
        for key, label, spec in (
            ("ttft_cold_s", "TTFT cold (s)", ".2f"),
            ("ttft_warm_s", "TTFT warm (s)", ".2f"),
            ("decode_tps_per_agent", "decode tok/s per agent", ".1f"),
        ):
            stats = v.get(key, {})
            parts = " / ".join("%s %s" % (k, _fmt(stats.get(k), spec)) for k in stats)
            lines.append("- %s: %s" % (label, parts or "-"))
        lines.append("- aggregate tok/s steady: %s" % _fmt(v.get("aggregate_tps_steady"), ".1f"))
        lines.append("- steady window s: %s" % _fmt(v.get("steady_window_s"), ".1f"))
        lines.append("- prefix hit rate: %s" % _fmt_rate(v.get("prefix_hit_rate")))
        lines.append("- cached token share: %s" % _fmt_rate(v.get("cached_token_share")))
        lines.append("- preemptions: %s" % _fmt(v.get("preemptions"), ".0f"))
        lines.append("- MTP acceptance length: %s" % _fmt(v.get("mtp_acceptance_length"), ".2f"))
        requests = v.get("requests", {})
        parts = ["%s %d" % (name, requests[name]) for name in RequestStatus.__members__ if name in requests]
        parts.append("rows %s" % _fmt(requests.get("rows"), ".0f"))
        parts.append("expected %s" % _fmt(requests.get("expected"), ".0f"))
        parts.append("missing %s" % _fmt(requests.get("missing"), ".0f"))
        lines.append("- requests: %s" % ", ".join(parts))
        lines.append("- wall s: %s" % _fmt(v.get("wall_s"), ".2f"))
        lines.append("- peak running: %s" % _fmt(v.get("peak_running"), ".1f"))
        reasons = v.get("reasons", {})
        for field, why in reasons.items():
            lines.append("- %s not reported: %s" % (field, why))
        lines.append("")
    return "\n".join(lines) + "\n"


def _format_startup(startup: Any) -> str:
    if not isinstance(startup, list):
        return "-"
    parts = []
    for entry in startup:
        if not isinstance(entry, dict):
            continue
        parts.append("server %s: max conc %s, kv tokens %s, weights GiB %s" % (
            entry.get("server"),
            _fmt(_as_float(entry.get("max_concurrency")), ".1f"),
            _fmt(_as_float(entry.get("kv_cache_tokens")), ".0f"),
            _fmt(_as_float(entry.get("weights_gib")), ".1f"),
        ))
    return "; ".join(parts) if parts else "-"


def _format_warm_pass(warm_pass: Any) -> str:
    """The warm-pass line of a variant section; ``none`` when the pass did not run."""
    if not isinstance(warm_pass, dict):
        return "none"
    return "%s requests in %s s, %s non-OK (graphs compiled on first use; shared block cached)" % (
        _fmt(_as_float(warm_pass.get("requests")), ".0f"),
        _fmt(_as_float(warm_pass.get("seconds")), ".1f"),
        _fmt(_as_float(warm_pass.get("non_ok")), ".0f"),
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a Flash-Next rental run folder")
    parser.add_argument("run_dir", help="the run folder to read and write summaries into")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print("not a directory: %s" % run_dir, flush=True)
        return 1

    summary = summarize_run(run_dir)
    (run_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")
    print("wrote %s and %s (%d variants)" % (SUMMARY_JSON, SUMMARY_MD, len(summary["variants"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
