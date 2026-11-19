# SPDX-License-Identifier: Apache-2.0
"""Tests for the rental run report (plan M4, RUK-2709). A synthetic run
folder is built in tmp_path with hand-computed expected values."""

from __future__ import annotations

import json
from pathlib import Path

from tools.flash_next.rental import serve
from tools.flash_next.rental.report import main as report_main
from tools.flash_next.rental.report import percentile, summarize_run, summarize_variant
from tools.flash_next.rental.schema import WORKLOAD_VERSION, GateStatus, RequestStatus


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_lines(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_row(
    agent: int,
    turn: int,
    t_send: float,
    t_first: float,
    t_last: float,
    output: int,
    prompt: int = 0,
    cached=None,
    status: int = int(RequestStatus.OK),
    server: int = 0,
) -> str:
    return json.dumps({
        "agent": agent,
        "turn": turn,
        "server": server,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "output_tokens": output,
        "t_send": t_send,
        "t_first": t_first,
        "t_last": t_last,
        "itl_p50_ms": None,
        "status": status,
        "error": None,
    })


def make_metrics(
    t: float,
    server: int,
    running=None,
    generation_tokens=None,
    prefix_hits=None,
    prefix_queries=None,
    preemptions=None,
    spec_accepted=None,
    spec_drafts=None,
) -> str:
    return json.dumps({
        "t": t,
        "server": server,
        "running": running,
        "waiting": None,
        "kv_usage": None,
        "prefix_hits": prefix_hits,
        "prefix_queries": prefix_queries,
        "preemptions": preemptions,
        "spec_accepted": spec_accepted,
        "spec_drafts": spec_drafts,
        "spec_draft_tokens": None,
        "generation_tokens": generation_tokens,
        "prompt_tokens": None,
    })


def startup_entry(server: int, max_concurrency: float) -> dict:
    return {
        "server": server,
        "usable_num_blocks": 40000,
        "kv_cache_tokens": 4194304,
        "max_concurrency": max_concurrency,
        "max_concurrency_len": 131072,
        "weights_gib": 21.6,
        "warmup_seconds": 30.0,
        "gdn_layers_detected": "48",
        "startup_seconds": 300.0,
    }


def build_run(tmp_path: Path) -> Path:
    """RUN_DIR with gates, manifest, variant A (full, one server), variant B
    (two servers, no steady window) and variant C (only a FAILED status.json).
    Turns are 2; A has 4 agents, B 2+2, C 4."""
    run = tmp_path / "run"
    write_json(
        run / "manifest.json", {
            "run_id":
            "run-7",
            "bundle_commit":
            "abc1234",
            "workload_sha":
            "cafef00d",
            "mode":
            "stub",
            "created":
            1700000000.0,
            "workload": {
                "version": WORKLOAD_VERSION,
                "shared_prefix_tokens": 16384,
                "target_initial_tokens": 124000,
                "turns": 2,
                "max_tokens": 1024,
                "agents": [{
                    "agent": i
                } for i in range(4)],
            },
            "variants": [
                {
                    "id": "A",
                    "title": "TP8 baseline",
                    "servers": [{
                        "agents": 4
                    }]
                },
                {
                    "id": "B",
                    "title": "2xTP4",
                    "servers": [{
                        "agents": 2
                    }, {
                        "agents": 2
                    }]
                },
                {
                    "id": "C",
                    "title": "failed variant",
                    "servers": [{
                        "agents": 4
                    }]
                },
            ],
        })
    write_json(run / "preflight" / "status.json", {"id": "preflight", "status": 1, "reason": None, "seconds": 12.0})
    write_json(run / "build" / "status.json", {"id": "build", "status": 1, "reason": None, "seconds": 30.0})
    write_json(
        run / "tiny" / "status.json",
        {"id": "tiny", "status": 2, "reason": "14/16 tokens matched", "seconds": 5.0})

    # variant A: cold TTFT 1,2,3,4 -> p50 2.0 p90 4.0 max 4.0;
    # warm TTFT 0.5,1.5,0.5,1.5 -> p50 0.5 p90 1.5;
    # per-agent decode 49.5, 30.0, 75.0, 10.0 -> p50 30.0 min 10.0;
    # steady window [1010..1014] (the dip at 1008 splits it) -> 100.0 tok/s over 4.0 s;
    # hit rate 80/100, preemptions 3, MTP 1 + 90/30 = 4.0; cached 2000/12000.
    a_rows = [
        make_row(0, 0, 1000.0, 1001.0, 1003.0, 100, prompt=1000, cached=500),
        make_row(1, 0, 1000.0, 1002.0, 1004.0, 61, prompt=1000, cached=500),
        make_row(2, 0, 1000.0, 1003.0, 1005.0, 151, prompt=1000, cached=500),
        make_row(3, 0, 1000.0, 1004.0, 1006.0, 21, prompt=1000, cached=500),
        make_row(0, 1, 1010.0, 1010.5, 1012.5, 100, prompt=2000),
        make_row(1, 1, 1010.0, 1011.5, 1013.5, 61, prompt=2000),
        make_row(2, 1, 1010.0, 1010.5, 1012.5, 151, prompt=2000),
        make_row(3, 1, 1010.0, 1011.5, 1013.5, 21, prompt=2000),
        make_row(4, 0, 1010.0, 1011.0, 1012.0, 0, status=int(RequestStatus.TIMEOUT)),
    ]
    a_metrics = [
        make_metrics(1000.0, 0, running=0, generation_tokens=0, prefix_hits=0, prefix_queries=0, preemptions=0,
                     spec_accepted=0, spec_drafts=0),
        make_metrics(1002.0, 0, running=2, generation_tokens=100, prefix_hits=10, prefix_queries=20),
        make_metrics(1004.0, 0, running=4, generation_tokens=200, prefix_hits=20, prefix_queries=40, preemptions=1,
                     spec_accepted=30, spec_drafts=10),
        make_metrics(1006.0, 0, running=4, generation_tokens=400, prefix_hits=40, prefix_queries=60, preemptions=2,
                     spec_accepted=60, spec_drafts=20),
        make_metrics(1008.0, 0, running=3, generation_tokens=600, prefix_hits=60, prefix_queries=80, preemptions=3,
                     spec_accepted=75, spec_drafts=25),
        make_metrics(1010.0, 0, running=4, generation_tokens=800, prefix_hits=70, prefix_queries=90, preemptions=3,
                     spec_accepted=80, spec_drafts=27),
        make_metrics(1012.0, 0, running=4, generation_tokens=1000, prefix_hits=75, prefix_queries=95, preemptions=3,
                     spec_accepted=85, spec_drafts=28),
        make_metrics(1014.0, 0, running=4, generation_tokens=1200, prefix_hits=80, prefix_queries=100,
                     preemptions=3, spec_accepted=90, spec_drafts=30),
        make_metrics(1016.0, 0, running=0, generation_tokens=1200, prefix_hits=80, prefix_queries=100,
                     preemptions=3, spec_accepted=90, spec_drafts=30),
    ]
    write_json(run / "A" / "status.json", {"id": "A", "status": 1, "reason": None, "seconds": 600.0})
    write_json(run / "A" / "startup.json", [startup_entry(0, 25.0)])
    # a killed run leaves a cut-off last line: it must be dropped, not fatal
    write_lines(run / "A" / "requests.jsonl", a_rows + ['{"agent": 9, "turn": 0, "serv'])
    write_lines(run / "A" / "metrics.jsonl", a_metrics)

    # variant B: two servers sampled 1 s apart land in one sample (running 2+1);
    # running never reaches 0.9 * 4 = 3.6, so there is no steady window.
    write_json(run / "B" / "status.json", {"id": "B", "status": 1, "reason": None, "seconds": 500.0})
    write_json(run / "B" / "startup.json", [startup_entry(0, 25.0), startup_entry(1, 22.0)])
    write_lines(run / "B" / "requests.jsonl", [
        make_row(0, 0, 2000.0, 2002.0, 2004.0, 81, prompt=1000, cached=100, server=0),
        make_row(1, 0, 2000.0, 2001.0, 2003.5, 61, prompt=1000, cached=100, server=1),
    ])
    write_lines(run / "B" / "metrics.jsonl", [
        make_metrics(2000.0, 0, running=2, prefix_hits=5, prefix_queries=25),
        make_metrics(2000.4, 1, running=1, prefix_hits=5, prefix_queries=25),
        make_metrics(2002.0, 0, running=1, prefix_hits=9, prefix_queries=40),
        make_metrics(2002.3, 1, running=1, prefix_hits=7, prefix_queries=40),
    ])

    # variant C: stopped at startup, only a FAILED status.json exists
    write_json(
        run / "C" / "status.json", {
            "id": "C",
            "status": int(GateStatus.FAILED),
            "reason": "server did not become healthy",
            "seconds": 1800.0
        })
    return run


def test_percentile_nearest_rank() -> None:
    assert percentile([4.0, 1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([4.0, 1.0, 2.0, 3.0], 0.9) == 4.0
    assert percentile([7.0], 0.5) == 7.0
    assert percentile([], 0.5) is None


def test_summarize_variant_full(tmp_path: Path) -> None:
    run = build_run(tmp_path)
    v = summarize_variant(run / "A", 4, turns=2)

    assert v["status"] == int(GateStatus.PASSED)
    assert v["seconds"] == 600.0
    assert v["startup"][0]["max_concurrency"] == 25.0

    assert v["ttft_cold_s"] == {"p50": 2.0, "p90": 4.0, "max": 4.0}
    assert v["ttft_warm_s"] == {"p50": 0.5, "p90": 1.5, "max": 1.5}
    assert v["decode_tps_per_agent"] == {"p50": 30.0, "min": 10.0}

    assert v["aggregate_tps_steady"] == 100.0
    assert v["steady_window_s"] == 4.0
    assert abs(v["prefix_hit_rate"] - 0.8) < 1e-12
    assert v["preemptions"] == 3.0
    assert v["mtp_acceptance_length"] == 4.0
    assert abs(v["cached_token_share"] - 2000.0 / 12000.0) < 1e-12

    assert v["requests"]["OK"] == 8
    assert v["requests"]["TIMEOUT"] == 1
    assert v["requests"]["HTTP_ERROR"] == 0 and v["requests"]["SHORT_OUTPUT"] == 0
    assert v["requests"]["rows"] == 9  # the truncated last line was dropped
    assert v["requests"]["expected"] == 8
    assert v["requests"]["missing"] == 0

    assert v["wall_s"] == 13.5
    assert v["peak_running"] == 4.0
    assert v["reasons"] == {}


def test_summarize_variant_two_servers_no_steady_window(tmp_path: Path) -> None:
    run = build_run(tmp_path)
    v = summarize_variant(run / "B", 4, turns=2)

    assert v["status"] == int(GateStatus.PASSED)
    assert len(v["startup"]) == 2
    assert v["ttft_cold_s"] == {"p50": 1.0, "p90": 2.0, "max": 2.0}
    assert v["ttft_warm_s"]["p50"] is None
    # agent 0: 80 / 2.0 = 40.0; agent 1: 60 / 2.5 = 24.0 -> p50 24.0, min 24.0
    assert v["decode_tps_per_agent"] == {"p50": 24.0, "min": 24.0}
    assert v["aggregate_tps_steady"] is None
    assert v["steady_window_s"] is None
    assert v["reasons"]["aggregate_tps_steady"] == "never reached 90% of agents running"
    assert v["peak_running"] == 3.0  # 2 + 1 servers in one sample
    assert abs(v["prefix_hit_rate"] - 0.2) < 1e-12  # (9+7-5-5) / (40+40-25-25)
    assert v["mtp_acceptance_length"] is None  # no drafts
    assert abs(v["cached_token_share"] - 0.1) < 1e-12
    assert v["requests"]["rows"] == 2
    assert v["requests"]["expected"] == 8  # 4 agents (2 per server) x 2 turns
    assert v["requests"]["missing"] == 6
    assert v["wall_s"] == 4.0


def test_summarize_variant_status_only(tmp_path: Path) -> None:
    run = build_run(tmp_path)
    v = summarize_variant(run / "C", 4, turns=2)

    assert v["status"] == int(GateStatus.FAILED)
    assert v["reason"] == "server did not become healthy"
    assert v["seconds"] == 1800.0
    assert v["startup"] is None
    assert v["ttft_cold_s"] == {"p50": None, "p90": None, "max": None}
    assert v["aggregate_tps_steady"] is None
    assert v["prefix_hit_rate"] is None
    assert v["mtp_acceptance_length"] is None
    assert v["wall_s"] is None
    assert v["requests"]["rows"] == 0
    assert v["requests"]["expected"] == 8
    assert v["requests"]["missing"] == 8
    assert v["reasons"]["startup"] == "startup.json missing"
    assert v["reasons"]["requests"] == "requests.jsonl missing"


def test_summarize_variant_empty_dir_never_crashes(tmp_path: Path) -> None:
    v = summarize_variant(tmp_path / "ghost", 4, turns=2)
    assert v["status"] is None
    assert v["startup"] is None
    assert v["requests"]["expected"] == 8
    assert v["requests"]["missing"] == 8
    for key in ("aggregate_tps_steady", "prefix_hit_rate", "preemptions", "mtp_acceptance_length",
                "cached_token_share", "wall_s", "peak_running"):
        assert v[key] is None


def test_summarize_run(tmp_path: Path) -> None:
    summary = summarize_run(build_run(tmp_path))
    assert summary["run"]["run_id"] == "run-7"
    assert summary["run"]["bundle_commit"] == "abc1234"
    assert summary["run"]["workload_sha"] == "cafef00d"
    assert summary["run"]["workload"]["turns"] == 2
    assert summary["gates"]["preflight"]["status"] == int(GateStatus.PASSED)
    assert summary["gates"]["tiny"]["status"] == int(GateStatus.FAILED)
    assert summary["gates"]["tiny"]["reason"] == "14/16 tokens matched"
    assert list(summary["variants"]) == ["A", "B", "C"]
    assert summary["variants"]["A"]["title"] == "TP8 baseline"


def test_main_writes_summaries(tmp_path: Path) -> None:
    run = build_run(tmp_path)
    assert report_main([str(run)]) == 0

    data = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert data["run"]["run_id"] == "run-7"
    assert list(data["variants"]) == ["A", "B", "C"]
    # statuses stay integers in the JSON
    assert data["gates"]["preflight"]["status"] == int(GateStatus.PASSED)
    assert data["variants"]["A"]["status"] == int(GateStatus.PASSED)
    assert data["variants"]["C"]["status"] == int(GateStatus.FAILED)
    assert isinstance(data["variants"]["C"]["status"], int)

    md = (run / "summary.md").read_text(encoding="utf-8")
    # statuses appear as names in the markdown, and the header carries the manifest
    for expected in ("run-7", "abc1234", "cafef00d", "PASSED", "FAILED", "TP8 baseline", "2xTP4", "## Variant A",
                     "## Variant C", "80%", "server did not become healthy"):
        assert expected in md
    assert "GateStatus." not in md and "RequestStatus." not in md
    # the comparison table shows the limiting server's concurrency for B
    b_row = [line for line in md.splitlines() if line.startswith("| B ")][0]
    assert "| 22.0 |" in b_row


def test_main_without_manifest(tmp_path: Path) -> None:
    run = build_run(tmp_path)
    (run / "manifest.json").unlink()

    assert report_main([str(run)]) == 0
    data = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    # the gate folders are not listed as variants even with no manifest
    assert set(data["variants"]) == {"A", "B", "C"}
    # agents and turns fall back to the rows: A has 5 distinct agents, turns 2
    assert data["variants"]["A"]["requests"]["expected"] == 10
    assert data["variants"]["C"]["requests"]["expected"] is None
    # the gate status files on disk are still read as gates
    assert data["gates"]["preflight"]["status"] == int(GateStatus.PASSED)
    assert data["gates"]["tiny"]["status"] == int(GateStatus.FAILED)
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "NOT_RUN" not in md


def test_metrics_delta_is_per_server(tmp_path: Path) -> None:
    """A server missing from the first sample must not contribute its whole
    cumulative counter: deltas are per server, summed over servers with two
    values."""
    var = tmp_path / "P"
    write_lines(var / "metrics.jsonl", [
        make_metrics(1000.0, 0, prefix_hits=0, prefix_queries=0),
        make_metrics(1002.0, 0, prefix_hits=50, prefix_queries=100),
        make_metrics(1002.0, 1, prefix_hits=5000, prefix_queries=10000),
        make_metrics(1004.0, 0, prefix_hits=80, prefix_queries=150),
        make_metrics(1004.0, 1, prefix_hits=5090, prefix_queries=10100),
    ])
    v = summarize_variant(var, 2, turns=2)
    # server 0: 80-0 hits / 150-0 queries; server 1: 90 hits / 100 queries
    # -> (80 + 90) / (150 + 100) = 0.68 (the summed first/last gives 0.504)
    assert abs(v["prefix_hit_rate"] - 0.68) < 1e-12


def test_null_status_row_counted_unknown(tmp_path: Path) -> None:
    var = tmp_path / "N"
    write_lines(var / "requests.jsonl", [make_row(0, 0, 1000.0, 1001.0, 1002.0, 10, status=None)])
    v = summarize_variant(var, 1, turns=1)
    assert v["requests"]["UNKNOWN"] == 1
    assert v["requests"]["rows"] == 1


def test_table_errors_column_counts_missing_rows(tmp_path: Path) -> None:
    """A short requests.jsonl must never show 0 errors in the comparison
    table (RUK-2923)."""
    run = tmp_path / "run"
    write_json(
        run / "manifest.json", {
            "run_id": "run-9",
            "workload": {
                "turns": 4
            },
            "variants": [{
                "id": "D",
                "title": "short run",
                "servers": [{
                    "agents": 2
                }]
            }],
        })
    write_json(run / "D" / "status.json", {"id": "D", "status": 1, "reason": None, "seconds": 10.0})
    write_lines(run / "D" / "requests.jsonl",
                [make_row(i % 2, 0, 1000.0 + i, 1001.0 + i, 1002.0 + i, 10, prompt=100) for i in range(5)])

    assert report_main([str(run)]) == 0
    md = (run / "summary.md").read_text(encoding="utf-8")
    d_row = [line for line in md.splitlines() if line.startswith("| D ")][0]
    # expected 2 agents x 4 turns = 8, rows 5, all OK -> errors 3
    assert "| 3 |" in d_row


def test_expected_rows_follow_workload_agents(tmp_path: Path) -> None:
    """A run with fewer workload agents than variants.json (--agents 2) expects only those agents' rows."""
    run = build_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["workload"]["agents"] = 2
    write_json(run / "manifest.json", manifest)
    summary = summarize_run(run)
    assert summary["variants"]["A"]["requests"]["expected"] == 4  # 2 agents x 2 turns, not 4 x 2
    assert summary["variants"]["B"]["requests"]["expected"] == 4


def test_summary_header_warmup_and_timeout(tmp_path: Path) -> None:
    run = build_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["skip_warmup"] = True
    manifest["server_timeout_s"] = 7200.0
    write_json(run / "manifest.json", manifest)
    assert report_main([str(run)]) == 0
    summary = summarize_run(run)
    assert summary["run"]["skip_warmup"] is True
    assert summary["run"]["server_timeout_s"] == 7200.0
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Warmup: skipped (graphs compiled on first use)" in md
    assert "- Server health timeout: 7200.0 s" in md

    manifest["skip_warmup"] = False
    del manifest["server_timeout_s"]
    write_json(run / "manifest.json", manifest)
    assert report_main([str(run)]) == 0
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Warmup: full" in md
    assert "- Server health timeout: n/a" in md


def test_warm_pass_line_and_absent_falls_back_to_none(tmp_path: Path) -> None:
    """warm-pass.json renders as one line per variant; a run without it (the
    --no-warm-pass case, or a run from before RUK-3203) renders "none"."""
    run = build_run(tmp_path)
    write_json(run / "A" / "warm-pass.json", {"requests": 4, "seconds": 250.0, "non_ok": 0})

    v = summarize_variant(run / "A", 4, turns=2)
    assert v["warm_pass"] == {"requests": 4, "seconds": 250.0, "non_ok": 0}
    assert v["reasons"] == {}  # an absent warm-pass.json is not a missing-field reason
    assert summarize_variant(run / "B", 4, turns=2)["warm_pass"] is None

    assert report_main([str(run)]) == 0
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Warm pass: 4 requests in 250.0 s, 0 non-OK (graphs compiled on first use; shared block cached)" in md
    assert "- Warm pass: none" in md  # variant B, which has no warm-pass.json


def test_recipe_cache_line_shapes(tmp_path: Path) -> None:
    """The `Recipe cache:` line (RUK-3242): none, the folder for all variants,
    the ids that got it with the reason for the rest, and off for all."""
    run = build_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    reason = serve.RECIPE_CACHE_OFF_REASON.removeprefix("off: ")

    assert report_main([str(run)]) == 0  # a run from before RUK-3242: no recipe_cache key at all
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Recipe cache: none" in md

    rc_dir = "/recipe-cache/abc123456789"
    manifest["recipe_cache"] = {
        "dir": rc_dir,
        "size_mb": 16384,
        "combined_kernel_db": True,
        "variants": {"A": rc_dir, "B": rc_dir, "C": rc_dir},
    }
    write_json(run / "manifest.json", manifest)
    assert report_main([str(run)]) == 0
    assert summarize_run(run)["run"]["recipe_cache"] == manifest["recipe_cache"]
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Recipe cache: %s (all variants)" % rc_dir in md

    manifest["recipe_cache"]["variants"] = {"A": rc_dir, "B": serve.RECIPE_CACHE_OFF_REASON, "C": "none"}
    write_json(run / "manifest.json", manifest)
    assert report_main([str(run)]) == 0
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Recipe cache: %s for A; off for B, C: %s" % (rc_dir, reason) in md

    manifest["recipe_cache"]["variants"] = {"A": serve.RECIPE_CACHE_OFF_REASON, "B": "none", "C": "none"}
    write_json(run / "manifest.json", manifest)
    assert report_main([str(run)]) == 0
    md = (run / "summary.md").read_text(encoding="utf-8")
    assert "- Recipe cache: off for all variants: %s" % reason in md
    assert "TTFT cold is turn 1 of the measured load, which runs after the warm pass." in md
