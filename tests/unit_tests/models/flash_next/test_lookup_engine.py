# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup (n-gram) speculation with prefix caching ON, through the real engine on CPU (``lookup_engine.py``).

The model-level n-gram tests (``test_ngram_end_to_end.py``) verify each ragged row in a ``1 + k``-wide forward
with hand-built metadata. Here the requests go through vLLM's scheduler, the HPU worker and model runner
(state lattices, the align-mode pre-step copies and block-completion commits, ragged rows, the ``[-1]`` empty
draft), and the speculative streams must be the streams of the same engine without speculation:

* oracle drafts (the baseline's own continuation) on a ragged schedule ``3, 1, 0, 2, 3, 3, 3`` with a tampered
  draft every fourth step: every accepted count ``0..k``, empty drafts, and steps narrower than the previous
  step's accepted count (``verify_width.verification_width`` widens them: without it the recurrent layers load
  the state of an earlier token and the stream breaks after the first full acceptance);
* oracle drafts accepted in full for whole streams, across the 256-token block boundaries (block-completion
  commits);
* the real ``NgramProposer`` (every step drafts through it);
* requests finishing mid-batch, and a second turn that starts with a prefix-cache hit on the first turn;
* the same oracle schedule with prefix caching off (compact GDN state).

The tiny model's next-token distribution is nearly flat (two logits are often equal in bfloat16), so a stream may
leave the baseline at a near-tie, where two correct computations in different batch shapes can pick different
tokens: the first difference must be such a tie, and enough tokens must match before it (``_compare``). One child
process of a few minutes runs every configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
K = 3
DEEP_K = 8  # the n-gram server's largest draft (platform.FLASH_NEXT_MAX_SPECULATIVE_TOKENS)
TIE = 0.0045  # nats: two bf16 steps of a logit near 0.3; a smaller top-2 gap can flip between batch shapes
MIN_MATCHED = 200  # tokens over all requests of a configuration (13 requests of up to 48 tokens)
MAX_TOKENS = 48
SCHEDULE = [3, 1, 0, 2, 3, 3, 3]
NGRAM = {"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 4, "prompt_lookup_min": 2}


def _plan(ckpt: Path) -> dict:
    reference = json.loads((ckpt / "reference.json").read_text(encoding="utf-8"))["prompts"]
    by_len = {len(p["prompt_ids"]): p["prompt_ids"] for p in reference}
    long_prompt = by_len[700]
    near_256, near_512 = long_prompt[:250], long_prompt[:505]
    motif = by_len[100][:40]
    repetitive = motif + motif + motif[:12]
    solo = [{"id": f"solo-{n}", "prompt_ids": by_len[n], "max_tokens": MAX_TOKENS} for n in (1, 7, 33, 100, 700)]
    scenarios = [{"name": f"solo-{r['id']}", "requests": [r]} for r in solo]
    scenarios += [
        {"name": "near-256", "requests": [{"id": "near-256", "prompt_ids": near_256, "max_tokens": MAX_TOKENS}]},
        {"name": "near-512", "requests": [{"id": "near-512", "prompt_ids": near_512, "max_tokens": MAX_TOKENS}]},
        {"name": "batch", "requests": [{"id": f"batch-{i}", "prompt_ids": p, "max_tokens": MAX_TOKENS - 6 * i}
                                       for i, p in enumerate((by_len[7], by_len[33], near_256, by_len[100]))]},
        {"name": "turn-2", "requests": [{"id": "turn-2", "turn_of": "near-256", "suffix_ids": by_len[33][:20],
                                         "max_tokens": MAX_TOKENS}]},
        {"name": "repetitive", "requests": [{"id": "repetitive", "prompt_ids": repetitive, "max_tokens": MAX_TOKENS}]},
    ]
    oracle = {"schedule": SCHEDULE, "tamper_every": 4}
    configs = [
        {"name": "ragged-pc", "speculative_config": NGRAM, "prefix_caching": True, "oracle": oracle},
        {"name": "full-pc", "speculative_config": NGRAM, "prefix_caching": True, "oracle": {"schedule": [K]}},
        {"name": "real-pc", "speculative_config": NGRAM, "prefix_caching": True, "oracle": None},
        {"name": "ragged-nopc", "speculative_config": NGRAM, "prefix_caching": False, "oracle": oracle},
        {"name": "no-widen-pc", "speculative_config": NGRAM, "prefix_caching": True, "oracle": oracle,
         "mutations": ["no_widen"], "scenarios": ["solo-solo-1", "solo-solo-7", "solo-solo-33"]},
        # depth: k = 8, a 9-row verification (the GDN verify past 6 rows is the torch recurrence)
        {"name": "ragged-pc-k8", "speculative_config": dict(NGRAM, num_speculative_tokens=DEEP_K),
         "prefix_caching": True, "oracle": {"schedule": [DEEP_K, 4, 0, 6, DEEP_K, 3, 7, 1, 5, 2], "tamper_every": 5},
         "scenarios": ["solo-solo-1", "solo-solo-7", "solo-solo-33", "near-256", "batch", "turn-2"]},
    ]
    return {"ckpt": str(ckpt), "tie": TIE, "max_model_len": 4096, "max_num_seqs": 4, "scenarios": scenarios,
            "configs": configs}


def _run_child(plan: dict, out_dir: Path) -> dict:
    plan_path, out_path = out_dir / "plan.json", out_dir / "out.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                                            if p])
    env.setdefault("VLLM_PLUGINS", "")
    done = subprocess.run([sys.executable, "-m", "tests.unit_tests.models.flash_next.lookup_engine", "--plan",
                           str(plan_path), "--out", str(out_path)],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=3000, check=False)
    assert out_path.exists(), f"the engine child wrote nothing (exit {done.returncode}):\n{done.stderr[-6000:]}"
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert "error" not in record, f"the engine child failed:\n{record['error']}\n{done.stderr[-3000:]}"
    return record


@pytest.fixture(scope="module")
def engine_run(tiny_checkpoint, tmp_path_factory) -> dict:
    """Run the child once for the module and return its record (the child's error, if any, fails here)."""
    record = _run_child(_plan(Path(tiny_checkpoint)), tmp_path_factory.mktemp("lookup_engine"))
    for name, config in record["configs"].items():
        assert "error" not in config, f"configuration {name} failed:\n{config['error']}"
    return record


@pytest.fixture(scope="module")
def hybrid_run(tiny_checkpoint, tmp_path_factory) -> dict:
    """The lookup + MTP hybrid (VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP=1, MTP k = 3) with an oracle lookup: on a schedule
    the lookup search returns the baseline's continuation (1..3 tokens), otherwise nothing (the MTP row stays)."""
    plan = _plan(Path(tiny_checkpoint))
    plan["configs"] = [{"name": "hybrid-pc", "speculative_config": {"method": "mtp", "num_speculative_tokens": K},
                        "prefix_caching": True, "env": {"VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP": "1"}, "oracle": None,
                        "lookup_oracle": {"schedule": [3, 0, 2, 0, 1, 3, 3]}}]
    return _run_child(plan, tmp_path_factory.mktemp("lookup_engine_hybrid"))


@pytest.fixture(scope="module")
def sampled_run(tiny_checkpoint, tmp_path_factory) -> dict:
    """Sampled requests (temperature 0.7, VLLM_HPU_SPEC_SAMPLED_VERIFY=1; no per-request seed: the fake HPU device has
    no generator) with lookup drafts: the n-gram server
    with ragged oracle drafts (the 1-token placeholder once a stream leaves the baseline) and the hybrid with an
    oracle lookup (lookup rows padded with -1 to k beside MTP rows)."""
    plan = _plan(Path(tiny_checkpoint))
    scenarios = ["solo-solo-7", "solo-solo-100", "near-256", "batch"]
    sampled = {"VLLM_HPU_SPEC_SAMPLED_VERIFY": "1"}
    plan["configs"] = [
        {"name": "ngram-sampled", "speculative_config": NGRAM, "prefix_caching": True, "env": sampled,
         "oracle": {"schedule": SCHEDULE, "tamper_every": 4}, "temperature": 0.7, "scenarios": scenarios},
        {"name": "hybrid-sampled", "speculative_config": {"method": "mtp", "num_speculative_tokens": K},
         "prefix_caching": True, "env": dict(sampled, VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP="1"), "oracle": None,
         "lookup_oracle": {"schedule": [3, 0, 2, 0, 1, 3, 3]}, "temperature": 0.7, "scenarios": scenarios},
    ]
    return _run_child(plan, tmp_path_factory.mktemp("lookup_engine_sampled"))


def test_sampled_requests_with_lookup_drafts(sampled_run, tiny_checkpoint):
    plan_requests = {r["id"]: r for s in _plan(Path(tiny_checkpoint))["scenarios"] for r in s["requests"]}
    for name in ("ngram-sampled", "hybrid-sampled"):
        config = sampled_run["configs"][name]
        assert "error" not in config, f"{name} failed:\n{config.get('error')}"
        streams = config["streams"]
        assert len(streams) == 7, (name, sorted(streams))
        for request_id, tokens in streams.items():
            # every stream runs to its length, and no padding draft (-1) ever comes out as a token
            assert len(tokens) == plan_requests[request_id]["max_tokens"], (name, request_id, len(tokens))
            assert min(tokens) >= 0, (name, request_id, [t for t in tokens if t < 0])
    # the n-gram server verified ragged widths (placeholder rows) for sampled requests
    widths = {w["width"] for w in sampled_run["configs"]["ngram-sampled"]["widths"]}
    assert len(widths) >= 2, widths


def _divergences(record: dict, config: str) -> tuple[int, list, list]:
    """``(matched, ties, breaks)`` of ``config`` against the baseline: per request the first token that differs is a
    tie when the speculative token is among the baseline's recorded best tokens there, at most ``TIE`` below the
    best log-probability (several tokens can tie); any other difference is a break."""
    base, streams = record["baseline"], record["configs"][config]["streams"]
    matched, ties, breaks = 0, [], []
    for request_id, got in streams.items():
        want = base[request_id]
        tokens = want["tokens"]
        assert len(got) == len(tokens), f"{config}: request {request_id} has {len(got)} tokens, not {len(tokens)}"
        first = next((i for i, (a, b) in enumerate(zip(got, tokens)) if a != b), None)
        if first is None:
            matched += len(tokens)
            continue
        matched += first
        below = {int(t): d for t, d in want["top"][first]}
        entry = (request_id, first, got[first], round(below.get(got[first], float("inf")), 4))
        (ties if below.get(got[first], float("inf")) <= TIE else breaks).append(entry)
    return matched, ties, breaks


def _compare(record: dict, config: str) -> int:
    """Every stream of ``config`` equals the baseline up to its first difference, which must be a near-tie of the
    baseline (``_divergences``); returns the tokens that matched."""
    matched, ties, breaks = _divergences(record, config)
    assert not breaks, (f"{config}: streams leave the engine without speculation away from a near-tie "
                        f"(request, token index, speculative token, its baseline log-probability below the best): "
                        f"{breaks}")
    assert matched >= MIN_MATCHED, f"{config}: only {matched} tokens matched before the first near-tie differences"
    return matched


def test_ragged_oracle_drafts_with_prefix_caching(engine_run):
    record = engine_run["configs"]["ragged-pc"]
    assert record["prefix_caching"] and record["recurrent_spec"] and record["drafter"] == "NgramProposer"
    compared = _compare(engine_run, "ragged-pc")
    accepted = {len(s["sampled"]) - 1 for s in record["steps"] if s["step"] > 0}
    assert set(range(K + 1)) <= accepted, f"not every accepted count 0..{K} ran: {sorted(accepted)}"
    assert any(not s["draft"] for s in record["steps"]), "no empty draft ([-1]) ran"
    assert any(s["tampered"] is not None for s in record["steps"]), "no tampered draft ran"
    widened = [w for w in record["widths"] if w["width"] > max(w["scheduled"])]
    assert widened, "no step was narrower than its previous accepted count: the widening was not exercised"
    print(f"ragged-pc: {compared} tokens compared, {len(record['steps'])} draft rows, {len(widened)} widened steps, "
          f"{record['boundary_commits']} block-completion commits")


def test_full_acceptance_across_block_boundaries(engine_run):
    record = engine_run["configs"]["full-pc"]
    compared = _compare(engine_run, "full-pc")
    assert record["boundary_commits"] > 0, "no accepted token completed a 256-token block"
    print(f"full-pc: {compared} tokens compared, {record['boundary_commits']} block-completion commits")


def test_real_ngram_drafter_with_prefix_caching(engine_run):
    """vLLM's NgramProposer itself drafts every step (the random tiny model rarely repeats an n-gram of its context,
    so its drafts are mostly empty and acceptance comes from the oracle configurations); the streams stay lossless."""
    record = engine_run["configs"]["real-pc"]
    assert record["drafter"] == "NgramProposer" and record["steps"], "the n-gram drafter was never called"
    compared = _compare(engine_run, "real-pc")
    drafted = [s for s in record["steps"] if s["draft"]]
    print(f"real-pc: {compared} tokens compared, {len(record['steps'])} draft rows, {len(drafted)} non-empty drafts")


def test_ragged_oracle_drafts_without_prefix_caching(engine_run):
    record = engine_run["configs"]["ragged-nopc"]
    assert not record["prefix_caching"]
    compared = _compare(engine_run, "ragged-nopc")
    print(f"ragged-nopc: {compared} tokens compared")


def test_deep_ngram_drafts_with_prefix_caching(engine_run):
    """k = 8: every accepted count 0..8 runs, steps are widened, and the streams stay lossless (9-row verifications
    through the torch GDN recurrence, the 12-row QSA key ring, the 17-row PLE short-conv state)."""
    record = engine_run["configs"]["ragged-pc-k8"]
    accepted = {len(s["sampled"]) - 1 for s in record["steps"] if s["step"] > 0}
    assert set(range(DEEP_K + 1)) <= accepted, f"not every accepted count 0..{DEEP_K} ran: {sorted(accepted)}"
    widened = [w for w in record["widths"] if w["width"] > max(w["scheduled"])]
    assert widened and max(w["width"] for w in record["widths"]) == 1 + DEEP_K
    matched, ties, breaks = _divergences(engine_run, "ragged-pc-k8")
    assert not breaks, f"ragged-pc-k8 breaks away from a near-tie: {breaks}"
    print(f"ragged-pc-k8: {matched} tokens matched, {len(widened)} widened steps, ties {ties}")


def test_tampered_drafts_were_rejected(engine_run):
    """The tamper token is never the target's choice: each tampered step accepts exactly the drafts before it."""
    for name in ("ragged-pc", "ragged-nopc"):
        steps = engine_run["configs"][name]["steps"]
        by_row = {}
        for s in steps:
            by_row.setdefault(s["row"], []).append(s)
        for row_steps in by_row.values():
            for prev, nxt in zip(row_steps, row_steps[1:]):
                if prev["tampered"] is not None and nxt["step"] == prev["step"] + 1:
                    assert len(nxt["sampled"]) - 1 <= prev["tampered"], (name, prev, nxt)


def test_the_comparison_catches_a_missing_widening(engine_run):
    """Mutation check: with the batch width of the program head (no ``verification_width``), a step narrower than the
    previous step's accepted count loads an earlier token's state and the stream breaks away from any tie."""
    _matched, _ties, breaks = _divergences(engine_run, "no-widen-pc")
    assert breaks, "the stale-state load went unnoticed: the comparison is too weak"
    print(f"no-widen-pc: breaks {breaks}")


def test_hybrid_lookup_beside_mtp_with_prefix_caching(hybrid_run):
    """The hybrid through the engine: lookup rows (padded to k with -1) and MTP rows in the same streams, every
    draft k long (the MTP drafter's row layout), lookup drafts accepted, the streams lossless."""
    record = hybrid_run["configs"]["hybrid-pc"]
    if "error" in record and "image_token_index" in record["error"]:
        pytest.skip("the MTP engine does not start on this tree (load_model: image_token_index): it needs the MTP "
                    "engine fixes of the mtp-card-ready track")
    assert "error" not in record, record.get("error")
    assert record["drafter"] == "HpuFlashNextMtpProposer"
    compared = _compare(hybrid_run, "hybrid-pc")
    sources = [s for row in record["hybrid_rows"] for s in row["sources"] if s]
    assert {"lookup", "mtp"} <= set(sources), f"the hybrid never mixed the two sources: {set(sources)}"
    assert all(len(d) == K for row in record["hybrid_rows"] for d in row["drafts"]), "a hybrid row is not k long"
    drafted, accepted, verified = record["hybrid_counts"]["lookup"]
    assert accepted > verified, f"the oracle lookup drafts were hardly accepted: {accepted} over {verified} rows"
    print(f"hybrid-pc: {compared} tokens compared, lookup rows {drafted} accepted {accepted} over {verified}, "
          f"MTP rows {record['hybrid_counts']['mtp']}")
