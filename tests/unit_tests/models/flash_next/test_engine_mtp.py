# SPDX-License-Identifier: Apache-2.0
"""MTP through the real vLLM engine with the HPU plugin, on CPU (``tests/unit_tests/hpu_cpu_engine.py``).

The layer and model tests (``test_mtp_end_to_end.py`` and friends) call the model directly; these start
``vllm.LLM`` with ``--speculative-config mtp`` on the tiny checkpoint, through the scheduler, the HPU worker,
the HPU model runner and ``HpuFlashNextMtpProposer``, with prefix caching on (the rental's setting). On the
program head before these tests the engine stopped before the first draft (``load_model``: the image token id,
the EAGLE3 aux layers; the first draft: M-RoPE on the draft, the prefill hidden shape; the fake-HPU block
mapping of the reused decode metadata), and the drafter read the block table of a GDN group.

* ``k = 3`` gives the same tokens as the same engine without speculation;
* the drafter gets the attention group's block table (group 0 of a Flash-Next engine is a GDN group);
* the first draft after a prefill is the HF reference's MTP argmax (``reference.json``, key ``mtp``);
* oracle drafts make the verification accept every count ``0..k``: the stream stays the non-speculative one
  and each step commits exactly the forced count plus one (the tiny model's random MTP head accepts nothing
  on its own, so without the oracle no accepted count above 0 would run).

Each engine run is a child process of about 15 s.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from tests.unit_tests import hpu_cpu_engine

K = 3


@pytest.fixture(scope="module")
def engine_runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_mtp")
    base = hpu_cpu_engine.run(tiny_checkpoint, out / "k0.json")
    spec = hpu_cpu_engine.run(tiny_checkpoint, out / "k3.json", "--k", str(K))
    return {"dir": out, "base": base, "spec": spec, "ckpt": Path(tiny_checkpoint)}


def _streams(record: dict) -> dict:
    return {tuple(p["prompt_ids"]): p["generated_ids"] for p in record["prompts"]}


def _oracle_run(engine_runs: dict, accept: list[int], *options: str) -> dict:
    name = "accept_" + "".join(str(a) for a in accept) + "".join(o.strip("-")[:3] for o in options)
    path = engine_runs["dir"] / f"{name}.oracle.json"
    path.write_text(json.dumps({"streams": engine_runs["base"]["prompts"], "accept": accept}), encoding="utf-8")
    return hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / f"{name}.json", "--k", str(K), "--oracle",
                              str(path), *options)


def test_k3_gives_the_tokens_of_the_engine_without_speculation(engine_runs):
    base, spec = _streams(engine_runs["base"]), _streams(engine_runs["spec"])
    assert base.keys() == spec.keys() and len(base) >= 4
    for prompt, tokens in base.items():
        assert spec[prompt] == tokens, f"prompt of {len(prompt)} tokens"
    assert engine_runs["spec"]["prefix_caching"] is True


def test_the_engine_runs_verification_steps_of_one_plus_k_rows(engine_runs):
    rows = [r for step in engine_runs["spec"]["decode_steps"] for r in step["rows"]]
    assert rows and max(rows) == 1 + K


def test_the_drafter_reads_the_attention_groups_block_table(engine_runs):
    facts = engine_runs["spec"]["runner"]
    assert facts["groups"][0]["spec"] == "MambaSpec"  # the premise: group 0 is not the attention group
    assert facts["groups"][facts["attention_group"]]["spec"] == "FullAttentionSpec"
    calls = engine_runs["spec"]["draft_calls"]
    assert calls
    assert all(call["is_attention_table"] for call in calls), calls[:3]


def test_the_first_draft_after_a_prefill_is_the_reference_mtp_argmax(engine_runs):
    run = _oracle_run(engine_runs, [K])
    engine_runs["oracle_all"] = run
    reference = json.loads((engine_runs["ckpt"] / "reference.json").read_text(encoding="utf-8"))
    first = {}
    for entry in run["oracle"]:
        first.setdefault(entry["req"], entry)
    by_length = {entry["prompt_len"]: entry for entry in first.values()}
    checked = 0
    for prompt in reference["prompts"]:
        entry = by_length.get(len(prompt["prompt_ids"]))
        if not prompt.get("mtp") or entry is None:
            continue
        assert entry["n_out"] == 1  # the draft that follows the prefill and its first sampled token
        top = prompt["mtp"]["last_row_top5"]["ids"]
        # bf16 engine against the fp32 HF reference: the top two logits of the tiny model are often within 1e-3
        assert entry["drafter"][0] in top[:2], (len(prompt["prompt_ids"]), entry["drafter"][0], top)
        checked += 1
    assert checked >= 4


@pytest.mark.parametrize("accept", [[K], [0, 1, 2, 3], [2, 0, 3, 1]], ids=["all", "rising", "mixed"])
def test_oracle_drafts_are_accepted_exactly_and_keep_the_stream(engine_runs, accept):
    run = engine_runs.get("oracle_all") if accept == [K] else None
    run = run or _oracle_run(engine_runs, accept)
    base = _streams(engine_runs["base"])
    for prompt, tokens in _streams(run).items():
        assert tokens == base[prompt], f"prompt of {len(prompt)} tokens, accept {accept}"
    max_tokens = len(next(iter(base.values())))
    by_req = defaultdict(list)
    for entry in run["oracle"]:
        by_req[entry["req"]].append(entry)
    checked = 0
    for entries in by_req.values():
        for before, after in zip(entries, entries[1:]):
            if before["n_out"] + K + 1 > max_tokens:  # the scheduler cuts the drafts at max_tokens
                continue
            assert after["n_out"] - before["n_out"] == before["accept"] + 1, (before, after)
            checked += 1
    assert checked >= len(by_req)
    # the decode batches load from the accepted counts of the step before
    loaded = {a for step in run["decode_steps"] if step["num_accepted"] for a in step["num_accepted"] if a > 0}
    assert loaded and loaded <= {a + 1 for a in accept}


def test_oracle_drafts_with_the_fused_kernels_reference_ops(engine_runs):
    # VLLM_HPU_GDN_CORE_DECODE / _VERIFY and VLLM_HPU_HC_FUSED run their Python reference kernels on CPU: the
    # speculative step through the code that calls the fused ops (lattice -> verify op, the HC fused path).
    env = {"VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_GDN_CORE_VERIFY": "1", "VLLM_HPU_HC_FUSED": "1"}
    fused_base = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "fused_k0.json", env=env)
    path = engine_runs["dir"] / "fused.oracle.json"
    path.write_text(json.dumps({"streams": fused_base["prompts"], "accept": [2, 0, 3, 1]}), encoding="utf-8")
    run = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "fused.json", "--k", str(K), "--oracle",
                             str(path), env=env)
    base = _streams(fused_base)
    for prompt, tokens in _streams(run).items():
        assert tokens == base[prompt], f"prompt of {len(prompt)} tokens"
    assert max(len(set(step["rows"])) for step in run["decode_steps"]) >= 1
    assert max(r for step in run["decode_steps"] for r in step["rows"]) == 1 + K
