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
# The tiny model's random weights give a flat distribution (logits about 0.4): its best two logits are often one
# bfloat16 step apart (0.002). A speculative row computed with other shapes may take the second one there, the
# near-tie the tiny gate already tolerates against the HF reference; after it the streams are unrelated.
NEAR_TIE = 0.0045


@pytest.fixture(scope="module")
def engine_runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_mtp")
    base = hpu_cpu_engine.run(tiny_checkpoint, out / "k0.json", "--logprobs", "2")
    spec = hpu_cpu_engine.run(tiny_checkpoint, out / "k3.json", "--k", str(K))
    return {"dir": out, "base": base, "spec": spec, "ckpt": Path(tiny_checkpoint)}


def _streams(record: dict) -> dict:
    return {tuple(p["prompt_ids"]): p["generated_ids"] for p in record["prompts"]}


def first_difference(base_prompt: dict, tokens: list[int]) -> int | None:
    """The first token where ``tokens`` leaves the base stream (None: equal). Fails unless the base's best two
    logprobs are within NEAR_TIE there and ``tokens`` took the second one."""
    want = base_prompt["generated_ids"]
    diff = next((i for i, (a, b) in enumerate(zip(tokens, want)) if a != b), None)
    if diff is None:
        assert len(tokens) == len(want), (len(tokens), len(want))
        return None
    top = sorted(base_prompt["top_logprobs"][diff].items(), key=lambda item: -item[1])
    margin = top[0][1] - top[1][1]
    assert margin <= NEAR_TIE and int(top[1][0]) == tokens[diff], (
        f"prompt of {len(base_prompt['prompt_ids'])} tokens leaves the non-speculative stream at token {diff}: "
        f"got {tokens[diff]}, want {want[diff]} (best two {top[:2]}, margin {margin:.4f})")
    return diff


def assert_same_streams(base: dict, run: dict) -> dict:
    """Every prompt's stream equals the base's up to a near-tie; returns {prompt: first difference or None}."""
    by_prompt = {tuple(p["prompt_ids"]): p for p in base["prompts"]}
    streams = _streams(run)
    assert streams.keys() == by_prompt.keys() and len(streams) >= 4
    return {prompt: first_difference(by_prompt[prompt], tokens) for prompt, tokens in streams.items()}


def _oracle_run(engine_runs: dict, accept: list[int], *options: str, k: int = K, env: dict | None = None,
                tag: str = "") -> dict:
    name = f"k{k}_accept_" + "".join(str(a) for a in accept) + "".join(o.strip("-")[:3] for o in options) + tag
    path = engine_runs["dir"] / f"{name}.oracle.json"
    path.write_text(json.dumps({"streams": engine_runs["base"]["prompts"], "accept": accept}), encoding="utf-8")
    return hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / f"{name}.json", "--k", str(k), "--oracle",
                              str(path), *options, env=env)


def _check_oracle(engine_runs: dict, run: dict, accept: list[int], k: int) -> None:
    """The stream is the base's (up to a near-tie) and every verification committed the forced count plus one."""
    diffs = {len(prompt): d for prompt, d in assert_same_streams(engine_runs["base"], run).items()}
    max_tokens = len(engine_runs["base"]["prompts"][0]["generated_ids"])
    by_req = defaultdict(list)
    for entry in run["oracle"]:
        by_req[entry["req"]].append(entry)
    checked = 0
    for entries in by_req.values():
        diff = diffs[entries[0]["prompt_len"]]
        for before, after in zip(entries, entries[1:]):
            if before["n_out"] + k + 1 > max_tokens:  # the scheduler cuts the drafts at max_tokens
                continue
            if diff is not None and before["n_out"] + k >= diff:  # past a near-tie the oracle drafts are wrong
                continue
            assert after["n_out"] - before["n_out"] == before["accept"] + 1, (before, after)
            checked += 1
    assert checked >= len(by_req)
    # the decode batches load from the accepted counts of the step before (1 after a prefill, fewer near
    # max_tokens, where the scheduler cuts the drafts)
    loaded = {a for step in run["decode_steps"] if step["num_accepted"] for a in step["num_accepted"] if a > 0}
    assert loaded and max(loaded) == max(accept) + 1 and loaded <= set(range(1, k + 2))
    assert max(r for step in run["decode_steps"] for r in step["rows"]) == 1 + k


def test_k3_gives_the_tokens_of_the_engine_without_speculation(engine_runs):
    diffs = assert_same_streams(engine_runs["base"], engine_runs["spec"])
    assert sum(d is None for d in diffs.values()) >= len(diffs) - 1  # at most one near-tie on the tiny model
    assert engine_runs["spec"]["prefix_caching"] is True


def test_the_engine_runs_verification_steps_of_one_plus_k_rows(engine_runs):
    rows = [r for step in engine_runs["spec"]["decode_steps"] for r in step["rows"]]
    assert rows and max(rows) == 1 + K


def test_the_drafter_reads_the_attention_groups_block_table(engine_runs):
    facts = engine_runs["spec"]["runner"]
    assert facts["groups"][0]["spec"] == "MambaSpec"  # the premise: group 0 is not the attention group
    assert facts["groups"][facts["attention_group"]]["spec"] == "FullAttentionSpec"
    calls = [call for call in engine_runs["spec"]["draft_calls"] if not call.get("fill")]
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
    engine_runs["oracle_" + "".join(str(a) for a in accept)] = run
    _check_oracle(engine_runs, run, accept, K)


def test_host_book_draft_positions_give_the_same_drafts(engine_runs):
    # VLLM_HPU_MTP_DRAFT_LOOP=1 (idea-sweep I02): the draft loop takes its positions from the runner's host books
    # instead of reading them back from the device; every draft and every token must be the same.
    accept = [2, 0, 3, 1]
    device = engine_runs.get("oracle_2031") or _oracle_run(engine_runs, accept)
    host = _oracle_run(engine_runs, accept, env={"VLLM_HPU_MTP_DRAFT_LOOP": "1"}, tag="_host")
    assert [e["drafter"] for e in host["oracle"]] == [e["drafter"] for e in device["oracle"]]
    assert [p["generated_ids"] for p in host["prompts"]] == [p["generated_ids"] for p in device["prompts"]]


@pytest.mark.parametrize("k,accept", [(1, [1, 0]), (2, [2, 0, 1]), (3, [2, 0, 3, 1]), (4, [4, 1, 3, 0, 2]),
                                      (5, [5, 2, 0, 4, 1, 3])],
                         ids=["k1", "k2", "k3", "k4", "k5"])
def test_the_draft_passes_in_one_graph_give_the_same_drafts(engine_runs, k, accept):
    # VLLM_HPU_MTP_DRAFT_GRAPH=1: the k passes of every decode draft run in one forward (MtpDraftLoop, one HPU graph
    # on a card) with the metadata of the later passes built first. Against the per-pass drafter with the same
    # host-book positions (VLLM_HPU_MTP_DRAFT_LOOP=1): the drafter's own ids and every stream are the same, at every
    # accepted count 0..k (oracle drafts), and every decode draft took the one-graph path (prefill drafts do not).
    loop = {"VLLM_HPU_MTP_DRAFT_LOOP": "1"}
    per_pass = _oracle_run(engine_runs, accept, k=k, env=loop, tag="_loop")
    graph = _oracle_run(engine_runs, accept, k=k, env={**loop, "VLLM_HPU_MTP_DRAFT_GRAPH": "1"}, tag="_graph")
    assert [e["drafter"] for e in graph["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in graph["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    decode_drafts = [call for call in graph["draft_calls"] if call.get("decode")]
    assert decode_drafts and graph["runner"]["draft_graph_calls"] == len(decode_drafts)
    assert any(not call.get("decode") and not call.get("fill") for call in graph["draft_calls"])  # prefill drafts
    assert per_pass["runner"]["draft_graph_calls"] == 0
    _check_oracle(engine_runs, graph, accept, k)


def test_the_draft_passes_in_one_graph_with_several_requests_at_once(engine_runs):
    # Every prompt in one generate call: decode drafts of several requests (padded batch rows, requests at different
    # positions, prefills joining a running batch). The drafter's ids and the streams equal the per-pass drafter's.
    loop = {"VLLM_HPU_MTP_DRAFT_LOOP": "1"}
    accept = [3, 1, 0, 2]
    per_pass = _oracle_run(engine_runs, accept, "--concurrent", env=loop, tag="_loop")
    graph = _oracle_run(engine_runs, accept, "--concurrent", env={**loop, "VLLM_HPU_MTP_DRAFT_GRAPH": "1"},
                        tag="_graph")
    assert [e["drafter"] for e in graph["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in graph["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    assert max(call["rows"] for call in graph["draft_calls"] if call.get("decode")) > 1
    assert graph["runner"]["draft_graph_calls"] == len([c for c in graph["draft_calls"] if c.get("decode")])


def test_the_one_graph_draft_switched_at_run_time(engine_runs):
    # VLLM_HPU_MTP_DRAFT_GRAPH in the runtime switch file (VLLM_HPU_RUNTIME_SWITCH_FILE, an A/B on one server):
    # the server starts with the per-pass drafter, takes the one-graph draft after the second prompt and drops it
    # after the fourth. Every draft and every stream equal the per-pass drafter's; the decode drafts of prompts 3 and
    # 4 ran in one graph, the others did not.
    loop = {"VLLM_HPU_MTP_DRAFT_LOOP": "1"}
    accept = [2, 0, 3, 1]
    per_pass = _oracle_run(engine_runs, accept, env=loop, tag="_loop")
    switch_file = engine_runs["dir"] / "draft_graph.switches"
    run = _oracle_run(engine_runs, accept, "--switch-after", "2", "VLLM_HPU_MTP_DRAFT_GRAPH=1", "--switch-after", "4",
                      "VLLM_HPU_MTP_DRAFT_GRAPH=0", env={**loop, "VLLM_HPU_RUNTIME_SWITCH_FILE": str(switch_file)},
                      tag="_switch")
    assert [e["drafter"] for e in run["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in run["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    start = 0
    for number, prompt in enumerate(run["prompts"], 1):
        calls = run["draft_calls"][start:prompt["draft_calls_end"]]
        start = prompt["draft_calls_end"]
        decode = [call for call in calls if call.get("decode")]
        assert decode, number
        assert all(call["graph"] == (number in (3, 4)) for call in decode), (number, decode)
        assert not any(call.get("graph") for call in calls if not call.get("decode"))
    assert 0 < run["runner"]["draft_graph_calls"] < len([c for c in run["draft_calls"] if c.get("decode")])


def test_the_draft_graph_and_the_sparse_metadata_switched_together(engine_runs):
    # The 2 x 2 of an operator's A/B on one server (the probe's switch set graph-md): VLLM_HPU_MTP_DRAFT_GRAPH and
    # VLLM_HPU_FLASH_NEXT_DECODE_METADATA (engine-step's I01: the target's verification steps and the drafter's later
    # passes without paged-attention buffers) switched between prompts at k = 2. Every draft and every stream equal
    # the per-pass drafter's; the one-graph path ran exactly where the switch file put it.
    loop = {"VLLM_HPU_MTP_DRAFT_LOOP": "1"}
    accept = [2, 0, 1]
    per_pass = _oracle_run(engine_runs, accept, k=2, env=loop, tag="_loop")
    switch_file = engine_runs["dir"] / "graph_md.switches"
    arms = {1: ("1", "0"), 2: ("0", "1"), 3: ("1", "1"), 4: ("0", "0")}
    options = []
    for after, (graph, md) in arms.items():
        options += ["--switch-after", str(after),
                    f"VLLM_HPU_MTP_DRAFT_GRAPH={graph};VLLM_HPU_FLASH_NEXT_DECODE_METADATA={md}"]
    run = _oracle_run(engine_runs, accept, *options, k=2,
                      env={**loop, "VLLM_HPU_RUNTIME_SWITCH_FILE": str(switch_file)}, tag="_graph_md")
    assert [e["drafter"] for e in run["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in run["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    start = 0
    for number, prompt in enumerate(run["prompts"], 1):
        calls = [c for c in run["draft_calls"][start:prompt["draft_calls_end"]] if c.get("decode")]
        start = prompt["draft_calls_end"]
        graph = arms.get(number - 1, ("0", "0"))[0] == "1"  # the switch after prompt n acts from prompt n + 1
        assert calls and all(c["graph"] == graph for c in calls), (number, calls)
    _check_oracle(engine_runs, run, accept, 2)


CHAIN_BASE = {"VLLM_HPU_MTP_DRAFT_LOOP": "1", "VLLM_HPU_FLASH_NEXT_DECODE_METADATA": "1"}
CHAIN = {**CHAIN_BASE, "VLLM_HPU_MTP_DRAFT_GRAPH": "1", "VLLM_HPU_MTP_DRAFT_CHAIN": "1"}


@pytest.mark.parametrize("k,accept", [(1, [1, 0]), (2, [2, 0, 1]), (3, [2, 0, 3, 1])], ids=["k1", "k2", "k3"])
def test_the_chained_draft_gives_the_drafts_of_the_per_pass_drafter(engine_runs, k, accept):
    # VLLM_HPU_MTP_DRAFT_CHAIN=1: the draft is issued behind the verification before the host reads its result;
    # its inputs (committed rows, last row, tokens, positions) and its later passes' block tables and slots come
    # from the device. Against the per-pass drafter with the same sparse decode metadata: the drafter's own ids and
    # every stream are the same at every accepted count, and no decode draft took the host-prepared path.
    per_pass = _oracle_run(engine_runs, accept, k=k, env=CHAIN_BASE, tag="_loop_md")
    chain = _oracle_run(engine_runs, accept, k=k, env=CHAIN, tag="_chain")
    assert [e["drafter"] for e in chain["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in chain["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    assert chain["runner"]["draft_chain_calls"] > 0 and per_pass["runner"]["draft_chain_calls"] == 0
    assert not [c for c in chain["draft_calls"] if c.get("decode")]
    _check_oracle(engine_runs, chain, accept, k)


def test_the_chained_draft_with_several_requests_at_once(engine_runs):
    # Every prompt in one generate call: padded batch rows, requests at different positions and acceptance counts.
    accept = [3, 1, 0, 2]
    per_pass = _oracle_run(engine_runs, accept, "--concurrent", env=CHAIN_BASE, tag="_loop_md")
    chain = _oracle_run(engine_runs, accept, "--concurrent", env=CHAIN, tag="_chain")
    assert [e["drafter"] for e in chain["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in chain["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    assert chain["runner"]["draft_chain_calls"] > 0
    assert not [c for c in chain["draft_calls"] if c.get("decode")]


def test_the_chain_runs_the_greedy_rejection_in_its_graph(engine_runs):
    # Stage B: with the verification's per-row greedy ids on the device (VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 here; the
    # verify graph's own ids with engine-step's VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1), the chained draft's graph runs
    # the greedy rejection too. Drafts and streams equal the per-pass drafter's with the same flags.
    accept = [2, 0, 3, 1]
    vp = {"VLLM_HPU_VOCAB_PARALLEL_LOGITS": "1"}
    per_pass = _oracle_run(engine_runs, accept, env={**CHAIN_BASE, **vp}, tag="_loop_md_vp")
    chain = _oracle_run(engine_runs, accept, env={**CHAIN, **vp}, tag="_chain_vp")
    assert [e["drafter"] for e in chain["oracle"]] == [e["drafter"] for e in per_pass["oracle"]]
    assert [p["generated_ids"] for p in chain["prompts"]] == [p["generated_ids"] for p in per_pass["prompts"]]
    assert chain["runner"]["draft_chain_calls"] > 0
    assert not [c for c in chain["draft_calls"] if c.get("decode")]
    _check_oracle(engine_runs, chain, accept, K)


def test_a_sampled_request_keeps_the_draft_after_the_host_read(engine_runs):
    # The chain is greedy only so far: a step with a sampled request keeps today's path (logged once), and its
    # stream gets its tokens.
    env = {**CHAIN, "VLLM_HPU_SPEC_SAMPLED_VERIFY": "1"}
    run = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "chain_sampled.json", "--k", str(K),
                             "--temperature", "1.0", "--max-tokens", "8", env=env)
    for prompt in run["prompts"]:
        assert len(prompt["generated_ids"]) == 8
        assert all(0 <= t < 248320 for t in prompt["generated_ids"])
    assert run["runner"]["draft_chain_calls"] == 0
    assert [c for c in run["draft_calls"] if c.get("decode")]
    log = (engine_runs["dir"] / "chain_sampled.json.log").read_text(encoding="utf-8", errors="replace")
    assert "a sampled request (greedy only so far)" in log


@pytest.mark.parametrize("k", [4, 5], ids=["k4", "k5"])
def test_oracle_drafts_at_the_sizes_above_the_recipe(engine_runs, k):
    # k + 1 lattice entries exceed the kernel blocks of a request's first block (two per 256-token block here, four
    # per 512-token block on the card): the attention group's rows must stay out of the lattice check.
    accept = list(range(k, -1, -1))
    _check_oracle(engine_runs, _oracle_run(engine_runs, accept, k=k), accept, k)


def test_oracle_drafts_with_the_fused_kernels_reference_ops(engine_runs):
    # VLLM_HPU_GDN_CORE_DECODE / _VERIFY and VLLM_HPU_HC_FUSED run their Python reference kernels on CPU: the
    # speculative step through the code that calls the fused ops (lattice -> verify op, the HC fused path).
    env = {"VLLM_HPU_GDN_CORE_DECODE": "1", "VLLM_HPU_GDN_CORE_VERIFY": "1", "VLLM_HPU_HC_FUSED": "1"}
    fused_base = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "fused_k0.json", "--logprobs", "2",
                                    env=env)
    path = engine_runs["dir"] / "fused.oracle.json"
    path.write_text(json.dumps({"streams": fused_base["prompts"], "accept": [2, 0, 3, 1]}), encoding="utf-8")
    run = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "fused.json", "--k", str(K), "--oracle",
                             str(path), env=env)
    assert_same_streams(fused_base, run)
    assert max(r for step in run["decode_steps"] for r in step["rows"]) == 1 + K


def test_sampled_requests_through_the_speculative_engine(engine_runs):
    # temperature 1.0: the engine's RejectionSampler runs the HPU sampled path (apply_sampling_constraints with
    # the torch expand, sampled_rejection_sample) and the bonus rows through the sampler. No per-request seed:
    # the runner makes a torch.Generator on its device, which the CPU stand-in has no backend for (the seeded rows
    # are checked in sampler/test_hpu_spec_sampled_verify.py).
    env = {"VLLM_HPU_SPEC_SAMPLED_VERIFY": "1"}
    run = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "sampled.json", "--k", str(K),
                             "--temperature", "1.0", "--max-tokens", "8", env=env)
    for prompt in run["prompts"]:
        assert len(prompt["generated_ids"]) == 8
        assert all(0 <= t < 248320 for t in prompt["generated_ids"])
    assert max(r for step in run["decode_steps"] for r in step["rows"]) == 1 + K


def test_sampled_requests_over_candidates(engine_runs):
    # VLLM_HPU_SPEC_SAMPLED_VERIFY=2 with the checkpoint's default sampling (temperature 1.0, top_k 20, top_p 0.95):
    # the verification steps take the candidate path (hpu_candidate_verify.py), with its host noise and its fp32 id
    # table, and every stream gets its tokens.
    env = {"VLLM_HPU_SPEC_SAMPLED_VERIFY": "2"}
    run = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "sampled_candidates.json", "--k", str(K),
                             "--temperature", "1.0", "--top-k", "20", "--top-p", "0.95", "--max-tokens", "8",
                             env=env)
    for prompt in run["prompts"]:
        assert len(prompt["generated_ids"]) == 8
        assert all(0 <= t < 248320 for t in prompt["generated_ids"])
    assert max(r for step in run["decode_steps"] for r in step["rows"]) == 1 + K
    assert run["runner"]["candidate_verify_steps"] > 0


def test_a_sampled_request_is_refused_without_the_flag(engine_runs):
    with pytest.raises(RuntimeError, match="VLLM_HPU_SPEC_SAMPLED_VERIFY=1"):
        hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "sampled_refused.json", "--k", str(K),
                           "--temperature", "1.0", "--max-tokens", "4", env={"VLLM_HPU_SPEC_SAMPLED_VERIFY": "0"})


def test_the_draft_runs_over_every_chunk_of_a_chunked_prompt(engine_runs):
    # VLLM_HPU_MTP_CHUNK_KV=1: with a 512-token chunk budget the 700- and 2600-token prompts take several chunks;
    # the draft now runs over the chunks that complete no prompt too (its KV then covers the whole prompt), and
    # the stream stays the non-speculative one.
    options = ("--k", str(K), "--max-num-batched-tokens", "512")
    off = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "chunked_off.json", *options)
    on = hpu_cpu_engine.run(engine_runs["ckpt"], engine_runs["dir"] / "chunked_on.json", *options,
                            env={"VLLM_HPU_MTP_CHUNK_KV": "1"})
    assert not [call for call in off["draft_calls"] if call.get("fill")]
    fills = [call for call in on["draft_calls"] if call.get("fill")]
    assert fills and min(call["first_position"] for call in fills) == 0
    assert max(call["last_position"] for call in fills) >= 2048  # the 2600-token prompt's chunks before its last
    assert_same_streams(engine_runs["base"], on)
    assert_same_streams(engine_runs["base"], off)


def test_requests_after_a_block_committed_during_speculative_decode(engine_runs):
    """Card job 00133 (spec-lookup, TP4, n-gram k = 5, fused verify op, prefix caching): the first request's decode
    completed a recurrent block (667 + 384 tokens across the 896-token block) and ran; the next request died in the
    verify op, its conv pool without data (``ValidateSyncInputTensors tensor_data is empty``). The align-mode copies
    of that block (the boundary commit after sampling, the pre-step copy before the forward) wrote the conv window
    through a slice of the GDN conv pool (``recurrent_commit.commit_conv``), which the lazy bridge turns into
    ``hpu::slice_insert``. On the CPU engine the recurrent block is 256 tokens:

    * turn 1: a 240-token prompt and 48 tokens: the decode crosses 256, so its block is committed;
    * turn 2: the same prompt: its decode completes a block whose hash is already cached;
    * turn 3: the agent's next turn, turn 1's prompt and answer and 30 more tokens: a prefix-cache hit on the
      committed block.

    MTP k = 3 with prefix caching gives the tokens of the engine without speculation, the copies ran, and none of
    them wrote through a slice of a pool (``--slice-write-probe``).
    """
    longest = max((p["prompt_ids"] for p in engine_runs["base"]["prompts"]), key=len)
    first, extra = longest[:240], longest[1000:1030]
    out = engine_runs["dir"]
    turns = out / "turns_prompts.json"
    turns.write_text(json.dumps({"prompts": [{"prompt_ids": first}, {"prompt_ids": first}]}), encoding="utf-8")
    # --next-turn: turn 3 is built from this engine's own turn-1 answer (greedy MTP and the engine without
    # speculation may part at a near tie, so a prompt built from the other engine's answer would share no block)
    options = ("--prompts", str(turns), "--max-tokens", "48", "--next-turn", json.dumps(extra))
    base = hpu_cpu_engine.run(engine_runs["ckpt"], out / "turns_k0.json", *options, "--logprobs", "2")
    spec = hpu_cpu_engine.run(engine_runs["ckpt"], out / "turns_k3.json", *options, "--k", str(K),
                              "--slice-write-probe")
    probe = spec["slice_write_probe"]
    assert probe["pools"] > 0 and probe["conv_pairs"] > 0 and probe["ssm_pairs"] > 0, probe
    assert probe["violations"] == []
    assert [len(p["prompt_ids"]) for p in spec["prompts"]] == [240, 240, 240 + 48 + 30]
    for base_prompt, spec_prompt in zip(base["prompts"][:2], spec["prompts"][:2]):
        first_difference(base_prompt, spec_prompt["generated_ids"])
    # the same prompt again on the same engine: the same tokens
    assert spec["prompts"][1]["generated_ids"] == spec["prompts"][0]["generated_ids"]
    # Turn 3's prefix-cache hit: 0 on this build with or without MTP (prompt chunks end at the engine's block, not
    # the recurrent block: prefill's VLLM_HPU_PREFIX_ALIGN_SPLIT, BRIEF 23:46)
    assert spec["prompts"][2]["num_cached_tokens"] == base["prompts"][2]["num_cached_tokens"]


def test_per_step_timing_records_through_the_runner(engine_runs, tmp_path):
    """``VLLM_HPU_MTP_TIMING=2`` through the runner (``mtp_step_timer.py``): one record per speculative step, with its
    rows and, per decode request, the tokens it committed in the step (the oracle's forced count plus one) and in its
    previous step."""
    path = tmp_path / "timing.oracle.json"
    path.write_text(json.dumps({"streams": engine_runs["base"]["prompts"], "accept": [2, 0, 3, 1]}), encoding="utf-8")
    hpu_cpu_engine.run(engine_runs["ckpt"], tmp_path / "timing.json", "--k", str(K), "--oracle", str(path),
                       env={"VLLM_HPU_MTP_TIMING": "2"}, cwd=tmp_path)
    files = list(tmp_path.glob("mtp_step_record_r*.jsonl"))
    assert len(files) == 1, files
    records = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    verify = [r for r in records if r["kind"] == "verify"]
    assert verify and max(r["rows"] for r in verify) == 1 + K
    assert all(r["rows"] <= r["requests"] * (1 + K) and r["verify_ms"] > 0 and r["draft_ms"] > 0 for r in verify)
    seen = set()
    for r in verify:
        seen |= set(r["committed"])
    assert seen <= {str(n) for n in range(1, K + 2)} and {"1", str(K + 1)} <= seen  # accept 0 and accept k ran
    chained = 0
    for before, after in zip(verify, verify[1:]):  # one request at a time: its steps follow each other
        if after["prev_committed"] != {"none": 1}:
            assert after["prev_committed"] == before["committed"], (before, after)
            chained += 1
    assert chained >= len(verify) // 2
