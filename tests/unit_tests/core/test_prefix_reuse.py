# SPDX-License-Identifier: Apache-2.0
"""Prefix reuse of the 32-agent workload under the rental settings, with the engine's real scheduler.

``flash_next_sched_harness.Sim``: TP8 layout of a card start (scheduler block 512, 896 recurrent pool slots),
prefix caching in align mode, the plugin's recurrent pool coordinator, 8,192 batched tokens and the 2,048-token
long-prefill cap, ``--prefix-cache-retention-interval`` 0 (vLLM's default; serve.py does not set it). The
stand-in runner records every read of a recurrent state slot that holds another prefix (``Sim.violations``).

What a GDN state can be reused from: with retention 0 only the "replay boundary" (the prompt's last token,
aligned down to the 512-token block) and the shared-prefix junctions are kept, and a state exists only where a
scheduled chunk ended (align mode ends every prompt chunk on a block).
"""

import pytest

from tests.unit_tests.core import flash_next_sched_harness as h

BLOCK = 512


def _turns(sim: h.Sim, first_prompt: list[int], turns: int, generate: int = 8, new_tokens: int = 600):
    prompts, prompt = [], first_prompt
    for turn in range(turns):
        rid = f"t{turn + 1}"
        sim.add(rid, prompt, max_tokens=generate)
        sim.run_until_finished([rid], 400 + generate)
        prompts.append(prompt)
        prompt = prompt + sim.outputs[rid] + h.own_tokens(0, new_tokens, salt=turn + 1)
    return prompts


def test_later_turns_hit_the_previous_prompt_end():
    sim = h.Sim()
    prompts = _turns(sim, h.agent_prompt(0, 24_000), turns=3)
    for turn in (2, 3):
        previous = prompts[turn - 2]
        cached = sim.admissions[f"t{turn}"].cached_tokens
        assert cached == (len(previous) - 1) // BLOCK * BLOCK, (turn, cached, len(previous))
        # what is prefilled again: the previous turn's last partial block, its output and the new tokens
        assert len(prompts[turn - 1]) - cached < 2 * BLOCK + 8 + 600
    assert sim.violations == []


def test_the_shared_block_is_reused_once_a_junction_is_known():
    sim = h.Sim()
    sim.add("a0", h.agent_prompt(0, 24_000), max_tokens=4)
    sim.run_until_finished(["a0"], 200)
    # the second agent finds the junction at the end of the shared block but computes it (no state there yet)
    sim.add("a1", h.agent_prompt(1, 24_000), max_tokens=4)
    sim.run_until_finished(["a1"], 200)
    assert sim.admissions["a1"].shared_prefix_boundary == h.SHARED_TOKENS
    assert sim.admissions["a1"].cached_tokens == 0
    # every agent after it starts behind the shared block, also four at once
    later = [f"b{i}" for i in range(4)]
    for i, rid in enumerate(later):
        sim.add(rid, h.agent_prompt(10 + i, 24_000), max_tokens=4)
    sim.run_until_finished(later, 400)
    assert [sim.admissions[r].cached_tokens for r in later] == [h.SHARED_TOKENS] * 4
    assert sim.violations == []


def test_a_block_aligned_prompt_misses_its_next_turn_at_retention_zero():
    """DEFECT (engine, reported by the prefill track): a prompt of a multiple of 512 tokens keeps its replay
    boundary one block before its end (``num_prompt - 1`` aligned down), where no chunk ended, so no state was
    ever written there and its next turn re-prefills the whole prompt. Any periodic retention (2,048 here)
    keeps the chunk-end state and the hit comes back."""
    aligned = h.agent_prompt(0, 24_576)
    sim = h.Sim()
    _turns(sim, aligned, turns=2)
    assert sim.admissions["t2"].cached_tokens == 0
    periodic = h.Sim(h.Settings(retention_interval=2048))
    _turns(periodic, aligned, turns=2)
    assert periodic.admissions["t2"].cached_tokens == len(aligned)
    assert sim.violations == [] and periodic.violations == []


# ----- VLLM_HPU_PREFIX_KEEP_PROMPT_END=1: the state at a prompt end that falls on a scheduler block is kept -----

ENV_KEEP = "VLLM_HPU_PREFIX_KEEP_PROMPT_END"


def test_keeping_the_prompt_end_gives_a_block_aligned_prompt_its_next_turn(monkeypatch):
    monkeypatch.setenv(ENV_KEEP, "1")
    aligned = h.agent_prompt(0, 24_576)  # 48 blocks of 512
    sim = h.Sim()
    prompts = _turns(sim, aligned, turns=3)
    # the next turn starts behind the whole first prompt: nothing of it is prefilled again
    assert sim.admissions["t2"].cached_tokens == len(aligned)
    assert len(prompts[1]) - sim.admissions["t2"].cached_tokens == 8 + 600
    # the turn after it follows the usual rule (its previous prompt does not end on a block)
    assert sim.admissions["t3"].cached_tokens == (len(prompts[1]) - 1) // BLOCK * BLOCK
    assert sim.violations == []


def test_keeping_the_prompt_end_changes_nothing_for_other_prompts(monkeypatch):
    def run() -> list[tuple[int, int, int]]:
        sim = h.Sim()
        _turns(sim, h.agent_prompt(0, 24_000), turns=3)
        sim.add("a1", h.agent_prompt(1, 24_000), max_tokens=4)
        sim.run_until_finished(["a1"], 200)
        assert sim.violations == []
        return [(a.prompt_tokens, a.cached_tokens, a.first_chunk) for a in sim.admissions.values()]

    monkeypatch.delenv(ENV_KEEP, raising=False)
    off = run()
    monkeypatch.setenv(ENV_KEEP, "1")
    assert run() == off


def test_keeping_the_prompt_end_at_the_tp4_block(monkeypatch):
    block = 896  # TP4 + EP4: twice the GDN state per rank, the same KV bytes per token
    aligned = h.agent_prompt(0, 28 * block)  # chunks of 2 blocks under the 2,048-token cap: no chunk ends at 27
    monkeypatch.delenv(ENV_KEEP, raising=False)
    miss = h.Sim(h.Settings(tp=4))
    assert miss.block_size == block
    _turns(miss, aligned, turns=2)
    assert miss.admissions["t2"].cached_tokens == 0  # the defect, at this layout too
    monkeypatch.setenv(ENV_KEEP, "1")
    hit = h.Sim(h.Settings(tp=4))
    _turns(hit, aligned, turns=2)
    assert hit.admissions["t2"].cached_tokens == len(aligned)
    assert miss.violations == [] and hit.violations == []


@pytest.mark.parametrize("tp,block,blocks", [(8, 512, 45), (4, 896, 27)])
def test_an_aligned_prompt_one_block_past_a_chunk_end_only_loses_a_block(monkeypatch, tp, block, blocks):
    """Which aligned prompts miss completely: those whose last-but-one block boundary is not a chunk end. A
    prompt that ends one block after a chunk end (chunks are 4 blocks at TP8 and 2 at TP4 under the 2,048-token
    cap) finds a state at its replay boundary and re-prefills one block; with the flag it re-prefills none."""
    aligned = h.agent_prompt(0, blocks * block)
    monkeypatch.delenv(ENV_KEEP, raising=False)
    plain = h.Sim(h.Settings(tp=tp))
    _turns(plain, aligned, turns=2)
    assert plain.admissions["t2"].cached_tokens == (blocks - 1) * block
    monkeypatch.setenv(ENV_KEEP, "1")
    kept = h.Sim(h.Settings(tp=tp))
    _turns(kept, aligned, turns=2)
    assert kept.admissions["t2"].cached_tokens == blocks * block
    assert plain.violations == [] and kept.violations == []


def test_how_often_an_aligned_prompt_misses_completely(monkeypatch):
    # TP8: of the four aligned lengths 44..47 blocks + 1, three miss (chunk ends are every 4 blocks)
    monkeypatch.delenv(ENV_KEEP, raising=False)
    cached = {}
    for blocks in (45, 46, 47, 48):
        sim = h.Sim()
        _turns(sim, h.agent_prompt(0, blocks * BLOCK), turns=2)
        cached[blocks] = sim.admissions["t2"].cached_tokens
        assert sim.violations == []
    assert cached == {45: 44 * BLOCK, 46: 0, 47: 0, 48: 0}


def test_keeping_the_prompt_end_costs_one_retained_state_per_group(monkeypatch):
    """What the flag costs: the recurrent blocks an aligned prompt leaves in the prefix cache when it ends."""
    def cached_recurrent_blocks() -> int:
        sim = h.Sim()
        sim.add("t1", h.agent_prompt(0, 24_576), max_tokens=8)
        sim.run_until_finished(["t1"], 400)
        assert sim.violations == []
        pool = sim.coordinator.recurrent_pool
        return sum(1 for block in pool.blocks if block.block_hash is not None)

    monkeypatch.delenv(ENV_KEEP, raising=False)
    without = cached_recurrent_blocks()
    monkeypatch.setenv(ENV_KEEP, "1")
    with_flag = cached_recurrent_blocks()
    groups = len(h.Sim().recurrent_groups)
    assert with_flag - without == groups  # one state per recurrent group, for this one prompt


def test_a_replay_of_the_same_block_aligned_prompt_still_misses(monkeypatch):
    # the same prompt again must compute its last token, so its hit could be one block before the end at
    # most, where no chunk ended: the flag does not change that (a retry of an aligned prompt stays a miss)
    monkeypatch.setenv(ENV_KEEP, "1")
    aligned = h.agent_prompt(0, 24_576)
    sim = h.Sim()
    for rid in ("first", "again"):
        sim.add(rid, aligned, max_tokens=4)
        sim.run_until_finished([rid], 400)
    assert sim.admissions["again"].cached_tokens == 0
    assert sim.violations == []


def test_the_previous_output_is_prefilled_again_at_retention_zero(monkeypatch):
    """What a turn costs today: the states a request produces while decoding are not kept, so the next turn
    hits at the previous PROMPT's end aligned down and prefills the previous output again. Dense retention
    keeps them (and every chunk-end state of every prompt)."""
    monkeypatch.delenv(ENV_KEEP, raising=False)
    generate, new_tokens = 1100, 600
    sparse = h.Sim()
    prompts = _turns(sparse, h.agent_prompt(0, 24_000), turns=2, generate=generate, new_tokens=new_tokens)
    cached = sparse.admissions["t2"].cached_tokens
    assert cached == (len(prompts[0]) - 1) // BLOCK * BLOCK == 23_552
    # the last partial block of the prompt, the 1,100 output tokens and the 600 new ones
    assert len(prompts[1]) - cached == 448 + generate + new_tokens
    dense = h.Sim(h.Settings(retention_interval=BLOCK))
    dense_prompts = _turns(dense, h.agent_prompt(0, 24_000), turns=2, generate=generate, new_tokens=new_tokens)
    # the last boundary the first request reached while decoding (its last output token is never computed)
    reached = (len(dense_prompts[0]) + generate - 1) // BLOCK * BLOCK
    assert dense.admissions["t2"].cached_tokens == reached == 25_088
    assert len(dense_prompts[1]) - reached == 12 + new_tokens
    assert sparse.violations == [] and dense.violations == []


# ----- VLLM_HPU_PREFIX_KEEP_DECODE_END=1: the last block boundary a request reached while decoding is kept -----

ENV_KEEP_DECODE = "VLLM_HPU_PREFIX_KEEP_DECODE_END"


def _cached_recurrent_blocks(sim: h.Sim) -> int:
    return sum(1 for block in sim.coordinator.recurrent_pool.blocks if block.block_hash is not None)


def test_keeping_the_decode_end_spares_the_next_turn_the_previous_output(monkeypatch):
    monkeypatch.setenv(ENV_KEEP_DECODE, "1")
    generate, new_tokens = 1100, 600
    sim = h.Sim()
    prompts = _turns(sim, h.agent_prompt(0, 24_000), turns=3, generate=generate, new_tokens=new_tokens)
    for turn in (2, 3):
        previous = prompts[turn - 2]
        # the last block boundary the previous request computed (its last output token is never fed back)
        reached = (len(previous) + generate - 1) // BLOCK * BLOCK
        cached = sim.admissions[f"t{turn}"].cached_tokens
        assert cached == reached, (turn, cached, reached)
        assert len(prompts[turn - 1]) - cached < BLOCK + new_tokens  # under one block of the old turn is left
    assert sim.admissions["t2"].cached_tokens == 25_088  # 23,552 without the flag: 1,536 tokens fewer to prefill
    assert sim.violations == []


def test_keeping_the_decode_end_keeps_one_decode_state_per_request(monkeypatch):
    """A long output crosses many block boundaries; only the last one stays in the prefix cache, next to the
    prompt's replay boundary, so the pool is not flooded with states no turn can use."""
    def cached_after(generate: int) -> tuple[int, int]:
        sim = h.Sim()
        sim.add("t1", h.agent_prompt(0, 24_000), max_tokens=generate)
        sim.run_until_finished(["t1"], 400 + generate)
        assert sim.violations == []
        return _cached_recurrent_blocks(sim), len(sim.recurrent_groups)

    monkeypatch.delenv(ENV_KEEP_DECODE, raising=False)
    plain, groups = cached_after(3000)
    monkeypatch.setenv(ENV_KEEP_DECODE, "1")
    kept, _ = cached_after(3000)  # the output crosses 6 block boundaries
    assert kept - plain == groups
    short, _ = cached_after(40)  # no boundary crossed while decoding: nothing more is kept
    assert short == plain


def test_keeping_the_decode_end_changes_nothing_when_no_boundary_is_crossed(monkeypatch):
    def run() -> list[tuple[int, int, int]]:
        sim = h.Sim()
        _turns(sim, h.agent_prompt(0, 24_000), turns=3)  # 8 output tokens per turn
        assert sim.violations == []
        return [(a.prompt_tokens, a.cached_tokens, a.first_chunk) for a in sim.admissions.values()]

    monkeypatch.delenv(ENV_KEEP_DECODE, raising=False)
    off = run()
    monkeypatch.setenv(ENV_KEEP_DECODE, "1")
    assert run() == off


def test_keeping_the_decode_end_with_several_agents_and_at_tp4(monkeypatch):
    monkeypatch.setenv(ENV_KEEP_DECODE, "1")
    generate, new_tokens = 1100, 600
    for tp, block in ((8, 512), (4, 896)):
        sim = h.Sim(h.Settings(tp=tp))
        prompts = {agent: h.agent_prompt(agent, 20_000 + 300 * agent) for agent in range(4)}
        for turn in range(3):
            ids = [f"a{agent}-t{turn}" for agent in prompts]
            for agent, rid in zip(prompts, ids):
                sim.add(rid, prompts[agent], max_tokens=generate)
            sim.run_until_finished(ids, 4000)
            for agent, rid in zip(list(prompts), ids):
                if turn:
                    reached = (previous_len[agent] + generate - 1) // block * block
                    assert sim.admissions[rid].cached_tokens == reached, (tp, rid)
            previous_len = {agent: len(prompt) for agent, prompt in prompts.items()}
            prompts = {agent: prompts[agent] + sim.outputs[rid] + h.own_tokens(agent, new_tokens, salt=turn + 1)
                       for agent, rid in zip(prompts, ids)}
        assert sim.violations == [], (tp, sim.violations[:3])


def test_both_keep_flags_together_on_a_block_aligned_prompt(monkeypatch):
    monkeypatch.setenv(ENV_KEEP, "1")
    monkeypatch.setenv(ENV_KEEP_DECODE, "1")
    aligned = h.agent_prompt(0, 24_576)
    short = h.Sim()
    _turns(short, aligned, turns=2)  # 8 output tokens: the prompt end is the furthest state
    assert short.admissions["t2"].cached_tokens == len(aligned)
    long = h.Sim()
    _turns(long, aligned, turns=2, generate=1100)
    assert long.admissions["t2"].cached_tokens == (len(aligned) + 1100 - 1) // BLOCK * BLOCK
    assert short.violations == [] and long.violations == []


def test_keeping_the_decode_end_for_speculative_blocks_needs_the_runner_s_boundary_commit():
    """Upstream, a step of 1 + k tokens that straddles a block boundary keeps its running state in the block of its
    last scheduled token, so the earlier block's slot is not the state at the boundary. The HPU runner commits that
    state into the completed block (``_commit_recurrent_boundaries``) whenever ``recurrent_spec_enabled`` with prefix
    caching, which the platform hook records (VLLM_HPU_RECURRENT_BOUNDARY_COMMITS): a speculative group is kept
    only then, and is left as it is when the runner does not commit or nothing was recorded."""
    from types import SimpleNamespace

    from vllm_gaudi.v1.core.recurrent_pool_coordinator import boundary_commits_known, keep_last_decode_boundary
    from vllm_gaudi.v1.worker.recurrent_lattice import align_running_block

    def untouched(*_args, **_kwargs):
        raise AssertionError("not called")

    def manager(speculative_blocks: int) -> SimpleNamespace:
        return SimpleNamespace(kv_cache_spec=SimpleNamespace(num_speculative_blocks=speculative_blocks),
                               cache_blocks=untouched, reachable_block_mask=untouched,
                               pop_blocks_for_free=untouched)

    for commits in (None, False):
        refused = manager(3)
        assert keep_last_decode_boundary(refused, commits) is False
        assert refused.cache_blocks is untouched and refused.reachable_block_mask is untouched
    kept = manager(3)
    assert keep_last_decode_boundary(kept, True) is True and kept.cache_blocks is not untouched
    plain = manager(0)  # no speculative blocks: kept whatever the runner does
    assert keep_last_decode_boundary(plain, None) is True
    # the platform hook's record
    assert boundary_commits_known({}) is None and boundary_commits_known({"VLLM_HPU_RECURRENT_BOUNDARY_COMMITS": "0"})\
        is False and boundary_commits_known({"VLLM_HPU_RECURRENT_BOUNDARY_COMMITS": "1"}) is True
    # the upstream reason, in the lattice's own terms: 4 tokens scheduled 2 before a boundary run in the NEXT block
    assert align_running_block(context_len=510, num_scheduled=4, block_size=BLOCK) == 1
    assert align_running_block(context_len=510, num_scheduled=1, block_size=BLOCK) == 0
    assert align_running_block(context_len=511, num_scheduled=1, block_size=BLOCK) == 0  # ends ON the boundary


def test_the_platform_hook_records_the_runner_s_boundary_commit(monkeypatch):
    from types import SimpleNamespace

    from vllm_gaudi.platform import _configure_recurrent_state
    from vllm_gaudi.v1.core.recurrent_pool_coordinator import ENV_BOUNDARY_COMMITS

    start = h.server_start(8)
    monkeypatch.setenv("VLLM_HPU_RECURRENT_STATE_POOL", "1")
    monkeypatch.setenv("VLLM_HPU_RECURRENT_STATE_POOL_SLOTS", str(start.pool_slots))
    for speculative, want in ((None, "0"), (SimpleNamespace(num_speculative_tokens=3), "1"),
                              (SimpleNamespace(num_speculative_tokens=0), "0")):
        monkeypatch.delenv(ENV_BOUNDARY_COMMITS, raising=False)
        config = SimpleNamespace(**{name: getattr(start.vllm_config, name)
                                    for name in ("model_config", "parallel_config", "cache_config",
                                                 "scheduler_config")}, speculative_config=speculative)
        _configure_recurrent_state(config)
        assert __import__("os").environ.get(ENV_BOUNDARY_COMMITS) == want, speculative


ENV_SPLIT = "VLLM_HPU_PREFIX_ALIGN_SPLIT"
CARD_ENGINE_BLOCK = 128  # cache_config.block_size of the engine-core process on the card (test_prefix_split_alignment)


def _speculative_next_turn(monkeypatch, commits: bool | None, keep_decode_end: bool = True) -> tuple[int, list, h.Sim]:
    monkeypatch.setenv(ENV_SPLIT, "1")
    if keep_decode_end:
        monkeypatch.setenv(ENV_KEEP_DECODE, "1")
    else:
        monkeypatch.delenv(ENV_KEEP_DECODE, raising=False)
    sim = h.Sim(h.Settings(speculative_blocks=3, eagle_drop=True, boundary_commits=commits))
    sim.scheduler.cache_config.block_size = CARD_ENGINE_BLOCK
    prompts = _turns(sim, h.agent_prompt(0, 24_000), turns=2, generate=1100)
    return sim.admissions["t2"].cached_tokens, prompts, sim


def test_keeping_the_decode_end_with_speculative_blocks_hits_one_block_before_the_last_decode_boundary(monkeypatch):
    """MTP's next turn: turn 1 decodes 1,100 tokens across two block boundaries (24,576 and 25,088). With the
    runner's boundary commit the last two decode boundaries are kept, and the EAGLE / MTP drop of the last matched
    block lands the hit on the earlier one. Without the commit nothing of the decode is kept (the same hit as with
    the flag off)."""
    cached, prompts, sim = _speculative_next_turn(monkeypatch, commits=True)
    last = (len(prompts[0]) + 1100 - 1) // BLOCK * BLOCK
    assert last == 25_088 and cached == last - BLOCK == 24_576
    assert sim.violations == []
    off, _, off_sim = _speculative_next_turn(monkeypatch, commits=False)
    flag_off, _, flag_off_sim = _speculative_next_turn(monkeypatch, commits=None, keep_decode_end=False)
    assert off == flag_off < len(prompts[0])  # the hit stays inside the first prompt: nothing of its output
    assert off_sim.violations == [] and flag_off_sim.violations == []
