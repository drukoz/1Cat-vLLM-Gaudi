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
        sim.run_until_finished([rid], 400)
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
