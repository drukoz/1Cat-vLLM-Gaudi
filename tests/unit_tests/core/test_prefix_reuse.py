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
