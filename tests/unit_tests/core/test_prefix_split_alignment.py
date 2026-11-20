# SPDX-License-Identifier: Apache-2.0
"""Prefill chunk ends and the recurrent block: what the card's scheduler did, and VLLM_HPU_PREFIX_ALIGN_SPLIT.

The engine's ``Scheduler._mamba_block_aligned_split`` aligns prompt chunk ends to ``self.cache_config.block_size``
in the engine-core process, while the runner keeps recurrent states in slots of the recurrent group's block (896
tokens at TP4, 512 at TP8: slot ``p`` holds the state after exactly ``(p + 1) * block`` tokens). On the card (prefill
job 00014, rank 0's per-forward events) prompts were cut into 2,048-token chunks with ends at multiples of 2,048, not
of 896, which the split can only produce with an engine-core block size of 128 or less (INFERRED: the plugin's
``check_and_update_config`` default is 128; the wrapper's one log line shows the value on a server). These tests
play that state, block 128, on the engine's scheduler (``flash_next_sched_harness``); it reproduces both card
misses exactly:

* without the flag, the state at a prompt's last block boundary exists only when a chunk happens to end there, so
  the next turn misses the prefix cache completely (00014: agents 0 to 3; engine-step 00110: a 120,832-token seed and
  then that prefix + 64 tokens, cached 0), and at TP4 a slot can hold the state of another position, which a later
  hit reads (``Sim.violations``: a silently wrong recurrent state);
* with ``VLLM_HPU_PREFIX_ALIGN_SPLIT=1`` chunk ends follow the recurrent block and every case hits, without stale
  reads; with the two block sizes equal the flag changes nothing.
"""

from tests.unit_tests.core import flash_next_sched_harness as h

ENV_SPLIT = "VLLM_HPU_PREFIX_ALIGN_SPLIT"
CARD_ENGINE_BLOCK = 128  # cache_config.block_size of the engine-core process on the card


def _sim(monkeypatch, tp: int, flag: bool, engine_block: int | None = CARD_ENGINE_BLOCK) -> h.Sim:
    if flag:
        monkeypatch.setenv(ENV_SPLIT, "1")
    else:
        monkeypatch.delenv(ENV_SPLIT, raising=False)
    sim = h.Sim(h.Settings(tp=tp, max_num_seqs=32))
    if engine_block is not None:
        sim.scheduler.cache_config.block_size = engine_block  # what the card's engine core holds
    return sim


def _chunk_ends(sim: h.Sim, request_id: str) -> list[int]:
    ends, position = [], 0
    for step in sim.steps:
        if request_id in step.prefill:
            position += step.prefill[request_id]
            ends.append(position)
    return ends


def _seed_then_check(sim: h.Sim, seed_tokens: int, extra: int = 64) -> int:
    seed = h.own_tokens(7, seed_tokens, salt=9)
    sim.add("seed", seed, max_tokens=2)
    sim.run_until_finished(["seed"], 5000)
    sim.add("check", seed + h.own_tokens(8, extra, salt=9), max_tokens=8)
    sim.run_until_finished(["check"], 5000)
    return sim.admissions["check"].cached_tokens


def _agents_turn(sim: h.Sim, agents: int = 32, context: int = 8192, new: int = 892, out: int = 4) -> dict[str, int]:
    prompts = {a: h.own_tokens(a, context) for a in range(agents)}
    setup = [f"a{a}-setup" for a in prompts]
    for a, rid in zip(prompts, setup):
        sim.add(rid, prompts[a], max_tokens=out)
    sim.run_until_finished(setup, 5000)
    turn = [f"a{a}-t1" for a in prompts]
    for a, rid in zip(prompts, turn):
        sim.add(rid, prompts[a] + sim.outputs[f"a{a}-setup"] + h.own_tokens(a, new, salt=1), max_tokens=out)
    sim.run_until_finished(turn, 5000)
    return {rid: sim.admissions[rid].cached_tokens for rid in turn}


# ----- the card's state, without the flag: the defect -----


def test_card_state_cuts_prompts_at_2048_multiples_and_the_next_turn_misses(monkeypatch):
    """00110 at TP4: a 120,832-token seed (59 x 2,048) keeps no state at 120,064 = 134 x 896, so its prefix misses."""
    sim = _sim(monkeypatch, tp=4, flag=False)
    assert _seed_then_check(sim, 120_832) == 0
    ends = _chunk_ends(sim, "seed")
    assert all(end % 2048 == 0 for end in ends), ends[:6]
    assert 120_064 not in ends


def test_card_state_most_agents_miss_and_some_hits_read_stale_states_at_tp4(monkeypatch):
    """00014: 32 conversations of 8,192 tokens, then 32 turns of +896: most miss, and hits can read a stale slot."""
    sim = _sim(monkeypatch, tp=4, flag=False)
    hits = _agents_turn(sim)
    misses = [rid for rid, cached in hits.items() if cached == 0]
    assert len(misses) >= 16, hits
    assert {"a0-t1", "a1-t1", "a2-t1", "a3-t1"} <= set(misses)  # the first four admitted, as on the card
    assert any(v.kind == "stale_state" for v in sim.violations), sim.violations[:3]


# ----- VLLM_HPU_PREFIX_ALIGN_SPLIT=1 -----


def test_flag_aligns_chunk_ends_to_the_recurrent_block_and_the_seed_hits(monkeypatch):
    sim = _sim(monkeypatch, tp=4, flag=True)
    assert _seed_then_check(sim, 120_832) == 120_064
    ends = _chunk_ends(sim, "seed")
    assert all(end % 896 == 0 for end in ends[:-1]), ends[:6]
    assert ends[-1] == 120_832 and 120_064 in ends
    assert sim.violations == []


def test_flag_every_agent_hits_without_stale_reads_at_tp4(monkeypatch):
    sim = _sim(monkeypatch, tp=4, flag=True)
    hits = _agents_turn(sim)
    assert set(hits.values()) == {8064}, hits
    assert sim.violations == []


def test_flag_at_tp8_hits_where_the_card_state_misses(monkeypatch):
    """TP8 (block 512 divides 2,048): the card's state keeps the replay boundary's state only when a 2,048-multiple
    chunk end or the 128-aligned last stop lands on it; the flag stops every prompt at its last 512 boundary."""
    lengths = (33_000, 70_500, 120_000)  # (N - 1) // 512 * 512: 32,768 (a 2,048 multiple), 70,144, 119,808
    plain = _sim(monkeypatch, tp=8, flag=False)
    plain_cached = []
    for i, n in enumerate(lengths):
        prompt = h.own_tokens(20 + i, n, salt=3)
        plain.add(f"p{i}", prompt, max_tokens=2)
        plain.run_until_finished([f"p{i}"], 5000)
        plain.add(f"q{i}", prompt + h.own_tokens(40 + i, 500, salt=3), max_tokens=2)
        plain.run_until_finished([f"q{i}"], 5000)
        plain_cached.append(plain.admissions[f"q{i}"].cached_tokens)
    fixed = _sim(monkeypatch, tp=8, flag=True)
    fixed_cached = []
    for i, n in enumerate(lengths):
        prompt = h.own_tokens(20 + i, n, salt=3)
        fixed.add(f"p{i}", prompt, max_tokens=2)
        fixed.run_until_finished([f"p{i}"], 5000)
        fixed.add(f"q{i}", prompt + h.own_tokens(40 + i, 500, salt=3), max_tokens=2)
        fixed.run_until_finished([f"q{i}"], 5000)
        fixed_cached.append(fixed.admissions[f"q{i}"].cached_tokens)
    assert fixed_cached == [(n - 1) // 512 * 512 for n in lengths], fixed_cached
    assert plain_cached[0] == 32_768  # a 2,048-multiple boundary is a chunk end anyway
    assert 0 in plain_cached[1:], plain_cached  # the others need luck
    assert plain.violations == [] and fixed.violations == []


def test_flag_changes_nothing_when_the_engine_block_is_the_recurrent_block(monkeypatch):
    def run(flag: bool) -> list[tuple[int, int, int]]:
        sim = _sim(monkeypatch, tp=4, flag=flag, engine_block=None)
        _agents_turn(sim, agents=8)
        assert sim.violations == []
        return [(a.prompt_tokens, a.cached_tokens, a.first_chunk) for a in sim.admissions.values()]

    assert run(False) == run(True)


def test_split_block_is_logged_once_per_scheduler(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO)
    sim = _sim(monkeypatch, tp=4, flag=False)
    _seed_then_check(sim, 4_000)
    lines = [r.getMessage() for r in caplog.records if "VLLM_HPU_PREFIX_ALIGN_SPLIT" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "128" in lines[0] and "896" in lines[0]


def test_block_aligned_prompts_need_the_prompt_end_keep_as_well(monkeypatch):
    """A prompt of a whole number of blocks keeps its replay boundary one block before its end: with the aligned
    split alone its next turn still misses (or hits a block short); VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 keeps the state
    after exactly N tokens, so the next turn starts behind the whole prompt."""
    for tp, n in ((4, 28_672), (8, 32_768), (8, 34_304)):
        cached = {}
        for keep in ("0", "1"):
            monkeypatch.setenv("VLLM_HPU_PREFIX_KEEP_PROMPT_END", keep)
            sim = _sim(monkeypatch, tp=tp, flag=True)
            prompt = h.own_tokens(3, n, salt=11)
            sim.add("cold", prompt, max_tokens=8)
            sim.run_until_finished(["cold"], 5000)
            sim.add("turn", prompt + sim.outputs["cold"] + h.own_tokens(4, 500, salt=11), max_tokens=8)
            sim.run_until_finished(["turn"], 5000)
            assert sim.violations == []
            cached[keep] = sim.admissions["turn"].cached_tokens
        assert cached["0"] < n and cached["1"] == n, (tp, n, cached)
