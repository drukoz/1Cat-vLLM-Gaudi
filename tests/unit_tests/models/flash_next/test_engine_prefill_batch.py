# SPDX-License-Identifier: Apache-2.0
"""Batched prefill across requests (``VLLM_HPU_PREFILL_BATCH``) through the real engine on CPU.

``prefill_batch_engine.py`` drives the in-process engine on the tiny checkpoint (the HPU platform, worker and model
runner, the plugin's ops) with arrivals at chosen steps. One plan, run with the flag off and on, without and with MTP
(k = 2 with build 5's MTP flags: the draft loop, the chunk KV fill, async mode 2, the prefix align flags):

* four 300-token prompts and two 1,200-token prompts at once (equal short chunks merge; the long ones are chunked by
  the step budget and the per-request cap), two 200-token prompts that arrive while the first ones decode, two
  prompts with one shared 600-token prefix, and the next turn of four of them (a prefix hit: chunks with context);
* every stream and every logprob is the flag-off one: the merged forwards compute what the one-row forwards did;
* the flag-on run has merged prompt forwards of each kind (no context, with context, completing their prompt or not)
  and the flag-off run has none;
* the state bookkeeping: every prompt chunk loads and stores the same recurrent state slots, per cache group, merged
  or not.

Plus the runtime switch: a server started at 8 whose switch file says 0 merges nothing, and one that says 2 merges at
most two rows. Each engine run is a child process of about 30 s.
"""

from __future__ import annotations

import random

import pytest

from tests.unit_tests.models.flash_next import prefill_batch_engine

ROWS = "8"
MTP_ENV = {"VLLM_HPU_FLASH_NEXT_ASYNC_SCHED": "2", "VLLM_HPU_MTP_DRAFT_LOOP": "1", "VLLM_HPU_MTP_CHUNK_KV": "1",
           "VLLM_HPU_PREFIX_ALIGN_SPLIT": "1", "VLLM_HPU_PREFIX_KEEP_PROMPT_END": "1"}


def mixed_plan() -> dict:
    rng = random.Random(11)

    def ids(n):
        return [rng.randrange(1000, 200000) for _ in range(n)]

    shared = ids(600)
    reqs = []
    reqs += [{"id": f"a{i}", "prompt_ids": ids(300), "max_tokens": 10, "arrive_step": 0} for i in range(4)]
    reqs += [{"id": f"L{i}", "prompt_ids": ids(1200), "max_tokens": 10, "arrive_step": 0} for i in range(2)]
    reqs += [{"id": f"c{i}", "prompt_ids": ids(200), "max_tokens": 10, "arrive_step": 4} for i in range(2)]
    reqs += [{"id": f"s{i}", "prompt_ids": shared + ids(50), "max_tokens": 10, "arrive_step": 6} for i in range(2)]
    reqs += [{"id": f"a{i}t2", "turn_of": f"a{i}", "extra_ids": ids(60), "max_tokens": 10} for i in range(4)]
    reqs += [{"id": f"L{i}t2", "turn_of": f"L{i}", "extra_ids": ids(60), "max_tokens": 10} for i in range(2)]
    return {"max_num_seqs": 8, "max_num_batched_tokens": 1024, "long_prefill_token_threshold": 512,
            "max_model_len": 4096, "k": 0, "logprobs": 2, "requests": reqs}


@pytest.fixture(scope="module")
def runs(tiny_checkpoint, tmp_path_factory):
    out = tmp_path_factory.mktemp("engine_prefill_batch")
    plan = mixed_plan()
    result = {}
    for name, k, env in (("off", 0, {}), ("on", 0, {"VLLM_HPU_PREFILL_BATCH": ROWS}),
                         ("off_mtp", 2, MTP_ENV), ("on_mtp", 2, {**MTP_ENV, "VLLM_HPU_PREFILL_BATCH": ROWS})):
        result[name] = prefill_batch_engine.run(tiny_checkpoint, {**plan, "k": k}, out / f"{name}.json", env=env)
    for name, cap in (("switch0", "0"), ("switch2", "2")):
        switch = out / f"{name}.switches"
        switch.write_text(f"VLLM_HPU_PREFILL_BATCH={cap}\n", encoding="utf-8")
        result[name] = prefill_batch_engine.run(
            tiny_checkpoint, plan, out / f"{name}.json",
            env={"VLLM_HPU_PREFILL_BATCH": ROWS, "VLLM_HPU_RUNTIME_SWITCH_FILE": str(switch)})
    return result


def _streams(record: dict) -> dict:
    return {s["id"]: s for s in record["streams"]}


def _merged(record: dict) -> list[dict]:
    return [f for f in record["prompt_forwards"] if len(f["req_ids"]) > 1]


@pytest.mark.parametrize("suffix", ["", "_mtp"], ids=["plain", "mtp2"])
def test_merged_prompt_forwards_give_the_same_tokens_and_logprobs(runs, suffix):
    off, on = _streams(runs["off" + suffix]), _streams(runs["on" + suffix])
    assert off.keys() == on.keys() and len(off) == 16
    assert not runs["on" + suffix]["unfinished"]
    for rid, stream in off.items():
        assert on[rid]["generated_ids"] == stream["generated_ids"], rid
        assert on[rid]["num_cached_tokens"] == stream["num_cached_tokens"], rid
        for a, b in zip(stream["top_logprobs"], on[rid]["top_logprobs"]):
            assert a.keys() == b.keys(), rid
            assert max(abs(a[t] - b[t]) for t in a) <= 1e-3, rid


@pytest.mark.parametrize("suffix", ["", "_mtp"], ids=["plain", "mtp2"])
def test_the_flag_merges_prompt_forwards_and_off_never_does(runs, suffix):
    on, off = runs["on" + suffix], runs["off" + suffix]
    assert on["prefill_batch_rows"] == int(ROWS) and off["prefill_batch_rows"] == 0
    assert not _merged(off)
    merged = _merged(on)
    assert merged
    assert len(on["prompt_forwards"]) < len(off["prompt_forwards"])
    for forward in merged:
        # one logits class per forward, and no row reads a block another row writes (the context of a merged row
        # is a prefix computed in an earlier step)
        assert len(set(bool(n) for n in forward["logits"])) == 1


def test_the_kinds_of_merged_forwards(runs):
    merged = _merged(runs["on"])
    assert any(all(c == 0 for c in f["context_lens"]) for f in merged)  # fresh prompts
    assert any(all(c > 0 for c in f["context_lens"]) for f in merged)  # chunks with context (prefix hits, tails)
    assert any(all(n > 0 for n in f["logits"]) for f in merged)  # chunks that complete their prompt
    arrivals = {s["id"]: s["added_at"] for s in runs["on"]["streams"]}
    # the two prompts that arrive while the first ones decode share a forward
    assert any({r.split("-")[0] for r in f["req_ids"]} == {"c0", "c1"} for f in merged), arrivals
    # under MTP a merged chunk pair that completes no prompt goes through the draft's chunk KV fill
    assert any(all(n == 0 for n in f["logits"]) for f in _merged(runs["on_mtp"]))


@pytest.mark.parametrize("suffix", ["", "_mtp"], ids=["plain", "mtp2"])
def test_every_chunk_loads_and_stores_the_same_state_slots_merged_or_not(runs, suffix):
    def slots(record):
        out = {}
        for forward in record["prompt_forwards"]:
            for i, rid in enumerate(forward["req_ids"]):
                key = (rid.split("-")[0], forward["context_lens"][i], forward["query_lens"][i])
                out[key] = (forward["load_slots"][i], forward["store_slots"][i])
        return out

    off, on = slots(runs["off" + suffix]), slots(runs["on" + suffix])
    assert off == on
    assert all(load is not None and store is not None for load, store in on.values())


def test_the_runtime_switch_caps_the_rows(runs):
    assert not _merged(runs["switch0"])
    two = _merged(runs["switch2"])
    assert two and all(len(f["req_ids"]) == 2 for f in two)
    assert _streams(runs["switch2"]) == _streams(runs["switch2"]) and {
        rid: s["generated_ids"] for rid, s in _streams(runs["switch2"]).items()
    } == {rid: s["generated_ids"] for rid, s in _streams(runs["off"]).items()}
