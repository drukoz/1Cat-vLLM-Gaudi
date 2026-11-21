# SPDX-License-Identifier: Apache-2.0
"""The whole serving loop on CPU against the Hugging Face reference (``engine_harness.py``).

Every other Flash-Next CPU test calls the model with metadata built by the test. Here the request goes through
vLLM's scheduler, ``EngineCore.step``, the real ``HPUModelRunner`` (state update, prefill and decode batch
builders, the bucketing, the recurrent state pool, the sparse QSA side caches, the sampler) and comes back as
tokens, which must be the reference's greedy tokens. The settings are the rental's: prefix caching on, the
recurrent state pool on, sparse QSA, 128-token kernel blocks under larger scheduler blocks.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


@pytest.fixture(scope="module")
def engine(tiny_checkpoint):
    """One synchronous server for the module; requests of different tests never overlap."""
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=8)) as opened:
        yield opened


def test_start_matches_the_rental_layout(engine):
    runner = engine.runner
    assert runner.attn_block_size == 128
    assert runner._recurrent_pool_slots > 0 and runner._MAMBA_PAD_BLOCK_ID == runner._recurrent_pool_slots + 1
    assert runner.use_prefix_caching and not runner.use_async_scheduling
    scheduler_block = engine.scheduler.block_size
    assert scheduler_block % 128 == 0 and scheduler_block > 128, "a scheduler block must span several kernel blocks"
    assert type(engine.scheduler).__name__ == "Scheduler"


def test_every_reference_prompt_alone(engine, reference):
    """1 to 2600 prompt tokens: single-chunk and chunked prefill, dense-equivalent and sparse selection."""
    for index, prompt in enumerate(reference):
        request_id = f"solo-{index}"
        engine.add(request_id, prompt["prompt_ids"], max_tokens=len(prompt["generated_ids"]))
        engine.run()
        assert engine.tokens[request_id] == prompt["generated_ids"], f"prompt of {len(prompt['prompt_ids'])} tokens"


def test_requests_finishing_mid_batch(engine, reference):
    """All prompts in one batch, each stopping one token earlier than the previous: the batch shrinks every step."""
    for index, prompt in enumerate(reference):
        engine.add(f"batch-{index}", prompt["prompt_ids"], max_tokens=len(prompt["generated_ids"]) - index)
    engine.run()
    for index, prompt in enumerate(reference):
        want = prompt["generated_ids"][:len(prompt["generated_ids"]) - index]
        assert engine.tokens[f"batch-{index}"] == want, f"prompt of {len(prompt['prompt_ids'])} tokens"


def test_prefix_cache_hit_gives_the_same_tokens(engine, reference):
    """The second request of a prompt reuses the first one's blocks (attention K/V and recurrent state)."""
    prompt = next(p for p in reference if len(p["prompt_ids"]) == 700)
    engine.add("cache-first", prompt["prompt_ids"], max_tokens=8)
    engine.run()
    cached_tokens = {}
    manager = engine.scheduler.kv_cache_manager
    original = manager.get_computed_blocks

    def spy(request):
        result = original(request)
        cached_tokens[request.request_id] = result[1]
        return result

    manager.get_computed_blocks = spy
    try:
        engine.add("cache-second", prompt["prompt_ids"], max_tokens=8)
        engine.run()
    finally:
        manager.get_computed_blocks = original
    assert cached_tokens["cache-second"] >= engine.scheduler.block_size, "the second request hit no cached block"
    assert engine.tokens["cache-first"] == prompt["generated_ids"][:8]
    assert engine.tokens["cache-second"] == prompt["generated_ids"][:8]
