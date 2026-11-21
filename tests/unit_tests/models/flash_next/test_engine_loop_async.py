# SPDX-License-Identifier: Apache-2.0
"""Async scheduling on the tiny Flash-Next model: the whole serving loop on CPU against the Hugging Face reference.

``engine_harness.py`` runs vLLM's ``AsyncScheduler`` (the plugin's ``HPUAsyncScheduler``) and
``EngineCore.step_with_batch_queue`` over the real ``HPUModelRunner``: the engine schedules step N+1 before step
N's tokens are on the host, the runner takes a decode row's input token from the previous step's samples on the
device, and the sampled tokens reach the engine only when it asks for that step's result, after the next step was
submitted. The runner's async path for Flash-Next (``VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=1`` on a server,
``v1/worker/flash_next_async.py``) must give the reference's greedy tokens exactly as the synchronous engine does
(``test_engine_loop.py`` on the engine-step branch): alone, batched with requests finishing mid-batch, with
requests joining while others decode, and after a prefix cache hit. The negative control shows the tests see the
defect the carry fixes: without it every prompt goes wrong from its third token.

Settings as the rental servers run: prefix caching, the recurrent state pool, sparse QSA, 128-token kernel blocks,
and ``VLLM_HPU_PLE_HOST_NGRAM_IDS=1`` (under async scheduling it falls back to the in-graph ids by design).
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402

from vllm_gaudi.v1.worker.flash_next_async import AsyncOutputLedger, PleContextCarry  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

RENTAL_ENV = {"VLLM_HPU_PLE_HOST_NGRAM_IDS": "1"}


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


@pytest.fixture(scope="module")
def engine(tiny_checkpoint):
    """One async server for the module; requests of different tests never overlap."""
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=8, async_scheduling=True, env=RENTAL_ENV)) as opened:
        yield opened


def test_start_is_the_async_engine(engine):
    runner = engine.runner
    assert runner.use_async_scheduling and runner.use_prefix_caching
    assert type(engine.scheduler).__name__ == "HPUAsyncScheduler"
    assert engine.core.batch_queue is not None and engine.core.batch_queue_size == 2
    assert engine.core.step_fn == engine.core.step_with_batch_queue


def test_every_reference_prompt_alone(engine, reference):
    """1 to 2600 prompt tokens: single-chunk and chunked prefill, dense-equivalent and sparse selection."""
    for index, prompt in enumerate(reference):
        request_id = f"solo-{index}"
        engine.add(request_id, prompt["prompt_ids"], max_tokens=len(prompt["generated_ids"]))
        engine.run()
        assert engine.tokens[request_id] == prompt["generated_ids"], f"prompt of {len(prompt['prompt_ids'])} tokens"
    runner = engine.runner
    assert isinstance(runner._ple_carry, PleContextCarry) and isinstance(runner._async_ledger, AsyncOutputLedger)


def test_requests_finishing_mid_batch(engine, reference):
    """All prompts in one batch, each stopping one token earlier than the previous: the batch shrinks every step
    (rows are condensed and reordered under the carried contexts)."""
    for index, prompt in enumerate(reference):
        engine.add(f"batch-{index}", prompt["prompt_ids"], max_tokens=len(prompt["generated_ids"]) - index)
    engine.run()
    for index, prompt in enumerate(reference):
        want = prompt["generated_ids"][:len(prompt["generated_ids"]) - index]
        assert engine.tokens[f"batch-{index}"] == want, f"prompt of {len(prompt['prompt_ids'])} tokens"


def test_requests_joining_while_others_decode(engine, reference):
    """A request arrives every second engine step: prefills enter a decoding batch, new rows take the host's
    context while the others keep the carried one."""
    order = [3, 0, 4, 1, 2, 5]
    pending = list(order)
    added = []
    for _ in range(10000):
        if pending and engine.steps % 2 == 0:
            index = pending.pop(0)
            engine.add(f"join-{index}", reference[index]["prompt_ids"],
                       max_tokens=len(reference[index]["generated_ids"]))
            added.append(index)
        if not pending and not engine.has_work():
            break
        engine.step()
    assert added == order
    for index in order:
        assert engine.tokens[f"join-{index}"] == reference[index]["generated_ids"], f"prompt {index}"


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


def test_output_token_ids_hold_the_tokens_and_one_placeholder(engine, reference):
    """At every sampler pass of a decode step, each request's output_token_ids is its tokens so far (all real) and
    one placeholder for the step in flight: the length synchronous scheduling has (min_tokens reads it)."""
    runner = engine.runner
    seen = []
    original = runner._prepare_sampling

    def spy(batch_changed, request_ids=None, *args, **kwargs):
        for request_id in request_ids or []:
            state = runner.requests.get(request_id)
            if state is not None and request_id.startswith("ledger-"):
                seen.append((request_id, list(state.output_token_ids)))
        return original(batch_changed, request_ids, *args, **kwargs)

    runner._prepare_sampling = spy
    try:
        for index in (1, 2):
            engine.add(f"ledger-{index}", reference[index]["prompt_ids"],
                       max_tokens=len(reference[index]["generated_ids"]))
        engine.run()
    finally:
        del runner._prepare_sampling
    assert len(seen) > 10
    for request_id, tokens in seen:
        want = reference[int(request_id.split("-")[1])]["generated_ids"]
        if not tokens:
            continue  # the prompt's own sampler pass
        assert tokens[-1] == -1 and -1 not in tokens[:-1], (request_id, tokens)
        assert tokens[:-1] == want[:len(tokens) - 1], (request_id, tokens)


def test_without_the_carry_the_tokens_go_wrong(engine, reference, monkeypatch):
    """Negative control: the host-built context alone (the defect the platform's old switch-off avoided)."""
    monkeypatch.setattr(PleContextCarry, "decode_context", lambda self, context, *args, **kwargs: context)
    prompt = reference[2]
    engine.add("no-carry", prompt["prompt_ids"], max_tokens=len(prompt["generated_ids"]))
    engine.run()
    got = engine.tokens["no-carry"]
    assert got[:2] == prompt["generated_ids"][:2] and got != prompt["generated_ids"]
