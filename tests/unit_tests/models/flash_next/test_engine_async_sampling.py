# SPDX-License-Identifier: Apache-2.0
"""Async scheduling against synchronous scheduling for sampler paths that are not plain greedy (tiny Flash-Next
model, the whole serving loop on CPU, ``engine_harness.py``).

* Penalties read every output token's value: under async scheduling the step in flight is a placeholder until
  its host copy is complete (``flash_next_async.AsyncOutputLedger``); the runner settles it before such a sampler
  pass, so the tokens equal the synchronous engine's. Before the ledger the HPU runner kept no output tokens at all
  under async scheduling (the negative control). The tiny model's greedy tokens never repeat, so the penalties
  here encourage repeats (negative presence and frequency penalties): then every token depends on the values.
* A seeded sampled request (the checkpoint's default sampling: temperature 1.0, top_k 20, top_p 0.95) draws the
  same random numbers in the same order, so it gives the same tokens in both modes, as long as its prompt is one
  chunk. A prompt of several chunks differs: under async scheduling the runner samples (and discards) a token for
  every chunk that does not finish the prompt (vLLM's GPU runner rewinds the generator there; the HPU runner does
  not), which moves the request's random stream by one draw per such chunk. The draws stay valid samples.

Each test runs one synchronous and then one async engine (one may be open at a time).
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402

from vllm_gaudi.v1.worker.flash_next_async import AsyncOutputLedger  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

RENTAL_ENV = {"VLLM_HPU_PLE_HOST_NGRAM_IDS": "1"}
REPEATS = dict(presence_penalty=-2.0, frequency_penalty=-1.0)
SEEDED = dict(temperature=1.0, top_k=20, top_p=0.95)


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _serve(tiny_checkpoint, async_scheduling: bool, requests, **options):
    with eh.open_engine(
            tiny_checkpoint,
            eh.Options(max_num_seqs=8, async_scheduling=async_scheduling, env=RENTAL_ENV, **options)) as engine:
        for request_id, prompt_ids, max_tokens, sampling in requests:
            engine.add(request_id, prompt_ids, max_tokens=max_tokens, **sampling)
        engine.run()
        return {request_id: list(engine.tokens[request_id]) for request_id, *_ in requests}


def _penalty_requests(reference):
    return [(f"pen-{index}", reference[index]["prompt_ids"], 20, REPEATS) for index in (0, 2, 3)]


def test_penalties_give_the_synchronous_tokens(tiny_checkpoint, reference):
    """Three requests decoding together with penalties on their output tokens."""
    requests = _penalty_requests(reference)
    sync = _serve(tiny_checkpoint, False, requests)
    asyn = _serve(tiny_checkpoint, True, requests)
    assert asyn == sync
    plain = {f"pen-{index}": reference[index]["generated_ids"] for index in (0, 2, 3)}
    assert all(sync[rid][:len(plain[rid])] != plain[rid] for rid in sync), "the penalties changed no token"


def test_penalties_without_the_ledger_go_wrong(tiny_checkpoint, reference, monkeypatch):
    """Negative control: no placeholders and no values (the HPU runner's async path before the ledger)."""
    requests = _penalty_requests(reference)
    sync = _serve(tiny_checkpoint, False, requests)
    monkeypatch.setattr(AsyncOutputLedger, "record", lambda self, *args, **kwargs: None)
    asyn = _serve(tiny_checkpoint, True, requests)
    assert asyn != sync


def test_seeded_sampling_gives_the_synchronous_tokens(tiny_checkpoint, reference):
    """Single-chunk prompts (7, 33 and 100 tokens), three requests sampling together."""
    requests = [(f"seed-{index}", reference[index]["prompt_ids"], 16, dict(SEEDED, seed=1000 + index))
                for index in (1, 2, 3)]
    sync = _serve(tiny_checkpoint, False, requests)
    asyn = _serve(tiny_checkpoint, True, requests)
    assert asyn == sync


def test_seeded_sampling_of_a_one_chunk_long_prompt(tiny_checkpoint, reference):
    """The 700-token prompt is one chunk without prefix caching (no block split): the same tokens again, so the
    difference of a chunked prompt is the discarded draw of each unfinished chunk, nothing else."""
    prompt = next(p for p in reference if len(p["prompt_ids"]) == 700)
    requests = [("seed-700", prompt["prompt_ids"], 12, dict(SEEDED, seed=4242))]
    sync = _serve(tiny_checkpoint, False, requests, prefix_caching=False)
    asyn = _serve(tiny_checkpoint, True, requests, prefix_caching=False)
    assert asyn == sync
