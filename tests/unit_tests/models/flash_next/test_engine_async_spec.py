# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding under async scheduling (``VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2``), the whole serving loop on
CPU (``engine_harness.py``).

Under async scheduling the engine schedules step N+1 before step N's output: step N+1's draft slots arrive as -1
placeholders and its ``num_computed_tokens`` counts every draft of step N as accepted. The runner keeps its
synchronous speculative step and ``flash_next_async.SpecDraftHandoff`` corrects the count and writes step N's
drafts, which never leave the device, into step N+1's draft slots. The drafts here are oracle drafts returned as a
tensor, as the MTP drafter returns them (the tiny model's own drafts are never accepted): the reference's greedy
continuation, with a wrong token every few steps so that every accepted count happens. The tokens must be the
reference's in both modes, and the async run must have corrected counts and written drafts. Negative controls:
without the correction the tokens go wrong; without the scatter they stay right (a wrong draft is rejected) but
fewer drafts are accepted.

The program head's speculative path runs here with whole-prompt prefills and prefix caching off (as in
engine-step's speculative harness tests); n-gram is the speculative method because the harness loads no drafter
model, and the oracle replaces the proposal, so what is tested is the handoff, the method's own drafting is not.
vLLM's config refuses async scheduling with the n-gram method (its GPU runner drafts n-grams on the host), so the
engine is configured without it and switched to it right after, as the harness does for every async run; the
platform's mode 2 keeps it for MTP only (``test_flash_next_async.py``).
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import dataclasses  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.worker.flash_next_async import SpecDraftHandoff  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

K = 3
SPEC = {"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
PROMPTS = (1, 2, 3)  # 7, 33 and 100 tokens


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _install_oracle(engine, want: dict, tamper_every: int = 3):
    """``propose_draft_token_ids`` returns the reference's next K tokens per input-batch row as a tensor (a
    device tensor on a card); one row in ``tamper_every`` steps gets one wrong token at a rotating place."""
    runner = engine.runner
    calls = {"n": 0}

    def propose(*args, **kwargs):
        calls["n"] += 1
        rows = []
        for index, req_id in enumerate(runner.input_batch.req_ids):
            done = list(runner.requests[req_id].output_token_ids)
            ref = want.get(req_id, [])
            draft = list(ref[len(done):len(done) + K])
            draft += [draft[-1] if draft else 1] * (K - len(draft))
            if (calls["n"] + index) % tamper_every == 0:
                spot = (calls["n"] + index) % K
                draft[spot] = (draft[spot] + 1) % 1000 + 1
            rows.append(draft)
        return torch.tensor(rows, dtype=torch.int64)

    runner.propose_draft_token_ids = propose


@pytest.fixture(autouse=True)
def async_after_config(monkeypatch):
    """Configure without async scheduling (vLLM refuses it for n-gram), then switch the engine's config to it."""
    original = eh._engine_config

    def engine_config(ckpt_dir, options):
        config = original(ckpt_dir, dataclasses.replace(options, async_scheduling=False))
        config.scheduler_config.async_scheduling = bool(options.async_scheduling)
        return config

    monkeypatch.setattr(eh, "_engine_config", engine_config)


def _serve(tiny_checkpoint, reference, async_scheduling: bool, monkeypatch=None, **patches):
    options = eh.Options(max_num_seqs=4, async_scheduling=async_scheduling, speculative_config=SPEC,
                         prefix_caching=False, max_num_batched_tokens=4096)
    with eh.open_engine(tiny_checkpoint, options) as engine:
        for name, value in patches.items():
            monkeypatch.setattr(SpecDraftHandoff, name, value)
        want = {}
        for index in PROMPTS:
            request_id = f"spec-{index}"
            want[request_id] = reference[index]["generated_ids"]
            engine.add(request_id, reference[index]["prompt_ids"], max_tokens=len(want[request_id]))
        _install_oracle(engine, want)
        engine.run()
        handoff = engine.runner._spec_draft_handoff
        stats = {"steps": engine.steps, "async": engine.runner.scheduler_config.async_scheduling,
                 "corrections": handoff.corrections if handoff is not None else None,
                 "scattered": handoff.scattered if handoff is not None else None,
                 "runner_async": engine.runner.use_async_scheduling}
        return dict(engine.tokens), want, stats


def test_async_speculation_gives_the_reference_tokens(tiny_checkpoint, reference):
    sync_tokens, want, sync_stats = _serve(tiny_checkpoint, reference, False)
    tokens, _, stats = _serve(tiny_checkpoint, reference, True)
    assert sync_tokens == want, "the synchronous speculative engine itself left the reference"
    assert tokens == want
    assert stats["async"] and not stats["runner_async"], stats
    assert stats["corrections"] > 0 and stats["scattered"] > 0, stats
    # Oracle drafts are accepted: far fewer engine steps than tokens (one token per step without speculation).
    assert sync_stats["steps"] < sum(len(v) for v in want.values()), sync_stats


def test_without_the_correction_the_tokens_go_wrong(tiny_checkpoint, reference, monkeypatch):
    tokens, want, stats = _serve(tiny_checkpoint, reference, True, monkeypatch,
                                 correct_computed=lambda self, *args, **kwargs: 0)
    assert tokens != want


def test_without_the_scatter_fewer_drafts_are_accepted(tiny_checkpoint, reference, monkeypatch):
    _, _, with_scatter = _serve(tiny_checkpoint, reference, True)
    tokens, want, without = _serve(tiny_checkpoint, reference, True, monkeypatch,
                                   scatter_drafts=lambda self, token_ids, *args, **kwargs: token_ids)
    assert tokens == want  # a draft slot left at -1 (token 0) is rejected: still the reference
    assert without["steps"] > with_scatter["steps"], (with_scatter, without)
