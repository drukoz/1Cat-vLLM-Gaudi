# SPDX-License-Identifier: Apache-2.0
"""Sampled verification of speculative steps in the candidate region (``VLLM_HPU_CANDIDATE_SAMPLING=2``).

With deterministic drafts (MTP's arg max, n-gram) the draft distribution is one-hot at the draft d, and speculative
sampling equals: draw every verification row and the bonus row from the target distribution, each with its own
noise, then verify the drafts against the draws (``greedy_rejection_sample_from_tokens``). A draw y accepted when
y = d is accepted with probability p(d); a rejected row's draw follows p without d, renormalized.

* The composition on the region's own functions: over many rows, the acceptance rate is p(d) and the rejected
  draws follow p(x) / (1 - p(d)), within sampling error.
* The serving loop (engine harness, tiny model, n-gram speculation with k = 3 drafts on every step): every decode
  step with a sampled request runs in the region with 1 + k rows per request, the rejection sampler never runs, a
  greedy request in the same batch gives the HF reference's tokens, and a sampled request advances by 1 to k + 1
  tokens per step.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.sample.candidate_sampling import (CANDIDATES, PARAM_COLUMNS, candidate_parameters,  # noqa: E402
                                                     candidate_probs, sample_candidates)
from vllm_gaudi.v1.sample.hpu_greedy_verify import greedy_rejection_sample_from_tokens  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

K = 3
FLAGS = {"VLLM_HPU_CANDIDATE_SAMPLING": "2", "VLLM_HPU_SPEC_SAMPLED_VERIFY": "2"}
SAMPLED = dict(temperature=1.0, top_k=20, top_p=0.95)


def test_draw_then_compare_is_speculative_sampling_for_a_one_hot_draft():
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(1, 4096, generator=gen) * 2.0
    values, ids = logits.topk(CANDIDATES, dim=-1)
    rows = 200_000
    params, top_k_index = candidate_parameters([1.0] * rows, [20] * rows, [0.95] * rows, rows,
                                               generator=torch.Generator().manual_seed(11))
    p = candidate_probs(values, torch.tensor([1.0]), torch.tensor([[19]]), torch.tensor([0.95]))[0]
    draws = sample_candidates(values.expand(rows, -1), ids.expand(rows, -1), params, top_k_index).view(-1)
    draft_column = 1  # the second most likely token: p(d) well inside (0, 1)
    draft = int(ids[0, draft_column])
    p_d = float(p[draft_column])
    accepted = draws == draft
    rate = float(accepted.float().mean())
    sigma = (p_d * (1 - p_d) / rows) ** 0.5
    assert abs(rate - p_d) < 5 * sigma, (rate, p_d)
    # A rejected row's draw: p without d, renormalized.
    rejected = draws[~accepted]
    want = p.clone()
    want[draft_column] = 0.0
    want = want / want.sum()
    for column in range(CANDIDATES):
        if float(want[column]) < 1e-3:
            continue
        got = float((rejected == int(ids[0, column])).float().mean())
        sd = (float(want[column]) * (1 - float(want[column])) / rejected.numel()) ** 0.5
        assert abs(got - float(want[column])) < 5 * sd, (column, got, float(want[column]))
    assert not bool((rejected == draft).any())


def test_the_verification_tokens_follow_the_greedy_rejection():
    """Rows of two requests with k = 3: request 0's draws equal its first two drafts, request 1's none."""
    draws = torch.tensor([5, 6, 9, 8, 1, 2, 3, 4], dtype=torch.int32)  # 1 + k rows per request
    drafts = torch.tensor([5, 6, 7, 7, 7, 7], dtype=torch.int32)
    target = torch.tensor([0, 1, 2, 4, 5, 6])
    bonus = torch.tensor([3, 7])
    cu = torch.tensor([3, 6])
    out = greedy_rejection_sample_from_tokens(draws, drafts, target, bonus, cu)
    assert out[0].tolist()[:3] == [5, 6, 9]  # two accepted, then the first unequal draw
    assert out[1].tolist()[0] == 1  # the first draw differs from the draft: it ends the step
    # A padded position (-1) is masked out, whatever row the clamp reads.
    padded = greedy_rejection_sample_from_tokens(draws, drafts, torch.tensor([0, 1, 2, -1, -1, -1]), bonus,
                                                 torch.tensor([3, 3]))
    assert padded[0].tolist()[:3] == [5, 6, 9]


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _drafts_of_k(engine, greedy_reference: dict):
    """k drafts for every request on every step: the greedy request's next reference tokens (every third step one
    of them wrong), and for a sampled request its last token repeated (accepted only when the draws repeat it)."""
    runner = engine.runner
    calls = iter(range(10 ** 6))

    def propose(sampled_token_ids):
        step = next(calls)
        drafts = []
        for index in range(len(sampled_token_ids)):
            req_id = runner.input_batch.req_ids[index] if index < len(runner.input_batch.req_ids) else None
            if req_id is None:
                drafts.append([-1])
                continue
            output = list(runner.requests[req_id].output_token_ids)
            if req_id in greedy_reference:
                ref = greedy_reference[req_id]
                draft = list(ref[len(output):len(output) + K])
                draft += [draft[-1] if draft else 1] * (K - len(draft))
                if step % 3 == 0:
                    draft[step % K] = (draft[step % K] + 1) % 1000 + 1
            else:
                draft = [output[-1] if output else 1] * K
            drafts.append(draft)
        return drafts

    runner.propose_ngram_draft_token_ids = propose


def test_speculative_steps_sample_every_row_in_the_region(tiny_checkpoint, reference, monkeypatch):
    for name, value in FLAGS.items():
        monkeypatch.setenv(name, value)
    greedy = next(p for p in reference if len(p["prompt_ids"]) == 7)  # no near tie on any path
    sampled_prompt = next(p for p in reference if len(p["prompt_ids"]) == 33)
    speculative = {"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    # As in test_decode_metadata_sparse._serve: the program head's speculative path needs whole-prompt prefills and
    # prefix caching off in this harness.
    options = eh.Options(max_num_seqs=4, speculative_config=speculative, env=FLAGS, prefix_caching=False,
                         max_num_batched_tokens=4096)
    with eh.open_engine(tiny_checkpoint, options) as engine:
        runner, model = engine.runner, engine.runner.model
        region = model.select_compute_logits_and_sample
        calls = []

        def recording_region(hidden_states, logits_indices, params, top_k_index, mixed, **kwargs):
            selected, tokens = region(hidden_states, logits_indices, params, top_k_index, mixed, **kwargs)
            calls.append((int(logits_indices.shape[0]), params.detach().clone(), tokens.detach().clone(), mixed))
            return selected, tokens

        class NoRejectionSampler:
            """The runner still parses the verified tokens through the sampler's parse_output; a call is wrong."""

            def __init__(self, sampler):
                self._sampler = sampler

            def __call__(self, *args, **kwargs):
                raise AssertionError("a verification step with a sampled request reached the rejection sampler")

            def __getattr__(self, name):
                return getattr(self._sampler, name)

        model.select_compute_logits_and_sample = recording_region
        monkeypatch.setattr(runner, "rejection_sampler", NoRejectionSampler(runner.rejection_sampler))
        _drafts_of_k(engine, {"g": greedy["generated_ids"]})
        engine.add("g", greedy["prompt_ids"], max_tokens=len(greedy["generated_ids"]))
        engine.add("s0", sampled_prompt["prompt_ids"], max_tokens=200, **SAMPLED)
        engine.add("s1", greedy["prompt_ids"], max_tokens=200, seed=7, **SAMPLED)
        lengths = []
        for _ in range(6):
            engine.step()
            lengths.append({r: len(engine.tokens[r]) for r in ("g", "s0", "s1")})
        greedy_tokens = list(engine.tokens["g"])
    assert calls, "no step ran in the region"
    verification = [c for c in calls if c[0] % (1 + K) == 0 and c[0] >= 3 * (1 + K)]
    assert verification, [c[0] for c in calls]
    for rows, params, tokens, mixed in verification:
        assert mixed  # the greedy request shares the batch
        assert tokens.shape == (rows, 1) and tokens.dtype == torch.int32
        assert params.shape == (rows, PARAM_COLUMNS + CANDIDATES)
        # Each request's 1 + k rows carry its parameters; each row has its own noise.
        noise = params[:, PARAM_COLUMNS:]
        assert len({tuple(row.tolist()) for row in noise[:3 * (1 + K)]}) == 3 * (1 + K)
    assert greedy_tokens == greedy["generated_ids"][:len(greedy_tokens)] and len(greedy_tokens) > 3
    for before, after in zip(lengths, lengths[1:]):
        for request in ("s0", "s1"):
            if before[request] and after[request] < 200:
                assert 1 <= after[request] - before[request] <= K + 1, lengths
