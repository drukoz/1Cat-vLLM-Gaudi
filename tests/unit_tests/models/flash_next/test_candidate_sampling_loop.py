# SPDX-License-Identifier: Apache-2.0
"""Sampling over candidates in the whole serving loop (``VLLM_HPU_CANDIDATE_SAMPLING=1``, ``engine_harness``).

* Requests with the checkpoint's default sampling (temperature 1.0, top_k 20, top_p 0.95) decode in the
  LM-head region: no decode step takes vLLM's sampler, every drawn token is one vLLM's sampler could draw (it
  has a probability above zero in vLLM's processed distribution of that step's logits), and a greedy request
  in the same batch still gives the HF reference's tokens.
* A seeded request repeats itself.
* A request outside the conditions (top_k unset, min_p) and a server without the flag keep vLLM's sampler.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch  # noqa: E402

from vllm_gaudi.v1.sample.candidate_sampling import PARAM_COLUMNS  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FLAG = {"VLLM_HPU_CANDIDATE_SAMPLING": "1"}
SAMPLED = dict(temperature=1.0, top_k=20, top_p=0.95)


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


class _Steps:
    """Which logits region each decode step took, and what the sampled region saw and drew."""

    def __init__(self, engine) -> None:
        self.sampled = 0
        self.general = 0
        self.draws: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        runner = engine.runner
        model = runner.model
        region, inner = model.select_compute_logits_and_sample, model.model
        compute_logits = inner.compute_logits
        generic = runner._execute_model_generic
        captured = {}

        def counting_generic(token_ids, position_ids, attn_metadata, *args, **kwargs):
            if (not attn_metadata.is_prompt and kwargs.get("candidate_sampling") is None
                    and not kwargs.get("fuse_greedy_sampling", False)):
                self.general += 1
            return generic(token_ids, position_ids, attn_metadata, *args, **kwargs)

        def recording_compute_logits(hidden_states):
            logits = compute_logits(hidden_states)
            captured["logits"] = logits.detach().clone()
            return logits

        def sampled_region(hidden_states, logits_indices, params, top_k_index, mixed, **kwargs):
            self.sampled += 1
            inner.compute_logits = recording_compute_logits
            try:
                selected, tokens = region(hidden_states, logits_indices, params, top_k_index, mixed, **kwargs)
            finally:
                inner.compute_logits = compute_logits
            self.draws.append((captured.pop("logits"), tokens.detach().clone(), params.detach().clone(),
                               top_k_index.detach().clone()))
            return selected, tokens

        model.select_compute_logits_and_sample = sampled_region
        runner._execute_model_generic = counting_generic


def vllm_probs(logits, temperature, top_k, top_p):
    scaled = logits.to(torch.float32).div(temperature.unsqueeze(dim=1))
    return apply_top_k_top_p_pytorch(scaled, top_k, top_p, allow_cpu_sync=False).softmax(dim=-1)


def test_sampled_requests_draw_in_the_logits_region(tiny_checkpoint, reference):
    prompts = [p for p in reference if len(p["prompt_ids"]) <= 700][:4]
    greedy = prompts[0]
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=4, env=FLAG)) as engine:
        steps = _Steps(engine)
        engine.add("greedy", greedy["prompt_ids"], max_tokens=len(greedy["generated_ids"]))
        for index, prompt in enumerate(prompts[1:]):
            engine.add(f"s{index}", prompt["prompt_ids"], max_tokens=12, **SAMPLED)
        engine.run()
        assert engine.tokens["greedy"] == greedy["generated_ids"]  # a greedy row of a mixed batch: the arg max
        for index in range(len(prompts) - 1):
            assert len(engine.tokens[f"s{index}"]) == 12
    assert steps.sampled > 0 and steps.general == 0, (steps.sampled, steps.general)
    checked = 0
    for logits, tokens, params, top_k_index in steps.draws:
        assert tokens.dtype == torch.int64 and tokens.shape == (logits.shape[0], 1)
        greedy_rows = params[:, 2] != 0
        want = vllm_probs(logits, params[:, 0], (top_k_index.view(-1) + 1).to(torch.int32), params[:, 1])
        drawn = want.gather(1, tokens)
        for row in range(logits.shape[0]):
            if greedy_rows[row]:
                assert int(tokens[row]) == int(logits[row].float().argmax())
            else:
                assert float(drawn[row]) > 0.0, "a token vLLM's sampler could not draw"
                checked += 1
        assert params.shape[1] > PARAM_COLUMNS and bool((params[:, PARAM_COLUMNS:] >= 0).all())
    assert checked > 0


def test_a_seeded_request_repeats_itself(tiny_checkpoint, reference):
    prompt = next(p for p in reference if len(p["prompt_ids"]) <= 700)
    runs = []
    for _ in range(2):
        with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2, env=FLAG)) as engine:
            steps = _Steps(engine)
            engine.add("seeded", prompt["prompt_ids"], max_tokens=16, seed=20261120, **SAMPLED)
            engine.run()
            runs.append(list(engine.tokens["seeded"]))
            assert steps.sampled > 0 and steps.general == 0
    assert runs[0] == runs[1] and len(runs[0]) == 16


@pytest.mark.parametrize("sampling", [dict(temperature=1.0, top_p=0.95), dict(SAMPLED, min_p=0.05), None])
def test_what_is_outside_the_conditions_keeps_vllms_sampler(tiny_checkpoint, reference, sampling):
    """top_k unset, min_p, and a server without the flag."""
    prompt = next(p for p in reference if len(p["prompt_ids"]) <= 700)
    env = FLAG if sampling is not None else {}
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2, env=env)) as engine:
        steps = _Steps(engine)
        engine.add("r", prompt["prompt_ids"], max_tokens=8, **(sampling or SAMPLED))
        engine.run()
        assert len(engine.tokens["r"]) == 8
    assert steps.sampled == 0 and steps.general > 0
