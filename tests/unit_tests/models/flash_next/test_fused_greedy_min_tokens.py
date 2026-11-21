# SPDX-License-Identifier: Apache-2.0
"""min_tokens inside the fused greedy region (``VLLM_HPU_FUSED_GREEDY_LOGITS=1``, ``v1/sample/fused_greedy.py``).

* The fixed-shape mask (a gather, a where, a scatter of numbers) gives the logits vLLM's
  ``MinTokensLogitsProcessor`` gives: ``-inf`` at every stop id of the rows below ``min_tokens``, nothing else.
* In the whole serving loop (``engine_harness``) a request with ``min_tokens`` now decodes in the fused region
  (the probes and the 32-agent load send ``min_tokens = max_tokens``), its tokens are the HF reference's, and a
  stop token the model would pick before ``min_tokens`` is masked: the token after the mask is the argmax of
  the masked logits.
* The general sampler path does not enforce ``min_tokens`` on HPU at all: the runner never hands the input
  batch's changes to vLLM's logits processors, so the min-tokens processor tracks no request (the last test
  holds that down until it is fixed).
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.v1.sample.logits_processor import MinTokensLogitsProcessor  # noqa: E402

from vllm_gaudi.v1.sample.fused_greedy import STOP_ID_WIDTH, mask_stop_tokens, min_tokens_stop_ids  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FUSED = {"VLLM_HPU_FUSED_GREEDY_LOGITS": "1"}


def test_mask_is_the_min_tokens_rule():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    logits[1, 7] = logits[1, 9] = 30.0  # a tie of two stop ids at the maximum
    stops = [[3, 7], [7, 9, 11], None, [], [49], None]
    rows = min_tokens_stop_ids(stops, 6)
    assert rows is not None
    want = logits.clone()
    for row, ids in enumerate(stops):
        for token in ids or []:
            want[row, token] = float("-inf")
    got = mask_stop_tokens(logits.clone(), *rows)
    assert torch.equal(got, want)
    assert torch.equal(got.argmax(dim=-1), want.argmax(dim=-1))


def test_stop_id_rows():
    ids, active = min_tokens_stop_ids([[5, 2], None, [9]], 4)
    assert ids.dtype == torch.int64 and ids.shape == (4, STOP_ID_WIDTH)  # host-made int64: no cast in the graph
    assert ids[0].tolist() == [2, 5] + [5] * (STOP_ID_WIDTH - 2)
    assert ids[1].tolist() == [0] * STOP_ID_WIDTH and ids[3].tolist() == [0] * STOP_ID_WIDTH
    assert active.tolist() == [True, False, True, False]
    assert min_tokens_stop_ids([None, []], 2) is None
    with pytest.raises(ValueError):
        min_tokens_stop_ids([list(range(STOP_ID_WIDTH + 1))], 1)
    with pytest.raises(ValueError):
        min_tokens_stop_ids([[1], [2]], 1)


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


class _Paths:
    """Which logits region each decode step of an engine took, and the fused region's logits."""

    def __init__(self, engine) -> None:
        self.fused = 0
        self.general = 0
        self.fused_steps: list[tuple[torch.Tensor, torch.Tensor]] = []  # (logits before the mask, tokens)
        runner = engine.runner
        model = runner.model
        fused, inner = model.select_compute_logits_and_greedy, model.model
        compute_logits = inner.compute_logits
        generic = runner._execute_model_generic
        captured = {}

        def counting_generic(token_ids, position_ids, attn_metadata, *args, **kwargs):
            if not attn_metadata.is_prompt and not kwargs.get("fuse_greedy_sampling", False):
                self.general += 1
            return generic(token_ids, position_ids, attn_metadata, *args, **kwargs)

        def recording_compute_logits(hidden_states):
            logits = compute_logits(hidden_states)
            captured["logits"] = logits.detach().clone()
            return logits

        def fused_region(hidden_states, logits_indices, **kwargs):
            self.fused += 1
            inner.compute_logits = recording_compute_logits
            try:
                selected, tokens = fused(hidden_states, logits_indices, **kwargs)
            finally:
                inner.compute_logits = compute_logits
            self.fused_steps.append((captured.pop("logits"), tokens.detach().clone(), kwargs))
            return selected, tokens

        model.select_compute_logits_and_greedy = fused_region
        runner._execute_model_generic = counting_generic


def test_min_tokens_requests_decode_in_the_fused_region(tiny_checkpoint, reference):
    """The probes' and the agent load's settings (min_tokens = max_tokens, ignore_eos): reference tokens, and
    every decode step in the fused region."""
    prompts = [p for p in reference if len(p["prompt_ids"]) <= 700][:4]
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=4, env=FUSED)) as engine:
        paths = _Paths(engine)
        for index, prompt in enumerate(prompts):
            n = len(prompt["generated_ids"])
            engine.add(f"m{index}", prompt["prompt_ids"], max_tokens=n, min_tokens=n, ignore_eos=True)
        engine.run()
        for index, prompt in enumerate(prompts):
            assert engine.tokens[f"m{index}"] == prompt["generated_ids"]
    assert paths.fused > 0 and paths.general == 0, (paths.fused, paths.general)
    assert any("stop_ids" in kwargs for _, _, kwargs in paths.fused_steps), "no step carried the min_tokens mask"


def test_a_stop_token_before_min_tokens_is_masked(tiny_checkpoint, reference):
    """The model's own third token made a stop token: below min_tokens the fused region must pick the next best
    token instead (the argmax of the masked logits), as vLLM's min-tokens processor does."""
    prompt = next(p for p in reference if 30 <= len(p["prompt_ids"]) <= 700 and len(p["generated_ids"]) >= 8)
    stop = prompt["generated_ids"][2]
    assert stop not in prompt["generated_ids"][:2]
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2, env=FUSED)) as engine:
        paths = _Paths(engine)
        engine.add("s", prompt["prompt_ids"], max_tokens=6, min_tokens=6, stop_token_ids=[stop])
        engine.run()
        tokens = engine.tokens["s"]
    assert tokens[:2] == prompt["generated_ids"][:2]
    assert stop not in tokens and len(tokens) == 6
    masked_steps = 0
    for logits, sampled, kwargs in paths.fused_steps:
        want = logits.float()
        if "stop_ids" in kwargs:
            masked_steps += 1
            want = mask_stop_tokens(want, kwargs["stop_ids"], kwargs["stop_rows"])
        assert torch.equal(sampled.flatten(), want.argmax(dim=-1).to(torch.int32))
    assert masked_steps == len(paths.fused_steps) > 0
    # the mask changed the third token: unmasked, the model picks the stop token there
    third_logits = paths.fused_steps[1][0]
    assert int(third_logits[0].float().argmax()) == stop and tokens[2] != stop


def test_general_sampler_path_tracks_no_min_tokens_request(tiny_checkpoint):
    """Known defect (engine-step report): the HPU input batch never calls the logits processors' update_state,
    so vLLM's min-tokens processor masks nothing on the general sampler path. When this fails, the defect is
    fixed: drop the test and compare both paths' min_tokens tokens instead."""
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2)) as engine:
        engine.add("a", list(range(1000, 1040)), max_tokens=6, min_tokens=6, ignore_eos=True)
        for _ in range(3):
            engine.step()
        processors = [p for p in engine.runner.input_batch.logitsprocs.non_argmax_invariant
                      if isinstance(p, MinTokensLogitsProcessor)]
        assert processors and all(not p.min_toks for p in processors)
