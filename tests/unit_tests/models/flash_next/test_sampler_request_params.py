# SPDX-License-Identifier: Apache-2.0
"""min_tokens, logit_bias and min_p on the HPU's general sampler path (``VLLM_HPU_SAMPLER_REQUEST_PARAMS=1``).

Without the flag the HPU ignores all three on the general sampler path (its input batch never feeds vLLM's logits
processors); with it the runner applies each row's own request's values before vLLM's sampler
(``v1/sample/request_params.py``).

* The adjustments equal vLLM's rules: the min-tokens mask, ``logit_bias`` added to the raw logit, and vLLM's min_p
  cut at the row's temperature (``MinPLogitsProcessor`` on temperature-scaled logits).
* The serving loop (engine harness, tiny model, the fused greedy region off so the general path runs): a
  ``logit_bias`` of +100 on one token makes a greedy request write only that token, and a stop token the model
  would emit before ``min_tokens`` is not emitted; without the flag both are ignored (the old behaviour).
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.sample.request_params import RowParams, apply_request_adjustments, request_adjustments  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FLAG = "VLLM_HPU_SAMPLER_REQUEST_PARAMS"
GENERAL = {"VLLM_HPU_FUSED_GREEDY_LOGITS": "0", "VLLM_HPU_CANDIDATE_SAMPLING": "0"}


def test_the_adjustments_are_vllms_rules():
    gen = torch.Generator().manual_seed(4)
    logits = torch.randn(5, 300, generator=gen) * 3.0
    rows = [
        RowParams(min_tokens=4, num_output_tokens=1, stop_token_ids=(7, 9, 9)),  # below min_tokens: masked
        RowParams(min_tokens=4, num_output_tokens=4, stop_token_ids=(7, )),  # reached: nothing
        RowParams(logit_bias={3: 5.0, 200: -2.5}),
        RowParams(min_p=0.2, temperature=0.7),
        RowParams(min_p=0.2, temperature=0.0),  # greedy: min_p does not apply
    ]
    out = apply_request_adjustments(logits, request_adjustments(rows))
    want = logits.clone()
    want[0, [7, 9]] = float("-inf")
    want[2, 3] += 5.0
    want[2, 200] -= 2.5
    scaled = logits[3] / 0.7
    probs = scaled.softmax(dim=-1)
    want[3, probs < 0.2 * probs.max()] = float("-inf")  # vLLM's MinPLogitsProcessor on the scaled row
    assert torch.equal(out, want)
    assert not bool(request_adjustments([RowParams(), RowParams(min_tokens=3, num_output_tokens=5)]))


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


@pytest.mark.parametrize("flag", [0, 1])
def test_logit_bias_on_the_general_path(tiny_checkpoint, reference, flag):
    prompt = next(p for p in reference if len(p["prompt_ids"]) == 7)
    token = 4242
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2, env=GENERAL | {FLAG: str(flag)})) as engine:
        engine.add("b", prompt["prompt_ids"], max_tokens=6, logit_bias={token: 100.0})
        engine.run()
        tokens = engine.tokens["b"]
    if flag:
        assert tokens == [token] * 6
    else:
        assert tokens == prompt["generated_ids"][:6]  # ignored, as on every HPU server before this flag


@pytest.mark.parametrize("flag", [0, 1])
def test_min_tokens_on_the_general_path(tiny_checkpoint, reference, flag):
    prompt = next(p for p in reference if 30 <= len(p["prompt_ids"]) <= 700 and len(p["generated_ids"]) >= 8)
    stop = prompt["generated_ids"][2]
    assert stop not in prompt["generated_ids"][:2]
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2, env=GENERAL | {FLAG: str(flag)})) as engine:
        engine.add("s", prompt["prompt_ids"], max_tokens=6, min_tokens=6, stop_token_ids=[stop])
        engine.run()
        tokens = engine.tokens["s"]
    assert tokens[:2] == prompt["generated_ids"][:2]
    if flag:
        assert stop not in tokens and len(tokens) == 6
    else:
        assert tokens[2] == stop  # emitted before min_tokens (and the request ends there): the old behaviour
