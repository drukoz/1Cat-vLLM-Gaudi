# SPDX-License-Identifier: Apache-2.0
"""Prompt logprobs on the HPU runner with the Flash-Next model (``engine_harness``, tiny checkpoint).

On the card (engine-step job 00130) a request with ``prompt_logprobs`` ended the engine: the runner kept the prefill
hidden states of request ``i`` as ``non_flattened_hidden_states[i]``, but Flash-Next returns them flattened
(``[batch * seq, hidden]``), so ``[i]`` was one token's vector and the LM head failed on a ``[1, seq]`` input
(``matrix 1 and matrix 2 shapes cannot be multiplied (1x6 and 128x248320)``). Now the request's rows are taken by
the prompt batch's ``[batch, seq]`` shape. The prompt logprobs must be those of the decoding: the row that
predicts the greedy token ranks it first.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _run_with_prompt_logprobs(engine, request_id, prompt_ids, max_tokens, prompt_logprobs):
    captured = {}
    step_fn = engine.core.step_fn

    def capturing_step():
        outputs, executed = step_fn()
        for engine_outputs in (outputs or {}).values():
            for output in engine_outputs.outputs:
                tensors = getattr(output, "new_prompt_logprobs_tensors", None)
                if tensors is not None:
                    captured.setdefault(output.request_id, []).append(tensors)
        return outputs, executed

    engine.core.step_fn = capturing_step
    try:
        engine.add(request_id, prompt_ids, max_tokens=max_tokens, prompt_logprobs=prompt_logprobs)
        engine.run()
    finally:
        engine.core.step_fn = step_fn
    return captured.get(request_id, [])


def test_prompt_logprobs_rank_the_greedy_token_first(tiny_checkpoint, reference):
    prompt = next(p for p in reference if len(p["prompt_ids"]) == 7)  # 16 of 16 on every path: no near tie
    ids, generated = prompt["prompt_ids"], prompt["generated_ids"]
    with eh.open_engine(tiny_checkpoint, eh.Options(max_num_seqs=2)) as engine:
        chunks = _run_with_prompt_logprobs(engine, "q", ids + generated[:2], max_tokens=2, prompt_logprobs=2)
        assert engine.tokens["q"] == generated[2:4]
    assert chunks, "no prompt logprobs came back"
    token_ids = [row for chunk in chunks for row in chunk.logprob_token_ids.tolist()]
    ranks = [rank for chunk in chunks for rank in chunk.selected_token_ranks.tolist()]
    prompt_ids = ids + generated[:2]
    assert len(token_ids) == len(prompt_ids) - 1
    # Column 0 is the prompt's own next token; columns 1 and 2 the top 2 of that position.
    assert [row[0] for row in token_ids] == prompt_ids[1:]
    assert all(0 <= t < 300000 for row in token_ids for t in row)
    # The rows that predict the two generated tokens of the prompt: greedy took them, so they rank first.
    for position in (len(ids) - 1, len(ids)):
        assert token_ids[position - 1][0] == prompt_ids[position]
    assert ranks[len(ids) - 1] == 1 and ranks[len(ids)] == 1, ranks
    assert token_ids[len(ids) - 1][1] == generated[0] and token_ids[len(ids)][1] == generated[1]
