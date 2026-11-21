# SPDX-License-Identifier: Apache-2.0
"""A greedy verification's logits region inside the captured forward (``VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1``).

Step-anatomy's 00205 trace: after the verify forward's replay the runner ran the LM head, the logits all-gather's
concatenation and the rejection sampler as three more graphs. With the flag the forward takes the logits indices and
returns the selected rows and every row's greedy token (int32 from an fp32 id table, card rule 9): the final mix, the
LM head and the arg max are one graph segment, and the drafts are verified from those tokens.

* ``greedy_ids`` is the arg max (ties to the lowest id) with ids from the fp32 table.
* The serving loop (engine harness, tiny model, n-gram k = 3 with oracle drafts, every third draft wrong, prompts of
  shrinking lengths so the batch pads): the same tokens with the flag off and on, every verification step through
  the forward's region with the flag (the rejection sampler never runs then), none without it.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from tests.unit_tests.models.flash_next.test_decode_metadata_sparse import _oracle_drafts  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FLAG = "VLLM_HPU_VERIFY_LOGITS_IN_GRAPH"
K = 3


def test_greedy_ids_are_the_arg_max_from_the_id_table():
    from vllm_gaudi.distributed import vocab_parallel

    class Head:

        def compute_logits(self, hidden):
            return hidden

    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(9, 300, generator=gen)
    logits[3, 17] = logits[3, 250] = logits[3].max() + 1.0  # a tie: the lowest id wins, as argmax
    ids = vocab_parallel.greedy_ids(Head(), logits.to(torch.bfloat16))
    assert ids.dtype == torch.int32 and ids.shape == (9, )
    assert ids.tolist() == logits.to(torch.bfloat16).to(torch.float32).argmax(dim=-1).tolist()
    assert int(ids[3]) == 17


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _serve(tiny_checkpoint, reference, flag: int):
    speculative = {"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    # As test_decode_metadata_sparse._serve: whole-prompt prefills and no prefix caching for the program head's
    # speculative path in this harness.
    options = eh.Options(max_num_seqs=8, speculative_config=speculative, env={FLAG: str(flag)}, prefix_caching=False,
                         max_num_batched_tokens=4096)
    with eh.open_engine(tiny_checkpoint, options) as engine:
        runner = engine.runner
        counts = {"in_graph": 0, "rejection": 0}
        forward = runner.model.forward

        def recording_forward(*args, **kwargs):
            if kwargs.get("verify_logits_indices") is not None:
                counts["in_graph"] += 1
            return forward(*args, **kwargs)

        runner.model.forward = recording_forward

        class CountingSampler:

            def __init__(self, sampler):
                self._sampler = sampler

            def __call__(self, *args, **kwargs):
                counts["rejection"] += 1
                return self._sampler(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._sampler, name)

        runner.rejection_sampler = CountingSampler(runner.rejection_sampler)
        want = {}
        for index, prompt in enumerate(reference):
            request_id = f"r{index}"
            want[request_id] = prompt["generated_ids"][:len(prompt["generated_ids"]) - index]
            engine.add(request_id, prompt["prompt_ids"], max_tokens=len(want[request_id]))
        _oracle_drafts(engine, want, K)
        engine.run()
        return dict(engine.tokens), want, counts


def test_the_same_tokens_with_the_region_in_the_forward(tiny_checkpoint, reference):
    off_tokens, want, off = _serve(tiny_checkpoint, reference, 0)
    on_tokens, _, on = _serve(tiny_checkpoint, reference, 1)
    assert on_tokens == off_tokens
    assert off["in_graph"] == 0 and off["rejection"] > 0
    assert on["in_graph"] >= 4 and on["rejection"] == 0, on
    # the oracle drafts keep the reference streams (up to the tiny model's near ties, compared off against on above)
    assert sum(off_tokens[r] == want[r] for r in want) >= len(want) - 1
