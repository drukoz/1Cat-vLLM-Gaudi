# SPDX-License-Identifier: Apache-2.0
"""A decode step's inputs in persistent tensors (``VLLM_HPU_PERSISTENT_INPUTS=1``), on the whole serving loop.

* The same tokens with the flag off and on: plain greedy decode, and n-gram k = 3 verification steps with oracle
  drafts (every third one wrong), over prompts of shrinking lengths so the batch pads.
* With the flag the forward's inputs keep their addresses: every decode forward of one shape gets the same
  ``input_ids`` / ``positions`` tensor (the persistent device tensor), filled anew each step.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.test_decode_metadata_sparse import _oracle_drafts  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FLAG = "VLLM_HPU_PERSISTENT_INPUTS"
K = 3


@pytest.fixture(scope="module")
def reference(tiny_checkpoint) -> list[dict]:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]


def _serve(tiny_checkpoint, reference, flag: int, spec: bool):
    extra = {}
    speculative = None
    if spec:
        speculative = {"method": "ngram", "num_speculative_tokens": K, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
        extra = {"prefix_caching": False, "max_num_batched_tokens": 4096}  # as test_decode_metadata_sparse._serve
    options = eh.Options(max_num_seqs=8, speculative_config=speculative, env={FLAG: str(flag)}, **extra)
    with eh.open_engine(tiny_checkpoint, options) as engine:
        runner = engine.runner
        seen = defaultdict(set)  # decode input shape -> the input_ids tensors' ids seen
        forward = runner.model.forward

        def recording_forward(*args, **kwargs):
            ids = kwargs.get("input_ids")
            md = kwargs.get("attn_metadata")
            if ids is not None and md is not None and getattr(md, "is_prompt", True) is False:
                seen[tuple(ids.shape)].add((id(ids), ids.data_ptr()))
            return forward(*args, **kwargs)

        runner.model.forward = recording_forward
        want = {}
        for index, prompt in enumerate(reference):
            request_id = f"r{index}"
            want[request_id] = prompt["generated_ids"][:len(prompt["generated_ids"]) - index]
            engine.add(request_id, prompt["prompt_ids"], max_tokens=len(want[request_id]))
        if spec:
            _oracle_drafts(engine, want, K)
        engine.run()
        return dict(engine.tokens), seen, runner._persistent_inputs


@pytest.mark.parametrize("spec", [False, True])
def test_the_same_tokens_with_persistent_inputs(tiny_checkpoint, reference, spec):
    off_tokens, _, off_inputs = _serve(tiny_checkpoint, reference, 0, spec)
    on_tokens, seen, on_inputs = _serve(tiny_checkpoint, reference, 1, spec)
    assert on_tokens == off_tokens
    assert off_inputs is None and on_inputs is not None and len(on_inputs) > 0
    # One shape, one device tensor: the address does not move from step to step.
    assert seen and all(len({ptr for _, ptr in tensors}) == 1 for tensors in seen.values()), dict(seen)
