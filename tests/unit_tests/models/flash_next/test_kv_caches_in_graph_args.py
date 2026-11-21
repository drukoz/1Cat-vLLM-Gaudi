# SPDX-License-Identifier: Apache-2.0
"""The caches left out of the model forward's arguments (``VLLM_HPU_KV_CACHES_IN_GRAPH_ARGS=0``), on the whole loop.

* With ``1`` (default) every forward (prompt, decode, verification) gets ``kv_caches``; with ``0`` none does.
* The same tokens either way: greedy decode over prompts that share a prefix (prefix-cache hits, the second turn's
  prompt extends the first one's text across a block boundary), and n-gram k = 3 verification steps with oracle drafts.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import json  # noqa: E402

import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.test_decode_metadata_sparse import _oracle_drafts  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

FLAG = "VLLM_HPU_KV_CACHES_IN_GRAPH_ARGS"
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
        calls = {"with": 0, "without": 0, "prompt": 0}
        forward = runner.model.forward

        def recording_forward(*args, **kwargs):
            calls["with" if "kv_caches" in kwargs else "without"] += 1
            md = kwargs.get("attn_metadata")
            if md is not None and getattr(md, "is_prompt", False):
                calls["prompt"] += 1
            return forward(*args, **kwargs)

        runner.model.forward = recording_forward
        want = {}
        if spec:
            for index, prompt in enumerate(reference):
                request_id = f"r{index}"
                want[request_id] = prompt["generated_ids"][:len(prompt["generated_ids"]) - index]
                engine.add(request_id, prompt["prompt_ids"], max_tokens=len(want[request_id]))
            _oracle_drafts(engine, want, K)
            engine.run()
            return dict(engine.tokens), calls
        # Two turns: the first prompts, then each prompt again with its first turn's text and more after it (the
        # second turn's prefill hits the prefix cache and its new tokens start past a block boundary).
        first = {}
        for index, prompt in enumerate(reference[:4]):
            first[f"a{index}"] = prompt["prompt_ids"]
            engine.add(f"a{index}", prompt["prompt_ids"], max_tokens=24)
        engine.run()
        tokens = dict(engine.tokens)
        for index, prompt in enumerate(reference[:4]):
            again = list(first[f"a{index}"]) + list(tokens[f"a{index}"]) + list(prompt["prompt_ids"][:40])
            engine.add(f"b{index}", again, max_tokens=20)
        engine.run()
        return dict(engine.tokens), calls


@pytest.mark.parametrize("spec", [False, True])
def test_the_same_tokens_without_the_caches_in_the_arguments(tiny_checkpoint, reference, spec):
    on_tokens, on = _serve(tiny_checkpoint, reference, 1, spec)
    off_tokens, off = _serve(tiny_checkpoint, reference, 0, spec)
    assert off_tokens == on_tokens
    assert on["with"] > 0 and on["without"] == 0 and on["prompt"] > 0, on
    assert off["with"] == 0 and off["without"] == on["with"], (on, off)
