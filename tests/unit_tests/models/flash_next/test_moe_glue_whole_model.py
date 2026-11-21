# SPDX-License-Identifier: Apache-2.0
"""The whole tiny Flash-Next model on CPU with ``VLLM_HPU_MOE_GLUE=1`` against the same model with 0.

The model runs the plugin's HPU MoE path with stand-ins for the vendor ops (``moe_block.py``), and the
fused glue ops run the torch oracle (``tests/unit_tests/hc_fused/moe_ops.py``), so the two builds must
agree bit for bit: the prefill hidden states of every committed reference prompt, and every token of a
greedy generation (prefill, then single-token decode steps carrying the GDN, PLE and attention state).
This proves that the real model gives every MoE block an input the fused path takes, that no block
falls back, and that prefill-sized and decode-sized calls go through the same wiring.

The first test checks the harness itself against an independent oracle: on the plugin's path with the
stand-ins the tiny model reproduces the Hugging Face reference, as it does on the engine's CPU path
(``test_prefill_parity.py``), in fp32 with the flag at 0 and in bf16 with the flag at 1.

The last test measures what the oracle stand-in cannot show: the routing kernel picks the lowest expert
id among equal logits, ``torch.topk`` picks as it likes, and bf16 logits tie at the cut in about one row
in a hundred on this model.

One ``cpu_model`` may be open at a time, so each test builds, runs and closes one model after the other.
Run this file in its own pytest process: importing it switches every model of the process to the plugin's
MoE path (``moe_block.use_plugin_fast_path``).
"""

import collections
import json

from tests.unit_tests.models.flash_next import moe_block

moe_block.use_plugin_fast_path()

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

import vllm_gaudi.ops.hpu_fused_moe as fused_moe  # noqa: E402
from tests.unit_tests.hc_fused import moe_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, dense_reference_prompts, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)
from tests.unit_tests.models.flash_next.moe_block import OpRecorder, moe_blocks  # noqa: E402
from tests.unit_tests.models.flash_next.test_decode_parity import (  # noqa: E402
    decode, install_cached_attention, prefill)
from tests.unit_tests.models.flash_next.test_prefill_parity import (  # noqa: E402
    BF16_LOGIT_TOL, FP32_LOGIT_TOL, check_last_prefill_logits)

FLAG = "VLLM_HPU_MOE_GLUE"
NEW_TOKENS = 16


def _generate(model, vllm_config, atts, prompt_ids, eos) -> list[int]:
    """Greedy generation of NEW_TOKENS tokens: one prefill, then one decode step per token."""
    reset_cpu_states(model)
    for att in atts:
        att.reset()
    seq = list(prompt_ids)
    logits = prefill(model, vllm_config, seq, 1, eos)[-1]
    tokens = []
    for _ in range(NEW_TOKENS):
        tokens.append(int(logits.argmax()))
        seq.append(tokens[-1])
        logits = decode(model, vllm_config, seq, len(seq) - 1, 1, eos)[-1]
    return tokens


def _run(tiny_checkpoint, prompts) -> dict:
    """Build the bf16 model with the environment as it is now; prefill and generate every prompt; close it."""
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        hidden = []
        with OpRecorder() as recorder:
            for prompt_ids in prompts:
                n = len(prompt_ids)
                reset_cpu_states(model)
                with torch.no_grad(), set_forward_context(prefill_metadata(n, 1), vllm_config):
                    out = model(input_ids=torch.tensor(prompt_ids), positions=torch.arange(n), **ple_kwargs(n, eos))
                hidden.append(out.clone())
            atts = install_cached_attention(model)
            tokens = [_generate(model, vllm_config, atts, prompt_ids, eos) for prompt_ids in prompts]
        states = [getattr(block.experts, "_hpu_moe_glue", None) for block in moe_blocks(model)]
        top1 = [int(fp32_logits(model, h)[-1].argmax()) for h in hidden]
    glue_ops = collections.Counter(
        name.split(".")[1] for name in recorder.names if name.startswith("flashinfer_gaudi.moe_"))
    return {"hidden": hidden, "tokens": tokens, "states": states, "glue_ops": glue_ops, "top1": top1,
            "vendor_calls": recorder.names.count("hpu.mixture_of_experts"),
            "hc_ops": recorder.count("flashinfer_gaudi.hc_")}


@pytest.fixture
def prompts(tiny_checkpoint):
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    return [list(prompt["prompt_ids"]) for prompt in dense_reference_prompts(reference)]


@pytest.fixture
def lines(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(fused_moe.logger, "warning", lambda message, *args: seen.append(message % args))
    return seen


def test_the_plugin_path_with_the_stand_ins_reproduces_the_hugging_face_reference(tiny_checkpoint, monkeypatch, lines):
    """The plugin's MoE fast path, ending in the vendor stand-in, is the reference model.

    fp32 with the flag at 0 (the kernels are bf16 only): the last-prefill top-1 is the reference's and its
    top-5 logits are within 1e-5.  bf16 with the flag at 1, every block fused: the top-1 is the reference's
    and the logits are within 1e-2, the tolerance the engine's CPU path is held to.
    """
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    reference_prompts = dense_reference_prompts(reference)

    monkeypatch.setenv(FLAG, "0")
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        blocks = moe_blocks(model)
        assert type(blocks[0].experts.routed_experts.quant_method).__name__ == "HPUUnquantizedFusedMoEMethod"
        assert type(blocks[0].experts).forward is fused_moe._patched_default_moe_runner_forward
        moe_block.VENDOR_CALLS.clear()
        check_last_prefill_logits(model, vllm_config, reference_prompts, FP32_LOGIT_TOL)
        assert len(moe_block.VENDOR_CALLS) == 4 * len(reference_prompts)  # the plugin's path really ran

    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        check_last_prefill_logits(model, vllm_config, reference_prompts, BF16_LOGIT_TOL)
        states = [block.experts._hpu_moe_glue for block in moe_blocks(model)]
    assert lines == [] and all(state.ok is True for state in states)


def test_hidden_states_and_generated_tokens_are_unchanged_by_the_flag(tiny_checkpoint, prompts, monkeypatch, lines):
    assert [len(p) for p in prompts] == [1, 7, 33, 100, 700]  # the committed reference prompts dense attention covers
    monkeypatch.setenv(FLAG, "0")
    plain = _run(tiny_checkpoint, prompts)
    assert plain["states"] == [None] * 4 and sum(plain["glue_ops"].values()) == 0

    monkeypatch.setenv(FLAG, "1")
    fused = _run(tiny_checkpoint, prompts)

    # every MoE block ran fused, and none wrote its "is off" line
    assert lines == []
    assert [state.ok is True and not state.logged for state in fused["states"]] == [True] * 4
    # per model forward: 4 blocks, each one routing op, one activation op and one combine op (bf16 experts);
    # per prompt: the prefill for the hidden states, then a prefill and NEW_TOKENS decode steps
    forwards = len(prompts) * (2 + NEW_TOKENS)
    assert fused["vendor_calls"] == plain["vendor_calls"] == 4 * forwards
    assert fused["glue_ops"] == {"moe_route": 4 * forwards, "moe_silu_mul": 4 * forwards, "moe_combine": 4 * forwards}

    for prompt_ids, a, b in zip(prompts, fused["hidden"], plain["hidden"]):
        assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape
        assert torch.equal(a, b), f"prefill of {len(prompt_ids)} tokens differs"
    assert fused["tokens"] == plain["tokens"]
    assert all(len(tokens) == NEW_TOKENS for tokens in fused["tokens"])
    assert [tokens[0] for tokens in fused["tokens"]] == fused["top1"]


def test_both_fused_paths_in_one_model(tiny_checkpoint, prompts, monkeypatch, lines):
    """VLLM_HPU_HC_FUSED=1 and VLLM_HPU_MOE_GLUE=1 together (one library, one extension): the bits of both at 0."""
    from tests.unit_tests.hc_fused import hc_ops  # noqa: F401  the hyper-connection ops' CPU stand-in

    short = prompts[:4]
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0")
    plain = _run(tiny_checkpoint, short)
    assert plain["hc_ops"] == 0 and sum(plain["glue_ops"].values()) == 0

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    fused = _run(tiny_checkpoint, short)

    assert lines == [] and all(state.ok is True for state in fused["states"])
    assert fused["hc_ops"] > 0 and fused["glue_ops"]["moe_route"] == 4 * len(short) * (2 + NEW_TOKENS)
    for a, b in zip(fused["hidden"], plain["hidden"]):
        assert torch.equal(a, b)
    assert fused["tokens"] == plain["tokens"]


def test_what_the_kernels_tie_rule_does_to_the_tokens(tiny_checkpoint, prompts, monkeypatch, lines):
    """The routing kernel's choice among equal logits (the lowest expert id) instead of torch.topk's.

    Not a requirement but a measurement for the card: a row that ties at the top-K cut may get another
    expert with the same weight, which is a valid routing and can still flip a greedy token of the
    random tiny model.  The test records how many rows tie, how many get another set, and whether the
    generated tokens of the reference prompts stay the same; it fails only on an invalid selection.
    """
    from flashinfer_gaudi import moe_glue
    from tests.unit_tests.hc_fused import moe_case

    monkeypatch.setenv(FLAG, "0")
    plain = _run(tiny_checkpoint, prompts)

    stats = collections.Counter()

    def kernel_route(gates, experts, top_k):
        case = moe_case.MoeCase("live", "route", gates.shape[0], experts=experts, top_k=top_k, columns=gates.shape[1])
        inputs = {"gates": gates}
        ids = moe_case.route_kernel_ids(case, inputs)
        weights = moe_case.route_expected_weights(case, inputs, ids)
        assert weights is not None, "the kernel's rule gave an invalid top-K"
        torch_ids, _ = moe_ops.CPU_KERNELS["moe_route"](gates, experts, top_k)
        other_set = (torch.sort(ids, dim=1).values != torch.sort(torch_ids, dim=1).values).any(dim=1)
        stats["rows"] += gates.shape[0]
        stats["tie_rows"] += int(moe_case.route_tie_rows(case, inputs).sum())
        stats["other_set_rows"] += int(other_set.sum())
        return ids, weights

    monkeypatch.setattr(moe_glue, "moe_route", kernel_route)
    monkeypatch.setenv(FLAG, "1")
    fused = _run(tiny_checkpoint, prompts)

    assert lines == [] and all(state.ok is True for state in fused["states"])
    same_prompts = sum(a == b for a, b in zip(fused["tokens"], plain["tokens"]))
    same_tokens = sum(x == y for a, b in zip(fused["tokens"], plain["tokens"]) for x, y in zip(a, b))
    same_hidden = sum(torch.equal(a, b) for a, b in zip(fused["hidden"], plain["hidden"]))
    print(f"\nMOETIE routed rows {stats['rows']}, rows that tie at the cut {stats['tie_rows']}, rows where the "
          f"lowest-id rule selects another set than torch.topk {stats['other_set_rows']}; prompts with unchanged "
          f"generated tokens {same_prompts}/{len(prompts)} ({same_tokens}/{len(prompts) * NEW_TOKENS} tokens), "
          f"prefills with unchanged hidden states {same_hidden}/{len(prompts)}")
    assert stats["other_set_rows"] <= stats["tie_rows"]  # only a tie at the cut can change the set
    if stats["other_set_rows"] == 0:
        assert fused["tokens"] == plain["tokens"]
