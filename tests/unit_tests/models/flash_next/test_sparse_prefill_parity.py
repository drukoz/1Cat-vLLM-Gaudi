# SPDX-License-Identifier: Apache-2.0
"""TP1 sparse-QSA prefill parity of the whole Flash-Next model against HF (M2-T3).

Builds the tiny model in ``QsaMode.SPARSE`` (``cpu_model(..., qsa_mode=QsaMode.SPARSE,
max_model_len=4096)``), prefills each reference prompt and compares the last-position logits
with ``reference.json``: the reference is the *sparse* HF oracle (indexer top-k, budget 2048),
so the 2600-token prompt is the case where a dense path cannot match it. The layer's
``HpuQsaIndexer.select`` picks the attended positions and the ``.sparse_attn`` seam
(:class:`SparseSdpaAttention`) attends over them, standing in for the HPU wrapper.

fp32 parity is tight (the selection reproduces HF's, top-1 and max |Δ| at the top-5 ids within
1e-5); bf16 is checked on the 2600-token prompt only (1e-2). A negative control prefills the
2600-token prompt in dense mode and asserts its logits are NOT within 1e-3 of the sparse
reference, so the parity checks above are sensitive to the selection.

Logits always come from ``fp32_logits``; one ``cpu_model`` per test (world-1 gloo).
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)

FP32_LOGIT_TOL = 1e-5  # the sparse selection reproduces HF's; the residual is fp32 reassociation
BF16_LOGIT_TOL = 1e-2  # bf16 sparse prefill on the 2600-token prompt
DENSE_NOT_SPARSE_GAP = 1e-3  # a dense prefill of the 2600-token prompt must NOT match the sparse reference
SPARSE_MAX_MODEL_LEN = 4096  # the compressed key cache has one row per 4 tokens; 2600 needs >= 650 rows
LONG_PROMPT = 2600


def model_sparse_attns(model) -> list:
    """The ``.sparse_attn`` stand-ins of every QSA layer (set ``.slot`` on them before a forward)."""
    return [m.sparse_attn for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]


def sparse_prefill_last_logits(model, vllm_config, eos: int, token_ids, slot: int = 1) -> torch.Tensor:
    """fp32 last-position logits ``(248320,)`` for one fresh sparse prefill of ``token_ids`` in slot ``slot``."""
    num_tokens = len(token_ids)
    reset_cpu_states(model)  # zero GDN/PLE state and the indexer side caches
    for att in model_sparse_attns(model):
        att.reset()
        att.slot = slot
    with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, slot), vllm_config):
        hidden = model(input_ids=torch.tensor(token_ids),
                       positions=torch.arange(num_tokens),
                       **ple_kwargs(num_tokens, eos))
    return fp32_logits(model, hidden)[-1]


def dense_prefill_last_logits(model, vllm_config, eos: int, token_ids, slot: int = 1) -> torch.Tensor:
    """fp32 last-position logits for a fresh dense (full-causal) prefill; the negative control."""
    num_tokens = len(token_ids)
    reset_cpu_states(model)
    with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, slot), vllm_config):
        hidden = model(input_ids=torch.tensor(token_ids),
                       positions=torch.arange(num_tokens),
                       **ple_kwargs(num_tokens, eos))
    return fp32_logits(model, hidden)[-1]


def top1_and_delta(logits: torch.Tensor, ref: dict) -> tuple[int, float]:
    """Last-position top-1 id and max |Δ| against the reference's five top-5 logits."""
    top1 = int(logits.argmax())
    delta = (logits[ref["ids"]] - torch.tensor(ref["logits"], dtype=torch.float32)).abs().max().item()
    return top1, delta


def test_sparse_prefill_fp32_matches_reference(tiny_checkpoint):
    """fp32 sparse prefill of EVERY reference prompt: top-1 equals the HF reference, top-5 |Δ| <= 1e-5."""
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    with cpu_model(tiny_checkpoint, fp32=True, max_model_len=SPARSE_MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (
            vllm_config, model):
        eos = eos_token_id(vllm_config)
        for prompt in reference["prompts"]:
            length = len(prompt["prompt_ids"])
            logits = sparse_prefill_last_logits(model, vllm_config, eos, prompt["prompt_ids"])
            ref = prompt["last_prefill_top5"]
            top1, delta = top1_and_delta(logits, ref)
            assert top1 == ref["ids"][0], (
                f"prompt {length}: sparse last-prefill top-1 {top1} != reference {ref['ids'][0]}")
            assert delta <= FP32_LOGIT_TOL, (
                f"prompt {length}: sparse prefill max |Δlogit| at the reference top-5 ids {delta:.3e} "
                f"> {FP32_LOGIT_TOL:.0e}")


def test_sparse_prefill_bf16_matches_reference(tiny_checkpoint):
    """bf16 sparse prefill of the 2600-token prompt: top-1 equals the HF reference, top-5 |Δ| <= 1e-2."""
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    long_prompt = next(p for p in reference["prompts"] if len(p["prompt_ids"]) == LONG_PROMPT)
    with cpu_model(tiny_checkpoint, max_model_len=SPARSE_MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (vllm_config,
                                                                                                        model):
        eos = eos_token_id(vllm_config)
        logits = sparse_prefill_last_logits(model, vllm_config, eos, long_prompt["prompt_ids"])
        ref = long_prompt["last_prefill_top5"]
        top1, delta = top1_and_delta(logits, ref)
        assert top1 == ref["ids"][0], f"prompt {LONG_PROMPT}: bf16 sparse top-1 {top1} != reference {ref['ids'][0]}"
        assert delta <= BF16_LOGIT_TOL, (
            f"prompt {LONG_PROMPT}: bf16 sparse prefill max |Δlogit| at the reference top-5 ids {delta:.3e} "
            f"> {BF16_LOGIT_TOL:.0e}")


def test_selection_width_and_causality(tiny_checkpoint, monkeypatch):
    """The selection contract M2-T5c depends on, captured on the 2600-token prompt.

    A monkeypatched wrapper on ``HpuQsaIndexer.select`` records the ``selected`` tensor of the
    sparse prefill: shape ``[2600, 2051]`` (``output_width = 512 * 4 + 3``), every entry ``-1``
    (padding) or ``<=`` its row's position (causality), every row selects its own position
    (a row with no valid entry gets an all ``-inf`` bias and FusedSDPA returns NaN for it),
    and every row below the width selects EVERY position ``<=`` its own (count == position + 1).
    """
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    long_prompt = next(p for p in reference["prompts"] if len(p["prompt_ids"]) == LONG_PROMPT)
    captured = []
    with cpu_model(tiny_checkpoint, fp32=True, max_model_len=SPARSE_MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (
            vllm_config, model):
        layer = next(m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention))
        orig_select = layer.indexer.select

        def record_select(*args, **kwargs):
            out = orig_select(*args, **kwargs)
            captured.append(out.detach().clone())
            return out

        monkeypatch.setattr(layer.indexer, "select", record_select)
        eos = eos_token_id(vllm_config)
        sparse_prefill_last_logits(model, vllm_config, eos, long_prompt["prompt_ids"])
    assert len(captured) == 1, f"indexer.select ran {len(captured)} times, expected 1"
    selected = captured[0]
    positions = torch.arange(LONG_PROMPT)
    width = layer.indexer.output_width
    assert width == 2051, f"output_width {width} != 512 blocks * 4 + 3"
    assert selected.shape == (LONG_PROMPT, width), f"selected shape {tuple(selected.shape)} != (2600, 2051)"
    valid = selected >= 0
    assert bool(((~valid) | (selected <= positions[:, None])).all()), "selected carries a future position"
    own = (selected == positions[:, None]).any(dim=-1)
    assert bool(own.all()), f"rows without their own position: {(~own).nonzero().flatten()[:8].tolist()}"
    counts = valid.sum(dim=-1)
    short = positions < width
    assert bool((counts[short] == positions[short] + 1).all()), (
        "a row below the width does not select every position <= its own")


def test_dense_prefill_is_not_the_sparse_reference(tiny_checkpoint):
    """Negative control: the dense prefill of the 2600-token prompt is NOT within 1e-3 of the sparse reference.

    If the dense gap ever drops to <= 1e-3 the reference no longer separates sparse from dense (the
    selection is a no-op there), so this test guards that the fp32/bf16 parity checks are sensitive.
    """
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    long_prompt = next(p for p in reference["prompts"] if len(p["prompt_ids"]) == LONG_PROMPT)
    with cpu_model(tiny_checkpoint, fp32=True, max_model_len=SPARSE_MAX_MODEL_LEN) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        logits = dense_prefill_last_logits(model, vllm_config, eos, long_prompt["prompt_ids"])
        ref = long_prompt["last_prefill_top5"]
        _, delta = top1_and_delta(logits, ref)
        assert delta > DENSE_NOT_SPARSE_GAP, (
            f"prompt {LONG_PROMPT}: dense-vs-sparse-reference gap {delta:.3e} <= {DENSE_NOT_SPARSE_GAP:.0e}; "
            "the reference is no longer sparse-sensitive, so the parity checks above prove nothing")
