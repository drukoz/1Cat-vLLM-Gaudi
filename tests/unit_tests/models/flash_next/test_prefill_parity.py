# SPDX-License-Identifier: Apache-2.0
"""TP1 prefill parity of the whole Flash-Next model against the HF reference (M0-T7b).

Builds ``HpuQwen4ExpForConditionalGeneration`` from the tiny checkpoint on CPU (``cpu_model.py``),
prefills each reference prompt and compares the last-position logits with ``reference.json``: tight
in fp32, loose in bf16. Greedy decoding re-prefills the whole sequence at every step (16 tokens,
fp32 only: bf16 ties flip greedy picks); decoding with carried state is T7c, not this file.

Logits always come from ``fp32_logits``: ``compute_logits`` returns bf16 even after ``to_fp32``, and
bf16 ties flip the argmax on the tiny model's flat logits. One ``cpu_model`` may be open at a time
(world-1 gloo), so each test opens its own build and loops over the 4 prompts inside.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)

FP32_LOGIT_TOL = 1e-5  # CTO probe on this branch: max |Δ| 4.5e-7 .. 6.6e-7
BF16_LOGIT_TOL = 1e-2  # CTO probe on this branch: max |Δ| 1.9e-3 .. 2.5e-3
SLOT_LOGIT_TOL = 1e-6  # a fresh prefill must not depend on the state-slot index

PROMPT_LENGTH_33 = 33


def prefill_last_logits(model, vllm_config, eos: int, token_ids, slot: int = 1) -> torch.Tensor:
    """fp32 logits ``(248320,)`` of the last position for one fresh prefill of ``token_ids`` in slot ``slot``."""
    num_tokens = len(token_ids)
    reset_cpu_states(model)  # every prefill here starts from zeroed GDN/PLE state
    with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, slot), vllm_config):
        hidden = model(input_ids=torch.tensor(token_ids),
                       positions=torch.arange(num_tokens),
                       **ple_kwargs(num_tokens, eos))
    return fp32_logits(model, hidden)[-1]


def check_last_prefill_logits(model, vllm_config, prompts, tol: float) -> None:
    """Per prompt: last-position top-1 equals the reference top-1 and max |Δ| at the 5 ids <= ``tol``."""
    eos = eos_token_id(vllm_config)
    for prompt in prompts:
        length = len(prompt["prompt_ids"])
        logits = prefill_last_logits(model, vllm_config, eos, prompt["prompt_ids"])
        ref = prompt["last_prefill_top5"]
        top1 = int(logits.argmax())
        delta = (logits[ref["ids"]] - torch.tensor(ref["logits"], dtype=torch.float32)).abs().max().item()
        assert top1 == ref["ids"][0], f"prompt {length}: last-prefill top-1 {top1} != reference {ref['ids'][0]}"
        assert delta <= tol, f"prompt {length}: max |Δlogit| at the reference top-5 ids {delta:.3e} > {tol:.0e}"


def test_prefill_fp32_matches_reference(tiny_checkpoint):
    """fp32 last-prefill logits: top-1 equals the HF reference and max |Δ| at its top-5 ids <= 1e-5."""
    prompts = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        check_last_prefill_logits(model, vllm_config, prompts, FP32_LOGIT_TOL)


def test_greedy_fp32_matches_reference(tiny_checkpoint):
    """fp32 greedy decoding by re-prefilling the growing sequence: the 16 tokens equal the reference."""
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        for prompt in reference["prompts"]:
            length = len(prompt["prompt_ids"])
            seq = list(prompt["prompt_ids"])
            generated = []
            for _ in range(reference["max_new_tokens"]):
                token = int(prefill_last_logits(model, vllm_config, eos, seq).argmax())
                generated.append(token)
                seq.append(token)
            expected = prompt["generated_ids"]
            first = next((i for i, (a, b) in enumerate(zip(generated, expected)) if a != b), len(expected))
            assert generated == expected, (
                f"prompt {length}: greedy tokens {generated} != reference {expected}, "
                f"first mismatch at index {first}")


def test_prefill_bf16_matches_reference(tiny_checkpoint):
    """bf16 last-prefill logits: top-1 equals the HF reference and max |Δ| at its top-5 ids <= 1e-2."""
    prompts = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        check_last_prefill_logits(model, vllm_config, prompts, BF16_LOGIT_TOL)


def test_state_slot_does_not_change_prefill(tiny_checkpoint):
    """The 33-token prompt prefilled into state slot 1 and slot 3 gives the same last-position logits."""
    prompts = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))["prompts"]
    prompt = next(p for p in prompts if len(p["prompt_ids"]) == PROMPT_LENGTH_33)
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        slot_1 = prefill_last_logits(model, vllm_config, eos, prompt["prompt_ids"], slot=1)
        slot_3 = prefill_last_logits(model, vllm_config, eos, prompt["prompt_ids"], slot=3)
        delta = (slot_1 - slot_3).abs().max().item()
        assert delta <= SLOT_LOGIT_TOL, (
            f"prompt {PROMPT_LENGTH_33}: slot 1 vs slot 3 last-prefill logits "
            f"max |Δ| {delta:.3e} > {SLOT_LOGIT_TOL:.0e}")
