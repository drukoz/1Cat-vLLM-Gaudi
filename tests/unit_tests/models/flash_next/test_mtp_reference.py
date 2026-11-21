# SPDX-License-Identifier: Apache-2.0
"""Tests for the MTP tensors and the MTP draft reference of the tiny Flash-Next checkpoint.

The checkpoint carries, on top of the 132 target tensors, the 31 ``mtp.*`` tensors of a one-layer multi-token
predictor; ``reference.json`` (schema 2) gains a per-prompt ``mtp`` block of teacher-forced draft ids. The target
model ignores ``mtp.*``. Runs on CPU on the session ``tiny_checkpoint`` fixture.
"""

import json

import torch
from safetensors import safe_open
from transformers import Qwen4ExpForConditionalGeneration

from tools.flash_next.make_tiny_checkpoint import MTP_PREFIX, MTP_TENSOR_NAMES, WEIGHTS_FILE

TOP_K = 5


def test_mtp_tensors_present(tiny_checkpoint) -> None:
    """(a) The checkpoint carries exactly the 31 ``mtp.*`` tensors, all bfloat16, without touching target names."""
    with safe_open(tiny_checkpoint / WEIGHTS_FILE, framework="pt") as handle:
        mtp = {name: handle.get_tensor(name) for name in handle.keys() if name.startswith(MTP_PREFIX)}

    assert set(mtp) == {MTP_PREFIX + name for name in MTP_TENSOR_NAMES}
    assert len(mtp) == 31
    assert all(tensor.dtype == torch.bfloat16 for tensor in mtp.values()), "MTP tensors must be saved in bfloat16"


def test_reference_has_mtp(tiny_checkpoint) -> None:
    """(b, c) ``reference.json`` is schema 2 with ``mtp_tensors=31`` and a well-formed ``mtp`` block per prompt."""
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    assert reference["schema"] == 2
    assert reference["mtp_tensors"] == 31

    for prompt in reference["prompts"]:
        length = len(prompt["prompt_ids"])
        mtp = prompt["mtp"]
        expected_rows = list(range(length)) if length <= 100 else list(range(length - 8, length))
        assert mtp["rows"] == expected_rows
        assert len(mtp["rows"]) == min(length, 8 if length > 100 else length)
        assert len(mtp["draft_argmax"]) == len(mtp["rows"])
        assert all(isinstance(token, int) for token in mtp["draft_argmax"])
        assert set(mtp["last_row_top5"]) == {"ids", "logits"}
        assert len(mtp["last_row_top5"]["ids"]) == TOP_K


def test_target_load_ignores_mtp(tiny_checkpoint) -> None:
    """(d) The Hugging Face target model loads the checkpoint and ignores every ``mtp.*`` tensor."""
    model = Qwen4ExpForConditionalGeneration.from_pretrained(tiny_checkpoint)
    assert not hasattr(model, "mtp")
    state = model.state_dict()
    assert not any(name.startswith(MTP_PREFIX) for name in state)
    assert "lm_head.weight" in state
