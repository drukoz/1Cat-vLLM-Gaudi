# SPDX-License-Identifier: Apache-2.0
"""Sparse QSA with the M-RoPE position layout the HPU runner sends: ``[3, num_tokens]``.

When the server runs WITHOUT ``--language-model-only``, ``model_config.uses_mrope`` is True (the checkpoint's
``rope_parameters`` carry ``mrope_section``), the QSA layer builds ``MRotaryEmbedding`` and ``HPUModelRunner``
sends the positions of a prompt (``_align_and_pad_mrope_positions``) and of a decode step as three rows
(temporal, height, width). The sparse branch of ``HpuQwen4ExpQSAAttention.forward`` flattened them
(``positions.reshape(-1)``), which gave the indexer ``3 * num_tokens`` positions for ``num_tokens`` rows and
failed in ``index_queries`` on shape.

With ``--language-model-only`` (the production setting, and what ``cpu_model`` builds) the engine strips the
M-RoPE keys (``_strip_qwen4_exp_mrope`` in ``vllm/model_executor/models/config.py``): ``uses_mrope`` is False
and the positions are ``[rows, bucket]``, for which flattening is right. So these tests swap in the
``MRotaryEmbedding`` (``hpu_bucket.use_mrope_rotary``) to build the multimodal server's layer.

What the engine does (``vllm/models/qwen4_exp/nvidia/indexer_qsa.py`` at the pin): the rotation of the indexer
queries and block keys applies the attention's M-RoPE composition (``apply_qsa_rope`` -> ``triton_mrope``, each
rotary section from its own row), and the block bookkeeping (which blocks are complete and visible) uses the
request's logical token index (``QSAForwardMetadata.logical_positions``), not a RoPE row. For a text token the
three rows are equal and equal to that index, so the composition is the 1-D rotation at row 0 and row 0 is the
logical position: the plugin's indexer takes row 0, which is exact for text tokens. These tests pin that:
three equal rows give the 1-D result bit for bit, in prefill and in decode.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, decode_metadata, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)
from tests.unit_tests.models.flash_next.hpu_bucket import use_mrope_rotary  # noqa: E402
from tests.unit_tests.models.flash_next.test_sparse_prefill_parity import (  # noqa: E402
    SPARSE_MAX_MODEL_LEN, model_sparse_attns)

PROMPT = 700  # a reference prompt long enough for several score tiles, short enough to be quick
SLOT = 1


def load_prompt(tiny_checkpoint, length: int) -> dict:
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    return next(p for p in reference["prompts"] if len(p["prompt_ids"]) == length)


def three_rows(positions: torch.Tensor) -> torch.Tensor:
    """The runner's M-RoPE layout of text positions: int32 ``[3, N]``, three equal rows."""
    return positions.to(torch.int32).reshape(1, -1).expand(3, -1).contiguous()


def record_selection(monkeypatch, model) -> list:
    layer = next(m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention))
    captured = []
    original = layer.indexer.select

    def select(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out.detach().clone())
        return out

    monkeypatch.setattr(layer.indexer, "select", select)
    return captured


def prefill(model, vllm_config, eos, token_ids, positions) -> torch.Tensor:
    reset_cpu_states(model)
    for att in model_sparse_attns(model):
        att.reset()
        att.slot = SLOT
    num_tokens = len(token_ids)
    with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, SLOT), vllm_config):
        hidden = model(input_ids=torch.tensor(token_ids), positions=positions, **ple_kwargs(num_tokens, eos))
    return fp32_logits(model, hidden)


def decode(model, vllm_config, eos, seq, pos, positions) -> torch.Tensor:
    with torch.no_grad(), set_forward_context(decode_metadata(SLOT, pos), vllm_config):
        hidden = model(input_ids=torch.tensor([seq[pos]]),
                       positions=positions,
                       **ple_kwargs(1, eos, context=(seq[pos - 2], seq[pos - 1])))
    return fp32_logits(model, hidden)[-1]


def test_indexer_positions_takes_row_zero():
    from vllm_gaudi.models.qwen4_exp.qsa import indexer_positions

    line = torch.tensor([5, 6, 7, -1], dtype=torch.int32)
    assert torch.equal(indexer_positions(three_rows(line)), line)
    assert torch.equal(indexer_positions(line), line)
    assert indexer_positions(three_rows(line[:1])).shape == (1, )  # a [3, 1] single-token decode step


def test_prefill_three_equal_rows_match_one_row(tiny_checkpoint, monkeypatch):
    """A sparse prefill with ``[3, N]`` positions equals the 1-D prefill: logits and selection, bit for bit."""
    prompt = load_prompt(tiny_checkpoint, PROMPT)["prompt_ids"]
    with cpu_model(tiny_checkpoint, fp32=True, max_model_len=SPARSE_MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (
            vllm_config, model):
        eos = eos_token_id(vllm_config)
        use_mrope_rotary(model)
        captured = record_selection(monkeypatch, model)
        line = torch.arange(PROMPT)
        one_row = prefill(model, vllm_config, eos, prompt, line)
        three = prefill(model, vllm_config, eos, prompt, three_rows(line))
        assert len(captured) == 2
        assert torch.equal(captured[0], captured[1]), "the selection changed with the [3, N] position layout"
        assert torch.equal(one_row, three), (
            f"[3, N] positions vs 1-D: max |delta logit| = {(one_row - three).abs().max().item():.3e}")


def test_decode_three_equal_rows_match_one_row(tiny_checkpoint, monkeypatch):
    """Decode steps with ``[3, 1]`` positions equal the 1-D steps (positions 700..703: one block completes)."""
    prompt = load_prompt(tiny_checkpoint, PROMPT)
    seq = list(prompt["prompt_ids"]) + list(prompt["generated_ids"])
    with cpu_model(tiny_checkpoint, fp32=True, max_model_len=SPARSE_MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (
            vllm_config, model):
        eos = eos_token_id(vllm_config)
        use_mrope_rotary(model)
        layer = next(m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention))
        captured = record_selection(monkeypatch, model)
        results = []
        for layout in (lambda p: torch.tensor([p]), lambda p: three_rows(torch.tensor([p]))):
            prefill(model, vllm_config, eos, seq[:PROMPT], torch.arange(PROMPT))
            captured.clear()
            logits = [decode(model, vllm_config, eos, seq, pos, layout(pos)) for pos in range(PROMPT, PROMPT + 4)]
            results.append((torch.stack(logits), [c.clone() for c in captured],
                            layer.indexer.compressed_key_cache.clone(), layer.indexer.key_ring.clone()))
        (logits_1, selected_1, comp_1, ring_1), (logits_3, selected_3, comp_3, ring_3) = results
        assert all(torch.equal(a, b) for a, b in zip(selected_1, selected_3))
        assert torch.equal(logits_1, logits_3)
        assert torch.equal(comp_1, comp_3) and torch.equal(ring_1, ring_3)
        assert bool(comp_1[PROMPT // 4].any()), "the decode chain did not complete block 175"
