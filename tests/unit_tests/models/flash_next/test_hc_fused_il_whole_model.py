# SPDX-License-Identifier: Apache-2.0
"""The whole tiny Flash-Next model on CPU with ``VLLM_HPU_HC_FUSED_IL=1`` (the interleaved hyper-connection
kernels, docs/design/flash-next/hc-kernels.md) against the torch path, in the copy mode the builds run (2).

On CPU the interleaved ops run the torch oracle, so the model must give the torch path's bits: this proves that
every combine_and_mix and every gated mix of the real model reaches the interleaved ops, that none falls back or
is handed a view, and that the MTP draft model's layer (handed a ``flatten`` view) works with them.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from tests.unit_tests.hc_fused import hc_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from tests.unit_tests.models.flash_next.cpu_model import cpu_draft_model, cpu_model, prefill_metadata  # noqa: E402
from tests.unit_tests.models.flash_next.test_hc_fused_whole_model import (  # noqa: E402,F401
    _FusedOpCounter, _prefill_all, token_lists)
from vllm_gaudi.models.qwen4_exp import hyperconnection as layer_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402


def test_the_tiny_model_is_bit_identical_with_the_interleaved_kernels(tiny_checkpoint, token_lists, monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(layer_module.logger, "warning", lambda message, *args: lines.append(message % args))
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0")
    torch_out, _, _ = _prefill_all(tiny_checkpoint, token_lists)

    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED_COPY_MODE", "2")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED_IL", "1")
    counters: list = []
    il_out, il_ops, layers = _prefill_all(tiny_checkpoint, token_lists, counters)
    assert lines == []
    assert [name for name, requested, ok, logged in layers if not (requested and ok is True and not logged)] == []
    calls = len(token_lists)
    assert il_ops == {
        "hc_norm": 2 * calls,
        "hc_combine": 1 * calls,
        "hc_combine_norm_il": 7 * calls,
        "hc_silu_split": 8 * calls,
        "hc_silu": 1 * calls,
        "hc_gate_mix_il": 9 * calls,
    }
    assert dict(counters[0].views) == {}, "interleaved ops were given views"
    for got, want, token_ids in zip(il_out, torch_out, token_lists):
        assert got.dtype == want.dtype == torch.bfloat16 and torch.equal(got, want), f"{len(token_ids)} tokens"


def test_the_mtp_draft_model_with_the_interleaved_kernels(tiny_checkpoint, monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(layer_module.logger, "warning", lambda message, *args: lines.append(message % args))
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv("VLLM_HPU_HC_FUSED_COPY_MODE", "2")
    rows = 3
    hidden_states = torch.randn(rows, 512, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    runs = {}
    for il in ("0", "1"):
        monkeypatch.setenv("VLLM_HPU_HC_FUSED_IL", il)
        with cpu_model(tiny_checkpoint, speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                       load_format="dummy") as (vllm_config, _):
            draft = cpu_draft_model(vllm_config)
            with _FusedOpCounter() as counter, set_forward_context(prefill_metadata(rows), vllm_config), \
                    torch.no_grad():
                sample, multi = draft(input_ids=torch.tensor([5, 6, 7]), positions=torch.arange(rows),
                                      hidden_states=hidden_states.clone(), spec_step_idx=0)
            layers = [module._hc_il for module in draft.modules() if isinstance(module, HpuGatedResidual)]
            runs[il] = (sample.clone(), multi.clone(), counter, layers)
    assert lines == []
    _, _, counter, layers = runs["1"]
    assert counter.ops == {"hc_norm": 1, "hc_combine_norm_il": 2, "hc_silu_split": 2, "hc_silu": 1,
                           "hc_gate_mix_il": 3}
    assert layers and all(layers)
    assert dict(counter.views) == {}
    assert runs["0"][2].clones == counter.clones
    assert torch.equal(runs["0"][0], runs["1"][0]) and torch.equal(runs["0"][1], runs["1"][1])
