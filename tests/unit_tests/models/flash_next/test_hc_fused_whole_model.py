# SPDX-License-Identifier: Apache-2.0
"""The whole tiny Flash-Next model on CPU with ``VLLM_HPU_HC_FUSED=1`` against the same model with 0.

On CPU the fused hyper-connection ops run the torch oracle (the stand-in of
``tests/unit_tests/hc_fused/hc_ops.py``), so the two builds must agree bit for bit: this proves
that the real model feeds every hyper-connection an input the fused path takes (2-D, bfloat16,
contiguous, the widths the kernels accept), that no layer falls back, and that the wiring is right
over the chain of all layers, the PLE layer's standalone combine and the final mixer.  The tiny
checkpoint has hidden size 128 and hc_count 4, the geometry of the ``hc`` smoke mode.

One ``cpu_model`` may be open at a time, so each test builds, runs and closes one model after the other.
"""

import collections
import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from tests.unit_tests.hc_fused import hc_ops  # noqa: E402,F401  the CPU stand-in: the oracle as the CPU kernels
from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, dense_reference_prompts, eos_token_id, ple_kwargs, prefill_metadata, reset_cpu_states)
from vllm_gaudi.models.qwen4_exp import hyperconnection as layer_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual  # noqa: E402

FUSED_PREFIX = "flashinfer_gaudi::hc_"


class _FusedOpCounter(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if name.startswith(FUSED_PREFIX):
            self.ops[name[len("flashinfer_gaudi::"):]] += 1
        return func(*args, **(kwargs or {}))


def _prefill_all(tiny_checkpoint, token_lists) -> tuple[list, collections.Counter, list]:
    """Build the bf16 model with the environment as it is now, prefill every token list, close the model.

    Returns the hidden states of each prefill, the fused ops dispatched, and the hyper-connection layers'
    (name, requested, check result, logged) records.
    """
    with cpu_model(tiny_checkpoint) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        outputs = []
        with _FusedOpCounter() as counter:
            for token_ids in token_lists:
                num_tokens = len(token_ids)
                reset_cpu_states(model)
                with torch.no_grad(), set_forward_context(prefill_metadata(num_tokens, 1), vllm_config):
                    hidden = model(input_ids=torch.tensor(token_ids),
                                   positions=torch.arange(num_tokens),
                                   **ple_kwargs(num_tokens, eos))
                outputs.append(hidden.clone())
        layers = [(module.layer_name, module.hc_fused_requested, module._hc_fused_ok, module._hc_fused_logged)
                  for module in model.modules() if isinstance(module, HpuGatedResidual)]
    return outputs, counter.ops, layers


@pytest.fixture
def token_lists(tiny_checkpoint):
    reference = json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))
    prompts = [list(prompt["prompt_ids"]) for prompt in dense_reference_prompts(reference)]
    return [prompts[0][:1]] + prompts  # one token (a decode-sized call), then the reference prompts


def test_the_tiny_model_is_bit_identical_with_the_fused_hyper_connections(tiny_checkpoint, token_lists, monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(layer_module.logger, "warning", lambda message, *args: lines.append(message % args))

    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0")
    torch_out, torch_ops, torch_layers = _prefill_all(tiny_checkpoint, token_lists)
    assert sum(torch_ops.values()) == 0
    assert len(torch_layers) == 9  # 4 decoder layers x (attention, MLP) and the final mixer
    assert all(requested is False for _, requested, _, _ in torch_layers)

    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    fused_out, fused_ops, fused_layers = _prefill_all(tiny_checkpoint, token_lists)

    # every hyper-connection ran fused, and none wrote its "is off" line
    assert lines == []
    assert [name for name, requested, ok, logged in fused_layers if not (requested and ok is True and not logged)] == []
    # per forward: layer 0 mixes (hc_norm), the PLE layer combines on its own (hc_combine) and mixes
    # (hc_norm), every other hyper-connection is a combine_and_mix (hc_combine_norm): 7 of them with
    # the final mixer, which is the one hc_silu; 9 gated mixes.
    calls = len(token_lists)
    assert fused_ops == {
        "hc_norm": 2 * calls,
        "hc_combine": 1 * calls,
        "hc_combine_norm": 7 * calls,
        "hc_silu_split": 8 * calls,
        "hc_silu": 1 * calls,
        "hc_gate_mix": 9 * calls,
    }
    assert len(fused_out) == len(torch_out) == calls
    for fused, plain, token_ids in zip(fused_out, torch_out, token_lists):
        assert fused.dtype == plain.dtype == torch.bfloat16 and fused.shape == plain.shape
        assert torch.equal(fused, plain), f"prefill of {len(token_ids)} tokens differs"


def test_the_layers_carry_their_checkpoint_names(tiny_checkpoint, token_lists, monkeypatch):
    """The name in the log line is the layer's prefix in the model, so a fallback line names the layer."""
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    _, _, layers = _prefill_all(tiny_checkpoint, token_lists[:1])
    names = [name for name, _, _, _ in layers]
    assert len(set(names)) == 9
    assert any(name.endswith("layers.0.attn_hyper_connection") for name in names)
    assert any(name.endswith("layers.3.mlp_hyper_connection") for name in names)
    assert any(name.endswith("hyper_connection_mixer") for name in names)
