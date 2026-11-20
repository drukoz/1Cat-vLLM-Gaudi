# SPDX-License-Identifier: Apache-2.0
"""The FP8 PLE table gets the Gaudi range conversion at load.

Gaudi2's FP8 goes up to 240, the checkpoint's E4M3FN to 448. On a card (rental 2026-11-20) the
conversion of an FP8 tensor to bfloat16 turned every code above 240 into inf or NaN (256 -> inf,
288 and 448 -> nan), the PLE table is read exactly that way (``embeddings.to(dtype) * weight_scale``)
and the first real-model server answered token 0 for every position. The FP8 linears and experts are
halved at load with their scale doubled (``gaudi_weight_wrapper``); the table was not.
"""

from types import SimpleNamespace

import torch

from vllm_gaudi.models.qwen4_exp.ple_layer import adapt_fp8_table_for_gaudi


class _Param:
    """A parameter stand-in with vLLM's ``weight_loader`` attribute."""

    def __init__(self, data):
        self.data = data
        self.dtype = data.dtype
        self.weight_loader = self._load

    def _load(self, param, loaded_weight):
        param.data = loaded_weight


def _gaudi_weight_wrapper(weight_loader):
    """What vllm_gaudi.extension.ops.gaudi_weight_wrapper does (that module needs the Habana runtime)."""

    def wrapper(param, loaded_weight):
        if loaded_weight.dtype == torch.float8_e4m3fn:
            loaded_weight = (loaded_weight.float() * 0.5).to(torch.float8_e4m3fn)
        else:
            loaded_weight = loaded_weight.data * 2.0
        return weight_loader(param, loaded_weight)

    return wrapper


def _table():
    values = torch.tensor([[0.5, 17.0, 240.0, 256.0], [448.0, -448.0, -300.0, 1.0]])
    return values.to(torch.float8_e4m3fn)


def test_an_fp8_table_is_halved_and_its_scale_doubled_at_load():
    embedding = SimpleNamespace(weight=_Param(torch.empty(2, 4, dtype=torch.float8_e4m3fn)),
                                weight_scale=_Param(torch.empty(1, dtype=torch.float32)))
    assert adapt_fp8_table_for_gaudi(embedding, wrapper=_gaudi_weight_wrapper) is True

    checkpoint_table, checkpoint_scale = _table(), torch.tensor([0.0123])
    embedding.weight.weight_loader(embedding.weight, checkpoint_table)
    embedding.weight_scale.weight_loader(embedding.weight_scale, checkpoint_scale)

    loaded = embedding.weight.data.float()
    assert float(loaded.abs().max()) <= 240.0, "a code above 240 becomes inf or NaN on Gaudi2"
    # the values the model sees are the checkpoint's: (w / 2) * (2 * s) == w * s
    assert torch.equal(loaded * embedding.weight_scale.data, checkpoint_table.float() * checkpoint_scale)


def test_a_bfloat16_table_is_left_alone():
    weight = _Param(torch.empty(2, 4, dtype=torch.bfloat16))
    loader = weight.weight_loader
    embedding = SimpleNamespace(weight=weight)
    assert adapt_fp8_table_for_gaudi(embedding, wrapper=_gaudi_weight_wrapper) is False
    assert weight.weight_loader == loader
