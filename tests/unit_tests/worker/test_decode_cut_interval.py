# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_DECODE_CUT_INTERVAL: the per-layer mark_step of ``modify_model_layers`` in decode forwards.

At load the runner hooks a ``mark_step()`` after every ``VLLM_CONFIG_HIDDEN_LAYERS``-th ``*DecoderLayer``
(``hpu_model_runner.modify_model_layers``). Inside a replayed HPU graph every such cut is one more recipe and one
more device hand-off per decode step (48 for Flash-Next). With the flag at its default 0 the hooks are the ones the
runner always registered. With N >= 1 a forward whose attention metadata says ``is_prompt=False`` (decode and
speculative verification) cuts only after every N-th layer, while a prompt forward keeps the
``VLLM_CONFIG_HIDDEN_LAYERS`` cuts. The hook runs while a graph is captured (or on a step outside graphs), so the
choice is made once per captured decode graph.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.config import VllmConfig  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from vllm_gaudi import envs as gaudi_envs  # noqa: E402
from vllm_gaudi.v1.worker import hpu_model_runner as runner  # noqa: E402

LAYERS = 6


class ToyDecoderLayer(torch.nn.Module):

    def forward(self, x):
        return x + 1


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(ToyDecoderLayer() for _ in range(LAYERS))
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


@pytest.fixture
def cuts(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.htorch.core, "mark_step", lambda: calls.append(1), raising=False)
    return calls


def _run(model, is_prompt):
    md = None if is_prompt is None else SimpleNamespace(is_prompt=is_prompt)
    with torch.no_grad(), set_forward_context(md, VllmConfig()):
        model(torch.zeros(1, 4))


def _hooked(n, decode_n):
    model = ToyModel()
    runner.modify_model_layers(model, ["DecoderLayer"], n, decode_n=decode_n)
    return model


def test_flag_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_DECODE_CUT_INTERVAL", raising=False)
    assert gaudi_envs.VLLM_HPU_DECODE_CUT_INTERVAL == 0
    monkeypatch.setenv("VLLM_HPU_DECODE_CUT_INTERVAL", "1000")
    assert gaudi_envs.VLLM_HPU_DECODE_CUT_INTERVAL == 1000


@pytest.mark.parametrize("is_prompt", [True, False, None])
def test_default_cuts_after_every_layer_in_every_forward(cuts, is_prompt):
    model = _hooked(1, 0)
    _run(model, is_prompt)
    assert len(cuts) == LAYERS


def test_default_registers_the_same_hooks_as_before():
    """decode_n=0 leaves the hook set unchanged: one hook on every n-th layer, none elsewhere."""
    model = _hooked(2, 0)
    hooked = [len(layer._forward_hooks) for layer in model.layers]
    assert hooked == [0, 1, 0, 1, 0, 1]


def test_decode_forward_without_layer_cuts(cuts):
    model = _hooked(1, 1000)
    _run(model, False)
    assert len(cuts) == 0


def test_prompt_forward_keeps_the_hidden_layers_cuts(cuts):
    model = _hooked(1, 1000)
    _run(model, True)
    assert len(cuts) == LAYERS
    cuts.clear()
    model = _hooked(3, 1000)
    _run(model, True)
    assert len(cuts) == LAYERS // 3


def test_decode_forward_cuts_every_nth_layer(cuts):
    model = _hooked(1, 2)
    _run(model, False)
    assert len(cuts) == LAYERS // 2


def test_forward_without_metadata_counts_as_prompt(cuts):
    """No forward context (a bare module call) or metadata without is_prompt keeps the prompt cuts."""
    model = _hooked(1, 1000)
    model(torch.zeros(1, 4))
    assert len(cuts) == LAYERS
    cuts.clear()
    _run(model, None)
    assert len(cuts) == LAYERS


def test_per_layer_dict_metadata_counts_as_prompt(cuts):
    model = _hooked(1, 1000)
    with torch.no_grad(), set_forward_context({"layer.0": SimpleNamespace(is_prompt=False)}, VllmConfig()):
        model(torch.zeros(1, 4))
    assert len(cuts) == LAYERS


def test_load_model_passes_the_flag_to_the_hook_installer():
    """The call site in load_model hands the flag over (source check: load_model needs a device to run)."""
    import inspect
    source = inspect.getsource(runner.HPUModelRunner.load_model)
    assert "decode_n=gaudi_envs.VLLM_HPU_DECODE_CUT_INTERVAL" in source


def test_the_interval_of_each_kind_of_forward_is_logged_once(cuts, monkeypatch):
    """The server log names the interval a decode and a prompt forward got, once per process and kind."""
    lines = []
    monkeypatch.setattr(runner, "_DECODE_CUT_LOGGED", set())
    monkeypatch.setattr(runner.logger, "info", lambda msg, *args: lines.append(msg % args))
    model = _hooked(1, 1000)
    _run(model, False)
    _run(model, False)
    _run(model, True)
    assert lines == [
        "VLLM_HPU_DECODE_CUT_INTERVAL=1000: a decode forward cuts after every 1000-th decoder layer",
        "VLLM_HPU_DECODE_CUT_INTERVAL=1000: a prompt forward cuts after every 1-th decoder layer",
    ]


def test_nothing_is_logged_with_the_flag_at_zero(cuts, monkeypatch):
    lines = []
    monkeypatch.setattr(runner, "_DECODE_CUT_LOGGED", set())
    monkeypatch.setattr(runner.logger, "info", lambda msg, *args: lines.append(msg % args))
    model = _hooked(1, 0)
    _run(model, False)
    assert lines == []
