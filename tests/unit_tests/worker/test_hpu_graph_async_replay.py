# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_GRAPH_ASYNC_REPLAY: whether the model's HPU graph replays asynchronously.

``wrap_in_hpu_graph`` defaults to ``asynchronous=False``: the caller's thread issues every captured recipe and
collective inside ``replayV3`` before the forward returns (bridge 1.24.1, ``graphs.py:635-770, 840``). With the flag
at 1 the runner wraps the model with ``asynchronous=True``: the replay is queued and the forward returns, so the
host can issue the logits and sampler ops while the replay is still being issued. The default keeps today's call.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import pytest  # noqa: E402

from vllm_gaudi import envs as gaudi_envs  # noqa: E402
from vllm_gaudi.v1.worker import hpu_model_runner as runner  # noqa: E402


def test_flag_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_GRAPH_ASYNC_REPLAY", raising=False)
    assert gaudi_envs.VLLM_HPU_GRAPH_ASYNC_REPLAY == 0


def test_default_wraps_as_before(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_GRAPH_ASYNC_REPLAY", raising=False)
    assert runner._hpu_graph_wrap_kwargs() == {"disable_tensor_cache": True}


@pytest.mark.parametrize("value", ["1"])
def test_flag_asks_for_an_asynchronous_replay(monkeypatch, value):
    monkeypatch.setenv("VLLM_HPU_GRAPH_ASYNC_REPLAY", value)
    assert runner._hpu_graph_wrap_kwargs() == {"disable_tensor_cache": True, "asynchronous": True}


def test_model_wrapper_uses_the_kwargs(monkeypatch):
    """_maybe_wrap_in_hpu_graph hands the kwargs to wrap_in_hpu_graph in lazy mode."""
    seen = {}

    def fake_wrap(module, **kwargs):
        seen.update(kwargs)
        return module

    class FakeAdapter:

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setenv("VLLM_HPU_GRAPH_ASYNC_REPLAY", "1")
    monkeypatch.setattr(runner, "HpuModelAdapter", FakeAdapter)
    monkeypatch.setattr(runner.htorch.hpu, "wrap_in_hpu_graph", fake_wrap, raising=False)
    monkeypatch.setattr(runner.htorch.utils.internal, "is_lazy", lambda: True, raising=False)
    runner._maybe_wrap_in_hpu_graph(object(), vllm_config=None)
    assert seen == {"disable_tensor_cache": True, "asynchronous": True}
