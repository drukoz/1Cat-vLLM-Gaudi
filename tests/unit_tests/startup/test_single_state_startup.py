# SPDX-License-Identifier: Apache-2.0
"""CPU test of the worker start-up path on the tiny fixture's single-state PLE cache.

On the 2026-11-18 rental the tiny TP8 server died twice with
``IndexError: tuple index out of range``: at ``vllm_gaudi/v1/worker/hpu_worker.py:952``
(``layer_spec.dtypes[1]`` in ``determine_available_memory``) and at
``vllm_gaudi/extension/defragmentation.py:64`` (``cache[1]`` in
``OnlineDefragmenter.__init__``). The Flash-Next PLE short-conv layer has one
recurrent state, so its ``MambaSpec`` has one dtype and the runner cache is a
one-element tuple. Commit ``f65f39f`` fixed both; this is the test that would
have caught it on the Mac.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import os  # noqa: E402

os.environ["TMPDIR"] = "/tmp"  # the gloo init file must not sit on a sandboxed TMPDIR

from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402

import vllm_gaudi.v1.worker.hpu_worker as hw  # noqa: E402
from vllm_gaudi.extension.defragmentation import OnlineDefragmenter  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import build_tiny_checkpoint, cpu_model  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "models" / "flash_next" / "tiny_tp8_config.json"


@pytest.fixture(scope="module")
def tiny_tp8_checkpoint(tmp_path_factory) -> Path:
    """The tiny TP8 checkpoint, built once per module in a subprocess (no distributed state)."""
    return build_tiny_checkpoint(tmp_path_factory.mktemp("tiny_tp8"), fixture=FIXTURE)


@pytest.fixture(scope="module")
def model_specs(tiny_tp8_checkpoint):
    """The tiny TP8 ``vllm_config`` and its ``MambaBase`` KV-cache specs, opened once per module."""
    with cpu_model(tiny_tp8_checkpoint) as (vllm_config, model):
        specs = {
            name: module.get_kv_cache_spec(vllm_config)
            for name, module in model.named_modules()
            if isinstance(module, MambaBase)
        }
        yield vllm_config, specs


def ple_name(specs: dict) -> str:
    """The one spec name of the PLE short-conv layer."""
    names = [name for name in specs if name.endswith(".ple")]
    assert len(names) == 1, names
    return names[0]


def test_fixture_has_a_single_state_spec(model_specs):
    """The fixture has exactly one single-state (PLE) spec next to two-state (GDN) specs."""
    _, specs = model_specs
    spec = specs[ple_name(specs)]
    assert len(spec.dtypes) == 1
    assert len(spec.shapes) == 1
    assert any(len(specs[name].dtypes) == 2 for name in specs if name.endswith(".linear_attn"))


def test_determine_available_memory_binds_every_recurrent_layer(model_specs):
    """The worker profiles the single-state PLE spec and binds a 4-tuple for every layer."""
    vllm_config, specs = model_specs
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):  # determine_available_memory allocates with device="hpu"
        if kwargs.get("device") == "hpu":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)

    worker = object.__new__(hw.HPUWorker)  # no __init__: no device, no distributed setup
    worker.model_runner = MagicMock()
    worker.model_runner.get_kv_cache_spec.return_value = specs
    worker.model_runner.uses_framework_kv_cache_layout.return_value = False
    worker.model_runner.profile_kv_cache_blocks.return_value = 1
    worker.vllm_config = vllm_config
    worker.model_config = vllm_config.model_config
    bound = {}
    with patch.object(hw, "bind_kv_cache", lambda caches, ctx, out: bound.update(caches)), \
         patch.object(hw, "is_fake_hpu", lambda: True), \
         patch.object(torch, "tensor", cpu_tensor):
        available = worker.determine_available_memory()  # 4 GiB: the fake-HPU early return after binding
    assert available == 4 * 2**30
    assert set(bound) == set(specs)
    for cache in bound.values():
        assert isinstance(cache, tuple) and len(cache) == 4
        assert all(isinstance(state, torch.Tensor) for state in cache)
    ple = specs[ple_name(specs)]
    for state in bound[ple_name(specs)]:
        assert state.dtype == ple.dtypes[0]


def test_online_defragmenter_accepts_runner_shaped_caches(model_specs):
    """The defragmenter accepts the runner's caches: a 1-tuple (PLE) among 2-tuples (GDN)."""
    _, specs = model_specs
    caches = [
        tuple(torch.zeros(4, *shape, dtype=dtype) for shape, dtype in zip(spec.shapes, spec.dtypes))
        for spec in specs.values()
    ]  # the runner keeps one tuple per layer, one tensor per state
    ple = caches[list(specs).index(ple_name(specs))]
    assert len(caches[0]) == 2  # model order: the GDN 2-tuple is first
    assert len(ple) == 1
    conf = MagicMock(defrag=True, VLLM_DEFRAG_THRESHOLD=32, VLLM_DEFRAG_WITH_GRAPHS=False, bridge_mode="eager")
    with patch("vllm_gaudi.extension.defragmentation.get_config", return_value=conf):
        assert OnlineDefragmenter(caches, 128).is_mla is False
        # The PLE 1-tuple first covers ``kv_caches[0][0].device`` on a one-element tuple.
        reordered = [ple] + [cache for cache in caches if cache is not ple]
        assert OnlineDefragmenter(reordered, 128).is_mla is False
