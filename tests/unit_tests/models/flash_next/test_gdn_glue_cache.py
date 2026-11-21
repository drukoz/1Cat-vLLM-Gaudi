# SPDX-License-Identifier: Apache-2.0
"""``VLLM_HPU_GDN_GLUE_CACHE`` (idea I25): the fused GDN decode op's glue without the per-step copies.

With the flag at 1 a GDN layer hands ``gdn_core_decode`` the transposed conv weight and dt_bias in fp32 from
buffers its weight-loader hooks filled at load, the owned int32 state indices made once per step and cache
group, and makes no output buffer that the op's output replaces. The op gets the same values, so on the
CPU tiny model the decode rows and both state pools must be the bits of the per-step glue, with fewer ops
per step. One model per test (built with the flag at 1, so the hooks exist); the per-step glue is the same
layers with ``gdn_glue_cache`` switched off.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.envs as gaudi_envs  # noqa: E402
from flashinfer_gaudi.gdn_core_decode import GdnCoreDecodeMode  # noqa: E402
from tests.unit_tests.models.flash_next.cpu_model import cpu_model, eos_token_id  # noqa: E402
from tests.unit_tests.models.flash_next.moe_block import OpRecorder  # noqa: E402
from tests.unit_tests.models.flash_next.test_decode_parity import decode, install_cached_attention, prefill  # noqa: E402
from tests.unit_tests.models.flash_next.test_gdn_core_decode_layer import (GDN_LAYERS, PREFILL_LEN, SLOT,  # noqa: E402
                                                                          gdn_layers, run, set_mode, tokens)

FLAG = "VLLM_HPU_GDN_GLUE_CACHE"


def test_flag_default_value_and_documentation(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert gaudi_envs.VLLM_HPU_GDN_GLUE_CACHE == 0
    monkeypatch.setenv(FLAG, "1")
    assert gaudi_envs.VLLM_HPU_GDN_GLUE_CACHE == 1
    root = Path(gaudi_envs.__file__).resolve().parents[1]
    assert f"    {FLAG}: int = 0" in Path(gaudi_envs.__file__).read_text()
    assert f"`{FLAG}`" in (root / "docs" / "configuration" / "env_variables.md").read_text()


def _states(layers):
    return [module.kv_cache[k].clone() for module in layers for k in range(2)]


def _ops_of_one_decode_step(model, vllm_config, eos, atts):
    seq = tokens()
    for att in atts:
        att.reset()
    from tests.unit_tests.models.flash_next.cpu_model import reset_cpu_states
    reset_cpu_states(model)
    prefill(model, vllm_config, seq[:PREFILL_LEN], SLOT, eos)
    with OpRecorder() as recorder:
        decode(model, vllm_config, seq, PREFILL_LEN, SLOT, eos)
    return recorder.names


@pytest.mark.parametrize("fp32", [True, False], ids=["fp32", "bf16"])
def test_cached_glue_gives_the_fused_op_the_same_bits_with_fewer_ops(tiny_checkpoint, monkeypatch, fp32):
    monkeypatch.setenv(FLAG, "1")
    with cpu_model(tiny_checkpoint, fp32=fp32) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        layers = gdn_layers(model)
        assert len(layers) == GDN_LAYERS and all(module.gdn_glue_cache for module in layers)
        # the constants were made at load, by the weight-loader hooks, and are what the per-step glue makes
        for module in layers:
            assert module._triton_conv_weight_ready and module._triton_dt_bias_ready
            weight = module.conv1d.weight
            assert torch.equal(module._triton_conv_weight_t, weight.view(weight.size(0), weight.size(2)).t())
            assert module._triton_dt_bias_f32.dtype == torch.float32
            assert torch.equal(module._triton_dt_bias_f32, module.dt_bias.float())
        set_mode(model, GdnCoreDecodeMode.FUSED)

        for module in layers:
            module.gdn_glue_cache = False
        per_step_rows = run(model, vllm_config, eos, atts)
        per_step_states = _states(layers)
        per_step_ops = _ops_of_one_decode_step(model, vllm_config, eos, atts)

        for module in layers:
            module.gdn_glue_cache = True
        cached_rows = run(model, vllm_config, eos, atts)
        cached_states = _states(layers)
        cached_ops = _ops_of_one_decode_step(model, vllm_config, eos, atts)

    assert torch.equal(cached_rows, per_step_rows), float((cached_rows - per_step_rows).abs().max())
    for got, want in zip(cached_states, per_step_states):
        assert torch.equal(got, want)
    assert per_step_ops.count("flashinfer_gaudi.gdn_core_decode.default") == GDN_LAYERS
    assert cached_ops.count("flashinfer_gaudi.gdn_core_decode.default") == GDN_LAYERS

    def count(names, op):
        return names.count(op)

    print(f"\nops per decode step: per-step glue {len(per_step_ops)}, cached {len(cached_ops)}")
    # per GDN layer: the conv weight's transpose and its copy, the dt_bias cast and the dead output buffer go;
    # the owned indices are made once for the step (one cache group here) instead of once per layer
    assert count(per_step_ops, "aten.t.default") - count(cached_ops, "aten.t.default") == GDN_LAYERS
    assert count(per_step_ops, "aten.zeros.default") - count(cached_ops, "aten.zeros.default") == GDN_LAYERS
    assert count(per_step_ops, "aten.clone.default") - count(cached_ops, "aten.clone.default") >= GDN_LAYERS + 2 * (
        GDN_LAYERS - 1)
    assert len(per_step_ops) - len(cached_ops) >= 4 * GDN_LAYERS


def test_without_the_flag_no_buffer_and_no_hook(tiny_checkpoint, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VLLM_HPU_TRITON_MODE", raising=False)
    with cpu_model(tiny_checkpoint, fp32=True) as (_, model):
        for module in gdn_layers(model):
            assert module.gdn_glue_cache is False
            assert not hasattr(module, "_triton_conv_weight_t") and not hasattr(module, "_triton_dt_bias_f32")
