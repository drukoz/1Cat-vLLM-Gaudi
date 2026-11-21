# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU parity tests for HpuGatedResidual (RUK-2743).

The pure-torch HPU variant must match upstream's own torch reference
(``vllm/models/qwen4_exp/common/hyperconnection.py::GatedResidual``) with
unmerged weights, given the documented weight correspondence:
NVIDIA merged rows ``[0:hc_lowrank]`` = common ``input_mix_weight_down``,
rows ``[hc_lowrank:hc_lowrank+hc_count]`` = common ``block_inject_weight``,
remaining pad rows zero; ``input_mix_weight_up``/``hc_norm`` identical.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    init_distributed_environment, initialize_model_parallel,
)
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel,
)
from vllm.models.qwen4_exp.common.hyperconnection import (  # noqa: E402
    GatedResidual, HyperConnectionConfig,
)

from vllm_gaudi.models.qwen4_exp.hyperconnection import (  # noqa: E402
    HpuGatedResidual, )

# (hidden_size, hc_lowrank): pad_size = (-(lora+hc)) % 16 -> 12 and 8.
SHAPES = [(128, 16), (64, 20)]


@pytest.fixture(scope="module")
def world1():
    """World-1 gloo group with a current VllmConfig (S3 pattern)."""
    _, temp_file = tempfile.mkstemp(dir="/tmp")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.fixture
def vllm_ctx():
    """vLLM linear layers read get_current_vllm_config() at construction."""
    with set_current_vllm_config(VllmConfig()):
        yield


def _config(hidden_size: int, hc_lowrank: int) -> HyperConnectionConfig:
    return HyperConnectionConfig(
        hc_count=4,
        hidden_size=hidden_size,
        params_dtype=torch.float32,
        hc_lowrank=hc_lowrank,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )


def _load_full(layer: torch.nn.Module, tensor: torch.Tensor, shard_id=None) -> None:
    param = layer.weight
    if shard_id is None:
        param.weight_loader(param, tensor)
    else:
        param.weight_loader(param, tensor, shard_id)
    layer.quant_method.process_weights_after_loading(layer)


def _build_pair(config: HyperConnectionConfig, use_combine: bool = True):
    """Seeded common module and an HpuGatedResidual loaded with its weights."""
    torch.manual_seed(0)
    common = GatedResidual(config)
    hpu = HpuGatedResidual(config, use_combine=use_combine)
    hpu.hc_norm.weight.data.copy_(common.hc_norm.weight.data)

    if use_combine:
        merged = hpu.input_mix_weight_down_block_inject
        _load_full(merged, common.input_mix_weight_down.weight.data, 0)
        _load_full(merged, common.block_inject_weight.weight.data, 1)
        if hpu.pad_size:
            pad = torch.zeros(hpu.pad_size, hpu.hyper_hidden_size, dtype=config.params_dtype)
            _load_full(merged, pad, 2)
    else:
        _load_full(hpu.input_mix_weight_down, common.input_mix_weight_down.weight.data)
    _load_full(hpu.input_mix_weight_up, common.input_mix_weight_up.weight.data)
    return hpu, common


def _inputs(config: HyperConnectionConfig, tokens: int = 5):
    torch.manual_seed(7)
    hyper = config.hc_count * config.hidden_size
    hidden = torch.randn(tokens, hyper, dtype=torch.float32)
    block = torch.randn(tokens, config.hidden_size, dtype=torch.float32)
    return hidden, block


@pytest.mark.parametrize("hidden_size,hc_lowrank", SHAPES)
def test_mix_matches_common_reference(world1, vllm_ctx, hidden_size, hc_lowrank):
    config = _config(hidden_size, hc_lowrank)
    hpu, common = _build_pair(config)
    hidden, _ = _inputs(config)

    # NVIDIA constructor weight shapes, incl. the 16-row merged padding.
    rows = config.hc_lowrank + config.hc_count + hpu.pad_size
    assert rows % 16 == 0
    assert hpu.input_mix_weight_down_block_inject.weight.shape == (rows, hpu.hyper_hidden_size)
    assert hpu.input_mix_weight_up.weight.shape == (hpu.hyper_hidden_size, config.hc_lowrank)
    assert hpu.hc_norm.weight.shape == (hpu.hyper_hidden_size, )

    passed, block_input, injection = hpu.mix(hidden)
    assert passed is hidden
    assert injection is not None

    mixed, (_, xn) = common.mix(hidden)
    torch.testing.assert_close(block_input, mixed, atol=1e-5, rtol=1e-5)

    # The combine kernel's weight formula on the HPU logits must match
    # common's 2 * sigmoid(inject(xn) / hc_count).
    inj_weights = 2.0 * torch.sigmoid(injection / config.hc_count)
    ref_weights = 2.0 * torch.sigmoid(F.linear(xn, common.block_inject_weight.weight) / config.hc_count)
    torch.testing.assert_close(inj_weights, ref_weights, atol=1e-5, rtol=1e-5)


def test_combine_matches_common_reference(world1, vllm_ctx):
    config = _config(128, 16)
    hpu, common = _build_pair(config)
    hidden, block = _inputs(config)

    _, (_, xn) = common.mix(hidden)
    injection = F.linear(xn, common.block_inject_weight.weight)

    out_hpu = hpu.combine(hidden, block, injection)
    out_common = common.combine(block, (hidden, xn))
    torch.testing.assert_close(out_hpu, out_common, atol=1e-5, rtol=1e-5)


def test_combine_and_mix_is_combine_then_mix(world1, vllm_ctx):
    config = _config(128, 16)
    hpu, _ = _build_pair(config)
    hidden, block = _inputs(config)
    _, _, injection = hpu.mix(hidden)

    h_cm, bi_cm, inj_cm = hpu.combine_and_mix(hidden, block, injection)
    h_seq = hpu.combine(hidden, block, injection)
    _, bi_seq, inj_seq = hpu.mix(h_seq)

    torch.testing.assert_close(h_cm, h_seq, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bi_cm, bi_seq, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(inj_cm, inj_seq, atol=1e-6, rtol=1e-6)


def test_3d_input_matches_2d(world1, vllm_ctx):
    config = _config(128, 16)
    hpu, _ = _build_pair(config)
    torch.manual_seed(11)
    x3 = torch.randn(2, 3, config.hc_count * config.hidden_size, dtype=torch.float32)
    b3 = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)
    x2, b2 = x3.reshape(6, -1), b3.reshape(6, -1)

    _, bi3, inj3 = hpu.mix(x3)
    _, bi2, inj2 = hpu.mix(x2)
    torch.testing.assert_close(bi3, bi2.reshape(2, 3, -1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(inj3, inj2.reshape(2, 3, -1), atol=1e-6, rtol=1e-6)

    h_cm3, bi_cm3, inj_cm3 = hpu.combine_and_mix(x3, b3, inj3)
    h_cm2, bi_cm2, inj_cm2 = hpu.combine_and_mix(x2, b2, inj2)
    torch.testing.assert_close(h_cm3, h_cm2.reshape(2, 3, -1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bi_cm3, bi_cm2.reshape(2, 3, -1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(inj_cm3, inj_cm2.reshape(2, 3, -1), atol=1e-6, rtol=1e-6)


def test_no_mix_variant(world1, vllm_ctx):
    config = _config(128, 16)
    hpu, common = _build_pair(config, use_combine=False)

    assert hasattr(hpu, "input_mix_weight_down")
    assert not hasattr(hpu, "input_mix_weight_down_block_inject")
    assert hpu.pad_size == 0

    hidden, _ = _inputs(config)
    _, block_input, injection = hpu.mix(hidden)
    assert injection is None

    mixed, _ = common.mix(hidden)
    torch.testing.assert_close(block_input, mixed, atol=1e-5, rtol=1e-5)
