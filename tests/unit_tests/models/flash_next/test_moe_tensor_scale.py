# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_MOE_TENSOR_SCALE_MAX_TOKENS: one per-tensor activation scale for short steps of the vendor MoE op.

The vendor op (``torch.ops.hpu.mixture_of_experts``) validates a per-row activation scale on the host: 4 extra graph
launches per MoE layer at any row count (qsa-fused job 00217: isfinite / nonzero, abs / ceil / min / max / ne, gt,
lt, each read back); with one per-tensor scale it runs in one graph. These tests check which steps get the
per-tensor scale and state the numeric change against per-row scales at 1, 4 and 16 rows on the CPU stand-in
(``moe_block``: the vendor op in fp32 over the FP8 inputs as given), for three activation regimes: rows of one
magnitude, rows whose magnitudes spread over 0.1x to 10x, and one outlier row at 100x.
"""

from tests.unit_tests.models.flash_next import moe_block  # noqa: F401  (CPU stubs first)

import math

import pytest
import torch

import vllm_gaudi.extension.ops as plugin_ops

H, I, LOCAL, GLOBAL, TOPK = 64, 32, 8, 32, 4
EXPERTS_MIN = 8


@pytest.fixture(autouse=True)
def _vendor_ops():
    moe_block.install_vendor_ops(fp8_max=float(plugin_ops.FP8_MAX))
    moe_block.VENDOR_CALLS.clear()


def _op(seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(LOCAL, 2 * I, H, generator=g) * 0.05
    w2 = torch.randn(LOCAL, H, I, generator=g) * 0.05
    s13 = (w13.abs().amax(dim=2) + 1e-8) / float(plugin_ops.FP8_MAX)
    s2 = (w2.abs().amax(dim=2) + 1e-8) / float(plugin_ops.FP8_MAX)
    op = plugin_ops.VllmMixtureOfExpertsOpFP8PerChannel(GLOBAL, LOCAL, EXPERTS_MIN, EXPERTS_MIN + LOCAL - 1, None)
    for e in range(LOCAL):
        op.w13_list[e].set_weight((w13[e] / s13[e].unsqueeze(-1)).to(torch.float8_e4m3fn))
        op.w13_list[e].set_scale_inv_fp8(s13[e])
        op.w2_list[e].set_weight((w2[e] / s2[e].unsqueeze(-1)).to(torch.float8_e4m3fn))
        op.w2_list[e].set_scale_inv_fp8(s2[e])
    op._cache_weight_lists()
    return op


def _batch(rows, regime, seed=1):
    g = torch.Generator().manual_seed(seed + rows)
    x = torch.randn(rows, H, generator=g)
    if regime == "spread":  # per-row magnitudes log-uniform over 0.1x .. 10x
        x = x * torch.exp(torch.empty(rows, 1).uniform_(math.log(0.1), math.log(10.0), generator=g))
    elif regime == "outlier" and rows > 1:  # one row 100x the others
        x[0] = x[0] * 100.0
    ids = torch.stack([torch.arange(EXPERTS_MIN, EXPERTS_MIN + LOCAL)[torch.randperm(LOCAL, generator=g)[:TOPK]]
                       for _ in range(rows)])  # every pair local: the whole output depends on the scale
    weights = torch.softmax(torch.randn(rows, TOPK, generator=g), dim=-1).to(torch.bfloat16)
    return x.to(torch.bfloat16), ids.to(torch.int64), weights


def _run(monkeypatch, op, batch, max_tokens):
    monkeypatch.setenv("VLLM_HPU_MOE_TENSOR_SCALE_MAX_TOKENS", str(max_tokens))
    moe_block.VENDOR_CALLS.clear()
    out = op(*batch).float()
    return out, moe_block.VENDOR_CALLS[-1]["d_scale_hidden_states"]


def test_default_keeps_per_row_scales(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_MOE_TENSOR_SCALE_MAX_TOKENS", raising=False)
    from vllm_gaudi import envs
    assert envs.VLLM_HPU_MOE_TENSOR_SCALE_MAX_TOKENS == 0
    op = _op()
    for rows in (1, 4, 16, 64):
        moe_block.VENDOR_CALLS.clear()
        op(*_batch(rows, "plain"))
        assert tuple(moe_block.VENDOR_CALLS[-1]["d_scale_hidden_states"].shape) == (rows, 1)


@pytest.mark.parametrize("rows,per_tensor", [(1, True), (4, True), (16, True), (17, False), (128, False)])
def test_steps_up_to_the_limit_take_one_scale(monkeypatch, rows, per_tensor):
    _, scale = _run(monkeypatch, _op(), _batch(rows, "plain"), 16)
    assert (scale.numel() == 1) == per_tensor
    if not per_tensor:
        assert tuple(scale.shape) == (rows, 1)


def test_one_row_is_unchanged(monkeypatch):
    op, batch = _op(), _batch(1, "plain")
    row, _ = _run(monkeypatch, op, batch, 0)
    tensor, _ = _run(monkeypatch, op, batch, 16)
    assert torch.equal(row, tensor)  # one row: the per-tensor scale is the row's scale


NUMBERS = {}


@pytest.mark.parametrize("regime", ["plain", "spread", "outlier"])
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_numeric_change_against_per_row_scales(monkeypatch, rows, regime):
    """The size of the change: |out(one scale) - out(per-row)| relative to each row's own output, max and mean over
    rows; and the same for per-row FP8 against the unquantized fp32 MoE, the error the per-row form already has."""
    op, batch = _op(), _batch(rows, regime)
    row, _ = _run(monkeypatch, op, batch, 0)
    tensor, _ = _run(monkeypatch, op, batch, 16)
    x, ids, weights = batch
    w12 = [w.float() * s.float().reshape(-1, 1) for w, s in zip(op._cached_w13_views, op._cached_w13_scale_views)]
    w3 = [w.float() * s.float().reshape(-1, 1) for w, s in zip(op._cached_w2_views, op._cached_w2_scale_views)]
    exact = moe_block._mixture_of_experts(hidden_states=x.float(), expert_routing_table=ids,
                                          router_weights=weights.float(), w12=w12, w3=w3, permuted_weights=True,
                                          activation="silu", experts_min=EXPERTS_MIN,
                                          experts_max=EXPERTS_MIN + LOCAL - 1).float()

    def rel(a, b):
        per_row = (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)
        return float(per_row.max()), float(per_row.mean())

    t_max, t_mean = rel(tensor, row)
    q_max, q_mean = rel(row, exact)
    qt_max, qt_mean = rel(tensor, exact)
    NUMBERS[(regime, rows)] = (t_max, t_mean, q_max, q_mean, qt_max, qt_mean)
    print("MOE_TENSOR_SCALE %-7s rows %2d: one scale vs per-row max %.4f mean %.4f | per-row vs fp32 max %.4f mean "
          "%.4f | one scale vs fp32 max %.4f mean %.4f" % (regime, rows, t_max, t_mean, q_max, q_mean, qt_max,
                                                            qt_mean))
    if rows == 1:
        assert t_max == 0.0
    # bounds that hold on this stand-in (FP8 E4M3 keeps a relative precision while values stay normal numbers)
    assert qt_mean < 0.2 and t_mean < 0.2
