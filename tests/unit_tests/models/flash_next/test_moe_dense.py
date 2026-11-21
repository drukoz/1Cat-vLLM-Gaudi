# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_MOE_DENSE: the per-channel FP8 MoE op's dense batched form against the vendor op's contract.

``moe_dense_fp8`` runs every local expert on every row as two batched FP8 GEMMs and weights each (row, expert)
pair by its routing weight (0 when not routed). On a CPU the vendor op is the tests' stand-in
(``moe_block.install_vendor_ops``: the sum over the routed local experts of ``weight * down(silu(gate) * up)``
in fp32) and ``fp8_gemm_v2`` a batched fp32 matmul. The dense form quantizes the intermediate to FP8 per
(expert, row) where the stand-in keeps fp32, so the two agree to FP8 rounding, not bit for bit.
"""

from tests.unit_tests.models.flash_next import moe_block  # noqa: F401  (CPU stubs first)

import pytest
import torch

import vllm_gaudi.extension.ops as plugin_ops
from vllm_gaudi import envs

H, I, LOCAL, GLOBAL, TOPK = 64, 32, 8, 32, 4
EXPERTS_MIN = 8  # this rank's experts are the global ids 8 .. 15


def _fp8_gemm_v2(A, trans_A, B, trans_B, D, out_dtype, A_scale_inv=None, B_scale_inv=None, bias=None,
                 accumulate=False):
    a = A.float().transpose(-1, -2) if trans_A else A.float()
    b = B.float().transpose(-1, -2) if trans_B else B.float()
    out = torch.matmul(a, b)
    if A_scale_inv is not None:
        out = out * A_scale_inv
    if B_scale_inv is not None:
        out = out * B_scale_inv
    return out.to(out_dtype)


@pytest.fixture(autouse=True)
def _vendor_ops(monkeypatch):
    moe_block.install_vendor_ops(fp8_max=float(plugin_ops.FP8_MAX))
    monkeypatch.setattr(torch.ops.hpu, "fp8_gemm_v2", _fp8_gemm_v2, raising=False)
    moe_block.VENDOR_CALLS.clear()


def _op(seed=0):
    """A per-channel FP8 MoE op over stacked experts (the per-expert weights are views, as on a card)."""
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(LOCAL, 2 * I, H, generator=g) * 0.05
    w2 = torch.randn(LOCAL, H, I, generator=g) * 0.05
    s13 = (w13.abs().amax(dim=2) + 1e-8) / float(plugin_ops.FP8_MAX)
    s2 = (w2.abs().amax(dim=2) + 1e-8) / float(plugin_ops.FP8_MAX)
    w13_8 = (w13 / s13.unsqueeze(-1)).to(torch.float8_e4m3fn)
    w2_8 = (w2 / s2.unsqueeze(-1)).to(torch.float8_e4m3fn)
    op = plugin_ops.VllmMixtureOfExpertsOpFP8PerChannel(GLOBAL, LOCAL, EXPERTS_MIN, EXPERTS_MIN + LOCAL - 1, None)
    for e in range(LOCAL):
        op.w13_list[e].set_weight(w13_8[e])
        op.w13_list[e].set_scale_inv_fp8(s13[e])
        op.w2_list[e].set_weight(w2_8[e])
        op.w2_list[e].set_scale_inv_fp8(s2[e])
    op._cache_weight_lists()
    return op, (w13_8, w2_8, s13, s2)


def _batch(rows, seed=1, local_only=None):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, H, generator=g)).to(torch.bfloat16)
    if local_only is False:
        pool = torch.tensor([i for i in range(GLOBAL) if not EXPERTS_MIN <= i < EXPERTS_MIN + LOCAL])
        ids = torch.stack([pool[torch.randperm(len(pool), generator=g)[:TOPK]] for _ in range(rows)])
    else:
        ids = torch.stack([torch.randperm(GLOBAL, generator=g)[:TOPK] for _ in range(rows)])
    weights = torch.softmax(torch.randn(rows, TOPK, generator=g), dim=-1).to(torch.bfloat16)
    return x, ids.to(torch.int64), weights


@pytest.fixture
def _dense_from_one_row(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_MOE_DENSE_MIN_TOKENS", "1")


@pytest.mark.parametrize("rows", [1, 7, 32])
def test_dense_form_matches_the_vendor_op_to_fp8_rounding(rows, _dense_from_one_row):
    op, dense = _op()
    x, ids, weights = _batch(rows)
    vendor = op(x, ids, weights).float()
    assert len(moe_block.VENDOR_CALLS) == 1
    op.set_dense_weights(*dense)
    got = op(x, ids, weights).float()
    assert len(moe_block.VENDOR_CALLS) == 1, "the dense form must not call the vendor op"
    scale = float(vendor.abs().max()) or 1.0
    assert float((got - vendor).abs().max()) / scale < 0.05
    # the rows without any local expert are exactly zero in both
    no_local = ~((ids >= EXPERTS_MIN) & (ids < EXPERTS_MIN + LOCAL)).any(dim=1)
    assert torch.equal(got[no_local], torch.zeros_like(got[no_local]))


def test_rows_routed_only_to_other_ranks_give_zeros(_dense_from_one_row):
    op, dense = _op()
    op.set_dense_weights(*dense)
    x, ids, weights = _batch(5, local_only=False)
    assert torch.equal(op(x, ids, weights), torch.zeros(5, H, dtype=torch.bfloat16))


def test_more_tokens_than_the_limit_keep_the_vendor_op(monkeypatch, _dense_from_one_row):
    op, dense = _op()
    op.set_dense_weights(*dense)
    monkeypatch.setenv("VLLM_HPU_MOE_DENSE_MAX_TOKENS", "16")
    op(*_batch(16))
    assert len(moe_block.VENDOR_CALLS) == 0
    op(*_batch(17))
    assert len(moe_block.VENDOR_CALLS) == 1


def test_fewer_tokens_than_the_minimum_keep_the_vendor_op():
    """Default minimum 17: a step of 1 to 16 rows reads only the hit experts through the vendor op."""
    op, dense = _op()
    op.set_dense_weights(*dense)
    assert envs.VLLM_HPU_MOE_DENSE_MIN_TOKENS == 17
    op(*_batch(16))
    assert len(moe_block.VENDOR_CALLS) == 1
    op(*_batch(17))
    assert len(moe_block.VENDOR_CALLS) == 1


def test_weights_are_kept_only_with_the_flag(monkeypatch):
    op, dense = _op()

    class Layer:
        pass

    layer = Layer()
    layer.moe_op = op
    layer.w13_weight = torch.nn.Parameter(dense[0], requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(dense[1], requires_grad=False)
    monkeypatch.setenv("VLLM_HPU_MOE_DENSE", "0")
    plugin_ops._set_moe_dense_weights(layer)
    assert op.dense_weights is None
    monkeypatch.setenv("VLLM_HPU_MOE_DENSE", "1")
    plugin_ops._set_moe_dense_weights(layer)
    w13, w2, s13, s2 = op.dense_weights
    # the layer's stacked tensors themselves (no copy of the experts)
    assert w13.data_ptr() == layer.w13_weight.data_ptr() and w2.data_ptr() == layer.w2_weight.data_ptr()
    assert torch.equal(s13, dense[2].float()) and torch.equal(s2, dense[3].float())
    op._apply(lambda t: t)  # model.to("hpu") on tensors already there (the runner does it after loading): kept
    assert op.dense_weights is not None and op.dense_weights[0].data_ptr() == layer.w13_weight.data_ptr()
    op._apply(lambda t: t.clone())  # a real move would copy the stacked experts: dropped, the vendor op serves
    assert op.dense_weights is None


@pytest.mark.parametrize("tokens,expected", [(1, {}), (16, {}), (17, {"chunk_size": 64, "total_experts": GLOBAL}),
                                             (512, {"chunk_size": 64, "total_experts": GLOBAL}),
                                             (513, {"chunk_size": 8192, "total_experts": GLOBAL})])
def test_a_zero_chunk_size_leaves_the_vendor_op_its_default(tokens, expected):
    """VLLM_MOE_CHUNK=0,64,8192 VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000: no chunk_size up to 16 tokens."""
    op, _ = _op()
    op.chunk_size_list, op.token_boundary_list = [0, 64, 8192], [16, 512, 1000000]
    assert op._get_extra_kwargs(tokens) == expected


def test_route_stats_record_the_pairs_per_local_expert(monkeypatch, tmp_path):
    """VLLM_HPU_MOE_ROUTE_STATS: an op with kept stacked experts writes the per-expert pair counts of its last call
    and the call's row count into its row of the buffer, on the dense form and on the vendor op alike; without the
    flag no row is allocated."""
    monkeypatch.setattr(plugin_ops, "_ROUTE_STATS", {"buffer": None, "ops": 0})
    op, dense = _op()

    class Layer:
        pass

    layer = Layer()
    layer.moe_op = op
    layer.w13_weight = torch.nn.Parameter(dense[0], requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(dense[1], requires_grad=False)
    monkeypatch.setenv("VLLM_HPU_MOE_DENSE", "1")
    plugin_ops._set_moe_dense_weights(layer)
    assert op.route_stats_slot is None and plugin_ops.moe_route_stats_rows() is None
    op.dense_weights = None
    monkeypatch.setenv("VLLM_HPU_MOE_ROUTE_STATS", str(tmp_path))
    plugin_ops._set_moe_dense_weights(layer)
    assert int(op.route_stats_slot) == 0
    for rows in (20, 5):  # 20: the dense form (17 to 512 rows); 5: the vendor op
        x, ids, weights = _batch(rows, seed=rows)
        op(x, ids, weights)
        expected = [int(((ids - EXPERTS_MIN) == e).sum()) for e in range(LOCAL)] + [rows]
        assert plugin_ops.moe_route_stats_rows() == [expected]
    assert len(moe_block.VENDOR_CALLS) == 1
