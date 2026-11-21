# SPDX-License-Identifier: Apache-2.0
"""CPU tests for flashinfer_gaudi.gdn_core_decode.gdn_core_decode_reference.

Prove the vectorized reference equals the per-row loop oracle (``_loop_oracle``) and the
composed torch path (``_composed_path``) at the TP1/TP2/TP4/TP8 head counts, with
load != store slots and padded rows.  Both functions are copied from
``spikes/gdn-core-decode/probe.py``; contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

import pytest
import torch

from flashinfer_gaudi.gdn_core_decode import GdnGateActivation, gdn_core_decode_reference
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating, hpu_fused_recurrent_gated_delta_rule

HEAD_COUNTS = [(16, 48), (8, 24), (4, 12), (2, 6)]
STORED_SLOTS = [2, 3, 6]
REAL_ROWS = [0, 1, 2]
UNTOUCHED_SLOTS = [0, 1, 4, 5, 7, 8]


def _loop_oracle(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, load, store,
                 scale, eps, gate):
    S = ssm_state.shape[0]
    B, C = mixed_qkv.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    H = (C - HV * D) // (2 * D)
    rep = HV // H
    out = torch.zeros(B, HV, D, dtype=torch.float32)
    for r in range(B):
        ls, ss = int(load[r]), int(store[r])
        if not (0 <= ls < S and 0 <= ss < S):
            continue
        hist = conv_state[ls, 0:3, :].float()
        x = mixed_qkv[r].float()
        w = conv_weight_t.float()
        conv = hist[0] * w[0] + hist[1] * w[1] + hist[2] * w[2] + x * w[3]
        act = conv * torch.sigmoid(conv)
        new_hist = torch.stack([conv_state[ls, 1], conv_state[ls, 2], mixed_qkv[r].to(conv_state.dtype)])
        for kh in range(H):
            q = act[kh * D:(kh + 1) * D]
            k = act[H * D + kh * D:H * D + (kh + 1) * D]
            q = q * torch.rsqrt((q * q).sum() + 1e-6) * scale
            k = k * torch.rsqrt((k * k).sum() + 1e-6)
            for j in range(rep):
                vh = kh * rep + j
                v = act[2 * H * D + vh * D:2 * H * D + (vh + 1) * D]
                xg = a[r, vh].float() + dt_bias[vh].float()
                sp = torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
                decay = torch.exp(-torch.exp(A_log[vh].float()) * sp)
                beta = torch.sigmoid(b[r, vh].float())
                St = ssm_state[ls, vh].float() * decay
                delta = (v - St @ k) * beta
                St = St + delta[:, None] * k[None, :]
                o = St @ q
                y = o * torch.rsqrt((o * o).mean() + eps) * norm_weight.float()
                zz = z[r, vh * D:(vh + 1) * D].float()
                y = y * (torch.sigmoid(zz) if gate == 1 else zz * torch.sigmoid(zz))
                out[r, vh] = y
                ssm_state[ss, vh] = St.to(ssm_state.dtype)
        conv_state[ss, 0:3, :] = new_hist
    return out


def _composed_path(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, load,
                   store, scale, eps, gate):
    B, C = mixed_qkv.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    H = (C - HV * D) // (2 * D)
    qsl = torch.arange(B + 1, dtype=torch.int32)
    g, beta = hpu_fused_gdn_gating(A_log, a, b, dt_bias)
    xc = hpu_causal_conv1d_update(x=mixed_qkv, conv_state=conv_state, weight=conv_weight_t.t().contiguous(),
                                  bias=None, activation="silu", conv_state_indices=load, query_start_loc=qsl,
                                  store_cache_indices=store)
    q, k, v = torch.split(xc, [H * D, H * D, HV * D], dim=-1)
    q = q.reshape(1, B, H, D)
    k = k.reshape(1, B, H, D)
    v = v.reshape(1, B, HV, D)
    o, _ = hpu_fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=ssm_state,
                                                inplace_final_state=True, cu_seqlens=qsl, ssm_state_indices=load,
                                                use_qk_l2norm_in_kernel=True, ssm_store_indices=store)
    o = o.reshape(B * HV, D)
    y = RMSNormGated.forward_static(o, z.reshape(B * HV, D), norm_weight, eps, o.dtype, None, True,
                                    "sigmoid" if gate == 1 else "silu")
    return y.reshape(B, HV, D).float()


def _make_inputs(H, HV, D, dtype, B=4, S=9, seed=0, load=None, store=None):
    torch.manual_seed(seed)
    C = 2 * H * D + HV * D
    bf = lambda *shape: torch.randn(*shape).to(torch.bfloat16).to(dtype)
    return {
        "conv_state": bf(S, 3, C),
        "ssm_state": (torch.randn(S, HV, D, D) * 0.1).float(),
        "mixed_qkv": bf(B, C),
        "z": bf(B, HV * D),
        "a": bf(B, HV),
        "b": bf(B, HV),
        "A_log": torch.randn(HV).float() * 0.5,
        "dt_bias": torch.randn(HV).float(),
        "conv_weight_t": bf(4, C) * 0.5,
        "norm_weight": bf(D) * 0.1 + 1,
        "load_indices": torch.tensor([1, 3, 5, -1] if load is None else load, dtype=torch.int32),
        "store_indices": torch.tensor([2, 3, 6, -1] if store is None else store, dtype=torch.int32),
        "scale": D ** -0.5,
        "eps": 1e-6,
    }


def _args(inp):
    return (inp["conv_state"], inp["ssm_state"], inp["mixed_qkv"], inp["z"], inp["a"], inp["b"], inp["A_log"],
            inp["dt_bias"], inp["conv_weight_t"], inp["norm_weight"], inp["load_indices"], inp["store_indices"],
            inp["scale"], inp["eps"])


def _fork(inp):
    """Positional args for one callee with its own state clones: (conv, ssm) are mutated, rest shared."""
    return [inp["conv_state"].clone(), inp["ssm_state"].clone()] + list(_args(inp)[2:])


@pytest.mark.parametrize("gate", [GdnGateActivation.SILU, GdnGateActivation.SIGMOID])
@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_matches_loop_oracle(H, HV, gate):
    D = 128
    inp = _make_inputs(H, HV, D, torch.float32)
    oracle_args = _fork(inp)
    exp = _loop_oracle(*oracle_args, gate)
    args = _fork(inp)
    out = gdn_core_decode_reference(*args, gate)
    torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(args[1], oracle_args[1], atol=1e-5, rtol=1e-5)
    assert torch.equal(args[0], oracle_args[0])


@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_matches_composed_fp32(H, HV):
    D = 128
    inp = _make_inputs(H, HV, D, torch.float32)
    comp_args = _fork(inp)
    exp = _composed_path(*comp_args, GdnGateActivation.SIGMOID)
    args = _fork(inp)
    out = gdn_core_decode_reference(*args, GdnGateActivation.SIGMOID)
    torch.testing.assert_close(out[REAL_ROWS], exp[REAL_ROWS], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(args[1][STORED_SLOTS], comp_args[1][STORED_SLOTS], atol=1e-5, rtol=1e-5)
    assert torch.equal(args[0][STORED_SLOTS], comp_args[0][STORED_SLOTS])


@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_matches_composed_bf16(H, HV):
    D = 128
    inp = _make_inputs(H, HV, D, torch.bfloat16)
    comp_args = _fork(inp)
    exp = _composed_path(*comp_args, GdnGateActivation.SIGMOID)
    args = _fork(inp)
    out = gdn_core_decode_reference(*args, GdnGateActivation.SIGMOID)
    torch.testing.assert_close(out[REAL_ROWS], exp[REAL_ROWS], atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(args[1][STORED_SLOTS], comp_args[1][STORED_SLOTS], atol=3e-2, rtol=1e-2)
    assert torch.equal(args[0][STORED_SLOTS], comp_args[0][STORED_SLOTS])


@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_padding_and_untouched_slots(H, HV):
    D = 128
    inp = _make_inputs(H, HV, D, torch.float32)
    c_init, s_init = inp["conv_state"].clone(), inp["ssm_state"].clone()
    args = _fork(inp)
    out = gdn_core_decode_reference(*args, GdnGateActivation.SIGMOID)
    assert torch.equal(out[3], torch.zeros(HV, D, dtype=torch.float32))
    assert torch.equal(args[0][UNTOUCHED_SLOTS], c_init[UNTOUCHED_SLOTS])
    assert torch.equal(args[1][UNTOUCHED_SLOTS], s_init[UNTOUCHED_SLOTS])


@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_padding_with_dummy_slot(H, HV):
    D = 128
    dummy = [1, 3, 8, 8]
    inp = _make_inputs(H, HV, D, torch.float32, load=dummy, store=dummy)
    oracle_args = _fork(inp)
    exp = _loop_oracle(*oracle_args, GdnGateActivation.SIGMOID)
    args = _fork(inp)
    out = gdn_core_decode_reference(*args, GdnGateActivation.SIGMOID)
    torch.testing.assert_close(out[0:2], exp[0:2], atol=1e-5, rtol=1e-5)
    for slot in (1, 3):
        torch.testing.assert_close(args[1][slot], oracle_args[1][slot], atol=1e-5, rtol=1e-5)
        assert torch.equal(args[0][slot], oracle_args[0][slot])


@pytest.mark.parametrize("gate", [GdnGateActivation.SILU, GdnGateActivation.SIGMOID])
def test_tiny_head_dim(gate):
    H, HV, D = 4, 12, 16
    inp = _make_inputs(H, HV, D, torch.float32)
    oracle_args = _fork(inp)
    exp = _loop_oracle(*oracle_args, gate)
    args = _fork(inp)
    out = gdn_core_decode_reference(*args, gate)
    torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(args[1], oracle_args[1], atol=1e-5, rtol=1e-5)
    assert torch.equal(args[0], oracle_args[0])


@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_rejects_bad_shapes(H, HV):
    D = 128
    inp = _make_inputs(H, HV, D, torch.float32)
    bad = dict(inp)
    bad["conv_state"] = torch.zeros(9, 4, 2 * H * D + HV * D, dtype=torch.float32)
    with pytest.raises(ValueError):
        gdn_core_decode_reference(*_args(bad), GdnGateActivation.SIGMOID)
    with pytest.raises(ValueError):
        gdn_core_decode_reference(*_args(inp), 2)
