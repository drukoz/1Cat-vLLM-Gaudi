# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the recurrence-only op ``flashinfer_gaudi::gdn_recurrent_decode`` (RT1).

The op runs the GDN delta-rule recurrence alone (activated q/k/v in, pre-norm ``o`` out, SSM
state updated from load slot to store slot).  The oracle is the composed torch path of
``test_reference.py`` without the conv (``qkv`` fed as ``xc``) and without the norm.
Contract: ``docs/design/flash-next/gdn-core-decode.md``, "Recurrence-only op".
"""

import pytest
import torch

from flashinfer_gaudi.gdn_core_decode import (GdnCoreDecodeMode, GdnGateActivation, gdn_core_decode_reference,
                                              gdn_recurrent_decode, gdn_recurrent_decode_available,
                                              gdn_recurrent_decode_reference)
from tests.unit_tests.gdn_core_decode.test_reference import (STORED_SLOTS, UNTOUCHED_SLOTS, _make_inputs)
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating, hpu_fused_recurrent_gated_delta_rule

HEAD_COUNTS = [(16, 48), (8, 24), (4, 12), (2, 6)]
DTYPES = [torch.float32, torch.bfloat16]
REAL_ROWS = [0, 1, 2]


def _silu_qkv(inp, dtype):
    """mixed_qkv through SiLU in fp32, cast to dtype: stands in for the conv update's output."""
    x = inp["mixed_qkv"].float()
    return (x * torch.sigmoid(x)).to(dtype)


def _rec_args(inp, dtype):
    """Positional args for one recurrence call, with its own ssm_state clone."""
    return [inp["ssm_state"].clone()] + [_silu_qkv(inp, dtype)] + [
        inp["a"], inp["b"], inp["A_log"], inp["dt_bias"], inp["load_indices"], inp["store_indices"], inp["scale"]
    ]


def _recurrence_oracle(ssm_state, qkv, a, b, A_log, dt_bias, load, store, scale):
    """The composed path's gating + recurrent GDN delta rule: no conv, no norm (mutates ssm_state)."""
    B, C = qkv.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    H = (C - HV * D) // (2 * D)
    qsl = torch.arange(B + 1, dtype=torch.int32)
    g, beta = hpu_fused_gdn_gating(A_log, a, b, dt_bias)
    q, k, v = torch.split(qkv, [H * D, H * D, HV * D], dim=-1)
    q = q.reshape(1, B, H, D)
    k = k.reshape(1, B, H, D)
    v = v.reshape(1, B, HV, D)
    o, _ = hpu_fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=ssm_state,
                                                inplace_final_state=True, cu_seqlens=qsl, ssm_state_indices=load,
                                                use_qk_l2norm_in_kernel=True, ssm_store_indices=store)
    return o.reshape(B, HV, D).float()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H,HV", HEAD_COUNTS)
def test_matches_composed_path(H, HV, dtype):
    D = 128
    inp = _make_inputs(H, HV, D, dtype)
    args = _rec_args(inp, dtype)
    out = gdn_recurrent_decode_reference(*args)
    oargs = _rec_args(inp, dtype)
    exp = _recurrence_oracle(*oargs)
    if dtype == torch.float32:
        torch.testing.assert_close(out[REAL_ROWS], exp[REAL_ROWS], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(args[0][STORED_SLOTS], oargs[0][STORED_SLOTS], atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(out[REAL_ROWS], exp[REAL_ROWS], atol=5e-2, rtol=1e-2)
        torch.testing.assert_close(args[0][STORED_SLOTS], oargs[0][STORED_SLOTS], atol=3e-2, rtol=1e-2)
    assert torch.equal(out[3], torch.zeros(HV, D, dtype=torch.float32))
    assert torch.equal(args[0][UNTOUCHED_SLOTS], inp["ssm_state"][UNTOUCHED_SLOTS])


def test_agrees_with_core_op_fp32():
    """One recurrence step: fed the core op's conv+SiLU output, the two ops write the same state."""
    H, HV, D = 8, 24, 128
    inp = _make_inputs(H, HV, D, torch.float32)
    load = inp["load_indices"].long().clamp(min=0)  # the padded row's act is never used
    hist = inp["conv_state"].index_select(0, load).float()
    w = inp["conv_weight_t"].float()
    conv = hist[:, 0] * w[0] + hist[:, 1] * w[1] + hist[:, 2] * w[2] + inp["mixed_qkv"].float() * w[3]
    act = conv * torch.sigmoid(conv)

    core_conv, core_ssm = inp["conv_state"].clone(), inp["ssm_state"].clone()
    gdn_core_decode_reference(core_conv, core_ssm, inp["mixed_qkv"], inp["z"], inp["a"], inp["b"], inp["A_log"],
                              inp["dt_bias"], inp["conv_weight_t"], inp["norm_weight"], inp["load_indices"],
                              inp["store_indices"], inp["scale"], inp["eps"], GdnGateActivation.SIGMOID)
    rec_ssm = inp["ssm_state"].clone()
    gdn_recurrent_decode_reference(rec_ssm, act, inp["a"], inp["b"], inp["A_log"], inp["dt_bias"],
                                   inp["load_indices"], inp["store_indices"], inp["scale"])
    torch.testing.assert_close(rec_ssm, core_ssm, atol=1e-6, rtol=1e-6)


def test_load_equals_store():
    """load == store on every row works: the state is read before the write."""
    H, HV, D = 4, 12, 128
    slots = [1, 2, 3, 4]
    inp = _make_inputs(H, HV, D, torch.float32, load=slots, store=slots)
    args = _rec_args(inp, torch.float32)
    out = gdn_recurrent_decode_reference(*args)
    oargs = _rec_args(inp, torch.float32)
    exp = _recurrence_oracle(*oargs)
    torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(args[0][slots], oargs[0][slots], atol=1e-5, rtol=1e-5)
    assert not torch.equal(args[0][slots], inp["ssm_state"][slots])


@pytest.mark.parametrize("dtype", DTYPES)
def test_op_matches_reference(dtype):
    inp = _make_inputs(16, 48, 128, dtype)
    eargs = _rec_args(inp, dtype)
    exp = gdn_recurrent_decode_reference(*eargs)
    args = tuple(_rec_args(inp, dtype))
    out = gdn_recurrent_decode(*args)
    assert torch.equal(out, exp)
    assert torch.equal(args[0], eargs[0])
    result = torch.library.opcheck(torch.ops.flashinfer_gaudi.gdn_recurrent_decode.default,
                                   args,
                                   test_utils=("test_schema", "test_faketensor"))
    assert result == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}


def test_fake_allocates_only():
    B, H, HV, D = 4, 16, 48, 128
    inp = _make_inputs(H, HV, D, torch.float32)
    args = tuple(_rec_args(inp, torch.float32))
    mode = torch._subclasses.fake_tensor.FakeTensorMode()
    with mode:
        fargs = [mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        out = torch.ops.flashinfer_gaudi.gdn_recurrent_decode(*fargs)
    assert out.shape == (B, HV, D)
    assert out.dtype == torch.float32


def test_available():
    assert gdn_recurrent_decode_available("cpu") is True
    assert gdn_recurrent_decode_available("meta") is False


def test_mode_enum():
    assert GdnCoreDecodeMode(2) is GdnCoreDecodeMode.RECURRENCE
    assert GdnCoreDecodeMode.OFF == 0
    assert GdnCoreDecodeMode.FUSED == 1
