# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the ``torch.library`` op ``flashinfer_gaudi::gdn_core_verify`` (Phase 2 contract, version 2).

The op wraps the T-node verification reference (contract: ``docs/design/flash-next/gdn-core-decode.md``,
"Phase 2 contract, version 2"): ``gdn_core_verify_reference`` matches ``verify_oracle`` (the contract's loop
form), the reference matches M3's composed speculative path (the probe's check, copied from
``spikes/gdn-core-verify/probe.py`` into this file) at all four TP head counts over 4 chained steps, with k = 3
draft tokens (T = 4) and with k = 5 (T = 6, the largest the op takes), the CPU
kernel matches the reference bit-exactly (output and both mutated pools), ``opcheck`` passes on schema and fake
tensor, the fake kernel allocates only, ``gdn_core_verify_available`` reports kernel presence per device type,
and ``GdnCoreVerifyMode`` carries the flag values.
"""

import pytest
import torch

import vllm_gaudi  # noqa: F401  (the plugin registers the platform the composed-path ops run on)
from vllm.model_executor.layers.layernorm import RMSNormGated

from flashinfer_gaudi.gdn_core_decode import GdnGateActivation
from flashinfer_gaudi.gdn_core_verify import (GDN_CORE_VERIFY_MAX_NODES, GdnCoreVerifyMode, gdn_core_verify,
                                              gdn_core_verify_available, gdn_core_verify_reference)
from flashinfer_gaudi.gdn_decode import gated_delta_rule_mtp_packed
from tests.unit_tests.gdn_core_decode.sim_case import D
from tests.unit_tests.gdn_core_decode.verify_sim_case import (VERIFY_ARG_ORDER, VERIFY_HPU_CASES, compare_verify,
                                                              make_verify_inputs, verify_oracle, verify_scalars)
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating

GATE = int(GdnGateActivation.SIGMOID)

# The probe's chain schedule (spikes/gdn-core-verify/probe.py): B = 3 requests, k draft tokens (T = 1 + k nodes,
# R = 3 + k conv rows; the probe ran k = 3), 4 chained steps; "full" accepts every draft, "trimmed" trims query
# lengths (step 1 row 2 is padding).
HEAD_COUNTS = [(16, 48), (8, 24), (4, 12), (2, 6)]  # TP1, TP2, TP4, TP8
DRAFT_COUNTS = [3, 5]  # k: Flash-Next's recipe so far, and the op's maximum (GDN_CORE_VERIFY_MAX_NODES - 1)
B = 3
STEPS = 4


def _full_qlens(k: int) -> list[list[int]]:
    return [[k + 1] * B] * STEPS


def _trimmed_qlens(k: int) -> list[list[int]]:
    return [[k + 1, 2, k + 1], [k + 1, k + 1, 1], [3, k + 1, k + 1], [k + 1] * B]


def _case(name: str):
    return next(c for c in VERIFY_HPU_CASES if c.name == name)


def _op_args(case, inputs):
    """Positional args for one op call, with its own state clones."""
    return ([inputs[k].clone() for k in VERIFY_ARG_ORDER] + list(verify_scalars(case)))


def composed(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, lat_load, lat_store,
             accepted, query_len, align, scale, eps, gate):
    """M3's speculative decode path, as qwen3_5.py runs it (is_spec_decode, lattice_store_differs = align)."""
    B_, T, C = mixed_qkv.shape
    HV, D_ = ssm_state.shape[1], ssm_state.shape[2]
    qsl = torch.zeros(B_ + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(query_len, 0)
    g, beta = hpu_fused_gdn_gating(A_log, a.reshape(B_ * T, HV), b.reshape(B_ * T, HV), dt_bias)
    xc = hpu_causal_conv1d_update(x=mixed_qkv.reshape(B_ * T, C),
                                  conv_state=conv_state,
                                  weight=conv_weight_t.t().contiguous(),
                                  bias=None,
                                  activation="silu",
                                  conv_state_indices=lat_load[:, 0],
                                  num_accepted_tokens=accepted,
                                  query_start_loc=qsl,
                                  max_query_len=T,
                                  store_cache_indices=lat_store[:, 0] if align else None)
    o, _ = gated_delta_rule_mtp_packed(packed_qkv=xc.reshape(B_, T, -1),
                                       log_decay=g.reshape(B_, T, -1),
                                       beta=beta.reshape(B_, T, -1),
                                       state_pool=ssm_state,
                                       state_indices=lat_load,
                                       num_accepted_tokens=accepted,
                                       query_lengths=query_len,
                                       scale=scale,
                                       use_qk_l2norm=True,
                                       store_indices=lat_store if align else None)
    o = o.reshape(B_ * T * HV, D_)
    y = RMSNormGated.forward_static(o, z.reshape(B_ * T * HV, D_), norm_weight, eps, o.dtype, None, True,
                                    "sigmoid" if gate == 1 else "silu")
    return y.reshape(B_, T, HV, D_).float()


def op_indices(lat_load, lat_store, accepted, query_len, align, R):
    """The layer's mapping from M3's lattice to the op's index columns (the probe's W1 wiring)."""
    T = lat_load.shape[1]
    acc_conv = accepted.long().clamp(1, R - 2) - 1
    acc_ssm = accepted.long().clamp(1, T) - 1
    conv_load = lat_load[:, 0]
    conv_store = lat_store[:, 0] if align else lat_load[:, 0]
    ssm_load = lat_load.gather(1, acc_ssm[:, None]).squeeze(1)
    ssm_store = (lat_store if align else lat_load).clone()
    return conv_load, acc_conv.to(torch.int32), conv_store, ssm_load, ssm_store


@pytest.mark.parametrize("case", VERIFY_HPU_CASES, ids=[c.name for c in VERIFY_HPU_CASES])
def test_reference_matches_oracle(case):
    """The reference vs verify_oracle (both run the contract's per-node math in fp32)."""
    inputs = make_verify_inputs(case)
    scale, eps, gate = verify_scalars(case)
    args = [inputs[k] for k in VERIFY_ARG_ORDER[2:]]

    conv_ref, ssm_ref = inputs["conv_state"].clone(), inputs["ssm_state"].clone()
    out_ref = gdn_core_verify_reference(conv_ref, ssm_ref, *args, scale, eps, gate)
    conv_or, ssm_or = inputs["conv_state"].clone(), inputs["ssm_state"].clone()
    out_or = verify_oracle(conv_or, ssm_or, *args, scale, eps, gate)

    torch.testing.assert_close(out_ref, out_or, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ssm_ref, ssm_or, atol=1e-6, rtol=1e-5)
    assert torch.equal(conv_ref, conv_or)
    assert compare_verify(case, inputs, {"out": out_ref, "conv_state": conv_ref, "ssm_state": ssm_ref}) == []


def test_draft_counts_end_at_the_op_maximum():
    assert max(DRAFT_COUNTS) + 1 == GDN_CORE_VERIFY_MAX_NODES


@pytest.mark.parametrize("H,HV", HEAD_COUNTS, ids=["tp1", "tp2", "tp4", "tp8"])
@pytest.mark.parametrize("align", [False, True], ids=["noalign", "align"])
@pytest.mark.parametrize("trimmed", [False, True], ids=["full", "trimmed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("K", DRAFT_COUNTS, ids=["k3", "k5"])
def test_matches_m3_composed_path(H, HV, dtype, align, trimmed, K):
    """The reference vs M3's composed speculative path, 4 chained steps, accepted counts threaded forward.

    The probe's check (spikes/gdn-core-verify/probe.py), adapted to the token-major op layout.  Tolerances from
    the contract: fp32 out/ssm ``atol=1e-5, rtol=1e-5``; bf16 out ``atol=5e-2, rtol=1e-2``, ssm ``atol=3e-2,
    rtol=1e-2``; conv state exact in both.
    """
    torch.manual_seed(0)
    C = 2 * H * D + HV * D
    T, R = 1 + K, 3 + K
    S = 2 + B * T * 2 + 1
    bf = lambda *s: torch.randn(*s).to(torch.bfloat16).to(dtype)  # noqa: E731
    conv0 = bf(S, R, C)
    ssm0 = (torch.randn(S, HV, D, D) * 0.1).float()
    A_log = torch.randn(HV).float() * 0.5
    dt_bias = torch.randn(HV).float()
    wt = bf(4, C) * 0.5
    nw = bf(D) * 0.1 + 1
    c1, s1 = conv0.clone(), ssm0.clone()  # the reference's pools
    c2, s2 = conv0.clone(), ssm0.clone()  # the composed path's pools
    out_tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=5e-2, rtol=1e-2)
    ssm_tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=3e-2, rtol=1e-2)
    qlens = _trimmed_qlens(K) if trimmed else _full_qlens(K)
    accepted = torch.ones(B, dtype=torch.int32)
    for step in range(STEPS):
        mixed = bf(B, T, C)
        z = bf(B, T, HV * D)
        a = bf(B, T, HV)
        b = bf(B, T, HV)
        base = 1 + torch.arange(B) * (2 * T)
        lat_load = (base[:, None] + torch.arange(T)[None, :]).to(torch.int32)
        if align:
            # Align mode: the write lattice is a different set of slots from the read lattice (alternating halves).
            lat_store = (lat_load + T) if step % 2 == 0 else lat_load.clone()
            if step % 2 == 1:
                lat_load = lat_load + T
        else:
            lat_store = lat_load
        ql = torch.tensor(qlens[step], dtype=torch.int32)
        if trimmed and step == 1:  # the trimmed schedule's padding row (load -1)
            lat_load = lat_load.clone()
            lat_load[B - 1] = -1
            lat_store = lat_store.clone()
            lat_store[B - 1] = -1
        cl, co, cs, sl, ss = op_indices(lat_load, lat_store, accepted, ql, align, R)
        # Nodes past the query length store nothing (the composed path's token mask).
        ss = torch.where(torch.arange(T)[None, :] < ql[:, None], ss, torch.full_like(ss, -1))
        row_indices = torch.stack([cl, co, cs, sl, ql], dim=1).to(torch.int32)
        y1 = gdn_core_verify_reference(c1, s1, mixed.reshape(B * T, C), z.reshape(B * T, HV * D), a.reshape(B * T, HV),
                                       b.reshape(B * T, HV), A_log, dt_bias, wt, nw, row_indices, ss, D**-0.5, 1e-6,
                                       GATE).view(B, T, HV, D)
        y2 = composed(c2, s2, mixed, z, a, b, A_log, dt_bias, wt, nw, lat_load, lat_store, accepted, ql, align, D**-0.5,
                      1e-6, GATE)
        for r in range(B):
            if int(lat_load[r, 0]) < 0:
                continue  # padding row
            n = int(ql[r])
            torch.testing.assert_close(y1[r, :n], y2[r, :n], **out_tol)
            for t in range(n):
                st = int(ss[r, t])
                torch.testing.assert_close(s1[st], s2[st], **ssm_tol)
            # Visible conv rows: the history window the next step can read (rows 0 .. n + 1), exact.
            cslot = int(cs[r])
            assert torch.equal(c1[cslot, :n + 2], c2[cslot, :n + 2])
        # The next step's accepted counts (a fixed schedule, threaded forward).
        nxt = [[K + 1, 1, 2], [2, K + 1, 1], [1, 2, K + 1], [K + 1, K + 1, 1]][step % 4]
        accepted = torch.tensor([min(nxt[r], int(ql[r])) for r in range(B)], dtype=torch.int32)


@pytest.mark.parametrize("name", ["ver_tp8_chain", "ver_tp4_t1", "ver_tp8_t6_chain", "ver_tp4_t5_b2"])
def test_op_matches_reference(name):
    """The op (torch.ops and the wrapper) matches the reference bit-exactly: out and both pools."""
    case = _case(name)
    inputs = make_verify_inputs(case)
    scale, eps, gate = verify_scalars(case)
    conv_exp, ssm_exp = inputs["conv_state"].clone(), inputs["ssm_state"].clone()
    out_exp = gdn_core_verify_reference(conv_exp, ssm_exp, *[inputs[k] for k in VERIFY_ARG_ORDER[2:]], scale, eps, gate)
    for run in (torch.ops.flashinfer_gaudi.gdn_core_verify, gdn_core_verify):
        args = _op_args(case, inputs)
        out = run(*args)
        assert torch.equal(out, out_exp)
        assert torch.equal(args[0], conv_exp)
        assert torch.equal(args[1], ssm_exp)


def test_opcheck():
    case = _case("ver_tp8_chain")
    args = tuple(_op_args(case, make_verify_inputs(case)))
    result = torch.library.opcheck(torch.ops.flashinfer_gaudi.gdn_core_verify.default,
                                   args,
                                   test_utils=("test_schema", "test_faketensor"))
    assert result == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}


def test_fake_shape():
    case = _case("ver_tp8_chain")
    args = _op_args(case, make_verify_inputs(case))
    mode = torch._subclasses.fake_tensor.FakeTensorMode()
    with mode:
        fargs = [mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        out = torch.ops.flashinfer_gaudi.gdn_core_verify(*fargs)
    assert out.shape == (case.B * case.T, case.HV, D)
    assert out.dtype == torch.float32


def test_available():
    assert gdn_core_verify_available("cpu") is True
    assert gdn_core_verify_available("meta") is False


def test_enum():
    assert GdnCoreVerifyMode.OFF == 0
    assert GdnCoreVerifyMode.FUSED == 1
