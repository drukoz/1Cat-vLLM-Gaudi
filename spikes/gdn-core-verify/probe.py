# SPDX-License-Identifier: Apache-2.0
# CPU probe of the gdn_core_verify contract (docs/design/flash-next/gdn-core-decode.md, "Phase 2 contract, version 2")
# against M3's composed speculative path in vllm_gaudi/models/qwen3_5.py (the is_spec_decode branch):
#   hpu_fused_gdn_gating -> hpu_causal_conv1d_update(num_accepted_tokens=...) -> gated_delta_rule_mtp_packed -> RMSNormGated
# Run from the repository root:
#   PYTHONPATH="$PWD" /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python -u spikes/gdn-core-verify/probe.py
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import vllm_gaudi  # noqa: E402

assert vllm_gaudi.__file__.startswith(os.getcwd()), vllm_gaudi.__file__
from vllm.model_executor.layers.layernorm import RMSNormGated  # noqa: E402

from flashinfer_gaudi.gdn_decode import gated_delta_rule_mtp_packed  # noqa: E402
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update  # noqa: E402
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating  # noqa: E402


def verify_ref(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, conv_load,
               conv_offset, conv_store, ssm_load, ssm_store, query_len, scale, eps, gate):
    """Loop oracle: the T-node fused step over M3's state protocol (expanded conv slot, SSM checkpoint lattice)."""
    S, R, C = conv_state.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    B, T = mixed_qkv.shape[0], mixed_qkv.shape[1]
    H = (C - HV * D) // (2 * D)
    rep = HV // H
    out = torch.zeros(B, T, HV, D, dtype=torch.float32)
    w = conv_weight_t.float()
    for r in range(B):
        cl, off, cs, sl = int(conv_load[r]), int(conv_offset[r]), int(conv_store[r]), int(ssm_load[r])
        if not (0 <= cl < S and 0 <= cs < S and 0 <= sl < S):
            continue  # padding row: zeros, no writes
        n = max(0, min(int(query_len[r]), T))
        assert 0 <= off <= R - 3
        hist = conv_state[cl, off:off + 3].clone()  # [3, C], read before any write
        xs = mixed_qkv[r].to(conv_state.dtype).clone()
        xs[n:] = 0  # nodes past the row's query length are masked to zero, as the composed path does
        seq = torch.cat([hist, xs], 0)  # [3 + T, C]
        state = ssm_state[sl].float().clone()  # [HV, D, D]
        for t in range(n):
            conv = seq[t].float() * w[0] + seq[t + 1].float() * w[1] + seq[t + 2].float() * w[2] + seq[t + 3].float() * w[3]
            act = conv * torch.sigmoid(conv)
            for kh in range(H):
                q = act[kh * D:(kh + 1) * D]
                k = act[H * D + kh * D:H * D + (kh + 1) * D]
                q = q * torch.rsqrt((q * q).sum() + 1e-6) * scale
                k = k * torch.rsqrt((k * k).sum() + 1e-6)
                for j in range(rep):
                    vh = kh * rep + j
                    v = act[2 * H * D + vh * D:2 * H * D + (vh + 1) * D]
                    xg = a[r, t, vh].float() + dt_bias[vh].float()
                    sp = torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
                    decay = torch.exp(-torch.exp(A_log[vh].float()) * sp)
                    beta = torch.sigmoid(b[r, t, vh].float())
                    St = state[vh] * decay
                    St = St + ((v - St @ k) * beta)[:, None] * k[None, :]
                    state[vh] = St
                    o = St @ q
                    y = o * torch.rsqrt((o * o).mean() + eps) * norm_weight.float()
                    zz = z[r, t, vh * D:(vh + 1) * D].float()
                    out[r, t, vh] = y * (torch.sigmoid(zz) if gate == 1 else zz * torch.sigmoid(zz))
            st = int(ssm_store[r, t])
            if 0 <= st < S:
                ssm_state[st] = state.to(ssm_state.dtype)
        new = torch.zeros(R, C, dtype=conv_state.dtype)
        new[0:2] = hist[1:3]
        new[2:2 + T] = xs[:min(T, R - 2)]
        conv_state[cs] = new
    return out


def composed(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, lat_load,
             lat_store, accepted, query_len, align, scale, eps, gate):
    """M3's speculative decode path, as qwen3_5.py runs it (is_spec_decode, lattice_store_differs = align)."""
    B, T, C = mixed_qkv.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    qsl = torch.zeros(B + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(query_len, 0)
    g, beta = hpu_fused_gdn_gating(A_log, a.reshape(B * T, HV), b.reshape(B * T, HV), dt_bias)
    xc = hpu_causal_conv1d_update(x=mixed_qkv.reshape(B * T, C), conv_state=conv_state,
                                  weight=conv_weight_t.t().contiguous(), bias=None, activation="silu",
                                  conv_state_indices=lat_load[:, 0], num_accepted_tokens=accepted,
                                  query_start_loc=qsl, max_query_len=T,
                                  store_cache_indices=lat_store[:, 0] if align else None)
    o, _ = gated_delta_rule_mtp_packed(packed_qkv=xc.reshape(B, T, -1), log_decay=g.reshape(B, T, -1),
                                       beta=beta.reshape(B, T, -1), state_pool=ssm_state, state_indices=lat_load,
                                       num_accepted_tokens=accepted, query_lengths=query_len, scale=scale,
                                       use_qk_l2norm=True, store_indices=lat_store if align else None)
    o = o.reshape(B * T * HV, D)
    y = RMSNormGated.forward_static(o, z.reshape(B * T * HV, D), norm_weight, eps, o.dtype, None, True,
                                    "sigmoid" if gate == 1 else "silu")
    return y.reshape(B, T, HV, D).float()


def op_indices(lat_load, lat_store, accepted, query_len, align, R):
    """The layer's mapping from M3's lattice to the op's five index tensors (the W1 wiring)."""
    T = lat_load.shape[1]
    acc_conv = accepted.long().clamp(1, R - 2) - 1
    acc_ssm = accepted.long().clamp(1, T) - 1
    conv_load = lat_load[:, 0]
    conv_store = lat_store[:, 0] if align else lat_load[:, 0]
    ssm_load = lat_load.gather(1, acc_ssm[:, None]).squeeze(1)
    ssm_store = (lat_store if align else lat_load).clone()
    return conv_load, acc_conv.to(torch.int32), conv_store, ssm_load, ssm_store


def run_case(H, HV, dt, k, seed, steps, align, schedule, qlens, B=3):
    torch.manual_seed(seed)
    D = 128
    C = 2 * H * D + HV * D
    T = 1 + k
    R = 3 + k
    S = 2 + B * T * 2 + 1
    bf = lambda *s: torch.randn(*s).to(torch.bfloat16).to(dt)  # noqa: E731
    conv0 = bf(S, R, C)
    ssm0 = (torch.randn(S, HV, D, D) * 0.1).float()
    A_log = torch.randn(HV).float() * 0.5
    dt_bias = torch.randn(HV).float()
    wt = bf(4, C) * 0.5
    nw = bf(D) * 0.1 + 1
    c1, s1 = conv0.clone(), ssm0.clone()
    c2, s2 = conv0.clone(), ssm0.clone()
    worst = {"out": 0.0, "ssm": 0.0, "conv": 0.0, "mag": 0.0}
    accepted = torch.ones(B, dtype=torch.int32)
    for step in range(steps):
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
        # row 2 of step 1 is a padding row (load -1) when B > 2 and the schedule says so
        if schedule == "pad" and step == 1:
            lat_load = lat_load.clone()
            lat_load[B - 1] = -1
            lat_store = lat_store.clone()
            lat_store[B - 1] = -1
        cl, co, cs, sl, ss = op_indices(lat_load, lat_store, accepted, ql, align, R)
        # nodes past the query length store nothing (the composed path's token mask)
        ss = torch.where(torch.arange(T)[None, :] < ql[:, None], ss, torch.full_like(ss, -1))
        y1 = verify_ref(c1, s1, mixed, z, a, b, A_log, dt_bias, wt, nw, cl, co, cs, sl, ss, ql, D**-0.5, 1e-6, 1)
        y2 = composed(c2, s2, mixed, z, a, b, A_log, dt_bias, wt, nw, lat_load, lat_store, accepted, ql, align,
                      D**-0.5, 1e-6, 1)
        real_rows = [r for r in range(B) if int(lat_load[r, 0]) >= 0]
        for r in real_rows:
            n = int(ql[r])
            worst["out"] = max(worst["out"], (y1[r, :n] - y2[r, :n]).abs().max().item())
            worst["mag"] = max(worst["mag"], y1[r, :n].abs().max().item())
            for t in range(n):
                st = int(ss[r, t])
                worst["ssm"] = max(worst["ssm"], (s1[st] - s2[st]).abs().max().item())
            # visible conv rows: the history window the next step can read (rows 0 .. n + 1)
            cslot = int(cs[r])
            worst["conv"] = max(worst["conv"], (c1[cslot, :n + 2].float() - c2[cslot, :n + 2].float()).abs().max().item())
        # the next step's accepted counts (a fixed schedule, threaded forward)
        nxt = [[k + 1, 1, 2], [2, k + 1, 1], [1, 2, k + 1], [k + 1, k + 1, 1]][step % 4]
        accepted = torch.tensor([min(nxt[r], int(ql[r])) for r in range(B)], dtype=torch.int32)
    return worst


if __name__ == "__main__":
    k = 3
    full = [[k + 1] * 3] * 4
    trimmed = [[k + 1, 2, k + 1], [k + 1, k + 1, 1], [3, k + 1, k + 1], [k + 1] * 3]
    for H, HV in [(16, 48), (8, 24), (4, 12), (2, 6)]:
        for dt in (torch.float32, torch.bfloat16):
            for align in (False, True):
                for name, ql in (("full", full), ("trimmed", trimmed)):
                    wv = run_case(H, HV, dt, k, 0, 4, align, "pad" if name == "trimmed" else "none", ql)
                    print(f"H={H:2d} HV={HV:2d} {str(dt)[6:]:8s} align={int(align)} {name:7s} "
                          f"out={wv['out']:.2e} (|out| {wv['mag']:.1f}) ssm={wv['ssm']:.2e} conv={wv['conv']:.2e}",
                          flush=True)


def transition_case(H, HV, dt, k, seed, B=3):
    """A plain decode step after a verification (qwen3_5.py is_dflash_transition): T = 1 through the same op."""
    from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_recurrent_gated_delta_rule
    torch.manual_seed(seed)
    D = 128
    C = 2 * H * D + HV * D
    R = 3 + k
    S = 2 + B * (1 + k) * 2 + 1
    bf = lambda *s: torch.randn(*s).to(torch.bfloat16).to(dt)  # noqa: E731
    conv0, ssm0 = bf(S, R, C), (torch.randn(S, HV, D, D) * 0.1).float()
    A_log, dt_bias, wt, nw = torch.randn(HV).float() * 0.5, torch.randn(HV).float(), bf(4, C) * 0.5, bf(D) * 0.1 + 1
    mixed, z, a, b = bf(B, C), bf(B, HV * D), bf(B, HV), bf(B, HV)
    accepted = torch.tensor([k + 1, 1, 2], dtype=torch.int32)
    load = torch.tensor([4, 9, 14], dtype=torch.int32)   # the accepted checkpoints (SSM)
    store = torch.tensor([1, 6, 11], dtype=torch.int32)  # the requests' base rows (conv read and write, SSM write)
    qsl = torch.arange(B + 1, dtype=torch.int32)
    c1, s1, c2, s2 = conv0.clone(), ssm0.clone(), conv0.clone(), ssm0.clone()
    co = (accepted.long().clamp(1, R - 2) - 1).to(torch.int32)
    y1 = verify_ref(c1, s1, mixed[:, None], z[:, None], a[:, None], b[:, None], A_log, dt_bias, wt, nw, store, co, store,
                    load, store[:, None], torch.ones(B, dtype=torch.int32), D**-0.5, 1e-6, 1)[:, 0]
    g, beta = hpu_fused_gdn_gating(A_log, a, b, dt_bias)
    xc = hpu_causal_conv1d_update(x=mixed, conv_state=c2, weight=wt.t().contiguous(), bias=None, activation="silu",
                                  conv_state_indices=store, num_accepted_tokens=accepted, query_start_loc=qsl,
                                  max_query_len=1)
    q, kk, v = torch.split(xc, [H * D, H * D, HV * D], dim=-1)
    o, _ = hpu_fused_recurrent_gated_delta_rule(q=q.reshape(1, B, H, D), k=kk.reshape(1, B, H, D),
                                                v=v.reshape(1, B, HV, D), g=g, beta=beta, scale=D**-0.5,
                                                initial_state=s2, inplace_final_state=True, cu_seqlens=qsl,
                                                ssm_state_indices=load, use_qk_l2norm_in_kernel=True,
                                                ssm_store_indices=store)
    y2 = RMSNormGated.forward_static(o.reshape(B * HV, D), z.reshape(B * HV, D), nw, 1e-6, o.dtype, None, True,
                                     "sigmoid").reshape(B, HV, D).float()
    sl = store.long()
    return ((y1 - y2).abs().max().item(), (s1[sl] - s2[sl]).abs().max().item(),
            (c1[sl, :3].float() - c2[sl, :3].float()).abs().max().item())


if __name__ == "__main__" and os.environ.get("PROBE_TRANSITION", "1") == "1":
    for H, HV in [(16, 48), (8, 24), (4, 12), (2, 6)]:
        for dt in (torch.float32, torch.bfloat16):
            o, s, c = transition_case(H, HV, dt, 3, 1)
            print(f"transition H={H:2d} HV={HV:2d} {str(dt)[6:]:8s} out={o:.2e} ssm={s:.2e} conv={c:.2e}", flush=True)
