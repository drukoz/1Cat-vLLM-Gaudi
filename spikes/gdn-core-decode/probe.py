# SPDX-License-Identifier: Apache-2.0
# CPU probe of the gdn_core_decode contract (docs/design/flash-next/gdn-core-decode.md) against the composed
# torch path. Run from the repository root:
#   PYTHONPATH="$PWD" /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python spikes/gdn-core-decode/probe.py
import os, sys, torch
sys.path.insert(0, os.getcwd())
import vllm_gaudi
assert vllm_gaudi.__file__.startswith(os.getcwd()), vllm_gaudi.__file__
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating, hpu_fused_recurrent_gated_delta_rule
from vllm.model_executor.layers.layernorm import RMSNormGated

def ref(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, load, store, scale, eps, gate):
    S = ssm_state.shape[0]; B, C = mixed_qkv.shape; HV, D = ssm_state.shape[1], ssm_state.shape[2]
    H = (C - HV * D) // (2 * D); rep = HV // H
    out = torch.zeros(B, HV, D, dtype=torch.float32)
    for r in range(B):
        ls, ss = int(load[r]), int(store[r])
        if not (0 <= ls < S and 0 <= ss < S):
            continue
        hist = conv_state[ls, 0:3, :].float(); x = mixed_qkv[r].float(); w = conv_weight_t.float()
        conv = hist[0] * w[0] + hist[1] * w[1] + hist[2] * w[2] + x * w[3]
        act = conv * torch.sigmoid(conv)
        new_hist = torch.stack([conv_state[ls, 1], conv_state[ls, 2], mixed_qkv[r].to(conv_state.dtype)])
        for kh in range(H):
            q = act[kh * D:(kh + 1) * D]; k = act[H * D + kh * D:H * D + (kh + 1) * D]
            q = q * torch.rsqrt((q * q).sum() + 1e-6) * scale; k = k * torch.rsqrt((k * k).sum() + 1e-6)
            for j in range(rep):
                vh = kh * rep + j
                v = act[2 * H * D + vh * D:2 * H * D + (vh + 1) * D]
                xg = a[r, vh].float() + dt_bias[vh].float()
                sp = torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
                decay = torch.exp(-torch.exp(A_log[vh].float()) * sp); beta = torch.sigmoid(b[r, vh].float())
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

def composed(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, load, store, scale, eps, gate):
    B, C = mixed_qkv.shape; HV, D = ssm_state.shape[1], ssm_state.shape[2]; H = (C - HV * D) // (2 * D)
    qsl = torch.arange(B + 1, dtype=torch.int32)
    g, beta = hpu_fused_gdn_gating(A_log, a, b, dt_bias)
    xc = hpu_causal_conv1d_update(x=mixed_qkv, conv_state=conv_state, weight=conv_weight_t.t().contiguous(), bias=None,
        activation="silu", conv_state_indices=load, query_start_loc=qsl, store_cache_indices=store)
    q, k, v = torch.split(xc, [H * D, H * D, HV * D], dim=-1)
    q = q.reshape(1, B, H, D); k = k.reshape(1, B, H, D); v = v.reshape(1, B, HV, D)
    o, _ = hpu_fused_recurrent_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=ssm_state,
        inplace_final_state=True, cu_seqlens=qsl, ssm_state_indices=load, use_qk_l2norm_in_kernel=True, ssm_store_indices=store)
    o = o.reshape(B * HV, D)
    y = RMSNormGated.forward_static(o, z.reshape(B * HV, D), norm_weight, eps, o.dtype, None, True, "sigmoid" if gate == 1 else "silu")
    return y.reshape(B, HV, D).float()

torch.manual_seed(0)
for H, HV in [(16, 48), (8, 24), (4, 12), (2, 6)]:
  for dt in (torch.float32, torch.bfloat16):
    D = 128; C = 2 * H * D + HV * D; B = 4; S = 9
    bf = lambda *s: torch.randn(*s).to(torch.bfloat16).to(dt)
    conv0 = bf(S, 3, C); ssm0 = (torch.randn(S, HV, D, D) * 0.1).float()
    mixed = bf(B, C); z = bf(B, HV * D); a = bf(B, HV); b = bf(B, HV)
    A_log = torch.randn(HV).float() * 0.5; dt_bias = torch.randn(HV).float(); wt = (bf(4, C) * 0.5)
    nw = bf(D) * 0.1 + 1
    load = torch.tensor([1, 3, 5, -1], dtype=torch.int32); store = torch.tensor([2, 3, 6, -1], dtype=torch.int32)
    c1, s1 = conv0.clone(), ssm0.clone(); c2, s2 = conv0.clone(), ssm0.clone()
    y1 = ref(c1, s1, mixed, z, a, b, A_log, dt_bias, wt, nw, load, store, D ** -0.5, 1e-6, 1)
    y2 = composed(c2, s2, mixed, z, a, b, A_log, dt_bias, wt, nw, load, store, D ** -0.5, 1e-6, 1)
    real = [0, 1, 2]; slots = [2, 3, 6]
    print(H, HV, dt, "out", (y1[real] - y2[real]).abs().max().item(), "outmag", y1[real].abs().max().item(),
          "ssm", (s1[slots] - s2[slots]).abs().max().item(), "conv", (c1[slots].float() - c2[slots].float()).abs().max().item(),
          "pad-out-zero", y1[3].abs().max().item(), "untouched", torch.equal(s1[[0,4,7]], ssm0[[0,4,7]]))
