# SPDX-License-Identifier: Apache-2.0
"""Torch reference for GDN multi-token verification over a draft chain or tree (Phase 2 contract).

Runs the fused GDN decode step of ``gdn_core_decode_reference`` (T = 1 per call) over ``T`` draft nodes per row,
where each node's input state is its **parent** node's output state: node 0 of row ``r`` loads slot
``load_indices[r]`` (conv history and SSM state); node ``t >= 1`` loads the state that node ``parent_idx[r, t]``
produced, with ``parent_idx[r, 0] == -1`` and ``0 <= parent_idx[r, t] < t`` for ``t >= 1``.  A row whose load slot
is out of range is padding: all its outputs are zero and it stores nothing.  Node ``t``'s state is stored where
``store_indices[r, t]`` is in range; node ``t``'s output does not depend on whether its state is stored.  M3's
chain is ``parent_idx[r, t] = t - 1``; a draft tree needs no kernel change, only a different ``parent_idx`` and
per-node stores.  Contract: ``docs/design/flash-next/gdn-core-decode.md``, "Phase 2 contract".  This is an
executable specification, not a fast path; imports on CPU with no Habana packages.
"""

from __future__ import annotations

import torch

from flashinfer_gaudi.gdn_core_decode import _require, gdn_core_decode_reference


def gdn_core_verify_reference(
    conv_state: torch.Tensor,  # [S, 3, C], mutated in place (stored nodes only)
    ssm_state: torch.Tensor,  # [S, HV, D, D] fp32, mutated in place (stored nodes only)
    mixed_qkv: torch.Tensor,  # [B, T, C]
    z: torch.Tensor,  # [B, T, HV * D]
    a: torch.Tensor,  # [B, T, HV]
    b: torch.Tensor,  # [B, T, HV]
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32
    conv_weight_t: torch.Tensor,  # [4, C]
    norm_weight: torch.Tensor,  # [D]
    parent_idx: torch.Tensor,  # [B, T] int32: parent_idx[:, 0] == -1; 0 <= parent_idx[r, t] < t for t >= 1
    load_indices: torch.Tensor,  # [B] int32: the slot node 0 loads (conv history and SSM state)
    store_indices: torch.Tensor,  # [B, T] int32: where node t's state is stored; out of range = not stored
    scale: float,
    eps: float,
    gate_activation: int,
) -> torch.Tensor:  # [B, T, HV, D] fp32
    """The fused GDN decode step over T draft nodes per row, exactly as in the Phase 2 contract.

    One ``gdn_core_decode_reference`` call per node ``t`` over all rows, in increasing ``t``: node 0 loads
    ``load_indices``, node ``t >= 1`` loads the scratch slot of its parent, every node stores to its own scratch
    slot.  Rows with an out-of-range load slot are padding: zero outputs, no stores, and their ``parent_idx``
    values are not checked.  After the loop the stored nodes' scratch slots are copied into ``conv_state`` /
    ``ssm_state`` at ``store_indices``.
    """
    _require(conv_state.dim() == 3 and conv_state.shape[1] == 3,
             f"conv_state must be [S, 3, C], got shape {tuple(conv_state.shape)}")
    _require(ssm_state.dim() == 4 and ssm_state.shape[2] == ssm_state.shape[3],
             f"ssm_state must be [S, HV, D, D], got shape {tuple(ssm_state.shape)}")
    S, _, C = conv_state.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    _require(ssm_state.shape[0] == S, f"ssm_state slot count {ssm_state.shape[0]} != conv_state slot count {S}")
    _require(HV > 0 and D > 0, f"ssm_state must have HV > 0 and D > 0, got shape {tuple(ssm_state.shape)}")
    _require(mixed_qkv.dim() == 3 and mixed_qkv.shape[2] == C,
             f"mixed_qkv must be [B, T, {C}], got shape {tuple(mixed_qkv.shape)}")
    B, T = mixed_qkv.shape[0], mixed_qkv.shape[1]
    _require(T >= 1, f"T must be >= 1, got {T}")
    _require(z.dim() == 3 and tuple(z.shape) == (B, T, HV * D),
             f"z must be [B, T, HV*D] = ({B}, {T}, {HV * D}), got {tuple(z.shape)}")
    _require(a.dim() == 3 and tuple(a.shape) == (B, T, HV),
             f"a must be [B, T, HV] = ({B}, {T}, {HV}), got {tuple(a.shape)}")
    _require(b.dim() == 3 and tuple(b.shape) == (B, T, HV),
             f"b must be [B, T, HV] = ({B}, {T}, {HV}), got {tuple(b.shape)}")
    _require(A_log.dim() == 1 and tuple(A_log.shape) == (HV,),
             f"A_log must be [HV] = ({HV},), got {tuple(A_log.shape)}")
    _require(dt_bias.dim() == 1 and tuple(dt_bias.shape) == (HV,),
             f"dt_bias must be [HV] = ({HV},), got {tuple(dt_bias.shape)}")
    _require(conv_weight_t.dim() == 2 and tuple(conv_weight_t.shape) == (4, C),
             f"conv_weight_t must be [4, C] = (4, {C}), got {tuple(conv_weight_t.shape)}")
    _require(norm_weight.dim() == 1 and tuple(norm_weight.shape) == (D,),
             f"norm_weight must be [D] = ({D},), got {tuple(norm_weight.shape)}")
    _require(parent_idx.dim() == 2 and tuple(parent_idx.shape) == (B, T),
             f"parent_idx must be [B, T] = ({B}, {T}), got {tuple(parent_idx.shape)}")
    _require(load_indices.dim() == 1 and tuple(load_indices.shape) == (B,),
             f"load_indices must be [B] = ({B},), got {tuple(load_indices.shape)}")
    _require(store_indices.dim() == 2 and tuple(store_indices.shape) == (B, T),
             f"store_indices must be [B, T] = ({B}, {T}), got {tuple(store_indices.shape)}")

    load = load_indices.long()
    row_valid = (load >= 0) & (load < S)
    p = parent_idx.long()
    _require(bool((p[row_valid, 0] == -1).all()),
             f"parent_idx[r, 0] must be -1 on non-padding rows, got {parent_idx[row_valid, 0].tolist()}")
    if T > 1:
        tvec = torch.arange(1, T, dtype=p.dtype, device=p.device)
        good = (p[:, 1:] >= 0) & (p[:, 1:] < tvec)
        _require(bool(good[row_valid].all()),
                 f"parent_idx[r, t] must satisfy 0 <= parent_idx[r, t] < t for t >= 1 on non-padding rows, "
                 f"got {parent_idx.tolist()}")

    # Scratch pool: the real slots, then one slot per node, node (r, t) at S + r * T + t.  Every node reads its
    # parent's scratch slot (already written, since parent < t) and writes its own; rows 0..S-1 stay untouched
    # until the final store below, so the caller's pools are read-only for the whole loop.
    device = ssm_state.device
    conv = torch.cat([conv_state, conv_state.new_zeros(B * T, 3, C)], dim=0)
    ssm = torch.cat([ssm_state, ssm_state.new_zeros(B * T, HV, D, D)], dim=0)
    out = torch.zeros(B, T, HV, D, dtype=torch.float32, device=device)
    scratch = S + torch.arange(B, dtype=torch.long, device=device) * T
    pad = torch.full((B,), -1, dtype=torch.long, device=device)
    load0 = torch.where(row_valid, load, pad)
    for t in range(T):
        load_t = load0 if t == 0 else torch.where(row_valid, scratch + p[:, t], pad)
        out[:, t] = gdn_core_decode_reference(conv, ssm, mixed_qkv[:, t], z[:, t], a[:, t], b[:, t], A_log, dt_bias,
                                              conv_weight_t, norm_weight, load_t, scratch + t, scale, eps,
                                              gate_activation)

    store = store_indices.long()
    stored = row_valid[:, None] & (store >= 0) & (store < S)
    rs, ts = torch.nonzero(stored, as_tuple=False).unbind(1)
    if rs.numel() > 0:
        dest = store[rs, ts]
        conv_state.index_copy_(0, dest, conv[S + rs * T + ts])
        ssm_state.index_copy_(0, dest, ssm[S + rs * T + ts])
    return out
