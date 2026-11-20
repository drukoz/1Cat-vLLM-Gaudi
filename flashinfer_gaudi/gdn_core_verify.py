# SPDX-License-Identifier: Apache-2.0
"""GDN multi-token verification over a draft chain (Phase 2 contract, version 2).

``gdn_core_verify_reference`` and the ``torch.library`` op ``flashinfer_gaudi::gdn_core_verify`` run the fused
GDN decode step of ``gdn_core_decode_reference`` (T = 1 per node) over ``T`` draft nodes per row, over M3's
state protocol: one **expanded conv slot** per request (``[S, R, C]``, ``R >= T + 2``; the slot's 3-row history
window starts at ``conv_offset = num_accepted - 1``) and a per-node **SSM checkpoint lattice** (node 0 starts
from the accepted checkpoint at ``ssm_load``, node ``t``'s state is stored at ``ssm_store_indices[r, t]``).
Activations are token-major: node ``t`` of row ``r`` is row ``r * T + t``.  A row with an out-of-range slot, an
offset outside ``[0, R - 3]`` or a query length below 1 is padding: zero outputs, no writes.  Nodes past a row's
query length are masked: zero output, no state change, nothing stored, their conv input rows are stored as zeros.
``T`` is at most ``GDN_CORE_VERIFY_MAX_NODES`` (6).  The reference (so the CPU kernel too) and the ``gdn_core_verify``
wrapper raise ``ValueError`` for a larger ``T``; the HPU op and the host glue reject it as well, so it never reaches
the Gaudi2 kernel.  The layer checks ``T`` first and takes the torch path.
Contract: ``docs/design/flash-next/gdn-core-decode.md``, "Phase 2 contract, version 2".  This is an executable
specification, not a fast path; imports on CPU with no Habana packages.

``gdn_core_verify_tree_reference`` is version 1 of the contract (RUK-2915, per-node conv checkpoints and a
``parent_idx``): kept as the executable form of a future draft tree, not the op.
"""

from __future__ import annotations

import enum

import torch

from flashinfer_gaudi.gdn_core_decode import (_DISPATCH_KEYS_BY_DEVICE_TYPE, _LIBRARY, _require,
                                              gdn_core_decode_reference)


# The largest T (= 1 + k draft tokens, so k <= 5) of ``gdn_core_verify``.  The Gaudi2 kernel keeps its per-node
# vectors in fixed arrays in the 16 KiB of vector local memory a kernel with lookup-table functions has, sized for
# this many nodes.  The same number: ``kGdnCoreVerifyMaxNodes`` (``csrc/gdn_fused/gdn_core_decode_params.h``),
# ``GDN_VERIFY_MAX_NODES`` (``csrc/gdn_fused/kernels/gdn_core_verify_gaudi2.c``) and ``MAX_NODES``
# (``tests/unit_tests/gdn_core_decode/verify_sim_case.py``).
GDN_CORE_VERIFY_MAX_NODES = 6


class GdnCoreVerifyMode(enum.IntEnum):
    """VLLM_HPU_GDN_CORE_VERIFY values (next free: 2)."""
    OFF = 0
    FUSED = 1


def gdn_core_verify_tree_reference(
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
    """Version 1 of the Phase 2 contract, draft trees; not the op.

    The fused GDN decode step over T draft nodes per row, one parent link per node.  Kept as the executable form
    of a future draft tree (the runner verifies chains; see the module docstring).

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
    _require(A_log.dim() == 1 and tuple(A_log.shape) == (HV, ),
             f"A_log must be [HV] = ({HV},), got {tuple(A_log.shape)}")
    _require(dt_bias.dim() == 1 and tuple(dt_bias.shape) == (HV, ),
             f"dt_bias must be [HV] = ({HV},), got {tuple(dt_bias.shape)}")
    _require(conv_weight_t.dim() == 2 and tuple(conv_weight_t.shape) == (4, C),
             f"conv_weight_t must be [4, C] = (4, {C}), got {tuple(conv_weight_t.shape)}")
    _require(norm_weight.dim() == 1 and tuple(norm_weight.shape) == (D, ),
             f"norm_weight must be [D] = ({D},), got {tuple(norm_weight.shape)}")
    _require(parent_idx.dim() == 2 and tuple(parent_idx.shape) == (B, T),
             f"parent_idx must be [B, T] = ({B}, {T}), got {tuple(parent_idx.shape)}")
    _require(load_indices.dim() == 1 and tuple(load_indices.shape) == (B, ),
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
        _require(
            bool(good[row_valid].all()),
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
    pad = torch.full((B, ), -1, dtype=torch.long, device=device)
    load0 = torch.where(row_valid, load, pad)
    for t in range(T):
        load_t = load0 if t == 0 else torch.where(row_valid, scratch + p[:, t], pad)
        out[:,
            t] = gdn_core_decode_reference(conv, ssm, mixed_qkv[:, t], z[:, t], a[:, t], b[:, t], A_log, dt_bias,
                                           conv_weight_t, norm_weight, load_t, scratch + t, scale, eps, gate_activation)

    store = store_indices.long()
    stored = row_valid[:, None] & (store >= 0) & (store < S)
    rs, ts = torch.nonzero(stored, as_tuple=False).unbind(1)
    if rs.numel() > 0:
        dest = store[rs, ts]
        conv_state.index_copy_(0, dest, conv[S + rs * T + ts])
        ssm_state.index_copy_(0, dest, ssm[S + rs * T + ts])
    return out


def gdn_core_verify_reference(
    conv_state: torch.Tensor,  # [S, R, C], mutated in place (real rows' store slots)
    ssm_state: torch.Tensor,  # [S, HV, D, D] fp32, mutated in place (stored nodes)
    mixed_qkv: torch.Tensor,  # [B*T, C] token-major: node t of row r is row r * T + t
    z: torch.Tensor,  # [B*T, HV * D]
    a: torch.Tensor,  # [B*T, HV]
    b: torch.Tensor,  # [B*T, HV]
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32
    conv_weight_t: torch.Tensor,  # [4, C]
    norm_weight: torch.Tensor,  # [D]
    row_indices: torch.Tensor,  # [B, 5] int32: conv load slot, conv offset, conv store slot, SSM load slot, query len
    ssm_store_indices: torch.Tensor,  # [B, T] int32: where node t's SSM state is stored; out of range = not stored
    scale: float,
    eps: float,
    gate_activation: int,
) -> torch.Tensor:  # [B*T, HV, D] fp32
    """The fused GDN step over T chain nodes per row, exactly as in the Phase 2 contract, version 2.

    Vectorized over rows and heads, with a Python loop over the T nodes: each node is one
    ``gdn_core_decode_reference`` call, so the arithmetic and rounding points are exactly phase 1's.  Two scratch
    pools of ``B * (T + 1)`` slots carry the state from node to node (node ``t`` loads slot ``r * (T + 1) + t`` and
    stores to ``r * (T + 1) + t + 1``); the caller's pools are read before any write and written only after the
    loop.  Padding rows and nodes past a row's query length are zero outputs and write nothing.
    """
    _require(conv_state.dim() == 3, f"conv_state must be [S, R, C], got shape {tuple(conv_state.shape)}")
    _require(ssm_state.dim() == 4 and ssm_state.shape[2] == ssm_state.shape[3],
             f"ssm_state must be [S, HV, D, D], got shape {tuple(ssm_state.shape)}")
    S, R, C = conv_state.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    _require(ssm_state.shape[0] == S, f"ssm_state slot count {ssm_state.shape[0]} != conv_state slot count {S}")
    _require(HV > 0 and D > 0, f"ssm_state must have HV > 0 and D > 0, got shape {tuple(ssm_state.shape)}")
    _require(row_indices.dim() == 2 and tuple(row_indices.shape[1:]) == (5, ),
             f"row_indices must be [B, 5], got shape {tuple(row_indices.shape)}")
    B = row_indices.shape[0]
    _require(ssm_store_indices.dim() == 2 and ssm_store_indices.shape[0] == B,
             f"ssm_store_indices must be [B, T] = ({B}, T), got {tuple(ssm_store_indices.shape)}")
    T = ssm_store_indices.shape[1]
    _require(1 <= T <= GDN_CORE_VERIFY_MAX_NODES, f"T must be in [1, {GDN_CORE_VERIFY_MAX_NODES}], got {T}")
    _require(T + 2 <= R, f"T + 2 = {T + 2} must be <= conv_state rows R = {R}")
    _require(mixed_qkv.dim() == 2 and tuple(mixed_qkv.shape) == (B * T, C),
             f"mixed_qkv must be [B*T, C] = ({B * T}, {C}), got shape {tuple(mixed_qkv.shape)}")
    _require(z.dim() == 2 and tuple(z.shape) == (B * T, HV * D),
             f"z must be [B*T, HV*D] = ({B * T}, {HV * D}), got {tuple(z.shape)}")
    _require(a.dim() == 2 and tuple(a.shape) == (B * T, HV),
             f"a must be [B*T, HV] = ({B * T}, {HV}), got {tuple(a.shape)}")
    _require(b.dim() == 2 and tuple(b.shape) == (B * T, HV),
             f"b must be [B*T, HV] = ({B * T}, {HV}), got {tuple(b.shape)}")
    _require(A_log.dim() == 1 and tuple(A_log.shape) == (HV, ),
             f"A_log must be [HV] = ({HV},), got {tuple(A_log.shape)}")
    _require(dt_bias.dim() == 1 and tuple(dt_bias.shape) == (HV, ),
             f"dt_bias must be [HV] = ({HV},), got {tuple(dt_bias.shape)}")
    _require(conv_weight_t.dim() == 2 and tuple(conv_weight_t.shape) == (4, C),
             f"conv_weight_t must be [4, C] = (4, {C}), got {tuple(conv_weight_t.shape)}")
    _require(norm_weight.dim() == 1 and tuple(norm_weight.shape) == (D, ),
             f"norm_weight must be [D] = ({D},), got {tuple(norm_weight.shape)}")

    device = ssm_state.device
    cl, off, cs, sl, n = (v.contiguous() for v in row_indices.long().unbind(1))
    real = (cl >= 0) & (cl < S) & (cs >= 0) & (cs < S) & (sl >= 0) & (sl < S) & (off >= 0) & (off <= R - 3) & (n >= 1)
    n = n.clamp(0, T)

    # Read every input slot before any write below: the 3-row conv history window, the SSM start state, and the
    # (masked) T input rows, all gathered per row; padding rows read and keep zeros.
    hist = torch.zeros(B, 3, C, dtype=conv_state.dtype, device=device)
    init = torch.zeros(B, HV, D, D, dtype=torch.float32, device=device)
    load_rows = torch.nonzero(real).reshape(-1)
    if load_rows.numel() > 0:
        r_cl, r_off, r_sl = cl[load_rows], off[load_rows], sl[load_rows]
        slot = conv_state.index_select(0, r_cl)  # [N, R, C]
        cols = (r_off[:, None] + torch.arange(3, dtype=torch.long, device=device)[None, :])  # [N, 3]
        hist.index_copy_(0, load_rows, slot.gather(1, cols[:, :, None].expand(-1, 3, C)))  # [N, 3, C], rows off..off+2
        init.index_copy_(0, load_rows, ssm_state.index_select(0, r_sl).float())
    xs = mixed_qkv.view(B, T, C).to(conv_state.dtype)  # [B, T, C]
    xs = torch.where((torch.arange(T, dtype=torch.long, device=device)[None, :] < n[:, None])[:, :, None], xs,
                     xs.new_zeros(()))  # nodes t >= n are zero, as the composed path masks them
    seq = torch.cat([hist, xs], dim=1)  # [B, 3 + T, C]

    # Scratch pools: node (r, t) at slot r * (T + 1) + t, so one index serves conv and SSM together.  The init
    # states start at each row's slot 0; the conv windows are written per node inside the loop.
    scr_ssm = torch.zeros(B * (T + 1), HV, D, D, dtype=torch.float32, device=device)
    scr_ssm.index_copy_(0, torch.arange(B, dtype=torch.long, device=device) * (T + 1), init)
    scr_conv = torch.zeros(B * (T + 1), 3, C, dtype=conv_state.dtype, device=device)

    out = torch.zeros(B, T, HV, D, dtype=torch.float32, device=device)
    r = torch.arange(B, dtype=torch.long, device=device)
    z4, a4, b4 = z.view(B, T, HV * D), a.view(B, T, HV), b.view(B, T, HV)
    for t in range(T):
        slot_t = r * (T + 1) + t  # node t's slot in both scratch pools
        scr_conv.index_copy_(0, slot_t, seq[:, t:t + 3])  # the node's history window seq[t : t + 3]
        live = real & (t < n)  # padding rows and nodes t >= n load -1: zero output, no write
        load = torch.where(live, slot_t, torch.full_like(slot_t, -1)).to(torch.int32)
        store = torch.where(live, slot_t + 1, torch.full_like(slot_t, -1)).to(torch.int32)
        out[:, t] = gdn_core_decode_reference(scr_conv, scr_ssm, seq[:, t + 3].to(mixed_qkv.dtype), z4[:, t], a4[:, t],
                                              b4[:, t], A_log, dt_bias, conv_weight_t, norm_weight, load, store, scale,
                                              eps, gate_activation)
        # Node t's state lands in scr_ssm slot r * (T + 1) + t + 1, which node t + 1 loads.  Its conv writes into
        # scr_conv are scratch and ignored (the window for node t + 1 is written over them).

    # After the loop, write the caller's pools.  SSM: node t's state to ssm_store_indices[r, t] when in range.
    sst = ssm_store_indices.long()
    node_real = torch.arange(T, dtype=torch.long, device=device)[None, :] < n[:, None]
    stored = real[:, None] & node_real & (sst >= 0) & (sst < S)
    rs, ts = torch.nonzero(stored, as_tuple=False).unbind(1)
    if rs.numel() > 0:
        ssm_state.index_copy_(0, sst[rs, ts], scr_ssm[rs * (T + 1) + ts + 1])
    # Conv: the expanded slot's R rows = 2 history rows, the T input rows (zeros past n), then zeros.
    if load_rows.numel() > 0:
        new = torch.zeros(len(load_rows), R, C, dtype=conv_state.dtype, device=device)
        lr = load_rows
        new[:, 0:2] = hist[lr, 1:3]
        new[:, 2:2 + T] = xs[lr]
        conv_state.index_copy_(0, cs[lr], new)
    return out.view(B * T, HV, D)


# The torch.library op wrapper.  This module is the only place that defines the schema and the Meta (fake) kernel;
# the HPU kernel comes from the C++ extension, which must not define them.  Same _LIBRARY fragment as
# gdn_core_decode.

_LIBRARY.define("gdn_core_verify("
                "Tensor(a!) conv_state, Tensor(b!) ssm_state, Tensor mixed_qkv, Tensor z, Tensor a, Tensor b, "
                "Tensor A_log, Tensor dt_bias, Tensor conv_weight_t, Tensor norm_weight, "
                "Tensor row_indices, Tensor ssm_store_indices, "
                "float scale, float eps, int gate_activation) -> Tensor")


def _gdn_core_verify_cpu(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight,
                         row_indices, ssm_store_indices, scale, eps, gate_activation):
    return gdn_core_verify_reference(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t,
                                     norm_weight, row_indices, ssm_store_indices, scale, eps, gate_activation)


_LIBRARY.impl("gdn_core_verify", _gdn_core_verify_cpu, "CPU")


@torch.library.register_fake("flashinfer_gaudi::gdn_core_verify", lib=_LIBRARY)
def _gdn_core_verify_fake(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight,
                          row_indices, ssm_store_indices, scale, eps, gate_activation):
    # Allocation only: no mutation, no data access.
    B_T = mixed_qkv.shape[0]
    HV = ssm_state.shape[1]
    D = ssm_state.shape[2]
    return mixed_qkv.new_empty((B_T, HV, D), dtype=torch.float32)


def gdn_core_verify(
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight_t: torch.Tensor,
    norm_weight: torch.Tensor,
    row_indices: torch.Tensor,
    ssm_store_indices: torch.Tensor,
    scale: float,
    eps: float,
    gate_activation: int,
) -> torch.Tensor:
    """The fused GDN verification step as ``torch.ops.flashinfer_gaudi.gdn_core_verify``.

    Mutates ``conv_state`` and ``ssm_state`` in place, returns ``[B*T, HV, D]`` fp32; semantics are exactly
    ``gdn_core_verify_reference`` (contract: ``docs/design/flash-next/gdn-core-decode.md``, "Phase 2 contract,
    version 2").  ``T`` outside ``[1, GDN_CORE_VERIFY_MAX_NODES]`` raises ``ValueError`` here, on every device,
    before the op is dispatched.
    """
    if ssm_store_indices.dim() == 2:
        T = ssm_store_indices.shape[1]
        _require(1 <= T <= GDN_CORE_VERIFY_MAX_NODES, f"T must be in [1, {GDN_CORE_VERIFY_MAX_NODES}], got {T}")
    return torch.ops.flashinfer_gaudi.gdn_core_verify(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias,
                                                      conv_weight_t, norm_weight, row_indices, ssm_store_indices, scale,
                                                      eps, int(gate_activation))


def gdn_core_verify_available(device: torch.device | str) -> bool:
    """True when ``flashinfer_gaudi::gdn_core_verify`` has a kernel for the device's type."""
    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    dispatch_key = _DISPATCH_KEYS_BY_DEVICE_TYPE.get(device_type)
    if dispatch_key is None:
        return False
    return torch._C._dispatch_has_kernel_for_dispatch_key("flashinfer_gaudi::gdn_core_verify", dispatch_key)
