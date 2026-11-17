# SPDX-License-Identifier: Apache-2.0
"""Torch reference for the fused GDN core decode step (one token per row, T = 1).

Vectorized (no per-row / per-head Python loops) implementation of the op contract in
``docs/design/flash-next/gdn-core-decode.md``: causal conv1d update + SiLU, decay/beta gates,
Q/K L2-norm and scale, the delta-rule recurrence, and the gated RMSNorm.  This is the portable
executable specification for the Gaudi2 TPC kernel; the ``torch.library`` op wrapper is defined
separately.  Imports on CPU with no Habana packages.
"""

from __future__ import annotations

import enum

import torch


class GdnGateActivation(enum.IntEnum):
    SILU = 0
    SIGMOID = 1


class GdnCoreDecodeMode(enum.IntEnum):
    """VLLM_HPU_GDN_CORE_DECODE values (next free: 2)."""
    OFF = 0
    FUSED = 1


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def gdn_core_decode_reference(
    conv_state: torch.Tensor,  # [S, 3, C], mutated in place
    ssm_state: torch.Tensor,  # [S, HV, D, D] fp32, mutated in place
    mixed_qkv: torch.Tensor,  # [B, C]
    z: torch.Tensor,  # [B, HV * D]
    a: torch.Tensor,  # [B, HV]
    b: torch.Tensor,  # [B, HV]
    A_log: torch.Tensor,  # [HV] fp32
    dt_bias: torch.Tensor,  # [HV] fp32
    conv_weight_t: torch.Tensor,  # [4, C]
    norm_weight: torch.Tensor,  # [D]
    load_indices: torch.Tensor,  # [B] int32
    store_indices: torch.Tensor,  # [B] int32
    scale: float,
    eps: float,
    gate_activation: int,
) -> torch.Tensor:  # [B, HV, D] fp32
    """One fused GDN decode step for every valid row, exactly as in the contract.

    Rows with an out-of-range load or store slot are padding: output zeros, no state write.
    Every read of a state slot happens before any write, so load == store on a row works.
    """
    _require(conv_state.dim() == 3 and conv_state.shape[1] == 3,
             f"conv_state must be [S, 3, C], got shape {tuple(conv_state.shape)}")
    _require(ssm_state.dim() == 4 and ssm_state.shape[2] == ssm_state.shape[3],
             f"ssm_state must be [S, HV, D, D], got shape {tuple(ssm_state.shape)}")
    S, _, C = conv_state.shape
    HV, D = ssm_state.shape[1], ssm_state.shape[2]
    _require(ssm_state.shape[0] == S, f"ssm_state slot count {ssm_state.shape[0]} != conv_state slot count {S}")
    _require(HV > 0 and D > 0, f"ssm_state must have HV > 0 and D > 0, got shape {tuple(ssm_state.shape)}")
    H = (C - HV * D) // (2 * D)
    _require(H > 0 and C == 2 * H * D + HV * D,
             f"conv_state channels C={C} must equal 2*H*D + HV*D with D={D}, HV={HV}")
    _require(HV % H == 0, f"HV={HV} must be a multiple of H={H}")
    rep = HV // H
    _require(mixed_qkv.dim() == 2 and mixed_qkv.shape[1] == C,
             f"mixed_qkv must be [B, {C}], got shape {tuple(mixed_qkv.shape)}")
    B = mixed_qkv.shape[0]
    _require(z.dim() == 2 and tuple(z.shape) == (B, HV * D),
             f"z must be [B, HV*D] = ({B}, {HV * D}), got {tuple(z.shape)}")
    _require(a.dim() == 2 and tuple(a.shape) == (B, HV), f"a must be [B, HV] = ({B}, {HV}), got {tuple(a.shape)}")
    _require(b.dim() == 2 and tuple(b.shape) == (B, HV), f"b must be [B, HV] = ({B}, {HV}), got {tuple(b.shape)}")
    _require(A_log.dim() == 1 and tuple(A_log.shape) == (HV,),
             f"A_log must be [HV] = ({HV},), got {tuple(A_log.shape)}")
    _require(dt_bias.dim() == 1 and tuple(dt_bias.shape) == (HV,),
             f"dt_bias must be [HV] = ({HV},), got {tuple(dt_bias.shape)}")
    _require(conv_weight_t.dim() == 2 and tuple(conv_weight_t.shape) == (4, C),
             f"conv_weight_t must be [4, C] = (4, {C}), got {tuple(conv_weight_t.shape)}")
    _require(norm_weight.dim() == 1 and tuple(norm_weight.shape) == (D,),
             f"norm_weight must be [D] = ({D},), got {tuple(norm_weight.shape)}")
    _require(load_indices.dim() == 1 and tuple(load_indices.shape) == (B,),
             f"load_indices must be [B] = ({B},), got {tuple(load_indices.shape)}")
    _require(store_indices.dim() == 1 and tuple(store_indices.shape) == (B,),
             f"store_indices must be [B] = ({B},), got {tuple(store_indices.shape)}")
    _require(gate_activation in (GdnGateActivation.SILU, GdnGateActivation.SIGMOID),
             f"gate_activation must be 0 (SILU) or 1 (SIGMOID), got {gate_activation}")

    out = torch.zeros(B, HV, D, dtype=torch.float32, device=ssm_state.device)
    load = load_indices.long()
    store = store_indices.long()
    valid = (load >= 0) & (load < S) & (store >= 0) & (store < S)
    rows = torch.nonzero(valid).reshape(-1)
    if rows.numel() == 0:
        return out
    load_valid = load.index_select(0, rows)
    store_valid = store.index_select(0, rows)
    N = rows.shape[0]

    # Read-before-write: gather every input row of the two states before any write below.
    raw = mixed_qkv.index_select(0, rows)  # original dtype, for the exact conv-history copy
    hist_raw = conv_state.index_select(0, load_valid)  # [N, 3, C], original dtype
    xf = raw.float()
    w = conv_weight_t.float()
    hist = hist_raw.float()
    conv = hist[:, 0] * w[0] + hist[:, 1] * w[1] + hist[:, 2] * w[2] + xf * w[3]  # [N, C]
    act = conv * torch.sigmoid(conv)
    q = act[:, 0:H * D].reshape(N, H, D)
    k = act[:, H * D:2 * H * D].reshape(N, H, D)
    v = act[:, 2 * H * D:2 * H * D + HV * D].reshape(N, HV, D)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(rep, dim=1)  # value head vh uses key head vh // rep
    k = k.repeat_interleave(rep, dim=1)
    xg = a.index_select(0, rows).float() + dt_bias.float()  # [N, HV]
    sp = torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
    decay = torch.exp(-torch.exp(A_log.float()) * sp)
    beta = torch.sigmoid(b.index_select(0, rows).float())
    St = ssm_state.index_select(0, load_valid).float()  # [N, HV, D, D]
    St = St * decay[..., None, None]
    delta = (v - torch.matmul(St, k[..., None]).squeeze(-1)) * beta[..., None]  # [N, HV, D]
    St = St + delta[..., None] * k[..., None, :]
    o = torch.matmul(St, q[..., None]).squeeze(-1)  # [N, HV, D]
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * norm_weight.float()
    zz = z.index_select(0, rows).float().reshape(N, HV, D)
    if gate_activation == GdnGateActivation.SIGMOID:
        gate = torch.sigmoid(zz)
    else:
        gate = zz * torch.sigmoid(zz)
    out.index_copy_(0, rows, y * gate)

    ssm_state.index_copy_(0, store_valid, St.to(ssm_state.dtype))
    new_hist = torch.cat([hist_raw[:, 1:3], raw[:, None, :]], dim=1)  # from original dtype values
    conv_state[store_valid] = new_hist.to(conv_state.dtype)
    return out


# The torch.library op wrapper.  This module is the only place that defines the schema and the Meta
# (fake) kernel; the HPU kernel comes from the C++ extension (K4b), which must not define them.

_LIBRARY = torch.library.Library("flashinfer_gaudi", "FRAGMENT")

_LIBRARY.define(
    "gdn_core_decode("
    "Tensor(a!) conv_state, Tensor(b!) ssm_state, Tensor mixed_qkv, Tensor z, Tensor a, Tensor b, "
    "Tensor A_log, Tensor dt_bias, Tensor conv_weight_t, Tensor norm_weight, "
    "Tensor load_indices, Tensor store_indices, "
    "float scale, float eps, int gate_activation) -> Tensor")


def _gdn_core_decode_cpu(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight,
                         load_indices, store_indices, scale, eps, gate_activation):
    return gdn_core_decode_reference(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t,
                                     norm_weight, load_indices, store_indices, scale, eps, gate_activation)


_LIBRARY.impl("gdn_core_decode", _gdn_core_decode_cpu, "CPU")


@torch.library.register_fake("flashinfer_gaudi::gdn_core_decode", lib=_LIBRARY)
def _gdn_core_decode_fake(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight,
                          load_indices, store_indices, scale, eps, gate_activation):
    # Allocation only: no mutation, no data access.
    B = mixed_qkv.shape[0]
    HV = ssm_state.shape[1]
    D = ssm_state.shape[2]
    return mixed_qkv.new_empty((B, HV, D), dtype=torch.float32)


_DISPATCH_KEYS_BY_DEVICE_TYPE = {"cpu": "CPU", "hpu": "HPU"}


def gdn_core_decode(
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
    load_indices: torch.Tensor,
    store_indices: torch.Tensor,
    scale: float,
    eps: float,
    gate_activation: int,
) -> torch.Tensor:
    """The fused GDN core decode step as ``torch.ops.flashinfer_gaudi.gdn_core_decode``.

    Mutates ``conv_state`` and ``ssm_state`` in place, returns ``[B, HV, D]`` fp32; semantics are
    exactly ``gdn_core_decode_reference`` (contract: ``docs/design/flash-next/gdn-core-decode.md``).
    """
    return torch.ops.flashinfer_gaudi.gdn_core_decode(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias,
                                                      conv_weight_t, norm_weight, load_indices, store_indices, scale,
                                                      eps, int(gate_activation))


def gdn_core_decode_available(device: torch.device | str) -> bool:
    """True when ``flashinfer_gaudi::gdn_core_decode`` has a kernel for the device's type."""
    device_type = device.type if isinstance(device, torch.device) else torch.device(device).type
    dispatch_key = _DISPATCH_KEYS_BY_DEVICE_TYPE.get(device_type)
    if dispatch_key is None:
        return False
    return torch._C._dispatch_has_kernel_for_dispatch_key("flashinfer_gaudi::gdn_core_decode", dispatch_key)


class GdnCoreDecodeDebug(enum.IntEnum):
    """VLLM_HPU_GDN_CORE_DECODE_DEBUG values (next free: 2)."""
    OFF = 0
    CHECK = 1


def check_gdn_core_decode_indices(load_indices: torch.Tensor, store_indices: torch.Tensor, num_slots: int,
                                  num_rows: int) -> None:
    """Debug check for the slot patterns the fused GDN core decode kernels cannot handle.

    Over the real rows (index ``< num_rows``) whose two slots are both in ``[0, num_slots)``:
    raises ``ValueError`` when two rows share a store slot, or when one row's store slot is
    another row's load slot (a reordered cache).  Rows from ``num_rows`` on are padding and
    out-of-range rows are ignored; a row loading and storing its own slot is the normal case.
    Copies the indices to the CPU, so it costs one host sync per call: debugging only.
    """
    load = load_indices.reshape(-1)[:num_rows].to("cpu", torch.int64).tolist()
    store = store_indices.reshape(-1)[:num_rows].to("cpu", torch.int64).tolist()
    real = [r for r in range(len(load)) if 0 <= load[r] < num_slots and 0 <= store[r] < num_slots]

    first_row_by_store: dict[int, int] = {}
    rows_by_load: dict[int, list[int]] = {}
    for r in real:
        s = store[r]
        r0 = first_row_by_store.get(s)
        if r0 is not None:
            raise ValueError(f"GDN core decode: rows {r0} and {r} both store slot {s}")
        first_row_by_store[s] = r
        rows_by_load.setdefault(load[r], []).append(r)

    for r in real:
        s = store[r]
        for r2 in rows_by_load.get(s, []):
            if r2 != r:
                raise ValueError(f"GDN core decode: row {r} stores slot {s}, which row {r2} loads")
