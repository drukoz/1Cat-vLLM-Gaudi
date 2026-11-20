# SPDX-License-Identifier: Apache-2.0
"""Shared kernel test cases for the fused GDN verification op (``gdn_core_verify``, phase 2).

Cases, inputs, the loop oracle (the executable form of the contract), the pass/fail comparison and the
raw-file I/O for ``flashinfer_gaudi::gdn_core_verify``.  The TPC simulator test, the HPU op test and the CPU
tests of ``gdn_core_verify_reference`` import these names.  The raw format is ``sim_case.py``'s (one header-less
little-endian file per tensor; bf16 as its 16-bit pattern).  ``params.txt`` holds ``S R H HV B T scale eps gate``.

Contract: ``docs/design/flash-next/gdn-core-decode.md``, "Phase 2 contract, version 2".  The oracle was checked
against M3's composed speculative path (``spikes/gdn-core-verify/probe.py``).
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

from flashinfer_gaudi.gdn_core_decode import GdnGateActivation
from flashinfer_gaudi.gdn_core_verify import GDN_CORE_VERIFY_MAX_NODES
from tests.unit_tests.gdn_core_decode.sim_case import D, OUT_ATOL, OUT_RTOL, SSM_ATOL, SSM_RTOL, _raw_bytes, _read_raw

# Columns of row_indices [B, 5] int32.
CONV_LOAD, CONV_OFFSET, CONV_STORE, SSM_LOAD, QUERY_LEN = 0, 1, 2, 3, 4
ROW_FIELDS = 5
# T <= 6 (T = 1 + k, k = 5 draft tokens): the kernel keeps per-node vectors in fixed __local__ arrays (16 KiB of
# vector local memory).
MAX_NODES = 6
assert MAX_NODES == GDN_CORE_VERIFY_MAX_NODES, "verify_sim_case.MAX_NODES must equal the op's GDN_CORE_VERIFY_MAX_NODES"

VERIFY_ARG_ORDER = ("conv_state", "ssm_state", "mixed_qkv", "z", "a", "b", "A_log", "dt_bias", "conv_weight_t",
                    "norm_weight", "row_indices", "ssm_store_indices")
VERIFY_OUTPUT_FILES = {"out": "out.bin", "conv_state": "conv_state.out.bin", "ssm_state": "ssm_state.out.bin"}


@dataclasses.dataclass(frozen=True)
class VerifyRow:
    conv_load: int  # conv slot read
    conv_offset: int  # first history row read in that slot (num_accepted - 1), 0 <= offset <= R - 3
    conv_store: int  # conv slot written (all R rows)
    ssm_load: int  # SSM slot node 0 starts from (the accepted checkpoint)
    query_len: int  # nodes 0 .. query_len - 1 are real
    ssm_store: tuple[int, ...]  # one per node: where node t's SSM state is stored (out of range = not stored)


@dataclasses.dataclass(frozen=True)
class VerifySimCase:
    name: str
    H: int  # key heads
    HV: int  # value heads
    S: int  # state slots
    R: int  # conv rows per slot (3 + k)
    rows: tuple[VerifyRow, ...]  # B = len(rows); T = len(rows[0].ssm_store)
    gate: GdnGateActivation
    seed: int

    @property
    def B(self) -> int:
        return len(self.rows)

    @property
    def T(self) -> int:
        return len(self.rows[0].ssm_store)


def _pad(T: int) -> VerifyRow:
    return VerifyRow(-1, 0, -1, -1, 0, (-1, ) * T)


def _lattice_rows(B: int, T: int) -> tuple[VerifyRow, ...]:
    """B in-place chain rows (load lattice == store lattice), accepted counts cycling 1..T."""
    rows = []
    for r in range(B):
        base = 1 + r * T
        off = r % T
        rows.append(VerifyRow(base, off, base, base + off, T, tuple(range(base, base + T))))
    return tuple(rows)


def _all_accepted_row(base: int, T: int) -> VerifyRow:
    """An in-place row whose previous step accepted all T nodes: the largest conv offset (T - 1) and the SSM load
    slot of node T - 1, which that node stores over."""
    return VerifyRow(base, T - 1, base, base + T - 1, T, tuple(range(base, base + T)))


def _trim_rows(T: int) -> tuple[VerifyRow, ...]:
    """Four rows with slots 1 .. 4T: trimmed in place with one unstored node, padding, moved with one real node, and
    a query length above T (the op clamps it to T)."""
    return (
        VerifyRow(1, 2, 1, 3, T - 2, (1, 99) + tuple(range(3, T + 1))),  # node 1 not stored (slot out of range)
        _pad(T),
        VerifyRow(T + 1, 1, 2 * T + 1, T + 2, 1, tuple(range(2 * T + 1, 3 * T + 1))),  # only slot 2T + 1 is stored
        VerifyRow(3 * T + 1, 0, 3 * T + 1, 3 * T + 1, T + 3, tuple(range(3 * T + 1, 4 * T + 1))),
    )


def _wide_cases(T: int, seed: int, trim_rows: int, silu: str) -> tuple[VerifySimCase, ...]:
    """The T = 5 and T = 6 simulator cases (k = 4 and k = 5 draft tokens): TP8 per-rank geometry (2 key heads,
    6 value heads) with batch 1, batch 16 and a padded/trimmed batch, and TP4 geometry (4, 12) with batch 2.
    ``R = T + 2`` conv rows as M3 sizes the pool, except the trimmed case (``trim_rows``).  The case named by
    ``silu`` uses the SiLU output gate."""

    def gate(kind: str) -> GdnGateActivation:
        return GdnGateActivation.SILU if kind == silu else GdnGateActivation.SIGMOID

    return (
        VerifySimCase(f"ver_tp8_t{T}_b1", 2, 6, T + 2, T + 2, (_all_accepted_row(1, T), ), gate("b1"), seed),
        VerifySimCase(f"ver_tp8_t{T}_b16", 2, 6, 1 + 16 * T, T + 2, _lattice_rows(16, T), gate("b16"), seed + 1),
        VerifySimCase(f"ver_tp8_t{T}_trim", 2, 6, 4 * T + 2, trim_rows, _trim_rows(T), gate("trim"), seed + 2),
        VerifySimCase(f"ver_tp4_t{T}_b2", 4, 12, 3 * T + 1, T + 2, (
            _all_accepted_row(1, T),
            VerifyRow(T + 1, 1, 2 * T + 1, T + 2, T, tuple(range(2 * T + 1, 3 * T + 1))),  # moved, all nodes real
        ), gate("tp4"), seed + 3),
    )


# tpc-clang may reorder loads and stores of one tensor (tpcsim.md), so the simulator cases cover both kinds of row:
# aliased rows (conv load == store, SSM load slot also a store slot) in every case but ver_tp1_move, and rows that
# move, are trimmed or are padding in the chain, t2, t3, trim and tp4 cases.  The batch-16 lattices are aliased rows
# with every accepted count 1 .. T.
VERIFY_CASES = (  # simulator + HPU
    VerifySimCase("ver_tp8_chain", 2, 6, 14, 6, (
        VerifyRow(1, 3, 1, 4, 4, (1, 2, 3, 4)),  # non-align, all accepted last step: node 3 stores over the load
        VerifyRow(5, 1, 9, 6, 2, (9, 10, 11, 12)),  # align (moved), trimmed to 2 nodes: 11, 12 untouched
        _pad(4),
    ), GdnGateActivation.SIGMOID, 20),
    VerifySimCase("ver_tp4_t1", 4, 12, 6, 6, (
        VerifyRow(1, 2, 1, 3, 1, (1, )),  # the plain step after a verification: T = 1, offset 2
        VerifyRow(2, 0, 2, 2, 1, (2, )),  # everything in place
    ), GdnGateActivation.SIGMOID, 21),
    VerifySimCase("ver_tp8_t2_silu", 2, 6, 8, 6, (
        VerifyRow(3, 0, 3, 3, 2, (3, 4)),
        VerifyRow(5, 1, 6, 5, 1, (6, 7)),  # one real node: only slot 6 is stored
    ), GdnGateActivation.SILU, 22),
    VerifySimCase("ver_tp1_move", 16, 48, 3, 6, (VerifyRow(0, 0, 1, 0, 2, (1, 2)), ), GdnGateActivation.SIGMOID, 23),
    VerifySimCase("ver_tp8_t3", 2, 6, 10, 6, (
        VerifyRow(1, 0, 1, 1, 3, (1, 2, 3)),  # T = 3, in place; node 0 stores over the load slot
        VerifyRow(4, 2, 5, 4, 2, (5, 99, 7)),  # moved, trimmed to 2 nodes, node 1 not stored (slot out of range)
        _pad(3),
    ), GdnGateActivation.SIGMOID, 26),
    # T = 6 (k = 5, R = 8): ver_tp8_chain's three rows at the largest T.
    VerifySimCase("ver_tp8_t6_chain", 2, 6, 20, 8, (
        VerifyRow(1, 5, 1, 6, 6, (1, 2, 3, 4, 5, 6)),  # non-align, all accepted last step: node 5 stores over the load
        VerifyRow(7, 2, 13, 9, 3, (13, 14, 15, 16, 17, 18)),  # align (moved), trimmed to 3 nodes: 16, 17, 18 untouched
        _pad(6),
    ), GdnGateActivation.SIGMOID, 30),
) + _wide_cases(5, 31, trim_rows=8, silu="tp4") + _wide_cases(6, 35, trim_rows=8, silu="trim")
# One node more than the op takes: the host glue and the HPU op callback must reject it, the reference raises.
VERIFY_TOO_MANY_NODES_CASE = VerifySimCase(
    "ver_too_many_nodes", 2, 6, 8, MAX_NODES + 3,
    (VerifyRow(1, 0, 1, 1, MAX_NODES + 1, tuple(range(1, MAX_NODES + 2))), ), GdnGateActivation.SIGMOID, 27)
# HPU only: the T = 4 lattice at batch 16 and 64 (kept out of the simulator run to bound its time; the simulator
# runs batch 16 at T = 5 and T = 6).
VERIFY_HPU_CASES = VERIFY_CASES + (
    VerifySimCase("ver_tp8_b16", 2, 6, 1 + 16 * 4, 6, _lattice_rows(16, 4), GdnGateActivation.SIGMOID, 24),
    VerifySimCase("ver_tp8_b64", 2, 6, 1 + 64 * 4, 6, _lattice_rows(64, 4), GdnGateActivation.SIGMOID, 25),
)


def make_verify_inputs(case: VerifySimCase) -> dict[str, torch.Tensor]:
    """CPU tensors keyed by ``VERIFY_ARG_ORDER``, in the kernel dtypes; deterministic from ``case.seed``.

    Token-major: ``mixed_qkv`` is ``[B*T, C]``, node ``t`` of row ``r`` at row ``r * T + t``; likewise ``z``,
    ``a`` and ``b``.
    """
    g = torch.Generator().manual_seed(case.seed)
    H, HV, S, R, B, T = case.H, case.HV, case.S, case.R, case.B, case.T
    C = 2 * H * D + HV * D

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    rows = torch.tensor([[r.conv_load, r.conv_offset, r.conv_store, r.ssm_load, r.query_len] for r in case.rows],
                        dtype=torch.int32)
    return {
        "conv_state": (randn(S, R, C) * 0.5).to(torch.bfloat16),
        "ssm_state": (randn(S, HV, D, D) * 0.1).float(),
        "mixed_qkv": randn(B * T, C).to(torch.bfloat16),
        "z": randn(B * T, HV * D).to(torch.bfloat16),
        "a": randn(B * T, HV).to(torch.bfloat16),
        "b": randn(B * T, HV).to(torch.bfloat16),
        "A_log": torch.log(torch.rand(HV, generator=g) * 15 + 1).float(),
        "dt_bias": (randn(HV) * 0.5).float(),
        "conv_weight_t": (randn(4, C) * 0.3).to(torch.bfloat16),
        "norm_weight": (1 + 0.1 * randn(D)).to(torch.bfloat16),
        "row_indices": rows,
        "ssm_store_indices": torch.tensor([r.ssm_store for r in case.rows], dtype=torch.int32),
    }


def verify_scalars(case: VerifySimCase) -> tuple[float, float, int]:
    """The op's three scalar parameters: (scale, eps, gate_activation)."""
    return D**-0.5, 1e-6, int(case.gate)


def row_is_real(row: torch.Tensor, S: int, R: int) -> bool:
    """A row of ``row_indices`` is real when its three slots are in ``[0, S)``, its offset is in ``[0, R - 3]``
    and its query length is at least 1; anything else is padding (zero output, no writes)."""
    cl, off, cs, sl, n = (int(v) for v in row)
    return 0 <= cl < S and 0 <= cs < S and 0 <= sl < S and 0 <= off <= R - 3 and n >= 1


def verify_oracle(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, row_indices,
                  ssm_store_indices, scale, eps, gate_activation) -> torch.Tensor:
    """The contract as a plain loop (slow; for tests).  Mutates the two pools, returns ``[B*T, HV, D]`` fp32."""
    S, R, C = conv_state.shape
    HV = ssm_state.shape[1]
    B, T = ssm_store_indices.shape
    H = (C - HV * D) // (2 * D)
    rep = HV // H
    out = torch.zeros(B * T, HV, D, dtype=torch.float32)
    w = conv_weight_t.float()
    for r in range(B):
        if not row_is_real(row_indices[r], S, R):
            continue
        cl, off, cs, sl, n = (int(v) for v in row_indices[r])
        n = min(n, T)
        hist = conv_state[cl, off:off + 3].clone()  # [3, C], read before any write
        xs = mixed_qkv[r * T:(r + 1) * T].to(conv_state.dtype).clone()
        xs[n:] = 0  # nodes past the query length are masked to zero
        seq = torch.cat([hist, xs], 0)  # [3 + T, C]
        state = ssm_state[sl].float().clone()  # [HV, D, D], read before any write
        for t in range(n):
            conv = (seq[t].float() * w[0] + seq[t + 1].float() * w[1] + seq[t + 2].float() * w[2] +
                    seq[t + 3].float() * w[3])
            act = conv * torch.sigmoid(conv)  # SiLU, fp32, no rounding
            tok = r * T + t
            for kh in range(H):
                q = act[kh * D:(kh + 1) * D]
                k = act[H * D + kh * D:H * D + (kh + 1) * D]
                q = q * torch.rsqrt((q * q).sum() + 1e-6) * scale
                k = k * torch.rsqrt((k * k).sum() + 1e-6)
                for j in range(rep):
                    vh = kh * rep + j
                    v = act[2 * H * D + vh * D:2 * H * D + (vh + 1) * D]
                    xg = a[tok, vh].float() + dt_bias[vh].float()
                    sp = torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
                    decay = torch.exp(-torch.exp(A_log[vh].float()) * sp)
                    beta = torch.sigmoid(b[tok, vh].float())
                    St = state[vh] * decay
                    St = St + ((v - St @ k) * beta)[:, None] * k[None, :]
                    state[vh] = St
                    o = St @ q
                    y = o * torch.rsqrt((o * o).mean() + eps) * norm_weight.float()
                    zz = z[tok, vh * D:(vh + 1) * D].float()
                    out[tok, vh] = y * (torch.sigmoid(zz) if gate_activation == int(GdnGateActivation.SIGMOID) else
                                        zz * torch.sigmoid(zz))
            st = int(ssm_store_indices[r, t])
            if 0 <= st < S:
                ssm_state[st] = state.to(ssm_state.dtype)
        new = torch.zeros(R, C, dtype=conv_state.dtype)
        new[0:2] = hist[1:3]
        new[2:2 + min(T, R - 2)] = xs[:min(T, R - 2)]
        conv_state[cs] = new
    return out


def expected_verify(case: VerifySimCase, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Oracle run on clones: {"out", "conv_state", "ssm_state"} (the two mutated state clones)."""
    conv = inputs["conv_state"].clone()
    ssm = inputs["ssm_state"].clone()
    out = verify_oracle(conv, ssm, *[inputs[k] for k in VERIFY_ARG_ORDER[2:]], *verify_scalars(case))
    return {"out": out, "conv_state": conv, "ssm_state": ssm}


def compare_verify(case: VerifySimCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> list[str]:
    """Failure messages for ``got`` (keys "out", "conv_state", "ssm_state"); [] means pass.

    Real nodes: out within ``OUT_ATOL``/``OUT_RTOL``.  Nodes past the query length and padding rows: out exactly 0.
    SSM: every stored slot within ``SSM_ATOL``/``SSM_RTOL``, every other slot unchanged.  Conv: every written slot
    (all R rows) and every other slot bit-exact.
    """
    S, R, B, T, HV = case.S, case.R, case.B, case.T, case.HV
    exp = expected_verify(case, inputs)
    out = got["out"]
    if tuple(out.shape) != (B * T, HV, D) or out.dtype != torch.float32:
        return [f"{case.name}: out must be [{B * T}, {HV}, {D}] fp32, got shape {tuple(out.shape)} dtype {out.dtype}"]
    msgs = []
    rows = inputs["row_indices"]
    stored, conv_written = set(), set()
    for r in range(B):
        real = row_is_real(rows[r], S, R)
        n = min(int(rows[r][QUERY_LEN]), T) if real else 0
        for t in range(T):
            tok = r * T + t
            if t < n:
                if not torch.allclose(out[tok], exp["out"][tok], atol=OUT_ATOL, rtol=OUT_RTOL):
                    e = (out[tok] - exp["out"][tok]).abs().max().item()
                    msgs.append(f"{case.name}: out row {r} node {t}: max abs err {e:.3e}")
                st = int(inputs["ssm_store_indices"][r, t])
                if 0 <= st < S:
                    stored.add(st)
            elif torch.count_nonzero(out[tok]) != 0:
                msgs.append(f"{case.name}: row {r} node {t} ({'padding row' if not real else 'past query_len'}): "
                            "output not zero")
        if real:
            conv_written.add(int(rows[r][CONV_STORE]))
    ssm, exp_ssm = got["ssm_state"], exp["ssm_state"]
    for s in range(S):
        if s in stored:
            if not torch.allclose(ssm[s], exp_ssm[s], atol=SSM_ATOL, rtol=SSM_RTOL):
                e = (ssm[s] - exp_ssm[s]).abs().max().item()
                msgs.append(f"{case.name}: ssm_state slot {s}: max abs err {e:.3e}")
        elif not torch.equal(ssm[s], inputs["ssm_state"][s]):
            msgs.append(f"{case.name}: ssm_state untouched slot {s} changed")
    conv, exp_conv = got["conv_state"], exp["conv_state"]
    for s in range(S):
        if not torch.equal(conv[s], exp_conv[s]):
            bad = [i for i in range(R) if not torch.equal(conv[s, i], exp_conv[s, i])]
            what = "written" if s in conv_written else "untouched"
            msgs.append(f"{case.name}: conv_state {what} slot {s}: rows {bad} differ")
    return msgs


def write_verify_inputs(case: VerifySimCase, inputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """One raw file ``f"{name}.bin"`` per ``VERIFY_ARG_ORDER`` tensor, plus ``params.txt`` =
    ``S R H HV B T scale eps gate``."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in VERIFY_ARG_ORDER:
        (directory / f"{name}.bin").write_bytes(_raw_bytes(inputs[name]))
    scale, eps, gate = verify_scalars(case)
    (directory / "params.txt").write_text(
        f"{case.S} {case.R} {case.H} {case.HV} {case.B} {case.T} {scale!r} {eps!r} {gate}\n")


def write_verify_outputs(outputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """Write ``VERIFY_OUTPUT_FILES`` (the format the simulator host program writes)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, filename in VERIFY_OUTPUT_FILES.items():
        (directory / filename).write_bytes(_raw_bytes(outputs[key]))


def read_verify_outputs(case: VerifySimCase, directory: pathlib.Path) -> dict[str, torch.Tensor]:
    """Read ``VERIFY_OUTPUT_FILES`` back; ``ValueError`` naming the file on a size mismatch."""
    directory = pathlib.Path(directory)
    C = 2 * case.H * D + case.HV * D
    return {
        "out": _read_raw(directory / VERIFY_OUTPUT_FILES["out"], torch.float32, (case.B * case.T, case.HV, D)),
        "conv_state": _read_raw(directory / VERIFY_OUTPUT_FILES["conv_state"], torch.bfloat16, (case.S, case.R, C)),
        "ssm_state": _read_raw(directory / VERIFY_OUTPUT_FILES["ssm_state"], torch.float32,
                               (case.S, case.HV, D, D)),
    }
