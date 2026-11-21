# SPDX-License-Identifier: Apache-2.0
"""Simulator and card cases of the fused QSA decode kernels: inputs, raw-file folders, comparison with the references.

A case is a kernel plus a geometry. ``make_inputs`` builds seeded inputs laid out as the layer builds them
(the row table of a decode step with real, padding, completing and moved rows; the selection of
``select_blocks`` for the emit kernel); ``write_inputs`` writes the folder ``tests/standalone/qsa_fused/
sim_host.cpp`` reads (TPC dims are the torch dims reversed: the bytes of a contiguous torch tensor are the
TPC tensor as it is); ``compare`` checks the kernel's outputs against ``qsa_reference``:

* emit: ``slots`` and ``valid`` bit-exact (integers); topk (the one-node top-k): ``ids`` bit-exact against
  ``qsa_index_topk_masked_ref`` (its workspace is scratch and not compared);
* prepare: the ring bit-exact (copies); every compressed row and query element either bit-exact or at most
  ``BF16_STEPS`` bf16 steps from the reference, and at least ``EXACT_FRACTION`` of them bit-exact (the
  kernel's ``rsqrt`` and its sum of squares round differently from torch's; the rest of the arithmetic
  keeps the torch path's rounding points); rows the step does not write unchanged, bit for bit.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from tests.unit_tests.qsa_fused.qsa_reference import (qsa_index_emit_ref, qsa_index_prepare_ref,
                                                      qsa_index_select_ref, qsa_index_threshold_ref,
                                                      qsa_index_topk_masked_ref, select_blocks, visible_blocks)

PREPARE_GUID = "qsa_fused_prepare_bf16_gaudi2"
EMIT_GUID = "qsa_fused_emit_i32_gaudi2"
THRESHOLD_GUID = "qsa_fused_threshold_f32_gaudi2"
SELECT_GUID = "qsa_fused_select_i32_gaudi2"
TOPK_GUID = "qsa_fused_topk_f32_gaudi2"
BF16_STEPS = 2
EXACT_FRACTION = 0.97
EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    kernel: str  # "prepare", "emit", "threshold", "select" or "topk"
    # prepare
    dim: int = 16
    heads: int = 4
    rotary_half: int = 4
    ratio: int = 4
    ring_rows: int = 8
    slots: int = 12
    positions: int = 4096
    requests: tuple = ()  # ((ctx, real rows, load slot, store slot), ...)
    t_cols: int = 1
    # emit
    topk: int = 8
    block_size: int = 128
    table_columns: int = 8
    emit_rows: tuple = ()  # positions of the rows (-1: a padding row under M-RoPE); also threshold and select
    score_columns: int = 1024  # threshold and select: N
    seed: int = 0


SIM_CASES = (
    Case("prepare_tiny_lattice", "prepare", dim=16, heads=4, rotary_half=4, ring_rows=8, slots=12, t_cols=4,
         requests=((61, 4, 1, 1), (126, 3, 2, 2), (2, 4, 3, 3), (40, 0, 4, 4)), seed=1),
    Case("prepare_tiny_moves", "prepare", dim=16, heads=4, rotary_half=4, ring_rows=8, slots=12, t_cols=4,
         requests=((63, 4, 1, 5), (130, 4, 2, 6), (3, 1, 3, 7)), seed=2),
    Case("prepare_real_decode32", "prepare", dim=128, heads=4, rotary_half=32, ring_rows=4, slots=40, t_cols=1,
         requests=tuple((c, 1, 1 + i, 1 + i) for i, c in enumerate(
             (3, 7, 11, 2050, 131071 % 4096, 99, 1023, 4, 5, 6, 3999, 1, 15, 19, 23, 27, 31, 0, 2047, 2051, 3071, 8,
              9, 10, 12, 13, 14, 16, 17, 18, 20, 21))), seed=3),
    Case("prepare_real_verify_k5", "prepare", dim=128, heads=4, rotary_half=32, ring_rows=12, slots=12, t_cols=6,
         requests=((3000, 6, 1, 1), (2051, 4, 2, 2), (9, 6, 3, 8), (100, 0, 4, 4), (4000, 6, 5, 5)), seed=4),
    Case("emit_tiny", "emit", topk=8, ratio=4, block_size=128, table_columns=8,
         emit_rows=(0, 1, 2, 3, 30, 31, 32, 33, 34, 35, 900, 1023, -1, 7), seed=5),
    Case("emit_real_decode32", "emit", topk=512, ratio=4, block_size=128, table_columns=1024,
         emit_rows=(3, 2047, 2048, 2051, 131071, 65536, 100000, 0, 1, 2, 4095, 8191, 12345, 99999, 77777, 1 << 16,
                    3000, 3001, 3002, 3003, 120000, 130000, 4, 5, 6, 7, 2052, 2053, 2054, -1, 50000, 2055), seed=6),
    Case("emit_ratio8", "emit", topk=16, ratio=8, block_size=64, table_columns=40,
         emit_rows=(0, 7, 8, 127, 128, 129, 2000, 2559, -1), seed=7),
)


SELECT_CASES = (
    Case("threshold_tiny", "threshold", topk=8, ratio=4, score_columns=128,
         emit_rows=(0, 3, 30, 31, 35, 300, 511, 600, -1), seed=10),
    Case("threshold_real32", "threshold", topk=512, ratio=4, score_columns=32768,
         emit_rows=(3, 2047, 2048, 2051, 131071, 65536, 100000, 0, 1, 2, 4095, 8191, 12345, 99999, 77777, 1 << 16,
                    3000, 3001, 3002, 3003, 120000, 130000, 4, 5, 6, 7, 2052, 2053, 2054, -1, 50000, 2055), seed=11),
    Case("select_tiny", "select", topk=8, ratio=4, score_columns=128,
         emit_rows=(0, 3, 30, 31, 35, 300, 511, 600, -1), seed=12),
    Case("select_real32", "select", topk=512, ratio=4, score_columns=32768,
         emit_rows=(3, 2047, 2048, 2051, 131071, 65536, 100000, 0, 1, 2, 4095, 8191, 12345, 99999, 77777, 1 << 16,
                    3000, 3001, 3002, 3003, 120000, 130000, 4, 5, 6, 7, 2052, 2053, 2054, -1, 50000, 2055), seed=13),
)
# The one-node top-k (VLLM_HPU_QSA_TOPK_FORM=4): the masked scores of select_blocks (-inf past a row's visible
# blocks). Rows: short (fewer than topk visible: -1 padding), long, padding (-1: nothing visible), the tiny
# model's width; every third row on a coarse grid (ties at the cut beyond topk: the two-bitmap path), others
# with exact zeros (relu) and distinct values (the one-bitmap path).
TOPK_CASES = (
    Case("topk_tiny", "topk", topk=8, ratio=4, score_columns=512,
         emit_rows=(0, 3, 30, 31, 35, 300, 511, 600, 2047, -1), seed=14),
    Case("topk_tiny_width1024", "topk", topk=64, ratio=4, score_columns=1024,
         emit_rows=(5, 255, 256, 1000, 4095, -1, 2000), seed=15),
    Case("topk_real32", "topk", topk=512, ratio=4, score_columns=32768,
         emit_rows=(3, 2047, 2048, 2051, 131071, 65536, 100000, 0, 1, 2, 4095, 8191, 12345, 99999, 77777, 1 << 16,
                    3000, 3001, 3002, 3003, 120000, 130000, 4, 5, 6, 7, 2052, 2053, 2054, -1, 50000, 2055), seed=16),
)
SIM_CASES = SIM_CASES + SELECT_CASES + TOPK_CASES

# Card cases: the simulator cases plus the target's verification shape (32 requests x 4 rows, k = 3) for both
# kernels, and a 128-row emit with the 1,024-column block table of --max-model-len 131072.
HPU_CASES = SIM_CASES + (
    Case("prepare_real_verify32x4", "prepare", dim=128, heads=4, rotary_half=32, ring_rows=8, slots=40, t_cols=4,
         requests=tuple((100 * i + 3 * (i % 5), 4 - (i % 3 == 2), 1 + i, 1 + i) for i in range(32)), seed=8),
    Case("emit_real_verify128", "emit", topk=512, ratio=4, block_size=128, table_columns=1024,
         emit_rows=tuple(2047 + 997 * i for i in range(128)), seed=9),
)


def case_by_name(name: str) -> Case:
    return next(case for case in HPU_CASES if case.name == name)


# ------------------------------------------------------------------------------------------------- inputs
def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)


def rope_table(positions: int, rotary_half: int, theta: float = 1e7) -> torch.Tensor:
    inv = 1.0 / (theta**(torch.arange(0, 2 * rotary_half, 2, dtype=torch.float32) / (2 * rotary_half)))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def make_inputs(case: Case) -> dict:
    gen = torch.Generator().manual_seed(20261120 + case.seed)
    if case.kernel == "prepare":
        D, H = case.dim, case.heads
        T = len(case.requests) * case.t_cols
        rows = []
        for ctx, n, load, store in case.requests:
            for j in range(case.t_cols):
                if j < n:
                    p = ctx + j
                    rows.append((p, 4096 + 128 * len(rows) + p % 128, load, store))  # a distinct attention slot
                else:
                    rows.append((0, -1, -1, -1))
        cache_rows = (4096 + 128 * T + 256) // case.ratio + 8
        return {
            "key_ring": _bf16(gen, case.slots * case.ring_rows, D, scale=1.5),
            "key_cache": _bf16(gen, cache_rows, D, scale=1.5),
            "qk": _bf16(gen, T, (H + 1) * D, scale=2.0),
            "rows": torch.tensor(rows, dtype=torch.int32),
            "q_norm_weight": _bf16(gen, D, scale=0.2),
            "k_norm_weight": _bf16(gen, D, scale=0.2),
            "cos_sin": rope_table(case.positions, case.rotary_half),
        }
    if case.kernel in ("threshold", "select", "topk"):
        T = len(case.emit_rows)
        positions = torch.tensor(case.emit_rows, dtype=torch.int32)
        # relu sums: non-negative, many exact zeros, exact ties on a coarse grid in some rows
        raw = torch.relu(torch.randn(T, case.score_columns, generator=gen))
        raw[::3] = (raw[::3] * 8).round() / 8
        scores = raw / 11.3
        if case.kernel == "threshold":
            return {"scores": scores.contiguous(), "positions": positions}
        if case.kernel == "topk":
            visible = visible_blocks(positions.long(), case.ratio)
            cols = torch.arange(case.score_columns)
            masked = scores.masked_fill(cols[None, :] >= visible[:, None], float("-inf"))
            return {"masked": masked.contiguous()}
        return {"meta": qsa_index_threshold_ref(scores, positions, case.ratio, case.topk)}
    T = len(case.emit_rows)
    positions = torch.tensor(case.emit_rows, dtype=torch.int32)
    columns = case.table_columns * (case.block_size // case.ratio)
    scores = torch.rand(T, columns, generator=gen)
    scores[:, ::7] = scores[:, ::7].round(decimals=1)  # some exact ties
    blocks = select_blocks(scores, visible_blocks(positions.long(), case.ratio), case.topk)
    table = torch.randint(0, 50000, (T, case.table_columns), generator=gen, dtype=torch.int32)
    return {"blocks": blocks.contiguous(), "positions": positions, "block_table": table}


def reference(case: Case, inputs: dict) -> dict:
    if case.kernel == "prepare":
        ring = inputs["key_ring"].clone()
        cache = inputs["key_cache"].clone()
        q = qsa_index_prepare_ref(ring, cache, inputs["qk"], inputs["rows"], inputs["q_norm_weight"],
                                  inputs["k_norm_weight"], inputs["cos_sin"], EPS, case.heads, case.ring_rows,
                                  case.ratio)
        return {"key_ring": ring, "key_cache": cache, "q_idx": q}
    if case.kernel == "threshold":
        return {"meta": qsa_index_threshold_ref(inputs["scores"], inputs["positions"], case.ratio, case.topk)}
    if case.kernel == "select":
        return {"blocks": qsa_index_select_ref(inputs["meta"], case.topk)}
    if case.kernel == "topk":
        return {"ids": qsa_index_topk_masked_ref(inputs["masked"], case.topk)}
    slots, valid = qsa_index_emit_ref(inputs["blocks"], inputs["positions"], inputs["block_table"], case.ratio,
                                      case.block_size)
    return {"slots": slots, "valid": valid}


# --------------------------------------------------------------------------------------------- raw files
_DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32"}


def _tpc_dims(t: torch.Tensor) -> str:
    return " ".join(str(d) for d in reversed(t.shape))


def manifest(case: Case, inputs: dict) -> list[str]:
    lines = []
    if case.kernel == "prepare":
        lines += [f"guid {PREPARE_GUID}", f"eps {EPS!r}", f"heads {case.heads}", f"ring_rows {case.ring_rows}",
                  f"ratio {case.ratio}", f"rotary_half {case.rotary_half}"]
        for name in ("key_ring", "key_cache"):
            lines.append(f"inout {name} bf16 {_tpc_dims(inputs[name])}")
        for name in ("qk", "rows", "q_norm_weight", "k_norm_weight", "cos_sin"):
            lines.append(f"in {name} {_DTYPES[inputs[name].dtype]} {_tpc_dims(inputs[name])}")
        T = inputs["qk"].shape[0]
        lines.append(f"out q_idx bf16 {case.heads * case.dim} {T}")
        return lines
    if case.kernel == "threshold":
        T = inputs["scores"].shape[0]
        return [f"guid {THRESHOLD_GUID}", f"ratio {case.ratio}", f"topk {case.topk}",
                f"in scores f32 {_tpc_dims(inputs['scores'])}", f"in positions i32 {_tpc_dims(inputs['positions'])}",
                f"out meta i32 {4 + case.score_columns // 16} {T}"]
    if case.kernel == "select":
        T = inputs["meta"].shape[0]
        return [f"guid {SELECT_GUID}", f"topk {case.topk}", f"in meta i32 {_tpc_dims(inputs['meta'])}",
                f"out blocks i32 {case.topk} {T}"]
    if case.kernel == "topk":
        T = inputs["masked"].shape[0]
        return [f"guid {TOPK_GUID}", f"topk {case.topk}", f"in masked f32 {_tpc_dims(inputs['masked'])}",
                f"out ids i32 {case.topk} {T}", f"out workspace i32 {4 + case.score_columns // 16} {T}"]
    lines += [f"guid {EMIT_GUID}", f"ratio {case.ratio}", f"block_size {case.block_size}"]
    for name in ("blocks", "positions", "block_table"):
        lines.append(f"in {name} i32 {_tpc_dims(inputs[name])}")
    T = inputs["blocks"].shape[0]
    width = case.topk * case.ratio + case.ratio - 1
    lines += [f"out slots i32 {width} {T}", f"out valid i32 {width} {T}"]
    return lines


def write_inputs(case: Case, inputs: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "case.txt").write_text("\n".join(manifest(case, inputs)) + "\n")
    for name, t in inputs.items():
        raw = t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous()
        (folder / f"{name}.bin").write_bytes(raw.numpy().tobytes())


def _read(path: Path, like: torch.Tensor) -> torch.Tensor:
    np_type = {torch.bfloat16: np.int16, torch.float32: np.float32}.get(like.dtype, np.int32)
    data = np.frombuffer(path.read_bytes(), dtype=np_type)
    t = torch.from_numpy(data.copy()).view(like.shape)
    return t.view(torch.bfloat16) if like.dtype == torch.bfloat16 else t


def read_outputs(case: Case, folder: Path, inputs: dict | None = None) -> dict:
    inputs = make_inputs(case) if inputs is None else inputs
    want = reference(case, inputs)
    return {name: _read(folder / f"{name}.out.bin", like) for name, like in want.items()}


# ------------------------------------------------------------------------------------------- comparison
def _bf16_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 steps between two bf16 tensors of the same sign pattern (ordered integer encoding)."""
    def ordered(x):
        bits = x.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    return (ordered(a) - ordered(b)).abs()


def compare(case: Case, inputs: dict, got: dict) -> list[str]:
    want = reference(case, inputs)
    errors = []
    if case.kernel == "threshold":
        # the header and the bitmap words of every 128-block tile a row touches; later words are unspecified
        words = (want["meta"].shape[1] - 4) // 2
        for t in range(want["meta"].shape[0]):
            g, w = got["meta"][t], want["meta"][t]
            visible = int(w[0])
            span = 0 if int(w[1]) < 0 else (visible + 127) // 128 * 4
            parts = [(0, 3), (4, 4 + span), (4 + words, 4 + words + span)]
            for lo, hi in parts:
                if not torch.equal(g[lo:hi], w[lo:hi]):
                    errors.append(f"meta row {t} words [{lo}, {hi}) differ")
        return errors
    if case.kernel in ("select", "topk"):
        name = "blocks" if case.kernel == "select" else "ids"
        if not torch.equal(got[name], want[name]):
            bad = (got[name] != want[name]).any(1).nonzero().reshape(-1).tolist()
            errors.append(f"{name} differ on rows {bad[:8]}")
        return errors
    if case.kernel == "emit":
        for name in ("slots", "valid"):
            if not torch.equal(got[name], want[name]):
                bad = (got[name] != want[name]).nonzero()
                errors.append(f"{name}: {bad.shape[0]} entries differ, first at {bad[0].tolist()}: "
                              f"{int(got[name][tuple(bad[0])])} vs {int(want[name][tuple(bad[0])])}")
        return errors
    if not torch.equal(got["key_ring"], want["key_ring"]):
        errors.append(f"key_ring: {int((got['key_ring'] != want['key_ring']).any(1).sum())} rows differ")
    written = (want["key_cache"].view(torch.int16) != inputs["key_cache"].view(torch.int16)).any(1)
    unwritten = ~written
    if not torch.equal(got["key_cache"][unwritten], want["key_cache"][unwritten]):
        errors.append("key_cache: a row the step does not write changed")
    for name, g, w in (("q_idx", got["q_idx"], want["q_idx"]),
                       ("key_cache rows", got["key_cache"][written], want["key_cache"][written])):
        if g.numel() == 0:
            continue
        steps = _bf16_steps(g, w)
        exact = float((steps == 0).float().mean())
        worst = int(steps.max())
        if worst > BF16_STEPS or exact < EXACT_FRACTION:
            errors.append(f"{name}: {exact:.4f} bit-exact, worst {worst} bf16 steps")
    return errors


def summary(case: Case, inputs: dict, got: dict) -> str:
    want = reference(case, inputs)
    parts = [case.name]
    for name in want:
        if want[name].dtype == torch.bfloat16:
            steps = _bf16_steps(got[name], want[name])
            parts.append(f"{name}: exact {float((steps == 0).float().mean()):.4f} worst {int(steps.max())}")
        else:
            parts.append(f"{name}: equal {bool(torch.equal(got[name], want[name]))}")
    return " | ".join(parts)
