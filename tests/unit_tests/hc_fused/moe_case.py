# SPDX-License-Identifier: Apache-2.0
"""Shared test cases for the fused MoE glue kernels (the ``moe_*`` kernels of ``csrc/hc_fused``).

One module defines the cases, their inputs, the expected outputs, the pass/fail comparison and the
raw-file I/O, as ``hc_case.py`` does for the hyper-connection kernels.  The TPC simulator test
(``tests/standalone/hc_fused/test_moe_simulation.py``), the HPU op test (``test_moe_hpu_op.py``) and
the CPU tests import these names.  The raw format (one header-less little-endian file per tensor)
plus ``case.txt`` is what ``tests/standalone/hc_fused/sim_host.cpp`` reads.

Contract: ``docs/design/flash-next/hc-fused.md``, "MoE glue kernels".
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch
import torch.nn.functional as F

from tests.unit_tests.hc_fused.hc_case import bf16_steps
from tests.unit_tests.hc_fused.moe_reference import (GAUDI2_FP8_MAX, moe_combine_reference, moe_route_reference,
                                                     moe_row_quant_reference, moe_silu_mul_reference)

BF16 = torch.bfloat16
F8 = torch.float8_e4m3fn
KINDS = ("route", "silu_mul", "combine", "row_quant")
# Kernel GUIDs; the same strings are in csrc/hc_fused/hc_fused_params.h (test_moe_sources.py checks it).
GUIDS = {
    "route": "moe_glue_route_bf16_gaudi2",
    "silu_mul": "moe_glue_silu_mul_bf16_gaudi2",
    "combine": "moe_glue_combine_bf16_gaudi2",
    "row_quant": "moe_glue_row_quant_bf16_gaudi2",
}
# The op of each kind: torch.ops.flashinfer_gaudi.<name> (schemas in flashinfer_gaudi/moe_glue.py).
OP_NAME = {kind: f"moe_{kind}" for kind in KINDS}
INPUT_ORDER = {
    "route": ("gates", ),
    "silu_mul": ("gate_up", ),
    "combine": ("shared", "gates", "routed"),
    "row_quant": ("x", ),
}
OUTPUT_ORDER = {
    "route": ("ids", "weights"),
    "silu_mul": ("act", ),
    "combine": ("out", ),
    "row_quant": ("x_fp8", "scale"),
}
OUTPUT_DTYPE = {"ids": torch.int32, "weights": BF16, "act": BF16, "out": BF16, "x_fp8": F8, "scale": torch.float32}
VECTOR = 128  # bf16 lanes of one TPC vector; combine and row_quant need hidden % VECTOR == 0
MAX_TOP_K = 64  # one 64-lane int32 vector holds the ids of a row

# Tolerances (one bf16 step per rounding point of the torch path):
#   weights    one bf16 step from the torch weight of the same probability, or both below the smallest
#              normal fp32 (the torch softmax keeps denormal probabilities, exp on the TPC flushes them);
#   act        one bf16 step, or the bits of silu-times-up when the rounded silu is one step away, or
#              UNDERFLOW times |up|: for a gate below -87.3 the fp32 sigmoid is a denormal, which torch
#              keeps (silu about -5e-37) and the TPC exp and reciprocal flush to zero;
#   out        one bf16 step of the row computed with the oracle's gate weight or with a gate weight one
#              bf16 step away, or FP32_SLACK of the operands where shared * gate + routed cancels, or
#              UNDERFLOW times |shared| (the same flush of a gate weight below the normal fp32 range);
#   scale      bit-exact;
#   x_fp8      bit-exact (as values: +0 and -0 are the same value).
# At most max(1, 1 %) of the elements of an output may differ from the oracle at all; an element that
# differs only by the UNDERFLOW rule (or, for the weights, below the smallest normal fp32) does not count.
MAX_ULP = 1
FLT_MIN = 1.1754943508222875e-38
FP32_SLACK = 1e-6
UNDERFLOW = 1e-36  # about 85 times the smallest normal fp32: |x * sigmoid(x)| where sigmoid(x) is a denormal
MAX_INEXACT_FRACTION = 0.01


@dataclasses.dataclass(frozen=True)
class MoeCase:
    name: str
    kind: str
    rows: int  # T: tokens
    experts: int = 512  # route, combine: E, the router columns of gates
    top_k: int = 10  # route: K
    columns: int = 513  # route, combine: C, the columns of gates (E, or E + 1 with the shared-gate logit)
    hidden: int = 2560  # combine, row_quant: H.  silu_mul: D, the output width (the input has 2 * D columns)
    flavor: str = "random"
    seed: int = 0
    exact: bool = False  # every element must be bit-exact (bf16 / fp8 rounding-edge cases)


def _c(name, kind, rows, **kw) -> MoeCase:
    return MoeCase(name=name, kind=kind, rows=rows, **kw)


_TINY = dict(experts=16, top_k=2, columns=17)  # tests/unit_tests/models/flash_next/tiny_config.json
# Geometries: the real model (512 experts, top 10, hidden 2560, shared expert 640 wide: 80 per rank at
# TP8) and the tiny test model (16 experts, top 2, hidden 128, shared expert 64 wide: 8 per rank at TP8).
SIM_CASES = (
    # ---- moe_route: softmax, top-K, renormalization, casts --------------------------------------
    _c("route_tiny_r1", "route", 1, seed=1, **_TINY),
    _c("route_tiny_r3", "route", 3, seed=2, **_TINY),
    _c("route_tiny_r64_router_only", "route", 64, experts=16, top_k=2, columns=16, seed=3),
    _c("route_real_r1", "route", 1, seed=4),
    _c("route_real_r2", "route", 2, seed=5),
    _c("route_real_r16", "route", 16, seed=6),
    _c("route_real_r32", "route", 32, seed=7),
    _c("route_real_r32_router_only", "route", 32, columns=512, seed=8),
    _c("route_prefill_r2048", "route", 2048, seed=9),
    _c("route_real_ties", "route", 64, flavor="ties", seed=10),
    _c("route_tiny_ties", "route", 64, flavor="ties", seed=11, **_TINY),
    _c("route_real_equal", "route", 8, flavor="equal", seed=12),
    _c("route_tiny_equal", "route", 8, flavor="equal", seed=13, **_TINY),
    _c("route_real_huge", "route", 16, flavor="huge", seed=14),
    _c("route_real_negative", "route", 16, flavor="negative", seed=15),
    _c("route_real_spread", "route", 32, flavor="spread", seed=16),
    _c("route_real_small", "route", 16, flavor="small", seed=17),
    _c("route_e200_k7_r5", "route", 5, experts=200, top_k=7, columns=201, seed=18),
    _c("route_e130_k64_r3", "route", 3, experts=130, top_k=64, columns=130, seed=19),
    # ---- moe_silu_mul: the shared expert's activation --------------------------------------------
    _c("silu_mul_tiny_r1", "silu_mul", 1, hidden=64, seed=21),
    _c("silu_mul_tiny_tp8_r3", "silu_mul", 3, hidden=8, seed=22),
    _c("silu_mul_real_tp8_r1", "silu_mul", 1, hidden=80, seed=23),
    _c("silu_mul_real_tp8_r2", "silu_mul", 2, hidden=80, seed=24),
    _c("silu_mul_real_tp8_r16", "silu_mul", 16, hidden=80, seed=25),
    _c("silu_mul_real_tp8_r32", "silu_mul", 32, hidden=80, seed=26),
    _c("silu_mul_real_tp1_r32", "silu_mul", 32, hidden=640, seed=27),
    _c("silu_mul_prefill_r2048", "silu_mul", 2048, hidden=80, seed=28),
    _c("silu_mul_real_stress", "silu_mul", 8, hidden=80, flavor="stress", seed=29),
    _c("silu_mul_real_edge", "silu_mul", 64, hidden=80, flavor="edge", seed=30, exact=True),
    _c("silu_mul_d130_r3", "silu_mul", 3, hidden=130, seed=31),
    _c("silu_mul_every_gate", "silu_mul", 816, hidden=80, flavor="every_gate", seed=32),
    # ---- moe_combine: gate * shared + routed -----------------------------------------------------
    _c("combine_tiny_r1", "combine", 1, hidden=128, seed=41, **_TINY),
    _c("combine_tiny_r3", "combine", 3, hidden=128, seed=42, **_TINY),
    _c("combine_real_r1", "combine", 1, seed=43),
    _c("combine_real_r2", "combine", 2, seed=44),
    _c("combine_real_r16", "combine", 16, seed=45),
    _c("combine_real_r32", "combine", 32, seed=46),
    _c("combine_prefill_r2048", "combine", 2048, seed=47),
    _c("combine_real_stress", "combine", 30, flavor="stress", seed=48),
    _c("combine_real_edge", "combine", 8, flavor="edge", seed=49, exact=True),
    _c("combine_gate_only_r4", "combine", 4, hidden=256, experts=0, columns=1, seed=50),
    _c("combine_every_gate", "combine", 65280, hidden=128, experts=0, columns=1, flavor="every_gate", seed=51),
    # ---- moe_row_quant: the routed experts' input, per-row FP8 ----------------------------------
    _c("row_quant_tiny_r1", "row_quant", 1, hidden=128, seed=61),
    _c("row_quant_tiny_r3", "row_quant", 3, hidden=128, seed=62),
    _c("row_quant_real_r1", "row_quant", 1, seed=63),
    _c("row_quant_real_r2", "row_quant", 2, seed=64),
    _c("row_quant_real_r16", "row_quant", 16, seed=65),
    _c("row_quant_real_r32", "row_quant", 32, seed=66),
    _c("row_quant_prefill_r2048", "row_quant", 2048, seed=67),
    _c("row_quant_real_stress", "row_quant", 16, flavor="stress", seed=68),
    _c("row_quant_real_edge", "row_quant", 8, flavor="edge", seed=69),
    _c("row_quant_every_amax", "row_quant", 32639, hidden=128, flavor="every_amax", seed=70),
)
# HPU only: every decode batch bucket and the largest prefill, per kernel.
HPU_CASES = SIM_CASES + tuple(
    _c(f"{kind}_real_r{rows}", kind, rows, hidden=80 if kind == "silu_mul" else 2560, seed=100 + 10 * i + j)
    for i, kind in enumerate(KINDS) for j, rows in enumerate((4, 8, 12, 20, 8192)))


def case_by_name(name: str) -> MoeCase:
    for case in HPU_CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def gate_column(case: MoeCase) -> int:
    """The column of gates that holds the shared-gate logit (combine)."""
    return case.columns - 1


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
_SIGMOID_STRESS = (0.0, -0.0, 1e-3, -1e-3, 0.5, -0.5, 1.0, -1.0, 4.0, -4.0, 8.0, -8.0, 17.0, -17.0, 20.0, -20.0, 36.0,
                   -36.0, 40.0, -40.0, 88.0, -88.0, 100.0, -100.0, 400.0, -400.0, 1e4, -1e4, 1e30, -1e30)


def _stress_values(g: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    table = torch.tensor(_SIGMOID_STRESS)
    return table[torch.randint(0, table.numel(), shape, generator=g)]


def _edge_values(g: torch.Generator, shape: tuple[int, ...], low: int = -6, high: int = 7) -> torch.Tensor:
    """Random values with a full 8-bit mantissa (every one is a bf16 value)."""
    mantissa = torch.randint(128, 256, shape, generator=g).float() / 128.0
    exponent = torch.randint(low, high, shape, generator=g).float()
    sign = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
    return sign * mantissa * torch.pow(2.0, exponent)


def every_finite_bf16() -> torch.Tensor:
    """The 65280 finite bf16 values, in bit-pattern order."""
    values = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(BF16)
    return values[torch.isfinite(values.float())]


def _route_logits(g: torch.Generator, case: MoeCase) -> torch.Tensor:
    T, E = case.rows, case.experts
    if case.flavor == "ties":
        # A coarse grid: many equal logits, at the top-K cut in most rows.
        return torch.randint(-8, 9, (T, E), generator=g).float() / 4.0
    if case.flavor == "equal":
        values = torch.tensor([0.0, 1.5, -3.0, 1e4, -1e4, 1e-30, 88.0, -0.0])
        return values[torch.arange(T) % values.numel()].reshape(T, 1).expand(T, E).clone()
    if case.flavor == "huge":
        logits = torch.randn(T, E, generator=g)
        for row in range(T):
            count = 1 + row % 3  # one, two or three winners far above the rest
            index = torch.randperm(E, generator=g)[:count]
            logits[row, index] = (1e4, 3e4, 200.0, 1e30)[row % 4]
        return logits
    if case.flavor == "negative":
        logits = -1e4 + 300.0 * torch.randn(T, E, generator=g)
        for row in range(T):
            masked = torch.randperm(E, generator=g)[:(E // 4 if row % 2 else E - 3)]
            logits[row, masked] = float("-inf")  # odd rows: a quarter masked; even rows: 3 finite experts
        return logits
    if case.flavor == "spread":
        return torch.randn(T, E, generator=g) * 40.0  # the top-K probabilities span exp(-100)
    if case.flavor == "small":
        # Distinct tiny logits whose probabilities are equal in fp32, and denormal logits.
        scale = torch.tensor([1e-30, 1e-20, 1e-10, 1e-38])[torch.arange(T) % 4].reshape(T, 1)
        return torch.randn(T, E, generator=g) * scale
    scale = torch.exp(0.7 * torch.randn(T, 1, generator=g))
    return torch.randn(T, E, generator=g) * 2.0 * scale


def _hidden(g: torch.Generator, rows: int, hidden: int) -> torch.Tensor:
    return torch.randn(rows, hidden, generator=g) * torch.exp(torch.randn(rows, 1, generator=g))


def make_inputs(case: MoeCase) -> dict[str, torch.Tensor]:
    """CPU tensors keyed by ``INPUT_ORDER[case.kind]``; deterministic from ``case.seed``."""
    g = torch.Generator().manual_seed(case.seed)
    T, H = case.rows, case.hidden

    if case.kind == "route":
        gates = torch.randn(T, case.columns, generator=g) * 3.0  # the columns after the experts are never read
        gates[:, :case.experts] = _route_logits(g, case)
        return {"gates": gates.to(BF16)}

    if case.kind == "silu_mul":
        gate_up = torch.randn(T, 2 * H, generator=g) * 4.0
        if case.flavor == "stress":
            gate_up = _stress_values(g, (T, 2 * H))
        if case.flavor == "edge":
            # Gates of 32 and more: sigmoid is 1 in fp32, so silu(gate) is the gate and the product of
            # two 8-bit mantissas sits exactly between two bf16 values in about 1 of 200 elements.
            gate_up = torch.cat([_edge_values(g, (T, H), 5, 9).abs(), _edge_values(g, (T, H))], dim=1)
        if case.flavor == "every_gate":
            # Every finite bf16 gate once, times 1: the output is the rounded silu of every possible input.
            gate = every_finite_bf16().reshape(T, H)
            return {"gate_up": torch.cat([gate, torch.ones(T, H, dtype=BF16)], dim=1)}
        return {"gate_up": gate_up.to(BF16)}

    if case.kind == "combine":
        shared = _hidden(g, T, H)
        routed = _hidden(g, T, H)
        gates = torch.randn(T, case.columns, generator=g) * 3.0
        if case.flavor == "stress":
            gates[:, gate_column(case)] = torch.tensor(_SIGMOID_STRESS)[torch.arange(T) % len(_SIGMOID_STRESS)]
            routed[T // 2:] = 0.0
        if case.flavor == "edge":
            # A gate logit of 100 is a weight of exactly 1; routed is a few quarter-steps of shared's
            # grid, so a quarter of the sums are exact ties.
            mantissa = torch.randint(128, 256, (T, H), generator=g).float()
            step = torch.pow(2.0, torch.randint(-12, 13, (T, H), generator=g).float() - 7.0)
            shared = torch.where(torch.rand(T, H, generator=g) < 0.5, -1.0, 1.0) * mantissa * step
            routed = torch.randint(-7, 8, (T, H), generator=g).float() * step / 4.0
            gates[:, gate_column(case)] = 100.0
        if case.flavor == "every_gate":
            # Every finite bf16 gate logit once, shared = 1 and routed = 0: the output is the gate weight.
            gates = every_finite_bf16().reshape(T, 1)
            return {"shared": torch.ones(T, H, dtype=BF16), "gates": gates, "routed": torch.zeros(T, H, dtype=BF16)}
        return {"shared": shared.to(BF16), "gates": gates.to(BF16), "routed": routed.to(BF16)}

    # row_quant
    x = _hidden(g, T, H)
    if case.flavor == "stress":
        for row in range(T):
            pattern = row % 8
            if pattern == 0:
                x[row].zero_()  # a zero row: the scale is the epsilon alone
            elif pattern == 1:
                x[row].mul_(1e-20)
            elif pattern == 2:
                x[row].mul_(1e-9)  # the maximum is about the epsilon
            elif pattern == 3:
                x[row].mul_(1e-2)
                x[row, H // 3] = -1e4  # one spike carries the scale
            elif pattern == 4:
                x[row].mul_(1e30)
            elif pattern == 5:
                x[row, ::2] = 0.0
            elif pattern == 6:
                x[row] = torch.where(torch.rand(H, generator=g) < 0.5, -1.0, 1.0) * 3.0  # every element is the maximum
            # pattern 7: plain
    if case.flavor == "edge":
        # A maximum of 240 makes the scale 1: the elements are cast as they are.  Odd multiples of half
        # an fp8 step are exact ties of the cast, at every exponent of the format.
        exponent = torch.randint(-9, 8, (T, H), generator=g).float()
        eighths = torch.randint(8, 16, (T, H), generator=g).float()  # the 1.xxx mantissa, in eighths
        ties = (2.0 * eighths + 1.0) / 16.0 * torch.pow(2.0, exponent)
        x = torch.where(torch.rand(T, H, generator=g) < 0.5, -1.0, 1.0) * ties.clamp(max=236.0)
        x[:, 0] = 240.0
    if case.flavor == "every_amax":
        # Row r holds one copy of the r-th positive finite bf16 value and smaller values elsewhere,
        # so every possible row maximum goes through the scale arithmetic.
        patterns = torch.arange(1, T + 1, dtype=torch.int32).to(torch.int16)
        amax = patterns.view(BF16).float().reshape(T, 1)
        x = amax * torch.rand(T, H, generator=g) * torch.where(torch.rand(T, H, generator=g) < 0.5, -1.0, 1.0)
        x[torch.arange(T), torch.arange(T) % H] = amax.reshape(T) * torch.where(torch.arange(T) % 2 == 0, 1.0, -1.0)
    return {"x": x.to(BF16)}


def scalars(case: MoeCase) -> tuple:
    """The op's scalar arguments, in schema order, after the tensors."""
    return {
        "route": (case.experts, case.top_k),
        "silu_mul": (),
        "combine": (gate_column(case), ),
        "row_quant": (GAUDI2_FP8_MAX, ),
    }[case.kind]


def kernel_params(case: MoeCase) -> tuple[int, int, int, float]:
    """The 16-byte node parameter struct of the four GUIDs: (experts, top_k, gate_column, fp8_max); unused are 0."""
    if case.kind == "route":
        return case.experts, case.top_k, 0, 0.0
    if case.kind == "combine":
        return 0, 0, gate_column(case), 0.0
    if case.kind == "row_quant":
        return 0, 0, 0, GAUDI2_FP8_MAX
    return 0, 0, 0, 0.0


def expected(case: MoeCase, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The oracle's outputs, keyed by ``OUTPUT_ORDER[case.kind]``."""
    args = [inputs[name] for name in INPUT_ORDER[case.kind]]
    reference = {
        "route": moe_route_reference,
        "silu_mul": moe_silu_mul_reference,
        "combine": moe_combine_reference,
        "row_quant": moe_row_quant_reference,
    }[case.kind]
    out = reference(*args, *scalars(case))
    if not isinstance(out, tuple):
        out = (out, )
    return dict(zip(OUTPUT_ORDER[case.kind], out))


def input_shapes(case: MoeCase) -> dict[str, tuple[int, ...]]:
    T, H, C = case.rows, case.hidden, case.columns
    return {
        "route": {"gates": (T, C)},
        "silu_mul": {"gate_up": (T, 2 * H)},
        "combine": {"shared": (T, H), "gates": (T, C), "routed": (T, H)},
        "row_quant": {"x": (T, H)},
    }[case.kind]


def output_shapes(case: MoeCase) -> dict[str, tuple[int, ...]]:
    T, H = case.rows, case.hidden
    return {
        "route": {"ids": (T, case.top_k), "weights": (T, case.top_k)},
        "silu_mul": {"act": (T, H)},
        "combine": {"out": (T, H)},
        "row_quant": {"x_fp8": (T, H), "scale": (T, 1)},
    }[case.kind]


# ---------------------------------------------------------------------------
# Routing: what a valid answer is, and the kernel's own rule
# ---------------------------------------------------------------------------
def route_probabilities(case: MoeCase, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """The fp32 softmax the torch path takes its top-K of."""
    return F.softmax(inputs["gates"][:, :case.experts], dim=1, dtype=torch.float32)


def route_tie_rows(case: MoeCase, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask of the rows whose K-th and (K+1)-th largest probabilities are equal: torch.topk may pick either."""
    if case.experts <= case.top_k:
        return torch.zeros(case.rows, dtype=torch.bool)
    ordered = torch.sort(route_probabilities(case, inputs), dim=1, descending=True).values
    return ordered[:, case.top_k - 1] == ordered[:, case.top_k]


def route_kernel_ids(case: MoeCase, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """The kernel's selection: the K largest logits in descending order, the lowest expert id first among equals."""
    logits = inputs["gates"][:, :case.experts].float()
    order = torch.sort(logits, dim=1, descending=True, stable=True).indices
    return order[:, :case.top_k].to(torch.int32)


def route_expected_weights(case: MoeCase, inputs: dict[str, torch.Tensor], ids: torch.Tensor) -> torch.Tensor | None:
    """The torch weights of the experts ``ids`` [T, K], or None when ``ids`` is not a valid top-K of any row.

    A selection is valid when its probabilities are, as a multiset, the K largest of the row.  Equal
    probabilities get equal torch weights, so the weight of a selected expert is the torch weight of
    any torch-selected expert with the same probability.
    """
    p = route_probabilities(case, inputs)
    exp = expected(case, inputs)
    top_p = torch.gather(p, 1, exp["ids"].long())
    got_p = torch.gather(p, 1, ids.long())
    if not torch.equal(torch.sort(got_p, dim=1, descending=True).values,
                       torch.sort(top_p, dim=1, descending=True).values):
        return None
    same = got_p.unsqueeze(2) == top_p.unsqueeze(1)  # [T, K got, K torch]
    slot = same.float().argmax(dim=2)
    return torch.gather(exp["weights"], 1, slot)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def bf16_neighbours(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The bf16 values one step below and one step above every element of the bf16 tensor ``t``."""
    bits = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    signed = torch.where(bits >= 0x8000, -magnitude, magnitude)

    def from_ordered(value: torch.Tensor) -> torch.Tensor:
        pattern = torch.where(value < 0, (-value) | 0x8000, value)
        pattern = torch.where(pattern >= 0x8000, pattern - 0x10000, pattern)
        return pattern.to(torch.int16).view(BF16)

    return from_ordered(signed - 1), from_ordered(signed + 1)


def _inexact_limit(numel: int) -> int:
    return max(1, int(MAX_INEXACT_FRACTION * numel))


def _shape_message(case: MoeCase, name: str, have, want) -> str | None:
    if have is None:
        return f"{case.name}: {name}: missing"
    if tuple(have.shape) != tuple(want.shape) or have.dtype != want.dtype:
        return f"{case.name}: {name}: must be {tuple(want.shape)} {want.dtype}, got {tuple(have.shape)} {have.dtype}"
    return None


def _bf16_messages(case: MoeCase, name: str, have: torch.Tensor, want: torch.Tensor, passes: torch.Tensor,
                   flushed: torch.Tensor | None = None) -> list[str]:
    """Messages for a bf16 output given the mask of the elements that pass by the kind's rules.

    ``flushed`` marks the elements that differ from the oracle only below the fp32 normal range.
    """
    steps = bf16_steps(have, want)
    differs = steps != 0
    if flushed is not None and not case.exact:
        differs = differs & ~flushed
    inexact = int(differs.sum())
    if case.exact:
        if inexact:
            return [f"{case.name}: {name}: {inexact} of {want.numel()} elements are not exact "
                    f"(max {int(steps.max())} bf16 steps)"]
        return []
    msgs = []
    if not bool(passes.all()):
        diff = (have.float() - want.float()).abs()
        diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
        msgs.append(f"{case.name}: {name}: {int((~passes).sum())} of {want.numel()} elements are outside the "
                    f"tolerance (max {int(steps.max())} bf16 steps, max abs err {float(diff.max()):.3e})")
    elif inexact > _inexact_limit(want.numel()):
        msgs.append(f"{case.name}: {name}: {inexact} of {want.numel()} elements differ from the oracle, "
                    f"at most {_inexact_limit(want.numel())} may")
    return msgs


def _route_verdict(case: MoeCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]):
    """(messages, expected weights or None) for the ids of a route result."""
    ids = got["ids"]
    msgs = []
    if int(ids.min()) < 0 or int(ids.max()) >= case.experts:
        return [f"{case.name}: ids: expert ids outside [0, {case.experts})"], None
    ordered = torch.sort(ids, dim=1).values
    if case.top_k > 1 and bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        return [f"{case.name}: ids: a row selects an expert twice"], None
    want_weights = route_expected_weights(case, inputs, ids)
    if want_weights is None:
        return [f"{case.name}: ids: a row does not hold the {case.top_k} largest probabilities"], None
    exp = expected(case, inputs)
    clear = ~route_tie_rows(case, inputs)
    same_set = (ordered == torch.sort(exp["ids"], dim=1).values).all(dim=1)
    if not bool(same_set[clear].all()):
        msgs.append(f"{case.name}: ids: {int((~same_set[clear]).sum())} rows without a tie at the cut select "
                    f"other experts than torch.topk")
    rule = route_kernel_ids(case, inputs)
    if not torch.equal(ids, rule):
        rows = int((ids != rule).any(dim=1).sum())
        msgs.append(f"{case.name}: ids: {rows} of {case.rows} rows are not the largest logits in descending order "
                    f"with the lowest expert id first among equals")
    return msgs, want_weights


def _silu_mul_passes(inputs: dict[str, torch.Tensor], have: torch.Tensor,
                     want: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mask of the elements that pass, mask of those that differ only by the UNDERFLOW rule)."""
    gate_up = inputs["gate_up"]
    d = gate_up.shape[-1] // 2
    silu = F.silu(gate_up[..., :d])
    below, above = bf16_neighbours(silu)
    up = gate_up[..., d:]
    moved = (have.view(torch.int16) == (below * up).view(torch.int16)) | (have.view(torch.int16) ==
                                                                           (above * up).view(torch.int16))
    flushed = (have.float() - want.float()).abs() <= UNDERFLOW * up.float().abs()
    return (bf16_steps(have, want) <= MAX_ULP) | moved | flushed, flushed


def _combine_passes(case: MoeCase, inputs: dict[str, torch.Tensor],
                    have: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mask of the elements that pass, mask of those that differ only by the UNDERFLOW rule).

    An element passes within a step of the row computed with the oracle's gate weight or a neighbouring one.
    """
    shared, routed = inputs["shared"], inputs["routed"]
    logit = inputs["gates"][:, gate_column(case):gate_column(case) + 1]
    weight = torch.sigmoid(logit)
    below, above = bf16_neighbours(weight)
    slack = FP32_SLACK * (shared.float().abs() + routed.float().abs()) + UNDERFLOW * shared.float().abs()
    row_ok = torch.zeros(case.rows, dtype=torch.bool)
    passes = torch.zeros(have.shape, dtype=torch.bool)
    for candidate in (weight, below, above):
        want = candidate * shared + routed
        near = (bf16_steps(have, want) <= MAX_ULP) | ((have.float() - want.float()).abs() <= slack)
        take = near.all(dim=1) & ~row_ok
        passes[take] = True
        row_ok |= take
    oracle = weight * shared + routed
    flushed = (have.float() - oracle.float()).abs() <= UNDERFLOW * shared.float().abs()
    return passes, flushed


def compare(case: MoeCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> list[str]:
    """Failure messages for the tensors in ``got`` (keys of ``OUTPUT_ORDER``); [] means pass."""
    exp = expected(case, inputs)
    msgs = []
    for name in OUTPUT_ORDER[case.kind]:
        message = _shape_message(case, name, got.get(name), exp[name])
        if message:
            msgs.append(message)
    if msgs:
        return msgs

    if case.kind == "route":
        id_msgs, want_weights = _route_verdict(case, inputs, got)
        msgs += id_msgs
        if want_weights is not None:
            have = got["weights"]
            tiny = (have.float().abs() < FLT_MIN) & (want_weights.float().abs() < FLT_MIN)
            passes = (bf16_steps(have, want_weights) <= MAX_ULP) | tiny
            msgs += _bf16_messages(case, "weights", have, want_weights, passes, tiny)
        return msgs

    if case.kind == "silu_mul":
        passes, flushed = _silu_mul_passes(inputs, got["act"], exp["act"])
        return _bf16_messages(case, "act", got["act"], exp["act"], passes, flushed)

    if case.kind == "combine":
        passes, flushed = _combine_passes(case, inputs, got["out"])
        return _bf16_messages(case, "out", got["out"], exp["out"], passes, flushed)

    # row_quant: both outputs are exact
    if not torch.equal(got["scale"], exp["scale"]):
        bad = int((got["scale"] != exp["scale"]).sum())
        msgs.append(f"{case.name}: scale: {bad} of {case.rows} rows are not bit-exact")
    have, want = got["x_fp8"].float(), exp["x_fp8"].float()
    if not torch.equal(have, want):
        bad = have != want
        msgs.append(f"{case.name}: x_fp8: {int(bad.sum())} of {want.numel()} elements differ "
                    f"(max abs err {float((have - want).abs()[bad].max()):.3e})")
    return msgs


def summary(case: MoeCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> str:
    """One line of measured error per output, for test logs."""
    exp = expected(case, inputs)
    parts = []
    if case.kind == "route":
        ties = route_tie_rows(case, inputs)
        same_set = (torch.sort(got["ids"], dim=1).values == torch.sort(exp["ids"], dim=1).values).all(dim=1)
        want = route_expected_weights(case, inputs, got["ids"])
        parts.append(f"ids: rows={case.rows} tie_rows={int(ties.sum())} same_set_as_topk={int(same_set.sum())} "
                     f"same_order_as_topk={int((got['ids'] == exp['ids']).all(dim=1).sum())} valid={want is not None}")
        if want is not None:
            steps = bf16_steps(got["weights"], want)
            parts.append(f"weights: n={want.numel()} exact={float((steps == 0).float().mean()):.5f} "
                         f"max_steps={int(steps.max())} one_step={int((steps == 1).sum())} "
                         f"max_abs={float((got['weights'].float() - want.float()).abs().max()):.3e}")
    elif case.kind in ("silu_mul", "combine"):
        name = OUTPUT_ORDER[case.kind][0]
        have, want = got[name], exp[name]
        if case.kind == "silu_mul":
            passes, flushed = _silu_mul_passes(inputs, have, want)
        else:
            passes, flushed = _combine_passes(case, inputs, have)
        steps = bf16_steps(have, want)
        flushed = flushed & (steps != 0)
        counted = torch.where(flushed, torch.zeros_like(steps), steps)  # the steps of what is not a flush
        diff = (have.float() - want.float()).abs()
        diff = torch.where(torch.isfinite(diff) & ~flushed, diff, torch.zeros_like(diff))
        parts.append(f"{name}: n={want.numel()} exact={float((steps == 0).float().mean()):.5f} "
                     f"one_step={int((counted == 1).sum())} more={int((counted > 1).sum())} "
                     f"max_steps={int(counted.max())} flushed={int(flushed.sum())} fail={int((~passes).sum())} "
                     f"max_abs={float(diff.max()):.3e}")
    else:
        have, want = got["x_fp8"].float(), exp["x_fp8"].float()
        sign = int((got["x_fp8"].view(torch.uint8) != exp["x_fp8"].view(torch.uint8)).sum()) - int((have != want).sum())
        parts.append(f"scale: n={case.rows} exact={float((got['scale'] == exp['scale']).float().mean()):.5f}")
        parts.append(f"x_fp8: n={want.numel()} exact={float((have == want).float().mean()):.5f} "
                     f"zero_sign_differs={sign}")
    return f"MOECASE {case.name} | " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------
_RAW = {BF16: ("bf16", torch.int16, 2), torch.float32: ("f32", torch.float32, 4), torch.int32: ("i32", torch.int32, 4),
        F8: ("f8", torch.uint8, 1)}


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Little-endian header-less bytes of a tensor (bf16 and fp8 as their bit patterns, torch row-major order)."""
    _, storage, _ = _RAW[t.dtype]
    return t.contiguous().view(storage).numpy().tobytes()


def _read_raw(path: pathlib.Path, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    _, storage, size = _RAW[dtype]
    data = path.read_bytes()
    numel = 1
    for dim in shape:
        numel *= dim
    if len(data) != size * numel:
        raise ValueError(f"{path.name}: file size {len(data)} bytes, expected {size * numel} bytes")
    return torch.frombuffer(bytearray(data), dtype=storage).view(dtype).reshape(shape)


def _tpc_dims(shape) -> str:
    """Torch shape -> TPC dims, innermost (fastest) first."""
    return " ".join(str(d) for d in reversed(tuple(shape)))


def write_inputs(case: MoeCase, inputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """One raw file ``<name>.bin`` per input tensor plus the ``case.txt`` manifest ``sim_host`` reads."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    experts, top_k, column, fp8_max = kernel_params(case)
    lines = [f"guid {GUIDS[case.kind]}", f"experts {experts}", f"top_k {top_k}", f"gate_column {column}",
             f"fp8_max {fp8_max!r}"]
    for name in INPUT_ORDER[case.kind]:
        (directory / f"{name}.bin").write_bytes(_raw_bytes(inputs[name]))
        lines.append(f"in {name} {_RAW[inputs[name].dtype][0]} {_tpc_dims(inputs[name].shape)}")
    for name, shape in output_shapes(case).items():
        lines.append(f"out {name} {_RAW[OUTPUT_DTYPE[name]][0]} {_tpc_dims(shape)}")
    (directory / "case.txt").write_text("\n".join(lines) + "\n")


def write_outputs(case: MoeCase, outputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """Write ``<name>.out.bin`` per output (the format the simulator host program writes)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_ORDER[case.kind]:
        (directory / f"{name}.out.bin").write_bytes(_raw_bytes(outputs[name]))


def read_outputs(case: MoeCase, directory: pathlib.Path) -> dict[str, torch.Tensor]:
    """Read ``<name>.out.bin`` back; ``ValueError`` naming the file on a size mismatch."""
    directory = pathlib.Path(directory)
    return {name: _read_raw(directory / f"{name}.out.bin", shape, OUTPUT_DTYPE[name])
            for name, shape in output_shapes(case).items()}
