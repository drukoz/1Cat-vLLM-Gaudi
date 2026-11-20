# SPDX-License-Identifier: Apache-2.0
"""Shared test cases for the fused hyper-connection kernels (``csrc/hc_fused``).

One module defines the cases, their inputs, the expected outputs, the pass/fail comparison and the
raw-file I/O.  The TPC simulator test (``tests/standalone/hc_fused/test_hc_simulation.py``), the HPU op
test (``test_hpu_op.py``) and the CPU tests all import these names.  The raw format (one header-less
little-endian file per tensor, bf16 stored as its 16-bit pattern) plus ``case.txt`` is what the C++
simulator host program reads, so it must not change without ``sim_host.cpp``.

Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

from tests.unit_tests.hc_fused.hc_reference import (hc_combine_norm_reference, hc_combine_norm_unrounded,
                                                    hc_combine_reference, hc_gate_mix_reference, hc_norm_reference,
                                                    hc_silu_reference, hc_silu_split_reference)

BF16 = torch.bfloat16
KINDS = ("norm", "silu", "silu_split", "gate_mix", "combine", "combine_norm")
# Kernel GUIDs; the same strings are in csrc/hc_fused/hc_fused_params.h (test_sources.py and
# test_hc_library.py check that they agree).
GUIDS = {
    "norm": "hc_fused_norm_bf16_gaudi2",
    "silu": "hc_fused_silu_bf16_gaudi2",
    "silu_split": "hc_fused_silu_split_bf16_gaudi2",
    "gate_mix": "hc_fused_gate_mix_bf16_gaudi2",
    "combine": "hc_fused_combine_bf16_gaudi2",
    "combine_norm": "hc_fused_combine_norm_bf16_gaudi2",
}
# The op of each kind: torch.ops.flashinfer_gaudi.<name> (schemas in hc_ops.py).
OP_NAME = {kind: f"hc_{kind}" for kind in KINDS}
# Op argument order (tensors only) and output names, per kind.
INPUT_ORDER = {
    "norm": ("x", "norm_weight"),
    "silu": ("down", ),
    "silu_split": ("down", ),
    "gate_mix": ("xn", "gate"),
    "combine": ("residual", "block_output", "injection"),
    "combine_norm": ("residual", "block_output", "injection", "norm_weight"),
}
OUTPUT_ORDER = {
    "norm": ("xn", ),
    "silu": ("lora", ),
    "silu_split": ("lora", "injection_out"),
    "gate_mix": ("block_input", ),
    "combine": ("residual_out", ),
    "combine_norm": ("residual_out", "xn"),
}
VECTOR = 128  # bf16 lanes of one TPC vector; the kernels need hidden % VECTOR == 0

# Tolerances.  The kernels work in fp32 like the oracle and round to bf16 at the oracle's rounding
# points, so an element passes when one of these holds:
#   1. it is at most one bf16 step from the oracle (a 1e-7 fp32 difference of the TPC
#      rsqrt/exp/reciprocal against torch can flip a rounding).  One bf16 step is at most
#      2**-7 = 0.78 % of the value, inside the bf16 tolerance of
#      docs/design/flash-next/gdn-core-decode.md (atol 5e-2, rtol 1e-2);
#   2. it is within ATOL (values that are zero for every purpose);
#   3. it is within FP32_SLACK of the magnitude of the operands of its last sum.  This is the
#      cancellation rule: where residual + block * weight (or the gated sum) cancels to a millionth
#      of its operands, a one-fp32-step difference in the sigmoid is many bf16 steps of the tiny
#      result.  About one element in a million of a random input passes only by this rule;
#   4. (xn of combine_norm only) it is within the difference of its own residual_out element times
#      the norm factor, plus one bf16 step.  This is the propagation rule: the norm is taken of the
#      kernel's rounded combine, so a combine element that rounded one step away (rule 1) moves its
#      xn by the same 0.4 to 0.8 %, which is one or two steps of xn.  About two elements in a
#      million.  The norm stage on its own is held to rules 1 and 2: xn must be the oracle's norm
#      of the kernel's own residual_out (compare()'s "norm of its own residual_out" message).
# On top of that, MIN_EXACT_FRACTION of every output must be bit-exact, and every element of every
# output must be inside the bf16 tolerance the GDN track documents (GDN_BF16_ATOL, GDN_BF16_RTOL),
# whatever rule it passed by.
MAX_ULP = 1
ATOL = 1e-6
FP32_SLACK = 1e-6  # about 8 fp32 steps of the operands
MIN_EXACT_FRACTION = 0.99
GDN_BF16_ATOL, GDN_BF16_RTOL = 5e-2, 1e-2  # docs/design/flash-next/gdn-core-decode.md, "Measured tolerances"


@dataclasses.dataclass(frozen=True)
class HcCase:
    name: str
    kind: str
    hidden: int  # HS: one stream; the multi-stream width is hc * hidden.  Unused by the silu kinds.
    rows: int  # T: tokens
    hc: int = 4
    lora: int = 16  # silu kinds: hc_lowrank
    pad: int = 12  # silu_split: padding columns after the injection logits
    shared_weight: bool = False  # norm kinds: one [hidden] affine for every stream instead of [hc * hidden]
    flavor: str = "random"  # "random" | "stress" | "edge" | "rounding"
    seed: int = 0
    eps: float = 1e-6
    exact: bool = False  # every element must be bit-exact (bf16 rounding-edge cases)


def _c(name, kind, hidden, rows, **kw) -> HcCase:
    return HcCase(name=name, kind=kind, hidden=hidden, rows=rows, **kw)


# Geometries: tiny test model 4 x 128 (hc_lowrank 16), real model 4 x 2560 (hc_lowrank 320, merged
# down+inject GEMM padded to 336 columns).  Decode rows 1..32, prefill-sized 2048.
SIM_CASES = (
    # ---- hc_norm: grouped Gemma RMSNorm --------------------------------------------------------
    _c("norm_tiny_r1", "norm", 128, 1, seed=1),
    _c("norm_tiny_r3_shared", "norm", 128, 3, shared_weight=True, seed=2),
    _c("norm_real_r1", "norm", 2560, 1, seed=3),
    _c("norm_real_r2_shared", "norm", 2560, 2, shared_weight=True, seed=4),
    _c("norm_real_r16", "norm", 2560, 16, seed=5),
    _c("norm_real_r32", "norm", 2560, 32, seed=6),
    _c("norm_prefill_r2048", "norm", 2560, 2048, seed=7),
    _c("norm_tiny_stress", "norm", 128, 3, flavor="stress", seed=8),
    _c("norm_real_stress_shared", "norm", 2560, 3, shared_weight=True, flavor="stress", seed=9),
    _c("norm_hc2_h384_r2", "norm", 384, 2, hc=2, seed=10),
    # ---- hc_silu_split / hc_silu ----------------------------------------------------------------
    _c("silu_split_tiny_r1", "silu_split", 128, 1, seed=11),
    _c("silu_split_tiny_r3", "silu_split", 128, 3, seed=12),
    _c("silu_split_real_r1", "silu_split", 2560, 1, lora=320, seed=13),
    _c("silu_split_real_r2", "silu_split", 2560, 2, lora=320, seed=14),
    _c("silu_split_real_r16", "silu_split", 2560, 16, lora=320, seed=15),
    _c("silu_split_real_r32", "silu_split", 2560, 32, lora=320, seed=16),
    _c("silu_split_prefill_r2048", "silu_split", 2560, 2048, lora=320, seed=17),
    _c("silu_split_tiny_stress", "silu_split", 128, 3, flavor="stress", seed=18),
    _c("silu_split_real_stress", "silu_split", 2560, 3, lora=320, flavor="stress", seed=19),
    _c("silu_split_lora130_hc3_r3", "silu_split", 128, 3, hc=3, lora=130, pad=11, seed=20),
    _c("silu_tiny_r3", "silu", 128, 3, seed=21),
    _c("silu_real_r32", "silu", 2560, 32, lora=320, seed=22),
    _c("silu_real_stress", "silu", 2560, 3, lora=320, flavor="stress", seed=23),
    # ---- hc_gate_mix ----------------------------------------------------------------------------
    _c("gate_mix_tiny_r1", "gate_mix", 128, 1, seed=31),
    _c("gate_mix_tiny_r3", "gate_mix", 128, 3, seed=32),
    _c("gate_mix_real_r1", "gate_mix", 2560, 1, seed=33),
    _c("gate_mix_real_r2", "gate_mix", 2560, 2, seed=34),
    _c("gate_mix_real_r16", "gate_mix", 2560, 16, seed=35),
    _c("gate_mix_real_r32", "gate_mix", 2560, 32, seed=36),
    _c("gate_mix_prefill_r2048", "gate_mix", 2560, 2048, seed=37),
    _c("gate_mix_tiny_stress", "gate_mix", 128, 3, flavor="stress", seed=38),
    _c("gate_mix_real_stress", "gate_mix", 2560, 3, flavor="stress", seed=39),
    _c("gate_mix_tiny_edge", "gate_mix", 128, 3, flavor="edge", seed=40, exact=True),
    _c("gate_mix_hc3_h256_r2", "gate_mix", 256, 2, hc=3, seed=41),
    # ---- hc_combine -----------------------------------------------------------------------------
    _c("combine_tiny_r1", "combine", 128, 1, seed=51),
    _c("combine_tiny_r3", "combine", 128, 3, seed=52),
    _c("combine_real_r1", "combine", 2560, 1, seed=53),
    _c("combine_real_r2", "combine", 2560, 2, seed=54),
    _c("combine_real_r16", "combine", 2560, 16, seed=55),
    _c("combine_real_r32", "combine", 2560, 32, seed=56),
    _c("combine_prefill_r2048", "combine", 2560, 2048, seed=57),
    _c("combine_tiny_stress", "combine", 128, 3, flavor="stress", seed=58),
    _c("combine_real_stress", "combine", 2560, 3, flavor="stress", seed=59),
    _c("combine_tiny_edge", "combine", 128, 3, flavor="edge", seed=60, exact=True),
    # ---- hc_combine_norm ------------------------------------------------------------------------
    _c("combine_norm_tiny_r1", "combine_norm", 128, 1, seed=71),
    _c("combine_norm_tiny_r3_shared", "combine_norm", 128, 3, shared_weight=True, seed=72),
    _c("combine_norm_real_r1", "combine_norm", 2560, 1, seed=73),
    _c("combine_norm_real_r2_shared", "combine_norm", 2560, 2, shared_weight=True, seed=74),
    _c("combine_norm_real_r16", "combine_norm", 2560, 16, seed=75),
    _c("combine_norm_real_r32", "combine_norm", 2560, 32, seed=76),
    _c("combine_norm_prefill_r2048", "combine_norm", 2560, 2048, seed=77),
    _c("combine_norm_tiny_stress", "combine_norm", 128, 3, flavor="stress", seed=78),
    _c("combine_norm_real_stress_shared", "combine_norm", 2560, 3, shared_weight=True, flavor="stress", seed=79),
    _c("combine_norm_tiny_rounding", "combine_norm", 128, 3, flavor="rounding", seed=80),
    _c("combine_norm_real_rounding_shared", "combine_norm", 2560, 2, shared_weight=True, flavor="rounding", seed=81),
    _c("combine_norm_hc2_h384_r2", "combine_norm", 384, 2, hc=2, seed=82),
)
# HPU only: every decode batch bucket and the largest prefill, per kernel that sits in the decode loop.
HPU_CASES = SIM_CASES + tuple(
    _c(f"{kind}_real_r{rows}", kind, 2560, rows, lora=320, seed=100 + 10 * i + j)
    for i, kind in enumerate(("norm", "silu_split", "gate_mix", "combine", "combine_norm"))
    for j, rows in enumerate((4, 8, 12, 20, 8192)))
ROUNDING_CASES = tuple(case for case in SIM_CASES if case.flavor == "rounding")


def case_by_name(name: str) -> HcCase:
    for case in HPU_CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def lora_columns(case: HcCase) -> int:
    """Width of the ``down`` input of the silu kinds."""
    return case.lora + case.hc + case.pad if case.kind == "silu_split" else case.lora


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# Values that stress the sigmoid (gate, injection logits, lora): zero, small, the bends, and far
# into both saturated tails.  All are finite; every one is a value bf16 can hold.
_SIGMOID_STRESS = (0.0, -0.0, 1e-3, -1e-3, 0.5, -0.5, 1.0, -1.0, 4.0, -4.0, 8.0, -8.0, 17.0, -17.0, 20.0, -20.0, 36.0,
                   -36.0, 40.0, -40.0, 88.0, -88.0, 100.0, -100.0, 400.0, -400.0, 1e4, -1e4, 1e30, -1e30)


def _hidden(g: torch.Generator, case: HcCase) -> torch.Tensor:
    """A multi-stream residual [T, hc, hidden] (fp32): unit normal times one log-normal scale per stream."""
    scale = torch.exp(torch.randn(case.rows, case.hc, 1, generator=g))
    return torch.randn(case.rows, case.hc, case.hidden, generator=g) * scale


def _stress_hidden(g: torch.Generator, case: HcCase) -> torch.Tensor:
    """Groups that stress rsqrt(mean square + eps): zero, far below eps, around eps, a spike, and huge."""
    x = _hidden(g, case)
    patterns = 8
    for row in range(case.rows):
        for stream in range(case.hc):
            pattern = (row * case.hc + stream) % patterns
            group = x[row, stream]
            if pattern == 0:
                group.zero_()  # mean square 0: rsqrt(eps)
            elif pattern == 1:
                group.mul_(1e-20)  # squares underflow fp32
            elif pattern == 2:
                group.mul_(1e-4)  # mean square 1e-8, far below eps
            elif pattern == 3:
                group.mul_(1e-3)  # mean square about eps
            elif pattern == 4:
                group.mul_(1e-2)
                group[case.hidden // 3] = 1e4  # one spike carries the whole norm
            elif pattern == 5:
                group.mul_(1e15)  # mean square about 1e30
            elif pattern == 6:
                group[::2] = 0.0  # half the elements exactly zero
            # pattern 7: plain
    return x


def _stress_values(g: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    table = torch.tensor(_SIGMOID_STRESS)
    index = torch.randint(0, table.numel(), shape, generator=g)
    return table[index]


def _norm_weight(g: torch.Generator, case: HcCase) -> torch.Tensor:
    size = case.hidden if case.shared_weight else case.hc * case.hidden
    return (0.25 * torch.randn(size, generator=g)).to(BF16)


def make_inputs(case: HcCase) -> dict[str, torch.Tensor]:
    """CPU bf16 tensors keyed by ``INPUT_ORDER[case.kind]``; deterministic from ``case.seed``."""
    g = torch.Generator().manual_seed(case.seed)
    T, HC, HS = case.rows, case.hc, case.hidden
    W = HC * HS
    stress = case.flavor == "stress"

    if case.kind == "norm":
        x = _stress_hidden(g, case) if stress else _hidden(g, case)
        return {"x": x.reshape(T, W).to(BF16), "norm_weight": _norm_weight(g, case)}

    if case.kind in ("silu", "silu_split"):
        columns = lora_columns(case)
        down = torch.randn(T, columns, generator=g) * 6.0
        if stress:
            down = _stress_values(g, (T, columns))
        return {"down": down.to(BF16)}

    if case.kind == "gate_mix":
        if case.flavor == "edge":
            # Saturated gates select streams exactly (sigmoid = 1 or 0), so the fp32 sum of the selected
            # bf16 values is exact and a quarter of the results sit exactly between two bf16 values.
            mantissa = torch.randint(128, 256, (T, HC, HS), generator=g).float() / 128.0
            exponent = torch.randint(-6, 7, (T, 1, HS), generator=g).float()
            sign = torch.where(torch.rand(T, 1, HS, generator=g) < 0.5, -1.0, 1.0)
            xn = sign * mantissa * torch.pow(2.0, exponent)
            # +100: sigmoid is 1 in fp32.  -1e4: exp overflows, sigmoid is exactly 0 (no denormal products).
            gate = torch.where(torch.rand(T, HC, HS, generator=g) < 0.7, 100.0, -1e4)
            return {"xn": xn.reshape(T, W).to(BF16), "gate": gate.reshape(T, W).to(BF16)}
        xn = _stress_hidden(g, case) if stress else torch.randn(T, HC, HS, generator=g) * 1.5
        gate = _stress_values(g, (T, W)) if stress else torch.randn(T, W, generator=g) * 3.0
        return {"xn": xn.reshape(T, W).to(BF16), "gate": gate.to(BF16)}

    # combine / combine_norm
    residual = _stress_hidden(g, case) if stress else _hidden(g, case)
    block = torch.randn(T, HS, generator=g) * torch.exp(torch.randn(T, 1, generator=g))
    injection = _stress_values(g, (T, HC)) if stress else torch.randn(T, HC, generator=g) * 3.0
    if case.flavor == "edge":
        # Zero logits give the weight 2 * sigmoid(0) = 1 exactly.  The residual is m * 2**(e - 7) with
        # m in [128, 256), the block output is k quarter-steps of that grid: k = 2 mod 4 is an exact tie.
        mantissa = torch.randint(128, 256, (T, HC, HS), generator=g).float()
        exponent = torch.randint(-12, 13, (T, 1, HS), generator=g).float()
        step = torch.pow(2.0, exponent - 7.0)
        sign = torch.where(torch.rand(T, 1, HS, generator=g) < 0.5, -1.0, 1.0)
        residual = sign * mantissa * step
        quarters = torch.randint(-7, 8, (T, 1, HS), generator=g).float()
        block = (quarters * step / 4.0).reshape(T, HS)
        injection = torch.zeros(T, HC)
    if case.flavor == "rounding":
        # Half the columns lose their block contribution when the combine is rounded to bf16 (it is just
        # under half a bf16 step of the residual), the other half have none.  The group's mean square
        # therefore depends on whether the combine was rounded before the norm.
        exponent = torch.randint(-3, 4, (T, 1, HS), generator=g).float()
        residual = torch.pow(2.0, exponent).expand(T, HC, HS).clone()
        lossy = (torch.arange(HS) % 2 == 0).reshape(1, HS)
        block = torch.where(lossy, 0.97 * torch.pow(2.0, exponent.reshape(T, HS) - 8.0), torch.zeros(T, HS))
        injection = torch.zeros(T, HC)
    inputs = {
        "residual": residual.reshape(T, W).to(BF16),
        "block_output": block.to(BF16),
        "injection": injection.to(BF16),
    }
    if case.kind == "combine_norm":
        inputs["norm_weight"] = _norm_weight(g, case)
    return inputs


def scalars(case: HcCase) -> tuple:
    """The op's scalar arguments, in schema order, after the tensors."""
    return {
        "norm": (case.eps, case.hc),
        "silu": (case.hc, ),
        "silu_split": (case.hc, case.lora),
        "gate_mix": (case.hc, ),
        "combine": (case.hc, ),
        "combine_norm": (case.eps, case.hc),
    }[case.kind]


def kernel_params(case: HcCase) -> tuple[float, int, int]:
    """The 12-byte node parameter struct every GUID takes: (eps, hc_count, lora_rank); unused fields are 0."""
    eps = case.eps if case.kind in ("norm", "combine_norm") else 0.0
    lora = case.lora if case.kind == "silu_split" else 0
    return eps, case.hc, lora


def expected(case: HcCase, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The oracle's outputs, keyed by ``OUTPUT_ORDER[case.kind]``."""
    args = [inputs[name] for name in INPUT_ORDER[case.kind]]
    reference = {
        "norm": hc_norm_reference,
        "silu": hc_silu_reference,
        "silu_split": hc_silu_split_reference,
        "gate_mix": hc_gate_mix_reference,
        "combine": hc_combine_reference,
        "combine_norm": hc_combine_norm_reference,
    }[case.kind]
    out = reference(*args, *scalars(case))
    if not isinstance(out, tuple):
        out = (out, )
    return dict(zip(OUTPUT_ORDER[case.kind], out))


def input_shapes(case: HcCase) -> dict[str, tuple[int, ...]]:
    """Torch shapes of ``make_inputs(case)``, without building the tensors."""
    T, W = case.rows, case.hc * case.hidden
    weight = (case.hidden if case.shared_weight else W, )
    return {
        "norm": {"x": (T, W), "norm_weight": weight},
        "silu": {"down": (T, lora_columns(case))},
        "silu_split": {"down": (T, lora_columns(case))},
        "gate_mix": {"xn": (T, W), "gate": (T, W)},
        "combine": {"residual": (T, W), "block_output": (T, case.hidden), "injection": (T, case.hc)},
        "combine_norm": {"residual": (T, W), "block_output": (T, case.hidden), "injection": (T, case.hc),
                         "norm_weight": weight},
    }[case.kind]


def output_shapes(case: HcCase) -> dict[str, tuple[int, ...]]:
    T, W = case.rows, case.hc * case.hidden
    return {
        "norm": {"xn": (T, W)},
        "silu": {"lora": (T, case.lora)},
        "silu_split": {"lora": (T, case.lora), "injection_out": (T, case.hc)},
        "gate_mix": {"block_input": (T, case.hidden)},
        "combine": {"residual_out": (T, W)},
        "combine_norm": {"residual_out": (T, W), "xn": (T, W)},
    }[case.kind]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def bf16_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance between two bf16 tensors in representable values (0 = same value; +0 and -0 are 0 apart)."""

    def ordered(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        magnitude = bits & 0x7FFF
        return torch.where(bits >= 0x8000, -magnitude, magnitude)

    return (ordered(a) - ordered(b)).abs()


def metrics(got: torch.Tensor, exp: torch.Tensor) -> dict[str, float]:
    """Error figures of one output against the oracle (both bf16)."""
    g, e = got.float(), exp.float()
    diff = (g - e).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    significant = e.abs() > 1e-3
    rel = torch.where(significant, diff / e.abs().clamp_min(1e-30), torch.zeros_like(diff))
    steps = bf16_steps(got, exp)
    return {
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "max_rel": float(rel.max()) if rel.numel() else 0.0,
        "max_steps": int(steps.max()) if steps.numel() else 0,
        "exact": float((steps == 0).float().mean()) if steps.numel() else 1.0,
    }


def _norm_factor(case: HcCase, inputs: dict[str, torch.Tensor], out: torch.Tensor) -> torch.Tensor:
    """rsqrt(mean square + eps) * |1 + w| of every element of a [T, hc * hidden] combine result `out`."""
    T, HC, HS = case.rows, case.hc, case.hidden
    grouped = out.float().reshape(T, HC, HS)
    rrms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + case.eps)
    weight = inputs["norm_weight"].float()
    weight = weight.view(1, 1, HS) if weight.numel() == HS else weight.view(1, HC, HS)
    return (rrms * (1.0 + weight).abs()).reshape(T, HC * HS)


def bf16_spacing(t: torch.Tensor) -> torch.Tensor:
    """The distance between neighbouring bf16 values at the magnitude of each (fp32) element."""
    _, exponent = torch.frexp(t.abs().clamp_min(1e-37))
    return torch.pow(2.0, (exponent - 8).float())


def allowance(case: HcCase, inputs: dict[str, torch.Tensor], exp: dict[str, torch.Tensor],
              got: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per output, the absolute error rules 3 and 4 of the tolerances allow each element.

    Outputs that come from products of the inputs only (the norm of an input, silu, the injection
    copy) have no entry: they pass by rules 1 and 2 or not at all.
    """
    T, HC, HS = case.rows, case.hc, case.hidden
    if case.kind == "gate_mix":
        xn = inputs["xn"].float().reshape(T, HC, HS)
        return {"block_input": FP32_SLACK * xn.abs().sum(dim=1) / HC}  # sigmoid <= 1
    if case.kind in ("combine", "combine_norm"):
        residual = inputs["residual"].float().reshape(T, HC, HS)
        block = inputs["block_output"].float().reshape(T, 1, HS)
        scale = (residual.abs() + 2.0 * block.abs()).reshape(T, HC * HS)  # the combine weight is at most 2
        allowed = {"residual_out": FP32_SLACK * scale}
        have = got.get("residual_out")
        if case.kind == "combine_norm":
            factor = _norm_factor(case, inputs, exp["residual_out"])
            allowed["xn"] = FP32_SLACK * scale * factor
            if have is not None and have.shape == exp["residual_out"].shape and have.dtype == BF16:
                moved = (have.float() - exp["residual_out"].float()).abs()
                shift = 1.01 * moved * factor  # what the moved combine element does to its xn
                propagated = shift + bf16_spacing(exp["xn"].float().abs() + shift)  # plus the rounding of xn
                allowed["xn"] = allowed["xn"] + torch.where(moved > 0, propagated, torch.zeros_like(moved))
        return allowed
    return {}


def _element_verdict(name: str, have: torch.Tensor, want: torch.Tensor,
                     allowed: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(mask of elements that fail, mask of elements that pass only by rule 3 or 4)."""
    steps = bf16_steps(have, want)
    diff = (have.float() - want.float()).abs()
    near = (steps <= MAX_ULP) | (diff <= ATOL)
    if name in allowed:
        extra = ~near & (diff <= allowed[name])
    else:
        extra = torch.zeros_like(near)
    return ~(near | extra), extra


def compare(case: HcCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> list[str]:
    """Failure messages for the tensors in ``got`` (keys of ``OUTPUT_ORDER``); [] means pass."""
    exp = expected(case, inputs)
    allowed = allowance(case, inputs, exp, got)
    msgs = []
    for name in OUTPUT_ORDER[case.kind]:
        want = exp[name]
        have = got.get(name)
        if have is None:
            msgs.append(f"{case.name}: {name}: missing")
            continue
        if tuple(have.shape) != tuple(want.shape) or have.dtype != want.dtype:
            msgs.append(f"{case.name}: {name}: must be {tuple(want.shape)} {want.dtype}, "
                        f"got {tuple(have.shape)} {have.dtype}")
            continue
        if not torch.isfinite(have.float()).all() and torch.isfinite(want.float()).all():
            msgs.append(f"{case.name}: {name}: has non-finite values")
            continue
        m = metrics(have, want)
        outside = (have.float() - want.float()).abs() > GDN_BF16_ATOL + GDN_BF16_RTOL * want.float().abs()
        if outside.any():
            msgs.append(f"{case.name}: {name}: {int(outside.sum())} of {want.numel()} elements are outside the GDN "
                        f"bf16 tolerance (atol {GDN_BF16_ATOL:g}, rtol {GDN_BF16_RTOL:g})")
        steps = bf16_steps(have, want)
        if name == "injection_out":  # a copy: the same bits, the sign of zero included
            if not torch.equal(have.view(torch.int16), want.view(torch.int16)):
                bad = int((have.view(torch.int16) != want.view(torch.int16)).sum())
                msgs.append(f"{case.name}: {name}: {bad} of {want.numel()} elements are not the same bits")
            continue
        if case.exact:  # the same value everywhere (+0 and -0 are the same value)
            if int(steps.max()) != 0:
                msgs.append(f"{case.name}: {name}: {int((steps != 0).sum())} of {want.numel()} elements are not "
                            f"exact (max {m['max_steps']} bf16 steps)")
            continue
        bad, _ = _element_verdict(name, have, want, allowed)
        if bad.any():
            msgs.append(f"{case.name}: {name}: {int(bad.sum())} of {want.numel()} elements are more than "
                        f"{MAX_ULP} bf16 step, {ATOL:g} and {FP32_SLACK:g} of their operands away "
                        f"(max abs err {m['max_abs']:.3e}, max rel err {m['max_rel']:.3e}, max {m['max_steps']} steps)")
        elif m["exact"] < MIN_EXACT_FRACTION:
            msgs.append(f"{case.name}: {name}: only {m['exact']:.4f} of the elements are bit-exact, "
                        f"need {MIN_EXACT_FRACTION}")
    if case.kind == "combine_norm" and all(
            got.get(name) is not None and tuple(got[name].shape) == tuple(exp[name].shape) and got[name].dtype == BF16
            for name in ("residual_out", "xn")):
        msgs += _own_norm_messages(case, inputs, got)
        if case.flavor == "rounding":
            msgs += _rounding_point_messages(case, inputs, got, exp)
    return msgs


def _own_norm_messages(case: HcCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> list[str]:
    """xn must be the oracle's grouped norm of the kernel's own (rounded, bf16) residual_out: rules 1 and 2 only."""
    want = hc_norm_reference(got["residual_out"], inputs["norm_weight"], case.eps, case.hc)
    bad, _ = _element_verdict("xn", got["xn"], want, {})
    if bad.any():
        m = metrics(got["xn"], want)
        return [f"{case.name}: xn is not the norm of its own residual_out: {int(bad.sum())} of {want.numel()} "
                f"elements are more than {MAX_ULP} bf16 step and {ATOL:g} away (max {m['max_steps']} steps)"]
    return []


def rounding_discriminators(case: HcCase, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask of the ``xn`` elements where rounding the combine to bf16 before the norm changes the result."""
    exp = expected(case, inputs)
    wrong = hc_combine_norm_unrounded(inputs["residual"], inputs["block_output"], inputs["injection"],
                                      inputs["norm_weight"], case.eps, case.hc)
    return exp["xn"].view(torch.int16) != wrong.view(torch.int16)


def _rounding_point_messages(case, inputs, got, exp) -> list[str]:
    """The oracle rounds the combine to bf16 before the norm; the kernel must do the same."""
    mask = rounding_discriminators(case, inputs)
    count = int(mask.sum())
    if count < 100:
        return [f"{case.name}: only {count} elements tell rounded from unrounded; the case does not pin the rounding"]
    agree = float((got["xn"].view(torch.int16)[mask] == exp["xn"].view(torch.int16)[mask]).float().mean())
    if agree < 0.95:
        return [f"{case.name}: xn matches the rounded-combine oracle on only {agree:.3f} of the {count} elements "
                f"that tell it from an unrounded combine: the kernel does not round the combine before the norm"]
    return []


def summary(case: HcCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> str:
    """One line of measured error per output, for test logs."""
    exp = expected(case, inputs)
    allowed = allowance(case, inputs, exp, got)
    parts = []
    for name in OUTPUT_ORDER[case.kind]:
        m = metrics(got[name], exp[name])
        bad, extra = _element_verdict(name, got[name], exp[name], allowed)
        steps = bf16_steps(got[name], exp[name])
        one = int((steps == 1).sum())  # rule 1
        tiny = int(((steps > MAX_ULP) & ~extra & ~bad).sum())  # rule 2: more than a step apart but within ATOL
        parts.append(f"{name}: n={exp[name].numel()} exact={m['exact']:.5f} max_steps={m['max_steps']} "
                     f"max_abs={m['max_abs']:.3e} max_rel={m['max_rel']:.3e} "
                     f"one_step={one} tiny={tiny} rule34={int(extra.sum())} fail={int(bad.sum())}")
    return f"HCCASE {case.name} | " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------
def _raw_bytes(t: torch.Tensor) -> bytes:
    """Little-endian header-less bytes of a bf16 tensor (its 16-bit patterns, torch row-major order)."""
    assert t.dtype == BF16
    return t.contiguous().view(torch.int16).numpy().tobytes()


def _read_raw(path: pathlib.Path, shape: tuple[int, ...]) -> torch.Tensor:
    data = path.read_bytes()
    numel = 1
    for dim in shape:
        numel *= dim
    if len(data) != 2 * numel:
        raise ValueError(f"{path.name}: file size {len(data)} bytes, expected {2 * numel} bytes")
    return torch.frombuffer(bytearray(data), dtype=torch.int16).view(BF16).reshape(shape)


def _tpc_dims(shape) -> str:
    """Torch shape -> TPC dims, innermost (fastest) first."""
    return " ".join(str(d) for d in reversed(tuple(shape)))


def write_inputs(case: HcCase, inputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """One raw file ``<name>.bin`` per input tensor plus the ``case.txt`` manifest ``sim_host`` reads."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    eps, hc_count, lora_rank = kernel_params(case)
    lines = [f"guid {GUIDS[case.kind]}", f"eps {eps!r}", f"hc_count {hc_count}", f"lora_rank {lora_rank}"]
    for name in INPUT_ORDER[case.kind]:
        (directory / f"{name}.bin").write_bytes(_raw_bytes(inputs[name]))
        lines.append(f"in {name} bf16 {_tpc_dims(inputs[name].shape)}")
    for name, shape in output_shapes(case).items():
        lines.append(f"out {name} bf16 {_tpc_dims(shape)}")
    (directory / "case.txt").write_text("\n".join(lines) + "\n")


def write_outputs(case: HcCase, outputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """Write ``<name>.out.bin`` per output (the format the simulator host program writes)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_ORDER[case.kind]:
        (directory / f"{name}.out.bin").write_bytes(_raw_bytes(outputs[name]))


def read_outputs(case: HcCase, directory: pathlib.Path) -> dict[str, torch.Tensor]:
    """Read ``<name>.out.bin`` back; ``ValueError`` naming the file on a size mismatch."""
    directory = pathlib.Path(directory)
    return {name: _read_raw(directory / f"{name}.out.bin", shape) for name, shape in output_shapes(case).items()}
