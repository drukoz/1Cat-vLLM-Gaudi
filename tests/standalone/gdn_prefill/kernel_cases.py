# SPDX-License-Identifier: Apache-2.0
"""Cases, torch references and raw-file I/O for the GDN prefill kernels of ``csrc/flashqla_hpu``.

Pure torch, no vLLM import: the TPC simulator test (``test_simulation.py`` in this folder, run in the
tpcsim container where vLLM is not installed) and the CPU test
``tests/unit_tests/ops/test_gdn_prefill_kernel_cases.py`` (which ties these references to the torch
code of ``vllm_gaudi/ops/hpu_gdn_pytorch.py`` that the kernels replace) both import this module as
``tests.standalone.gdn_prefill.kernel_cases``.

Geometry of one rank (head dim 128 everywhere): TP1 16 key / 48 value heads, packed q|k|v width 10240;
TP2 8 / 24, 5120; TP4 4 / 12, 2560; TP8 2 / 6, 1280.  Every key head is shared by 3 value heads.

Raw format read by ``sim_host.cpp``: ``case.txt`` (``guid``, then ``in``/``out`` lines with the TPC
dims, fastest first, which is the torch shape reversed) and one header-less little-endian
``<name>.bin`` per input; bf16 is stored as its 16-bit pattern.
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

BF16 = torch.bfloat16
F32 = torch.float32
HEAD_DIM = 128
VALUE_HEADS_PER_KEY_HEAD = 3
# key heads per rank -> (tensor-parallel size, packed q|k|v width)
GEOMETRIES = {16: (1, 10240), 8: (2, 5120), 4: (4, 2560), 2: (8, 1280)}

GUIDS = {
    "compact_qk": "qwen38_post_conv_qk_compact_bf16_gaudi2",
    "expanded_qk": "qwen38_post_conv_qk_expanded_bf16_gaudi2",
    "f32_qk": "qwen38_post_conv_qk_bf16_gaudi2",
    "compact_kkt": "qwen38_compact_kkt_bf16_gaudi2",
    "rmsnorm_gated": "qwen38_rmsnorm_gated_bf16_gaudi2",
    "recurrent": "flashqla_recurrent_prefill_f32_gaudi2",
    "inverse16": "flashqla_unit_lower_inverse16_f32_gaudi2",
    "kkt_form": "flashqla_kkt_form_bf16_gaudi2",
    "pair_transform": "flashqla_pair_transform_bf16_gaudi2",
    "conv_qkv": "qwen38_conv_qkv_prep_bf16_gaudi2",
}
INPUT_ORDER = {
    "compact_qk": ("packed", ),
    "expanded_qk": ("packed", ),
    "f32_qk": ("packed", ),
    "compact_kkt": ("compact_dot", "grouped_beta"),
    "rmsnorm_gated": ("x", "z", "weight"),
    "recurrent": ("q", "k", "v", "decay", "beta", "initial_state"),
    "inverse16": ("lower", ),
    "kkt_form": ("dot", "gate", "beta"),
    "pair_transform": ("a0", "scores", "g_even", "g_odd", "beta"),
    "conv_qkv": ("packed", "initial_state", "weight", "bias"),
}
OUTPUT_ORDER = {
    "compact_qk": ("q", "k"),
    "expanded_qk": ("q", "k"),
    "f32_qk": ("q", "k"),
    "compact_kkt": ("lmat", ),
    "rmsnorm_gated": ("output", ),
    "recurrent": ("output", "final_state"),
    "inverse16": ("inverse", ),
    "kkt_form": ("kkt", ),
    "pair_transform": ("ag", "attention"),
    "conv_qkv": ("q", "k", "v"),
}
# Pass rule per kind: an element passes when |got - want| <= atol + rtol * |want|.
#   compact/expanded/f32 qk  the hardware test's own tolerance (tests/unit_tests/ops/test_qwen38_native_qk.py):
#                            the kernel uses the TPC fast reciprocal square root
#   compact_kkt              exact: one bf16 product per element
#   rmsnorm_gated, conv_qkv  fast rsqrt and the bf16 sigmoid table against torch's exact functions
#   recurrent, inverse16     fp32 with another summation order
#   kkt_form, pair_transform fp32 Cephes exp against torch.exp, then bf16 rounding
TOLERANCE = {
    "compact_qk": (0.002, 0.01),
    "expanded_qk": (0.002, 0.01),
    "f32_qk": (0.002, 0.01),
    "compact_kkt": (0.0, 0.0),
    "rmsnorm_gated": (0.002, 0.02),
    "recurrent": (1e-4, 1e-3),
    "inverse16": (1e-5, 1e-4),
    "kkt_form": (0.002, 0.02),
    "pair_transform": (0.002, 0.02),
    "conv_qkv": (0.004, 0.03),
}
RECURRENT_SCALE = 0.08838834764831845  # 128 ** -0.5, a constant of the kernel


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    kind: str
    key_heads: int = 2  # key heads of the rank (2 = TP8); value heads = 3 * key_heads
    tokens: int = 128
    chunk: int = 64  # rows of one chunk matrix (the kkt kernels take 64)
    outer: int = 2  # sequences x chunks
    scale: float = 1.0  # input magnitude (compact_qk)
    seed: int = 20261120
    outputs_declared: int = 0  # 0 = as the kernel has them (the inverse16 glue took 2 before 2026-11-20)

    @property
    def value_heads(self) -> int:
        return self.key_heads * VALUE_HEADS_PER_KEY_HEAD

    @property
    def packed_width(self) -> int:
        return (2 * self.key_heads + self.value_heads) * HEAD_DIM


def _gen(case: Case) -> torch.Generator:
    return torch.Generator().manual_seed(case.seed)


def make_inputs(case: Case) -> dict[str, torch.Tensor]:
    """The op's input tensors in torch layout, in ``INPUT_ORDER[case.kind]`` order."""
    g = _gen(case)
    kh, vh, t = case.key_heads, case.value_heads, case.tokens
    if case.kind in ("compact_qk", "expanded_qk", "f32_qk"):
        return {"packed": (torch.randn(t, case.packed_width, generator=g) * case.scale).to(BF16)}
    if case.kind == "compact_kkt":
        c = case.chunk
        return {
            "compact_dot": torch.randn(case.outer, kh, c, c, generator=g).to(BF16),
            "grouped_beta": torch.rand(case.outer, kh, VALUE_HEADS_PER_KEY_HEAD, c, generator=g).to(BF16),
        }
    if case.kind == "rmsnorm_gated":
        rows = t * vh
        return {
            "x": (torch.randn(rows, HEAD_DIM, generator=g) * 0.7).to(BF16),
            "z": (torch.randn(rows, HEAD_DIM, generator=g) * 2.0).to(BF16),
            "weight": (1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g)).to(BF16),
        }
    if case.kind == "recurrent":
        q = torch.nn.functional.normalize(torch.randn(t, vh, HEAD_DIM, generator=g), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(t, vh, HEAD_DIM, generator=g), dim=-1)
        return {
            "q": q.contiguous(),
            "k": k.contiguous(),
            "v": torch.randn(t, vh, HEAD_DIM, generator=g) * 0.5,
            "decay": torch.exp(-torch.rand(t, vh, generator=g) * 0.5),
            "beta": torch.rand(t, vh, generator=g),
            "initial_state": torch.randn(vh, HEAD_DIM, HEAD_DIM, generator=g) * 0.01,
        }
    if case.kind == "inverse16":
        return {"lower": torch.randn(case.outer, vh, case.chunk // 16, 16, 16, generator=g) * 0.3}
    if case.kind == "kkt_form":
        matrices = case.outer * vh
        gate = -torch.cumsum(torch.rand(matrices, 64, generator=g) * 0.3, dim=-1)
        return {
            "dot": torch.randn(matrices, 64, 64, generator=g).to(BF16),
            "gate": gate.to(BF16),
            "beta": torch.rand(matrices, 64, generator=g).to(BF16),
        }
    if case.kind == "pair_transform":
        gate = -torch.cumsum(torch.rand(case.outer, vh, 64, generator=g) * 0.3, dim=-1)
        return {
            "a0": torch.randn(case.outer, vh, 64, 64, generator=g).to(BF16),
            "scores": torch.randn(case.outer, vh, 64, 64, generator=g).to(BF16),
            "g_even": gate[..., 0::2].contiguous(),
            "g_odd": gate[..., 1::2].contiguous(),
            "beta": torch.rand(case.outer, vh, 64, generator=g).to(BF16),
        }
    if case.kind == "conv_qkv":
        w = case.packed_width
        return {
            "packed": (torch.randn(t, w, generator=g) * 0.5).to(BF16),
            "initial_state": (torch.randn(3, w, generator=g) * 0.5).to(BF16),
            "weight": (torch.randn(4, w, generator=g) * 0.4).to(BF16),
            "bias": (torch.randn(w, generator=g) * 0.1).to(BF16),
        }
    raise ValueError(f"unknown kind {case.kind}")


def l2norm_heads(packed: torch.Tensor, offset: int, heads: int) -> torch.Tensor:
    """fp32 ``x * rsqrt(sum(x^2) + 1e-6)`` over each 128-wide head of ``packed[:, offset : offset + heads * 128]``:
    ``[tokens, heads, 128]`` fp32 (``_l2norm_last_dim`` of hpu_gdn_pytorch.py on the head's fp32 values)."""
    x = packed[:, offset:offset + heads * HEAD_DIM].reshape(packed.shape[0], heads, HEAD_DIM).float()
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def compact_kkt_reference(compact_dot: torch.Tensor, grouped_beta: torch.Tensor) -> torch.Tensor:
    """``I + strict_lower(dot[o, h, r, c] * beta[o, h, rep, r])`` in the inputs' dtype:
    ``[outer, heads, repeats, C, C]`` (the compact branch of ``hpu_flashqla_chunk_gdr_phase_a``)."""
    c = compact_dot.shape[-1]
    product = compact_dot.unsqueeze(2) * grouped_beta.unsqueeze(-1)
    return torch.tril(product, diagonal=-1) + torch.eye(c, dtype=compact_dot.dtype)


def rmsnorm_gated_reference(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, gate: str = "silu") -> torch.Tensor:
    """fp32 ``x * rsqrt(mean(x^2) + 1e-6) * weight * gate(z)`` rounded to bf16 (``gate``: silu or sigmoid)."""
    xf, zf = x.float(), z.float()
    normed = xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6) * weight.float()
    activation = torch.nn.functional.silu(zf) if gate == "silu" else torch.sigmoid(zf)
    return (normed * activation).to(BF16)


def recurrent_reference(q, k, v, decay, beta, initial_state):
    """Token-by-token gated delta rule in fp64, returned in fp32: ``output [T, H, V]``, ``final_state [H, V, K]``.

    Per token: ``S *= decay``; ``v_new = (v - S k) * beta``; ``S += v_new k^T``; ``out = S q * 128^-0.5``.
    """
    state = initial_state.double().clone()
    outputs = []
    for t in range(q.shape[0]):
        state = state * decay[t].double()[:, None, None]
        projection = torch.einsum("hvk,hk->hv", state, k[t].double())
        v_new = (v[t].double() - projection) * beta[t].double()[:, None]
        state = state + v_new[:, :, None] * k[t].double()[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q[t].double()) * RECURRENT_SCALE)
    return torch.stack(outputs).float(), state.float()


def inverse16_reference(lower: torch.Tensor) -> torch.Tensor:
    """Inverse of ``I + strict_lower(lower)`` per 16x16 block (the diagonal and the upper part are ignored)."""
    eye = torch.eye(16, dtype=torch.float64)
    return torch.linalg.inv(eye + torch.tril(lower.double(), diagonal=-1)).float()


def kkt_form_reference(dot: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``I + strict_lower(dot[m, r, c] * bf16(bf16(exp(g_r - g_c)) * beta_r))`` in bf16."""
    gf = gate.float()
    decay = torch.exp(gf.unsqueeze(-1) - gf.unsqueeze(-2)).to(BF16)
    coefficient = decay * beta.unsqueeze(-1)
    return torch.tril(dot * coefficient, diagonal=-1) + torch.eye(dot.shape[-1], dtype=BF16)


def pair_transform_reference(a0, scores, g_even, g_odd, beta):
    """``ag = lower(a0 * exp(g_r - g_c) * beta_c)`` and ``attention = lower(scores * exp(g_r - g_c))``, fp32 -> bf16."""
    gate = torch.stack((g_even, g_odd), dim=-1).flatten(-2)
    decay = torch.exp(torch.tril(gate.unsqueeze(-1) - gate.unsqueeze(-2)))
    ag = torch.tril(a0.float() * decay * beta.float().unsqueeze(-2))
    attention = torch.tril(scores.float() * decay)
    return ag.to(BF16), attention.to(BF16)


def conv_qkv_reference(packed, initial_state, weight, bias, key_heads: int):
    """Width-4 causal conv with a 3-token history, SiLU, then the q/k normalization: fp32 -> bf16.

    Returns ``q, k [T, 3 * key_heads, 128]`` (each key head three times) and ``v [T, value_heads, 128]``.
    """
    tokens = packed.shape[0]
    history = torch.cat((initial_state.float(), packed.float()), dim=0)
    acc = sum(history[tap:tap + tokens] * weight[tap].float() for tap in range(4)) + bias.float()
    conv = torch.nn.functional.silu(acc).to(BF16)
    q = l2norm_heads(conv, 0, key_heads).to(BF16).repeat_interleave(VALUE_HEADS_PER_KEY_HEAD, dim=1)
    k = l2norm_heads(conv, key_heads * HEAD_DIM, key_heads).to(BF16).repeat_interleave(VALUE_HEADS_PER_KEY_HEAD,
                                                                                    dim=1)
    v = conv[:, 2 * key_heads * HEAD_DIM:].reshape(tokens, key_heads * VALUE_HEADS_PER_KEY_HEAD, HEAD_DIM)
    return q, k, v.contiguous()


def expected(case: Case, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The reference outputs, by output name, in torch layout."""
    kh = case.key_heads
    if case.kind in ("compact_qk", "expanded_qk", "f32_qk"):
        q = l2norm_heads(inputs["packed"], 0, kh)
        k = l2norm_heads(inputs["packed"], kh * HEAD_DIM, kh)
        if case.kind == "compact_qk":
            return {"q": q.to(BF16), "k": k.to(BF16)}
        q = q.repeat_interleave(VALUE_HEADS_PER_KEY_HEAD, dim=1)
        k = k.repeat_interleave(VALUE_HEADS_PER_KEY_HEAD, dim=1)
        dtype = BF16 if case.kind == "expanded_qk" else F32
        return {"q": q.to(dtype), "k": k.to(dtype)}
    if case.kind == "compact_kkt":
        return {"lmat": compact_kkt_reference(inputs["compact_dot"], inputs["grouped_beta"])}
    if case.kind == "rmsnorm_gated":
        return {"output": rmsnorm_gated_reference(inputs["x"], inputs["z"], inputs["weight"])}
    if case.kind == "recurrent":
        output, final_state = recurrent_reference(**inputs)
        return {"output": output, "final_state": final_state}
    if case.kind == "inverse16":
        return {"inverse": inverse16_reference(inputs["lower"])}
    if case.kind == "kkt_form":
        return {"kkt": kkt_form_reference(**inputs)}
    if case.kind == "pair_transform":
        ag, attention = pair_transform_reference(**inputs)
        return {"ag": ag, "attention": attention}
    if case.kind == "conv_qkv":
        q, k, v = conv_qkv_reference(key_heads=kh, **inputs)
        return {"q": q, "k": k, "v": v}
    raise ValueError(f"unknown kind {case.kind}")


def _dtype_name(tensor: torch.Tensor) -> str:
    return {BF16: "bf16", F32: "f32", torch.int32: "i32"}[tensor.dtype]


def _raw(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous()
    if tensor.dtype == BF16:
        return tensor.view(torch.int16).numpy().tobytes()
    return tensor.numpy().tobytes()


def write_inputs(case: Case, inputs: dict[str, torch.Tensor], folder: pathlib.Path) -> None:
    """Write ``case.txt`` and one ``<name>.bin`` per input into ``folder`` (created)."""
    folder.mkdir(parents=True, exist_ok=True)
    want = expected(case, inputs)
    lines = [f"guid {GUIDS[case.kind]}"]
    for name in INPUT_ORDER[case.kind]:
        tensor = inputs[name]
        dims = " ".join(str(d) for d in reversed(tensor.shape))
        lines.append(f"in {name} {_dtype_name(tensor)} {dims}")
        (folder / f"{name}.bin").write_bytes(_raw(tensor))
    names = list(OUTPUT_ORDER[case.kind])
    for name in names:
        tensor = want[name]
        dims = " ".join(str(d) for d in reversed(tensor.shape))
        lines.append(f"out {name} {_dtype_name(tensor)} {dims}")
    for extra in range(len(names), case.outputs_declared):
        tensor = want[names[0]]
        dims = " ".join(str(d) for d in reversed(tensor.shape))
        lines.append(f"out spare{extra} {_dtype_name(tensor)} {dims}")
    (folder / "case.txt").write_text("\n".join(lines) + "\n")


def read_outputs(case: Case, inputs: dict[str, torch.Tensor], folder: pathlib.Path) -> dict[str, torch.Tensor]:
    """The kernel's outputs from ``<name>.out.bin``, shaped and typed like the reference."""
    want = expected(case, inputs)
    got = {}
    for name in OUTPUT_ORDER[case.kind]:
        reference = want[name]
        data = bytearray((folder / f"{name}.out.bin").read_bytes())
        if reference.dtype == BF16:
            tensor = torch.frombuffer(data, dtype=torch.int16).view(BF16)
        else:
            tensor = torch.frombuffer(data, dtype=reference.dtype)
        got[name] = tensor.reshape(reference.shape).clone()
    return got


def compare(case: Case, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> list[str]:
    """Failure texts ([] when every output matches the reference within ``TOLERANCE[case.kind]``)."""
    atol, rtol = TOLERANCE[case.kind]
    failures = []
    for name, want in expected(case, inputs).items():
        actual = got[name]
        if actual.shape != want.shape or actual.dtype != want.dtype:
            failures.append(f"{name}: got {tuple(actual.shape)} {actual.dtype}, want {tuple(want.shape)} {want.dtype}")
            continue
        a, w = actual.double(), want.double()
        bad = ~((a - w).abs() <= atol + rtol * w.abs())
        bad |= ~torch.isfinite(a)
        if bool(bad.any()):
            index = tuple(int(i) for i in torch.nonzero(bad)[0])
            failures.append(f"{name}: {int(bad.sum())} of {bad.numel()} elements differ (atol {atol}, rtol {rtol}); "
                            f"first at {index}: got {float(a[index])!r}, want {float(w[index])!r}")
    return failures


def summary(case: Case, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> str:
    """One line per output: the largest absolute and relative difference from the reference."""
    lines = []
    for name, want in expected(case, inputs).items():
        a, w = got[name].double(), want.double()
        diff = (a - w).abs()
        relative = (diff / w.abs().clamp_min(1e-30))[w.abs() > 1e-3]
        lines.append(f"{case.name} {name}: max |diff| {float(diff.max()):.3e}, max relative diff "
                     f"{float(relative.max()) if relative.numel() else 0.0:.3e}, exact {int((diff == 0).sum())}"
                     f"/{diff.numel()}")
    return "\n".join(lines)


# Simulator cases.  TP8 (2 key / 6 value heads) at one chunk (128 tokens) and at a 2048-token step for the
# kernels that take token rows; the other three compact layouts at one chunk.
SIM_CASES = (
    Case("compact_qk_tp8_128", "compact_qk", key_heads=2, tokens=128),
    Case("compact_qk_tp8_2048", "compact_qk", key_heads=2, tokens=2048),
    Case("compact_qk_tp8_5_tokens", "compact_qk", key_heads=2, tokens=5),
    Case("compact_qk_tp8_zeros", "compact_qk", key_heads=2, tokens=128, scale=0.0),
    Case("compact_qk_tp8_tiny_values", "compact_qk", key_heads=2, tokens=128, scale=1e-8),
    Case("compact_qk_tp4_128", "compact_qk", key_heads=4, tokens=128),
    Case("compact_qk_tp4_2048", "compact_qk", key_heads=4, tokens=2048),
    Case("compact_qk_tp2_128", "compact_qk", key_heads=8, tokens=128),
    Case("compact_qk_tp1_128", "compact_qk", key_heads=16, tokens=128),
    Case("compact_kkt_tp8_one_chunk_pair", "compact_kkt", key_heads=2, outer=2),
    Case("compact_kkt_tp8_2048_tokens", "compact_kkt", key_heads=2, outer=32),
    Case("compact_kkt_tp4_one_chunk_pair", "compact_kkt", key_heads=4, outer=2),
    Case("rmsnorm_gated_tp8_128", "rmsnorm_gated", key_heads=2, tokens=128),
    Case("rmsnorm_gated_tp8_2048", "rmsnorm_gated", key_heads=2, tokens=2048),
    Case("recurrent_tp8_128", "recurrent", key_heads=2, tokens=128),
    Case("inverse16_tp8", "inverse16", key_heads=2, outer=2, chunk=128),
    Case("kkt_form_tp8", "kkt_form", key_heads=2, outer=2),
    Case("pair_transform_tp8", "pair_transform", key_heads=2, outer=2),
    Case("expanded_qk_tp1_128", "expanded_qk", key_heads=16, tokens=128),
    Case("f32_qk_tp1_128", "f32_qk", key_heads=16, tokens=128),
    Case("conv_qkv_tp1_64", "conv_qkv", key_heads=16, tokens=64),
)
# Shapes the glue must refuse (GLUE_UNSUPPORTED_LAYER_CONFIGURATION): a packed width that is no rank's.
REFUSED_CASES = (
    Case("compact_qk_width_of_no_rank", "compact_qk", key_heads=3, tokens=8),
    Case("expanded_qk_tp8", "expanded_qk", key_heads=2, tokens=8),
    Case("f32_qk_tp8", "f32_qk", key_heads=2, tokens=8),
    Case("conv_qkv_tp8", "conv_qkv", key_heads=2, tokens=8),
    Case("compact_kkt_chunk_128", "compact_kkt", key_heads=2, outer=2, chunk=128),
)
