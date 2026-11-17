# SPDX-License-Identifier: Apache-2.0
"""Shared kernel test cases for the fused GDN core decode step.

One module defines the cases, their inputs, the expected outputs, the pass/fail comparison and
the raw-file I/O.  The TPC simulator test (K5b), the HPU op test and the replay tool (K6) all
import these names.  The raw format (one header-less little-endian file per tensor, bf16 stored
as its 16-bit pattern, fp32 as float32, int32 as int32) is what the C++ simulator host program
reads and writes, so it must not change.

Contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

from flashinfer_gaudi.gdn_core_decode import GdnGateActivation, gdn_core_decode_reference

D = 128
ARG_ORDER = ("conv_state", "ssm_state", "mixed_qkv", "z", "a", "b", "A_log", "dt_bias", "conv_weight_t", "norm_weight",
             "load_indices", "store_indices")
OUTPUT_FILES = {"out": "out.bin", "conv_state": "conv_state.out.bin", "ssm_state": "ssm_state.out.bin"}


@dataclasses.dataclass(frozen=True)
class SimCase:
    name: str
    H: int  # key heads
    HV: int  # value heads
    S: int  # state slots
    load: tuple[int, ...]  # one per row; B = len(load)
    store: tuple[int, ...]
    gate: GdnGateActivation
    seed: int


CASES = (  # simulator + HPU
    SimCase("tp8_mixed", 2, 6, 4, (0, 1, -1), (0, 2, -1), GdnGateActivation.SIGMOID, 0),
    SimCase("tp4_silu", 4, 12, 3, (2, 0), (1, 0), GdnGateActivation.SILU, 1),
    SimCase("tp1_move", 16, 48, 2, (1, ), (0, ), GdnGateActivation.SIGMOID, 2),
)
HPU_CASES = CASES + (  # HPU only (too slow for the simulator)
    SimCase("tp8_b16", 2, 6, 17, tuple(range(16)), tuple(range(16)), GdnGateActivation.SIGMOID, 3),
    SimCase("tp8_b64", 2, 6, 65, tuple(range(1, 65)), tuple(range(1, 65)), GdnGateActivation.SIGMOID, 4),
)
OUT_ATOL, OUT_RTOL = 1e-3, 1e-3  # TPC exp/log/sigmoid/rsqrt approximations against torch fp32
SSM_ATOL, SSM_RTOL = 1e-5, 1e-4


def make_inputs(case: SimCase) -> dict[str, torch.Tensor]:
    """CPU tensors keyed by ``ARG_ORDER``, in the kernel dtypes; deterministic from ``case.seed``."""
    g = torch.Generator().manual_seed(case.seed)
    H, HV, S = case.H, case.HV, case.S
    C = 2 * H * D + HV * D
    B = len(case.load)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return {
        "conv_state": (randn(S, 3, C) * 0.5).to(torch.bfloat16),
        "ssm_state": (randn(S, HV, D, D) * 0.1).float(),
        "mixed_qkv": randn(B, C).to(torch.bfloat16),
        "z": randn(B, HV * D).to(torch.bfloat16),
        "a": randn(B, HV).to(torch.bfloat16),
        "b": randn(B, HV).to(torch.bfloat16),
        "A_log": torch.log(torch.rand(HV, generator=g) * 15 + 1).float(),
        "dt_bias": (randn(HV) * 0.5).float(),
        "conv_weight_t": (randn(4, C) * 0.3).to(torch.bfloat16),
        "norm_weight": (1 + 0.1 * randn(D)).to(torch.bfloat16),
        "load_indices": torch.tensor(case.load, dtype=torch.int32),
        "store_indices": torch.tensor(case.store, dtype=torch.int32),
    }


def scalars(case: SimCase) -> tuple[float, float, int]:
    """The op's three scalar parameters: (scale, eps, gate_activation)."""
    return D**-0.5, 1e-6, int(case.gate)


def expected(case: SimCase, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reference run on clones: {"out", "conv_state", "ssm_state"} (the two mutated state clones)."""
    conv = inputs["conv_state"].clone()
    ssm = inputs["ssm_state"].clone()
    args = [conv, ssm] + [inputs[k] for k in ARG_ORDER[2:]]
    out = gdn_core_decode_reference(*args, *scalars(case))
    return {"out": out, "conv_state": conv, "ssm_state": ssm}


def _valid_rows(case: SimCase) -> list[bool]:
    return [0 <= ls < case.S and 0 <= ss < case.S for ls, ss in zip(case.load, case.store)]


def compare(case: SimCase, inputs: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> list[str]:
    """Failure messages for the tensors in ``got`` (same keys as ``expected``); [] means pass."""
    S, B = case.S, len(case.load)
    valid = _valid_rows(case)
    exp = expected(case, inputs)
    out = got["out"]
    if tuple(out.shape) != (B, case.HV, D) or out.dtype != torch.float32:
        return [f"{case.name}: out must be [{B}, {case.HV}, {D}] fp32, got shape {tuple(out.shape)} dtype {out.dtype}"]
    msgs = []
    for r in range(B):
        if valid[r]:
            if not torch.allclose(out[r], exp["out"][r], atol=OUT_ATOL, rtol=OUT_RTOL):
                e = (out[r] - exp["out"][r]).abs().max().item()
                msgs.append(f"{case.name}: out row {r}: max abs err {e:.3e}")
        elif torch.count_nonzero(out[r]) != 0:
            msgs.append(f"{case.name}: padded row {r}: output not zero")
    conv, exp_conv = got["conv_state"], exp["conv_state"]
    if not torch.equal(conv, exp_conv):
        if conv.shape == exp_conv.shape and conv.dtype == exp_conv.dtype:
            n = int(torch.count_nonzero(conv != exp_conv).item())
        else:
            n = int(exp_conv.numel())
        msgs.append(f"{case.name}: conv_state: {n} elements differ")
    ssm, exp_ssm = got["ssm_state"], exp["ssm_state"]
    written = {ss for (ls, ss), v in zip(zip(case.load, case.store), valid) if v}
    for s in sorted(written):
        if not torch.allclose(ssm[s], exp_ssm[s], atol=SSM_ATOL, rtol=SSM_RTOL):
            e = (ssm[s] - exp_ssm[s]).abs().max().item()
            msgs.append(f"{case.name}: ssm_state slot {s}: max abs err {e:.3e}")
    for s in range(S):
        if s in written:
            continue
        if not torch.equal(ssm[s], inputs["ssm_state"][s]):
            msgs.append(f"{case.name}: ssm_state untouched slot {s} changed")
    return msgs


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Little-endian header-less bytes: bf16 as its 16-bit pattern, fp32/int32 as themselves."""
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def _read_raw(path: pathlib.Path, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    data = path.read_bytes()
    numel = 1
    for dim in shape:
        numel *= dim
    want = numel * dtype.itemsize
    if len(data) != want:
        raise ValueError(f"{path.name}: file size {len(data)} bytes, expected {want} bytes")
    if dtype == torch.bfloat16:
        tensor = torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(data), dtype=dtype)
    return tensor.reshape(shape)


def write_inputs(case: SimCase, inputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """One raw file ``f"{name}.bin"`` per ``ARG_ORDER`` tensor, plus ``params.txt``."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARG_ORDER:
        (directory / f"{name}.bin").write_bytes(_raw_bytes(inputs[name]))
    scale, eps, gate = scalars(case)
    B = len(case.load)
    (directory / "params.txt").write_text(f"{case.S} {case.H} {case.HV} {B} {scale!r} {eps!r} {gate}\n")


def write_outputs(outputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """Write ``OUTPUT_FILES`` (the format the simulator host program writes)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, filename in OUTPUT_FILES.items():
        (directory / filename).write_bytes(_raw_bytes(outputs[key]))


def read_outputs(case: SimCase, directory: pathlib.Path) -> dict[str, torch.Tensor]:
    """Read ``OUTPUT_FILES`` back; ``ValueError`` naming the file on a size mismatch."""
    directory = pathlib.Path(directory)
    C = 2 * case.H * D + case.HV * D
    B = len(case.load)
    return {
        "out": _read_raw(directory / OUTPUT_FILES["out"], torch.float32, (B, case.HV, D)),
        "conv_state": _read_raw(directory / OUTPUT_FILES["conv_state"], torch.bfloat16, (case.S, 3, C)),
        "ssm_state": _read_raw(directory / OUTPUT_FILES["ssm_state"], torch.float32, (case.S, case.HV, D, D)),
    }
