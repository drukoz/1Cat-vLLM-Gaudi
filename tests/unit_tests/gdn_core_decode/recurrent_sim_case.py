# SPDX-License-Identifier: Apache-2.0
"""Shared kernel test cases for the recurrence-only GDN decode op (``gdn_recurrent_decode``).

The recurrence-only counterpart of ``sim_case.py``: cases, inputs, expected outputs, the pass/fail
comparison and the raw-file I/O for ``flashinfer_gaudi::gdn_recurrent_decode``.  The TPC simulator
test and the HPU op test import these names.  The raw format is ``sim_case.py``'s (one header-less
little-endian file per tensor; bf16 as its 16-bit pattern), which the C++ simulator host program
reads and writes.  ``params.txt`` holds ``S H HV B scale``.

Contract: ``docs/design/flash-next/gdn-core-decode.md``, "Recurrence-only op".
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

from flashinfer_gaudi.gdn_core_decode import gdn_recurrent_decode_reference
from tests.unit_tests.gdn_core_decode.sim_case import D, OUT_ATOL, OUT_RTOL, SSM_ATOL, SSM_RTOL, _raw_bytes, _read_raw

RECURRENT_ARG_ORDER = ("ssm_state", "qkv", "a", "b", "A_log", "dt_bias", "load_indices", "store_indices")
RECURRENT_OUTPUT_FILES = {"out": "out.bin", "ssm_state": "ssm_state.out.bin"}


@dataclasses.dataclass(frozen=True)
class RecurrentSimCase:
    name: str
    H: int  # key heads
    HV: int  # value heads
    S: int  # state slots
    load: tuple[int, ...]  # one per row; B = len(load)
    store: tuple[int, ...]
    seed: int


# Every simulator case has a row with load == store (in place) and a row that moves or is padding,
# because tpc-clang may reorder loads and stores of one tensor (docs/design/flash-next/tpcsim.md).
RECURRENT_CASES = (  # simulator + HPU
    RecurrentSimCase("rec_tp8_mixed", 2, 6, 4, (0, 1, -1), (0, 2, -1), 10),
    RecurrentSimCase("rec_tp4_move", 4, 12, 3, (2, 0), (1, 0), 11),
    RecurrentSimCase("rec_tp1_move", 16, 48, 3, (1, 2), (0, 2), 12),
)
RECURRENT_HPU_CASES = RECURRENT_CASES + (  # HPU only (too slow for the simulator)
    RecurrentSimCase("rec_tp8_b16", 2, 6, 17, tuple(range(16)), tuple(range(16)), 13),
    RecurrentSimCase("rec_tp8_b64", 2, 6, 65, tuple(range(1, 65)), tuple(range(1, 65)), 14),
)


def make_recurrent_inputs(case: RecurrentSimCase) -> dict[str, torch.Tensor]:
    """CPU tensors keyed by ``RECURRENT_ARG_ORDER``, in the kernel dtypes; deterministic from ``case.seed``."""
    g = torch.Generator().manual_seed(case.seed)
    H, HV, S = case.H, case.HV, case.S
    C = 2 * H * D + HV * D
    B = len(case.load)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return {
        "ssm_state": (randn(S, HV, D, D) * 0.1).float(),
        "qkv": randn(B, C).to(torch.bfloat16),
        "a": randn(B, HV).to(torch.bfloat16),
        "b": randn(B, HV).to(torch.bfloat16),
        "A_log": torch.log(torch.rand(HV, generator=g) * 15 + 1).float(),
        "dt_bias": (randn(HV) * 0.5).float(),
        "load_indices": torch.tensor(case.load, dtype=torch.int32),
        "store_indices": torch.tensor(case.store, dtype=torch.int32),
    }


def recurrent_scalars(case: RecurrentSimCase) -> tuple[float]:
    """The op's one scalar parameter: (scale,)."""
    return (D**-0.5, )


def expected_recurrent(case: RecurrentSimCase, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reference run on a clone of the state: {"out", "ssm_state"} (the mutated clone)."""
    ssm = inputs["ssm_state"].clone()
    out = gdn_recurrent_decode_reference(ssm, *[inputs[k] for k in RECURRENT_ARG_ORDER[1:]], *recurrent_scalars(case))
    return {"out": out, "ssm_state": ssm}


def compare_recurrent(case: RecurrentSimCase, inputs: dict[str, torch.Tensor],
                      got: dict[str, torch.Tensor]) -> list[str]:
    """Failure messages for ``got`` (keys "out" and "ssm_state"); [] means pass."""
    S, B = case.S, len(case.load)
    valid = [0 <= ls < S and 0 <= ss < S for ls, ss in zip(case.load, case.store)]
    exp = expected_recurrent(case, inputs)
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
    ssm, exp_ssm = got["ssm_state"], exp["ssm_state"]
    written = {ss for ss, v in zip(case.store, valid) if v}
    for s in sorted(written):
        if not torch.allclose(ssm[s], exp_ssm[s], atol=SSM_ATOL, rtol=SSM_RTOL):
            e = (ssm[s] - exp_ssm[s]).abs().max().item()
            msgs.append(f"{case.name}: ssm_state slot {s}: max abs err {e:.3e}")
    for s in range(S):
        if s not in written and not torch.equal(ssm[s], inputs["ssm_state"][s]):
            msgs.append(f"{case.name}: ssm_state untouched slot {s} changed")
    return msgs


def write_recurrent_inputs(case: RecurrentSimCase, inputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """One raw file ``f"{name}.bin"`` per ``RECURRENT_ARG_ORDER`` tensor, plus ``params.txt`` = ``S H HV B scale``."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in RECURRENT_ARG_ORDER:
        (directory / f"{name}.bin").write_bytes(_raw_bytes(inputs[name]))
    (scale, ) = recurrent_scalars(case)
    (directory / "params.txt").write_text(f"{case.S} {case.H} {case.HV} {len(case.load)} {scale!r}\n")


def write_recurrent_outputs(outputs: dict[str, torch.Tensor], directory: pathlib.Path) -> None:
    """Write ``RECURRENT_OUTPUT_FILES`` (the format the simulator host program writes)."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, filename in RECURRENT_OUTPUT_FILES.items():
        (directory / filename).write_bytes(_raw_bytes(outputs[key]))


def read_recurrent_outputs(case: RecurrentSimCase, directory: pathlib.Path) -> dict[str, torch.Tensor]:
    """Read ``RECURRENT_OUTPUT_FILES`` back; ``ValueError`` naming the file on a size mismatch."""
    directory = pathlib.Path(directory)
    B = len(case.load)
    return {
        "out": _read_raw(directory / RECURRENT_OUTPUT_FILES["out"], torch.float32, (B, case.HV, D)),
        "ssm_state": _read_raw(directory / RECURRENT_OUTPUT_FILES["ssm_state"], torch.float32,
                               (case.S, case.HV, D, D)),
    }
