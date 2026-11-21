# SPDX-License-Identifier: Apache-2.0
"""Replay N decode steps of each GDN op in the bridge mode of this process: one process runs one mode.

CPU (``--device cpu``) and eager on a card (``PT_HPU_LAZY_MODE=0``) run eager and
``torch.compile`` (``aot_eager`` on CPU, ``hpu_backend`` on the card) and report the HPU-graph
variant as ``skipped (cpu)`` / ``skipped (eager mode)``; lazy on a card (``PT_HPU_LAZY_MODE=1``)
runs eager and the HPU graph and reports ``torch.compile`` and the recompile count as
``skipped (lazy mode)``: the HPU graph exists only in lazy mode and ``hpu_backend`` only in
eager mode, so one process can run only one of the two.  The states are carried from step to
step (the ops mutate them in place), so an exact match with eager over every step output and the
final states rules out aliasing, stale graph buffers and dropped state writes on replay.  On a
card run the tool twice: ``PT_HPU_LAZY_MODE=1`` for the HPU graph and ``PT_HPU_LAZY_MODE=0`` for
``torch.compile`` (run sheet row 1b-3, ``docs/design/flash-next/run-sheets/qualification.md``,
``--steps 1000``).

Usage: ``PYTHONPATH=$PWD python tools/flash_next/gdn_core_decode_replay.py [--steps 1000]
[--device auto|cpu|hpu] [--case tp8_b16|ver_tp8_b16] [--ops core,recurrent|verify]``; ``--ops
verify`` runs alone on a ``ver_`` case (default ``ver_tp8_b16``, others default to ``tp8_b16``).
Exit code 0 only when every variant that ran is exact against eager, every measured recompile
count after warmup is 0, and at least one variant (HPU graph or ``torch.compile``) ran for every
op; argument errors exit 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import sys

import torch
import torch.nn as nn

from flashinfer_gaudi.gdn_core_decode import gdn_core_decode, gdn_recurrent_decode
from flashinfer_gaudi.gdn_core_verify import gdn_core_verify
from tests.unit_tests.gdn_core_decode.sim_case import HPU_CASES, make_inputs, scalars
from tests.unit_tests.gdn_core_decode.verify_sim_case import VERIFY_HPU_CASES, make_verify_inputs, verify_scalars

CASES_BY_NAME = {case.name: case for case in HPU_CASES}
CASES_BY_NAME.update({case.name: case for case in VERIFY_HPU_CASES})
OPS = ("core", "recurrent", "verify")
DEFAULT_CASE = "tp8_b16"
DEFAULT_VERIFY_CASE = "ver_tp8_b16"
WARMUP_STEPS = 3  # steps after which the recompile count baseline is read


class BridgeMode(enum.IntEnum):
    """The bridge mode of the running process: one process runs exactly one of these."""
    CPU = 0    # --device cpu: no HPU graph; torch.compile runs with aot_eager
    EAGER = 1  # PT_HPU_LAZY_MODE=0 on a card: no HPU graph; torch.compile runs with hpu_backend
    LAZY = 2   # PT_HPU_LAZY_MODE=1 on a card: the HPU graph runs; no torch.compile


BRIDGE_NAMES = {BridgeMode.CPU: "cpu", BridgeMode.EAGER: "eager", BridgeMode.LAZY: "lazy"}


def bridge_mode(device: torch.device) -> BridgeMode:
    """CPU for a CPU device; on HPU, LAZY or EAGER from the bridge itself (not from the environment)."""
    if device.type != "hpu":
        return BridgeMode.CPU
    import habana_frameworks.torch as htorch
    return BridgeMode.LAZY if htorch.utils.internal.is_lazy() else BridgeMode.EAGER


def graph_skip_reason(mode: BridgeMode) -> str | None:
    """None when the HPU-graph variant runs in this mode, else the text after 'skipped '."""
    if mode == BridgeMode.LAZY:
        return None
    return "(cpu)" if mode == BridgeMode.CPU else "(eager mode)"


def compile_skip_reason(mode: BridgeMode) -> str | None:
    """None when the torch.compile variant runs in this mode, else the text after 'skipped '."""
    return "(lazy mode)" if mode == BridgeMode.LAZY else None


def wrap_in_hpu_graph(module: nn.Module) -> nn.Module:
    import habana_frameworks.torch as htorch
    return htorch.hpu.wrap_in_hpu_graph(module)


@dataclasses.dataclass
class RunResult:
    """One replay: the per-step outputs and the final states, all moved to the CPU."""
    outputs: list[torch.Tensor]
    states: dict[str, torch.Tensor]


class CoreStep(nn.Module):
    """One fused ``gdn_core_decode`` step; constant inputs come from the constructor."""

    def __init__(self, A_log, dt_bias, conv_weight_t, norm_weight, load_indices, store_indices, scale, eps,
                 gate_activation):
        super().__init__()
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.conv_weight_t = conv_weight_t
        self.norm_weight = norm_weight
        self.load_indices = load_indices
        self.store_indices = store_indices
        self.scale = float(scale)
        self.eps = float(eps)
        self.gate_activation = int(gate_activation)

    def forward(self, conv_state, ssm_state, mixed_qkv, z, a, b):
        return gdn_core_decode(conv_state, ssm_state, mixed_qkv, z, a, b, self.A_log, self.dt_bias,
                               self.conv_weight_t, self.norm_weight, self.load_indices, self.store_indices, self.scale,
                               self.eps, self.gate_activation)


class RecurrentStep(nn.Module):
    """One ``gdn_recurrent_decode`` step; constant inputs come from the constructor."""

    def __init__(self, A_log, dt_bias, load_indices, store_indices, scale):
        super().__init__()
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.load_indices = load_indices
        self.store_indices = store_indices
        self.scale = float(scale)

    def forward(self, ssm_state, qkv, a, b):
        return gdn_recurrent_decode(ssm_state, qkv, a, b, self.A_log, self.dt_bias, self.load_indices,
                                    self.store_indices, self.scale)


class VerifyStep(nn.Module):
    """One fused ``gdn_core_verify`` step; constant inputs come from the constructor."""

    def __init__(self, A_log, dt_bias, conv_weight_t, norm_weight, row_indices, ssm_store_indices, scale, eps,
                 gate_activation):
        super().__init__()
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.conv_weight_t = conv_weight_t
        self.norm_weight = norm_weight
        self.row_indices = row_indices
        self.ssm_store_indices = ssm_store_indices
        self.scale = float(scale)
        self.eps = float(eps)
        self.gate_activation = int(gate_activation)

    def forward(self, conv_state, ssm_state, mixed_qkv, z, a, b):
        return gdn_core_verify(conv_state, ssm_state, mixed_qkv, z, a, b, self.A_log, self.dt_bias,
                               self.conv_weight_t, self.norm_weight, self.row_indices, self.ssm_store_indices,
                               self.scale, self.eps, self.gate_activation)


def hpu_present() -> bool:
    """True when ``habana_frameworks.torch`` imports and a Gaudi device is visible."""
    try:
        import habana_frameworks.torch  # noqa: F401
    except Exception:
        return False
    try:
        # Importing habana_frameworks.torch registers torch.hpu (Gaudi is not a CUDA device).
        return bool(hasattr(torch, "hpu") and torch.hpu.is_available())
    except Exception:
        return False


def draw_activations(base: dict[str, torch.Tensor], steps: int) -> list[dict[str, torch.Tensor]]:
    """Per-step activations on the CPU, seed ``1000 + step``, same shapes and dtypes as ``base``.

    Drawn once up front so every variant sees the same inputs; ``qkv`` is ``mixed_qkv`` through
    SiLU in fp32, back to bf16 (what the conv stage feeds the recurrence op).
    """
    activations = []
    for step in range(steps):
        g = torch.Generator().manual_seed(1000 + step)
        drawn = {
            name: torch.randn(base[name].shape, generator=g).to(base[name].dtype)
            for name in ("mixed_qkv", "z", "a", "b")
        }
        drawn["qkv"] = torch.nn.functional.silu(drawn["mixed_qkv"].float()).to(torch.bfloat16)
        activations.append(drawn)
    return activations


def run(step_fn,
        states: dict[str, torch.Tensor],
        activations: list[tuple[torch.Tensor, ...]],
        device: torch.device,
        progress=None) -> RunResult:
    """Carry cloned starting states through every step; collect each step output and the finals, on CPU.

    ``step_fn`` is the module (or its compiled / graph-wrapped form); the op mutates the carried
    states in place.  ``mark_step()`` runs after every step on HPU (lazy mode runs nothing until
    it is called).  ``progress(i)`` is called with the 1-based step number after each step.
    """
    hpu = device.type == "hpu"
    if hpu:
        import habana_frameworks.torch as htorch
    carried = {name: tensor.clone() for name, tensor in states.items()}
    outputs = []
    for i, act in enumerate(activations, start=1):
        out = step_fn(*[carried[name] for name in states], *act)
        if hpu:
            htorch.core.mark_step()
        outputs.append(out.cpu())
        if progress is not None:
            progress(i)
    return RunResult(outputs, {name: carried[name].cpu() for name in carried})


def run_compiled(step_fn, states: dict[str, torch.Tensor], activations: list[tuple[torch.Tensor, ...]],
                 device: torch.device) -> tuple[RunResult, int]:
    """Run through ``torch.compile`` and count the recompiles after warmup (steps 1-3).

    ``aot_eager`` stands in for ``hpu_backend`` on CPU.  The count is the growth of
    ``counters["stats"]["unique_graphs"]`` between after step 3 and the last step.
    """
    from torch._dynamo.utils import counters

    backend = "hpu_backend" if device.type == "hpu" else "aot_eager"
    compiled = torch.compile(step_fn, backend=backend, dynamic=False)
    counters.clear()
    warm_after = min(WARMUP_STEPS, len(activations))
    baseline: list[int] = []

    def progress(i: int) -> None:
        if i == warm_after:
            baseline.append(counters["stats"]["unique_graphs"])

    result = run(compiled, states, activations, device, progress=progress)
    recompiles = counters["stats"]["unique_graphs"] - (baseline[0] if baseline else 0)
    return result, recompiles


def compare_runs(eager: RunResult, other: RunResult) -> str | None:
    """``None`` when every step output and every final state is ``torch.equal``; else the first mismatch."""
    if len(other.outputs) != len(eager.outputs):
        return f"MISMATCH: {len(other.outputs)} steps, eager has {len(eager.outputs)}"
    for i, (a, b) in enumerate(zip(eager.outputs, other.outputs), start=1):
        if not torch.equal(a, b):
            err = (a.float() - b.float()).abs().max().item()
            return f"MISMATCH at step {i}: max abs err {err:.3e}"
    for name, state in eager.states.items():
        if name not in other.states or not torch.equal(state, other.states[name]):
            return f"MISMATCH in final {name}"
    return None


def _make_step(op: str, base: dict[str, torch.Tensor], args: tuple[float, float, int]) -> nn.Module:
    scale, eps, gate = args
    if op == "core":
        return CoreStep(base["A_log"], base["dt_bias"], base["conv_weight_t"], base["norm_weight"],
                        base["load_indices"], base["store_indices"], scale, eps, gate)
    if op == "verify":
        return VerifyStep(base["A_log"], base["dt_bias"], base["conv_weight_t"], base["norm_weight"],
                          base["row_indices"], base["ssm_store_indices"], scale, eps, gate)
    return RecurrentStep(base["A_log"], base["dt_bias"], base["load_indices"], base["store_indices"], scale)


def _op_states(op: str, base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if op in ("core", "verify"):
        return {"conv_state": base["conv_state"], "ssm_state": base["ssm_state"]}
    return {"ssm_state": base["ssm_state"]}


def _op_activations(op: str, activations: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, ...]]:
    if op in ("core", "verify"):
        return [(d["mixed_qkv"], d["z"], d["a"], d["b"]) for d in activations]
    return [(d["qkv"], d["a"], d["b"]) for d in activations]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "hpu"), default="auto")
    parser.add_argument("--case", default=None, choices=sorted(CASES_BY_NAME))
    parser.add_argument("--ops", default="core,recurrent")
    ns = parser.parse_args(argv)

    if ns.steps < 1:
        print(f"--steps must be >= 1, got {ns.steps}", file=sys.stderr)
        return 2
    ops = [op.strip() for op in ns.ops.split(",") if op.strip()]
    if not ops or any(op not in OPS for op in ops):
        print(f"--ops must be a comma list of {','.join(OPS)}, got {ns.ops!r}", file=sys.stderr)
        return 2
    if "verify" in ops and len(ops) > 1:
        print(f"--ops verify must run alone: the verification cases run only the verify op, got {ns.ops!r}",
              file=sys.stderr)
        return 2
    default_case = DEFAULT_VERIFY_CASE if ops == ["verify"] else DEFAULT_CASE
    case = CASES_BY_NAME[ns.case if ns.case is not None else default_case]
    if case.name.startswith("ver_") != ("verify" in ops):
        if "verify" in ops:
            print(f"--ops verify runs only verification cases (names starting with ver_), not --case {case.name}",
                  file=sys.stderr)
        else:
            print(f"--case {case.name} is a verification case (ver_*): it runs only with --ops verify",
                  file=sys.stderr)
        return 2
    device_str = ns.device
    if device_str == "auto":
        device_str = "hpu" if hpu_present() else "cpu"
    elif device_str == "hpu" and not hpu_present():
        print("--device hpu requested but no Gaudi device is visible", file=sys.stderr)
        return 2
    device = torch.device(device_str)
    if device_str == "hpu":
        import habana_frameworks.torch  # noqa: F401  (registers the HPU device with torch)
        from flashinfer_gaudi._native import load_gdn_fused_extension

        if not load_gdn_fused_extension():  # nothing else loads it in a standalone script
            print("the gdn_fused_ops extension did not load: build it with tools/build_flashinfer_gaudi.py "
                  "--gdn-fused-only, or point GDN_FUSED_GAUDI_KERNEL_LIBRARY and GDN_FUSED_GAUDI_NATIVE_LIBRARY "
                  "at the built files", file=sys.stderr)
            return 2
    mode = bridge_mode(device)

    if case.name.startswith("ver_"):
        base = {name: tensor.to(device) for name, tensor in make_verify_inputs(case).items()}
        args = verify_scalars(case)
    else:
        base = {name: tensor.to(device) for name, tensor in make_inputs(case).items()}
        args = scalars(case)
    activations = [{k: v.to(device) for k, v in d.items()} for d in draw_activations(base, ns.steps)]
    print(f"device={device_str} bridge={BRIDGE_NAMES[mode]} case={case.name} steps={ns.steps} "
          f"ops={','.join(ops)}", flush=True)

    ok = True
    for op in ops:
        states = _op_states(op, base)
        acts = _op_activations(op, activations)
        eager = run(_make_step(op, base, args), states, acts, device)
        variants_run = 0

        graph_reason = graph_skip_reason(mode)
        if graph_reason is None:
            graphed = wrap_in_hpu_graph(_make_step(op, base, args))
            message = compare_runs(eager, run(graphed, states, acts, device))
            print(f"{op} lazy+hpu_graph: {'exact' if message is None else message}", flush=True)
            ok = ok and message is None
            variants_run += 1
        else:
            print(f"{op} lazy+hpu_graph: skipped {graph_reason}", flush=True)

        compile_reason = compile_skip_reason(mode)
        if compile_reason is None:
            result, recompiles = run_compiled(_make_step(op, base, args), states, acts, device)
            message = compare_runs(eager, result)
            print(f"{op} torch.compile: {'exact' if message is None else message}", flush=True)
            print(f"{op} recompiles after warmup: {recompiles}", flush=True)
            ok = ok and message is None and recompiles == 0
            variants_run += 1
        else:
            print(f"{op} torch.compile: skipped {compile_reason}", flush=True)
            print(f"{op} recompiles after warmup: skipped {compile_reason}", flush=True)

        if variants_run == 0:
            print(f"{op}: no replay variant runs in bridge mode {BRIDGE_NAMES[mode]}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
