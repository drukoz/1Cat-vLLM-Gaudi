# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the GDN decode replay tool (K6, run sheet row 3f step 2).

Small-step runs of the tool's ``main`` on CPU, direct checks of ``compare_runs``, and a check
that the eager replay actually advances the carried ``ssm_state`` (the mutation check for
row 3f: a step module that hands the op a clone would leave the carried state frozen).
"""

from __future__ import annotations

import torch

from tests.unit_tests.gdn_core_decode.sim_case import HPU_CASES, make_inputs, scalars
from tools.flash_next import gdn_core_decode_replay as replay
from tools.flash_next.gdn_core_decode_replay import (RunResult, RecurrentStep, compare_runs, draw_activations,
                                                     main, run)

CASE_BY_NAME = {case.name: case for case in HPU_CASES}


def _recurrent_result(steps: int) -> tuple[RunResult, torch.Tensor]:
    """An eager recurrent replay on the padding-row case; also returns the starting ``ssm_state``."""
    case = CASE_BY_NAME["tp8_mixed"]
    base = make_inputs(case)
    scale, _, _ = scalars(case)
    activations = [(d["qkv"], d["a"], d["b"]) for d in draw_activations(base, steps)]
    module = RecurrentStep(base["A_log"], base["dt_bias"], base["load_indices"], base["store_indices"], scale)
    start = base["ssm_state"].clone()
    result = run(module, {"ssm_state": base["ssm_state"]}, activations, torch.device("cpu"))
    return result, start


def _copy(result: RunResult) -> RunResult:
    return RunResult([o.clone() for o in result.outputs], {k: v.clone() for k, v in result.states.items()})


def test_main_small_cpu(capsys):
    assert main(["--steps", "5", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "core lazy+hpu_graph: skipped (cpu)" in out
    assert "recurrent lazy+hpu_graph: skipped (cpu)" in out
    assert "core torch.compile: exact" in out
    assert "recurrent torch.compile: exact" in out
    assert "recompiles after warmup: 0" in out


def test_main_padding_rows():
    assert main(["--steps", "3", "--device", "cpu", "--case", "tp8_mixed"]) == 0


def test_compare_runs_step_mismatch():
    result, _ = _recurrent_result(3)
    other = _copy(result)
    other.outputs[1][0, 0, 0] += 1.0
    message = compare_runs(result, other)
    assert message is not None and "MISMATCH at step 2" in message


def test_compare_runs_final_state_mismatch():
    result, _ = _recurrent_result(3)
    other = _copy(result)
    other.states["ssm_state"][0, 0, 0, 0] += 1.0
    assert compare_runs(result, other) == "MISMATCH in final ssm_state"


def test_compare_runs_identical():
    result, _ = _recurrent_result(3)
    assert compare_runs(result, _copy(result)) is None


def test_eager_run_advances_state():
    result, start = _recurrent_result(3)
    assert not torch.equal(result.states["ssm_state"], start)


def test_main_verify_small_cpu(capsys):
    assert main(["--device", "cpu", "--steps", "3", "--ops", "verify"]) == 0
    out = capsys.readouterr().out
    assert "case=ver_tp8_b16" in out
    assert "verify lazy+hpu_graph: skipped (cpu)" in out
    assert "verify torch.compile: exact" in out
    assert "verify recompiles after warmup: 0" in out


def test_main_verify_t1_case():
    assert main(["--device", "cpu", "--steps", "3", "--ops", "verify", "--case", "ver_tp4_t1"]) == 0


def test_main_verify_arg_rules(capsys):
    for argv in (["--device", "cpu", "--steps", "2", "--ops", "core,verify"],
                 ["--device", "cpu", "--steps", "2", "--ops", "verify", "--case", "tp8_b16"],
                 ["--device", "cpu", "--steps", "2", "--ops", "core", "--case", "ver_tp8_b16"]):
        assert main(argv) == 2, argv
    assert capsys.readouterr().err.count("\n") == 3


def test_main_default_case_unchanged(capsys):
    assert main(["--device", "cpu", "--steps", "2", "--ops", "core"]) == 0
    out = capsys.readouterr().out
    assert "case=tp8_b16" in out


class _OffsetStep(torch.nn.Module):
    """Wraps a step module and shifts every step output by +1 (a graph variant that must MISMATCH)."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        return self.inner(*args) + 1


def test_lazy_mode_runs_graph_and_skips_compile(monkeypatch, capsys):
    monkeypatch.setattr(replay, "bridge_mode", lambda device: replay.BridgeMode.LAZY)
    monkeypatch.setattr(replay, "wrap_in_hpu_graph", lambda module: module)
    assert main(["--device", "cpu", "--steps", "3"]) == 0
    out = capsys.readouterr().out
    assert "bridge=lazy" in out
    for op in ("core", "recurrent"):
        assert f"{op} lazy+hpu_graph: exact" in out
        assert f"{op} torch.compile: skipped (lazy mode)" in out
        assert f"{op} recompiles after warmup: skipped (lazy mode)" in out


def test_eager_mode_runs_compile_and_skips_graph(monkeypatch, capsys):
    monkeypatch.setattr(replay, "bridge_mode", lambda device: replay.BridgeMode.EAGER)
    assert main(["--device", "cpu", "--steps", "3"]) == 0
    out = capsys.readouterr().out
    assert "core lazy+hpu_graph: skipped (eager mode)" in out
    assert "core torch.compile: exact" in out
    assert "core recompiles after warmup: 0" in out


def test_graph_mismatch_fails_in_lazy_mode(monkeypatch, capsys):
    monkeypatch.setattr(replay, "bridge_mode", lambda device: replay.BridgeMode.LAZY)
    monkeypatch.setattr(replay, "wrap_in_hpu_graph", lambda module: _OffsetStep(module))
    assert main(["--device", "cpu", "--steps", "3"]) == 1
    out = capsys.readouterr().out
    assert "lazy+hpu_graph: MISMATCH" in out


def test_no_variant_ran_is_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(replay, "graph_skip_reason", lambda mode: "(test)")
    monkeypatch.setattr(replay, "compile_skip_reason", lambda mode: "(test)")
    assert main(["--device", "cpu", "--steps", "3"]) == 1
    err = capsys.readouterr().err
    assert "no replay variant runs in bridge mode cpu" in err


def test_bridge_mode_cpu():
    assert replay.bridge_mode(torch.device("cpu")) is replay.BridgeMode.CPU
    assert (int(replay.BridgeMode.CPU), int(replay.BridgeMode.EAGER), int(replay.BridgeMode.LAZY)) == (0, 1, 2)
