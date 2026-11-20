# SPDX-License-Identifier: Apache-2.0
"""Layer-level tests for the fused GDN verification op (W1 of track RUK-2969).

With ``VLLM_HPU_GDN_CORE_VERIFY=1`` every MTP verification step (the 2-D ``[B, T]`` state lattice)
and the plain decode step that follows one (1-D load indices with ``num_accepted`` set) of
``HPUGatedDeltaNetAttention`` runs ``torch.ops.flashinfer_gaudi.gdn_core_verify`` instead of the
torch MTP path. The M3 acceptance harnesses are the parity engine: the same oracle/tamper chains
as ``test_mtp_end_to_end.test_accept_paths_state_parity`` and the S6 align scenario must equal the
straight fp32 chain with the flag on, and the op must be the code that produced the result:
a counting wrapper records every call's ``ssm_store_indices`` shape (``args[11]``, ``[B, T]``),
and zeroing the op's output must break the row-0 logits parity. A layer whose op is unavailable
logs exactly one warning per GDN layer and runs the torch path; with the flag off the op is
never called.

Every test runs at each size of ``flash_next_spec_sizes.SPEC_SIZES`` (k = 3, 4, 5): the op must
take the ``1 + k``-row lattice of each (``T`` = 4, 5 and 6, the op's maximum), not fall back to
the torch path above the recipe's size.

One ``cpu_model`` per test function, never nested; logits only from ``fp32_logits``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402

import vllm_gaudi.models.qwen3_5 as qwen3_5  # noqa: E402

from flashinfer_gaudi.gdn_core_verify import GDN_CORE_VERIFY_MAX_NODES  # noqa: E402
from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup  # noqa: E402
from tests.unit_tests.models.flash_next.test_mtp_end_to_end import (  # noqa: E402
    FP32_TOL, K, MUT_TOL, TAMPER, StepLayout, _chain, _dense_context, _load_prompt)
from tests.unit_tests.models.flash_next.test_mtp_align_continuity import (  # noqa: E402
    _SCENARIOS, _chain as _align_chain, _check, _check_sized, _sized_chain)

SSM_STORE_INDEX_ARG = 11  # gdn_core_verify(conv, ssm, qkv, z, a, b, A_log, dt_bias, weight, norm_w,
                          # row_indices, ssm_store_indices, ...)


class VerifyOpSpy:
    """Monkeypatch of ``qwen3_5.gdn_core_verify`` recording ``args[11].shape`` and forwarding."""

    def __init__(self, real) -> None:
        self.real = real
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, *args, **kwargs):
        self.shapes.append(tuple(int(v) for v in args[SSM_STORE_INDEX_ARG].shape))
        return self.real(*args, **kwargs)

    @property
    def calls(self) -> int:
        return len(self.shapes)

    @property
    def ts(self) -> list[int]:
        """The node count ``T`` of every recorded call (``ssm_store_indices`` is ``[B, T]``)."""
        return [int(shape[1]) for shape in self.shapes]

    def log(self, label: str) -> None:
        print(f"{label}: {self.calls} gdn_core_verify calls, T counts {self.ts}")


class RecordingLogger:
    """Stand-in for ``qwen3_5.logger`` that records the raw ``warning(*args)`` and ``debug(*args)`` tuples."""

    def __init__(self) -> None:
        self.warnings: list[tuple] = []
        self.debugs: list[tuple] = []

    def warning(self, *args) -> None:
        self.warnings.append(args)

    def debug(self, *args) -> None:
        self.debugs.append(args)


def test_the_sizes_reach_the_largest_lattice_the_op_takes():
    """The largest size the tests run is the op's limit: ``1 + k = GDN_CORE_VERIFY_MAX_NODES``."""
    assert 1 + max(SPEC_SIZES) == GDN_CORE_VERIFY_MAX_NODES


def _accept_path_step(tiny_checkpoint, steps: int, k: int = K, **chain_kwargs):
    """One dense M3 accept-path chain with the flag-independent plumbing; ``(chain result, gdn)``."""
    lay = StepLayout(k)
    prompt = _load_prompt(tiny_checkpoint, 33)
    ids, gen = list(prompt["prompt_ids"]), list(prompt["generated_ids"])
    ctx, *rest = _dense_context(tiny_checkpoint, lay)
    try:
        args = (rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7])
        out = _chain(*args, ids, gen, steps, lay=lay, **chain_kwargs)
    finally:
        ctx.__exit__(None, None, None)
    return out, len(rest[6])


def test_accept_paths_run_the_verify_op(tiny_checkpoint, monkeypatch, spec_k):
    """The M3 accept-path chain with the flag on: parity, and every step ran the op with T = 1 + k.

    Same chain, acceptance and parity asserts as ``test_mtp_end_to_end.test_accept_paths_state_parity``
    (a = k, 1, 0, 2 with the tampered drafts); added here, the counting wrapper must have seen
    ``gdn_core_verify`` calls, every one of them on the ``1 + k``-row lattice, a whole number of layers.
    """
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "1")
    spy = VerifyOpSpy(qwen3_5.gdn_core_verify)
    monkeypatch.setattr(qwen3_5, "gdn_core_verify", spy)
    (seq, accepts, gaps, log0, bonus), num_gdn = _accept_path_step(tiny_checkpoint,
                                                                   4,
                                                                   spec_k,
                                                                   oracle_steps={0, 1, 3},
                                                                   tamper_steps={1: (1, TAMPER),
                                                                                 3: (2, TAMPER)})
    state_worst = [max(g) for g in gaps]
    print(f"MTP accept-path acceptance counts: {accepts}, state gaps {[f'{g:.3e}' for g in state_worst]}, "
          f"row-0 logits gaps {[f'{g:.3e}' for g in log0]}, bonus gaps {[f'{g:.3e}' for g in bonus]}")
    assert accepts[0] == spec_k, f"the oracle step did not commit K drafts: {accepts}"
    assert accepts[1] == 1, f"the tampered step did not commit exactly one draft: {accepts}"
    assert accepts[3] == 2, f"the second tampered step did not commit two drafts: {accepts}"
    stream = seq[len(_load_prompt(tiny_checkpoint, 33)["prompt_ids"]):]
    assert stream == _load_prompt(tiny_checkpoint, 33)["generated_ids"][:len(stream)], \
        f"stream {stream} is not the reference prefix"
    for i, (g, l0) in enumerate(zip(state_worst, log0)):
        assert g <= FP32_TOL, f"step {i} committed state diverges from the straight chain: {g:.3e}"
        assert l0 <= FP32_TOL, f"step {i} row-0 logits diverge from the straight chain: {l0:.3e}"
        assert bonus[i] <= FP32_TOL, f"step {i} bonus logits diverge from the straight chain: {bonus[i]:.3e}"
    spy.log("accept-path chain")
    assert spy.calls > 0, "the verification steps never called gdn_core_verify"
    assert all(t == 1 + spec_k for t in spy.ts), f"a verification step ran the op with T != {1 + spec_k}: {spy.ts}"
    assert spy.calls == 4 * num_gdn, f"{spy.calls} calls, expected one per step and GDN layer (4 x {num_gdn})"


def test_align_plain_steps_run_the_verify_op(tiny_checkpoint, monkeypatch):
    """S6 (verify, plain, verify, plain, verify) with the flag on: the trace, and T = 4 and T = 1 calls.

    The op must serve both shapes of the align chain: the verification steps as ``[1, 4]`` lattice
    calls, and the plain step after one (1-D load indices with ``num_accepted`` set) as ``[1, 1]``.
    """
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "1")
    spy = VerifyOpSpy(qwen3_5.gdn_core_verify)
    monkeypatch.setattr(qwen3_5, "gdn_core_verify", spy)
    exp = _SCENARIOS["S6"]
    records, final, gen = _align_chain(tiny_checkpoint, "S6", exp["tampers"], len(exp["a"]), exp["modes"])
    _check("S6", records, final, gen)
    spy.log("S6 chain")
    assert 4 in spy.ts, f"no verification step (T = 4) ran the op: {spy.ts}"
    assert 1 in spy.ts, f"no plain step after a verification (T = 1) ran the op: {spy.ts}"


def test_align_plain_steps_run_the_verify_op_at_every_size(tiny_checkpoint, monkeypatch, spec_k):
    """The verify, plain, verify, plain, verify align chain at each size: ``[1, 1 + k]`` and ``[1, 1]`` calls only.

    Three verification steps and two plain steps, each through every GDN layer, and none of them on the
    torch path: ``T = 6`` (k = 5) is the op's largest lattice.
    """
    require_startup(spec_k, prefix_caching=True)
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "1")
    spy = VerifyOpSpy(qwen3_5.gdn_core_verify)
    monkeypatch.setattr(qwen3_5, "gdn_core_verify", spy)
    scenario, label, records, final, reference = _sized_chain(tiny_checkpoint, "plain_between", spec_k, 4)
    _check_sized(scenario, label, records, final, reference, 4)
    spy.log(label)
    assert set(spy.ts) == {1, 1 + spec_k}, f"the op ran lattices other than T = 1 and T = {1 + spec_k}: {spy.ts}"
    # The plain reference stream and the straight chain decode without num_accepted: they never reach the op.
    assert spy.ts.count(1 + spec_k) * 2 == spy.ts.count(1) * 3, f"not three verification steps per two plain: {spy.ts}"


def test_zero_output_breaks_parity(tiny_checkpoint, monkeypatch, spec_k):
    """Negative control: zeroing the op's output moves the row-0 logits, so the chain uses its result.

    The states still update (the wrapper calls the real op and zeroes only the returned rows); with
    the verification's logits flattened to zeros the greedy accept loop commits nothing, and the
    row-0 logits gap against the straight chain must exceed ``MUT_TOL``.
    """
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "1")
    real = qwen3_5.gdn_core_verify

    def zeroing_op(*args, **kwargs):
        return torch.zeros_like(real(*args, **kwargs))

    monkeypatch.setattr(qwen3_5, "gdn_core_verify", zeroing_op)
    (seq, accepts, gaps, log0, bonus), _ = _accept_path_step(tiny_checkpoint, 1, spec_k, oracle_steps={0})
    state_worst = [max(g) for g in gaps]
    print(f"zeroed verify output: accepts {accepts}, row-0 logits gap {log0[0]:.3e}, state gaps "
          f"{[f'{g:.3e}' for g in state_worst]}")
    assert log0[0] > MUT_TOL, (f"zeroing the op output moved the row-0 logits by only {log0[0]:.3e}: "
                               "the layer ignores the op's output")


def test_unavailable_op_falls_back_once(tiny_checkpoint, monkeypatch, spec_k):
    """``gdn_core_verify_available`` False: one warning per GDN layer, the torch path, parity intact."""
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "1")
    spy = VerifyOpSpy(qwen3_5.gdn_core_verify)
    monkeypatch.setattr(qwen3_5, "gdn_core_verify", spy)
    monkeypatch.setattr(qwen3_5, "gdn_core_verify_available", lambda device: False)
    log = RecordingLogger()
    monkeypatch.setattr(qwen3_5, "logger", log)
    (seq, accepts, gaps, log0, bonus), num_gdn = _accept_path_step(tiny_checkpoint, 1, spec_k, oracle_steps={0})
    assert spy.calls == 0, "the fallback path called the verification op"
    # Only the verify hook's lines: the phase-1 decode hook logs its own (mode 1 on a 2-D lattice step).
    verify_warnings = [args for args in log.warnings if "GDN core verify" in args[0]]
    assert len(verify_warnings) == num_gdn, (f"expected one verify warning per GDN layer ({num_gdn}), got "
                                             f"{len(verify_warnings)}: {[a[0] % a[1:] for a in verify_warnings]}")
    for args in verify_warnings:
        message = args[0] % args[1:]
        assert "GDN core verify (VLLM_HPU_GDN_CORE_VERIFY=1) is off for" in message, f"warning {message!r}"
        assert "no gdn_core_verify kernel for cpu" in message, f"warning {message!r}"
    worst = max([max(g) for g in gaps] + log0 + bonus)
    print(f"verify-unavailable fallback: accepts {accepts}, worst gap {worst:.3e}, warnings {len(verify_warnings)}")
    assert worst <= FP32_TOL, f"the fallback step diverges from the straight chain: {worst:.3e}"


def test_flag_off_keeps_the_torch_path(tiny_checkpoint, monkeypatch, spec_k):
    """With ``VLLM_HPU_GDN_CORE_VERIFY`` unset the layer never reaches the op: the M3 path is untouched."""
    monkeypatch.delenv("VLLM_HPU_GDN_CORE_VERIFY", raising=False)
    spy = VerifyOpSpy(qwen3_5.gdn_core_verify)
    monkeypatch.setattr(qwen3_5, "gdn_core_verify", spy)
    (seq, accepts, gaps, log0, bonus), _ = _accept_path_step(tiny_checkpoint, 1, spec_k, oracle_steps={0})
    print(f"flag off: accepts {accepts}, gdn_core_verify calls {spy.calls}")
    assert spy.calls == 0, f"the torch path called the verification op {spy.calls} times"
