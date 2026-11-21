# SPDX-License-Identifier: Apache-2.0
"""GDN prefill triangular solve: accuracy of the cheaper settings against today's 14 Newton steps (idea G6-2).

Every 128-token chunk of a GDN prefill inverts L = I + N, N strictly lower (beta * K K^T * decay). The default
runs 14 Newton steps X <- X (2I - L X). In exact arithmetic step 7 is exact (N^128 = 0), which is why the idea
sweep proposed VLLM_GDN_NEUMANN_ITERS=7. In fp32 that is wrong for ill-conditioned chunks (keys that nearly
repeat inside a chunk, beta near 1, slow decay: repeated code lines or indentation): the iterates pass through
values near 1e35 and step 7 is far from the inverse; steps 8 to 14 act as iterative refinement and repair it.
The recursive block inverse with base 16 (VLLM_GDN_RECURSIVE_SOLVER_BASE=16, the fork's promoted tactic) does
not blow up but leaves an error near 3e-4 on such chunks. One Newton step after it
(VLLM_GDN_RECURSIVE_SOLVER_REFINE=1) brings it back to today's accuracy at about a seventh of the solve's work.

The tests run the plugin's own solver and chunk rule (fp32 compute, the server default) at one TP8 rank's
geometry (2 key heads, 6 value heads, head dim 128, chunk 128) and compare against float64.
"""

from __future__ import annotations

import importlib
import os
from unittest import mock

import pytest
import torch

N = 128
H_K, H_V, D = 2, 6, 128  # one rank at TP8: 16 / 48 heads over 8 ranks


def _import_gdn(env: dict[str, str] | None = None):
    base = {"VLLM_GDN_LEGACY_PHASE_B": "0", "VLLM_GDN_COMPUTE_FP32": "1", "VLLM_GDN_EXACT_SOLVE": "0",
            "VLLM_GDN_RECURSIVE_SOLVER_REFINE": "0"}
    base.update(env or {})
    with mock.patch.dict(os.environ, base):
        import vllm_gaudi.ops.hpu_gdn_pytorch as mod
        return importlib.reload(mod)


@pytest.fixture(scope="module")
def gdn():
    return _import_gdn()


@pytest.fixture
def refine(gdn, monkeypatch):
    """Set the module's refinement steps (the flag is read at import; test_refine_flag_is_read_at_import)."""

    def set_steps(steps: int):
        monkeypatch.setattr(gdn, "_RECURSIVE_SOLVER_REFINE", steps)
        return gdn

    return set_steps


def _chunk_matrix(seed: int, spread: float, beta_logit: float = 3.0) -> torch.Tensor:
    """L = I + strict_lower(beta * K K^T) for one chunk of nearly repeated, normalized keys (no decay)."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(1, D, generator=gen)
    keys = base + spread * torch.randn(N, D, generator=gen)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    beta = torch.sigmoid(beta_logit + 0.5 * torch.randn(N, generator=gen))
    return (torch.eye(N) + torch.tril((keys @ keys.T) * beta[:, None], -1)).unsqueeze(0)


STRESSED = [(seed, spread) for seed in range(8) for spread in (0.02, 0.05, 0.1, 0.3, 1.0)]


def _worst_error(solve) -> float:
    worst = 0.0
    for seed, spread in STRESSED:
        lmat = _chunk_matrix(100 + seed, spread)
        exact = torch.linalg.inv(lmat.double())
        err = (solve(lmat).double() - exact).abs().max().item()
        worst = max(worst, err if err == err else float("inf"))
    return worst


def _solve(mod, iters: int, base: int = 0):
    eye = torch.eye(N)
    return lambda lmat: mod._hpu_solve_lower_triangular_batched(lmat, eye, use_vectorized=True, neumann_iters=iters,
                                                                recursive_base=base)


def test_today_14_steps_are_accurate_on_ill_conditioned_chunks(gdn):
    assert _worst_error(_solve(gdn, 14)) < 1e-6


def test_seven_steps_are_not_enough_in_fp32(gdn):
    # The exact-arithmetic argument (N^128 = 0 after 7 squarings) does not survive fp32 rounding.
    assert _worst_error(_solve(gdn, 7)) > 1.0


def test_recursive_base_16_alone_loses_accuracy(gdn):
    assert 1e-5 < _worst_error(_solve(gdn, 14, base=16)) < 1e-2


def test_recursive_base_16_with_one_refinement_step_matches_today(refine):
    assert _worst_error(_solve(refine(1), 14, base=16)) < 1e-6


def test_refinement_leaves_the_newton_path_alone(gdn, refine):
    lmat = _chunk_matrix(3, 0.3)
    eye = torch.eye(N)
    ref = gdn._hpu_solve_lower_triangular_batched(lmat, eye, use_vectorized=True, neumann_iters=14)
    out = refine(1)._hpu_solve_lower_triangular_batched(lmat, eye, use_vectorized=True, neumann_iters=14)
    assert torch.equal(out, ref)


def test_refine_flag_is_read_at_import():
    try:
        assert _import_gdn()._RECURSIVE_SOLVER_REFINE == 0
        assert _import_gdn({"VLLM_GDN_RECURSIVE_SOLVER_REFINE": "1"})._RECURSIVE_SOLVER_REFINE == 1
        with pytest.raises(ValueError, match="VLLM_GDN_RECURSIVE_SOLVER_REFINE"):
            _import_gdn({"VLLM_GDN_RECURSIVE_SOLVER_REFINE": "-1"})
    finally:
        _import_gdn()


def _gdn_inputs(T: int, regime: str, seed: int):
    gen = torch.Generator().manual_seed(seed)
    if regime == "repeated":
        base = torch.randn(1, 1, H_K, D, generator=gen)
        k = base + 0.05 * torch.randn(1, T, H_K, D, generator=gen)
        q = base + 0.05 * torch.randn(1, T, H_K, D, generator=gen)
        beta = torch.sigmoid(3.0 + 0.3 * torch.randn(1, T, H_V, generator=gen))
        g = -0.001 * torch.rand(1, T, H_V, generator=gen)
    else:
        k = torch.randn(1, T, H_K, D, generator=gen)
        q = torch.randn(1, T, H_K, D, generator=gen)
        beta = torch.sigmoid(torch.randn(1, T, H_V, generator=gen))
        decay_rate = torch.empty(H_V).uniform_(1, 16, generator=gen)  # exp(A_log), A_log ~ log U(1, 16)
        g = -decay_rate * torch.nn.functional.softplus(torch.randn(1, T, H_V, generator=gen) - 4.0)
    v = torch.randn(1, T, H_V, D, generator=gen)
    state = 0.1 * torch.randn(1, H_V, D, D, generator=gen)
    return q.bfloat16(), k.bfloat16(), v.bfloat16(), g.float(), beta.float(), state


def _chunk_rule(mod, inputs, iters: int, base: int):
    q, k, v, g, beta, state = inputs
    return mod.hpu_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state.clone(), output_final_state=True,
                                          use_qk_l2norm_in_kernel=True, chunk_size=N, prefill_num_seqs=1,
                                          prefill_seq_len=q.shape[1], neumann_iters=iters,
                                          recursive_solver_base=base)


@pytest.mark.parametrize("regime", ["decaying", "repeated"])
def test_chunk_rule_error_against_float64_recurrence(gdn, refine, regime):
    inputs = _gdn_inputs(512, regime, seed=7)
    q, k, v, g, beta, state = inputs
    ref_out, ref_state = gdn.hpu_fused_recurrent_gated_delta_rule(q.double(), k.double(), v.double(), g.double(),
                                                                  beta.double(), initial_state=state.double().clone(),
                                                                  use_qk_l2norm_in_kernel=True,
                                                                  inplace_final_state=False)

    def errors(mod, iters, base):
        out, final = _chunk_rule(mod, inputs, iters, base)
        return ((out.double() - ref_out).abs().max().item(), (final.double() - ref_state).abs().max().item())

    today = errors(refine(0), 14, 0)
    refined = errors(refine(1), 14, 16)
    assert today[0] < 2e-3 and today[1] < 1e-5
    # one refinement step after the base-16 inverse: no worse than today (output and final state)
    assert refined[0] <= 1.1 * today[0] + 1e-6
    assert refined[1] <= 2.0 * today[1] + 1e-6
