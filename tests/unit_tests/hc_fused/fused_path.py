# SPDX-License-Identifier: Apache-2.0
"""The fused hyper-connection path of ``HpuGatedResidual``, as free functions.

This is the executable form of the layer patch described in ``docs/design/flash-next/hc-fused.md``
("Layer patch"): the three methods below are what ``HpuGatedResidual.mix`` /
``combine_and_mix`` / ``combine`` run when ``VLLM_HPU_HC_FUSED=1`` and the layer's precondition
holds.  ``module`` is an ``HpuGatedResidual`` (duck-typed: ``hc_norm.weight``, ``config.rms_norm_eps``,
``hc_count``, ``lora_rank``, ``use_combine`` and the vLLM linear layers).  The GEMMs stay the
module's own linear layers; everything between and around them is one fused op each.

``test_fused_path.py`` checks on CPU that these return exactly what the torch methods return and
counts the torch ops of both paths.
"""

from __future__ import annotations

import torch

from tests.unit_tests.hc_fused.hc_ops import hc_combine, hc_combine_norm, hc_gate_mix, hc_norm, hc_silu, hc_silu_split


def fused_precondition(module, hidden_states: torch.Tensor) -> str | None:
    """Why the fused path cannot serve this module and input, or None when it can.

    Shape and dtype facts only (static in a compiled graph): nothing here reads tensor data.
    """
    if hidden_states.dim() != 2:
        return f"hidden states are {hidden_states.dim()}-D, the fused ops take [tokens, width]"
    if hidden_states.dtype != torch.bfloat16:
        return f"hidden states are {hidden_states.dtype}, the fused ops take bfloat16"
    if module.hidden_size % 128 != 0:
        return f"hidden_size {module.hidden_size} is not a multiple of 128 (one TPC vector)"
    weight = module.hc_norm.weight
    if weight.dtype != torch.bfloat16 or weight.dim() != 1:
        return f"hc_norm.weight is {weight.dtype} {tuple(weight.shape)}, the fused ops take a 1-D bfloat16 weight"
    return None


def _mix_tail(module, hidden_states: torch.Tensor,
              xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if module.use_combine:
        down_and_injection = module.input_mix_weight_down_block_inject(xn)  # GEMM 1 (matrix engine)
        lora, injection = hc_silu_split(down_and_injection, module.hc_count, module.lora_rank)
    else:
        lora = hc_silu(module.input_mix_weight_down(xn), module.hc_count)
        injection = None
    gate = module.input_mix_weight_up(lora)  # GEMM 2 (matrix engine)
    block_input = hc_gate_mix(xn, gate, module.hc_count)
    return hidden_states, block_input, injection


def fused_mix(module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``HpuGatedResidual.mix`` with the fused ops."""
    xn = hc_norm(hidden_states, module.hc_norm.weight, module.config.rms_norm_eps, module.hc_count)
    return _mix_tail(module, hidden_states, xn)


def fused_combine_and_mix(module, hidden_states: torch.Tensor, prev_block_output: torch.Tensor,
                          prev_injection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``HpuGatedResidual.combine_and_mix`` with the fused ops."""
    hidden_states, xn = hc_combine_norm(hidden_states, prev_block_output, prev_injection, module.hc_norm.weight,
                                        module.config.rms_norm_eps, module.hc_count)
    return _mix_tail(module, hidden_states, xn)


def fused_combine(module, hidden_states: torch.Tensor, block_output: torch.Tensor,
                  injection: torch.Tensor) -> torch.Tensor:
    """``HpuGatedResidual.combine`` with the fused op."""
    return hc_combine(hidden_states, block_output, injection, module.hc_count)
