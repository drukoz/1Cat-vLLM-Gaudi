# SPDX-License-Identifier: Apache-2.0
"""The fused hyper-connection path of ``HpuGatedResidual``, called directly.

The fused path is part of the layer (``vllm_gaudi/models/qwen4_exp/hyperconnection.py``): with
``VLLM_HPU_HC_FUSED=1`` its ``mix`` / ``combine_and_mix`` / ``combine`` run the functions below
when the layer's preconditions hold.  This module was the draft of that code; it now calls the
layer's own methods, so there is one implementation.  ``module`` is an ``HpuGatedResidual`` built
with any value of the flag: these functions run the fused path whatever the flag says.

The ops are the stand-in of ``hc_ops.py`` (the torch oracle as their CPU kernels), so
``test_fused_path.py`` can check on CPU that the fused path returns exactly what the torch methods
return, and count the torch ops of both paths.  The layer's own tests (flag, preconditions, log
line, fallback, wiring) are ``tests/unit_tests/models/flash_next/test_hc_fused_layer.py``.
"""

from __future__ import annotations

import torch

from tests.unit_tests.hc_fused import hc_ops


def fused_precondition(module, hidden_states: torch.Tensor) -> str | None:
    """Why the fused path cannot serve this module and input, or None when it can (the layer's two checks).

    Shape, dtype and layout facts only (static in a compiled graph): nothing here reads tensor data.
    """
    return (module._hc_fused_layer_reason(hidden_states.device)
            or module._hc_fused_call_reason(hidden_states, False, None, None))


def fused_mix(module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``HpuGatedResidual.mix`` with the fused ops."""
    return module._hc_fused_mix(hc_ops, hidden_states)


def fused_combine_and_mix(module, hidden_states: torch.Tensor, prev_block_output: torch.Tensor,
                          prev_injection: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``HpuGatedResidual.combine_and_mix`` with the fused ops."""
    return module._hc_fused_combine_and_mix(hc_ops, hidden_states, prev_block_output, prev_injection)


def fused_combine(module, hidden_states: torch.Tensor, block_output: torch.Tensor,
                  injection: torch.Tensor) -> torch.Tensor:
    """``HpuGatedResidual.combine`` with the fused op."""
    return hc_ops.hc_combine(hidden_states, block_output, injection, module.hc_count)
