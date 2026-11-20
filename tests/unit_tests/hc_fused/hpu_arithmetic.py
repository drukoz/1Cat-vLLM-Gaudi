# SPDX-License-Identifier: Apache-2.0
"""One arithmetic rule of a Gaudi2 card that CPU torch does not have, as a CPU emulation.

**The rule.**  A bf16 tensor divided by a Python number is, on the card, a multiplication by the
number's reciprocal rounded to bf16: ``x / 240.0`` is ``bf16(x * bf16(1 / 240))``.  CPU torch divides:
``bf16(x / 240)``.  ``bf16(1 / 240)`` is ``0x1.12p-8``, 0.34 % above ``1 / 240``, so the two results
differ by one bf16 step for about two values in three.

**Evidence.**

* The first card runs of the MoE glue kernels (Gaudi2, lazy mode, 2026-11-20: twice with the bridge's
  parallel graph accumulation on, once with it off).  ``test_moe_hpu_op.py`` printed, every time,
  ``MOEHPU row_quant_real_r32 against dynamic_quant on the card: 22 of 32 scales and 3378 of 81920 fp8
  values differ (max abs diff 1.600e+01)``.  The kernel of that build computed ``bf16(x * fp32(1 / 240))``,
  which is CPU torch's quotient for every bf16 ``x``, and its case passed against the CPU oracle in the
  same run.  The plugin's ``dynamic_quant`` (``vllm_gaudi/extension/ops.py``, the default branch) run on
  CPU under this rule differs from it in exactly 22 scales and 3378 fp8 values, with a largest difference
  of 16 (``test_moe_card_arithmetic.py``).  No neighbouring model gives these counts: the constant one
  bf16 step lower gives 19 scales, one step higher 32, a truncated product 6, an unrounded quotient 31.
* ``csrc/deepseek_v4/kernels/deepseek_v41_dynamic_quant_bf16_gaudi2.c``, an earlier kernel written to match
  the vendor's scale op (the other branch of ``dynamic_quant``), multiplies the row maximum by
  ``(float)(bf16)(1.0f / 240.0f)``.

**Scope.**  What is known is the one expression of ``dynamic_quant``: per-row maxima of ordinary size
divided by ``FP8_MAX``.  The mode below applies the rule to every ``bf16 tensor / Python number`` it
sees and to nothing else: not to a tensor divided by a tensor, not to other dtypes, not to the in-place
form.  Whether the card treats those the same way is not known.

The arithmetic itself is ``moe_reference.hpu_div_by_scalar``; this module only applies it to torch code
that was written with ``/``, so that a test can run the plugin's own function the way a card does.
"""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.unit_tests.hc_fused.moe_reference import hpu_div_by_scalar


class HpuScalarDivision(TorchDispatchMode):
    """Inside this mode ``bf16_tensor / python_number`` returns what a Gaudi2 card returns for it."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.div.Tensor and len(args) == 2 and not kwargs:
            dividend, divisor = args
            if (isinstance(dividend, torch.Tensor) and dividend.dtype == torch.bfloat16
                    and isinstance(divisor, (int, float)) and not isinstance(divisor, bool)):
                return hpu_div_by_scalar(dividend, divisor)
        return func(*args, **kwargs)


def card_arithmetic() -> HpuScalarDivision:
    """``with card_arithmetic(): ...`` runs torch code on CPU with the card's scalar division."""
    return HpuScalarDivision()
