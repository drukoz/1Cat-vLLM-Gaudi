# SPDX-License-Identifier: Apache-2.0
"""HPU registration for the Qwen4Exp (Flash-Next) architectures.

Import order is load-bearing: ``vllm_gaudi.models.qwen3_5`` replaces
``QwenGatedDeltaNetAttention`` in the upstream GDN module and
``vllm_gaudi.models.qwen3_next`` patches ``Qwen3NextSparseMoeBlock.forward``,
both at their import time. Upstream ``vllm.models.qwen4_exp.nvidia.model``
binds those names by ``from ... import`` at its own import time, so the two
plugin modules must be imported before it.
"""

import vllm_gaudi.models.qwen3_5 as _qwen3_5  # noqa: F401  (must precede the upstream model import)
import vllm_gaudi.models.qwen3_next as _qwen3_next  # noqa: F401  (must precede the upstream model import)
import vllm.models.qwen4_exp.nvidia.model as _upstream_model
from vllm.model_executor.models.registry import ModelRegistry

FLASH_NEXT_ARCHS = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")


class HpuQwen4ExpForConditionalGeneration(_upstream_model.Qwen4ExpForConditionalGeneration):
    """HPU specialization of the Flash-Next multimodal model. No overrides yet."""


class HpuQwen4ExpForCausalLM(_upstream_model.Qwen4ExpForCausalLM):
    """HPU specialization of the Flash-Next text-only model. No overrides yet."""


def apply_flash_next_patches() -> None:
    """Make the fork's GDN/MoE replacements effective on the Flash-Next model.

    The replacements are installed by importing ``vllm_gaudi.models.qwen3_5``
    and ``vllm_gaudi.models.qwen3_next`` (see module docstring); this asserts
    the GDN binding reached the upstream model module. Later tasks add their
    replacements (HyperConnection, PLE, QSA) here.
    """
    if _upstream_model.QwenGatedDeltaNetAttention is not _qwen3_5.HPUGatedDeltaNetAttention:
        raise RuntimeError("Flash-Next GDN replacement did not take: vllm_gaudi.models.qwen3_5 and "
                           "vllm_gaudi.models.qwen3_next must be imported before "
                           "vllm.models.qwen4_exp.nvidia.model")


def register() -> None:
    apply_flash_next_patches()
    for arch in FLASH_NEXT_ARCHS:
        ModelRegistry.register_model(arch, f"vllm_gaudi.models.qwen4_exp:Hpu{arch}")
