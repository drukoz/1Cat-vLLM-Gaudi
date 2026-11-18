# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU registration for the Qwen4Exp (Flash-Next) architectures.

Import order is load-bearing: ``vllm_gaudi.models.qwen3_5`` replaces
``QwenGatedDeltaNetAttention`` in the upstream GDN module and
``vllm_gaudi.models.qwen3_next`` patches ``Qwen3NextSparseMoeBlock.forward``,
both at their import time. Upstream ``vllm.models.qwen4_exp.nvidia.model``
binds those names by ``from ... import`` at its own import time, so the two
plugin modules must be imported before it. ``vllm.models.qwen4_exp.nvidia.mtp``
imports from ``.model``, so it follows the upstream model import too.
"""

import torch
import vllm_gaudi.models.qwen3_5 as _qwen3_5  # noqa: F401  (must precede the upstream model import)
import vllm_gaudi.models.qwen3_next as _qwen3_next  # noqa: F401  (must precede the upstream model import)
import vllm.models.qwen4_exp.nvidia.model as _upstream_model
import vllm.models.qwen4_exp.nvidia.mtp as _upstream_mtp  # must follow the upstream model import
from vllm_gaudi.models.qwen4_exp.ple_layer import HpuQwen4ExpPLELayer  # must follow the upstream model import
from vllm.config import VllmConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.sequence import IntermediateTensors
from vllm_gaudi.models.qwen4_exp.hyperconnection import HpuGatedResidual
from vllm_gaudi.models.qwen4_exp.mtp import mtp_aux_hidden
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention

FLASH_NEXT_ARCHS = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")

MTP_ARCHS = ("Qwen4ExpMTP",)


class _MtpTargetOutputs:
    """Return ``(hidden, [multi])`` from a Flash-Next target forward under ``method == "mtp"``.

    ``multi`` is the pre-final-mixer multi-stream state ``Qwen4ExpModel._mtp_hidden_buffer[:T]``
    (viewed with ``hidden``'s leading shape). The tuple feeds the fork's aux-hidden channel
    (``use_aux_hidden_state_outputs`` in ``v1/worker/hpu_model_runner.py``), which the MTP
    drafter reads on step 0. Without an MTP speculative config, or when the upstream forward
    returns ``IntermediateTensors`` (a non-last PP rank), the return is unchanged.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        sc = vllm_config.speculative_config
        self._hpu_mtp_outputs = sc is not None and getattr(sc, "method", None) == "mtp"

    def forward(self,
                input_ids: torch.Tensor | None,
                positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs: object) -> torch.Tensor | IntermediateTensors:
        # The runner schedules [-1] for an empty n-gram draft (always rejected); id 0 keeps ids in embedding range.
        if input_ids is not None:
            input_ids = input_ids.clamp(min=0)
        hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if self._hpu_mtp_outputs and isinstance(hidden, torch.Tensor):
            return hidden, [mtp_aux_hidden(hidden, self.get_mtp_target_hidden_states())]
        return hidden


class HpuQwen4ExpForConditionalGeneration(_MtpTargetOutputs, _upstream_model.Qwen4ExpForConditionalGeneration):
    """HPU specialization of the Flash-Next multimodal model; the MTP target output tuple is added above."""
    # Sparse QSA decode gathers K/V through the per-request block_table (M2).
    requires_block_table = True


class HpuQwen4ExpForCausalLM(_MtpTargetOutputs, _upstream_model.Qwen4ExpForCausalLM):
    """HPU specialization of the Flash-Next text-only model; the MTP target output tuple is added above."""
    # Sparse QSA decode gathers K/V through the per-request block_table (M2).
    requires_block_table = True


def apply_flash_next_patches() -> None:
    """Make the fork's GDN/MoE/HyperConnection/QSA/PLE replacements effective on the Flash-Next model.

    The GDN and MoE replacements are installed by importing ``vllm_gaudi.models.qwen3_5``
    and ``vllm_gaudi.models.qwen3_next`` (see module docstring); this asserts
    the GDN binding reached the upstream model module, and rebinds the
    ``GatedResidual``, ``Qwen4ExpQSAAttention`` and ``Qwen4ExpPLELayer`` names the
    upstream decoder layer builds through at construction time.
    """
    _upstream_model.GatedResidual = HpuGatedResidual
    _upstream_model.Qwen4ExpQSAAttention = HpuQwen4ExpQSAAttention
    if _upstream_model.QwenGatedDeltaNetAttention is not _qwen3_5.HPUGatedDeltaNetAttention:
        raise RuntimeError("Flash-Next GDN replacement did not take: vllm_gaudi.models.qwen3_5 and "
                           "vllm_gaudi.models.qwen3_next must be imported before "
                           "vllm.models.qwen4_exp.nvidia.model")
    _upstream_model.Qwen4ExpPLELayer = HpuQwen4ExpPLELayer
    # The MTP draft's final mixer is built in nvidia/mtp.py through its own GatedResidual import.
    _upstream_mtp.GatedResidual = HpuGatedResidual


def register() -> None:
    apply_flash_next_patches()
    for arch in FLASH_NEXT_ARCHS:
        ModelRegistry.register_model(arch, f"vllm_gaudi.models.qwen4_exp:Hpu{arch}")
    for arch in MTP_ARCHS:
        ModelRegistry.register_model(arch, f"vllm_gaudi.models.qwen4_exp.mtp:Hpu{arch}")
