# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU registration wrapper for the Qwen4Exp MTP draft model (Flash-Next, M3-T2).

Upstream's ``Qwen4ExpMTP`` reuses the Qwen4Exp decoder layer, so the fork's GDN/MoE/QSA/PLE
and per-layer ``GatedResidual`` replacements (installed by ``apply_flash_next_patches()``
into ``vllm.models.qwen4_exp.nvidia.model``) already reach it. Its own final mixer is the
exception: ``nvidia/mtp.py`` binds ``GatedResidual`` through its own ``from .hyperconnection
import GatedResidual``, so ``apply_flash_next_patches()`` also rebinds
``vllm.models.qwen4_exp.nvidia.mtp.GatedResidual``. The subclass carries no code of its own;
it is the registration target and the class those module-name seams act on.
"""

import torch

import vllm.models.qwen4_exp.nvidia.mtp as _upstream_mtp


class HpuQwen4ExpMTP(_upstream_mtp.Qwen4ExpMTP):
    """Upstream's Qwen4ExpMTP; the HPU seams are the module names ``apply_flash_next_patches`` rebinds."""
    # The draft's sparse QSA decode gathers K/V through the per-request block_table (M2).
    requires_block_table = True


def mtp_aux_hidden(hidden: torch.Tensor, buffer: torch.Tensor | None) -> torch.Tensor:
    """View ``buffer[:T]`` with ``hidden``'s leading shape, for the drafter's aux-hidden channel.

    ``T = hidden.numel() // hidden.shape[-1]`` is the number of rows ``hidden`` carries
    (its leading dimensions flatten to ``T``); the buffer is the target's
    ``_mtp_hidden_buffer`` of ``[max_num_batched_tokens, hc_count * H]`` rows, so the first
    ``T`` rows are this forward's pre-final-mixer multi-stream state. Raises ``RuntimeError``
    when ``buffer`` is ``None`` (the target was built without ``speculative_config.method ==
    "mtp"``) or holds fewer than ``T`` rows.
    """
    if buffer is None:
        raise RuntimeError("MTP aux-hidden requested but the target has no _mtp_hidden_buffer "
                           "(it was built without speculative_config.method == 'mtp')")
    num_tokens = hidden.numel() // hidden.shape[-1]
    if buffer.shape[0] < num_tokens:
        raise RuntimeError(f"the MTP target buffer holds {buffer.shape[0]} rows, fewer than the "
                           f"{num_tokens} rows of the hidden states")
    return buffer[:num_tokens].view(*hidden.shape[:-1], -1)


__all__ = ["HpuQwen4ExpMTP", "mtp_aux_hidden"]
