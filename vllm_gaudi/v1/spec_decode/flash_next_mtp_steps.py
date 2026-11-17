# SPDX-License-Identifier: Apache-2.0
"""Pure CPU helpers for the Flash-Next MTP drafter (M3).

``HpuFlashNextMtpProposer`` runs the ``Qwen4ExpMTP`` draft on HPU, so it
cannot be imported (let alone tested) on a machine without one. Every piece
of step logic that needs no HPU lives here: the draft-model detection, the
``(sample, multi)`` split of the draft's output, the per-step hidden state,
the QSA ring slots (the GDN state lattice's column 0) and the per-step
kernel ``block_table`` built from the CPU block tables.

``hpu_flash_next_mtp.py`` stays a thin copy of ``HpuEagleProposer.propose``
that calls these helpers. Imports stay torch + ``flash_next_inputs`` only,
so this module is unit-testable on CPU (rule from the plan's conventions).
"""

import torch

from vllm_gaudi.v1.worker.flash_next_inputs import build_block_table

FLASH_NEXT_MTP_ARCH = "Qwen4ExpMTP"


def is_flash_next_mtp(speculative_config) -> bool:
    """method == "mtp" and "Qwen4ExpMTP" in draft_model_config.hf_config.architectures."""
    if getattr(speculative_config, "method", None) != "mtp":
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    return FLASH_NEXT_MTP_ARCH in architectures


def split_draft_output(ret) -> tuple[torch.Tensor, torch.Tensor]:
    """(sample, multi) from the draft's return.

    TypeError unless ret is a 2-tuple with multi.shape[-1] % sample.shape[-1] == 0.
    """
    if not isinstance(ret, tuple) or len(ret) != 2:
        raise TypeError("the Flash-Next MTP draft must return a (sample, multi) 2-tuple, "
                        f"got {type(ret).__name__}"
                        + (f" of length {len(ret)}" if isinstance(ret, tuple) else ""))
    sample, multi = ret
    if multi.shape[-1] % sample.shape[-1] != 0:
        raise TypeError(f"draft multi stream {tuple(multi.shape)} is not a whole multiple of the "
                        f"sample hidden size {sample.shape[-1]}")
    return sample, multi


def ring_slots_from_lattice(indices: torch.Tensor) -> torch.Tensor:
    """[groups, bs, T] -> [groups, bs] (column 0); [groups, bs] unchanged; ValueError otherwise."""
    if indices.dim() == 3:
        return indices[:, :, 0]
    if indices.dim() == 2:
        return indices
    raise ValueError(f"expected a [groups, bs, T] state lattice or [groups, bs] ring slots, "
                     f"got shape {tuple(indices.shape)}")


def next_step_hidden(multi: torch.Tensor, last_token_indices: torch.Tensor) -> torch.Tensor:
    """multi viewed as [vbs, hc*H], rows last_token_indices -> [bs, 1, hc*H]."""
    hidden_size = multi.shape[-1]
    rows = multi.view(-1, hidden_size)[last_token_indices]
    return rows.view(-1, 1, hidden_size)


def draft_block_table(block_table_cpu_tensor: torch.Tensor, num_blocks: list[int], batch_size: int,
                      max_blocks: int, pad_block_id: int) -> torch.Tensor:
    """Per-request kernel block ids for one draft step: row i = block_table_cpu_tensor[i, :num_blocks[i]] via
    build_block_table, padded to batch_size rows."""
    rows = [block_table_cpu_tensor[i, :n].tolist() for i, n in enumerate(num_blocks)]
    return build_block_table(rows, batch_size, max_blocks, pad_block_id)
