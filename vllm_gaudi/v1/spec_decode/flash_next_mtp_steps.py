# SPDX-License-Identifier: Apache-2.0
"""Pure CPU helpers for the Flash-Next MTP drafter (M3).

``HpuFlashNextMtpProposer`` runs the ``Qwen4ExpMTP`` draft on HPU, so it
cannot be imported (let alone tested) on a machine without one. Every piece
of step logic that needs no HPU lives here: the draft-model detection, the
``(sample, multi)`` split of the draft's output, the per-step hidden state
and the QSA ring slots (the GDN state lattice's column 0). The per-step
kernel ``block_table`` reuses ``flash_next_inputs.build_block_table`` on the
defragmenter-resolved rows, exactly as the target's decode does.

``hpu_flash_next_mtp.py`` stays a thin copy of ``HpuEagleProposer.propose``
that calls these helpers. Imports stay torch/os/vllm-logger only,
so this module is unit-testable on CPU (rule from the plan's conventions).
``DraftRankStats`` (M3b, ``VLLM_FLASH_NEXT_DRAFT_RANK_LOG=1``) histograms the
target token's rank among the drafter's top-4 at the first rejected position.
"""

import os

import torch
from vllm.logger import init_logger

FLASH_NEXT_MTP_ARCH = "Qwen4ExpMTP"

logger = init_logger(__name__)


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


DRAFT_RANK_LOG_ENV = "VLLM_FLASH_NEXT_DRAFT_RANK_LOG"
DRAFT_RANK_TOP = 4


def draft_rank_log_enabled() -> bool:
    """VLLM_FLASH_NEXT_DRAFT_RANK_LOG is "1" or "true" (case-insensitive, stripped)."""
    return os.environ.get(DRAFT_RANK_LOG_ENV, "0").strip().lower() in ("1", "true")


class DraftRankStats:
    """Histogram of the target token's rank among the drafter's top-4 candidates.

    The runner remembers the drafter's top-4 rows by request id right after the
    decode propose (``remember``), then after the next step's sampling hands
    over the batch's accepted streams (``observe``): for a request with ``k``
    drafts and ``L = len(ids)`` committed tokens (accepted drafts + the target's
    bonus), ``L - 1 < k`` means the draft at index ``j = L - 1`` was rejected
    and ``ids[-1]`` is the target's own token there; ``L == k + 1`` means every
    draft was accepted. Requests without a remembered row (prefill-only drafts,
    first steps) are ignored. Under greedy sampling ``rank1`` stays 0: a target
    token equal to the top-1 draft would have been accepted.
    """

    def __init__(self, log_every: int = 200) -> None:
        self.counts = [0] * (DRAFT_RANK_TOP + 1)  # counts[r]: rank r+1; counts[4]: beyond top-4
        self.all_accepted = 0
        self.log_every = log_every
        self.observations = 0
        self._rows: dict[str, torch.Tensor] = {}

    def remember(self, req_ids: list[str], top_ids: torch.Tensor) -> None:
        """Store top_ids row i (a [k, DRAFT_RANK_TOP] top-4 row, any device) under req_ids[i]."""
        if len(req_ids) != top_ids.shape[0]:
            raise ValueError(f"remember: {len(req_ids)} req_ids for {top_ids.shape[0]} top-4 rows")
        top_ids = top_ids.cpu()
        for i, req_id in enumerate(req_ids):
            self._rows[req_id] = top_ids[i]

    def observe(self, req_ids: list[str], sampled: list[list[int]]) -> None:
        """Count the target token's rank at the first rejected position of each request."""
        for req_id, ids in zip(req_ids, sampled):
            row = self._rows.pop(req_id, None)
            if row is None or not ids:
                continue
            k = row.shape[0]
            length = len(ids)
            if length - 1 < k:
                target = ids[-1]
                rank = DRAFT_RANK_TOP
                for position, token in enumerate(row[length - 1].tolist()):
                    if token == target:
                        rank = position
                        break
                self.counts[rank] += 1
            else:
                self.all_accepted += 1
            self.observations += 1
            if self.observations % self.log_every == 0:
                logger.info(self.summary_line())

    def summary_line(self) -> str:
        return ("Flash-Next draft rank at the first rejected position: "
                f"rank1={self.counts[0]} rank2={self.counts[1]} rank3={self.counts[2]} "
                f"rank4={self.counts[3]} beyond4={self.counts[4]} all_accepted={self.all_accepted}")

