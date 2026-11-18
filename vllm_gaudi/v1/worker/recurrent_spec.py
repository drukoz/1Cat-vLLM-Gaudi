# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the recurrent-state verification lattice (M3-T5a).

These helpers generalise the per-token state lattice that the HPU model
runner used to build for DFlash2 only to every speculative method (MTP in
particular). They are plain Python so they can be unit-tested on CPU; the
runner keeps only thin call sites.
"""

from __future__ import annotations

from typing import Any, Optional


def recurrent_spec_enabled(speculative_config: Any) -> bool:
    """True when speculation with >= 1 draft token is active (any method)."""
    return (speculative_config is not None
            and int(speculative_config.num_speculative_tokens or 0) > 0)


def state_slots_per_request(speculative_config: Any) -> int:
    """GDN state slots retained per request: ``1 + k`` under speculation, else 1."""
    if recurrent_spec_enabled(speculative_config):
        return 1 + int(speculative_config.num_speculative_tokens)
    return 1


def verification_state_width(num_tokens: int, enabled: bool) -> int:
    """State-lattice width of a decode batch: ``num_tokens`` for a verification
    batch, else the single-slot decode width 1."""
    return num_tokens if (enabled and num_tokens > 1) else 1


def accepted_counts(req_ids: list, counts_by_req: dict, padded_batch_size: int) -> list:
    """Per-request ``num_accepted_tokens`` for the decode batch.

    Unknown requests count as 1 (fresh request / post-prefill); the list is
    padded with 1 up to ``padded_batch_size``.
    """
    counts = [int(counts_by_req.get(req_id, 1)) for req_id in req_ids]
    counts.extend([1] * max(0, padded_batch_size - len(counts)))
    return counts


def load_checkpoint_offsets(counts: list, is_spec_batch: bool) -> Optional[list]:
    """Per-request load offsets into the state lattice.

    A verification batch carries the offsets in the lattice itself, so it
    returns ``None``; a plain decode loads the committed checkpoint written
    by the previous verification, i.e. ``count - 1``.
    """
    if is_spec_batch:
        return None
    return [count - 1 for count in counts]


def qsa_modules(*models: Any) -> list:
    """Every ``HpuQwen4ExpQSAAttention`` in the given modules, in argument
    order (target first, then drafter). ``None`` entries are skipped."""
    from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention

    found: list = []
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if isinstance(module, HpuQwen4ExpQSAAttention):
                found.append(module)
    return found
