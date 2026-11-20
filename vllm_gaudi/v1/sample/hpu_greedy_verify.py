# SPDX-License-Identifier: Apache-2.0
"""Greedy verification of a speculative step from the target's greedy token of every verification row.

Greedy verification needs one thing from the target's logits: the arg max of every row. The general path
gathers the logits of every verification row on every rank (``[32 x (1 + k), 248,320]`` bf16 for Flash-Next at
32 streams) and runs the rejection sampler over them. With ``VLLM_HPU_VOCAB_PARALLEL_LOGITS`` the HPU runner gets
the arg max of every row from ``vllm_gaudi.distributed.vocab_parallel.top_tokens`` instead (no logits gather)
and verifies the drafts here, with the same prefix comparison the HPU rejection sampler and the DFlash2 greedy
path use (``_rejection_sample_pytorch``). This module is separate on purpose: ``hpu_rejection_sampler.py`` is
not changed by the collective cuts.
"""

import torch


def greedy_rejection_sample_from_tokens(
    all_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """The verified tokens ``[requests, max_drafts + 1]`` (placeholders past the accepted prefix).

    ``all_token_ids`` is the greedy token of every row the logits would have had (int32 ``[rows]``, the order
    of ``logits_indices``); the other arguments are the fields of ``SpecDecodeMetadata`` with the same names.
    The same result as ``dflash2_greedy_rejection_sample_packed(logits, ...)`` when
    ``all_token_ids == logits.argmax(-1)``.
    """
    from vllm_gaudi.v1.sample.hpu_rejection_sampler import _rejection_sample_pytorch

    target_token_ids = torch.index_select(all_token_ids, 0, target_logits_indices)
    bonus_token_ids = torch.index_select(all_token_ids, 0, bonus_logits_indices).view(-1, 1)
    return _rejection_sample_pytorch(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        bonus_logits_indices.shape[0],
        cu_num_draft_tokens,
    )
