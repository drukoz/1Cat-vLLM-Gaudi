# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from vllm.v1.sample import rejection_sampler
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def _dflash2_greedy_fastpath_supported(sampling_metadata: SamplingMetadata, ) -> bool:
    """Return whether argmax can bypass the general sampler losslessly."""
    if not sampling_metadata.all_greedy:
        return False
    if sampling_metadata.max_num_logprobs is not None or sampling_metadata.logprob_token_ids:
        return False
    if not sampling_metadata.no_penalties:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None or sampling_metadata.bad_words_token_ids:
        return False

    # Spec decode currently installs only MinTokens as an argmax-changing
    # processor. It is a true no-op once its active request map is empty.
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        if not isinstance(processor, MinTokensLogitsProcessor) or processor.min_toks:
            return False

    holder = sampling_metadata.thinking_budget_state_holder
    return holder is None or not holder.has_tracked_requests()


def dflash2_greedy_rejection_sample(
    logits: torch.Tensor,
    metadata: SpecDecodeMetadata,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor | None:
    """Fuse DFlash2's two greedy sampler passes into one vocabulary argmax.

    The general rejection sampler separately gathers/casts/samples bonus and
    target rows. For the common unconstrained greedy request, casting cannot
    change the argmax, so one argmax over the already-packed verification
    logits is exactly equivalent and avoids materializing FP32 vocabulary
    tensors twice.
    """
    if not _dflash2_greedy_fastpath_supported(sampling_metadata):
        return None

    return dflash2_greedy_rejection_sample_packed(
        logits,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.cu_num_draft_tokens,
    )


def dflash2_greedy_rejection_sample_packed(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Tensor-only DFlash2 greedy sampler suitable for regional compile."""
    all_token_ids = logits.argmax(dim=-1).to(torch.int32)
    target_token_ids = torch.index_select(all_token_ids, 0, target_logits_indices)
    bonus_token_ids = torch.index_select(all_token_ids, 0, bonus_logits_indices).view(-1, 1)
    return _rejection_sample_pytorch(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        bonus_logits_indices.shape[0],
        cu_num_draft_tokens,
    )


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Performs vectorized rejection sampling on a batch of token sequences.

    This function compares draft tokens to target tokens and accepts them up to 
    the first mismatch. If an entire sequence of draft tokens is accepted, a 
    bonus token is appended. This version handles variable numbers of draft 
    tokens per sequence.

    The current HPU implementation of spec decode will flatten the num_draft_tokens
    to 1. And so the shape size of padded_draft_token_ids will be
    the [real batch size * num_draft_tokens, 1].

    Args:
        padded_draft_token_ids (torch.Tensor): A 2D tensor of draft tokens.
            Shape: (num_seqs * max_draft_tokens, 1)
        padded_target_token_ids (torch.Tensor): A 1D tensor of target tokens
            predicted by the main model.
            Shape: (num_seqs * max_draft_tokens)
        bonus_token_ids (torch.Tensor): A single bonus token for each sequence,
            to be used if all draft tokens are accepted.
            Shape: (num_seqs, 1)
        num_draft_tokens: list[int]: List of number draft tokens for each sequence.
            Shape: (num_seqs)
        cu_num_draft_tokens (torch.Tensor): The cumulative sum of the number of
            draft tokens for each request. Used to determine actual sequence 
            lengths. Shape: (num_seqs,)

    Returns:
        torch.Tensor: The resulting tensor of accepted tokens.
            Shape: (num_seqs, max_draft_tokens + 1)
    """
    return _rejection_sample_pytorch(
        padded_draft_token_ids,
        padded_target_token_ids,
        bonus_token_ids,
        len(num_draft_tokens),
        cu_num_draft_tokens,
    )


def _rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    num_seqs: int,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    # Keep the prefix comparison and output packing on the input device.  The
    # caller already performs one final D2H copy when parsing engine output;
    # copying all draft/target rows here adds an extra synchronization to every
    # speculative step and is especially expensive for short decode batches.
    padded_draft_token_ids = padded_draft_token_ids.to(torch.int32)
    padded_target_token_ids = padded_target_token_ids.to(torch.int32)
    bonus_token_ids = bonus_token_ids.to(torch.int32)
    padded_draft_token_ids = padded_draft_token_ids.view(num_seqs, -1)
    max_draft_tokens = padded_draft_token_ids.shape[-1]
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device

    # Calculate the number of draft tokens for each request without moving the
    # cumulative lengths to the host.
    cu_num_draft_tokens = cu_num_draft_tokens.to(device=device)
    start_indices = torch.cat((torch.zeros(1, device=device,
                                           dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # Find the first mismatch while ignoring padded draft positions.
    pos = torch.arange(max_draft_tokens, device=device, dtype=num_draft_tokens_per_seq.dtype)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)
    mismatches = (padded_draft_token_ids != padded_target_token_ids) & valid_token_mask
    any_mismatch = mismatches.any(dim=1)
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    prefix_len = torch.where(any_mismatch, first_mismatch_idx + 1, num_draft_tokens_per_seq)
    prefix_mask = (pos < prefix_len.unsqueeze(-1)) & valid_token_mask
    placeholder = torch.full_like(padded_target_token_ids, PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    output_prefix = torch.where(prefix_mask, padded_target_token_ids, placeholder)

    output_tokens = torch.cat(
        (output_prefix, placeholder[:, :1]),
        dim=1,
    )
    output_pos = torch.arange(max_draft_tokens + 1, device=device, dtype=num_draft_tokens_per_seq.dtype)
    bonus_mask = (~any_mismatch).unsqueeze(-1) & (output_pos.unsqueeze(0) == num_draft_tokens_per_seq.unsqueeze(-1))
    return torch.where(bonus_mask, bonus_token_ids[:, :1], output_tokens)


SAMPLING_EPS = 1e-5  # vLLM's _SAMPLING_EPS: a temperature below it is greedy (the HPU batch stores -1.0)


def sampled_verify_enabled() -> bool:
    """VLLM_HPU_SPEC_SAMPLED_VERIFY=1: verify the drafts of sampled requests (temperature > 0) on the HPU."""
    from vllm_gaudi import envs as gaudi_envs

    return int(gaudi_envs.VLLM_HPU_SPEC_SAMPLED_VERIFY) == 1


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Upstream's ``expand_batch_to_tokens`` (a Triton kernel) in torch, for the HPU's padded draft layout:
    ``[batch] -> [num_tokens]`` with request i's value on rows ``i * k .. i * k + k - 1``, ``k = num_tokens //
    batch``, ``replace_from`` becoming ``replace_to``.

    The HPU runner lays every request's draft rows out padded to the step's ``k`` (``target_logits_indices`` holds
    ``-1`` for a request's missing drafts), where upstream packs them by ``cu_num_tokens``; with uneven draft
    counts the packed rule would give a request the temperature, top-k and top-p of the next. Padded rows get
    their request's value and are never read. One ``index_select``: no host sync, no advanced indexing.
    """
    batch_size = x.shape[0]
    if batch_size == 0 or num_tokens % batch_size:
        raise ValueError(f"HPU draft rows are padded per request: {num_tokens} rows for {batch_size} requests")
    request = torch.arange(num_tokens, device=x.device, dtype=torch.int32) // (num_tokens // batch_size)
    expanded = torch.index_select(x, 0, request)
    return torch.where(expanded == replace_from, torch.full_like(expanded, replace_to), expanded)


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Upstream's ``apply_sampling_constraints`` with the torch ``expand_batch_to_tokens`` and the HPU batch's
    greedy temperature: a greedy row (temperature below SAMPLING_EPS; the HPU batch stores -1.0 where upstream's
    replace value is 0) is divided by 1, so its argmax stays the raw logits' argmax."""
    if sampling_metadata.all_greedy:
        return logits
    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(sampling_metadata.temperature, cu_num_draft_tokens, num_tokens)
    temperature = torch.where(temperature < SAMPLING_EPS, torch.ones_like(temperature), temperature)
    logits.div_(temperature.to(logits.dtype).unsqueeze(-1))
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(sampling_metadata.top_k, cu_num_draft_tokens, num_tokens)
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(sampling_metadata.top_p, cu_num_draft_tokens, num_tokens)
    return rejection_sampler.apply_top_k_top_p(logits, top_k, top_p)


def _uniform(num_tokens: int, max_draft: int, num_draft_tokens: list[int], generators: dict,
             device: torch.device) -> torch.Tensor:
    """fp32 uniforms in (0, 1] (1 - rand is never 0, so p(d) >= u accepts with probability exactly p(d) without
    upstream's float64); a seeded request's own rows from its generator, as upstream."""
    uniform = 1.0 - torch.rand((num_tokens, ), dtype=torch.float32, device=device)
    for req_idx, generator in generators.items():
        n = num_draft_tokens[req_idx] if req_idx < len(num_draft_tokens) else 0
        if n > 0:
            start = req_idx * max_draft
            uniform[start:start + n] = 1.0 - torch.rand((n, ), dtype=torch.float32, device=device,
                                                        generator=generator)
    return uniform


def sampled_rejection_sample(
    draft_token_ids: torch.Tensor,  # [batch * max_draft] (or [.., 1]): the HPU's padded layout
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,  # [batch * max_draft, vocab], after apply_sampling_constraints
    bonus_token_ids: torch.Tensor,  # [batch, 1]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Speculative sampling of argmax drafts (MTP: no draft probabilities), exact in distribution.

    Upstream's ``NO_DRAFT_PROBS`` path in torch: draft ``d`` at a position is accepted with probability ``p(d)``
    of the target distribution there (temperature, top-k and top-p applied); at the first rejection the output is
    drawn from ``p`` with ``d`` removed (an exponential race, ``argmax p(v) / q(v)`` with ``q ~ Exp(1)`` per
    request); with every draft accepted the bonus token (the caller samples it from the bonus row) follows.
    Greedy rows of a mixed batch keep the greedy rule: accept while the draft is the target's argmax, then the
    argmax. Output ``[batch, max_draft + 1]`` int32, PLACEHOLDER_TOKEN_ID after the last token.

    The HPU lays request i's drafts out at rows ``i * max_draft .. i * max_draft + n_i - 1`` (padded rows hold
    any id and are never read). Fixed-shape tensor ops only: no host sync, no advanced indexing, int32 ids.
    Every TP rank draws the same randoms from the same seed, as upstream's sampler does.
    """
    batch_size = len(num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape
    device = target_logits.device
    if batch_size == 0 or num_tokens == 0:
        return torch.full((batch_size, 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device)
    max_draft = num_tokens // batch_size
    assert max_draft * batch_size == num_tokens, (num_tokens, batch_size)
    draft = draft_token_ids.reshape(batch_size, max_draft).to(torch.int32)
    cumulative = cu_num_draft_tokens.to(device=device, dtype=torch.int32)
    n_draft = cumulative - torch.cat((torch.zeros(1, dtype=torch.int32, device=device), cumulative[:-1]))
    pos = torch.arange(max_draft, device=device, dtype=torch.int32).unsqueeze(0)
    valid = pos < n_draft.unsqueeze(1)
    draft_ids = draft.reshape(-1).clamp(0, vocab_size - 1)

    probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    flat = torch.arange(num_tokens, device=device, dtype=torch.int32) * vocab_size + draft_ids
    p_draft = torch.index_select(probs.reshape(-1), 0, flat).view(batch_size, max_draft)
    uniform = _uniform(num_tokens, max_draft, num_draft_tokens, sampling_metadata.generators, device)
    accepted_sampled = p_draft >= uniform.view(batch_size, max_draft)

    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    for req_idx, generator in sampling_metadata.generators.items():
        if req_idx < batch_size and num_draft_tokens[req_idx] > 0:
            q[req_idx].exponential_(generator=generator)
    request = torch.arange(num_tokens, device=device, dtype=torch.int32) // max_draft
    scores = probs / torch.index_select(q, 0, request)
    vocab = torch.arange(vocab_size, device=device, dtype=torch.int32)
    scores = torch.where(vocab.unsqueeze(0) == draft_ids.unsqueeze(1), torch.zeros_like(scores), scores)
    recovered = scores.argmax(dim=-1).to(torch.int32).view(batch_size, max_draft)

    target_argmax = target_logits.argmax(dim=-1).to(torch.int32).view(batch_size, max_draft)
    if sampling_metadata.all_random or sampling_metadata.temperature is None:
        greedy = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    else:
        greedy = (sampling_metadata.temperature.to(device) < SAMPLING_EPS).view(batch_size, 1)
    accepted = torch.where(greedy, draft == target_argmax, accepted_sampled)
    replacement = torch.where(greedy, target_argmax, recovered)

    rejected = (~accepted) & valid
    any_rejected = rejected.any(dim=1, keepdim=True)
    first = torch.argmax(rejected.to(torch.int32), dim=1, keepdim=True).to(torch.int32)
    placeholder = torch.full_like(draft, PLACEHOLDER_TOKEN_ID)
    kept = valid & ((pos < first) | ~any_rejected)
    tokens = torch.where(kept, draft, placeholder)
    tokens = torch.where(any_rejected & (pos == first), replacement, tokens)
    out = torch.cat((tokens, placeholder[:, :1]), dim=1)
    out_pos = torch.arange(max_draft + 1, device=device, dtype=torch.int32).unsqueeze(0)
    bonus_here = (~any_rejected) & (out_pos == n_draft.unsqueeze(1))
    return torch.where(bonus_here, bonus_token_ids.reshape(batch_size, 1).to(torch.int32), out)


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    # `target_probs` is upstream's `target_logits` (after apply_sampling_constraints). `use_fp64_gumbel` is
    # accepted for signature parity: the sampled path draws fp32 uniforms in (0, 1].
    if not sampling_metadata.all_greedy:
        if not sampled_verify_enabled():
            raise ValueError("Speculative decoding on HPU verifies greedy requests only (temperature 0); set "
                             "VLLM_HPU_SPEC_SAMPLED_VERIFY=1 to verify sampled requests")
        if draft_probs is not None:
            raise NotImplementedError("HPU sampled verification takes argmax drafts (no draft probabilities)")
        return sampled_rejection_sample(draft_token_ids, num_draft_tokens, cu_num_draft_tokens, target_probs,
                                        bonus_token_ids, sampling_metadata)

    # Rejection sampling for greedy sampling requests.

    target_argmax = target_probs.argmax(dim=-1)
    output_token_ids = rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                                cu_num_draft_tokens)
    return output_token_ids


rejection_sampler.rejection_sample = rejection_sample
# The sampled path's logits processing: upstream's expand_batch_to_tokens is a Triton kernel, and upstream's
# greedy temperature (0) is not the HPU batch's (-1.0).
rejection_sampler.expand_batch_to_tokens = expand_batch_to_tokens
rejection_sampler.apply_sampling_constraints = apply_sampling_constraints
