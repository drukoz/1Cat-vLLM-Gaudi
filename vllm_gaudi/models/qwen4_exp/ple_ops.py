# SPDX-License-Identifier: Apache-2.0
"""Pure-torch Qwen4Exp PLE gate and short convolution (Flash-Next T4a).

Reproduces Hugging Face's ``Qwen4ExpTextPLELayer`` math (``transformers/models/qwen4_exp/
modeling_qwen4_exp.py:1179-1253``) with the formulas upstream's Triton kernels use, so the PLE layer can run on
HPU without them:

- gate: ``d = dot(RMSNorm(key), RMSNorm(query)) / sqrt(H)``, ``g = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))``,
  ``gated = g * value`` for every hyper-connection stream, then ``normed = RMSNorm(gated)``.
- short convolution: ``silu(sum_k weight[k] * history[t + k * dilation])`` over ``[state ; normed]``, added to
  ``gated``; the state carried to the next call is the last ``(kernel_size - 1) * dilation`` rows of
  ``[state ; normed]``.
- speculative short convolution (M3): ``ple_short_conv_spec_torch`` computes the T verification rows of an MTP step
  over the committed history window selected by ``num_accepted`` and stores ``[history[1:] ; candidates]``, the
  rollback convention of ``_hpu_causal_conv1d_spec_update``.

Every tensor is 2-D and token-major (``[T, ...]``): callers flatten the batch. The math is done in fp32 and the
outputs are cast back to the input dtype.
"""

import math

import torch
import torch.nn.functional as F


def grouped_rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """Grouped Gemma-style RMSNorm, the formula of ``Qwen4ExpTextRMSNorm`` with ``group_size``.

    ``x`` is ``[..., C]`` with ``C % group_size == 0``; each ``group_size``-sized group is normalized
    independently and scaled by ``(1 + weight)``.
    """
    input_dtype = x.dtype
    x_float = x.float()
    grouped = x_float.unflatten(-1, (-1, group_size))
    normalized = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(input_dtype)


def ple_gate_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate the PLE values with the stream activations.

    ``key`` and ``hidden`` are ``[T, hc * H]``, ``value`` is ``[T, H]`` and the three norm weights are ``[hc * H]``
    (grouped over ``H``). Returns ``(gated, normed)``, both ``[T, hc * H]``, with ``gated`` the gated values
    flattened over the streams and ``normed`` their grouped RMSNorm.
    """
    hidden_size = value.shape[-1]
    hc_count = hidden.shape[-1] // hidden_size

    key_normed = grouped_rmsnorm_torch(key, norm_key_w, hidden_size, eps).unflatten(-1, (hc_count, hidden_size))
    query_normed = grouped_rmsnorm_torch(hidden, norm_query_w, hidden_size, eps).unflatten(-1, (hc_count, hidden_size))
    gate = (key_normed.float() * query_normed.float()).sum(-1, keepdim=True) / math.sqrt(hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()

    gated_value = torch.sigmoid(gate) * value.float().unsqueeze(-2)
    gated = gated_value.flatten(-2)
    normed = grouped_rmsnorm_torch(gated, norm_conv_w, hidden_size, eps)
    return gated.to(key.dtype), normed.to(key.dtype)


def ple_short_conv_torch(
    normed: torch.Tensor,
    conv_weight: torch.Tensor,
    state: torch.Tensor,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise dilated short convolution over the history ``[state ; normed]``.

    ``normed`` is ``[T, C]``, ``conv_weight`` is ``conv1d.weight`` (``[C, 1, K]``, ``[C, K]`` is accepted too),
    ``state`` is the carried history ``[state_len, C]`` with ``state_len = (K - 1) * dilation`` (zeros for a fresh
    sequence). Returns ``(out, new_state)``: ``out`` is ``[T, C]`` with
    ``out[t] = silu(sum_k conv_weight[:, 0, k] * history[t + k * dilation])`` and ``new_state`` is the last
    ``state_len`` rows of ``[state ; normed]``.
    """
    channels = normed.shape[-1]
    kernel = conv_weight.reshape(channels, -1)
    kernel_size = kernel.shape[-1]
    state_len = (kernel_size - 1) * dilation
    if state.shape[0] != state_len:
        raise ValueError(f"state must hold (kernel_size - 1) * dilation = {state_len} rows, "
                         f"got {state.shape[0]}")

    history = torch.cat([state, normed], dim=0).float()
    num_tokens = normed.shape[0]
    accumulated = torch.zeros_like(normed, dtype=torch.float32)
    for k in range(kernel_size):
        accumulated = accumulated + history[k * dilation:k * dilation + num_tokens] * kernel[:, k]
    out = F.silu(accumulated).to(normed.dtype)
    new_state = history[-state_len:].to(state.dtype) if state_len > 0 else history[:0]
    return out, new_state


def ple_short_conv_batched_torch(
    normed: torch.Tensor,
    conv_weight: torch.Tensor,
    state: torch.Tensor,
    dilation: int,
    lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched :func:`ple_short_conv_torch` over padded rows (fixed shapes, graph-safe).

    ``normed`` is ``[B, S, C]``, ``conv_weight`` is ``conv1d.weight`` (``[C, 1, K]``, ``[C, K]`` is accepted
    too), ``state`` is the carried history ``[B, L, C]`` with ``L = (K - 1) * dilation`` and ``lengths`` is
    int64 ``[B]`` with the number of real tokens of every row (default ``S``). Returns ``(out, new_state)``,
    ``[B, S, C]`` and ``[B, L, C]``, with ``out[b, t] = silu(sum_k conv_weight[:, k] * history[b, t +
    k * dilation])`` over ``history[b] = [state[b] ; normed[b]]`` and ``new_state[b] = history[b, lengths[b]
    : lengths[b] + L]``. ``out`` at ``t >= lengths[b]`` is computed over padding and must be ignored by the
    caller. Math in fp32, outputs cast to ``normed.dtype`` / ``state.dtype``; one loop over the kernel taps,
    ``torch.gather`` for the state window, no per-row Python loop and no boolean indexing (HPU graphs).
    """
    channels = normed.shape[-1]
    kernel = conv_weight.reshape(channels, -1)
    kernel_size = kernel.shape[-1]
    state_len = (kernel_size - 1) * dilation
    rows, seq_len = normed.shape[0], normed.shape[1]
    if state.shape[0] != rows or state.shape[1] != state_len:
        raise ValueError(f"state must be [B, (kernel_size - 1) * dilation, C] = "
                         f"[{rows}, {state_len}, {channels}], got {tuple(state.shape)}")
    if lengths is None:
        lengths = torch.full((rows,), seq_len, dtype=torch.long, device=normed.device)
    lengths = lengths.reshape(-1).long()

    history = torch.cat([state.float(), normed.float()], dim=1)
    accumulated = torch.zeros((rows, seq_len, channels), dtype=torch.float32, device=normed.device)
    for k in range(kernel_size):
        tap = kernel[:, k].to(torch.float32).view(1, 1, channels)
        accumulated = accumulated + history[:, k * dilation:k * dilation + seq_len] * tap
    out = F.silu(accumulated).to(normed.dtype)

    gather_index = lengths.view(rows, 1, 1) + torch.arange(state_len, device=normed.device).view(1, state_len, 1)
    gather_index = gather_index.expand(rows, state_len, channels)
    new_state = torch.gather(history, 1, gather_index).to(state.dtype)
    return out, new_state


def ple_short_conv_spec_torch(
    normed: torch.Tensor,
    conv_weight: torch.Tensor,
    state: torch.Tensor,
    num_accepted: torch.Tensor,
    query_lens: torch.Tensor,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Speculative (MTP verification) short conv with per-request rollback (M3, T6).

    ``normed`` is ``[B, T, C]``; ``state`` is ``[B, S, C]`` with ``S >= L + T - 1`` and
    ``L = (kernel_size - 1) * dilation``; ``num_accepted`` is int ``[B]`` (1..T_prev, the rows
    the previous step committed) and ``query_lens`` is int ``[B]`` (0..T, the real token count
    of every padded row). With ``a = clamp(num_accepted, 1, S - L + 1) - 1`` the committed
    window is ``history = state[b, a : a + L]`` (gather) and ``x`` is ``normed`` masked to zero
    at ``t >= query_lens[b]``. Over ``seq = cat(history, x)`` (``[B, L + T, C]`` the output is
    ``out[b, t] = silu(sum_k conv_weight[:, k] * seq[b, t + k * dilation])`` for every ``t < T``
    (padding rows computed, then masked to zero); ``new_state[b] = cat(history[b, 1:], x[b])``
    padded with zeros to ``S`` rows, so the next step reads its window at ``num_accepted - 1``
    and rejected candidates leave no trace. Rows with ``query_lens == 0`` keep ``state[b]``.
    Returns ``(out [B, T, C] in normed.dtype, new_state [B, S, C] in state.dtype)``; fp32 math,
    no Python loop over B.
    """
    channels = normed.shape[-1]
    kernel = conv_weight.reshape(channels, -1)
    kernel_size = kernel.shape[-1]
    state_len = (kernel_size - 1) * dilation
    rows, width = normed.shape[0], normed.shape[1]
    capacity = state.shape[1]
    if state.shape[0] != rows:
        raise ValueError(f"state must be [B, S, C] = [{rows}, S, {channels}], got {tuple(state.shape)}")
    if capacity < state_len + width - 1:
        raise ValueError(f"state must hold at least (kernel_size - 1) * dilation + T - 1 = "
                         f"{state_len + width - 1} rows, got {capacity}")

    device = normed.device
    accepted = num_accepted.reshape(-1).long().clamp(min=1, max=capacity - state_len + 1)
    window = (accepted - 1).view(rows, 1) + torch.arange(state_len, device=device).view(1, state_len)
    history = torch.gather(state.float(), 1, window.unsqueeze(-1).expand(rows, state_len, channels))

    lengths = query_lens.reshape(-1).long().clamp(min=0, max=width)
    token_mask = torch.arange(width, device=device).view(1, width) < lengths.view(rows, 1)
    x = normed.float() * token_mask.unsqueeze(-1)
    sequence = torch.cat([history, x], dim=1)

    accumulated = torch.zeros((rows, width, channels), dtype=torch.float32, device=device)
    for k in range(kernel_size):
        tap = kernel[:, k].to(torch.float32).view(1, 1, channels)
        accumulated = accumulated + sequence[:, k * dilation:k * dilation + width] * tap
    out = F.silu(accumulated).to(normed.dtype)
    out = out * token_mask.unsqueeze(-1).to(out.dtype)

    new_state = torch.cat([history[:, 1:], x], dim=1)
    if new_state.shape[1] < capacity:
        new_state = F.pad(new_state, (0, 0, 0, capacity - new_state.shape[1]))
    elif new_state.shape[1] > capacity:
        new_state = new_state[:, :capacity]
    new_state = torch.where((lengths > 0).view(rows, 1, 1), new_state, state.float())
    return out, new_state.to(state.dtype)


def ple_forward_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
    conv_weight: torch.Tensor,
    state: torch.Tensor,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whole PLE layer body: gate the values, add the short convolution, carry the conv state.

    ``key`` is ``key_proj(embeddings)`` (the n-gram embedding lookup is the caller's), ``hidden`` is the
    ``[T, hc * H]`` stream activation, ``state`` the ``[state_len, C]`` conv history of this sequence. Returns
    ``(output, new_state)`` with ``output = gated + short_conv(normed)`` of shape ``[T, hc * H]``.
    """
    gated, normed = ple_gate_torch(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps)
    conv_out, new_state = ple_short_conv_torch(normed, conv_weight, state, dilation)
    return gated + conv_out, new_state
