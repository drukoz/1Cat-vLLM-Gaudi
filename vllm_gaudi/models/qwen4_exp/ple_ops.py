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
- int32-only n-gram hash (RUK-3116): ``ple_ngram_ids_int32`` reproduces the engine's int64 n-gram hash
  (``vllm/models/qwen4_exp/nvidia/ple_layer.py:426-440``) with 5-bit limb arithmetic on int32 tensors only, so it runs
  on the Gaudi lazy bridge, which narrows every int64 device tensor to int32.
- n-gram ids on the host (``VLLM_HPU_PLE_HOST_NGRAM_IDS=1``): ``host_ngram_ids`` computes the same ids in numpy
  from the batch's token ids where the runner builds the batch, for every PLE layer at once
  (``collect_ngram_host_tables``), so the id computation leaves the HPU graphs.

Every tensor is 2-D and token-major (``[T, ...]``): callers flatten the batch. The math is done in fp32 and the
outputs are cast back to the input dtype.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
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


NGRAM_HASH_LIMB_BITS = 5
NGRAM_HASH_LIMBS = 13  # 13 * 5 = 65 bits cover every product below 2**63
NGRAM_HASH_MAX_VOCAB = 1 << 25
NGRAM_HASH_MAX_SIZE = 1 << 25

_NGRAM_HASH_LIMB_MASK = (1 << NGRAM_HASH_LIMB_BITS) - 1
_NGRAM_HASH_MAX_VALUE = 1 << (NGRAM_HASH_LIMB_BITS * NGRAM_HASH_LIMBS)


def ngram_hash_limbs(value: int) -> list[int]:
    """The 13 five-bit limbs of a non-negative int below 2**65, least significant first."""
    if value < 0 or value >= _NGRAM_HASH_MAX_VALUE:
        raise ValueError(f"value {value} does not fit in {NGRAM_HASH_LIMBS} limbs of {NGRAM_HASH_LIMB_BITS} bits")
    return [(value >> (NGRAM_HASH_LIMB_BITS * j)) & _NGRAM_HASH_LIMB_MASK for j in range(NGRAM_HASH_LIMBS)]


def ngram_mod_weights(sizes: Sequence[int]) -> list[list[int]]:
    """``[[pow(2, 5 * j, p) for j in range(13)] for p in sizes]``."""
    return [[pow(2, NGRAM_HASH_LIMB_BITS * j, p) for j in range(NGRAM_HASH_LIMBS)] for p in sizes]


def check_ngram_hash_bounds(multipliers: Sequence[int], sizes: Sequence[int], offsets: Sequence[int],
                            vocab_size: int) -> None:
    """Raise ValueError unless the int32 hash is exact for these tables."""
    if len(sizes) != len(offsets):
        raise ValueError(f"sizes has {len(sizes)} entries, offsets has {len(offsets)}")
    if vocab_size > NGRAM_HASH_MAX_VOCAB:
        raise ValueError(f"vocab_size {vocab_size} exceeds NGRAM_HASH_MAX_VOCAB {NGRAM_HASH_MAX_VOCAB}")
    for index, size in enumerate(sizes):
        if size < 1 or size > NGRAM_HASH_MAX_SIZE:
            raise ValueError(f"sizes[{index}] = {size} is outside [1, {NGRAM_HASH_MAX_SIZE}]")
    for index, offset in enumerate(offsets):
        if offset < 0:
            raise ValueError(f"offsets[{index}] = {offset} is negative")
        if offset + sizes[index] > (1 << 31) - 1:
            raise ValueError(f"offsets[{index}] = {offset} plus sizes[{index}] = {sizes[index]} exceeds 2**31 - 1")
    for index, multiplier in enumerate(multipliers):
        if multiplier < 1:
            raise ValueError(f"multipliers[{index}] = {multiplier} is below 1")
        if multiplier >= _NGRAM_HASH_MAX_VALUE:
            raise ValueError(f"multipliers[{index}] = {multiplier} does not fit in {NGRAM_HASH_LIMBS} limbs")
        if multiplier * (vocab_size - 1) >= 1 << 63:
            raise ValueError(f"multipliers[{index}] = {multiplier} times vocab_size - 1 = {vocab_size - 1} "
                             f"reaches 2**63")


def ple_ngram_ids_int32(tokens: Sequence[torch.Tensor], multiplier_limbs: torch.Tensor, mod_weights: torch.Tensor,
                        sizes: torch.Tensor, offsets: torch.Tensor, heads_per_ngram: int) -> torch.Tensor:
    """N-gram ids ``[N, (len(tokens) - 1) * heads_per_ngram]`` int32, equal to the engine's int64 hash.

    ``tokens[i]`` is the int32 ``[N]`` tensor of the token ``i`` positions back (``len(tokens) == ngram_size``);
    ``multiplier_limbs`` is int32 ``[ngram_size, 13]`` with row ``i`` = ``ngram_hash_limbs(multiplier[i])``,
    ``mod_weights`` int32 ``[H, 13]`` (``ngram_mod_weights(sizes)``), ``sizes`` and ``offsets`` int32 ``[H]`` with
    ``H = (ngram_size - 1) * heads_per_ngram``, the heads ordered as in the engine (bigram heads first). The product
    limbs of ``tokens[i]`` with row ``i`` are computed once and XORed cumulatively; every intermediate is int32, so
    the whole op runs on the Gaudi lazy bridge. Callers check the tables with :func:`check_ngram_hash_bounds`.
    """
    mult = multiplier_limbs.to(torch.int32)
    weights = mod_weights.to(torch.int32)
    head_sizes = sizes.to(torch.int32)
    head_offsets = offsets.to(torch.int32)

    blocks: list[torch.Tensor] = []
    mixed: torch.Tensor | None = None
    for index, token in enumerate(tokens):
        tok = token.to(torch.int32)
        carry = torch.zeros_like(tok)
        limb_rows: list[torch.Tensor] = []
        for limb in range(NGRAM_HASH_LIMBS):
            acc = tok * mult[index, limb] + carry
            limb_rows.append(torch.bitwise_and(acc, _NGRAM_HASH_LIMB_MASK))
            carry = torch.bitwise_right_shift(acc, NGRAM_HASH_LIMB_BITS)
        product = torch.stack(limb_rows, dim=-1)
        mixed = product if mixed is None else torch.bitwise_xor(mixed, product)
        if index == 0:
            continue
        start = (index - 1) * heads_per_ngram
        end = start + heads_per_ngram
        size = head_sizes[start:end]
        terms = torch.remainder(mixed[:, None, :] * weights[None, start:end, :], size[None, :, None])
        ids = torch.remainder(terms.sum(-1, dtype=torch.int32), size[None, :])
        blocks.append(ids + head_offsets[start:end][None, :])
    return torch.cat(blocks, dim=-1)


# The model input that carries the host-computed n-gram ids (VLLM_HPU_PLE_HOST_NGRAM_IDS=1): the runner sends it,
# ``_MtpTargetOutputs.forward`` hands it to the PLE layers.
HOST_NGRAM_IDS_KWARG = "ple_ngram_ids"


@dataclass(frozen=True)
class NgramHostTables:
    """The n-gram hash tables of every PLE layer of one model, on the host, for :func:`host_ngram_ids`.

    Row ``i`` of the three int64 arrays is the PLE layer with ``ple_dense_layer_id == i``: ``multipliers``
    ``[layers, ngram_size]``, ``sizes`` and ``offsets`` ``[layers, (ngram_size - 1) * heads_per_ngram]``, the heads
    ordered as in the engine (bigram heads first). The values are the ones ``HpuQwen4ExpNGramEmbedding`` checked
    with :func:`check_ngram_hash_bounds` and mirrors into its int32 limb buffers.
    """

    eos_token_id: int
    heads_per_ngram: int
    multipliers: np.ndarray
    sizes: np.ndarray
    offsets: np.ndarray

    @property
    def num_layers(self) -> int:
        return int(self.multipliers.shape[0])

    @property
    def ngram_size(self) -> int:
        return int(self.multipliers.shape[1])

    @property
    def ngram_heads(self) -> int:
        return int(self.sizes.shape[1])


def collect_ngram_host_tables(model: torch.nn.Module) -> NgramHostTables | None:
    """The host hash tables of the PLE layers under ``model``, or ``None`` when it has none.

    A PLE layer is a module with a ``ple_dense_layer_id`` and a ``ple_embedding`` that keeps its tables on the host
    (``hash_multipliers``, ``hash_head_sizes``, ``hash_head_offsets``: ``HpuQwen4ExpNGramEmbedding``). Call it
    after the weights are loaded: a checkpoint replaces the seed's tables. Raises ``ValueError`` when the dense
    ids are not ``0 .. layers - 1`` or the layers disagree on the n-gram geometry or the EOS id.
    """
    layers = {}
    for module in model.modules():
        embedding = getattr(module, "ple_embedding", None)
        if embedding is None or not hasattr(module, "ple_dense_layer_id") or not hasattr(embedding, "hash_multipliers"):
            continue
        dense_id = int(module.ple_dense_layer_id)
        if dense_id in layers:
            raise ValueError(f"two PLE layers share ple_dense_layer_id {dense_id}")
        layers[dense_id] = embedding
    if not layers:
        return None
    if sorted(layers) != list(range(len(layers))):
        raise ValueError(f"PLE dense layer ids {sorted(layers)} are not 0..{len(layers) - 1}")
    ordered = [layers[index] for index in range(len(layers))]
    geometry = {(int(e.eos_token_id), int(e.ngram_size), int(e.heads_per_ngram)) for e in ordered}
    if len(geometry) != 1:
        raise ValueError(f"PLE layers disagree on (eos_token_id, ngram_size, heads_per_ngram): {sorted(geometry)}")
    eos_token_id, _, heads_per_ngram = next(iter(geometry))
    return NgramHostTables(
        eos_token_id=eos_token_id,
        heads_per_ngram=heads_per_ngram,
        multipliers=np.array([e.hash_multipliers for e in ordered], dtype=np.int64),
        sizes=np.array([e.hash_head_sizes for e in ordered], dtype=np.int64),
        offsets=np.array([e.hash_head_offsets for e in ordered], dtype=np.int64),
    )


def _host_int64(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values).astype(np.int64, copy=False)


def host_ngram_ids(tables: NgramHostTables, input_ids: torch.Tensor | np.ndarray,
                   query_start_loc: torch.Tensor | np.ndarray,
                   ngram_context: torch.Tensor | np.ndarray) -> torch.Tensor:
    """The PLE n-gram ids of one model call, computed on the host: int32 ``[layers, num_tokens, heads]``.

    Takes what ``HpuQwen4ExpNGramEmbedding.compute_ngram_ids`` takes, as CPU tensors or arrays: ``input_ids`` (any
    shape, read row-major as the flat ``[num_tokens]`` batch), ``query_start_loc`` ``[num_reqs + 1]`` (starts at 0,
    non-decreasing) and ``ngram_context`` ``[>= num_reqs, ngram_size - 1]``. Row ``i`` of the result equals the ids
    PLE layer ``i`` computes on the device, and the engine's int64 hash, for every token:

    - the request of a token and its column come from ``query_start_loc`` exactly as the engine's ``searchsorted``
      gives them (tokens past the last start continue the last request);
    - the token ``k`` back is the row's own token when the column is at least ``k``, else ``ngram_context``;
      it is replaced by EOS when any of the ``k`` tokens before the current one is EOS (the engine's
      ``_shift_precompute`` / ``_shift_apply`` cut);
    - the hash is the engine's: the int64 products ``token * multiplier`` XORed cumulatively, each n-gram order
      reduced modulo its heads' sizes plus their offsets. numpy int64 multiplies wrap as torch's do, and for ids
      below the vocabulary size nothing wraps (``check_ngram_hash_bounds``), which is where the device's int32 limb
      arithmetic is exact too.

    Ids below 0 are hashed as id 0: the runner pads a prompt bucket with ``-1`` and schedules ``-1`` for an empty
    draft, and the model forward clamps both to 0 before the PLE layer sees them. Vectorised over the tokens (no
    Python loop over tokens or requests); pure numpy, so it runs where the batch is built.
    """
    tokens = np.maximum(_host_int64(input_ids).reshape(-1), 0)
    starts = _host_int64(query_start_loc).reshape(-1)
    context = _host_int64(ngram_context)
    num_reqs = starts.size - 1
    width = tables.ngram_size - 1
    if num_reqs < 1 or context.ndim != 2 or context.shape[0] < num_reqs or context.shape[1] != width:
        raise ValueError(f"host n-gram ids need query_start_loc [num_reqs + 1] and ngram_context [>= num_reqs, "
                         f"{width}], got {starts.shape} and {context.shape}")
    if starts[0] != 0 or bool((starts[1:] < starts[:-1]).any()):
        raise ValueError(f"host n-gram ids need a query_start_loc that starts at 0 and never decreases, got "
                         f"{starts.tolist()}")

    num_tokens = tokens.size
    positions = np.arange(num_tokens, dtype=np.int64)
    requests = np.minimum(np.searchsorted(starts, positions, side="right") - 1, num_reqs - 1)
    columns = positions - starts[requests]
    eos = tables.eos_token_id

    shifted = [tokens]
    cut = np.zeros(num_tokens, dtype=bool)
    for back in range(1, width + 1):
        in_row = columns >= back
        previous = np.where(in_row, tokens[np.maximum(positions - back, 0)],
                            context[requests, np.clip(columns + width - back, 0, width - 1)])
        cut = cut | (previous == eos)
        shifted.append(np.where(cut, eos, previous))

    per_ngram = tables.heads_per_ngram
    ids = np.empty((tables.num_layers, num_tokens, tables.ngram_heads), dtype=np.int32)
    for layer in range(tables.num_layers):
        multipliers = tables.multipliers[layer]
        mixed = shifted[0] * multipliers[0]
        for index in range(1, tables.ngram_size):
            mixed = mixed ^ (shifted[index] * multipliers[index])
            heads = slice((index - 1) * per_ngram, index * per_ngram)
            reduced = np.remainder(mixed[:, None], tables.sizes[layer, heads])
            ids[layer, :, heads] = reduced + tables.offsets[layer, heads]
    return torch.from_numpy(ids)
