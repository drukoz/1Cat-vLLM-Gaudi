# SPDX-License-Identifier: Apache-2.0
"""Pure CPU helpers for the Flash-Next PLE forward kwargs.

Upstream Flash-Next (``vllm.models.qwen4_exp.nvidia.model``) declares
``requires_raw_input_tokens = True`` and its PLE layer needs two extra
forward kwargs: ``query_start_loc`` (int32 ``[num_reqs + 1]``, cumulative
token offsets of the flattened batch) and ``ngram_context`` (int32
``[num_reqs, ngram_size - 1]``, the token ids immediately before each
request's first scheduled token, ``eos_token_id`` where no such token
exists). Without them the model raises
``RuntimeError("PLE inputs were not prepared")``.

HPU batches are 2-D padded tensors ``[padded_batch_size, num_tokens]``; the
model flattens ``input_ids.reshape(-1)``, so each padded row is one
"request" of ``num_tokens`` tokens and
``query_start_loc = arange(padded_batch_size + 1) * num_tokens``. Padding
rows and padded tail tokens receive dummy values that are discarded with
the rest of the padded outputs.

This module imports only ``torch`` and ``numpy`` so it is unit-testable on
machines without an HPU; ``hpu_model_runner.py`` cannot be imported there.
"""

from typing import Optional, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]


def model_requires_raw_input_tokens(model) -> bool:
    """True when the model declares ``requires_raw_input_tokens``.

    Checks the given object first and falls back to its ``.model``
    attribute, so it works both with the ``HpuModelAdapter`` wrapper and
    with the unwrapped model.
    """
    if model is None:
        return False
    if hasattr(model, "requires_raw_input_tokens"):
        return bool(model.requires_raw_input_tokens)
    inner = getattr(model, "model", None)
    return inner is not None and bool(getattr(inner, "requires_raw_input_tokens", False))


def _to_numpy(tensor: TensorLike) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def build_ple_inputs(token_ids_cpu: torch.Tensor, num_computed_tokens_cpu, batch_indices: list[int],
                     padded_batch_size: int, num_tokens: int, eos_token_id: int,
                     ngram_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(query_start_loc, ngram_context)`` as CPU int32 tensors.

    Row ``r < len(batch_indices)`` of ``ngram_context`` holds
    ``token_ids_cpu[b, start - (ngram_size - 1) : start]`` with
    ``b = batch_indices[r]`` and ``start = int(num_computed_tokens_cpu[b])``,
    left-filled with ``eos_token_id`` where ``start - k < 0``. Rows
    ``>= len(batch_indices)`` are all ``eos_token_id``.
    """
    context_len = max(int(ngram_size) - 1, 0)
    query_start_loc = torch.arange(padded_batch_size + 1, dtype=torch.int32) * int(num_tokens)
    ngram_context = torch.full((padded_batch_size, context_len), int(eos_token_id), dtype=torch.int32)
    token_ids_np = _to_numpy(token_ids_cpu)
    computed_np = _to_numpy(num_computed_tokens_cpu)
    for row, b in enumerate(batch_indices):
        if row >= padded_batch_size:
            break
        start = int(computed_np[b])
        row_ids = token_ids_np[b]
        for k in range(context_len):
            pos = start - context_len + k
            if pos >= 0:
                ngram_context[row, k] = int(row_ids[pos])
    return query_start_loc, ngram_context


def dummy_ple_inputs(padded_batch_size: int,
                     num_tokens: int,
                     eos_token_id: int,
                     ngram_size: int,
                     token_ids_cpu: Optional[TensorLike] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """All-eos ``(query_start_loc, ngram_context)`` for warmup batches."""
    if token_ids_cpu is None:
        token_ids_cpu = np.zeros((0, 0), dtype=np.int32)
    return build_ple_inputs(token_ids_cpu, np.zeros(0, dtype=np.int32), [], padded_batch_size, num_tokens, eos_token_id,
                            ngram_size)
