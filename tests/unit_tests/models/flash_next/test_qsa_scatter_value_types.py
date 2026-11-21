# SPDX-License-Identifier: Apache-2.0
"""The sparse QSA path never scatters a Python bool.

On a Gaudi2 card (rental 2026-11-20) the first sparse prefill died when the lazy graph was
flushed: "We don't have an op for aten::scatter_ but it isn't a special case. Argument types:
Tensor, int, Tensor, bool". The Habana bridge's graph accepts ``scatter_`` with a tensor source or
a numeric scalar, not with a bool value, and CPU torch accepts all three, so only a test that
looks at the argument types can see it here.
"""

import pytest
import torch

from vllm_gaudi.models.qwen4_exp import qsa_hpu_attention, qsa_sparse_ops


@pytest.fixture
def scatter_values(monkeypatch):
    """Record the type of every scalar value passed to Tensor.scatter_ / Tensor.scatter."""
    seen = []
    real_inplace = torch.Tensor.scatter_
    real_copy = torch.Tensor.scatter

    def record(args, kwargs):
        value = kwargs.get("value", args[2] if len(args) > 2 else None)
        if not isinstance(value, torch.Tensor):
            seen.append(type(value))

    def scatter_(self, *args, **kwargs):
        record(args, kwargs)
        return real_inplace(self, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        record(args, kwargs)
        return real_copy(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "scatter_", scatter_)
    monkeypatch.setattr(torch.Tensor, "scatter", scatter)
    return seen


def _reference_bias(selected, context_lens, n_ctx_cols, num_chunk_cols, dtype):
    """Loop reference of prefill_selection_bias."""
    bs, num_queries, _ = selected.shape
    cols = n_ctx_cols + num_chunk_cols
    bias = torch.full((bs, 1, num_queries, cols), float("-inf"), dtype=dtype)
    for b in range(bs):
        ctx = int(context_lens[b])
        for t in range(num_queries):
            picked = [int(p) for p in selected[b, t] if int(p) >= 0]
            if not picked:
                bias[b, 0, t, :] = 0.0  # a padding row attends to everything instead of to nothing
                continue
            for p in picked:
                bias[b, 0, t, p if p < ctx else n_ctx_cols + (p - ctx)] = 0.0
    return bias


def _selected():
    g = torch.Generator().manual_seed(7)
    selected = torch.randint(0, 40, (2, 6, 5), generator=g, dtype=torch.int32)
    selected[0, 1, 3:] = -1  # a partly padded row
    selected[1, 4, :] = -1  # a row that selects nothing
    return selected, torch.tensor([24, 8], dtype=torch.long)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_selection_bias_scatters_no_bool_and_matches_the_reference(scatter_values, dtype):
    selected, context_lens = _selected()
    bias = qsa_hpu_attention.prefill_selection_bias(selected, context_lens, n_ctx_cols=32, num_chunk_cols=32,
                                                    dtype=dtype)
    assert scatter_values, "the helper no longer scatters: this test must look at its new op instead"
    assert bool not in scatter_values, "scatter with a Python bool value has no op in the Habana graph"
    assert bias.shape == (2, 1, 6, 64) and bias.dtype == dtype
    assert torch.equal(bias, _reference_bias(selected, context_lens, 32, 32, dtype))


def test_selection_mask_scatters_no_bool_and_matches_the_reference(scatter_values):
    selected = torch.tensor([[0, 3, -1], [-1, -1, -1], [5, 5, 2]], dtype=torch.int32)
    mask = qsa_sparse_ops.selection_mask(selected, num_columns=6)
    assert bool not in scatter_values, "scatter with a Python bool value has no op in the Habana graph"
    expected = torch.zeros(3, 6, dtype=torch.bool)
    expected[0, [0, 3]] = True
    expected[2, [5, 2]] = True
    assert mask.dtype == torch.bool and torch.equal(mask, expected)
