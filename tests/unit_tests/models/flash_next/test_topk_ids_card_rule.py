# SPDX-License-Identifier: Apache-2.0
"""Top-k token ids the host reads as data: only out of an fp32 id table (``vllm_gaudi/v1/sample/topk_ids.py``).

On the card (engine-step job 00093, collectives job 00108) top-k indices cast in the graph that made them come back
garbled, and top-k indices read back to the host came back as the top-k VALUES in one graph sequence; consumed
inside the graph as the index of a gather they are right. A ``logprobs=5`` request killed a tiny-model server
(spec-lookup job 00086: ``OverflowError`` in the detokenizer) because ``gather_logprobs`` cast and returned the
top-k indices as ids.

``CardTopkRule`` is a CPU model of that rule: every top-k index tensor it hands out may only be reshaped and used
as the index operand of a gather (``torch.gather``, ``take_along_dim``, ``index_select``); any other use (a cast,
a concatenation, arithmetic, a read to the host, a gather FROM it) raises. The upstream form of
``gather_logprobs`` raises under it; the plugin's logprobs paths and the sampled region pass and give the CPU's
ids.
"""

from tests.unit_tests.models.flash_next import engine_harness as eh  # first: installs the Gaudi runtime stand-ins

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

from vllm_gaudi.v1.sample.candidate_sampling import (CANDIDATES, candidate_parameters,  # noqa: E402
                                                     sample_candidates, take_candidates)
from vllm_gaudi.v1.sample.topk_ids import FP32_EXACT_IDS, ids_from_topk_index, topk_with_ids  # noqa: E402

pytestmark = pytest.mark.skipif(not eh.hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

VOCAB = 248320  # the real vocabulary: ids above 2**16, below 2**24

_TOPK = {torch.topk, torch.Tensor.topk}
# (function, position of the index operand)
_INDEX_USES = {torch.gather: 2, torch.Tensor.gather: 2, torch.take_along_dim: 1, torch.index_select: 2,
               torch.Tensor.index_select: 2}
_VIEWS = {torch.Tensor.view, torch.Tensor.reshape, torch.reshape, torch.Tensor.unsqueeze, torch.unsqueeze,
          torch.Tensor.squeeze, torch.squeeze, torch.Tensor.flatten, torch.flatten, torch.Tensor.contiguous,
          torch.Tensor.expand, torch.Tensor.expand_as, torch.Tensor.__getitem__, torch.Tensor.narrow, torch.narrow}
_METADATA = {"__get__", "dim", "size", "numel", "stride", "is_contiguous", "element_size", "__len__", "__hash__",
             "data_ptr", "storage_offset", "is_floating_point", "__format__"}


class CardTopkRule(TorchFunctionMode):
    """Top-k indices may only be reshaped and used as the index operand of a gather; anything else raises."""

    def __init__(self):
        super().__init__()
        self.marked: list[torch.Tensor] = []
        self.index_uses = 0

    def _is_marked(self, value) -> bool:
        return isinstance(value, torch.Tensor) and any(value is m for m in self.marked)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat, _ = tree_flatten((args, kwargs))
        marked = [i for i, value in enumerate(flat) if self._is_marked(value)]
        out = func(*args, **kwargs)
        if func in _TOPK:
            self.marked.append(out.indices)
            return out
        if not marked:
            return out
        name = getattr(func, "__name__", str(func))
        if name in _METADATA:
            return out
        if func in _VIEWS and marked == [0]:
            self.marked.append(out)
            return out
        position = _INDEX_USES.get(func)
        if position is not None:
            index = args[position] if len(args) > position else kwargs.get("index", kwargs.get("indices"))
            if self._is_marked(index) and not self._is_marked(args[0]):
                self.index_uses += 1
                return out
        raise AssertionError(f"card rule: top-k indices used as data by {name}")


def upstream_gather_logprobs(logprobs, num_logprobs, token_ids):
    """vLLM's Sampler.gather_logprobs without mark_unbacked: what the card ran before the fix."""
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
    token_ids = token_ids.unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, token_ids)
    indices = torch.cat((token_ids, topk_indices), dim=1)
    return indices.to(torch.int32), torch.cat((token_logprobs, topk_logprobs), dim=1)


def _logprobs(rows: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, VOCAB, generator=gen).log_softmax(dim=-1)  # fp32: no ties among the largest


def test_the_card_model_refuses_upstream_and_accepts_the_id_table():
    logprobs = _logprobs(4)
    token_ids = logprobs.argmax(dim=-1)
    with pytest.raises(AssertionError, match="top-k indices used as data"):
        with CardTopkRule():
            upstream_gather_logprobs(logprobs, 5, token_ids)
    with CardTopkRule() as rule:
        values, ids = topk_with_ids(logprobs, 5)
    assert rule.index_uses == 1
    want_values, want_ids = logprobs.topk(5, dim=-1)
    assert ids.dtype == torch.int32 and torch.equal(ids.long(), want_ids) and torch.equal(values, want_values)
    assert int(ids.max()) > 65535  # ids beyond 16 bits come through exactly


def test_ids_from_topk_index_is_exact_up_to_two_to_the_24():
    index = torch.tensor([[0, 1, VOCAB - 1], [FP32_EXACT_IDS - 1, 7, FP32_EXACT_IDS - 2]])
    ids = ids_from_topk_index(index, FP32_EXACT_IDS)
    assert torch.equal(ids.long(), index)
    with pytest.raises(ValueError):
        ids_from_topk_index(index, FP32_EXACT_IDS + 1)


@pytest.mark.parametrize("rows,num_logprobs", [(1, 5), (8, 20), (32, 5)])
def test_gather_logprobs_patch_passes_the_card_rule(rows, num_logprobs, monkeypatch):
    import vllm.v1.sample.sampler as sampler_module

    from vllm_gaudi.patches import _hpu_batched_count_greater_than, _hpu_gather_logprobs

    # The plugin swaps in the uncompiled rank count at load time (patches._patch_batched_count_greater_than).
    monkeypatch.setattr(sampler_module, "batched_count_greater_than", _hpu_batched_count_greater_than)
    logprobs = _logprobs(rows, seed=rows)
    token_ids = logprobs.argmax(dim=-1)  # int64, as the sampler hands them over
    with CardTopkRule() as rule:
        got = _hpu_gather_logprobs(logprobs, num_logprobs, token_ids)
    assert rule.index_uses >= 1
    want_ids, want_logprobs = upstream_gather_logprobs(logprobs, num_logprobs, token_ids)
    assert got.logprob_token_ids.dtype == torch.int32
    assert torch.equal(got.logprob_token_ids, want_ids)
    assert torch.equal(got.logprobs, want_logprobs)
    assert torch.equal(got.selected_token_ranks, (logprobs >= logprobs.gather(-1, token_ids[:, None])).sum(-1))


def test_prompt_logprobs_path_passes_the_card_rule():
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    logprobs = _logprobs(6, seed=3)
    token_ids = torch.tensor([5, 70000, 248000, 1, 2, 3], dtype=torch.int64)  # host-made prompt ids
    with CardTopkRule() as rule:
        got = HPUModelRunner._gather_prompt_logprobs_hpu(None, logprobs, 2, token_ids)
    assert rule.index_uses >= 1
    want_ids, want_logprobs = upstream_gather_logprobs(logprobs, 2, token_ids)
    assert got.logprob_token_ids.dtype == torch.int32
    assert torch.equal(got.logprob_token_ids, want_ids) and torch.equal(got.logprobs, want_logprobs)


class _Head(torch.nn.Module):
    """A stand-in LM head: ``compute_logits`` of a fixed weight, no lm_head (vocab_parallel stays off)."""

    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        gen = torch.Generator().manual_seed(11)
        self.weight = torch.randn(vocab, hidden, generator=gen)

    def compute_logits(self, hidden_states):
        return hidden_states.float() @ self.weight.T


@pytest.mark.parametrize("mixed", [False, True])
def test_sampled_region_passes_the_card_rule(mixed, monkeypatch):
    from vllm_gaudi.v1.worker.hpu_model_runner import HpuModelAdapter

    monkeypatch.setenv("VLLM_HPU_VOCAB_PARALLEL_LOGITS", "0")
    rows, hidden = 8, 16
    adapter = type("Adapter", (), {})()
    adapter.model = _Head(hidden, VOCAB)
    gen = torch.Generator().manual_seed(5)
    hidden_states = torch.randn(rows, hidden, generator=gen)
    temperatures = [0.0 if (mixed and r % 3 == 0) else 1.0 for r in range(rows)]
    params, top_k_index = candidate_parameters(temperatures, [20] * rows, [0.95] * rows, rows,
                                               generator=torch.Generator().manual_seed(9))
    with CardTopkRule() as rule:
        _, tokens = HpuModelAdapter.select_compute_logits_and_sample(adapter, hidden_states, torch.arange(rows),
                                                                    params, top_k_index, mixed)
    assert rule.index_uses >= 1
    assert tokens.dtype == torch.int32 and tokens.shape == (rows, 1)
    # The same draw from the CPU oracle's candidates (int64 ids from topk, read on the CPU).
    logits = adapter.model.compute_logits(hidden_states)
    values, ids = take_candidates(logits)
    greedy = logits.argmax(dim=-1, keepdim=True) if mixed else None
    want = sample_candidates(values, ids, params, top_k_index, greedy)
    assert torch.equal(tokens.long(), want)
    assert int(tokens.max()) < VOCAB and CANDIDATES == values.shape[1]
