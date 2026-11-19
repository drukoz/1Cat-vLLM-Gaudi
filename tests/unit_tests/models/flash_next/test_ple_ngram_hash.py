# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the int32-only PLE n-gram hash (RUK-3116).

``ple_ngram_ids_int32`` must equal the engine's int64 n-gram hash (``vllm/models/qwen4_exp/nvidia/
ple_layer.py:426-440``) on 20,000 random tokens with the real Flash-Next tables (vocab 248320, n-gram 3, 16 heads,
base 20,000,000) and the tiny layout (base 2048), create no int64 intermediate, and reject out-of-range tables.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.utils import _pytree as pytree  # noqa: E402

from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding  # noqa: E402

from vllm_gaudi.models.qwen4_exp.ple_ops import (  # noqa: E402
    check_ngram_hash_bounds,
    ngram_hash_limbs,
    ngram_mod_weights,
    ple_ngram_ids_int32,
)

VOCAB_SIZE = 248320
EOS_TOKEN_ID = 248044
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NUM_TOKENS = 20_000
REAL_BASE = 20_000_000
TINY_BASE = 2048
REAL_MULT = [35243655749357, 23198521423797, 25737225394503]


def _tables(ngram_vocab_size_base: int) -> tuple[list[int], list[int], list[int]]:
    """The engine's multipliers, head sizes and offsets for one layout, pinned to the spec's values."""
    mult = Qwen4ExpNGramEmbedding._make_layer_multipliers(
        ngram_size=NGRAM_SIZE, unigram_vocab_size=VOCAB_SIZE, seed=1234, ple_dense_layer_id=2)
    sizes, offsets, total = Qwen4ExpNGramEmbedding._make_vocab_layout(
        ngram_vocab_size_base=ngram_vocab_size_base, ngram_heads=2 * HEADS_PER_NGRAM, ple_dense_layer_id=2)
    assert mult == REAL_MULT
    if ngram_vocab_size_base == REAL_BASE:
        assert max(sizes) == 20_000_689 and total == 320_009_088
    else:
        assert max(sizes) == 2399 and total == 37_678
    return mult, sizes, offsets


def _tokens() -> list[torch.Tensor]:
    """20,000 random tokens per position, with 0, vocab - 1 and eos forced into the first rows."""
    generator = torch.Generator().manual_seed(0)
    tokens = [torch.randint(0, VOCAB_SIZE, (NUM_TOKENS,), generator=generator) for _ in range(NGRAM_SIZE)]
    for row, token_id in enumerate((0, VOCAB_SIZE - 1, EOS_TOKEN_ID)):
        for position in tokens:
            position[row] = token_id
    return tokens


def _reference(tokens: list[torch.Tensor], mult: list[int], sizes: list[int], offsets: list[int],
               heads_per_ngram: int) -> torch.Tensor:
    """The engine's int64 n-gram hash math (``ple_layer.py:426-440``) on int64 CPU tensors."""
    shifted = [token.to(torch.int64) for token in tokens]
    multipliers = torch.tensor(mult, dtype=torch.int64)
    head_sizes = torch.tensor(sizes, dtype=torch.int64)
    head_offsets = torch.tensor(offsets, dtype=torch.int64)
    blocks = []
    for ngram in range(2, len(tokens) + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[index] * multipliers[index])
        ids = torch.remainder(mixed.unsqueeze(-1), head_sizes[start:end]) + head_offsets[start:end]
        blocks.append(ids)
    return torch.cat(blocks, dim=-1)


def _int32_inputs(tokens: list[torch.Tensor], mult: list[int], sizes: list[int],
                  offsets: list[int]) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
    limbs = torch.tensor([ngram_hash_limbs(m) for m in mult], dtype=torch.int32)
    weights = torch.tensor(ngram_mod_weights(sizes), dtype=torch.int32)
    return ([token.to(torch.int32) for token in tokens], limbs, weights,
            torch.tensor(sizes, dtype=torch.int32), torch.tensor(offsets, dtype=torch.int32))


@pytest.mark.parametrize("base", [REAL_BASE, TINY_BASE], ids=["real", "tiny"])
def test_matches_int64_reference(base):
    mult, sizes, offsets = _tables(base)
    tokens = _tokens()
    args = _int32_inputs(tokens, mult, sizes, offsets)
    got = ple_ngram_ids_int32(*args, heads_per_ngram=HEADS_PER_NGRAM)
    ref = _reference(tokens, mult, sizes, offsets, HEADS_PER_NGRAM)
    assert got.dtype == torch.int32
    assert got.shape == (NUM_TOKENS, (NGRAM_SIZE - 1) * HEADS_PER_NGRAM)
    assert torch.equal(got.long(), ref)


class _Int64Recorder(torch.utils._python_dispatch.TorchDispatchMode):
    """Records every dispatched op whose output contains an int64 tensor."""

    def __init__(self):
        super().__init__()
        self.int64_ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for value in pytree.tree_leaves(out):
            if isinstance(value, torch.Tensor) and value.dtype == torch.int64:
                self.int64_ops.append(str(func))
        return out


def test_no_int64_intermediates():
    mult, sizes, offsets = _tables(REAL_BASE)
    tokens = _tokens()
    args = _int32_inputs(tokens, mult, sizes, offsets)
    with _Int64Recorder() as recorder:
        ple_ngram_ids_int32(*args, heads_per_ngram=HEADS_PER_NGRAM)
    assert recorder.int64_ops == []


def test_bounds_rejected():
    mult, sizes, offsets = _tables(REAL_BASE)
    check_ngram_hash_bounds(mult, sizes, offsets, VOCAB_SIZE)  # the real tables are accepted
    with pytest.raises(ValueError):
        check_ngram_hash_bounds(mult, sizes, offsets, (1 << 25) + 1)
    bad_sizes = list(sizes)
    bad_sizes[0] = (1 << 25) + 1
    with pytest.raises(ValueError):
        check_ngram_hash_bounds(mult, bad_sizes, offsets, VOCAB_SIZE)
    bad_mult = list(mult)
    bad_mult[0] = (1 << 63) // (VOCAB_SIZE - 1) + 1
    with pytest.raises(ValueError):
        check_ngram_hash_bounds(bad_mult, sizes, offsets, VOCAB_SIZE)
    bad_offsets = list(offsets)
    bad_offsets[-1] = (1 << 31) - 1
    with pytest.raises(ValueError):
        check_ngram_hash_bounds(mult, sizes, bad_offsets, VOCAB_SIZE)


@pytest.mark.parametrize("device", ["hpu"])
def test_matches_on_device(device):
    pytest.importorskip("habana_frameworks.torch")
    import habana_frameworks.torch  # noqa: F401,E402  registers the hpu device with torch
    if not (hasattr(torch, "hpu") and torch.hpu.is_available()):  # test_registration.py stubs habana_frameworks on CPU
        pytest.skip("no HPU device: habana_frameworks is a CPU stub")
    mult, sizes, offsets = _tables(REAL_BASE)
    tokens = _tokens()
    args = _int32_inputs(tokens, mult, sizes, offsets)
    device_args = ([token.to(device) for token in args[0]], *[tensor.to(device) for tensor in args[1:]])
    got = ple_ngram_ids_int32(*device_args, heads_per_ngram=HEADS_PER_NGRAM).cpu()
    ref = _reference(tokens, mult, sizes, offsets, HEADS_PER_NGRAM)
    assert torch.equal(got.long(), ref)
