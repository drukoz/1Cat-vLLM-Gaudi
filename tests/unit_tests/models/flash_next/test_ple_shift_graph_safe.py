# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the graph-safe n-gram shift of the HPU PLE layer (rental run 2).

``HpuQwen4ExpNGramEmbedding._shift_precompute`` replaces the engine's
``torch.cummax`` (no HPU kernel: the Gaudi bridge runs it on the CPU and HPU
graph capture refuses) with a doubling ``torch.maximum`` scan, and
``_shift_apply`` replaces ``where(valid, shifted, tokens.new_full((), eos))``
with ``masked_fill``. These tests pin both static methods to the engine's
``Qwen4ExpNGramEmbedding`` versions:

1. ``test_shift_matches_engine_random``: 300 random grids (EOS-heavy by
   construction), asserting ``torch.equal`` on both ``_shift_precompute``
   outputs and on ``_shift_apply`` for shifts 0..3 (each side applied to its
   own precompute),
2. ``test_shift_edge_cases``: an all-EOS grid, a grid with no EOS, and the
   ``L = 1`` grids ``[[eos]]`` and ``[[3]]``,
3. ``test_no_cummax_or_new_full_in_ple_modules``: the PLE modules contain no
   ``torch.cummax`` or ``new_full`` call outside docstrings and comments.
"""

import inspect
import os
import re

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402

from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding  # noqa: E402

from vllm_gaudi.models.qwen4_exp import ple_layer as ple_layer_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp import ple_ops as ple_ops_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_layer import HpuQwen4ExpNGramEmbedding  # noqa: E402

EOS = 7


def _assert_shift_matches_engine(tokens):
    """Ours and the engine agree on the precompute and on every shift 0..3."""
    ours_pos, ours_pis = HpuQwen4ExpNGramEmbedding._shift_precompute(tokens, EOS)
    engine_pos, engine_pis = Qwen4ExpNGramEmbedding._shift_precompute(tokens, EOS)
    assert torch.equal(ours_pos, engine_pos)
    assert torch.equal(ours_pis, engine_pis)
    for shift in (0, 1, 2, 3):
        ours = HpuQwen4ExpNGramEmbedding._shift_apply(tokens, ours_pos, ours_pis, shift, EOS)
        engine = Qwen4ExpNGramEmbedding._shift_apply(tokens, engine_pos, engine_pis, shift, EOS)
        assert torch.equal(ours, engine), f"shift {shift} differs for a {tuple(tokens.shape)} grid"


def test_shift_matches_engine_random():
    gen = torch.Generator().manual_seed(0)
    for case in range(300):
        b_size = int(torch.randint(1, 5, (1,), generator=gen))
        seq_len = int(torch.randint(1, 71, (1,), generator=gen))
        vocab = int(torch.randint(8, 41, (1,), generator=gen))  # small, so EOS appears often
        tokens = torch.randint(0, vocab, (b_size, seq_len), generator=gen, dtype=torch.int64)
        if case % 3 == 0:
            tokens[:, 0] = EOS
        if case % 5 == 0:
            tokens[:, -2:] = EOS
        _assert_shift_matches_engine(tokens)


def test_shift_edge_cases():
    no_eos = torch.arange(8, 8 + 3 * 6, dtype=torch.int64).reshape(3, 6)  # vocab starts above EOS
    for tokens in (torch.full((2, 9), EOS, dtype=torch.int64), no_eos,
                   torch.tensor([[EOS]], dtype=torch.int64), torch.tensor([[3]], dtype=torch.int64)):
        _assert_shift_matches_engine(tokens)


def _strip_comments_and_docstrings(src):
    """Module source with triple-quoted strings and comment-only lines removed."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))


def test_no_cummax_or_new_full_in_ple_modules():
    for module in (ple_layer_module, ple_ops_module):
        code = _strip_comments_and_docstrings(inspect.getsource(module))
        assert "cummax" not in code, f"{module.__name__} calls torch.cummax"
        assert "new_full" not in code, f"{module.__name__} calls new_full"
