# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_PREFILL_INDEXER_SPLIT=1 (idea G6-4): the prompt indexer scores only this rank's rows.

The indexer projection is replicated and the compressed key cache holds the same keys on every rank, so today
each of the 8 TP ranks scores all query rows of a prompt chunk against all keys (fp32 ``[512, H, N]`` tiles and
a top-k over ``N``, N = context / 4) and gets the same answer. With the flag, rank r scores rows
``[r*T/tp, (r+1)*T/tp)`` and one all-gather collects the block ids (as fp32: exact below 2^24). The selection
must be the one the unsplit indexer makes, on every rank, at contexts that need the top-k and at bucket padding.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.qsa_indexer_ranks import select_rank  # noqa: E402

from harness import run_ranks  # noqa: E402

BUDGET = 64  # 16 blocks of 4: contexts of a few hundred tokens already need the top-k
CASES = [
    (0, 256, 256),  # cold first chunk
    (1024, 256, 256),  # a later chunk: context keys from the cache, top-k over 320 blocks
    (2048, 512, 300),  # bucket padding after the real tokens
    (4096, 1024, 1024),  # two score tiles of 512 rows unsplit, 1024 / tp rows per rank split
]


@pytest.fixture(scope="module", params=[2, 4])
def ranks(request) -> tuple[int, list]:
    return request.param, run_ranks(request.param, select_rank, CASES, BUDGET, timeout_s=600)


def test_split_selection_equals_the_unsplit_one_on_every_rank(ranks):
    tp, results = ranks
    for case_index, case in enumerate(CASES):
        reference = results[0][case_index]["0"]["selected"]
        for rank in range(tp):
            assert results[rank][case_index]["0"]["selected"] == reference, (tp, rank, case)
            assert results[rank][case_index]["1"]["selected"] == reference, (tp, rank, case)


def test_split_adds_one_all_gather_per_row_and_unsplit_none(ranks):
    tp, results = ranks
    for rank_results in results:
        for case_result in rank_results:
            assert case_result["0"]["collectives"] == []
            assert case_result["1"]["collectives"] == ["all_gather"]
            assert case_result["1"]["dtype"] == "torch.int32"


def test_the_selection_uses_the_top_k():
    # a guard on the cases: at these contexts the budget is smaller than the visible blocks
    assert any((ctx + seq) // 4 > BUDGET // 4 for ctx, seq, _ in CASES)


# ----- card rule 9: the top-k's int64 ids are never cast on the device -----

_CASTS = ("_to_copy", "to", "type_as", "int", "long", "float")
_ALIASES = ("view", "reshape", "_reshape_alias", "flatten", "contiguous", "clone", "alias", "squeeze", "unsqueeze",
            "slice", "select", "expand", "cat")


def _op_name(func) -> str:
    return func._schema.name.split("::")[-1]


def _topk_id_casts(fn) -> tuple[list[str], int]:
    """Run ``fn`` and return the cast ops that read a top-k's indices (or a view or copy of them), and how
    many top-k calls there were. A dispatch-mode trace: every aten op ``fn`` runs is seen."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class Trace(TorchDispatchMode):

        def __init__(self) -> None:
            super().__init__()
            self.ids: list = []  # tensors that hold top-k indices (kept alive so their identity stays unique)
            self.casts: list[str] = []
            self.topks = 0

        def _tainted(self, value) -> bool:
            if isinstance(value, torch.Tensor):
                return any(value is t for t in self.ids)
            if isinstance(value, (list, tuple)):
                return any(self._tainted(v) for v in value)
            return False

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = _op_name(func)
            if name == "topk":
                self.topks += 1
                self.ids.append(out[1])
            elif self._tainted(args) or self._tainted(list((kwargs or {}).values())):
                tainted_first = self._tainted(args[:1])
                if name in _CASTS and tainted_first:
                    self.casts.append(f"{name}: {args[0].dtype} -> {out.dtype}")
                elif name in _ALIASES and tainted_first and isinstance(out, torch.Tensor):
                    self.ids.append(out)
            return out

    trace = Trace()
    with trace:
        fn()
    return trace.casts, trace.topks


def _scores_case(t: int, n: int, seed: int = 7):
    import torch

    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand(t, n, generator=gen)
    visible = torch.randint(0, n + 1, (t, ), generator=gen)  # some rows see nothing, some everything
    visible[0] = 0
    visible[-1] = n
    return scores, visible


@pytest.mark.parametrize("t,n,topk", [(8, 64, 16), (8, 8, 16), (5, 40, 16), (16, 320, 16)])
def test_fp32_block_ids_equal_select_blocks(t, n, topk):
    import torch

    from vllm_gaudi.models.qwen4_exp.qsa import select_block_ids_fp32
    from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import select_blocks

    scores, visible = _scores_case(t, n)
    got = select_block_ids_fp32(scores, visible, topk)
    assert got.dtype == torch.float32 and got.shape == (t, topk)
    assert torch.equal(got.to(torch.int32), select_blocks(scores, visible, topk))


def test_the_trace_sees_the_cast_that_card_rule_9_forbids():
    # the guard on the guard: top-k ids narrowed in the same graph show as the forbidden cast
    import torch

    scores, _ = _scores_case(8, 64)
    casts, topks = _topk_id_casts(lambda: torch.topk(scores, 16, dim=-1).indices.to(torch.int32))
    assert topks == 1
    assert casts == ["_to_copy: torch.int64 -> torch.int32"]


def test_the_served_selection_no_longer_casts_the_top_k_ids():
    # program head (qsa-fused): select_blocks narrows through topk_ids_int32, never through a cast (card rule 9)
    from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import select_blocks

    scores, visible = _scores_case(8, 64)
    casts, _ = _topk_id_casts(lambda: select_blocks(scores, visible, 16))
    assert casts == []


def test_fp32_block_ids_never_cast_the_top_k_ids():
    from vllm_gaudi.models.qwen4_exp.qsa import select_block_ids_fp32

    scores, visible = _scores_case(8, 64)
    casts, topks = _topk_id_casts(lambda: select_block_ids_fp32(scores, visible, 16))
    assert topks == 1
    assert casts == []


def test_the_split_selection_never_casts_the_top_k_ids(monkeypatch):
    """The whole split path of one prompt row (scoring, top-k, the all-gather, the expansion) on one rank."""
    from types import SimpleNamespace

    import torch

    from vllm_gaudi.models.qwen4_exp import qsa

    # rank 0 of 2 (no TP group in this process): the other rank's half is stood in for by a copy
    monkeypatch.setattr(qsa, "tensor_model_parallel_all_gather", lambda x, dim=0: torch.cat((x, x), dim=dim))
    indexer = SimpleNamespace(SCORE_TILE=16, compress_ratio=4)  # all the method reads of its indexer
    gen = torch.Generator().manual_seed(5)
    seq, n, dim = 64, 96, 16
    q_row = torch.randn(seq, 4, dim, generator=gen)
    row_keys = torch.randn(n, dim, generator=gen)
    pos_row = torch.arange(320, 320 + seq)
    selected = []
    casts, topks = _topk_id_casts(lambda: selected.append(
        qsa.HpuQsaIndexer._select_prompt_rows_split(indexer, q_row, row_keys, pos_row, BUDGET // 4, (0, 2))))
    assert topks == 2  # two score tiles of 16 rows for this rank's 32 rows
    assert casts == []
    assert selected[0].dtype == torch.int32 and selected[0].shape[0] == seq
