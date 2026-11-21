# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_TOPK_FORM: the mode-1 decode's top-k forms (qsa_sparse_ops.topk_by_form / select_blocks).

Every form must return exactly form 0's block ids, in form 0's order, when no two scores at the cut are equal: at
the served geometry (32,768 block scores, top 512), with masked columns (rows that see fewer blocks than the row is
wide, rows that see fewer than 512, a row that sees none), with zero scores of either sign, and through the
request-major path that mode 1 serves (select_blocks_request_major, one rank).
"""
import pytest
import torch

from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import TOPK_CHUNK, select_blocks, threshold_pick, topk_by_form


def _scores(rows, n, seed, zeros=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randperm(rows * n, generator=g).to(torch.float32).view(rows, n) / 7.0  # distinct, >= 0
    if zeros:
        s[0, :zeros] = 0.0
        s[0, zeros:2 * zeros] = -0.0
    return s


@pytest.mark.parametrize("form", [1, 2, 3])
@pytest.mark.parametrize("rows,n,k", [(4, 32768, 512), (16, 32768, 512), (3, 8192, 512), (5, 1024, 64)])
def test_select_blocks_forms_equal_form0(form, rows, n, k):
    scores = _scores(rows, n, seed=rows + n + form)
    g = torch.Generator().manual_seed(9)
    visible = torch.randint(0, n + 1, (rows, ), generator=g)
    visible[0] = n  # every column
    if rows > 2:
        visible[1] = k // 2  # fewer than k: -1 padding
        visible[2] = 0  # nothing visible
    want = select_blocks(scores, visible, k, form=0)
    got = select_blocks(scores, visible, k, form=form)
    assert got.dtype == torch.int32
    assert torch.equal(got, want)


@pytest.mark.parametrize("form", [1, 2, 3])
def test_zero_scores_and_order(form):
    rows, n, k = 2, 2 * TOPK_CHUNK, 512
    scores = _scores(rows, n, seed=3)
    scores[1] = 0.0  # a row of zeros only: ties everywhere, form 0 picks its own; compare as sets of values
    scores[1, :5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    visible = torch.tensor([n, n])
    want_v, want_i = topk_by_form(scores, k, 0)
    got_v, got_i = topk_by_form(scores, k, form)
    assert torch.equal(got_v, want_v)  # values in the same (descending) order
    assert torch.equal(got_i[0], want_i[0])  # distinct scores: the same ids
    assert torch.equal(got_i[1, :5], torch.arange(5))  # the five positive scores first, in value order
    del visible


def test_threshold_pick_lowest_columns_among_ties():
    scores = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0, 0.0, 3.0]])
    picked = threshold_pick(scores, 3)
    assert picked.tolist() == [[1, 2, 4]]  # three of the four 3.0s: the lowest columns, ascending
    masked = torch.tensor([[2.0, float("-inf"), 1.0]])
    assert threshold_pick(masked, 3).tolist() == [[0, 2, -1]]  # fewer unmasked columns than k: -1


def test_two_stage_falls_back_on_other_widths():
    scores = _scores(2, 3000, seed=4)  # not a multiple of TOPK_CHUNK
    assert torch.equal(topk_by_form(scores, 64, 2)[1], topk_by_form(scores, 64, 0)[1])


@pytest.mark.parametrize("form", [1, 2, 3])
def test_request_major_path(monkeypatch, form):
    from vllm_gaudi.models.qwen4_exp import qsa_fused_path as fp
    monkeypatch.setattr(fp, "shard_requests", lambda bs, shard: (0, 1, bs))  # one rank
    g = torch.Generator().manual_seed(21)
    bs, t_cols, heads, dim, rpb, mb, ratio, topk = 4, 2, 4, 16, 8, 64, 4, 64
    q = torch.randn(bs * t_cols, heads, dim, generator=g)
    cache = torch.randn(mb * rpb * bs + rpb, dim, generator=g)
    table = torch.randperm(mb * bs, generator=g)[:bs * mb].view(bs, mb).repeat_interleave(t_cols, 0).to(torch.int32)
    positions = torch.randint(ratio * topk, mb * rpb * ratio, (bs * t_cols, ), generator=g)
    lens = torch.full((bs, ), t_cols)
    args = (q, positions, table, lens, bs, t_cols, cache, rpb, ratio, topk, fp.QsaShardMode.REPLICATED)
    monkeypatch.setenv("VLLM_HPU_QSA_TOPK_FORM", "0")
    want = fp.select_blocks_request_major(*args)
    monkeypatch.setenv("VLLM_HPU_QSA_TOPK_FORM", str(form))
    assert torch.equal(fp.select_blocks_request_major(*args), want)


def test_flag_default_is_form0(monkeypatch):
    import vllm_gaudi.envs as envs
    monkeypatch.delenv("VLLM_HPU_QSA_TOPK_FORM", raising=False)
    assert envs.VLLM_HPU_QSA_TOPK_FORM == 0
