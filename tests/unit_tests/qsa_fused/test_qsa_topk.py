# SPDX-License-Identifier: Apache-2.0
"""The one-node QSA top-k (``qsa_index_topk``, ``VLLM_HPU_QSA_TOPK_FORM=4``): its reference against ``torch.topk`` and
the op's plumbing (schema, fake, CPU kernel, GUID, glue, build), on CPU.

The kernel itself runs in the simulator (``tests/standalone/qsa_fused/test_qsa_simulation.py``, the ``topk_*`` cases:
its ids bit-exact against ``qsa_index_topk_masked_ref``) and on a card (``test_hpu_op.py``). The selected set is
``select_blocks``' (``torch.topk`` of the masked row) whenever the scores at the cut are distinct; among scores equal at
the cut the lowest columns are taken.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

import flashinfer_gaudi.qsa_fused as qsa_fused
from tests.unit_tests.qsa_fused import qsa_ops  # noqa: F401  (the CPU kernels)
from tests.unit_tests.qsa_fused.qsa_reference import (masked_score_keys, qsa_index_topk_masked_ref, select_blocks,
                                                      visible_blocks)

ROOT = Path(__file__).resolve().parents[3]
CSRC = ROOT / "csrc" / "qsa_fused"
K = 512
N = 4096


def _masked(rows: int, n: int, seed: int, visible=None) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    scores = torch.rand(rows, n, generator=gen) + 1e-3  # distinct with probability ~1
    if visible is None:
        visible = torch.randint(1, n + 1, (rows, ), generator=gen)
    cols = torch.arange(n)
    return scores.masked_fill(cols[None, :] >= torch.as_tensor(visible)[:, None], float("-inf"))


def test_distinct_scores_select_the_set_of_torch_topk() -> None:
    masked = _masked(8, N, seed=1, visible=[N, N - 1, 3000, K + 1, K, K - 1, 37, 1])
    got = qsa_index_topk_masked_ref(masked, K)
    want = select_blocks(masked, torch.full((8, ), N), K)  # the mask is already in the scores
    for t in range(8):
        g = got[t][got[t] >= 0]
        w = want[t][want[t] >= 0]
        assert sorted(g.tolist()) == sorted(w.tolist()), t
        assert int((got[t] < 0).sum()) == int((want[t] < 0).sum()), t
        # one bitmap (no ties beyond the cut): ascending
        assert torch.equal(g, torch.sort(g).values), t


def test_ties_at_the_cut_take_the_lowest_columns_after_every_column_above() -> None:
    masked = torch.zeros(2, 1024)
    masked[0, 100:130] = 2.0  # 30 above the cut
    masked[0, 500:600] = 1.0  # 100 tied at the cut for 34 places
    masked[1, ::2] = 1.0  # 512 tied values for 40 places, nothing above
    got = qsa_index_topk_masked_ref(masked, 64)
    assert got[0].tolist() == list(range(100, 130)) + list(range(500, 534))
    assert got[1, :40].tolist() == list(range(0, 80, 2)) and got[1, 40:].tolist() == list(range(80, 128, 2))
    got1 = qsa_index_topk_masked_ref(masked[1:], 40)
    assert got1[0].tolist() == list(range(0, 80, 2))


def test_zero_scores_nan_and_masked_columns() -> None:
    masked = torch.full((1, 512), float("-inf"))
    masked[0, :10] = -0.0  # keys of 0
    masked[0, 10:20] = 0.0
    masked[0, 20] = float("nan")  # sign bit clear: above +inf, taken first
    masked[0, 21] = float("inf")
    masked[0, 22:30] = 0.5
    keys = masked_score_keys(masked)
    assert int(keys[0, 0]) == 0 and int(keys[0, 10]) == 0 and int(keys[0, 21]) == 0x7F800000
    assert int(keys[0, 20]) > int(keys[0, 21]) and int(keys[0, 40]) < 0
    got = qsa_index_topk_masked_ref(masked, 12)
    # above the cut (key 0): 20, 21, 22..29 (10 columns); then the two lowest zeros, 0 and 1 (two bitmaps)
    assert got[0].tolist() == [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1]
    # fewer keys than k: every key, ascending, then -1
    got = qsa_index_topk_masked_ref(masked, 64)
    assert got[0, :30].tolist() == list(range(30)) and (got[0, 30:] == -1).all()


def test_a_padding_row_selects_nothing() -> None:
    masked = torch.full((1, 512), float("-inf"))
    assert (qsa_index_topk_masked_ref(masked, 8) == -1).all()


def test_the_op_runs_the_reference_on_cpu_and_its_fake_has_the_output_shape() -> None:
    masked = _masked(4, 1024, seed=2)
    got = qsa_fused.qsa_index_topk(masked, 64)
    assert got.dtype == torch.int32 and torch.equal(got, qsa_index_topk_masked_ref(masked, 64))
    assert "qsa_index_topk" in qsa_fused.SCHEMAS and qsa_fused.GUIDS["qsa_index_topk"] == "qsa_fused_topk_f32_gaudi2"
    assert not qsa_fused.qsa_topk_missing_op("cpu")
    with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
        fake = mode.from_tensor(masked)
        out = torch.ops.flashinfer_gaudi.qsa_index_topk(fake, 64)
        assert tuple(out.shape) == (4, 64) and out.dtype == torch.int32


@pytest.mark.parametrize("shape,topk,reason", [((4, 32768), 512, None), ((4, 1024), 8, None),
                                               ((4, 1000), 8, "1000 columns"), ((4, 65536), 512, "65536 columns"),
                                               ((4, 1024), 0, "topk 0")])
def test_shape_reason(shape, topk, reason) -> None:
    got = qsa_fused.qsa_topk_shape_reason(torch.zeros(shape), topk)
    assert (got is None) if reason is None else (reason in got)


def test_the_guid_glue_and_build_agree() -> None:
    params = (CSRC / "qsa_fused_params.h").read_text()
    assert 'kQsaFusedTopkGuid = "qsa_fused_topk_f32_gaudi2"' in params
    assert "kQsaFusedGuidCount = 5" in params
    assert re.search(r"kQsaFusedTopkMaxColumns = 32768;", params) and "kQsaFusedTopkColumnStep = 512;" in params
    glue = (CSRC / "host" / "qsa_fused_gaudi2.cpp").read_text()
    assert "{Kind::Topk, kQsaFusedTopkGuid, 1, 2}," in glue
    assert "_binary_qsa_fused_topk_gaudi2_o_start" in glue
    assert "Topk" in (CSRC / "host" / "qsa_fused_gaudi2.hpp").read_text()
    cmake = (CSRC / "CMakeLists.txt").read_text()
    assert "qsa_fused_topk_gaudi2)" in cmake
    op = (CSRC / "pytorch" / "hpu_qsa_fused.cpp").read_text()
    assert 'm.impl("qsa_index_topk", run_topk);' in op and "kQsaFusedTopkGuid" in op
    kernel = (CSRC / "kernels" / "qsa_fused_topk_gaudi2.c").read_text()
    assert "void main(tensor masked, tensor ids, tensor workspace, int topk)" in kernel
    # the vector stores must be visible to the scalar walk: the aso pair before each read-back
    assert kernel.count("aso(SW_INC | SW_VPU);") == 2 and kernel.count("aso(SW_DEC | SW_SPU);") == 2
