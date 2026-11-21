# SPDX-License-Identifier: Apache-2.0
"""The sparse-QSA block selection at a long context, and card rule 9 (top-k ids are never cast to int32).

Two things no test covered before 2026-11-20:

* a selection where top-k really selects (more than ``block_topk`` = 512 visible blocks, over 2,048 tokens of
  context) at the real model's geometry, against an oracle that runs no top-k (``tests/unit_tests/qsa_select/
  select_case.py``): mode 0 (the default path), mode 1 (request-major keys) and mode 1 sharded by request, on
  plain decode rows and on the verification lattice;
* the card's behaviour found by kernel-fixes' job 00038: ``torch.topk``'s int64 ids, narrowed with
  ``.to(torch.int32)`` in the same lazy graph, came back as every second id of the table with the second half
  out of range (a ``[16, 512]`` input, ``k = 10``; at the QSA shapes the cast was right on the card, jobs 00062
  and 00063). ``CardIds`` models the defect on the CPU; with it ``select_blocks`` must still select the right
  blocks, and the cast it replaced must not (which shows the model would catch the old line).

The same case runs on a card in ``tests/unit_tests/qsa_select/test_hpu_select_ids.py``.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402

from tests.unit_tests.qsa_select import select_case as sc  # noqa: E402
from vllm_gaudi.models.qwen4_exp import qsa_sparse_ops  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import expand_selection, select_blocks  # noqa: E402


@pytest.mark.parametrize("name,mode", [("decode8", 0), ("decode8", 1), ("lattice4", 0), ("lattice4", 1)])
def test_long_context_selection_is_the_oracle(name, mode):
    case = sc.build_case(name)
    cmp = sc.compare(case, sc.decode_selection(case, "cpu", mode))
    assert cmp["rows"] == len(sc.real_rows(case))
    assert cmp["ok"], cmp


@pytest.mark.parametrize("shard", [1, 2])
@pytest.mark.parametrize("name,world", [("decode8", 4), ("decode8", 8), ("lattice4", 4), ("lattice4", 2)])
def test_sharded_long_context_selection_is_the_oracle(name, world, shard):
    case = sc.build_case(name)
    want = sc.oracle(case)
    rows = 0
    for rank in range(world):
        cmp = sc.compare(case, sc.decode_selection(case, "cpu", 1, shard, (rank, world)), want)
        rows += cmp["rows"]
        assert cmp["ok"], (rank, cmp)
    assert rows == len(sc.real_rows(case))


@pytest.mark.parametrize("mode,world", [(0, 1), (1, 1), (1, 4)])
def test_a_context_sized_block_table_selects_the_same(mode, world):
    """VLLM_HPU_QSA_TABLE_WIDTH=1: the table as wide as the batch's longest request needs, not as the engine's."""
    from vllm_gaudi.v1.worker.flash_next_inputs import decode_block_table_width
    case = sc.build_case("decode8_32k")
    want = sc.oracle(case)
    blocks = [(ctx + n + sc.BLOCK - 1) // sc.BLOCK for ctx, n in case.specs]
    width = decode_block_table_width(blocks, sc.MB, 1)
    assert width == 256 and decode_block_table_width(blocks, sc.MB, 0) == sc.MB
    assert bool((case.block_table[:, width:] == sc.PAD_PAGE).all())  # the dropped columns hold the pad block only
    rows = 0
    for rank in range(world):
        got = sc.decode_selection(case, "cpu", mode, int(world > 1), (rank, world) if world > 1 else None, width=width)
        cmp = sc.compare(case, got, want)
        rows += cmp["rows"]
        assert cmp["ok"], (rank, cmp)
    assert rows == len(sc.real_rows(case))


@pytest.mark.parametrize("blocks,mode,expected", [([1, 3, 9], 1, 64), ([64], 1, 64), ([65, 2], 1, 128),
                                                  ([257], 1, 512), ([1000, 3], 1, 1024), ([1024], 1, 1024),
                                                  ([5], 0, 1024), ([5], 2, 1024), ([], 1, 1024)])
def test_decode_block_table_width(blocks, mode, expected):
    from vllm_gaudi.v1.worker.flash_next_inputs import decode_block_table_width
    assert decode_block_table_width(blocks, 1024, mode) == expected
    assert decode_block_table_width([900], 1000, 1) == 1000  # an engine whose longest context is not a power of two


def test_the_case_really_selects():
    """The case is a top-k test: most rows see more blocks than the budget, and the oracle drops visible blocks."""
    case = sc.build_case("decode8")
    want = sc.oracle(case)
    sparse = [r for r in want.values() if r["visible"] > sc.TOPK]
    assert len(sparse) >= 5
    assert all(len(r["blocks"]) == sc.TOPK for r in sparse)
    assert max(r["position"] for r in want.values()) > 131000


# ----------------------------------------------------------------------------- the card's cast, modelled
def card_cast_int32(ids: torch.Tensor) -> torch.Tensor:
    """What a Gaudi card returns for ``ids.to(torch.int32)`` when ``ids`` are top-k's int64 ids in the same graph.

    The bridge holds the ids as 4 bytes each and the cast reads 8: element ``j`` of the result is id ``2 * j`` of
    the flat table, and the second half reads past the table (job 00038; modelled as an out-of-range value).
    """
    flat = ids.reshape(-1)
    out = torch.full((flat.numel(), ), 2**30, dtype=torch.int32)
    half = (flat.numel() + 1) // 2
    out[:half] = flat[0::2].to(torch.int32)
    return out.view(ids.shape)


class CardIds(torch.Tensor):
    """int64 ids as a card's lazy graph holds them: usable as an index, wrong when cast to int32 in the same graph.

    A tensor derived from them (``masked_fill``, ``where``, ``clone``) is still such a tensor.
    """

    def to(self, *args, **kwargs):
        plain = self.as_subclass(torch.Tensor)
        target = kwargs.get("dtype", next((a for a in args if isinstance(a, torch.dtype)), None))
        if plain.dtype == torch.int64 and target is not None and target != torch.int64:
            # Any conversion of the ids' type reads them as 8-byte elements (the model's assumption for the
            # types job 00038 did not try: a cast through fp32 is not a way around the rule).
            return card_cast_int32(plain).to(target)
        return plain.to(*args, **kwargs).as_subclass(CardIds)

    def int(self):
        return self.to(torch.int32)

    def type(self, dtype=None, *args, **kwargs):
        if dtype is None:
            return self.as_subclass(torch.Tensor).type()
        return self.to(dtype)


@pytest.fixture
def card_topk(monkeypatch):
    real_topk = torch.topk

    def topk(*args, **kwargs):
        vals, idx = real_topk(*args, **kwargs)
        return vals, idx.as_subclass(CardIds)

    monkeypatch.setattr(torch, "topk", topk)


CASTS = {
    "cast": lambda idx, n: idx.to(torch.int32),  # the line select_blocks had before 2026-11-20
    "through_fp32": lambda idx, n: idx.to(torch.float32).to(torch.int32),
}


def host_selection(case, narrow=None, monkeypatch=None):
    """``select_blocks`` on the case's scores; ``narrow`` replaces ``topk_ids_int32`` with one of ``CASTS``."""
    if narrow is not None:
        monkeypatch.setattr(qsa_sparse_ops, "topk_ids_int32", CASTS[narrow])
    scores, visible = sc.host_scores(case)
    blocks = select_blocks(scores, visible, sc.TOPK).as_subclass(torch.Tensor)
    return {"selected": expand_selection(blocks.clamp(max=sc.N_KEYS - 1), case.positions, sc.RATIO)}


def test_select_blocks_is_right_on_a_device_that_casts_like_the_card(card_topk):
    case = sc.build_case("decode8")
    cmp = sc.compare(case, host_selection(case))
    assert cmp["ok"], cmp


@pytest.mark.parametrize("narrow", sorted(CASTS))
def test_the_model_catches_a_cast_of_the_ids(card_topk, monkeypatch, narrow):
    """A cast of the ids (the line before 2026-11-20, or a cast through fp32) selects wrongly under the model."""
    case = sc.build_case("decode8")
    cmp = sc.compare(case, host_selection(case, narrow, monkeypatch))
    assert len(cmp["selected_wrong_rows"]) >= 6, cmp


def test_whole_decode_step_under_the_card_model(card_topk):
    """Mode 0 and mode 1 of the indexer's decode step with top-k ids that cannot be cast."""
    for name, mode in (("decode8", 0), ("lattice4", 1)):
        case = sc.build_case(name)
        got = sc.decode_selection(case, "cpu", mode)
        got = {k: (v.as_subclass(torch.Tensor) if isinstance(v, torch.Tensor) else v) for k, v in got.items()}
        cmp = sc.compare(case, got)
        assert cmp["ok"], (name, mode, cmp)
