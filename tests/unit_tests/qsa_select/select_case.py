# SPDX-License-Identifier: Apache-2.0
"""A long-context decode step of the sparse-QSA indexer whose selection is known without running a top-k.

Card rule 9 (kernel-fixes job 00038, 2026-11-20): on a Gaudi card ``torch.topk``'s int64 ids, narrowed with
``.to(torch.int32)`` in the same lazy graph, come back as every second id of the table. ``select_blocks`` did
that, and no card run had ever checked a selection against the CPU at a context where top-k really selects (more
than ``block_topk`` = 512 blocks, 2,048 tokens). This module builds such a step at the real model's geometry
(block 128, ratio 4, ``--max-model-len 131072``: 32,768 block keys per request, 512 selected, 2,051 attended
positions), an oracle of the right answer, and the two ways a device's answer is read:

* the selection itself, read on the host (int32 ``[T, W]`` attended positions);
* what an **in-graph consumer** of the selection gets: the decode attention's own slot arithmetic
  (``decode_selected_slots``) and one gather from a table that holds each slot's number as fp32, so the host reads
  floats and no integer of the selection is ever converted for the read.

Scores are exact on every device: each block key has two non-zero elements, ``hi`` (an integer in [1, 128]) and
``lo`` (a multiple of 1/256 below 1), and every query is all ones, so a block's score is ``H * (hi + lo) /
sqrt(D)``: sums of two bf16 values with exact fp32 results, distinct for every block of a request, whatever the
order a GEMM adds its products in. The top-k set and its order therefore do not depend on rounding, and a
difference from the oracle is an indexing or integer-type defect, not noise. ``k_layernorm`` has the weight -1, so
the key of a block that the step itself completes is exactly zero on every device (the lowest score).

Used by the CPU test ``tests/unit_tests/models/flash_next/test_qsa_select_ids.py`` and by the card test
``test_hpu_select_ids.py`` next to this file.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

BLOCK = 128  # tokens per attention kernel block
RATIO = 4  # tokens per compressed block key
RPB = BLOCK // RATIO  # compressed rows per kernel block
MB = 1024  # block-table columns at --max-model-len 131072
TOPK = 512  # indexer_budget 2048 / ratio
DIM = 128  # indexer head dim
HEADS = 4  # indexer query heads
N_KEYS = MB * RPB  # 32,768 block keys a table row can address
WIDTH = TOPK * RATIO + RATIO - 1  # 2,051 attended positions per row
NUM_PAGES = 4096  # kernel blocks of the paged cache; the pad block is the next one
PAD_PAGE = NUM_PAGES
RING_ROWS = 8
EPS = 1e-6

# name: (requests [(tokens before the step, real rows)], rows per request). No row of "decode8" completes a block
# (position % 4 != 3); "lattice4" is the verification layout (k = 3) with ragged requests and completing rows.
CASES = {
    "decode8": ([(9, 1), (2045, 1), (2053, 1), (4001, 1), (20002, 1), (65533, 1), (100001, 1), (131069, 1)], 1),
    "lattice4": ([(100000, 4), (3000, 2), (2040, 4), (60001, 3)], 4),
}


def build_case(name: str, seed: int = 20261120) -> SimpleNamespace:
    """The CPU tensors of case ``name`` and the plain-Python facts the oracle needs (``pages``, ``codes``)."""
    specs, t_cols = CASES[name]
    gen = torch.Generator().manual_seed(seed)
    bs = len(specs)
    rows = bs * t_cols
    perm = torch.randperm(NUM_PAGES, generator=gen).tolist()
    cache = torch.zeros((NUM_PAGES + 1) * RPB, DIM, dtype=torch.bfloat16)
    table = torch.full((rows, MB), PAD_PAGE, dtype=torch.int32)
    positions, slot_mapping, lens, pages_of, codes_of = [], [], [], [], []
    cursor = 0
    for b, (ctx, n) in enumerate(specs):
        count = (ctx + max(n, 1) + BLOCK - 1) // BLOCK
        pages = perm[cursor:cursor + count]
        cursor += count
        if len(pages) != count:
            raise ValueError("case geometry: not enough pages")
        pages_of.append(pages)
        lens.append(n)
        # One distinct score per block the request's pages hold, also for the blocks past its visible prefix
        # (a selection that ignores the visibility mask then shows): code = hi * 256 + lo * 256, hi >= 1.
        covered = count * RPB
        codes = torch.randperm(N_KEYS, generator=gen)[:covered] + 256
        codes_of.append(codes.tolist())
        block = torch.arange(covered)
        cache_rows = torch.tensor(pages, dtype=torch.long)[block // RPB] * RPB + block % RPB
        keys = torch.zeros(covered, DIM)
        keys[:, 0] = (codes // 256).float()
        keys[:, 1] = (codes % 256).float() / 256.0
        cache[cache_rows] = keys.to(torch.bfloat16)
        for j in range(t_cols):
            row = b * t_cols + j
            if j < n:
                pos = ctx + j
                table[row, :count] = torch.tensor(pages, dtype=torch.int32)
                positions.append(pos)
                slot_mapping.append(pages[pos // BLOCK] * BLOCK + pos % BLOCK)
            else:  # a padding row of a ragged request: the pad table, position 0, the pad slot
                positions.append(0)
                slot_mapping.append(-1)
    ring_slots = bs + 2  # slot 0 unused, the last one is the ring's scratch slot
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    slots_per_row = [1 + b for b in range(bs) for _ in range(t_cols)]
    freqs = torch.outer(torch.arange(MB * BLOCK, dtype=torch.float32), 1.0 / (1e7**(torch.arange(0, 8, 2) / 8.0)))
    return SimpleNamespace(
        name=name, specs=specs, t_cols=t_cols, bs=bs, rows=rows, lens=lens, pages=pages_of, codes=codes_of,
        cache=cache,
        ring=torch.randn(ring_slots * RING_ROWS, DIM, generator=gen).to(torch.bfloat16),
        q=torch.ones(rows, HEADS, DIM, dtype=torch.bfloat16),
        raw_k=torch.randn(rows, DIM, generator=gen).to(torch.bfloat16),
        positions=torch.tensor(positions, dtype=torch.int64),
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64),
        block_table=table,
        query_start_loc_p=None if t_cols == 1 else torch.tensor(starts, dtype=torch.int32),
        load=torch.tensor(slots_per_row, dtype=torch.int64),
        k_weight=torch.full((DIM, ), -1.0, dtype=torch.bfloat16),
        rope=torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16),
        slot_values=torch.arange((NUM_PAGES + 1) * BLOCK, dtype=torch.float32),
    )


def real_rows(case: SimpleNamespace) -> list[int]:
    return [b * case.t_cols + j for b, (_, n) in enumerate(case.specs) for j in range(n)]


def oracle(case: SimpleNamespace) -> dict[int, dict]:
    """Per real row: the selected ``blocks`` (best first), the attended ``tokens`` and their cache ``slots`` (sorted).

    Plain integer work: a block's rank is its code; a block completed by a row of this step holds the zero key
    (below every code). No float, no ``torch.topk``, none of the plugin's functions.
    """
    out = {}
    for b, (ctx, n) in enumerate(case.specs):
        completed = {(ctx + j) // RATIO for j in range(n) if (ctx + j) % RATIO == RATIO - 1}
        for j in range(n):
            pos = ctx + j
            visible = (pos + 1) // RATIO
            value = [(-1 if blk in completed else case.codes[b][blk]) for blk in range(visible)]
            blocks = sorted(range(visible), key=lambda blk: -value[blk])[:TOPK]
            tokens = [blk * RATIO + r for blk in blocks for r in range(RATIO)]
            tokens += [t for t in range(visible * RATIO, visible * RATIO + RATIO - 1) if t <= pos]
            slots = [case.pages[b][t // BLOCK] * BLOCK + t % BLOCK for t in tokens]
            out[b * case.t_cols + j] = {"blocks": blocks, "tokens": sorted(tokens), "slots": sorted(slots),
                                        "position": pos, "visible": visible}
    return out


def host_scores(case: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[rows, N_KEYS]`` scores of a case without completing rows, and its int64 ``visible`` counts."""
    scores = torch.zeros(case.rows, N_KEYS)
    for b in range(case.bs):
        for j in range(case.t_cols):
            codes = torch.tensor(case.codes[b], dtype=torch.float32)
            scores[b * case.t_cols + j, :codes.numel()] = codes / 256.0
    return scores, (case.positions + 1) // RATIO


def consumer(selected: torch.Tensor, block_table: torch.Tensor, slot_values: torch.Tensor) -> torch.Tensor:
    """What the decode attention does with a selection, ending in fp32: ``[T, W]`` slot numbers, -1 where masked."""
    from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import decode_selected_slots, rows_without_selection
    slots = decode_selected_slots(selected, block_table, BLOCK)
    valid = (selected >= 0) | rows_without_selection(selected)
    got = slot_values.index_select(0, slots.reshape(-1)).view(slots.shape)
    return torch.where(valid, got, torch.full_like(got, -1.0))


def on_device(case: SimpleNamespace, device) -> SimpleNamespace:
    """Fresh copies of the case's tensors on ``device`` and the decode metadata of the step."""
    def dev(t):
        return None if t is None else t.clone().to(device)

    rope = dev(case.rope)
    md = SimpleNamespace(is_prompt=False, query_start_loc_p=dev(case.query_start_loc_p),
                         block_table=dev(case.block_table), block_size=BLOCK)
    return SimpleNamespace(md=md, cache=dev(case.cache), ring=dev(case.ring), q=dev(case.q), raw_k=dev(case.raw_k),
                           positions=dev(case.positions), slot_mapping=dev(case.slot_mapping), load=dev(case.load),
                           k_weight=dev(case.k_weight), slot_values=dev(case.slot_values),
                           cos_sin_at=lambda p: rope.index_select(0, p.clamp_min(0)).chunk(2, dim=-1))


@contextlib.contextmanager
def as_rank(rank: int, world: int, sink: list):
    """Run the sharded indexer as TP rank ``rank`` of ``world`` on one device: its share goes to ``sink``.

    The all-gather is replaced by ``world`` copies of the rank's own share (the right shape, the right values on
    the rank's own rows only); the caller joins the ranks' shares itself. The real all-gather of int32 selections
    was checked on 8 cards in job 00011.
    """
    import vllm_gaudi.models.qwen4_exp.qsa_fused_path as fused
    saved = fused.tp_rank_and_size, fused.all_gather_rows

    def gather(local):
        sink.append(local)
        return torch.cat([local] * world, dim=0)

    fused.tp_rank_and_size, fused.all_gather_rows = (lambda: (rank, world)), gather
    try:
        yield
    finally:
        fused.tp_rank_and_size, fused.all_gather_rows = saved


def decode_selection(case: SimpleNamespace, device, mode: int, shard: int = 0, rank_world=None) -> dict:
    """One decode step of ``HpuQsaIndexer._select_decode`` on ``device``; host copies of what it produced.

    ``mode``/``shard`` are ``VLLM_HPU_QSA_FUSED`` / ``VLLM_HPU_QSA_INDEXER_SHARD``. Returns ``selected`` (int32
    ``[rows, W]``), ``consumer`` (fp32 ``[rows, W]``, built in the same graph as the selection) and ``first_row``;
    with ``rank_world = (rank, world)`` the rows are that rank's share only.
    """
    from vllm_gaudi.models.qwen4_exp.qsa import HpuQsaIndexer
    from vllm_gaudi.models.qwen4_exp.qsa_fused_path import QsaFusedMode, QsaShardMode
    from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import expand_selection

    dev = on_device(case, device)
    indexer = SimpleNamespace(index_head_dim=DIM, compress_ratio=RATIO, block_topk=TOPK,
                              fused_mode=QsaFusedMode(mode), shard_mode=QsaShardMode(shard),
                              key_ring_slots=int(dev.ring.shape[0]) // RING_ROWS, _key_ring_flat=dev.ring,
                              _key_ring_rows=RING_ROWS, k_layernorm=SimpleNamespace(weight=dev.k_weight), eps=EPS,
                              _decode_rows=HpuQsaIndexer._decode_rows)
    sink: list = []
    with (as_rank(rank_world[0], rank_world[1], sink) if rank_world else contextlib.nullcontext()):
        selected = HpuQsaIndexer._select_decode(indexer, dev.q, dev.raw_k, dev.positions, dev.cos_sin_at, dev.md,
                                                dev.load, dev.load, dev.slot_mapping, dev.cache, RPB)
    first_row = 0
    table = dev.md.block_table
    if rank_world:
        if len(sink) != 1:
            raise RuntimeError(f"the bucket did not shard: {len(sink)} all-gathers")
        local = sink[0]  # [share rows, topk] block ids of this rank's requests
        first_row = rank_world[0] * int(local.shape[0])
        mine = torch.arange(first_row, first_row + int(local.shape[0]), dtype=torch.int32, device=local.device)
        selected = expand_selection(local, dev.positions.index_select(0, mine), RATIO)
        table = table.index_select(0, mine)
    probe = consumer(selected, table, dev.slot_values)
    return {"consumer": probe.cpu(), "selected": selected.cpu(), "first_row": first_row}


def compare(case: SimpleNamespace, got: dict, want: dict[int, dict] | None = None) -> dict:
    """Rows of ``got`` (``decode_selection`` or the same keys) that differ from the oracle, read both ways."""
    want = oracle(case) if want is None else want
    first = int(got.get("first_row", 0))
    selected, probe = got.get("selected"), got.get("consumer")
    count = int((selected if selected is not None else probe).shape[0])
    rows = [r for r in real_rows(case) if first <= r < first + count]
    bad_selected, bad_consumer, detail = [], [], None
    for r in rows:
        if selected is not None:
            row = selected[r - first].to(torch.int64)
            have = sorted(row[row >= 0].tolist())
            if have != want[r]["tokens"]:
                bad_selected.append(r)
                if detail is None:
                    detail = (f"row {r} (position {want[r]['position']}, {want[r]['visible']} visible blocks): the host "
                              f"read {len(have)} positions, {len(set(have) & set(want[r]['tokens']))} of the right "
                              f"{len(want[r]['tokens'])}; first got {have[:6]}, want {want[r]['tokens'][:6]}; largest "
                              f"got {have[-1] if have else None}")
        if probe is not None:
            row = probe[r - first]
            have = sorted(int(v) for v in row[row >= 0].tolist())
            if have != want[r]["slots"]:
                bad_consumer.append(r)
    return {"rows": len(rows), "selected_wrong_rows": bad_selected, "consumer_wrong_rows": bad_consumer,
            "detail": detail, "ok": not bad_selected and not bad_consumer}
