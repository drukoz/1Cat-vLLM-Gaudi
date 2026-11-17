# SPDX-License-Identifier: Apache-2.0
"""CPU equivalence tests for the QSA sparse selection ops (M2-T1).

Checks ``vllm_gaudi/models/qwen4_exp/qsa_sparse_ops.py`` against the HF
``Qwen4ExpTextQSAIndexer`` oracle on the tiny config:

1. ``test_prefill_mask_matches_hf``: the whole-prefill selection mask for
   300- and 2600-token sequences, two seeds; exact ``torch.equal`` on every
   row whose 512/513 block-score boundary is a strict inequality, and on
   exact-tie rows (per CTO decision ``97c84a7b``) the selected-block sets
   must agree outside the boundary-tie group,
2. ``test_selection_width``: the 2051-wide selection row and ``ring_rows``,
3. ``test_replay_matches_hf``: token-by-token decode selection through the
   raw-key ring and the compressed-key cache equals HF's rows for tokens
   2050..2599,
4. ``test_compressed_rows_equal_prefill``: the cache built by the replay equals
   the batch-computed block keys.

HF modules are built from the explicit classes, never through
``AutoConfig``/``CONFIG_MAPPING`` (vLLM replaces that mapping in-process).
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
)

from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import (  # noqa: E402
    compress_block_keys,
    compressed_row_slots,
    expand_selection,
    index_queries,
    ring_block_rows,
    ring_rows,
    ring_write,
    score_blocks,
    select_blocks,
    selection_mask,
    visible_blocks,
)

FIXTURE = Path(__file__).resolve().parent / "tiny_config.json"


def _hf_setup(L, seed):
    """HF oracle on the tiny config: (indexer, cos_full, sin_full, hidden,
    ref_mask) as in the task spec; ``ref_mask`` is bool ``[L, L]``, True where
    the query row attends the column."""
    cfg = Qwen4ExpTextConfig(**json.loads(FIXTURE.read_text())["text_config"])
    torch.manual_seed(seed)
    indexer = Qwen4ExpTextQSAIndexer(cfg, layer_idx=3).float().eval()
    # HF inits these weights to 0 (1 + 0 = 1): randomize so the norm weight is on the tested path.
    indexer.q_layernorm.weight.data.normal_(0.0, 0.1)
    indexer.k_layernorm.weight.data.normal_(0.0, 0.1)
    rotary = Qwen4ExpTextRotaryEmbedding(cfg).float().eval()
    hidden = torch.randn(1, L, cfg.hidden_size)
    positions = torch.arange(L)
    cos_full, sin_full = rotary(hidden, positions[None])  # each [1, L, rotary_dim], the two halves equal
    causal = torch.ones(L, L, dtype=torch.bool).tril()[None, None]  # True = visible (HF's bool-mask convention)
    ref_mask = indexer(hidden, (cos_full, sin_full), causal, None)[0, 0]
    return indexer, cos_full, sin_full, hidden, ref_mask


def _project(indexer, hidden):
    """index_qk_proj split: q [L, H, D] and raw_k [L, D] from hidden [1, L, hidden]."""
    qk = hidden[0] @ indexer.index_qk_proj.weight.T
    q, raw_k = torch.split(qk, [indexer.index_n_heads * indexer.index_head_dim,
                                indexer.index_kv_heads * indexer.index_head_dim], dim=-1)
    return q.view(q.shape[0], indexer.index_n_heads, -1), raw_k.view(q.shape[0], -1)


def _query_states(indexer, cos_full, sin_full, hidden):
    """Indexer projections for the whole sequence: roped queries [L, H, D] and raw keys [L, D]."""
    r2 = cos_full.shape[-1] // 2
    q, raw_k = _project(indexer, hidden)
    q = index_queries(q, indexer.q_layernorm.weight, indexer.q_layernorm.eps, cos_full[0, :, :r2],
                      sin_full[0, :, :r2])
    return q, raw_k


def _block_keys(indexer, cos_full, sin_full, raw_k, n_blocks):
    """Batch-compressed keys [N, D] for the first n_blocks complete blocks."""
    ratio = indexer.compress_ratio
    r2 = cos_full.shape[-1] // 2
    return compress_block_keys(raw_k[:n_blocks * ratio].view(n_blocks, ratio, -1), indexer.k_layernorm.weight,
                               indexer.k_layernorm.eps, cos_full[0, 0:n_blocks * ratio:ratio, :r2],
                               sin_full[0, 0:n_blocks * ratio:ratio, :r2])


def _ours_prefill(indexer, cos_full, sin_full, hidden):
    """Our prefill selection: (mask [L, L] bool, selected [L, 2051] int32,
    blocks [L, block_topk] int32, scores [L, L // ratio] fp32)."""
    ratio = indexer.compress_ratio
    L = hidden.shape[1]
    positions = torch.arange(L)
    q, raw_k = _query_states(indexer, cos_full, sin_full, hidden)
    keys = _block_keys(indexer, cos_full, sin_full, raw_k, L // ratio)
    scores = score_blocks(q, keys)
    blocks = select_blocks(scores, visible_blocks(positions, ratio), indexer.block_topk)
    selected = expand_selection(blocks, positions, ratio)
    return selection_mask(selected, L), selected, blocks, scores


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("L", [300, 2600])
def test_prefill_mask_matches_hf(L, seed):
    indexer, cos_full, sin_full, hidden, ref_mask = _hf_setup(L, seed)
    ours, selected, blocks, scores = _ours_prefill(indexer, cos_full, sin_full, hidden)
    ratio, topk = indexer.compress_ratio, indexer.block_topk

    # Per CTO decision 97c84a7b: rows whose k-th/(k+1)-th visible block scores are strictly
    # ordered must equal HF exactly; rows with an exact boundary tie may only swap blocks
    # scored exactly at the boundary value (topk tie order is a layout artifact, not a defect).
    tie_rows = []
    min_gap = None
    for t in range(L):
        nc = (t + 1) // ratio
        if nc > topk:  # a (k+1)-th visible column exists only on these rows
            srt = torch.sort(scores[t, :nc].detach(), descending=True).values
            gap = float(srt[topk - 1] - srt[topk])
            min_gap = gap if min_gap is None else min(min_gap, gap)
        else:  # every visible block is selected: no tie ambiguity
            gap, srt = float("inf"), None
        if gap == 0.0:
            tie_rows.append(t)
            boundary = scores[t, :nc] == srt[topk - 1]  # exact fp32 equality at the k-th score
            ours_b = set(blocks[t][blocks[t] >= 0].tolist())
            hf_b = set(torch.nonzero(ref_mask[t, :nc * ratio].view(nc, ratio).all(-1)).flatten().tolist())
            boundary_b = set(torch.nonzero(boundary).flatten().tolist())
            assert ours_b ^ hf_b <= boundary_b, f"row {t}: block sets diverge outside the boundary-tie group"
        else:
            assert torch.equal(ours[t], ref_mask[t]), f"row {t}: gap > 0 rows must equal HF exactly"

    if not tie_rows:  # no boundary ties anywhere: the whole tensor must equal HF exactly
        assert torch.equal(ours, ref_mask)
    gap_txt = "N/A (no row has a 513rd visible block)" if min_gap is None else f"{min_gap}"
    print(f"prefill L={L} seed={seed}: tie rows {len(tie_rows)}, min 512/513 boundary gap: {gap_txt}")

    positions = torch.arange(L)
    assert bool(torch.all((selected == -1) | (selected <= positions[:, None])))
    expected = torch.tensor([min(indexer.block_topk, (t + 1) // 4) * 4 + (t + 1) % 4 for t in range(L)])
    assert torch.equal(ours.sum(-1).to(torch.int64), expected)


def test_selection_width():
    blocks = torch.full((3, 512), -1, dtype=torch.int32)
    selected = expand_selection(blocks, torch.tensor([100, 200, 2599]), 4)
    assert selected.shape == (3, 2051)  # 512 * 4 + 3
    assert ring_rows(4, 0) == 4
    assert ring_rows(4, 3) == 8


_REPLAY: dict = {}


def _replay(L, seed, P):
    """Token-by-token decode selection for p in P..L-1 through the ring + cache.

    Prefills the first P tokens with the prefill path and maintains the decode
    state: the compressed-key cache ``comp`` (identity slot mapping, one row
    per complete block) and the raw-key ring (slot 1) holding the P % ratio
    tail keys. Returns the state and the per-row comparison against HF.
    """
    indexer, cos_full, sin_full, hidden, ref_mask = _hf_setup(L, seed)
    ratio = indexer.compress_ratio
    topk = indexer.block_topk
    D = indexer.index_head_dim
    r2 = cos_full.shape[-1] // 2
    k_norm_w, eps_k = indexer.k_layernorm.weight, indexer.k_layernorm.eps

    q, raw_k = _query_states(indexer, cos_full, sin_full, hidden)
    prefill_keys = _block_keys(indexer, cos_full, sin_full, raw_k, L // ratio)

    comp = torch.zeros(L // ratio, D)
    slots_p = compressed_row_slots(torch.arange(P, dtype=torch.int64), ratio)  # identity slot = position
    comp[slots_p[ratio - 1::ratio]] = prefill_keys[:P // ratio]  # one row per complete block
    ring = torch.zeros(2, ring_rows(ratio, 0), D)
    tail_pos = torch.arange(P - (P % ratio), P, dtype=torch.int64)
    ring_write(ring, torch.ones(tail_pos.shape, dtype=torch.int64), tail_pos, raw_k[tail_pos])

    slot = torch.ones(1, dtype=torch.int64)
    mismatches = []
    t0 = time.perf_counter()
    for p in range(P, L):
        pos_p = torch.tensor([p], dtype=torch.int64)
        raw_k_p = raw_k[p:p + 1]
        if p % ratio == ratio - 1:  # complete the block from the ring first, then write the raw key
            rows = ring_block_rows(ring, slot, pos_p, raw_k_p, ratio)
            b = p // ratio
            comp[b:b + 1] = compress_block_keys(rows, k_norm_w, eps_k, cos_full[0, ratio * b:ratio * b + 1, :r2],
                                                sin_full[0, ratio * b:ratio * b + 1, :r2])
        ring_write(ring, slot, pos_p, raw_k_p)

        n_vis = (p + 1) // ratio
        scores = score_blocks(q[p:p + 1], comp[:n_vis])
        blocks = select_blocks(scores, visible_blocks(pos_p, ratio), topk)
        selected = expand_selection(blocks, pos_p, ratio)
        if not torch.equal(selection_mask(selected, L)[0], ref_mask[p]):
            mismatches.append(p)
    elapsed = time.perf_counter() - t0
    return {"comp": comp, "prefill_keys": prefill_keys, "mismatches": mismatches, "elapsed": elapsed}


def _run_replay():
    if "state" not in _REPLAY:
        _REPLAY["state"] = _replay(2600, 0, 2050)
    return _REPLAY["state"]


def test_replay_matches_hf():
    state = _run_replay()
    print(f"replay loop wall time: {state['elapsed']:.3f} s for {2600 - 2050} rows")
    assert state["mismatches"] == []
    assert state["elapsed"] < 60.0


def test_compressed_rows_equal_prefill():
    state = _run_replay()
    torch.testing.assert_close(state["comp"][:650], state["prefill_keys"], atol=1e-6, rtol=0)
