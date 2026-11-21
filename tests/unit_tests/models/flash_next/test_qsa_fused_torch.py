# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_QSA_FUSED=1: the restructured sparse-QSA decode data flow equals the torch path on CPU.

Mode 1 (``qsa_fused_path``) scores each request's compressed keys once for all its rows (one
whole-block gather per request instead of a ``[rows, N]`` gather per row, I14) and the decode
attention folds the grouped query heads into the GEMM (I21, G5-1). These tests run the tiny
checkpoint's indexer (its real weights, ``HpuQsaIndexer.select``) in mode 0 and in mode 1 on the same
inputs and require the same selection, bit for bit, and the same side caches:

* decode rows laid out as the HPU runner lays them out (``_prepare_decode_inputs``): ``bs`` requests
  of ``T_cols`` rows, one block-table row per query row (the request's table, or per-token prefixes
  when the step crosses a page), ``[[_PAD_BLOCK_ID]]`` rows for padding, ``query_start_loc_p`` of the
  real lengths, the lattice state indices ``[groups, bs, T_cols]``;
* shuffled kernel-block ids (a fragmented paged cache, not the identity table of ``cpu_model``),
  compressed keys and ring rows filled with seeded random values (the history of earlier steps);
* "dense" rows (at most ``block_topk`` visible blocks: the selection is the causal prefix) and
  "sparse" rows (more visible blocks than ``block_topk``: a true top-k);
* 1 row, 32 rows, the verification lattice at k = 3, 4 and 5, ragged and padding requests, slot moves
  (load slot != store slot), a step that crosses a page, and metadata without ``query_start_loc_p``.

The TP1/2/4/8 sharded indexer (``VLLM_HPU_QSA_INDEXER_SHARD=1``, I18) is in
``test_qsa_fused_torch_tp.py``.
"""

import contextlib
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"


from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_fused_path import (  # noqa: E402
    QsaFusedMode, QsaShardMode, qsa_fused_mode, qsa_shard_mode, request_table_rows, score_request_major)
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import (  # noqa: E402
    decode_selected_slots, rows_without_selection, sparse_decode_attention, sparse_decode_attention_grouped)
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import score_blocks, select_blocks, visible_blocks  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import cpu_model  # noqa: E402
from tests.unit_tests.models.flash_next.qsa_fused_ranks import (  # noqa: E402
    BLOCK, MAX_MODEL_LEN, MB, RATIO, RING_SLOTS, RPB, bind_random_caches, build_indexer, make_requests, real_rows,
    rope_table, runner_metadata)

_ = score_blocks, select_blocks, visible_blocks  # the torch path's ops, named in the docstrings

SEED = 20261120


@contextlib.contextmanager
def cpu_world():
    """A world-1 gloo group and a plain vLLM config: what a standalone indexer's linear layer needs."""
    import tempfile

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel,
                                                 init_distributed_environment, initialize_model_parallel,
                                                 model_parallel_is_initialized)
    if model_parallel_is_initialized():  # the module's tiny model is open: its world-1 group serves
        yield
        return
    fd, init_file = tempfile.mkstemp(dir="/tmp")
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(1, 0, f"file://{init_file}", 0, "gloo")
            initialize_model_parallel(1)
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(init_file)


def indexer_of(model):
    layer = next(m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention))
    return layer, layer.indexer


@pytest.fixture(scope="module")
def tiny_layer(tiny_checkpoint):
    """The tiny checkpoint's sparse QSA layer (bf16, real weights), built once for the module."""
    with cpu_model(tiny_checkpoint, max_model_len=MAX_MODEL_LEN, qsa_mode=QsaMode.SPARSE) as (_, model):
        yield indexer_of(model)


def cos_sin_at(layer):
    return lambda p: layer.rotary_emb.cos_sin_cache[p.clamp_min(0)].chunk(2, dim=-1)








def run_select(indexer, csa, hidden, md, mode: QsaFusedMode):
    indexer.fused_mode = mode
    indexer.shard_mode = QsaShardMode.REPLICATED
    return indexer.select(hidden, md.positions, csa, md)


def compare_modes(tiny_layer, specs, t_cols, *, moves=False, with_qsl=True, seed=SEED, csa=None):
    """Select with mode 0 and mode 1 from the same caches: ``(sel0, sel1, md, caches0, caches1, requests)``."""
    layer, indexer = tiny_layer
    csa = cos_sin_at(layer) if csa is None else csa
    gen = torch.Generator().manual_seed(seed)
    ring_rows = max(4, RATIO * -(-(RATIO + t_cols - 1) // RATIO))
    dtype = next(indexer.parameters()).dtype
    bind_random_caches(indexer, ring_rows, gen, dtype)
    requests = make_requests(gen, specs, moves=moves)
    md = runner_metadata(requests, t_cols, with_qsl=with_qsl)
    hidden = torch.randn(md.positions.numel(), layer.hidden_size, generator=gen).to(dtype)
    comp0, ring0 = indexer.compressed_key_cache.clone(), indexer._key_ring_flat.clone()
    sel0 = run_select(indexer, csa, hidden, md, QsaFusedMode.OFF)
    caches0 = (indexer.compressed_key_cache.clone(), indexer._key_ring_flat.clone())
    indexer.compressed_key_cache.copy_(comp0)
    indexer._key_ring_flat.copy_(ring0)
    sel1 = run_select(indexer, csa, hidden, md, QsaFusedMode.TORCH)
    caches1 = (indexer.compressed_key_cache.clone(), indexer._key_ring_flat.clone())
    return sel0, sel1, md, caches0, caches1, requests




def assert_same(sel0, sel1, md, caches0, caches1, requests, t_cols):
    rows = real_rows(requests, t_cols)
    idx = torch.tensor(rows, dtype=torch.long)
    a, b = sel0.index_select(0, idx), sel1.index_select(0, idx)
    assert torch.equal(a, b), (f"mode 1 selected differently on {int((a != b).any(1).sum())} of {len(rows)} real rows; "
                               f"first differing row {int((a != b).any(1).nonzero()[0])}")
    # Side caches: the writes are the same code in both modes (scratch row and scratch slot excluded).
    assert torch.equal(caches0[0][:-1], caches1[0][:-1])
    ring_rows = caches0[1].shape[0] // RING_SLOTS
    assert torch.equal(caches0[1][:-ring_rows], caches1[1][:-ring_rows])


# ------------------------------------------------------------------------------------------- flags
def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_QSA_FUSED", raising=False)
    monkeypatch.delenv("VLLM_HPU_QSA_INDEXER_SHARD", raising=False)
    assert qsa_fused_mode() is QsaFusedMode.OFF and qsa_shard_mode() is QsaShardMode.REPLICATED
    monkeypatch.setenv("VLLM_HPU_QSA_FUSED", "1")
    monkeypatch.setenv("VLLM_HPU_QSA_INDEXER_SHARD", "1")
    assert qsa_fused_mode() is QsaFusedMode.TORCH and qsa_shard_mode() is QsaShardMode.BY_REQUEST
    monkeypatch.setenv("VLLM_HPU_QSA_FUSED", "7")
    monkeypatch.setenv("VLLM_HPU_QSA_INDEXER_SHARD", "3")
    assert qsa_fused_mode() is QsaFusedMode.OFF and qsa_shard_mode() is QsaShardMode.REPLICATED


def test_request_table_rows():
    lens = torch.tensor([4, 2, 0, 1], dtype=torch.int32)
    assert request_table_rows(lens, 4, 4, torch.device("cpu")).tolist() == [3, 5, 8, 12]
    assert request_table_rows(None, 3, 2, torch.device("cpu")).tolist() == [1, 3, 5]
    assert request_table_rows(lens, 4, 4, torch.device("cpu")).dtype == torch.int32


@pytest.mark.parametrize("t_cols", [1, 4, 6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_request_major_scores_equal_row_major(t_cols, dtype):
    """``score_request_major`` against ``score_blocks`` of every row on its own gathered keys."""
    gen = torch.Generator().manual_seed(SEED + t_cols)
    bs, heads, dim, mb, rpb = 3, 4, 16, 8, 32
    cache = (torch.randn((bs * mb + 1) * rpb, dim, generator=gen) * 2).to(dtype)
    table = torch.randperm(bs * mb, generator=gen).to(torch.int32).view(bs, mb)
    q = torch.randn(bs * t_cols, heads, dim, generator=gen).to(dtype)
    got = score_request_major(q, cache, table, rpb, t_cols)
    for t in range(bs * t_cols):
        keys = cache.view(-1, rpb, dim).index_select(0, table[t // t_cols]).reshape(-1, dim)
        want = score_blocks(q[t:t + 1], keys)[0]
        torch.testing.assert_close(got[t], want, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------------------------- the indexer, modes 0 / 1
def test_one_row_dense_and_sparse(tiny_layer):
    for ctx in (37, 1500, 3001):  # visible 9 and 375 (the causal prefix), 750 (> 512: a true top-k)
        out = compare_modes(tiny_layer, [(ctx, 1)], 1)
        assert_same(*out, 1)
        n_sel = int((out[1][0] >= 0).sum())
        assert n_sel == (512 * RATIO + (ctx + 1) % RATIO if ctx == 3001 else ctx + 1)


def test_thirty_two_rows(tiny_layer):
    """32 single-token requests: 8 long (sparse) contexts, 24 short (dense) ones, shuffled pages."""
    gen = torch.Generator().manual_seed(SEED + 32)
    long_ctx = torch.randint(2100, 3900, (8,), generator=gen).tolist()
    short_ctx = torch.randint(0, 600, (24,), generator=gen).tolist()
    specs = [(c, 1) for c in long_ctx + short_ctx]
    order = torch.randperm(32, generator=gen).tolist()
    out = compare_modes(tiny_layer, [specs[i] for i in order], 1)
    assert_same(*out, 1)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_verification_lattice(tiny_layer, k):
    """bs requests of 1 + k rows: full, ragged and padding requests; dense and sparse contexts."""
    t_cols = 1 + k
    specs = [(3000, t_cols), (2499, t_cols - 2), (126, t_cols), (61, 1), (2900, 0), (777, t_cols)]
    out = compare_modes(tiny_layer, specs, t_cols)
    assert_same(*out, t_cols)


def test_thirty_two_requests_verification(tiny_layer):
    """32 requests x 4 rows (k = 3), the target's verification step shape."""
    gen = torch.Generator().manual_seed(SEED + 128)
    ctx = torch.randint(2050, 3000, (6,), generator=gen).tolist() + torch.randint(0, 700, (26,), generator=gen).tolist()
    lens = torch.randint(0, 5, (32,), generator=gen).tolist()
    out = compare_modes(tiny_layer, list(zip(ctx, lens)), 4)
    assert_same(*out, 4)


def test_step_crossing_a_page(tiny_layer):
    """The rows of one step on two pages (per-token prefix tables): the longest table serves every row."""
    specs = [(BLOCK * 20 - 2, 4), (BLOCK * 3 - 1, 4), (BLOCK * 17 - 3, 3)]
    out = compare_modes(tiny_layer, specs, 4)
    md = out[2]
    assert not torch.equal(md.block_table[0], md.block_table[3]), "the harness must give per-token tables"
    assert_same(*out, 4)


def test_slot_moves(tiny_layer):
    """Prefix-cache slot moves: every request reads ``load`` and writes ``store``."""
    assert_same(*compare_modes(tiny_layer, [(1203, 4), (3002, 3), (55, 4)], 4, moves=True), 4)
    assert_same(*compare_modes(tiny_layer, [(2222, 1), (40, 1)], 1, moves=True), 1)


def test_without_query_start_loc(tiny_layer):
    assert_same(*compare_modes(tiny_layer, [(2100, 4), (901, 4)], 4, with_qsl=False), 4)


def test_fp32_indexer():
    """The fp32 regime of the other Flash-Next parity tests: a standalone fp32 indexer, seeded weights."""
    with cpu_world():
        indexer = build_indexer(SEED + 3, torch.float32)
        table = rope_table(MAX_MODEL_LEN)
        csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
        layer = SimpleNamespace(hidden_size=128)
        assert_same(*compare_modes((layer, indexer), [(3100, 4), (1999, 2), (300, 4)], 4, csa=csa), 4)


def test_mode_one_has_no_row_index(tiny_layer, monkeypatch):
    """Mode 1 never builds the ``[T, N]`` int64 key index: no ``gather`` over the block table."""
    calls = []
    real_gather = torch.Tensor.gather

    def spy(self, dim, index, *args, **kwargs):
        calls.append(tuple(index.shape))
        return real_gather(self, dim, index, *args, **kwargs)

    layer, indexer = tiny_layer
    gen = torch.Generator().manual_seed(SEED + 9)
    bind_random_caches(indexer, 8, gen, next(indexer.parameters()).dtype)
    md = runner_metadata(make_requests(gen, [(2500, 4), (700, 4)]), 4)
    hidden = torch.randn(md.positions.numel(), layer.hidden_size, generator=gen).to(torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "gather", spy)
    run_select(indexer, cos_sin_at(layer), hidden, md, QsaFusedMode.TORCH)
    assert calls == [], f"mode 1 gathered through an index of shape {calls}"
    run_select(indexer, cos_sin_at(layer), hidden, md, QsaFusedMode.OFF)
    assert (8, MB * RPB) in calls, "the spy must see mode 0's [T, N] index"


# ------------------------------------------------------------------------------------ the attention fold
@pytest.mark.parametrize("heads,kv", [(3, 1), (2, 1), (1, 1), (4, 2), (6, 2), (8, 2)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_grouped_attention_equals_repeat(heads, kv, dtype):
    gen = torch.Generator().manual_seed(SEED + heads * 10 + kv)
    bs, width, dim, slots_total = 5, 19, 32, 600
    key_cache = torch.randn(slots_total, kv, dim, generator=gen).to(dtype)
    value_cache = torch.randn(slots_total, kv, dim, generator=gen).to(dtype)
    selected = torch.randint(0, 500, (bs, width), generator=gen).to(torch.int32)
    selected[1, 7:] = -1
    selected[4] = -1  # a padding row: no position at all
    table = torch.arange(8, dtype=torch.int32).repeat(bs, 1)
    slots = decode_selected_slots(selected, table, 128)
    valid = (selected >= 0) | rows_without_selection(selected)
    query = torch.randn(bs, heads * dim, generator=gen).to(dtype)
    want = sparse_decode_attention(query, key_cache, value_cache, slots, valid, heads, kv, dim, dim**-0.5)
    got = sparse_decode_attention_grouped(query, key_cache, value_cache, slots, valid, heads, kv, dim, dim**-0.5)
    assert got.shape == want.shape and got.dtype == want.dtype
    if dtype == torch.bfloat16:
        # The same products and sums; the GEMM forming them (M = G rows instead of 1) may sum in another order
        # in fp32, which the bf16 rounding of the output hides here (bit-exact on this CPU).
        assert torch.equal(got, want), f"max |delta| {float((got.float() - want.float()).abs().max())}"
    else:
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(got).all()
