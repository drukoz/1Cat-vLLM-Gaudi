# SPDX-License-Identifier: Apache-2.0
"""The references of the fused QSA ops equal the torch path of the indexer and the decode attention (CPU).

``qsa_index_prepare_ref`` against ``HpuQsaIndexer._select_decode``'s writes and ``index_queries`` (the side caches
outside the scratch slot and row, the queries), and ``qsa_index_emit_ref`` against ``expand_selection`` +
``decode_selected_slots`` + the valid mask, on the runner's decode layout (tiny-model indexer geometry, seeded
weights): 1 and 32 rows, the verification lattice at k = 3 and 5, ragged and padding requests, slot moves.
Then the mode-2 indexer (the references registered as the ops' CPU kernels) equals mode 1 end to end.
"""

import contextlib
import os
import tempfile

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("vllm")

import tests.unit_tests.qsa_fused.qsa_ops  # noqa: E402,F401  the references as the CPU kernels
from tests.unit_tests.models.flash_next.qsa_fused_ranks import (  # noqa: E402
    MAX_MODEL_LEN, RATIO, bind_random_caches, build_indexer, make_requests, real_rows, rope_table, runner_metadata)
from tests.unit_tests.qsa_fused.qsa_reference import qsa_index_emit_ref, qsa_index_prepare_ref, rows_table  # noqa: E402
from vllm_gaudi.models.qwen4_exp import qsa_fused_path  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_fused_path import QsaDecodeSelection, QsaFusedMode, QsaShardMode  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import decode_selected_slots, rows_without_selection  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import expand_selection, index_queries  # noqa: E402

SEED = 20261121
CASES = {
    "one_row_sparse": ([(3001, 1)], 1),
    "thirty_two_rows": ([(c, 1) for c in (3, 7, 2050, 3999, 1, 15, 2047, 2051, 3071, 8, 9, 10, 12, 13, 14, 16, 17, 18,
                                          20, 21, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)], 1),
    "verify_k3": ([(3000, 4), (2499, 2), (126, 4), (61, 1), (2900, 0), (777, 4)], 4),
    "verify_k5": ([(3000, 6), (2051, 4), (9, 6), (100, 0)], 6),
}


@contextlib.contextmanager
def cpu_world():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel,
                                                 init_distributed_environment, initialize_model_parallel)
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


@pytest.fixture(scope="module")
def world():
    with cpu_world():
        yield


def setup_case(name, moves=False):
    specs, t_cols = CASES[name]
    indexer = build_indexer(SEED, torch.bfloat16)
    gen = torch.Generator().manual_seed(SEED + len(name))
    ring_rows = max(4, RATIO * -(-(RATIO + t_cols - 1) // RATIO))
    bind_random_caches(indexer, ring_rows, gen, torch.bfloat16)
    requests = make_requests(gen, specs, moves=moves)
    md = runner_metadata(requests, t_cols)
    hidden = torch.randn(md.positions.numel(), 128, generator=gen).to(torch.bfloat16)
    table = rope_table(MAX_MODEL_LEN).to(torch.bfloat16)
    return indexer, md, hidden, table, requests, t_cols


@pytest.mark.parametrize("moves", [False, True])
@pytest.mark.parametrize("name", sorted(CASES))
def test_prepare_reference_equals_torch_path(world, name, moves):
    indexer, md, hidden, table, requests, t_cols = setup_case(name, moves)
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
    H, D = indexer.index_n_heads, indexer.index_head_dim
    ring0, cache0 = indexer._key_ring_flat.clone(), indexer.compressed_key_cache.clone()
    indexer.fused_mode = QsaFusedMode.OFF
    indexer.select(hidden, md.positions, csa, md)  # the torch path writes the caches
    ring_t, cache_t = indexer._key_ring_flat.clone(), indexer.compressed_key_cache.clone()
    qk, _ = indexer.index_qk_proj(hidden)
    T = qk.shape[0]
    load = indexer._state_indices(md, "load_indices_tensor")
    store = indexer._state_indices(md, "store_indices_tensor")
    bs, lens, T_cols, valid, load_rows, store_rows = indexer._decode_rows(md, md.positions, load, store)
    rows = rows_table(md.positions, md.slot_mapping.reshape(-1)[:T], load_rows, store_rows, valid)
    assert torch.equal(rows, qsa_fused_path.rows_table(md.positions, md.slot_mapping.reshape(-1)[:T], load_rows,
                                                       store_rows, valid))
    ring_r, cache_r = ring0.clone(), cache0.clone()
    q_ref = qsa_index_prepare_ref(ring_r, cache_r, qk, rows, indexer.q_layernorm.weight, indexer.k_layernorm.weight,
                                  table, indexer.eps, H, indexer._key_ring_rows, RATIO)
    cos, sin = csa(md.positions)
    q_torch = index_queries(qk[:, :H * D].view(T, H, D), indexer.q_layernorm.weight, indexer.eps, cos, sin)
    assert torch.equal(q_ref, q_torch.reshape(T, H * D))
    R = indexer._key_ring_rows
    assert torch.equal(ring_r[:-R], ring_t[:-R]), "ring rows outside the scratch slot differ"
    assert torch.equal(cache_r[:-1], cache_t[:-1]), "compressed rows outside the scratch row differ"


@pytest.mark.parametrize("name", sorted(CASES))
def test_emit_reference_equals_torch_path(world, name):
    indexer, md, hidden, table, requests, t_cols = setup_case(name)
    gen = torch.Generator().manual_seed(7)
    T = md.positions.numel()
    blocks = torch.randint(-1, 900, (T, indexer.block_topk), generator=gen, dtype=torch.int32)
    slots, valid = qsa_index_emit_ref(blocks, md.positions, md.block_table, RATIO, 128)
    selected = expand_selection(blocks, md.positions, RATIO)
    assert torch.equal(slots, decode_selected_slots(selected, md.block_table, 128).to(torch.int32))
    assert torch.equal(valid.bool(), (selected >= 0) | rows_without_selection(selected))


@pytest.mark.parametrize("name", sorted(CASES))
def test_mode_two_equals_mode_one(world, name, monkeypatch):
    """The mode-2 indexer (references as CPU kernels) selects what mode 1 selects and leaves the same caches."""
    monkeypatch.setattr(qsa_fused_path, "KERNEL_DEVICE_TYPES", ["hpu", "cpu"])
    indexer, md, hidden, table, requests, t_cols = setup_case(name)
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
    ring0, cache0 = indexer._key_ring_flat.clone(), indexer.compressed_key_cache.clone()
    indexer.fused_mode, indexer.shard_mode = QsaFusedMode.TORCH, QsaShardMode.REPLICATED
    selected = indexer.select(hidden, md.positions, csa, md, cos_sin=table)
    ring1, cache1 = indexer._key_ring_flat.clone(), indexer.compressed_key_cache.clone()
    indexer._key_ring_flat.copy_(ring0)
    indexer.compressed_key_cache.copy_(cache0)
    indexer.fused_mode = QsaFusedMode.KERNELS
    got = indexer.select(hidden, md.positions, csa, md, cos_sin=table)
    assert isinstance(got, QsaDecodeSelection), indexer.kernels_off_reason
    rows = torch.tensor(real_rows(requests, t_cols))
    want_slots = decode_selected_slots(selected, md.block_table, 128).to(torch.int32)
    want_valid = ((selected >= 0) | rows_without_selection(selected)).to(torch.int32)
    assert torch.equal(got.slots.index_select(0, rows), want_slots.index_select(0, rows))
    assert torch.equal(got.valid.index_select(0, rows), want_valid.index_select(0, rows))
    R = indexer._key_ring_rows
    assert torch.equal(indexer._key_ring_flat[:-R], ring1[:-R])
    assert torch.equal(indexer.compressed_key_cache[:-1], cache1[:-1])


def test_mode_two_falls_back_off_hpu(world):
    """Without a kernel for the device (no stand-in registered as a kernel device), mode 2 runs mode 1."""
    indexer, md, hidden, table, requests, t_cols = setup_case("verify_k3")
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
    indexer.fused_mode = QsaFusedMode.KERNELS
    got = indexer.select(hidden, md.positions, csa, md, cos_sin=table)
    assert not isinstance(got, QsaDecodeSelection)
    assert "device cpu" in indexer.kernels_off_reason


@pytest.mark.parametrize("ties", [False, True])
def test_topk_reference_equals_select_blocks(ties):
    """threshold + select (mode 3) == select_blocks as sets; with ties at the cut, the lowest blocks win."""
    from tests.unit_tests.qsa_fused.qsa_reference import qsa_index_topk_ref
    from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import select_blocks, visible_blocks
    gen = torch.Generator().manual_seed(11)
    T, N = 12, 1024
    if ties:  # relu sums: exact zeros, and a coarse grid of equal scores at the cut
        scores = (torch.relu(torch.randn(T, N, generator=gen)) * 4).round() / 12
    else:  # continuous and positive: no two scores are equal
        scores = torch.rand(T, N, generator=gen) + 0.01
    positions = torch.tensor([3, 2047, 2051, 4095, 100, 0, -1, 3000, 2500, 2048, 4000, 1999])
    got = qsa_index_topk_ref(scores, positions, RATIO, 512)
    want = select_blocks(scores, visible_blocks(positions, RATIO), 512)
    keys = scores.view(torch.int32) & 0x7FFFFFFF
    for t in range(T):
        g = {int(b) for b in got[t] if b >= 0}
        w = {int(b) for b in want[t] if b >= 0}
        assert len(g) == len(w)
        assert got[t][got[t] >= 0].tolist() == sorted(g), "ascending order"
        if not ties:
            assert g == w, t
        else:
            visible = int(visible_blocks(positions[t:t + 1], RATIO)[0])
            if visible > 512:  # the same keys above the cut; at the cut the lowest blocks
                k = keys[t, :visible]
                theta = int(torch.sort(k, descending=True).values[511])
                above = {n for n in range(visible) if int(k[n]) > theta}
                at_cut = [n for n in range(visible) if int(k[n]) == theta][:512 - len(above)]
                assert g == above | set(at_cut), t


@pytest.mark.parametrize("name", sorted(CASES))
def test_mode_three_selects_the_same_set(world, name, monkeypatch):
    """Mode 3 (the top-k kernels' references as CPU kernels) selects mode 2's blocks except among scores equal at
    the cut (the lowest blocks win there; torch.topk's choice is open), and K4 maps them the same way."""
    monkeypatch.setattr(qsa_fused_path, "KERNEL_DEVICE_TYPES", ["hpu", "cpu"])
    recorded = {"scores": [], "blocks": []}
    real_score, real_select, real_topk = (qsa_fused_path.score_request_major, qsa_fused_path.select_blocks,
                                          qsa_fused_path.topk_kernels)

    def score(*a, **k):
        out = real_score(*a, **k)
        recorded["scores"].append(out)
        return out

    def select(*a, **k):
        out = real_select(*a, **k)
        recorded["blocks"].append(out)
        return out

    def topk(*a, **k):
        out = real_topk(*a, **k)
        recorded["blocks"].append(out)
        return out

    monkeypatch.setattr(qsa_fused_path, "score_request_major", score)
    monkeypatch.setattr(qsa_fused_path, "select_blocks", select)
    monkeypatch.setattr(qsa_fused_path, "topk_kernels", topk)
    indexer, md, hidden, table, requests, t_cols = setup_case(name)
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
    ring0, cache0 = indexer._key_ring_flat.clone(), indexer.compressed_key_cache.clone()
    indexer.fused_mode, indexer.shard_mode = QsaFusedMode.KERNELS, QsaShardMode.REPLICATED
    two = indexer.select(hidden, md.positions, csa, md, cos_sin=table)
    indexer._key_ring_flat.copy_(ring0)
    indexer.compressed_key_cache.copy_(cache0)
    indexer.fused_mode = QsaFusedMode.KERNELS_TOPK
    three = indexer.select(hidden, md.positions, csa, md, cos_sin=table)
    assert isinstance(two, QsaDecodeSelection) and isinstance(three, QsaDecodeSelection)
    scores2, scores3 = recorded["scores"]
    assert torch.equal(scores2, scores3)
    blocks2, blocks3 = recorded["blocks"]
    keys = scores2.view(torch.int32) & 0x7FFFFFFF
    visible = ((md.positions + 1) // RATIO).clamp(0, scores2.shape[1])
    tied_rows = 0
    for t in real_rows(requests, t_cols):
        a = {int(b) for b in blocks2[t] if b >= 0}
        b = {int(x) for x in blocks3[t] if x >= 0}
        assert len(a) == len(b), t
        if a != b:  # only blocks whose score equals the score at the cut may differ
            k = keys[t, :int(visible[t])]
            theta = int(torch.sort(k, descending=True).values[indexer.block_topk - 1])
            assert all(int(k[n]) == theta for n in a ^ b), t
            tied_rows += 1
    assert tied_rows < len(real_rows(requests, t_cols))
