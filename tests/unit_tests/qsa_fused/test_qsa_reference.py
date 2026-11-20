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
