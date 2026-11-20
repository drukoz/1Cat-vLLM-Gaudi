# SPDX-License-Identifier: Apache-2.0
"""Rank function and decode-layout helpers for the sharded QSA indexer tests (``VLLM_HPU_QSA_INDEXER_SHARD``).

Plain module (no pytest import): ``harness.run_ranks`` spawns children that re-import it by its module
path, and every value a rank returns is a numpy array or a plain Python value (see ``tp_ranks``).

``shard_select_rank`` builds a standalone ``HpuQsaIndexer`` (the tiny checkpoint's indexer geometry,
seeded weights identical on every rank), binds seeded random side caches, and runs
``HpuQsaIndexer.select`` on the HPU runner's padded decode layout in three settings from the same
caches: mode 0 (the torch path), mode 1 replicated, mode 1 sharded by request. It records the TP
collectives of each call.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("TMPDIR", "/tmp")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[4] / "spikes" / "cpu-tp"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import harness  # noqa: E402

BLOCK = 128  # attention kernel block (block_size)
RATIO = 4
RPB = BLOCK // RATIO
MAX_MODEL_LEN = 4096
MB = MAX_MODEL_LEN // BLOCK
NUM_PAGES = 400  # kernel blocks of the paged cache; the pad block is the last one
PAD_BLOCK = NUM_PAGES
RING_SLOTS = 80  # GDN state rows; the last one is the ring's scratch slot


def bind_random_caches(indexer, ring_rows: int, gen, dtype) -> None:
    """Seeded history: every compressed row and ring row holds a random key (scale of the real keys)."""
    import torch
    dim = indexer.index_head_dim
    indexer.compressed_key_cache = (torch.randn((NUM_PAGES + 1) * RPB, dim, generator=gen) * 2).to(dtype)
    indexer.bind_key_ring((torch.randn(RING_SLOTS * ring_rows, dim, generator=gen)).to(dtype), ring_rows)
    indexer.ring_group_idx = torch.tensor(0, dtype=torch.long)


def runner_metadata(requests: list[dict], t_cols: int, *, with_qsl: bool = True) -> SimpleNamespace:
    """The HPU runner's padded decode metadata for ``requests`` (``_prepare_decode_inputs``).

    Each request: ``ctx`` (tokens before the step), ``n`` (real rows, ``0 <= n <= t_cols``), ``pages``
    (its kernel blocks in logical order), ``load``/``store`` (state slots). One block-table row per
    query row: the request's whole table when the step stays on one page, else the per-token prefix
    ``pages[:(ctx + j + B) // B]``; padding rows get ``[[_PAD_BLOCK_ID]]``, slot ``-1``, position 0.
    ``query_start_loc_p`` holds the cumulative real lengths (``_prepare_decode_inputs``).
    """
    import torch
    bs = len(requests)
    rows = bs * t_cols
    table = torch.full((rows, MB), PAD_BLOCK, dtype=torch.int32)
    slot_mapping, positions, lens = [], [], []
    for b, req in enumerate(requests):
        ctx, n, pages = req["ctx"], req["n"], req["pages"]
        lens.append(n)
        n_blocks = (ctx + n + BLOCK - 1) // BLOCK
        first_page_count = (ctx + BLOCK) // BLOCK
        for j in range(t_cols):
            row = b * t_cols + j
            if j < n:
                pos = ctx + j
                count = n_blocks if first_page_count == n_blocks else (ctx + j + BLOCK) // BLOCK
                table[row, :count] = torch.tensor(pages[:count], dtype=torch.int32)
                slot_mapping.append(pages[pos // BLOCK] * BLOCK + pos % BLOCK)
                positions.append(pos)
            else:
                slot_mapping.append(-1)
                positions.append(0)
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    if t_cols == 1:
        load = torch.tensor([[r["load"] for r in requests]], dtype=torch.int64)
        store = torch.tensor([[r["store"] for r in requests]], dtype=torch.int64)
    else:
        load = torch.tensor([[[r["load"]] * t_cols for r in requests]], dtype=torch.int64)
        store = torch.tensor([[[r["store"]] * t_cols for r in requests]], dtype=torch.int64)
    return SimpleNamespace(is_prompt=False,
                           load_indices_tensor=load,
                           store_indices_tensor=store,
                           query_start_loc_p=torch.tensor(starts, dtype=torch.int32) if with_qsl else None,
                           slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64),
                           block_table=table,
                           block_size=BLOCK,
                           context_lens_tensor=None,
                           block_list=None,
                           positions=torch.tensor(positions, dtype=torch.int64))


def make_requests(gen, specs: list[tuple[int, int]], *, moves: bool = False) -> list[dict]:
    """Requests with ``(ctx, n)`` from ``specs``, disjoint shuffled pages and distinct state slots."""
    import torch
    perm = torch.randperm(NUM_PAGES, generator=gen).tolist()
    out, cursor = [], 0
    for i, (ctx, n) in enumerate(specs):
        need = (ctx + max(n, 1) + BLOCK - 1) // BLOCK
        pages = perm[cursor:cursor + need]
        cursor += need
        if len(pages) != need:
            raise ValueError("test geometry: not enough pages")
        load = 1 + i
        store = (1 + len(specs) + i) if moves else load
        out.append({"ctx": ctx, "n": n, "pages": pages, "load": load, "store": store})
    if 1 + (2 if moves else 1) * len(specs) >= RING_SLOTS - 1:
        raise ValueError("test geometry: not enough ring slots")
    return out


def real_rows(requests, t_cols) -> list[int]:
    return [b * t_cols + j for b, r in enumerate(requests) for j in range(r["n"])]


def tiny_indexer_config() -> SimpleNamespace:
    """The tiny checkpoint's indexer geometry (``reports/qsa-fused/tiny_ckpt/config.json``)."""
    return SimpleNamespace(indexer_n_heads=4,
                           indexer_kv_heads=1,
                           indexer_head_dim=16,
                           indexer_compress_ratio=4,
                           indexer_budget=2048,
                           rms_norm_eps=1e-6,
                           hidden_size=128)


def rope_table(positions: int, rotary_dim: int = 8, theta: float = 1e7):
    """A ``cos_sin_cache`` ``[positions, rotary_dim]`` (cosines then sines), as vLLM's RoPE builds it."""
    import torch
    inv = 1.0 / (theta**(torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def build_indexer(seed: int, dtype):
    """A standalone ``HpuQsaIndexer`` with seeded weights (the same on every rank)."""
    import torch
    from vllm_gaudi.models.qwen4_exp.qsa import HpuQsaIndexer
    indexer = HpuQsaIndexer(config=tiny_indexer_config(), prefix="model.layers.3.self_attn.indexer")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in indexer.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (0.08 if p.dim() == 2 else 0.1))
    indexer = indexer.to(dtype)
    # What the model loader does after loading (the CPU GEMM binds its kernel here).
    for module in indexer.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and hasattr(method, "process_weights_after_loading"):
            method.process_weights_after_loading(module)
    return indexer


CASES = {
    # name: (specs [(ctx, real rows)], t_cols)
    "decode8": ([(3001, 1), (37, 1), (2600, 1), (1500, 1), (3800, 1), (0, 1), (2051, 1), (900, 1)], 1),
    "decode16": ([(c, 1) for c in (3100, 12, 2222, 700, 3999, 2052, 64, 1990, 3333, 5, 2400, 1024, 3700, 300, 2100,
                                   1)], 1),
    "verify8x4": ([(3000, 4), (2499, 2), (126, 4), (61, 1), (2900, 0), (777, 4), (3500, 3), (2049, 4)], 4),
    "decode6": ([(3001, 1), (37, 1), (2600, 1), (1500, 1), (3800, 1), (5, 1)], 1),
}


def shard_select_rank(rank: int, tp: int, case: str, seed: int) -> dict:
    """Run one case on this rank in mode 0, mode 1 replicated and mode 1 sharded; return selections + collectives."""
    import numpy as np
    import torch

    from vllm_gaudi.models.qwen4_exp.qsa_fused_path import QsaFusedMode, QsaShardMode

    specs, t_cols = CASES[case]
    dtype = torch.bfloat16
    indexer = build_indexer(seed, dtype)
    gen = torch.Generator().manual_seed(seed + 1)
    ring_rows = max(4, RATIO * -(-(RATIO + t_cols - 1) // RATIO))
    bind_random_caches(indexer, ring_rows, gen, dtype)
    requests = make_requests(gen, specs)
    md = runner_metadata(requests, t_cols)
    hidden = torch.randn(md.positions.numel(), 128, generator=gen).to(dtype)
    table = rope_table(MAX_MODEL_LEN).to(dtype)
    csa = lambda p: table[p.clamp_min(0)].chunk(2, dim=-1)  # noqa: E731
    snap = (indexer.compressed_key_cache.clone(), indexer._key_ring_flat.clone())
    out = {}
    for name, mode, shard in (("mode0", QsaFusedMode.OFF, QsaShardMode.REPLICATED),
                              ("replicated", QsaFusedMode.TORCH, QsaShardMode.REPLICATED),
                              ("sharded", QsaFusedMode.TORCH, QsaShardMode.BY_REQUEST)):
        indexer.compressed_key_cache.copy_(snap[0])
        indexer._key_ring_flat.copy_(snap[1])
        indexer.fused_mode, indexer.shard_mode = mode, shard
        with harness.record_collectives() as recorded:
            sel = indexer.select(hidden, md.positions, csa, md)
        out[name] = sel.numpy().copy()
        out[name + "_collectives"] = [(m, tuple(s), str(d)) for m, s, d in recorded]
        out[name + "_cache"] = indexer.compressed_key_cache[:-1].float().numpy().copy()
    out["real_rows"] = real_rows(requests, t_cols)
    out["rank"] = rank
    return {k: (np.asarray(v) if k.endswith("_cache") else v) for k, v in out.items()}
