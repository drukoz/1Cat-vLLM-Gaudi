# SPDX-License-Identifier: Apache-2.0
"""Rank function for the QSA prefill indexer split test, on the gloo CPU ranks of ``spikes/cpu-tp``.

Plain module (no pytest import): ``run_ranks`` spawns children that re-import it by its module path, and a
rank returns plain Python values only.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("TMPDIR", "/tmp")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_HARNESS_DIR = Path(__file__).resolve().parents[4] / "spikes" / "cpu-tp"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

ENV = "VLLM_HPU_QSA_PREFILL_INDEXER_SPLIT"
BLOCK = 128  # kernel block of the bucket layout
DIM = 16
RING_ROWS = 4


def _indexer(budget: int):
    import torch
    from types import SimpleNamespace

    from vllm_gaudi.models.qwen4_exp.qsa import HpuQsaIndexer

    config = SimpleNamespace(indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=DIM, indexer_compress_ratio=4,
                             indexer_budget=budget, hidden_size=64, rms_norm_eps=1e-6)
    indexer = HpuQsaIndexer(config=config)
    gen = torch.Generator().manual_seed(1234)  # the same weights on every rank (ReplicatedLinear)
    with torch.no_grad():
        for param in indexer.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.5)
    proj = indexer.index_qk_proj
    proj.quant_method.process_weights_after_loading(proj)  # what the loader does (sets the CPU GEMM)
    return indexer


def _prompt(ctx: int, seq: int, length: int, n_blocks: int):
    """One prompt row of ``length`` real tokens in a ``seq``-token bucket after ``ctx`` context tokens."""
    import torch
    from types import SimpleNamespace

    gen = torch.Generator().manual_seed(99)
    hidden = torch.randn(seq, 64, generator=gen)
    real = torch.arange(ctx, ctx + length)
    positions = torch.cat((real, torch.full((seq - length, ), -1, dtype=torch.long)))
    slot_mapping = torch.cat((real, torch.full((seq - length, ), -1, dtype=torch.long))).view(1, seq)
    n_ctx = -(-ctx // BLOCK)
    md = SimpleNamespace(
        is_prompt=True,
        load_indices_tensor=torch.tensor([1], dtype=torch.int32),
        store_indices_tensor=torch.tensor([1], dtype=torch.int32),
        slot_mapping=slot_mapping,
        block_size=BLOCK,
        context_lens_tensor=torch.tensor([ctx], dtype=torch.int32),
        block_list=torch.arange(n_ctx, dtype=torch.int32) if n_ctx else None,
        query_start_loc_p=torch.tensor([0, length], dtype=torch.int32),
        seq_lens_tensor=torch.tensor([length], dtype=torch.int32),
    )
    cache = torch.randn(n_blocks * (BLOCK // 4), DIM, generator=gen)  # context keys; the last row is the scratch row
    return hidden, positions, md, cache


def _cos_sin_at(positions):
    import torch

    freq = torch.tensor([1.0, 0.01])
    angles = positions.clamp_min(0).float()[:, None] * freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def select_rank(rank: int, cases: list, budget: int) -> list:
    """Per case ``(ctx, seq, length)``: the selection with the split off and on, as lists, and the collectives."""
    import torch

    import harness

    out = []
    for ctx, seq, length in cases:
        n_blocks = (ctx + seq) // BLOCK + 2
        results = {}
        for flag in ("0", "1"):
            os.environ[ENV] = flag
            indexer = _indexer(budget)
            hidden, positions, md, cache = _prompt(ctx, seq, length, n_blocks)
            indexer.compressed_key_cache = cache
            indexer.bind_key_ring(torch.zeros(3 * RING_ROWS, DIM), rows=RING_ROWS)
            with harness.record_collectives() as recorded:
                selected = indexer.select(hidden, positions, _cos_sin_at, md)
            results[flag] = {"selected": selected.tolist(), "collectives": [c[0] for c in recorded],
                             "dtype": str(selected.dtype)}
        out.append(results)
    os.environ.pop(ENV, None)
    return out
