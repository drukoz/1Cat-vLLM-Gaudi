# SPDX-License-Identifier: Apache-2.0
"""Validation of the HPU hybrid KV cache capacity model against the user's startup logs
and against vLLM's own page-size accounting (S0 CPU venv, no HPU needed)."""

import os

import kv_capacity

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def geom_27b():
    import json
    with open(os.path.join(CONFIG_DIR, "Qwen3.8-27B.json"), encoding="utf-8") as f:
        return kv_capacity.geometry(json.load(f))


def test_run1_log_reproduced():
    # Run 1: 27B, tp 8, bf16 KV, budget 72.98 GiB -> 5293 blocks, block 512, 20.20x at 131072.
    lay = kv_capacity.layout(geom_27b(), tp=8, kv_bytes=2, budget_gib=72.98)
    assert lay["today"]["blocks"] == 5293
    assert lay["block"] == 512
    assert abs(lay["today"]["concurrency"] - 20.20) <= 0.01


def test_run2_log_reproduced():
    # Run 2: 27B, tp 4, bf16 KV, budget 75.36 GiB -> block 896, log concurrency 19.08x (accept 1%).
    lay = kv_capacity.layout(geom_27b(), tp=4, kv_bytes=2, budget_gib=75.36)
    assert lay["block"] == 896
    assert abs(lay["today"]["concurrency"] - 19.08) <= 0.01 * 19.08


def test_vllm_spec_cross_check():
    # The 27B-at-tp-8 page sizes must equal vLLM's own FullAttentionSpec / MambaSpec values.
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    geom = geom_27b()
    lay = kv_capacity.layout(geom, tp=8, kv_bytes=2, budget_gib=72.98)
    attn = FullAttentionSpec(block_size=128,
                             num_kv_heads=lay["kv_heads_rank"],
                             head_size=geom["head_dim"],
                             dtype=torch.bfloat16)
    assert lay["attn_page128"] == attn.page_size_bytes
    mamba = MambaSpec(
        shapes=((geom["linear_num_value_heads"] // 8, geom["linear_key_head_dim"], geom["linear_value_head_dim"]),
                (lay["conv_dim"] // 8, geom["linear_conv_kernel_dim"] - 1)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=-1,
    )
    assert lay["gdn_real"] == mamba.page_size_bytes
