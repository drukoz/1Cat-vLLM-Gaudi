# SPDX-License-Identifier: Apache-2.0
"""Validation of the HPU hybrid KV cache capacity model against the user's startup logs
and against vLLM's own page-size accounting (S0 CPU venv, no HPU needed)."""

import os

import kv_capacity

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


GIB = 1024**3


def geom_27b():
    import json
    with open(os.path.join(CONFIG_DIR, "Qwen3.8-27B.json"), encoding="utf-8") as f:
        return kv_capacity.geometry(json.load(f))


def geom_fn():
    import json
    with open(os.path.join(CONFIG_DIR, "Qwen3.8-Flash-Next.json"), encoding="utf-8") as f:
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


def test_flash_next_ple_geometry():
    # Flash-Next has one PLE layer (ple_layer_ids [2]); its state per slot is
    # hc_count * hidden_size * (ple_conv_kernel_size - 1) * ngram_size * 2 bytes, bf16,
    # replicated on every rank. The 27B has no PLE.
    fn = geom_fn()
    assert fn["ple_layers"] == 1
    assert fn["ple_state"] == 4 * 2560 * 3 * 3 * 2 == 184320
    assert geom_27b()["ple_layers"] == 0
    assert geom_27b()["ple_state"] == 0


def test_flash_next_today_both_factors():
    # Flash-Next TP8 BF16 40 GiB, today's allocation: the GDN factor page/(page+gdn_real)
    # and the PLE standard-mamba factor page/(page+ple_state), each truncated as in
    # hpu_worker.py:1128 and :1155. Page 524288 = 4 * 131072, gdn_real 400896, 12 layers.
    page, gdn_real, ple_state, positions = 524288, 400896, 184320, 12
    expected = int(int(40 * GIB * page / (page + gdn_real)) * page / (page + ple_state)) // (positions * page)
    assert expected == 2862  # the spec's worked example
    lay = kv_capacity.layout(geom_fn(), tp=8, kv_bytes=2, budget_gib=40)
    assert (lay["page"], lay["gdn_real"], lay["ple_state"], lay["positions"]) == (page, gdn_real, ple_state,
                                                                                   positions)
    assert lay["today"]["blocks"] == expected


def test_flash_next_pool():
    # The recurrent state pool: slots = max_num_seqs * groups * 3 + cache_slots (32 * 4 * 3
    # + 512 = 896 for Flash-Next: 3 GDN groups + 1 PLE group), pool bytes =
    # (slots + 2) * (positions * gdn_real + ple_layers * ple_state); the attention blocks
    # the budget leaves must exceed today's, and concurrency reports both terms.
    lay = kv_capacity.layout(geom_fn(), tp=8, kv_bytes=2, budget_gib=40)
    assert lay["groups"] == 4
    assert lay["slots"] == 32 * 4 * 3 + 512 == 896
    assert lay["pool"]["bytes"] == 898 * (12 * 400896 + 184320)
    assert lay["pool"]["blocks"] > lay["today"]["blocks"]
    assert lay["pool"]["concurrency"] == min(lay["pool"]["attn_term"], lay["pool"]["slots_term"])
    assert lay["pool"]["attn_term"] == lay["pool"]["blocks"] / 256  # cdiv(131072 * 1024, 524288)
    assert lay["pool"]["slots_term"] == 896 / (4 * 3)
    # --pool-slots 896 is the default for this layout; an override changes the pool blocks.
    override = kv_capacity.layout(geom_fn(), tp=8, kv_bytes=2, budget_gib=40, pool_slots=1024)
    assert override["slots"] == 1024
    assert override["pool"]["blocks"] < lay["pool"]["blocks"]
