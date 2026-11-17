# SPDX-License-Identifier: Apache-2.0
"""CPU model of the HPU hybrid (attention + GDN) KV cache capacity.

Predicts the scheduler block size, bytes per block, usable blocks, KV tokens and max
concurrent requests for a hybrid GDN model and layout, for today's HPU allocation and the
planned GDN-state-pool fix. Validated against the Run 1 / Run 2 startup logs (see README).
"""

import argparse
import json
import os

GIB = 1024**3
KERNEL_BLOCK = 128  # HPU kernel block is 128 tokens (vllm_gaudi/platform.py:360)


def cdiv(a, b):
    return -(-a // b)


def geometry(config_json):
    """Model geometry from a HF config dict (fields read from text_config when present)."""
    tc = config_json.get("text_config", config_json)
    layer_types = tc["layer_types"]
    return {
        "num_hidden_layers": tc["num_hidden_layers"], "n_attn": layer_types.count("full_attention"),
        "n_gdn": layer_types.count("linear_attention"), "kv_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"], "linear_num_key_heads": tc["linear_num_key_heads"],
        "linear_num_value_heads": tc["linear_num_value_heads"], "linear_key_head_dim": tc["linear_key_head_dim"],
        "linear_value_head_dim": tc["linear_value_head_dim"], "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
    }


def layout(geom, tp, kv_bytes, budget_gib, max_len=131072, checkpoints=0):
    """Every capacity quantity for one layout; res['today'] / res['fixed'] sub-dicts."""
    n_attn, n_gdn = geom["n_attn"], geom["n_gdn"]
    kv_heads_rank = max(1, geom["kv_heads"] // tp)  # vLLM replicates KV heads when tp > kv_heads
    attn_tok = 2 * kv_heads_rank * geom["head_dim"] * kv_bytes  # K+V bytes per token per layer per rank
    attn_page128 = KERNEL_BLOCK * attn_tok
    conv_dim = (2 * geom["linear_num_key_heads"] * geom["linear_key_head_dim"] +
                geom["linear_num_value_heads"] * geom["linear_value_head_dim"])
    ssm_bytes = ((geom["linear_num_value_heads"] // tp) * geom["linear_key_head_dim"] *
                 geom["linear_value_head_dim"] * 4)  # recurrent state, float32
    conv_bytes = (conv_dim // tp) * (geom["linear_conv_kernel_dim"] - 1) * 2  # conv state, bf16
    gdn_real = ssm_bytes + conv_bytes
    # platform.py:394-419 pads the mamba page to a multiple of the attention page;
    # vLLM's unify_kv_cache_spec_page_size then raises the attention block size to match.
    page = cdiv(gdn_real, attn_page128) * attn_page128
    block = page // attn_tok
    positions = n_attn  # 1 attention group + (n_gdn / n_attn) GDN groups, each of n_attn layers
    gpu_bytes_per_block = positions * page  # vLLM shared-buffer accounting (kv_cache_utils)
    hpu_bytes_per_block = positions * (page + gdn_real)  # HPU: separate tensors, both num_blocks+1 deep
    budget_bytes = budget_gib * GIB
    blocks_today = int(budget_bytes * page / (page + gdn_real)) // gpu_bytes_per_block  # hpu_worker factor
    blocks_fixed = int(budget_bytes) // gpu_bytes_per_block  # GDN slots sized to the real need
    # align mode: MambaSpec.max_memory_usage_bytes = page * (2 + checkpoints) per GDN group
    per_request = cdiv(max_len * attn_tok, page) + (n_gdn // n_attn) * (2 + checkpoints)
    res = {
        "kv_heads_rank": kv_heads_rank, "attn_tok": attn_tok, "attn_page128": attn_page128,
        "conv_dim": conv_dim, "ssm_bytes": ssm_bytes, "conv_bytes": conv_bytes, "gdn_real": gdn_real,
        "page": page, "block": block, "positions": positions, "gpu_bytes_per_block": gpu_bytes_per_block,
        "hpu_bytes_per_block": hpu_bytes_per_block, "gdn_groups": n_gdn // n_attn, "per_request": per_request,
    }
    for mode, blocks in (("today", blocks_today), ("fixed", blocks_fixed)):
        res[mode] = {"blocks": blocks, "tokens": blocks * block, "concurrency": blocks / per_request}
    return res


def _config_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def load_geometry(name):
    with open(os.path.join(_config_dir(), name), encoding="utf-8") as f:
        return geometry(json.load(f))


RUN1 = ("Qwen3.8-27B.json", 8, 2, 72.98)  # tp, kv_bytes, budget GiB (user's Run 1 log)
RUN2 = ("Qwen3.8-27B.json", 4, 2, 75.36)  # user's Run 2 log

TABLE_ROWS = [
    ("FN TP8 BF16 weights", "Qwen3.8-Flash-Next.json", 8, 2, 40.0),
    ("FN TP4 FP8 weights", "Qwen3.8-Flash-Next.json", 4, 2, 39.0),
    ("FN TP8 + FP8 KV", "Qwen3.8-Flash-Next.json", 8, 1, 40.0),
    ("FN TP4 FP8 weights + FP8 KV", "Qwen3.8-Flash-Next.json", 4, 1, 39.0),
    ("27B TP8 BF16 (Run 1)", "Qwen3.8-27B.json", 8, 2, 72.98),
    ("27B TP4 BF16 (Run 2)", "Qwen3.8-27B.json", 4, 2, 75.36),
]

TABLE_HDR = ("row", "tp", "kv", "budget GiB", "block", "blocks today", "tokens today", "agents@131K today",
             "blocks fixed", "tokens fixed", "agents@131K fixed")


def table_row(name, geom, tp, kv_bytes, budget_gib, max_len=131072):
    lay = layout(geom, tp, kv_bytes, budget_gib, max_len=max_len)
    t, f = lay["today"], lay["fixed"]
    return (name, tp, "bf16" if kv_bytes == 2 else "fp8", f"{budget_gib:g}", lay["block"], t["blocks"],
            f"{t['tokens'] / 1e6:.2f}M", f"{t['concurrency']:.1f}", f["blocks"], f"{f['tokens'] / 1e6:.2f}M",
            f"{f['concurrency']:.1f}")


def print_table(rows):
    widths = [max(len(str(r[i])) for r in (TABLE_HDR, *rows)) for i in range(len(TABLE_HDR))]
    for row in (TABLE_HDR,) + tuple(rows):
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)).rstrip())


def check():
    ok = True
    r1 = layout(load_geometry(RUN1[0]), *RUN1[1:])
    good = (r1["today"]["blocks"] == 5293 and r1["block"] == 512 and abs(r1["today"]["concurrency"] - 20.20) <= 0.01)
    print(f"Run 1: blocks={r1['today']['blocks']} block={r1['block']} "
          f"concurrency={r1['today']['concurrency']:.2f} {'OK' if good else 'FAIL'}")
    ok &= good
    r2 = layout(load_geometry(RUN2[0]), *RUN2[1:])
    good = r2["block"] == 896 and abs(r2["today"]["concurrency"] - 19.08) <= 0.01 * 19.08
    print(f"Run 2: block={r2['block']} concurrency={r2['today']['concurrency']:.1f} (log 19.08) "
          f"{'OK' if good else 'FAIL'}")
    ok &= good
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="path to a HF config.json (text_config aware)")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--kv", choices=("bf16", "fp8"))
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--checkpoints", type=int, default=0)  # prefill checkpoint blocks
    parser.add_argument("--table", action="store_true", help="Flash-Next rows plus the two 27B validation rows")
    parser.add_argument("--check", action="store_true", help="reproduce the Run 1 / Run 2 logs, exit 0 on success")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check())
    if args.table:
        rows = [table_row(n, load_geometry(c), tp, kv, b) for n, c, tp, kv, b in TABLE_ROWS]
        print_table(rows)
        return
    if not all(v is not None for v in (args.config, args.tp, args.kv, args.budget_gib)):
        parser.error("--config, --tp, --kv and --budget-gib are required together")
    with open(args.config, encoding="utf-8") as f:
        geom = geometry(json.load(f))
    kv_bytes = 2 if args.kv == "bf16" else 1
    name = os.path.basename(args.config)
    print_table([table_row(name, geom, args.tp, kv_bytes, args.budget_gib, args.max_model_len)])


if __name__ == "__main__":
    main()
