# S4: KV cache capacity calculator for hybrid GDN models on HPU (CPU only)

`kv_capacity.py` predicts, for a hybrid GDN model (Qwen3.8-27B / Qwen3.8-Flash-Next) and a
layout (TP size, KV dtype, HPU KV memory budget):

- the scheduler block size and the page bytes per layer group,
- bytes per block (vLLM's shared-buffer view and the HPU's separate-tensor view),
- usable `num_blocks`, KV tokens and the maximum number of concurrent requests at
  131072 tokens per request (configurable via `--max-model-len`, `--checkpoints` adds
  prefill checkpoint blocks),

both for **today's HPU allocation** (the worker's `page / (page + gdn_real)` factor and, when
the model has a PLE short-conv standard-mamba spec, its second factor
`page / (page + ple_state)`, `vllm_gaudi/v1/worker/hpu_worker.py:1102-1166`) and for the
**M1 recurrent state pool** (slots = `max_num_seqs * groups * 3 + cache_slots`, attention
blocks from what the budget leaves). Pure CPU arithmetic; no `VllmConfig`, no weights.

## Usage

```bash
PY=/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python   # S0 env, vLLM at the pin
$PY spikes/kv-capacity/kv_capacity.py --check
$PY spikes/kv-capacity/kv_capacity.py --table
$PY spikes/kv-capacity/kv_capacity.py --config spikes/kv-capacity/configs/Qwen3.8-Flash-Next.json \
    --tp 8 --kv fp8 --budget-gib 40 [--max-model-len L] [--checkpoints C] \
    [--pool-slots N] [--max-num-seqs S]
```

`--pool-slots` overrides the pool slot count (default from the formula with
`--max-num-seqs`, default 32, and `cache_slots = 512`).

## Validation (user's startup logs)

`$PY spikes/kv-capacity/kv_capacity.py --check` (exit 0):

```text
Run 1: blocks=5293 block=512 concurrency=20.20 OK
Run 2: block=896 concurrency=19.2 (log 19.08) OK
```

- Run 1: 27B, tp 8, bf16 KV, budget 72.98 GiB. Log: `HPU KV cache size: 2,647,954 tokens`,
  `Maximum concurrency for 131,072 tokens per request: 20.20x`, `Usable num_blocks: 5293`.
- Run 2: 27B, tp 4, bf16 KV, budget 75.36 GiB. Log concurrency 19.08x; model 19.2 (within 1%).

## `--table` output

`$PY spikes/kv-capacity/kv_capacity.py --table`:

```text
row                          tp  kv    budget GiB  block  blocks today  tokens today  agents@131K today  slots  pool GiB  blocks pool  tokens pool  agents@131K pool
FN TP8 BF16 weights          8   bf16  40          512    2862          1.47M         10.9               896    4.18      6113         3.13M        23.9
FN TP4 FP8 weights           4   bf16  39          896    1690          1.51M         11.0               896    8.20      3003         2.69M        20.4
FN TP8 + FP8 KV              8   fp8   40          896    2970          2.66M         19.4               896    4.18      6987         6.26M        47.5
FN TP4 FP8 weights + FP8 KV  4   fp8   39          1664   1734          2.89M         20.4               896    8.20      3234         5.38M        40.9
27B TP8 BF16 (Run 1)         8   bf16  72.98       512    5293          2.71M         20.2               800    4.79      8728         4.47M        34.1
27B TP4 BF16 (Run 2)         4   bf16  75.36       896    2941          2.64M         19.2               800    9.58      4811         4.31M        32.7
```

`today` = the worker's shrink factors (the Flash-Next rows pay the PLE factor too, which is 1
for the 27B); the pool columns use the slots formula with `max_num_seqs = 32` (the 27B has
`groups = 3`, no PLE: 800 slots).

## Tests

`$PY -m pytest spikes/kv-capacity -q` → `6 passed`: Run 1 (5293 / 512 / 20.20 ± 0.01),
Run 2 (896 / 19.08 ± 1%), a cross-check of the two page sizes against vLLM's own
`FullAttentionSpec` / `MambaSpec` page-size accounting for the 27B at tp 8, the Flash-Next PLE
geometry (1 layer, 184,320 B per slot), Flash-Next TP8 BF16 40 GiB `today` with both factors
(2862 blocks), and the pool (896 slots, `(slots + 2)` bytes, more blocks than today, both
concurrency terms).

## Model (formulas and code references)

From the HF config `text_config`: `layer_types` gives `n_attn` (`full_attention`) and
`n_gdn` (`linear_attention`); plus `num_key_value_heads`, `head_dim` and the GDN geometry
(`linear_*` fields). With `tp` = tensor-parallel size and `kv_bytes` = 2 (bf16) / 1 (fp8):

| quantity | formula | reference |
|---|---|---|
| `kv_heads_rank` | `max(1, kv_heads // tp)` | vLLM replicates KV heads when tp > kv_heads |
| `attn_tok` | `2 * kv_heads_rank * head_dim * kv_bytes` | K+V bytes per token per layer per rank |
| `attn_page128` | `128 * attn_tok` | HPU kernel block is 128 tokens (`vllm_gaudi/platform.py:360`) |
| `conv_dim` | `2*linear_num_key_heads*linear_key_head_dim + linear_num_value_heads*linear_value_head_dim` | = 10240 for both models |
| `ssm_bytes` | `(linear_num_value_heads // tp) * linear_key_head_dim * linear_value_head_dim * 4` | recurrent state, float32 |
| `conv_bytes` | `(conv_dim // tp) * (linear_conv_kernel_dim - 1) * 2` | conv state, bf16 |
| `gdn_real` | `ssm_bytes + conv_bytes` | cross-checked vs `MambaSpec.page_size_bytes` |
| `page` | `ceil(gdn_real / attn_page128) * attn_page128` | mamba page padded to a multiple of the attention page (`vllm_gaudi/platform.py:394-419`) |
| `block` | `page // attn_tok` | vLLM `unify_kv_cache_spec_page_size` raises the attention block size to match |
| `positions` | `n_attn` | 1 attention group + `n_gdn / n_attn` GDN groups, each of `n_attn` layers |
| `gpu_bytes_per_block` | `positions * page` | vLLM shared-buffer accounting (`kv_cache_utils._get_kv_cache_bytes_per_block`) |
| `hpu_bytes_per_block` | `positions * (page + gdn_real)` | HPU allocates separate attention and GDN tensors, both `num_blocks+1` deep |
| `ple_layers` | `len(ple_layer_ids)` (0 when the config has no PLE: the 27B) | `text_config.ple_layer_ids`, `[2]` for Flash-Next |
| `ple_state` | `hc_count * hidden_size * (ple_conv_kernel_size - 1) * ngram_size * 2` (bf16, replicated: not `// tp`) | PLE short-conv state per slot = 184,320 B for Flash-Next |
| `blocks_today` | `int(int(budget * page / (page + gdn_real)) * page / (page + ple_state)) // gpu_bytes_per_block` | the two `hpu_worker.py:1102-1166` factors (GDN, then standard-mamba/PLE), each int-truncated, then vLLM's division; `ple_state = 0` ⇒ factor 1 |
| `groups` | `n_gdn // n_attn + (1 if ple_layers)` | recurrent state groups: GDN groups + the PLE layer (Flash-Next 3 + 1 = 4; 27B 3) |
| `slots` | `max_num_seqs * groups * 3 + cache_slots` (`cache_slots = 512`; `--pool-slots` overrides) | M1-T1 pool size: live slots per request 3 + cache; Flash-Next @32: 896 |
| `pool_bytes` | `(slots + 2) * (positions * gdn_real + ple_layers * ple_state)` | M1-T3 pool reservation (+2 slack slots); Flash-Next: 898 × 4.76 MiB = 4.18 GiB |
| `blocks_pool` | `(budget - pool_bytes) // gpu_bytes_per_block` | attention blocks from what the budget leaves |
| `per_request(L)` | `cdiv(L * attn_tok, page) + (n_gdn // n_attn) * (2 + checkpoints)` | align mode: `MambaSpec.max_memory_usage_bytes = page * (2 + checkpoints)` |
| `concurrency_today(L)` | `blocks_today / per_request(L)` | vLLM's "Maximum concurrency" line |
| `concurrency_pool(L)` | `min(blocks_pool / cdiv(L * attn_tok, page), slots / (groups * 3))` | the pool charges no per-request GDN blocks; both terms reported |
| `kv_tokens` | `blocks * block` | vLLM's "KV cache size ... tokens" line |

The two page sizes equal vLLM's own classes at the pin (verified in
`test_vllm_spec_cross_check`): `attn_page128 == FullAttentionSpec(block_size=128, ...).page_size_bytes`
and `gdn_real == MambaSpec(shapes=((V//tp, dk, dv), (conv_dim//tp, kernel-1)),
dtypes=(float32, bfloat16), block_size=-1).page_size_bytes`.

Out of scope: the PLE-table-on-host layout (budget not settled), the GDN-state-pool fix
itself (M1), anything under `vllm_gaudi/`.
