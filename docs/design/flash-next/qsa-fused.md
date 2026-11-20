# Fused QSA decode: data flow, kernels and contract

Version 2, 2026-11-20. The sparse QSA decode path of Flash-Next (`vllm_gaudi/models/qwen4_exp/qsa.py`,
`HpuQsaIndexer._select_decode`, and `qsa_hpu_attention.py`, `HpuQsaSparseAttention`) behind one flag,
`VLLM_HPU_QSA_FUSED` (`QsaFusedMode`, **default 0**), plus `VLLM_HPU_QSA_INDEXER_SHARD` (`QsaShardMode`, default 0):

| `VLLM_HPU_QSA_FUSED` | What runs |
|---|---|
| 0 | the torch path as before (nothing of this page is imported, loaded or logged) |
| 1 | the restructured **torch data flow**: one key gather per request (I14), the grouped attention (I21, G5-1), and with `VLLM_HPU_QSA_INDEXER_SHARD=1` the indexer scored by request on each TP rank (I18) |
| 2 | mode 1 plus two **TPC kernels**: `qsa_index_prepare` (K1) and `qsa_index_emit` (K4); each layer falls back to mode 1 with one log line when the kernels cannot serve it |

Status words: **VERIFIED** (by the named test, probe or source line) and **INFERRED** (reasoned, not measured).
Nothing here has run on a card yet; "Unproven without a card" lists what that leaves open. Version 1 of this page
(the first agent, 2026-11-20 morning) planned four kernels with a per-row TPC key scan; version 2 keeps its row
table, K1 and its rounding analysis, moves the scores to the matrix engine and changes the data flow (below, "Why
the cut moved").

## Why

### Nodes

Torch ops dispatched by one QSA layer forward in one decode step of the tiny checkpoint (sparse mode, bf16, CPU
stand-ins of the HPU attention), counted with a `TorchDispatchMode` that attributes every op to its source line
(`reports/qsa-fused/measure/qsa_ops.py`, `qsa_ops_torch.log`; VERIFIED):

| Stage (source) | 1 row: total (compute) | 32 rows: total (compute) |
|---|---|---|
| Indexer projection GEMM | 2 (1) | 2 (1) |
| Indexer queries: split, RoPE table rows, norm, RoPE (`index_queries`) | 27 (19) | 27 (19) |
| Index prep: state slots, row validity | 18 (8) | 18 (8) |
| Key ring: slot selects, write, read of the block's raw keys | 26 (25) | 26 (25) |
| Compressed row: pool, norm, RoPE, `index_copy_` | 34 (29) | 34 (29) |
| Indexer scoring: gather through the block table, `q.k`, relu, head sum | 30 (13) | 30 (13) |
| Block top-k (`select_blocks`) | 13 (10) | 13 (10) |
| Expansion to token positions (`expand_selection`) | 20 (11) | 20 (11) |
| **Indexer `select`** | **170 (116)** | **170 (116)** |
| Attention: K/V cache write, selected positions -> slots and valid mask, gather, scores, softmax, values | 57 (28) | 59 (30) |
| Layer: QKV GEMM, q/gate split, q and k norm, RoPE, gate, output GEMM | 29 (13) | 29 (13) |
| **One QSA layer** | **256 (157)** | **258 (159)** |

A verification step (`bs` requests of `1 + k` rows) dispatches 182 (118) in `select` at 4, 128 and 192 rows. The
node count does not depend on the rows; the bytes do.

The same count for each mode (`reports/qsa-fused/measure/qsa_ops_v2.py`, VERIFIED; mode 2 with the references
registered as the ops' CPU kernels, one node per op as on the card):

| Mode | One QSA layer, 1 row: total (compute) | 32 rows | 12 layers at 32 rows |
|---|---|---|---|
| 0 (torch path) | 256 (157) | 258 (159) | 3,096 (1,908) |
| 1 (torch data flow) | 244 (155) | 246 (157) | 2,952 (1,884) |
| 2 (K1 + K4) | 136 (68) | 138 (70) | 1,656 (840) |

Mode 2 by stage at 32 rows: the row table and K1 31 ops (the ring, cache and query work is one node), the
request-major scores and top-k 36, K4 1, the attention 41 (cache write, gather, grouped GEMMs, softmax), the
layer's projections, norms, RoPE and gate 29.

### Bytes: the block table is as wide as the engine's longest context

VERIFIED (`hpu_model_runner.py:7204`, `flash_next_inputs.build_block_table`): the decode block table has
`MB = ceil(max_model_len / 128)` columns, fixed per engine, so with `--max-model-len 131072` every decode row
addresses `N = MB * rpb = 32,768` compressed keys **whatever its context**. The torch path gathers all of them for
every row (`_select_decode`: an int64 `[T, N]` index, `cache.index_select`, then `.float()` of the gathered keys),
and the `1 + k` rows of one request gather the same keys `1 + k` times (one block-table row per query row,
`hpu_model_runner.py:4946-4961`). All eight TP ranks repeat it (`index_qk_proj` is replicated).

Bytes per rank per QSA layer, real geometry at TP8 (indexer `H = 4`, `D = 128`; attention 3 query heads and 1 KV
head per rank, head dim 256, `W = 2,051` attended positions). INFERRED from the shapes, every torch op read and
written once:

| Per QSA layer, per rank | today, 32 x 1 rows | mode 1, 32 x 1 | mode 1 + shard, 32 x 1 | today, 32 x 4 (k = 3) | mode 1, 32 x 4 | mode 1 + shard, 32 x 4 |
|---|---|---|---|---|---|---|
| Indexer: `[T, N]` index | 96 MiB | 0 | 0 | 384 MiB | 0 | 0 |
| Indexer: gather (read + write) | 512 MiB | 512 MiB | 64 MiB | 2.0 GiB | 512 MiB | 64 MiB |
| Indexer: fp32 copy (read + write) + GEMM read | 1.25 GiB | 1.25 GiB | 160 MiB | 5.0 GiB | 1.25 GiB | 160 MiB |
| Indexer: scores, relu, sum, mask, top-k | 70 MiB | 70 MiB | 9 MiB | 280 MiB | 280 MiB | 35 MiB |
| **Indexer total** | **1.9 GiB** | **1.8 GiB** | **0.23 GiB** | **7.6 GiB** | **2.0 GiB** | **0.25 GiB** |
| Attention: gather K and V, 3x repeat, permute/transpose copies, GEMM reads | 1.0 GB | 0.20 GB | 0.20 GB | 4.0 GB | 0.81 GB | 0.81 GB |
| **12 layers, one step** | **37 GB** | **26 GB** | **5.3 GB** | **146 GB** | **36 GB** | **13 GB** |
| ms at 1.6 to 2.4 TB/s | 15 to 23 | 11 to 16 | 2.2 to 3.3 | 61 to 91 | 15 to 22 | 5.4 to 8.1 |

What each change buys (INFERRED, same arithmetic): the request-major gather divides the indexer bytes of a
verification step by `1 + k` and removes the int64 index; sharding by request divides what is left by the TP size;
the grouped attention removes the 3x copies of K and V and their layout copies (about 25 of 31 MB per row and layer).
The fp32 copy of the gathered keys stays in mode 1: plain torch has no bf16 x bf16 -> fp32 GEMM, and a bf16 output
changes the scores (I15; "Next"). Whether the graph compiler fuses that cast is measured by card job
`qsa-select-attn-bench` (`reports/qsa-fused/card-bench/qsa_bench.py`, case `part_cast_fp32`).

### Why the cut moved from version 1

Version 1 scored the keys with a TPC kernel (K2) reading each visible key once per row. The first agent's
simulator probes give about 40 instruction words per key and row (`reports/qsa-fused/probe/score_row_{a,b}.c`,
93 and 121 words with one key per iteration; INFERRED 40 with four keys in flight): at 32 x 128K that is 20 to 40 ms
per step on 24 TPC engines, compute-bound, and it does not shrink when rows of one request share a key load (I14).
On the matrix engine the same product is `[bs, (1 + k) H, D] x [bs, D, N]`: memory-bound, 8 MiB of keys per request
and layer. With the request-major gather and sharding (mode 1) the scores cost about 0.3 GiB per layer and rank
at 32 x 4 rows, under 0.2 ms at the band above. So version 2 keeps the scores, and the top-k, in torch on the MME and
the vendor TopK, and spends the TPC kernels where torch spends nodes: the per-row prepare work (K1, about 100 ops per
layer) and the expansion of the selection to cache slots (K4, about 35 ops per layer).

## I17 or I18: shard by context range or by request

Both cut the replicated indexer by the TP size. INFERRED, at TP8, 32 streams, `N = 32,768`:

| | I18, by request | I17, by context range |
|---|---|---|
| Applies when | `bs % 8 == 0` (every decode bucket from 8 streams up) | any row count, also 1 stream |
| Exact | yes: each request is scored whole by one rank; no merge | only with an id tie-break in the merge |
| Collective per layer | all-gather of int32 `[bs / 8 * (1 + k), 512]`: 8 KiB per rank at 32 x 1 | all-gather of `[T, 1024]` (512 values + 512 ids) fp32: 128 KiB per rank at 32 x 1 |
| Extra work per layer | none | a second top-512 over 4,096 candidates per row on every rank |
| Load balance on the MME path | perfect: every request costs `N` keys (fixed `MB`) | perfect |
| Gain at 1 stream | none (not sharded) | 7/8 of one row's 56 MiB per layer, about 0.3 ms per step, against 12 all-gathers (unmeasured; 0.25 to 0.7 ms at 20 to 60 us) |

At the target (32 streams, 1 + k rows) both remove the same bytes; I18 sends 16x less, merges nothing and is exact,
so **I18 is built** (`select_blocks_request_major`, `VLLM_HPU_QSA_INDEXER_SHARD=1`). I17 helps only below 8 streams,
where its gain is about the price of its collectives. If a later kernel reads only the visible keys (context-
proportional cost), I18's balance needs a host-sorted request order across ranks; on today's fixed-`MB` MME path
it does not.

Mechanics (VERIFIED by `tests/unit_tests/models/flash_next/test_qsa_fused_torch_tp.py`, 1/2/4/8 gloo ranks): rank
`r` takes requests `[r * bs / tp, (r + 1) * bs / tp)` with `index_select` (all ranks build the same graph shape),
scores and selects them, and one `tensor_model_parallel_all_gather` along dim 0 returns the selections in request
order on every rank. The cache writes stay replicated: every rank keeps complete caches, because the request-to-
rank map follows the batch order of each step. A bucket whose `bs` is not a multiple of `tp` stays replicated, so
the decision is static per bucket (no host read).

## The cut (mode 2)

```
torch path (flag 0)                                     mode 2
  qk = GEMM_index(hidden)                    (MME)        qk = GEMM_index(hidden)                         (MME)
  q = rope(norm(qk[:, :H*D]))                 27 ops      rows = rows_table(positions, slots, ring slots)    ~20 ops
  slots, validity                             18 ops      q_idx = qsa_index_prepare(ring!, cache!, qk, rows, [K1]
  ring write, ring read                       26 ops                       weights, cos_sin)
  pooled key: norm, rope, cache write         34 ops      keys = cache.view(-1, rpb, D)[table of each request]
  gather n keys per row, bmm, relu, sum       30 ops      scores = relu(q_idx @ keys^T).sum(heads)     (MME, torch)
  mask, topk(512)                             13 ops      blocks = topk(mask(scores), 512)  [all-gather if sharded]
  expand to token positions                   20 ops      slots, valid = qsa_index_emit(blocks, positions,  [K4]
  positions -> cache slots, valid mask        13 ops                              block_table)
  gather K, V; repeat heads; bmm; mask;       36 ops      gather K, V; grouped bmm; mask; softmax; bmm  (MME, torch)
    softmax; bmm
```

K1 before the scores: K1 writes the step's raw keys and compressed rows, the scores read the compressed rows
(write-then-score, `_select_decode` docstring); the data dependency is `q_idx`. K4 after the top-k.

What stays torch: the GEMMs (`index_qk_proj`, `qkv_proj`, `o_proj`), the request-major scores and the top-k, the
layer's q/gate split, q and k norm and RoPE, the K/V cache write, the attention, the prefill branch of `select` and
the prefill attention (they share the side caches with the kernels, in the same layout).

## Names

| What | Name |
|---|---|
| Ops | `torch.ops.flashinfer_gaudi.qsa_index_prepare`, `qsa_index_emit` |
| Schemas, fake kernels, wrappers (the only definition) | `flashinfer_gaudi/qsa_fused.py`; imported only with the flag at 2 |
| Torch references (the executable definition) | `tests/unit_tests/qsa_fused/qsa_reference.py` |
| CPU stand-in (the references as the ops' CPU kernels) | `tests/unit_tests/qsa_fused/qsa_ops.py` |
| Data flow of modes 1 and 2 | `vllm_gaudi/models/qwen4_exp/qsa_fused_path.py`; `HpuQsaIndexer.select` / `_select_decode_kernels`; `HpuQsaSparseAttention.forward` |
| TPC kernels | `csrc/qsa_fused/kernels/qsa_fused_{prepare,emit}_gaudi2.c` + `qsa_fused_common.h` |
| GUIDs | `qsa_fused_prepare_bf16_gaudi2`, `qsa_fused_emit_i32_gaudi2` (`csrc/qsa_fused/qsa_fused_params.h`) |
| Kernel database | `libqsa_fused_gaudi2.so` (CMake target `qsa_fused_gaudi2`); its own library; a member of the combined kernel database (`csrc/kernel_db_adapter/adapter.cpp`) |
| Host glue | `csrc/qsa_fused/host/qsa_fused_gaudi2.cpp`, `host/entry_points.cpp` |
| PyTorch extension | `qsa_fused_ops*.so` (`csrc/qsa_fused/pytorch/hpu_qsa_fused.cpp`, `setup.py`); registers only the `HPU` kernels |
| Loader | `flashinfer_gaudi/_native.py::load_qsa_fused_extension`, `qsa_fused_extension_errors`, `qsa_fused_extension_loaded`; overrides `QSA_FUSED_GAUDI_KERNEL_LIBRARY`, `QSA_FUSED_GAUDI_NATIVE_LIBRARY`; called by `vllm_gaudi/__init__.py::_load_qsa_fused_if_requested` |
| Flags | `VLLM_HPU_QSA_FUSED` (`QsaFusedMode`: 0, 1, 2), `VLLM_HPU_QSA_INDEXER_SHARD` (`QsaShardMode`: 0, 1) in `vllm_gaudi/envs.py` |
| Build switch | `VLLM_GAUDI_BUILD_QSA_FUSED=1` (the rental's install sets it with the other native builds), or `tools/build_flashinfer_gaudi.py --qsa-fused-only` |

## Geometry and symbols

`T` = query rows (`bs` requests of `T_cols >= 1` rows, padded), `H` = `indexer_n_heads` (4), `D` =
`indexer_head_dim` (128; 16 on the tiny model), `ratio` = `indexer_compress_ratio` (4), `topk` = `indexer_budget /
ratio` (512), `W = topk * ratio + ratio - 1` (2,051), `R2` = half the rotary dims of the attention's
`cos_sin_cache` (32; 4 on the tiny model), `S` = ring slots, `R` = ring rows per slot (`ring_rows(ratio, k)`: 4, 8,
8, 12 at k = 0, 3, 4, 5), `C` = rows of the compressed key cache, `B` = `block_size` (128), `rpb = B / ratio` (32),
`MB` = block-table columns (1,024 at 128K), `N = MB * rpb`. `floordiv`/`floormod` round towards minus infinity, as
torch's `//` and `%`.

### The 32 x (1 + k) row case

A verification step of 32 requests at k = 3 is `T = 128` rows, `T_cols = 4`. Mode 1 takes one table row per
request (`request_table_rows`: row `b * T_cols + clamp(len_b - 1, 0)`, the longest causal prefix; a padding request
takes its first row, the runner's pad row) and gathers `[32, N, D]` once; the product is `[32, 16, 128] x [32, 128,
N]` (`M = 16` per request). Exact (VERIFIED, `test_qsa_fused_torch.py::test_step_crossing_a_page`,
`test_thirty_two_requests_verification`): a row's visible columns `n < (p + 1) // ratio` lie on pages its own table
shares with the longest one, and every other column is masked before the top-k. With sharding, each rank scores 4
requests (16 rows of `M = 16`) and gathers `[16, 512]` int32. K1 and K4 run on all 128 rows (index space `{T}`).
The ring holds `R >= ratio + k` rows per slot, so the `1 + k` rows of a request never write the same ring row.

## The row table

One int32 tensor `rows` `[T, 4]` (`qsa_fused_path.rows_table`, built per layer from the same metadata as
`_select_decode`):

| Column | Value | Torch path |
|---|---|---|
| 0 | position `p` of the row | `positions` (`indexer_positions`) |
| 1 | attention slot of the row's token | `md.slot_mapping.reshape(-1)[:T]` |
| 2 | ring slot the row reads, or `-1` when the row is padding | `where(valid, load_rows, scratch)` |
| 3 | ring slot the row writes, or `-1` when the row is padding | `where(valid, store_rows, scratch)` |

The torch path parks padding rows on the ring's scratch slot and on the cache's scratch row; K1 writes nothing for
them. **The ring's scratch slot (its last `R` rows) and the cache's scratch row (its last row) are unspecified
after either path** and no test compares them (on the torch path they hold whichever duplicate index of one
`index_copy_` won).

## Schemas

```
qsa_index_prepare(Tensor(a!) key_ring, Tensor(b!) key_cache, Tensor qk, Tensor rows, Tensor q_norm_weight,
                  Tensor k_norm_weight, Tensor cos_sin, float eps, int heads, int ring_rows, int ratio) -> Tensor
qsa_index_emit(Tensor blocks, Tensor positions, Tensor block_table, int ratio, int block_size) -> (Tensor, Tensor)
```

| Arg | Torch shape, dtype | TPC dims (fastest first) | Meaning |
|---|---|---|---|
| `key_ring` | `[S * R, D]` bf16 | `{D, S * R}` | the flat raw-key ring. **Written by K1** |
| `key_cache` | `[C, D]` bf16 | `{D, C}` | the compressed key cache. **Written by K1**, read by the scores |
| `qk` | `[T, (H + 1) * D]` bf16 | `{(H + 1) * D, T}` | `index_qk_proj` output: `H` query heads, then the raw key |
| `rows` | `[T, 4]` int32 | `{4, T}` | the row table |
| `q_norm_weight`, `k_norm_weight` | `[D]` bf16 | `{D}` | `q_layernorm.weight`, `k_layernorm.weight` |
| `cos_sin` | `[P, 2 * R2]` bf16 | `{2 * R2, P}` | the attention's `rotary_emb.cos_sin_cache` (bf16: `RotaryEmbedding` casts it to the model dtype) |
| K1 -> `q_idx` | `[T, H * D]` bf16 | `{H * D, T}` | indexer queries after norm and RoPE |
| `blocks` | `[T, topk]` int32 | `{topk, T}` | `select_blocks` output (`-1` = none), any order |
| `positions` | `[T]` int32 | `{T}` | position of each row (`-1` on a padding row under M-RoPE) |
| `block_table` | `[T, MB]` int32 | `{MB, T}` | kernel block ids per query row (`md.block_table`) |
| K4 -> `slots`, `valid` | `[T, W]` int32, `[T, W]` int32 | `{W, T}` | flat K/V cache row and 0/1 flag of every attended column |

Gates, checked by the op's validation callback (`TORCH_CHECK`), by the kernel glue (`GlueCodeReturn`) and by the
layer before it calls (`kernel_shape_reason`, `kernel_ops_reason`): the dtypes and ranks above; contiguous tensors on
one device; `1 <= D <= 128`; `1 <= H <= 8`; `ratio` in {2, 4, 8}; `ratio <= R <= 64` and `S * R` the ring's rows;
`R2` a power of two with `2 * R2 <= D`; `B` a power of two `>= ratio`; positions below `2^24`. Everything else
comes from the tensor dims.

## Semantics and rounding points

The definition is `tests/unit_tests/qsa_fused/qsa_reference.py`, written with the torch path's own functions
(verbatim copies; `test_reference_copy.py` checks the source). Every bf16 tensor op of the torch path is "fp32, then
one rounding to bf16, round half to even" on CPU; the kernels keep those rounding points (products and sums are
separate instructions, `fp contract(off)`, every bf16 result through `v_convert_f32_to_bf16_all_b(..., SW_RHNE)`).

### K1 `qsa_index_prepare`

For every row `t` with `p, sm, ls, ss = rows[t]` and `raw = qk[t, H*D:]`:

```python
for h in range(H):                                   # index_queries
    x = qk[t, h*D:(h+1)*D].float()
    y = bf16(x * rsqrt(sum(x*x) * (1/D) + eps) * (1.0 + q_norm_weight.float()))
    q_idx[t, h] = rope(y, cos_sin[max(p, 0)])
if 0 <= ss < S:                                      # a real row
    key_ring[ss*R + p mod R] = raw                   # exact copy
    c = sm >> log2(ratio)
    if p mod ratio == ratio - 1 and 0 <= c < C:
        blk = [ring_row(ls*R + (p - ratio + 1 + j) mod R) for j < ratio - 1] + [raw]
        pooled = bf16(((blk[0] + blk[1]) + blk[2] + ...) * (1 / ratio))      # fp32, in order
        key_cache[c] = rope(bf16(norm(pooled, k_norm_weight)), cos_sin[max(p - ratio + 1, 0)])

def rope(y, cs):          # rope_neox_partial; c = cs[:R2], s = cs[R2:]
    out[j]      = bf16(bf16(y[j] * c[j]) - bf16(y[R2+j] * s[j]))      for j < R2
    out[R2 + j] = bf16(bf16(y[R2+j] * c[j]) + bf16(y[j] * s[j]))      for j < R2
    out[2*R2:]  = y[2*R2:]
```

- **Write-then-read without a race.** `ring_row(i)` is the raw key of the last real row of this call that writes
  ring row `i`, read from that row's `qk`, else the ring's row as it was before the call. No point reads a ring row
  that a point of the same call writes, so the in-place hazard of `tpcsim.md` does not arise; no two real rows write
  the same ring row (`R >= ratio + k`, distinct slots per request).
- The RoPE half swap is `v_element_shift_xor_bf16(y, R2)` (lane `j` <-> lane `R2 + j` inside every `2 R2` lanes), in
  the bf16 lane domain where the order is linear; the products and sums go through fp32 and one RHNE rounding.
- `1.0 + weight` is its own fp32 addition, as in `gemma_rms_norm`. The mean of squares is the kernel's own sum
  times `1 / D`, and `rsqrt` is the TPC table function: both differ from CPU torch by a few fp32 steps before the
  rounding to bf16 (the tolerance below).
- The modulo by `R` (12 at k = 5 is not a power of two) is a float quotient with two integer corrections; exact for
  `n < 2^24`.

### K4 `qsa_index_emit`

For every row `t` with `p = positions[t]`: column `i * ratio + r` (`r < ratio`) is position `b * ratio + r` of
`b = blocks[t, i]` with valid 1 when `b >= 0`, else position 0 with valid 0; column `topk * ratio + j` (`j < ratio -
1`) is the tail position `floordiv(p + 1, ratio) * ratio + j` with valid 1 when it is `<= p`, else position 0 with
valid 0. A position `q` maps to `block_table[t, q >> log2(B)] * B + (q & (B - 1))` (a column outside `[0, MB)` is
clamped: the kernel never reads outside its tensors). A row with `p < 0` gets valid 1 in every column, as
`rows_without_selection` gives it (for the indexer's own `blocks`, the expanded selection of a row is empty exactly
when `p < 0`). This is `expand_selection` + `decode_selected_slots` + `(selected >= 0) | rows_without_selection`,
in the same column order, so the attention sums in the same order as mode 1.

### What each mode is equal to

| Quantity | Mode 1 against mode 0 (CPU) | Mode 2 with the references as kernels, against mode 1 (CPU) | TPC kernel against its reference (simulator) |
|---|---|---|---|
| Ring and compressed cache, outside the scratch slot / row | bit-exact (same code) | bit-exact | bit-exact (all cases) |
| `q_idx` | (not formed) | bit-exact | 99.9 % or more bit-exact, the rest 1 bf16 step (`prepare_tiny_lattice`; the three other cases 100 %) |
| Block scores | the same products and sums; GEMM blocking may sum in another fp32 order (bit-exact on this CPU) | the same | (torch on both sides) |
| Selection of a real row | bit-exact on every tested case | bit-exact | (torch) |
| `slots`, `valid` | (not formed) | bit-exact | bit-exact |
| Attention output | grouped GEMM: bit-exact in bf16 on this CPU, fp32 within `1e-6` | the same as mode 1 | (torch) |

VERIFIED by `test_qsa_fused_torch.py`, `test_qsa_fused_torch_tp.py`, `tests/unit_tests/qsa_fused/test_qsa_reference.py`
and `tests/standalone/qsa_fused/test_qsa_simulation.py` (see "Tests"). On a card the MME may sum the scores in an
order other than the CPU's; a changed selection then needs two scores equal within that rounding at the cut.

## Kernel layout

Scalar arguments: K1 `float eps, inv_dim, inv_ring, inv_ratio; int heads, ring_rows, ratio, rotary_half`
(`QsaFusedPrepareArgs`, 32 bytes, the reciprocals from the host: no kernel divides); K4 `int ratio_log2,
block_log2, topk` (12 bytes). Node parameters: `QsaFusedParams {float eps; int32 heads, ring_rows, ratio,
block_size, rotary_half}`, 24 bytes, the same for both GUIDs (unused fields 0; K1 derives `rotary_half` from
`cos_sin`).

| Kernel | Tensors (in + out) | Index space | One point owns | Special functions | VLM |
|---|---|---|---|---|---|
| `qsa_fused_prepare_bf16_gaudi2` | 7 + 1 | `{T}` | one row: its `q_idx` row, its ring row, its compressed row | `rsqrt` | 0 bytes |
| `qsa_fused_emit_i32_gaudi2` | 3 + 2 | `{T}` | one row of `slots` and `valid` | none | 0 bytes |

- **In-place tensors.** `key_ring` and `key_cache` are inputs K1 stores to, declared as whole tensors read by
  every point: the form the card-proven `gdn_core_decode` uses for its state pools.
- **Access patterns.** K1: every input whole (data-dependent rows of the ring, the cache, `cos_sin`; `qk` and
  `rows` of other rows for write-then-read); `q_idx` by row. K4: `blocks`, `positions`, `block_table`, `slots`,
  `valid` by row. The simulator runs every case with its index-space mapping test and partition shuffle on.
- **Instruction counts** (simulator, VERIFIED): K1 5.6 K words per row at the real geometry (180,746 for 32 rows)
  and 8.8 K at 32 x 4 rows (1,122,610 for 128 rows: the write-then-read search scans the 128 rows); K4 11.8 K
  words per row (378,328 for 32 rows, 1,513,048 for 128: about 23 words per selected block, four blocks' scalar
  loads in flight; 60 with one). INFERRED at 1.5 to 2 GHz: K1 about 6 to 8 us and K4 about 12 to 16 us per layer at
  32 rows (two rows per engine), K1 30 to 40 us and K4 35 to 50 us at 128 rows.

## Card rules (BRIEF.md) and how each is met

| Rule | How |
|---|---|
| 1. A custom op never gets a view | `qsa_fused_path.owned` copies any input that is a view, not contiguous or offset (`qk`, `cos_sin`, `blocks`, `block_table`); `rows` is the output of `torch.stack`; `positions` goes through `to(int32, copy=True)`; the ring, the cache and the weights are the bound buffers and parameters themselves |
| 2. int32 only | `rows`, `blocks`, `positions`, `block_table`, `slots`, `valid` are int32; the mode-1 block gather uses the int32 table as the `index_select` index; the sharded all-gather sends int32 (checked on 8 ranks by card job `qsa-select-attn-bench`) |
| 3. No op without an HPU kernel | modes 1 and 2 add `index_select`, `matmul`, `relu`, `sum`, `masked_fill`, `topk`, `stack`, `where`, `arange`, casts, views and one all-gather; no `cummax`, advanced indexing, `index_put_` or bool scatter |
| 4. Accumulation mode 0 | the rental sets `PT_HPU_LAZY_ACC_PAR_MODE=0` for every process |
| 5. Card arithmetic | no kernel divides: `1 / D`, `1 / R`, `1 / ratio` come from the host; the card test compares with the CPU reference |
| 6. FP8 range | no FP8 value is read or written (G5-3 below would add one) |
| 7. One `GC_KERNEL_PATH` file | `libqsa_fused_gaudi2.so` is a member of the combined kernel database; the loader activates it only when the combined file serves the library |
| 8. Fallback and switch | layer: `kernel_ops_reason` / `kernel_shape_reason`, one log line, mode 1; process: the loader never raises, one warning at plugin start; flags in `vllm_gaudi/envs.py`, default 0, integer enums |
| A single node per op | each op is one `registerUserCustomOp` GUID: one TPC node |

## Tests

| Command (from the worktree root) | What | Result |
|---|---|---|
| `pytest tests/unit_tests/models/flash_next/test_qsa_fused_torch.py` | mode 1 vs mode 0 on the tiny indexer: 1 row, 32 rows, lattice k = 3/4/5, 32 x 4, page crossing, slot moves, no `query_start_loc_p`, fp32; grouped attention vs repeat (KV 1 and 2, G 1 to 4) | see REPORT |
| `pytest tests/unit_tests/models/flash_next/test_qsa_fused_torch_tp.py` | sharded vs replicated vs mode 0 on 1/2/4/8 gloo ranks, the collective issued | 16 passed |
| `pytest tests/unit_tests/qsa_fused` | references vs torch path, mode 2 (references as kernels) vs mode 1, fallback, op callbacks (ASan/UBSan), reference copies | 37 passed |
| `tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest tests/standalone/qsa_fused/test_qsa_simulation.py'` | both kernels compile with the default flags (0 bytes VLM) and match the references on 7 cases | 10 passed |

## Unproven without a card

1. The extension against the real bridge (the schema with `Tensor(a!)`, an int32 pair of outputs).
2. The graph compiler's reading of an input that the kernel stores to (`key_cache` in K1) next to a later node
   that reads it (the scores' gather): the dependency is `q_idx`.
3. `index_select` with an int32 index, and HCCL `all_gather_into_tensor` of int32 (card job `qsa-select-attn-bench`).
4. `v_rsqrt_f32` on hardware against the simulator.
5. Speed: every ms in this page is arithmetic.

## G5-3: FP8 KV cache with sparse QSA (not built)

Today the layer refuses an FP8 main KV cache (`HpuQwen4ExpQSAAttention.__init__`). With `kv_cache_dtype` FP8 the
decode gather would read 1 byte per element and dequantize right after it: `k = cast_from_fp8(key_cache.index_select
(0, slots)) * scale` before the grouped GEMMs (2,051 x 256 bytes per row and tensor instead of x 2). INFERRED: about
0.1 GB per 32-row step less after the grouped attention; the larger gain is KV capacity at 128K and FP8 prefill
attention. Rule 6 applies (Gaudi2 range 240).

## Next

- **The fp32 copy of the gathered keys** (mode 1's largest byte term): a bf16 x bf16 -> fp32 GEMM (the HPU
  `fp8_gemm_v2` takes bf16 operands with an fp32 output in some releases; unverified), or a TPC kernel that gathers
  and widens in one pass.
- **K4 at fewer words per block** (23 now): split a row over several index-space points (24 engines for 32 rows
  leave most engines at two rows).
- **K3**, a radix-select threshold for the top-512 (the DeepSeek V4.1 `index_threshold` precedent), only if card
  job `qsa-select-attn-bench` shows `torch.topk` over `[T, 32768]` costs more than the rest of the indexer.
- **Tile layout of the compressed key cache** and a TPC scan (version 1's "Next"): only worth it if the MME path
  stays bandwidth-bound above the TPC scan's cost.
- **The row table once per step** for all 12 layers (about 20 ops per layer today), keyed to the forward's metadata.
