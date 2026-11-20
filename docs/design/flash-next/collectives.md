# Flash-Next collectives at TP8 + EP8: the count, the cuts, and what cannot be cut

Status 2026-11-20. Track `collectives`. Every number in the tables is the recorded output of
`tools/flash_next/collective_count.py` (8 gloo CPU ranks, the 8-head tiny model through vLLM's real loader, the
plugin's MoE path and `HpuCommunicator`), scaled to the real model as the tool's docstring says. Labels:
**VERIFIED** (a test or the code), **INFERRED** (arithmetic on verified inputs).

## 1. The count (VERIFIED, `test_collective_cuts.py::test_baseline_count_at_tp8`)

Every collective of a step goes through `vllm_gaudi/distributed/device_communicators/hpu_communicator.py`.

| Phase (real model, TP8 + EP8, k = 3) | Collectives | Kinds |
|---|---|---|
| prefill chunk | 99 | embedding 1, PLE table 1, mixer 48, MoE 48, logits all-gather 1 (last row of the chunk) |
| decode step | 99 | the same; the logits all-gather carries every decode row |
| verification step (32 x (1 + k) rows) | 99 | the same; the logits of every verification row are gathered |
| MTP drafts of one step | 6 per drafted token (18 at k = 3) | embedding 1, `fc_embedding` and `fc_hidden` all-gathers 2, mixer 1, MoE 1, logits all-gather 1 |

Call sites (rank 0, the same order on every rank):

| Kind | Method | Payload per row on every rank | Issued at | By |
|---|---|---|---|---|
| embedding | all-reduce | 2560 bf16 (5 KB); one rank holds the row, seven send zeros | `vllm_gaudi/ops/hpu_vocab_parallel_embedding.py` forward | `model.py` `embed_input_ids` (target and draft share the table) |
| PLE table | all-reduce | 16 x 160 values (bf16 on the tiny model; the real table is FP8) | upstream `vocab_parallel_embedding.py` forward | `vllm_gaudi/models/qwen4_exp/ple_layer.py` forward (layer 1) |
| mixer | all-reduce | 2560 bf16 | upstream `linear.py` `RowParallelLinear.forward` | GDN `out_proj` (`vllm_gaudi/models/qwen3_5.py` forward), QSA `o_proj` |
| MoE | all-reduce | 2560 bf16 | upstream `moe_runner.py` `_maybe_reduce_final_output` | `vllm_gaudi/ops/hpu_fused_moe.py` `patched_fused_moe_forward` |
| logits | all-gather | 31,040 bf16 sent, 248,320 bf16 (497 KB) held | upstream `logits_processor.py` `_gather_logits` | `compute_logits` (runner, proposer) |
| `mtp_fc` | all-gather | 320 + 1280 bf16 sent, 2560 + 4 x 2560 held | upstream `linear.py` `ColumnParallelLinear.forward` | `mtp.py` `fc_embedding`, `fc_hidden` |

Held bytes per rank per step at the real geometry (INFERRED from the recorded shapes): decode at 32 rows 30.5 MiB,
of which the logits all-gather is 15.2 MiB; verification at 128 rows 121.9 MiB, logits 60.6 MiB; the three
drafts of a k = 3 step at 32 rows 49.2 MiB, logits 45.5 MiB. No collective of a target step carries a payload
that is the same on every rank (the recorder hashes every rank's input).

## 2. The cuts (each behind its flag, default 0; `docs/configuration/env_variables.md`)

| Flag | Removes | Exact? | Cost |
|---|---|---|---|
| `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1` | the logits all-gather of plain greedy decode, greedy MTP verification and every MTP draft: each rank takes the largest logit of its own vocabulary shard (lowest id among equal values) and `[rows, 2]` fp32 per rank is gathered (8 B per row per rank instead of 62 KB sent and 497 KB held) | yes for every arg-max consumer: the largest element of a set is the largest of the largest elements of its parts; the values compared are the bf16 logits widened to fp32 (exact). Ties go to the lowest id, which is what `torch.argmax` returns on the CPU | none; the count stays (one small all-gather) |
| `VLLM_HPU_REPLICATED_EMBEDDING=1` | the embedding all-reduce of every target step and every draft step | yes, bit for bit (one rank held the row, the others added zeros) | 1.11 GB more per card (the whole 248,320 x 2560 bf16 table) |
| `VLLM_HPU_MTP_REPLICATED_FC=1` | the two all-gathers of every draft step | yes, bit for bit (each output column is the same dot product) | about 11 MiB more per card for each of the two 2560 x 2560 matrices |

Tests (8, 4, 2 and 1 gloo ranks, `tests/unit_tests/models/flash_next/test_collective_cuts.py`): with each flag
and with all three, a greedy stream (a prefill, six decode steps, one speculative step of k = 3 drafts and their
verification) gives the same decode tokens, drafts, verified tokens and the same bits of every hidden state as
without; the collective order is the same on every rank; the cut collectives are gone; TP1 issues none. The
runner gate (`test_hpu_model_runner.py::test_vocab_parallel_logits_opens_the_greedy_region_for_flash_next_mtp`)
and the verification from tokens (`tests/unit_tests/sampler/test_hpu_greedy_verify.py`, equal to the DFlash2
packed path and to the HPU rejection sampler's greedy rule) are CPU tests too.

What the runner does with flag 1 (`hpu_model_runner.py`): `_can_fuse_plain_greedy_sampling` opens the existing
fused greedy region (`select_compute_logits_and_greedy`) for plain greedy requests, and also under a Flash-Next
MTP speculative config; the decode branch then verifies the drafts with
`vllm_gaudi/v1/sample/hpu_greedy_verify.py` from the arg max of every row, with no sampler pass over logits. A
batch with one sampled request, logprobs, `min_tokens`, penalties, logit bias, token masks, structured output or any other
speculative method keeps today's path. `hpu_rejection_sampler.py` is not changed (the mtp-card-ready track builds
sampled verification there). The MTP proposer takes its drafts through `HpuFlashNextMtpProposer._draft_tokens`
(the draft-rank log keeps the gathered logits, it needs their top 4).

Mode 2 (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`, `vocab_parallel.top_k`) gathers the per-rank top `K` instead of the
logits. No serving path calls it yet. When it is exact for sampling:

- the `k` largest logits of the vocabulary are among the `k` largest of each rank's shard, so for `k <= K` the
  returned values are exactly the gathered top-`k` values. Among tokens tied at the `k`-th value which ones are
  returned is not defined (it is not for `torch.topk` either).
- top-k sampling with `1 <= top_k <= K`, any temperature above 0 and top-p applied after top-k (vLLM's order) is
  exact in distribution over the candidates: top-p and the softmax only see the top-k set.
- it is not exact without a top-k (`top_k` unset or above `K`), with min-p (the kept set can exceed `K`), with
  logprobs (they need the full-vocabulary normalizer; a log-sum-exp all-reduce of per-rank `(max, sum exp)` pairs
  would give it), or with penalties, logit bias, allowed or bad words applied to the full logits first (each
  could be applied per shard first, not done).
- realized tokens equal the gathered path's only in distribution: a sampler that draws one exponential per
  candidate draws different random numbers than one that draws per vocabulary entry. Every rank must draw the
  same numbers (a shared seed), because speculative bookkeeping (accepted counts, recurrent state slots) is
  computed on every rank.

## 3. What cannot be cut: two all-reduces per layer (VERIFIED in the code, algebra INFERRED)

The decoder layer (`vllm/models/qwen4_exp/nvidia/model.py` `Qwen4ExpDecoderLayer.forward`, hyper-connections in
`nvidia/hyperconnection.py` and `nvidia/ops/hc.py`): `R` is the 4-stream residual `[rows, 4 x 2560]`.

    mix:      xn_s = RMSNorm(R_s) (per stream),  [lora, inj] = xn @ W_down,  gate = silu(lora) @ W_up,
              block_in = 1/4 sum_s sigmoid(gate_s) * xn_s                                       (replicated)
    block:    out = sum over ranks r of out_r          (TP-sharded mixer, or EP experts + TP shared expert)
    combine:  R'_s = R_s + out * 2 sigmoid(inj_s / 4)                     (delayed into the next mix's norm)

- **Merge the mixer and MoE all-reduces of a layer: no.** The MoE input is `mix(R')` and `R'` contains the full
  mixer output; the per-stream RMSNorm needs `mean((R_s + c_s sum_r out_r)^2)`, whose cross terms
  `out_r . out_r'` exist on no single rank, and the silu and sigmoid follow. Every rank needs the whole mixer
  output before the MoE input exists. The same holds from the MoE output to the next layer's mixer input, and
  from the last MoE output to the final mixer and the vocabulary-parallel LM head.
- **Defer one: no.** The combine is linear in `out` given `R` and `inj`, but the first nonlinear op (the norm)
  comes right after it; deferring the reduction past the combine would reduce `R'`, 4 times the payload.
- **Replace an all-reduce by reduce-scatter + all-gather with the hyper-connection row-sharded between them:** the
  count doubles (96 to 192 per step) for the same bytes. In decode it saves nothing: the hyper-connection GEMMs
  are bound by their weights (1.30 GB per rank per step, read whatever the row count), and every rank still reads
  all of them. In prefill (8,192 rows) it cuts 7/8 of the replicated hyper-connection MACs: a prefill idea
  (target-model.md section 8, idea 15), not a decode one.
- **Replicate a block to drop its all-reduce:** the GDN weights and states would be 8 times larger per rank
  (about 3.6 GB more weights read per step, and 48 value heads of fp32 state per request per layer instead of 6):
  far more than 48 all-reduces cost.
- **Send the MoE output only from ranks that hold a routed expert: no.** Every rank also holds a shard of the
  shared expert, so every rank's partial is nonzero for every row. Without the shared expert's shard a rank has no
  routed expert for a row with probability `(1 - 64/512)^10 = 0.26`; at 32 rows the probability that a rank has
  no routed expert in the whole step is `0.875^320`, about `3e-19`. A captured HPU graph cannot make a rank's part
  in a collective depend on the routing anyway; an all-to-all dispatch and combine is two collectives per layer
  instead of one.
- **Overlap a collective with independent compute:** within a layer there is none (each op needs the previous
  collective's result); two half-batches would read every weight twice.

So a decode step keeps 96 layer all-reduces. With the cuts: 98 per target step (the embedding replicated, or the
PLE table rows on the embedding all-reduce, section 4), the logits payload goes from 497 KB per row to 8 B per row,
and a k = 3 draft pass from 6 to 3 collectives per token. What is left is the cost of each collective: section 5.

## 4. The PLE table all-reduce (`VLLM_HPU_PLE_TABLE_RIDER=1`)

The n-gram table is vocabulary-parallel and far too large to replicate (about 12.8 GB per card at TP4). Its lookup
needs only the ids, so with the flag the target's forward (`_MtpTargetOutputs._ple_table_rider` in
`vllm_gaudi/models/qwen4_exp/__init__.py`) looks up, before the embedding, the rows each rank's shards own for the
tokens and for the n-gram ids (`HpuQwen4ExpPLELayer.ngram_ids`, `local_table_rows`), all-reduces both as the columns
of one `[rows, 2560 + 16 x 160]` tensor, and leaves the two sums for the embedding's and the PLE layer's next call.
Every id has one owner, so each sum is the owner's row: the result is the same bit for bit (an FP8 table's rows are
widened to bf16 before the sum, which is exact, and its scale is applied after, as `_dequantize_embeddings` does).
One collective fewer per target step at no memory cost, the same count as `VLLM_HPU_REPLICATED_EMBEDDING=1`; with
the replicated embedding there is no embedding all-reduce to ride on and nothing changes. The draft has no PLE
layer. Riding on layer 0's first all-reduce instead (97 with the replicated embedding) would need the rider inside
`RowParallelLinear`; not built.

A cut that is not built: the arg-max gather of a draft step could carry, next to `(value, id)`, the embedding row of
each rank's candidate (`[rows, 2 + 2560]` per rank); the winner's row is then the next draft step's input
embedding, so the draft's embedding all-reduce of steps 1 to k - 1 goes without the 1.11 GB replicated table.

## 5. Cost per collective (C4, C6, the two-server layout)

`VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS=N` (`HpuCommunicator._all_reduce_as_gather`) is C4 without the fused kernel:
an all-reduce of at most `N` rows becomes an all-gather of the `world` partials and one local fp32 sum, the same
op on every rank, so every rank holds the same bits; an all-reduce with one owner per element (embedding, PLE
table) stays exact. It moves 7 times the payload per rank at 8 ranks against about 1.75 times for a ring
all-reduce, and adds the sum (one reduction over `[8, rows, 2560]`, about one graph op). It can pay only where an
all-gather is faster than an all-reduce by more than that op; the row gate is the I34 crossover. The fused form
(the sum inside `hc_combine_norm`, which would also take the sum op away) is worth building only after the card
shows the torch form pays. Tests: `tests/unit_tests/distributed/test_all_reduce_as_gather.py` (form, gate, exact
one-owner sums), `test_collective_cuts.py::test_small_all_reduces_as_gather` (fp32 tiny model on 2 and 8 ranks:
same tokens, hidden states within fp32 rounding, the same bits on every rank).

Measured so far (MEASURED, job 00009, `bench_collectives.py`, lazy mode, 4 ranks on cards 0-3; its 8-rank run
failed at device acquire, card 0 still held by the step before it): 98 all-reduces of `[rows, 2560]` bf16 in one
lazy graph cost, on the device (wait after `mark_step`, minus the same graph without them), 43 us each at 1 and 8
rows and 62 us at 32 rows, that is 4.2 to 6.0 ms per 98; one all-reduce as its own launch (with its `mark_step`
and sync) 0.16 to 0.24 ms at 8 to 32 rows, an all-gather of the same rows 0.19 to 0.21 ms. A `mark_step` before
every collective (`VLLM_HPU_COLLECTIVE_MARK_STEP=1`, the plugin's default) added 2.1 to 11 ms per 98.

The decision between C4, C6 and the two-server layout (G4-1) comes from the collectives track's job 00019, which
opens with the same payloads on 8 and then 4 ranks, lazy and HPU-graph replay, all-reduce against all-gather plus
local sum (`reports/collectives/job/bench_c4.py`), and from the track report.
