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
| `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1` | the logits all-gather of plain greedy decode, greedy MTP verification and every MTP draft: each rank takes the largest logit of its own vocabulary shard (lowest id among equal values) and `[rows, 2]` fp32 per rank is gathered (8 B per row per rank instead of 62 KB sent and 497 KB held) | yes for every arg-max consumer: the largest element of a set is the largest of the largest elements of its parts; the values compared are the bf16 logits widened to fp32 (exact). Ties go to the lowest id, which is what `torch.argmax` returns on the CPU (how the card's arg max over the gathered logits breaks an exact bf16 tie is not documented, so two servers may part at such a tie) | none; the count stays (one small all-gather) |
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
(the draft-rank log keeps the gathered logits, it needs their top 4). Prompt chunks are sampled by the general
sampler in the runner, so a prefill step keeps its logits all-gather (one row per prompt chunk, 497 KB); the gloo
harness's prefill phase picks its token through `top_tokens`, so its "prefill" count shows the reduced gather.

Mode 2 (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`) is the head's side of a sampled step. Real agents sample: the
checkpoint's generation_config (temperature 1.0, top_k 20, top_p 0.95) applies to every request that sets no
temperature, and such a request gets nothing from the arg max region. Today its step gathers `[rows, 248,320]`
logits on every rank and runs vLLM's sampler on them (a sort of the whole row, about fourteen passes over it; the
engine-step track's design note, section 7). The engine-step track builds the sampler stage that draws from
candidates instead (`vllm_gaudi/v1/sample/candidate_sampling.py`, `VLLM_HPU_CANDIDATE_SAMPLING`); this track gives
it the candidates without the logits all-gather:

- `vocab_parallel.top_k(model, hidden_states, k)`: `(values fp32 [rows, k] largest first, global ids int32
  [rows, k])`. With the flag at 2: each rank takes `torch.topk` of its shard, one all-gather of `[rows, 2k]` fp32 per
  rank (values, then ids; 512 B per row per rank at `k` = 64 against 62 KB of logits), and `topk` over the
  `ranks x k` candidates restores the order. With the flag at 0 or 1: `compute_logits` and `topk`. One call for the
  consumer, the same result (`test_vocab_parallel_candidates.py`, 1, 2, 4 and 8 ranks).
- `vocab_parallel.first_maximum(values, ids)`: a greedy row's token from the same candidates (the lowest id among
  the largest values, what the arg max region gives), so a mixed batch needs no second pass over the head.
- Card rule 9: no int64 id is cast. The local ids are read out of an fp32 id table with the `topk` indices as they
  are, ids cross the ranks as fp32, and the last cast is fp32 to int32 (a test watches every dispatched op for a
  cast of an int64 tensor; job 00064 runs the same ops on a card).

When the candidates are enough for sampling (VERIFIED by the tests for the value and id claims, INFERRED for the
rest):

- the `k` largest logits of the vocabulary are among the `k` largest of each rank's shard, so the returned values
  are exactly the gathered top-`k` values. Among tokens tied at the `k`-th value which ones are returned is not
  defined (it is not for `torch.topk` either);
- top-k keeps every token at or above the `top_k`-th largest value. For `top_k <= k` all of them are candidates
  unless more than `k` of them lie in one rank's shard, which needs a tie group at the boundary that reaches past
  column `k` (the sampler stage checks `values[:, k - 1] < kth` in the graph and such a batch keeps the gathered
  path);
- top-p (applied after top-k, vLLM's order), the softmax and the draw only see the kept logits, so their result is
  the same in distribution; the realized token differs, the noise being drawn per candidate;
- not enough: a request without a top-k or with one above `k`, min-p (its kept set is not bounded), logprobs (they
  need the full-vocabulary normalizer; a log-sum-exp all-reduce of per-rank `(max, sum exp)` pairs would give it),
  penalties, a logit bias, allowed or bad words (applied to the full logits first).

Sampled verification of an MTP step (mtp-card-ready, idea I07) needs the same candidates for every verification
row: `[32 x (1 + k), 128]` fp32 per rank instead of a 64 MB logits gather per step at k = 3. There every rank must
draw the same noise, because the accepted counts and the recurrent state slots are computed on every rank.

## 3. The structural question: two all-reduces per layer is the floor of this layout

Asked at 19:00 UTC on 2026-11-20, when the host cost of a collective was known (section 5): can the two all-reduces
of a layer become one, can the expert-parallel MoE send only from ranks that hold a routed expert, can several
collectives be issued as one. Short answer: **no, no, and only where two collectives do not depend on each other
(one pair per target step, already merged by section 4).** The count of a target step is 96 + 1 at best in any
layout that shards both the mixers and the experts; only a layout with fewer sharded blocks has fewer.

### 3.1 The layer, exactly (VERIFIED in the code)

`Qwen4ExpDecoderLayer.forward` (`vllm/models/qwen4_exp/nvidia/model.py:276-327`), `GatedResidual.mix` and
`combine_and_mix` (`nvidia/hyperconnection.py:131-190`), the kernels' arithmetic (`nvidia/ops/hc.py`:
`_grouped_gemma_rmsnorm_kernel`, `_hc_silu_kernel`, `_hc_gate_mix_kernel`, `_hc_combine_norm_kernel`). `R` is the
residual of 4 streams, `[rows, 4 x 2560]`; every weight below is replicated (`disable_tp=True`,
`ReplicatedLinear`), so every rank computes the same glue.

    combine + norm:  R'_s = bf16(R_s + c_s * a),  c_s = 2 sigmoid(inj_s / 4)      a = the pending block output
                     xn_s = R'_s * rsqrt(mean(R'_s^2) + eps) * (1 + w_s)                         (per stream s)
    down, inject:    [lora, inj'] = xn @ W_down          (10240 -> 320 + 4),  lora <- silu(lora / 4)
    up, gate mix:    gate = lora @ W_up  (320 -> 10240),  block_in = 1/4 sum_s sigmoid(gate_s) * xn_s
    block:           a' = sum over ranks r of a'_r(block_in)

The block is the mixer (GDN: `in_proj_qkvz` column-parallel, the recurrence per head, `out_proj` row-parallel;
QSA: heads split over the ranks, `o_proj` row-parallel) or the MoE (rank `r` holds experts `64r .. 64r + 63`
whole, and an 80-wide shard of the 640-wide shared expert: `Qwen3NextSparseMoeBlock`,
`vllm/model_executor/models/qwen3_next.py:128-260`; one reduction of experts and shared expert together,
upstream `moe_runner.py` `_maybe_reduce_final_output`). A layer is two such sub-blocks, 96 per step, and after
the last one the final mixer and the vocabulary-parallel head.

### 3.2 Why consecutive sharded sub-blocks cannot share a collective (algebra, INFERRED from 3.1)

Rank `r` needs `block_in` to run its shard of the next sub-block, and `block_in` is a function of the whole
`a = sum_r a_r`, not of any rank's part:

- the norm's scalar is `mean((R_s + c_s a)^2) = mean(R_s^2) + 2 c_s mean(R_s a) + c_s^2 mean(a^2)`. The middle
  term is linear in `a` (each rank could send one number per stream), but `mean(a^2) = sum_r sum_r' mean(a_r a_r')`
  has the products of different ranks' parts, which exist on no rank;
- after the norm come `silu`, `sigmoid` and the gated mean over the streams, which need the whole normalized
  vector, 2560 numbers per row, on every rank.

So between two consecutive sharded sub-blocks every rank must receive something that determines `a`: one
collective per sharded sub-block, whatever its form (all-reduce, all-gather of the parts, reduce-scatter and
all-gather). Consequences, each checked against the code:

- **Mixer and MoE of a layer in one all-reduce: no.** The MoE input is `mix(R')` with `R'` holding the mixer
  output; likewise from a MoE output to the next layer's mixer, and from the last MoE to the final mixer.
- **Defer a reduction past the combine: no.** The combine is linear in `a`, the norm right after it is not, and
  reducing `R'` instead of `a` is 4 times the payload for the same count.
- **Send the MoE output only from ranks that hold a routed expert: no**, for three independent reasons. Every rank
  holds a shard of the shared expert, so every rank's part is nonzero for every row (`shared_expert` is built
  with `reduce_results=False` and reduced with the experts). Without that shard a rank has no routed expert for
  a row with probability `C(448,10) / C(512,10) = 0.26`, so at one row about 2 of 8 ranks would have nothing to
  send, and at 32 rows the chance that a rank has none in the whole step is `0.26^32`, about `2e-19`. And a
  captured HPU graph cannot make a rank's part in a collective depend on the routing: the group and the order of
  collectives are fixed at capture (section 1: the same order on every rank). The cost is per collective, not per
  byte (section 5), so sending zeros costs what sending values costs.
- **Token dispatch and combine instead of all-gather and all-reduce (the classic expert-parallel pattern):** two
  collectives per MoE for one, plus the mixer's; more, not fewer.
- **Several collectives issued as one:** only collectives that do not depend on each other can share a call. In a
  target step that is exactly one pair, the embedding and the PLE table rows (both depend on the ids only):
  section 4 merges them. Every layer all-reduce depends on the one before it. (What the bridge does per call, and
  whether a grouped or coalesced call is cheaper, is the native-replay track's measurement: its job 00027,
  `docs/design/flash-next/native-replay.md` on its branch.)
- **Overlap a collective with independent compute:** within a layer there is none (each op needs the previous
  collective's result); two half-batches would read every weight twice.

### 3.3 Layouts with fewer sharded blocks (INFERRED; costs at 150 to 190 us per collective, section 5)

The floor follows the layout: a target step has `(sharded sub-blocks) + 1` collectives (the arg-max gather; the
embedding replicated or its all-reduce carrying the PLE rows).

| Layout | Layer collectives per step | What it costs | Verdict |
|---|---|---|---|
| TP8 mixers + EP8 experts (today) | 96 at 8 ranks | the reference | 97 or 98 with the cuts |
| Two servers, TP4 + EP4 each | 96 per server step at 4 ranks | the same count; a gain only if a 4-rank collective is much cheaper than an 8-rank one (the bench of section 5, the coordinator's job 00048); expert bytes per step rise (target-model.md section 7) | measured, not built here |
| PP2 x TP4, PP4 x TP2 | 96 per token (48 or 24 per stage) | the same count, fewer ranks per collective, the stages serial | no gain in count |
| Mixers replicated, experts EP8 | 48 | every rank computes all heads: GDN projections 8 times the MACs (0.26 -> 2.1 GMAC per row per rank), about 5.1 GB more weights read per step (GDN 3.6, QSA 1.5), 8 times the GDN state (113 MB per request against 14, the state pool 4.2 -> 33.6 GiB per card) and both QSA KV heads on every rank (12.3 -> 24.6 KB per token) | saves about 7 to 9 ms of host time at one stream and a short context; **does not fit the target** (32 x 128K needs 52 GB of KV per card at one KV head) |
| GDN mixers replicated only | 60 | the GDN part of the row above: the state pool at 33.6 GiB per card next to the KV cache | does not fit 32 x 128K either; a small-context, few-stream setup only |
| Experts replicated | 48 | 512 experts x 48 layers = 120 GB of FP8 weights per card | impossible |
| PP8 x TP1 (8 single-card stages of 6 layers) | 0, plus 7 card-to-card hand-offs of `[rows, 10240]` | serial stages: a step is the sum of 8 stage times, so at one micro-batch 7 of 8 cards idle; all cards busy only with 8 micro-batches in rotation (4 streams each), which vLLM's engine does not do, and each micro-batch reads its own routed experts (about 2.3 times the expert bytes of TP8 at 4 rows per stream); the hand-off through the host costs more than the collectives it removes, so it needs a device-to-device send | the one layout without the collective term; an engine project, not a flag (estimate: about 2.3 ms per stage at one stream, 18 to 22 ms per token against 37 today; with 8 rotating micro-batches of 4 streams about 27 ms per token per stream without speculation) |

So for the target (32 streams, 128K, speculation) the layer floor is 96 collectives per target step, and the two
levers left are the cost of each collective (section 5: its form, the rank count, the replay's host path) and
the number of target steps per token (speculation). What this track could still remove is counted in
section 6.

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

At 8 ranks (MEASURED, native-replay job 00012, `tools/flash_next/replay/bench_replay.py` on its branch, a 48-layer
body with GEMMs): one all-reduce costs 177 / 129 / 195 us at 1 / 32 / 128 rows under HPU-graph replay as the runner
does it (`wrap_in_hpu_graph`, `asynchronous=False`), 73 to 93 % of it host time inside the replay; 89 to 164 us in
plain lazy mode; 74 to 158 us under `torch.compile`. The real step agrees (MEASURED, step-anatomy job 00013, the
step timer on worker rank 0, TP8 + EP8): the replay call holds the host 22.8 ms of a 37.1 ms step at one stream,
27.1 of 46.4 ms at 8 and 35.2 of 83.6 ms at 32 streams; the wait for the device after it is 11.6, 15.0 and
42.7 ms. So the collectives of a decode step, 98 x about 150 to 190 us = 15 to 19 ms, are most of the fixed term of
today's step, and **every collective removed is worth about 0.15 to 0.19 ms per step** at TP8, whatever its
payload. In that light (INFERRED from the counts of section 1):

- the cuts of sections 2 and 4 remove 1 collective per target step and, with the replicated embedding and the
  unsharded MTP entry linears, 9 per k = 3 draft pass set (about 1.4 to 1.7 ms per MTP step); the logits gather's
  bytes come on top where they are large (16 MB per rank at 32 rows, 64 MB at 128 verification rows: an own-launch
  logits all-gather of 32 rows cost 0.8 ms at 4 ranks in job 00009);
- the per-layer floor of section 3 stays at 2;
- the cost of each collective is therefore the lever: its form (this section's bench), the rank count (the same
  bench, and the coordinator's two-server job 00048), and the host's work per collective inside the replay
  (native-replay: asynchronous replay, the bridge's host path).

**Job 00073 (MEASURED, 2026-11-20 21:24-21:29 UTC, `bench_c4.py` at commit 684b54ad, all 8 cards):** the cost of
one reduction of `[rows, 2560]` bf16 in a body of 96 (one element-wise op, then the reduction; `none` = the same
96 ops in one recipe), the slowest rank's median of 12 calls after 3 warm ones, each call behind a one-element
all-reduce barrier and timed until `torch.hpu.synchronize()` returns; per reduction = (form - none) / 96, so it
includes the recipe split each collective forces, as in a served step. Microseconds at 1 / 8 / 32 / 128 rows:

| ranks, regime | all-reduce | all-gather + local sum (the flag) | all-gather alone | `none` body (ms per 96) |
|---|---|---|---|---|
| 8, replay | 168 / 164 / 164 / 172 | 167 / 163 / 161 / 160 | 113 / 169 / 172 / 161 | 0.26 / 0.55 / 0.48 / 0.48 |
| 4 (modules 0-3), replay | 147 / 162 / 165 / 155 | 128 / 142 / 146 / 153 | 86 / 134 / 141 / 146 | 0.27 / 0.48 / 0.27 / 0.54 |
| 4, replay, both halves at once (1 and 32 rows) | 155 and 147 / 127 and 122 | 188 and 136 / 191 and 140 | | |
| 8, lazy | 59 / 179 / 181 / 72 | 280 / 308 / 208 / 175 | 204 / 244 / 155 / 120 | 15.9 / 11.7 / 11.6 / 12.3 |
| 4, lazy | 121 / 63 / 66 / 52 | 211 / 221 / 221 / 213 | 142 / 144 / 146 / 178 | 14.0 / 13.1 / 13.0 / 13.5 |

Readings (INFERRED from the table unless marked):
- Under replay a collective costs about 165 us at 8 ranks, flat in the payload (6.9 us per MB): host cost per
  collective. 98 per step = about 16 ms of every TP8 step, which agrees with the replay call of the real step
  (step-anatomy 00013: 22.8 ms on rank 0 at one stream = 98 x 165 us + 47 cut launches + about 2 ms).
- **Four ranks are not cheaper per collective under replay** (165 against 164 us at 32 rows): the two-server
  layout gains nothing on the collective term (coordinator job 00048: +10 % for the node, from fewer rows per
  server).
- **C4 does not pay at TP8:** gather + sum is within 1 to 8 % of the all-reduce at every row count, and the
  all-gather alone (the fused-kernel form) saves 55 us at one row and nothing from 8 rows on.
  `VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS` stays 0 and the fused form is not built. (At 4 ranks gather + sum is 12 to
  14 % cheaper; irrelevant while TP8 stays.)
- Lazy mode (no replay) is noisy and not what a decode step runs.
- Against native-replay's 00065 (45 to 74 us at one row, 109 to 151 at 32): the same slowest-rank, barrier and
  "without" definitions; its body carries GEMMs (up to 18 to 32 us per collective can hide behind them), its
  run had a stack sampler on rank 0 and a load average of 228 to 270, its regimes of equal cost spread from 24 to
  205 us, and its one-row point contradicts its own run (the segmented body without collectives took longer
  than the body with them: -18 us per collective alone). What a served step pays is about 150 to 170 us per
  collective (this job; 00012: 129 to 195; 00065 at 32 rows: 145), of which about 65 to 90 us is the recipe launch
  the collective forces (00065's `replay_seg`). A host path that made the collective itself free would save
  about 8 to 10 ms per TP8 step; collectives inside one recipe up to about 14 to 16 ms.

The step's last collective (the logits all-gather against the `[rows, 2]` pair gather, `--tail-rows`) is not in
00073 (its image predates it) and is not run: one collective and its payload per step is small next to the
floor (coordinator, 2026-11-20).

## 6. Card jobs of this track, their readings and their results

Every job runs the committed tree in the farm's per-commit image (`CARD.md`); scripts in
`tools/flash_next/collectives/`.

| Job | What | Reading | Result |
|---|---|---|---|
| 00064 `idcheck-rule9`, 1 card, image 3abbd8df, 20:26 UTC | `id_check.py`: every id pattern of the token path against the CPU, one lazy graph and a replayed one, rows 1, 2, 5, 32, 128, untied and tied, at a shard's width and at a whole row | serving cases (`vp_argmax`, `fp32_ids_row`) must equal the CPU; the probes say which patterns card rule 9 breaks | **PASS.** Both serving cases equal the CPU in every case. Every arg max form is right everywhere (`argmax -> int32`: the fused greedy region and the sampler; `argmax -> .int()`: the MTP draft loop; int64; two graphs; fp32). `topk -> int32` read to the host: every second id at every row count, but only in the first lazy pass (a freshly compiled graph); right in the replayed graph and in the later lazy pass at the same shapes. `max(dim).indices -> int32`: the same at 2 rows and more. `topk` ids as they are, cast to fp32, or read out of an fp32 id table: right everywhere |
| 00108 `idcheck-shapes`, 1 card, image 1c9abfef, 20:28 UTC (replaced 00081) | `id_check.py`: the top-k forms at k = 20 and 64 on a shard of 31,040 columns with 64 known largest logits, the arg max forms, and the cases whose narrowed ids are consumed inside the graph | this branch's top-k must equal the CPU at every row count | **PASS.** `vp_topk@20/@64` (this branch's `vocab_parallel_topk`), `topk_gather_fp32`, `stage2_topk_gather`, `first_maximum` at k = 20 and 64: equal at every row, both regimes, untied and tied, shard and whole row. `topk_to_int32@k` and `max_dim_to_int32` wrong again only in the first lazy pass. In-graph consumers right even there: `argmax_int_embed` (the MTP draft loop's ids into the embedding) and `topk_int32_embed@k` (the QSA block selection's pattern) |
| 00073 `c4-bench-r8-r4`, 8 cards, image 684b54ad, 21:24-21:29 UTC | `bench_job.sh`: the id check through the all-gather on 8 ranks; `bench_c4.py` on 8 ranks, on the module group 0-3, and on both groups at once | alpha per rank count, regime and form (section 5) | **Exit 2 from the first stage**: the 8 id-check ranks hit the stage's 150 s limit and the script killed them (the "killed while waiting" kernel lines on cards 0, 4, 5, 6, 7; not a card fault; load average up to 414). Before that `vp_argmax` through the 8-rank all-gather equalled the CPU at every row, regime and data set, and `vp_topk` (the earlier form) in every case that ran. The bench stages ran to the end: section 5 |
| 00112 `cuts-cardproof-tp4`, 4 cards (modules 0-3), commit d1cbe91d (the head of PR #14), 23:08-23:12 UTC | `cardproof_job.sh`: `id_check.py` on 4 hccl ranks; the tiny model at TP4 with fused hyper-connections, arm `logits-rider` (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 VLLM_HPU_PLE_TABLE_RIDER=1`) and arm `logits2-embed` (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=2 VLLM_HPU_REPLICATED_EMBEDDING=1`) | the serving id cases equal the CPU; every arm's tokens equal the reference | **Ids PASS at 4 ranks**: `vp_argmax` and `vp_topk@20/@64` (the merged top-k head through the 4-rank all-gather) equal at rows 1, 2, 32, 128, both regimes, untied and tied. **Tiny: exit 1.** Both arms give identical tokens: prompts of 7, 33, 2,600 tokens 16/16; 1 and 100 tokens near ties at token 0 (excused, as in every baseline); the 700-token prompt 4/16, first difference at token 4. The same 700-token answer, token for token, comes from engine-step's 00093 `tiny-flags` at TP1 (`VLLM_HPU_FUSED_GREEDY_LOGITS=1`, none of this track's flags) and from mtp-card-ready's 00105 fused-verify arms at TP2, while every baseline tiny run matches 16/16. So it comes with the in-graph greedy region (or any small numerical change), not with the cuts: a near tie of the reference at token 4 (coordinator, 2026-11-20; the tiny model's logits are near-flat: top-2 margins of 0.0005 to 0.02 at values near 0.27, where one bf16 step is 0.002, on a same-recipe checkpoint built on the Mac) |
| 00082 `real4-ab2-off`, 00083 `real4-ab2-logits-rider` (commit d158c6d5) | the real-model A/B against the plugin's default path | | replaced by the pair below: against the default path the lone texts can part at a bf16 tie that the greedy region itself moves, which says nothing about the cuts |
| **00124** `real4-ab3-region` / **00125** `real4-ab3-cuts`, 4 cards each, 25 min, commit 0a66722b, in the places of 00082 / 00083; **cancelled at 23:44 UTC before running** (coordinator: 00109 first); commands for the next rental in the handoff | `ab_job.sh` on this branch's head: the real model at TP4 + EP4, 1 and 32 streams, 256 tokens, no `min_tokens`, every stream's text kept; 6 lone requests; one 32-stream round with the server's default sampling. Arm `region`: `VLLM_HPU_FUSED_GREEDY_LOGITS=1` (the greedy region over the gathered logits); arm `cuts`: `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 VLLM_HPU_PLE_TABLE_RIDER=1` (the same region, the vocabulary-parallel arg max, the PLE rows on the embedding all-reduce); both `VLLM_GRAPH_RESERVED_MEM=0.5` | `compare_arms.py <region> <cuts>`: the 6 lone requests' texts must be equal token for token (both arms compute the same bf16 logits; the cuts are exact); the ms per step at 1 and 32 streams is the cuts' own effect (expected under 1 ms: one all-gather of `[rows, 62,080]` bf16 per rank and one all-reduce fewer) | pending |

A probe that sends `min_tokens` measures none of this: a request with `min_tokens` is kept out of the greedy region
(`_can_fuse_plain_greedy_sampling`, `hpu_model_runner.py`), so the logits are gathered and sampled as before.

## 7. Our own build of the PyTorch bridge in farm jobs (deployment; from 2026-11-21 00:40 UTC)

Why (coordinator, 00:40 UTC): about 150 of every 188 us of host time per graph launch is the PyTorch bridge's
own work (step-anatomy, RESULTS.md "Integration card build"); native-replay writes a bridge-only "prepared replay"
patch against gaudi-pytorch-bridge v1.24.1 (5176b1b2, the node's exact version, `habana-torch-plugin 1.24.1.482`
in the image). This track deploys native-replay's builds into farm jobs and qualifies them on the card.

Interface (agreed with native-replay, 00:45 UTC):
- **Builds** (native-replay, no card): `~/farm/native-replay/bridge-build/out/<build-id>/` on the node with
  `wheels/*.whl`, `manifest.json` (`build_id`, `bridge_commit`, `patch_sha256`, `built_at`, `wheels` {file:
  sha256}, optional `files` {file: [sha256, destination]}, `env_flag`) and `build.log`; a copy under
  `~/farm/native-replay/results/bridge-builds/<build-id>/`. Ids: `stock-5176b1b2` (unpatched),
  `lean-<12 hex of the patch's sha256>` (patched).
- **Switch:** `bin/farm job|tiny|serve ... --bridge <build-id>` (this track's change to `bin/farm`, 2026-11-21;
  the diff is `tools/flash_next/bridge/farm-bridge-option.diff` against card-farm's `farm.sh` at a2fd0cf8: a
  `farm install` from wt/card-farm would undo it). The job script copies the build into the job's own results
  folder when the job starts (the container's `$FARM_OUT/bridge`) and wraps the job's command with
  `tools/flash_next/bridge/with_bridge.py` (deployed to `~/farm/collectives/bridge/with_bridge.py`): every
  wheel's sha256 against the manifest, `pip install --no-deps --force-reinstall --no-index`, then a fresh process
  imports the bridge, runs one HPU op and must map only the wheels' `.so` files (md5 from `/proc/self/maps`,
  also for a bridge library loaded from outside the package, e.g. through `LD_LIBRARY_PATH`); `bridge.json` in
  the results, one `BRIDGE <build-id> ok: ...` line in the log; exit 7 (cannot install) or 8 (does not load, or
  not what was installed); the compiled-recipe cache moves to `$FARM_RECIPE_CACHE/bridge-<build-id>`. No image
  per combination; the install costs seconds of the job's hold. CPU test:
  `tests/unit_tests/tools/test_with_bridge.py` (a fake bridge in a scratch venv; 5 cases).
- **Flag:** `PT_HPU_HPUGRAPH_PREPARED_REPLAY` (0 = stock replay, the default), read by the bridge itself; set it
  with `--env` (a stock bridge ignores it).
- The image's bridge (VERIFIED on the node, 00:50 UTC): `/usr/local/lib/python3.10/dist-packages/habana_frameworks/
  torch/lib/` holds `libhabana_pytorch{,2}_plugin.so`, `libhabana_pytorch_backend.so` and their `.upstream.so`
  twins; `python` and `python3` are both `/usr/bin/python3.10`; `/usr/lib/habanalabs` (on `LD_LIBRARY_PATH` and
  `PYTHONPATH`) holds no bridge library in the image (the build container's build system links its own there).

Qualification, in order (each decides the next):

| Step | Job | Decides |
|---|---|---|
| a1 | the tiny model at TP1, `--gdn hc`: stock (plain `farm tiny`) and `--bridge stock-5176b1b2`, two 1-card jobs side by side | the deployment path: the rebuilt, unpatched bridge loads (BRIDGE ok, md5s) and gives the tokens of the stock bridge |
| a2 | the same with `--bridge stock-5176b1b2` at TP8 (all) | the same through HCCL at 8 ranks |
| a3 | `--bridge lean-<x>`, tiny at TP1 and TP8, `PT_HPU_HPUGRAPH_PREPARED_REPLAY=0` and `=1` | the patch keeps the tokens with the flag off and on |
| b | native-replay's `bench_replay.py` at TP8 (`replay`, `replay_seg`), lean build, flag 0 against 1 | the host cost per launch the patch removes |
| c | the real model at TP8, 1 and 32 streams (`serve_probe.sh` + `decode_scaling.py`): stock bridge, lean flag 0, lean flag 1 | the step time; texts equal at 1 stream |

**Job 00160 `tiny8-bridge-check` (MEASURED, all 8 cards, 02:03-02:08 UTC on 2026-11-21; a1 and a2 merged into one
job at the coordinator's word): PASS.** The tiny model at TP8 (`--gdn hc`) with the image's bridge, then with
native-replay's unpatched rebuild `stock-5176b1b2` installed by `--bridge`: `BRIDGE stock-5176b1b2 ok:
habana_frameworks.torch 1.24.1+git5176b1b ...; 23 .so from the wheels, 14 mapped, 18 differ from the image's; torch
2.11.0a0+git009...` (every bridge library the process maps is the wheel's; the `.upstream` twins the build does
not make were not needed); both smokes passed (prompts of 7, 33, 700 and 2,600 tokens 16/16, the usual near ties
at token 0 for 1 and 100 tokens) and **the tokens are equal prompt by prompt, near-tie lists included**. The install
and check took under 3 minutes of the arm (stock arm 84 s, bridge arm 167 s). So a rebuilt bridge reaches the
card through `--bridge` and is what runs. Results: `reports/collectives/card/00160-tiny8-bridge-check-{stock,build}/`.

**Job 00174 `real8-bridge-phase` (MEASURED, all 8 cards, 02:33-03:06 UTC, exit 0, all six arms ok):**
native-replay's `timers-c281d8b53d08` (phase timers) and `lean-e1dbb4942fb8` (timers + prepared replay step 1),
and the timers build with `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1` (no accumulation thread, so no spinning CPU per
worker). Tiny model at TP8: the lean (flag 1) and accsync arms give the timers arm's tokens prompt by prompt. Real
model at TP8, steady ms per step (1 stream: one round; 32 streams: two rounds), the 1-stream text identical in all
three arms:

| arm | 1 stream | 32 streams |
|---|---|---|
| timers (reference) | 38.2 | 84.2, 84.7 |
| lean, `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1` | 30.8 (-7.4) | 83.7, 83.6 (-0.8) |
| accsync, `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1` | 34.6 (-3.6) | 82.4, 82.2 (-2.2) |

Host time per graph launch from the `HPUGRAPH_REPLAY_TIMING` lines (median over every line after each pid's first;
each line averages one thread's last 5,000 launches; us; `reports/collectives/probes/replay_timing_table.py`):

| phase | timers | lean | accsync |
|---|---|---|---|
| total | 155.4 | 137.8 | 138.8 |
| exec_setup | 6.78 | 1.52 | 5.83 |
| launch_hashes, create_launcher, run_spec, run_lookup | 10.92 | 0.00 | 9.50 |
| run_prep | 1.75 | 14.05 | 1.38 |
| run_patching | 22.4 | 16.9 | 19.2 |
| launcher_inputs | 29.7 | 33.3 | 29.6 |
| syn_launch | 48.9 | 41.0 | 43.7 |
| launcher_post | 14.6 | 13.0 | 12.8 |

Readings (INFERRED from the tables): the prepared path engages (the hash, launcher, spec and lookup phases read 0)
but adds 12 us in `run_prep`, so it saves 17.6 us per launch net; at one stream (host-bound) that is the largest
step gain measured tonight, at 32 streams (device-bound) little. Without the accumulation thread almost every phase
is 10 to 20 % cheaper (the spinner's CPU contention). The launch's host time is now mostly the Synapse launch (41 to
49 us), input handling (about 30), patching (17 to 22) and post-launch work (13 to 15). Not yet run: lean and accsync
together; accsync on the image's bridge (it is a stock setting). Results:
`reports/collectives/card/00174-real8-bridge-phase-*/`.

**Job 00181 `real8-bridge-headline2` (MEASURED, all 8 cards, 04:24-04:56 UTC, exit 0, six arms ok):** integration
build 3's image and no-speculation env with the accumulation thread off in every arm, build 3's probe; the reference
is the image's own (vendor-built) bridge. Tokens: tiny-lean and tiny-lean-inline equal tiny-stock prompt by prompt;
the 1-stream texts (both rounds) are equal in all three real arms. Steady ms per step, two rounds each:

| arm | greedy 32 | greedy 8 | greedy 1 | sampled 32 | sampled 8 | sampled 1 |
|---|---|---|---|---|---|---|
| vendor bridge | 49.4, 56.4 | 53.3, 53.5 | 37.6, 37.2 | 57.6, 63.5 | 59.3, 57.7 | 36.2, 38.2 |
| lean3t (prepared replay) | 46.2, 47.0 | 63.1, 61.3 | 43.5, 43.6 | 47.9, 50.7 | 59.4, 59.6 | 44.5, 44.7 |
| lean3t + `PT_HPU_DISABLE_ASYNC_COLLECTIVE=1` | 45.5, 46.4 | 61.1, 59.9 | 42.2, 43.4 | 51.8, 51.0 | 63.2, 60.5 | 44.0, 44.5 |

Per launch (us; lean3t / inline): total 143.8 / 133.7; `run_prep` 2.14 / 2.20 (native-replay's fix works);
unattributed 5.0 / 5.3; `launcher_inputs` 40.1 / 6.0 (the waits on the async collectives' futures: `sub_in_waits`
39.0, `sub_in_future` 34.2, gone inline); `launcher_post` 13.7 / 35.9 (the collective runs there inline);
`syn_launch` 43.6 / 48.3 (`sub_syn_call` 27.9, `sub_syn_lock` 9.9); `run_patching` 24.4 (`sub_patch_table` 22.8,
`sub_patch_alloc` 14.4). Readings (INFERRED): against the vendor's bridge, lean is faster at 32 streams (3 to 10 ms
greedy, 8 to 13 sampled; the vendor arm's own rounds differ by 6 to 7 ms) but about 6 ms slower at 1 stream and 8 ms
at 8 streams, the opposite of 00174 (where the reference was our own rebuild); per-launch host time cannot explain a
6 ms step difference, so either our rebuild as such is slower than the vendor's build in this env or the cost lies
outside the timed launch path. The vendor bridge stays in the headline builds (coordinator).

**The per-rank spread (native-replay, from 00174's and 00181's timing lines; MEASURED per rank):** in every real arm
2 of the 8 ranks run about 2x slower per launch for long stretches, every phase 2x (a slower CPU, not more bridge
work; INFERRED: two busy worker threads on hyperthread siblings of one host core); which two changes with each server
start. Split by level with vLLM's `Running: N reqs` metrics lines (00181 real-lean, median us per launch): the two
slow ranks 223 and 235 at 1 stream, 179 and 202 at 8, the other six 140 to 157 and 114 to 142; at 32 streams all
eight 128 to 155. At 1 stream the slowest rank sets the step (about 150 launches x 90 us = 13 ms), so the step-time
flips between 00174 and 00181 are this spread, not lean; the per-launch medians are attributable (lean -17 us, the
`run_prep` fix -12 us, inline collectives -10 us net). Our rebuild and the vendor's build are the same code
generation (native-replay: GCC 13.4.0, no LTO in either, .text sizes within 0.002 %).

**Job 00189 `real8-bridge-vendor2`** (all, 95 min, queued 05:03 UTC in the place of 00188 after integration's 00186
and 00187; build 4's image `farm-integration:30f4e37ef08c`, second tag `farm-collectives-int:30f4e37ef08c`;
`bridge_vendor_job.sh`, coordinator 05:00 and 05:10 UTC): build 4's no-speculation env without the
accumulation-thread setting; tiny gates (vendor, lean3a with the prepared flag 0, lean3a with the flag 1); then the
vendor bridge with the accumulation thread on (default) against off (00178 ran 29.4 to 29.7 ms at 1 stream with it
on, 00181's vendor arm 37 ms with it off: possibly the spread); the job keeps the setting with the lower sum of
greedy ms per step at 32 + 8 + 1 (`<job>-<name>-spinner.txt`) for real-rebuild (lean3a with the flag 0 and timers:
our build on the stock path, the control 00181 lacked), real-lean (`lean3a-8f6f7faf88dc`, flag 1: lean3t + no GIL
release for inputs whose event is done, no argument building for logs and traces that are off) and real-pin (the
same + `VLLM_HPU_WORKER_CPU_PIN=1`; mode 2 would put the eight main threads on CPUs 19m, possibly four sibling pairs).
Native-replay's sibling-map probe (its job 00193) found that vCPUs 2k and 2k+1 share a host core (all 12 tested
a^1 pairs slow each other, no other pair does); its placement script `place.sh` (md5 7e54f2de, read in full: inline
Python, it only sets thread affinities inside its own container) gives each worker's main thread one vCPU of its
share with the sibling kept free, keeps every other thread off those, and re-applies every 2 s. Staged at 05:43 UTC as
`~/farm/collectives/results/bench/place.sh` with `PIN_PLACE=/results/bench/place.sh` in
`~/farm/collectives/bridge/late/00189.env`: the pin arm runs last, on its own, through `place_wrap.sh` (the server and
probe start at once; the script runs once the server answers `/health`; `placement.log` in the arm's results), with
`VLLM_HPU_WORKER_CPU_PIN=1` kept. Pending.

**Job 00198 `real8-bridge-reuse`** (all, 90 min, queued 06:32 UTC behind 00189; `bridge_reuse_job.sh`): the
qualification of native-replay's `lean4-44fe39c26127` (lean3a + output-buffer reuse: `PT_HPU_HPUGRAPH_PREPARED_REUSE=1`
with `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`, a launcher reuses its outputs from two launches ago when it holds the only
references; `=2` also fills every reused buffer with NaN before the launch, so a stale read shows as garbage) on build
4's image and env. Tiny gates tiny-stock / tiny-lean4-nan / tiny-lean4; real arms real-ref (REUSE=0, exactly lean3a),
real-lean4-nan, real-lean4, each running the headline probe and then native-replay's `text_check.py` (md5 1d8cb0c9)
at 1 stream (6 prompts) and 32 streams, 600 tokens each (the probe dir `~/farm/collectives/results/bench/intprobe4r/`:
build 4's `serve_probe.sh`, its `decode_scaling.py` as `decode_scaling_real.py`, `decode_scaling_textcheck.py` as
`decode_scaling.py`, `text_check.py`). Verdicts: the texts_1 of both reuse arms equal real-ref's; no texts_32 with
NaN garbage (replacement characters over 0.5 % or a run of 40 or more). The spinner (`SPIN`), the pin (`PIN_KV`) and
the placement (`PLACE`) are late-bound from 00189's result through `~/farm/collectives/bridge/late/00198.env`.
Pending.

**The card queue's CPU layout (coordinator, 06:5x UTC; live since 06:58:07 UTC):** the image-build CPUs 18, 37, 56,
75, 94, 113, 132, 151 had their hyperthread siblings (vCPUs 2k / 2k+1, native-replay's 00193) inside job cpusets;
in 00190 four builds during one arm raised the graph-launch p50 from 143 to 232 us. `~/farm/cardq.sh` (installed by
rename; the previous one is `~/farm/.cardq.sh.before-cpu-layout`) now gives builds whole cores when
`~/farm/q/cpu_layout` says `cores <epoch>`: in each pair of modules the middle core (18-19, 56-57, 94-95, 132-133),
and every module 18 vCPUs of whole cores (0-17, 20-37, 38-55, 58-75, 76-93, 96-113, 114-131, 134-151). A job that
started before the switch keeps the old layout for every container it starts; while such a job runs, builds keep to
18, 56, 94, 132 (build CPUs in both layouts). `cardq.sh adopt` gave the 18 waiting jobs supervisors with the new code.
Tests: the old outputs unchanged without the file (20 queries), card-farm's `test_sh_cardq.py` and
`test_sh_farm_job.py` against the new script (64 passed, 1 skipped), a new `test_sh_cardq_cpu_layout.py` (4 pass).
The patch for card-farm's source (`farm install` would undo it): `tools/flash_next/bridge/cardq-cpu-layout.diff`.
Rollback: `mv ~/farm/q/cpu_layout ~/farm/q/cpu_layout.off`. The plugin's `VLLM_HPU_WORKER_CPU_PIN` and native-replay's
`place.sh` cut the container's own CPU mask into equal shares, so they follow the new layout.

## Handoff (if this agent stops)

State at 07:05 UTC on 2026-11-21 (fourth agent of this track, Opus 5.5). **The node stays until about 00:00 UTC
on 2026-11-22** (BRIEF 00:50 UTC). Branch `feat/flash-next-gaudi-collectives` (and `-dev`, the same head), worktree
`/Volumes/Storage/Claude/gaudi-work/wt/collectives`; card results and logs in
`/Volumes/Storage/Claude/gaudi-work/reports/collectives/` (`card/<id>-<name>/`, `logs/`, `probes/`). This section is
the track's report: the harness refuses a REPORT.md.

**Current task (coordinator, 00:40 UTC): deploy native-replay's builds of the PyTorch bridge in farm jobs and
qualify them on the card** (section 7; native-replay, agent `abe5ca8da150f80ba`, writes the "prepared replay"
patch and makes the builds). Done:
- `with_bridge.py` (in-container install and load check) and its CPU test (5 pass); `ab_bridge_job.sh` (TP8
  real-model A/B, one server per arm in one all-card job; dry-run on the Mac with a stub `farm_job.sh`). Both
  deployed to `~/farm/collectives/bridge/` on the node (md5 `3933df6a...` and `fbac8df0...`).
- `bin/farm` has `--bridge <build-id>` (job, tiny, serve), installed by rename at 00:49 UTC; the diff is
  `tools/flash_next/bridge/farm-bridge-option.diff` (card-farm's `farm.sh` at a2fd0cf8 needs the same change: a
  `farm install` from wt/card-farm would undo it).
- Interface agreed with native-replay (section 7). The coordinator has the plan and was asked for places.

The unpatched build `stock-5176b1b2` landed at 00:57 UTC (wheels `habana_torch_plugin-1.24.1+git5176b1b`, 44 MB,
23 `.so`: the three main libraries and `fork_pybind/`, no `.upstream` twins; `habana_torch_dataloader`).
**Job 00160 `tiny8-bridge-check`** (all, 20 min, queued 00:58 UTC; the coordinator places it after the 8-card job
running then) is the deployment check (the coordinator merged a1 and a2 into it: 8-card jobs first):
`ab_bridge_job.sh --sha 0a66722b7a55 --cmd 'python /farm/farm_tiny.py --gdn hc' stock:- build:stock-5176b1b2`.
Results: `card/00160-tiny8-bridge-check-{stock,build}/` (fetched by this agent's waiter, or `bin/farm fetch
collectives 00160 00160-tiny8-bridge-check-stock 00160-tiny8-bridge-check-build`). Pass: the job's exit code 0 =
both smokes pass, `BRIDGE stock-5176b1b2 ok`, and the line `tokens EQUAL to the stock arm's`.
Step b is staged: `~/farm/collectives/results/bench/{bench_replay.py (md5 a62658ce, native-replay's late copy),
run_bench.sh}`; with the lean build: `printf 'exec bash ~/farm/collectives/bridge/ab_bridge_job.sh --sha
0a66722b7a55 --cmd "bash /results/bench/run_bench.sh" stock:- lean:<lean id>\n' | bin/farm submit collectives
bench8-bridge all 30m` (each arm pairs the flag off and on in one bench run; summary:
`python3 reports/native-replay/probes/knob_table.py bench_r8.json`).

Next, in order (section 7's table):
1. Card results so far: 00160 PASS, 00174 and 00181 ok (section 7), each sent to the coordinator and
   native-replay. **Waiting: 00189** (section 7; this agent's waiter fetches its eight arms, the verdicts and
   `00189-real8-bridge-vendor2-spinner.txt` into `reports/collectives/card/`; else `bin/farm fetch collectives 00189
   00189-real8-bridge-vendor2-<arm>...` and `bin/node-scp :farm/collectives/results/00189-real8-bridge-vendor2-
   {verdicts,spinner}.txt <dir>/`). Report the spinner pair and the rebuild against the vendor to the coordinator
   first; then, for every timed arm, `python3 -I reports/collectives/probes/replay_timing_table.py <arm folder>`:
   per-rank medians per level (the headline now), the pooled phase and `sub_*` medians, and the pin arm's eight
   `VLLM_HPU_WORKER_CPU_PIN` lines (`grep WORKER_CPU_PIN <real-pin>/server0.log`), to native-replay. Drop the
   00189's pin verdict (do the 2x-slow ranks at 1 and 8 streams go away under the placement?) also goes to
   async-sched (agent `a58988ba328b40a9b`): its TP8 jobs 00196 / 00197 can take the placement behind a switch of
   their own (told at 06:45 UTC how: their job copies `place_wrap.sh` and `place.sh` from
   `~/farm/collectives/results/bench/` into their results folder and wraps each server's `serve_probe.sh` with
   `PLACE_ARGS="--log <out>/server0.log"`; default off until 00189's verdict).
   **Then: set 00198's late file from 00189's result** (`SPIN=on|off`; if the pin arm closed the per-rank spread:
   `PIN_KV=VLLM_HPU_WORKER_CPU_PIN=1` and `PLACE=/results/bench/place.sh`), before 00198 starts; its waiter fetches
   its six arms and verdicts; to native-replay: the three texts_1 comparisons, the hit rates (`n_reuse_*` in the
   parser), the per-rank per-level medians and the `sub_*` medians. Drop the
   protective image tags when no job needs them: `sudo -n docker rmi farm-collectives-int:d00c0a9a3e18
   farm-collectives-int:30f4e37ef08c` on the node.
2. **PR #19 merged at 04:05 UTC** (program head 22ec5ea3): the gated snapshot 04f690be (gate 88 files, 0 red; tool
   tests 10 pass; card 00160 and 00174) from its own branch `feat/flash-next-gaudi-collectives-bridge-gated`. The
   branch has merged the new head (214125e7). The next PR: the scripts' later features (late-bound settings,
   `--ref-arm`, the inline and `EXTRA` pairs) and this note, after 00181 proves them.
1b. native-replay's next builds: `timers-c281d8b53d08` (phase timers: `PT_HPU_HPUGRAPH_REPLAY_TIMING=N` prints one
   `HPUGRAPH_REPLAY_TIMING` line per N replayed launches per thread; about 01:47 UTC) and, about 20 minutes later,
   `lean-e1dbb4942fb8` (the timers plus the first prepared-replay step behind `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`;
   patch `tools/flash_next/bridge/patches/lean1.patch` on native-replay's branch, d9a1f8f1). One all-card job of
   six arms, `bridge_phase_job.sh TIMERS LEAN SHA` (deployed to `~/farm/collectives/bridge/`; approved by the
   coordinator, hold 65 min; **submitted as 00174 at 02:09 UTC**): tiny-timers (reference), tiny-accsync, tiny-lean
   (`PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`: no accumulation thread, no spinning CPU per worker), then
   real-timers, and real-lean / real-accsync only where their tiny arm passed with the reference's tokens
   (verdicts in `<id>-real8-bridge-phase-verdicts.txt`). A background task of this agent
   (`scratchpad/collectives/auto_submit_phase.sh`, started 01:33 UTC) submits it once 00160 ended with exit 0 and
   the timers build exists (waiting up to 25 more minutes for the lean build; without it the lean arms drop).
   If the agent is gone and nothing was submitted:
   `printf 'exec bash ~/farm/collectives/bridge/bridge_phase_job.sh timers-c281d8b53d08 lean-e1dbb4942fb8 0a66722b7a55\n' | bin/farm submit collectives real8-bridge-phase all 65m`.
   Results: `<id>-real8-bridge-phase-<arm>/` (real arms: `server0.log` with the timing lines, `scaling.json`,
   `bridge.json`; tiny arms: under `hc/`); native-replay parses the timing lines.
2. (Replaced by 00160: the 1-card pair a1 and the single TP8 arm a2.)
3. a3 and b with the lean build (flag 0 and 1); c with `ab_bridge_job.sh` through `bin/farm submit`:
   `printf 'exec bash ~/farm/collectives/bridge/ab_bridge_job.sh --sha 0a66722b7a55 stock:- lean0:<id>:PT_HPU_HPUGRAPH_PREPARED_REPLAY=0 lean1:<id>:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1\n' | bin/farm submit collectives real8-bridge-ab all 45m`,
   then `bin/farm wait <id>` and `bin/farm fetch collectives <id> <id>-real8-bridge-ab-stock <id>-real8-bridge-ab-lean0 <id>-real8-bridge-ab-lean1`.
4. Every result into section 7 and to the coordinator and native-replay.

Earlier work of this track (sections 1 to 6): the cuts are merged (PRs #13, #14) behind their flags; the card
proved the arg max and the top-k head (00064, 00108, 00112, 00073); a collective costs about 165 us of host time
under replay at TP8, flat in the payload, no cheaper at 4 ranks or as gather plus sum (00073). The real-model
pair for mode 1 + rider (00124 / 00125) was cancelled at 23:44 UTC to let prefill's 00109 run; its commands (in
section 6's last row and below) can run on this node now that it stays:
`bin/farm job collectives real4-ab3-region any:4 25m --env VLLM_HPU_FUSED_GREEDY_LOGITS=1 --env VLLM_GRAPH_RESERVED_MEM=0.5 -- bash tools/flash_next/collectives/ab_job.sh --levels 1,32 --tokens 256 --warm-tokens 320 --repeat-top 2`
and the same with `real4-ab3-cuts` and `--env VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 --env VLLM_HPU_PLE_TABLE_RIDER=1`
instead of the first `--env` (submit the second after the first prints its job id). Pull requests are allowed
again (BRIEF 00:50 UTC): the note and the bridge tools (no runtime code; CPU test green, card check 00160 passed)
go into the program branch through `bin/gate` when its queue clears (02:12 UTC: engine-step's gate running for 90
minutes, four tracks waiting): merge `origin/feat/flash-next-gaudi` (about 20 commits ahead) first.
