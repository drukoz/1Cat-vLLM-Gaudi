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
| **00124** `real4-ab3-region` (cards 0,1,2,7) / **00125** `real4-ab3-cuts` (cards 3,4,5,6), 4 cards each, 25 min, commit 0a66722b, side by side, in the places of 00082 / 00083 | `ab_job.sh` on this branch's head: the real model at TP4 + EP4, 1 and 32 streams, 256 tokens, no `min_tokens`, every stream's text kept; 6 lone requests; one 32-stream round with the server's default sampling. Arm `region`: `VLLM_HPU_FUSED_GREEDY_LOGITS=1` (the greedy region over the gathered logits); arm `cuts`: `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 VLLM_HPU_PLE_TABLE_RIDER=1` (the same region, the vocabulary-parallel arg max, the PLE rows on the embedding all-reduce); both `VLLM_GRAPH_RESERVED_MEM=0.5` | `compare_arms.py <region> <cuts>`: the 6 lone requests' texts must be equal token for token (both arms compute the same bf16 logits; the cuts are exact); the ms per step at 1 and 32 streams is the cuts' own effect (expected under 1 ms: one all-gather of `[rows, 62,080]` bf16 per rank and one all-reduce fewer) | pending |

A probe that sends `min_tokens` measures none of this: a request with `min_tokens` is kept out of the greedy region
(`_can_fuse_plain_greedy_sampling`, `hpu_model_runner.py`), so the logits are gathered and sampled as before.

## Handoff (if this agent stops)

State at 23:35 UTC on 2026-11-20 (fourth agent of this track, from 23:15 UTC, Opus 5.5). Branch
`feat/flash-next-gaudi-collectives`, worktree `/Volumes/Storage/Claude/gaudi-work/wt/collectives`; scratch, logs
and card results in `/Volumes/Storage/Claude/gaudi-work/reports/collectives/` (`card/<id>-<name>/`, `logs/`,
`probes/`). This section is the track's report: the harness refuses a REPORT.md.

Where things are:

- PRs #13 and #14 are merged (program head 938dada4): the cuts behind their flags (default 0), the card tools
  (`tools/flash_next/collectives/`), the top-k head (`vocab_parallel.top_k`, `first_maximum`). The branch head has
  only this note on top. **No pull request tonight** (BRIEF 23:45 UTC: card evidence first, pull requests after
  04:00 UTC; the integration track builds from pushed branches). The second worktree
  `reports/collectives/dev-wt` (branch `feat/flash-next-gaudi-collectives-dev`) is an ancestor of the branch head
  and has nothing of its own.
- Card results, all read and fetched (section 6): 00064, 00108, 00112 (ids), 00073 (collective cost, section 5).
  Sent to the coordinator (23:25 to 23:35 UTC): the id verdicts, why 00073 failed and what it measured, the cuts'
  card state, the bench comparison with native-replay's 00065 (native-replay agrees: a served TP8 step pays about
  150 to 170 us per collective), the pair 00124 / 00125 and the flag state below.

What is proven on the card (VERIFIED by the jobs named):
- `vocab_parallel_argmax` (mode 1's arg max): equal to the CPU at 1 rank (00064, 00108), through the 4-rank
  (00112) and the 8-rank (00073) all-gather, lazy and replay, tied and untied, rows 1 to 128.
- The top-k head (mode 2, `vocab_parallel_topk` / `top_k`) at k = 20 and 64: equal at 1 rank (00108) and
  through the 4-rank all-gather (00112); its second stage (`first_maximum`, the top k of the gathered candidates)
  at 1 rank (00108). Not run: through an 8-rank gather (the same gather as the arg max's, which passed at 8).
- The tiny model at TP4 with mode 1 + PLE rider and with mode 2 + replicated embedding: identical tokens, and
  identical to the greedy region without any cut (00093 at TP1); the 700-token prompt's token 4 is a near tie
  that the region moves (section 6). Not formally a pass of the smoke.
- Not on any card: `VLLM_HPU_MTP_REPLICATED_FC=1`; mode 1 or 2 under MTP (greedy verification from the arg max,
  `hpu_greedy_verify.py`; this branch lacks mtp-card-ready's state-write fixes, so an MTP run belongs on the
  integration tree).
- Not worth it (00073): `VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS` (keep 0) and its fused form.

Card jobs: the real-model pair **00124** `real4-ab3-region` / **00125** `real4-ab3-cuts` (commit 0a66722b, any:4,
25 min each, side by side on the two halves, in the places of 00082 / 00083; at 23:33 UTC both "start by 02:50
UTC at the latest"). This agent handed back at about 23:45 UTC, before the pair started: its two background
`bin/farm job` commands (logs `reports/collectives/logs/farm_real4_ab3_{region,cuts}.log`) copy each job's log and
results to `reports/collectives/card/<id>-<name>/` if they are still alive when the jobs end; otherwise
`bin/farm fetch collectives 00124 00124-real4-ab3-region` and `bin/farm fetch collectives 00125
00125-real4-ab3-cuts` (or the mirror `node-mirror/farm/collectives/results/`). Then, from the worktree root:
`python3 -I tools/flash_next/collectives/compare_arms.py reports-folder/00124-real4-ab3-region
reports-folder/00125-real4-ab3-cuts` (`reports-folder` = `/Volumes/Storage/Claude/gaudi-work/reports/collectives/card`).
Reading: the line `texts ... lone requests: N/6 identical` decides. 6/6: mode 1 + rider are token-exact on the real
model, and the steady ms per step of the two arms at 1 and 32 streams is their effect (expected under 1 ms per
step at TP4, inside the noise between the halves). Fewer than 6: look at the first differing token of that
request (the two arms compute the same bf16 logits, so a difference is a defect of the cuts, unless a server
error or a compile pause explains it; check both `server0.log`). Also read the `TEXT` lines (no garbage) and the
`SAMPLED` lines (the sampled round runs neither the region nor the arg max: the same in both arms but for the
rider's one all-reduce). Cancel the pair if it has not started by about 03:00 UTC.

Pending, in order:

1. When the pair ends: compare the arms (above), write the result into section 6, tell the coordinator (`main`)
   whether mode 1 + rider are proven (the integration build keeps every flag of this track off tonight; the
   verdict is for the next run and the pull request), push.
2. Pull requests after 04:00 UTC: this note (sections 5 and 6, the handoff); nothing else is unmerged.
3. Count work left, small (0.15 to 0.19 ms each at TP8, coordinator: not worth card time tonight): the PLE rows on
   layer 0's mixer all-reduce when the embedding is replicated; the draft's arg-max gather carrying the winner's
   embedding row. The per-layer floor of 2 is algebraic (section 3); the cost per collective is the host path
   inside the replay call (section 5), native-replay's domain.
