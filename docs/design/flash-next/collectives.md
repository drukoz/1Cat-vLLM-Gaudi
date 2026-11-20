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

## 6. Card jobs of this track, their readings and their results

Every job runs the committed tree in the farm's per-commit image (`CARD.md`); scripts in
`tools/flash_next/collectives/`.

| Job | What | Reading | Result |
|---|---|---|---|
| 00064 `idcheck-rule9`, 1 card | `id_check.py`: every id pattern of the token path against the CPU, one lazy graph and a replayed one, rows 1, 2, 5, 32, 128, untied and tied, at a shard's width and at a whole row | serving cases (`vp_argmax`, `fp32_ids_row`) must equal the CPU; the probes say which patterns card rule 9 breaks (`argmax -> int32` is the fused greedy region and vLLM's sampler, `argmax -> .int()` the MTP draft loop, `topk -> int32` the control) | pending |
| 00073 `c4-bench-r8-r4`, 8 cards, in the place of 00019 | `bench_job.sh`: the id check through the all-gather on 8 ranks; `bench_c4.py` on 8 ranks, on the module group 0-3, and on both groups at once | `decide_g41.py`: alpha and us per MB per rank count, regime and form; C4 pays where `gather_sum` (the flag) or `gather_first` (the collective alone, the fused-kernel form) is below `all_reduce` under replay; alpha8 against alpha4 for the layout | pending |
| 00074 `tiny-tp4-logits-rider`, 4 cards | the tiny model at TP4 with `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 VLLM_HPU_PLE_TABLE_RIDER=1`, fused hyper-connections | tokens equal the CPU reference (one request at a time: one row) | pending |
| 00075 `real4-ab-off`, 00076 `real4-ab-logits-rider`, 4 cards each | `ab_job.sh`: the real model at TP4 + EP4, 1 and 32 streams, 256 tokens, no `min_tokens`, every stream's text kept | `compare_arms.py <off> <logits-rider>`: the same texts (a wrong id parts at once), and the ms per token of both | pending |

A probe that sends `min_tokens` measures none of this: a request with `min_tokens` is kept out of the greedy region
(`_can_fuse_plain_greedy_sampling`, `hpu_model_runner.py`), so the logits are gathered and sampled as before.

## Handoff (if this agent stops)

State at 19:35 UTC 2026-11-20 (third agent of this track, started 19:00 UTC). Branch
`feat/flash-next-gaudi-collectives`, worktree `/Volumes/Storage/Claude/gaudi-work/wt/collectives`, scratch, logs and
card results in `/Volumes/Storage/Claude/gaudi-work/reports/collectives/` (`logs/`, `card/`). This section is the
track's report (the harness refuses a REPORT.md).

Where things are:

- `origin` has the branch up to `fc71ef83` (pull request #13's head). **Local commits after it are not pushed on
  purpose** until #13 is merged, so that the gate result covers the pull request's head: `3abbd8df` (id check),
  `684b54ad` (bench, A/B and reader tools, a unit test), and the commits of this note. After the merge:
  `git push origin feat/flash-next-gaudi-collectives`, then a second pull request for them.
- Pull request #13 (the five cuts, default off): the program head `b0cd0bb` is already merged into the branch
  (`6b265d50`); the cut suite gave 35 passed on the merged tree (`logs/cuts_m2.log`); the common gate
  (`bin/gate <worktree> --tools`) was started at 19:00 UTC on `fc71ef83`: its lines are in `/tmp/gate.rqDQHN/summary.txt`
  and at the end in `logs/gate_fc71ef83.log` (`rc=0` in the last line = green). Green: `gh pr merge 13 --merge`.
  A red line: run that file alone (`reports/collectives/pt2.sh <name> <file>`) and compare with the program head
  before deciding (two files were order-dependent in one process; the gate runs one process per file).
- Card rule 9 (BRIEF 19:15) on this track's paths, by reading: the vocabulary-parallel arg max has no int64 on the
  device (fp32 id table, max / where / min, an fp32 to int32 cast); mode 2 (`vocab_parallel_topk`, no serving
  caller) casts topk indices to fp32 in the graph: change it to a gather from an fp32 id table when job 00064 shows
  which form the card computes right (`topk_to_fp32` against `topk_gather_fp32`). The fused greedy region
  (`select_compute_logits_and_greedy`) and its gathered fallback belong to engine-step (its commit `121c88b0` and
  later): do not edit them here; when both branches meet, its region keeps its own path and calls
  `vocab_parallel.vocab_parallel_argmax` when `vocab_parallel.enabled(model)`.
- Reported to the coordinator at 19:12 and 19:20 UTC: the MTP draft loop (`hpu_flash_next_mtp.py:118, 141` on the
  program branch) is `argmax -> .int()` in one graph (wrong from the second row on if rule 9 holds for argmax; with
  `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1` the drafts are int32 from the start); the farm's probe sent `min_tokens`
  (fixed on the node at 19:20).

Card jobs (section 6 has what each decides): 00064 (bumped, runs after 00048 and the QSA pair, about 20:20 to
21:15 UTC), 00073 (all cards, 15 min, in the place of 00019, by 00:26 UTC at the latest), 00074, 00075, 00076
(any:4, at the end of the queue: they run only if the queue moves faster than its maximum holds; cancel them if
they have not started by about 03:00 UTC). Each `farm` command waits in a background shell of this agent and
copies the results to `reports/collectives/card/<id>-<name>/` when its job ends; if the agent is gone:
`bin/farm ledger | grep -w collectives`, then `bin/farm fetch collectives <id> <id>-<name>`. A job can be
changed while it waits by a new submit with `--replaces <id>` (00073 passes on the place of 00019).

Pending, in order:

1. Merge #13 when the gate is green; push; open the second pull request (tools and the id check: no flag, no
   plugin code) after its own gate.
2. 00064's lines to the coordinator, mtp-card-ready and engine-step the moment it ends
   (`grep '^idcheck:' reports/collectives/card/00064-idcheck-rule9.log`).
3. 00073: `python3 -I tools/flash_next/collectives/decide_g41.py reports/collectives/card/00073-c4-bench-r8-r4 --json ...`;
   alpha and us per MB for 8 and 4 ranks, lazy and replay, into `reports/collectives/collective_alpha.json` (its
   `pending` fields) and to the coordinator (`main`); the decisions into section 5.
4. C4 only where the replay regime of 00073 shows the gather form below the all-reduce (I34 row gate): then a
   real-model arm with `VLLM_HPU_ALL_REDUCE_AS_GATHER_ROWS=<rows>` (`ab_job.sh`), and the fused form (the sum
   inside `hc_combine_norm`) only if `gather_first` beats `gather_sum` by more than the sum op.
5. Count work left (each collective about 0.15 to 0.19 ms at TP8): the PLE rows on layer 0's mixer all-reduce when
   the embedding is replicated (98 to 97); the draft's arg-max gather carrying the winner's embedding row (the
   draft's embedding all-reduce without the 1.11 GB replicated table). The per-layer floor of 2 is algebraic
   (section 3).
6. When MTP runs on the card (mtp-card-ready's fixes merged): an MTP k = 3 A/B of
   `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 VLLM_HPU_REPLICATED_EMBEDDING=1 VLLM_HPU_MTP_REPLICATED_FC=1` (9 collectives
   and four logits gathers fewer per MTP step, the verification from tokens).
