# Flash-Next on Gaudi: host time per decode step (track `engine-step`)

What the host does in one decode step that is neither a graph op nor a collective, what of it grows with the
rows, and the fast paths that remove the largest parts: the sparse-QSA decode metadata (idea I01), the in-graph
greedy region with `min_tokens` (idea I05), and the design of a sampled step that sends one small tensor to the
host. Every fast path ships default off behind a flag.

Labels: VERIFIED (by the named test, line or card job), MEASURED (where), INFERRED.

This track's harness refuses report files outside the repository, so the handoff of the track lives here.

## Handoff (if this agent stops)

Last update: **2026-11-21 09:33 UTC** (fourth agent, Opus 5.5; the node was extended to about 00:00 UTC on
2026-11-22). Branch `feat/flash-next-gaudi-engine-step`, pushed; merged into the program branch up to `bcef8aa5` (PR
#20); since then, not yet in a PR, all default-off: `ff160409` (`VLLM_HPU_WORKER_GC_FREEZE`), `ca52fa33`
(`VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS`), `844a37f2` (`VLLM_HPU_VERIFY_LOGITS_IN_GRAPH`), `8efc6969`
(`VLLM_HPU_PERSISTENT_INPUTS` and the block-table width guard), the probe's `--spec-arms` (`6d5b8eef`), docs; the
program head `adbcff30` (PRs #18, #21, #22) merged in at `a0bd1753` (two conflicts in `hpu_model_runner.py`, both
sides kept), and `83a3fa35` (the logger wraps the `sys.modules` graphs module; its test failed after the host
n-gram test in one session), `e4c442ef` (`VLLM_HPU_VERIFY_LOGITS_IN_GRAPH` runtime-switchable, the entry
mtp-draft-graph's build-5 card branch has as `64656082`; when that branch merges, the `SWITCHABLE` tuple conflicts
on one line: take the union). Next PR: gate of `e4c442ef` in `reports/engine-step/merge-wt`
(`logs/gate_e4c442ef.log`); the worker tests by hand at `83a3fa35` (`logs/worker_tests_83a3fa35.log`) are as on the
program head (9 files need the card, `test_gdn_state_layout` has 1 card-only test, `test_draft_rank_stats` expects
3 `.topk(`, stale since collectives' `2d644ce9`), the rest green; open and merge the PR when the gate is green. A fix
branch, `fix/flash-next-gaudi-block-table-width-guard` (`33eb7cb1`, on `adbcff30`), waits for its gate
(section 15). A second branch of this track,
**`feat/flash-next-gaudi-engine-step-mtp-record`** (`2b90707f`, worktree `reports/engine-step/trees/mtp-record`):
mtp-card-ready's head `7c4d65e3` plus `VLLM_HPU_MTP_TIMING=2` (one record per speculative step, section 14), for
integration build 4 in place of mtp-card-ready's branch (the timer exists only there). Scratch files, probes and
card results: `/Volumes/Storage/Claude/gaudi-work/reports/engine-step/` (`card_job/` = submit logs, `card/` =
fetched results, `logs/` = CPU test and gate logs, `probes/` = Mac probes (`mtp_step_ops.py` and `mtp_ops/` for
section 14), `merge-wt/` = the gate's snapshot of the branch, `trees/b1`, `trees/b2` = builds 1 and 2 detached).

### State in one paragraph

Card-proven: **`VLLM_HPU_CANDIDATE_SAMPLING=1`** (00110: -4.9 ms per sampled step at 32 streams on TP4; 00144:
-1.2 ms, noisy) and **with `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`** (`candvp`, 00144: -3.55 ms, -3.1 % at 32 sampled
streams on TP4); **I01** token-correct (00130) and neutral at a short context (00110, 00144); **`logprobs=5` and
`prompt_logprobs` no longer kill the server** (00130, 00144); **`VLLM_HPU_SAMPLER_REQUEST_PARAMS=1`** (`299856cd`,
00167 on the tiny model: `logit_bias`, `min_tokens` and `min_p` take effect on the general sampler path).
CPU-proven, card job queued: **`VLLM_HPU_CANDIDATE_SAMPLING=2`** (sampled MTP verification in the candidate
region, 00164). Host time per decode step outside the forward is spread thin (section 13; the copy bench 00173:
packing the 11 copies saves 0.2 ms at 32 rows, not built; the coordinator leaves the rest to async-sched). The
candidate region no longer refuses async scheduling (`01d66e31`, async-sched's request). Build 2's 1-stream MTP
question answered on the CPU (section 14). Section 15 (the host's serial work of a 32-stream MTP step): CPU-proven,
default off: the verify's LM head and arg max inside the captured forward, the persistent per-step inputs, the
collector freeze, the unmatched-input logger. 00223 (1 card): the verify forward and the draft passes have no cut a
collective would not explain (only the per-layer hook, which follows the MoE all-reduce); the nine step-level cuts
between the graphs are listed there; 00226 (TP2) is to name the TP8 front cuts.

### If you take over, in this order

1. **Done: PR #20 merged** into `feat/flash-next-gaudi` at 04:46 UTC (merge commit `2e9f9196`): this branch up to
   `bcef8aa5` (program head with PR 16 and PR 19 merged in). Gate of `ba1441b7`: 69 files, 0 red (one file failed
   only in the parallel pass, `test_gdn_core_decode_batch.py`, green alone). Owner's rule (03:07 UTC): merge your
   own PR as soon as its gate is green and the change is card-proven or default-off, then tell the coordinator the
   number. Next PR: gate the branch (`bin/gate <snapshot>`, no `--tools`: the rental tool tests fail on this Mac for
   every track), merge the program head first.
2. Read 00207 and 00169 when they land (table below); section 9 and the coordinator.
3. Section 15 (the host's serial work, coordinator 08:10 UTC), in this order: (a) integration's build-5 round with
   the unmatched-input logger (`HPU_GRAPH_UNMATCHED` lines): fix every named input (persistent base tensor, sliced
   inside the forward; a slice that reaches a custom op gets an owned copy inside the graph, card rule 1); (b) I01
   in the serve setting once 00207 and 00169 agree; (c) last, the packed upload (`VLLM_HPU_PACKED_INPUTS`) with
   async-sched's per-input switch. Rank skew: native-replay's pin mode 3, seen per rank with `replay_first_ms`.
   (d) 00226's TP2 cut log: name the cuts of 00205's `embed` and `ple` segments (`python3
   reports/engine-step/probes/cutlog/cuts.py reports/engine-step/card/00226-*/habana_logs`, then `seq.py` on the
   verify and draft captures) and tell the coordinator. (e) The chain job (coordinator 09:19 UTC) is
   mtp-draft-graph's 00230 `real8-chain-b5b` (it replaced 00228), after 00222 (about 13:00 UTC), on its branch
   `feat/flash-next-gaudi-mtp-draft-graph-b5` (build 5 `eb21ac75` + the chain + async-sched's M1 + this track's
   `844a37f2` and `ca52fa33`); arms `on-md`, `chain-md`, `vlg`, `chain-vlg`, `all`, `chain-all` on one server (`vlg`
   against `on-md` is this track's flag alone). Its tiny gate `tiny-chain-vlg` (TP8 tiny, k = 2, draft graph + I01 +
   chain + VLG, VP 0, `REPLICATED_FC` and `REPLICATED_EMBEDDING` on, the unmatched-input log) carries
   `LOG_LEVEL_PT_IRGRAPH=1` and `LOG_LEVEL_PT_LAZY=1`: `window.py` on
   `reports/mtp-draft-graph/card/00230-real8-chain-b5b/tiny-chain-vlg/habana_logs` (one rank's main thread: 8 ranks
   write there) shows which of the nine step-level cuts of section 15 are gone; tell the coordinator. The
   unmatched-input lines of the three chain gates are in the job log, after each gate's `arm ... start` line.
4. Build 4's code with this track's card tools and flags is the local branch `engine-step-int4-card` (`4b180729`);
   the CPU census tree is `reports/engine-step/trees/b4`.

### Card jobs

| Job | What | State | To do when it ends |
|---|---|---|---|
| 00093, 00110, 00130, 00144 | (earlier jobs) | ended | done: section 9 |
| 00146 `real8-candvp-short` | candvp at TP8 | replaced by 00166 | - |
| 00158 `real4-mtp3-cand2-sampled` | | cancelled | - |
| 00167 `tiny-request-params` (1 card, `9a3ed1d1`) | `tiny_checks.py --arms params` | ended 01:46 UTC, exit 0 | done: section 9 (`PARAMS ok`) |
| 00173 `h2d-copies-1card` (1 card, `01d66e31`) | `h2d_copies_bench.py` at 32 and 1 rows | ended 02:02 UTC, exit 0 | done: section 13 |
| 00164 `real8-mtp2-cand2-sampled`, 00166 `real8-candvp-short2` | (TP8 jobs) | folded into 00206 at 07:03 UTC (00164 replaced, 00166 cancelled) | - |
| 00206 `real8-mtp2-candsw` | | replaced by 00207 at 07:58 UTC (the collector arms added) | - |
| **00207** `real8-mtp2-candsw-gc` (all, TP8 + EP8, 35m, `4b180729` = build 4 `30f4e37e` + the probe's `--spec-arms` + `VLLM_HPU_WORKER_GC_FREEZE`; local branch `engine-step-int4-card`) | one MTP k = 2 server, build 4's mtp2 env (00186's `versions.txt`) with `VLLM_HPU_MTP_TIMING=2`: `serve_long_probe.sh --arms both --no-long --spec-arms cand,cand2,cand2vp --gc-arms --logprobs-check` (`SERVE_SPEC_K=2`) | queued | as 00206 (greedy base / both; sampled cand / cand2 / cand2vp at 32 and 1 streams: step ms, tokens per chunk, acceptance per position; the `Candidate sampling` lines; seeded; logprobs under MTP), then the collector rounds (`gc32`, `gc1`: off / on, twice): `python3 reports/engine-step/probes/gc_steps.py reports/engine-step/card/00207-*` gives p50 / p99 per rank of the replay call and the step and the generation-2 counts; the `SCHED:` line (may the container take SCHED_FIFO) |
| 00223 `tiny-cutlog` (1 card, `4b18072924bb`, `LOG_LEVEL_PT_IRGRAPH=1`, `LOG_LEVEL_PT_LAZY=1`) | `farm_tiny.py --gdn hc --spec 3`, every `step marker due to` reason | ended 09:01 UTC, exit 0 | done: section 15 (the cut reasons; parsers `reports/engine-step/probes/cutlog/`) |
| **00226** `tiny-cutlog-tp2` (any:2, 10m, `--no-sync` = `4b18072924bb`; coordinator GO 09:19 UTC, runs beside 00225 and 00119 after build 5, about 13:00 UTC) | the same at `--tp 2` with 00205's flags through tiny_smoke's `--env` (`PLE_HOST_NGRAM_IDS=1`, `COLLECTIVE_MARK_STEP=0`, `DECODE_CUT_INTERVAL=1000`, `QSA_FUSED=1`, `QSA_INDEXER_SHARD=1`, `GDN_KERNEL_INTERLEAVE=1`, `GDN_GLUE_CACHE=1`, `HC_FUSED_COPY_MODE=2`, `CANDIDATE_SAMPLING=1`, `PT_HPU_LAZY_ACC_PAR_MODE=0`, `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`); submit log `card_job/submit_cutlog_tp2.log` | queued | `cuts.py` / `seq.py` on `card/00226-tiny-cutlog-tp2/habana_logs`: every cut inside the verify and draft captures with its op, which are collectives (with `COLLECTIVE_MARK_STEP=0`), and what cuts after the embedding and the PLE cast; tell the coordinator |
| 00218 `real8-build5` | | cancelled 08:57 UTC; build 5 is 00222 | - |
| 00222 `real8-build5b` (integration's, starts by 11:22 UTC) | this track's asks inside build 5 (as planned for 00218) | queued | `reports/integration/card/00222-real8-build5b/` (arm folders as integration names them; planned: `mtp2-async-fp8kv-hpulog-32-greedy-dg32-kvlp-long/server0.log` (`HPU_GRAPH_UNMATCHED`, draft graph off then on; probe_dg32.log, scaling_dg32.json) -> mtp-draft-graph and section 15; `mtp2-async-sw-cap-kvlp/scaling_sw.json` (I01 off / dg / md); `mtp2-kvlpfb/mtp_step_record_r*.jsonl` (per-rank `replay_first_ms` skew); if ss0 is on, `mtp2-async-ss0-greedy/` and `mtp2-async-irlog-lv1-greedy/habana_logs/` (integration sends the two greps)) |
| **00169** `real8-i01-long120k` (all, 50m) | I01 at 32 x 120K over a shared prefix with prefill's prefix flags, and an MTP k = 2 part (`probe.log`, `mtp2/probe.log`) | queued | `prepare_inputs` and step time base against I01 at a long context |

### Pending subtasks after those steps, in order

0. **For the next rental (the coordinator's list):**
   - **I01 at a long context.** 32 streams over one shared 120,832-token prefix, base against I01 (and I01 + I05),
     on TP4 or TP8: `farm job engine-step real-i01-long any:4 45m --env VLLM_HPU_PREFIX_ALIGN_SPLIT=1 --env
     VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 --env VLLM_GRAPH_RESERVED_MEM=0.5 -- bash
     tools/flash_next/engine_step/serve_long_probe.sh --arms both,i01 --no-sampled` (prefill's two flags make the
     prefix cache hit: without them every long round re-prefills 120K per stream). Expected from the Mac (32 x 120K,
     CPU-timed): `prepare_inputs` 5.83 -> 0.39 ms per step without speculation, 21.8 -> 1.09 ms with MTP k = 3. The
     first long round prefills 32 suffixes over the cached prefix (about 6 to 20 s each, serialized): plan the hold.
   - **`candvp` at TP8**: the sampled rounds with `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2` (each rank's top 64 of a
     31,040-column shard, one `[rows, 128]` fp32 all-gather) against `cand` and vLLM's sampler on an 8-card server:
     `farm job engine-step real8-candvp all 35m --env VLLM_GRAPH_RESERVED_MEM=0.4 -- bash
     tools/flash_next/engine_step/serve_long_probe.sh --arms both --no-long --candvp --logprobs-check` (also checks
     the logprobs and prompt-logprobs fixes on the real model, and the candidate path's log lines).
1. **Time to first token of a turn with a cached prefix** (coordinator: 6 to 9 s at TP4 for 300 to 500 new tokens
   on a 34.5K or 64K cached prefix): the runner's host part is 2 to 4 ms (section 10), so the time is in the model's
   prompt forward (device, or the lazy issue of the prompt graph): prefill's. A step timer or a device trace of
   one such chunk on the card would split it; not run.
2. **The replay call grows from 23 ms at 1 stream to 35 ms at 32** (step-anatomy's split): what in the runner makes
   the host wait inside it; split with native-replay. Not started.
3. **Next week: sampled MTP verification through the candidate region** (design from mtp-card-ready, 00:40 UTC).
   With MTP's arg-max drafts no separate rejection sampler is needed: draw every verification row and the bonus row
   from the target distribution (temperature, top-k, top-p), each row with its own noise, then run the greedy
   rejection on those tokens (`greedy_rejection_sample_from_tokens`, on the program branch since PR 13): accept while
   a row's draw equals the draft, output the first unequal draw, or the bonus row's draw when every draft is
   accepted. Exact: the draft distribution is one-hot at d, so speculative sampling accepts with probability p(d)
   and on rejection draws from p without d, renormalized; a draw y from p accepted when y = d gives P(accept) = p(d)
   and P(y = x | y != d) = p(x) / (1 - p(d)); the rows are independent. Work: `_candidate_sampling_inputs` accepts a
   speculative step and expands each request's parameters to its 1 + k rows in the HPU's padded layout (row r
   belongs to request r // (1 + k)), one noise row per row (a seeded request draws its 1 + k rows from its own
   generator in a fixed order), greedy rows keep the arg max; the region returns per-row tokens as
   `fused_sampled_token_ids`, which the runner's speculative branch already hands to the greedy rejection. Cost of a
   sampled verification step then about that of a greedy one (no `[128, 248320]` logits all-gather with
   `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`, no sort, no second pass). Estimate: one day for the code and CPU tests (a
   distribution test: first-draft acceptance equals p(d) within sampling error; tokens equal the greedy rejection's
   for greedy rows), then one card job: first-draft acceptance against p(d) as in mtp-card-ready's 00135, and the
   32-stream sampled MTP step time against `VLLM_HPU_SPEC_SAMPLED_VERIFY=2`. Until then sampled verification is
   mtp-card-ready's (`VLLM_HPU_SPEC_SAMPLED_VERIFY`, rows of `candidate_probs`); keep `candidate_probs`' signature
   stable.
4. The brief's later items, none started: device tensors kept in place, async scheduling with speculation (the PLE
   layer needs the sampled tokens on the device), the worker round trip. CPU pinning belongs to native-replay.
   MEASURED on the Mac (01:30 UTC, the engine harness forcing `async_scheduling=True` past the platform's refusal,
   tiny model, four reference prompts): every prompt's tokens leave the reference at token 2, with the PLE n-gram
   ids on the host and on the device alike: under async scheduling the step's n-gram context (built from the host's
   token table) lacks the previous step's token, which is still on the device. Async scheduling for Flash-Next
   needs that context on the device first (the last n - 1 tokens of each request kept and shifted in the graph);
   it would hide the host time outside the replay call (about 5 ms per step at 32 streams on TP8, step-anatomy),
   but not under speculation, which the HPU runner refuses with async scheduling.
5. **Done on the CPU at `299856cd`** (`VLLM_HPU_SAMPLER_REQUEST_PARAMS=1`, default 0): the general sampler path's
   ignored `min_tokens`, `logit_bias` and `min_p`. A card check is a 1-card tiny job (a `logit_bias` of +100 must
   make a greedy request write only that token; a stop token before `min_tokens` must not appear; `min_p` must not
   crash), not yet submitted: a new sync prunes the oldest bundle, and 00146 waits on `e9032e68` (replace 00146
   onto the new commit in the same go).
6. For mtp-card-ready (not this track's file): `hpu_flash_next_mtp.py` reads `logits.topk(DRAFT_RANK_TOP).indices`
   back as data for its draft-rank statistics (off by default); under the card rule those ids may be wrong.
   kernel-fixes' 00123 widened the rule: any `.to(int32)` of ids from `topk`, `sort` or `max(dim).indices` read on
   the host is garbage on the first run of each new graph; `argmax` forms are right. This track's id paths follow it
   (`v1/sample/topk_ids.py`; vLLM's sampler uses its sort indices only as a scatter index inside the graph).

### How to run the CPU tests

From `wt/engine-step`:
`PYTHONPATH="$PWD" /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python -m pytest -q -p no:cacheprovider tests/unit_tests/models/flash_next/test_topk_ids_card_rule.py tests/unit_tests/models/flash_next/test_runtime_switches.py tests/unit_tests/models/flash_next/test_candidate_sampling.py`
(fast, 18 passed at `0cb55ebb`); the serving-loop files (`test_engine_loop.py`, `test_engine_step_budget.py`,
`test_decode_metadata_sparse.py`, `test_decode_metadata_graphs.py`, `test_fused_greedy_min_tokens.py`,
`test_candidate_sampling_loop.py`) take 10 to 20 minutes each on the shared Mac. The probes check themselves
without a card: `python3 tools/flash_next/engine_step/hpu_sampling_ops.py --selftest --out /tmp/x`.

## 1. The serving-loop harness

`tests/unit_tests/models/flash_next/engine_harness.py` (commit `84aace3a`) runs vLLM's scheduler and EngineCore
step functions over the real `HPUModelRunner` and the tiny Flash-Next model on the CPU: prefix caching, the
recurrent state pool, sparse QSA and 128-token kernel blocks, as the rental servers run. Greedy tokens must equal
the Hugging Face reference alone, batched with requests finishing mid-batch, and after a prefix-cache hit
(`test_engine_loop.py`). The same harness counts what one decode step moves across the host/device boundary
(`test_engine_step_budget.py`: 14 copies to the device and one read per step, at 1 and at 32 rows) and, with a
stub model of the real geometry, times the host phases (`reports/engine-step/probes/`).

## 2. Jobs 00030 and 00031: why they ended (VERIFIED from the queue's job files and the fetched logs)

Both ran from 18:21:55 UTC on commit `5995d8b5`, the base on cards 0,1,2,7 and the flag-on server on 3,4,5,6.
Each server loaded checkpoint shards at about 1.6 s per shard for a while and then at 40 to 60 s per shard
(`card/00030-real4-base-longctx/server0.log`, lines 454-505). The queue's cancel files read `by engine-step at
2026-11-20T18:41:20Z` and `...18:42:56Z`; the ledger reason is `cancelled`. Exit code 143 is the SIGTERM of that
cancel, not the maximum hold, which was 19:01:55 UTC. No probe ran. The cause, found by card-farm (BRIEF.md,
18:45 UTC): two real-model servers loading at once shared every CPU, each worker with 152 / TP OpenMP threads;
since 18:42 UTC the farm pins each partial-card container to its cards' CPUs.

## 3. The sampling stage and the host path of a greedy step

1. **The min-tokens mask does not run on HPU** (VERIFIED: code and CPU test).
   `HpuInputBatch.refresh_sampling_metadata` (`vllm_gaudi/v1/worker/hpu_input_batch.py:554-562`) never calls the
   logits processors' `update_state`, so `MinTokensLogitsProcessor.min_toks` stays empty and `apply()` returns the
   logits untouched (vLLM `v1/sample/logits_processor/builtin.py:277-286`): no `index_put_`, no fallback, no host
   round trip. Test: `test_fused_greedy_min_tokens.py::test_general_sampler_path_tracks_no_min_tokens_request`.
   A `min_tokens = max_tokens` request still has the exact length, because the scheduler's `check_stop` honours
   `min_tokens` (vLLM `v1/core/sched/utils.py:122`); a stop token before `min_tokens` is emitted as a token. For
   the same reason `logit_bias` and `min_p` are ignored on the general sampler path. A defect, not a cost.
   Confirmed on the card by mtp-card-ready's reference arm (job 00021): 84 ms per step at 32 streams with and
   without `min_tokens`.
2. **The sampler stage of a greedy step** (VERIFIED by reading): the logits stay on the device; `_run_sampling`
   is `mark_step`, `_prepare_sampling` (host; nothing is copied to the device while the batch is unchanged), one
   lazy graph of an fp32 cast of `[rows, 248320]`, an arg max and an int32 cast (vLLM `v1/sample/sampler.py:97-140`
   and `:241-242`), `mark_step`. The only read is rows x 4 bytes of token ids. No op runs outside a graph.
3. **Host path of a step at a 64-token context** (MEASURED on the Mac, one thread, the real runner and scheduler
   in the harness, stub model; `probes/host_phases_anyctx.py`, output `probes/host_phases_short64_flag0.txt`),
   CPU ms per step:

   | rows | engine step, all | `_prepare_inputs` | `scheduler.schedule` | `update_from_output` | sampling |
   |---|---|---|---|---|---|
   | 1 | 0.36 | 0.19 | 0.03 | 0.01 | 0.03 |
   | 8 | 0.49 | 0.21 | 0.10 | 0.02 | 0.03 |
   | 16 | 0.74 | 0.29 | 0.20 | 0.04 | 0.04 |
   | 32 | 1.36 | 0.47 | 0.46 | 0.08 | 0.07 |

   About 0.03 ms per extra row. Not in the harness: the API server process (detokenize and SSE per chunk; it
   overlaps the engine) and the workers' IPC (1.5 KB out, 1 KB back at 32 rows).
4. The card agrees (MEASURED, step-anatomy job 00013, TP8, rank 0, ms per step at 1 / 8 / 32 streams): prepare
   1.67 / 1.77 / 2.07; the replay call (host) 22.8 / 27.1 / 35.2; logits and sample (host) 0.66 / 0.78 / 0.81;
   wait for the device 11.6 / 15.0 / 42.7; engine turn 0.56 / 0.88 / 1.89; step 37.1 / 46.4 / 83.6. So at a short
   context the host outside the replay call is 2.9 ms at 1 stream and 4.8 ms at 32: what I05 can remove is under
   1 ms (logits and sample), what async scheduling can hide is under 5 ms; I01's part is the growth of `prepare`
   with the context (section 4).

## 4. I01: sparse-QSA decode metadata (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`)

A sparse-QSA decode or verification step reads, of the attention metadata, only `block_table`, `slot_mapping`
and `block_size`; yet every step built the dense paged-attention buffers (per-row block lists, an identity block
resolve, `block_list` / `block_groups` / `block_usage`, their three copies to the device, and in the graph the
`block_mapping` one-hot and `attn_bias`), and its decode graph followed the block bucket. With the flag: none of
that, one block-table row per request built with numpy and repeated over its verification rows, the decode
bucket by batch size only, the recurrent state slots without the per-request loop, and the MTP drafter's
per-step metadata built the same way. A runner that cannot use it (another model, dense QSA, the defragmenter
on, a draft method other than MTP or n-gram) logs the reason once and keeps the buffers.

Files: `vllm_gaudi/v1/worker/flash_next_decode_metadata.py`, `hpu_model_runner.py`
(`flash_next_sparse_decode_metadata`, `_create_decode_input_data`, `prepare_mamba_state_idxs_vectorized`),
`v1/spec_decode/hpu_flash_next_mtp.py` (`sparse_attn_metadata`).

VERIFIED on the CPU (`test_decode_metadata_sparse.py`, `test_decode_metadata_graphs.py`): tokens and logits
bit-identical with the flag on and off, for decode and for k = 3 verification steps; one decode graph signature
per batch size; three fewer copies per step.

MEASURED on the Mac (real geometry, 32 rows x 120K tokens, stub model; `probes/rg2_*.txt`), CPU ms per step:

| | flag 0 | flag 1 |
|---|---|---|
| `prepare_inputs`, no speculation | 5.83 | 0.39 |
| `prepare_inputs`, k = 3 verification | 21.8 | 1.09 |
| drafter `prepare_attn_metadata`, per draft step | 5.2 | 0.07 |
| copied to the device per step, no speculation | 723 KB in 14 copies | 134 KB in 11 |
| the same, k = 3 | 2.89 MB | 0.53 MB |

Integer casts (card rule 9): I01 adds none. Its block table is the same host-made int32 table as before, and
the sparse QSA code that reads it is unchanged (`qsa.py:392` widens it with `.long()` in the graph, flag on or
off; the one-card probe checks that pattern as `table_int32_widened`).

## 5. I05: in-graph greedy with `min_tokens` (`VLLM_HPU_FUSED_GREEDY_LOGITS=1`)

The fused region computes the LM head and the arg max in one lazy graph, without the sampler's own graph, its
two `mark_step` calls and the per-step sampling metadata. A request with `min_tokens` used to keep the general
sampler path; now its stop ids get `-inf` before the arg max through a fixed-shape gather, where and scatter
(`vllm_gaudi/v1/sample/fused_greedy.py`, up to 8 stop ids per row).

Card rule 9 (`torch.topk` ids narrowed to int32 in the graph that made them came back wrong): the stop ids are
made on the host as int64 and used as they are. For arg max the narrowing in one graph is sound in vLLM's
sampler at 8 and 32 rows (MEASURED by text, mtp-card-ready job 00021), so the region returns int32 ids as the
program branch's `vocab_parallel.top_tokens` does; the one-card probe checks every pattern against the CPU.

After the merge of the program branch (`1aa017aa`): without a mask the region is `vocab_parallel.top_tokens`
(the gathered arg max, or the vocabulary-parallel one with `VLLM_HPU_VOCAB_PARALLEL_LOGITS` at 1 or 2, which
also turns the region on by itself and lets it serve the greedy verification of an MTP step); a step with a
row below its `min_tokens` gathers the logits, writes the mask and takes the arg max; a `min_tokens` request is
refused under a speculative config (the mask has one row per request, a verification step has 1 + k).

VERIFIED on the CPU (`test_fused_greedy_min_tokens.py`): the mask equals vLLM's min-tokens rule; every decode step
of a `min_tokens` request runs in the fused region; tokens equal the HF reference; a stop token the model would
pick before `min_tokens` is replaced by the arg max of the masked logits.

It helps greedy requests only: the probes, not the owner's agents (section 7).

## 6. An A/B on one server: the switch file (`VLLM_HPU_RUNTIME_SWITCH_FILE`)

A real-model server takes about ten minutes to start, and two servers on two halves of the node differ by more
than a host-side fast path saves. With `VLLM_HPU_RUNTIME_SWITCH_FILE=<path>` every worker looks at the file
before each step (one `stat`) and, when it changed, takes the flags it lists (`NAME=VALUE` lines). Only the two
flags the runner reads at every step can be switched (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA`,
`VLLM_HPU_FUSED_GREEDY_LOGITS`); another name is reported and the file ignored. The file is replaced with a
rename while no request runs, so every worker switches at the same step.

Files: `vllm_gaudi/v1/worker/runtime_switches.py`, `hpu_model_runner.py` (`_poll_runtime_switches` at the start
of `execute_model`), `tools/flash_next/engine_step/long_context_probe.py` (arms base / both / i01 / i05, rounds
in turn), `serve_long_probe.sh` (the job). VERIFIED on the CPU (`test_runtime_switches.py`): a server takes both
flags in the middle of its requests and drops them again, every decode step runs the path in force, the tokens
stay the HF reference's.

**Texts of two rounds.** On the real model two rounds of the same 32 prompts on one server already give
different texts (coordinator, job 00021: stream 0 leaves its one-stream text after 55 characters at 8 streams,
after 23 at 32). Why (INFERRED from the runner's flow, not measured): a request's first tokens are decoded in
batches of whatever size the other requests' prefills have reached (1, 2, 4, ... 32 rows as they join; the
prompt batch is one request per step), and in the row its arrival order gives it; each padded batch size is its
own compiled graph with its own kernel shapes, and the low bits of bf16 results differ between them, so a
near-tie in the logits flips one token and the texts part ways. Arrival order differs from round to round. The
probe therefore compares the arms' texts at ONE stream (`text1`, must be equal), scores N streams by a task
(`TASK`: each stream goes on with its own numbers) and counts, for N streams with one prompt, how many different
texts come back (`SAME`: more than one means a row's result depends on its place or its neighbours).
Bit-exactness itself is proven on the tiny model (CPU tests; on a card by the tiny run of the one-card job).

## 7. The sampled step (real traffic): what runs today and the design

A request without a temperature gets the checkpoint's default sampling: temperature 1.0, top_k 20, top_p 0.95.

**Today** (VERIFIED by reading; costs on the card unknown until the one-card probe). The decode graph ends with
the logits `[rows, 248320]` gathered on every rank. Then `_run_sampling`: `mark_step`; `_prepare_sampling`
builds the metadata with three device gathers by a Python list (`self.temperature[req_indices]`, `top_p`,
`top_k`: three small copies to the device per step); vLLM's `Sampler.forward` in one lazy graph: fp32 cast,
temperature (`div_`), `apply_top_k_top_p_pytorch` (no Triton on the card, `allow_cpu_sync=False`: **a full sort
of `[rows, 248320]` with indices**, the top-k mask by a gather and a compare, the top-p mask by a softmax and a
cumsum over the sorted row, **a scatter back to vocabulary order**), a softmax, **exponential noise over the
whole vocabulary** (`q.exponential_()`), a division and an arg max, the int32 cast; `mark_step`. About fourteen
passes over 32 MB at 32 rows plus the sort. One graph and two `mark_step` calls, as for greedy. To the host:
rows x 4 bytes of ids. A seeded request draws its row of noise from a host generator (`q[i].exponential_`).
Files: vLLM `v1/sample/sampler.py:244-300`, `v1/sample/ops/topk_topp_sampler.py:131-153, 349-403, 445-467`;
`hpu_input_batch.py:635-706`.

**Built: sampling over candidates** (`VLLM_HPU_CANDIDATE_SAMPLING=1`, default 0; commit `4e5900c3`; the
one-card probe times the same ops as `sampled_candidates*`). Files: `vllm_gaudi/v1/sample/candidate_sampling.py`
(`take_candidates`, `candidate_probs`, `sample_candidates`, `candidate_parameters`, `candidate_refusal`),
`hpu_model_runner.py` (`select_compute_logits_and_sample`, `_candidate_sampling_inputs`). VERIFIED on the CPU:
`test_candidate_sampling.py` (the kept tokens and their probabilities are those of vLLM's
`apply_top_k_top_p_pytorch` for eight rows of different temperature, top-k and top-p; 40,000 draws follow the
distribution within five sigma and never leave the kept set; a row's draw depends only on its own noise; a
greedy row of a mixed batch gets the arg max over the whole row; refusals) and
`test_candidate_sampling_loop.py` (the serving loop: every decode step of sampled requests runs in the region,
every drawn token has a probability above zero in vLLM's distribution of that step's logits, a greedy request
in the same batch still gives the HF reference's tokens, a seeded request repeats itself, a request outside the
conditions and a server without the flag keep vLLM's sampler). Two differences from vLLM, both without weight:
when the top-p cut falls inside a group of tokens with equal logits, which of the tied tokens stay follows
top-k's order instead of the sort's (same count, same probabilities); a tie group at the k-th value that
reaches past 64 candidates is cut at 64. The design:
1. Candidates: `values, ids = logits.topk(C)` per row, sorted, C = 64 (at least the largest `top_k` of the
   batch); the ids read out of an fp32 id table with the top-k indices (since `330f9413`). With the collectives track's vocabulary-parallel head each rank takes the top C of its shard and the
   ranks gather `[rows, ranks x C]` (value, id) pairs instead of the logits; a final `topk(C)` restores the order.
2. On `[rows, C]`: temperature; top-k (keep every value at or above the k-th); top-p as vLLM does it (drop the
   smallest tokens whose summed probability stays at or below `1 - top_p`, never the largest); softmax; the
   exponential race `argmax(probs / noise)`; the token id by a gather from `ids` (int64 as top-k made it: no
   narrowing, card rule 9).
3. The noise `[rows, C]` comes from the host with the step's inputs (8 KB at 32 rows, one copy; a seeded
   request's row from its own generator), or from the device if `exponential_` is native there.
4. It runs in the LM-head region like the greedy region: no sampler graph of its own. To the host: rows x 8
   bytes of ids, one small tensor. Under async scheduling the ids stay on the device.
5. Same distribution as vLLM's path whenever `top_k <= C` and no tie group at the k-th value reaches past
   column C (checked in the graph: `values[:, C-1] < kth`; such a batch, and requests with `top_k` unset or
   above C, `min_p`, penalties, `logit_bias`, logprobs or a custom logits processor, keep the general path).
   The random stream differs, so a seeded request does not reproduce its old output.

**Interfaces for the other tracks.**
- collectives (vocabulary-parallel top-k): the sampler stage takes `(values fp32 [rows, C], ids [rows, C])`,
  sorted by value, largest first, global token ids, C fixed per server. Agreed at 20:05 UTC: collectives'
  `vocab_parallel.top_k(model, hidden_states, C)` returns exactly that with int32 ids (its commit `ea3bc30a`:
  per-rank `topk(C)`, one all-gather of `[rows, 2C]` fp32 per rank, a final `topk(C)`; with the flag at 0 or 1
  the gathered logits and `topk(C)`), and `vocab_parallel.first_maximum(values, ids)` gives the greedy row's
  token from the same candidates (the lowest id among the largest values). This track's functions never cast
  an id, so either id dtype passes through. Collectives dropped its own sampler stage for this one.
  **Done at `330f9413`** (after PR 14 put `vocab_parallel.top_k` on the program branch): the region calls
  `vocab_parallel.top_k(self.model, hidden_states, CANDIDATES)` and `vocab_parallel.first_maximum` for the greedy
  rows of a mixed batch; a step with a row below its `min_tokens` masks the gathered logits and takes
  `topk_ids.topk_with_ids`. The drawn ids are int32 out of an fp32 id table: no top-k index is cast or read back
  as data (the card evidence is in `v1/sample/topk_ids.py`). `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2` turns the logits
  all-gather of a sampled step into one `[rows, 128]` fp32 all-gather per rank (job 00144 measures it).
- mtp-card-ready (sampled verification, idea I07): one function `candidate_probs(values, temperature, top_k,
  top_p) -> probs [rows, C]` (zero outside the kept set) serves both the pick (`argmax(probs / noise)`) and the
  rejection sampler (`p(draft token)` = the probability at the column whose id equals the draft id, zero when
  the draft is no candidate; the recovery token from the same row with that column zeroed), so target
  probabilities agree between a plain step and a verification step.

## 8. For other tracks

- qsa-fused: `qsa_sparse_ops.py:116-117` (`select_blocks`) narrows `torch.topk` ids in the graph that made them
  (the rule-9 pattern) on the served sparse path (in BRIEF.md since 19:15 UTC).

## 9. Card results

**00093 `hpu-ids-sampler-tiny-flags`** (one card, commit `383288c5`, 20:26 to 20:29 UTC, exit 1). MEASURED.
- Ids, `[rows, 248320]` on one card, rows 1 to 32: `argmax -> int32` in the same graph, read as int64, or narrowed
  in a second graph: right at every row count. `topk(4) -> int32` in the same graph: garbled on the first run of a
  new graph (every second id), right on its replays except the order of tied values. `topk(4).indices` read as
  int64 or narrowed in a second graph: right except tie order in the id pass; **in the timing pass, which had run
  `sort` and `topk(64).values` graphs first, they returned the top-k VALUES (40, 30, 4, 4) on every iteration.**
  In-graph consumers of top-k ids were right in both passes (the candidate draw gathers its id in the graph).
  Hence the rule since 00:10 UTC: never read top-k indices back as data (`v1/sample/topk_ids.py`). The min-tokens
  mask forms and the block-table forms: right. vLLM's sampled path and the candidate path draw ids inside the
  row's top 20 (one id outside at 32 rows once in each: a tie at the 20th value, which vLLM keeps, INFERRED).
- The sampler stage on one card, ms per step at 32 / 16 / 8 / 1 rows: vLLM's sampled path (sort, top-k, top-p,
  noise) 7.46 / 4.90 / 3.61 / 1.60, of which the sort alone 4.31 at 32 rows; sampling over 64 candidates 3.18 /
  2.16 / 1.65 / 1.18, of which `topk(64)` alone 2.52 at 32 rows; the greedy sampler 0.47 / 0.48 / 0.49 / 0.50.
  vLLM's `index_put_` min-tokens mask 0.37 to 0.44 (it never runs on HPU, section 3). A logits round trip to the
  host (what a CPU fallback would cost) 11.4 ms at 32 rows.
- Tiny model at TP1 with I01 + I05: 5 of 6 prompts right; the 700-token prompt took token 139797 at its 5th
  position (4 of 16 tokens match); the same server without the flags: 16 of 16. The same alternative token stream
  came from collectives' vocabulary-parallel arg max (TP4) and mtp-card-ready's TP2 verify arms; the tiny model's
  top-2 logit margins there are 0.0005 to 0.02 at values near 0.27 (one bf16 step is 0.002): a near tie that the
  in-graph greedy region resolves the other way (INFERRED; job 00130 runs I05 alone).

**00110 `real4-ab3-greedy-sampled-long`** (TP4 + EP4 on modules 0-3, commit `9bf1920f`, 22:11 to 22:29 UTC, exit 5).
One server, flags flipped by the switch file, rounds in turn. Healthy after 426 s. MEASURED, steady ms per step
(median gap between streamed chunks) at 32 streams, 64-token prompts, 128 tokens:

| kind | base | arm | arm against base |
|---|---|---|---|
| greedy, I01 + I05 (`both`) | 106.66, 112.85 | 113.41, 112.09 | +3.0 ms (noise) |
| greedy, I01 alone | | 107.98 | -1.8 ms (noise) |
| greedy, I05 alone | | 107.03 | -2.7 ms (noise) |
| sampled (no temperature: T 1.0, top_k 20, top_p 0.95), `cand` = `VLLM_HPU_CANDIDATE_SAMPLING=1` | 115.68, 113.59 | 109.78, 109.60 | **-4.9 ms (-4.3 %)** |

- Every stream of the text rounds goes on with its own numbers in every arm, greedy and sampled (TASK 32 of 32).
  The 1-stream text with I01 + I05 equals base's (48 tokens). A seeded request under `cand` repeats itself, and
  under vLLM's sampler too (the device generator works on the card). 32 streams with one prompt: 2 different
  texts in both arms (31 equal the 1-stream text). With `min_tokens` in the fused region: 108.5 ms against 112.1.
- Exit 5 was the probe's counting: stream 24 (a random-token prompt after which the model emits special tokens)
  came back with fewer than two visible text chunks in 4 rounds of base and the arms alike, and the 2-token seed
  request too; the probe called those errors (fixed in `0cb55ebb`: every streamed chunk counts).
- **The prefix cache served 0 tokens of the 120,832-token shared prefix** (the second request's TTFT 150 s
  against 133 s for the first; hit rate 0.0 % all job), so no long-context round ran. Prefill owns it.
- 1-stream steps varied from 27 to 44 ms within the job (host-bound; a neighbour job loaded its weights, load
  average up to 132).

**00130 `tiny-i01-i05-logprobs`** (one card, TP1, commit `0cb55ebb`, 00:23 to 00:25 UTC, exit 1). MEASURED.
- I01 alone: the tiny smoke PASSES (16 of 16 on the 7-, 33-, 700- and 2600-token prompts; the 1- and 100-token
  prompts are the reference's own near ties at token 0, the same as every baseline). I05 alone: the 700-token prompt
  takes 139797 at token 4, the stream of job 00093 with both flags: the near tie is the in-graph greedy region's.
- `logprobs=5` on all six reference prompts, both arms: HTTP 200, every top-logprob id valid, the generated token
  first among its position's top logprobs, the first position's top 5 equal to the CPU reference's (5 of 5, or 4
  of 5 where the reference's 4th and 5th are near ties). The fix of `330f9413` works on the card.
- `prompt_logprobs=2` ended the engine in both arms (`matrix 1 and matrix 2 shapes cannot be multiplied (1x6 and
  128x248320)` in `_get_prompt_logprobs_dict`; `/health` 503 after). Fixed at `e9032e68`
  (`_prompt_hidden_states_per_request`, test `test_prompt_logprobs.py`, which reproduced the crash on the CPU).

**00144 `real4-ab6-short-candvp`** (TP4 + EP4 on modules 4-7, commit `e9032e68`, 00:42 to 00:56 UTC, exit 0).
MEASURED, steady ms per step at 32 streams, rounds in turn on one server. Card 0's hardware reset (00:46:19 to
00:47:12, links of cards 3-6 to module 3 down and up) fell inside the server's start (healthy at about 00:49:30): no
round overlapped it.

| kind | base | arm |
|---|---|---|
| sampled, `cand` | 114.65, 113.83 | 113.50, 110.79, 114.76 (-1.2 ms on average, noisy) |
| sampled, `candvp` (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`) | (same base) | 110.79, 110.60 (**-3.55 ms, -3.1 %**; -2.3 ms against `cand`) |
| greedy, I01 + I05 | 108.31, 109.50 | 107.99, 108.80 (-0.5 ms) |
| greedy I01 + I05 with `min_tokens` | | 111.07 (+2.3 ms for the mask at 32 rows) |

Every stream on task in every arm, greedy and sampled; the 1-stream text with I01 + I05 equals base's; seeded
requests repeat themselves (cand and vLLM's sampler). The `Candidate sampling` lines: "drew in the logits region ...
from the gathered logits" (cand) and "... from each rank's vocabulary shard (VLLM_HPU_VOCAB_PARALLEL_LOGITS=2)"
(candvp). Logprobs check: `logprobs=5` and `prompt_logprobs=2` answered 200 with valid ids, `/health` fine after.

**00167 `tiny-request-params`** (one card, `9a3ed1d1`, 01:45 UTC, exit 0): the tiny model with
`VLLM_HPU_SAMPLER_REQUEST_PARAMS=1` on the general sampler path: `logit_bias` +100 gives 6 of 6 biased tokens,
`min_tokens` 6 with a stop token the model emits at token 3 gives 6 tokens without it, `min_p` 0.5 answers,
`/health` 200 after; the greedy smoke still matches the reference; logprobs and prompt logprobs fine.

## 10. Time to first token of a cached-prefix turn: the runner's host part (MEASURED on the Mac)

`reports/engine-step/probes/ttft_cached_prefix.py` (output `ttft_cached_prefix_out.txt`): the engine harness with
the stub model at the real geometry, TP4 (scheduler block 896, kernel block 128, align mode, recurrent pool, sparse
QSA), `max_num_batched_tokens` 8192, `long_prefill_token_threshold` 2048. Request A prefills a prefix, then B and C
send the same prefix plus their own new tokens.

| prefix | new tokens | cached for B and C | steps to B's first token | host ms of that step |
|---|---|---|---|---|
| 34,048 | 501 | 34,048 | 1 | 2.2 (C: 1.8) |
| 120,832 | 64 | 120,064 (134 x 896) | 1 | 3.5 (C: 3.6) |

So the runner and the scheduler spend 2 to 4 ms on the host before such a turn's first token; qsa-fused's 6.5 to
7.6 s per 501-token chunk over a 34K cached prefix (job 00063, warm) is in the model's prompt forward. The harness
hits the 120,832-token prefix because its engine core cuts chunks at multiples of 896; the card's engine cut them
at multiples of 2,048 and missed (prefill's root cause, `VLLM_HPU_PREFIX_ALIGN_SPLIT`).

## 11. Sampled speculative steps in the candidate region (`VLLM_HPU_CANDIDATE_SAMPLING=2`, `d5ad5e67`)

Design from mtp-card-ready, interface agreed on 2026-11-21. With deterministic drafts (MTP's arg max, n-gram) the
draft distribution is one-hot at the draft d, and speculative sampling equals: draw every verification row and
the bonus row from the target distribution, each with its own noise, then verify the drafts against those draws.
A draw y accepted when y = d is accepted with probability p(d); a rejected row's draw follows p(x) / (1 - p(d)) for
x != d, which is the residual distribution max(p - q, 0) renormalized when q is one-hot at d; the rows draw
independently.

- `_candidate_sampling_decision` serves a decode step of a speculative server when the flag is 2, the method is
  `mtp` or `ngram`, `VLLM_HPU_SPEC_SAMPLED_VERIFY` (mtp-card-ready's) is 1 or 2, and every request has k drafts:
  request i owns rows `i * (1 + k) .. i * (1 + k) + k` (the layout of `_prepare_spec_decode_inputs`), each with
  the request's temperature, top-k and top-p and its own noise row (a seeded request draws its rows from its own
  generator, in row order). A greedy request's rows take `first_maximum`, its arg max. Anything else (fewer
  drafts, `min_tokens` below its count, logprobs, ...) is refused and logged once; mtp-card-ready's flag then
  serves the step.
- The region returns the per-row tokens as `fused_sampled_token_ids`; the runner's speculative branch already
  passes them to `greedy_rejection_sample_from_tokens` (accept while a row's draw equals the draft, then the first
  unequal draw, or the bonus row's draw). That function now clamps the `-1` of padded draft positions before its
  `index_select` (the CPU refuses `-1`; the positions are masked out anyway).
- The non-seeded noise comes from one host generator per server (`candidate_sampling.NOISE_SEED`), seeded alike on
  every TP rank and used by nothing else, so every rank draws the same tokens (also for `=1`).
- VERIFIED on the CPU (`test_candidate_verification.py`): over 200,000 rows the acceptance rate equals p(d) and
  the rejected draws equal p without d, renormalized, within 5 sigma; the greedy rejection on draws, padded
  positions included; the serving loop with n-gram k = 3 and drafts on every step: every verification step with a
  sampled request runs in the region, the rejection sampler never runs, a greedy request in the batch gives the HF
  reference's tokens. Card check: job 00164 (TP8, MTP k = 2, integration build 2 + `d5ad5e67`).

## 12. `min_tokens`, `logit_bias` and `min_p` on the general sampler path (`VLLM_HPU_SAMPLER_REQUEST_PARAMS=1`, `299856cd`)

vLLM applies the three through logits processors whose per-row state the input batch feeds (`update_state`); the
HPU input batch never does, and the HPU samples prefills and decodes in separate calls whose rows are not the
batch's rows, so all three were ignored on HPU's general path (section 3). With the flag the runner builds them
per row from each row's own request at every sampling call (`_apply_sampler_request_params`,
`v1/sample/request_params.py`) and applies them before vLLM's sampler: the min-tokens mask of `fused_greedy`
(gather, where, scatter of a host-made int64 index), `logit_bias` by `scatter_add`, and vLLM's min_p cut at the
row's temperature (equal to vLLM's `MinPLogitsProcessor` on the temperature-scaled logits; greedy rows untouched).
One difference: a request with logprobs gets them after its own bias and masks (vLLM's raw logprobs come first).
VERIFIED on the CPU (`test_sampler_request_params.py`): the adjustments equal vLLM's rules; in the serving loop a
`logit_bias` of +100 makes a greedy request write only that token and a stop token before `min_tokens` is not
emitted, and without the flag both are ignored. Card check: job 00167 (tiny model, one card).

## 13. Host time per decode step outside the forward (MEASURED on the Mac, 2026-11-21 01:45 UTC)

`reports/engine-step/probes/step_phases.py` and `step_profile.py`: the engine harness with the stub model at the
real geometry (TP8: five hybrid KV groups, 4 Mamba and 1 full attention, block 512), 32 requests at a 1K context,
median wall ms per engine step over 60 steps, flags as noted.

| configuration | step | `schedule` | `update_from_output` | runner `sample_tokens` | of which `_prepare_inputs` |
|---|---|---|---|---|---|
| greedy, vLLM's sampler | 2.14 | 0.56 | 0.09 | 1.14 | |
| greedy, `VLLM_HPU_FUSED_GREEDY_LOGITS=1` | 0.98 | 0.40 | 0.06 | 0.50 | |
| greedy, I05 + I01 | 1.00 | 0.40 | 0.06 | 0.50 | 0.32 |
| sampled, `VLLM_HPU_CANDIDATE_SAMPLING=1` | 1.18 | 0.41 | 0.06 | 0.67 | |
| 1 request, I05 | 0.30 | 0.03 | 0.01 | 0.24 | |

- About 70 % of `schedule` is `KVCacheManager.allocate_slots` for each of the 32 requests across the five groups,
  also on the steps where nothing changes (no block boundary, no newly full block). A decode fast path would have
  to reproduce Mamba align mode's per-step partial-tail caching (`MambaManager.cache_blocks` re-marks the prompt
  tail in `cached_blocks_this_step`); not done.
- The card's `prepare` is about 2 ms at 32 rows (step-anatomy 00013) against 0.32 ms here. **Job 00173**
  (`h2d_copies_bench.py`, one card, MEASURED, host us per step, median of 200): the eleven small copies of a
  32-row step one by one cost 473 us to issue (about 43 us each) plus 790 us at the `mark_step` (with the bench's
  consumer ops); one packed int32 copy with device slices costs 409 + 638 us (with clones of the slices 650 + 573);
  at 1 row 345 + 571 against 362 + 587. So packing saves about 0.2 ms per step at 32 rows and nothing at 1 row: the
  host-side packing and the slice ops eat most of what one copy saves. Not built (about 0.3 % of a step).

## 14. Build 2's 1-stream MTP step (coordinator's question, 2026-11-21 02:30 UTC)

00156 (build 2, MTP k = 2, `VLLM_HPU_SPEC_SAMPLED_VERIFY=2`) ran 1 stream at 84 ms per step with first-position
acceptance 0.60, against 00128 (build 1, k = 3) at 66 ms and 0.83. Integration's 00161 (build 2, k = 3, `=2` unset)
ran like build 1. CPU checks with `reports/engine-step/probes/mtp_step_ops.py` (the CPU engine: vLLM's engine, the
HPU runner, the MTP proposer, the tiny checkpoint; one request, 240-token prompt, 600 greedy tokens, oracle drafts
accepting 1,0,2,1,1, the prefix flags on; runs in `probes/mtp_ops/`):

- **`=2` changes nothing for greedy rows (MEASURED):** with `VLLM_HPU_SPEC_SAMPLED_VERIFY` 0, 1 and 2 all 300 steps
  take `_prepare_sampling`, vLLM's rejection sampler and its greedy `rejection_sample`; `_candidate_verify` is
  never called; `all_greedy` is True on every step; 433 host ops per step outside the two forwards, equal op by op;
  the same 600 tokens, equal to the stream without speculation. A sampled request on `=2` calls
  `_candidate_verify` every step.
- **The align-mode copies run once per recurrent block (MEASURED):** a boundary commit after sampling and a pre-step
  copy when the running block moves, at contexts 252/253, 508 and 763 on the CPU's 256-token block (once per 512
  tokens at TP8). The headline's short rounds (23-token prompts, 128 tokens) never leave the first block: no copy
  ran in 00156's 1-, 8- or 32-stream rounds. `5c770d96` adds 24 ops per copy event (66 against 42).
- Host ops per step outside the forwards: k = 2 433, k = 3 493 (one more draft pass).
- **The 1-stream acceptance compares two texts (MEASURED on the card's records):** stream 0's greedy text changes
  with the batch and with k (00128 nospec N=1 against N=8; 00156 mtp2 N=8 rep 0 against rep 1; k = 3 against k = 2
  from about token 13). At 8 and 32 streams the first-position rate is equal: k = 3 0.77 / 0.74, k = 2 0.76 / 0.76.
- **Where the time went (MEASURED, the card's MTP timer files):** the verify span (step start to the drafter call),
  54.4 ms at k = 3 against 67-71 ms in k = 2's greedy rounds; the draft is 14.5 against 13.2-14.2 ms. On the same
  k = 2 server the sampled 1-stream rounds verified at 51.5 ms (rep 0) and 64-66 ms (rep 1): not a fixed cost of k
  or of a path. INFERRED from four rounds: the span tracks the share of steps whose first draft was rejected (0.25:
  51.5 ms, 0.37: 64-66, 0.40: 67-71; k = 3 0.17: 54). The timer keeps 25-step means; a per-step record keyed by
  the previous step's accepted count would show it directly.

## 15. The host's serial work in a 32-stream MTP step (coordinator, 08:10 UTC; step-anatomy's 00205 trace)

00205 (integration build 4b, TP8, dense MoE, sync MTP k = 3, 32 streams, I01 off, no worker CPU pin): 84.9 ms per
step, prepare 6.2 ms, 154 launches, 178 host-to-device DMA rows, 13 copy-only graphs; the device idles about 30 ms per
step while the host works.

- **Rank skew (MEASURED, step-anatomy's per-rank extracts `reports/step-anatomy/trace_work/ext_00205d_r*.json`):** the
  8 ranks' step markers are within 80 us every step (the broadcast is not the cause), but ranks 0 and 1 reach their
  first graph launch 18.5 ms after the step start against 10.8-11.6 ms on ranks 2-7, every step, and then launch
  20-30 % slower (rank 1 is 22 ms behind rank 7 at the verify forward's 119th launch): two systematically slow hosts.
  Fix to measure: native-replay's `VLLM_HPU_WORKER_CPU_PIN=3` + `VLLM_HPU_WORKER_CPU_SIBLINGS` (the model thread on
  its own vCPU, its hyperthread sibling free). The per-step record has `t` + `replay_first_ms` per rank (`07932956`
  on the `mtp-record` branch, for build 5) to see the skew in every job.
- **H2D census (MEASURED on the CPU engine, `reports/engine-step/probes/h2d_census.py`, build 4's code):** our Python
  issues 31 host-to-device copies per steady MTP k = 3 step at 4 and at 32 streams (13 in `_create_decode_input_data`,
  4 in `_prepare_spec_decode_inputs`, 2 PLE, 10 from the drafter's `prepare_attn_metadata` on 2 passes, 2 blocking
  in `hpu_eagle.prepare_inputs`), 29 non-blocking, none from a pinned buffer, each with a fresh device allocation (25
  per step); 4 scalar operands in lazy ops between the graphs. With I01: 22 copies, 16 allocations. So the trace's
  178 rows are about 5 DMA commands per copy call (INFERRED; native-replay's H2D timer build will count both).
- **Copy-only graphs:** between the captured graphs our Python issues 8 recipes per step, all with compute (the
  speculative input gather, the LM head, the rejection sampler, 2 in `parse_output`, 1 per draft pass). The 13
  copy-only graphs are the replay wrapper's copies of unmatched inputs (native-replay, bridge source: an input that is
  not the captured graph's own input value is copied by `graphs.copy_to`, flushed as its own recipe before the
  replay). To name them: `VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS=1` (`ca52fa33`), and the same logger as a Python-only
  sitecustomize for build 5 (`reports/engine-step/probes/hpu_graph_input_log/`, sent to integration at 08:19 UTC).
- **The 13, predicted on CPU (MEASURED, the census's forward-input check, 08:22 UTC):** every input of the target and
  draft forwards that is a view made outside the forward or a lazy result still pending at the replay (both unmatched
  on a card). Exactly 13 per step, all on the draft forwards: pass 0 (16 rows) `input_ids`, `positions`,
  `hidden_states` (arg max, +1, index_select just before the pass) and the lattice's `load/store_indices_tensor`
  (`ring_slots_from_lattice`, pending); passes 1 and 2 the same three plus the two lattice tensors as non-contiguous
  views. The target forward's inputs are all matched. Fix: persistent base tensors into the draft forward and that
  arithmetic inside the captured draft graph, or mtp-draft-graph's one-graph chain (asked the coordinator).
- **The verify forward's wrapper graphs (coordinator, 08:22 UTC; mtp-draft-graph's follow-up c).** Step 2 of the
  rank-0 trace in order (`reports/engine-step/probes/trace/seq.py`, `seq_00205d_s2.txt`): inside the replay an
  "embed" segment and a cast-only "ple" segment that no collective ends, the final hyper-connection mix last; after
  it the LM head, a 500-op memcpy (the logits all-gather's concatenation) and the rejection sampler (324 ops).
  - Tail, done on CPU: **`VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1`** (`844a37f2`, default 0): for a greedy verification
    step the forward takes the logits indices and returns the selected rows and int32 per-row greedy ids from an
    fp32 id table (`vocab_parallel.greedy_ids`, card rule 9); the final mix, LM head and arg max are one segment,
    the runner verifies from the ids; mtp-draft-graph's chain (Stage B) takes the greedy rejection into its graph
    and reads the ids. CPU: same tokens off and on (n-gram k = 3, oracle drafts), the rejection sampler never runs.
  - Front: every lazy flush during capture ends a segment (native-replay, bridge source). Stage submission is not
    the cutter (qsa-fused's 00209: `PT_HPU_ENABLE_STAGE_SUBMISSION=0` leaves the MoE layer's launches as they are).
- **The cut reasons of the verify forward and the draft passes (MEASURED, 00223 `tiny-cutlog`, 1 card, the tiny
  model (3 GDN + 1 QSA layers, MoE, HC, PLE) with MTP k = 3 on the old draft loop, image `4b18072924bb`,
  `LOG_LEVEL_PT_IRGRAPH=1`):** parsers `reports/engine-step/probes/cutlog/` (`cuts.py`: the cuts inside each
  capture with the ops around them; `seq.py`: a capture's ops with its cuts; `window.py`: every cut between the
  graphs in a time window). `host step marker` is logged by PT_IRGRAPH and PT_LAZY both: count one. The verify
  forward (797 lazy ops): 4 cuts, all `host step marker`, one after each decoder layer, after `mixture_of_experts`
  and `add` (shared + routed): `modify_model_layers`' hook (`VLLM_CONFIG_HIDDEN_LAYERS=1`; the tiny run had
  `VLLM_HPU_DECODE_CUT_INTERVAL=0`). Nothing else: no cut between the embedding, the PLE and layer 0, no host read,
  data-dependent op, shape change or CPU fallback. The draft passes (6 captures, 223-254 ops): no cut. At TP > 1
  the hook follows the MoE runner's all-reduce (the layer returns `mlp_out`; the HC combine waits for the next
  layer), so it adds no launch; 00205 had `DECODE_CUT_INTERVAL=1000` anyway. So 00205's boundaries with no
  collective within 150 us (verify: `embed`, row 23, and the `ple` cast, row 31; drafts: embed | 25-op | 65-100-op
  segments, and `qsa_attention` | `moe` in passes 1 and 2) come from TP-only code or from 00205 flags the tiny run
  lacked (`PLE_HOST_NGRAM_IDS=1`, `QSA_FUSED=1`, `QSA_INDEXER_SHARD=1`, `HC_FUSED_COPY_MODE=2`, ...). `seq.py`'s
  `coll` needs the collective to start within 150 us of the segment's end, so an all-reduce a slow host issues
  late reads as none: row 23 to 25 is most likely the vocabulary-parallel embedding's all-reduce (INFERRED). 00226
  (TP2, 00205's flags) names them.
- **The step-level cuts between the graphs (MEASURED, 00223, 161 steady steps; per step; none is a collective):**
  (1) `host step marker` launches 3 pending PLE input ops (cast, add, index) before the verify replay; (2)
  `HPUGraph::replayV3`, the verify; (3) 3 x `host step marker` after it: the first launches `index` + `linear` (the
  logits rows and the LM head outside the graph), two are empty; (4) `d2h_maybe_eval` after the 34-op sampler /
  rejection (`argmax`, `any`, `where`): the sampled ids to the host; (5) `replayV3`, draft pass 0: the pending
  draft-input ops (`clamp index_select*3 cat`) and the copy-only D2D inputs launch first; (6)
  `EvaluateTensorData-PT_USE_MARKSTEP` after pass 0's `index_select linear argmax index_select*2`:
  `_draft_positions_cpu` reads `target_positions` (`hpu_flash_next_mtp.py:260`; gone with
  `VLLM_HPU_MTP_DRAFT_LOOP=1`'s host positions); (7), (8) `replayV3`, passes 1 and 2: the positions + 1 (7 ops) and
  the LM head + arg max + next inputs (9 ops) launch first; (9) `d2h_maybe_eval` after `linear argmax cat`: the
  drafts to the host. Prefill only: `out shape changed` twice per GDN layer (the chunked path: after the `cumsum`,
  before the `eye` / `tril` solve, and after the solve's `matmul`). Startup only: `bindings-synchronize_device`
  (626), `local scalar` (2). mtp-draft-graph's chain and `VLLM_HPU_VERIFY_LOGITS_IN_GRAPH` take out (3)'s flush,
  the pending-op launches of (5) to (8) and the read (6) (coordinator, 09:19 UTC): check with `window.py` when the
  chain job runs.
- **Stage 1 of the upload, done on CPU: `VLLM_HPU_PERSISTENT_INPUTS=1`** (`8efc6969`, default 0; coordinator GO
  08:38 UTC): every per-step input of the decode builder, the speculative indices and the PLE inputs through
  `HPUModelRunner._upload`, a persistent pinned host tensor and a persistent device tensor per (name, shape, dtype),
  copied non-blocking (two pairs alternating per input under async scheduling). Same tokens off and on (greedy, n-gram
  k = 3 with oracle drafts); one device address per input shape, which lean5's static launch plan needs (asked
  collectives for the plan_hit per rank, now from 00234). async-sched's device-made inputs replace five of them in chained
  verification steps (its own tensors; agreed 08:46 UTC).
- **The lean5 A/B of the persistent inputs:** collectives runs `VLLM_HPU_PERSISTENT_INPUTS=1` against `=0` on one
  image with lean5 on (its ab_bridge template), build 6 preferred, this track's image only if the coordinator or
  collectives says so (do not sync it for this unasked). Best on native-replay's `lean6m-43fdd6850644` with
  `PT_HPU_HPUGRAPH_STATIC_PLAN_MISS_LOG=200` (needs `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1
  PT_HPU_HPUGRAPH_PREPARED_REUSE=1 PT_HPU_HPUGRAPH_STATIC_PLAN=1`; it also has lean6h2's H2D timer):
  `HPUGRAPH_PLAN_MISS` lines per launcher, `input_moved` and `first_input <idx> <dtype[shape]>` name the inputs that
  still move. The baseline (without this flag) is collectives' 00234 (10:00 UTC: 00210's lean6m qualification moved
  by the coordinator onto build 5's image `eb21ac75` with the dense MoE from 1 token, 100 launches per 1-stream step;
  starts by 16:25 UTC): collectives sends the miss table when its `real-lean5-verify` arm ends.
- **A regression of this track on the program branch, being fixed:** the PR 16 merge resolution (`fe4cac24`, in
  PR 20) computed the block table's width before the sparse / dense branch; stub runners without
  `_block_table_max_blocks` fail (`tests/unit_tests/worker/test_flash_next_host_ngram.py`, 4 tests; served models
  unaffected). Fix branch `fix/flash-next-gaudi-block-table-width-guard` (`33eb7cb1`), gate
  `reports/engine-step/logs/gate_fixbt.log` (wrapper pid 87075, worktree `reports/engine-step/trees/fixbt`); open
  and merge its PR when green. Also fixed on this branch in `8efc6969`. Note: `bin/gate` does not run
  `tests/unit_tests/worker/`; run the worker tests a change touches by hand.
- **Plan (coordinator's order, 08:18 UTC):** (1) name and remove the copy-only graphs (persistent base tensors,
  sliced inside the forward), about 13 launches, 2-4 ms per step (INFERRED); (2) I01 in the serve setting once 00207
  (MTP, 32 streams) and 00169 (32 x 120K) agree (integration got the runtime-switch recipe); (3) last: the packed
  upload (`VLLM_HPU_PACKED_INPUTS`, 22-31 calls to 1-3 from pinned buffers, about 0.5-1.5 ms per step), with
  async-sched's per-input host-made / device-made switch agreed (08:29 UTC).
