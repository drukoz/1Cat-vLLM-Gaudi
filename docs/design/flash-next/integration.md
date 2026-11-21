# Flash-Next integration card build: one build of every card-proven branch, the TP8 headline numbers

This file is the integration track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `integration/card-build-20261120b` (worktree
`/Volumes/Storage/Claude/gaudi-work/wt/integration`), pushed to origin under its own name. **A card build only:
never merged into `feat/flash-next-gaudi`** (each track merges its own pull requests). Raw card results:
`/Volumes/Storage/Claude/gaudi-work/reports/integration/card/`; CPU runs: `reports/integration/cpu/`.
Labels: MEASURED (card, with its file), VERIFIED (CPU test or code line), INFERRED.

## Handoff (if this agent stops)

**Build 5 (fifth agent of the track, Opus 5.5, started 08:04 UTC on 2026-11-21; updated 12:10 UTC: job 00222 measured, results below).**

- **Tree.** Base = build 4 (`8cdc5999`) with the program head `adbcff30` (PRs #18 to #22) merged in, so the tree is the
  program head plus what build 4 carried that the head lacks (qsa-fused's branch, mtp-card-ready + the per-step record,
  prefill `92e65144`, async-sched `fb5cefca`, this kit). Then, in order: `f6aba529` merge mtp-draft-graph-b4 `100f4bf0`
  (`VLLM_HPU_MTP_DRAFT_GRAPH` + runtime switch + the kit's switch arms); kernel-fixes `11bc2f6d`, `a5786c55`, `b92402ef`
  (dense MoE that survives `model.to("hpu")`, chunk size 0); native-replay `773a7f7c` (`VLLM_HPU_WORKER_CPU_PIN=3`, default
  off); qsa-fused `f3b63114` (`VLLM_HPU_QSA_KV_FP8`, default off); `125429d3` merge engine-step-mtp-record `07932956`;
  qsa-fused `a09aad30` (kv_logprob_probe.py), `03156364` (`VLLM_HPU_QSA_SLOT_SHIFT`, default 0, OFF in every arm),
  `703b3cf7` + `9b4faf7c` (qsa_ab_probe.py long rounds); async-sched `d3fb1b11` (mode 1 only: no device wait in a steady
  step; mode 2 returns before that code, VERIFIED by reading `_flash_next_async_begin_step`); then the kit commit.
  Merge log below. Image of the code commit `125429d3`: `farm-integration:125429d31188` (built 08:18 UTC, 278 s).
- **CPU.** dense + FP8 K/V + pin tests 63 passed; slot shift + FP8 K/V 24 passed; async worker test 39 passed; 54
  non-gate files on a snapshot of `125429d3` (`reports/integration/cpu/out-b10x/summary.txt`): 51 green, 3 card-only
  skips. **`bin/gate --tools` on a snapshot of `716ff094` (the job's code; later commits touch only the kit, a probe
  tool and docs): 119 files, 0 red** (09:02 UTC; `reports/integration/b5/gate-716ff094.txt`).
- **Kit (this commit's tools/flash_next/integration).** An MTP arm is `mtp<K>` plus words: server `async graph rfc noshard
  spin inline fp8kv pin3 hpulog ss0 irlog`, probe `32 lv<N>_<N> greedy switch`, and steps on the same server after the
  probe `sw dg32 cap kvlp kvlpfb long` (serve_probe.sh `--post-cmd`). `HEADLINE_TRACE_ENV=1` starts every MTP server
  with step-anatomy's capture setup; `HEADLINE_REQUIRE_DENSE=1` stops the job with exit 8 when the server log lacks
  "keep their stacked experts" at load or "runs the dense MoE form" after the first warm round (a silent fallback voided
  00177 and 00190). Late switches read at the job's start: `~/farm/integration/results/switches/pin` ("place" =
  every real-model server under collectives' place_wrap.sh + native-replay's place.sh once healthy, rule a^1, NO pin
  flag (00189: the pin flag doubled the load to 14.9 min); "pin3" = `VLLM_HPU_WORKER_CPU_PIN=3` everywhere; no file =
  nothing; the scripts are `~/farm/integration/results/late/place/` when both exist there (uploaded 08:49 UTC:
  place_wrap.sh 363bcf9d = collectives' b9aac4db, place.sh 7e54f2de), else the kit's `place/` copies; the job log
  prints the md5s it used) and `.../switches/ss0` ("on" = the stage-submission arms run and the inline arm does not;
  the coordinator keeps it off: 00209 showed stage submission does not cause the MoE cuts).
  Dry runs against a fake server on the Mac: `scratchpad/integration/dry-*` (all arms, post steps, the dense stop,
  the switches).
- **The job: 00222 `real8-build5b`** (queued 08:58 UTC with `--replaces 00218`, keeping its place and bump: after
  async-sched's 00221; commit `eb21ac75` = the kit `716ff094` + kv_logprob_probe's PASSNOISE (the teacher-logprob
  |delta| of its two passes on one server = the noise floor of the FP8 comparison) + the `place` switch + collectives'
  fixed place_wrap.sh; image `farm-integration:eb21ac75a7c8`), all cards, hold 110 min. 00218 (commit `385cf2b7`) was
  the first submission, cancelled by the replacement (the coordinator's 08:49 change: placement without the pin flag).
  Set before the start (VERIFIED on the node at 09:46 UTC): `switches/pin` = `place` (00219: place-only gives a
  7.7 min load, no slow rank, 1-stream greedy 24 ms per step); no `ss0` file (00209: stage submission is not the
  cause of the MoE cuts); `late/place/` = place_wrap.sh 363bcf9d + place.sh 7e54f2de; `late/run5.sh` (md5 d62ff800)
  = `export VLLM_HPU_MOE_DENSE_MIN_TOKENS=1` + the same arms (coordinator, 09:50: kernel-fixes' 00213 measured dense
  from 1 token at 1 stream 74.0 -> 43.5 ms per sync MTP k = 3 step, launches 256 -> 124, 8 streams plain 49.1 -> 32.4,
  logprob mean |delta| 0.0072, top-1 99.81 %). So every 1-stream and 8-stream row of build 5 against 00186 carries
  dense-at-1 and the placement on top of the build changes. Submitted by
  `NAME=real8-build5b reports/integration/late/submit_b5.sh --replaces 00218` (log `reports/integration/card/submit-real8-build5b.log`; it copies the results
  when the job ends while it still runs). Read: `python3 -I reports/integration/b5/b5_compare.py
  reports/integration/card/00222-real8-build5b` (the table against 00186 and 00205), the TEXT lines, the REQUIRE-LOG
  lines, PASSNOISE next to COMPARE in the FP8 arm's kvlp_fp8.log, and the job's end line (kernel lines, node load).
  Arms: `tiny mtp2-async-sw-cap-kvlp mtp2-async-spin mtp3-async mtp2-kvlpfb ifnot-ss0:mtp2-async-inline-greedy
  if-ss0:mtp2-async-ss0-greedy if-ss0:mtp2-async-irlog-lv1-greedy mtp2-async-fp8kv-hpulog-32-greedy-dg32-kvlp-long`.
  Late-bound: `reports/integration/late/make_run5.sh` (run5.sh; `--pin on|off`). Results:
  `reports/integration/card/<id>-real8-build5/` (or `bin/farm fetch integration <id> <id>-real8-build5`).

Earlier (fourth agent, updated 2026-11-21 06:30 UTC; started 23:12 UTC):

**Now (02:25 UTC), for whoever takes the track next.** Builds 1 and 2 are measured (tables below: 00128, 00156).
Three jobs of this track wait or run on the node; the `farm job` commands that submitted them ran in the fourth
agent's background shells and may be gone, so fetch each result with `bin/farm fetch integration <id> <id>-<name>`
when `bin/farm ledger | grep -w integration` shows it ended:
- **00161** `moe-chunk-ab` (running since about 02:00 UTC; kernel-fixes' A/B, image `b2a7f62f` = build 2): kernel-fixes'
  `moe_chunk_ab.sh` (f8fe517c) runs the kit's `mtp3` arm twice, base and `VLLM_MOE_CHUNK=64,8192
  VLLM_MOE_TOKEN_BOUNDARY=512,1000000`, 32 and 1 streams + a 1 x 16K round. Its base arm already shows MTP k = 3 on
  build 2 (prefix flags on, `SPEC_SAMPLED_VERIFY` unset) at **24.9 / 24.2 ms per token at 1 stream and 43.2 / 40.8 at
  32 streams**: build 2's code and the prefix flags do not cause the 1-stream slowdown of 00156's arm C (k = 2 with
  `SPEC_SAMPLED_VERIFY=2`: 41.7 ms per token). Kernel-fixes reads the A/B.
- **00177** `moe-ab` (waiting; image `d00c0a9a` = build 3): kernel-fixes' `moe_ab.sh` (6b960231, md5 1c34d795) with the
  arms `dense-mtp3 base-mtp3 dense-nospec` (`VLLM_HPU_MOE_DENSE=1`, kernel-fixes' 86104c9d + b773ad72 in the build).
- **00178** `real8-build3` (waiting, placed after the async A/B 00171, about 05:00 UTC; 75 min): build 3's headline.
  Late-bound: `reports/integration/late/make_run3.sh KEY=VALUE...` writes `~/farm/integration/results/late/run3.sh`
  (exports + the image kit's `headline_job.sh tiny nospec mtp3 nospec-noshard nospec-noprefix nospec-noprefix-nocut`);
  switch on what landed before it starts (MoE dense or chunk from 00177 / 00161, `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`
  if the bridge job shows it, an async-scheduling setting if 00171 shows it). Its `--env`: prefix flags on,
  `HEADLINE_TINY_SPEC=3 HEADLINE_MTP_SAMPLED=1 HEADLINE_MTP_SAMPLED_LEVELS=32,1 VLLM_HPU_SPEC_SAMPLED_VERIFY=2
  HEADLINE_HOLD_S=4500`. Arms in priority order: tiny gate (`--spec 3`); no spec 32 / 8 / 1 (2 greedy + 2 sampled
  rounds); MTP k = 3 (greedy 32 / 8 / 1, then sampled 32 and 1 with `=2`); the low-stream bisect (greedy 8 and 1, 96
  tokens): `VLLM_HPU_QSA_INDEXER_SHARD=0`, then prefix flags off, then also `VLLM_HPU_DECODE_CUT_INTERVAL=0`.
**Build 4 and the layout re-check (submitted 04:55 UTC; the fourth agent's last step):**
- **00186** `real8-build4` (all, 80 min, image `farm-integration:30f4e37ef08c`; plugin code `bcaa3ed6` = builds 1-3 +
  program head a16ebdde + engine-step ba1441b7 + prefill 92e65144 + engine-step-mtp-record 2b90707f + async-sched
  fb5cefca; CPU `reports/integration/cpu/out-b9/summary.txt`: 164 files, 160 green, 3 card-only skips,
  test_moe_glue_layer's order test fixed in bcaa3ed6 and green alone). Arms: `tiny mtp2-async nospec-async mtp2 nospec`
  (32 / 8 / 1 streams, two greedy and two sampled rounds; async arms first: `VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2` with
  `MTP_TIMING=0`, `=1`; the synchronous mtp2 arm with `MTP_TIMING=2`, the per-step record). `--env`: prefix flags, MoE
  chunk, `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`, health / request timeouts 900 / 600 s. Late-bound:
  `reports/integration/late/make_run4.sh KEY=VALUE...` (`ARMS=...` to change the arms, e.g. without the async arms if
  async-sched's 00179 shows async wrong). The submitting `farm job` copies the results to
  `reports/integration/card/00186-real8-build4/` if it still runs; otherwise `bin/farm fetch integration 00186
  00186-real8-build4`.
- **00187** `layout-2xtp4` (all, 50 min, stage-1, same image): `tools/flash_next/integration/layout_2xtp4.sh` (on the
  node: `~/farm/integration/results/late/layout_2xtp4.sh 30f4e37ef08c`): two TP4 + EP4 containers (cards 0,1,2,7 /
  3,4,5,6, their module groups' CPUs), each `headline_job.sh mtp3 mtp1` at 16 streams, two greedy + two sampled rounds,
  every round barrier-aligned across the two servers (`--barrier-dir`). Nothing copies its results: `bin/farm fetch
  integration 00187 00187-layout-2xtp4`. Compare per-server ms per token at 16 streams and the sum of the two
  aggregate tok/s with build 4's TP8 at 32 streams.

**For build 4 (notes received after the hand-back, 02:40 UTC):**
- MoE: 00161 (build 2 image, MTP k = 3) measured `VLLM_MOE_CHUNK=64,8192 VLLM_MOE_TOKEN_BOUNDARY=512,1000000` at
  38.7 / 38.0 ms per token at 32 streams against 43.1 / 40.8 without (about -12 ms per step), 26.2 / 26.4 against
  24.9 / 24.2 at 1 stream (+3 ms), 23.9 against 29.3 at one 14.8K-token prompt; acceptance unchanged. 00177 decides
  chunk against kernel-fixes' dense form (`VLLM_HPU_MOE_DENSE=1`, in build 3).
- Async scheduling (async-sched track): its code is NOT in build 3's image, so an env switch in `run3.sh` does nothing
  there; its own 00176 measures build 3 + its code. For build 4 merge commit `fb5cefca` of
  `feat/flash-next-gaudi-async-sched` (not the branch head, which re-merges the program head with PR #16's QSA data
  flow; it merges cleanly into this branch). Flags: `VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=1` (async without speculation),
  `=2` (with MTP: drafts stay on the device); with `=2` set `VLLM_HPU_MTP_TIMING=0` (the timer's device wait hides the
  overlap). Under async, engine-step's candidate sampling needs engine-step `01d66e31` (otherwise sampled requests fall
  back to vLLM's sampler), and `VLLM_HPU_PLE_HOST_NGRAM_IDS=1` has no effect. 00176's numbers arrive about 06:00 UTC.
- Keep out: `VLLM_HPU_WORKER_CPU_PIN`, `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK` (00156 arm B: slower).

**Build 3** = commit `d00c0a9a` (plugin `848deb15`): build 2 (`b2a7f62f`) + engine-step `3e568842` (both server-killer
fixes e9032e68 / logprobs, candidates on the vocabulary shard `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2` (not set), sampled
speculative steps in the candidate region, request parameters on the general sampler) + kernel-fixes 86104c9d and
b773ad72 (`VLLM_HPU_MOE_DENSE`, default 0). CPU: `reports/integration/cpu/out-b8/summary.txt` (157 files: 152 green, 3
card-only skips, `test_main_evidence.py` and `test_main_moe_glue.py` failed once in the parallel run and pass alone:
`out-b8/rerun_alone.txt`; `test_moe_dense.py` 6 passed on 848deb15). The program branch has since merged the QSA data
flow (PR #16, d9791183): not merged into this branch (the same code via qsa-fused).
**Open questions for the next builds** (coordinator, 01:50-02:00 UTC): the no-speculation 8-stream step (70 ms in
build 2, 58.7 in build 1, 45.9 on the program head) and MTP at 1 stream under k = 2 with `=2` (84 ms per step against
66): 00178's bisect and its MTP arm answer them. Keep a build N+1 ready and submit a TP8 headline every 3 to 4 hours
(the node runs until about 00:00 UTC on 2026-11-22); the coordinator gives TP8 headlines the head of the queue.

**Build 1 commit `fbb685ac`** (pushed; job 00128 ran it). The plugin code is final since `b691faa8`; the later commits only add the
job kit `tools/flash_next/integration/` and this file. What is in it (first-parent merges on top of the qsa-fused
branch the first agent started from):

| Merged | Head | Flags the build's servers set |
|---|---|---|
| qsa-fused (branch base of this build) | `a5f837ca` | `VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1` (never 2 or 3; `VLLM_HPU_QSA_TABLE_WIDTH` stays 0) |
| mtp-card-ready | `f1aa2292` | MTP server only: `--speculative-config {"method": "mtp", "num_speculative_tokens": 3}`, `VLLM_HPU_GDN_CORE_VERIFY=1 VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1` |
| step-anatomy | `2ef735dc` | none (the step timer stays off) |
| program head (PRs #13, #14) | `938dada4` | none |
| kernel-fixes | `821253cf` | `VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2` (MoE glue stays OFF) |
| native-replay | `1a1e1fcf` | `VLLM_HPU_DECODE_CUT_INTERVAL=1000` (never `VLLM_HPU_GRAPH_ASYNC_REPLAY`, never `HCCL_PRIM_*`) |
| engine-step | `f0fd0a2b` | `VLLM_HPU_CANDIDATE_SAMPLING=1` (sampled requests only; `VLLM_HPU_FLASH_NEXT_DECODE_METADATA` stays 0) |

Every server also gets the farm preset (`PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1
VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 VLLM_HPU_PLE_HOST_NGRAM_IDS=1
VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true`) and `VLLM_GRAPH_RESERVED_MEM=0.4` (coordinator: run-time
graphs ran out of device memory in prefill's job 00014 at the default 0.1). Server command: the farm's
(`--tensor-parallel-size 8 --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32 --enable-prefix-caching
--gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048`).

Left out: the prefill branch (`VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS`: card-proven for prefill only, not needed for the
decode headline), spec-lookup (n-gram drafting, not asked for), card-farm (tools only), step-anatomy's newer head
`f6b5c425` (anatomy job tools only). Collectives' work is in through the program head (PRs #13 and #14), flags off.

**Card jobs (track `integration`).** Results land in `reports/integration/card/<id>-<name>/` when the waiting `farm`
command on the Mac copies them; if this agent is gone: `bin/farm ledger | grep -w integration`, then
`bin/farm fetch integration <id> <id>-<name>`.
- (a) **00121** `tiny8-build` (all, 20 min, commit `b54ca4fd`: same plugin code as `fbb685ac`): `bash
  tools/flash_next/rental/mtp_tiny_card.sh 3 "build:<every flag above>"` = the tiny model at TP8, mode `hc` without
  speculation, then `hc+mtp3` (fused verification), tokens against the CPU reference and against the plain server.
  Read the `ARM build:` lines and `summary.json`. Job id: first lines of `reports/integration/card/submit-tiny8-build.log`.
  (Coordinator: a mismatch at token 4 of the tiny model's 700-token prompt is a known near tie, not a defect.)
- (b) **00128** `real8-headline` (all, 40 min, commit `fbb685ac`): `tools/flash_next/integration/headline_job.sh nospec mtp3`,
  late-bound: if `~/farm/integration/results/late/run.sh` exists on the node when the job starts, the job runs that
  file instead (upload with `bin/node-scp` to change arms or flags without losing the queue place). Job id: first
  lines of `reports/integration/card/submit-real8-headline.log`. Results: `<id>-real8-headline/headline.md` (the
  table), `nospec/` and `mtp3/` (`scaling.json`, `probe.log`, `server0.log`, `versions.txt`), `hlsmi_mem.log`.
- Both need a bump by the coordinator (ids sent to `main`).
- If MTP crashes or garbles at 8 or 32 streams: the job reruns that server with `VLLM_HPU_QSA_INDEXER_SHARD=0` by
  itself when 15 minutes of the hold are left (qsa-fused: the shard never ran in a served speculative step).

**How to read (b).** ms per step = `steady_ms_per_token_p50 x tokens_per_chunk_p50` (median gap between a stream's
chunks, one chunk per engine step); with MTP tokens per step = `tokens_per_chunk_p50` and acceptance per position =
the `SPEC` lines (the `/metrics` counters around each measured round); the MTP split = `MTP step timing` lines in
`mtp3/server0.log` (Ft = verify part, Fd = draft / k). Read the TEXT lines: a round whose later streams decode garbage
is not a number. Real-model greedy texts differ across batch sizes (compare meaning at 8 and 32 streams, not text).
`headline_table.py <results folder>` prints the table.

**CPU tests** (one pytest process per file, 4 at once, `reports/integration/cpu/run_tests.sh CHECKOUT OUT LIST 4`):
the run on the final code `b691faa8` is `reports/integration/cpu/out-b4/summary.txt` (list `list_b4.txt`, 136 files:
native-replay's and engine-step's test files first, then the worker, sampler, MTP and whole-model tests, then every
Flash-Next model, QSA, GDN, HC, MoE and rental test of the earlier list). Known failures, both pre-existing on
mtp-card-ready (its own gate `reports/mtp-card-ready/gate/merge-6285cfae-gate.txt`, lines 37 and 110):
`test_engine_mtp.py::test_sampled_requests_through_the_speculative_engine` (sampled verification,
`VLLM_HPU_SPEC_SAMPLED_VERIFY=1`, not used by this build: `output with shape [1, 248320] doesn't match the broadcast
shape [4, 248320]`) and `test_sh_run_sheet.py` (`tiny_smoke.sh --spec` is not named in the run sheet).

**Next, in order**
1. When (a) ends: the `ARM build:` lines. A failure on a speculative rung: do not trust (b)'s MTP numbers; tell the
   coordinator with the error lines.
2. When (b) ends: the table (`headline.md`, also at the end of the job log), the texts, `hlsmi_mem.log`; send the table
   to the coordinator at once and write it below with the build commit and flags.

### 00128 `real8-headline`, server 2 (MTP k = 3), 23:46:36-23:57:05 UTC (same build and flags)

Plus `--speculative-config {"method": "mtp", "num_speculative_tokens": 3}` and `VLLM_HPU_GDN_CORE_VERIFY=1
VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1`; greedy (`--no-sampled-check`: sampled
verification is off in build 1); one measured round per level after a warm round. MEASURED
(`reports/integration/card/00128-real8-headline/mtp3/`, `headline.md`):

| streams | tokens per step | acceptance per position | ms per step | ms per token | tok/s per stream | no speculation, same build |
|---|---|---|---|---|---|---|
| 32 | 2.61 (acceptance length 2.68) | 0.74 / 0.54 / 0.40 | 106 | **40.7** | **24.6** | 51.1 ms (19.6 tok/s) |
| 8 | 2.76 (2.80) | 0.77 / 0.60 / 0.43 | 81 | 29.3 | 34.1 | 58.7 ms |
| 1 | 2.72 (2.78) | 0.83 / 0.57 / 0.39 | 66 | 24.3 | 41.1 | 43.6 ms |

- **MTP pays at 32 streams with the QSA data flow** (1.26 times; at TP4 on the torch QSA path it lost: 134 against
  108, coordinator's 00099/00104). Best per-stream number of the night: 24.6 tok/s at 32 streams (target 100).
- The 32-stream round still compiled verification shapes (warm round 91 s; pauses up to 47 s in the measured round,
  span 138 ms per token, aggregate 75 tok/s): the steady median gap is the reading, a second round confirms it
  (build 2 runs two).
- Split (`MTP step timing`, rank 0, mean of 25 steps, second record per shape): 128 rows verify 85.5 ms + draft
  38.1 ms (pass0 4.0, pass1 19.3 (compile-inflated), pass2 6.8, final wait 6.0); 32 rows verify 62.3 + draft 18.1;
  4 rows verify 54.4 + draft 14.6.
- Acceptance equals 00104 (TP4) and 00021 (TP8, 1 stream): the request-sharded indexer and grouped attention in the
  verification step do not hurt the drafts. Texts coherent at 32, 8 and 1. No fallback line.
- Card memory (`hlsmi_mem.log`): about 65.5 GB per card used while serving, both servers (98 GB cards). After server
  1 stopped, its memory was released about 20 s after its processes ended (the MTP server's load started 10 s after
  the stop and was not hurt).

## Headline across builds (TP8 + EP8, steady ms per step and tok/s per stream; the owner's target is 10 ms per token)

ms per token = ms per step / tokens per step. Greedy = temperature 0; sampled = no temperature in the request (the
checkpoint's temperature 1.0, top_k 20, top_p 0.95). Reference (program head, 00021, greedy, no speculation): 41.6 /
45.9 / 84.1 ms per step at 1 / 8 / 32 streams.

| build (job) | server | streams | greedy ms/step | greedy ms/token (tok/s) | sampled ms/step | sampled ms/token (tok/s) | tokens per step (MTP) |
|---|---|---|---|---|---|---|---|
| 1 (00128) | no spec | 1 | 43.6 | 43.6 (22.9) | 39.5 | 39.5 (25.3) | |
| 1 (00128) | no spec | 8 | 58.7 (one round) | 58.7 (17.0) | 57.4 | 57.4 (17.4) | |
| 1 (00128) | no spec | 32 | 51.1 (2nd round; 1st 67.3) | 51.1 (19.6) | 68.4 (one round) | 68.4 (14.6) | |
| 1 (00128) | MTP k = 3 | 1 | 66 | 24.3 (41.1) | not run | | 2.72 |
| 1 (00128) | MTP k = 3 | 8 | 81 | 29.3 (34.1) | not run | | 2.76 |
| 1 (00128) | MTP k = 3 | 32 | 106 | **40.7 (24.6)** | not run | | 2.61 |
| 2 (00156) | no spec (A) | 1 | 44.0 / 33.0 | 33.0 (30.3) | 43.2 / 31.6 | 31.6 (31.7) | |
| 2 (00156) | no spec (A) | 8 | 70.0 / 69.9 | 69.9 (14.3) | 58.8 / 69.3 | 58.8 (17.0) | |
| 2 (00156) | no spec (A) | 32 | 53.7 / 62.6 | 53.7 (18.6) | 58.9 / 52.7 | 52.7 (19.0) | |
| 2 (00156) | B: A + CPU pin + hash check off | 1 | 41.2 / 41.1 | 41.1 (24.3) | not run | | |
| 2 (00156) | B: A + CPU pin + hash check off | 32 | 68.5 / 69.0 | 68.5 (14.6) | not run | | |
| 2 (00156) | MTP k = 2 (C) | 1 | 84 / 84 | 41.7 (24.0) | 65 / 77 | 29.0 (34.5) | 2.00 greedy, 2.25 / 1.88 sampled |
| 2 (00156) | MTP k = 2 (C) | 8 | 99 / 91 | 39.8 (25.1) | not run | | 2.29 |
| 2 (00156) | MTP k = 2 (C) | 32 | 91 / 104 | **40.1 (24.9)** | 97 / 111 | **46.7 (21.4)** | 2.27 / 2.29 greedy, 2.08 sampled |
| 2 (00156) | MTP k = 2 (C), 8 x 14.8K prompts | 8 | 86 | 35.0 (28.5) steady | | | 2.46 (acceptance 0.80 / 0.63) |
| 3 (00178) | no spec | 1 | 29.4 / 29.7 | 29.4 (34.0) | 41.0 / 31.4 | 31.4 (31.9) | |
| 3 (00178) | no spec | 8 | 53.4 / 52.9 | 52.9 (18.9) | 51.2 / 50.4 | 50.4 (19.8) | |
| 3 (00178) | no spec | 32 | 69.1 / 66.7 | 66.7 (15.0) | 54.2 / 55.1 | 54.2 (18.4) | |
| 3 (00178) | MTP k = 3 | 1 | 72 / 63 | 24.5 (40.8) | 68 / 69 | 27.2 (36.7) | 2.56 |
| 3 (00178) | MTP k = 3 | 8 | 79 / 80 | 29.3 (34.1) | not run | | 2.67 / 2.72 |
| 3 (00178) | MTP k = 3 | 32 | 104 / 105 | **39.4 (25.4)** | 121 / 119 | 49.0 (20.4) | 2.56 / 2.67 greedy, 2.46 / 2.37 sampled |
| 3 (00178) bisect, greedy | no spec, QSA shard off | 8 / 1 | 57.5, 43.4 / 36.6, 29.3 | | | | |
| 3 (00178) bisect, greedy | no spec, prefix flags off | 8 / 1 | 48.7, 48.7 / 38.4, 37.7 | | | | |
| 3 (00178) bisect, greedy | no spec, prefix off, cut 0 | 8 / 1 | 51.7, 48.3 / 40.6, 37.1 | | | | |
| MoE A/B 00161 (build 2 image) | MTP k = 3, base / VLLM_MOE_CHUNK | 32 | | 40.8 / **38.0 (26.3)** | | | |
| MoE A/B 00177 (build 3 image) | MTP k = 3, base / VLLM_HPU_MOE_DENSE | 32 | | 40.2 / 43.4 | | | |
| 4 (00186) | no spec (sync) | 1 / 8 / 32 | 33.1, 34.3 / 52.5, 48.4 / 64.1, 62.3 | 33.1 / 48.4 / 62.3 | 38.5, 34.3 / 54.1, 48.6 / 60.0, 55.9 | 34.3 / 48.6 / 55.9 | |
| 4 (00186) | no spec, async (ASYNC_SCHED=1) | 1 / 8 / 32 | 53.5, 55.7 / 69.6, 63.8 / 54.4, 72.4 | 53.5 / 63.8 / 54.4 | 52.0, 54.3 / 66.3, 67.5 / 53.9, 49.8 | 52.0 / 66.3 / 49.8 | |
| 4 (00186) | MTP k = 2 (sync, MTP_TIMING=2) | 1 / 8 / 32 | 63, 64 / 71, 72 / 88, 79 | 26.1 / 30.3 / 35.0 (28.6) | 63, 64 / 69, 72 / 103, 85 | 26.5 / 33.2 / 40.0 (25.0) | 2.42 / 2.29-2.35 / 2.27-2.29 |
| 4 (00186) | MTP k = 2, async (ASYNC_SCHED=2) | 1 / 8 / 32 | 55, 57 / 65, 67 / 77, 77 | 27.5 / 27.4 / **33.7 (29.7)** | 81, 77 / 94, 90 / 82, 98 | 35.3 / 42.1 / **38.9 (25.7)** | 2.00 / 2.29-2.37 / 2.29 |
| 5 (00222) | A: MTP k = 2, async =2, dense from 1 token, chunk, placement | 1 / 8 / 32 | 35.1, 34.9 / 42.5, 42.0 / 65.3, 64.7 | 15.3 / 17.6 / **28.3 (35.3)** | 35.8, 36.0 / 43.0, 43.2 / 70.5, 70.8 | 18.0 / 19.6 / **33.3 (30.0)** | 2.29 / 2.33-2.39 / 2.29-2.31 |
| 5 (00222) | A-spin (spinner on) | 1 / 8 / 32 | 34.5, 34.9 / 40.3, 40.7 / 64.0, 63.8 | 15.1 / 17.3 / 27.9 | 35.6, 36.0 / 42.2, 43.2 / 70.7, 70.4 | 16.0 / 20.8 / 32.3 | 2.29 / 2.33 / 2.29 |
| 5 (00222) | B: MTP k = 3, async =2 | 1 / 8 / 32 | 40.7, 39.7 / 48.0, 48.2 / 80.9, 80.0 | 15.2 / 17.6 / 30.3 | 40.0, 40.1 / 50.2, 50.7 / 89.9, 88.8 | 15.0 / 21.2 / 37.5 | 2.61 / 2.59-2.72 / 2.64-2.67 |
| 5 (00222) | C: MTP k = 2 sync (MTP_TIMING=2) | 1 / 8 / 32 | 36.7, 36.8 / 43.8, 42.7 / 68.5, 67.6 | 16.1 / 18.7 / 29.6 | 38.6, 38.4 / 45.0, 48.3 / 73.5, 73.6 | 17.5 / 21.1 / 34.5 | 2.29 / 2.29-2.33 / 2.29 |
| 5 (00222) | A-inline (PT_HPU_DISABLE_ASYNC_COLLECTIVE=1), greedy | 1 / 8 / 32 | 34.7, 34.3 / 40.6, 40.4 / 62.8, 63.6 | 15.0 / 17.4 / **27.0** | | | 2.29 / 2.33 / 2.29-2.33 |
| 5 (00222) | A's server, per round: off / draft graph / I01 | 1 / 8 / 32 | off 35.5, 35.6 / 41.2, 40.9 / 64.8, 65.0; dg 34.3, 34.3 / 39.5, 39.2 / 63.5, 63.7; md 34.9, 35.9 / 41.2, 41.3 / 64.9, 64.9 | dg 15.0 / 16.2 / 27.4 | | | |
| 5 (00222) | FP8 K/V (A + fp8_inc + the copy_to logger), greedy | 32 | 65.2, 66.3 | 29.5 | | | 2.21-2.25 |

Build 2 readings (arm A, two measured rounds per level, `a / b` = round 1 / round 2; the better round in the
ms-per-token column): sampled equals greedy now (server0.log: `Candidate sampling (VLLM_HPU_CANDIDATE_SAMPLING=1): a
sampled decode step drew in the logits region ... from the gathered logits`, no fallback line); a 32-stream step varies
by about 10 ms between rounds of one server (also in build 1: 67.3 / 51.1); **8 streams is slower than 32 in both
builds** (70.0 / 69.9 here, 58.7 in build 1, 45.9 on the program head): systematic, unexplained (candidates: the QSA
request shard at 1 request per rank with its 12 all-gathers, grouped attention from 8 rows). Arm B's eight
`VLLM_HPU_WORKER_CPU_PIN=1: rank r module m: share r of 8, CPUs 19r-(19r+17)` lines are in its server0.log.

Build 2, arms B and C: B (native-replay's `VLLM_HPU_WORKER_CPU_PIN=1` + `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1`)
is slower (32 streams 68.5 / 69.0 against A's 53.7 / 62.6; server load 811 s against 477): drop both. C (MTP k = 2,
`VLLM_HPU_SPEC_SAMPLED_VERIFY=2`): at 32 streams 40.1 / 45.7 ms per token greedy and 46.7 / 53.1 sampled, the
candidate verification path's first real-model run (no fallback line, coherent texts, acceptance 0.66 / 0.44); timer at
96 rows verify 82-86 ms + draft 16-19 ms (k = 3 at 128 rows verified in about 85 ms: k barely moves the verification
cost); the 8 x 14.8K long-prompt round ran without a crash (5c770d96 in the image) with acceptance 0.80 / 0.63. At 1
and 8 streams k = 2 on build 2 is slower than k = 3 on build 1 (part of it the 8-stream regression of arm A).

Build 3 (00178) and the MoE A/Bs: the 8-stream regression of build 2 is not reproduced (53 ms with every flag; the
bisect arms land at 43 to 57 ms with 5 to 10 ms between rounds and servers: no flag explains build 2's 70);
MTP k = 3 at 1 stream is 24.5 to 27.9 ms per token with `VLLM_HPU_SPEC_SAMPLED_VERIFY=2` set, so build 2's 1-stream
slowdown was k = 2 on that server; `VLLM_MOE_CHUNK` wins at 32 streams (00161: -12 ms per MTP step), the dense form
loses on the served model (00177). Single-server numbers at 32 streams vary by about 10 ms (00178: greedy 69 / 67
against sampled 54 / 55 on the same server).

Build 5 (00222, image eb21ac75a7c8, late: `VLLM_HPU_MOE_DENSE_MIN_TOKENS=1`, placement `place`; 10:32-11:57 UTC,
0 of 7 arms failed, no kernel error lines; table `reports/integration/b5/table-00222.md`, made by
`reports/integration/b5/b5_compare.py`). MEASURED: **A (serve candidate) 28.3 / 28.3 ms per token at 32 streams greedy
(35.3 tok/s per stream; -16 % against build 4's 33.7, -13 % against 00205's 32.4), 17.6 to 18.3 at 8 (-36 %), 15.3 at
1 (-44 %: 65 tok/s); sampled 33.3 / 33.7, 19.6 / 19.8, 18.0 / 19.0** (build 4: 38.9, 42.1, 35.3). Spinner on against
off: within about 1 ms both ways (a wash). k = 3 (B) loses at 32 streams (30.3 greedy, 37.5 to 40.4 sampled) and ties
at 8 and 1. Async against sync (C, with the per-step record): about -1.5 ms per token at 32 and 8; sampled at 8 async is
now faster (43 against 45 to 48 ms per step; build 4 had 92 against 71). Inline collectives: -1.8 / -1.7 / -0.5 ms per
step at 32 / 8 / 1. On A's server, rounds alternated through the switch file: the draft graph -1.3 / -1.7 / -1.2 ms per
step at 32 / 8 / 1 with identical 1-stream texts; I01 no change at short context (engaged: 13 QSA layers). FP8 K/V:
speed at 32 streams equal to A (65.2 / 66.3 ms per step); teacher-forced logprobs against A's bf16 at 16K / 64K / 120K:
mean |delta| 0.357 / 0.085 / 0.174, p99 4.12 / 0.72 / 1.51, top-1 91.3 / 96.3 / 90.5 %, perplexity 6.10 / 3.816 / 5.26
against 6.56 / 3.817 / 5.29; PASSNOISE (two passes, one server) exactly 0 on both servers; the cross-server bf16 floor
is unmeasured (the kvlpfb fallback did not fire); the probe's own needle is void (1 token, empty text, both runs); the
chat needle at 64K hit 1 of 1 and 32 of 32 on the FP8 server; 1-stream decode at 64K 16.1 ms per token; the 32 x 64K
round (153 ms per token, TTFT 33 s) overlaps the prefill tails (not a decode number). Tiny gate passed; dense kept and
running in all six servers; texts coherent; peak card memory 78.9 GiB; the only traceback is C's shutdown
(EngineDeadError at the stop). **Defect (VERIFIED, fixed in b71b396f): place.sh moved the job script off its reserved
vCPUs, so the servers ran on 144, 128, 112, 96, 80, 64 CPUs (helpers per rank 16 to 6); loads 328 to 471 s; decode did
not visibly suffer (the 80-CPU arm was the fastest).** Sampled MTP still takes vLLM's sampler
(`VLLM_HPU_CANDIDATE_SAMPLING=1`; =2 is engine-step's lever). Recommended serve setting (INFERRED, not yet run as one
combination): MTP k = 2 + async =2 + dense from 1 token + chunk + placement (place only, with the CPU fix) +
`PT_HPU_DISABLE_ASYNC_COLLECTIVE=1` + `VLLM_HPU_MTP_DRAFT_GRAPH=1` (each about -1.5 ms per step alone; INFERRED 26 to 27
ms per token at 32 streams together), bf16 K/V until a cross-server floor puts the FP8 differences in context.

Build 4 (00186, image 30f4e37ef08c): **best at 32 streams so far: MTP k = 2 with async scheduling
(`VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2`, `VLLM_HPU_MTP_TIMING=0`), greedy 33.7 ms per token = 29.7 tok/s per stream (803
tok/s aggregate in round 2), sampled 38.9**; acceptance unchanged under async (0.75 / 0.55). Async without speculation is
worse at 1 and 8 streams (1 stream 54 against 33 ms per step): under async `VLLM_HPU_PLE_HOST_NGRAM_IDS` has no effect
(the n-gram ids go back into the graph; INFERRED cause). Every arm had the MoE chunk and synchronous queue flushing.

## Merge log (what the conflicts were)

- `b38ffaa1` (first agent) mtp-card-ready into qsa-fused: one conflict in `hpu_flash_next_mtp.py` (vocabulary-parallel
  draft arg max against `draft_argmax`): `_draft_tokens` takes the vocabulary-parallel arg max when
  `VLLM_HPU_VOCAB_PARALLEL_LOGITS` is on and `draft_argmax` otherwise (as spec-lookup resolved it).
- `bbf1d16e` (first agent) step-anatomy `2ef735dc`: clean.
- `8ef34b34` program head `938dada4`: clean.
- `6a956cb4` kernel-fixes `821253cf`: `envs.py`, `env_variables.md`: both sides kept.
- `356d0866` native-replay `1a1e1fcf`: `env_variables.md` only (the program head's newer vocabulary-parallel row
  kept, the branch's two new rows added).
- `b691faa8` engine-step `f0fd0a2b`: `hpu_model_runner.py` (the decode block table is now built inside engine-step's
  dense-metadata branch, with qsa-fused's table width; the step timer's setup lap after engine-step's sampler
  inputs); `envs.py` and `env_variables.md`: both sides kept.

- Build 2: `c02bfd25` engine-step `3757e7e0` (clean), `3a4aee22` native-replay `9b0e7655` (`env_variables.md`: rows of
  both sides), `46b3ce5b` prefill `d072b775` (`qsa.py`: qsa-fused's decode-row helpers and prefill's split prompt-row
  selection, both kept; `hpu_model_runner.py`: the runtime-switch poll and the prompt-graph knob, both kept),
  `ff58bb6b` engine-step `ddc6bbbf` (clean), `4b602948` prefill `2abf636a` (clean, a test only), `312c9299`
  mtp-card-ready `6f5004f6` (`envs.py`: the newer description of `VLLM_HPU_SPEC_SAMPLED_VERIFY`;
  `hpu_flash_next_mtp.py`: a comment). `a39d1f48` adjusts two of prefill's tests to the merged tree (a stub of
  engine-step's `_poll_runtime_switches`; the rule-9 cast guard skipped because qsa-fused's `select_blocks` no longer
  casts): for prefill's own merge after engine-step and qsa-fused.
- Build 5: `9b971279` program head `adbcff30` (`envs.py`: the step timer's four variables had moved on the head; both
  copies identical, the head's placement kept); `f6aba529` mtp-draft-graph-b4 `100f4bf0` (clean); kernel-fixes
  `11bc2f6d` / `a5786c55` / `b92402ef` and native-replay `773a7f7c` cherry-picked (clean); `ecee792f` = qsa-fused
  `f3b63114` (`qsa.py`, `envs.py`, `env_variables.md`: both sides had added code at the same place, both kept);
  `125429d3` engine-step-mtp-record `07932956` (`hpu_model_runner.py` after the target forward: the record's replay
  timer inside the profiler block, then the step timer's forward lap, both kept); `a09aad30`, `03156364`, `703b3cf7`,
  `9b4faf7c`, `d3fb1b11` cherry-picked (clean).

## Card results

### 00128 `real8-headline`, server 1 (no speculation), 23:36:27-23:46:27 UTC (commit `fbb685ac`, plugin `b691faa8`)

TP8 + EP8, the build's flags (handoff table), 128 tokens per stream, about 25-token prompts, a warm round (48 tokens)
before every level. MEASURED (`reports/integration/card/partial-00128/nospec/scaling.json`, copied from the node while
the job ran; the job's own copy lands in `reports/integration/card/00128-real8-headline/nospec/`). Steady ms per step =
the median gap between a stream's streamed tokens, medianed over the streams.

| streams | greedy ms per step | greedy tok/s per stream | sampled ms per step | sampled tok/s per stream | reference (00021, greedy) |
|---|---|---|---|---|---|
| 1 | 43.6 | 22.9 | 39.5 | 25.3 | 41.6 |
| 8 | 58.7 | 17.0 | 57.4 | 17.4 | 45.9 |
| 32 | **51.1** (second round; 463 tok/s aggregate); first round 67.3 with compile pauses up to 4.6 s | **19.6** | 68.4 (one round; 376 tok/s aggregate) | 14.6 | 84.1 |

- 32 streams greedy: 84.1 -> 51.1 ms per step (-39 %). The target is 10 ms per token.
- 8 streams is slower than the reference (58.7 against 45.9) and than 32 streams: one measured round only; the
  32-stream level shows how much a first round after the warm round can be off (67.3 against 51.1). Unexplained.
- Sampled at 32 streams costs 17 ms more than the best greedy round, in one first round; at 1 and 8 streams sampled
  equals greedy. Whether `VLLM_HPU_CANDIDATE_SAMPLING=1` took the sampled rows is not visible in the log (the path
  logs nothing when it acts).
- Texts: every stream at 8 and 32 streams, greedy and sampled, writes a coherent text about its own prompt.
- Server: healthy after 476 s; 26.2 GiB per card reserved for graphs, KV cache 2.79M tokens (maximum concurrency 21x
  at 131K); the cut-interval line is in the log; no fallback line; no traceback.

### 00121 `tiny8-build`, 23:32:52-23:36:27 UTC (commit `b54ca4fd`, TP8, every build flag): PASSED

MEASURED (`reports/integration/card/00121-tiny8-build/build/summary.json`, job log `00121-tiny8-build.log`):
- mode `hc` (fused GDN decode + fused hyper-connections + all build flags, no speculation): tokens equal the CPU
  reference; decode 9.9 ms per token (shortest prompt, 16 tokens).
- mode `hc+mtp3` (k = 3, fused GDN verification, draft loop, chunk KV): PASSED on the first rung (`verify`): tokens
  equal the CPU reference and the server without speculation; 0 of 270 drafts accepted (random weights: only the
  all-rejected verification path is proven); decode 24.3 ms per token.
- Kernel lines: only "killed while waiting" (servers stopped), not a card fault.
- Not covered: the tiny server serves at most 4 requests and the smoke sends one at a time, so the request-sharded
  QSA indexer (8 or more rows at TP8) did not run here; 00128 runs it at 8 and 32 streams.
