# Flash-Next integration card build: one build of every card-proven branch, the TP8 headline numbers

This file is the integration track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `integration/card-build-20261120b` (worktree
`/Volumes/Storage/Claude/gaudi-work/wt/integration`), pushed to origin under its own name. **A card build only:
never merged into `feat/flash-next-gaudi`** (each track merges its own pull requests). Raw card results:
`/Volumes/Storage/Claude/gaudi-work/reports/integration/card/`; CPU runs: `reports/integration/cpu/`.
Labels: MEASURED (card, with its file), VERIFIED (CPU test or code line), INFERRED.

## Handoff (if this agent stops)

Updated 2026-11-21 02:25 UTC (fourth agent of the track, Opus 5.5, started 23:12 UTC).

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
