# Flash-Next integration card build: one build of every card-proven branch, the TP8 headline numbers

This file is the integration track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `integration/card-build-20261120b` (worktree
`/Volumes/Storage/Claude/gaudi-work/wt/integration`), pushed to origin under its own name. **A card build only:
never merged into `feat/flash-next-gaudi`** (each track merges its own pull requests). Raw card results:
`/Volumes/Storage/Claude/gaudi-work/reports/integration/card/`; CPU runs: `reports/integration/cpu/`.
Labels: MEASURED (card, with its file), VERIFIED (CPU test or code line), INFERRED.

## Handoff (if this agent stops)

Updated 2026-11-20 23:40 UTC (fourth agent of the track, Opus 5.5, started 23:12 UTC).

**Now:** job **00121** `tiny8-build` PASSED (23:36 UTC, below). Job **00128** `real8-headline` (commit `fbb685ac`)
runs since 23:36:27 UTC (SIGTERM at 00:16:27 UTC). Both
`farm job` commands wait in background shells of the agent's session and copy the results to
`reports/integration/card/` when the jobs end. **Do not run `farm sync`, `farm job` or `farm serve` for this track
while 00121 or 00128 waits**: the node keeps three images per track, a fourth build prunes the oldest (00121's).
A first submit of the headline job (commit `5ec92ead`) failed at its image build: two syncs of this track ran at once,
the later bundle folder `~/farm/integration/b/5ec92eadbe9b` vanished and was re-created empty and root-owned
(reported to the coordinator as a queue problem; INFERRED cause: the first sync pruned it).

**Build commit `fbb685ac`** (pushed; the headline job runs it). The plugin code is final since `b691faa8`; the later commits only add the
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
