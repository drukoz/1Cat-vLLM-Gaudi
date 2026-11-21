# Flash-Next MTP on the card: state, card results and handoff (track `mtp-card-ready`)

This file is the track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `feat/flash-next-gaudi-mtp-card-ready`. Raw card logs:
`/Volumes/Storage/Claude/gaudi-work/reports/mtp-card-ready/card/`; job scripts: `reports/mtp-card-ready/job/`.

## Handoff (if this agent stops)

Last update: 2026-11-21 01:00 UTC (fourth agent of the track, Opus 5.5, started 23:15 UTC). The node is extended to about
for tonight (BRIEF 23:21): card evidence first, no pull request before 04:00 UTC; push the branch after each green
step and tell the coordinator the commit and the flags (the integration track builds from this branch).

**State (00:13 UTC).** With the QSA data flow and the kernel-fixes flags, MTP k = 3 pays at 1 and 8 streams and
breaks even at 32 on TP4 (00126 / 00127: 25.5 / 34.0 / 61.9 ms per token against 29.8 / 43.2 / 63.7; a 128-row
verification costs Ft 131 ms, it was 303 on the torch QSA path); on TP8 (integration 00128) 40.7 against 51.1 ms per
token at 32 streams. The draft loop is sound (acceptance 0.72 / 0.50 / 0.34 at 32 streams, as with it off).
**Sampled verification (I07) works on the card** (exact path, `VLLM_HPU_SPEC_SAMPLED_VERIFY=1`: coherent texts,
acceptance length 2.62 at 1 stream, 2.23 at 32) but adds about 26 ms per step, so sampled MTP loses at 32 streams on
TP4 (82.4 against 76.2 ms per token); the candidate path (`=2`) targets that cost (jobs 00135, 00136). **At 32
streams on TP4 k = 1 is the best k** (00126 arm k1: 55.2 ms per token greedy, 65.8 sampled, against 63.7 / 76.2
without speculation and 61.9 / 82.4 at k = 3); at 1 and 8 streams k = 3. On TP8 (Ft grows about 0.24 ms per row)
k = 2 to 4 are all about 40-41 ms per token at 32 streams by arithmetic.

**The node runs until about 00:00 UTC on 2026-11-22 (BRIEF 00:50); merges allowed again.** Coordinator's plan for
this track (00:42 UTC): (1) the verify-op crash of 00133, cause and fix, CPU first; (2) the pull request of the
card-proven MTP fixes (gate, merge); (3) with engine-step (agent a8a87f4c2008bde9c): sampled MTP verification
through its candidate region (`VLLM_HPU_CANDIDATE_SAMPLING=2`, admission stays on `VLLM_HPU_SPEC_SAMPLED_VERIFY`
1 or 2); (4) the k choice at TP8 from integration's build 2 (00142) and MTP acceptance at long prompts.

**00133 crash (spec-lookup, TP4, n-gram k = 5, fused verify, prefix caching): cause and fix, `5c770d96`.** The
request after one whose decode completed a recurrent block died with `ValidateSyncInputTensors tensor_data is
empty` at `gdn_core_verify`'s first input (the GDN conv pool), in a graph captured after the block commit. The
align-mode copies (`recurrent_commit.commit_conv`, boundary commit and pre-step copy) wrote the conv window through
a slice of the pool (`state[dst].narrow(...).copy_`, `hpu::slice_insert`); `commit_ssm` (whole rows,
`index_copy_`) was fine and its pool was not the failing input (INFERRED from code and log). Fix: whole new rows
built from `index_select` copies, one `index_copy_` on the pool. CPU: `test_align_commit_slice_writes.py` 10 passed
(8 failed before), `test_mtp_align_continuity.py` 76 passed, the 3-turn CPU engine test
(`test_requests_after_a_block_committed_during_speculative_decode`: decode across the block, the same prompt
again, the next turn) tokens equal to k = 0, copies ran, no slice write (`hpu_cpu_engine --slice-write-probe`).
Turn 3 gets no prefix-cache hit on this build, with or without MTP. Card check proposed: spec-lookup reruns 00133's
case with `5c770d96` (or a TP4 MTP arm decoding across 896 tokens, then new batch sizes). The prefix flags of
prefill (`VLLM_HPU_PREFIX_ALIGN_SPLIT` and the two KEEP flags) on and off: scratch tree
`reports/mtp-card-ready/scratch-prefill` (prefill merged, qsa.py ours), `reports/mtp-card-ready/turns_prefix_check.py`,
output `reports/mtp-card-ready/cpu4/turns_prefix/summary.json`.

**Owner's directive (BRIEF 00:28): 8 cards first.** Every TP4 job waits behind the TP8 jobs and probably will
not run tonight. Sampled MTP at TP8 runs in the integration track's build 2 (MTP k = 2, sampled rounds last with
`VLLM_HPU_SPEC_SAMPLED_VERIFY=2`; its build 4681ee37 has this branch up to 6f5004f6, which contains `=2`; the
optional log line is `409c83e5`).

**Branch** (pushed: `origin/feat/flash-next-gaudi-mtp-card-ready` = `5c770d96` and this handoff):
- `5c770d96` the 00133 fix (above) with its tests; `hpu_cpu_engine.py --slice-write-probe` and `num_cached_tokens`.
- 00139 (ended 00:38 UTC, ok): the bench with id checks by value passes on the card at 1 / 8 / 32
  requests (every candidate id holds its value, greedy holds the max); candidate path 3.3 / 10.3 / 19.1 ms against
  5.0 / 18.5 / 33.3 ms exact (a busier node than 00135).
- `409c83e5` one log line at the first candidate step (`VLLM_HPU_SPEC_SAMPLED_VERIFY=2: sampled verification over
  the top 64 candidates of every row (first step: N requests, k = K)`); a refused step logs once per reason.
- `b66e1131` candidate_verify_bench.py: id checks by value (bf16 ties made 00135's set check fail).
- `fb4f4d50` spec_scaling.py: the long prompts of the warm and the measured round start with different text (no
  round reads another's prefix cache: a TP4 prefix-cache hit can read another position's GDN state, BRIEF 23:46).
- `1f2dc38e` I07 fast path: `VLLM_HPU_SPEC_SAMPLED_VERIFY=2`, sampled verification over the top-64 candidates of
  every verification and bonus row (`vllm_gaudi/v1/sample/hpu_candidate_verify.py`, hook
  `HPUModelRunner._candidate_verify`); steps with a seeded request, logprobs or top_k outside 1..64 take the exact
  path (`=1`). `tools/flash_next/candidate_verify_bench.py`: both paths on one card. CPU: 10 + 14 sampler tests,
  60 platform tests, the CPU engine with `=2` and top_k 20 verifies over candidates (4 engine tests passed,
  `reports/mtp-card-ready/cpu4/engine_mtp-cand.log`). Never ran on a card yet (job below).
- `6a0a046e` spec_scaling.py + mtp_real_card.sh (the previous agent's uncommitted probe, 18 tests passed).
- `fc43e7de` merge of the program head `938dada4` (PR #13, #14), clean; `697bffed` merge of
  `origin/feat/flash-next-gaudi-qsa-fused` (`a5f837ca`), clean; `26d9ee1a` merge of
  `origin/feat/flash-next-gaudi-kernel-fixes` (`821253cf`), two both-added conflicts (envs.py,
  env_variables.md), both sides kept.
- `0d1be6fc` FIX: the sampling metadata of a verification step has one row per request (it was padded to the
  logits' 1 + k rows per request, so a sampled request crashed the sampler: test_engine_mtp's sampled test, red
  since the merge of PR 13); the step no longer runs a discarded sampler pass over the bonus rows. Run sheet names
  `tiny_smoke.sh --spec`. CPU: `test_engine_mtp.py` 14 passed on a snapshot of `0d1be6fc`
  (`reports/mtp-card-ready/cpu4/engine_mtp-0d1be6fc.log`).
- `13d7393a` `tools/flash_next/rental/mtp_arms.sh`: several real-model arms in one farm job (each in
  `$FARM_OUT/NAME`, a fallback run without COPY_MODE / DRAFT_LOOP / CHUNK_KV when an arm's server dies or measures
  nothing), hl-smi memory every 15 s; spec_scaling.py exits 1 when nothing was measured. 20 tests passed.
- Not gated (`bin/gate` has five tracks queued: hours). The previous agent's gate of `6285cfae` was red in
  test_engine_mtp (fixed in `0d1be6fc`), test_sh_run_sheet (fixed), and tools/rental signal / git tests that fail
  the same way on other tracks' trees (prefill's gate log `reports/prefill/logs/gate_5d8af094.log`).

**Card jobs queued (23:35 UTC, both from `13d7393a`, image `farm-mtp-card-ready:13d7393a3b8f`; bump asked):**
- **00126 `real4-qsa-mtp-k3-k1`** (any:4, 40 min): `bash tools/flash_next/rental/mtp_arms.sh 'k3|3||--levels
  1,8,32 --tokens 256 --sampled-levels 1,32' 'k1|1||--levels 1,32 --tokens 256 --repeat-top 1 --sampled-levels
  32'` with `VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1 VLLM_HPU_GDN_KERNEL_INTERLEAVE=1
  VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2 VLLM_GRAPH_RESERVED_MEM=0.5 VLLM_HPU_GDN_CORE_VERIFY=1
  VLLM_HPU_MTP_TIMING=1 VLLM_HPU_SPEC_SAMPLED_VERIFY=1 VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1`.
- **00127 `real4-qsa-ref-nospec`** (any:4, 25 min): the same server without speculation, `'k0|0||--levels 1,8,32
  --tokens 256 --sampled-levels 1,32'`, the first six flags.
- **00135 `cand-verify-bench`** (any:1, 15 min, `1f2dc38e`, bumped into the 1-card wave after 00126):
  `python tools/flash_next/candidate_verify_bench.py --device hpu --requests 1,8,32 --out "$FARM_OUT/bench.json"`:
  the exact sampled verification (`=1`) against the candidate path (`=2`) per call on one card, the candidate ids
  against the CPU's topk, valid tokens, first-draft acceptance of both, greedy = arg max.
- **00135 ended 00:23 UTC (ok, 22 s)**: candidate path 3.7 / 6.0 / 13.3 ms per call against 5.4 / 11.9 / 27.6 ms on
  the exact path at 1 / 8 / 32 requests (k = 3, one card); tokens valid, k-th value equal to the CPU's, first-draft
  acceptance of both paths within 1.5 sigma of the CPU's exact expectation (0.097 at 32 requests). Its id-set and
  greedy checks failed on bf16 ties only (88-100 % of rows tie at the 64th value); fixed in `b66e1131`.
- **00139 `cand-verify-bench2`** (any:1, 10 min, `b66e1131`): the same bench with the id checks by value (every
  candidate id holds its value in the CPU's logits). Queued 00:35 UTC; runs only if a card is idle.
- **00138 `real4-qsa-cand-k3-k1`** (any:4, 45 min, `fb4f4d50`, replaced 00136 at 00:26 UTC, keeps its bump): 00126's
  flags with `VLLM_HPU_SPEC_SAMPLED_VERIFY=2`, `mtp_arms.sh 'cand3|3||--levels 32 --tokens 256 --repeat-top 1
  --sampled-levels 1,32 --ctx 8:16000' 'cand1|1||--levels 32 --tokens 256 --repeat-top 1 --sampled-levels 1,32'`.
  Waits behind the TP8 jobs; THE PLAN FOR THE NEXT RENTAL if it does not run.
- (cancelled) **00136 `real4-qsa-mtp3-cand-ctx`** (any:4, 30 min, `fb4f4d50`, bump asked after 00135): 00126's flags with
  `VLLM_HPU_SPEC_SAMPLED_VERIFY=2`, `mtp_arms.sh 'cand|3||--levels 32 --tokens 256 --repeat-top 1 --sampled-levels
  1,32 --ctx 8:16000'` (greedy 32 in the same server, sampled 1 and 32 over candidates, 8 x 16K-token prompts with
  CHUNK_KV). If 00135 shows the candidate path wrong on the card: `--replaces 00136` with `=1`.
- Local waiters write `reports/mtp-card-ready/card/submit-*.log` and copy the results to
  `reports/mtp-card-ready/card/<id>-<name>/`; if they are gone: `bin/farm fetch mtp-card-ready 00126
  00126-real4-qsa-mtp-k3-k1` (and 00127). What to read per arm folder (`k3/`, `k1/`, `k0/`, `*-fallback/`):
  `probe.log` (MEAS / SAMPL lines: ms per step, tokens per step, acceptance per position, TEXT lines),
  `scaling.json`, `server0.log` (`MTP step timing` lines: Ft, Fd; `PT_DEVMEM` = out of graph memory),
  `mtp_step_timing_r*.jsonl`, `hlsmi.log` in the job folder. Report to the coordinator: ms per token at 1 / 8 / 32
  streams greedy and sampled, k = 3 against k = 1 against the reference, Ft / Fd at 128 rows, which verification
  rung ran (the fused one, `VLLM_HPU_GDN_CORE_VERIFY=1`).

**Pending, in order**
1. DONE 00:24 UTC: 00126 / 00127 read, sent to the coordinator, in "Card results" (digest:
   `python3 reports/mtp-card-ready/analyze_arms.py reports/mtp-card-ready/card/00126-real4-qsa-mtp-k3-k1`).
2. I07 fast path on the card: the 1-card bench job (`cand-verify-bench`, below), then a real-model arm with
   `VLLM_HPU_SPEC_SAMPLED_VERIFY=2` (sampled rounds at 1 and 32 streams) against 00126's `=1` sampled rounds:
   `mtp_arms.sh 'cand|K||--levels 32 --tokens 256 --repeat-top 1 --sampled-levels 1,32'` with 00126's flags,
   `VLLM_HPU_SPEC_SAMPLED_VERIFY=2` and K the best k at 32 streams.
3. k from arithmetic (`tools/flash_next/mtp_k_choice.py`): E(k) = 1 + r1 + ... + rk with the unconditional
   per-position rates; cost Ft(rows(k)) + k Fd. k = 4 / 5 only where Fd stays small against Ft (1 stream).
4. Pull request after 04:00 UTC: gate (`bin/gate <worktree> --tools`) on a snapshot, PR to
   `feat/flash-next-gaudi` (the merges of qsa-fused and kernel-fixes are then on the program branch or not: if
   not, the PR carries them, say so).

**Results of the jobs the third agent left (read 23:20 UTC)** are in "Card results" (00099, 00104, 00105).
The fused verify rung's mismatch on the tiny model at TP2 (00105, 700-token prompt, token 4) is a near tie: the
CPU engine rehearsal with the QSA flags diverges on the same prompt at a reference margin of 0.002 (random
weights: every logit within about 0.05), and the coordinator confirmed it (BRIEF 23:23, 23:24).

## Earlier handoff (third agent, 20:20 UTC; superseded, kept for its details)

Last update: 2026-11-20 20:10 UTC (third agent of the track, started 19:00 UTC).

**State in one paragraph.** MTP speculative decoding (k = 3) WORKS on the card. Tiny model (TP8, fused GDN
verification): tokens equal the non-speculative server for prompts of 1 to 2,600 tokens. Real model (TP8 + EP8,
fused verification, greedy, one stream, job 00021 run 3): **2.78 tokens per step, acceptance per draft position
0.83 / 0.57 / 0.39, 84.3 ms per step against 41.6 ms without speculation, 31.0 ms per token steady against 41.6
(1.34 times faster)**. The 32-stream round did not fit the hold (its warm round: acceptance length 2.13,
positions 2 and 3 not collapsed, so the draft loop's `argmax -> .int()` is sound at 32 rows). The overhead of a
speculative step (43 ms at one stream), not the acceptance, is the limiter now. Three card failures were found
and fixed on the way (6, 7 below; the CPU-only break of the first fix).

**Done and verified**
- CPU (commits up to `77af8a5e`, previous agents): engine start with the drafter, the drafter's block table, two
  verification crashes at real block sizes, the recurrent pool's page count with MTP, k = 1..5, sampled verification
  behind `VLLM_HPU_SPEC_SAMPLED_VERIFY` (0 = a clean HTTP 400 for a sampled request), `VLLM_HPU_MTP_DRAFT_LOOP`,
  `spec_probe.py`, `tiny_smoke.py --spec K`, `mtp_k_choice.py`. Gate lines: `reports/mtp-card-ready/gate/`.
- Card failure 6 (real model, first prompt): `Neither storage attached to input tensor, not its view`; the PLE
  conv state was written through `view[:, :state_len, :]` of the state a speculative config widens. Fixed in
  `f0e7d02f` (PLE, GDN conv prefill and plain decode: whole rows with `index_copy_`); `9d013a36` gives the GDN
  copies an int64 index (the CPU engine needs it; found by spec-lookup). Card-verified (tiny and real model).
- Card failure 7 (tiny model, first draft after a decode step): `Empty tensor optional` in the drafter's gathers;
  the MTP aux hidden state was a view of `_mtp_hidden_buffer`, which upstream fills in the forward with a slice
  copy. Fixed in `9d013a36` (`qwen4_exp/__init__.py` `_hpu_tap_mtp_multi`: the final mixer's own output; the
  buffer is dropped on an HPU device). Card-verified (tiny and real model).
- **Card rule (new): inside an HPU graph a persistent tensor is never written, and never leaves the graph,
  through a slice of itself.** CPU model of the write half: `tests/unit_tests/models/flash_next/slice_write_probe.py`.
- Card, reference without speculation (same bundle, TP8, no `min_tokens`): 41.6 ms per step at 1 stream, 45.9 at
  8, 84.1 / 84.6 at 32 (85.1 ms per token effective, 346 tok/s total). `argmax -> .to(int32)` in one lazy graph
  is sound at 8 and 32 rows (the streams' texts): card rule 9 does not hit the sampler or the draft loop.
- `317c33c5`: `VLLM_HPU_MTP_TIMING=1`, the speculative step's timer (`vllm_gaudi/v1/spec_decode/mtp_step_timer.py`):
  per step the verification part and the draft with a lap per draft pass; log lines `MTP step timing: ...` and
  `mtp_step_timing_r<rank>_<pid>.jsonl` in the server's working directory, with `Ft_ms` and `Fd_ms`.

**Branch state (20:20 UTC).** Pushed: `origin/feat/flash-next-gaudi-mtp-card-ready` = `f1aa2292` (all card
fixes, the timer; `test_engine_mtp.py` 14 passed on `9d013a36`). LOCAL, NOT PUSHED (`git log
origin/feat/flash-next-gaudi-mtp-card-ready..HEAD`): `6285cfae` = the merge of the program head `1b28bbff`
(PR 12, PR 13; one conflict in `hpu_flash_next_mtp.py`: PR 13's `_draft_tokens` kept, `draft_argmax` inside it,
the row gather stays `index_select`; the stale `test_proposer_only_reaches_topk_under_the_gate` updated) and
`015003a3` = `tools/flash_next/rental/mtp_sampled_card.py` (I07's first card check). Push them when the common
gate is green on the merged tree: it runs detached on a snapshot (`bin/gate` now queues gates, mine waits behind
other tracks'), output `reports/mtp-card-ready/gate/merge-6285cfae-gate.txt`; the worker tests on the merged tree
passed (`gate/merge-6285cfae-worker.txt`; `test_hpu_model_runner.py` needs the real bridge, as before the merge).
Then: pull request to `feat/flash-next-gaudi` (the card-proven fixes are what every other track's MTP needs).
Dry-run merges onto the merged head: qsa-fused, engine-step and collectives-dev merge clean; step-anatomy
conflicts in `envs.py` and in `sample_tokens` next to the three timer hooks (keep both).
Not yet verified on a card: commit `9722b888` (`VLLM_HPU_MTP_CHUNK_KV`, default 0, inert at 0; its engine test
passes on CPU in the 14). Needed for every prompt above about two attention blocks on an MTP server with prefix
caching (the scheduler ends a prefill chunk at the last cacheable block boundary, one block earlier with MTP;
the block is 512 tokens at TP8 with MTP, 896 at TP4 + EP4 without speculation: read it from the pair's log), so
the draft layer has KV for the whole prompt. Without it acceptance at long prompts will be far below the
short-prompt numbers.

**Card jobs of this track**
- 00021 ended 19:54 UTC (exit 0). Everything is fetched: `reports/mtp-card-ready/card/mtp3-fix/` (and
  `00021-mtp3-fix/` for the early copies): `mtp3-torchverify`, `mtp3-fusedverify` (failure 6), `base-nospec`
  (reference), `run1` (tiny, failure 7), `run2` (tiny, PASSED), `run3` + `mtp3-retry` (real model, the numbers).
- 00022 is cancelled; its place in the queue went to the matched 4-card pair (TP4 + EP4, 1 / 8 / 32 streams,
  256 tokens), bumped by the coordinator to run after 00014 (it starts by 21:59 UTC at the latest):
  - **00099 `real4-nospec-ref`** (any:4, 30 min, commit `9d013a36`): the reference. Its local `farm` waiter is
    gone: when it ends, `bin/farm fetch mtp-card-ready 00099 00099-real4-nospec-ref`.
  - **00104 `real4-mtp3-timed`** (any:4, 35 min, commit `317c33c5`, `--mtp 3`, `VLLM_HPU_GDN_CORE_VERIFY=1`,
    `VLLM_HPU_MTP_TIMING=1`). A local waiter copies its results if it still runs; else `bin/farm fetch
    mtp-card-ready 00104 00104-real4-mtp3-timed`.
  What to read: `scaling.json` / `probe.log` (ms per step = `steady_ms_per_token_p50 x tokens_per_chunk_p50`,
  tokens per step = `tokens_per_chunk_p50`), the `SpecDecoding metrics` lines of `server0.log` (per-position
  acceptance per 10 s window), the `MTP step timing` lines and `mtp_step_timing_r*.jsonl` (Ft, Fd, the draft
  passes), the TEXT lines, and for engine-step the lines `Recurrent state pool: ... slots x ... MiB` and
  `Maximum concurrency` of the TP4 k = 3 server. Report to the coordinator: Ft, Fd and acceptance per position
  at 1, 8, 32 streams against the reference; then `python tools/flash_next/mtp_k_choice.py` for the best k.
- **00105 `tiny2-mtp-arms`** (any:2, 12 min, image `317c33c5`, at the end of the queue: ask the coordinator for a
  bump): `bash tools/flash_next/rental/mtp_tiny_card.sh 3`: arm `base` (the tree's defaults: the int64 index of
  the GDN copies on a card), arm `loop-chunk` (`VLLM_HPU_MTP_DRAFT_LOOP=1`, `VLLM_HPU_MTP_CHUNK_KV=1`, neither
  ever ran on a card). Once the merged head is pushed, replace it by one job from the head that also runs
  `python tools/flash_next/rental/mtp_sampled_card.py --k 3` (arms plain, refuse, sampled: I07 on the tiny
  model): `bin/farm job mtp-card-ready tiny2-mtp-all any:2 20m --replaces 00105 -- bash -c 'bash
  tools/flash_next/rental/mtp_tiny_card.sh 3; python tools/flash_next/rental/mtp_sampled_card.py --k 3'`.
- spec-lookup's 00097 (`tiny2-mtp3-lookup-oracle`): accepted counts 1 to 3 on the tiny model on a card; read it.

**Pending, in order**
1. The pair above: read, report, write the numbers here.
2. Tiny model, `any:2`, 12 minutes: `farm job mtp-card-ready tiny2-mtp-arms any:2 12m -- bash
   tools/flash_next/rental/mtp_tiny_card.sh 3` (arms: the tree's defaults with the int64 index; then
   `VLLM_HPU_MTP_DRAFT_LOOP=1` + `VLLM_HPU_MTP_CHUNK_KV=1`). spec-lookup's job 00097 runs accepted counts 1 to 3
   on the tiny model with an oracle: read its result.
3. 32 streams with the qsa-fused data flow: `git merge origin/feat/flash-next-gaudi-qsa-fused` (clean onto the
   merged head), flags `VLLM_HPU_QSA_FUSED=1`, `VLLM_HPU_QSA_INDEXER_SHARD=1`, `VLLM_HPU_QSA_TABLE_WIDTH=1` as its
   A/B pair 00062 / 00063 confirms them; a CPU engine rehearsal of k = 3 against k = 0 with those flags on a
   scratch merge was started 20:17 UTC (results in the agent's scratchpad `qsa_k3/`, `qsa_k0/`; repeat it with
   `python -m tests.unit_tests.hpu_cpu_engine --ckpt <tiny checkpoint> --out k3.json --k 3`). Then an `any:4`
   MTP arm at 32 streams with the timer. engine-step's I01 (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`, on the
   program branch after its gate) is one more `--env` on the same arm: the timer's `passN_meta` laps and `Ft_ms`
   read its effect.
4. I07 sampled verification on the card (the owner's agents sample: temperature 1.0, top_k 20, top_p 0.95).
   First the tiny model (`mtp_sampled_card.py`, above), then the real model: `VLLM_HPU_SPEC_SAMPLED_VERIFY=1`
   and sampled rounds (the farm's probe sends none: use `reports/mtp-card-ready/job/spec_scaling.py --sampled
   N:T` from a `farm job`, or add the round to a probe in the tree). Today's `=1` path is exact in distribution
   over the full vocabulary in torch (`hpu_rejection_sampler.sampled_rejection_sample`). Target form: engine-step's
   `vllm_gaudi/v1/sample/candidate_sampling.py` (its branch `9bf1920f`: `take_candidates(logits, C=64)`,
   `candidate_probs(values, temperature, top_k_index, top_p)`) or collectives' `vocab_parallel.top_k(model,
   hidden_states, C)` (branch `collectives-dev`, `ea3bc30a`) give `(values fp32 [rows, C], ids)`: p(draft) is
   the column whose id equals the draft (int32 compare; zero when the draft is no candidate), the recovery token
   comes from the same row with that column zeroed, the noise is host-made and identical on every rank.
5. I02 (the timer shows what each draft pass costs; engine-step's I01 removes the drafter's per-step
   paged-attention buffers), I37 memory (the k = 3 pool is 6.147 GiB per card at TP8; read `Maximum
   concurrency`), I43 / k = 4 and 5 (`tools/flash_next/mtp_k_choice.py`).
6. Merge proven steps per the protocol: merge `origin/feat/flash-next-gaudi` into the branch (one conflict, in
   `hpu_flash_next_mtp.py`), `bin/gate <worktree> --tools`, PR to `feat/flash-next-gaudi`.

## I07, sampled verification: what is built and where it should end

- `VLLM_HPU_SPEC_SAMPLED_VERIFY=1` (exact, `hpu_rejection_sampler.sampled_rejection_sample`): card-proven in 00126 (TP4,
  coherent sampled texts, acceptance length 2.62 at 1 stream, 2.23 at 32), but about 26 ms per step at k = 3 (13 at
  k = 1), mostly upstream's RejectionSampler / Sampler pass: the one-card cost of the rejection maths is 5.4 ms at 4
  rows and 27.6 ms at 128 rows (00135).
- `=2` (`hpu_candidate_verify.py`): the same distribution over the top-64 candidates after the logits all-gather,
  13.3 ms at 128 rows on one card (00135); skips upstream's sampler pass. Real-model run pending (00138, integration
  build 2).
- **End state (proposed to engine-step, 00:47 UTC):** for arg-max drafts (MTP), speculative sampling equals "sample
  every verification row and the bonus row from the target distribution, each with its own noise, then the greedy
  rejection on the sampled tokens". Proof: with q one-hot at d, speculative sampling accepts with p(d) and on a
  rejection draws from p without d; a draw y ~ p accepted when y = d gives P(accept) = p(d) and
  P(y = x | y != d) = p(x) / (1 - p(d)); rows are independent. So engine-step's sampled region
  (`VLLM_HPU_CANDIDATE_SAMPLING=1`, candidates inside the compiled LM-head region, vocabulary-parallel in its candvp
  form) can serve a verification step: expand the request parameters to rows (row r is request r // (1 + k) in the
  HPU's padded layout), one noise row per row, return the per-row tokens as `fused_sampled_token_ids`; the runner's
  spec branch already hands them to `greedy_rejection_sample_from_tokens` (PR 13). Sampled verification would then
  cost about what greedy verification costs. Card check needed: the first-draft acceptance against p(d).

## Card results

| Job | What | Result |
|---|---|---|
| 00008 (coord) | first MTP start, program head | died in `load_model` (`image_token_index`), fixed in `8951c003` |
| 00021 attempt 1 | TP8, k = 3, torch verification | healthy after 435 s; first prompt: failure 6 (PLE slice write) |
| 00021 attempt 2 | TP8, k = 3, fused verification, host-book draft positions, sampled verify on | healthy after 435 s; same failure 6 |
| 00021 reference | TP8, same bundle, no speculation, no `min_tokens`, 256 tokens | ms per step: 41.6 (1 stream), 45.9 (8), 84.1 / 84.6 (32); 85.1 ms per token at 32 streams = 11.7 tok/s per stream |
| 00021 run 1 | tiny model TP8, `hc` then `hc+mtp3`, failure-6 fix | `hc` passed (7.8 ms/token); both speculative rungs: failure 7 (aux hidden view of the buffer) |
| 00021 run 2 | tiny model TP8, `hc+mtp3`, both fixes | PASSED on the fused verification rung: tokens equal the non-speculative server for 1, 7, 33, 100, 700 and 2,600-token prompts; 0 of 270 drafts accepted (random weights); 22.0 ms/token against 8.2 without speculation |
| 00021 run 3 | REAL model TP8 + EP8, k = 3, fused verification, greedy, both fixes, 128 tokens | healthy after 466 s. 1 stream: acceptance per position 0.83 / 0.57 / 0.39, 2.78 tokens per step (2.72 per streamed chunk), 84.3 ms per step, 31.0 ms per token steady (55.9 with the round's 2.8 s compile pause). 32 streams: only the warm round fitted (24 tokens: acceptance length 2.13; the server's window: per-position 0.548 / 0.299 / 0.201 over the first tokens of 32 prompts) |
| 00099 `real4-nospec-ref` | TP4 + EP4, no speculation, `9d013a36`, modules 0-3, greedy, 256 tokens, torch QSA path | steady ms per step 39.8 (1 stream), 53.7 (8), 108.1 / 108.1 (32, two rounds) |
| 00104 `real4-mtp3-timed` | TP4 + EP4, k = 3, fused verify rung (`VLLM_HPU_GDN_CORE_VERIFY=1`), timer, `317c33c5`, modules 4-7, greedy, 256 tokens | ms per token 23.5 / 49.0 / 133.7 (1 / 8 / 32 streams), tokens per step 2.56 / 2.57 / 2.53, step 60 / 126 / 338 ms. Timer (means of 25 steps, 4 ranks within 1 ms): Ft 47.0-48.4 / 109.2-109.5 / 302.5-303.4 ms at 4 / 32 / 128 rows (415-438 in the first 32-stream rounds, compiles); draft 12.8-13.5 / 17.8 / 32.1-32.5 ms, so Fd 4.3-4.5 / 5.9-6.0 / 10.7-10.8 ms per pass. Acceptance per position (server windows): 0.75 / 0.55 / 0.38 at 8 streams (1,959 drafted), 0.72 / 0.51 / 0.36 at 32 (16,134 drafted), 0.64 / 0.39 / 0.23 at 1 stream (windows with warm rounds). Texts coherent; the 1-stream text leaves the reference's after 4 words (a near tie under 4 rows). Pauses up to 24.6 s in the first 32-stream round (compiles) |
| 00105 `tiny2-mtp-arms` | tiny model TP2, k = 3, image `317c33c5`; arm base, arm `VLLM_HPU_MTP_DRAFT_LOOP=1` + `VLLM_HPU_MTP_CHUNK_KV=1` | both arms PASS on the torch-verification rung (tokens equal the non-speculative server, 1 to 2,600-token prompts; 0 of 270 drafts accepted, random weights), 20.0 and 16.2 ms per token; the fused verify rung missed the 700-token prompt at token 4 in both arms (a near tie, see the handoff). First card run of DRAFT_LOOP and CHUNK_KV: no error |
| 00127 `real4-qsa-ref-nospec` | TP4 + EP4, no speculation, `13d7393a`, QSA data flow (`VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1`) + kernel-fixes flags (`VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2`), `VLLM_GRAPH_RESERVED_MEM=0.5`, 256 tokens | greedy ms per step 29.8 (1 stream), 43.2 (8), 64.5 / 63.7 (32); sampled (server default sampling) 38.7 (1), 76.2 (32); hl-smi peak 70.7 GiB per card; texts coherent |
| 00126 arm `k3` | the same build and flags with k = 3, fused verify rung (`VLLM_HPU_GDN_CORE_VERIFY=1`), `VLLM_HPU_MTP_DRAFT_LOOP=1`, `VLLM_HPU_MTP_CHUNK_KV=1`, `VLLM_HPU_SPEC_SAMPLED_VERIFY=1`, timer | greedy: 65.3 / 88.3 / 157.9 and 157.6 ms per step at 1 / 8 / 32 streams, 2.56 / 2.60 / 2.53 and 2.55 tokens per step, **25.5 / 34.0 / 62.3 and 61.9 ms per token (1.17 / 1.27 / 1.03 times the reference)**, acceptance 0.74 / 0.47 / 0.38, 0.73 / 0.52 / 0.39, 0.73 / 0.50 / 0.35. Timer: Ft 47-52 / 65 / 130-132 ms at 4 / 32 / 128 rows, Fd 4.4 / 5.7 / 6.9-8.5 ms (pass-0 wait gone with the draft loop). **Sampled (I07, exact path, first card run): works**, 91.0 ms per step at 1 stream (2.59 tokens per step, 35.2 ms per token, acceptance 0.73 / 0.53 / 0.36), 184.3 at 32 streams (2.24, 82.4 ms per token, 0.64 / 0.38 / 0.22): the exact path adds about 26 ms per step at 4 and at 128 rows; texts coherent |
| 00126 arm `k1` | the same with k = 1 | 1 stream: 50.1 ms per step, 1.75 tokens per step, 28.6 ms per token, acceptance 0.76; **32 streams greedy: 95.7 ms per step, 1.74 tokens per step, 55.2 ms per token (1.15 times the reference's 63.7; k = 3: 61.9)**, acceptance 0.75; 32 streams sampled (`=1`): 109.1 ms per step, 1.66, **65.8 ms per token (reference 76.2, k = 3: 82.4)**, acceptance 0.66. Timer: Ft 42-47 / 84-86 (sampled 98-100) ms at 2 / 64 rows, Fd 4.8-5.9 / 5.2-7.6 ms. KV cache 868,867 tokens (k = 3: 657,714). hl-smi peak 73.7 GiB per card (MTP), 70.7 without |

## How the jobs of this track run (late-bound)

`farm`'s stage-1 form: the job's `cmd.sh` runs `~/farm/mtp-card-ready/job/plan.sh` (read when the job starts),
which starts `farm-base:latest` containers with the bundle `~/farm/mtp-card-ready/next` mounted at `/bundle`; the
container script `job/serve_probe_spec.sh` copies the bundle's Python files over the image's `vllm_gaudi`
(Python-only changes since `e238f84`; a native change needs a `farm serve` image instead). Replace a file on the
node only with `scp` to `name.new` and `mv` (a running bash keeps reading the old file). The probe never sends
`min_tokens`.
