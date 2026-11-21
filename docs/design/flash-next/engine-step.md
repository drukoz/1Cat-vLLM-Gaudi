# Flash-Next on Gaudi: host time per decode step (track `engine-step`)

What the host does in one decode step that is neither a graph op nor a collective, what of it grows with the
rows, and the fast paths that remove the largest parts: the sparse-QSA decode metadata (idea I01), the in-graph
greedy region with `min_tokens` (idea I05), and the design of a sampled step that sends one small tensor to the
host. Every fast path ships default off behind a flag.

Labels: VERIFIED (by the named test, line or card job), MEASURED (where), INFERRED.

This track's harness refuses report files outside the repository, so the handoff of the track lives here.

## Handoff (if this agent stops)

Last update: **2026-11-21 00:45 UTC** (fourth agent, Opus 5.5, stopping). Branch `feat/flash-next-gaudi-engine-step`,
head `89cd8692` plus this document (program head `938dada4` merged in at `eaf2131f`), everything pushed. Scratch
files, probes and card results: `/Volumes/Storage/Claude/gaudi-work/reports/engine-step/` (`card_job/` = submit logs,
`card/` = fetched results, `logs/` = CPU test logs, `probes/` = Mac probes). Node mirror:
`node-mirror/farm/engine-step/results/`.

### State in one paragraph

Card-proven (section 9): **`VLLM_HPU_CANDIDATE_SAMPLING=1`**: -4.9 ms per sampled step at 32 streams on TP4
(00110: 114.6 -> 109.7 ms; every stream on task; a seeded request repeats itself; 00093: the sampler stage 3.18
against 7.46 ms at 32 rows on one card); integration builds with it. **I01 (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`)
is token-correct on a card** (00130, tiny model, 16 of 16) and neutral at a short context (00110); its host saving
at a long context is unmeasured on a card (needs prefill's prefix-cache fix, see the next rental's jobs). I05
(`VLLM_HPU_FUSED_GREEDY_LOGITS=1`) gave no measurable gain and resolves one near tie of the random-weight tiny model
the other way (00130; not a defect). **`logprobs=5` no longer kills the server** (top-k ids out of an fp32 id table,
`v1/sample/topk_ids.py`, `330f9413`; card-proven in 00130). **`prompt_logprobs` killed the engine** (an older
defect: Flash-Next's flattened prefill hidden states were indexed per request): fixed at `e9032e68`, CPU-proven (the
test reproduced the crash), not yet on a card. The sampled region now takes `vocab_parallel.top_k` (int32 ids) and
logs once per server which path a sampled step took (`21d9eace`). `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2` with candidates
(`candvp`) has not run on a card: 00144 waits but gets no slot tonight. No pull request tonight (BRIEF.md 23:45).

### If you take over, in this order

1. After 04:00 UTC (node gone, Mac free): `bin/gate /Volumes/Storage/Claude/gaudi-work/wt/engine-step --tools`
   (merge `origin/feat/flash-next-gaudi` into the branch first if it moved), then one pull request into
   `feat/flash-next-gaudi` (`gh pr create --base feat/flash-next-gaudi`, `gh pr merge --merge`), all flags default
   off. CPU tests green at `e9032e68`: `logs/t_fast_eaf.log` (18), `logs/t_logline.log` (22), `logs/t_plp_after.log`
   (9), `logs/t_loop_plp.log` (18), `logs/t_track_3757.log` (29), `logs/t_platform_vp_3757.log` (73).
2. A stale gate of this track (pid 25332, started 20:10 UTC on `reports/engine-step/merge-wt` = the old merged tree
   `f0fd0a2b`) and a `bin/farm job` waiter for 00144 (pid 28597) are still running on this Mac; stopping them was
   refused to this agent by the permission check (the owner was told). Both are harmless.
3. The next rental: the two jobs under "Pending subtasks" item 0 (I01 at 32 x 120K with prefill's two prefix flags;
   `candvp` at TP8), then 00144's content at TP4 if TP8 is not free.

### Card jobs

| Job | What | State | To do when it ends |
|---|---|---|---|
| 00093 `hpu-ids-sampler-tiny-flags` (1 card) | id verdicts, sampler-stage times, tiny model with I01 + I05 | ended 20:29 UTC, exit 1 (the tiny 700-token prompt, a near tie) | done: section 9 |
| 00110 `real4-ab3-greedy-sampled-long` (TP4) | greedy base / I01 / I05 / both, sampled base / cand, long prefix | ended 22:29 UTC, exit 5 (probe counting artifact; no prefix-cache hit) | done: section 9 |
| 00130 `tiny-i01-i05-logprobs` (1 card, commit `0cb55ebb`) | `tiny_checks.py --arms i01,i05` | ended 00:25 UTC, exit 1 (the I05 arm's near tie, and both servers died on the `prompt_logprobs` request) | done: section 9; the prompt-logprobs defect fixed at `e9032e68` |
| **00144** `real4-ab6-short-candvp` (any:4, 28m, commit `e9032e68`, `VLLM_GRAPH_RESERVED_MEM=0.5`; replaced 00131 (23:55 UTC) and 00137 (00:36 UTC) to carry the log lines and the prompt-logprobs fix, kept the place) | `serve_long_probe.sh --arms both --no-long --candvp --logprobs-check`: 32 greedy streams base / both (twice, in turn), text1 compared at 1 stream, 32 sampled streams base / cand / candvp (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=2`), seeded requests, min_tokens round, the logprobs check last; hl-smi memory in `hlsmi.log` | waiting; the owner's 8-card runs go first (coordinator, 00:28 UTC): probably not tonight, then it is the next rental's first TP4 job | `probe.log`: `SUMMARY` (sampled cand and candvp against base, candvp against cand), `TASK` for candvp (must be 32 of 32), `TEXTS text1`, `LOGPROBS ok`; the job log's `Candidate sampling` lines ("drew in the logits region ... from the gathered logits" for cand, "... from each rank's vocabulary shard" for candvp); then tell the coordinator whether `VLLM_HPU_VOCAB_PARALLEL_LOGITS=2` with candidates is proven |

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
5. **Functional fix, after the speed work**: the general sampler path on HPU ignores `min_tokens`, `logit_bias` and
   `min_p` (section 3); `test_general_sampler_path_tracks_no_min_tokens_request` pins it.
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
