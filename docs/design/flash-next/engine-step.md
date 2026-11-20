# Flash-Next on Gaudi: host time per decode step (track `engine-step`)

What the host does in one decode step that is neither a graph op nor a collective, what of it grows with the
rows, and the fast paths that remove the largest parts: the sparse-QSA decode metadata (idea I01), the in-graph
greedy region with `min_tokens` (idea I05), and the design of a sampled step that sends one small tensor to the
host. Every fast path ships default off behind a flag.

Labels: VERIFIED (by the named test, line or card job), MEASURED (where), INFERRED.

This track's harness refuses report files outside the repository, so the handoff of the track lives here.

## Handoff (if this agent stops)

Last update: **2026-11-20 19:56 UTC**. Branch `feat/flash-next-gaudi-engine-step`, pushed (code head `383288c5`,
then documents). Scratch files, probes and card results: `/Volumes/Storage/Claude/gaudi-work/reports/engine-step/`
(`probes/`, `card_job/` = submit logs, `card/` = fetched results, `logs/` = CPU test logs).

### State in one paragraph

I01 (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`) and I05 (`VLLM_HPU_FUSED_GREEDY_LOGITS=1` with `min_tokens`) are
implemented, default off, CPU-proven bit-exact on the whole serving loop, pushed, **not merged and never run on a
card**. The first card A/B (jobs 00030 and 00031) produced no number: both were cancelled by this track's second
agent while the two servers still loaded the checkpoint. A switch file (`VLLM_HPU_RUNTIME_SWITCH_FILE`) now lets
ONE server measure both arms. Two card jobs wait: a one-card probe (ids, the sampler stage, the tiny model with
both flags) and the real-model A/B.

### Card jobs

Check the ids first: `bin/farm status | grep engine-step`, and the submit logs in `card_job/` (each ends with
`job NNNNN queued` or `replaces job ...`).

| Job | What | State | To do when it ends |
|---|---|---|---|
| 00030, 00031 | first A/B pair, two TP4 + EP4 servers at once | cancelled by engine-step at 18:41:20 and 18:42:56 UTC during the checkpoint load; no numbers (section 2) | nothing |
| **00093** `hpu-ids-sampler-tiny-flags` (any:1, 14 min; replaced 00085 <- 00077 <- 00072, place of 19:25:12 UTC). The coordinator bumped 00077 into the single-card wave expected between 20:15 and 21:25 UTC and was asked at 19:54 UTC to bump 00093 instead (a replacement loses the bump). | `tools/flash_next/engine_step/hpu_sampling_ops.py` at `383288c5`: `VERDICT` lines first (arg max and top-k ids narrowed in the same graph, in a second graph, read as int64; the stop-id mask; the block-table gather; the sampled paths), then the times of the sampler stage at 1, 8, 16, 32 rows (greedy, vLLM's sampled path and its parts, sampling over 64 candidates), vLLM's `index_put_` mask, the bridge's fallback lines; then the tiny model at TP = 1 card with both flags on against the CPU reference | waiting | **send the `VERDICT` lines to the coordinator (`main`), mtp-card-ready and collectives: three tracks wait for them**; write the lines and the sampler table into section 9; if the tiny model fails with the flags and passes without, I01 or I05 breaks on a card: read `tiny-flags/hc/server0.log` before the A/B runs |
| **00094** `real4-i01-i05-ab2` (any:4, 36 min; replaced 00061, place of 19:07:58 UTC) | one TP4 + EP4 server, `serve_long_probe.sh`: arms base / both / i01 / i05 through the switch file, 32 streams at a short context and over a shared 120,832-token prefix | waiting | `card/00094-real4-i01-i05-ab2/probe.log`: the `SUMMARY` block (steady ms per token per arm, each arm against base), the `TEXTS text1` line (must be 1 of 1), `TASK` and `SAME` lines, `SWITCH` lines (worker lines must be above 0), `NOTE` lines; write them into section 9 and tell the coordinator |

After any job: `bin/farm ledger | grep -w engine-step`, then `bin/farm fetch engine-step <id> <id>-<name>` if
`reports/engine-step/card/` lacks it. The node ends about 04:00 UTC on 2026-11-21 and loses everything; a job not
started by about 03:15 UTC will not finish.

### Pending subtasks, in order

1. **The one-server A/B of I01 + I05.** Built, pushed (`383288c5`) and queued as 00094. To submit it again
   (in its place while it waits: add `--replaces 00094`):
   `bin/farm job engine-step real4-i01-i05-ab3 any:4 36m -- bash tools/flash_next/engine_step/serve_long_probe.sh`
2. **Merge I01, I05 and the switch file** once the card shows equal one-stream texts and no slowdown:
   `git fetch origin && git merge origin/feat/flash-next-gaudi`, `bin/gate /Volumes/Storage/Claude/gaudi-work/wt/engine-step --tools`,
   push, `gh pr create --base feat/flash-next-gaudi`, `gh pr merge --merge`. The flags stay default 0.
3. **The sampled step** (coordinator, 19:45 UTC: real agents send no temperature and get temperature 1.0,
   top_k 20, top_p 0.95; I05 does not help them). Section 7 has what the code does today and the design; the
   one-card probe measures both forms. Next: `vllm_gaudi/v1/sample/candidate_sampling.py` (pure functions,
   CPU tests of the kept set and of the distribution against vLLM's `apply_top_k_top_p_pytorch` +
   `random_sample`), then the region `select_compute_logits_and_sample` in the runner behind a flag. Interfaces
   for collectives (vocabulary-parallel top-k) and mtp-card-ready (sampled verification) are in section 7.
4. **I01 under MTP k = 3** (its largest predicted effect, about 29 ms of host time per step at 32 rows x 120K):
   only after mtp-card-ready shows MTP runs on the card; then the same A/B with a speculative config. The
   drafter's metadata path is CPU-proven with oracle drafts only.
5. **The replay call grows from 23 ms at 1 stream to 35 ms at 32** (step-anatomy's split): what in the runner
   makes the host wait inside it. Split the work with native-replay (its design document
   `docs/design/flash-next/native-replay.md` on its branch). Not started.
6. The brief's later items, none started: device tensors kept in place (one packed copy instead of 11 to 14 per
   step), async scheduling with speculation (the PLE layer needs the sampled tokens: keep that context on the
   device), the worker round trip, CPU pinning.
7. **Functional fix, after the speed work** (coordinator's decision): the general sampler path on HPU ignores
   `min_tokens`, `logit_bias` and `min_p` (section 3). The test
   `test_general_sampler_path_tracks_no_min_tokens_request` pins the defect and must be replaced by a parity
   test of both paths when it is fixed.

### How to run the CPU tests

From the worktree root:
`PYTHONPATH="$PWD" /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python -m pytest -q -p no:cacheprovider tests/unit_tests/models/flash_next/test_engine_loop.py tests/unit_tests/models/flash_next/test_engine_step_budget.py tests/unit_tests/models/flash_next/test_decode_metadata_sparse.py tests/unit_tests/models/flash_next/test_decode_metadata_graphs.py tests/unit_tests/models/flash_next/test_fused_greedy_min_tokens.py tests/unit_tests/models/flash_next/test_runtime_switches.py`
(34 passed at `6697fe04`; `test_fused_greedy_min_tokens.py` 5 passed at `3c640728`; `test_runtime_switches.py`
3 passed on the tree of `383288c5`). With ten tracks testing on the same Mac a file takes 10 to 20 minutes.
The probes check themselves without a card: `python3 tools/flash_next/engine_step/hpu_sampling_ops.py --selftest --out /tmp/x`.

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

Card rule 9 (`torch.topk` ids narrowed to int32 in the graph that made them came back wrong): since `3c640728`
the region has no integer cast of its own. The stop ids are made on the host as int64 and used as they are;
without async scheduling the arg max ids leave the graph as int64 and the host narrows them. For arg max the
narrowing in one graph is sound in vLLM's sampler at 8 and 32 rows (MEASURED by text, mtp-card-ready job 00021);
the one-card probe checks every pattern against the CPU. The collectives track leaves
`select_compute_logits_and_greedy` to this track (coordinator, 19:17 UTC).

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

**Design: sample over candidates** (to build; the one-card probe times it as `sampled_candidates*`):
1. Candidates: `values, ids = logits.topk(C)` per row, sorted, C = 64 (at least the largest `top_k` of the
   batch). With the collectives track's vocabulary-parallel head each rank takes the top C of its shard and the
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
- collectives (vocabulary-parallel top-k): the sampler stage takes `(values fp32 [rows, C], ids int64 [rows, C])`,
  sorted by value, largest first, global token ids, C fixed per server. Whether they come from `logits.topk(C)`
  or from per-shard top-k and a small gather is the head's matter.
- mtp-card-ready (sampled verification, idea I07): one function `candidate_probs(values, temperature, top_k,
  top_p) -> probs [rows, C]` (zero outside the kept set) serves both the pick (`argmax(probs / noise)`) and the
  rejection sampler (`p(draft token)` = the probability at the column whose id equals the draft id, zero when
  the draft is no candidate; the recovery token from the same row with that column zeroed), so target
  probabilities agree between a plain step and a verification step.

## 8. For other tracks

- qsa-fused: `qsa_sparse_ops.py:116-117` (`select_blocks`) narrows `torch.topk` ids in the graph that made them
  (the rule-9 pattern) on the served sparse path (in BRIEF.md since 19:15 UTC).

## 9. Card results

None yet (the one-card probe and the A/B wait).
