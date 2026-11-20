# Flash-Next on Gaudi: host time per decode step (track `engine-step`)

What the host does in one decode step that is neither a graph op nor a collective, what of it grows with the
rows, and the two fast paths that remove the largest parts: the sparse-QSA decode metadata (idea I01) and the
in-graph greedy region with `min_tokens` (idea I05). Both ship default off behind a flag.

Labels: VERIFIED (by the named test, line or card job), MEASURED (where), INFERRED.

This track's harness refuses report files outside the repository, so the handoff of the track lives here.

## Handoff (if this agent stops)

Last update: **2026-11-20 19:35 UTC**. Branch `feat/flash-next-gaudi-engine-step`, pushed. Scratch files, probes
and card results: `/Volumes/Storage/Claude/gaudi-work/reports/engine-step/` (`probes/`, `card_job/`, `card/`,
`logs/`).

### State in one paragraph

I01 (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`) and I05 (`VLLM_HPU_FUSED_GREEDY_LOGITS=1` with `min_tokens`) are
implemented, default off, CPU-proven bit-exact on the whole serving loop, pushed, **not merged and never run on a
card**. The first card A/B (jobs 00030 and 00031) produced no number: both were cancelled by this track's
second agent while the two servers still loaded the checkpoint. Two card jobs wait in the queue: the one-card
id probe (00072) and the real-model A/B (00061, a placeholder to replace).

### Card jobs

| Job | What | State | To do when it ends |
|---|---|---|---|
| 00030 `real4-base-longctx`, 00031 `real4-i01-i05-longctx` | first A/B pair, two TP4 + EP4 servers at once | **cancelled by engine-step** at 18:41:20 and 18:42:56 UTC during the checkpoint load (00030: 51 of 131 shards after 18.5 min; 00031: 88 of 131 after 20 min; no `SERVER-HEALTHY`, no `probe.json`) | nothing: no numbers exist |
| **00072** `hpu-sampling-ops-rule9` (any:1, 10 min) | `tools/flash_next/engine_step/hpu_sampling_ops.py` at commit `3c640728`: card rule 9 for arg max and top-k ids, the stop-id mask, the block-table gather, each against the CPU at 1, 2, 8, 16, 32 rows; times of the sampler stage | waiting (submitted 19:25 UTC) | read the table at the end of `card/00072-hpu-sampling-ops-rule9.log` and `card/00072-*/sampling_ops.json`; **send the `ok` / `WRONG` lines of `sampler`, `sampler_int64`, `sampler_cast_second_graph`, `topk_*`, `fused_mask_*`, `table_*` to the coordinator (`main`), mtp-card-ready and collectives: three tracks wait for them**; write them into section 6 |
| **00061** `real4-i01-i05-ab` (any:4, 40 min) | **placeholder** at commit `6697fe04` (both flags on, the old probe that sends `min_tokens`); it holds the queue place of 19:07 UTC | waiting | **replace it** (`--replaces 00061`) with the one-server A/B of subtask 1; if it starts before that, its numbers are one arm only |

After any job: `bin/farm ledger | grep -w engine-step`, then `bin/farm fetch engine-step <id> <id>-<name>` if
`reports/engine-step/card/` lacks it. The node ends about 04:00 UTC on 2026-11-21 and loses everything; a job not
started by about 03:15 UTC will not finish.

### Pending subtasks, in order

1. **One-server A/B of I01 + I05** (replaces 00061). A switch file that a running server rereads
   (`VLLM_HPU_RUNTIME_SWITCH_FILE`, the two per-step flags only), so one TP4 + EP4 server measures base and
   flag-on rounds in turn: half the card time of two servers and no server-to-server noise. Probe without
   `min_tokens` (`max_tokens` with `ignore_eos`), the texts of both arms compared, 32 streams at a short context
   and over a shared 120K-token prefix. Files: `vllm_gaudi/v1/worker/runtime_switches.py` (new),
   `hpu_model_runner.py` (poll at the start of `execute_model`), `tools/flash_next/engine_step/
   long_context_probe.py`, `serve_long_probe.sh`. Submit:
   `bin/farm job engine-step real4-i01-i05-ab2 any:4 35m --replaces 00061 -- bash tools/flash_next/engine_step/serve_long_probe.sh --prefix 120000`
2. **Merge I01 and I05** once the card shows equal texts and no slowdown: `git fetch origin && git merge
   origin/feat/flash-next-gaudi`, `bin/gate /Volumes/Storage/Claude/gaudi-work/wt/engine-step --tools`, push,
   `gh pr create --base feat/flash-next-gaudi`, `gh pr merge --merge`. The flags stay default 0.
3. **I01 under MTP k = 3** (its largest predicted effect, about 29 ms of host time per step at 32 rows x 120K):
   only after mtp-card-ready's job 00021 shows MTP runs on the card; then the same A/B with
   `--speculative-config`. The drafter's metadata path is CPU-proven with oracle drafts only.
4. **The replay call grows from 23 ms at 1 stream to 35 ms at 32** (step-anatomy's split, coordinator 19:17
   UTC): what in the runner makes the host wait inside it. Split the work with native-replay (its design
   document `docs/design/flash-next/native-replay.md` on its branch).
5. The brief's later items, none started: device tensors kept in place, device-side sampling, async scheduling
   with speculation (the PLE layer needs the sampled tokens: keep that context on the device), the worker round
   trip, CPU pinning.
6. **Functional fix, after the speed work** (coordinator's decision): the general sampler path on HPU ignores
   `min_tokens`, `logit_bias` and `min_p` (section 3). The test
   `test_general_sampler_path_tracks_no_min_tokens_request` pins the defect and must be replaced by a parity
   test of both paths when it is fixed.

### How to run the CPU tests

From the worktree root:
`PYTHONPATH="$PWD" /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python -m pytest -q -p no:cacheprovider tests/unit_tests/models/flash_next/test_engine_loop.py tests/unit_tests/models/flash_next/test_engine_step_budget.py tests/unit_tests/models/flash_next/test_decode_metadata_sparse.py tests/unit_tests/models/flash_next/test_decode_metadata_graphs.py tests/unit_tests/models/flash_next/test_fused_greedy_min_tokens.py`
(34 passed at `6697fe04` in 5 min 37 s; `test_fused_greedy_min_tokens.py` 5 passed at `3c640728`). With ten tracks
testing on the same Mac a file can take 10 to 20 minutes.

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

## 3. The sampling stage and the host path of a step

1. **The min-tokens mask does not run on HPU** (VERIFIED: code and CPU test).
   `HpuInputBatch.refresh_sampling_metadata` (`vllm_gaudi/v1/worker/hpu_input_batch.py:554-562`) never calls the
   logits processors' `update_state`, so `MinTokensLogitsProcessor.min_toks` stays empty and `apply()` returns the
   logits untouched (vLLM `v1/sample/logits_processor/builtin.py:277-286`): no `index_put_`, no fallback, no host
   round trip. Test: `test_fused_greedy_min_tokens.py::test_general_sampler_path_tracks_no_min_tokens_request`.
   A `min_tokens = max_tokens` request still has the exact length, because the scheduler's `check_stop` honours
   `min_tokens` (vLLM `v1/core/sched/utils.py:122`); a stop token before `min_tokens` is emitted as a token. For
   the same reason `logit_bias` and `min_p` are ignored on the general sampler path. A defect, not a cost.
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
4. So the growth with the rows at a short context is not the runner's Python (INFERRED here; step-anatomy's card
   split agrees: host Python outside the replay call 3.3 ms at 1 stream and 5.4 ms at 32, the replay call 23 and
   35 ms, the device 12 and 43 ms).

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
off).

## 5. I05: in-graph greedy with `min_tokens` (`VLLM_HPU_FUSED_GREEDY_LOGITS=1`)

The fused region computes the LM head and the arg max in one lazy graph, without the sampler's own graph, its
two `mark_step` calls and the per-step sampling metadata. A request with `min_tokens` used to keep the general
sampler path; now its stop ids get `-inf` before the arg max through a fixed-shape gather, where and scatter
(`vllm_gaudi/v1/sample/fused_greedy.py`, up to 8 stop ids per row).

Card rule 9 (`torch.topk` ids narrowed to int32 in the graph that made them came back wrong): since `3c640728`
the region has no integer cast of its own. The stop ids are made on the host as int64 and used as they are;
without async scheduling the arg max ids leave the graph as int64 and the host narrows them. The same
`argmax -> .to(torch.int32)` in one lazy graph runs in every decode step today (vLLM's sampler, section 3), so
it is unlikely to be wrong for arg max (INFERRED); job 00072 checks it on a card. The collectives track leaves
`select_compute_logits_and_greedy` to this track.

VERIFIED on the CPU (`test_fused_greedy_min_tokens.py`): the mask equals vLLM's min-tokens rule; every decode step
of a `min_tokens` request runs in the fused region; tokens equal the HF reference; a stop token the model would
pick before `min_tokens` is replaced by the arg max of the masked logits.

## 6. Card results

None yet (jobs 00072 and 00061 wait).
