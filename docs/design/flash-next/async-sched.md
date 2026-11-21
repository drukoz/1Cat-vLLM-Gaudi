# Async scheduling for Flash-Next on HPU (track `async-sched`)

Branch `feat/flash-next-gaudi-async-sched` (from program head `5c1e8219`), worktree `wt/async-sched`. The
track's handoff lives in this file (the agent harness refuses `reports/async-sched/REPORT.md`); card logs and
results are under `/Volumes/Storage/Claude/gaudi-work/reports/async-sched/card/`.

## Handoff (if this agent stops)

Updated 2026-11-21 01:45 UTC.

- **On the branch** (pushed): `8f4816f8` engine-step's `engine_harness.py`, unchanged (same blob as engine-step and
  the integration build); `7a818007` the flag `VLLM_HPU_FLASH_NEXT_ASYNC_SCHED` (default 0) with the PLE context
  carry, the output-token ledger and the async scheduler's split fix (section 2.2). CPU: 27 + 7 + 4 tests of this
  track pass (`tests/unit_tests/worker/test_flash_next_async.py`,
  `tests/unit_tests/models/flash_next/test_engine_loop_async.py`, `test_engine_async_sampling.py`); the
  neighbours (host n-gram ids, platform, inputs, engine-step's sync loop) pass. No `bin/gate` run yet.
- **Card job 00171** `real8-async-ab` (all, 45 min hold), queued 01:37 UTC, waiting for a place from the
  coordinator. It runs job branch `feat/flash-next-gaudi-async-sched-int` (`0dcc7ddb` = integration `2af8a8fe` +
  `7a818007` + job arms; never merge it): `headline_job.sh tiny-async async nospec nospec-devids` with
  `HEADLINE_LEVELS=32,1`. A background `bin/farm job` (pid 72620 on this Mac, log
  `reports/async-sched/logs/farm_real8_async_ab.out`) waits and fetches into
  `reports/async-sched/card/00171-real8-async-ab/`. If the session is gone: `bin/farm fetch async-sched 00171
  00171-real8-async-ab`, then read `headline.md` (arms async against nospec at 32 and 1 streams, greedy and
  sampled), `tiny-async/summary.json` (must be PASSED), and grep `async/server0.log` for
  `Async scheduling on for Flash-Next` (the flag took) and `Traceback|Error`.
- **Next** (coordinator 01:40 UTC: MTP first): MTP under async scheduling, stage 1 (section 2.4): the drafts stay on
  the device, the accepted counts keep their one host read per step, the worker corrects the scheduler's optimistic
  `num_computed_tokens`. Then the pull request of `7a818007` (after `bin/gate`), and tell the coordinator the flag
  and the card result for the integration track.
- Messages sent: engine-step (runner lines I touch; its candidate sampling refusal under async is not needed on
  CPU, please drop it), coordinator (status, job 00171, MTP plan).

Labels: **MEASURED** (card, with the file), **VERIFIED** (code or CPU test, with the line or test), **INFERRED**.

## 1. The size of the prize: host time per step outside the model call

Source: step-anatomy's step timer, rank 0, ms p50 per step, program-head flags (fused GDN decode, fused
hyper-connections, host n-gram ids, sparse QSA mode 1, recurrent pool, no speculation, 30-token prompts).
"Outside" = period - forward (the host replay call) - wait (host blocked on the device at the sampled-token read).
"Worker Python" = rpc_gap + execute_model + setup + prepare + launch_inputs + logits + launch_model + sample +
launch_sample + post (the worker's host time outside the replay call); "engine turn" = the worker's `gap`
(the engine core's scheduler, `update_from_output` and the RPC between two steps).

TP8 + EP8, job 00117 (`reports/step-anatomy/card/00117-real8-timer-components2/s-timer/timing_summary.md`), MEASURED:

| streams (round) | period | forward (host) | wait (device) | outside | engine turn | worker Python | of it prepare |
|---|---|---|---|---|---|---|---|
| 1 (`1/N=1/perf`, steady) | 36.66 | 22.89 | 10.04 | **3.73** | 0.57 | 3.21 | 1.64 |
| 1 (`1/N=1`) | 43.17 | 35.94 | 0.56 | 6.67 | 0.66 | 3.71 | 1.93 |
| 8 (`1/N=8`) | 46.33 | 38.56 | 2.81 | **4.96** | 1.00 | 4.37 | 2.33 |
| 32 (`1/N=32`) | 83.20 | 37.74 | 38.47 | **6.99** | 2.03 | 4.83 | 2.59 |

TP4 + EP4, job 00113 (`reports/step-anatomy/card/00113-real4-timer-ctx-trace2/s-timer/timing_summary.md`), MEASURED:

| streams | period | forward (host) | wait (device) | outside | engine turn | worker Python |
|---|---|---|---|---|---|---|
| 1 | 35.38 | 25.38 | 6.65 | 3.35 | 0.54 | 3.12 |
| 8 | 55.69 | 34.36 | 16.89 | 4.44 | 0.89 | 3.36 |
| 32 | 106.79 | 35.09 | 66.17 | 5.53 | 1.99 | 3.93 |

00113's device trace (TP4, 32 streams) shows 7.3 to 7.8 ms of device idle at the step boundary (step-anatomy
REPORT section 0c; the profiler adds host time).

What async scheduling can hide (INFERRED from the tables; the card decides):

- With async scheduling the engine schedules step N+1 while step N runs, and the worker returns from step N
  without waiting for the device (the sampled tokens go to the host on a side stream), so it prepares and issues
  step N+1 while the device still executes step N. The steady period becomes max(host time per step, device time
  per step) instead of their sum minus the overlap inside the replay call.
- **32 streams (device-bound): about 6 to 7 ms per TP8 step** (the boundary idle: outside 6.99 minus the logits and
  sampler issue that already overlaps the device, 1.1 ms). That is 8 % of the head's 83 ms and **about 13 % of the
  integration build's 51.1 ms** (00128 greedy, 32 streams), if its outside time is the head's (its timer split was
  not recorded: INFERRED).
- **8 streams: about 3 ms** (host per step = worker Python 4.4 + replay call 38.6 = 43 ms against a 46.3 ms period).
- **1 stream (host-bound): about 1 ms** (the engine turn, 0.57 ms, and the slowest rank's short wait). The worker's
  own Python (3.2 ms) runs on the same thread as its replay call and stays serial.
- Under MTP the prize is the same in kind (host work of the verification step, the drafts and the engine turn), but
  async scheduling with speculation needs the accepted counts on the device (section 2): later.

## 2. Design

### 2.1 What async scheduling is at the pin (VERIFIED from the code)

- Engine: `EngineCore.step_with_batch_queue` with `max_concurrent_batches = 2` (`vllm/config/vllm.py:567`):
  schedule step N+1 and submit it, then block on step N-1's output. The scheduler is `AsyncScheduler` (the plugin
  rebinds it to `HPUAsyncScheduler`, `vllm_gaudi/utils.py:288`): it counts the in-flight tokens as
  `num_output_placeholders` and caches blocks only up to the known tokens.
- Worker (`multiproc_executor.py:1001`): an `AsyncModelRunnerOutput` goes to a side thread, which calls
  `get_output()` (waits for the host copy) and sends it; the worker's main thread takes the next RPC at once.
- HPU runner (upstream vllm-gaudi plumbing, `hpu_model_runner.py`): `AsyncHPUModelRunnerOutput` copies the
  sampled ids on `async_output_copy_stream` after `wait_stream(default)`; `prev_sampled_token_ids` +
  `prev_req_id_to_index` keep the step's samples on the device and `_prepare_input_ids` writes them into the next
  decode batch's input ids; partial prefill chunks sample a discarded token (`invalid_req_indices`); a step with
  prefills of a hybrid model returns synchronously (`should_synchronize_hybrid_prefill_output`). The host's
  `token_ids_cpu` and `output_token_ids` are never told the sampled tokens.
- Platform (`platform.py`): vLLM resolves `async_scheduling` (default on) before `check_and_update_config`;
  the HPU hook turns it off for any speculative config and, until this track, for Flash-Next.

### 2.2 What Flash-Next needed, and what the branch does

| Host dependency on sampled tokens | Effect under async scheduling (before) | On the branch |
|---|---|---|
| PLE n-gram context: `build_ple_inputs` reads the two tokens before each row's input from `token_ids_cpu` | wrong ids from a row's second decode step on: every reference prompt wrong from its 3rd token (VERIFIED, `test_without_the_carry_the_tokens_go_wrong`) | `PleContextCarry`: the next context of every decode row is kept on the device, `next = (context[1:], input token)`; rows that decoded in the previous step take it (one `cat`, or one `index_select` when rows moved), every other row keeps the host's (complete for it) |
| `output_token_ids` (penalties and bad words read the values; min_tokens masks of the engine-step branch read the length) | stayed empty: penalties saw no output, min_tokens never reached | `AsyncOutputLedger`: one placeholder per sampled token (vLLM's GPU runner does the same), settled from the step's host copy two steps later (no wait) or before a sampler pass that reads values (waits only for the step in flight, after the next forward was issued) |
| `HPUAsyncScheduler._mamba_block_aligned_split` aligned prompt chunks to `mamba_chunk_size` (written for the 27B) | Flash-Next prompts were not split at the recurrent block: no recurrent state cached at block ends, prefix hits 0 (VERIFIED: 700 tokens gave one chunk instead of 512 + 188) | Flash-Next keeps the base `Scheduler` rule, the one its synchronous server runs (so prefill's `VLLM_HPU_PREFIX_ALIGN_SPLIT` wrapper applies too) |
| `VLLM_HPU_PLE_HOST_NGRAM_IDS=1` hashes the input ids on the host | already falls back to the in-graph ids under async scheduling (logged once) | unchanged: the in-graph id ops (about 318 compute ops of the PLE layer, INFERRED 0.5 to 1 ms of device time per step) come back. A later option: hash the context part on the host, only the input token's part on the device |
| Logprobs | the step's logprobs are read on the host inside the step (correct, no overlap for that step) | unchanged |
| A seeded request with a prompt of several chunks | each unfinished chunk's discarded sample takes one draw from the request's generator (vLLM's GPU runner rewinds it) | documented; the draws stay valid samples (VERIFIED equal to sync for one-chunk prompts) |
| Recurrent state, QSA side caches, block tables | positions and slots come from `num_computed_tokens` (the scheduler's optimistic count) and the device runs the steps in order | nothing needed (VERIFIED by the engine-loop tests: chunked prompts, batches that shrink and grow, a prefix hit) |

The flag only decides whether the platform keeps async scheduling for Flash-Next; the runner and scheduler parts
are active whenever async scheduling is on for a Flash-Next model (it is wrong without them).

### 2.3 What only a card can show

- Whether the side-stream copy of the sampled ids (`.to('cpu', non_blocking=True)` under
  `torch.hpu.stream(...)`) returns before the device finishes in lazy mode with HPU-graph replay. If it blocks,
  the worker cannot run ahead: only the engine turn (0.6 to 2 ms) hides. vllm-gaudi's own CI runs granite with
  async scheduling (`tests/full_tests/ci_e2e_discoverable_tests.sh:294`), so it at least works; its overlap on
  this model is unmeasured.
- The per-step cost of the in-graph n-gram ids against the host ids (device time, about 0.5 to 1 ms INFERRED).
- Steps with a prefill (an agent's new turn) synchronize (hybrid rule above): no overlap on those steps.

### 2.4 Speculation (MTP) under async scheduling: stage 1 on the branch (`VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2`)

The coordinator's priority (01:40 UTC): the 32-agent plan is MTP (k = 2 or 3 at TP8), so async scheduling without
speculation only helps a no-MTP server.

What an MTP step does today (VERIFIED from the code, integration tree; the timer of 00128 for the sizes): the
runner verifies 1 + k rows per request, reads the verification result on the host (`parse_output`), updates its
books (accepted counts, the GDN lattice, align-mode commits), runs the drafter, returns. The engine then reads the
drafts back (`take_draft_token_ids`: a host read that waits for the draft passes, `final_wait` 4.5 / 6.5 / 6.0 ms
at 1 / 8 / 32 streams), schedules the next step with them and sends it.

Upstream's async speculation (GPU) keeps the drafts and the accepted counts on the device and corrects positions
there. Stage 1 keeps the host read of the verification result that the HPU step has anyway and drops the rest:

- Platform: mode 2 keeps vLLM's async scheduling with a Flash-Next MTP config (vLLM itself keeps it on for MTP;
  the HPU hook turned it off).
- Runner (`SpecDraftHandoff`): with async scheduling and a speculative config the runner keeps its synchronous
  step (`use_async_scheduling` False inside the runner). The engine schedules step N+1 during step N: its draft
  slots arrive as -1 placeholders and its `num_computed_tokens` counts every draft of step N as accepted. At the end
  of step N the handoff keeps the drafter's output (on the device, never read back) and, per verification row,
  the scheduler's optimistic count and the true one (start + committed tokens). At the start of step N+1 it gives
  every verification row whose count is still the optimistic one the true count (a count that is neither is left:
  preemption, resume); in the decode batch it writes step N's drafts into the draft slots (one `index_select` +
  `index_copy`, out of place) before the verification metadata reads them from the input ids.
- The host n-gram ids cannot hash drafts only the device has: the in-graph ids run (logged once).
- The runner's output copies `req_id_to_index` (the engine reads it after the next step's state update; the
  live dict gave a `KeyError` in the CPU loop).

What it saves (INFERRED from 00128's MTP timer, TP8): `final_wait` (about 6 ms at 32 streams) and the engine turn
(about 2 ms) per step, and the next step's input preparation overlaps the draft passes on the device: about 8 ms of
a 106 ms MTP step at 32 streams (8 %), 5 to 6 ms of 66 ms at 1 stream. What stays: the host read of the
verification result inside the step and the drafter's host work after it (the device idles for the parse, the
books and the first draft pass's issue, about 4 to 5 ms). Stage 2 would move the accepted counts to the device
(lattice offsets, positions, slot mapping of the next verification, align commits): 2 to 4 days, only if stage 1's
card number shows the remaining in-step gap is worth it.

CPU evidence (engine-step's harness on this branch; n-gram as the vehicle because the harness loads no drafter
model, with oracle drafts returned as a tensor the way the MTP drafter returns them, one wrong draft every few
steps): `test_engine_async_spec.py`: the async run gives the reference tokens, corrects counts and writes drafts;
without the correction the tokens go wrong; without the scatter they stay right with fewer accepted drafts. The real
MTP drafter under async runs in mtp-card-ready's rehearsal engine (`tests/unit_tests/hpu_cpu_engine.py`, on the
integration tree only), checked on the job branch before the card.

## 3. What is on the branch

See the handoff. Files: `vllm_gaudi/v1/worker/flash_next_async.py` (carry, ledger), `vllm_gaudi/platform.py`
(`resolve_flash_next_async_scheduling`), `vllm_gaudi/v1/core/sched/hpu_async_scheduler.py` (split),
`vllm_gaudi/envs.py`, `docs/configuration/env_variables.md`, seven small hooks in `hpu_model_runner.py` (import;
`_ple_forward_kwargs`; `_flash_next_async_begin_step`; the start of `sample_tokens`; `_create_decode_input_data`
after `_with_host_ngram_ids`; after `AsyncHPUModelRunnerOutput(...)`; the top of `_prepare_sampling`).

## 4. Card jobs

| Job | What | State |
|---|---|---|
| 00171 `real8-async-ab` | TP8, integration build + async: tiny gate with the flag, then async / nospec / nospec-devids at 32 and 1 streams | queued 01:37 UTC |
