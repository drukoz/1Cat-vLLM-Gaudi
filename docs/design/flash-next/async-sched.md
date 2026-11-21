# Async scheduling for Flash-Next on HPU (track `async-sched`)

Branch `feat/flash-next-gaudi-async-sched` (from program head `5c1e8219`), worktree `wt/async-sched`. The
track's handoff lives in this file (the agent harness refuses `reports/async-sched/REPORT.md`); card logs and
results are under `/Volumes/Storage/Claude/gaudi-work/reports/async-sched/card/`.

## Handoff (if this agent stops)

Updated 2026-11-21 13:29 UTC.

- **Card results so far** (placement on, MTP timers 0, compare arms within a job only):
  - 00220, mode 1 with `d3fb1b11` (MEASURED). ms/token sync → async: 1 stream greedy 27.9 → 29.4, sampled
    32.0 → 31.6; 8 streams greedy 43.7 → 50.1, sampled 53.0 → 48.8 (00186 had +16 to +21 ms/step). tiny-async
    passed. The census's `_tensor_str` reads (29 `.item()` per step) are capture-only (engine-step, VERIFIED in the
    records): all of them are in record 0 of a 50-step window, the step that captured that bucket's graph, and
    `per_step` is total/50. Steady steps wait in the D2H of `get_output` and 14 H2D per step (`utils.py:60`,
    `async_h2d_copy`).
  - 00221, mode 2 against synchronous MTP k = 2 (MEASURED; `reports/async-sched/card/00221-real8-async-m2u`).
    ms/step sync → mode 2: 8 streams greedy 61.9/61.9 → 62.2/65.3, sampled 67.5/67.1 → 67.5/69.5; 1 stream greedy
    53.4/53.7 → 55.0/57.6, sampled 57.9/56.9 → 53.3/56.2. Mode 2 is even with sync per step. At 1 stream greedy it
    gets 2.00 tokens/step against 2.42: its text leaves sync's after 13 equal tokens (sync opens a `<think>` block
    the drafter predicts well), INFERRED content, not a defect: greedy texts move with the batch's numerics even
    within the sync arm (8 streams, stream 0: round 1 "He had seen ships", round 2 "found a rope"), and mode 2's
    1-stream text is sync's 8-stream round-1 text. Spans: mode 2 hides the engine gap (sync 2.2-4.6 ms
    → 0.26) but the host's forward + wait (blocked on the device) grows by as much (1 row: sync 38-40 ms, mode 2
    41-44). INFERRED: the per-step pageable H2D copies (about 20 per step) wait behind the step in flight.
    engine-step's `VLLM_HPU_PERSISTENT_INPUTS` (pinned host buffers, non-blocking copies, two alternating slots under
    async scheduling) is the matching fix: an async arm with it when it is on the program head.
- **Branches.**
  - `feat/flash-next-gaudi-async-sched`: done, merged as PR #25 (`4f070c75`): modes 1 and 2 on the program head.
  - `feat/flash-next-gaudi-async-sched-s2`: stage 2 (section 2.6), all default-off, CPU-green. M1 `97a3a08a`
    `54c9f552` `959890bf`; M2 `cef6fd98`; M3 `a0f29355` `f7bb35ce`; `f616e3e1` (M1 with prefix caching keeps the
    align mode's load count). `78478ead` merges the program head `4f070c75` (PRs #23 chain, #24, #25; conflicts only
    in `SWITCHABLE` and its test: the union); `915638c2` (= `35d9d4d4` of the job branches) makes mode 3 issue early
    after the chain's stage B too; `9d2ee2e3` tests it on the engine (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=1`).
  - **Stage 2 is on the program head**: PR #27 merged as `4d66755e` (11:48 UTC). Gate on the frozen
    `9d2ee2e3` (s2 + head `4f070c75`): 80 files, 0 red (`reports/async-sched/logs/gate_9d2ee2e3.out`); the worker
    files, not in the gate's set, green on the same merge. Body: `reports/async-sched/pr_body_s2.md`.
  - engine-step's persistent-inputs branch merged the program head as `887266b1` (the device-context passthrough
    kept, as asked). s2 merges it with the `SWITCHABLE` union only (scratch branch `s2-pi` in `scratchpad/m1clone`).
    With `VLLM_HPU_PERSISTENT_INPUTS=1` (MEASURED on CPU, `reports/async-sched/logs/tests_s2_pi.log`): loop_async 7
    and early 4 pass, but device_inputs fails 6 of 8 and align 2 of 3. Cause (VERIFIED): M1 kept references to the
    step's host-made inputs and the next step's uploads rewrite those buffers first (one buffer per input in a
    synchronous engine). engine-step also found that an input uploaded twice in a step shared a pair; they fixed it
    (a pair per upload of a step).
  - `feat/flash-next-gaudi-async-sched-s3` (follow-ups) at `412a568d`: `59b54a77` no draft scatter when M1 made
    the step's ids (two H2D copies and two device ops less per step); `a4f1e896` M1 keeps copies of host-made inputs
    (the fix above); `3e98b623` its unit tests (`tests/unit_tests/worker/test_flash_next_async_spec_keep.py`, 3);
    `412a568d` class defaults for the async-scheduling state: PR #27 broke
    `tests/unit_tests/worker/test_flash_next_host_ngram.py` on the head (15 of 22, AttributeError `_verify_advance`
    in stub runners built without `__init__`; found by engine-step; bin/gate does not run `tests/unit_tests/worker/`).
    It merges the head `2c84deea` (`2b063e35`). CPU: every worker file that imports on CPU passes
    (`reports/async-sched/logs/tests_s3_worker.log`; `test_step_timing_wiring` fails 1 on the clean head too, since
    the chain's `73e9fbf8`; ten files need habana_frameworks); with PI on s3 + engine-step's `8a5bc621`: M1 8, M3
    3, early 4, loop_async 7 (`tests_s3_pi2.log`). **Gate**: a gate on the older s3 (`57339426`) runs and was not
    stopped (the kill was denied); `reports/async-sched/logs/gate_412a568d.out` starts the gate of the frozen
    `412a568d` when it ends. If green: `gh pr create --base feat/flash-next-gaudi --head
    feat/flash-next-gaudi-async-sched-s3` (title "Stage 2 follow-ups: class defaults for the async state (fixes
    test_flash_next_host_ngram), M1 keeps copies of host-made inputs, no draft scatter under M1"), then
    `gh pr merge --merge`. Integration takes the four commits for build 6 (they cherry-pick cleanly onto
    `2c84deea`).
- **00230** (mtp-draft-graph's chain job, MEASURED, `reports/mtp-draft-graph/card/00230-real8-chain-b5b`): M1's
  `tiny-m1` gate passed (208 of 223 verification steps advanced on the device, 0 differences); `m1sw` (one server,
  mode 2, chain stage A, M1 switched at runtime): M1 costs ms/token 24.7 → 26.6 and 25.0 → 25.6 at 32 streams
  (+1.2, +4.9 %), 12.7 → 13.4 and 12.7 → 13.6 at 1 stream (+0.8); texts identical, acceptance unchanged; 87 % of
  verification steps advanced. In mode 2 M1 is overhead only: mode 3 has to win more than that.
- **Card job 00256** `real8-async-m3e` (all, 85 min; replaced 00240, which replaced 00237, 00233 and 00229; ran from
  13:16 UTC). Its `tiny-m3` gate failed: the tiny server ran with prefix caching on (my opt-out was only in
  `rental/serve.py`; the tiny server is `rental/main.py`'s `tiny_serve_command`), mode 3 logged "acts as 2", and
  `1a70d353`'s rule left out every `async3` arm, the m3a arm too: no mode-3 numbers from it, only the card M3 gates
  (`tiny-m3pc1`, `tiny-m3pc2`) and mode-2 baselines. **00269** `real8-async-m3f` (the same arms and env, branch
  `e51a697f`: the tiny command honors `FARM_SERVE_NO_PREFIX_CACHING`; an m3a arm answers only to the tiny-m3pc gates)
  was queued at 13:26 at the end of the queue; the coordinator was asked to bump it right after 00256. Waiter
  `reports/async-sched/logs/wait_00269.out`. The rest of this entry describes both jobs.
  - Job branch `feat/flash-next-gaudi-async-sched-m3-b5` at `cd49d9f7` (never merge it; `1a70d353` makes a failed
    `tiny-m3` gate leave out the `async3` arms; `cd49d9f7` adds the word `m1off` and `59b54a77`, the scatter skip):
    the M2 job branch
    `44a51fcd` (build 5 + their chain + M1 + M2) + M3 + the M1 fix + harness: gates `tiny-m3pc1` (prefix caching on,
    mode 2, M3 = 1; passes only with M3's check line at 0 differences) and `tiny-m3pc2` (mode 3 + M3 = 2; passes only
    if steps were issued early and nothing acted as 2), word `m3a` (M3 = 2; the arm is left out when a `tiny-m3pc`
    gate failed), word `turns` (`tools/flash_next/rental/commit_turns_probe.py` after the probe, block
    `HEADLINE_TURNS_BLOCK=512`). CPU on the branch: align 50, spec 41, engine align 3, engine device inputs 8, early 3.
  - Arms in order: `tiny-m3 mtp2-async3-cuts-nopc-greedy mtp2-async-cuts-nopc-greedy` (M2 isolated, prefix caching
    off), `tiny-m3pc1 tiny-m3pc2`, `mtp2-async3-cuts-greedy-m3a-turns mtp2-async-cuts-greedy-turns` (prefix caching on),
    `mtp2-async-cuts-greedy-m1off` (the best mode 2: M1 off).
  - Env: build 5's dense env (`VLLM_HPU_MOE_DENSE=1 VLLM_HPU_MOE_DENSE_MIN_TOKENS=1 VLLM_MOE_CHUNK=0,64,8192
    VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000 HEADLINE_REQUIRE_DENSE=1`), prefix flags,
    `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`, `VLLM_HPU_QSA_SLOT_SHIFT=0`, chain + draft graph + I01 +
    `SPEC_DEVICE_INPUTS=2`, levels 32/8/1, greedy, 2 rounds, step timer mode 1, MTP timers 0,
    `HEADLINE_PLACE_SWITCH=/results/switches/place`, `HEADLINE_HOLD_S=5040`. The waiter writes
    `reports/async-sched/logs/wait_00256.out`; else `bin/farm fetch async-sched 00256 00256-real8-async-m3e`.
  - Read: the `tiny-m3`, `tiny-m3pc1`, `tiny-m3pc2` lines in the job log; `python3 -I
    reports/async-sched/tools/ab_table.py <dir> --pairs mtp2-async3-cuts-nopc-greedy:mtp2-async-cuts-nopc-greedy,
    mtp2-async3-cuts-greedy-m3a-turns:mtp2-async-cuts-greedy-turns,mtp2-async3-cuts-greedy-m3a-turns:mtp2-async-cuts-greedy-m1off`;
    `turns_table.py <dir> --pair mtp2-async3-cuts-greedy-m3a-turns:mtp2-async-cuts-greedy-turns`; `timer_spans.py <dir>` (mode 3's gain shows as
    prepare and forward issue under the device's busy time); `turns.json` in both turns arms: `again` equals `first`,
    `eight_again` equals `eight`, `next_turn` has `cached_tokens` > 0.
- **00230** is mtp-draft-graph's chain job (build 5 + chain + my M1 as cherry-picks): it carries M1's A/B, the
  `tiny-m1` gate at `=1` and `m1sw` (one server, arms `=0` and `=2`). Their results folder, read when it lands.
- **Placement switch**: `~/farm/async-sched/results/switches/place` on the node, read once at each job's start, now
  `place` (the coordinator sets it). `on` = `VLLM_HPU_WORKER_CPU_PIN=1` + `place_wrap.sh` + `place.sh` (rule a^1);
  `place` = the placement without the pin flag. Since 12:05 UTC (coordinator, via integration) the fixed pair is in
  `~/farm/async-sched/results/late/place/` and `bench/`: `place.sh` 816292de (native-replay `3a733c26`: moves only
  the server's own process tree) and `place_wrap.sh` 7b159d11 (collectives `98213f8c`: each server starts from the
  container's whole cpuset). The build 5 kit (00240's branch) reads `late/place/` first. Before the fix every later
  server of a job had 16 CPUs fewer: 00220 and 00221 ran their arms at 144, 128 and 112 CPUs of 152 (first to
  third), so their sync baselines had 16 fewer than the async arms (collectives and integration saw no decode loss
  from it in 00222, only longer loads). The old wrapper: `reports/async-sched/bench/place_wrap_363bcf9d.sh`.
- **Next**: read 00256; s3's PR when its gate is green; the persistent-inputs
  job (mode 2 and mode 3, PI off/on, prefix caching on, 32/8/1; needs PI on the program head and s3); sampled with
  mtp-draft-graph's D2 (they have no D2 yet: the chain refuses sampled requests).
- The 1-token prompt note: spec-lookup's `005e1ff9` is not on the program branch yet. When it is, put prompt 0 back
  into `_serve_staggered`.

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
  the worker cannot run ahead: only the engine turn (0.6 to 2 ms) hides. The bridge source at the node's version
  (v1.24.1 head, native-replay's checkout `reports/native-replay/bridge-src`) says it does not block: with
  `PT_ENABLE_HABANA_STREAMASYNC` (default true, `scripts/env_flags.yaml:235`) `copy_data_to_host` enqueues the DMA
  with a completion callback and returns; only a blocking copy spins until it is done
  (`backend/helpers/tensor_utils.cpp:408-446`). VERIFIED in the source, not yet on a card; vllm-gaudi's CI runs
  granite with async scheduling (`tests/full_tests/ci_e2e_discoverable_tests.sh:294`).
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
books and the first draft pass's issue, about 2 to 4 ms). Stage 2 would move the accepted counts to the device
(lattice offsets, positions, slot mapping of the next verification, align commits, the drafter's inputs): 2 to 4
days. Assessment (INFERRED): at 32 streams an MTP step is device-bound (verification about 85 ms of device work on
TP8 against about 55 ms of host work), so once stage 1 lets the host run ahead, stage 2 can only remove the
in-step gap at the verification-to-draft boundary, 2 to 4 ms of about 100 ms; at 1 stream the step is host-bound
and stage 2 changes nothing the host must do. Not recommended unless 00176 shows a larger gap.

**Revised 08:15 UTC: stage 2 is worth up to about 30 ms per step at 32 streams, not 2 to 4.** Step-anatomy split
the dense build's synchronous MTP k = 3 step at 32 streams (84.9 ms; the coordinator's message): prepare 6.2,
forward issue 27.2, wait for the device 31.3, draft 14.6, sample and post 2.6, gap 2.8; the device is busy 54.1 ms
and idle 30.8 ms while the host works serially. The assumption above (a device-bound step) was wrong. From the code
(int3, `hpu_model_runner.py`; 00203's census checks it on the card): mode 2 overlaps the step boundary (the engine
schedules during the step) and the draft read-back (no `take_draft_token_ids`, so nothing waits for the draft
passes), so step N+1's prepare and the start of its forward issue overlap step N's draft passes on the device,
about 6 ms (00186: 77.2 against 83.4 ms per step at 32 streams, greedy). Everything else waits for the host's
read of the verification result (`parse_output`, :7440): the accepted counts per request (:7455), the align
commits (:7461), the drafter's inputs (`propose_draft_token_ids`, :7664, takes the host token lists), and every
input of step N+1. Overlapping the rest takes, in order: (1) the draft chained behind the verification on the
device (mtp-draft-graph's task), so verification and draft are issued back to back without the read; (2) stage 2,
step N+1's inputs from step N's accepted counts on the device (positions, sequence and context lengths, KV slot
mapping and block-table offsets, the recurrent lattice offsets and align commits, the PLE context of the k + 1
verification rows, input ids), as upstream's GPU runner does for async speculation, 2 to 4 days; (3) mode 1's
asynchronous output for the verification result, so no read of a step's tokens stays on the main thread. Fully
overlapped, a step becomes max(device about 54 ms, host issue about 50 ms); below that the host issue itself must
shrink (engine-step's batched input upload, native-replay's graph replay for the 27 ms forward issue).

For the integration build (02:45 UTC): take commit `fb5cefca` (merges cleanly into integration `8df22e1e`), not the
branch head (which merges the program head with PR 16, carried there as qsa-fused). Flags `=1` / `=2`; with `=2`
also `VLLM_HPU_MTP_TIMING=0`; engine-step's `01d66e31` keeps candidate sampling under async scheduling.

CPU evidence (engine-step's harness on this branch; n-gram as the vehicle because the harness loads no drafter
model, with oracle drafts returned as a tensor the way the MTP drafter returns them, one wrong draft every few
steps): `test_engine_async_spec.py`: the async run gives the reference tokens, corrects counts and writes drafts;
without the correction the tokens go wrong; without the scatter they stay right with fewer accepted drafts. The real
MTP drafter under async runs in mtp-card-ready's rehearsal engine (`tests/unit_tests/hpu_cpu_engine.py`, on the
integration tree only), checked on the job branch before the card.

### 2.5 Mode 1 slower than synchronous at 1 and 8 streams (00186) and the fix `d3fb1b11`

MEASURED (00186, integration build 4 at TP8, mean of two rounds, ms per step, async minus synchronous):

| server | streams | greedy | sampled |
|---|---|---|---|
| mode 1 (`nospec-async` vs `nospec`) | 32 | 63.4 vs 63.2 (+0.2) | 51.9 vs 58.0 (-6.1) |
| mode 1 | 8 | 66.7 vs 50.5 (+16.2) | 66.9 vs 51.4 (+15.5) |
| mode 1 | 1 | 54.6 vs 33.7 (+20.9) | 53.2 vs 36.4 (+16.8) |
| mode 2 (`mtp2-async` vs `mtp2`) | 32 | 77.2 vs 83.4 (-6.2) | 89.6 vs 94.3 (-4.7) |
| mode 2 | 8 | 66.0 vs 71.4 (-5.4) | 92.1 vs 70.6 (+21.5) |
| mode 2 | 1 | 56.0 vs 63.5 (-7.5) | 78.5 vs 63.3 (+15.2) |

The sampled rounds use the checkpoint's defaults (temperature 1.0, top_k 20, top_p 0.95, no penalties). The two
MTP arms also differ in `VLLM_HPU_MTP_TIMING` (2 synchronous, 0 async). mtp2-async's lower 1-stream greedy
acceptance (2.05 against 2.44) follows its text (INFERRED): greedy text differs between arms on the card.

VERIFIED in the bridge source at the node's version (`reports/native-replay/bridge-src`):
`HPUEvent::synchronize` waits for its event and then calls `device::flush_host_events()`
(`backend/habana_device/HPUEvent.cpp:131`). That takes a snapshot of every pending host event and waits for each
(`backend/synapse_helpers/device.cpp:1107-1118`). Every device-to-host copy into pageable memory registers one,
completed by a callback after the staging copy (`device.cpp:1520-1590`). And `register_host_event` first waits for
a pending event on the same host address (`device.cpp:1166-1176`).

Build 4's ledger called `settle(keep=1)` at the start of every `sample_tokens`, on the main thread: an
`event.synchronize()` of the step before last, which through the flush waited for the copy of the step in flight
too. And in this runner `execute_model` only updates the batch and keeps the scheduler output (VERIFIED,
`hpu_model_runner.py` `execute_model`: `self.scheduler_output = scheduler_output; return None`); the step's inputs,
its forward and its sampling are all issued in `sample_tokens`, after that hook. So the worker waited for all of
step N on the device before it prepared step N+1: no overlap at all, plus the cost of the async machinery (the
output thread, the engine's batch queue), which fits mode 1 being slower than synchronous. Also VERIFIED:
`device::synchronize_event` holds `usr_event_mutex_` during `synEventSynchronize` (`device.cpp:827`), so the output
thread's wait in `AsyncHPUModelRunnerOutput.get_output` blocks event operations of the main thread (creating and
recording the next copy's event, `wait_stream`) until the device reaches the event; those come after the next
step's forward is issued. Mode 2 never makes that call (its runner step is synchronous), and mode 2 greedy is not slower. `d3fb1b11`:
no wait in a steady decode step. Placeholders keep `output_token_ids` at their synchronous length (`min_tokens`),
the values are settled only before a sampler pass that reads them (penalties, bad words) from a window of the last
64 steps, and older placeholders become the penalty pad (CPU: `test_output_token_ids_have_the_synchronous_length_and_no_wait`
and the penalty tests). INFERRED: mode 1's async step is nearly flat in the stream count (54 to 67 ms) while the
synchronous step grows (34 to 51 ms), which a wait for the step in flight alone may not explain; 00196 times the
spans of every step (and counts the host synchronisation points) with the fix. Mode 2 sampled has a gap of the same
size without any ledger: cause unknown, 00197. CPU first (06:27 UTC, mtp-card-ready's rehearsal engine with the real
MTP drafter on the int3 tree, k = 2, `VLLM_HPU_SPEC_SAMPLED_VERIFY=2`, the step timer in mode 1, six prompts at
once; records in `reports/async-sched/cpu/`, read with `reports/async-sched/tools/timer_spans.py`): the verification step's host spans outside the model and the drafter (setup, prepare, sample, post, wait,
gap) are within noise of each other in synchronous MTP and mode 2, greedy and sampled (for example 2 rows,
sampled: prepare 1.98 against 1.41 ms, sample 2.89 against 2.61, gap 0.27 against 0.23). No host work runs only
for sampled requests under mode 2, and the drafter has no sampled path (VERIFIED). So the card's gap is an
interaction with the device, which only the card's spans and census can show (INFERRED).

### 2.6 Stage 2: step N+1's verification inputs on the device (branch `feat/flash-next-gaudi-async-sched-s2`)

The coordinator's order (08:15 UTC): stage 2 now, the largest item behind the 32-agent target (up to about 30 ms of
the 84.9 ms synchronous MTP step at 32 streams, section 2.4). Plan approved 08:24 UTC (also in `TASKS.md`); every
stage behind its own default-off flag, its CPU proof a test file in the tree, then a card A/B, greedy before
sampled:

| Stage | Owner | What | Removes from the device's idle | Estimate |
|---|---|---|---|---|
| D1 | mtp-draft-graph | `VLLM_HPU_MTP_DRAFT_CHAIN=1`: verification, in-graph greedy acceptance and the one-graph draft issued back to back; leaves `runner._mtp_chain_out` | the gap between verification and draft (the host's read, the drafter's input preparation) | theirs |
| M1 | async-sched | `VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS` (1 check, 2 use): step N+1's acceptance-dependent inputs made on the device | nothing (enabler; prices the extra ops) | 1 day (done on CPU) |
| M2 | async-sched | issue N+1 before reading N: N's tokens copied on a side stream and read on the output thread, the host's books one step late | N+1's preparation and forward issue, as far as N's device work covers them: up to about 20 ms at 32 streams | 1 to 1.5 days |
| M3 | async-sched | prefix-cache align mode exact under M2 (pre-step copy and boundary commits from the device acceptance); requests joining or leaving through a row map | nothing (production has prefix caching on) | 0.5 to 1 day |
| D2 + M4 | both | sampled requests: the chain through the in-graph candidate sampler; tests on my side | the same for sampled steps | 0.25 day mine |

Interfaces agreed 08:20 to 08:30 UTC:

- mtp-draft-graph: `VLLM_HPU_MTP_DRAFT_CHAIN` (needs `VLLM_HPU_MTP_DRAFT_GRAPH=1` and
  `VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`); the per-step decision `runner._mtp_chain_step` (set before the forward);
  after a chained step `runner._mtp_chain_out` = (accepted [B], next_ids [B, 1 + k] = bonus + k drafts, sampled
  [B, 1 + k] with -1 after the bonus, last_row [B], num_decodes, k), int32, the step's padded decode batch,
  input-batch row order, padded rows accepted 0 and sampled -1, every id rule-9-safe (from an fp32 id table, never a
  `.to(int32)` of arg max indices); their host read and books in a method of its own (`_mtp_chain_settle`) that M2
  calls one step late.
- engine-step (`VLLM_HPU_PACKED_INPUTS`): a per-input switch host-made / device-made; in a verification step that
  continues a chained one, the device-made inputs (ids, positions, the verification slot mapping, the GDN accepted
  counts and, with prefix caching, the lattice rows, the PLE context, the draft ids) are read from my tensors and
  left out of the packed upload; the I01 block table, `query_start_loc`, the logits indices, the target / bonus
  indices, `cu_num_draft_tokens` and the sampling parameters stay in the packed buffer. Native-replay's bridge facts
  (through engine-step): a graph input binds without a copy only if it is the tensor the capture saw; any other is
  copied in by a lazy D2D copy, one launch each; a view handed to a custom op fails at launch.

M1 (VERIFIED on CPU, `vllm_gaudi/v1/worker/flash_next_async_spec.py`, runner hooks in `_create_decode_input_data`
and `sample_tokens`): after a verification step, its device inputs and its acceptance are kept (mtp-draft-graph's
`_mtp_chain_out` when the step chained; else a stand-in from the host's read, its draft columns refreshed from the
drafts the scheduler scheduled, which is what a synchronous engine verifies). In the next verification step, if its
decode rows are the same requests in the same order with the same padded batch and k, the device makes: ids =
next_ids; positions = `positions_N[r, 0] + a + 1 + j`; slots = `block_table[r, pos // bs] * bs + pos % bs` (an int32
`index_select` over the one-row-per-request table of `VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`; padded rows the pad
block's slots in row-major order, as `fill_pad_slots`); GDN `num_accepted_tokens` = a + 1 (padded rows 1); the PLE
context = `advance_ple_context(context_N, ids_N, a + 1)` (padded rows eos). Tests:
`tests/unit_tests/worker/test_flash_next_async_spec.py` (21: the device-made inputs equal the host-made ones from the
runner's own helpers over 12 steps of random acceptance, k = 1..3, padded rows; the non-continuation cases; the
stand-in), `tests/unit_tests/models/flash_next/test_engine_spec_device_inputs.py` (the engine rehearsal with the real
MTP drafter, k = 2, oracle acceptance 2,0,1,2,1,0, prefix caching on and off: `=1` checks 32 steps with no
difference in any of the five tensors; `=2` gives the model the device-made inputs and the streams equal the
engine's without the flag). The branch is stacked on mtp-draft-graph's (its MTP fixes and the rehearsal engine are
not on the program head yet). Not covered by M1: the align-mode lattice rows (host-made; M3), M-RoPE positions and
the dense decode metadata (refused, logged once).

M2 (VERIFIED on CPU, `VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3`, cef6fd98): mode 3 is mode 2 for the engine; in a chained
verification step (`_mtp_chain_step`) the runner returns `AsyncSpecModelRunnerOutput` right after the chain is issued
(stage A behind the verification, or stage B with the greedy rejection inside the chain's graph; `35d9d4d4`): the
step's tokens `[B, 1 + k]` (-1 after the bonus) are copied on a side stream and read by the worker's output
thread; `_mtp_chain_settle`, the post-processing and the drafter call are skipped (the chain drafted on the device,
M1 makes the next step's inputs from its output). The books (the batch's token rows and `num_tokens(_no_spec)`, the
requests' `output_token_ids`, `_gdn_num_accepted_by_req`) are queued (`SpecEarlyBooks`) and applied at the start of a
later `execute_model`, in step order, without waiting when the step continues the batch (no request new, resumed or
finished; every request with its 1 + k tokens; at most two books stay pending). A step whose batch changes waits for
every pending book first and its `num_computed_tokens` become the true ones (each request's count seeded at the
first early step and advanced by every applied book; the scheduler's still count the early steps' drafts as
accepted). A chained step that needs its tokens (logprobs, penalties, bad words, `min_tokens`, structured output, a
KV transfer, a prefill in the batch) reads them as before, after every pending book. Mode 3 needs the chain,
`SPEC_DEVICE_INPUTS=2`, the sparse decode metadata and prefix caching off or `VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN=2`
(the align-mode pre-step copy and boundary commits from the device: M3); otherwise it acts as 2 and logs why. Tests:
`tests/unit_tests/models/flash_next/test_engine_async_spec_early.py` (4: the streams equal mode 2's, prompts one at a
time and all at once, 128 steps issued before the previous step's tokens reached the host, 122 books applied late
and 5 after a wait; the same with the chain's stage B, `VLLM_HPU_VOCAB_PARALLEL_LOGITS=1`; with prefix caching and
no M3 mode 3 acts as 2), the books in `test_flash_next_async_spec.py`. On CPU every
copy is done when it is issued, so this proves the books, not the overlap; the rehearsal's oracle cannot force
acceptance under async (it hooks `take_draft_token_ids`, which an async engine never calls), so M2's runs mostly
accept 0 (M1's oracle runs cover every count).

M3 (VERIFIED on CPU, `VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN`, a0f29355): with prefix caching (align mode) a
verification step's running block is `cdiv(ctx + 1 + k, bs) - 1` of its TRUE context, so under mode 3 the host (one
step behind) cannot make the align mode. On the device, from the step's device positions and the carried state (the
last step's running block and next load count; seeded from the host's books when the step does not continue the
last one): the pre-step plan (`align_precopy_plan`'s rule: a copy when the window moved), the copies (conv from
row[prev] at row offset n - 1, ssm from row[prev + n - 1], into row[curr]; masked rows copy the pad slot onto itself;
whole rows with one `index_copy_`, never a write through a slice), the recurrent groups' lattices row[curr : curr + 1
+ k] (other groups keep the host's rows) and the load counts; after the step's acceptance (the chain's or the
stand-in's) the boundary commits of the committed token whose position completes a block (ssm from store[t], conv
window from store[0] at row offset t) and the self commits (next load count 1). The host's books follow: now on a
synchronous step, with the async copy (`AsyncSpecModelRunnerOutput`'s extra rows) under mode 3. Tests:
`tests/unit_tests/worker/test_flash_next_async_align.py` (50: plan, lattices, commits, self commits and both conv
layouts equal `recurrent_lattice` / `recurrent_commit`), `tests/unit_tests/models/flash_next/test_engine_spec_device_align.py`
(prefix caching on, 160-token generations crossing 256-token blocks; oracle acceptance: `=1` without a difference,
`=2` the host's streams; mode 3 with `=2` serves with prefix caching and gives the synchronous streams, the agent's
next turn included; no copy through a slice). On this build the CPU engine's next turn gets no prefix-cache hit, so the
committed slots are checked by the unit tests and the check mode, not by a hit (the card's multi-turn probe should).

M1 for mtp-draft-graph's chain card job on build 5: `feat/flash-next-gaudi-async-sched-m1-b5` (28b113fa, 12b7d8c6,
ad21795c on build 5's 57ca7c31; the M1 flag is read every step, so one server started at `=2` runs the `=0` and
`=2` arms through the runtime switch file).

## 3. What is on the branch

See the handoff. Files: `vllm_gaudi/v1/worker/flash_next_async.py` (carry, ledger), `vllm_gaudi/platform.py`
(`resolve_flash_next_async_scheduling`), `vllm_gaudi/v1/core/sched/hpu_async_scheduler.py` (split),
`vllm_gaudi/envs.py`, `docs/configuration/env_variables.md`, seven small hooks in `hpu_model_runner.py` (import;
`_ple_forward_kwargs`; `_flash_next_async_begin_step`; the start of `sample_tokens`; `_create_decode_input_data`
after `_with_host_ngram_ids`; after `AsyncHPUModelRunnerOutput(...)`; the top of `_prepare_sampling`).

## 4. Card jobs

| Job | What | State |
|---|---|---|
| 00171 `real8-async-ab` | TP8, integration build + mode 1: tiny gate, async / nospec / nospec-devids at 32 and 1 streams | replaced by 00175 (02:20 UTC) |
| 00175 `real8-async-mtp` | as 00176 with k = 3 | replaced by 00176 (02:35 UTC) |
| 00176 `real8-async-mtp2` | as 00179 with the tiny gate aborting the job | replaced by 00179 (02:50 UTC) |
| 00179 `real8-async-mtp2b` | TP8, integration `d00c0a9a` + modes 1 and 2: tiny gate (no speculation and MTP k = 2), mtp2-async, mtp2, async, nospec at 32 and 1 streams, greedy and sampled | cancelled (05:45 UTC): 00186 answered it |
| 00186 `real8-build4` (integration) | the coordinator's build 4 with modes 1 and 2: mtp2-async, nospec-async, mtp2, nospec at 32, 8 and 1 streams, greedy and sampled | done; section 2.5 |
| 00195 `real8-async-m1` | mode 1 with `d3fb1b11`: nospec-async, nospec, nospec-devids, nospec-async-spin at 8 and 1 streams, step timer | replaced by 00196 (06:21 UTC) |
| 00196 `real8-async-m1b` | TP8, int3 `5ccf0604`: tiny-async, nospec-async, nospec, nospec-async-census at 8 and 1 streams, greedy and sampled, step timer | replaced by 00202 (06:44 UTC) |
| 00197 `real8-async-m2s` | TP8, int3 `5ccf0604`: mtp2-async, mtp2, mtp2-async-census (MTP k = 2, sampled verification 2, MTP timers 0) at 8 and 1 streams, greedy and sampled, step timer | replaced by 00203 (06:47 UTC) |
| 00202 `real8-async-m1c` | as 00196 from int3 `f2a5ff0b` (the placement switch, read at the start) | queued after 00189 |
| 00203 `real8-async-m2t` | as 00197 from int3 `f2a5ff0b` | queued after 00202 |
