# Flash-Next MTP: the k draft passes in one HPU graph (track `mtp-draft-graph`)

This file is the track's report. The harness of the track's agent does not write report files under `reports/`, so
the handoff lives here, on branch `feat/flash-next-gaudi-mtp-draft-graph` (worktree
`/Volumes/Storage/Claude/gaudi-work/wt/mtp-draft-graph`). Raw CPU runs and card logs:
`/Volumes/Storage/Claude/gaudi-work/reports/mtp-draft-graph/` (`cpu/`, `card/`). Labels: MEASURED (card, with its
file), VERIFIED (CPU test or code line), INFERRED.

## Handoff (if this agent stopped)

Last update: 2026-11-21 09:35 UTC (`date -u`), first agent of the track (Opus 5.5), started 04:35 UTC. The notes
after the next horizontal rule are history (06:50 UTC and before).

**Where things are (09:35 UTC).**
- **PR #23 merged** into `feat/flash-next-gaudi` as merge commit `e5188341`, from the gated head `d44ca56d` (gate: 74
  files, 0 red). It contains:
  - the one-graph draft (`VLLM_HPU_MTP_DRAFT_GRAPH`, card-verified in 00194);
  - the chained draft (`VLLM_HPU_MTP_DRAFT_CHAIN`, stage A `73e9fbf8`, stage B `c0bef8d8`, runtime switch
    `214cdcf4`; CPU-verified, not yet on a card).

  All flags are default 0.
- Feature branch `feat/flash-next-gaudi-mtp-draft-graph`, after PR #23:
  - `a7508d00`: `_mtp_chain_step` and `_mtp_chain_out` reset at the top of `sample_tokens`. Before, a step
    without decodes kept the last chained step's values: its timer records were labelled `+chain`, and async
    scheduling's mode 3 would have read a stale chain. Red-then-green engine check.
  - An engine test of the chain switched at run time.
  - This handoff.

  Next PR after its own gate.
- The gate of `436855e7` had one red file, `test_gdn_core_decode_batch.py::test_padded_batch_matches_off[negative]`
  ("RECURRENCE pad=-1: conv slot 5 differs from OFF"). It passed alone on the program head `adbcff30` and on this
  branch, and it was green in the gate of `d44ca56d`. INFERRED: intermittent, and not this branch's code.
- **Card job 00230 `real8-chain-b5b`** (all, 100 min; it replaced 00228, which had replaced 00212; the coordinator
  places it right after build 5's 00222, about 13:00 UTC).
  - Branch and image: `feat/flash-next-gaudi-mtp-draft-graph-b5` at `915d8863` (`ea1715fa` + `HEADLINE_TINY_IRLOG_ARMS`:
    the bridge's IR-graph and lazy logs of the tiny gate `tiny-chain-vlg` for engine-step's census of the step's
    cuts), image `farm-mtp-draft-graph:915d88638866`. It is build 5 (`integration/card-build-20261120b` at `eb21ac75`) plus:
    - the chain (`f3d82f32`, `357f220c`, `2a07e787`, `6a7a87ce`);
    - engine-step's `844a37f2` (`VLLM_HPU_VERIFY_LOGITS_IN_GRAPH`) and `ca52fa33` (`VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS`);
    - `64656082`: the verify graph's ids handed to the chain whole, VLG runtime-switchable;
    - async-sched's M1 (`249647f7`, `c1906d33`, `70d41eb9`), with `0cc01f43` putting its flag in SWITCHABLE;
    - the harness (`7fe2bea8`, `ea1715fa`).
  - Submit script and log: `reports/mtp-draft-graph/card/submit-chain-b5-3.log` (the script in this agent's scratchpad,
    `submit_chain_b5.sh`: the `--env` list below plus `HEADLINE_TINY_IRLOG_ARMS=tiny-chain-vlg`).
  - Env: build 5's serve-candidate env as 00222 runs it, every server async (`ASYNC_SCHED=2`, MTP timer 0), levels
    32 and 1, 2 rounds.
  - Arms:
    1. `tiny`;
    2. `mtp2-async-cuts-switch` and `mtp2-async-switch`: `graph-md-all` (off, on, md, on-md, all). The
       REPLICATED_FC + REPLICATED_EMBEDDING levers and the KV capacity with and without (server logs).
    3. `tiny-chain`, `tiny-chain-vlg`, `tiny-chain-vp`: the chain's gates.
    4. `mtp2-async-cuts-chainsw`: on-md, chain-md, vlg, chain-vlg, all, chain-all on one server.
    5. `tiny-m1` and `mtp2-async-cuts-m1sw`: async-sched's M1, 0 against 2, with the chain on.
  - Placement: async-sched's switch, through the hard link `~/farm/mtp-draft-graph/results/switches/place`.
  - Late-bound (coordinator 09:50 UTC): `~/farm/mtp-draft-graph/results/late/run-chain.sh` on the node exports
    `VLLM_HPU_MOE_DENSE_MIN_TOKENS=1` (the new serve baseline: the dense MoE form from 1 token, kernel-fixes'
    00213) for the whole job, then runs the arms as submitted. Remove that file after 00230, or a later job of
    this track with the same wrapper picks it up.
  - async-sched builds its M2 job on `ea1715fa` or `915d8863` (the same code).
- **What to read when 00230 ends** (`bin/farm fetch mtp-draft-graph 00230 00230-real8-chain-b5b` if no waiter
  copied it):
  - `headline.md`;
  - per arm folder, `tools/ab_summary.py <arm folder>` (rounds per arm, means, text identity);
  - the tiny gates' lines in the job log: the chain's stage, `HPU graph replay copies ...` (the chain graph's
    copied inputs), M1's `=1: ... 0 with differences`;
  - the servers' `GPU KV cache size` (cuts against plain: the token capacity with and without
    REPLICATED_EMBEDDING);
  - the chain arms against their twins at 32 and 1 stream: chain-md vs on-md, chain-vlg vs vlg, chain-all vs all;
  - m1on vs m1off.

**The chained draft (`VLLM_HPU_MTP_DRAFT_CHAIN=1`, the coordinator's item (b), 08:10 UTC).**

Today the runner reads the verification's result on the host (`parse_output`), then builds the draft's inputs and
issues the draft. So the draft's host issue (8 to 15 ms per step at 32 streams on TP8) sits between two device
phases.

With the chain, right after the rejection sampler and before any host read, the drafter issues `MtpDraftChain`
(one HPU graph, `flash_next_mtp_steps.py`):
- `chain_draft_rows`: each request's committed rows, last row, accepted count and bonus token from the device
  result (`HpuEagleProposer.prepare_inputs` on the device);
- the pass-0 inputs gathered from the target's rows;
- `draft_step_slots`: the later passes' block tables and slots from the device positions (engine-step's I01 builder
  on the device);
- then `MtpDraftLoop`'s passes. The ids are int32 from fp32 id tables (`first_max_ids`, card rule 9).

Stage B: when the verification's per-row greedy ids are on the device (VP, or engine-step's VLG), the greedy
rejection (`greedy_rejection_sample_from_tokens`) runs inside the chain's graph too.

The host then reads the result and keeps its books as before (`HPUModelRunner._mtp_chain_settle`). It takes the
drafts from `_mtp_chain_out`: `MtpChainOut` with accepted, next_ids, sampled, last_row, num_decodes, k and drafts,
the interface agreed with async-sched.

Refused (today's path, each reason logged once): a prefill in the step, a sampled request, structured output, the
draft-rank log, no `VLLM_HPU_MTP_DRAFT_GRAPH=1`, I01 not in force. The runner logs once which stage ran. Timer kind
`+chain`.
- CPU (VERIFIED): `test_mtp_draft_chain.py` 10 passed. In the engine rehearsal, every variant's drafts and streams
  equal the per-pass drafter's:
  - k = 1, 2, 3, every prompt at once, and stage B with VP;
  - on the job branch also: stage B on the verify graph's ids, and the six chain arms switched at run time in one
    run;
  - M1 switched 2 → 0 → 2 with the chain.
- INFERRED gain at 32 streams: about 12 ms per step. The draft's host issue overlaps the verification's device
  time, and the host read now also waits for the draft's device time (2.6 ms at 32 streams).
- Split of item (c) with engine-step (their message, 08:55 UTC):
  - engine-step takes the logits region into the captured verify forward (`VLLM_HPU_VERIFY_LOGITS_IN_GRAPH`, their
    `844a37f2`), plus the front (embedding, PLE cast) and the final-mix cut;
  - this track takes the greedy rejection into the chain (stage B, done).

---

### Earlier handoff (06:50 UTC)

Last update: 2026-11-21 06:50 UTC (`date -u`), first agent of the track (Opus 5.5), started 04:35 UTC.

**Goal (coordinator, 04:45 UTC).** Run the k draft passes of an MTP decode draft in ONE HPU graph behind a default-off
flag. Basis: build 3's timer (00178, TP8, k = 3) at 32 streams: the draft costs 16.3 to 18.2 ms of a 104 ms step
(pass 0 issue 4.9 to 5.1, passes 1 and 2 metadata 1.2 to 1.3 each and issue 3.9 to 4.1 each, final device wait 0.7 to
2.8); at 1 stream 13.5 to 14.7 of about 68. INFERRED saving 7 to 10 ms per step at 32 streams, 6 to 8 at 1 stream
(`docs/design/flash-next/mtp-card-ready.md`, "Next MTP speed work").

**State.**
- Branch commits: `4648f8c7` merge of the program head `22ec5ea3` (PR #17 spec padding, PR #19 bridge tools; one doc
  conflict, `docs/design/flash-next/qsa-fused.md`: the program head's card-results page kept); `fe52724c` the flag
  `VLLM_HPU_MTP_DRAFT_GRAPH` (default 0; 1 = one graph), `MtpDraftLoop`
  (`vllm_gaudi/v1/spec_decode/flash_next_mtp_steps.py`), the proposer's graph path (`hpu_flash_next_mtp.py`
  `_propose_in_one_graph`, `_draft_graph_loop`), docs in `docs/configuration/env_variables.md`, tests.
- CPU (VERIFIED): `tests/unit_tests/worker/test_mtp_draft_graph.py` 21 passed (the loop equals a transcription of
  the per-pass proposer at k = 1..5, batch 1 / 3 / 8: same calls, inputs and ids; no host read in its forward).
  Engine rehearsal k = 3, oracle accept 2,0,3,1: the drafter's own ids and the streams equal the per-pass drafter's
  with `VLLM_HPU_MTP_DRAFT_LOOP=1`; 47 of 47 decode drafts in one graph, the 6 prefill drafts per pass
  (`reports/mtp-draft-graph/cpu/k3/`). All six engine cases pass (k = 1, 2, 3, 4, 5 with oracle acceptance patterns
  covering every accepted count, and every prompt at once: padded batch rows, requests at different positions):
  `pytest tests/unit_tests/models/flash_next/test_engine_mtp.py -k one_graph` 6 passed in 193 s
  (`reports/mtp-draft-graph/cpu/engine_graph_all.log`).
- `e3dbc22a` merge of the program head `2e9f9196` (PR #20 engine-step: its sparse-QSA draft metadata, which the
  one-graph path calls too); `d3bbf2a6` the flag as a runtime switch (`runtime_switches.SWITCHABLE`) and the MTP
  timer's kind `verify+graph`; CPU engine `--switch-after` and a test that switches the flag mid-run (8 passed).
- `bf7cf32c` an engine test of the one-graph draft with engine-step's I01 (`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`,
  in force in both runs): same drafts and streams as the per-pass drafter with I01 (1 passed).
- Worker tests on the merged branch: `test_mtp_draft_graph.py`, `test_draft_rank_stats.py`, `test_mtp_draft_ids.py`,
  `test_mtp_step_timer.py` 51 passed.
- Gate of `bf7cf32c` (the merge protocol's common gate): `bin/gate reports/mtp-draft-graph/gate-bf7cf32c` (a snapshot
  worktree), output `reports/mtp-draft-graph/gate/gate-bf7cf32c.txt`; it waits for a gate slot (two other tracks'
  gates hold both). When green: `gh pr create --base feat/flash-next-gaudi --head feat/flash-next-gaudi-mtp-draft-graph`
  and `gh pr merge --merge` (default off), then tell the coordinator the PR number.
- Test branch `feat/flash-next-gaudi-mtp-draft-graph-b4` also has, for a follow-up job (not in 00194's image):
  `0a72517d` arm `mtp<K>-async-switch` (the A/B on the async MTP server, `VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2`; set
  `HEADLINE_ASYNC_MTP_TIMING=0`) and `43b5431c` `HEADLINE_SWITCH_SET=graph-md` (the 2 x 2 with I01). Analysis
  scripts: `reports/mtp-draft-graph/tools/ab_summary.py <arm folder>` (rounds per arm, means, on - off, text
  identity per stream) and `tools/timer_ab.py <arm folder>` (the timer's records per kind and rows, first record of a
  key dropped). Build 3's per-pass draft by that script (00178 mtp3): 19.0 ms per step at 32 streams (pass0 4.9,
  pass1_meta 1.1, pass1 3.8, pass2_meta 1.0, pass2 3.7, final_wait 4.4), 15.3 ms at 1 stream.
- **Card (00194, MEASURED): correct; -8 % per token at 1 stream; no measurable gain at 32 streams on TP8.** Details
  in "Card results". Verdict: keep it default-off, a low-concurrency (1 to 8 streams) setting.
- `344acd9c` engine test: the draft graph and I01 switched together at run time (the 2 x 2, k = 2): same drafts and
  streams as the per-pass drafter (1 passed; on build 4's code too, 2 passed with the switch test).
- **Next job, approved by the coordinator (06:35 UTC: fewer collectives per draft pass first, then I01; k = 2):
  00201 `real8-draft-levers2`** (all, 60 min; replaced 00199 at 06:44 UTC and kept its place, near the end of the
  queue: the coordinator expects it about 20:00 to 21:00 UTC; it moves up if jobs ahead end early). Test branch at
  `100f4bf0` (image `farm-mtp-draft-graph:100f4bf032fe`; `d64734c5` cherry-picked: the timer's kind takes `+md` while
  I01 is in force, so the four arms have records of their own). Earlier test-branch head `4b296df9` (adds
  `0e1abfb9`: arms `mtp<K>-rfc` / `mtp<K>-rfc-switch` = `VLLM_HPU_MTP_REPLICATED_FC=1`, the tiny gate's
  `HEADLINE_TINY_MTP_REPLICATED_FC` / `HEADLINE_TINY_DECODE_METADATA`), image `farm-mtp-draft-graph:4b296df99177`,
  00194's env with `HEADLINE_TINY_SPEC=2 HEADLINE_TINY_DRAFT_GRAPH=1 HEADLINE_TINY_MTP_REPLICATED_FC=1
  HEADLINE_TINY_DECODE_METADATA=1 HEADLINE_SWITCH_SET=graph-md HEADLINE_ROUNDS=3 HEADLINE_LEVELS=32,1`, arms
  `tiny mtp2-rfc-switch mtp2-switch`: the tiny gate with all three levers on (REPLICATED_FC was never on a card), then
  two MTP k = 2 servers (REPLICATED_FC 1, then 0), each alternating the 2 x 2 of draft graph and I01 per round (arms
  `off`, `on`, `md`, `on-md`). Read: `tools/ab_summary.py <job>/mtp2-rfc-switch` and `.../mtp2-switch` (rounds per
  arm, means), `tools/timer_ab.py` (kinds `verify`, `verify+graph`, `verify+md`, `verify+graph+md`), acceptance per arm,
  the 1-stream text identity across arms within a server. Fetch if no waiter copied it: `bin/farm fetch
  mtp-draft-graph 00201 00201-real8-draft-levers2`.
- `d64734c5` the timer's `+md` kind (CPU engine with the timer and the four arms switched: all four kinds, no error).
  Gate of `d64734c5`: `reports/mtp-draft-graph/gate/gate-d64734c5.txt` (snapshot `reports/mtp-draft-graph/gate-d64734c5`).

**Card job 00194 `real8-draft-graph-switch`** (all, 55 min; replaced 00191 at 05:37 UTC and kept its place: after
integration's 00187, start by 06:46 UTC at the latest). Test branch `feat/flash-next-gaudi-mtp-draft-graph-b4` at
`8b27af1b` = integration build 4 (`integration/card-build-20261120b` at `5dc6993f`, plugin code `bcaa3ed6`) + this
branch's `fe52724c` and `d3bbf2a6` (cherry-picked as `f4bcd11d`) + the kit changes `f2fa0677` and `8b27af1b`; image
`farm-mtp-draft-graph:8b27af1b3b1c`. Command (from `wt/mtp-draft-graph` with the test branch checked out):
`bin/farm job mtp-draft-graph real8-draft-graph-switch all 55m --env <build 4's env> --env HEADLINE_LEVELS=32,1
--env HEADLINE_ROUNDS=3 --env HEADLINE_MTP_FALLBACK=0 --env HEADLINE_TINY_SPEC=3 --env HEADLINE_TINY_DRAFT_GRAPH=1
-- bash tools/flash_next/integration/headline_job.sh tiny mtp3-switch mtp3-graph` (full line in
`reports/mtp-draft-graph/card/submit-switch.log`'s process; build 4's env as integration gave it: `VLLM_HPU_QSA_FUSED=1
VLLM_HPU_QSA_INDEXER_SHARD=1 VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2
VLLM_HPU_DECODE_CUT_INTERVAL=1000 VLLM_HPU_CANDIDATE_SAMPLING=1 VLLM_GRAPH_RESERVED_MEM=0.4
VLLM_HPU_PREFIX_ALIGN_SPLIT=1 VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 VLLM_MOE_CHUNK=64,8192
VLLM_MOE_TOKEN_BOUNDARY=512,1000000 PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1 VLLM_HPU_GDN_CORE_VERIFY=1
VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_SPEC_SAMPLED_VERIFY=2 VLLM_HPU_MTP_TIMING=1`; no async
scheduling). Arms:
1. `tiny`: the tiny model at TP8, `--spec 3`, flag on: tokens against the CPU reference and the mode without
   speculation (the job ends with exit 7 if they differ).
2. `mtp3-switch`: ONE real-model server, MTP k = 3, started with the flag off; `decode_scaling.py --switch-arm
   off=VLLM_HPU_MTP_DRAFT_GRAPH=0 --switch-arm on=VLLM_HPU_MTP_DRAFT_GRAPH=1` through the server's runtime switch
   file: per level (32, then 1 stream) one warm round per arm, then off, on, off, on, off, on (128 tokens). No
   server-to-server spread; at 1 stream the off and on texts must be the same (the drafts are the same).
3. `mtp3-graph`: a server started with the flag on (the production form), the same levels and rounds.
References without the flag: build 3 (00178) 39.4 to 40.6 ms per token at 32 streams, 24.5 to 27.9 at 1 stream; with
the MoE chunk (00161) 38.0 at 32. Build 4's async MTP k = 2 (00186): 33.7 at 32 streams.

**What to read when 00194 ends** (`bin/farm fetch mtp-draft-graph 00194 00194-real8-draft-graph-switch` if the
waiter is gone): `headline.md` (rows `mtp3-switch/off`, `mtp3-switch/on`, `mtp3-graph`: ms per step, tokens per step,
ms per token, acceptance per position), `mtp3-switch/probe.log` (MEAS / SPEC / TEXT lines per arm; a `SWITCH` line
means a round may not be its arm), `mtp3-switch/server0.log`: `Runtime switches from` lines, the line `Flash-Next MTP:
the 3 draft passes of a decode step run in one HPU graph`, no traceback or `PT_DEVMEM`, the `MTP step timing` lines:
kind `verify` (flag off: laps `pass0`, `pass0_wait`, `pass1_meta`, `pass1`, `pass2_meta`, `pass2`, `final_wait`)
against `verify+graph` (flag on: `graph_positions`, `graph_meta`, `graph`, `final_wait`) at rows 128 (32 streams)
and 4 (1 stream); the draft span is the `draft` value, the timer's JSON in `mtp_step_timing_r*.jsonl`. The
coordinator wants: the draft span per step and ms per token at 32 and 1 stream with and without the flag, the
rounds, acceptance per position.

## Design (VERIFIED in code unless marked)

- `MtpDraftLoop(pass_forward, draft_tokens, k, max_model_len)`: pass 0 over the verification rows, `index_select` of
  each request's last accepted row, LM head + arg max, then passes 1 .. k-1 (the previous ids `.int()`, positions + j
  clamped at `max_model_len`, the previous pass's multi-stream hidden), each with its own metadata; returns
  `[batch, k]`. The same ops in the same order as the per-pass `propose`.
- The proposer (`_propose_in_one_graph`) takes the base positions (the runner's books with
  `VLLM_HPU_MTP_DRAFT_LOOP=1`, else a device read), builds the metadata of passes 1 .. k-1 with
  `prepare_attn_metadata` (as today, only before pass 0 instead of between the passes), then calls the loop once.
  The loop is wrapped with `htorch.hpu.wrap_in_hpu_graph(loop, disable_tensor_cache=True)` in lazy mode (the
  parameters of `_maybe_wrap_in_hpu_graph`). Each pass calls `type(adapter).forward(adapter, ...)`: the draft
  adapter's own forward (metadata processing, RoPE tables, forward context per pass), not the graph-wrapped
  `adapter.forward`, so the passes are captured into the loop's graph, not into graphs of their own.
- Graph key (the bridge's `graphs.py` `input_hash`): tensors by dtype and shape, other values by value; the loop
  takes no Python number that changes per step (the batch size comes from `last_token_indices.shape`).
- Pool writes of a pass: the draft KV cache (`VLLMKVCache.forward`: `cache.index_copy_`), the QSA key ring
  (`ring_write`: `index_copy_`) and the compressed keys (`cache.index_copy_`): torch ops that the lazy graph orders
  against the next pass's reads (write-then-read already happens inside one QSA forward on the card). No custom op
  writes a pool in the draft layer with QSA modes 0 and 1, so rule 10b does not apply (mode 2's K1 kernel would:
  INFERRED risk; mode 2 is not served). Rule 11: the loop writes no persistent tensor itself.
- Not taken: drafts after a prefill (their shapes follow the prompt bucket) and the draft-rank log. Card rule 9: the
  ids leave the graph in the arg max's own dtype (int64 by default, as today); the int32 narrowing feeds the next
  pass inside the graph.
- The proposer module now logs through the "vllm" logger (`vllm_gaudi.extension.logger`): its module logger had no
  handler, so its lines never reached a server log.

## Card results

**00230 `real8-chain-b5b`** (12:01 to 13:00 UTC, exit 0, 9 arms, the only failed gate `tiny-chain-vp`; build 5's tree
plus the chain, engine-step's VLG and async-sched's M1; build 5's serve env, every server with async scheduling
(mode 2, MTP timer 0), dense MoE from 1 token (late-bound `VLLM_HPU_MOE_DENSE_MIN_TOKENS=1`), MTP k = 2, greedy,
levels 32 and 1, 2 rounds per arm; results `reports/mtp-draft-graph/card/00230-real8-chain-b5b/`). The placement was
on, with the place.sh bug that takes 16 CPUs from each later server (144, 128, 112, 96 of 152): read only A/Bs on one
server as clean. MEASURED, steady ms per token, mean of 2 rounds:

| server (CPUs) | arm | 32 streams | 1 stream |
|---|---|---|---|
| cuts (144) | off / on (graph) / md (I01) / on-md / all (+VP) | 27.91 / 27.14 / 27.83 / 27.13 / 27.51 | 15.92 / 15.07 / 15.03 / 14.23 / 15.70 |
| plain (128) | off / on / md / on-md / all | 28.51 / 27.68 / 28.07 / 27.85 / 27.79 | 15.70 / 14.84 / 15.32 / 14.54 / 15.79 |
| chain, cuts (112) | on-md / **chain-md** / vlg / **chain-vlg** / all | 27.22 / **24.94** / 26.59 / **24.99** / 26.57 | 14.08 / **13.03** / 14.04 / **12.68** / 15.40 |
| M1, cuts (96) | m1off (chain-md) / m1on | 24.84 / 26.06 | 12.73 / 13.50 |

- **The chain:**
  - Stage A (chain-md against on-md): -2.28 ms per token at 32 streams (-8.4 %, about -4.7 ms per step at 2.29
    tokens per step); -1.05 at 1 stream (-7.5 %).
  - Stage B on the verify graph's ids (chain-vlg against vlg): -1.60 at 32 streams (-6.0 %); -1.36 at 1 stream
    (-9.7 %).
  - Best at 1 stream: chain-vlg, 12.68 ms per token = 78.9 tok/s per stream.
  - The M1 server repeats chain-md: 24.84 / 12.73.
  - Same 1-stream text in all four chain and non-chain arms, same acceptance (2.309 tokens per step, 55 drafts at
    1 stream; 2.25 to 2.33 at 32 streams). No traceback.
  - On all 8 ranks the log says stage A and stage B were chained; the only refusal logged was steps with a prefill.
  - The gain is about 40 % of the INFERRED 12 ms per step: the host read now waits for the chain graph's device
    time too.
- **The chain graph's copied inputs:** the tiny gates with `VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS=1` show one copy
  flush per replay.
  - Stage A: 4 tensors, the verification's hidden rows `[rows, H]` bf16 and three small int32.
  - Stage B on VLG: the same plus the `[rows, 1]` view of the verify ids. The `_base` unwrap does not take on the
    card.
  - engine-step's `runner._verify_ids_raw` (`bd2e731b`, their kvargs branch) gives the raw ids: read it in
    `_mtp_chain_issue` when it lands.
- **VLG** (engine-step) against on-md: -0.6 ms at 32 streams, flat at 1 stream.
- **VP (`VLLM_HPU_VOCAB_PARALLEL_LOGITS=1`, not this track's):** wrong or tie-different greedy ids.
  - `tiny-chain-vp` FAILED in its non-speculative `[hc]` mode: a 700-token prompt diverges at token 4.
  - On the real model at 1 stream it changes the greedy text and lowers tokens per step from 2.29 to 2.10.
  - So the chain-all arm was dropped. engine-step is checking the card's argmax tie rule (job 00252).
- **Draft graph + I01** against off (same server): -0.78 / -0.66 ms at 32 streams, -1.69 / -1.16 at 1 stream.
- **REPLICATED_FC + REPLICATED_EMBEDDING** (cuts against plain, cross-server, so 16 CPUs and server spread
  confound it): -0.6 to -0.7 ms at 32 streams, within noise at 1 stream.
  - KV capacity: 2,296,169 tokens with REPLICATED_EMBEDDING against 2,378,571 without (-82,402 tokens, -3.5 %).
- **async-sched's M1:**
  - `tiny-m1` passed: 208 of 223 steps advanced on the device and checked, 0 differences.
  - On the server, m1on against m1off: +1.22 ms at 32 streams (+4.9 %), +0.77 at 1 stream (+6.0 %).
  - About 87 % of the steps used device-made inputs; texts were identical.
  - It is an enabler for M2's early issue, not a gain by itself.
- engine-step's IR logs of `tiny-chain-vlg` (their census of the step's cuts) are in `tiny-chain-vlg/habana_logs`.

### Earlier card results

**00194 `real8-draft-graph-switch`** (06:00 to 06:22 UTC, exit 0, all three arms ok; TP8 + EP8; build 4's code and env;
MTP k = 3; greedy; 128 tokens; results `reports/mtp-draft-graph/card/00194-real8-draft-graph-switch/`, `headline.md`;
analysis `tools/ab_summary.py`, `tools/timer_ab.py`). MEASURED:

1. Tiny model at TP8, flag on, `--spec 3`: PASSED (`hc` and `hc+mtp3` on the fused verify rung: tokens equal the
   reference). The workers log `Flash-Next MTP: the 3 draft passes of a decode step run in one HPU graph
   (VLLM_HPU_MTP_DRAFT_GRAPH=1, lazy mode True)`.
2. One server, the flag switched per round (`mtp3-switch`), steady ms per token (ms per step):

| level | off (per-pass) rounds | on (one graph) rounds | mean off | mean on | change |
|---|---|---|---|---|---|
| 32 streams | 35.94 / 35.53 / 40.09 | 38.76 / 37.69 / 38.65 | 37.19 (97.8) | 38.37 (99.9) | +2.1 ms per step, inside the round spread |
| 1 stream | 33.38 / 29.47 / 30.29 | 28.00 / 29.29 / 28.39 | 31.05 (86.4) | 28.56 (79.5) | **-6.9 ms per step, -8 % per token** (pairs -15.0 / -0.5 / -5.3) |

   Acceptance at 32 streams off 0.756 / 0.533 / 0.384, 0.741 / 0.545 / 0.405, 0.745 / 0.526 / 0.371; on 0.747 /
   0.526 / 0.382, 0.747 / 0.532 / 0.383, 0.748 / 0.537 / 0.391. At 1 stream all six rounds: 2.822 tokens per step,
   0.756 / 0.644 / 0.422, 45 drafts, and stream 0's text identical: **the drafts are the same on the card**.
   Every switch was taken (8 worker lines per switch, `switch_lines`).
3. The MTP timer (steady records: the last two per rank and key, median of the 8 ranks):

| | 1 stream (4 rows) | 32 streams (128 rows) |
|---|---|---|
| draft, per-pass | 17.1 ms (pass0 3.7, meta 0.8 + 0.7, pass1 2.8, pass2 2.7, final wait 6.4) | 17.7 ms (pass0 4.4, meta 1.1 + 1.0, pass1 3.4, pass2 3.4, final wait 3.9) |
| draft, one graph | **12.4 ms** (positions 0.6, meta 1.1, graph 6.3, final wait 4.6) | **15.2 ms** (positions 0.7, meta 1.5, graph 8.2, final wait 4.6) |
| verify, per-pass rounds / graph rounds | 65.6 / 67.5 ms | 78.0 / 81.5 ms |

   The loop graph's first three 25-step records at 32 streams hold its compiles (draft 54 / 45 / 27 ms: the warm
   round and the first measured round, new batch sizes as streams finish), which made the `on` arm's first 32-stream
   round slow (pauses up to 8.2 s). The verification span of the graph rounds was 1.9 / 3.5 ms longer (cause not
   known; INFERRED candidates: both graph sets live in the switch server, allocator state after the loop's replay):
   it cancels the draft's 2.5 ms at 32 streams.
4. A server started with the flag on (`mtp3-graph`): 32 streams 36.87 / 38.78 / 34.52 ms per token (mean 36.72,
   95.6 ms per step; acceptance 0.73 to 0.75 / 0.53 to 0.55 / 0.37 to 0.40); 1 stream 21.03 / 20.67 / 21.34 (mean
   21.01 = 47.6 tok/s, 57.2 ms per step; acceptance 0.826 / 0.565 / 0.391). Its 1-stream verification span was 47.8 ms
   against 65.6 ms on the switch server in both arms: server-to-server spread (the host-bound 1-stream step), not
   the flag. Draft span 9.2 ms (1 stream), 13.2 ms (32 streams). Device memory about 66 GB per card in both servers,
   no `PT_DEVMEM`, no traceback.

**Why short of the INFERRED 7 to 10 ms per 32-stream step:** merging the passes keeps every recipe launch (each
collective splits a captured graph into recipes, and the bridge's host cost is per launch: about 150 us, BRIEF
22:23 and 00174); it removes only the per-replay overhead of the two extra replay calls and the lazy LM-head and
arg-max segments between them. The single replay still issues 8.2 ms at 32 streams.

**Next levers for the draft (not built):** fewer collectives per draft pass (`VLLM_HPU_MTP_REPLICATED_FC=1`, the
collectives track's flag: the two fc all-gathers of each pass), engine-step's I01 for the later passes' metadata
(`VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1`: 1.1 to 1.5 ms; CPU-proven together with the flag, `bf7cf32c`), k = 2 (one
pass fewer), the bridge's per-launch path (native-replay). A follow-up job is ready on the test branch:
`mtp2-async-switch` (the A/B on build 4's recommended serving form, `HEADLINE_ASYNC_MTP_TIMING=0`) and
`HEADLINE_SWITCH_SET=graph-md` (the 2 x 2 with I01).
