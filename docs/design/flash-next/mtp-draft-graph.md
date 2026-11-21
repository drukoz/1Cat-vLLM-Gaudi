# Flash-Next MTP: the k draft passes in one HPU graph (track `mtp-draft-graph`)

This file is the track's report. The harness of the track's agent does not write report files under `reports/`, so
the handoff lives here, on branch `feat/flash-next-gaudi-mtp-draft-graph` (worktree
`/Volumes/Storage/Claude/gaudi-work/wt/mtp-draft-graph`). Raw CPU runs and card logs:
`/Volumes/Storage/Claude/gaudi-work/reports/mtp-draft-graph/` (`cpu/`, `card/`). Labels: MEASURED (card, with its
file), VERIFIED (CPU test or code line), INFERRED.

## Handoff (if this agent stopped)

Last update: 2026-11-21 05:15 UTC (`date -u`), first agent of the track (Opus 5.5), started 04:35 UTC.

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
- Card: nothing yet. Plan below.

**Card plan (integration agent's advice, 05:05 UTC).** Test branch `feat/flash-next-gaudi-mtp-draft-graph-b4` =
integration build 4 (`integration/card-build-20261120b` at `7901b089`; its plugin code is final at `bcaa3ed6`) + this
branch. One all-card job running `tools/flash_next/rental/mtp_draft_graph_ab.sh` (on the test branch): (1) the tiny
model at TP8 with the flag on, `--spec 3` (headline_job.sh arm `tiny`: the job ends if tokens differ); (2) the real
model, MTP k = 3, flag off; (3) the same with the flag on. `HEADLINE_LEVELS=32,1 HEADLINE_ROUNDS=2
VLLM_HPU_MTP_TIMING=1 HEADLINE_MTP_FALLBACK=0` and build 4's env (`VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1
VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2
VLLM_HPU_DECODE_CUT_INTERVAL=1000 VLLM_HPU_CANDIDATE_SAMPLING=1 VLLM_GRAPH_RESERVED_MEM=0.4
VLLM_HPU_PREFIX_ALIGN_SPLIT=1 VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 VLLM_MOE_CHUNK=64,8192
VLLM_MOE_TOKEN_BOUNDARY=512,1000000 PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`, the MTP arm adds
`VLLM_HPU_GDN_CORE_VERIFY=1 VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_SPEC_SAMPLED_VERIFY=2`); no
async scheduling (not card-proven: async-sched's 00179 decides). References without the flag: build 3 (00178) 39.4 to
40.6 ms per token at 32 streams, 24.5 to 27.9 at 1 stream; with the MoE chunk (00161) 38.0 at 32.

**What to read when the job ends.** Per arm, `server0.log`: the line `Flash-Next MTP: the 3 draft passes of a decode
step run in one HPU graph` (the path is on), no traceback, no `PT_DEVMEM`; the `MTP step timing` lines (the draft span
and its laps: `graph_positions`, `graph_meta`, `graph`, `final_wait` with the flag; `pass0`, `pass0_wait`,
`pass1_meta`, `pass1`, ... without); `probe.log` / `scaling.json` (ms per token, tokens per step, acceptance per
position: with the flag it must equal the flag-off arm's within noise, which shows the drafts are right on the card).

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

(none yet)
