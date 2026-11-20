# Flash-Next MTP on the card: state, card results and handoff (track `mtp-card-ready`)

This file is the track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `feat/flash-next-gaudi-mtp-card-ready`. Raw card logs:
`/Volumes/Storage/Claude/gaudi-work/reports/mtp-card-ready/card/`; job scripts: `reports/mtp-card-ready/job/`.

## Handoff (if this agent stops)

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

**Branch state.** `9d013a36` and `317c33c5` are committed; they are pushed when the CPU engine run on the fixed
tree is green (`reports/mtp-card-ready/cpu/fix2-engine.log`; a k = 3 engine rehearsal against k = 0 was the
quick check). spec-lookup, qsa-fused and engine-step build on this branch and wait for the push: tell the
coordinator the head (engine-step's agent id is `ad4fd8b387ee2314a`).
Not yet verified: commit `9722b888` (WIP, `VLLM_HPU_MTP_CHUNK_KV`, default 0, inert at 0). Needed for every
prompt above about two attention blocks on an MTP server with prefix caching (the scheduler ends a prefill
chunk at the last cacheable block boundary, one block earlier with MTP; the block is 512 tokens at TP8 with
MTP, 896 at TP4 + EP4 without speculation: read it from the pair's log), so the draft layer has KV for the
whole prompt. Its engine test is `test_the_draft_runs_over_every_chunk_of_a_chunked_prompt`.

**Card jobs of this track**
- 00021 ended 19:54 UTC (exit 0). Everything is fetched: `reports/mtp-card-ready/card/mtp3-fix/` (and
  `00021-mtp3-fix/` for the early copies): `mtp3-torchverify`, `mtp3-fusedverify` (failure 6), `base-nospec`
  (reference), `run1` (tiny, failure 7), `run2` (tiny, PASSED), `run3` + `mtp3-retry` (real model, the numbers).
- 00022 is cancelled; its place in the queue went to the matched 4-card pair (TP4 + EP4, 1 / 8 / 32 streams,
  256 tokens):
  - **00099 `real4-nospec-ref`** (any:4, 30 min, commit `9d013a36`): the reference.
  - **`real4-mtp3-timed`** (any:4, 35 min, commit `317c33c5`, `--mtp 3`, `VLLM_HPU_GDN_CORE_VERIFY=1`,
    `VLLM_HPU_MTP_TIMING=1`): the id is in `reports/mtp-card-ready/card/submit-real4-mtp3-timed.log`.
  The coordinator bumps both once it has the ids. The `farm serve` commands that submitted them wait in the
  background of the agent's session and copy the results when the jobs end; if the agent is gone:
  `bin/farm ledger | grep -w mtp-card-ready`, then `bin/farm fetch mtp-card-ready <id> <id>-<name>`.
  What to read: `scaling.json` / `probe.log` (ms per step = `steady_ms_per_token_p50 x tokens_per_chunk_p50`,
  tokens per step = `tokens_per_chunk_p50`), the `SpecDecoding metrics` lines of `server0.log` (per-position
  acceptance per 10 s window), the `MTP step timing` lines (Ft, Fd, the draft passes), the TEXT lines. Report
  to the coordinator: Ft, Fd and acceptance per position at 1, 8, 32 streams against the reference; then
  `python tools/flash_next/mtp_k_choice.py` for the best k.

**Pending, in order**
1. The pair above: read, report, write the numbers here.
2. Tiny model, `any:2`, 12 minutes: `farm job mtp-card-ready tiny2-mtp-arms any:2 12m -- bash
   tools/flash_next/rental/mtp_tiny_card.sh 3` (arms: the tree's defaults with the int64 index; then
   `VLLM_HPU_MTP_DRAFT_LOOP=1` + `VLLM_HPU_MTP_CHUNK_KV=1`). spec-lookup's job 00097 runs accepted counts 1 to 3
   on the tiny model with an oracle: read its result.
3. 32 streams with the qsa-fused data flow: merge `origin/feat/flash-next-gaudi-qsa-fused` into this branch (it
   contains the program head; `git merge-tree` shows one conflict, in `hpu_flash_next_mtp.py`), flags
   `VLLM_HPU_QSA_FUSED`, `VLLM_HPU_QSA_INDEXER_SHARD`, `VLLM_HPU_QSA_TABLE_WIDTH` as its A/B pair 00062 / 00063
   confirms them; then an `any:4` MTP arm at 32 streams.
4. I07 sampled verification on the card (the owner's agents sample: temperature 1.0, top_k 20, top_p 0.95):
   `VLLM_HPU_SPEC_SAMPLED_VERIFY=1`, acceptance and step time with the server's default sampling. engine-step
   will provide `candidate_probs(values, temperature, top_k, top_p)` (the processed distribution on the 64
   largest logits per row, `vllm_gaudi/v1/sample/candidate_sampling.py` on its branch): build the sampled
   rejection path so it can take that in place of the full-vocabulary distribution.
5. I02 (the timer shows what each draft pass costs; engine-step's I01 removes the drafter's per-step
   paged-attention buffers), I37 memory (the k = 3 pool is 6.147 GiB per card at TP8; read `Maximum
   concurrency`), I43 / k = 4 and 5 (`tools/flash_next/mtp_k_choice.py`).
6. Merge proven steps per the protocol: merge `origin/feat/flash-next-gaudi` into the branch (one conflict, in
   `hpu_flash_next_mtp.py`), `bin/gate <worktree> --tools`, PR to `feat/flash-next-gaudi`.

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

## How the jobs of this track run (late-bound)

`farm`'s stage-1 form: the job's `cmd.sh` runs `~/farm/mtp-card-ready/job/plan.sh` (read when the job starts),
which starts `farm-base:latest` containers with the bundle `~/farm/mtp-card-ready/next` mounted at `/bundle`; the
container script `job/serve_probe_spec.sh` copies the bundle's Python files over the image's `vllm_gaudi`
(Python-only changes since `e238f84`; a native change needs a `farm serve` image instead). Replace a file on the
node only with `scp` to `name.new` and `mv` (a running bash keeps reading the old file). The probe never sends
`min_tokens`.
