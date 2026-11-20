# Flash-Next MTP on the card: state, card results and handoff (track `mtp-card-ready`)

This file is the track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `feat/flash-next-gaudi-mtp-card-ready`. Raw card logs:
`/Volumes/Storage/Claude/gaudi-work/reports/mtp-card-ready/card/`; job scripts: `reports/mtp-card-ready/job/`.

## Handoff (if this agent stops)

Last update: 2026-11-20 19:50 UTC (third agent of the track, started 19:00 UTC).

**State in one paragraph.** MTP speculative decoding (k = 3) runs correctly on the card on the TINY model (TP8,
mode `hc`, fused GDN verification): tokens equal the non-speculative server for prompts of 1 to 2,600 tokens
(job 00021, run 2, 19:44 UTC). Two card failures of one family were found and fixed on the way (failures 6 and 7
below). The real model starts with the drafter (healthy after 435 s) but had not decoded with speculation when
this was written: its first run with both fixes was loading inside job 00021's hold (ends 19:56). No real
acceptance or ms-per-step number exists yet; the tiny checkpoint accepts nothing (random weights), so only the
all-rejected verification path is card-proven.

**Done and verified**
- CPU (commits up to `77af8a5e`, previous agents): engine start with the drafter, the drafter's block table, two
  verification crashes at real block sizes, the recurrent pool's page count with MTP, k = 1..5, sampled verification
  behind `VLLM_HPU_SPEC_SAMPLED_VERIFY` (0 = a clean HTTP 400 for a sampled request), `VLLM_HPU_MTP_DRAFT_LOOP`,
  `spec_probe.py`, `tiny_smoke.py --spec K`, `mtp_k_choice.py`. Gate lines: `reports/mtp-card-ready/gate/`.
- Card, real model TP8 + EP8, k = 3 (job 00021 attempts 1 and 2): start OK,
  `Recurrent state pool: 1280 slots x 4.91 MiB per slot = 6.147 GiB reserved; 52.11 GiB left for attention blocks`.
- Card failure 6 (real model, first prompt, all 8 ranks): `Neither storage attached to input tensor, not its view`
  from `graph.replayV3`; the failing tensor is the output of `model/1/ple/hpu__slice_insert` (VERIFIED from the
  failing-graph dump). `ple_layer.py` `_hpu_short_conv` wrote the widened conv state through
  `view[:, :state_len, :]`. Fixed in `f0e7d02f` (PLE, and the GDN conv prefill and plain-decode writes): whole
  rows with `index_copy_`. Card-verified on the tiny model (run 1: the prompt and the first verification pass).
- Card failure 7 (tiny model, the first draft after a decode step, run 1, both verification rungs):
  `Empty tensor optional` when `propose_eagle_decode`'s gathers are flushed. The MTP aux hidden state was
  `_mtp_hidden_buffer[:T].view(...)`: a view of a persistent buffer that upstream fills inside the forward with
  `buffer[:T].copy_(multi)`; outside the HPU graph that buffer version has no value. Fixed in the working tree
  (`vllm_gaudi/models/qwen4_exp/__init__.py`, `_hpu_tap_mtp_multi`): the aux output is the final mixer's own
  output tensor, and on an HPU device the buffer is dropped so the in-graph slice copy does not run.
  Card-verified on the tiny model (run 2).
- **Card rule (new): inside an HPU graph a persistent tensor is never written, and never leaves the graph,
  through a slice of itself.** Read whole rows with `index_select` and slice the copy; write whole rows with
  `index_copy_`; return computed tensors, not views of module buffers. CPU model of the write half:
  `tests/unit_tests/models/flash_next/slice_write_probe.py`.
- Card, reference without speculation (same bundle, TP8, no `min_tokens`): 41.6 ms per step at 1 stream, 45.9 at
  8, 84.1 / 84.6 at 32 (85.1 ms per token effective, 346 tok/s total). `argmax -> .to(int32)` in one lazy graph
  is sound at 8 and 32 rows in the sampler (stream 1's text is its own story): card rule 9 does not hit it.

**In the working tree, not committed yet** (the node's bundle `~/farm/mtp-card-ready/next` has them, see its
`BUNDLE.txt`; commit and push when `tests/unit_tests/models/flash_next/test_engine_mtp.py` is green, log
`reports/mtp-card-ready/cpu/fix2-engine.log`)
- `vllm_gaudi/models/qwen4_exp/__init__.py`: failure 7 (above).
- `vllm_gaudi/ops/causal_conv1d_pytorch.py`: the two new `index_copy_` calls take `.to(torch.long)` indices
  (`f0e7d02f` broke the CPU engine at the first prefill of a speculative config: `index_copy_(): Expected a long
  tensor for index, but got Int`; found by spec-lookup, which waits for this fix). The card ran the int32 index;
  the widened one needs its tiny-model card run.
- `tests/unit_tests/models/flash_next/test_state_slice_writes.py`: the GDN tests with int32 and int64 indices.
- Not yet verified: commit `9722b888` (WIP, `VLLM_HPU_MTP_CHUNK_KV`, default 0). Needed for every prompt of
  1,024 tokens or more on an MTP server with prefix caching (the scheduler ends a prefill chunk at the last
  cacheable block boundary, one block earlier on an MTP server), so the draft layer has KV for the whole prompt.
  Its engine test `test_the_draft_runs_over_every_chunk_of_a_chunked_prompt` runs in the same file.

**Card jobs of this track**
- 00021 `real8-mtp3-fixed-torch-then-fused-verify` (all 8 cards, hold ends 19:56 UTC). Results on the node under
  `~/farm/mtp-card-ready/results/mtp3-fix/`: `mtp3-torchverify`, `mtp3-fusedverify` (failure 6), `base-nospec`
  (reference), `run1` (tiny, failure 7), `run2` (tiny, PASSED), `run3` + `mtp3-retry` (real model k = 3, fused
  verification, 1 and 32 streams, 128 tokens: only if the load finished before 19:53). Copies:
  `reports/mtp-card-ready/card/00021-mtp3-fix/`. Fetch what is missing with `bin/node-scp` before 04:00 UTC.
- 00022 `real8-mtp-k4-k5-scaling` (all 8 cards, 45 min, waiting): runs `job/plan2.sh`, which exits at once
  unless `results/mtp3-fix/mtp3-torchverify/scaling.json` holds a k = 3 result (it does not). Cancel it
  (`bin/farm cancel mtp-card-ready 00022`) and submit k = 4 / k = 5 as `any:4` jobs once k = 3 has numbers.

**Pending, in order**
1. Real model k = 3, greedy: accepted tokens per step and per draft position, ms per step, effective ms per token
   at 1 and 32 streams against the reference above; texts of streams 0 and 1 (the probe prints them). If run 3
   of job 00021 did not get there: a new job. `any:4` (TP4 + EP4) answers acceptance and correctness; the TP8
   step time needs `all`. Use `farm serve mtp-card-ready <name> any:4 30m --mtp 3 --levels 1,8,32 --tokens 128`
   from the committed head (per-commit image), or the late-bound scripts in `reports/mtp-card-ready/job/`. Ask
   the coordinator for a bump. Tell the coordinator the numbers at once (SendMessage to `main`). Expect a slow
   verification step at 32 streams on the torch QSA path (qsa-fused measured about 125 ms of QSA device time at
   32 x 4 rows); report acceptance and step time separately; the next multi-stream arm runs with the qsa-fused
   data-flow change merged and its flags on.
2. Tiny model, `any:4` or `any:2`, 10-minute holds: the committed tree (the int64 index), then a prompt above
   1,024 tokens with prefix caching and `VLLM_HPU_MTP_CHUNK_KV=1`, then `VLLM_HPU_MTP_DRAFT_LOOP=1`.
3. I07 sampled verification (the owner's agents sample: temperature 1.0, top_k 20, top_p 0.95 from the
   checkpoint's generation_config, so this decides whether MTP gives real agents anything): on the card with
   `VLLM_HPU_SPEC_SAMPLED_VERIFY=1` and the probe's `--sampled 32:1.0`; coordinate with collectives
   (vocabulary-parallel top-k) and engine-step (the sampler stage).
4. I02 device-resident draft loop; I37 memory (the k = 3 pool is 6.147 GiB per card; read `Maximum concurrency`);
   I43 choice of k (`tools/flash_next/mtp_k_choice.py` with the measured step cost and acceptance).
5. Merge proven steps per the protocol (`bin/gate <worktree> --tools`, PR to `feat/flash-next-gaudi`); merge
   `origin/feat/flash-next-gaudi` (head `b0cd0bb`) into the branch first. spec-lookup builds on this branch.

## Card results

| Job | What | Result |
|---|---|---|
| 00008 (coord) | first MTP start, program head | died in `load_model` (`image_token_index`), fixed in `8951c003` |
| 00021 attempt 1 | TP8, k = 3, torch verification | healthy after 435 s; first prompt: failure 6 (PLE slice write) |
| 00021 attempt 2 | TP8, k = 3, fused verification, host-book draft positions, sampled verify on | healthy after 435 s; same failure 6 |
| 00021 reference | TP8, same bundle, no speculation, no `min_tokens`, 256 tokens | ms per step: 41.6 (1 stream), 45.9 (8), 84.1 / 84.6 (32); 85.1 ms per token at 32 streams = 11.7 tok/s per stream |
| 00021 run 1 | tiny model TP8, `hc` then `hc+mtp3`, failure-6 fix | `hc` passed (7.8 ms/token); both speculative rungs: failure 7 (aux hidden view of the buffer) |
| 00021 run 2 | tiny model TP8, `hc+mtp3`, both fixes | PASSED on the fused verification rung: tokens equal the non-speculative server for 1, 7, 33, 100, 700 and 2,600-token prompts; 0 of 270 drafts accepted (random weights); 22.0 ms/token against 8.2 without speculation |

## How the jobs of this track run (late-bound)

`farm`'s stage-1 form: the job's `cmd.sh` runs `~/farm/mtp-card-ready/job/plan.sh` (read when the job starts),
which starts `farm-base:latest` containers with the bundle `~/farm/mtp-card-ready/next` mounted at `/bundle`; the
container script `job/serve_probe_spec.sh` copies the bundle's Python files over the image's `vllm_gaudi`
(Python-only changes since `e238f84`; a native change needs a `farm serve` image instead). Replace a file on the
node only with `scp` to `name.new` and `mv` (a running bash keeps reading the old file). The probe never sends
`min_tokens`.
