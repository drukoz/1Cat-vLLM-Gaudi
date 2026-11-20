# Flash-Next MTP on the card: state, card results and handoff (track `mtp-card-ready`)

This file is the track's report. The harness of the track's agents does not write report files under
`reports/`, so the handoff lives here, on branch `feat/flash-next-gaudi-mtp-card-ready`. Raw card logs:
`/Volumes/Storage/Claude/gaudi-work/reports/mtp-card-ready/card/`; job scripts: `reports/mtp-card-ready/job/`.

## Handoff (if this agent stops)

Last update: 2026-11-20 19:30 UTC (third agent of the track, started 19:00 UTC).

**State in one paragraph.** MTP speculative decoding (k = 3) now starts on the real model at TP8 + EP8 (server
healthy after 435 s, twice: the five start failures are fixed on the card, VERIFIED by job 00021). It has not
decoded a token on a card yet: the first prompt of a speculative server failed in the PLE conv state write
(card failure 6, below). The fix is in the tree and on the node; its first card run (tiny model, TP8) is running
inside job 00021's hold as this is written. No acceptance or ms-per-step number exists yet.

**Done and verified**
- CPU (commits up to `77af8a5e`, previous agents): engine start with the drafter, the drafter's block table, two
  verification crashes at real block sizes, the recurrent pool's page count with MTP, k = 1..5, sampled verification
  behind `VLLM_HPU_SPEC_SAMPLED_VERIFY`, `VLLM_HPU_MTP_DRAFT_LOOP`, `spec_probe.py`, `tiny_smoke.py --spec K`,
  `mtp_k_choice.py`. Gate lines: `reports/mtp-card-ready/gate/`.
- Card, job 00021 (TP8, k = 3, bundle `77af8a5e` as Python overlay on image `farm-base` = `e238f84`): start OK,
  `Recurrent state pool: 1280 slots x 4.91 MiB per slot = 6.147 GiB reserved; 52.11 GiB left for attention blocks`.
- Card failure 6 (job 00021, both attempts, first prompt, all 8 ranks): `RuntimeError: Neither storage attached to
  input tensor, not its view. Uniqueid: ... Strided Params Has Value: 0` from `graph.replayV3` of the prompt bucket
  `('prompt', 1, 128, 0)`. The failing tensor is the output of `model/1/ple/hpu__slice_insert` on every rank
  (VERIFIED from the failing-graph dump: the node's id per rank equals the failing id per rank). Cause:
  `ple_layer.py` `_hpu_short_conv` took `view = view[:, :state_len, :]` of the conv state that a speculative config
  widens, and `_save_conv_state` wrote it with `index_copy_`: an in-place write through a slice view of a
  persistent state. **New card rule: inside an HPU graph never write in place through a slice of a persistent
  tensor; read whole rows with `index_select`, slice the copy, write whole rows with `index_copy_`.**

**In the working tree, tested on CPU, first card run in progress** (commit them when the tiny model passes on
the card; the node's bundle `~/farm/mtp-card-ready/next` already has these five files, noted in its `BUNDLE.txt`)
- `vllm_gaudi/models/qwen4_exp/ple_layer.py`: `_widen_conv_state`, no slice of the state tensor.
- `vllm_gaudi/ops/causal_conv1d_pytorch.py`: the same for the GDN conv state under a widened state, in
  `hpu_causal_conv1d_fn` (prefill) and `hpu_causal_conv1d_fn_update` (plain decode). INFERRED to be the next
  failure (`linear_attn/hpu__slice_insert` is in the same failing graph); non-speculative behaviour unchanged.
- `vllm_gaudi/v1/spec_decode/flash_next_mtp_steps.py` `draft_argmax`, `vllm_gaudi/envs.py`
  `VLLM_HPU_MTP_DRAFT_IDS` (default 0; 1 = int32 draft ids from an fp32 id table, no int64 id on the device), used
  by `hpu_flash_next_mtp.py`; test `tests/unit_tests/worker/test_mtp_draft_ids.py`. For card rule 9 (topk ids
  narrowed to int32 in one lazy graph are wrong): whether `argmax` ids are affected is not measured; the upstream
  sampler does the same narrowing in every decode step, so it is probably sound. Set the flag only if the card
  shows draft positions 2..k collapsing at 8 or 32 streams while 1 stream is fine.
- Not yet verified: commit `9722b888` (WIP, `VLLM_HPU_MTP_CHUNK_KV`, default 0): its engine test was running
  when it was committed. Needed for every prompt of 1,024 tokens or more on an MTP server (the scheduler ends a
  prefill chunk one block early there), so before real agents: run
  `tests/unit_tests/models/flash_next/test_engine_mtp.py`, then a tiny-model prompt above 1,024 tokens on a card.

**Card jobs of this track**
- 00021 `real8-mtp3-fixed-torch-then-fused-verify` (all 8 cards, hold ends 19:56 UTC): attempts 1 and 2 failed at
  the first prompt (failure 6). Its tail is late-bound: a no-speculation reference of the same bundle
  (`results/mtp3-fix/base-nospec/`), then command files `~/farm/mtp-card-ready/job/run/NN.sh` run in order inside
  the container (`job/after_probe.sh`); `run/01.sh` = tiny model TP8, `--gdn hc --spec 3`, with the hotfix files.
  When it ends: `bin/farm fetch mtp-card-ready 00021 mtp3-fix` (or `bin/node-scp`), results under
  `reports/mtp-card-ready/card/`.
- 00022 `real8-mtp-k4-k5-scaling` (all 8 cards, 45 min, waiting): runs `job/plan2.sh`, which exits at once
  unless job 00021 left a k = 3 probe result. Cancel it (`bin/farm cancel mtp-card-ready 00022`) once a k = 3
  number exists and submit k = 4 / k = 5 as `any:4` jobs; do not let it take 8 cards for 45 minutes blind.

**Pending, in order**
1. Tiny model on the card until speculation runs clean (start, first prompt, drafts, verification, tokens equal
   to the non-speculative reference): `farm tiny`-sized jobs on `any:4` or `any:2`, 10-minute holds, bump by the
   coordinator; inside 00021's hold use `job/run/NN.sh`. Then a prompt above 1,024 tokens with
   `VLLM_HPU_MTP_CHUNK_KV=1`.
2. Real model k = 3: accepted tokens per step per draft position, ms per step, effective ms per token at 1 and 32
   streams against the same bundle without speculation; print the texts of streams 0 and 1 (the probe does:
   `job/spec_scaling.py` + pinned `job/decode_scaling.py`, no `min_tokens`). Tell the coordinator at once
   (SendMessage to `main`). Expect the verification step to be slow at 32 streams on the torch QSA path (qsa-fused:
   about 125 ms of QSA device time at 32 x 4 rows): report acceptance and step time separately; the next
   multi-stream arm runs with the qsa-fused data-flow change merged and its flags on (`any:4`).
3. I07 sampled verification on the card (`VLLM_HPU_SPEC_SAMPLED_VERIFY=1`, probe `--sampled 32:1.0`).
4. I02 device-resident draft loop (`VLLM_HPU_MTP_DRAFT_LOOP=1` is the first step, never run on a card).
5. I37 memory: read `Maximum concurrency` of the k = 3 server log; k = 3 pool measured: 6.147 GiB per card.
6. I43 choice of k: `tools/flash_next/mtp_k_choice.py` with the measured step cost and acceptance.
7. Merge proven steps per the protocol (`bin/gate <worktree> --tools`, PR to `feat/flash-next-gaudi`); merge
   `origin/feat/flash-next-gaudi` (head `b0cd0bb`) into the branch first. spec-lookup waits for the PLE fix.

## Card results

| Job | What | Result |
|---|---|---|
| 00008 (coord) | first MTP start, program head | died in `load_model` (`image_token_index`), fixed in `8951c003` |
| 00021 attempt 1 | TP8, k = 3, torch verification | healthy after 435 s; first prompt: failure 6 (PLE slice write) |
| 00021 attempt 2 | TP8, k = 3, fused verification, host-book draft positions, sampled verify on | healthy after 435 s; same failure 6 |

## How the jobs of this track run (late-bound)

`farm`'s stage-1 form: the job's `cmd.sh` runs `~/farm/mtp-card-ready/job/plan.sh` (read when the job starts),
which starts `farm-base:latest` containers with the bundle `~/farm/mtp-card-ready/next` mounted at `/bundle`; the
container script `job/serve_probe_spec.sh` copies the bundle's Python files over the image's `vllm_gaudi`
(Python-only changes since `e238f84`; a native change needs a `farm serve` image instead). Replace a file on the
node only with `scp` to `name.new` and `mv` (a running bash keeps reading the old file). The probe never sends
`min_tokens`.
