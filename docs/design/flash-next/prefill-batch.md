# Flash-Next on Gaudi2: batched prefill across requests (track prefill-batch)

Track `prefill-batch`, worktree `/Volumes/Storage/Claude/gaudi-work/wt/prefill-batch`, branch
`feat/flash-next-gaudi-prefill-batch`. Started 12:15 UTC 2026-11-21. Goal: several requests' prompt chunks in ONE
forward for Flash-Next's hybrid stack, behind a default-off flag. The track's report lives here (the harness refuses
report files under `reports/`); scratch tools are in `/Volumes/Storage/Claude/gaudi-work/reports/prefill-batch/tools/`.

## Handoff (if this agent stops) - updated 13:27 UTC 2026-11-21

- **v1 is built and CPU-proven; two card jobs are queued.** Flag `VLLM_HPU_PREFILL_BATCH=N` (default 0), switchable
  at run time through `VLLM_HPU_RUNTIME_SWITCH_FILE` (a server started at N; the switch file caps it, 0 = off).
- **Branches.** `feat/flash-next-gaudi-prefill-batch` (base `8ead88c7` = program head 4d66755e + prefill's work,
  pushed): design note, v1 (`41dae6de`), chunk KV fill shape (`0c482379`), runtime switch + merge counter + refusal
  under `VLLM_HPU_PREFILL_CHECKPOINT` (`3a8b710c`), card tools (`f3286aaf`, `818f1dcb`), engine test (`2d977977`),
  op counts (`e5bf9b8b`), switch hardening (`74c9cd41`). Card branch `feat/flash-next-gaudi-prefill-batch-b5`
  (pushed) = build 5 `eb21ac75` + the code and tool commits cherry-picked (the switch-list conflict resolved to build
  5's list + this flag); head `46c3d2ad`, image `farm-prefill-batch:46c3d2adb11c`; worktree
  `reports/prefill-batch/b5-tree`.
- **Card jobs (both `all`; the coordinator placed them after integration's 00263, about 18:00 to 21:30 UTC):**
  - `00257 tiny8-gate-b5` (25 min, image e79272b6dab4; same code): `python3 tools/flash_next/prefill_batch/
    tiny_gate.py --mtp 0,2`. Pass = every merged reference copy reproduces the HF reference, on equals off (near
    ties recorded), merged forwards in on and none in off. Read `summary.json`, `<server>/streams.json`.
  - `00267 real8-b5-pb-ab2` (75 min hold, replaced 00258): `ab_job.sh b5_ab.plan` under the placement wrap
    (`~/farm/prefill-batch/results/place/`, place_wrap 7b159d11 + place.sh 44273de4), 00243's settings
    (`reports/prefill/slot2/b5_common.env` + `b5_i01off.env`, MTP k = 2, graphs 64,000), `PB_ROWS=8`. Lines: burst
    probes `b2k` (context 2,048, K = 1, 2, 4, 8) and `b10k` (10,240, K = 1, 2, 4, knob 70,000); agent loops
    `a32-off`, `a32-on`, `a16-on`, `a16-off`; last (may run out of memory) `b10k8` (K = 8, knob 140,000) and
    `a32-on-g128k` (arm on, knob 128,000). Digest: `python3 -I reports/prefill-batch/tools/digest_ab.py <results>`
    (burst spans, saved per removed forward, agent tables, merge lines); per-forward costs from the rank-0
    `server_events_*.json` with `tools/bucket_cost.py`; memory `hlsmi_mem.csv`.
  - Submit waiters (my pids 36972 for 00257 and 56583 for 00267, files `reports/prefill-batch/logs/submit_*`) copy
    the results to `reports/prefill-batch/card/` when the jobs end; else `bin/farm fetch prefill-batch <id> <id>-<name>`.
- **To submit again from the card branch:** `farm sync` takes the committed HEAD of `wt/prefill-batch`; detach it at
  the b5 commit (`git -C wt/prefill-batch checkout --detach <b5 sha>`), submit, wait for the `farm: job` line, then
  `git checkout feat/flash-next-gaudi-prefill-batch` again (never two syncs of this track at once).
- **Condition (prefill):** the flag is refused while `VLLM_HPU_PREFILL_CHECKPOINT` is on, until prefill's per-row
  checkpoint maps ([groups, target_bs, nphysical_chunks]) land; then drop `prefill_checkpoint_on()` from the refusal.
- **PR:** after prefill's PR (its branch is my base) is merged: merge the program head, gate (`bin/gate`), open and
  merge (default off). Gate notes: `tests/unit_tests/test_qwen3_next.py` has two gated-FP8 tests red on the base too.
- **Split agreed with prefill (a9a36e0116ed6cb7d), sent to the coordinator at about 12:25 UTC.** prefill (a, one
  forward per turn, `VLLM_HPU_PREFILL_CHECKPOINT`): `vllm_gaudi/v1/core/*`, `compute_prefix_caching_block_indices`
  and the mamba / prefix-caching branch of `_form_prefill_batch` (made per row, [target_bs, K]),
  `ops/hpu_gdn_pytorch.py`, `ops/causal_conv1d_pytorch.py`, a new `ops/gdn_prefill_checkpoint.py`,
  `get_kv_cache_spec`, `tests/unit_tests/core/*`, `tools/flash_next/rental/prefill_*`. prefill-batch (b):
  `_can_merge_prefill_contents`, `_extract_prefill_batch_contents`, prompt bs buckets and `max_prefill_batch_size`,
  the batch-level parts of `_form_prefill_batch`, the prompt graph key for bs > 1, the multi-row prefill fields and
  `_save_ssm_state` in `models/qwen3_5.py`, the QSA prompt path for several rows (qsa-fused asked first),
  `ple_layer.py` per request, the flag, new test files only. Seam: the 2D prompt bucket [bs, T_bucket] of padded
  rows (the form every model path already has); (a)'s checkpoint metadata is per row with row-local offsets.
- **Design note**: below (sent to the coordinator 12:31 UTC; v1 approved 12:36 UTC).
- **Next**: read 00257 / 00258 when they end; report the per-forward or per-row answer (burst probe) and the agent-loop
  A/B to the coordinator; if merges are rare at 16K to 56K under the 64,000 graph bound, say what bound v1 needs and
  what it costs in KV capacity (the owner's trade). Then the PR of v1 (default off; gate: models/flash_next,
  platform, worker/test_prefill_batch_rules.py, test_qwen3_next.py with its two pre-existing red gated-FP8 tests).

## Design note (12:45 UTC 2026-11-21)

### 1. What a prompt forward costs today (MEASURED, 00150 / 00151, TP4 + EP4, rank-0 events)

Source: prefill's `reports/prefill/card/s2-deon/server_events_vllm-instance-588-*.json` (00150) and
`s2-deoff/...-589-*` (00151), read with `reports/prefill-batch/tools/bucket_cost.py`. A replayed forward's host call
returns as soon as the device queue has room; in a burst the forwards run back to back and the time from one
forward's start to the next one's start is the device's time per forward (pacing rule INFERRED from the pattern: the
first forward after an idle gap returns in 48 to 87 ms of host time, the ones after it take 175 to 282 ms each).
Medians of these paced periods (00150 / 00151):

| bucket (bs, query, context blocks of 128) | paced period, ms | n |
|---|---|---|
| 1, 128, 123 (about 10K context) | 105.9 / 104.2 | 3 / 3 |
| 1, 256, 123 | 138.0 / 135.0 | 14 / 13 |
| 1, 512, 123 | 177.9 / 179.8 | 12 / 11 |
| 1, 768, 123 | 235.2 / 233.8 | 13 / 12 |
| 1, 1024, 123 | 275.3 / 278.0 | 26 / 69 |
| 1, 1024, 0 | 207.3 / 204.0 | 4 / 4 |
| 1, 2048, 0 / 8 / 16 / 31 / 62 | 398 / 406 / 405 / 419 / 392 (00150) | 9 to 19 |

Fit (INFERRED from these medians): at context bucket 123, **t = about 80 ms + 0.192 ms per padded query token**
(128 to 1,024 tokens); at context 0, about 10 ms + 0.19 ms per token (two points). A decode step at the same
contexts costs about 40 ms (bs 16). So a 1,024-token turn chunk at 10K context is about 30 % "per forward" (the
80 ms) and 70 % per token. The 0.19 ms per token is far above the FLOP cost of the model at TP4 (a few
microseconds per token from the matmul shapes, INFERRED): the prompt path is neither compute-bound nor
fixed-cost-bound.

The burst of 00150 (08:55:59.85 to 08:56:07.2 UTC): all 16 agents' turns arrive within 10 ms, nobody decodes, the
steps carry only prompt chunks (one step: 8 chunks of 896 tokens; the next: 8 tails of 512 + 4 chunks of 896; ...),
each chunk is its own forward: 16 x (277 + 178) ms = 7.3 s of device time; TTFT p50 7.72 s, p99 9.77 s, ITL max
3.8 s (1,280 uncached tokens per turn; 00151 without the decode-end keep: 2,176, TTFT p50 8.30 s, p99 14.46 s).

### 2. Why merging is refused today (VERIFIED by reading, line numbers at 8ead88c7)

1. `hpu_model_runner.py:4117` `_can_merge_prefill_contents`: `if self.num_mamba_like_layers > 0: return False`
   (written for Mamba2 / Granite: `_form_prefill_batch` asserts one request under prefix caching and
   `granite_causal_conv1d_fn` asserts one padded row without it).
2. Even for attention-only models, Conditions A and B (`:4124`, `:4133`): a chunk with history (context > 0)
   never joins a non-empty batch, and a batch with history takes nothing more. Every agent turn has history.
3. The merge test calls the bucketing with the summed token lists and no context blocks
   (`bucketing_fn(combined_num_tokens, [])`): it never looks at the context or at the prompt-graph limit.
4. `max_prefill_batch_size = VLLM_PROMPT_BS_BUCKET_MAX`, default 1 (`:2613`), so the prompt bs buckets are [1]
   (00150 log: `bs:[1, 2, 1, 1]`).
5. Prefix caching: `_form_prefill_batch` builds the Mamba2 checkpoint maps for one request and asserts it
   (`:5400`, `:5402`). Flash-Next's GDN path does not read those maps (only `hpu_mamba_mixer2.py` does).
6. Code after the forward that assumes one request per prompt forward:
   - `qwen3_5.py:153` `_save_ssm_state` with more than one row uses `torch.nonzero(valid)` (a data-dependent shape
     and a host sync inside the captured forward); the one-row path is a fixed-shape `remainder` + `index_copy_`.
   - MTP: `propose_eagle_prefill` (`:11601`) shifts the target tokens with `token_ids.squeeze()` and
     `target_token_ids[:-1].copy_(token_ids[1:])` (`:11635`): for a [bs, T] batch that shifts rows, not tokens. The
     prefill-draft loop (`:11056`, "Currently, prefill is done one by one") and `prefill_batch_start_idx` assume
     each prompt batch is a contiguous run of input-batch rows.
   - `_use_graphs` (`:8627`): a prompt forward replays a graph only while `bs x query + context blocks x 128` stays
     under `VLLM_HPU_PROMPT_GRAPH_MAX_TOKENS`; `num_blocks()` counts the whole `block_list`, so a merged batch reaches
     the limit bs times sooner, and an unreplayed prompt forward costs about 1.7 s of host time (prefill 00014).

Already row-ready (VERIFIED by reading; the CPU tests check each): the GDN chunk rule's compile path takes
`prefill_num_seqs` uniform rows with row-local chunks and one initial state per row (`hpu_gdn_pytorch.py:2031`;
padding has g = 0 and beta = 0, so a row's state passes through its padding); the prompt conv's batched path gathers
each row's conv state at the row's real length from `query_start_loc` (`causal_conv1d_pytorch.py:442`);
`_extract_metadata` loads one initial state per row (`qwen3_5.py:677-689`); the QSA indexer takes rows with per-row
context, block list and lengths (`qsa.py:332`, a Python loop over rows at `:383`); the QSA selection-bias attention
takes [rows, seq] with per-row context columns (`qsa_hpu_attention.py:215`); the PLE short conv runs over padded
rows with per-row lengths (`ple_short_conv_batched_torch`); `build_ple_inputs` takes several requests; logits
indices, sampling and prompt logprobs are per request in the 2D layout (attention models use bs > 1 prompt buckets).

### 3. The smallest first version (v1): `VLLM_HPU_PREFILL_BATCH=N` (default 0 = off; N = most rows per forward)

Merge the prompt chunks of one step into one forward when they have **the same padded shape** (the same query bucket
and the same context bucket: every row is padded exactly as its own one-row forward would be), the same logits class
(all complete their prompt, or none), and no row reads a state slot another row of the batch writes. Group sizes come
from the bs buckets {1, 2, 4, 8} (capped by N) with no padded rows (7 equal chunks = 4 + 2 + 1), and a group must fit
the prompt-graph limit (`bs x (query + context)` tokens; otherwise smaller groups, never an unreplayed forward).
Consecutive requests of the input batch only (no reordering in v1: the drafter and the async-scheduling paths pair
outputs with input-batch positions).

Since every row keeps its own padded shape, a merged forward does exactly the padded work of the forwards it
replaces: it can only remove per-forward costs (collective latencies, launches, per-layer fixed work), never add
padding. Changes: the merge rule and its guards in `_extract_prefill_batch_contents` / `_can_merge_prefill_contents`
(flag path only), the bs buckets, the fixed-shape multi-row `_save_ssm_state`, the per-row token shift in
`propose_eagle_prefill` (without it, under speculation, only chunks that complete no prompt merge), and prefill's
per-row prefix-caching maps (its side of the seam). One-row batches run today's code unchanged.

Stays serial in v1: chunks of different (query, context) buckets (one forward per distinct bucket group, not per
request); requests with structured output, prompt logprobs, LoRA or multimodal inputs; a row whose load slot another
row of the batch writes (a shared prefix computed in the same step); groups beyond the graph limit; DP and
KV-transfer setups.

### 4. Expected effect (INFERRED from the fit in section 1; the card decides)

The open question: is the 80 ms "per forward" part at 10K context paid once per forward or once per row? It is
mostly context-dependent (10 ms at context 0, 80 ms at bucket 123), so it may be per-row work of the QSA prompt path
(gathering the whole context's K/V and compressed keys per layer, dense attention over context + chunk), which
merging does not remove.

| case | today | v1, per-forward part shared | v1, per-forward part per row |
|---|---|---|---|
| 16-turn burst, TP4, 00150 shape (896 + 512 per turn, 10K context) | 32 forwards, 7.3 s (MEASURED) | 4 to 5 forwards, about 5.0 s (-31 %) | about 7.0 s (-4 %) |
| same, per-token floor (16 x 1,536 padded tokens x 0.192 ms) | | 4.7 s | 4.7 s |
| 32-turn burst, TP8, 1,024 new tokens, 16K to 56K context (00243's shape, about 2,300 uncached tokens per turn) | about 64 forwards (no TP8 prompt-forward data yet) | about -20 to -35 % | about -5 % |

Decode stall: a step's decode rows wait for all of that step's prompt work; v1 shortens it by the same share (ITL max
3.8 s -> about 2.6 to 3.6 s at TP4). With prefill's (a) as well, a 16-turn burst becomes 2 to 3 forwards of
8 x 1,536 padded tokens and the per-token part (4.7 s at TP4) is what is left. **So after (a) + (b) the larger lever
on TTFT is the per-token device cost of the prompt path (0.19 ms per padded token at TP4, far above its FLOP cost),
which neither (a) nor (b) changes** (prefill's GDN stack, a sparse prefill attention; a device trace of one prompt
forward at 10K context would say where it goes).

### 5. Risks

- Card rules 10, 10b, 11: v1 adds no custom op and no second call on a pool inside a forward (each layer still makes
  one state write per pool for all rows). The multi-row state save must stay fixed-shape (`remainder` + one
  `index_copy_`, no `nonzero`). The prompt conv writes its pool as `conv_states[idx, a:b, :] = rows`
  (`causal_conv1d_pytorch.py:502`, prefill's file): the rule-11 pattern, on the card today for one row; with bs rows
  it gets more indices (no padded rows in v1, so no duplicate indices). If the card shows a `slice_insert` problem
  there, the fix is a whole-row `index_copy_` (prefill's file).
- Graph memory: a prompt graph's workspace grows with bs x (query + context) (the dense QSA prompt attention's scores
  and bias); with 00150's 64,000-token bound at TP4, a 10K-context bucket allows bs <= 3 and a 56K one bs = 1. At the
  agents' contexts v1 merges little unless the bound (device memory) is raised: this, not the code, may decide v1.
- Compiles: every new (bs, query, context) bucket compiles live when warmup is skipped (10 to 15 s at TP4); bs buckets
  stay {1, 2, 4, 8}; the first bursts after a start pay them.
- Numerics: a merged forward is bit-different from one-row forwards (GEMM shapes, MoE grouping). Token parity is
  proven on the CPU tiny model and the tiny TP8 gate; real-model texts are compared at 1 stream only (BRIEF 08:41).
- Same-step shared prefix: a request that hits blocks another request of the same step computes would read a state
  slot before it is written in a merged forward: guarded (never merged).
- MTP and async scheduling: the drafter's prefill pass and `invalid_req_indices` per row; covered by engine tests.

### 6. Plan

1. CPU: tests of the grouping rule and its guards; per-row state bookkeeping (load / store / conv slots per row equal
   the one-row forwards'); the fixed-shape multi-row state save; the CPU rehearsal engine with mixed bursts
   (different lengths, prefix hits and misses, a prompt arriving while others decode, with and without MTP): the same
   tokens with the flag on as off.
2. Card: the tiny TP8 token gate with the flag; then a real-model A/B on build 5 + these commits with prefill's agents
   plan (16 and 32 agents, 1,024 new tokens per turn, contexts 16K to 56K), placement on: TTFT p50 / p90, the longest
   decode stall, uncached tokens per turn, decode ms per token, and the merged forwards' paced periods against the sum
   of the one-row ones (this answers the per-forward or per-row question of section 4).

## v1 as built (13:13 UTC 2026-11-21)

- `vllm_gaudi/v1/worker/prefill_batch.py`: the pure rules (flag, refusal, `group_sizes`, the read-after-write
  guard, the graph limit). `hpu_model_runner.py`: `_resolve_prefill_batch` (at init; the prompt bs buckets go up to
  N), `_prefill_batch_rows_now` (per step: the switch file can cap it), `_prefill_chunk_shape`,
  `_prefill_batch_sizes`, `_group_prefill_contents` (called from `_extract_prefill_batch_contents` for runs of
  consecutive one-request chunks), `_count_prefill_batch` (log line at the first merged forward and every 32nd:
  `VLLM_HPU_PREFILL_BATCH: N merged prompt forwards so far, ...`); the Mamba2 checkpoint maps only for one-row batches
  (`one_row_prefix_maps`); `propose_eagle_prefill` shifts the target tokens per row; `_fill_draft_kv_for_chunk` takes
  every row of a no-logits batch. `qwen3_5._save_ssm_state`: several rows in one fixed-shape `index_copy_` (padding
  rows to the padding row, as the PLE layer does; the test of the old `nonzero` behaviour is updated).
  `hpu_flash_next_mtp.fill_prefill_chunk`: one last index and next token per row, ids in the batch's [rows, T] shape.
  `runtime_switches.SWITCHABLE` gains the flag.
- What v1 does NOT change: the model's prompt path (GDN chunk rule rows, conv rows, QSA rows, PLE rows were already
  there), the scheduler, one-row forwards (bit-identical path), any flag-off behaviour.

## CPU proof (VERIFIED, tiny checkpoint, CPU engine; 13:10 UTC)

Driver `tests/unit_tests/models/flash_next/prefill_batch_engine.py` (step-driven arrivals, records every prompt forward
with its rows' state slots). Scratch runs in `reports/prefill-batch/runs/` (`tools/exp1.py`, `exp2.py`, `exp3.py`,
`cmp.py`), on this branch and on the build-5 card tree:

| plan | flag off -> on prompt forwards | streams equal | logprob drift |
|---|---|---|---|
| exp1: 4 x 300 + 2 x 700 at once, then 4 next turns (prefix hits) | 16 -> 5 (all merged) | 10 / 10 | 0.0 |
| exp2: + chunked 1,200-token prompts, 2 arrivals while decoding, a shared 600 prefix, 6 next turns | 26 -> 18 | 16 / 16 | 0.0 |
| exp2 with MTP k = 2 + async 2 + draft loop + chunk KV + align split / keep prompt end | 26 -> 23 | 16 / 16 | 0.0 |
| exp2 with async mode 1 (partial chunks carry one logit) | 26 -> 18 | 16 / 16 | 0.0 |
| exp3: 4 requests with one shared 600 prefix + 4 with one 512 system prompt | 10 -> 4 | 8 / 8 | 0.0 |

The graph limit is what cut most groups on CPU (vLLM's capture size = the 1,024-token step budget of these plans):
e.g. the four 44-token tails at context 256 ran as 2 + 2. The hybrid scheduler never gave a request a prefix hit on
blocks another request computes in the same step (exp3: s1 to s3 waited one step for s0's first chunk), so the
read-after-write guard did not fire on CPU; it is pinned by `test_prefill_batch_rules.py`.
Tests: `tests/unit_tests/worker/test_prefill_batch_rules.py` (rules), `tests/unit_tests/models/flash_next/
test_engine_prefill_batch.py` (the engine plan off / on, without and with MTP; state slots per chunk equal; the
runtime switch at 0 and 2), `test_runtime_switches.py`, `tests/unit_tests/test_qwen3_next.py` (state save; its two
gated-FP8 tests are red on the base `8ead88c7` too, not this track's).

## Ops of a merged forward (VERIFIED by CPU dispatcher count, 13:21 UTC; device time INFERRED)

`prefill_batch_engine.py --count-ops` counts the ops every model forward dispatches (on a card each one is a node of
the forward's graph). Tiny model (4 layers: 3 GDN + 1 QSA), bursts of K equal turns (`reports/prefill-batch/tools/
exp4.py`): a one-row prompt forward [1, 512] dispatches 2,592 ops; merged [2, 512] 2,676 (+84), [4, 512] 2,786
(+194), [8, 512] 3,002 (+410, +16 %); [1, 256] 2,535, [8, 256] 2,945. About 52 ops per added row, all from the QSA
indexer's per-row loop (`qsa.py`, `for b in range(rows)`): on the real model about 12 x 52 = 620 ops per added row
against about 35,000 per forward (prefill's count at TP4), i.e. a merged forward of 8 rows has about 1.12 times the
graph nodes of a one-row forward, not 8 times. If the 80 ms intercept of section 1 is mostly per-node device overhead
(35,000 x 2.4 us = 84 ms, coord 00009's per-op cost), merging removes most of it per removed forward: the upper
column of section 4. Byte- and FLOP-bound per-row work (context gathers, attention) is not counted here; the burst
probe of 00267 measures the whole.

Without prefix caching (`tools/exp5.py`, `--no-prefix-caching`: load = store = the request's own slot): 9 of 9
streams and logprobs equal, 14 -> 11 prompt forwards, one merged pair with contexts 512 and 333 in one context bucket.
Side finding (not this track's, both arms alike): without prefix caching a chunk can start at a context that is not
a multiple of 4 (333 here), while the QSA prompt indexer assumes 4-aligned contexts (`qsa.py`, "ctx is 4-aligned by
construction"); only the align mode guarantees it, which every served configuration uses.
