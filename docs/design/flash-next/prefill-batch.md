# Flash-Next on Gaudi2: batched prefill across requests (track prefill-batch)

Track `prefill-batch`, worktree `/Volumes/Storage/Claude/gaudi-work/wt/prefill-batch`, branch
`feat/flash-next-gaudi-prefill-batch`. Started 12:15 UTC 2026-11-21. Goal: several requests' prompt chunks in ONE
forward for Flash-Next's hybrid stack, behind a default-off flag. The track's report lives here (the harness refuses
report files under `reports/`); scratch tools are in `/Volumes/Storage/Claude/gaudi-work/reports/prefill-batch/tools/`.

## Handoff (if this agent stops) - updated 12:45 UTC 2026-11-21

- **Branch** based on `8ead88c7` (prefill's head = program head 4d66755e + prefill's work; the prefill agent chose
  this base; its PR into the program branch follows its gate). Commits of this track: this document only so far.
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
- **Design note**: below (sent to the coordinator).
- **Next**: CPU tests first (grouping rule, per-row state bookkeeping, fixed-shape state save, engine rehearsal with
  mixed bursts), then the flag, then the tiny TP8 gate and the real-model A/B on build 5 (coordinator places it).

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
