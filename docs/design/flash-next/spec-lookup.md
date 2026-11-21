# Flash-Next prompt lookup (n-gram) drafting and the lookup + MTP hybrid

Track `spec-lookup`, branch `feat/flash-next-gaudi-spec-lookup`. Labels: **VERIFIED** (by the named test, log or
code line), **INFERRED** (reasoning on verified inputs), **MEASURED** (a card job, with its id).

This file is also the track's report and handoff: the agent's harness does not write report files, so the
handoff lives here, on the branch (BRIEF.md, 19:00 UTC entry, point 5). Raw material that is not in the
repository is under `/Volumes/Storage/Claude/gaudi-work/reports/spec-lookup/` (`logs/`, `replay/`, `card/`,
`realcfg/`).

## Handoff (if this agent stops)

State at **2026-11-21 02:55 UTC** (fourth agent, Opus 5.5). Branch head: the commit that holds this text, pushed.
It contains the program head `5c1e8219` (PR #15 kernel-fixes, merged as `f4dad13a`), mtp-card-ready through
`f1aa2292` and every commit of this track. **The node was extended (BRIEF 00:50 UTC): it ends about 00:00 UTC on
2026-11-22**; everything on it is lost then. Merges are allowed again (one gate at a time).

**Card jobs** (`bin/farm status`, `bin/farm ledger | grep -w spec-lookup`; results in
`reports/spec-lookup/card/<id>-<name>/`, mirrored under `node-mirror/farm/spec-lookup/results/`)

| Job | What | State |
|---|---|---|
| 00086, 00087, 00107, 00114 | tiny-model rehearsals (n-gram k = 3 and 8, vLLM's search, the hybrid with the torch and the fused GDN verification) | ended ok; section 7.1 |
| 00095 `real4-nospec-edit-prose` | real model, no speculation, TP4 + EP4: the reference and the token ids for the replay | ended ok; sections 7.2 to 7.4 |
| 00133 `real4-ngram5-edit3` | real model, n-gram k = 5, TP4: 85 tok/s for one editing agent, then the conv-commit crash on the second request (fused and torch verification alike) | ended rc 5 at 00:44 UTC; section 7.5 |
| **00157 `real8-lookup-nospec-hybrid3-pf`** (all, 45 min; replaced 00153 / 00154) | TP8 on the integration build 2 + this branch + mtp-card-ready's conv-commit fix `5c770d96` (build `e85a3292`, branch `feat/flash-next-gaudi-spec-lookup-tp8`, pushed); build 2's flags with the prefix flags; arms `nospec` and `hybrid3` (`lookup_arms.sh`), rounds edit:1, edit:32, mix:32 and a repeat of edit:1 (the 00133 case at TP8) | **waiting**: placed by the coordinator after prefill's agent headline (`farm status` at 01:12 UTC: starts by 04:38 UTC at the latest); local waiter `card/submit-real8-lookup-nospec-hybrid3-pf.log` |
| **00159 `real8-lookup-ngram5-pf`** (all, 35 min; replaced 00155) | the same build and rounds, arm `ngram5` | **waiting**, after 00157 (starts by 06:45 UTC at the latest); local waiter `card/submit-real8-lookup-ngram5-pf.log` |

The exact submit commands are in section 7.6. Both jobs run the build `e85a3292`: to resubmit, check out
`feat/flash-next-gaudi-spec-lookup-tp8` in this worktree first (the farm runs the worktree's HEAD), submit with
`--replaces <id>`, wait for the `farm: job` line, then check out `feat/flash-next-gaudi-spec-lookup` again.

When they end:
```bash
cd /Volumes/Storage/Claude/gaudi-work/wt/spec-lookup
C=/Volumes/Storage/Claude/gaudi-work/reports/spec-lookup/card
python3 tools/flash_next/lookup_summary.py $C/00157-real8-lookup-nospec-hybrid3-pf/nospec \
    $C/00157-real8-lookup-nospec-hybrid3-pf/hybrid3 $C/00159-real8-lookup-ngram5-pf/ngram5
grep -h -E "MEAS|TEXT|fallback|DONE|lookup_arms" $C/00157-*/*/probe.log $C/00159-*/*/probe.log $C/00157-*.log $C/00159-*.log
```
and report to the coordinator (SendMessage `main`) and in section 7.6: per arm and round tokens per step, ms per
step, ms per token and tok/s per stream against `nospec`; the hybrid's mode 1 against mode 0 (lookup's gain over
MTP alone); the mix round's halves (`by_kind` in probe.json: edit and prose streams in one batch); `c` at 32
streams = (ngram5 edit:32 ms per step - nospec edit:32 ms per step) / 160 and at 1 stream / 5; and whether the
repeat round (the last edit:1, a second request on prompt 100 past the 512-token block) ran and equals the first
edit:1 stream token for token (TEXT lines; at TP8 a difference points at the commit path, coordinator 00:57 UTC).

**Defect fixed at 02:50 UTC (coordinator's request, found by async-sched): a one-token request padded by the
scheduler.** vLLM's scheduler pads a waiting request with exactly one token left to compute to `1 + k` tokens
with `k` draft placeholders (-1) when speculation is on and other requests run in the step (or the request has
computed tokens): "pad new decode requests to uniform spec decoding size to preserve full cudagraph". The HPU
runner took it on the prompt path, computed the placeholders as tokens (the recurrent state ran over them) and
sampled every row: three garbage tokens after the first, accepted. Fix `005e1ff9`
(`vllm_gaudi/v1/worker/spec_padding.py`, four lines in `_prepare_inputs`): the prompt path computes only a
request's real tokens; the scheduler rejects the placeholders. VERIFIED: `test_lookup_engine.py::
test_one_token_prompt_joining_a_verifying_batch` failed in all three configurations (n-gram with prefix caching
off and on, MTP k = 3 with prefix caching on) and passes; async-sched's staggered schedule with the 1-token
prompt passes with the same change on their branch. Serving triggers: a 1-token prompt beside running requests,
a prompt whose cached prefix leaves exactly one token, a resumed request with one token left. Cherry-pickable
alone (the engine test needs this branch's harness).

**The gate.** The third agent's waiting gate was stopped at 23:41 UTC (BRIEF 23:21, since withdrawn by the 00:50
entry). Run `bin/gate <worktree> --tools` when the merge is due (step 2 below); it waits for the lock.

**Pending, in order**

1. 00157 and 00159's numbers (above).
2. **Merge after mtp-card-ready's own pull request** (merges are allowed again, BRIEF 00:50 UTC; one gate at a time).
   This branch contains mtp-card-ready through `f1aa2292`; on its merged head `test_engine_mtp.py` fails one test
   alone (`test_sampled_requests_through_the_speculative_engine`, "output with shape [1, 248320] doesn't match the
   broadcast shape [4, 248320]", `reports/spec-lookup/logs/m8_test_engine_mtp_54990f1f.log`), fixed by their
   `0d1be6fc`; their `5c770d96` fixes the conv-commit crash of 00133. Both are on
   `origin/feat/flash-next-gaudi-mtp-card-ready`, whose branch also carries merges of qsa-fused, so do not merge
   their branch here: when their pull request has landed, `git fetch origin && git merge origin/feat/flash-next-gaudi`,
   `bin/gate <worktree> --tools`, push, `gh pr create --base feat/flash-next-gaudi` (body draft:
   `reports/spec-lookup/pr/pr1_body.txt`; fill in the gate line and the card lines), `gh pr merge --merge`. My
   `2dd6bc75` changes their `sampled_rejection_sample` (a lookup padding draft -1 is never accepted for a sampled
   request; told the coordinator for them at 01:00 UTC): resolve a conflict there in favour of keeping `real`.
   Card-proven so far: n-gram with prefix caching (tiny 00086 k = 3, 00087 k = 8 and vLLM's search; real model
   00133 at 1 stream until the conv-commit crash, which `5c770d96` fixes), the hybrid (tiny 00107 torch and 00114
   fused verification). All flags of this track default off (`VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP`,
   `..._LOOKUP_MIN_NGRAM`, `..._LOOKUP_MAX_NGRAM`, `..._LOOKUP_ORACLE`); n-gram itself is chosen by
   `--speculative-config`. **Sampled requests with lookup drafts** (agents sample): `test_lookup_engine.py -k sampled`
   (the n-gram server and the hybrid, temperature 0.7, `VLLM_HPU_SPEC_SAMPLED_VERIFY=1`: every stream to its
   length, no -1 token, ragged widths) passes on the TP8 build tree (`e85a3292` + this branch: 15 passed with the
   sampler tests, `reports/spec-lookup/logs/m10_sampled_on_tp8_build.log`) and fails on this branch alone like
   mtp-card-ready's own sampled test did before their `0d1be6fc` (`logs/m9_test_lookup_engine_sampled.log`): it
   needs that commit, which comes with their pull request.
3. **Card check of `VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS=1`** (`be141e1a`, default 0, CPU-proven: a step in which
   no request drafts runs as a plain decode step instead of a 2-row placeholder verification; at one stream on
   TP4 the placeholder row cost about 8 ms, 00133). It is not in the TP8 build `e85a3292`; the next n-gram job
   should carry it (merge `be141e1a` into a new job build) with a `prose:1` and a `mix:32` round, where it acts.
4. **Rows at 32 streams** (section 7): an n-gram step is as wide as its longest draft, an empty draft is the
   2-row placeholder, so 32 streams carry `32 (1 + k)` rows as soon as one stream drafts. Two code steps would
   fix the economics of a mixed load: per-request (ragged) verification rows (rows `sum(1 + d_i)`, not
   `32 (1 + max d_i)`), and no placeholder row for a request without a draft. Both need the runner's decode
   batch to carry per-request query lengths; not started. A smaller step, needed before an n-gram server runs
   with warm-up in production: a flag that pins the n-gram verification width to `1 + k` (in
   `_create_decode_input_data`, `num_tokens = max(num_tokens, 1 + k)` for a verification step), so every step
   uses the warmed `batch x (1 + k)` buckets instead of compiling the narrower widths on first use (section 2);
   it costs `k - 1` padding rows per request on steps whose longest draft is short (no effect at 32 editing
   streams, which run at full width anyway).
5. Depth beyond 5 with the fused verification op needs a 9-node lattice (`GDN_CORE_VERIFY_MAX_NODES = 6`);
   k = 8 runs the torch recurrence today (rehearsed on the tiny model, 00087).

**Done and verified on CPU** (earlier agents; tests on the tree of each named commit)

| What | Evidence |
|---|---|
| n-gram lookup with prefix caching through the real engine (ragged oracle drafts, tampered drafts, every accepted count 0..3, empty drafts, widened steps, block-completion commits, k = 8, vLLM's own `NgramProposer`, prefix caching off) | `tests/unit_tests/models/flash_next/test_lookup_engine.py`: 8 passed (`3d536265`), `reports/spec-lookup/logs/m1_test_lookup_engine.log` |
| The lookup + MTP hybrid through the same engine | the same run: `test_hybrid_lookup_beside_mtp_with_prefix_caching` passed |
| Hybrid proposer, modes, per-request mode, log line, rehearsal oracle | `tests/unit_tests/worker/test_flash_next_lookup.py` 19 passed |
| The rehearsal oracle through the engine (n-gram server and hybrid) | `tests/unit_tests/models/flash_next/test_lookup_oracle_engine.py` 2 passed (`c809b15e` + the test), `logs/m7_test_lookup_oracle_engine.log` |
| Verification width helpers | `tests/unit_tests/worker/test_verify_width.py` 12 passed |
| Platform: n-gram with prefix caching served, k limits `{"mtp": 5, "ngram": 8}` | `tests/unit_tests/test_flash_next_platform.py` 75 passed |
| Card job tools | `tests/unit_tests/tools/rental/test_lookup_probe.py` 11, `test_lookup_tiny.py` 9, `test_spec_probe.py` 13 passed |
| The replay tool (its `earliest` policy equals vLLM's n-gram kernel on 400 random streams) | `tests/unit_tests/tools/test_lookup_replay.py` 5 passed |

Not yet rerun on the merged head: the CPU engine tests (`test_lookup_engine.py` about 19 min under load,
`test_engine_mtp.py` about 30 min); the gate above runs the Flash-Next model tests and the platform and tool tests.

## 1. What is on the branch

| Commit | What |
|---|---|
| `cfeb7e5d` | n-gram lookup with prefix caching: the platform refusal is lifted, the align-mode lattice loop skips the attention group, ragged verification batches are widened (`verify_width.py`), a prefill-only step no longer leaves `sampling_metadata` unbound |
| `80590c1a` | `flash_next_lookup.py`: the lookup + MTP hybrid behind `VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP` (two call sites in the runner) |
| `e7d676ef` | a step whose prefill chunks sample nothing drafts nothing (`torch.cat` of an empty list) |
| `2716b63f` | the hybrid through the engine on CPU |
| `7dcc65a3` | n-gram drafts up to k = 8 |
| `3d536265`, `628d9716` | merges of `feat/flash-next-gaudi-mtp-card-ready` (`9722b888`, then `f0e7d02f`) |
| `64e10aa6` | `tools/flash_next/rental/lookup_serve.sh`, `lookup_probe.py`: the real-model card job |
| `bf19334a` | the rehearsal oracle, `tools/flash_next/rental/lookup_tiny.py`, the hybrid's per-load log line |
| `0953885a` | merge of the program head `1b28bbff` (PR #12, PR #13) |
| `ffb4d87e`, `c13d1b97` | `lookup_probe.py`: whole answers (`editf`), the token ids of every stream and prompt |
| `8c5ac747`, `877cab29` | `tools/flash_next/lookup_replay.py` (lookup drafters replayed over recorded streams), `lookup_summary.py` (the table of the card jobs) |
| `e121d61b` | merge of mtp-card-ready `f1aa2292` |
| `aa51e8b9`, `f0e2b5dd`, `074fb2b6` | the hybrid's counters through the `vllm` logger; `lookup_tiny.py` with the fused GDN verification op; the rehearsal oracle through the engine on CPU |
| `49eb18cb` | merge of the program head `938dada4` (PR #14 collectives) |
| `32f54e44`, `ab513444` | `lookup_probe.py` round kinds `mix` and the first prompt per round (no prompt twice), `lookup_arms.sh` |
| `f4dad13a` | merge of the program head `5c1e8219` (PR #15 kernel-fixes) |
| `2dd6bc75` | sampled verification: a lookup padding draft (-1) is never accepted (mtp-card-ready's function) |
| `e6d8b4d6` | the sampled-request engine test of both lookup paths (needs mtp-card-ready's `0d1be6fc`) |
| `be141e1a` | `VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS`: plain decode steps when no request drafts |
| `005e1ff9` | the prompt path never computes the scheduler's draft placeholders of a padded one-token request (all speculative methods) |

## 2. n-gram lookup with prefix caching

The platform refused `--speculative-config '{"method": "ngram", ...}'` with prefix caching on. With prefix
caching the recurrent layers (GDN, the PLE short conv) keep their state in align mode: one state slot per
block, a state lattice `row[curr : curr + 1 + k]` per verification step, pre-step copies and block-completion
commits (the machinery MTP uses). Lifting the refusal showed three defects, none specific to n-gram (VERIFIED
by `test_lookup_engine.py`):

1. The lattice loop built a lattice for the attention group too, whose block row is kernel blocks: a short
   context hit the null block. The attention group now takes its scalar rows (mtp-card-ready's form of the same
   fix is the one on the branch).
2. **Ragged widths.** A verification batch is as wide as its widest scheduled row, and the recurrent layers
   load each request's state from lattice column `num_accepted - 1` (the previous step's accepted count),
   clamped to the batch width. An MTP batch is always `1 + k` wide. An n-gram batch is not: after a step that
   accepted every draft, a step with shorter or empty drafts (the `[-1]` placeholder) loaded an earlier token's
   state and the stream broke. `verify_width.verification_width` widens such a step to the previous accepted
   count; the extra columns are padding rows (`pad_query_rows`). Prefix caching off had the same defect.
3. A step whose only work is a prefill chunk that completes no prompt left `sampling_metadata` unbound.

Every decode step of an n-gram server is a verification step: a request without a match gets the placeholder
draft `[-1]` (never accepted), so the narrowest batch is 2 tokens wide and the width changes from step to step
(2 to `1 + k`). Consequence for the card: each (batch bucket, width, block bucket) is its own HPU graph, so
an n-gram server compiles up to `k` times the decode graphs of an MTP server (INFERRED); the hybrid keeps
MTP's one width. The warm-up does not cover the narrower widths: `generate_spec_decode_buckets`
(`vllm_gaudi/extension/bucketing/common.py`) adds decode buckets for `batch x (1 + k)` flattened rows only
(VERIFIED by reading the code), so a production n-gram server with warm-up would still compile every narrower
width on its first use (the longest gaps of 00095's rounds at TP4, compiles and prefills: 12.7 to 32.8 s).

## 3. The lookup + MTP hybrid

`VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP` (with the `mtp` method): `0` off, `1` hybrid, `2` lookup only; a request may
pick its own mode with `vllm_xargs {"spec_lookup": 0 | 1 | 2}`. The MTP drafter runs every step for every
request; then, per decode request, its `k` drafts are replaced by the n-gram continuation when the request's
last `n` tokens (`VLLM_HPU_FLASH_NEXT_LOOKUP_MAX_NGRAM` = 8 down to `..._MIN_NGRAM` = 3) occur earlier in its
prompt or output (vLLM's search: the earliest occurrence of the longest match). A lookup draft shorter than
`k` is padded with `-1`, so every verification row stays `1 + k` wide: the MTP drafter's `prepare_inputs`
stays aligned with the target's rows and the HPU graphs are the MTP graphs. The drafter's own KV, compressed
keys and key ring are written by its first pass over the verification rows as with MTP alone.

The server logs the rows and accepted drafts of each source (`lookup`, `mtp`, `none`) every 200 steps and
whenever none of the previous step's requests is left (one line per round of a probe).

What the hybrid does not do yet: lookup drafts deeper than the MTP `k`; a search that costs less than
O(context) per request and step (section 4).

## 4. Depth beyond 5

- MTP: `k <= 5`, the largest lattice of the fused GDN verify op (`GDN_CORE_VERIFY_MAX_NODES = 6`).
- n-gram: `k <= 8` (`platform.FLASH_NEXT_MAX_SPECULATIVE_TOKENS`). A verification of more than 6 rows runs the
  torch GDN recurrence (the fused verify op's gate returns None). VERIFIED on CPU at k = 8:
  `test_ngram_end_to_end.py` (fp32 gaps about 1e-7, state gaps about 1e-6) and the k = 8 engine configuration.
- Memory (INFERRED from the pool formula and MEASURED slot sizes): the recurrent state pool holds
  `max_num_seqs x (3 + k) x 4 groups + 512` slots. At TP8 a slot is 4.91 MiB (job 00021: 1,280 slots at k = 3 =
  6.15 GiB of the 58.25 GiB KV budget per card); k = 8 is 1,920 slots = 9.2 GiB. At TP4 a slot is 9.35 MiB
  (card-farm 00024) and the budget 40.6 GiB: k = 3 takes 11.7 GiB, k = 8 takes 17.5 GiB, leaving 23.1 GiB of
  attention blocks (about 1.8 million tokens at the density 00024 reported: 35.04 GiB held 2,718,686 tokens).
- What limits a deeper lookup draft on the card is the step cost of the rows (32 streams x 9 rows = 288 rows
  per step), which no job has measured.

## 5. Acceptance estimates by replay

Greedy verification accepts a draft token exactly when it equals the token the target emits, so for a recorded
greedy stream the outcome of any lookup drafter at any depth is arithmetic. `tools/flash_next/lookup_replay.py`
plays the streams of a probe record (`lookup_probe.py` keeps the token ids) through an n-gram index drafter:
`earliest` (vLLM's search, what the server uses; VERIFIED equal to vLLM's kernel on 400 random streams),
`recent`, `follow`. A run without speculation is enough; MTP's acceptance is not replayable.

**Synthetic preview (REPLAY ESTIMATE, not the model's answers).** 24 files of this tree returned with a
synthetic edit (one-line docstrings, type hints, renamed locals; 56,667 output tokens, prompts of 783 to 5,106
tokens), n-gram 3 to 8, `reports/spec-lookup/replay/synthetic_replay.json`. Share of ALL steps that accepted at
least j drafted tokens (the server's per-position rate counts the same way), `earliest` policy:

| Depth k | tokens per step | steps with a draft | j=1 | 2 | 3 | 4 | 5 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 3.61 | 92 % | 0.89 | 0.87 | 0.86 | | | | | |
| 4 | 4.38 | 91 % | 0.87 | 0.85 | 0.84 | 0.83 | | | | |
| 5 | 5.12 | 89 % | 0.85 | 0.84 | 0.82 | 0.81 | 0.80 | | | |
| 8 | 7.09 | 85 % | 0.80 | 0.78 | 0.77 | 0.76 | 0.75 | 0.73 | | |
| 12 | 9.30 | 80 % | 0.75 | 0.73 | 0.71 | 0.70 | 0.70 | 0.68 | 0.66 | |
| 16 | 11.18 | 76 % | 0.71 | 0.68 | 0.66 | 0.65 | 0.65 | 0.63 | 0.62 | 0.61 |

`follow` gives 3.70, 4.50, 5.26, 7.35, 9.70 and 11.69 tokens per step at the same depths (2 to 5 % more);
`recent` slightly less than `earliest` (3.58 to 10.94). Prose (one 2,252-token document): 1.20 to 1.25 tokens
per step at every depth, a draft on 10 to 13 % of the steps. The rates fall with depth because a deeper draft
makes the steps longer, so a larger share of the steps starts at an edit. These figures say what lookup accepts
when the model copies as the synthetic edits do; the same table from the real model's answers comes from the
no-speculation job (section 7).

## 6. Card jobs

**Tiny-model rehearsal** (`tools/flash_next/rental/lookup_tiny.py`, inside `bin/farm job ... any:2`). It serves
the tiny checkpoint without speculation, records the streams of eight scenarios (single prompts of 7 to 700
tokens, prompts ending just before a 256- and a 512-token boundary, four requests in one batch, a second turn
with a prefix-cache hit), then serves each configuration and compares its streams with the baseline up to a
near-tie. A random tiny model never repeats its context, so drafts come from the rehearsal oracle
(`VLLM_HPU_FLASH_NEXT_LOOKUP_ORACLE`: the baseline's own continuation on a ragged schedule with tampered
drafts): accepted counts 0..k, widened steps and block-completion commits run on the card before the real
model does. Configurations: `ngram3`, `ngram8`, `mtp3-lookup` (three passes: hybrid, lookup-only, MTP alone, by
the request's mode), `ngram3-search` (vLLM's own search, no oracle: the numba kernel in the workers).

**Real model** (`tools/flash_next/rental/lookup_serve.sh` + `lookup_probe.py`, inside `bin/farm job ... any:4`):
one server per job (TP4 + EP4, the card farm's serve command and preset, `VLLM_HPU_GDN_CORE_VERIFY=0`), rounds
`KIND:STREAMS:TOKENS:MODE`:

- `edit`: an instruction plus one small file of this tree, the answer is the whole edited file. 664 to 859
  prompt tokens (VERIFIED with the checkpoint's tokenizer and chat template). Short on purpose: with prefix
  caching the scheduler's `_mamba_block_aligned_split` ends a prefill chunk at the last cacheable block
  boundary, one block earlier on an MTP server (`use_eagle_block_drop`), and the MTP drafter never sees a
  chunk that completes no prompt (`VLLM_HPU_MTP_CHUNK_KV=0`). The block is 512 tokens at TP8 with MTP (job
  00021's log) and 896 at TP4 + EP4 without speculation (job 00024's log: "Setting attention block size to 896
  tokens"), so below 1,024 tokens an MTP server prefills the prompt in one chunk in either layout and MTP
  alone is measured with its full context. At TP4 these prompts are shorter than one block: no round gets a
  prefix-cache hit (every round prefills its prompts again).
- `editl`: the same with files of 3,164 to 3,601 tokens (on an MTP server most of the prompt is in chunks the
  drafter did not see: this round shows what that costs MTP, and what lookup accepts on longer copies).
- `editf`: the edit prompts answered to the model's own end (no `ignore_eos`): whole answers for the replay.
- `prose`: a story request, 39 to 41 tokens, nothing to copy.

No `min_tokens`: `max_tokens` with `ignore_eos`. Stream `i` of a round always gets the same prompt, so the
streams of all rounds and jobs compare token for token (`return_token_ids`: the probe keeps the ids). Per
round: tokens per verification step per request and the rate of each draft position (the server's counters),
ms per engine step (the median gap between a stream's chunks), ms per token over the decode span and steady,
the client-side step sizes. `tools/flash_next/lookup_summary.py` turns the result folders into the table of
section 7.

**CPU check with the real checkpoint's config** (`reports/spec-lookup/realcfg/check_real_config.py`, the
`config.json` copied from the node with `bin/node-scp`, no weights; VERIFIED 2026-11-20 20:00 UTC on
`877cab29`): the engine config of the three servers (no speculation, n-gram k = 8, MTP k = 3 with the lookup
hook) builds with the job's serve arguments (131,072 tokens, 32 sequences, prefix caching, align mode), the
speculative validators pass, the target's config has `image_token_id` (248056) and no `image_token_index`
(the line job 00008 died on), the proposer looks the target up as `Qwen3_5ForConditionalGeneration`, the
draft's M-RoPE keys are removed, and `HybridLookup.from_env` gives HYBRID, n-gram 3..8, k = 3. What CPU cannot
show without the weights is the load itself: on the card the MTP server is healthy after 435 s (job 00021).

## 7. Card results

### 7.1 Tiny-model rehearsals (jobs 00086, 00087, 00107, 00114; 2026-11-20 20:30 to 20:38 UTC)

`lookup_tiny.py`, TP2 on two cards, prefix caching on, the real-model job's decode flags, drafts from the
rehearsal oracle (the baseline's own continuation on a ragged schedule, every 4th or 5th draft tampered).
MEASURED (`reports/spec-lookup/card/<id>-<name>/lookup_tiny.json`, the job logs):

| Job, commit | Configuration (pass) | Drafts | Accepted | Tokens per step | Rate per position | Streams equal to the baseline |
|---|---|---|---|---|---|---|
| 00086, `628d9716` | n-gram k = 3, torch GDN verification | 328 | 153 | 1.47 | 0.22, 0.18, 0.07 | 1 of 11 |
| 00087, `628d9716` | n-gram k = 8 (9-row torch verification) | 200 | 285 | 2.42 | 0.24, 0.24, 0.23, 0.23, 0.19, 0.14, 0.10, 0.05 | 5 of 11 |
| 00087 | n-gram k = 3, vLLM's own search (`prompt_lookup_min` 1), no oracle | 481 | 0 | 1.00 | 0 | 3 of 11 |
| 00107, `e121d61b` | MTP k = 3 + lookup, torch verification: hybrid / lookup only / MTP alone | 293 / 293 / 481 | 190 / 190 / 0 | 1.65 / 1.65 / 1.00 | 0.30, 0.25, 0.10 | 2 / 2 / 4 of 11 |
| 00114, `074fb2b6` | the same with the fused GDN verification op | 264 / 264 / 481 | 221 / 221 / 0 | 1.84 / 1.84 / 1.00 | 0.39, 0.32, 0.13 | 5 / 5 / 7 of 11 |

("Drafts" counts verification rows with a draft; the hybrid's own counter in 00114: lookup rows 216, 2.08
accepted per row; MTP rows 0 accepted, a random model.)

- **MEASURED: accepted drafts beyond what MTP drafts run on the card** through the n-gram verification (ragged
  widths, widened steps, the torch GDN recurrence at k = 3 and k = 8) and through the MTP-shaped verification
  (torch and fused GDN verification op), with prefix caching on: no error line, no fallback line, every request
  answered. 00114 is the first card run of accepted drafts through the fused verification op.
- Token equality on the random tiny model: no stream accepted a tampered draft (its first difference is never
  the tampered token, VERIFIED over every pass); the four baselines are identical token for token; most first
  differences sit where the passes without any accepted draft differ too (MTP alone, vLLM's search) and where
  the baseline's own twins differ (the 7-token prompt alone and in a batch, token 40): batch-shape near-ties of
  a random model with flat logits (INFERRED). Two (prompt, position) pairs differ only with accepted drafts
  (the 100-token prompt at token 28 in three jobs, the second turn at token 14 in two): **not classified**,
  because the baseline's log-probability request crashed the server (below). The real-model run at one stream
  (00133 against 00095) is the check.
- Defect found on the way (reported to the coordinator, owner engine-step / the sampler): a `/v1/completions`
  request with `logprobs: 5` killed the HPU server's output handler: `OverflowError: out of range integral type
  conversion attempted` in the detokenizer of the top-logprob ids (`00086-.../baseline/server0.log`, about line
  426), so the ids it got were outside the u32 range: the shape of card rule 9 (top-k ids narrowed in one graph)
  in the logprobs path. Every request in flight then failed with HTTP 500 and the API server shut down.

### 7.2 The real model without speculation (job 00095, 2026-11-20 23:03 to 23:22 UTC)

TP4 + EP4 on cards 3 to 6, commit `c13d1b97`, `lookup_serve.sh --label nospec`, greedy, thinking off, healthy
after 442 s, no fallback line. MEASURED, ms per engine step (median gap between a stream's chunks):

| Round | Streams | Tokens | ms per step | tok/s per stream |
|---|---|---|---|---|
| edit (whole small file returned, ignore_eos) | 32 | 384 | 108.0 | 9.3 |
| prose | 32 | 256 | 107.2 | 9.3 |
| edit | 1 | 384 | 29.9 | 33.4 |
| prose | 1 | 256 | 30.9 | 32.4 |
| editf (answered to the model's own end: 617 to 831 tokens) | 24 | up to 1,400 | 110.8 | 9.0 |
| editl (3,168 to 3,600-token prompts) | 8 | 512 | 52.8 | 18.9 |

The model returns the whole file in a code fence (`texts` in `probe.json`).

### 7.3 Lookup acceptance on the real model's answers (REPLAY of 00095)

`tools/flash_next/lookup_replay.py .../00095-real4-nospec-edit-prose/probe.json --depths 3,4,5,6,8,12,16` ->
`reports/spec-lookup/card/00095-replay.json`. Greedy verification accepts a draft token exactly when it equals
the target's token, so for these recorded greedy streams the acceptance of vLLM's n-gram search (`earliest`,
n-gram 3..8) is exact arithmetic; time is not in it. Tokens per step per stream, and the share of ALL steps
that accepted at least j tokens:

| k | editf (24 whole answers, 17,727 tokens) | j = 1 .. k | edit (32 x 384) | editl (8 x 512) | prose (32 x 256) | rows per step at 32 streams: edits / prose (n-gram server) |
|---|---|---|---|---|---|---|
| 3 | 3.48 | 0.84 0.82 0.81 | 3.44 | 2.90 | 1.00 | 128 / 76 |
| 4 | 4.16 | 0.81 0.79 0.78 0.77 | 4.11 | 3.35 | 1.00 | 160 / 82 |
| 5 | 4.78 | 0.78 0.76 0.75 0.74 0.74 | 4.72 | 3.72 | 1.00 | 192 / 88 |
| 6 | 5.36 | 0.76 0.74 0.73 0.72 0.71 0.70 | 5.27 | 4.02 | 1.00 | 224 / 94 |
| 8 | 6.41 | 0.72 0.70 0.68 0.68 0.67 0.66 0.66 0.65 | 6.29 | 4.57 | 1.00 | 288 / 105 |
| 12 | 8.07 | 0.65 .. 0.56 | 7.87 | 5.23 | 1.00 | 416 / 125 |
| 16 | 9.44 | 0.59 .. 0.49 | 9.09 | 5.74 | 1.00 | 544 / 143 |

- A draft on 87 % (k = 3) to 64 % (k = 16) of the steps of the whole answers; on prose on 1 % of the steps.
- The `follow` policy adds 0 to 0.1 tokens per step; `recent` is slightly worse: the search policy is not the lever.
- The shortest match is not a lever either (`--min-ngram`, k = 3 / 5 on the whole answers): 2 -> 3.56 / 4.94,
  3 -> 3.48 / 4.78, 4 -> 3.40 / 4.63, 5 -> 3.33 / 4.49, 8 -> 3.16 / 4.17 tokens per step; on prose a minimum of 4
  already removes every draft (2: 7 % of the steps draft, 3: 1 %). For the hybrid a minimum of 4 keeps MTP's drafts
  on prose and costs the edits about 1 % (INFERRED); for the n-gram server it turns 32 prose streams' 76 rows into
  the 64 of the placeholder.
- Rows: an n-gram step is as wide as its longest draft and an empty draft runs as the 2-row placeholder, so 32
  editing streams carry `32 (1 + k)` rows; 32 prose streams still carry 76 to 143 rows (expected, streams drafting
  independently) for nothing. The MTP and hybrid shapes are always `32 (1 + k)`.
- The real answers accept slightly less than the synthetic preview of section 5 (3.48 against 3.61 at k = 3,
  6.41 against 7.09 at k = 8): the model rewrites more than the synthetic edits did. MTP alone on the same
  model: 2.62 to 2.78 tokens per step at k = 3 (mtp-card-ready, job 00021, one stream).

### 7.4 Depth as a function of the cost of a verification row (INFERRED from 7.3 and the cost model)

Model: a step at 32 streams that carries `R` rows costs `T32 + c (R - 32)`, `T32` the plain step, `c` the cost
of one extra row; per stream the n-gram server then spends `(T32 + c (R - 32)) / a` per token, `a` the tokens
per step (7.3). With `rho = c / T32` the speed-up on edits is `a / (1 + 32 k rho)`
(`reports/spec-lookup/replay/depth_by_row_cost.py`, output `card/00095-depth-by-row-cost.txt`):

| rho = c / T32 | best k on edits at 32 streams | speed-up there | k = 3 | k = 8 |
|---|---|---|---|---|
| 0.022 (MTP k = 3 at 32 streams, TP4, head's QSA path: 338 ms at 128 rows with the draft passes; upper bound) | 3 | x1.11 | x1.11 | x0.96 |
| 0.017 (TP4 head's QSA path: 1.8 ms per decode row, qsa-fused 00062, T32 = 108) | 3 | x1.34 | x1.34 | x1.22 |
| 0.011 (TP4 with the QSA data-flow change: 0.84 / 73.4) | 4 to 5 | x1.69 | x1.66 | x1.63 |
| 0.005 | 8 | x2.8 | x2.3 | x2.8 |
| 0.0025 | 12 to 16 | x4.1 | x2.8 | x3.9 |

- Break-even `rho*` = `(a - 1) / (32 k)`: 0.026 (k = 3), 0.024 (k = 5), 0.021 (k = 8), 0.017 (k = 16).
- The optimum is flat: k = 3 to 6 lie within 3 % of each other for rho 0.01 to 0.017.
- One stream (T1 = 30 ms at TP4, a host-bound step): every depth pays; with an assumed 0.5 ms per extra row,
  k = 5 gives about 6.7 ms per token (x4.5) and k = 8 about 5.2 (x5.8). 00133 measures it.
- Prose at 32 streams: the n-gram server loses at every depth (x0.58 at rho 0.017, k = 3; x0.67 at rho 0.011).
- Mixed load at 32 streams, a share `f` of the streams in an edit span: the n-gram server runs
  `(T32 + c (R - 32)) / (f a_L + (1 - f))` (R = `32 (1 + k)` once any stream drafts; 76 rows at k = 3 when all
  write prose); the hybrid (MTP rows on prose, lookup rows on edits, always `1 + k` wide, plus the MTP draft
  passes and the rest of a speculative step, `D`) runs `(T32 + 32 k c + D) / (f a_H + (1 - f) a_M)`, with
  `a_L` the lookup tokens per step (3.48 at k = 3), `a_M` MTP's (about 2.5 at 32 streams, coordinator 23:19
  UTC) and `a_H` the hybrid's on edits (about `0.87 x 3.86 + 0.13 x a_M` = 3.68 at k = 3: lookup on the 87 % of
  steps with a match, MTP on the rest; INFERRED, the step boundaries move with the acceptance). Inputs at TP4,
  k = 3: on the head's QSA path `c` = (303 - 109) / 96 = 2.0 ms per verification row and `D` = 338 - 303 =
  about 35 ms (00104's step timer: forward 109 / 303 ms at 32 / 128 rows, an MTP step 338 ms, a draft pass 4 to
  11 ms); with the QSA data-flow change `T32` = 73.4 and `c` about 0.84 (decode rows, 00063; for verification
  rows not yet measured), `D` assumed unchanged. Speed-up against no speculation (INFERRED):

  | Path (TP4, k = 3) | Server | f = 0 (all prose) | f = 0.5 | f = 1 (all edits) |
  |---|---|---|---|---|
  | head's QSA path | n-gram | x0.55 | x0.80 | x1.24 |
  | head's QSA path | hybrid | x0.80 | x0.99 | x1.18 |
  | QSA data-flow change | n-gram | x0.66 | x1.07 | x1.66 (k = 4 to 5: x1.69) |
  | QSA data-flow change | hybrid | x0.97 | x1.20 | x1.43 |

  **TP8 with every card-proven flag** (integration 00128, 00:00 UTC, MEASURED: plain step 51.1 ms at 32 streams,
  MTP k = 3 step 106.4 ms with 2.61 tokens per step, 40.7 ms per token): the hybrid runs MTP's shapes, so its step
  is MTP's (plus the lookup search on the host, about 0.1 ms per step at these contexts), and only its acceptance
  differs: edits about 106.4 / 3.68 = 29 ms per token (x1.77 against 51.1), prose 106.4 / 2.61 = 41 (x1.25),
  half and half 34 (x1.51) (INFERRED from the measured steps and the replay). The cost per verification row there
  is about `c` = (106.4 - D - 51.1) / 96 = 0.2 to 0.4 ms (D, the draft passes and the rest of the step at TP8, 15
  to 35 ms, not measured), `rho` = 0.004 to 0.008: on edits the n-gram server then gains x2.1 (rho 0.008) to
  x3.2 (rho 0.004) at its best depth, k = 5 to 12 (k = 8 within 3 % of the best across that band); on prose it
  still loses (x0.6 to x0.85).
  The hybrid is the server for a mixed load once the verification rows are cheap; the n-gram server wins only on
  a pure copy load, until its rows become per-request (Handoff, pending 3). At one stream both pay (the step is
  host-bound): MTP alone 1.69x at TP4 (00099 / 00104), n-gram on edits x4 to x6 by 7.4.

### 7.5 n-gram on the real model (job 00133, TP4 + EP4, 2026-11-21 00:25 to 00:44 UTC)

n-gram k = 5 with prefix caching, commit `f2b86e91`, `VLLM_GRAPH_RESERVED_MEM=0.5`, the fused GDN verification op
first, then (the job's own fallback) the torch verification; rounds as 00095. MEASURED
(`reports/spec-lookup/card/00133-real4-ngram5-edit3/`, `probe.log` and `torch-verify/probe.log`):

| Server | Request | tokens per step (rate per position) | ms per step (p90) | ms per token, steady | tok/s per stream | against 00095 (29.9 ms) |
|---|---|---|---|---|---|---|
| n-gram k = 5, fused verification | 1 stream, edit prompt 0, 384 tokens | 5.87 (0.97 at every position) | 68.9 (87.7) | 11.7 | 85 | x2.55 |
| n-gram k = 5, torch verification | the same | 5.88 (0.97 at every position) | 84.8 (104.1) | 14.4 | 69 | x2.07 |

- These are the warm rounds (the measured rounds never ran, below); the median gap between chunks is one engine
  step and is not moved by the few compile pauses (longest 7.8 and 9.6 s).
- One stream on the head's QSA path at TP4: an extra verification row costs (68.9 - 29.9) / 5 = 7.8 ms with the
  fused op, 11.0 ms with the torch recurrence: rows are not cheap at one stream there (the torch QSA path works
  per query row). The replay's average edit stream (4.72 tokens per step at k = 5) would give about
  (29.9 + 5 x 7.8) / 4.72 = 14.6 ms per token there (INFERRED); stream 0 copies its file almost verbatim.
- **Then both servers died on the second request on the same prompt**, at the first recurrent block the decode
  completed (896 tokens at TP4): the fused one on the step starting at position 896 (`ValidateSyncInputTensors
  tensor_data is empty`: the first input of `gdn_core_verify`, the conv pool), the torch one a step later
  (`Neither storage attached to input tensor` in `replayV3`). Cause, found by mtp-card-ready (`5c770d96`, BRIEF rule
  11): `commit_conv`, the align-mode copy of a completed block's conv state (the boundary commit and the pre-step
  copy, prefix caching with speculation only), wrote the pool through a slice; the fix writes whole rows with
  `index_copy_`. 00095 (no speculation) ran the same request sequence without a crash. MTP had never completed a
  block during decode on a card (prompts and outputs under one block).
- **The replay is exact on the card**: replaying 00095's greedy answer to the same prompt (edit prompt 0, 384
  tokens) through vLLM's search at k = 5 gives 5.82 tokens per step (rates 0.97 0.97 0.97 0.95 0.95); the server
  measured 5.87 and 5.88 (VERIFIED, `lookup_replay.replay` on `00095 .../probe.json`). At k = 8 the same stream
  would give 8.53, at k = 3 3.92.
- 32 streams: not measured (the server was gone). Prompts repeat no more in this track's probes (`ab513444`).

### 7.6 n-gram and the hybrid at TP8 on the integration build (jobs 00157, 00159): pending

Build `e85a3292` (branch `feat/flash-next-gaudi-spec-lookup-tp8`, pushed): the integration build 2's code
(`a9814f72`), this branch (`ab513444`) and mtp-card-ready's `5c770d96`. Submitted from that build (00157 replaced
00153 and 00154, 00159 replaced 00155):
```bash
F=/Volumes/Storage/Claude/gaudi-work/bin/farm
E="--env VLLM_HPU_QSA_FUSED=1 --env VLLM_HPU_QSA_INDEXER_SHARD=1 --env VLLM_HPU_GDN_KERNEL_INTERLEAVE=1
   --env VLLM_HPU_GDN_GLUE_CACHE=1 --env VLLM_HPU_HC_FUSED_COPY_MODE=2 --env VLLM_HPU_DECODE_CUT_INTERVAL=1000
   --env VLLM_HPU_CANDIDATE_SAMPLING=1 --env VLLM_GRAPH_RESERVED_MEM=0.4 --env VLLM_HPU_GDN_CORE_VERIFY=1
   --env VLLM_HPU_PREFIX_ALIGN_SPLIT=1 --env VLLM_HPU_PREFIX_KEEP_PROMPT_END=1"
R=edit:1:384:-:100,edit:32:384:-:0,mix:32:384:-:400,edit:1:384:-:100
$F job spec-lookup real8-lookup-nospec-hybrid3-pf all 45m $E -- bash -c \
    "LOOKUP_ARMS_ROUNDS=$R bash tools/flash_next/rental/lookup_arms.sh nospec hybrid3"
$F job spec-lookup real8-lookup-ngram5-pf all 35m $E -- bash -c \
    "LOOKUP_ARMS_ROUNDS=$R bash tools/flash_next/rental/lookup_arms.sh ngram5"
```
The last round repeats prompt 100: a second request decoding past the 512-token block the first one cached (the
00133 case at TP8; at TP8 a stale prefix state cannot occur, coordinator 00:57 UTC, so a text difference there
points at the commit path). `hybrid3` runs every round in mode 1 (hybrid) and, on prompts + 300, in mode 0 (MTP
alone). CPU on that tree: `test_align_commit_slice_writes.py` 10, `test_lookup_oracle_engine.py` 2,
`test_engine_mtp.py` 15 passed (`test_lookup_probe.py`'s editl prompt-size guard fails there: a large file of the
integration tree grew; editl is not used by these jobs).

## 8. Notes for other tracks

- mtp-card-ready: every prompt of 1,024 tokens or more gets a prefill chunk that completes no prompt on an MTP
  server with prefix caching (not only prompts above the 2,048-token threshold): `vllm/v1/core/sched/
  scheduler.py` `_mamba_block_aligned_split`, `last_cache_position = num_tokens - num_tokens % block_size`,
  minus one block with `use_eagle_block_drop`. Without `VLLM_HPU_MTP_CHUNK_KV=1` the draft layer has no KV for
  those blocks (VERIFIED in the scheduler code; the effect on acceptance is not measured).
- mtp-card-ready: `causal_conv1d_pytorch.py:498` needed a long index for `index_copy_` on CPU (fixed in their
  `9d013a36`).
- Every track: an INFO line of a logger named `vllm_gaudi.<module>` (`init_logger(__name__)`) reaches no handler
  on a server: vLLM configures only the `vllm` logger. Job 00021's server log has no INFO line of
  `hpu_flash_next_mtp.py` or `flash_next_mtp_steps.py` (VERIFIED: "the draft runs with the target's plain RoPE"
  is absent). Use `vllm_gaudi.extension.logger.logger()` (the `vllm` logger) for lines a card log must show.
