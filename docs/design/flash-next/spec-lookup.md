# Flash-Next prompt lookup (n-gram) drafting and the lookup + MTP hybrid

Track `spec-lookup`, branch `feat/flash-next-gaudi-spec-lookup`. Labels: **VERIFIED** (by the named test, log or
code line), **INFERRED** (reasoning on verified inputs), **MEASURED** (a card job, with its id).

This file is also the track's report and handoff: the agent's harness does not write report files, so the
handoff lives here, on the branch (BRIEF.md, 19:00 UTC entry, point 5). Raw material that is not in the
repository is under `/Volumes/Storage/Claude/gaudi-work/reports/spec-lookup/` (`logs/`, `replay/`, `card/`,
`realcfg/`).

## Handoff (if this agent stops)

State at **2026-11-20 20:10 UTC**. Branch head: the commit that holds this text, pushed. It contains
mtp-card-ready through `f1aa2292` (both card fixes of job 00021, the int64 conv index, the MTP step timer) and
the program head `1b28bbff`.

**Card jobs in the queue** (`bin/farm status`; results land in `reports/spec-lookup/card/<id>-<name>/`)

| Job | What | State |
|---|---|---|
| 00086 `tiny2-ngram3-oracle` (any:2, 10 min) | tiny model: baseline, then n-gram k = 3 with prefix caching and the rehearsal oracle | waiting, bumped by the coordinator to the tail of the 1-card wave (bound 21:09 UTC); runs commit `628d9716` |
| 00087 `tiny2-ngram8-search` (any:2, 14 min) | tiny model: baseline, n-gram k = 8 with the oracle, n-gram k = 3 with vLLM's own search | the same (bound 21:14 UTC) |
| `tiny2-mtp3-lookup-oracle2` (any:2, 12 min, `--replaces 00097`) | tiny model: baseline, then MTP k = 3 + the lookup hook with the oracle, three passes (hybrid, lookup-only, MTP alone) | resubmitted 20:08 UTC from `e121d61b` (00097 ran a commit without mtp-card-ready's second fix `9d013a36` and would have died at the first draft); the id is in `reports/spec-lookup/card/submit-tiny2-mtp3-lookup-oracle2.log`; 00097 had a bump, the replacement needs the coordinator's again |
| 00095 `real4-nospec-edit-prose` (any:4, 28 min) | real model without speculation: the edit, prose, whole-answer and large-file rounds (section 6) | waiting in 00029's place (bound 02:34 UTC, realistically hours earlier); runs commit `c13d1b97` |

The local `farm job` commands that submitted 00086, 00087 and 00095 were killed with the agent's shell at 20:05
UTC; the jobs are on the node. Detached waiters copy their results when they end
(`reports/spec-lookup/card/wait-<id>.txt` says so); if those are gone too:
`bin/farm fetch spec-lookup <id> <id>-<name>`.

When a tiny job ends: send its result lines (the `lookup tiny [...]` lines: tokens per step, the rate of every
draft position, drafts and accepted; any failure with its traceback from `<config>/server0.log`) to the
coordinator (`main`) and to the mtp-card-ready track at once: it is the first card run with accepted drafts
(mtp-card-ready's tiny and real MTP runs accept only what the MTP layer drafts: 0 of 270 on the tiny model).
When the no-speculation job ends: run
`python3 tools/flash_next/lookup_replay.py reports/spec-lookup/card/00095-real4-nospec-edit-prose/probe.json --out reports/spec-lookup/card/00095-replay.json`
and send the acceptance per draft position for depths 3 to 16 on the edit prompts with the rows a verification
step carries at 32 streams (the tool prints both), and
`python3 tools/flash_next/lookup_summary.py reports/spec-lookup/card/<run folders>` for the table of section 7.

**Done and verified on CPU**

| What | Evidence |
|---|---|
| n-gram lookup with prefix caching through the real engine (scheduler, HPU worker and runner, tiny model): ragged oracle drafts with tampered ones (every accepted count 0..3, empty drafts, 11 widened steps, 3 block-completion commits, 440 tokens compared), full acceptance across 256-token block boundaries, vLLM's own `NgramProposer`, prefix caching off, k = 8 (every accepted count 0..8, 21 widened steps), the mutation without the widening breaks | `tests/unit_tests/models/flash_next/test_lookup_engine.py`: **8 passed** in 1,156 s on the tree of `3d536265` (this branch + mtp-card-ready `9722b888`), log `reports/spec-lookup/logs/m1_test_lookup_engine.log` |
| The lookup + MTP hybrid through the same engine (lookup rows padded with -1 and MTP rows in the same streams, lossless up to a bf16 near-tie) | the same run: `test_hybrid_lookup_beside_mtp_with_prefix_caching` passed (it was skipped before the merge: the MTP engine did not start) |
| Hybrid proposer, its modes, the per-request mode, the per-load log line, the rehearsal oracle | `tests/unit_tests/worker/test_flash_next_lookup.py` 19 passed (head) |
| Verification width helpers | `tests/unit_tests/worker/test_verify_width.py` 12 passed |
| Platform: n-gram with prefix caching served, k limits `{"mtp": 5, "ngram": 8}` | `tests/unit_tests/test_flash_next_platform.py` 75 passed (head) |
| Card job tools | `tests/unit_tests/tools/rental/test_lookup_probe.py` 11 passed, `test_lookup_tiny.py` 9 passed, `test_spec_probe.py` 13 passed (head) |
| The replay tool; its `earliest` policy equals vLLM's n-gram kernel on 400 random streams | `tests/unit_tests/tools/test_lookup_replay.py` 5 passed (head) |
| Model-level n-gram tests at k = 3, 4, 5, 8 (earlier agent, `7dcc65a3`; not rerun after the merges) | `tests/unit_tests/models/flash_next/test_ngram_end_to_end.py`, logs `reports/spec-lookup/logs/ngram_k8*.log` |

**Not verified yet**

- The CPU engine tests on the head (`test_lookup_engine.py`, 19 min under load; `test_engine_mtp.py`, 30 min).
  `test_lookup_engine.py` is running on a copy of `7ec81b17` with the conv index cast by hand
  (`reports/spec-lookup/logs/m4_scratch_test_lookup_engine.log`); mtp-card-ready's own cast (`9d013a36`) is
  merged since `e121d61b`. An earlier `test_engine_mtp.py` run on a tree that changed under it had 12 passed
  and 2 artefacts (`m1_test_engine_mtp.log`).
- `bin/gate /Volumes/Storage/Claude/gaudi-work/wt/spec-lookup --tools` on the head (needed before a pull request).
- **Nothing of this track has run on a card yet.** MTP on the card (mtp-card-ready's
  `docs/design/flash-next/mtp-card-ready.md`, job 00021): two card failures fixed (`f0e7d02f`, `9d013a36`),
  then MTP k = 3 on the real model at TP8 + EP8, one stream: 2.78 tokens per step, acceptance per position
  0.83 / 0.57 / 0.39, 84.3 ms per step against 41.6 ms without speculation (31.0 against 41.6 ms per token).
  On the tiny model every MTP draft is rejected. Accepted drafts beyond what MTP drafts, widened steps and
  block-completion commits have not run on a card: jobs 00086 and 00087 are the first.

**Pending, in order**

1. **Read the tiny rehearsals** (00086, 00087) and report them (above). If a configuration fails: the traceback
   is in `reports/spec-lookup/card/<id>-<name>/<config>/server0.log`; `lookup_tiny.json` has every stream.
   Then the hybrid's rehearsal, which needs nothing new:
   ```bash
   F=/Volumes/Storage/Claude/gaudi-work/bin/farm
   $F job spec-lookup tiny2-mtp3-lookup any:2 12m --no-model -- python tools/flash_next/rental/lookup_tiny.py --configs mtp3-lookup
   ```
2. **Real-model speculative jobs, TP4 + EP4** (section 6 has the rounds), each after its tiny rehearsal passed.
   One of them can still take the other half of 00029's place until 21:20 UTC (`--replaces 00029`; the
   no-speculation job took the first half); after that they queue at the end:
   ```bash
   $F job spec-lookup real4-ngram8 any:4 30m --replaces 00029 -- bash tools/flash_next/rental/lookup_serve.sh \
       --label ngram8 --ngram 8 --rounds edit:32:384:-,prose:32:256:-,edit:1:384:-,prose:1:256:-,editl:8:512:-
   $F job spec-lookup real4-mtp3-lookup any:4 30m -- bash tools/flash_next/rental/lookup_serve.sh \
       --label mtp3lookup --mtp 3 --lookup 1 --rounds edit:32:384:0,edit:32:384:1,edit:32:384:2,prose:32:256:0,prose:32:256:1,edit:1:384:0,edit:1:384:1,edit:1:384:2,prose:1:256:0,prose:1:256:1,editl:8:512:0,editl:8:512:1
   ```
   Reread CARD.md before each submit. Results: `probe.json`, `probe.log`, `server0.log`; compare texts across
   jobs with `python3 tools/flash_next/rental/lookup_probe.py --compare A/probe.json B/probe.json ...`.
3. **Numbers to section 7 and to the coordinator** (SendMessage to `main`): accepted tokens per step, ms per
   step, effective ms per token for lookup and the hybrid against MTP alone and against no speculation, on
   edit-and-return and on fresh prose, at 1 and 32 streams.
4. **Depth beyond 5 and the search's host cost** (sections 4 and 5): the hybrid's lookup rows are capped at
   the MTP k (the rows are the MTP shapes); deeper lookup drafts need either the n-gram server (k = 8 proven on
   CPU) or hybrid rows wider than MTP's (not built). `HybridLookup.merge` runs vLLM's O(context) search once
   per request and step: MEASURED on the workstation 0.10 ms per search at 128,000 tokens, 3.2 ms per step for
   32 requests (0.1 ms per step at 4,000 tokens); the index drafter of `tools/flash_next/lookup_replay.py`
   is O(n) per step and its `follow` policy accepts more in the replay. Not yet in the server.
5. **Merge** per the protocol: after mtp-card-ready's pull request (this branch contains that branch), run the
   gate on the merged tree, `gh pr create --base feat/flash-next-gaudi`, `gh pr merge --merge`. The flags of
   this track are its own (`VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP`, `..._LOOKUP_MIN_NGRAM`, `..._LOOKUP_MAX_NGRAM`,
   `..._LOOKUP_ORACLE`), all default off.

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
(2 to `1 + k`). INFERRED consequence for the card: each (batch bucket, width, block bucket) is its own HPU
graph, so an n-gram server compiles up to `k` times the decode graphs of an MTP server; the hybrid keeps
MTP's one width.

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

None yet (2026-11-20 19:50 UTC).

## 8. Notes for other tracks

- mtp-card-ready: every prompt of 1,024 tokens or more gets a prefill chunk that completes no prompt on an MTP
  server with prefix caching (not only prompts above the 2,048-token threshold): `vllm/v1/core/sched/
  scheduler.py` `_mamba_block_aligned_split`, `last_cache_position = num_tokens - num_tokens % block_size`,
  minus one block with `use_eagle_block_drop`. Without `VLLM_HPU_MTP_CHUNK_KV=1` the draft layer has no KV for
  those blocks (VERIFIED in the scheduler code; the effect on acceptance is not measured).
- mtp-card-ready: `causal_conv1d_pytorch.py:498` needs a long index for `index_copy_` on CPU (see the handoff).
