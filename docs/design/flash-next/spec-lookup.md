# Flash-Next prompt lookup (n-gram) drafting and the lookup + MTP hybrid

Track `spec-lookup`, branch `feat/flash-next-gaudi-spec-lookup`. Labels: **VERIFIED** (by the named test, log or
code line), **INFERRED** (reasoning on verified inputs), **MEASURED** (a card job, with its id).

This file is also the track's report and handoff: the agent's harness does not write report files, so the
handoff lives here, on the branch (BRIEF.md, 19:00 UTC entry, point 5). Raw material that is not in the
repository is under `/Volumes/Storage/Claude/gaudi-work/reports/spec-lookup/` (`logs/`, `replay/`, `card/`,
`realcfg/`).

## Handoff (if this agent stops)

State at **2026-11-20 19:50 UTC**. Branch head `628d9716` plus this file, pushed.

**Done and verified on CPU**

| What | Evidence |
|---|---|
| n-gram lookup with prefix caching through the real engine (scheduler, HPU worker and runner, tiny model): ragged oracle drafts with tampered ones (every accepted count 0..3, empty drafts, 11 widened steps, 3 block-completion commits, 440 tokens compared), full acceptance across 256-token block boundaries, vLLM's own `NgramProposer`, prefix caching off, k = 8 (every accepted count 0..8, 21 widened steps), the mutation without the widening breaks | `tests/unit_tests/models/flash_next/test_lookup_engine.py`: **8 passed** in 1,156 s on the tree of `3d536265` (this branch + mtp-card-ready `9722b888`), log `reports/spec-lookup/logs/m1_test_lookup_engine.log` |
| The lookup + MTP hybrid through the same engine (lookup rows padded with -1 and MTP rows in the same streams, lossless up to a bf16 near-tie) | the same run: `test_hybrid_lookup_beside_mtp_with_prefix_caching` passed (it was skipped before the merge: the MTP engine did not start) |
| Hybrid proposer, its modes, the per-request mode, the per-load log line, the rehearsal oracle | `tests/unit_tests/worker/test_flash_next_lookup.py` 19 passed (head) |
| Verification width helpers | `tests/unit_tests/worker/test_verify_width.py` 12 passed |
| Platform: n-gram with prefix caching served, k limits `{"mtp": 5, "ngram": 8}` | `tests/unit_tests/test_flash_next_platform.py` 75 passed (head) |
| Card job tools | `tests/unit_tests/tools/rental/test_lookup_probe.py` 10 passed, `test_lookup_tiny.py` 9 passed, `test_spec_probe.py` 13 passed (head) |
| Model-level n-gram tests at k = 3, 4, 5, 8 (earlier agent, `7dcc65a3`; not rerun after the merges) | `tests/unit_tests/models/flash_next/test_ngram_end_to_end.py`, logs `reports/spec-lookup/logs/ngram_k8*.log` |

**Not verified yet**

- The CPU engine tests on the head. mtp-card-ready's fix `f0e7d02f` passes an int32 index to `index_copy_`
  (`vllm_gaudi/ops/causal_conv1d_pytorch.py:498`), which the CPU kernel refuses, so every CPU engine test of a
  speculative config stops at its first prefill until that track casts the index (reported 19:45 UTC; do not
  change those lines here, take their commit). Then rerun `test_lookup_engine.py` (19 min under load) and
  `test_engine_mtp.py` (30 min; a run that started on the tree of `3d536265` while later merges changed the
  files under it had 12 passed, 1 failure from this defect and 1 from a file read mid-merge:
  `reports/spec-lookup/logs/m1_test_engine_mtp.log`).
- `bin/gate /Volumes/Storage/Claude/gaudi-work/wt/spec-lookup --tools` on the head (needed before a pull request).
- **Nothing of this track has run on a card.** The MTP server starts on the card since job 00021 (healthy after
  435 s); its first prompt died (the conv-state slice write, fixed by `f0e7d02f`); mtp-card-ready's tiny run with
  the fix then failed with `Empty tensor optional` (19:34 UTC, their job 00021 hold). The n-gram server shares
  the target's speculative path (widened PLE and GDN conv state, state lattices), not the drafter.

**Pending, in order**

1. **Tiny-model rehearsal on the card** (coordinator, 19:26 UTC: the real model only after the tiny model runs
   drafts and verification cleanly). One server is about a minute; each job is a baseline server plus the named
   configurations (`tools/flash_next/rental/lookup_tiny.py`, section 6):
   ```bash
   F=/Volumes/Storage/Claude/gaudi-work/bin/farm
   $F job spec-lookup tiny2-ngram3 any:2 10m --no-model -- python tools/flash_next/rental/lookup_tiny.py --configs ngram3
   $F job spec-lookup tiny2-ngram8-search any:2 14m --no-model -- python tools/flash_next/rental/lookup_tiny.py --configs ngram8,ngram3-search
   $F job spec-lookup tiny2-mtp3-lookup any:2 10m --no-model -- python tools/flash_next/rental/lookup_tiny.py --configs mtp3-lookup
   ```
   Results: `reports/spec-lookup/card/<id>-<name>/lookup_tiny.json` and one `server0.log` per server. The
   `mtp3-lookup` job needs mtp-card-ready's MTP path to run on the tiny model first.
2. **Real-model jobs, TP4 + EP4, replacing the queued 8-card job 00029** (do not let 00029 run: it is the old
   stage-1 plan with `min_tokens` and a tree without the MTP fixes). Section 6 has the rounds. With the tiny
   rehearsal of the matching configuration passed:
   ```bash
   $F job spec-lookup real4-ngram8 any:4 30m --replaces 00029 -- bash tools/flash_next/rental/lookup_serve.sh \
       --label ngram8 --ngram 8 --rounds edit:32:384:-,prose:32:256:-,edit:1:384:-,prose:1:256:-,editl:8:512:-
   $F job spec-lookup real4-mtp3-lookup any:4 30m --replaces 00029 -- bash tools/flash_next/rental/lookup_serve.sh \
       --label mtp3lookup --mtp 3 --lookup 1 --rounds edit:32:384:0,edit:32:384:1,edit:32:384:2,prose:32:256:0,prose:32:256:1,edit:1:384:0,edit:1:384:1,edit:1:384:2,prose:1:256:0,prose:1:256:1,editl:8:512:0,editl:8:512:1
   $F job spec-lookup real4-nospec any:4 25m -- bash tools/flash_next/rental/lookup_serve.sh \
       --label nospec --rounds edit:32:384:-,prose:32:256:-,edit:1:384:-,prose:1:256:-,editl:8:512:-
   ```
   (`--replaces` lets two 4-card jobs take the place of the 8-card job; the third queues at the end.) Reread
   CARD.md before each submit. Results: `reports/spec-lookup/card/<id>-<name>/probe.json`, `probe.log`,
   `server0.log`; compare texts across jobs with
   `python3 tools/flash_next/rental/lookup_probe.py --compare A/probe.json B/probe.json ...`.
3. **Numbers to the report (section 7) and to the coordinator** (SendMessage to `main`): accepted tokens per
   step, ms per step, effective ms per token for lookup and the hybrid against MTP alone and against no
   speculation, on edit-and-return and on fresh prose, at 1 and 32 streams.
4. **Depth beyond 5 and the search's host cost** (section 4 and 5): the hybrid's lookup rows are capped at the
   MTP k (the rows are the MTP shapes); deeper lookup drafts need either the n-gram server (k = 8 proven on
   CPU) or rows wider than MTP's. `HybridLookup.merge` runs vLLM's O(context) search once per request and step
   in Python: at 32 x 128K tokens that is INFERRED to cost several ms per step; an incremental n-gram index
   (the `IndexDrafter` of `reports/spec-lookup/replay/replay.py`, "most recent match, follow the copy") makes it
   O(k) and accepts more in the replay (section 5).
5. **Acceptance estimates by replay** on the real model's own outputs: the texts of the card jobs
   (`probe.json`) replayed through the drafters of `replay.py` (today's estimates use synthetic edits).
6. **Merge** per the protocol: after mtp-card-ready's pull request (this branch contains that branch), run the
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

## 5. Acceptance estimates by replay (REPLAY ESTIMATE, synthetic outputs)

`reports/spec-lookup/replay/replay.py` plays known (prompt, output) token streams through lookup drafters
(greedy verification accepts a draft token exactly when it equals the output's token). The outputs are
synthetic coding-agent answers (a repository file returned with one-line docstrings, type hints or renamed
locals), a second turn editing the first answer, and prose written from a one-line request; real tokenizer.
Tokens per step (`replay.log`):

| Drafter | edit k=3 | k=4 | k=5 | k=8 | k=16 | prose k=3 |
|---|---|---|---|---|---|---|
| vLLM search, n 2..4 (earliest match) | 3.40 | 4.08 | 4.73 | 6.48 | 10.24 | 1.13 |
| most recent match, n 3..8 | 3.53 | 4.25 | 4.94 | 6.72 | 10.45 | 1.09 |
| most recent match, n 3..8, follow the copy | 3.66 | 4.44 | 5.19 | 7.18 | 11.26 | 1.09 |
| earliest match, n 3..8 | 3.57 | 4.31 | 5.03 | 6.89 | 10.78 | 1.09 |

A lookup draft is on the table for 82 to 94 % of the steps of an edit (k up to 8) and for 6 to 18 % of the steps
of prose (with n >= 3: 6 to 7 %, and 1.2 to 1.8 tokens accepted per hit). These figures say what lookup can accept when
the model does copy; whether the real model copies as these synthetic answers do is what the card job measures.

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
  prompt tokens (VERIFIED with the checkpoint's tokenizer and chat template). Under 1,024 tokens on purpose:
  with prefix caching the scheduler's `_mamba_block_aligned_split` ends a prefill chunk at the last cacheable
  512-token block boundary, one block earlier on an MTP server (`use_eagle_block_drop`), and the MTP drafter
  never sees a chunk that completes no prompt (`VLLM_HPU_MTP_CHUNK_KV=0`); below 1,024 tokens an MTP server
  prefills the prompt in one chunk, so MTP alone is measured with its full context.
- `editl`: the same with files of 3,164 to 3,601 tokens (on an MTP server most of the prompt is in chunks the
  drafter did not see: this round shows what that costs MTP, and what lookup accepts on longer copies).
- `prose`: a story request, 39 to 41 tokens, nothing to copy.

No `min_tokens`: `max_tokens` with `ignore_eos`. Stream `i` of a round always gets the same prompt, so the
texts of all rounds and jobs compare token for token. Per round: tokens per verification step per request and
the rate of each draft position (the server's counters), ms per engine step (the median gap between a stream's
chunks), ms per token over the decode span and steady, the client-side step sizes.

## 7. Card results

None yet (2026-11-20 19:50 UTC).

## 8. Notes for other tracks

- mtp-card-ready: every prompt of 1,024 tokens or more gets a prefill chunk that completes no prompt on an MTP
  server with prefix caching (not only prompts above the 2,048-token threshold): `vllm/v1/core/sched/
  scheduler.py` `_mamba_block_aligned_split`, `last_cache_position = num_tokens - num_tokens % block_size`,
  minus one block with `use_eagle_block_drop`. Without `VLLM_HPU_MTP_CHUNK_KV=1` the draft layer has no KV for
  those blocks (VERIFIED in the scheduler code; the effect on acceptance is not measured).
- mtp-card-ready: `causal_conv1d_pytorch.py:498` needs a long index for `index_copy_` on CPU (see the handoff).
