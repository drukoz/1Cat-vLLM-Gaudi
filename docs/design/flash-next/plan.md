# RUK-2709 Flash-Next on Gaudi: plan

Kind: **Program**. Version 0.2, 2026-11-17: v0.1 was written from the system map (v0.5) and the reuse ledger (v0.3) after all Phase 0 research (R1–R7, R4b) and spikes (S0, S3, S4) reported; v0.2 rewrites the M1 section after M0 merged (`383e5ab`) with the facts checked for it. The working copy is the `plan` document on RUK-2709; this file is refreshed with the pull request.

## What changed between the map and this plan

Three facts, each checked on this Mac in the S0 environment (vLLM `8f816a3f`, torch 2.13.0, Python 3.12), reshape the first milestone:

1. **Upstream's Flash-Next model imports on CPU at the pin.** `vllm.models.qwen4_exp.nvidia.model` and `nvidia.mtp` import without Triton (the kernels are reached through `vllm.triton_utils` stubs and custom ops that are only *called* in forward). R4's "not importable on HPU" holds at `v0.34.0`, not at the pin. The fork's existing class-replacement mechanism reaches the upstream model unchanged: after `import vllm_gaudi.models.qwen3_5` and `qwen3_next`, `nvidia.model.QwenGatedDeltaNetAttention is HPUGatedDeltaNetAttention` and `Qwen4ExpSparseMoeBlock.forward` is the fork's HPU MoE forward (CPU probe, 2026-11-17). So the HPU model is **a subclass of upstream's class plus three module-level replacements** (HyperConnection, PLE layer, QSA attention), not a vendored copy. R4b's 21 fixes are applied as subclass overrides or patches; a fix that cannot be expressed that way vendors the one file it touches, decided per fix in M1+.
2. **Hugging Face transformers has a pure-PyTorch Flash-Next.** `transformers.models.qwen4_exp.modeling_qwen4_exp` (5.23.0 in the S0 venv; the node pins 5.16.1, and the checkpoint's own `transformers_version` is 5.8.0.dev0) runs on CPU with reference fallbacks for the GDN chunk rule, causal conv1d, QSA block top-k and PLE. A random tiny model built from the fixture below ran a CPU forward (67M parameters, 256 MiB in fp32). This is the program's **CPU oracle**: every HPU path is compared with HF on the tiny model on CPU at TP1, on 2/4/8 CPU ranks through S3, and in stage 2 of the hardware run sheet.
3. **Dense attention equals QSA for short prompts.** The indexer selects up to `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens plus the current block, so for prompts of at most 2048 tokens every block is selected and dense attention is exact. M0's dense QSA fallback is therefore correct for the short-prompt parity checks; real sparsity (M2) is a long-context performance feature, not a correctness prerequisite. vLLM's config validation fixes the block budget at 512 or 2048, so the tiny fixture keeps the real indexer budget.

Instantiating upstream's class through vLLM's loader on CPU with the tiny config (probe, 2026-11-17) gets through the GDN, PLE, HyperConnection and MoE layers and stops only at `Qwen4ExpQSAAttention.__init__` (`nvidia/qsa.py:108`, "Qwen4Exp QSA requires FlashAttention"). The PLE layer declares a `MambaSpec` with `mamba_type` `SHORT_CONV`, one bf16 tensor of shape `(9, hc_count × hidden)` per slot (9 = `(ple_conv_kernel_size − 1) × ngram_size`; 180 KiB per slot on the real model); the plugin has no short-conv code today (`grep -ri short_conv vllm_gaudi` is empty). No runner at the pin produces the `ngram_context` / `query_start_loc` kwargs the PLE path needs (`nvidia/model_state.py` does it only for the GPU Model Runner V2, which the fork does not use).

## Assumptions

1. The first hardware layout is BF16 at TP8 on one server (system map 2a); the two-server FP8 layout is M4.
2. `--language-model-only` is the production setting; the vision tower is loaded as a `StageMissingLayer` and gets no work.
3. transformers 5.16.1 on the node contains `qwen4_exp` (the checkpoint declares 5.8.0.dev0). The CPU reference is produced on this Mac, so the node's transformers version only matters for the tokenizer. *(unverified)*
4. The real tokenizer files (`Qwen/Qwen3.8-Flash-Next`) can be downloaded on this Mac; the generator also accepts a local path. If the repository is gated, the user provides the files.

## Target architecture

| Piece | Where | What it is |
|---|---|---|
| Model package | `vllm_gaudi/models/qwen4_exp/` (new) | `HpuQwen4ExpForConditionalGeneration` and `HpuQwen4ExpForCausalLM` subclasses of upstream's classes; the package's `__init__` applies the three replacements into `vllm.models.qwen4_exp.nvidia.model` the way `vllm_gaudi/models/qwen3_5.py:757-765` replaces the GDN class, and registers the names through `vllm_gaudi/models/__init__.py::register_model` |
| HyperConnection | `vllm_gaudi/models/qwen4_exp/hyperconnection.py` | `GatedResidual` with upstream NVIDIA's constructor, weight layout (`input_mix_weight_down_block_inject` merged linear, `hc_norm`, `input_mix_weight_up`) and `mix` / `combine` / `combine_and_mix` API, implemented with plain torch instead of the five Triton kernels in `nvidia/ops/hc.py`. Reference: upstream's `common/hyperconnection.py` torch `GatedResidual` and HF `Qwen4ExpTextGatedResidual` |
| PLE layer | `vllm_gaudi/models/qwen4_exp/ple_layer.py` | Subclass of `Qwen4ExpPLELayer`: torch gate (`ple_gate` formula from `nvidia/ops/ple.py` docstring) and dilated short conv with its state in the SHORT_CONV mamba slot the fork's runner provides; n-gram ids already have a torch path (`nvidia/ple_layer.py:380-437`) |
| QSA attention | `vllm_gaudi/models/qwen4_exp/qsa.py` | Subclass of `Qwen4ExpQSAAttention` with `QsaMode` (`IntEnum`: `DENSE = 0`, `SPARSE = 1`, env `VLLM_HPU_FLASH_NEXT_QSA_MODE`, default 0). Dense mode: no indexer modules or caches, the fork's `HPUAttentionBackend`, forward as `_hpu_qwen3next_attention_forward` with q/k norm and interleaved mRoPE through `HPUMRotaryEmbedding` (`hpu_rotary_embedding.py:628`, interleaved sections supported at :44-47). Sparse mode is M2 |
| Runner | `vllm_gaudi/v1/worker/hpu_model_runner.py`, `hpu_worker.py`, `platform.py` | Architecture allow-lists learn `Qwen4ExpForConditionalGeneration` / `Qwen4ExpForCausalLM`; `ngram_context` and `query_start_loc` kwargs for models with `requires_raw_input_tokens`; SHORT_CONV mamba group allocation and state views; later the GDN state pool (M1) and indexer caches (M2) |
| Test fixture | `tests/unit_tests/models/flash_next/tiny_config.json` (committed), `tools/flash_next/make_tiny_checkpoint.py` | A 4-layer Flash-Next with the real head counts (16/48 GDN heads, 4/2 attention heads, 4 indexer heads, 16 experts) and tiny dims, random seeded weights saved under HF names, the real tokenizer, and `reference.json` with HF's greedy outputs and top-5 logits |
| CPU tests | `tests/unit_tests/models/flash_next/` | Component parity (HC, PLE gate/conv, QSA dense vs torch SDPA), whole-model parity vs HF at TP1, TP1/2/4/8 equivalence through `spikes/cpu-tp/harness.py`, kv-cache-spec enumeration, runner kwargs |
| Run sheets | `docs/design/flash-next/run-sheets/` | Template plus `qualification.md`, stages 1–6, with S4's predicted `Usable num_blocks` / `Maximum concurrency` per layout |

Conventions for every coding task in this program:

- The shared CPU environment is `/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv` (S0). Nobody runs `pip install` into it: it is shared by every parallel task. Tests run from the worktree root with `PYTHONPATH="$PWD"`, which makes the worktree's `vllm_gaudi` win over the venv's editable install (checked 2026-11-17); every spec's Setup proves it with `vllm_gaudi.__file__`.
- Stock macOS has no `timeout(1)`: Verify commands use `perl -e 'alarm N; exec @ARGV'` when a cap is needed.
- New behaviour that is not the plain torch path ships behind a flag in `vllm_gaudi/envs.py`, default off, with an integer enum when the value is a mode.
- Enums are integers everywhere (`IntEnum`); no string modes.
- Nothing is committed under `bin/`, `obj/`, `.paperclip-runtime/`, or the venv.

## What we do not build

- The vision tower path, LoRA, EPLB, pipeline parallel (only if M4 measurements call for 2×TP4 with PP, through #60234 from the carry list).
- Model Runner V2's `Qwen4ExpModelState`: the fork runs the V1 runner; the same bookkeeping goes into the HPU runner.
- DFlash2 for Flash-Next (no draft checkpoint exists); MTP is the speculative method (M3).
- A vendored copy of upstream's model files, unless a backport needs it (see above).
- Sparse QSA kernels before the dense path is proven (M2).

## Verification model

From the system map (section 14): all code passes the CPU gates before the server is rented, and one qualification run sheet runs on the hardware. The gates, in the order a task runs them:

1. `PYTHONPATH="$PWD" $PY -m pytest -q -p no:cacheprovider <the task's test files>`.
2. The S0 regression: `spikes/mac-cpu-env/run_cpu_tests.sh` on the files it recorded as `pass` (66 files, 1550 tests); run at milestone verification, not per task.
3. The S3 eight-rank harness for every sharded layer a task adds or changes.
4. The tiny-model parity test against HF's reference (T7), from M0 on.

The oracle chain: HF transformers on CPU → HPU model class on CPU (TP1, fp32) → the same on 2/4/8 CPU ranks → the tiny checkpoint on the Gaudi node at TP1 and TP8 in bf16 (TP8 uses the 8-head variant `tiny_tp8_config.json`, since 4 attention heads cannot split over 8 ranks; system map §17 fact 22) (stage 2 of the run sheet, greedy token agreement over 16 tokens on 4 prompts) → the real checkpoint on the node, coherence on 8 short prompts and a 50-item GSM8K sample against the published score.

## Milestones

### M0 Foundations (branch `feat/flash-next-gaudi-m0`, cut from `feat/flash-next-gaudi`)

Delivers: the Flash-Next model class instantiates and runs on CPU through the plugin, matches HF on the tiny model at TP1 and on 8 CPU ranks (8-head variant at TP8), and the first run sheet (stages 1–2) is ready for a rental session. Subtasks and their real dependencies:

| # | Subtask | Depends on | Owner |
|---|---|---|---|
| T1 | Tiny checkpoint generator and HF CPU reference (`tools/flash_next/make_tiny_checkpoint.py`, `reference.json`) | the committed `tiny_config.json` | Engineer 1 |
| T2 | Registration, architecture allow-lists, package skeleton (`vllm_gaudi/models/qwen4_exp/__init__.py`, `platform.py`, runner `mamba_like_arch`) with CPU tests | none | Engineer 2 |
| T3 | Torch `GatedResidual` with the NVIDIA layout and API, tested against upstream's `common` torch class and HF's | none | Engineer 3 |
| T4a | Torch PLE gate and dilated short conv as pure functions with state in/out, tested against HF's `Qwen4ExpTextPLELayer` | none | Engineer 4 |
| T4b | PLE layer subclass wired to the fork's mamba state views (SHORT_CONV group) | T2, T4a, and the T5 instantiation report | later |
| T5 | HPU QSA attention, dense mode, plus the whole-model CPU instantiation and kv-cache-spec enumeration | T2 (registration) | Engineer 5 |
| T6 | Runner kwargs: `query_start_loc` and `ngram_context` for models with `requires_raw_input_tokens`, including warmup | none | General Agent |
| T7 | Whole-model parity vs HF at TP1 and TP1/2/4/8 equivalence through S3 | T1–T6 | later |
| T8 | Run-sheet template and `qualification.md` stages 1–2 (build, TP8 BF16 bring-up, S4 predictions, capture lines) | none (final numbers from T7) | Engineer 6 |

Exit: T1–T8 merged into the milestone branch, the S0 regression green, the QA Engineer's verification task passed, the milestone merged into the program branch and the PR description updated.

### M1 Admission, prefix reuse and the recurrent state pool (`feat/flash-next-gaudi-m1`, RUK-2759)

Cut from the program branch at `383e5ab` (M0 merged). Four facts checked on this Mac on 2026-11-17 changed the M1 scope from v0.1:

1. **The fork's GDN layer writes the conv state to the wrong slot under prefix caching.** `vllm_gaudi/models/qwen3_5.py` already resolves load and store slots and writes the SSM prefill state to the store slot, but the prefill conv (`:461-477`) reads and writes at the load slot, and the decode SSM and conv (`:629-647`, `:710-720`) write where they read. This is upstream vllm-gaudi #1831's bug (LongBench 0.23 -> 0.51 when fixed); it hits every prefill chunk after the first and every 512th decode token, so prefix reuse could never have produced correct output on the fork. M1-T4 fixes it; M1-T5 proves it with a chunked-prefill parity test on CPU (load != store), a case M0's tests did not cover.
2. **#1828, #1853 and #1890 are not needed at the pin.** vLLM `8f816a3f`'s base `Scheduler._mamba_block_aligned_split` (`vllm/v1/core/sched/scheduler.py:394-470`) already clips every chunk end to the mamba block in `align` mode, and Flash-Next runs the base scheduler (async scheduling is off, M0-T6b). The fork's `HPUAsyncScheduler` override (the 27B with async scheduling) falls through to the base for GDN models. The runner's `context_lens % mamba_chunk_size` assert stays as a guard.
3. **The PLE short-conv group doubles the waste.** `hpu_worker.py:1142-1166` applies a second shrink factor `page / (page + ple_state)` for the SHORT_CONV spec, so Flash-Next today gets `40 GiB x 0.567 x 0.740` for attention at TP8 BF16: 2,862 blocks, 1.47M KV tokens, about 11 agents at 131K, not the 1.98M of system map v0.5. The recurrent state pool covers the GDN and PLE groups alike.
4. **Admission needs no plugin code.** R7 found no plugin cap beyond the zero-memory default (`platform.py:276-291`, which keeps vLLM's 2048-token step budget) and prefills that never merge for hybrids; the knobs are CLI flags, which stage 4 measures (variants A-C) before any default changes.

**Design of the recurrent state pool** (contract M1-T1, scheduler M1-T2, worker/runner M1-T3): two `BlockPool`s inside an `HybridKVCacheCoordinator` subclass installed through a plugin patch of `vllm.v1.core.kv_cache_manager.get_kv_cache_coordinator`. Every `MambaSpec` group's manager allocates from a pool of `slots + 1` blocks (id 0 null); attention keeps `num_blocks`; a routing attention pool forwards deferred frees, resets and events; `get_num_blocks_to_allocate` returns the attention need or "not enough" when the recurrent pool is short. The pool size is resolved once by the platform hook into `VLLM_HPU_RECURRENT_STATE_POOL_SLOTS` (`max_num_seqs x groups x 3 + 512` cache slots: 896 for Flash-Next at 32 sequences) so the engine core and the workers agree; the worker reserves `(slots + 2) x (12 x 391.5 KiB + 180 KiB)` and gives attention the rest with no shrink factors; the runner allocates `slots + 2` rows per recurrent tensor and pads with `slots + 1`. Mode flag `VLLM_HPU_RECURRENT_STATE_POOL` (`RecurrentStatePoolMode`: 0 off, 1 pool), default off until stage 4 qualifies it. Compact GDN with prefix caching becomes a startup error, and unset defaults to 0 under prefix caching. Prediction (S4 with the pool model, M1-T8): TP8 BF16, 40 GiB, 896 slots -> 6,113 attention blocks, 3.13M KV tokens, 24 agents at 131K, against 2,862 blocks / 1.47M tokens / 11 agents today.

| # | Subtask | Depends on | Owner |
|---|---|---|---|
| T1 | Pool contract module, env flags, platform resolution, compact-GDN rule, CPU tests | none | Engineer 2 |
| T2 | Scheduler side: `HpuRecurrentPoolCoordinator`, routing pool, factory patch, `KVCacheManager` tests on CPU | T1 | Engineer 5 |
| T3 | Worker memory budget and runner tensor sizing for the pool | T1 | Engineer 4 |
| T4 | Conv read/write slot split, decode store slots, null-block masks (#1831, #1895) with op tests | none | Engineer 3 |
| T5 | Chunked-prefill and block-crossing decode parity on CPU | T4 | Engineer 6 |
| T6 | 700-token prompt in the HF reference (multi-chunk GDN at TP1 to TP8) | none | Engineer 6 |
| T7 | Bucket coverage test for 128K-262K prompts, bucket counts | none | General Agent |
| T8 | S4 pool model, run sheet stage 4 (variants A-F, hit check, pool), client prefix guidance | none | Engineer 1 |

Exit: T1-T8 merged; the QA verification passed (the Flash-Next suite with the fifth prompt and the chunked test, `tests/unit_tests/core`, the S0 regression, and two mutation checks: the conv write slot reverted fails T5, the pool factory disabled fails T2's tests); the milestone merged into the program branch and the PR updated. Hardware proof is run sheet stage 4 on the user's node.

### M2 Decode batching and sparse attention (`feat/flash-next-gaudi-m2`)

Fused GDN decode tactics for the TP8 and TP4 shapes (native, compile-unchecked, behind flags; A/B in stage 3); QSA sparse mode: indexer (torch rope and scoring, top-k; the HPU DSA decode indexer `hpu_sparse_attn_indexer.py` from upstream vllm-gaudi #1870 as the model) with the raw and compressed key caches on the HPU worker, block-gather sparse GQA, long-context parity against HF on prompts above 2048 tokens; MoE decode for 512 experts (`MAX_EXPERTS_PER_SLICE`, R6); graph replay and host overhead at batch 16–32.

### M3 MTP speculative decoding (`feat/flash-next-gaudi-m3`)

`Qwen4ExpMTP` through the fork's EAGLE proposer (`hpu_eagle.py`, method `mtp`), draft verification and GDN/PLE state rollback (DFlash2's `gdn_state_layout.py` checkpoints), #55513 and #61770 from the carry list, IndexShare top-k reuse once sparse mode exists; CPU parity of the draft's logits against HF.

### M4 Two servers and eight cards (`feat/flash-next-gaudi-m4`)

FP8 at TP4 needs expert parallelism (R6: 640/4 = 160 per rank is not a multiple of the 128-wide FP8 blocks; `--enable-expert-parallel` gives each rank 128 whole experts), so: TP4 + EP4 block-FP8 MoE, INC FP8 with the conv1d blocklist as the alternative; PLE table on the host (backport #54371's structure with an HPU gather, or the fork's Engram host gather) to fit BF16 at TP4; TP8 fused AllReduce + residual + RMSNorm (`tp2_fused_ar_norm.py` pattern); run-sheet stage 5 comparisons (TP8, 2×TP4, TP4+EP).

### M5 Gaudi2 extras

Only what stages 3–6 show: FP8 KV cache (block 896 or 1664), warmup time, sampler, communication overlap, the compiled decoder (system map section 16, items 2c and 3a–3c).

## Qualification run sheet (stages, fail-fast; T8 writes stages 1–2)

1. Build the image at the program branch and compile native code; record `flashinfer_gaudi.get_capabilities()`.
2. TP8 BF16 bring-up on the **tiny checkpoint** first (greedy agreement with `reference.json`, TP1 then TP8), then the real checkpoint: startup lines (`Usable num_blocks`, `Maximum concurrency`, `kv cache group sizes`, `GDN layers detected`), compared with S4's prediction; 8 short prompts for coherence; the GSM8K sample.
3. Flag-by-flag A/B of each native path against its torch path.
4. R7's variants A–D and the prefix-cache hit rate.
5. TP1/2/4/8, PP2×TP4 and TP+EP throughput and latency.
6. 128K–262K contexts and the eight-agent stress run.

## Spikes

| Spike | Result | Run against the real thing? |
|---|---|---|
| S0 Mac CPU environment (RUK-2717) | pass: venv at the pin, 66 of 175 test files pass on CPU | CPU only, by design |
| S3 eight-rank CPU harness (RUK-2732) | pass: TP1/2/4/8 equivalence and collective order | CPU only, by design |
| S4 KV capacity calculator (RUK-2734) | pass: reproduces Run 1 exactly and Run 2 within 1% | against the user's logs; the Flash-Next rows are predictions until stage 2 |
| S1 model instantiation through the plugin | folded into T2 and T5 (probe done by the CTO: stops at the QSA class) | CPU |

## Checklist (programs.md)

- [x] Two sentences: the plugin serves Qwen3.8-Flash-Next on the company's Gaudi2 node for the Paperclip agent fleet (long shared prefixes, many concurrent 128K contexts). Users are the agents and the user who runs the node.
- [x] External systems: vLLM at the pin, the checkpoints on the node, the Gaudi stack, INC, HF hub for the tokenizer (system map section 5).
- [x] Every capability has a ledger row (reuse ledger v0.3: 19 rows), every build row says why.
- [x] Nothing hand-written could be generated or taken from a library: the model skeleton, GDN, MoE, attention backend and HC reference come from upstream; only HPU seams are written.
- [x] Every "use" has a passing spike: S0, S3, S4; the model-import and HF-reference facts were checked by CPU probes recorded above.
- [x] M0 ends in something a user can run: the tiny model served on the node (stage 2) and the real checkpoint bring-up.

## Risks

- **Short-conv state on the HPU runner** (T4b): new state kind for the fork's hybrid allocation; the fallback is to hold the PLE conv history in a per-request host-side buffer managed by the model (correct, slower), behind the same interface.
- **QSA indexer caches** use `CircularBufferSpec` / `MLAAttentionSpec` layouts the HPU worker does not know; M0 avoids creating them (dense mode), M2 adds them.
- **Numerics in bf16 on HPU** versus fp32 CPU references: stage 2 compares greedy tokens, not logits; the tiny fixture's prompts are short so one divergence is visible.
- **Tokenizer download** needs network; the generator takes `--tokenizer-from <dir>`.
- **The real checkpoint cannot be loaded on this Mac** (335 GiB, 64 GiB RAM): the real-model checks exist only on the hardware.
