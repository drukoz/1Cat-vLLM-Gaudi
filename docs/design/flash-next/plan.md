# RUK-2709 Flash-Next on Gaudi: plan

Kind: **Program**. Version 0.5, 2026-11-18: adds "Track: fused GDN decode kernel" (RUK-2812, merged through #3), which makes the fused kernel the GDN decode path. Version 0.4, 2026-11-17: the M3 section is rewritten from seven facts checked on this Mac after M2 merged (`e4b265f`; system map §17, facts 30-36): upstream's MTP class builds on CPU through the fork's seams after one more rebinding, the target already exports the multi-stream hidden, the GDN verification lattice exists in the fork's layer but the runner builds it only for DFlash2, PLE and sparse QSA refuse speculation today, and prefix caching changes what the lattice means, so MTP ships for prefix caching off first and the align-mode lattice last. Version 0.3, 2026-11-17: the M2 section is rewritten from five facts checked on this Mac after M1 merged (`d635485`; system map §17, facts 25-29): sparse QSA is the correctness requirement for every prompt beyond 2050 tokens, the side caches become plugin-owned tensors, the GDN tactics are torch-level so the TP4/TP8 gates are a Python change. Version 0.2, 2026-11-17: v0.1 was written from the system map (v0.5) and the reuse ledger (v0.3) after all Phase 0 research (R1–R7, R4b) and spikes (S0, S3, S4) reported; v0.2 rewrites the M1 section after M0 merged (`383e5ab`) with the facts checked for it. The working copy is the `plan` document on RUK-2709; this file is refreshed with the pull request.

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

### M2 Sparse QSA, GDN tactics for TP4/TP8 and run sheet stage 3 (`feat/flash-next-gaudi-m2`)

Cut from the program branch at `d635485` (M1 merged, plan v0.2 documents included). Five facts checked on this Mac on 2026-11-17 (system map §17, facts 25-29) set the scope:

1. **Sparse QSA is a correctness requirement, not a performance feature.** HF's `Qwen4ExpTextQSAIndexer` (`transformers/models/qwen4_exp/modeling_qwen4_exp.py:670-775`) defines the model: for the query at position p, the complete blocks are the `nc = (p + 1) // 4` groups of four consecutive tokens `[4b, 4b + 4)`; a block's key is `k_layernorm(mean_fp32(raw_k[4b .. 4b + 3]))` RoPE'd at position `4b`; the score is `sum over the 4 indexer heads of relu(q_h . k_b) / sqrt(128)` with the indexer query normed and RoPE'd at p; the top 512 blocks (all of them while nc <= 512) plus the tail tokens `[4 nc, p]` form the attended set (2048 + 3 = 2051 positions at most), and attention over that set is ordinary GQA with the sigmoid output gate. Dense attention equals this only while nc <= 512, that is p <= 2050. Every 128K agent prompt is far beyond that, so M0's dense mode is wrong for the use case and sparse mode is M2's first deliverable. The HF reference costs 1.0 s for a 2600-token prompt on the tiny model (CPU, this Mac), so the fixture gains a 2600-token prompt and the parity tests run against it in sparse mode.
2. **Upstream's side caches do not transfer to HPU; plugin-owned side tensors do.** Upstream (`vllm/models/qwen4_exp/common/qsa_cache.py`, `nvidia/indexer_qsa.py`) keeps a per-request ring of raw keys as a `CircularBufferSpec` KV-cache group and the compressed keys as an `MLAAttentionSpec(tokens_per_state=4)` group that shares the attention block table, and runs Triton kernels for metadata, scoring, top-k and sparse attention. The HPU worker knows neither spec and has no Triton. On HPU the compressed key cache is a plugin-owned tensor per QSA layer, `[key_cache_rows // 4, 128]` bf16, addressed by `attention_slot // 4` (exact because the kernel block of 128 tokens is a multiple of 4, so a request's compressed row r sits in the block that holds its tokens; 64 B per token per layer, 6.25% of the attention cache at TP >= 2), and the raw-key ring is `[gdn_state_rows, R, 128]` bf16 per QSA layer addressed by the first GDN group's load/store slot, `R = 4 ceil((4 + num_speculative_tokens) / 4)` (4 without speculation; upstream's formula). The ring needs no checkpointing: every slot change happens at a 128-aligned position (chunk ends are aligned by the base scheduler and asserted by the runner, decode slot changes happen at 512-token block boundaries, prefix hits are block-aligned), and at a 4-aligned position the ring is empty. The runner allocates both tensors next to the KV cache and the worker's budget reserves them.
3. **Attention over the selected set reuses the fork's attention.** Prefill: dense paged prompt attention (`vllm_gaudi/extension/ops.py` `prompt_attention`; `_include_past` concatenates the context blocks before the chunk) with an explicit bias that is -inf outside the selection and `is_causal=False`, tiled over queries (default 2048; FusedSDPA's bias plane is limited to 2^31 bytes, `ops.py:470-486`). Cost: the same O(T x L) the 27B pays today for dense attention, plus a T x L x 2 B mask per layer per tile. Decode: a gather of the 2051 selected K/V rows per request through the block table (`block_table[b, pos // 128] * 128 + pos % 128`) and a small SDPA with fixed shapes `[bs, 2051]`: 2 MiB of KV per request per layer instead of the whole context (128 MiB at 131K). Decode scoring gathers the request's compressed rows through the same block table (`[bs, max_model_len // 4, 128]`, fixed shape: 16 MiB per request at 262K, about 3 ms per step at batch 32 and 12 layers; bucketing by context is a stage-3 follow-up). The runner adds a per-request `block_table [bs, max_model_len // 128]` (kernel block ids) to the decode metadata; prefill already has `block_list` per request.
4. **The GDN "FlashInfer-Gaudi" tactics are torch compositions with Python shape gates.** `flashinfer_gaudi/_reference.py` (`qwen38_fused_decode_step_direct`; TP1 and TP2 variants that differ only in constants), `gdn_prefill.py` (`_chunk_gated_delta_rule_log_gate`, tuned by `tactics/gaudi2_qwen38_gdn.json`); the native kernels stay unpromoted (`stable_default: pytorch`). Opening TP4 (4 key / 12 value heads per rank, packed width 2560) and TP8 (2 / 6, 1280) is a Python change with CPU parity tests against the general path; whether the tactics beat the general path at those shapes is a stage-3 measurement, behind `VLLM_HPU_FLASHINFER_GDN_TP4` / `VLLM_HPU_FLASHINFER_GDN_TP8` (default off, like the existing `_TP2`).
5. **MoE with 512 experts has no cap but one untested knob.** `MAX_EXPERTS_PER_SLICE` (default -1: one `mixture_of_experts` call over all 512 experts; `vllm_gaudi/extension/ops.py:58-63, 740`) is the first thing to vary if the stage-2b warmup fails or crawls (R6: 64 experts per slice is the tested point). Stage 2b gains that note; stage 3 measures -1 / 64 / 128.

Flags (all default off, `vllm_gaudi/envs.py`): `VLLM_HPU_FLASH_NEXT_QSA_MODE` (`QsaMode`: 0 dense, 1 sparse; sparse becomes the default after stage 3 passes), `VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE` (query tile of the sparse prefill attention, default 2048), `VLLM_HPU_FLASHINFER_GDN_TP4`, `VLLM_HPU_FLASHINFER_GDN_TP8`.

**Contract of the sparse path** (pasted into every M2 spec that touches it):

- `vllm_gaudi/models/qwen4_exp/qsa_sparse_ops.py`, pure torch, tested on CPU against HF's indexer: `ring_rows`, `gemma_rms_norm`, `rope_neox_partial`, `index_queries`, `compress_block_keys`, `visible_blocks`, `score_blocks`, `select_blocks`, `expand_selection`, `selection_mask`, `compressed_row_slots`, `ring_write`, `ring_block_rows`.
- `HpuQsaIndexer` (`qsa.py`; the M0 `HpuQsaIndexerWeights` extended, same checkpoint names): attributes `compressed_key_cache` (`[rows, 128]` bf16, bound by the runner; `None` on an unbound engine), `key_ring` (`[gdn_state_rows, R, 128]` bf16), `ring_group_idx` (scalar tensor set by the runner: the first GDN group's row in the 2-D state-index tensors); method `select(hidden, positions, cos, sin, md) -> int32 [T, 2051]` of absolute token positions, -1 padded, writing the compressed rows and the ring as a side effect.
- `HpuQwen4ExpQSAAttention.sparse_attn(q, k, v, selected, positions, md) -> [T, H D]`: attention over the selected positions; on HPU `HpuQsaSparseAttention` (`qsa_hpu_attention.py`: prefill bias, decode gather), in the CPU tests `SparseSdpaAttention` (`cpu_model.py`).
- Metadata: `HPUAttentionMetadata.block_table: Optional[Tensor]` `[bs, max_blocks]` int32 of kernel block ids (128 tokens each, pad `_PAD_BLOCK_ID`), filled for Flash-Next decode through `make_decode_metadata(block_table=...)`; prefill uses `block_list` (context blocks per request, padded, logical order), `context_lens_tensor`, `slot_mapping` and `block_size`.
- Invariant asserted by the sparse prefill: `context_len % 4 == 0` on every row (`NotImplementedError` otherwise).

| # | Subtask | Depends on | Owner |
|---|---|---|---|
| T1 | `qsa_sparse_ops.py` and its HF-equivalence tests (prefill mask at 300 and 2600 tokens; token-by-token replay through the ring and the compressed rows) | none | Engineer 1 |
| T2 | Fixture: a 2600-token prompt in `reference.json`; the dense-mode tests restricted to prompts within the exact range | none | Engineer 3 |
| T3 | Sparse forward in `qsa.py` (`HpuQsaIndexer.select`, the `sparse_attn` seam), CPU harness (side-cache binding, `SparseSdpaAttention`, metadata fields), 2600-token prefill parity vs HF with a negative control | T1, T2 | after T1 and T2 |
| T4 | Sparse decode parity (16 teacher-forced tokens after the 2600 prompt), compressed-row and ring state checks, a slot-move test, mutation checks | T3 | after T3 |
| T5a | `block_table` in the HPU attention metadata, `build_block_table` helper with CPU tests, the runner's decode call site | none | Engineer 4 |
| T5b | Side-cache sizing helpers (`qsa_side_caches.py`) with CPU tests; runner allocation and binding; worker budget | none | Engineer 5 |
| T5c | `HpuQsaSparseAttention`: prefill selection bias (CPU-tested) and the tiled `ops.prompt_attention` call; decode gather and SDPA (CPU-tested) | none (names from the contract) | Engineer 6 |
| T6 | GDN tactic gates for TP4/TP8 in `_reference.py` and the adapter, two flags, CPU parity tests | none | General Agent |
| T7 | Run sheet stage 3 (3a sparse QSA A/B, 3b GDN tactics at TP8 with a decode sweep, 3c MoE slices, 3d decode batch baseline) and the stage 2b MoE note | none | Engineer 2 |
| T8 | TP1/2/4/8 equivalence of the sparse path on CPU ranks | T3 | after T3 |

Exit: T1-T8 merged; the QA verification passed (the Flash-Next suite with the sparse prefill and decode parity on the 2600-token prompt, fp32 exact and bf16 loose; `test_qsa_sparse_ops.py` against HF; the TP ranks; `tests/unit_tests/ops/test_flashinfer_gaudi_gdn*.py`; the S0 regression; the mutation checks named in T3 and T4); the milestone merged into the program branch and the PR updated. Hardware proof is run sheet stage 3. Stage 2 (bring-up) runs in dense mode and does not wait for M2.

What M2 does not build: native TPC kernels for QSA (torch ops over the fork's attention kernels are the first version); bucketing of the decode indexer gather by context length (stage-3 follow-up); FP8 or INT8 compressed keys (#54890 on the carry list); the compiled decoder (system map §16, 3a-3c).

### M3 MTP speculative decoding (`feat/flash-next-gaudi-m3`, RUK-2798)

Cut from the program branch at `e4b265f` (M2 merged). Seven facts checked on this Mac on 2026-11-17 (system map §17, facts 30-36) set the scope:

1. **Flash-Next's MTP is one more Flash-Next layer, and upstream's class builds on CPU through the fork's seams.** `vllm/models/qwen4_exp/nvidia/mtp.py` at the pin: `Qwen4ExpMultiTokenPredictor` = `embed_tokens` -> `pre_fc_norm_embedding` (GemmaRMSNorm over H) -> `fc_embedding` (H x H) on the token side, `pre_fc_norm_hidden` (GemmaRMSNorm over hc x H, ungrouped) -> `fc_hidden` (H x H, applied per branch) on the backbone side, summed per branch into `[T, hc x H]`, then one `Qwen4ExpDecoderLayer(layer_type="full_attention")` (QSA attention with its own indexer, MoE with shared expert, two HC `GatedResidual`s; prefix `mtp.layers.{num_hidden_layers}`) and `hyper_connection_mixer.combine_and_mix`, which returns `(sample_hidden [T, H], multi_hidden [T, hc x H])`. `Qwen4ExpMTP` adds `lm_head` and `compute_logits`; `forward(input_ids, positions, hidden_states, ..., spec_step_idx)`. CPU probe (`spikes/flash-next-parity/mtp_probe.py`): with `speculative_config={"method": "mtp", "num_speculative_tokens": 3}` and `load_format="dummy"`, `get_model(replace(vllm_config, model_config=draft), model_config=draft, load_config=draft_load_config)` builds `Qwen4ExpMTP` (64,166,240 parameters on the tiny config) whose decoder layer already carries `HpuQwen4ExpQSAAttention` and `HpuGatedResidual` (the M0 patches of `nvidia.model` reach it), but `mtp.py` binds `GatedResidual` through its own `from .hyperconnection import`, so the final mixer stays the Triton class and the forward fails in `hc_combine_norm` (`'function' object is not subscriptable`). With `nvidia.mtp.GatedResidual = HpuGatedResidual` the forward returns `([3, 128], [3, 512])` and `compute_logits` gives `[3, 248320]`. That rebinding is the whole model-side port.
2. **The target already exports what the drafter needs.** `Qwen4ExpModel.__init__` allocates `_mtp_hidden_buffer [max_num_batched_tokens, hc x H]` when `speculative_config.method == "mtp"`, the forward copies the pre-final-mixer multi stream into it (`nvidia/model.py:546-559`) and `get_mtp_target_hidden_states()` returns it (probe: `(2048, 512)` bf16). On HPU the padded token count of a forward is bounded by `max_num_batched_tokens` for prompts (`extension/bucketing/common.py:452`, `bs * query <= max_num_batched_tokens`) and by `max_num_seqs x (1 + k)` for decode, so the buffer holds; the HPU subclass asserts it. The fork's runner already has a channel for a second hidden tensor: `use_aux_hidden_state_outputs` (eagle3) unpacks `(hidden, aux_list)` from the model (`hpu_model_runner.py:5459-5464`), keeps the aux tensors per decode and prefill batch, and `propose_eagle_decode` / `propose_eagle_prefill` hand `cat(aux, -1)` to the drafter (`:9836-9840`, `:9871-9876`). So the HPU target returns `(sample_hidden, [multi_hidden])` under MTP and the runner sets that flag; no new plumbing.
3. **Weights: 31 `mtp.*` tensors, no head, no embedding.** The real checkpoint's index (`Qwen/Qwen3.8-Flash-Next`, fetched 2026-11-17): `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}.weight`, and `mtp.layers.0.` x {`self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `self_attn.indexer.{index_qk_proj, q_layernorm, k_layernorm}`, `mlp.gate`, `mlp.experts.{gate_up_proj, down_proj}`, `mlp.shared_expert.{gate,up,down}_proj`, `mlp.shared_expert_gate`, `attn_hyper_connection.{block_inject_weight, hc_norm, input_mix_weight_down, input_mix_weight_up}`, `mlp_hyper_connection.{the same four}`} `.weight`. No `mtp.shared_head.*`, no `mtp.embed_tokens`: the draft shares the target's `lm_head` and `embed_tokens` (`llm_base_proposer.py` `_maybe_share_embeddings`: "Detected MTP model. Sharing target model embedding weights"; `_remap_mtp_weight_name` feeds the target's `embed_tokens` / `lm_head` tensors into the draft at load time). HF transformers ignores `mtp.*` (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`) and has no MTP module, so the tiny checkpoint has none of these today and HF alone cannot be the oracle. The oracle is an **HF-module reference** built in the generator: `Qwen4ExpTextRMSNorm`, `nn.Linear`, `Qwen4ExpTextDecoderLayer` (from a one-layer full-attention copy of the text config, no PLE), `Qwen4ExpTextGatedResidual(use_combine=False)`, the target's rotary embedding, causal mask and `lm_head`, driven exactly as `Qwen4ExpTextModel.forward` drives a layer (`modeling_qwen4_exp.py:1383-1478`), with the formula of fact 1. The multi stream the MTP consumes is the input of HF's `hyper_connection_mixer`, captured with a forward pre-hook. Upstream's MTP is vLLM's own day-0 implementation, so a disagreement between the two is adjudicated by the CTO, not papered over.
4. **The GDN verification lattice exists; the runner builds it only for DFlash2.** `qwen3_5.py:568-700`: with a 2-D `load_indices_tensor [bs, T]` and `num_accepted_tokens`, the decode runs `hpu_causal_conv1d_update(..., num_accepted_tokens, max_query_len=T)` (`causal_conv1d_pytorch.py` `_hpu_causal_conv1d_spec_update`: the conv state holds `width - 2` history rows followed by the T candidate rows, and `num_accepted - 1` selects the committed window on the next pass) and `gated_delta_rule_mtp_packed(state_pool, state_indices [bs, T], num_accepted_tokens, query_lengths)` (`flashinfer_gaudi/gdn_decode.py:416`: loads `idx[b, num_accepted[b] - 1]`, writes the state after token t to `idx[b, t]`; the portable torch recurrence on every shape but the DFlash2 native `B <= 16, T = 8` contract, so k = 3 runs the torch path). A 1-D index with `num_accepted_tokens` (`is_dflash_transition`) is the plain decode after a verification. The runner creates this lattice from compact-GDN base slots only when `use_dflash()` (`_gdn_state_slots_per_req` `:2233`, `is_dflash_decode` `:4907-4993`, `_gdn_num_accepted_by_req` `:6397/:6445`). Upstream's generic rule is `MambaSpec.num_speculative_blocks = num_speculative_tokens`, which the fork's GDN and PLE specs already report through `MambaBase.get_kv_cache_spec` (probe: 3 at k = 3), with the scratch blocks placed after the running-state block in the request's block list (`single_type_kv_cache_manager.py:1715-1760`, `mamba_get_block_table_tensor`: "align: the last 1 + num_speculative_blocks of each request"). The state shapes already carry k: GDN conv `(3 + k, 1280)`, PLE conv `(9 + k, hc x H)` (probe: `(6, 1280)`, `(12, 512)`).
5. **PLE and sparse QSA refuse speculation today.** `ple_layer.py:102` raises `NotImplementedError("PLE speculative decode is M3")` when `num_spec_tokens > 0`; `qsa.py` `select` raises `"speculative decode is M3"` when a decode has more rows than state slots; `HpuQsaSparseAttention` decode gathers through `md.block_table`, which the runner already builds with one row per query row for multi-token decode (`hpu_model_runner.py:4715-4730`), so only the indexer changes. The multi-token decode branch of `select` must write the step's raw keys into the ring and the completed blocks' compressed rows **before** scoring (write-then-score): the rows are position-addressed (`ring[slot, pos % R]`, R = 8 at k = 3; `compressed_row_slots(slot_mapping)`), so every row a rejected draft wrote is rewritten by the corrected token before any query can see it, and rollback needs no extra state.
6. **The drafter needs a Flash-Next branch.** `HpuEagleProposer.propose` (`hpu_eagle.py:14-135`) treats `mtp` as a single-tensor model and feeds `target_hidden_states` to every later step; Flash-Next returns `(sample, multi)` and the next step consumes `multi` (upstream `llm_base_proposer.py:598-603` and `model_returns_tuple`, whose list has DeepSeek, GLM and Kimi MTPs but not `Qwen4ExpMTP` at the pin). Its per-step `prepare_attn_metadata` builds decode metadata without `block_table` and without state indices, both of which the draft's sparse QSA needs (the draft's QSA layer has its own compressed-key cache and ring, addressed like the target's). `speculative_config.use_eagle()` is True for `mtp`, async scheduling is forced off with speculation (`platform.py:546-549`) and `supports_v1_speculative_method` has no caller at the pin.
7. **Prefix caching changes the lattice's meaning.** Under align mode the position block of the last computed token can be a completed block whose state is a prefix-cache checkpoint; a verification step that writes token 0's state into it (as the lattice does) would corrupt it, and a token that completes a block inside the step must copy its state into that position block once accepted (vLLM hashes decode-produced full blocks, so a later hit reads them). M3 therefore ships MTP for prefix caching **off** first (the compact-GDN lattice DFlash2 already proved), guards MTP plus prefix caching with a startup error, and adds the align-mode lattice last: load slot `block(last computed token)`, token-0 store slot `block(token 0's position)`, scratch slots `S_0 .. S_{k-1}` from the block table, boundary commit copies after acceptance, pool slots `+k` per request.

**Design (v1, prefix caching off):**

- `vllm_gaudi/models/qwen4_exp/mtp.py`: `HpuQwen4ExpMTP(Qwen4ExpMTP)` with `requires_block_table = True`; `apply_flash_next_patches` rebinds `nvidia.mtp.GatedResidual`; `register()` registers `Qwen4ExpMTP`. The target subclasses override `forward`: under `speculative_config.method == "mtp"` they return `(hidden, [multi])` with `multi = get_mtp_target_hidden_states()[:T].view(*hidden.shape[:-1], hc x H)` and raise when T exceeds the buffer.
- Drafter: `vllm_gaudi/v1/spec_decode/hpu_flash_next_mtp.py` `HpuFlashNextMtpProposer(HpuEagleProposer)`: `propose` with tuple outputs, `multi` as the next step's hidden, `spec_step_idx=j`; per-step metadata gains `block_table` and the per-group ring slots; the pure helpers live in `vllm_gaudi/v1/spec_decode/flash_next_mtp_steps.py` (CPU-tested; `hpu_eagle.py` cannot import on this Mac). The runner picks the class when the draft architecture is `Qwen4ExpMTP` and sets `use_aux_hidden_state_outputs = True`.
- Runner: `_gdn_state_slots_per_req = 1 + k`, the lattice and the `num_accepted_tokens` bookkeeping for every speculative method on a model with mamba-like layers (helpers in `vllm_gaudi/v1/worker/recurrent_spec.py`); `dflash_full_query` stays False outside DFlash2; the QSA side caches and `ring_group_idx` are set for the drafter's QSA layer too.
- PLE: `ple_short_conv_spec_torch` (history + candidate rows, `num_accepted - 1` window, dilation 3) and the spec branch of `_hpu_short_conv`.
- QSA: `select` gets the multi-token decode branch (write-then-score, ragged rows through `query_start_loc_p`, padded rows parked on the ring's scratch row); `_state_indices` takes column 0 of a lattice and repeats it per row; `decode_selected_slots` gets per-row block tables.
- Platform: `check_and_update_config` raises for Flash-Next MTP with prefix caching on until T10b lands (the message names `--no-enable-prefix-caching`) and requires compact GDN.
- No new flags. The recipe is `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`. IndexShare (`index_share_for_mtp_iteration`) is not implemented: every draft step selects.

**Contract of the MTP step** (pasted into every M3 spec that touches it):

- Target forward under MTP returns `(hidden [..., H], [multi [..., hc x H]])`; `multi[i]` is the pre-final-mixer stream of row i.
- Draft step 0: `input_ids[i]` = the token sampled after row i (the fork's `target_token_ids`), `positions[i]` = row i's position, `hidden_states[i] = multi[i]`; the draft predicts the token after `input_ids[i]`; it returns `(sample [T, H], multi' [T, hc x H])`; `logits = compute_logits(sample[last_token_indices])`; the draft token is the argmax. Step j >= 1: `input_ids` = the previous draft tokens, positions + 1, `hidden_states = multi'[last_token_indices]`, `spec_step_idx = j`.
- Verification (target decode): per request `T = 1 + k` rows `[t_0 = the last sampled token, d_1 .. d_k]`, padded to T; `query_start_loc_p` = cumulative real lengths; `load_indices_tensor [groups, bs, T]` is the lattice and `store_indices_tensor` is identical; `num_accepted_tokens [bs]` = 1 + the drafts accepted in the previous step (1 after a prefill); GDN and PLE load from `idx[num_accepted - 1]` and write token t's state to `idx[t]`; conv states live in `idx[:, 0]` with `history + T` rows. A lattice `[groups, bs, T]` yields per-row slots `idx[g][:, 0].repeat_interleave(T)` for the ring and the PLE conv.
- Reference (`reference.json`, key `mtp` per prompt): rows i = 0 .. L-1 over the prompt with `input_ids[i] = prompt[i + 1]` (row L-1 gets HF's first greedy token), positions 0 .. L-1, teacher-forced; `draft_argmax` over the recorded rows (all rows of prompts up to 100 tokens, the last 8 rows otherwise) and `last_row_top5` {ids, logits}.

| # | Subtask | Depends on | Owner |
|---|---|---|---|
| T1 | Tiny checkpoint gains the 31 `mtp.*` tensors and `reference.json` gains the HF-module MTP reference | none | Engineer 1 |
| T2 | `HpuQwen4ExpMTP`, the mixer rebinding, registration, the target's `(hidden, [multi])` output; construction and forward tests with dummy weights | none | Engineer 2 |
| T3 | MTP prefill parity against the reference (dense fp32 exact and bf16 loose on prompts up to 100 tokens; sparse on the 2600-token prompt), the 31-name loading check, a routing negative control | T1, T2 | after T1 and T2 |
| T4 | `HpuFlashNextMtpProposer` plus pure step helpers with CPU tests; runner selection | none | Engineer 3 |
| T5a | Runner: recurrent-spec generalization (lattice, `num_accepted` bookkeeping, side caches and ring group for the drafter); helpers CPU-tested | none | Engineer 5 |
| T5b | Platform guard: Flash-Next speculation is MTP only, prefix caching off, compact GDN, k in 1..4; CPU tests | none | Engineer 4 |
| T6 | PLE short-conv speculative path with rollback tests | none | Engineer 6 |
| T7 | QSA sparse multi-token decode (write-then-score), per-row block table, lattice-aware state indices, rollback tests | none | General Agent |
| T8 | End-to-end MTP CPU test: draft, verify, accept, state parity against non-speculative decode, with a forced rejection | T2, T3, T4, T6, T7 | after them |
| T9 | Run sheet stage 3e (MTP A/B: acceptance length, decode sweep at 1/8/16/32 agents, k in {1, 2, 3}) and the design docs | none | Engineer 4 |
| T10a | Align mode, part 1: pool slots `+k`, the align lattice builder, `store_indices` in the torch MTP recurrence and the speculative conv update, CPU tests | T5a, T8 | after them |
| T10b | Align mode, part 2: runner call sites, boundary commit copies after acceptance, the guard lifted, `KVCacheManager` tests with `num_speculative_blocks` | T10a | after T10a |

Exit: T1-T10b merged; the QA verification passed (the Flash-Next suite including `test_mtp_*`, `tests/unit_tests/core`, `tests/unit_tests/worker`, the S0 regression, and the mutation checks named in T3, T6, T7 and T8); the milestone merged into the program branch and the PR updated. Hardware proof is run sheet stage 3e.

What M3 does not build: IndexShare, a native kernel for `T = 4` (the torch recurrence runs inside the compiled graph), DFlash2 for Flash-Next (no draft checkpoint), #55513 (block-FP8 MTP scales: M4 with the FP8 checkpoint), non-greedy acceptance tuning, and draft-side bucketing or warmup changes beyond what the fork's EAGLE path already does (checked on hardware in stage 3e).

### M4 Two servers and eight cards (`feat/flash-next-gaudi-m4`)

FP8 at TP4 needs expert parallelism (R6: 640/4 = 160 per rank is not a multiple of the 128-wide FP8 blocks; `--enable-expert-parallel` gives each rank 128 whole experts), so: TP4 + EP4 block-FP8 MoE, INC FP8 with the conv1d blocklist as the alternative; PLE table on the host (backport #54371's structure with an HPU gather, or the fork's Engram host gather) to fit BF16 at TP4; TP8 fused AllReduce + residual + RMSNorm (`tp2_fused_ar_norm.py` pattern); run-sheet stage 5 comparisons (TP8, 2×TP4, TP4+EP).

### M5 Gaudi2 extras

Only what stages 3–6 show: FP8 KV cache (block 896 or 1664), warmup time, sampler, communication overlap, the compiled decoder (system map section 16, items 2c and 3a–3c).

### Track: fused GDN decode kernel (RUK-2812, `feat/flash-next-gaudi-gdn-fused`, merged through #3)

A separate track, planned on RUK-2733 and RUK-2812, wrote native Gaudi2 TPC kernels for the GDN decode step and
merged them into the program branch on 2026-11-18. Contract and layouts: [gdn-core-decode.md](gdn-core-decode.md);
simulator setup: [tpcsim.md](tpcsim.md).

- `torch.ops.flashinfer_gaudi.gdn_core_decode` is one TPC kernel per GDN layer per decode step (T = 1), between the
  in-projection and `out_proj`: the conv update and SiLU, the gates, the Q/K L2 norm, the delta rule, the gated
  RMSNorm and both state writes, with separate load and store slots. It builds as its own library,
  `libgdn_fused_gaudi2.so` (`VLLM_GAUDI_BUILD_GDN_FUSED=1`); if that build fails, the rest of the install is kept.
- **It is the decode path.** `VLLM_HPU_GDN_CORE_DECODE` defaults to `1` (FUSED) for every model that uses this GDN
  layer; `0` selects the torch path, and `2` (RECURRENCE, the recurrence only) is a bisect tool. Without the
  library, the torch path runs and the process logs one warning at startup; a layer the op does not fit runs the
  torch path with one warning for that layer. Speculative decode steps (M3) keep the torch path.
- Checked before the rental: CPU parity at TP1/2/4/8 head counts (load != store, padded batches, slot reuse), and
  both kernels against the torch reference on the Gaudi2 TPC simulator.
- On the rental: stage 1 builds the library; stage 1b gates it before any model runs (simulator, op numerics on the
  card, a 1,000-step graph replay; a failure sets `VLLM_HPU_GDN_CORE_DECODE=0` for the session); stage 3 runs on
  mode 0, so its A/Bs keep their meaning; row 3f measures the kernel against its targets.
- Phase 2, after M3 merges: the T = 1 + k verification kernel. Its contract and torch reference are in place
  (`gdn_core_verify_reference`, RUK-2915): each draft node loads its `parent_idx` node's state, so a draft tree
  needs no kernel change.
- Merging M3 into the program branch: the run sheet conflicts once, because M3's stage 3e and this track's row 3f
  were both added after 3d. Keep both, 3e first. A new run-sheet section takes the next free letter (3g).

## Qualification run sheet (stages, fail-fast; T8 writes stages 1–2)

1. Build the image at the program branch and compile native code; record `flashinfer_gaudi.get_capabilities()`.
   1b (fused GDN track): the simulator, op numerics on the card and a 1,000-step graph replay, as pass/fail gates before any model runs.
2. TP8 BF16 bring-up on the **tiny checkpoint** first (greedy agreement with `reference.json`, TP1 then TP8), then the real checkpoint: startup lines (`Usable num_blocks`, `Maximum concurrency`, `kv cache group sizes`, `GDN layers detected`), compared with S4's prediction; 8 short prompts for coherence; the GSM8K sample.
3. Flag-by-flag A/B of each native path against its torch path; 3e (M3): MTP on/off, acceptance length and the decode sweep at k = 1, 2, 3; 3f (fused GDN track): the kernel against its targets, with mode 0 once as the baseline.
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

- **The HPU sparse attention wrapper cannot run on this Mac** (M2-T5c): the prefill bias layout must match `_include_past`'s block order and the decode gather must match the kernel-block units; both are checked by reading and by CPU tests on fake caches, then by run sheet stage 3. The fallback is dense mode (`VLLM_HPU_FLASH_NEXT_QSA_MODE=0`), exact only within 2050 tokens.
- **bf16 selection ties**: the indexer's top-512 cut in bf16 can differ from HF's at near-ties, so bf16 parity on the 2600-token prompt is loose (top-1 and a logit tolerance); fp32 parity is exact.
- **The ring-by-GDN-slot invariant** relies on the runner's 128-aligned chunking assert; if a later change relaxes it, the ring moves to its own `MambaSpec` state group (the M1 pool already handles any `MambaSpec`).
- **Decode indexer gather at 262K**: about 3 ms per step at batch 32 with the fixed-shape gather; bucketing by context length is the stage-3 follow-up if the profile shows it.
- **Short-conv state on the HPU runner** (T4b): new state kind for the fork's hybrid allocation; the fallback is to hold the PLE conv history in a per-request host-side buffer managed by the model (correct, slower), behind the same interface.
- **QSA indexer caches** use `CircularBufferSpec` / `MLAAttentionSpec` layouts the HPU worker does not know; M0 avoids creating them (dense mode), M2 adds them.
- **Numerics in bf16 on HPU** versus fp32 CPU references: stage 2 compares greedy tokens, not logits; the tiny fixture's prompts are short so one divergence is visible.
- **Tokenizer download** needs network; the generator takes `--tokenizer-from <dir>`.
- **The real checkpoint cannot be loaded on this Mac** (335 GiB, 64 GiB RAM): the real-model checks exist only on the hardware.
