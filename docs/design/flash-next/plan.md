# RUK-2709 Flash-Next on Gaudi: plan

Kind: **Program**. Version 0.1, 2026-11-17, written from the system map (v0.5) and the reuse ledger (v0.3) after all Phase 0 research (R1–R7, R4b) and spikes (S0, S3, S4) reported. The working copy is the `plan` document on RUK-2709; this file is refreshed with the pull request.

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

### M1 Admission, prefill, prefix reuse and the GDN state pool (`feat/flash-next-gaudi-m1`)

R7's config variants first (run-sheet stage 4), then code: prefill chunk alignment to the GDN block under prefix caching (re-implement upstream vllm-gaudi #1831, #1853, #1828, #1890 by hand: the fork rewrote the runner and the async scheduler); `VLLM_COMPACT_GDN=0` qualification with a loud startup error when compact mode meets prefix caching; the **GDN state pool** (system map 2a: reserved block-id range or second pool in the scheduler's `BlockPool` through `vllm_gaudi/v1/engine/core_patch.py`, worker shrink factor 1, state tensors sized to the `align` need; S4 predicts the new `num_blocks`); prompt bucket shapes for 128K–262K; the GDN PyTorch prefill path for the TP8 shape (2 key / 6 value heads per rank) checked on 8 CPU ranks; #1895 (non-finite initial-state masking). Measured by first-token latency and hit rate in run-sheet stage 4.

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
