# RUK-2709 Flash-Next on Gaudi: system map

Kind: **Program**. Version 0.1, 2026-11-17, written by the CTO from a first reading of the repository, the model's public configuration, and upstream vLLM / vllm-gaudi. Rows marked *(unverified)* wait for a research subtask or for the user's answers to the questions in section 13. Facts carry a file path or a URL.

## Assumptions

1. The base of all work is this fork (`drukoz/1Cat-vLLM-Gaudi`, identical to `1CatAI/1Cat-vLLM-Gaudi` `main` at `e2266e0`), not upstream `vllm-project/vllm-gaudi`. The request asks to audit 1Cat's kernels and reuse its DFlash2 work; upstream fixes are cherry-picked, not rebased onto.
2. The model is **Qwen3.8-Flash-Next** (`Qwen/Qwen3.8-Flash-Next`, architecture `Qwen4ExpForConditionalGeneration`), served from an FP8 checkpoint with BF16 KV, two servers of four Gaudi2 cards each. "Swift-1.5 Flash-Next" is the company's name for it.
3. No Gaudi hardware is reachable from this Mac, and no `torch` or `vllm` is installed here. Everything is verified first with CPU unit tests (the fork's own pattern: hardware tests skip without an HPU), then on the user's machines with exact commands we hand over.
4. The repository has no Dev branch; `main` is the base branch and the program's pull request targets it as a draft until the program is done.
5. Existing routes (Qwen3.5 / Qwen3.8-27B GDN, DeepSeek V4 / V4.1, MiniMax M3 / H3) must not regress, but get no new work.

## 1. Purpose and users

The repository is the Intel Gaudi hardware plugin for vLLM (`vllm_gaudi`, Apache-2.0), in 1CatAI's Gaudi2-focused fork, which adds Qwen GDN / DFlash2 paths, DeepSeek V4 / V4.1 native execution, FlashInfer-Gaudi operators and MiniMax support on top of upstream `vllm-project/vllm-gaudi` (`AGENTS.md`, `README.md`, `docs/1cat_gaudi_guide.md`).

For Rukoz Ent it serves one thing: the company's agent fleet. Paperclip's engineers, tech leads and reviewers call an OpenAI-compatible endpoint with long agentic contexts (about 128K tokens each: the servers held 2.5M KV tokens for about 19 agents), many concurrent sequences, long shared prefixes (system prompts, repository guides), and tool calls. The baseline in section 11 is that workload.

## 2. The model: Qwen3.8-Flash-Next

Sources: the Hugging Face config and model card (https://huggingface.co/Qwen/Qwen3.8-Flash-Next), the LMSYS architecture note (https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/), the vLLM serving recipe (https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) and upstream vLLM `vllm/models/qwen4_exp/` at `v0.34.0`.

| Fact | Value |
|---|---|
| Architecture / model type | `Qwen4ExpForConditionalGeneration`, `qwen4_exp` (text: `qwen4_exp_text`) |
| Size | 125B total, 6B active per token, plus a 51B n-gram embedding table and a 4B MTP layer |
| Layers | 48, hidden 2560; `full_attention_interval` 4: 36 Gated DeltaNet layers and 12 Qwen Sparse Attention (QSA) layers, each followed by MoE |
| Gated DeltaNet | 16 key heads, 48 value heads, head dim 128 / 128, conv kernel 4, `mamba_ssm_dtype` float32. Same head geometry as the fork's Qwen3.8-27B tactic (TP1 16 / 48), but hidden 2560, so `in_proj` is 2560 to 10240 |
| QSA attention | 24 query heads, 2 KV heads, head dim 256, RoPE dim 64 (`partial_rotary_factor` 0.25), interleaved mRoPE sections [11, 11, 10], `rope_theta` 1e7 |
| QSA indexer | 4 query heads, 1 shared key head, dim 128, `indexer_compress_ratio` 4 (one compressed key per 4 tokens), `indexer_budget` 2048 tokens (512 blocks), score `relu(q·k)/sqrt(128)` summed over heads; attention then runs over the original K/V of the selected blocks plus up to 3 tokens of the current block |
| N-gram embedding (PLE / Engram) | `ngram_size` 3, 20M rows base, 8 heads per n-gram (8 bigram + 8 trigram hashes = 16 rows of 160 values per token), `split_ngram_parts` 128, at `ple_layer_ids` [2]; about 95.4 GiB in BF16; gated into the 4 HyperConnection branches |
| HyperConnection ("gated residual") | `hc_count` 4 branches, `hc_lowrank` 320, Gemma-style grouped RMSNorm |
| MoE | 512 experts, top-10 plus 1 shared expert, intermediate 640 |
| MTP | 1 hybrid layer (full attention), `mtp_use_dedicated_embeddings` false; vLLM recipe uses `{"method":"mtp","num_speculative_tokens":3}` |
| Context | 262,144 native; 1M with YaRN overrides |
| Vocab | 248,320 |
| Checkpoints | BF16 335 GiB; `Qwen/Qwen3.8-Flash-Next-FP8` 172.8 GiB (block FP8, 128-wide blocks: plain TP8 is incompatible, TP4 or TP+EP recommended) |

KV cache per token (BF16, 12 attention layers, 2 KV heads, dim 256): 12 × 2 × 2 × 256 × 2 B = 24.6 KB, plus about 0.8 KB of indexer key cache. GDN state per sequence: 36 × (48 × 128 × 128 × 4 B + conv) ≈ 115 MB.

## 3. Where it runs

- Two servers × four Gaudi2 (HL-225, 96 GB HBM2e, about 2.45 TB/s each; `evidence/ROOFLINE.md`), TP4 per server. *(unverified)* whether both servers are one 8-card HLS-2 node or two nodes.
- Gaudi software: the fork's baseline is Gaudi Software 1.24.1 with Gaudi PyTorch 2.11 (`docs/1cat_gaudi_guide.md`); upstream CI moved to 1.24.2 / PyTorch 2.13 on 2026-10-12. The user's version is *(unverified)*.
- vLLM engine: the fork's last upstream sync targets vLLM commit `17da48596c98946d3e3e6896e2ebd341e809f3bd` (fork commit `71ee4e4`, 2026-08-28). Qwen3.8-Flash-Next support merged into vLLM on 2026-08-31 (`e126687a`, PR #53896), after that sync. The engine the user ran is *(unverified)*.
- This Mac: macOS 26, Python 3.14 (Homebrew), no `torch`, no `vllm`, no HPU. The plugin requires Python 3.9–3.12 (`pyproject.toml`).

## 4. Build, test and release

- Install: a paired vLLM checkout installed with `VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e ../vllm`, then `pip install --no-build-isolation -e .` (`docs/1cat_gaudi_guide.md`, installation section). Native kernels: `tools/build_flashinfer_gaudi.py` (CMake, TPC compiler) and `tools/build_deepseek_v4*.py` (needs matching Gaudi PyTorch bridge sources).
- Tests: `tests/unit_tests/` (pytest; many run on CPU with mocks, hardware tests skip unless an HPU is present or `FLASHINFER_GAUDI_RUN_HARDWARE_TESTS=1`), `tests/full_tests/` (model cards, HPU only), `tests/upstream_tests/`. The fork's PRs report e.g. "494 unit tests passed, 58 skipped" on CPU (`docs/benchmarks/deepseek-records.md`).
- Lint: `yapf` (120 columns) + `ruff`; `pre-commit`; DCO `Signed-off-by` on commits (`AGENTS.md`).
- CI: upstream GitHub workflows need self-hosted HPU runners (`.github/workflows/pre-merge.yaml`) and will not run on this fork. Our gate is CPU unit tests plus the user's hardware runs.
- Release and deployment: none in the repository. The user installs the plugin on the Gaudi servers by hand *(unverified)*. Nothing we merge deploys by itself.
- Dangerous: `tools/sync_deepseek_v41_checkpoint.py` writes into model directories; `tools/run_deepseek_v41.py` takes device locks. Not on our path.

## 5. External systems

| System | How | Notes |
|---|---|---|
| vLLM engine | Python dependency at a pinned commit; plugin entry points `vllm_gaudi/__init__.py` (`register_ops`, `register_models`, `register_tool_parsers`) | Upstream API moves weekly; upstream vllm-gaudi carries `[FIX_FOR_VLLM_CUSTOM=<sha>]` commits for it |
| Hugging Face checkpoints | local paths on the servers | FP8 checkpoint identity *(unverified)* |
| Gaudi software stack | `habana_frameworks.torch`, Synapse, HCCL, TPC compiler | Versions *(unverified)* |
| Intel Neural Compressor | FP8 calibration / quantization (`calibration/`, `docs/features/floating_point_8.md`) | Only if the FP8 weights were produced with INC *(unverified)* |
| NIXL / KV transfer | `vllm_gaudi/distributed/kv_transfer/` | Not used by the agents' servers *(unverified)* |

No databases, no outbound services.

## 6. Surfaces

- `vllm serve` OpenAI-compatible API (chat and completions, tool-call parsers, reasoning parsers) — the agents' surface.
- Dedicated entry points `python -m vllm_gaudi.entrypoints.deepseek_v4` / `deepseek_v41` and the multi-model API server (`vllm_gaudi/entrypoints/`) — not on our path.
- Operator benchmarks and audits: `tools/benchmark_flashinfer_gaudi_*.py`, `tools/benchmark_gdn_*.py`, `tools/profiler/`, `tools/trace_anatomy/`, `tools/analyze_*.py` — the profiling toolkit for the kernel audit.
- `flashinfer_gaudi.get_capabilities()` — reports which native operators are promoted (`flashinfer_gaudi/_capabilities.py`, `flashinfer_gaudi/tactics/*.json`).

## 7. Workflows this program touches

1. **Startup.** `HpuPlatform.check_and_update_config` (`vllm_gaudi/platform.py`) forces block size 128, realigns `mamba_page_size_padded` for hybrid models, turns `mamba_cache_mode` "all" into "align" when prefix caching is on, disables prefix caching under contiguous PA, and validates DFlash2 (TP1 only, prefix caching off, exactly 7 draft tokens, Qwen3.5/3.8 hybrids only). Then model load with OOT overrides (`vllm_gaudi/models/__init__.py`), KV-cache sizing, warmup over buckets (`vllm_gaudi/extension/bucketing/`, `docs/features/warmup.md`), graph compilation.
2. **Long-prompt prefill.** Scheduler chunks by `max_num_batched_tokens`; `HPUModelRunner` (`vllm_gaudi/v1/worker/hpu_model_runner.py`, 10,240 lines) pads to a prompt bucket; GDN layers run `hpu_chunk_gated_delta_rule` (`vllm_gaudi/ops/hpu_gdn_pytorch.py`) or the FlashInfer-Gaudi prefill tactic when the shape is TP1 16/48 or TP2 8/24 (`vllm_gaudi/ops/flashinfer_gaudi_adapter.py:58-94`); attention layers run FusedSDPA / paged prefill (`vllm_gaudi/attention/backends/hpu_attn.py`); MoE runs `vllm_gaudi/ops/hpu_fused_moe.py`. Prefix-cache hits reuse attention blocks; GDN state is checkpointed at mamba block boundaries (`compute_prefix_caching_block_indices`, runner line 1314).
3. **Decode step.** Decode bucket by batch and blocks; fused GDN decode (`gdn_fused_decode_step`, promoted for TP1 16/48 batches 1–32, TP2 8/24 behind `VLLM_HPU_FLASHINFER_GDN_TP2`); paged attention; MoE; sampler (device sampling since fork PR #58); async scheduling.
4. **Speculative decoding.** EAGLE-style proposer (`vllm_gaudi/v1/spec_decode/hpu_eagle.py`) and DFlash2 (`hpu_dflash2.py`, `flashinfer_gaudi/dflash2.py`, Qwen3.8-27B draft, TP1 only); DeepSeek DSpark is separate.
5. **Multi-card.** HCCL collectives; TP2 fused AllReduce-norm boundaries (`vllm_gaudi/distributed/tp2_fused_ar_norm.py`); TP4 native replay only for DeepSeek V4.1.

## 8. Data and configuration

- Settings: 534 `VLLM_*` names in `vllm_gaudi/envs.py` (GDN: `VLLM_GDN_*`, `VLLM_HPU_GDN_*`, `VLLM_HPU_FLASHINFER_*`, `VLLM_COMPACT_GDN`), `PT_HPU_*` runtime variables, feature flags in `vllm_gaudi/extension/features.py`, bucketing in `vllm_gaudi/extension/bucketing/`, INC JSON for FP8, tactic manifests in `flashinfer_gaudi/tactics/`.
- Runtime state: paged KV cache (block 128), GDN recurrent state (FP32 by default) and conv state per sequence, index caches (QSA, to add), the PLE table (95 GiB BF16, to place), HPU graphs / recipes.

## 9. Where the weight is

| Area | Size | Nature |
|---|---|---|
| `vllm_gaudi/v1/worker/hpu_model_runner.py` | 10,240 lines | The engine of the plugin; heavily modified by the fork (GDN state views, async DMA, TP2 plans) |
| GDN | `ops/hpu_gdn_pytorch.py` 2,684; `flashinfer_gaudi/gdn_*.py` 964; `csrc/flashqla_hpu/` 55 files; `csrc/flashinfer_gaudi/` 47 files | Real kernels and dispatch; audit target |
| DeepSeek V4 / V4.1 | about 30K Python lines across `ops/deepseek_v41_*`, `models/deepseek_v41*`, `v1/worker/deepseek_v41_*`; `csrc/deepseek_v4*` 976 files | Not on our path; reuse candidates: Engram host gather, indexer kernels, DSpark MTP, TP4 replay |
| MoE / FP8 | `ops/hpu_fused_moe.py` 983; `ops/hpu_fp8.py` 380; `ops/hpu_compressed_tensors.py` 1,321 | Generic; needs 512-expert / shared-expert / block-FP8 checks |
| Attention | `attention/backends/hpu_attn.py` 969; `v1/attention/backends/hpu_attn.py` 183; `extension/ops.py` 2,602 | Dense paged attention; no sparse path |
| Bucketing / warmup | `extension/bucketing/` 1,516 | Long-context bucket shapes live here |
| Speculative | `v1/spec_decode/` 666; `ops/hpu_dflash2.py` 208; `flashinfer_gaudi/dflash2.py` 547 | DFlash2 is target-specific (Qwen3.8-27B draft) |
| `tools/` | 96K lines | Mostly DeepSeek V4.1 audits and benchmarks; the profiling method to copy |
| `tests/unit_tests/` | 527 files | CPU-runnable with skips; our first gate |

Dead or irrelevant for us: MiniMax H3 (`vllm_gaudi/omni/`), DeepSeek OCR, Gemma, LoRA.

## 10. What must not change

- The vLLM plugin contract (`register_ops`, `register_models`, `register_tool_parsers`) and the OpenAI API.
- Existing qualified routes and their flags (`docs/features/1cat_gaudi_execution.md`, `docs/features/flashinfer_gaudi.md`): new behaviour sits behind architecture checks or new flags, default off until qualified.
- Block size 128 and the FP8 numerics contracts in `docs/features/floating_point_8.md`.
- `AGENTS.md` conventions: yapf/ruff, SPDX headers, no raw performance numbers in anything we may upstream (our own program documents keep the user's measurements).

## 11. Baseline: Run 2 (user's measurement, 2026-11-17)

FP8 weights, BF16 KV, two servers × four Gaudi2 cards, 2.5M KV tokens per server (about 19 agents each).

| Test | First token median / p90 / p99 | Decode per agent | Total output | Wall |
|---|---|---|---|---|
| 1 agent | 24.9 s | 28 tok/s | 17 tok/s | 123 s |
| 16 agents, cold | 136 s / 237 s / 263 s | 7.1 tok/s | 51 tok/s | 321 s |
| 16 agents, prime (also cold) | 161 s / 251 s | 6.3 tok/s | 50 tok/s | 329 s |

Reading, pending the exact prompt sizes and launch flags:

1. **Prefill dominates.** One agent waits 24.9 s for its first token; at 16 agents the p99 first token is 263 s of a 321 s run. If the prompts are around 100K tokens, four cards deliver roughly 4K tokens/s of prefill, about what the fork's own record shows for one card on 2K-token prompts (`docs/benchmarks/qwen38-a800-gaudi2.md`). Long-prompt prefill at TP4 is the first bottleneck: chunk size, GDN prefill tactic (none exists for the TP4 shape 4 / 12 heads), attention prefill at 128K, and recompiles.
2. **Decode does not batch.** 28 tok/s alone, 7.1 tok/s each for 16 agents: 16× the work for 4× the aggregate. A step at batch 16 takes about 140 ms; the FP8 weight and KV traffic for that step is far below what Gaudi2's HBM moves in 140 ms, so the step is overhead-bound (host work, graph launches, per-sequence loops, prefill chunks stealing steps), not bandwidth-bound.
3. **Prefix reuse did nothing.** Priming the cache made the cold run slower, not faster. Either prefix caching was off (DFlash2 and the GDN async-DMA path both require it off) or it never hit for a hybrid model. The startup log will say which.

## 12. Upstream state that matters

- **vLLM.** Qwen3.8-Flash-Next merged 2026-08-31 (PR #53896, 124 files); PLE host offload and Engram TP merged 2026-09-09 (PR #54371); block-FP8 MTP fix 2026-09-08 (PR #55513); minimum 0.30.0; latest release v0.34.0 (2026-11-16). The model is the package `vllm/models/qwen4_exp/` with `common/` (HyperConnection, n-gram embedding base classes, PLE loading, QSA metadata and index-cache ownership), `nvidia/` and `amd/`. `vllm/models/qwen4_exp/__init__.py` picks `amd` on ROCm, raises on XPU/TPU, and **falls back to `nvidia` for every other platform, including HPU**. The registry maps `Qwen4ExpForConditionalGeneration` and `Qwen4ExpMTP` there (`vllm/model_executor/models/registry.py`). An HPU implementation therefore has to be registered out-of-tree by the plugin, as the fork does for other models.
- **vllm-gaudi upstream** (`vllm-project/vllm-gaudi`, about 216 commits since the fork's base of 2026-08-28). Relevant: Qwen3.5 GDN prefix caching with `COMPACT_GDN=0` (#1831), prefill chunk alignment to the mamba block (#1853), mamba chunk alignment fixes (#1828, #1890), non-finite initial-state masking in GDN prefill (#1895), per-request top-k in the DeepSeek sparse-attention decode indexer (#1870, `vllm_gaudi/ops/hpu_sparse_attn_indexer.py`), FP8 KV decode softmax on Gaudi2 (#1931), RoPE numerics for very long contexts (#1816), explicit MoE overloads for FP8 per-channel (#1921), MoE top-k routing options (#1922), vendored V1 model runner after upstream removal (#1970). No Qwen3.8-Flash-Next support upstream.
- **1CatAI upstream.** No commits after 2026-10-12; no Flash-Next code. The user's fork is identical to it.

## 13. Open questions for the user (asked as an interaction on RUK-2709)

1. Exact launch command and environment of Run 2 (one server).
2. Checkpoint identity and how the FP8 weights were produced.
3. Gaudi software, vLLM, plugin, PyTorch and Python versions.
4. Benchmark client: prompt and output lengths, shared prefix, how "prime" was done, how the rates were computed.
5. One 8-card node or two 4-card servers.
6. Whether an engineer or I can get access to a Gaudi2 machine for profiling, or whether the user runs the commands we write.
7. Startup and run log excerpts (KV blocks, prefix caching, mamba cache mode, warmup buckets, recompilation warnings).

## 14. Program shape (to be confirmed by the reuse ledger and plan)

Milestones as vertical slices, in the order the baseline dictates:

- **M0 Foundations.** Mac CPU development environment and test gate (spike S0); exact baseline reproduction script; Flash-Next OOT model registration on HPU built from upstream `common/` pieces with the fork's GDN path, dense attention as the first QSA fallback, PLE table on device, MoE through the existing HPU path.
- **M1 Prefill and prefix reuse.** Long-prompt prefill at TP4 (chunking, GDN TP4 prefill tactic, attention prefill at 128K, bucket shapes, recompiles) and working prefix caching for the hybrid model (cherry-picked upstream fixes), measured by first-token latency on the agent workload.
- **M2 Decode batching.** Fused GDN decode for the TP4 shape, MoE FP8 decode for 512 experts, graph replay and host overhead at batch 16–32, QSA sparse attention proper (indexer + block gather) so long contexts stop paying dense attention.
- **M3 Speculative decoding.** Flash-Next MTP on HPU (upstream `Qwen4ExpMTP` through the fork's EAGLE proposer), IndexShare top-k reuse; DFlash2 only if a Flash-Next draft exists.
- **M4 Eight cards.** TP4 ×2 versus TP8 versus TP + EP, HCCL topology, PLE host offload to free HBM for KV.
- **M5 Gaudi2 extras.** Whatever the profiles show (FP8 KV, native fusions already in FlashInfer-Gaudi, warmup time, sampler).
