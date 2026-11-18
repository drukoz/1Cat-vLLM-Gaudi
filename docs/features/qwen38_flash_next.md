# Serving Qwen3.8-Flash-Next on Gaudi

Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) is a hybrid model: 36
Gated DeltaNet (GDN) layers, 12 Qwen Sparse Attention (QSA) layers, a
512-expert MoE with one shared expert, a 51B-parameter n-gram embedding table
(PLE), and one MTP (multi-token prediction) layer used for speculative
decoding. This plugin serves it on Intel Gaudi2. This page is the single
reference for how to serve it, which flags matter, and what is supported.

## Status

- Supported on Gaudi2 through the Flash-Next program (RUK-2709); milestones
  M0 to M4 are merged on the feature branch.
- Everything on this page is checked by CPU parity tests against Hugging Face
  transformers and on the Gaudi TPC simulator.
- **Hardware qualification is pending: no number from a Gaudi card exists
  yet.** The first hardware run is the rental described in
  `docs/design/flash-next/run-sheets/rental.md`. Read every statement here as
  "tested on CPU and in simulation", not "measured on the card".

## Checkpoints and layouts

| Checkpoint | Precision | Size | Layout |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-Flash-Next` | BF16 | 335 GiB | TP8 only, on one 8-card Gaudi2 node (about 42 GiB of weights per card) |
| `Qwen/Qwen3.8-Flash-Next-FP8` | block-FP8 routed experts, everything else BF16 | about 185 GB | the production checkpoint; TP8 or TP4, `--enable-expert-parallel` required |

The FP8 checkpoint **requires `--enable-expert-parallel`** at TP8 and at
TP4: with expert parallelism each rank keeps whole experts (intermediate size
640 = five 128-wide FP8 blocks); without it the 640 columns split into 80 or
160 per rank, which breaks vLLM's 128-block rule.
`tools/flash_next/check_fp8_layout.py` checks a checkpoint against a layout
(exit 0 passes).

The KV cache stays BF16. An FP8 KV cache needs INC calibration on the card
and is not qualified.

`--language-model-only` is the production setting; the vision tower is not
supported on HPU in this program.

## Serving recipe

Variant A of `tools/flash_next/rental/variants.json` is the production
recipe: one TP8 + EP8 server, 128K context, 32 agents. The exact command
and environment:

```bash
export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7
export VLLM_HPU_FLASH_NEXT_QSA_MODE=1
export VLLM_HPU_RECURRENT_STATE_POOL=1
export PT_HPU_LAZY_MODE=1
unset VLLM_COMPACT_GDN

vllm serve Qwen/Qwen3.8-Flash-Next-FP8 \
  --served-model-name flash-next \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --language-model-only \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 2048 \
  --enable-prompt-tokens-details
```

Variant C in the same file is the two-server TP4 + EP4 layout: two servers
of four cards each, 16 agents per server.

## Flags

Defaults are as defined in `vllm_gaudi/envs.py`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `VLLM_HPU_FLASH_NEXT_QSA_MODE` | `0` | QSA attention mode. 0 = dense attention, exact only for prompts up to 2050 tokens. 1 = sparse Qwen Sparse Attention (the indexer's top-512 blocks of 4 tokens plus the current tail), required for correct output on any prompt beyond 2050 tokens. Production runs 1. |
| `VLLM_HPU_FLASH_NEXT_QSA_PREFILL_TILE` | `2048` | Query tile of the sparse prefill attention. |
| `VLLM_HPU_RECURRENT_STATE_POOL` | `0` | 1 gives the GDN and PLE short-conv state groups their own pool of slots, so attention blocks no longer each carry an unused recurrent state slot; under prefix caching this roughly doubles the KV capacity. |
| `VLLM_HPU_RECURRENT_STATE_POOL_SLOTS` | `0` | Size of the state pool; 0 is resolved at startup from `max_num_seqs`. |
| `VLLM_HPU_RECURRENT_STATE_POOL_CACHE_SLOTS` | `512` | Cache slots of the state pool. |
| `VLLM_COMPACT_GDN` | — | Must be 0 when prefix caching is on; the server refuses to start otherwise. |
| `VLLM_HPU_GDN_CORE_DECODE` | `1` | 1 = the fused Gaudi2 GDN decode kernel, the default decode path; 0 = the PyTorch path; 2 = recurrence only, a bisect tool. |
| `VLLM_HPU_GDN_CORE_DECODE_DEBUG` | `0` | 1 = index check before each fused decode step. |
| `VLLM_HPU_FLASHINFER_GDN_TP4` | `false` | Opt-in GDN tactic geometry for 4 cards; unmeasured. |
| `VLLM_HPU_FLASHINFER_GDN_TP8` | `false` | Opt-in GDN tactic geometry for 8 cards; unmeasured. |
| `VLLM_FLASH_NEXT_DRAFT_RANK_LOG` | `0` | With MTP, logs every 200 readings where the target's token ranked in the drafter's top 4 at the first rejected position (read in `vllm_gaudi/v1/spec_decode/flash_next_mtp_steps.py`). |

The fused GDN decode kernel is built at install time with
`VLLM_GAUDI_BUILD_GDN_FUSED=1` (needs `tpc-clang` from `habanalabs-tools`);
a failed kernel build keeps the rest of the install. Without the
`gdn_fused_ops` library the server logs one warning and runs the PyTorch
path. Details: `docs/design/flash-next/gdn-core-decode.md`.

## Speculative decoding

MTP (the model's own draft layer):

```bash
vllm serve Qwen/Qwen3.8-Flash-Next-FP8 ... \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

MTP runs 1 to 4 speculative tokens (the recipe is 3). The draft shares the
target's embeddings and LM head and loads the checkpoint's `mtp.*` tensors.

n-gram (prompt lookup):

```bash
vllm serve Qwen/Qwen3.8-Flash-Next-FP8 ... \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 3}' \
  --no-enable-prefix-caching
```

n-gram requires `--no-enable-prefix-caching`; the server refuses to start
otherwise ("Flash-Next n-gram speculation runs with prefix caching disabled
for now").

The server enforces the combination rules at startup (in
`vllm_gaudi/platform.py`):

- Flash-Next supports speculative decoding with its own MTP layer or n-gram
  prompt lookup only.
- `VLLM_COMPACT_GDN=1` does not support prefix caching: unset it (prefix
  caching uses the recurrent state pool) or disable prefix caching with
  `--no-enable-prefix-caching`.
- Without prefix caching, Flash-Next MTP needs compact GDN state for the
  per-token verification checkpoints: unset `VLLM_COMPACT_GDN` or set it to
  1.
- Flash-Next MTP on HPU runs 1 to 4 speculative tokens (the recipe is 3).

DFlash2 is not available for Flash-Next (no draft checkpoint exists).

## Prefix caching for clients

A prefix hit needs whole blocks identical from token 0. The block is 512
tokens at TP8 (the GDN state page sets it), so a shared system prefix padded
to a multiple of 512 tokens reuses best.

## Not supported

The vision tower, LoRA, EPLB, pipeline parallel, DFlash2, and FP8 KV cache.

## Developer notes

- `docs/design/flash-next/README.md` — the program documents (system map,
  reuse ledger, plan, run sheets).
- The "Developing on this Mac without an HPU" section of `AGENTS.md` — CPU
  test environment, the tiny fixture under
  `tests/unit_tests/models/flash_next/`.
- `tools/flash_next/` — fixture generator, rental harness, step anatomy,
  agent prompt set.
