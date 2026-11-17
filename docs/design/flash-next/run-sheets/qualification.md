# Flash-Next qualification run sheet

Stages are ordered fail-fast (system map §14): a stop rule ends the rental session for the day; the captures come back and the fix is made offline. Startup costs about 400 s per launch (system map §11, Reading 5), so share a launch between checks wherever possible. Sections follow [the template](README.md). Stages 1–2 are written (M0); stages 3–6 are titled placeholders filled by later milestones.

Checkpoint paths on the node are assumptions: replace `/models/Qwen3.8-Flash-Next` with the real path and say so in your post-back (system map §5 marks the Flash-Next checkpoint unverified).

## Stage 1: build and native compile

### Purpose
Prove the fork stack builds at the program-branch commit: vLLM at the engine pin imports, the plugin imports, and the native FlashInfer-Gaudi kernels compile — the code the Mac gates mark *compile-unchecked* (system map §14, gate 3).

### Prerequisites
- Base image `vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest` plus `habanalabs-tools` 1.24.1 (tpc-clang), Python 3.10 — the recipe in `~/1cat-build/` (on the node) (system map §3, fork stack).
- Plugin repository `drukoz/1Cat-vLLM-Gaudi`, branch `feat/flash-next-gaudi-m0`, at its current head.
- vLLM engine source at commit `8f816a3f665489d7f0d222115d4f72ebab01076b` (system map, assumption 6).

### Launch
```bash
# on the node, inside a container from the base image (the ~/1cat-build/ recipe, system map §3)
git clone https://github.com/drukoz/1Cat-vLLM-Gaudi.git && cd 1Cat-vLLM-Gaudi
git checkout feat/flash-next-gaudi-m0 && git rev-parse HEAD   # record this in the results table
cd .. && git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 8f816a3f665489d7f0d222115d4f72ebab01076b && cd ..
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation ./vllm
VLLM_TARGET_DEVICE=hpu VLLM_GAUDI_BUILD_FLASHINFER=1 pip install ./1Cat-vLLM-Gaudi --no-build-isolation
pip install --no-deps transformers==5.16.1 huggingface_hub==1.33.0 tokenizers==0.23.2 safetensors==0.8.0
```
Build sequence from system map §3 (fork stack: `VLLM_TARGET_DEVICE=hpu VLLM_GAUDI_BUILD_FLASHINFER=1 pip install . --no-build-isolation`, vLLM `8f816a3f` with `VLLM_TARGET_DEVICE=empty`, pins `--no-deps`).

### Expected startup lines
```bash
python -c "import vllm, vllm_gaudi; print(vllm.__version__)"
python -c "import flashinfer_gaudi; print(flashinfer_gaudi.get_capabilities())"
```
```text
0.28.1rc1.dev392+g8f816a3f6   # system map §3 (fork stack: vLLM 8f816a3f = 0.28.1rc1.dev392+g8f816a3f6)
```
The capabilities command prints the mapping of promoted native operators without raising (system map §6; fused GDN decode is promoted for TP1 16/48 batches 1–32 on a qualified build, system map §7.3). The install built `libflashinfer_gaudi_kernels.so` and `flashinfer_gaudi_ops*.so` (system map §3; names in `flashinfer_gaudi/_native.py:32,46`).

### Capture commands
```bash
docker logs <container> 2>&1 | grep -i -E "recompil|graph compile|bucket|warmup finished|max_num_batched_tokens|GDN layers detected|Compact GDN|mamba"   # R7, RUK-2728
# for the build stage also post: the tail of each pip install log, the plugin git rev-parse HEAD,
# and the full get_capabilities() output.
```

### Results table
| Check | Expected | Observed | Pass? |
|---|---|---|---|
| Plugin commit built | head of `feat/flash-next-gaudi-m0` | | |
| `vllm.__version__` | `0.28.1rc1.dev392+g8f816a3f6` | | |
| `import vllm_gaudi` | no error | | |
| `flashinfer_gaudi.get_capabilities()` | prints a mapping, no exception | | |
| native build | `libflashinfer_gaudi_kernels.so` + `flashinfer_gaudi_ops*.so` present | | |

### Stop rule
Stop the session and post the full traceback and the pip install logs if: the plugin `pip install` fails, the native (TPC/C++) build fails, `import vllm_gaudi` raises, or the printed vLLM version is not `0.28.1rc1.dev392+g8f816a3f6`. A `get_capabilities()` mapping with no promoted ops is *not* a stop — record it and continue.

### What to post back
- the capture output above, complete (build logs included);
- the results table, filled in;
- an `hl-smi` snapshot (on the node).

## Stage 2a: tiny model bring-up

### Purpose
Prove the M0 model class serves the tiny Flash-Next checkpoint and reproduces the HF CPU reference token-for-token (plan.md, Verification model: "the same tiny checkpoint on the Gaudi node at TP1 and TP8 in bf16 … greedy token agreement over 16 tokens on 4 prompts").

### Prerequisites
- The stage 1 image, with the stage 1 checkout mounted (the generator script lives in the repository).
- `tools/flash_next/make_tiny_checkpoint.py` (added by M0-T1, plan.md "Test fixture"): writes `config.json`, `model.safetensors`, the tokenizer files and `reference.json` — `{"prompts": [{"prompt_ids": [...], "generated_ids": [16 ids], "last_prefill_top5": {...}}, ...]}` over 4 prompts of 1 / 7 / 33 / 100 tokens (CPU-generated fp32 reference).
- Output dir `/models/tiny-flash-next` (on the node, arbitrary path).

### Launch
```bash
# in the container, CPU, about a minute
python tools/flash_next/make_tiny_checkpoint.py --out /models/tiny-flash-next
# TP8 needs the 8-head variant: the T1 fixture has 4 attention heads, and vLLM refuses
# "Total number of attention heads (4) must be divisible by tensor parallel size (8)"
# (system map §17 fact 22). Same model otherwise, with its own reference.json.
python tools/flash_next/make_tiny_checkpoint.py --out /models/tiny-flash-next-tp8 \
  --fixture tests/unit_tests/models/flash_next/tiny_tp8_config.json
```
```bash
# TP1 first
vllm serve /models/tiny-flash-next --served-model-name tiny --tensor-parallel-size 1 \
  --max-model-len 512 --max-num-seqs 4 --language-model-only --trust-remote-code --port 8010
```
Send the prompts (below), stop that server, then serve the 8-head checkpoint at TP8 and send the prompts again with `CKPT=/models/tiny-flash-next-tp8`:
```bash
vllm serve /models/tiny-flash-next-tp8 --served-model-name tiny --tensor-parallel-size 8 \
  --max-model-len 512 --max-num-seqs 4 --language-model-only --trust-remote-code --port 8010
```

### Expected startup lines
| Expected line (quoted) | Source | Observed |
|---|---|---|
| `Warmup finished in … secs` | system map §11 (line seen in Run 1/Run 2) | |
| (server answers `GET /v1/models` with model `tiny`) | — | |

### Send the prompts and compare
```bash
# vLLM accepts a list of token ids as the prompt; return_token_ids makes the response
# carry choices[0].token_ids, the generated ids (vllm entrypoints/openai/completion/protocol.py)
# CKPT is the checkpoint being served: /models/tiny-flash-next (TP1) or /models/tiny-flash-next-tp8 (TP8)
CKPT=${CKPT:-/models/tiny-flash-next} python - <<'EOF'
import json, os, urllib.request
ref = json.load(open(os.path.join(os.environ['CKPT'], 'reference.json')))
for i, p in enumerate(ref['prompts']):
    req = {'model': 'tiny', 'prompt': p['prompt_ids'], 'max_tokens': 16,
           'temperature': 0, 'return_token_ids': True}
    r = urllib.request.urlopen(urllib.request.Request(
        'http://localhost:8010/v1/completions',
        data=json.dumps(req).encode(), headers={'Content-Type': 'application/json'}))
    got = json.load(r)['choices'][0]['token_ids']
    exp = p['generated_ids']
    first_diff = next((j for j, (a, b) in enumerate(zip(got, exp)) if a != b), None)
    print(f"prompt {i + 1} ({len(p['prompt_ids'])} tok): "
          f"{sum(a == b for a, b in zip(got, exp))}/16 match, first diff at "
          f"{first_diff if first_diff is not None else '—'}")
EOF
```

### Capture commands
```bash
docker logs <container> 2>&1 | grep -E "Avg prompt throughput|Prefix cache hit rate|Running: |Waiting: "   # R7, RUK-2728
docker logs <container> 2>&1 | grep -i -E "recompil|graph compile|bucket|warmup finished|max_num_batched_tokens|GDN layers detected|Compact GDN|mamba"   # R7, RUK-2728
```

### Results table
| Prompt | prompt tokens | TP1 match | TP8 match | TP8 first differing position |
|---|---|---|---|---|
| 1 | 1 | /16 | /16 | |
| 2 | 7 | /16 | /16 | |
| 3 | 33 | /16 | /16 | |
| 4 | 100 | /16 | /16 | |

TP1 compares with `/models/tiny-flash-next/reference.json`, TP8 with `/models/tiny-flash-next-tp8/reference.json`.

### Stop rule
The generator crashing, the server failing to start at TP1, the server crashing at TP8, any TP1 prompt not matching 16/16, or TP8 prompts 1–3 not matching 16/16 → stop, run the capture commands, post everything. bf16 on HPU may diverge from the fp32 reference after a few tokens: a single late divergence on prompt 4 (the 100-token prompt) is a warning, not a stop — record the first differing position and continue.

### What to post back
- the two `docker logs` greps, complete, for both the TP1 and TP8 containers;
- the results table, filled in;
- the printed per-prompt comparison lines;
- an `hl-smi` snapshot (on the node).

## Stage 2b: real checkpoint TP8 BF16 bring-up

### Purpose
Bring up `/models/Qwen3.8-Flash-Next` at TP8 BF16 on one server (the only layout the 335 GiB checkpoint fits, system map §2a) and compare the KV-cache startup lines with S4's prediction, then check coherence on short prompts.

### Prerequisites
- The stage 1 image and checkpoint `/models/Qwen3.8-Flash-Next` (on the node; assumed path — replace with the real one and say so).
- S4 prediction, run on the Mac before the session (paste the row into your notes):

```bash
# on the Mac (S0 environment), prediction for today's allocation:
/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python \
  spikes/kv-capacity/kv_capacity.py --config spikes/kv-capacity/configs/Qwen3.8-Flash-Next.json \
  --tp 8 --kv bf16 --budget-gib 40
```
```text
row                      tp  kv    budget GiB  block  blocks today  tokens today  agents@131K today  blocks fixed  tokens fixed  agents@131K fixed
Qwen3.8-Flash-Next.json  8   bf16  40          512    3868          1.98M         14.8               6826          3.49M         26.1
```
Predicted for today's allocation (the `today` columns; the `fixed` columns are the M1 GDN-state-pool prediction, system map §2a): block 512, `Usable num_blocks: 3868`, 1.98M KV tokens, 14.8× concurrency at 131,072 tokens per request.

### Launch
```bash
# on the node, cards 0–7, one server (system map §3: one server = the whole node)
export VLLM_COMPACT_GDN=0                 # prefix caching needs it; system map §7.1
export VLLM_HPU_FLASH_NEXT_QSA_MODE=0     # dense QSA fallback = QsaMode.DENSE; plan.md, Target architecture
vllm serve /models/Qwen3.8-Flash-Next --served-model-name Qwen3.8-Flash-Next --port 8000 \
  --tensor-parallel-size 8 --max-model-len 131072 --max-num-seqs 32 \
  --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92
```

### Expected startup lines
| Expected line (quoted) | Source | Observed |
|---|---|---|
| `Loading model weights took … GB` (≈42 GB per rank) | format: system map §11; 335 GiB / 8 = 42 GiB per card, §2a | |
| `Usable num_blocks: 3868` | S4 prediction above, predicted for today's allocation | |
| `HPU KV cache size: … tokens` (≈1.94M; the log reports about 2% below blocks × 512: Run 1 logged 2,647,954 for 5293 blocks) | S4 prediction; line format and ratio: system map §11, Run 1 and §2a | |
| `Maximum concurrency for 131,072 tokens per request: 14.8x` | S4 prediction above | |
| `kv cache group sizes [512, 512, 512, 512]` | block 512 at TP8, 1 attention + 3 GDN groups, system map §2a; line: §11, Run 1 | |
| `Mamba cache mode is set to 'align' … when prefix caching is enabled` | system map §11, Run 1 | |
| `GDN layers detected (36): VLLM_USE_HYBRID_CACHE=…, VLLM_COMPACT_GDN=0` | line format: §11, Run 1 (the 27B logs 48); 36 GDN layers of 48: §2 | |
| `Warmup finished in … secs` (about 400 s) | system map §11, Reading 5 | |

### Coherence check
```bash
# 8 short English prompts (each under 60 tokens), greedy, 64 tokens out;
# prompt 7 asks for a JSON object, prompt 8 for Python code
for p in \
  "Explain in two sentences why the sky is blue during the day." \
  "Write a haiku about the sea." \
  "List the three primary colors and one use of each." \
  "What is the capital of France? Answer in one sentence." \
  "Summarize in one sentence what a compiler does." \
  "Continue this sentence: The best way to learn programming is" \
  "Reply with a JSON object only, keys name, age, city, for a fictional person." \
  "Write a Python function fib(n) that returns the n-th Fibonacci number." ; do
  payload=$(python -c "import json,sys; print(json.dumps({'model':'Qwen3.8-Flash-Next','prompt':sys.argv[1],'max_tokens':64,'temperature':0}))" "$p")
  curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "$payload" \
    | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'].strip())"
  echo '================'
done
```
Pass: every answer is on-topic, grammatical text with no repetition loops.

```bash
# optional quality check — only if lm_eval installs in the container (pip install lm_eval)
lm_eval --model local-completions \
  --model_args model=Qwen3.8-Flash-Next,base_url=http://localhost:8000/v1/completions,num_concurrent=8 \
  --tasks gsm8k --limit 50
```

### Capture commands
```bash
docker logs <container> 2>&1 | grep -E "Avg prompt throughput|Prefix cache hit rate|Running: |Waiting: "   # R7, RUK-2728
docker logs <container> 2>&1 | grep -i -E "recompil|graph compile|bucket|warmup finished|max_num_batched_tokens|GDN layers detected|Compact GDN|mamba"   # R7, RUK-2728
docker logs <container> 2>&1 | grep -E "Usable num_blocks|HPU KV cache size|Maximum concurrency|kv cache group sizes|Mamba cache mode|GDN layers detected|Loading model weights|Warmup finished"   # the startup lines of this stage
```

### Results table
| Check | Expected | Observed | Pass? |
|---|---|---|---|
| Attention block size | 512 | | |
| `Usable num_blocks` | 3868 (predicted, ±5%) | | |
| `HPU KV cache size` | ≈1.94M tokens (S4 1.98M × 0.977) | | |
| `Maximum concurrency … 131,072 tokens` | ≈14.8x | | |
| `kv cache group sizes` | `[512, 512, 512, 512]` | | |
| `GDN layers detected` | `(36)` with `VLLM_COMPACT_GDN=0` | | |
| `Mamba cache mode` | `align` | | |
| `Warmup finished in … secs` | about 400 s | | |
| Coherence, 8 prompts | all on-topic, grammatical, no loops | | |
| GSM8K, 50 items (optional) | record the score | | |

### Stop rule
Stop the session and post the captures if: startup crashes or OOMs at any rank; the attention block size is not 512; `Usable num_blocks` is more than 10% below the prediction 3868 (check `VLLM_COMPACT_GDN=0` took effect — the `GDN layers detected` line must show `VLLM_COMPACT_GDN=0`); any coherence answer is gibberish or falls into a repetition loop. A single prompt mildly off-topic is a warning: record the full text and continue. GSM8K is optional at this stage.

### What to post back
- the three `docker logs` greps, complete;
- the results table, filled in;
- the eight coherence answers in full;
- an `hl-smi` snapshot (on the node).

## Stages 3–6 (placeholders)

- **Stage 3: flag-by-flag A/B of each native kernel and fusion against its torch path** (system map §14) — filled by M2.
- **Stage 4: R7's scheduler variants A–D and the prefix-cache hit rate** (system map §15, R7) — filled by M1.
- **Stage 5: TP1/TP2/TP4/TP8, PP2×TP4 and TP+EP throughput and latency** (system map §16, items 5b/5c) — filled by M4.
- **Stage 6: 128K–262K contexts and the eight-agent stress run with quality checks** (system map §16, item 5d) — filled by M1/M4.
