# Flash-Next qualification run sheet

Stages are ordered fail-fast (system map §14): a stop rule ends the rental session for the day; the captures come back and the fix is made offline. Startup costs about 400 s per launch (system map §11, Reading 5), so share a launch between checks wherever possible. Sections follow [the template](README.md). Stages 1, 2a, 2b are written (M0), stages 3 and 4 are written (M2, M1); stages 5, 6 are titled placeholders filled by later milestones.

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
Prove the M0 model class serves the tiny Flash-Next checkpoint and reproduces the HF CPU reference token-for-token (plan.md, Verification model: "the same tiny checkpoint on the Gaudi node at TP1 and TP8 in bf16 … greedy token agreement over 16 tokens on 5 prompts").

### Prerequisites
- The stage 1 image, with the stage 1 checkout mounted (the generator script lives in the repository).
- `tools/flash_next/make_tiny_checkpoint.py` (added by M0-T1, plan.md "Test fixture"): writes `config.json`, `model.safetensors`, the tokenizer files and `reference.json` — `{"prompts": [{"prompt_ids": [...], "generated_ids": [16 ids], "last_prefill_top5": {...}}, ...]}` over 5 prompts of 1 / 7 / 33 / 100 / 700 tokens (CPU-generated fp32 reference).
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
| 5 | 700 | /16 | /16 | |

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
row                      tp  kv    budget GiB  block  blocks today  tokens today  agents@131K today  slots  pool GiB  blocks pool  tokens pool  agents@131K pool
Qwen3.8-Flash-Next.json  8   bf16  40          512    2862          1.47M         10.9               896    4.18      6113         3.13M        23.9
```
Predicted for today's allocation (the `today` columns): block 512, `Usable num_blocks: 2862`, 1.47M KV tokens, 10.9× concurrency at 131,072 tokens per request. S4 applies both worker shrink factors — the GDN factor and, for Flash-Next's PLE short conv (a standard-mamba spec), `page / (page + ple_state)` (`hpu_worker.py:1142-1166`). The `today` numbers are lower than the ones this sheet quoted at M0 (3868 blocks) because S4 then missed the PLE factor; the 27B validation rows are unchanged (no PLE). The `pool` columns are the M1 recurrent state pool prediction (896 slots, 4.18 GiB; measured in stage 4 variant E; system map §2a).

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
| `Usable num_blocks: 2862` | S4 prediction above, predicted for today's allocation (both worker factors, PLE included) | |
| `HPU KV cache size: … tokens` (≈1.43M; the log reports about 2% below blocks × 512: Run 1 logged 2,647,954 for 5293 blocks) | S4 prediction; line format and ratio: system map §11, Run 1 and §2a | |
| `Maximum concurrency for 131,072 tokens per request: 10.9x` | S4 prediction above | |
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
| `Loading model weights took … GB` | ≈42 GB per rank | | |
| `Usable num_blocks` | 2862 (predicted, ±5%) | | |
| `HPU KV cache size` | ≈1.43M tokens (S4 1.47M × 0.977) | | |
| `Maximum concurrency … 131,072 tokens` | ≈10.9x | | |
| `kv cache group sizes` | `[512, 512, 512, 512]` | | |
| `GDN layers detected` | `(36)` with `VLLM_COMPACT_GDN=0` | | |
| `Mamba cache mode` | `align` | | |
| `Warmup finished in … secs` | about 400 s | | |
| Coherence, 8 prompts | all on-topic, grammatical, no loops | | |
| GSM8K, 50 items (optional) | record the score | | |

### Stop rule
Stop the session and post the captures if: startup crashes or OOMs at any rank; the attention block size is not 512; `Usable num_blocks` is more than 10% below the prediction 2862 (check `VLLM_COMPACT_GDN=0` took effect — the `GDN layers detected` line must show `VLLM_COMPACT_GDN=0`); any coherence answer is gibberish or falls into a repetition loop. A single prompt mildly off-topic is a warning: record the full text and continue. GSM8K is optional at this stage.
Warmup failing inside `mixture_of_experts`, or crawling (far past the 400 s the 27B needed), is not a direct stop: relaunch once
with `MAX_EXPERTS_PER_SLICE=64` — the tested slice point; the default -1 runs one `mixture_of_experts` call over all 512 experts
(`vllm_gaudi/extension/ops.py:58-63`) — and record that in the results; stop only if the 64-slice launch fails the same way.

### What to post back
- the three `docker logs` greps, complete;
- the results table, filled in;
- the eight coherence answers in full;
- an `hl-smi` snapshot (on the node).

## Stage 4: admission variants A–D, the prefix-hit check, and the recurrent state pool

### Purpose
Run R7's scheduler variants A–D (the prefix-cache hit check is D) and variant E, the M1 recurrent state pool, against S4's prediction: prove admission is a scheduler-budget setting and that the pool's `Usable num_blocks` matches the model, and rule out a pool that over-reserves memory or an `align`-mode state-alignment crash (R7, RUK-2728; system map §15).

### Prerequisites
- The stage 1 fork build; plugin branch `feat/flash-next-gaudi-m1` at its current head, with M1-T1/T3 (the recurrent state pool) merged; the engine pin unchanged.
- Checkpoint `/models/Qwen3.8-Flash-Next` (on the node; stage 2b's path).
- S4's predictions, run on the Mac before the session (paste into your notes):

```bash
# on the Mac (S0 environment):
/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python \
  spikes/kv-capacity/kv_capacity.py --config spikes/kv-capacity/configs/Qwen3.8-Flash-Next.json \
  --tp 8 --kv bf16 --budget-gib 40 --pool-slots 896
```
```text
row                      tp  kv    budget GiB  block  blocks today  tokens today  agents@131K today  slots  pool GiB  blocks pool  tokens pool  agents@131K pool
Qwen3.8-Flash-Next.json  8   bf16  40          512    2862          1.47M         10.9               896    4.18      6113         3.13M        23.9
```
Predicted: today's allocation (variants A–D, no pool): block 512, `Usable num_blocks: 2862`, 1.47M KV tokens, 10.9× concurrency; with the pool (variant E): 896 slots, 4.18 GiB reserved, `Usable num_blocks: 6113`, 3.13M KV tokens, 23.9 agents at 131,072 tokens (`blocks / 256`; the slots term `896 / 12 = 74.7` is not binding).

### Launch
One variant per launch (cards 0–7, one server; startup about 400 s, system map §11, Reading 5). Variant A, in full:
```bash
# on the node, in the stage 1 container; name the container fn4-a (fn4-b1, fn4-c, fn4-e, ...)
unset VLLM_COMPACT_GDN                  # compact GDN is 0 by default under prefix caching since M1-T1
export VLLM_HPU_FLASH_NEXT_QSA_MODE=0   # dense QSA fallback, as in stage 2b
vllm serve /models/Qwen3.8-Flash-Next --served-model-name Qwen3.8-Flash-Next --port 8000 \
  --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching \
  --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 --max-num-seqs 32
```
Each variant adds to that command:

| Variant | Added flags | What it tests |
|---|---|---|
| A (control) | `--max-num-batched-tokens 2048 --long-prefill-token-threshold 0` (the pin's defaults, made explicit) | the pin's defaults, stated in the command line (R7, RUK-2728) |
| B1 | `--max-num-batched-tokens 8192` | a bigger per-step token budget (R7) |
| B2 | `--max-num-batched-tokens 16384` | the budget raised further (R7) |
| C | `--max-num-batched-tokens 8192 --long-prefill-token-threshold 2048` | several prefills share a step; 2048 is a multiple of the 512-token GDN block (R7) |
| D | no launch: the prefix-hit check on C's server | the prefix-cache hit check (see below) |
| E | `export VLLM_HPU_RECURRENT_STATE_POOL=1` (plus C's flags) | the M1 recurrent state pool (M1-T1, M1-T3) |
| F (optional) | E plus `--prefix-cache-retention-interval 4096` | sparse mamba boundary states vs pool slots; confirm the flag exists first: `vllm serve --help \| grep retention` |

Variant D (on C's server): send prompt `a` of at least 1536 tokens (three 512-token blocks), `a` again, then `b` sharing `a`'s first 1024 tokens; `vllm bench serve` and the user's client both work. The script below builds the prompts and prints each request's first-token arrival:
```bash
# on the node, against the running server (variant C); needs aiohttp (pip install aiohttp)
python - <<'EOF'
import asyncio, time
import aiohttp

def prompt(n_words, tail):
    sysblk = ("You are a precise operations assistant. Follow the policy exactly. " * n_words).strip()
    return sysblk + tail

a = prompt(300, " End of the shared system block.\n")          # 3000 words, about 3900 tokens (>= 1536)
b = prompt(100, " This agent adds its own note. " + "Agent specific detail follows here. " * 300)  # ~1300 shared tokens (2 whole blocks), then diverges; about a's length
prompts = [("a cold", a), ("a again", a), ("b, shares a", b)]

def session():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800))

async def timed(sess, tag, text):
    payload = {"model": "Qwen3.8-Flash-Next", "prompt": text, "max_tokens": 16,
               "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    async with sess.post("http://localhost:8000/v1/completions", json=payload) as r:
        r.raise_for_status()
        async for _ in r.content:  # first streamed chunk ~= first token
            print(f"{tag}: first token at {time.perf_counter() - t0:.2f} s")
            break

async def main():
    async with session() as sess:
        for tag, text in prompts:
            await timed(sess, tag, text)
    print("words: a", len(a.split()), "b", len(b.split()), "(about 1.3 tokens per word)")

asyncio.run(main())
EOF
# after the second request, check the server's rate line:
docker logs fn4-c 2>&1 | grep "Prefix cache hit rate" | tail -n 3
```
Expect a `Prefix cache hit rate` above 0% after the second request and `b`'s first token faster than `a`'s cold run.

Variant E, then the 16-agent run of system map §11 (the user's agent workload, as in Run 2's 16-agent rows), recording first-token median/p90/p99, decode per agent, total output, wall time, `Running:` / `Waiting:` at 60 s and 120 s, and the final hit rate.

Variant F (optional): the E launch with `--prefix-cache-retention-interval 4096` (vLLM at the pin keeps sparse mamba boundary states; more retained states trade pool slots). Confirm the flag exists first:
```bash
vllm serve --help 2>&1 | grep retention   # absent on the pin's vllm -- skip F and record that
```

### Expected startup lines
| Expected line (quoted) | Source | Observed |
|---|---|---|
| `Usable num_blocks: 2862` (variants A–D, ±5%) | S4 prediction above (today's allocation, both worker factors) | |
| `HPU hybrid cache: reducing available KV cache memory by … (factor=0.567)` | format: `hpu_worker.py:1145-1153`; page 524288, GDN 400896 (S4) | |
| `[HPU] Recurrent state pool: 896 slots (max_num_seqs=32, recurrent groups=4, live slots per request=3, cache slots=512)` (E) | M1-T1 log format; 896 predicted by S4's slots formula | |
| `Recurrent state pool: 896 slots x 4.76 MiB per slot = 4.18 GiB reserved; … left for attention blocks` (E) | M1-T3 log format; 4.18 GiB predicted by S4 | |
| `Usable num_blocks: 6113` (E, within 2%: 5991–6235) | S4 pool prediction above | |
| `Maximum concurrency for 131,072 tokens per request: 23.9x` (E; record if it differs: vLLM's own formula may still charge GDN blocks per request) | S4 pool prediction above | |

### Capture commands
```bash
docker logs <container> 2>&1 | grep -E "Avg prompt throughput|Prefix cache hit rate|Running: |Waiting: "   # R7, RUK-2728
docker logs <container> 2>&1 | grep -i -E "recompil|graph compile|bucket|warmup finished|max_num_batched_tokens|GDN layers detected|Compact GDN|mamba|Recurrent state pool|Usable num_blocks|Maximum concurrency"   # R7, RUK-2728, extended (M1-T1/T3 pool lines, S4 lines)
docker logs fn4-e 2>&1 | grep -E "Recurrent state pool|Usable num_blocks|HPU KV cache size|Maximum concurrency"   # variant E startup block (M1-T1/T3 formats, S4)
```

### Results table
| Check | Expected | Observed | Pass? |
|---|---|---|---|
| A, `Usable num_blocks` | 2862 (predicted, ±5%) | | |
| A/B1/B2/C, `Running:`/`Waiting:` during the run | record (R7's point: more running, less waiting as the budget grows) | | |
| A/B1/B2/C, final `Prefix cache hit rate` | record | | |
| D, hit rate after the second request | > 0% | | |
| D, first token: `b` vs `a` cold | `b` faster | | |
| E, pool line 1 (slots 896, groups 4, live 3, cache 512) | as quoted above | | |
| E, pool line 2 (`896 slots x 4.76 MiB per slot = 4.18 GiB reserved`) | as quoted above | | |
| E, `Usable num_blocks` | 6113 (predicted, ±2%: 5991–6235) | | |
| E, `Maximum concurrency` | 23.9x (predicted) | | |
| E, 16-agent first-token median / p90 / p99 | record (Run 2's 16-agent cold rows are the reference, system map §11) | | |
| E, decode per agent / total output / wall time | record | | |
| E, `Running: N reqs` / `Waiting: N reqs` at 60 s and 120 s | record | | |
| E, final prefix cache hit rate | record | | |
| F, `Usable num_blocks` (optional) | record | | |

### Stop rule
A `ValueError` naming `VLLM_COMPACT_GDN` at startup means the environment forces compact GDN on: unset the variable and relaunch (this is an environment fix, not a stop). Stop **variant E** (you may keep the A–D results) and post the third capture command's output if its `Usable num_blocks` is more than 5% off the prediction 6113. Stop the **stage** if any traceback contains `assert context_lens[0] % self.mamba_chunk_size`: post the full traceback and the scheduler lines around it (the capture greps plus `docker logs <container> 2>&1 | tail -n 200`). A server crash at any variant: stop the session and post the captures.

### What to post back
- the capture greps, complete, for every variant launched;
- the three D timing lines and the hit-rate lines after the second request;
- the results table, filled in;
- the E 16-agent run numbers (first-token median/p90/p99, decode per agent, total output, wall time, `Running`/`Waiting` at 60 s and 120 s, final hit rate);
- an `hl-smi` snapshot (on the node).

### Client guidance (for D and E's 16-agent run)
A prefix-cache hit needs whole block-sized prefixes: 512 tokens at TP8 (896 at TP4), identical from token 0. GDN state is cached only at block boundaries, so pad the shared system prefix to a multiple of the block and put the per-agent text after it. `--prefix-cache-retention-interval` trades retained boundary states for pool slots.

## Stage 3: flag-by-flag A/B — sparse QSA, GDN tactics at TP8, MoE expert slices, decode batch baseline

### Purpose
A/B every M2 flag group against the stage 2b plain launch on the real checkpoint — sparse QSA (3a), the GDN tactics at TP8 (3b), the
MoE expert slices (3c) — and record the decode batch baseline (3d) that the M3/M5 decode work is measured against (system map §14;
plan v0.3, M2 exit rule: sparse QSA becomes the default only after this stage passes).

### Prerequisites
- The stage 1 fork build; plugin branch `feat/flash-next-gaudi-m2` at its current head (every flag below exists only there), engine pin unchanged.
- Checkpoint `/models/Qwen3.8-Flash-Next` (on the node; stage 2b's path). Stage 3 runs before stage 4 in the session (fail-fast order 1, 2a, 2b, 3, 4).
- Stage 2b's S4 row in your notes (`Usable num_blocks: 2862`). 3a predicts 0.94 x that (~2690) on the sparse launch: sparse mode reserves
  about 6% of the attention budget for the compressed keys (plan v0.3, M2 fact 2).
- `vllm bench serve` in the container (ships with vLLM). 3a re-generates the tiny checkpoint: the M2 generator writes a `reference.json`
  with six prompts, the sixth 2600 tokens — past the 2050-token boundary where dense equals the model's sparse attention (plan v0.3, M2 fact 1).

### Launch
Five launches, about 400 s of warmup each (system map §11, Reading 5), run in this order: launch 1 is stage 2b's command verbatim and
serves 3d plus the "off" side of 3a-3c, every later launch adds exports to it. Launch 1:

```bash
# on the node, cards 0–7, one server; name the containers fn3-plain, fn3-sparse, fn3-tactics, fn3-slice64, fn3-slice128
export VLLM_COMPACT_GDN=0                 # prefix caching needs it; system map §7.1
export VLLM_HPU_FLASH_NEXT_QSA_MODE=0     # dense QSA fallback = QsaMode.DENSE; plan.md, Target architecture
vllm serve /models/Qwen3.8-Flash-Next --served-model-name Qwen3.8-Flash-Next --port 8000 \
  --tensor-parallel-size 8 --max-model-len 131072 --max-num-seqs 32 \
  --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92
```

Launches 2–5 repeat that command with the extra exports (set inside each launch block, never inherited from the shell):

```bash
# launch 2, fn3-sparse (3a):
export VLLM_HPU_FLASH_NEXT_QSA_MODE=1
# launch 3, fn3-tactics (3b) — all four are needed; the TP8 geometry is 2 key / 6 value heads per rank,
# packed width 1280 (plan v0.3, M2 fact 4; VLLM_HPU_FLASHINFER_GDN_TP8 is named by the plan, added by M2-T6):
export VLLM_HPU_FLASHINFER_GDN=1 VLLM_HPU_FLASHINFER_GDN_PREFILL=1
export VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE=1 VLLM_HPU_FLASHINFER_GDN_TP8=1
# launch 4, fn3-slice64 (3c):   export MAX_EXPERTS_PER_SLICE=64
# launch 5, fn3-slice128 (3c):  export MAX_EXPERTS_PER_SLICE=128
```

`MAX_EXPERTS_PER_SLICE` is a plain environment variable read at `vllm_gaudi/extension/ops.py:58-63` (default -1 = one
`mixture_of_experts` call over all 512 experts; the code comment names 64 as the tested slice). 3d benches a 32768-token input:
launch 1 above has `--max-model-len 131072` (stage 2b), which fits 32768 + 256; if your stage 2b launch used less, launch
fn3-plain with `--max-model-len 65536`.

### Expected startup lines
| Expected line (quoted) | Source | Observed |
|---|---|---|
| `Warmup finished in … secs` (about 400 s; 3c records the seconds per slice) | system map §11, Reading 5 | |
| `Usable num_blocks: 2862` (fn3-plain, fn3-tactics, fn3-slice64, fn3-slice128; ±5%) | stage 2b's S4 prediction | |
| `Usable num_blocks: ≈2690` (fn3-sparse: 0.94 x 2862, the compressed keys reserve ~6% of the attention budget) | plan v0.3, M2 fact 2 | |
| `GDN layers detected (36): … VLLM_COMPACT_GDN=0` (every launch; the line names no QSA/tactic mode — 3a and 3b say what to record) | stage 2b's line | |

### Capture commands
```bash
docker logs <name> 2>&1 | grep -E "Usable num_blocks|Warmup finished|Prefix cache hit rate|Avg generation throughput|Traceback"   # post for every launch
docker logs <name> 2>&1 | grep -i -E "recompil|graph compile|bucket|warmup finished|max_num_batched_tokens|GDN layers detected|Compact GDN|mamba"   # R7, RUK-2728
docker logs <name> 2>&1 | grep -i "qsa"   # 3a: no startup line names the QSA mode today (the layer logs none) — record what prints, plus the launch's VLLM_HPU_FLASH_NEXT_QSA_MODE value
```

### 3a: sparse QSA A/B (tiny 2600-token prompt, 20K needle prompt)
Tiny side first, port 8010 (the tiny server boots in seconds; it is stage 2a's server with `--max-model-len 4096` — 2600 + 16
tokens do not fit stage 2a's 512). Sparse side, then dense side:

```bash
# on the node, CPU, about a minute: the M2 generator adds the 2600-token sixth prompt
python tools/flash_next/make_tiny_checkpoint.py --out /models/tiny-flash-next
export VLLM_HPU_FLASH_NEXT_QSA_MODE=1    # sparse side (the M2 default after this stage passes)
vllm serve /models/tiny-flash-next --served-model-name tiny --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 4 --language-model-only --trust-remote-code --port 8010
# run stage 2a's "Send the prompts and compare" block (CKPT=/models/tiny-flash-next), saving the printed lines:
#   <the stage 2a block> | tee /tmp/3a-tiny-sparse.txt
# stop that server, then the dense side: export VLLM_HPU_FLASH_NEXT_QSA_MODE=0, same serve command,
# run the compare block again, saving to /tmp/3a-tiny-dense.txt
```

Expect: sparse mode, 6/6 prompts match 16/16 with `reference.json`. Dense mode: prompts 1–5 match 16/16 and prompt 6 (2600 tokens)
is expected to differ — dense equals the model's sparse attention only up to 2050 tokens (plan v0.3, M2 fact 1). Record both files.

Real checkpoint: the 20K-token needle prompt below, run against fn3-sparse (launch 2) and fn3-plain (launch 1). Filler sentences
repeat, the secret sits at 30% depth, about 1.3 tokens per word (stage 4's client note); the print gives first-token seconds
(first streamed chunk ~= first token, stage 4's convention), decode ms/token over 16 tokens, and the answer:

```bash
python - <<'EOF'
import json, time, urllib.request
filler = "The quarterly report describes logistics, staffing, and maintenance schedules in detail.\n"
lines = [filler] * 1460                                      # about 20K tokens of filler
lines[int(0.3 * len(lines))] = "The secret code is 7418.\n"  # needle at 30% depth
prompt = "".join(lines) + "Question: What is the secret code? Answer with the digits only.\n"
payload = {"model": "Qwen3.8-Flash-Next", "prompt": prompt, "max_tokens": 16, "temperature": 0, "stream": True}
t0, first, text = time.perf_counter(), 0.0, ""
with urllib.request.urlopen("http://localhost:8000/v1/completions", data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"}) as r:
    for chunk in r:
        line = chunk.decode(errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        first = first or time.perf_counter() - t0
        if line != "data: [DONE]":
            text += json.loads(line[5:])["choices"][0].get("text", "")
print(f"first token {first:.2f} s, decode {1000 * (time.perf_counter() - t0 - first) / 16:.0f} ms/token: {text!r}")
EOF
```

Expect on fn3-sparse: the answer names 7418. On fn3-plain record the answer and the timings — behaviour past 2050 tokens is the
known dense divergence, that is the A/B. Then run stage 2b's 8-prompt coherence loop against fn3-sparse, saving the answers, and,
if stage 2b ran the GSM8K sample, run it on fn3-sparse too (same `lm_eval` command, 50 items).

```bash
# capture for 3a (the stage greps, plus):
docker logs fn3-sparse 2>&1 | grep -E "Usable num_blocks|Warmup finished"   # ≈2690 blocks expected (0.94 x 2862)
docker logs fn3-sparse 2>&1 | grep -i "qsa"   # record what prints: the mode-naming startup line does not exist today
```

Results table (3a):

| Check | Expected | Observed | Pass? |
|---|---|---|---|
| Tiny prompt 6 (2600 tok), sparse mode | 16/16 | | |
| Tiny prompt 6, dense mode | differs — record the first-differing position | | |
| Tiny prompts 1–5, sparse mode | 16/16 | | |
| 20K needle, fn3-sparse | answer contains 7418 | | |
| 20K needle, fn3-plain | record (divergence past 2050 tokens expected) | | |
| 8 short prompts, fn3-sparse | all on-topic, grammatical, no loops | | |
| GSM8K, 50 items (only if 2b ran it) | record the score | | |
| `Usable num_blocks`, fn3-sparse | ≈2690 (0.94 x 2862) | | |
| 20K first token, plain / sparse | record both | | |
| 20K decode ms/token, plain / sparse | record both | | |

Stop rule (3a): fn3-sparse crashes at startup, the tiny prompt 6 does not match 16/16 in sparse mode, or the sparse 20K needle
answer misses the code → stop the session and post the captures, both tiny files and both needle prints. Dense prompt 6 differing
is the expected result, not a stop.

Post back (3a): `/tmp/3a-tiny-sparse.txt`, `/tmp/3a-tiny-dense.txt`, the two needle prints, the sparse 8-prompt answers (and the
GSM8K score if run), the fn3-sparse startup capture, the filled 3a table.

### 3b: GDN tactics at TP8 (four flags, decode sweep)
Correctness first: stage 2b's 8-prompt loop, greedy 64 tokens, written to a file on fn3-plain (launch 1) and fn3-tactics (launch 3):

```bash
OUT=/tmp/3b-plain.txt      # on fn3-tactics: OUT=/tmp/3b-tactics.txt
: > "$OUT"
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
    | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['text'].strip())" >> "$OUT"
  echo '================' >> "$OUT"
done
diff /tmp/3b-plain.txt /tmp/3b-tactics.txt    # expected: no output
```

The 8 greedy outputs must be identical across the two launches (both bf16; report any difference). Expected startup lines for
fn3-tactics: stage 2b's set — the four GDN exports change no startup line (R1). Performance sweep, on both servers, N = 4 x C:

```bash
NAME=plain               # on fn3-tactics: NAME=tactics
for C in 1 8 16 32; do
  vllm bench serve --backend vllm --model Qwen3.8-Flash-Next --dataset-name random \
    --random-input-len 4096 --random-output-len 256 --num-prompts $((4 * C)) --max-concurrency "$C" \
    > "/tmp/3b-$NAME-c$C.log" 2>&1
  grep -E "Output token throughput|Mean TPOT|P99 TPOT" "/tmp/3b-$NAME-c$C.log"
done
```

The fused decode step only runs on the direct-state GDN layout and the batch buckets {1, 2, 4, 8, 16, 32}
(`vllm_gaudi/ops/flashinfer_gaudi_adapter.py:237-260`) — the sweep's C values are exactly the buckets. No startup line proves the
tactic engaged (R1): the proof is the TPOT difference between the two tables, plus, if you can, profiling later without
`PT_HPU_LOG_LEVEL`. Capture for 3b: the stage greps for fn3-tactics. Results tables (3b), one per launch:

| C | N = 4C | Output token throughput (tok/s) | Mean TPOT (ms) | P99 TPOT (ms) |
|---|---|---|---|---|
| 1 | 4 | | | |
| 8 | 32 | | | |
| 16 | 64 | | | |
| 32 | 128 | | | |

(Repeat the table for fn3-tactics; add the rows below to one of them.)

| Check | Expected | Observed | Pass? |
|---|---|---|---|
| `diff` of the two greedy files | empty (report any difference) | | |
| fn3-tactics Mean/P99 TPOT vs plain | record both; the delta is the gate's measurement, a loss is a finding, not a stop | | |

Stop rule (3b): any difference between the two greedy files → stop the session, post both files and the capture greps of both
launches; fn3-tactics crashes at startup → stop, post the traceback. A tactics loss in the tables is a result, not a stop.

Post back (3b): the two greedy files, the two filled bench tables, the eight `/tmp/3b-*.log` files, the fn3-tactics capture greps.

### 3c: MoE expert slices (unset / 64 / 128)
Three points, one per launch: unset (fn3-plain: the -1 default, one `mixture_of_experts` call over all 512 experts), 64 (launch 4),
128 (launch 5). Expected line per launch: `Warmup finished in … secs` — the seconds are this sub-stage's measurement:

```bash
docker logs <name> 2>&1 | grep -E "Warmup finished|Usable num_blocks|Traceback"   # per launch; the warmup seconds are the point
```

Then stage 2b's 8-prompt loop into a file (the 3b loop with `OUT=/tmp/3c-slice64.txt`, then `OUT=/tmp/3c-slice128.txt`) — slicing
re-chunks the expert loop, so the answers must equal `/tmp/3b-plain.txt` — and the C = 16 bench row on each launch:

```bash
diff /tmp/3b-plain.txt /tmp/3c-slice64.txt     # expected: empty; same check for /tmp/3c-slice128.txt
NAME=slice-unset       # on fn3-slice64: NAME=slice64; on fn3-slice128: NAME=slice128
vllm bench serve --backend vllm --model Qwen3.8-Flash-Next --dataset-name random \
  --random-input-len 4096 --random-output-len 256 --num-prompts 64 --max-concurrency 16 \
  > "/tmp/3c-$NAME-c16.log" 2>&1
grep -E "Output token throughput|Mean TPOT|P99 TPOT" "/tmp/3c-$NAME-c16.log"
```

Results table (3c):

| Launch | `Warmup finished` (s) | 8 prompts identical to plain? | Output token throughput (C=16) | Mean TPOT (ms) | P99 TPOT (ms) |
|---|---|---|---|---|---|
| unset, -1 (fn3-plain) | | | | | |
| 64 (fn3-slice64) | | | | | |
| 128 (fn3-slice128) | | | | | |

Stop rule (3c): warmup fails inside `mixture_of_experts` at any point (the traceback names it) → stop the session and post the
traceback and the launch's capture greps; a crawl (far past the 400 s the 27B needed) is the measurement, record it. Answers not
identical to plain → stop and post the diffs.

Post back (3c): the filled 3c table, the three Warmup lines, the two diff results, the three C=16 rows and `/tmp/3c-*.log` files,
the fn3-slice64 / fn3-slice128 capture greps.

### 3d: decode batch baseline (4K and 32K input)
On fn3-plain (dense QSA, no tactics) — the before-picture for the M3/M5 decode work. Expected startup lines: stage 2b's set
(launch 1 is stage 2b's launch). Two sweeps, N = 4 x C, 256 tokens out:

```bash
NAME=4k; IN=4096         # then the second sweep: NAME=32k; IN=32768 (fits --max-model-len 131072; see Launch)
for C in 1 8 16 32; do
  vllm bench serve --backend vllm --model Qwen3.8-Flash-Next --dataset-name random \
    --random-input-len "$IN" --random-output-len 256 --num-prompts $((4 * C)) --max-concurrency "$C" \
    > "/tmp/3d-$NAME-c$C.log" 2>&1
  grep -E "Output token throughput|Mean TPOT|P99 TPOT" "/tmp/3d-$NAME-c$C.log"
done
```

During each sweep, from a second shell, sample the engine stats lines (the stage 4 capture grep):

```bash
docker logs fn3-plain 2>&1 | grep -E "Avg generation throughput|Running: " | tail -n 20
```

Results table (3d) — per-request TPOT, aggregate output throughput, and the engine stats lines:

| Sweep | C | N | Output token throughput (tok/s) | Mean TPOT (ms) | P99 TPOT (ms) | `Avg generation throughput` / `Running:` during the run |
|---|---|---|---|---|---|---|
| 4K in / 256 out | 1 | 4 | | | | |
| 4K in / 256 out | 8 | 32 | | | | |
| 4K in / 256 out | 16 | 64 | | | | |
| 4K in / 256 out | 32 | 128 | | | | |
| 32K in / 256 out | 1 | 4 | | | | |
| 32K in / 256 out | 8 | 32 | | | | |
| 32K in / 256 out | 16 | 64 | | | | |
| 32K in / 256 out | 32 | 128 | | | | |

Stop rule (3d): fn3-plain crashes, OOMs, or a sweep dies mid-run → stop the session, post the captures and the log of the failing
sweep. A low 32K throughput is data, record it.

Post back (3d): the filled 3d table, the eight `/tmp/3d-*.log` files, the engine-stats lines taken during the sweeps.

### What to post back
- the results tables (3a, 3b's two bench tables, 3c, 3d), filled in;
- the greedy-output files: `/tmp/3a-tiny-sparse.txt`, `/tmp/3a-tiny-dense.txt`, `/tmp/3b-plain.txt`, `/tmp/3b-tactics.txt`,
  `/tmp/3c-slice64.txt`, `/tmp/3c-slice128.txt`;
- `docker logs <name> 2>&1 | grep -E "Usable num_blocks|Warmup finished|Prefix cache hit rate|Avg generation throughput|Traceback"`
  for each of the five launches, complete;
- the two 20K needle prints (answer, first-token s, decode ms/token — sparse and plain) and the raw `/tmp/3b-*.log`, `/tmp/3c-*.log`,
  `/tmp/3d-*.log` bench logs;
- an `hl-smi` snapshot (on the node).

## Stages 5, 6 (placeholders)

- **Stage 5: TP1/TP2/TP4/TP8, PP2×TP4 and TP+EP throughput and latency** (system map §16, items 5b/5c) — filled by M4.
- **Stage 6: 128K–262K contexts and the eight-agent stress run with quality checks** (system map §16, item 5d) — filled by M1/M4.
