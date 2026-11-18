# Flash-Next rental run sheet (one command)

This is the sheet you run in the rental session. Everything is automated: you build one bundle on
the Mac, copy it to the node, and run **one command**. The script starts the container, installs
the stack, runs every gate, and leaves **one results tarball** to send back. You do not need to
read [qualification.md](qualification.md) — it stays as background for the manual stages.

## What the run is

The bundle carries the plugin (this repository, at a clean commit) and vLLM at the engine pin
`8f816a3f665489d7f0d222115d4f72ebab01076b`. On the node, `run_rental.sh` starts the Habana
container, `container_entry.sh` installs the stack inside it, and the orchestrator
(`python -m tools.flash_next.rental.main`) runs, in order: preflight, build check, tiny TP8, then
one phase per variant (A, B, C), then the report. Each gate has a stop rule (table below); the run
stops at the first gate that trips, and you send the tarball back.

## Before the rental (on the Mac, any time)

1. Make sure the checkout is clean (`git status --porcelain` empty) at the commit you want to test,
   then build the bundle:

   ```bash
   bash tools/flash_next/rental/make_rental_bundle.sh --out dist
   # prints: bundle: dist/flash-next-rental-<sha>.tar.gz   (about 56 MB)
   ```

   The bundle contains `run_rental.sh` (top level), `plugin/` (this repository at its commit),
   `vllm/` (at the engine pin) and `BUNDLE.txt` (plugin sha, branch, vLLM commit, build date).

2. Copy the tarball to the node (on the node, "scp" from your workstation is fine):

   ```bash
   scp dist/flash-next-rental-*.tar.gz <node>:~/
   ```

3. Disk on the node: the FP8 checkpoint is about 185.5 GB and the results add a few GB — allow
   **at least 250 GB free** for a run with `--download`, at least 20 GB free otherwise. The run
   refuses to start when the model parent folder does not exist.

4. The checkpoint: if the node does not have it yet, add `--download` to the command below; the
   script downloads `Qwen/Qwen3.8-Flash-Next-FP8` (public, not gated) into the `--model` directory
   inside the container (`hf download`, about 185.5 GB — plan 1–2 h of rental time for it).

## The one command (on the node)

```bash
tar xzf ~/flash-next-rental-*.tar.gz -C ~/rental && cd ~/rental
bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8
```

That is all. Useful options:

| Option | Meaning |
|---|---|
| `--model DIR` | checkpoint directory (required); mounted at `/models/<name>` in the container |
| `--download` | download the checkpoint into `--model` first (needs `hf`, installed by the run) |
| `--results DIR` | results folder on the host (default `./results`) |
| `--variants A,B,C` | run only these variants (default: all in the run) |
| `--with-profile` | enable profiling |
| `--from PHASE` | resume from a phase (`preflight`, `build`, `tiny`, a variant id, `report`) |
| `--run-id ID` | fixed run id (default: UTC timestamp) |
| `--image IMG` | container image (default the 1.24.1 base image) |
| `--dry-run` | print the `docker run` command and exit |

When the run stops at a stop rule, restart after fixing the cause with `--run-id <same id>
--from <phase>` to keep the results in one folder.

## Expected duration per phase

| Phase | Expected duration |
|---|---|
| preflight | about 1 min |
| install (vLLM, plugin with native kernels, model deps) | 20–40 min |
| tiny TP8 | about 10 min |
| each variant | about 35–45 min (startup up to 30 min including warmup, context fill, four turns) |

A full run (three variants, no `--download`) is about 3–4 h; add the download on top.

## What each gate checks and its stop rule

| Gate | Checks | Stop when |
|---|---|---|
| G0 preflight | 8 cards in `hl-smi`; the checkpoint config is block-FP8 [128, 128]; the FP8 layout check passes for each layout in the run; free disk; records the bundle commit, the image, `hl-smi -q` and the environment | any check fails |
| G1 build | stage 1's install from the bundle; `vllm.__version__` is `0.28.1rc1.dev392+g8f816a3f6`; `import vllm_gaudi`; `flashinfer_gaudi.get_capabilities()` | install or import fails, or the version is wrong |
| G2 tiny TP8 | stage 2a's tiny 8-head checkpoint at TP8: greedy agreement with `reference.json` | fewer than 16/16 tokens match on any prompt |
| G3 startup, per server | `Maximum concurrency for 131,072 tokens per request` ≥ agents on that server; `Usable num_blocks`, KV tokens, weights and warmup recorded | the concurrency is too low, or no healthy `/v1/models` within 1,800 s |
| G4 coherence (variant A's server) | the 8 short prompts of stage 2b; one 120K-token needle prompt | an empty answer, repetition, a JSON prompt that does not parse, or a missed needle |
| G5 load, per variant | the workload finishes in 5,400 s with no request error and `Running` reaching the agent count | any of these (then the variant's "on failure" rule applies: A stops the run, B and C continue) |

Notes:

- When `tpc-clang` (from `habanalabs-tools` 1.24.1) is missing in the container, the plugin is
  built without the fused GDN kernel (`VLLM_GAUDI_BUILD_GDN_FUSED=0`) and the script prints a
  warning: the run continues on the PyTorch GDN path. This is a warning, not a stop.
- The install logs are in the results folder as `install-<n>.log`; the per-phase records
  (`startup.json`, `requests.jsonl`, `metrics.jsonl`, `status.json`, `summary.*`) are all in the
  results folder for the run id.

## What to send back

`run_rental.sh`'s last line prints `SEND THIS FILE: <path>` — the one tarball
`<results>/<run-id>.tar.gz` with every log and record of the run, made even when the run failed.
Attach that single file to RUK-2709. Nothing else is needed from the node.
