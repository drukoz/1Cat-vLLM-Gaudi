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

   The bundle contains `run_rental.sh` and `prepare_host.sh` (top level), `plugin/` (this
   repository at its commit), `vllm/` (at the engine pin) and `BUNDLE.txt` (plugin sha, branch,
   vLLM commit, build date).

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

5. The host: a rented node ships bare Ubuntu 22.04. The first thing to run on it is
   `sudo bash prepare_host.sh` (from the unpacked bundle), which installs the driver, Docker, the
   Habana runtime and the image — see [The host (bare Ubuntu 22.04)](#the-host-bare-ubuntu-2204).
   Plan 15–30 min for it, once per node.

## The host (bare Ubuntu 22.04)

The rental ships bare: Ubuntu 22.04 and nothing else — no Gaudi driver, no Docker, no container
runtime. Before the run, `prepare_host.sh` (at the top of the unpacked bundle) installs the four
things `run_rental.sh` needs, pinned to the image's release **1.24.1, build 482**:

1. the Intel Gaudi driver stack (`habanalabs-installer.sh install --type base -y`),
2. Docker (`docker.io`),
3. the Habana container runtime, registered with Docker as runtime `habana`, and
4. the container image (pulled once, about 10 GB, so the run itself does not pay for it).

Run it once per node, as root, from the unpacked bundle (15–30 min, including the image pull):

```bash
tar xzf ~/flash-next-rental-*.tar.gz -C ~/rental && cd ~/rental
sudo bash prepare_host.sh
```

It is idempotent — rerun it freely. A reboot is needed only when its **step 4** (verify driver)
reports that `hl-smi` does not see all cards; reboot and run it again. Two remedies it detects and
prints:

- **Kernel headers.** On a custom cloud kernel, `linux-headers-$(uname -r)` is absent and the driver
  build cannot start; the script stops. Run `apt-get install -y linux-headers-generic
  linux-image-generic`, reboot into the new kernel, and rerun.
- **IOMMU on kernel 6.8.** On Ubuntu 22.04.5 (kernel 6.8) the driver needs `iommu=pt intel_iommu=on`
  in `GRUB_CMDLINE_LINUX_DEFAULT` (`/etc/default/grub`), then `update-grub` and a reboot. The script
  prints this instruction; it does not edit GRUB.

Useful options: `--version V` / `--build N` (the release pin, default `1.24.1-482`), `--image IMG`,
`--no-pull` (skip the image pull and the smoke test), `--reinstall-driver` (replace a driver whose
version differs from the pin — it never silently downgrades), `--force` (continue on a non-22.04
Ubuntu), and `--dry-run` (print every step's commands; needs no root, apt, docker, or network). When
the host is not ready, `run_rental.sh` stops and points back at this script.

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
| prepare_host.sh | 15–30 min (once per node) |
| preflight | about 1 min |
| install (vLLM, plugin with native kernels, model deps) | 20–40 min |
| fused GDN gate | about 15 min |
| tiny TP8 | about 10 min |
| each variant | about 35–45 min (startup up to 30 min including warmup, context fill, four turns) |

A full run (three variants, no `--download`) is about 3–4 h; add the download on top.

## What each gate checks and its stop rule

| Gate | Checks | Stop when |
|---|---|---|
| G0 preflight | 8 cards in `hl-smi`; the checkpoint config is block-FP8 [128, 128]; the FP8 layout check passes for each layout in the run; free disk; records the bundle commit, the image, `hl-smi -q` and the environment | any check fails |
| G1 build | stage 1's install from the bundle; `vllm.__version__` is `0.28.1rc1.dev392+g8f816a3f6`; `import vllm_gaudi`; `flashinfer_gaudi.get_capabilities()` | install or import fails, or the version is wrong |
| G1b fused GDN kernel | op numerics of both GDN decode ops on the card against the CPU reference (`test_hpu_op.py`), 1,000-step graph replay against eager (two cases), then the verification op's numerics and two replays (only after the decode checks pass), then the speed measurement | never stops: on a failure, a skip or a missing library, every later server runs with `VLLM_HPU_GDN_CORE_DECODE=0` (the PyTorch GDN path), recorded in the manifest and the summary; a verify failure leaves `VLLM_HPU_GDN_CORE_VERIFY=0`, only a pass sets it to 1 |
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
