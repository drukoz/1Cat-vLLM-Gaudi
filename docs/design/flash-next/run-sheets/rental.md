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

3. Disk on the node: the checkpoint is 185.5 GB and a `--download` needs **250 GB free** under
   `--model`. Root disks of rented nodes are often smaller (92 GB on 2026-11-18).
   `prepare_host.sh` prints the disks and the free space and prepares a data disk; `run_rental.sh`
   refuses a download that does not fit. See [Disks and the data disk](#disks-and-the-data-disk).

4. The checkpoint: if the node does not have it yet, add `--download` to the command below; the
   script downloads `Qwen/Qwen3.8-Flash-Next-FP8` (public, not gated) into the `--model` directory
   inside the container (`hf download`, about 185.5 GB — plan 1–2 h of rental time for it).

5. The host: a rented node ships bare Ubuntu 22.04. The first thing to run on it is
   `sudo bash prepare_host.sh` (from the unpacked bundle), which installs the driver, Docker, the
   Habana runtime and the image — see [The host (bare Ubuntu 22.04)](#the-host-bare-ubuntu-2204).
   Plan 15–30 min for it, once per node.

**Known failure, fixed in the script on 2026-11-19 (apt repository "is not signed", `NO_PUBKEY 1E676CE836A2DE65` in `step-3.log`).** Intel's installer writes its apt repository and runs `apt update` before it imports the signing key, so the update fails on the installer's own repository and the key is never imported. `prepare_host.sh` now imports the key into `/etc/apt/trusted.gpg.d/habanalabs.gpg` before step 2's `apt-get update` and before the installer. On a node that already hit the failure with an older bundle, import the key by hand and rerun:

```bash
curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/habanalabs.gpg >/dev/null
sudo apt-get update
cd ~/rental && sudo bash prepare_host.sh
```

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
mkdir -p ~/rental && tar xzf ~/flash-next-rental-*.tar.gz -C ~/rental && cd ~/rental
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

### Disks and the data disk

On 2026-11-18 `/models` was a folder on the 92 GB root disk, and the 185.5 GB download filled it
(97%) and stopped; the node had a second, blank 26.4 TB disk (`/dev/vdc`) that nothing used,
because nothing showed it to us and nothing prepared it. Every `prepare_host.sh` run now prints
`lsblk`, the free space (`df -h`) and the unused disks, so a data disk is visible before the
download starts. To use one:

```bash
sudo bash prepare_host.sh --data-disk /dev/vdc
```

Use the device name the list printed; `--mount DIR` puts the filesystem somewhere other than
`/models`. Try it first with `--dry-run` (it runs the checks and prints what the real run would
do). The script formats a device only with `--data-disk`, and only when `blkid` and `wipefs` both
see it as blank, with no partitions and not mounted: any signature, any partition, a mount point,
an unreadable device or a non-empty mount folder is a refusal (exit 1, a `REFUSED` line). It never
formats without the option. The `/etc/fstab` line uses the device UUID with `nofail`, so a missing
disk does not stop the boot, and the rerun after a reboot finds the disk already prepared: the
script's own `fn-models` ext4 is recognized and `mkfs` is skipped.

### Fallback: the checkpoint in RAM (no data disk)

Only for a node without a usable data disk and with plenty of memory (2026-11-18: 925 GB RAM).
Conditions: `df -h /dev/shm` shows at least 250 GB free (tmpfs is half the RAM by default) and
`free -g` shows at least 400 GB available.

```bash
sudo mkdir -p /dev/shm/models /models
sudo mount --bind /dev/shm/models /models
sudo bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8 --download
```

It is lost on reboot: after a reboot, redo the bind mount and the download. It worked on
2026-11-18 (gate `preflight` passed).

## First on the node: the tiny smoke test

Before the checkpoint download and before `run_rental.sh`, check that a Flash-Next server starts and
decodes on this node at all. It needs no checkpoint:

```bash
tar xzf ~/flash-next-rental-*.tar.gz -C ~/rental && cd ~/rental
sudo bash tiny_smoke.sh
```

What it does: starts the container `fn-smoke`, installs the stack from the bundle in it
(`container_entry.sh --install-only`), builds the tiny checkpoint and its CPU reference, then serves
it twice with the environment of a rental variant (lazy mode, sparse QSA, the recurrent state pool)
and compares the served tokens with the reference each time:

| Mode | GDN flags | A pass means |
|---|---|---|
| `pytorch` | `VLLM_HPU_GDN_CORE_DECODE=0`, `VLLM_HPU_GDN_CORE_VERIFY=0` | the server builds, warms up and decodes on the torch GDN path |
| `fused` | `VLLM_HPU_GDN_CORE_DECODE=1`, `VLLM_HPU_GDN_CORE_VERIFY=0` | the same with the fused decode kernel, and no layer fell back to the torch path |

It prints one line per mode (`tiny smoke [pytorch]: PASSED ...`) and, on a failure, the error lines of
the server log and the path of that log. Results are in `results/smoke-<UTC>/` (`summary.json`,
`<mode>/server0.log`, `console.log`). The exit code is 0 only when every mode passed.

Reading the result:

- Both pass: go on to the one command below.
- `pytorch` fails: no variant can start. Fix it here first; `run_rental.sh` would stop at gate `tiny`
  for the same reason, an hour of download later.
- `pytorch` passes and `fused` fails: the run would work with the kernel off. Gate G1b decides that
  for the run; read the `fused` log before spending the run on it.

The container stays, with the stack installed, every server stopped and the cards free. A second call
reuses it and skips the install and the checkpoint build, so the loop "change, check" takes a few
minutes instead of a full run:

| Option | Meaning |
|---|---|
| `--reinstall` | install the stack again in the same container; needed after any change under `plugin/` (the servers run the installed plugin, not the source tree) |
| `--fresh` | remove the container first: new install, new tiny checkpoint |
| `--gdn LIST` | the modes to serve, in order (default `pytorch,fused`) |
| `--results DIR`, `--run-id ID`, `--image IMG`, `--dry-run` | as for `run_rental.sh`; `--results` is fixed once the container exists |

It refuses to start while the rental run's container (`fn-rental`) is running, because that run holds
the cards. `Ctrl-C` stops the check inside the container too, so the tiny server does not keep them.
`sudo docker exec -it fn-smoke bash` opens a shell in the container; `sudo docker rm -f fn-smoke`
removes it when you are done.

`tiny_smoke.sh` first ran on a card on 2026-11-19 (rental run 2): the install, the checkpoint build and
the server start worked as designed, and it caught two prompt-warmup failures that no Mac test could see
(the MoE fast path's engine drift, `ca676dd`; `torch.cummax` without an HPU kernel, `b89bdd7`). Both were
applied to the running container one file at a time, as below, and rerun in minutes.

### Updating the bundle on a prepared node

When the branch changes after the node was prepared, copy the new tarball and extract it over the same
folder:

```bash
# workstation
scp -i <key> dist/flash-next-rental-<sha>.tar.gz <node>:~/
# node
tar xzf ~/flash-next-rental-<sha>.tar.gz -C ~/rental && cat ~/rental/BUNDLE.txt
```

The tarball holds only `BUNDLE.txt`, the three scripts, `plugin/` and `vllm/`, so `results/`, the
containers and the data disk are untouched, and `BUNDLE.txt` names the plugin commit the run's manifest
records. `run_rental.sh` installs from the bundle folder into its own container (`fn-rental`), so it always
runs the replaced `plugin/`. The smoke container keeps its installed stack: run
`sudo bash tiny_smoke.sh --reinstall` when the change is on the smoke test's path, or patch one file into it
without a reinstall (`/bundle` is the bind-mounted bundle folder, so the copied file is the one the bundle
now carries):

```bash
sudo docker exec fn-smoke bash -c 'cd / && d=$(python -c "import importlib.util as u; print(u.find_spec(\"vllm_gaudi\").submodule_search_locations[0])") && cp /bundle/plugin/vllm_gaudi/<path>.py "$d/<path>.py" && echo "patched $d/<path>.py"'
```

## The one command (on the node)

```bash
tar xzf ~/flash-next-rental-*.tar.gz -C ~/rental && cd ~/rental
sudo bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8
```

That is all. Three things about the host it runs on:

- **Root.** `prepare_host.sh` does not add your user to the `docker` group, so run it with `sudo`
  (or `sudo usermod -aG docker $USER`, then log in again). Without either, Docker refuses the socket
  and the script stops with a message about the `habana` runtime. `--download` also creates
  `/models`, which needs root.
- **A terminal that survives the SSH session.** The run takes hours; run it inside `tmux`
  (`tmux new -s rental`; detach with `Ctrl-b d`, reattach with `tmux attach -t rental`). A dropped
  SSH session otherwise kills the run, and the tarball at the end is never written.
- **Disk.** `run_rental.sh` checks the disks itself. With `--download` it stops when the folder for
  `--model` has less than 250 GB free (what is already downloaded there counts), and every run stops
  when Docker's root (`/var/lib/docker`) has less than 30 GB free. Each stop is one line naming the
  folder, its free space and the fix. On a node whose root disk is small, prepare the data disk
  first: see [Disks and the data disk](#disks-and-the-data-disk).

Useful options:

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
| `--no-disk-check` | skip both free-space checks |

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
| G1b fused GDN kernel | op numerics of both GDN decode ops on the card against the CPU reference (`test_hpu_op.py`), 1,000-step graph replay against eager (two cases), then the verification op's numerics and two replays (only after the decode checks pass), then the speed measurement; all of these run with `PT_HPU_LAZY_MODE=1`, the servers' bridge mode, and only they decide the kernel. The two decode replays then run once more with `PT_HPU_LAZY_MODE=0` for `torch.compile` and are recorded only (`replay_eager`, `eager_compile_ok`): an eager failure never turns the kernel off. Every step runs in its own process group and is killed with its whole group at its timeout (op tests 600 s, each replay 900 s, bench 1200 s). When `test_hpu_verify_op.py` crashes or times out, G1b runs `gdn_verify_crash_probe.sh` (at most 4500 s) into `fused_gdn/verify_crash/`; recorded only | never stops: on a failure, a skip or a missing library, every later server runs with `VLLM_HPU_GDN_CORE_DECODE=0` (the PyTorch GDN path), recorded in the manifest and the summary; a verify failure leaves `VLLM_HPU_GDN_CORE_VERIFY=0`, only a pass sets it to 1 |
| G2 tiny TP8 | stage 2a's tiny 8-head checkpoint at TP8: greedy agreement with `reference.json` | fewer than 16/16 tokens match on any prompt |
| card_tests (after the gates) | the two HPU-only unit-test files on the card, in lazy mode (`PT_HPU_LAZY_MODE=1`), 600 s each: `tests/unit_tests/test_defragmentation.py` and `tests/unit_tests/ops/test_hpu_fused_moe.py`; each file's output is kept as `<stem>.log` and the pytest counts are recorded in `details.json` | never stops the run: a failure, a timeout or a skip is recorded only |
| G3 startup, per server | `Maximum concurrency for 131,072 tokens per request` ≥ agents on that server; `Usable num_blocks`, KV tokens, weights and warmup recorded | the concurrency is too low, or no healthy `/v1/models` within 1,800 s |
| G4 coherence (variant A's server) | the 8 short prompts of stage 2b; one 120K-token needle prompt | an empty answer, repetition, a JSON prompt that does not parse, or a missed needle |
| G5 load, per variant | the workload finishes in 5,400 s with no request error and `Running` reaching the agent count | any of these (then the variant's "on failure" rule applies: A stops the run, B and C continue) |

Notes:

- The install adds `habanalabs-tools` 1.24.1-482 (tpc-clang) to the container when the image
  lacks it, so the fused GDN decode and verification kernels and the flashinfer kernels are
  compiled from the bundle (`install-0-toolchain.log`, `toolchain.json`). Only if that install
  fails does the run continue without native kernels on the PyTorch GDN path, with
  `native_build: 0` in `toolchain.json`, the G1b gate recording the missing library, and a
  warning in the console.
- The install logs are in the results folder as `install-<n>.log`; the per-phase records
  (`startup.json`, `toolchain.json`, `requests.jsonl`, `metrics.jsonl`, `status.json`,
  `summary.*`) are all in the results folder for the run id.
- Every server runs lazy (`PT_HPU_LAZY_MODE=1`). The native extensions link only the Habana
  backend library so one build loads in both bridge modes; the checks that prove it on the card
  are in [native-extensions.md](../native-extensions.md#run-sheet-lines-for-the-next-rental).

## If the GDN verify op crashes (RUK-3105)

Run the crash probe when G1b's verify step ends by a signal (a segfault) or a timeout.
R1-d (RUK-3104) will make G1b run the probe on its own, into
`<results>/<run-id>/fused_gdn/verify_crash/`; until then, run it by hand while a run is going:

```bash
docker exec fn-rental bash /bundle/plugin/tools/flash_next/gdn_verify_crash_probe.sh /results/verify-crash
```

(`fn-rental` is the container `run_rental.sh` starts.) The probe runs six short checks, each in
its own process under `gdb -batch`, bounded by `PROBE_TIMEOUT_S` (default 300 s): the decode
control, the verify op at T = 1 and T = 4 (the crashing `ver_tp8_chain` case) in lazy mode,
T = 4 in eager mode, and the T = 1 / T = 4 checks again against a verify kernel rebuilt with its
`.llvmbc` section stripped (`-DGDN_FUSED_VERIFY_STRIP_BITCODE=ON`).

Send `summary.txt`, the six logs and the `*.habana_logs` folders back with the results.

How to read `summary.txt`:

| Result | Points to |
|---|---|
| `verify_t1_lazy` crashes too | not the T = 4 geometry: the kernel ELF (H1 or H2) |
| the strip runs pass where the shipped ones crash | H2: ship the stripped ELF |
| the backtrace names libSynapse or the graph compiler, and the strip runs crash | H1 or H3: the kernel must fit the default 16 KB VLM |

## Checks carried from rental run 1 (2026-11-18)

- `PT_HPU_LAZY_MODE=1`: gate G2's tiny TP8 server reaches `/health` (run 1 failed while building
  the model: the PLE hash multipliers were an int64 device buffer; fixed in RUK-3101).
- On the card, in lazy mode, before G2:
  `python -m pytest -q tests/unit_tests/models/flash_next/test_ple_ngram_hash.py` runs its `hpu`
  case (not skipped) and passes; if it fails, the int32 hash is not exact on the card: stop and
  send the log.

## What to send back

Everything the run writes is under `<results>/<run-id>/` on the host (default `~/rental/results/`),
and `run_rental.sh` ends with two lines:

- `PASTE THIS FILE: <results>/<run-id>/summary.md` — the result, as text. It has the gates table
  (status and reason per gate), the variants table (max concurrency, TTFT cold and warm, decode
  tok/s per agent, aggregate tok/s, prefix-cache hit rate, preemptions, MTP acceptance length,
  errors, wall time), one section per variant, the GDN decode line (fused kernel or PyTorch path)
  and the native-kernel line from `toolchain.json`. Paste it whole into a comment on RUK-2709. It
  is written at the end of every run that reached the orchestrator, passed or failed.
- `SEND THIS FILE: <results>/<run-id>.tar.gz` — the whole run folder, made even when the run
  failed: `console.log` (everything the run printed, kept across `--from` resumes),
  `manifest.json`, `toolchain.json`, the `install-<n>.log` files, each gate's and variant's
  `status.json` and `details.json`, `startup.json`, `server<k>.log`, `requests.jsonl`,
  `metrics.jsonl`, `hlsmi.csv`, `workload.json` and `summary.json`. Copy it to the Mac at
  `/Volumes/Storage/paperclip/tmp/rental-results/` (scp from the node to your workstation, then to
  the Mac). The deeper readings of the plan's M5 table (the drafter-rank log, the profiler trace, a
  failing server's log) are read from there.

If the run stopped early (an install step failed, or a gate tripped), also paste the last 40 lines
of `<results>/<run-id>/console.log`: they name the phase that stopped and the reason, and an install
failure prints the last 30 lines of its `install-<n>.log` there. A failure before the orchestrator
(the toolchain or install steps) leaves no `summary.md`; the console log and the tarball are then
the whole result.
