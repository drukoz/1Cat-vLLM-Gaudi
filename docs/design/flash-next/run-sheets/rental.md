# Flash-Next rental run sheet (one command)

This is the sheet you run in the rental session. Everything is automated: you build one bundle on
the Mac, copy it to the node, and run **one command**. The script starts the container, installs
the stack, runs every gate, and leaves **one results tarball** to send back. You do not need to
read [qualification.md](qualification.md) — it stays as background for the manual stages.

## What the run is

The bundle carries the plugin (this repository, at a clean commit) and vLLM at the engine pin
`8f816a3f665489d7f0d222115d4f72ebab01076b`. On the node, `run_rental.sh` starts the Habana
container, `container_entry.sh` installs the stack inside it, and the orchestrator
(`python -m tools.flash_next.rental.main`) runs these phases, in order: `preflight`, `build`,
`fused_gdn` (the fused GDN kernel gate), `tiny` (tiny TP8), `card_tests`, then one phase per variant
(`A`, `B`, `C`; `P` with `--with-profile`), then `report`. Each gate has a stop rule (table below);
the run stops at the first gate that trips, and you send the tarball back.

The order of a whole session, with the cheap checks first, is in
[Cheapest session order](#cheapest-session-order).

## Cheapest session order

The node bills by the hour from the first minute, so the order is: cheap checks first, the long run
last, nothing waiting on a download. Every command runs from the unpacked bundle (`cd ~/rental`).

1. **Prepare the host** (15–30 min, once per node): `sudo bash prepare_host.sh`, with
   `--data-disk DEV` when the root disk is small.
2. **Start the checkpoint download at once, detached** (about 7–10 min at the measured 0.3 GB/s or
   more):

   ```bash
   sudo bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8 --download-only
   ```

   It returns at once. Progress: `sudo docker logs --tail 3 fn-download` or
   `du -sh /models/Qwen3.8-Flash-Next-FP8`. Finished: `sudo docker wait fn-download` prints `0`.
   `--download-only` is new in this bundle and has not run on a node yet; if it fails, add
   `--download` to the run in step 4, which downloads inside the run container as on 2026-11-18.
3. **In parallel, the tiny smoke test, one mode per call** (each call, and each retry, about 4–6
   minutes; the first one also installs the stack, about 2 minutes, and builds the tiny checkpoint):

   ```bash
   sudo bash tiny_smoke.sh --gdn pytorch
   sudo bash tiny_smoke.sh --gdn fused
   ```

   Fix what fails here before going on; a retry costs minutes, a failed run costs its install and
   gates. After a change under `plugin/` add `--reinstall`.
4. **The run, inside `tmux`**, once both modes pass and the download is complete:

   ```bash
   tmux new -s rental
   sudo bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8 --quick-gates
   ```

   `--quick-gates` leaves out the recorded-only steps in front of the first measurement. Watch for
   the first proof that the model decodes: one console line per server of a variant,
   `variant A server0: first token after <s> s: '<text>'`, and the file
   `<results>/<run-id>/A/first-token.json` (one short completion per server, sent before the warm
   pass and the load). A line `no first token after ...` fails the variant; for A that stops the run.
5. **The compile probe only after variant A has numbers** (`A` passed in `summary.md`), and when
   the run has ended or been stopped: the probe needs cards the run holds. See
   [Where the compile time goes](#where-the-compile-time-goes-compile_probepy).
6. **Copy the results off the node, then stop the node.** The run's tarball is
   `<results>/<run-id>.tar.gz`; the smoke test writes none, so pack its folders:
   `cd ~/rental/results && tar czf ~/smoke-results.tar.gz smoke-*`. Stop (release) the node as soon
   as the files are on your workstation: an idle node costs the same as a busy one.

If a server fails at start with the combined kernel database, rerun with `--no-combined-kernel-db`
(`tiny_smoke.sh` and `run_rental.sh` both take it; the orchestrator also retries by itself at the
kernel gate G1b and at the tiny gate).

## Before the rental (on the Mac, any time)

1. Make sure the checkout is clean (`git status --porcelain` empty) at the commit you want to test,
   then build the bundle:

   ```bash
   bash tools/flash_next/rental/make_rental_bundle.sh --out dist
   # prints: bundle: dist/flash-next-rental-<sha>.tar.gz   (about 56 MB)
   ```

   The bundle contains `run_rental.sh`, `prepare_host.sh` and `tiny_smoke.sh` (top level),
   `plugin/` (this repository at its commit), `vllm/` (at the engine pin) and `BUNDLE.txt` (its
   first line is `plugin sha: <commit>`; then branch, vLLM commit, build date). Build it from a
   clean tree: with `--allow-dirty` the three top-level scripts are the working tree's while
   `plugin/` is the commit, and the script warns when they differ.

2. Copy the tarball to the node (on the node, "scp" from your workstation is fine):

   ```bash
   scp dist/flash-next-rental-*.tar.gz <node>:~/
   ```

3. Disk on the node: the checkpoint is 185.5 GB and a `--download` needs **250 GB free** under
   `--model`. Root disks of rented nodes are often smaller (92 GB on 2026-11-18).
   `prepare_host.sh` prints the disks and the free space and prepares a data disk; `run_rental.sh`
   refuses a download that does not fit. See [Disks and the data disk](#disks-and-the-data-disk).

4. The checkpoint: if the node does not have it yet, start the download first and on its own with
   `run_rental.sh --download-only` (see [Cheapest session order](#cheapest-session-order)), or add
   `--download` to the one command below. Both download `Qwen/Qwen3.8-Flash-Next-FP8` (public, not
   gated) into the `--model` directory with `hf download`, about 185.5 GB. On 2026-11-18 the disk
   filled at 0.3 GB/s or more, so plan about 7–10 min for it, not hours.

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

Before `run_rental.sh`, check that a Flash-Next server starts and decodes on this node at all. It
needs no checkpoint, so it runs while the checkpoint downloads (`run_rental.sh --download-only`):

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
| `pytorch` | `VLLM_HPU_GDN_CORE_DECODE=0`, `VLLM_HPU_GDN_CORE_VERIFY=0` | the server builds and decodes on the torch GDN path |
| `fused` | `VLLM_HPU_GDN_CORE_DECODE=1`, `VLLM_HPU_GDN_CORE_VERIFY=0` | the same with the fused decode kernel, and no layer fell back to the torch path |

The tiny checkpoint's GDN layers are built with head dim 128 (`make_tiny_checkpoint.py
--gdn-head-dim 128`), the head dim of the real model and the only one the fused kernel accepts, so
the `fused` mode really runs the fused kernel (with the fixture's own head dim 16 every layer fell
back to the torch path). The reference holds six prompts of 1, 7, 33, 100, 700 and 2,600 tokens, and
the tiny server's `--max-model-len` is sized to fit the longest one (4,096). A checkpoint that an
older bundle left in the container is rebuilt on the next call: its `reference.json` records the
settings it was built with.

By default the servers skip the bucket warmup (`VLLM_SKIP_WARMUP=true`) and the health wait is 1800 s:
the warmup alone takes about 15 minutes per mode on the tiny model, while the six prompts need only a
handful of graphs, which compile on first use. A run then takes about 4–6 minutes per mode. The
servers share a compiled-recipe cache under `results/recipe-cache/<plugin commit, 12 characters>`,
reused on every start, so a second run is faster. The `pytorch` mode's server always gets it. The
`fused` mode's server gets it only with the combined kernel database (the install builds it): without
that database, or with `--no-combined-kernel-db`, its kernel database is registered in a colon-list
`GC_KERNEL_PATH`, and the Habana 1.24.1 bridge accepts only one kernel library there next to a
recipe cache.

If a server fails at start with the combined kernel database, run the mode again with
`--no-combined-kernel-db` (`run_rental.sh` takes the same option; the orchestrator also retries by
itself, see [The one command](#the-one-command-on-the-node)).

It prints one line per mode (`tiny smoke [pytorch]: PASSED ...`) and, on a failure, the error lines of
the server log and the path of that log. Results are in `results/smoke-<UTC>/` (`summary.json`,
`<mode>/server0.log`, `console.log`, and `habana_logs/` with the Habana runtime logs of the check
and its servers). The exit code is 0 only when every mode passed.

Reading the result:

- Both pass: go on to the one command below.
- `pytorch` fails: no variant can start. Fix it here first; `run_rental.sh` would stop at gate `tiny`
  for the same reason, after its install and the gates before `tiny`.
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
| `--full-warmup` | run the real bucket warmup (about 15 minutes per mode); the default skips it, see above |
| `--health-timeout S` | seconds to wait for a server to get healthy, passed to the check (default 1800) |
| `--env KEY=VALUE` | extra server environment variable, repeatable; applied after the mode's own flags, so it can override them (a `KEY` without `=` is a usage error) |
| `--no-recipe-cache` | give no server the compiled-recipe cache under `results/recipe-cache` |
| `--no-combined-kernel-db` | keep the combined kernel database out of the smoke servers (`FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0` for the check and its servers). The check then treats the database as not installed: the `fused` mode's server runs the colon-list `GC_KERNEL_PATH` and gets no recipe cache; the `pytorch` mode keeps its cache |
| `--results DIR`, `--run-id ID`, `--image IMG`, `--dry-run` | as for `run_rental.sh`; `--results` is fixed once the container exists |

It refuses to start while the rental run's container (`fn-rental`) is running, because that run holds
the cards. `Ctrl-C`, `SIGTERM` and a dropped SSH session (`SIGHUP`) stop the check inside the
container too, so the tiny server does not keep the cards: the check gets `SIGINT` and up to 90 s to
stop its server; if it is still there after that, the script stops the whole container (the next
call then starts a new one and installs again). The exit code is 130, 143 or 129.
`sudo docker exec -it fn-smoke bash` opens a shell in the container; `sudo docker rm -f fn-smoke`
removes it when you are done.

The container is started with `--init`, so an init process reaps the workers a crashed server leaves
behind (without it the check waited its full 65 s stop time for them), and with
`PYTHONFAULTHANDLER=1`. A container that an older bundle created has neither: pass `--fresh` once
after updating to this bundle.

`tiny_smoke.sh` first ran on a card on 2026-11-19 (rental run 2): the install, the checkpoint build and
the server start worked as designed, and it caught two prompt-warmup failures that no Mac test could see
(the MoE fast path's engine drift, `ca676dd`; `torch.cummax` without an HPU kernel, `b89bdd7`). Both were
applied to the running container one file at a time, as below, and rerun in minutes.

### Where the compile time goes (compile_probe.py)

The tiny model's prompt graphs take 13–35 s to compile on the card, and nothing on the Mac can time
the graph compiler. The probe times each Flash-Next component in isolation on the card: it captures
one eager prefill's inputs per component (the whole model, every decoder layer, the GDN layer, the
QSA layer and its indexer `select`, the PLE layer and its n-gram embedding, a HyperConnection, the
MoE block) and times the first (compiling) and second (replaying) call of each, wrapped in an HPU
graph, at 128, 512 and 2048 tokens. With the `fn-smoke` container prepared by `tiny_smoke.sh`:

```bash
sudo docker exec -w /bundle/plugin -e PT_HPU_LAZY_MODE=1 fn-smoke python -m tools.flash_next.compile_probe \
    --ckpt /tmp/fn-smoke-ckpt --device hpu --out /results/probe-$(date -u +%Y%m%dT%H%M%SZ)
```

The run takes about 15 minutes; the probe refuses to start without `-e PT_HPU_LAZY_MODE=1`. Paste
back the printed markdown table (one row per component and shape: compile and replay seconds,
sorted by compile time); `probe.json` in the `--out` folder carries the same data if the table is
cut off.

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
runs the replaced `plugin/`.

`plugin/build/` (the build folder of the plugin install, native kernels included) is not part of the
tarball and survives the extraction. Its rebuilds go by file time, and the files of a bundle carry
their commit's time, so objects built from the older commit could pass for current.
`container_entry.sh` therefore records the `plugin sha:` of `BUNDLE.txt` in
`plugin/build/.bundle-sha` after a plugin install, and removes the folder before the next install
whenever `BUNDLE.txt` names another commit (the console shows a `phase: stale build ...` line). The
first install after a bundle update builds the native kernels from scratch (48 s on 2026-11-18);
later starts of the same bundle reuse the build (16–17 s). Extraction also keeps files that the new
commit deleted; for an exact tree remove the two folders first:
`sudo rm -rf ~/rental/plugin ~/rental/vllm`, then extract.

The smoke container keeps its installed stack: run
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
  (`tmux new -s rental`; detach with `Ctrl-b d`, reattach with `tmux attach -t rental`). Outside
  `tmux` a dropped SSH session ends the run. `Ctrl-C`, `SIGTERM` and that dropped session
  (`SIGHUP`) are handled the same way: the script asks the orchestrator to stop (it stops its
  servers and writes the report; up to 90 s), then uses `docker stop` and `docker kill` if the
  container is still there, so no server keeps the cards; it still writes the tarball and the two
  closing lines and exits with 130, 143 or 129. Resume with `--run-id <id> --from <phase>`.
- **Disk.** `run_rental.sh` checks the disks itself. With `--download` or `--download-only` it stops
  when the folder for `--model` has less than 250 GB free (what is already downloaded there counts), and every run stops
  when Docker's root (`/var/lib/docker`) has less than 30 GB free. Each stop is one line naming the
  folder, its free space and the fix. On a node whose root disk is small, prepare the data disk
  first: see [Disks and the data disk](#disks-and-the-data-disk).

Useful options:

| Option | Meaning |
|---|---|
| `--model DIR` | checkpoint directory (required); mounted at `/models/<name>` in the container |
| `--download` | download the checkpoint into `--model` first (needs `hf`, installed by the run) |
| `--download-only` | only start the download into `--model`, in a detached container of its own (`fn-download`: no card, nothing installed from the bundle), and return; the 250 GB free-space check applies. `sudo docker wait fn-download` prints 0 when the checkpoint is complete; `run_rental.sh` refuses to start a run while it is running |
| `--results DIR` | results folder on the host (default `./results`) |
| `--variants A,B,C` | run only these variants (default: all in the run) |
| `--with-profile` | enable profiling |
| `--from PHASE` | resume the run `--run-id ID` from a phase: `preflight`, `build`, `fused_gdn`, `tiny`, `card_tests`, a variant id of the run (`A`, `B`, `C`; `P` with `--with-profile`), or `report`. `--run-id` is required with it, and that run folder must hold the `manifest.json` of an earlier start. A missing `--run-id`, an unknown run id or a misspelt phase stops the script at once (exit 2, before the container starts) with the run ids found under `--results` or the list of phases |
| `--run-id ID` | fixed run id (default: UTC timestamp) |
| `--quick-gates` | skip the recorded-only steps before the first measurement: G1b's eager replays and its bench, the verify crash probe, and the `card_tests` phase (recorded as skipped). The checks that decide the GDN kernel and the verify op still run |
| `--no-verify-probe` | skip only the automatic crash probe that follows a crashed or hung GDN verify op test |
| `--load-read-timeout S` | measured load: seconds a request may go without receiving data (default 1800) |
| `--warm-request-timeout S` | warm pass: seconds a request may go without receiving data, first graph compiles included (default 3600) |
| `--image IMG` | container image (default the 1.24.1 base image) |
| `--full-warmup` | run the full bucket warmup at server start (default: skipped — see [Warmup and the recipe cache](#warmup-and-the-recipe-cache-ruk-3200)) |
| `--server-timeout S` | seconds to wait for a variant's server to become healthy (default 1800, enough with the default skipped warmup; raise it with `--full-warmup`, see below). The tiny gate does not use it: its health wait is fixed at 1,200 s |
| `--recipe-cache DIR` | host folder for the HPU recipe cache, mounted at `/recipe-cache` and handed to the orchestrator with `--recipe-cache` (default: a `recipe-cache` folder next to `--model`) |
| `--no-recipe-cache` | no recipe cache: no `/recipe-cache` mount and no `--recipe-cache` for the orchestrator |
| `--no-combined-kernel-db` | keep the combined kernel database out of the servers (`FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0` in the container). The orchestrator then treats the database as not installed: servers that run the fused GDN kernel use the colon-list `GC_KERNEL_PATH` and get no recipe cache; servers on the PyTorch GDN path keep the cache |
| `--dry-run` | print the `docker run` command and exit |
| `--no-disk-check` | skip both free-space checks |
| `--no-warm-pass` | skip the warm pass (a one-turn pass that compiles the variant's graphs and caches the shared block before the coherence gate; on by default — see [Warm pass](#warm-pass-ruk-3203)) |

When the run stops at a stop rule, restart after fixing the cause with `--run-id <same id>
--from <phase>` to keep the results in one folder.

**If a server fails at start with the combined kernel database** (the adapter that puts Intel's
kernels and ours into one `GC_KERNEL_PATH` file), rerun with `--no-combined-kernel-db`. The
orchestrator also retries by itself at two gates: at G1b, when the decode checks fail with the
database and pass without it, and at the tiny gate, when the tiny server fails or times out with it
(the first attempt is kept under `tiny/attempt-combined/` and the gate runs once more without it).
Either way the database then stays out of every later server of the run, and the manifest records
who took it out (`combined_kernel_db_disabled_by`: `operator`, `g1b` or `tiny`).

### The container

`run_rental.sh` starts the container `fn-rental` with `--rm`, `--init` (an init process reaps the
workers a crashed server leaves behind; on 2026-11-18 every crashed server cost the orchestrator its
full 65 s stop wait without it), `PYTHONFAULTHANDLER=1` (every Python process writes its stack on a
crash) and `HABANA_LOGS=/results/<run-id>/habana_logs`, so the Habana runtime logs land in the run
folder instead of vanishing with the container. Every `phase:` line that `container_entry.sh`
prints ends with its UTC time (`phase: install 2 (plugin (vllm-gaudi)) [2026-11-19T22:31:05Z]`), so
`console.log` can be timed.

## Warmup and the recipe cache (RUK-3200)

Bucket warmup is **off by default**: every variant server starts with `VLLM_SKIP_WARMUP=true`,
and so does the tiny gate's checkpoint build. `--full-warmup` starts the servers as before, and
`summary.md` then prints `Warmup: full` at the top (with `Server health timeout: <S> s` next to
it; the manifest records both).

Skipping warmup means the graphs are not compiled at server start — they compile **on first
use**, inside the first requests. Two consequences:

- **Startup is minutes, not tens of minutes.** With the warmup skipped a start is weight loading,
  the cache setup and the sampler warmup, with no graph compiled, so the default 1800 s health
  wait holds; `--server-timeout` matters with `--full-warmup` (below). The coherence gate allows
  3,600 s per request when the warmup is skipped, so a first request that triggers compilation
  is not cut off.
- **The recipe cache keeps the compiled graphs.** `run_rental.sh` mounts the cache at
  `/recipe-cache` and hands it to the orchestrator with `--recipe-cache /recipe-cache`; the
  orchestrator then sets `PT_HPU_RECIPE_CACHE_CONFIG` **per server** (RUK-3242). Only servers
  that run the PyTorch GDN path get the variable: a server that runs the fused GDN kernel
  registers its native kernel database in `GC_KERNEL_PATH`, which the Habana 1.24.1 bridge
  accepts only when no recipe cache is set. With the combined kernel database in use (installed,
  not switched off by `--no-combined-kernel-db`, G1b passed with the fused kernel, and neither
  G1b nor the tiny gate took it out), every server gets the cache. The folder gets a per-bundle
  subfolder `<dir>/<plugin commit, first 12 characters>`. The commit is the `plugin sha:` line of
  `/bundle/BUNDLE.txt` (`unknown` only when there is none), read from the bundle that runs now: a
  `--from` resume after a bundle update uses the new bundle's subfolder, not the one the run began
  with. `summary.md` records the policy on its `Recipe cache:`
  line. The default folder is `recipe-cache` **next to the checkpoint**, so it lives on the
  checkpoint's disk: on the prepared data disk it survives reboots and container rebuilds, and
  the second run on the same checkpoint and node starts with the graphs already compiled. Keep
  it on the data disk, never under Docker's root (which has its own 30 GB free-space check).
  `--recipe-cache DIR` picks another folder; `--no-recipe-cache` drops the mount and the
  orchestrator option.

The TPC toolchain is cached on the data disk the same way (RUK-3239): the first run on a node
installs `habanalabs-tools` with `apt-get update` (slow on the rental node) and keeps the
downloaded debs in an `apt-cache` folder next to the checkpoint; later runs install from there
with no `apt-get update` at all. Delete that folder to force a fresh download.

Graph-count guidance for the report-back: the first server start on a cold cache compiles every
shape the run touches, and the console shows the graph builders working through the buckets;
the same server with a warm cache reports far fewer compiled graphs. With `--full-warmup` the
servers compile the buckets up front — 249 graphs for variant A (TP8, 32 agents), 397 for
variant B (A + MTP k=3) and 237 per server for variant C (TP4, 16 agents); at the observed
13–35 s per prompt graph that is hours per server, so run `--full-warmup` with a
`--server-timeout` of several hours (e.g. 14400–21600). The G3 line records the
startup seconds per server (`startup.json`), so a startup that is "only" graph compilation is
visible there. A **startup of tens of minutes with a cold recipe cache is not a hang** — do not
stop the run while the graphs compile. For a variant's server the health timeout
(`--server-timeout`) is the line, and `hl-smi` showing TPC activity is the proof of life. The tiny
gate does not use `--server-timeout`: its server gets a fixed 1,200 s to become healthy (the smoke
test's own wait is `--health-timeout`, default 1800 s).

## Warm pass (RUK-3203)

Bucket warmup is skipped by default, so graphs compile **on first use**, inside the first
requests. The warm pass puts that first use before the measurement instead of inside it: after a
variant's servers pass G3, the orchestrator runs **a one-turn copy of the workload** — every
agent, the same shared system block, each agent's files shifted by one (`file_offset=1`, so no
measured prompt is pre-cached past the shared block), `max_tokens=64` — on that variant's own
servers, and only then runs the coherence gate (variant A) and the measured load.

- **What it buys:** the shapes the workload touches are compiled once (the recipe cache keeps
  them), and the shared system block is in the prefix cache before G4 and G5. Cold TTFT then
  measures the first turn of the real load, not graph compilation.
- **Cost:** 5–35 min per variant (32 prompts of about 120K tokens plus the graph compiles; the
  low end with a warm recipe cache). On a cold cache the variant's graph compilation happens
  inside its warm pass — minutes, not the hours of `--full-warmup`, because only the shapes the
  workload touches are compiled.
- **Files:** `<variant>/warm-pass.json` (requests, seconds, non-OK), `<variant>/warm-requests.jsonl`
  (the rows), `warm-workload.json` at the run level, and `warm_pass: true` in the manifest.
  `summary.md` prints one `Warm pass: ... requests in ... s, ... non-OK` line per variant
  (`Warm pass: none` when it is switched off).
- **Stop rule:** a non-OK warm-pass row fails the variant with `warm pass: <n> non-OK rows`,
  the same rule as G5 (A stops the run, B and C continue).
- **`--no-warm-pass`** skips the pass and prints `Warm pass: none`: the first measured request
  compiles the graphs, so cold TTFT includes compilation (the pre-RUK-3203 behavior).

## Expected duration per phase

| Phase | Expected duration |
|---|---|
| prepare_host.sh | 15–30 min (once per node) |
| checkpoint download (`--download-only` or `--download`) | about 7–10 min (measured fill rate 0.3 GB/s or more, 185.5 GB) |
| tiny smoke test | about 4–6 min per mode and per retry |
| install (toolchain, vLLM, plugin with native kernels, model deps), at every container start | about 2 min: 97–128 s measured on 2026-11-18 (toolchain 20–48 s, vLLM 44–53 s, plugin with native kernels 48 s on the first build and 16–17 s after, model deps 7–13 s). The first start after a bundle update pays the 48 s again (the build folder is removed) |
| preflight | about 1 min |
| fused GDN gate | about 15 min; less with `--quick-gates` (no eager replays, no bench, no crash probe) |
| tiny TP8 | about 10 min |
| card tests | two files, at most 600 s each; skipped with `--quick-gates` |
| each variant | about 35–45 min; with the default skipped warmup the server startup is minutes (graphs compile on first use, the recipe cache makes this cheap from the second run — see [Warmup and the recipe cache](#warmup-and-the-recipe-cache-ruk-3200)), then the [warm pass](#warm-pass-ruk-3203) (5–35 min; it absorbs the graph compilation), then context fill and four turns |

A full run (three variants, no `--download`) is about 3–5 h with the warm pass. Only the install and
download rows are measurements; the others are estimates, because no rental run has reached a
variant yet.

## What each gate checks and its stop rule

| Gate | Checks | Stop when |
|---|---|---|
| G0 preflight | 8 cards in `hl-smi`; the checkpoint config is block-FP8 [128, 128]; the FP8 layout check passes for each layout in the run; free disk; records the bundle commit, the image, `hl-smi -q` and the environment | any check fails |
| G1 build | stage 1's install from the bundle; `vllm.__version__` is `0.28.1rc1.dev392+g8f816a3f6`; `import vllm_gaudi`; `flashinfer_gaudi.get_capabilities()` | install or import fails, or the version is wrong |
| G1b fused GDN kernel | op numerics of both GDN decode ops on the card against the CPU reference (`test_hpu_op.py`), 1,000-step graph replay against eager (two cases), then the verification op's numerics and two replays (only after the decode checks pass), then the speed measurement; all of these run with `PT_HPU_LAZY_MODE=1`, the servers' bridge mode, and only they decide the kernel. The two decode replays then run once more with `PT_HPU_LAZY_MODE=0` for `torch.compile` and are recorded only (`replay_eager`, `eager_compile_ok`): an eager failure never turns the kernel off. Every step runs in its own process group and is killed with its whole group at its timeout (op tests 600 s, each replay 900 s, bench 1200 s). When `test_hpu_verify_op.py` crashes or times out, G1b runs `gdn_verify_crash_probe.sh` (at most 4500 s) into `fused_gdn/verify_crash/`; recorded only. `--quick-gates` skips the recorded-only steps (the eager replays, the bench and the crash probe); `--no-verify-probe` skips only the probe | never stops: on a failure, a skip or a missing library, every later server runs with `VLLM_HPU_GDN_CORE_DECODE=0` (the PyTorch GDN path), recorded in the manifest and the summary; a verify failure leaves `VLLM_HPU_GDN_CORE_VERIFY=0`, only a pass sets it to 1 |
| G2 tiny TP8 | stage 2a's tiny 8-head checkpoint (GDN head dim 128) at TP8: greedy agreement with `reference.json` on its six prompts (1 to 2,600 tokens). The server gets a fixed 1,200 s to become healthy (`--server-timeout` does not apply). When the gate fails or times out with the combined kernel database loaded, that attempt is kept under `tiny/attempt-combined/` and the gate runs once more without the database | fewer than 16/16 tokens match on any prompt, or the server is not healthy in time (after the retry, when there is one) |
| card_tests (after the gates) | the two HPU-only unit-test files on the card, in lazy mode (`PT_HPU_LAZY_MODE=1`), 600 s each: `tests/unit_tests/test_defragmentation.py` and `tests/unit_tests/ops/test_hpu_fused_moe.py`; each file's output is kept as `<stem>.log` and the pytest counts are recorded in `details.json`. Skipped with `--quick-gates` | never stops the run: a failure, a timeout or a skip is recorded only |
| G3 startup, per server | `Maximum concurrency for 131,072 tokens per request` ≥ agents on that server; `Usable num_blocks`, KV tokens, weights and warmup recorded | the concurrency is too low, or no healthy `/v1/models` within the health timeout (default 1,800 s, `--server-timeout`; warmup is skipped by default, so startup is usually minutes) |
| G4w warm pass, per variant (RUK-3203) | a one-turn copy of the workload, files shifted by one, on the variant's own servers (`max_tokens=64`): compiles the workload's graphs on first use and caches the shared system block; `warm-pass.json` records requests, seconds and non-OK rows; `--no-warm-pass` switches it off | any non-OK row fails the variant with `warm pass: <n> non-OK rows` (the variant's "on failure" rule applies: A stops the run, B and C continue) — see [Warm pass](#warm-pass-ruk-3203) |
| G4 coherence (variant A's server, after A's warm pass) | the 8 short prompts of stage 2b; one 120K-token needle prompt | an empty answer, repetition, a JSON prompt that does not parse, or a missed needle |
| G5 load, per variant | the workload finishes in 5,400 s with no request error and `Running` reaching the agent count | any of these (then the variant's "on failure" rule applies: A stops the run, B and C continue) |

Notes:

- The install adds `habanalabs-tools` 1.24.1-482 (tpc-clang) to the container when the image
  lacks it, so the fused GDN decode and verification kernels and the flashinfer kernels are
  compiled from the bundle (`install-0-toolchain.log`, `toolchain.json`). Only if that install
  fails does the run continue without native kernels on the PyTorch GDN path, with
  `native_build: 0` in `toolchain.json`, the G1b gate recording the missing library, and a
  warning in the console.
- The install logs are in the results folder as `install-<n>.log`, with `pip-freeze.txt` (what
  the install steps left in the container's Python; pytest is pinned to 9.2.0, the version the
  2026-11-18 rental resolved); the per-phase records (`startup.json`, `toolchain.json`,
  `requests.jsonl`, `metrics.jsonl`, `status.json`, `summary.*`) are all in the results folder for
  the run id.
- Every server runs lazy (`PT_HPU_LAZY_MODE=1`). The native extensions link only the Habana
  backend library so one build loads in both bridge modes; the checks that prove it on the card
  are in [native-extensions.md](../native-extensions.md#run-sheet-lines-for-the-next-rental).

## If the GDN verify op crashes (RUK-3105)

G1b runs the crash probe on its own when its verify step ends by a signal (a segfault) or a
timeout, into `<results>/<run-id>/fused_gdn/verify_crash/` (its console output is
`fused_gdn/verify_crash_probe.log`). `--quick-gates` and `--no-verify-probe` leave it out. To run
it by hand, use a container that has the stack installed while the cards are free: the smoke
container when no run is going,

```bash
sudo docker exec fn-smoke bash /bundle/plugin/tools/flash_next/gdn_verify_crash_probe.sh /results/verify-crash
```

or `fn-rental` in place of `fn-smoke` while a run is still in its gate phases (once a variant's
server is up it holds every card).

The probe runs six short checks, each in its own process under `gdb -batch`, bounded by
`PROBE_TIMEOUT_S` (default 300 s): the decode control, the verify op at T = 1 and T = 4 (the
crashing `ver_tp8_chain` case) in lazy mode, T = 4 in eager mode, and the T = 1 / T = 4 checks
again against a verify kernel rebuilt with its `.llvmbc` section stripped
(`-DGDN_FUSED_VERIFY_STRIP_BITCODE=ON`).

pytest runs with `-s` (no capture) in every check. Without it pytest swallowed what the graph
compiler writes to stderr, the only place its own error message appears; that is why the segfault
logs of 2026-11-18 show no compiler message. When gdb is absent and cannot be installed
(`gdb: missing` in `summary.txt`), every check still runs, without gdb: its log holds the same
output and the Python stack of the fault handler (`Fatal Python error: ...`, also for a check that
hits the time limit), and `summary.txt` quotes that stack in place of the gdb backtrace; a crash
then reads `crashed:exit139` (the exit code) instead of `crashed:SIGSEGV`.

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
  errors, wall time), one section per variant — each with its `Warm pass:` line, the numbers of
  the [warm pass](#warm-pass-ruk-3203) (`none` with `--no-warm-pass`) —, the GDN decode line
  (fused kernel or PyTorch path) and the native-kernel line from `toolchain.json`. Its header
  also carries the run's warmup mode (`Warmup: skipped ...` / `Warmup: full`) and the server
  health timeout (RUK-3200). Paste it whole into a comment on RUK-2709. It is written at the end
  of every run that reached the orchestrator, passed or failed.
- `SEND THIS FILE: <results>/<run-id>.tar.gz` — the whole run folder, made even when the run
  failed or was interrupted (`Ctrl-C`, `SIGTERM`, a dropped SSH session): `console.log`
  (everything the run printed, kept across `--from` resumes), `manifest.json`, `toolchain.json`,
  the `install-<n>.log` files, `pip-freeze.txt`, each gate's and variant's `status.json` and
  `details.json`, `startup.json`, `first-token.json` (the first completion of each server of the
  variant), `warm-pass.json` and `warm-requests.jsonl` (the variant's warm pass), `server<k>.log`,
  `requests.jsonl`, `metrics.jsonl`, `hlsmi.csv`, `workload.json`, `warm-workload.json` and
  `summary.json`. It also holds `habana_logs/` (the Habana runtime logs of the run's processes).
  `run_rental.sh` adds, best effort, just before it writes the tarball: `dmesg-tail.txt` (the last
  400 kernel log lines: OOM kills, driver errors; needs root, skipped without it),
  `image-digest.txt` (the repo digest and id of the image the run used) and, only when the
  container outlived its docker client or was interrupted, `container-state.json` (`docker
  inspect` state: `ExitCode`, `OOMKilled`; a container that finished by itself is already removed
  by `--rm`, and its exit code is the `container exit code:` line of `console.log`). The tiny
  gate's weights
  (`tiny/ckpt/*.safetensors`, about 136 MB) are left out of the tarball; the checkpoint's config
  and `reference.json` are in it. Copy it to the Mac at
  `/Volumes/Storage/paperclip/tmp/rental-results/` (scp from the node to your workstation, then to
  the Mac). The deeper readings of the plan's M5 table (the drafter-rank log, the profiler trace, a
  failing server's log) are read from there.

If the run stopped early (an install step failed, or a gate tripped), also paste the last 40 lines
of `<results>/<run-id>/console.log`: they name the phase that stopped and the reason, and an install
failure prints the last 30 lines of its `install-<n>.log` there. A failure before the orchestrator
(the toolchain or install steps) leaves no `summary.md`; the console log and the tarball are then
the whole result.
