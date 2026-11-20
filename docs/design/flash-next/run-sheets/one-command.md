# Flash-Next rental: the whole session in one command

On the workstation, as soon as the provider shows the node's IP address:

```bash
bash tools/flash_next/rental/rent.sh <ip>
```

It does the six manual steps of [rental.md, Cheapest session order](rental.md#cheapest-session-order)
by itself, and it is safe to run again: the same command continues the session, re-attaches after
`Ctrl-C`, a sleeping laptop or a dropped connection, and skips a finished stage after checking on
the node that its result is still there. Before renting, try both halves without a node: `--dry-run`
prints every local and remote command in order and contacts nothing, and
`bash tools/flash_next/rental/make_rental_bundle.sh --out dist` shows that the bundle builds (it needs
a clean tree and a vLLM checkout with the engine pin: `--vllm-src PATH` or `FN_VLLM_SRC`).

**Only the provider's console stops the billing.** The command cannot release the node; it says so,
with the session's time and cost, every time it stops.

## The stages

| Stage | What it does | Time |
|---|---|---|
| 1 connect | SSH as `sesterce` with `~/.ssh/sesterce_ed25519` (`--user`, `--key`); checks passwordless sudo; prints the node (OS, CPUs, memory, disks, cards) and, once, the host key fingerprint | seconds |
| 2 bundle | builds `dist/flash-next-rental-<sha>.tar.gz` from this checkout (clean tree, or `--allow-dirty`; `--bundle FILE` uses a built one), copies it, unpacks it over `~/rental`; skipped when the node has this commit | under a minute |
| 3 host | `sudo bash prepare_host.sh`, with the data disk when the checkpoint needs one | 15–30 min, once per node |
| 4 download | starts the 185.5 GB checkpoint download (`run_rental.sh --download-only`) and goes on | seconds |
| 5 smoke | `tiny_smoke.sh --gdn pytorch`, then `--gdn fused` (`--smoke-modes pytorch,fused,hc` adds the fused hyper-connection mode, for a run with `--hc-fused`) | 4–6 min per mode, plus about 2 min of install the first time |
| 6 wait | waits for the download: bytes on disk, rate and ETA every 30 s | 7–10 min in all at 0.3 GB/s, most of it already spent in stage 5 |
| 7 run | `run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8 --quick-gates --variants A,C` | hours (an estimate: about 2 h; no run has reached a variant yet) |
| 8 fetch | copies the run's tarball, `summary.md`, `console.log`, the smoke folders (no `*.safetensors`) and the `prepare_host.sh` logs; prints `summary.md` | under a minute |

Stages 3, 5 and 7 run on the node as jobs that are detached from the SSH connection (their output is
in `~/rental/.jobs/<name>.log` there), so nothing on the node ends when the workstation does.

## What you see when it works

One `stage N/8 <name>: start` and `ok` line per stage, each with the session's time and cost
(`--rate`, default 10 dollars per hour; the clock starts at the first connection). During the run the
lines that decide something are marked with `>>>`: `phase:` lines, gate results, `RUN ... PASSED` or
`FAILED at`, and the first proof that the model decodes,
`>>> FIRST TOKEN >>> variant A server0: first token after <s> s: '<text>'`. At the end: `summary.md`
(paste it into the issue), the local paths, and the reminder.

## The four things that stop it, and the rerun

| It stops with | Why | Then |
|---|---|---|
| `stage 3/8 host: FAILED (no data disk chosen; nothing was changed on the node)`, exit 3 | the checkpoint folder has less than 250 GB free, and there is not exactly one blank disk of at least 300 GB. Nothing was changed; the `lsblk` table is printed | the printed command: `bash tools/flash_next/rental/rent.sh <ip> --data-disk /dev/<name>` (a disk with a filesystem or partitions is never formatted) |
| `stage 3/8 host: FAILED (prepare_host.sh asks for a reboot)`, exit 3 | the driver does not see all cards, or the kernel needs headers | reboot the node yourself (`sudo reboot` on it; this command never does), then the same command |
| `stage 5/8 smoke: FAILED (the <mode> mode failed: ...)`, exit 1 | a tiny server did not start or decode. The error lines and the local path of `server0.log` are printed; the run was not started | fix, commit, the same command (it sends the new bundle and reinstalls). Only for a failed `fused` (or `hc`) mode: add `--continue-after-smoke-failure fused` |
| `stage 6/8 wait: FAILED (the download container fn-download ended with exit code N)`, exit 1 | the download ended with an error; its last log lines are printed | the same command: the download starts again and resumes |

A run that has ended is never started a second time by itself, and its bundle stays on the node:
after a gate stopped it, add `--from <phase>` to resume the same run id, or `--new-run` to start
another run; both send the checkout first when it has a new commit. Exit codes: 0 done, 1 a stage
failed, 2 usage, 3 stopped for your decision, 130 detached.

## While it runs

- `bash tools/flash_next/rental/rent.sh <ip> --status` prints each job's state and the last lines of
  its log, and changes nothing.
- `bash tools/flash_next/rental/rent.sh <ip> --stop` interrupts the run (`run_rental.sh` stops its
  servers and writes the report and the tarball), then fetches.
- `Ctrl-C` only detaches. The same command attaches again.

## Where the results land

`~/rental-sessions/<ip>-<UTC date>/` (`--session DIR`): `results/<run-id>.tar.gz` (send it),
`results/<run-id>/summary.md` and `console.log`, `results/smoke-*/` (with each mode's `server0.log`),
`results/prepare-host-logs/`, `session.log` (everything printed, with UTC timestamps) and `jobs/`
(the node-side job logs).

All options: `bash tools/flash_next/rental/rent.sh --help`. Not checked on a node yet: the command
is tested against stand-in `ssh` and `scp` only.
