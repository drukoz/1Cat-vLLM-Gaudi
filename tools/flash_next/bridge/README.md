# Our own build of the Gaudi PyTorch bridge (native-replay track)

The prepared HPU-graph replay (`docs/design/flash-next/native-replay.md`, Handoff) is a change in the Gaudi PyTorch
bridge (`https://github.com/HabanaAI/gaudi-pytorch-bridge`, commit `5176b1b2`, the head of `v1.24.1`: the SDK the
node runs). These files build it inside the farm's base image and publish the wheels for farm jobs.

| File | What |
|---|---|
| `build_bridge.sh` | the first, unpatched build in a throwaway container (`stock-5176b1b2`, 2026-11-20 23:43-00:57 UTC) |
| `dev_first_build.sh` | inside the long-lived dev container `nr-bridge-dev` (farm-base, build CPUs, no card): fetch the source once, apply one patch, a full configure and release build, package |
| `dev_build.sh` | on the node: an incremental build with another patch in the same container (only what the patch touches recompiles) |
| `package_out.py` | the wheels into `~/farm/native-replay/bridge-build/out/<build-id>/` (`wheels/`, `build.log`, `manifest.json` written last, renamed into place): what `bin/farm ... --bridge <build-id>` installs in a job's container |
| `patches/timers.patch` | phase timers of a replayed launch: `PT_HPU_HPUGRAPH_REPLAY_TIMING=N` prints every N replayed launches per thread one line `HPUGRAPH_REPLAY_TIMING ...` (16 phases, tensors per launch); off unless set |
| `patches/lean1.patch` | the timers plus the first prepared-replay step: `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1` keeps each captured segment's launcher (`HabanaLaunchOpPT`) and recipe from its first replay and replays later through `HabanaLaunchOpPT::RunPrepared` (no `HlExec`, no new launcher, no recipe spec, no recipe lookup); off unless set |

Both patches apply to a clean checkout of `5176b1b2` with `git apply`. Neither changes `hpu_lazy_tensors.h` or
`HPUGraph.h` (a third of the bridge's sources include them), so an incremental build after either recompiles about 45
files. Rules for running anything on a card with a custom bridge: `CARD.md`; never `perf` with call graphs on a
process that has a card open (BRIEF rule 12).
