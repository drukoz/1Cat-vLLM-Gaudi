#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Tiny Flash-Next smoke test on the rented Gaudi node (RUK-2709): can a Flash-Next server start and
# decode on this node at all. Run it first on a new node, after prepare_host.sh and before the
# checkpoint download and run_rental.sh. It needs no checkpoint. From the unpacked rental bundle:
#
#     sudo bash tiny_smoke.sh
#
# Starts the Habana container "fn-smoke", installs the stack from the bundle in it, then runs
# tools/flash_next/rental/tiny_smoke.py there: the tiny checkpoint is built, served with the
# environment of a rental variant and compared with its CPU reference, once on the PyTorch GDN path
# and once with the fused GDN kernel. Every server is stopped again, so the cards are free when the
# script returns.
#
# A third mode, hc, is the fused mode plus the fused hyper-connection kernels (VLLM_HPU_HC_FUSED=1).
# It is not in the default list: --gdn hc (or --gdn pytorch,fused,hc) selects it. It fails when the
# server log says a hyper-connection ran the torch path. The kernels are built by the install of
# this bundle: in a container an older bundle installed, pass --reinstall first.
#
# A fourth mode, moe, is the fused mode plus the MoE glue kernels (VLLM_HPU_MOE_GLUE=1), which are in
# the library of the hyper-connection kernels. It is not in the default list either: --gdn moe (or
# --gdn pytorch,fused,hc,moe) selects it. It fails when the server log says a MoE block ran the torch
# path.
#
# By default the servers skip the bucket warmup (VLLM_SKIP_WARMUP=true): the graphs compile on first
# use and a run takes about 4-6 minutes per mode. --full-warmup keeps the real warmup, about 15 minutes
# per mode on the tiny model. The servers get a compiled-recipe cache under /results/recipe-cache,
# reused on the next start, so a second run is faster: the pytorch mode's always, the fused mode's
# only with the combined kernel database (without that database, or with --no-combined-kernel-db, its
# kernel database is in a colon-list GC_KERNEL_PATH, which the Habana bridge accepts only without a
# recipe cache). --no-recipe-cache drops the cache. --health-timeout S and --env KEY=VALUE
# (repeatable) are passed through to the check.
#
# The container stays. A second call reuses it, skips the install and the checkpoint build, and takes
# a few minutes. After a change under plugin/ pass --reinstall: the servers run the installed plugin,
# not the source tree.
#
# Ctrl-C, SIGTERM and a dropped SSH session (SIGHUP) stop the check inside the container, so its
# server releases the cards; the exit code is then 130, 143 or 129. The Habana runtime logs of a
# call go to <results>/<run-id>/habana_logs.

set -euo pipefail

DEFAULT_IMAGE="vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest"
NAME="fn-smoke"
RENTAL_NAME="fn-rental"
MARKER="/tmp/fn-smoke.installed"
CKPT="/tmp/fn-smoke-ckpt"
ENTRY="plugin/tools/flash_next/rental/container_entry.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") [options]"
    echo "  --gdn LIST      GDN modes to serve, in order (default: pytorch,fused); the third mode, hc, is fused plus the fused hyper-connection kernels (VLLM_HPU_HC_FUSED=1) and runs only when named: --gdn hc; the fourth, moe, is fused plus the MoE glue kernels (VLLM_HPU_MOE_GLUE=1) and runs only when named: --gdn moe"
    echo "  --results DIR   results directory on the host (default: ./results); fixed once the container exists"
    echo "  --run-id ID     folder name under the results directory (default: smoke-<UTC timestamp>)"
    echo "  --image IMG     container image (default: $DEFAULT_IMAGE)"
    echo "  --reinstall     install the stack again in the existing container (after a change under plugin/)"
    echo "  --fresh         remove the container first: new install, new tiny checkpoint"
    echo "  --full-warmup   run the real bucket warmup (about 15 minutes per mode); the default skips it"
    echo "  --health-timeout S  seconds to wait for the server to get healthy (default in the check: 1800)"
    echo "  --env KEY=VALUE extra server environment variable, repeatable; applied after the mode's own flags"
    echo "  --no-recipe-cache   give no server the compiled-recipe cache under /results/recipe-cache (by default the pytorch mode's server gets it, and the fused mode's with the combined kernel database)"
    echo "  --no-combined-kernel-db  keep the combined kernel database out of the smoke servers (FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 for the check and its servers); the fused mode's server then runs the colon-list GC_KERNEL_PATH without a recipe cache"
    echo "  --dry-run       print the docker commands and exit 0"
}

IMAGE="$DEFAULT_IMAGE"
RESULTS=""
RUN_ID=""
GDN="pytorch,fused"
REINSTALL=0
FRESH=0
DRY_RUN=0
FULL_WARMUP=0
HEALTH_TIMEOUT=""
NO_RECIPE_CACHE=0
NO_COMBINED_DB=0
EXTRA_ENV=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gdn)
            GDN="$2"
            shift 2
            ;;
        --results)
            RESULTS="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --reinstall)
            REINSTALL=1
            shift
            ;;
        --fresh)
            FRESH=1
            shift
            ;;
        --full-warmup)
            FULL_WARMUP=1
            shift
            ;;
        --health-timeout)
            HEALTH_TIMEOUT="$2"
            shift 2
            ;;
        --env)
            EXTRA_ENV+=("$2")
            shift 2
            ;;
        --no-recipe-cache)
            NO_RECIPE_CACHE=1
            shift
            ;;
        --no-combined-kernel-db)
            NO_COMBINED_DB=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

RUN_ID="${RUN_ID:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS="${RESULTS:-$PWD/results}"
case "$RESULTS" in
    /*) ;;
    *) RESULTS="$PWD/$RESULTS" ;;
esac

# The bundle root holds plugin/ and vllm/. This script sits at its top level, or four folders down
# in plugin/tools/flash_next/rental/.
BUNDLE_DIR=""
if [ -f "$SCRIPT_DIR/$ENTRY" ]; then
    BUNDLE_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../../../../$ENTRY" ]; then
    BUNDLE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

# The three docker commands of a first call; a later call skips the first two.
build_commands() {
    # --init: an init process as PID 1 reaps the workers a crashed server leaves behind (without it
    # they stay zombies and the check waits its full 65 s stop time for them). PYTHONFAULTHANDLER
    # gives every Python process a stack on a crash. HABANA_LOGS keeps the Habana runtime logs of
    # anything started in the container by hand under the results folder; the check's own logs go
    # to its run folder (below).
    RUN_CMD=(
        docker run -d --name "$NAME"
        --init
        --runtime=habana
        -e HABANA_VISIBLE_DEVICES=all
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none
        -e PYTHONUNBUFFERED=1
        -e PYTHONFAULTHANDLER=1
        -e HABANA_LOGS=/results/habana_logs
        --cap-add=sys_nice
        --net=host
        --ipc=host
        -v "${BUNDLE_DIR:-$SCRIPT_DIR}:/bundle"
        -v "$RESULTS:/results"
        "$IMAGE"
        sleep infinity
    )
    INSTALL_CMD=(docker exec "$NAME" bash "/bundle/$ENTRY" --install-only --results /results --run-id "$RUN_ID")
    # The container outlives one call, so what belongs to this call travels with the docker exec:
    # the Habana logs of the check and its servers go to the call's run folder. The fault handler is
    # repeated here for a container that an older bundle created without it.
    SMOKE_CMD=(
        docker exec -w /bundle/plugin
        -e "HABANA_LOGS=/results/$RUN_ID/habana_logs"
        -e PYTHONFAULTHANDLER=1
    )
    if [ "$NO_COMBINED_DB" -eq 1 ]; then
        # docker exec -e reaches the check and the servers it starts: they then treat the
        # combined kernel database as absent and use the colon-list GC_KERNEL_PATH (RUK-3249).
        SMOKE_CMD+=(-e FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0)
    fi
    SMOKE_CMD+=(
        "$NAME"
        python -m tools.flash_next.rental.tiny_smoke
        --out "/results/$RUN_ID" --ckpt "$CKPT" --gdn "$GDN"
    )
    if [ "$NO_RECIPE_CACHE" -eq 0 ]; then
        # The check hands the cache to the servers that may carry it: the pytorch mode's, and the fused
        # mode's with the combined kernel database (a colon-list GC_KERNEL_PATH and a recipe cache
        # together are refused by the Habana bridge).
        SMOKE_CMD+=(--recipe-cache /results/recipe-cache)
    fi
    if [ "$FULL_WARMUP" -eq 1 ]; then
        SMOKE_CMD+=(--full-warmup)
    fi
    if [ -n "$HEALTH_TIMEOUT" ]; then
        SMOKE_CMD+=(--health-timeout "$HEALTH_TIMEOUT")
    fi
    if [ "${#EXTRA_ENV[@]}" -gt 0 ]; then
        local kv
        for kv in "${EXTRA_ENV[@]}"; do
            SMOKE_CMD+=(--env "$kv")
        done
    fi
}
build_commands

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${RUN_CMD[@]}"
    printf '\n'
    printf '%q ' "${INSTALL_CMD[@]}"
    printf '\n'
    printf '%q ' "${SMOKE_CMD[@]}"
    printf '\n'
    exit 0
fi

if [ -z "$BUNDLE_DIR" ]; then
    echo "error: run this from the unpacked rental bundle (no $ENTRY next to or above $SCRIPT_DIR)" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found on this host" >&2
    exit 1
fi

docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"' || {
    echo "error: the habana container runtime is not registered with Docker; run: sudo bash $BUNDLE_DIR/prepare_host.sh" >&2
    exit 1
}

# True when the name list $1 (one per line) has the line $2.
has_name() {
    local nl=$'\n'
    case "$nl$1$nl" in
        *"$nl$2$nl"*) return 0 ;;
    esac
    return 1
}

RUNNING="$(docker ps --format '{{.Names}}')"
EXISTING="$(docker ps -a --format '{{.Names}}')"

if has_name "$RUNNING" "$RENTAL_NAME"; then
    echo "error: the rental run (container $RENTAL_NAME) is running and holds the cards; wait for it, or stop it: sudo docker stop $RENTAL_NAME" >&2
    exit 1
fi

if command -v hl-smi >/dev/null 2>&1; then
    if [ "$(hl-smi -Q index,name -f csv,noheader 2>/dev/null | grep -c . || true)" -eq 0 ]; then
        echo "warning: hl-smi lists no cards; the servers will likely fail (check the driver, or reboot and rerun prepare_host.sh)" >&2
    fi
fi

NEW=0
if [ "$FRESH" -eq 1 ] || ! has_name "$RUNNING" "$NAME"; then
    if has_name "$EXISTING" "$NAME"; then
        docker rm -f "$NAME" >/dev/null
    fi
    NEW=1
else
    # /results was mounted when the container was created; a later --results cannot move it.
    MOUNTED="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/results"}}{{.Source}}{{end}}{{end}}' "$NAME" 2>/dev/null || true)"
    if [ -n "$MOUNTED" ]; then RESULTS="$MOUNTED"; fi
fi

mkdir -p "$RESULTS/$RUN_ID/habana_logs" "$RESULTS/habana_logs"  # HABANA_LOGS of the check, and of the container
CONSOLE_LOG="$RESULTS/$RUN_ID/console.log"
say() {
    echo "$@" | tee -a "$CONSOLE_LOG"
}

INSTALLED=0
if [ "$NEW" -eq 1 ]; then
    say "container $NAME: starting"
    if ! "${RUN_CMD[@]}" >/dev/null; then
        # --init needs Docker's docker-init binary on the host; a host without it leaves a created,
        # never started container behind. Try once more without the flag: any other cause fails
        # again here, with Docker's own message.
        say "container $NAME: the start failed; trying once more without --init"
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        NO_INIT_CMD=()
        for arg in "${RUN_CMD[@]}"; do
            [ "$arg" = "--init" ] || NO_INIT_CMD+=("$arg")
        done
        "${NO_INIT_CMD[@]}" >/dev/null
    fi
elif docker exec "$NAME" test -f "$MARKER"; then
    INSTALLED=1
fi

if [ "$INSTALLED" -eq 0 ] || [ "$REINSTALL" -eq 1 ]; then
    if [ "$INSTALLED" -eq 1 ]; then
        docker exec "$NAME" rm -f "$MARKER"  # a failed reinstall must not look installed
    fi
    say "container $NAME: installing the stack from the bundle (a few minutes)"
    set +e
    "${INSTALL_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" -ne 0 ]; then
        echo "error: install failed (exit $RC); logs: $RESULTS/$RUN_ID/install-*.log. The container is kept: sudo docker exec -it $NAME bash" >&2
        exit "$RC"
    fi
    docker exec "$NAME" touch "$MARKER"
else
    say "container $NAME: reusing the installed stack (--reinstall after a change under plugin/, --fresh for a new container)"
fi

# A signal to this script does not reach the check in the container: Ctrl-C and a dropped SSH
# session end the docker client at most, and the tiny server would keep all cards. on_signal stops
# the check there (SIGINT: it stops its server and waits for the workers), waits until it is gone,
# and after STOP_WAIT_S seconds stops the whole container instead, so nothing can keep the cards.
STOP_WAIT_S="${FN_SMOKE_STOP_WAIT_S:-90}"
RUN_PID=""

# True while the check runs in the container.
check_running() {
    docker exec "$NAME" pgrep -f tools.flash_next.rental.tiny_smoke >/dev/null 2>&1
}

# The docker client and tee end when the check does. Give them 15 s to copy its last lines into
# console.log, then end what is left of them.
wait_client() {
    local left=15
    if [ -z "$RUN_PID" ]; then
        return 0
    fi
    while kill -0 "$RUN_PID" 2>/dev/null; do
        if [ "$left" -le 0 ]; then
            pkill -P "$RUN_PID" 2>/dev/null
            kill "$RUN_PID" 2>/dev/null
            break
        fi
        sleep 1
        left=$((left - 1))
    done
    wait "$RUN_PID" 2>/dev/null
}

# on_signal NAME EXIT_CODE — Ctrl-C, SIGTERM, or SIGHUP from a dropped SSH session.
on_signal() {
    local left="$STOP_WAIT_S"
    trap '' INT TERM HUP PIPE
    set +e
    say "interrupted by SIG$1: asked the check in $NAME to stop its server (at most $STOP_WAIT_S s)"
    docker exec "$NAME" pkill -INT -f tools.flash_next.rental.tiny_smoke >/dev/null 2>&1
    while check_running; do
        if [ "$left" -le 0 ]; then
            say "the check did not stop in $STOP_WAIT_S s: stopping container $NAME so nothing keeps the cards (the next call starts a new container and installs again)"
            docker stop -t 10 "$NAME" >/dev/null 2>&1
            break
        fi
        sleep 1
        left=$((left - 1))
    done
    wait_client
    say "results so far: $RESULTS/$RUN_ID"
    exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
set +e
# The check runs in the background and the script waits for it with `wait`, which a signal
# interrupts at once (a foreground command would hold the trap back until the check ends).
(
    # The docker client and tee ignore Ctrl-C and a dropped terminal: on_signal stops the check, and
    # they go on copying its last lines into console.log until it is gone.
    trap '' INT HUP
    "${SMOKE_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
    exit "${PIPESTATUS[0]}"
) &
RUN_PID=$!
wait "$RUN_PID"
RC=$?
set -e
trap - INT TERM HUP

say "results: $RESULTS/$RUN_ID (summary.json, <mode>/server0.log, console.log, habana_logs/)"
say "container $NAME is still running, the stack installed and the cards free: sudo docker exec -it $NAME bash"
say "run again: sudo bash $0 (a few minutes). Remove it when done: sudo docker rm -f $NAME"
exit "$RC"
