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
# By default the servers skip the bucket warmup (VLLM_SKIP_WARMUP=true): the graphs compile on first
# use and a run takes about 5 minutes per mode. --full-warmup keeps the real warmup, about 15 minutes
# per mode on the tiny model. The pytorch mode's server gets a compiled-recipe cache under
# /results/recipe-cache, reused on the next start, so its second run is faster; the fused mode runs
# without it (its kernel database must be in GC_KERNEL_PATH, which the Habana bridge accepts only
# without a recipe cache). --no-recipe-cache drops the cache. --health-timeout S and --env KEY=VALUE
# (repeatable) are passed through to the check.
#
# The container stays. A second call reuses it, skips the install and the checkpoint build, and takes
# a few minutes. After a change under plugin/ pass --reinstall: the servers run the installed plugin,
# not the source tree.

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
    echo "  --gdn LIST      GDN modes to serve, in order (default: pytorch,fused)"
    echo "  --results DIR   results directory on the host (default: ./results); fixed once the container exists"
    echo "  --run-id ID     folder name under the results directory (default: smoke-<UTC timestamp>)"
    echo "  --image IMG     container image (default: $DEFAULT_IMAGE)"
    echo "  --reinstall     install the stack again in the existing container (after a change under plugin/)"
    echo "  --fresh         remove the container first: new install, new tiny checkpoint"
    echo "  --full-warmup   run the real bucket warmup (about 15 minutes per mode); the default skips it"
    echo "  --health-timeout S  seconds to wait for the server to get healthy (default in the check: 1800)"
    echo "  --env KEY=VALUE extra server environment variable, repeatable; applied after the mode's own flags"
    echo "  --no-recipe-cache   do not give the pytorch mode's server the compiled-recipe cache under /results/recipe-cache (the fused mode never gets it)"
    echo "  --no-combined-kernel-db  keep the combined kernel database out of the smoke servers (FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 in the container)"
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
    RUN_CMD=(
        docker run -d --name "$NAME"
        --runtime=habana
        -e HABANA_VISIBLE_DEVICES=all
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none
        -e PYTHONUNBUFFERED=1
        --cap-add=sys_nice
        --net=host
        --ipc=host
        -v "${BUNDLE_DIR:-$SCRIPT_DIR}:/bundle"
        -v "$RESULTS:/results"
        "$IMAGE"
        sleep infinity
    )
    INSTALL_CMD=(docker exec "$NAME" bash "/bundle/$ENTRY" --install-only --results /results --run-id "$RUN_ID")
    SMOKE_CMD=(docker exec -w /bundle/plugin)
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
        # The check hands the cache to the pytorch mode's server only: the fused mode's kernel database
        # must be registered in GC_KERNEL_PATH, which the Habana bridge accepts only without a cache.
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

mkdir -p "$RESULTS/$RUN_ID"
CONSOLE_LOG="$RESULTS/$RUN_ID/console.log"
say() {
    echo "$@" | tee -a "$CONSOLE_LOG"
}

INSTALLED=0
if [ "$NEW" -eq 1 ]; then
    say "container $NAME: starting"
    "${RUN_CMD[@]}" >/dev/null
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

# Ctrl-C ends the docker client, not the process in the container: stop the check there, so the
# tiny server and its workers release the cards.
on_interrupt() {
    docker exec "$NAME" pkill -INT -f tools.flash_next.rental.tiny_smoke >/dev/null 2>&1 || true
    echo "interrupted: asked the check in $NAME to stop its server" >&2
    exit 130
}
trap on_interrupt INT TERM
set +e
"${SMOKE_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
RC=${PIPESTATUS[0]}
set -e
trap - INT TERM

say "results: $RESULTS/$RUN_ID (summary.json, <mode>/server0.log, console.log)"
say "container $NAME is still running, the stack installed and the cards free: sudo docker exec -it $NAME bash"
say "run again: sudo bash $0 (a few minutes). Remove it when done: sudo docker rm -f $NAME"
exit "$RC"
