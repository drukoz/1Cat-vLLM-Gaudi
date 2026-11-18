#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the Flash-Next rental test on the rented Gaudi node (RUK-2709 M4). Run it from the unpacked
# rental bundle: bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8
#
# Starts the Habana container, runs container_entry.sh inside it (install + orchestrator), and
# leaves <results>/<run-id>.tar.gz on the host to send back.

set -euo pipefail

DEFAULT_IMAGE="vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") --model DIR [options]"
    echo "  --model DIR       checkpoint directory (required; /models/<name> inside the container)"
    echo "  --image IMG       container image (default: $DEFAULT_IMAGE)"
    echo "  --results DIR     results directory on the host (default: ./results)"
    echo "  --variants LIST   variant ids, e.g. A,B,C (passed to the orchestrator)"
    echo "  --with-profile    enable profiling (passed to the orchestrator)"
    echo "  --download        download the checkpoint into --model inside the container first"
    echo "  --from PHASE      resume the orchestrator from PHASE"
    echo "  --run-id ID       run id (default: UTC timestamp)"
    echo "  --dry-run         print the docker command and exit 0"
}

MODEL=""
IMAGE="$DEFAULT_IMAGE"
RESULTS=""
VARIANTS=""
WITH_PROFILE=0
DOWNLOAD=0
FROM_PHASE=""
RUN_ID=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --results)
            RESULTS="$2"
            shift 2
            ;;
        --variants)
            VARIANTS="$2"
            shift 2
            ;;
        --with-profile)
            WITH_PROFILE=1
            shift
            ;;
        --download)
            DOWNLOAD=1
            shift
            ;;
        --from)
            FROM_PHASE="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
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

if [ -z "$MODEL" ]; then
    echo "error: --model is required" >&2
    usage >&2
    exit 2
fi
case "$MODEL" in
    /*) ;;
    *)
        echo "error: --model must be an absolute path" >&2
        exit 2
        ;;
esac

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS="${RESULTS:-$PWD/results}"
MODEL_PARENT="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL")"

if [ "$DOWNLOAD" -eq 1 ]; then
    mkdir -p "$MODEL_PARENT"
fi
if [ ! -d "$MODEL_PARENT" ]; then
    echo "error: model parent $MODEL_PARENT does not exist (pass --download to fetch the checkpoint)" >&2
    exit 1
fi

mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"
mkdir -p "$RESULTS/$RUN_ID"

ENTRY_ARGS=(--model "/models/$MODEL_NAME" --results /results --run-id "$RUN_ID")
if [ -n "$VARIANTS" ]; then ENTRY_ARGS+=(--variants "$VARIANTS"); fi
if [ "$WITH_PROFILE" -eq 1 ]; then ENTRY_ARGS+=(--with-profile); fi
if [ -n "$FROM_PHASE" ]; then ENTRY_ARGS+=(--from "$FROM_PHASE"); fi
if [ "$DOWNLOAD" -eq 1 ]; then ENTRY_ARGS+=(--download); fi

DOCKER_CMD=(
    docker run --rm --name fn-rental
    --runtime=habana
    -e HABANA_VISIBLE_DEVICES=all
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none
    --cap-add=sys_nice
    --net=host
    --ipc=host
    -v "$SCRIPT_DIR:/bundle"
    -v "$MODEL_PARENT:/models"
    -v "$RESULTS:/results"
    "$IMAGE"
    bash /bundle/plugin/tools/flash_next/rental/container_entry.sh
    "${ENTRY_ARGS[@]}"
)

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${DOCKER_CMD[@]}"
    printf '\n'
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found on this host" >&2
    exit 1
fi

docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"' || {
    echo "error: the habana container runtime is not registered with Docker; run: sudo bash $SCRIPT_DIR/prepare_host.sh" >&2
    exit 1
}

if command -v hl-smi >/dev/null 2>&1; then
    if [ "$(hl-smi -Q index,name -f csv,noheader 2>/dev/null | grep -c . || true)" -eq 0 ]; then
        echo "warning: hl-smi lists no cards; the run will likely fail (check the driver, or reboot and rerun prepare_host.sh)" >&2
    fi
fi

echo "run id: $RUN_ID"
set +e
"${DOCKER_CMD[@]}"
RC=$?
set -e

TARBALL="$RESULTS/$RUN_ID.tar.gz"
tar czf "$TARBALL" -C "$RESULTS" "$RUN_ID"
echo "container exit code: $RC"
echo "SEND THIS FILE: $TARBALL"
exit "$RC"
