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
    echo "  --no-disk-check   skip the free-space checks (download folder and Docker root)"
}

# The nearest existing folder at or above $1 (a folder that does not exist yet lives on its parent's disk).
existing_dir() {
    local dir="$1"
    while [ ! -d "$dir" ]; do dir="$(dirname "$dir")"; done
    echo "$dir"
}

# Prints "<free bytes> <mount point>" for the filesystem holding folder $1, or nothing when df fails.
disk_free() {
    local kb="" mnt=""
    read -r kb mnt < <(df -Pk "$1" 2>/dev/null | awk 'NR==2 {print $4, $6}') || true
    case "$kb" in
        '' | *[!0-9]*) return 0 ;;
    esac
    echo "$((kb * 1024)) $mnt"
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
NO_DISK_CHECK=0

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
        --no-disk-check)
            NO_DISK_CHECK=1
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

DOWNLOAD_NEED_GB="${FN_RENTAL_DOWNLOAD_NEED_GB:-250}"
DOCKER_NEED_GB="${FN_RENTAL_DOCKER_NEED_GB:-30}"

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
    if [ "$NO_DISK_CHECK" -eq 1 ]; then
        echo "warning: free-space checks skipped (--no-disk-check)" >&2
    else
        need=$((DOWNLOAD_NEED_GB * 1000000000))
        if [ -d "$MODEL" ]; then
            used_kb="$(du -sk "$MODEL" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
            case "$used_kb" in
                '' | *[!0-9]*) ;;
                *) need=$((need - used_kb * 1024)) ;;
            esac
        fi
        if [ "$need" -lt 0 ]; then need=0; fi
        dir="$(existing_dir "$MODEL")"
        free="" mnt=""
        read -r free mnt < <(disk_free "$dir") || true
        if [ -z "$free" ]; then
            echo "warning: cannot read the free space of $dir; continuing" >&2
        elif [ "$free" -lt "$need" ]; then
            echo "error: $dir (filesystem mounted at $mnt) has $((free / 1000000000)) GB free; --download needs \
$((need / 1000000000)) GB there (the 185.5 GB checkpoint plus room for the results). Fix: sudo bash \
$SCRIPT_DIR/prepare_host.sh --data-disk DEV (an unused disk from lsblk, mounted at /models), or point --model \
at a disk with room, or pass --no-disk-check." >&2
            exit 1
        else
            echo "disk: $dir has $((free / 1000000000)) GB free, --download needs $((need / 1000000000)) GB: ok"
        fi
    fi
fi
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
    -e PYTHONUNBUFFERED=1
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

if [ "$NO_DISK_CHECK" -eq 0 ]; then
    DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    if [ -z "$DOCKER_ROOT" ] || [ ! -d "$DOCKER_ROOT" ]; then
        echo "warning: docker info gave no root folder; skipping its free-space check" >&2
    else
        free="" mnt=""
        read -r free mnt < <(disk_free "$DOCKER_ROOT") || true
        if [ -z "$free" ]; then
            echo "warning: cannot read the free space of $DOCKER_ROOT; continuing" >&2
        elif [ "$free" -lt "$((DOCKER_NEED_GB * 1000000000))" ]; then
            echo "error: Docker's root $DOCKER_ROOT (filesystem mounted at $mnt) has $((free / 1000000000)) GB \
free; the container's installs need $DOCKER_NEED_GB GB. Fix: free space there (docker system prune removes \
stopped containers and unused images), or move Docker's \"data-root\" in /etc/docker/daemon.json to a bigger \
disk and rerun prepare_host.sh, or pass --no-disk-check." >&2
            exit 1
        else
            echo "disk: Docker's root $DOCKER_ROOT has $((free / 1000000000)) GB free: ok"
        fi
    fi
fi

if command -v hl-smi >/dev/null 2>&1; then
    if [ "$(hl-smi -Q index,name -f csv,noheader 2>/dev/null | grep -c . || true)" -eq 0 ]; then
        echo "warning: hl-smi lists no cards; the run will likely fail (check the driver, or reboot and rerun prepare_host.sh)" >&2
    fi
fi

# Everything the run prints is also kept in the results folder (so in the tarball), to be read
# or pasted after the fact; tee -a keeps one file across --from resumes of the same run id.
CONSOLE_LOG="$RESULTS/$RUN_ID/console.log"
echo "run id: $RUN_ID" | tee -a "$CONSOLE_LOG"
set +e
"${DOCKER_CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
RC=${PIPESTATUS[0]}
set -e
echo "container exit code: $RC" | tee -a "$CONSOLE_LOG"

TARBALL="$RESULTS/$RUN_ID.tar.gz"
tar czf "$TARBALL" -C "$RESULTS" "$RUN_ID"
echo "PASTE THIS FILE: $RESULTS/$RUN_ID/summary.md (plus the last 40 lines of $CONSOLE_LOG if the run stopped early)"
echo "SEND THIS FILE: $TARBALL"
exit "$RC"
