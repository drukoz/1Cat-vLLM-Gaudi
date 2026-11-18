#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run a command inside the amd64 tpcsim container (Gaudi TPC toolchain +
# Gaudi2 instruction simulator) with the current directory mounted at /work.
# Starts the colima VM and builds the image when needed. See
# docs/design/flash-next/tpcsim.md.
set -euo pipefail

VM_ROOT=/Volumes/Storage/paperclip/linux-vm
export COLIMA_HOME="$VM_ROOT/colima"
export DOCKER_CONFIG="$VM_ROOT/docker-config"
export XDG_CONFIG_HOME="$VM_ROOT"
export DOCKER_HOST="unix://$COLIMA_HOME/tpcsim/docker.sock"
IMAGE=tpcsim:1.24.1

if [ "${1:-}" = "--stop" ]; then
    exec colima stop --profile tpcsim
fi

if [ "$#" -eq 0 ]; then
    echo "usage: tools/flash_next/tpcsim.sh <command...> | --stop" >&2
    exit 2
fi

command -v colima >/dev/null || { echo "colima is not installed: brew install colima docker" >&2; exit 1; }
command -v docker >/dev/null || { echo "docker CLI is not installed: brew install colima docker" >&2; exit 1; }

if ! colima status --profile tpcsim >/dev/null 2>&1; then
    colima start --profile tpcsim --vm-type vz --vz-rosetta --arch aarch64 \
        --cpu 6 --memory 12 --disk 80 --mount /Volumes/Storage:w >&2
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build --platform linux/amd64 -t "$IMAGE" "$(dirname "$0")/tpcsim" >&2
fi

exec docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work -e PYTHONPATH=/work \
    "$IMAGE" bash -lc "$*"
