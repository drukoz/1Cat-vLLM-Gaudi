#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# build_base.sh [BUNDLE_DIR]: the farm's base image, on the node. One stack install of a rental
# bundle (container_entry.sh --install-only: the TPC toolchain, vLLM at the engine pin, the plugin
# with its native kernels, the model and test packages) in a container that holds no card, kept as
# the image farm-base:<plugin commit, 12 characters> and farm-base:latest. A job that starts from
# this image needs no install. Needs no card and no queue slot; a node start gives a new machine,
# so it runs once per node start. BUNDLE_DIR defaults to ~/rental.

set -euo pipefail

BUNDLE="${1:-$HOME/rental}"
FROM_IMAGE="${FARM_FROM_IMAGE:-vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest}"
OUT="${FARM_BASE_DIR:-$HOME/farm/base}"
NAME="farm-base-build"
DOCKER="${FARM_DOCKER:-sudo -n docker}"

sha="$(sed -n 's/^plugin sha: *//p' "$BUNDLE/BUNDLE.txt" 2>/dev/null | head -n 1 || true)"
if [ -z "$sha" ]; then
    echo "error: no 'plugin sha:' line in $BUNDLE/BUNDLE.txt (BUNDLE_DIR must be an unpacked rental bundle)" >&2
    exit 1
fi
TAG="farm-base:${sha:0:12}"
mkdir -p "$OUT"

if $DOCKER image inspect "$TAG" >/dev/null 2>&1; then
    $DOCKER tag "$TAG" farm-base:latest
    echo "BASE-IMAGE-DONE $TAG (it was there already)"
    exit 0
fi

echo "base image $TAG: installing the stack of $BUNDLE in the container $NAME [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
$DOCKER run -d --name "$NAME" --init -e PYTHONUNBUFFERED=1 --net=host \
    -v "$BUNDLE:/bundle" -v "$OUT:/results" "$FROM_IMAGE" sleep infinity >/dev/null
$DOCKER exec "$NAME" bash /bundle/plugin/tools/flash_next/rental/container_entry.sh \
    --install-only --results /results --run-id install
# The marker tiny_smoke.sh looks for in a container it reuses; the commit the image was built from.
$DOCKER exec "$NAME" bash -c "touch /tmp/fn-smoke.installed && printf '%s\n' '$sha' >/farm-base-sha"
$DOCKER commit "$NAME" "$TAG" >/dev/null
$DOCKER tag "$TAG" farm-base:latest
$DOCKER rm -f "$NAME" >/dev/null
echo "BASE-IMAGE-DONE $TAG [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
