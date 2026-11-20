#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Make the Flash-Next rental bundle (RUK-2709 M4): one tarball holding the plugin at its commit,
# vLLM at the engine pin, and run_rental.sh, prepare_host.sh and tiny_smoke.sh at the top level. Copy it to the
# rented node, unpack, prepare a bare host with `sudo bash prepare_host.sh`, then run
# `bash run_rental.sh --model <checkpoint dir>`.
#
# Usage: bash tools/flash_next/rental/make_rental_bundle.sh [--out DIR] [--vllm-src PATH] [--allow-dirty]

set -euo pipefail

VLLM_PIN="8f816a3f665489d7f0d222115d4f72ebab01076b"
DEFAULT_VLLM_SRC="/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/vllm-src"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") [--out DIR] [--vllm-src PATH] [--allow-dirty]"
    echo "  --out DIR        output directory (default: dist/)"
    echo "  --vllm-src PATH  git checkout of vLLM (default: $DEFAULT_VLLM_SRC)"
    echo "  --allow-dirty    bundle HEAD even when the working tree is not clean"
}

OUT_DIR="dist"
VLLM_SRC="$DEFAULT_VLLM_SRC"
ALLOW_DIRTY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --vllm-src)
            VLLM_SRC="$2"
            shift 2
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
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

if [ "$ALLOW_DIRTY" -eq 0 ] && [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    echo "error: the working tree is not clean; commit first or pass --allow-dirty" >&2
    exit 1
fi

if [ ! -d "$VLLM_SRC/.git" ]; then
    echo "error: no git checkout of vLLM at $VLLM_SRC (pass --vllm-src)" >&2
    exit 1
fi
if ! git -C "$VLLM_SRC" cat-file -e "${VLLM_PIN}^{commit}" 2>/dev/null; then
    echo "error: vLLM commit $VLLM_PIN not found in $VLLM_SRC" >&2
    exit 1
fi

PLUGIN_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
PLUGIN_SHA12="${PLUGIN_SHA:0:12}"
PLUGIN_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/plugin"
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$STAGE/plugin"
git -C "$VLLM_SRC" archive --prefix=vllm/ "$VLLM_PIN" | tar -x -C "$STAGE"
cp "$SCRIPT_DIR/run_rental.sh" "$STAGE/run_rental.sh"
cp "$SCRIPT_DIR/prepare_host.sh" "$STAGE/prepare_host.sh"
cp "$SCRIPT_DIR/tiny_smoke.sh" "$STAGE/tiny_smoke.sh"

# On the node plugin/build is the build folder of the plugin install: container_entry.sh removes it
# whenever BUNDLE.txt names another commit than the one that made it. The bundle must therefore
# carry nothing there (git archive holds tracked files only, and build/ is ignored; this keeps it so).
if [ -e "$STAGE/plugin/build" ]; then
    echo "error: the commit tracks files under build/; the node removes plugin/build on a bundle change, move them" >&2
    exit 1
fi
# plugin/ is the commit; the three top-level scripts are the working tree's. With --allow-dirty they
# can differ from the commit's, and run_rental.sh would then meet another container_entry.sh.
for name in run_rental.sh prepare_host.sh tiny_smoke.sh; do
    if ! cmp -s "$STAGE/$name" "$STAGE/plugin/tools/flash_next/rental/$name"; then
        echo "warning: $name at the bundle top is the uncommitted working-tree file; plugin/ (container_entry.sh, the orchestrator) is commit $PLUGIN_SHA12" >&2
    fi
done

{
    echo "plugin sha: $PLUGIN_SHA"
    echo "branch: $PLUGIN_BRANCH"
    echo "vllm commit: $VLLM_PIN"
    echo "build date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$STAGE/BUNDLE.txt"

mkdir -p "$OUT_DIR"
TARBALL="$(cd "$OUT_DIR" && pwd)/flash-next-rental-$PLUGIN_SHA12.tar.gz"
tar czf "$TARBALL" -C "$STAGE" BUNDLE.txt run_rental.sh prepare_host.sh tiny_smoke.sh plugin vllm

echo "bundle: $TARBALL"
echo "size: $(du -h "$TARBALL" | cut -f1)"
