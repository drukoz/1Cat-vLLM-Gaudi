#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# build_track_image.sh TRACK SHA12: on the node, the image farm-TRACK:SHA12 of one synced commit of a
# track: the farm's base image (farm-base:latest, the toolchain and vLLM at the engine pin) with the
# track's plugin and its native kernels installed by the commit's own container_entry.sh
# --install-only, and the tiny checkpoint of the commit built in /tmp/fn-smoke-ckpt. A job from that
# commit then starts in seconds. Needs no card: the build container has no habana runtime, at most
# FARM_BUILD_CPUS cores (default 24) and runs niced, so it can run next to card jobs. One build per
# track at a time (a lock) and at most FARM_IMAGE_SLOTS builds on the node (default 3), so builds do not
# crowd the host side of card jobs; a second call for a commit that has its image returns at once. Keeps the
# newest FARM_KEEP_IMAGES images of the track (default 3) and removes older ones that no container uses.
# The bundle is ~/farm/TRACK/b/SHA12 (farm_unpack.sh --sha); logs go to ~/farm/TRACK/images/SHA12/.

set -euo pipefail

die() {
    echo "build_track_image.sh: error: $*" >&2
    exit 1
}

[ $# -eq 2 ] || die "usage: build_track_image.sh TRACK SHA12"
track="$1"
sha="$2"
case "$track" in '' | *[!abcdefghijklmnopqrstuvwxyz0123456789-]*) die "the track name is lower-case letters, digits and '-', got '$track'" ;; esac
case "$sha" in '' | *[!0123456789abcdef]*) die "SHA12 is the 12 hex digits of a commit, got '$sha'" ;; esac

FARM="${FARM_DIR:-$HOME/farm}"
DOCKER="${FARM_DOCKER:-sudo -n docker}"
BASE="${FARM_BASE_IMAGE:-farm-base:latest}"
CPUS="${FARM_BUILD_CPUS:-24}"
KEEP="${FARM_KEEP_IMAGES:-3}"
NEED_GB="${FARM_IMAGE_NEED_GB:-15}"
BUNDLE="$FARM/$track/b/$sha"
TAG="farm-$track:$sha"
NAME="farm-img-$track-$sha"
LOG_DIR="$FARM/$track/images/$sha"

[ -f "$BUNDLE/BUNDLE.txt" ] || die "no bundle $BUNDLE (the workstation's 'farm sync $track' puts it there)"
mkdir -p "$LOG_DIR"
exec 8>>"$FARM/$track/images/.lock"
if ! flock -n 8; then
    echo "build_track_image: another image build of track $track is running; waiting for it"
    flock 8
fi

if $DOCKER image inspect "$TAG" >/dev/null 2>&1; then
    $DOCKER tag "$TAG" "farm-$track:latest"
    echo "IMAGE-READY $TAG (it was there already)"
    exit 0
fi
$DOCKER image inspect "$BASE" >/dev/null 2>&1 || die "no base image $BASE on the node (bash ~/farm/build_base.sh makes it, once per node start)"

root="$($DOCKER info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
free_kb="$(df -Pk "$root" 2>/dev/null | awk 'NR == 2 {print $4}')" || free_kb=""
case "$free_kb" in
    '' | *[!0-9]*) ;;
    *)
        if [ "$free_kb" -lt $((NEED_GB * 1024 * 1024)) ]; then
            die "only $((free_kb / 1024 / 1024)) GB free under $root, $NEED_GB GB wanted: tell the card-farm track (old track images can go)"
        fi
        ;;
esac

# One of FARM_IMAGE_SLOTS node-wide build slots (lock files under ~/farm/images-slots), held to the end.
SLOTS="${FARM_IMAGE_SLOTS:-3}"
mkdir -p "$FARM/images-slots"
slot=""
waited=0
while [ -z "$slot" ]; do
    for ((i = 1; i <= SLOTS; i++)); do
        exec 7>>"$FARM/images-slots/slot$i.lock"
        if flock -n 7; then
            slot="$i"
            break
        fi
        exec 7>&-
    done
    if [ -z "$slot" ]; then
        if [ "$waited" -eq 0 ]; then echo "build_track_image: $SLOTS image builds are running on the node; waiting for a slot"; fi
        waited=1
        sleep 5
    fi
done

started="$(date -u +%s)"
echo "build_track_image: $TAG from $BASE: installing the plugin of $BUNDLE (about three minutes) [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
cleanup() { $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
$DOCKER run -d --name "$NAME" --label "farm.image-build=$track" --cpus "$CPUS" --net=host \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$BUNDLE:/bundle" -v "$LOG_DIR:/results" "$BASE" sleep infinity >/dev/null
if ! $DOCKER exec "$NAME" nice -n 10 bash /bundle/plugin/tools/flash_next/rental/container_entry.sh \
    --install-only --results /results --run-id install >"$LOG_DIR/install.log" 2>&1; then
    tail -n 30 "$LOG_DIR/install.log" >&2
    die "the install failed; log: $LOG_DIR/install.log and $LOG_DIR/install/"
fi
# The tiny checkpoint of this commit, so a tiny smoke from the image skips its one-minute build. A
# failure here only costs that minute later.
if $DOCKER exec -w /bundle/plugin -e OMP_NUM_THREADS=8 "$NAME" nice -n 10 python -c \
    "from pathlib import Path; from tools.flash_next.rental import main as m; import sys; r = m.ensure_tiny_checkpoint(Path('/tmp/fn-smoke-ckpt')); print('tiny checkpoint:', r or 'built'); sys.exit(1 if r else 0)" \
    >"$LOG_DIR/tiny-ckpt.log" 2>&1; then
    tiny=prebuilt
else
    tiny="not prebuilt (see $LOG_DIR/tiny-ckpt.log)"
fi
$DOCKER exec "$NAME" bash -c "touch /tmp/fn-smoke.installed && printf '%s\n' '$track $sha' >/farm-image"
$DOCKER commit --change 'CMD ["bash"]' "$NAME" "$TAG" >/dev/null
$DOCKER tag "$TAG" "farm-$track:latest"
echo "IMAGE-READY $TAG ($(($(date -u +%s) - started)) s, tiny checkpoint $tiny)"

# The newest KEEP images of the track stay; an older one goes only when no container uses it. Never fails
# the build.
prune() {
    local n=0 old
    $DOCKER images "farm-$track" --format '{{.Tag}} {{.CreatedAt}}' 2>/dev/null | grep -v '^latest ' \
        | sort -k2,3r | while read -r old _; do
        n=$((n + 1))
        if [ "$n" -le "$KEEP" ] || [ "$old" = "$sha" ]; then continue; fi
        if [ -n "$($DOCKER ps -aq --filter "ancestor=farm-$track:$old" 2>/dev/null)" ]; then continue; fi
        if $DOCKER rmi "farm-$track:$old" >/dev/null 2>&1; then
            echo "build_track_image: removed the old image farm-$track:$old"
        fi
    done
}
prune || true
exit 0
