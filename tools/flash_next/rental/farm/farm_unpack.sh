#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# farm_unpack.sh TRACK TARBALL: on the node, unpack a track's rental bundle into
# ~/farm/TRACK/rental, replacing the one that is there. Refused while a job of the track waits or
# runs in the card queue (it would lose its files). The old folder holds files the containers wrote
# as root, so it is removed with sudo; that is the only thing this script removes.

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $(basename "$0") TRACK TARBALL" >&2
    exit 2
fi
track="$1"
tarball="$2"
case "$track" in
    '' | *[!a-z0-9-]*)
        echo "error: the track name is lower-case letters, digits and '-', got '$track'" >&2
        exit 2
        ;;
esac
if [ ! -f "$tarball" ]; then
    echo "error: no file $tarball" >&2
    exit 1
fi

Q="${CARDQ_DIR:-$HOME/farm/q}"
for d in "$Q"/jobs/*/; do
    [ -f "$d/state" ] || continue
    case "$(cat "$d/state")" in waiting | running) ;; *) continue ;; esac
    if [ "$(cat "$d/track" 2>/dev/null)" = "$track" ] && kill -0 "$(cat "$d/pid" 2>/dev/null || echo 0)" 2>/dev/null; then
        echo "error: job $(basename "$d") of track $track is $(cat "$d/state"): it runs from the bundle; wait for it or cancel it first" >&2
        exit 3
    fi
done

dir="$HOME/farm/$track"
new="$dir/rental.new.$$"
mkdir -p "$new"
tar xzf "$tarball" -C "$new"
if [ ! -f "$new/BUNDLE.txt" ] || [ ! -f "$new/run_rental.sh" ]; then
    rm -rf "$new"
    echo "error: $tarball is not a rental bundle (no BUNDLE.txt and run_rental.sh at its top)" >&2
    exit 1
fi
find "$new" -name '._*' -type f -delete # AppleDouble files of a tar made on macOS
if [ -d "$dir/rental" ]; then
    old="$dir/rental.old.$$"
    mv "$dir/rental" "$old"
    sudo -n rm -rf -- "$old"
fi
mv "$new" "$dir/rental"
mkdir -p "$dir/results"
echo "bundle of track $track unpacked in $dir/rental:"
sed 's/^/  /' "$dir/rental/BUNDLE.txt"
