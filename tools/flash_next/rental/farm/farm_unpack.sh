#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# farm_unpack.sh TRACK TARBALL: on the node, unpack a track's rental bundle into
# ~/farm/TRACK/rental, replacing the one that is there. Refused while a job of the track waits or
# runs in the card queue (it would lose its files). The old folder holds files the containers wrote
# as root, so it is removed with sudo; that is the only thing this script removes.
#
# farm_unpack.sh --sha TRACK TARBALL: unpack it into ~/farm/TRACK/b/<plugin commit, 12 characters> (the
# per-commit bundles of `farm sync`), unless that folder is there already; then delete TARBALL. A
# per-commit folder is never replaced, so jobs of the track that wait or run keep their files.

set -euo pipefail

SHA_MODE=0
if [ "${1:-}" = --sha ]; then
    SHA_MODE=1
    shift
fi
if [ $# -ne 2 ]; then
    echo "usage: $(basename "$0") [--sha] TRACK TARBALL" >&2
    exit 2
fi
track="$1"
tarball="$2"
case "$track" in
    '' | *[!abcdefghijklmnopqrstuvwxyz0123456789-]*)
        echo "error: the track name is lower-case letters, digits and '-', got '$track'" >&2
        exit 2
        ;;
esac
if [ ! -f "$tarball" ]; then
    echo "error: no file $tarball" >&2
    exit 1
fi

if [ "$SHA_MODE" -eq 1 ]; then
    root="$HOME/farm/$track/b"
    mkdir -p "$root" "$HOME/farm/$track/results"
    new="$root/.unpack.$$"
    rm -rf "$new"
    mkdir -p "$new"
    tar xzf "$tarball" -C "$new"
    if [ ! -f "$new/BUNDLE.txt" ] || [ ! -f "$new/run_rental.sh" ]; then
        rm -rf "$new"
        echo "error: $tarball is not a rental bundle (no BUNDLE.txt and run_rental.sh at its top)" >&2
        exit 1
    fi
    find "$new" -name '._*' -type f -delete # AppleDouble files of a tar made on macOS
    sha="$(sed -n 's/^plugin sha: *//p' "$new/BUNDLE.txt" | head -n 1)"
    case "$sha" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
        *)
            rm -rf "$new"
            echo "error: $tarball names no plugin commit in its BUNDLE.txt" >&2
            exit 1
            ;;
    esac
    sha="${sha:0:12}"
    if [ -d "$root/$sha" ]; then
        rm -rf "$new"
        echo "bundle of track $track at $sha: already in $root/$sha"
    else
        mv "$new" "$root/$sha"
        echo "bundle of track $track at $sha unpacked in $root/$sha:"
        sed 's/^/  /' "$root/$sha/BUNDLE.txt"
    fi
    rm -f "$tarball"
    exit 0
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
