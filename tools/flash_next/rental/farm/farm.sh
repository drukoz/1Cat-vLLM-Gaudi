#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# farm: the workstation side of the card farm (the queue cardq.sh on the rented node). It reaches
# the node only through bin/node-ssh and bin/node-scp of the work folder.
#
#   farm status                              who holds which card, who waits
#   farm bundle TRACK [--allow-dirty]        build the rental bundle of wt/TRACK (the committed tree),
#                                            copy it to the node and unpack it in ~/farm/TRACK/rental
#   farm submit TRACK NAME CARDS [MAX_HOLD]  queue the job whose script is on standard input;
#                                            prints the job id. CARDS: all, a list (0,1,2,3), any:N
#   farm wait ID                             wait for the job; the exit code is the job's
#   farm log ID [-f]                         the job's log (-f: follow it until the job ends)
#   farm fetch TRACK ID [RUN_ID...]          copy the job's log, and the named folders of
#                                            ~/farm/TRACK/results, into reports/TRACK/card/
#   farm run TRACK NAME CARDS [MAX_HOLD] [-- RUN_ID...]   submit, wait, fetch in one command
#   farm cancel TRACK ID                     stop a job of your own track
#   farm ledger                              every ended job: track, cards, start, end, seconds

# shellcheck disable=SC2088  # a ~ in a command for the node is expanded there, by the node's shell

set -euo pipefail

G="${FARM_HOME:-/Volumes/Storage/Claude/gaudi-work}"
SSH="$G/bin/node-ssh"
SCP="$G/bin/node-scp"

die() {
    echo "error: $*" >&2
    exit 1
}

usage() {
    sed -n '6,19p' "$0" | sed 's/^# \{0,1\}//'
}

need_track() {
    case "${1:-}" in '' | *[!a-z0-9-]*) die "TRACK is your track's name (lower-case letters, digits, '-'), got '${1:-}'" ;; esac
}

need_word() {
    # need_word WHAT VALUE: letters, digits, '.', '_', '-', ',' and ':' only, so no quoting is needed on the node
    case "${2:-}" in '' | *[!A-Za-z0-9._,:-]*) die "$1: letters, digits and . _ - , : only, got '${2:-}'" ;; esac
}

# node COMMAND: run it on the node; a dropped connection (exit 255) is tried again for five minutes.
node() {
    local rc=0 tries=0
    while :; do
        "$SSH" "$1" && return 0
        rc=$?
        if [ "$rc" -ne 255 ] || [ "$tries" -ge 30 ]; then return "$rc"; fi
        tries=$((tries + 1))
        echo "farm: no connection to the node, trying again in 10 s ($tries/30)" >&2
        sleep 10
    done
}

cmd_bundle() {
    local track="${1:-}" out log tarball
    need_track "$track"
    shift
    [ -d "$G/wt/$track" ] || die "no worktree $G/wt/$track"
    out="$G/dist/$track"
    mkdir -p "$out"
    log="$(cd "$G/wt/$track" && bash tools/flash_next/rental/make_rental_bundle.sh --out "$out" "$@")"
    printf '%s\n' "$log"
    tarball="$(printf '%s\n' "$log" | sed -n 's/^bundle: //p')"
    [ -f "$tarball" ] || die "make_rental_bundle.sh named no bundle"
    node "mkdir -p ~/farm/$track"
    "$SCP" "$tarball" ":farm/$track/bundle.tar.gz"
    node "bash ~/farm/farm_unpack.sh $track ~/farm/$track/bundle.tar.gz"
}

cmd_submit() {
    local track="${1:-}" name="${2:-}" cards="${3:-}" hold="${4:-45m}"
    need_track "$track"
    need_word NAME "$name"
    need_word CARDS "$cards"
    need_word MAX_HOLD "$hold"
    if [ -t 0 ]; then die "the job's script comes on standard input (a here-document or a file)"; fi
    "$SSH" "~/farm/cardq.sh submit --track $track --name $name --cards $cards --max-hold $hold"
}

cmd_wait() {
    local id="${1:-}" rc=0
    need_word ID "$id"
    while :; do
        rc=0
        node "~/farm/cardq.sh wait $id --timeout 1500" || rc=$?
        if [ "$rc" -ne 124 ]; then return "$rc"; fi
    done
}

cmd_fetch() {
    local track="${1:-}" id="${2:-}" dest name run
    need_track "$track"
    need_word ID "$id"
    shift 2
    dest="$G/reports/$track/card"
    mkdir -p "$dest"
    name="$(node "cat ~/farm/q/jobs/\$(printf %05d \$((10#$id)))/name")"
    node "~/farm/cardq.sh log $id" >"$dest/$id-$name.log"
    echo "farm: log of job $id: $dest/$id-$name.log"
    for run in "$@"; do
        need_word RUN_ID "$run"
        node "cd ~/farm/$track/results && sudo -n tar czf - --exclude='*.safetensors' $run" | tar xzf - -C "$dest"
        echo "farm: results folder: $dest/$run"
    done
}

cmd_run() {
    local track="${1:-}" name="${2:-}" cards="${3:-}" hold="45m" id rc=0
    [ $# -ge 3 ] || die "usage: farm run TRACK NAME CARDS [MAX_HOLD] [-- RUN_ID...]"
    shift 3
    if [ $# -gt 0 ] && [ "$1" != "--" ]; then
        hold="$1"
        shift
    fi
    if [ "${1:-}" = "--" ]; then shift; fi
    id="$(cmd_submit "$track" "$name" "$cards" "$hold")"
    echo "farm: job $id"
    cmd_wait "$id" || rc=$?
    cmd_fetch "$track" "$id" "$@" || echo "farm: the fetch failed; run: farm fetch $track $id $*" >&2
    return "$rc"
}

case "${1:-}" in
    status | ledger)
        node "~/farm/cardq.sh $1"
        ;;
    bundle)
        shift
        cmd_bundle "$@"
        ;;
    submit)
        shift
        cmd_submit "$@"
        ;;
    wait)
        shift
        cmd_wait "$@"
        ;;
    log)
        need_word ID "${2:-}"
        if [ "${3:-}" = -f ]; then node "~/farm/cardq.sh log $2 -f"; else node "~/farm/cardq.sh log $2"; fi
        ;;
    fetch)
        shift
        cmd_fetch "$@"
        ;;
    run)
        shift
        cmd_run "$@"
        ;;
    cancel)
        need_track "${2:-}"
        need_word ID "${3:-}"
        node "~/farm/cardq.sh cancel --track $2 $3"
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
