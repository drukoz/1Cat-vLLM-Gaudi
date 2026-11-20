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
#
# Stage 2: one command per test. Each syncs the committed HEAD of wt/TRACK first (farm sync), runs
# the job in a container of the track's image on the granted cards, waits, and copies the job's log
# and its results folder ~/farm/TRACK/results/<job id>-<NAME> to reports/TRACK/card/.
#   farm sync TRACK                          bundle HEAD of wt/TRACK, unpack it in ~/farm/TRACK/b/<sha12>
#                                            and build the image farm-TRACK:<sha12> (once per commit)
#   farm job TRACK NAME CARDS MAX_HOLD [OPTIONS] -- COMMAND...      any command in the track's tree
#   farm tiny TRACK NAME CARDS MAX_HOLD [OPTIONS] [TINY OPTIONS]    the tiny model, TP = card count
#   farm serve TRACK NAME CARDS MAX_HOLD [OPTIONS] [SERVE OPTIONS] [-- VLLM SERVE ARGS...]
#                                            the real model, TP = card count, EP, decode probe
#   OPTIONS: --env KEY=VALUE (repeatable), --image IMG, --no-model, --no-sync, --replaces ID (take the
#     place in the queue of your own waiting job ID, which is cancelled; up to its card count in total)
#   TINY OPTIONS: --gdn MODES (pytorch,fused,hc,moe; default pytorch,fused), --full-warmup
#   SERVE OPTIONS (serve_probe.sh): --mtp K, --spec JSON, --levels LIST, --tokens N, --warm-tokens N,
#     --repeat-top N, --ctx N:TOKENS, --no-probe, --max-model-len N, --max-num-seqs N,
#     --health-timeout S, --bare, --no-ep, --tp N, --no-mintok-check (skip the probe's one round with
#     min_tokens), --min-tokens (send min_tokens in every round)
#   farm install                             (card-farm only) put the farm scripts of wt/card-farm on
#                                            the node and in bin/farm

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
    sed -n '6,37p' "$0" | sed 's/^# \{0,1\}//'
}

need_track() {
    case "${1:-}" in '' | *[!abcdefghijklmnopqrstuvwxyz0123456789-]*) die "TRACK is your track's name (lower-case letters, digits, '-'), got '${1:-}'" ;; esac
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
    local track="${1:-}" name="${2:-}" cards="${3:-}" hold="${4:-45m}" replaces="${5:-}"
    need_track "$track"
    need_word NAME "$name"
    need_word CARDS "$cards"
    need_word MAX_HOLD "$hold"
    if [ -n "$replaces" ]; then need_word REPLACES "$replaces"; fi
    if [ -t 0 ]; then die "the job's script comes on standard input (a here-document or a file)"; fi
    "$SSH" "~/farm/cardq.sh submit --track $track --name $name --cards $cards --max-hold $hold${replaces:+ --replaces $replaces}"
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

# ------------------------------------------------------------------ stage 2

FARM_FILES="cardq.sh farm_job.sh build_track_image.sh farm_unpack.sh build_base.sh"
FARM_TOOLS="serve_probe.sh decode_scaling.py farm_tiny.py kernel_test.sh"
SYNCED=""

# sync TRACK: the committed HEAD of wt/TRACK on the node, as bundle and image; its sha12 in SYNCED.
cmd_sync() {
    local track="${1:-}" sha out log tarball
    need_track "$track"
    [ -d "$G/wt/$track" ] || die "no worktree $G/wt/$track"
    sha="$(git -C "$G/wt/$track" rev-parse HEAD)" || die "wt/$track is not a git worktree"
    sha="${sha:0:12}"
    if [ -n "$(git -C "$G/wt/$track" status --porcelain --untracked-files=no)" ]; then
        echo "farm: note: wt/$track has uncommitted changes; the card runs the committed HEAD $sha without them" >&2
    fi
    if node "test -f ~/farm/$track/b/$sha/BUNDLE.txt"; then
        echo "farm: bundle of $track at $sha: on the node already"
    else
        out="$G/dist/$track"
        mkdir -p "$out"
        log="$(cd "$G/wt/$track" && bash tools/flash_next/rental/make_rental_bundle.sh --out "$out" --allow-dirty 2>&1)" \
            || { printf '%s\n' "$log" >&2; die "make_rental_bundle.sh failed in wt/$track"; }
        tarball="$(printf '%s\n' "$log" | sed -n 's/^bundle: //p')"
        [ -f "$tarball" ] || die "make_rental_bundle.sh named no bundle"
        node "mkdir -p ~/farm/$track"
        "$SCP" "$tarball" ":farm/$track/incoming-$sha.tar.gz"
        node "bash ~/farm/farm_unpack.sh --sha $track ~/farm/$track/incoming-$sha.tar.gz"
    fi
    node "bash ~/farm/build_track_image.sh $track $sha && ln -sfn b/$sha ~/farm/$track/current"
    SYNCED="$sha"
}

need_hold() {
    case "${1:-}" in
        [0-9]*s | [0-9]*m | [0-9]*h | [0-9]*) need_word MAX_HOLD "$1" ;;
        *) die "MAX_HOLD is seconds or a number with s, m or h (20m), got '${1:-}'" ;;
    esac
}

# submit_job TRACK NAME CARDS HOLD SCRIPT: queue SCRIPT, wait, fetch the log and results/<id>-NAME,
# print the end of the log; returns the job's exit code.
submit_job() {
    local track="$1" name="$2" cards="$3" hold="$4" script="$5" replaces="${6:-}" id rc=0 dest
    id="$(printf '%s\n' "$script" | cmd_submit "$track" "$name" "$cards" "$hold" "$replaces")" || die "the submit failed"
    echo "farm: job $id (log: farm log $id -f; results: reports/$track/card/$id-$name/)"
    cmd_wait "$id" || rc=$?
    dest="$G/reports/$track/card"
    if node "test -d ~/farm/$track/results/$id-$name"; then
        cmd_fetch "$track" "$id" "$id-$name" || echo "farm: the fetch failed; run: farm fetch $track $id $id-$name" >&2
    else
        cmd_fetch "$track" "$id" || echo "farm: the fetch failed; run: farm fetch $track $id" >&2
    fi
    if [ -f "$dest/$id-$name.log" ]; then
        echo "farm: the end of the log of job $id:"
        tail -n 25 "$dest/$id-$name.log" | sed 's/^/  /'
    fi
    echo "farm: job $id ended with exit code $rc"
    return "$rc"
}

# job_line WORDS...: one shell line that runs WORDS (each quoted for bash).
job_line() {
    local w line=""
    for w in "$@"; do line="$line $(printf '%q' "$w")"; done
    printf '%s\n' "${line# }"
}

# cmd_stage2 KIND TRACK NAME CARDS HOLD ARGS...: farm job, farm tiny and farm serve.
cmd_stage2() {
    local kind="$1" track="${2:-}" name="${3:-}" cards="${4:-}" hold="${5:-}" sync=1 script replaces=""
    local -a fj=() inner=() extra=()
    [ $# -ge 5 ] || die "usage: farm $kind TRACK NAME CARDS MAX_HOLD [OPTIONS] ... (farm help)"
    shift 5
    need_track "$track"
    need_word NAME "$name"
    need_word CARDS "$cards"
    need_hold "$hold"
    while [ $# -gt 0 ]; do
        case "$1" in
            --env)
                [ $# -ge 2 ] || die "--env needs KEY=VALUE"
                case "$2" in [A-Za-z_]*=*) ;; *) die "--env takes KEY=VALUE, got '$2'" ;; esac
                fj+=(--env "$2")
                shift 2
                ;;
            --image)
                [ $# -ge 2 ] || die "--image needs a value"
                fj+=(--image "$2")
                shift 2
                ;;
            --no-model)
                fj+=(--no-model)
                shift
                ;;
            --no-sync)
                sync=0
                shift
                ;;
            --replaces)
                [ $# -ge 2 ] || die "--replaces needs the id of your waiting (or just cancelled) job"
                replaces="$2"
                shift 2
                ;;
            --)
                shift
                extra=("$@")
                break
                ;;
            *)
                case "$kind" in
                    tiny)
                        case "$1" in
                            --gdn | --health-timeout | --tp)
                                [ $# -ge 2 ] || die "$1 needs a value"
                                inner+=("$1" "$2")
                                shift 2
                                ;;
                            --full-warmup | --no-recipe-cache)
                                inner+=("$1")
                                shift
                                ;;
                            *) die "farm tiny: unknown option $1" ;;
                        esac
                        ;;
                    serve)
                        case "$1" in
                            --mtp | --spec | --levels | --tokens | --warm-tokens | --repeat-top | --ctx | \
                                --max-model-len | --max-num-seqs | --health-timeout | --tp)
                                [ $# -ge 2 ] || die "$1 needs a value"
                                inner+=("$1" "$2")
                                shift 2
                                ;;
                            --no-probe | --bare | --no-ep | --no-mintok-check | --min-tokens)
                                inner+=("$1")
                                shift
                                ;;
                            *) die "farm serve: unknown option $1 (vllm serve arguments follow --)" ;;
                        esac
                        ;;
                    *) die "farm job: unknown option $1 (the command follows --)" ;;
                esac
                ;;
        esac
    done
    case "$kind" in
        job) [ "${#extra[@]}" -gt 0 ] || die "farm job: the command follows -- (farm job TRACK NAME CARDS MAX_HOLD -- python -m pytest ...)" ;;
        tiny) [ "${#extra[@]}" -eq 0 ] || die "farm tiny takes no command" ;;
    esac
    if [ "$sync" -eq 1 ]; then
        cmd_sync "$track"
    else
        SYNCED="$(node "basename \$(readlink ~/farm/$track/current)")" || die "nothing synced for $track: drop --no-sync"
    fi
    fj=(--sha "$SYNCED" ${fj[@]+"${fj[@]}"})
    script="# farm $kind $track $name $cards $hold (commit $SYNCED), written $(date -u +%Y-%m-%dT%H:%M:%SZ)"$'\n'
    case "$kind" in
        job) script="${script}exec ~/farm/farm_job.sh $(job_line "${fj[@]}" -- "${extra[@]}")" ;;
        tiny) script="${script}exec ~/farm/farm_job.sh $(job_line "${fj[@]}" -- python /farm/farm_tiny.py ${inner[@]+"${inner[@]}"})" ;;
        serve) script="${script}exec ~/farm/farm_job.sh $(job_line "${fj[@]}" -- bash /farm/serve_probe.sh ${inner[@]+"${inner[@]}"} -- ${extra[@]+"${extra[@]}"})" ;;
    esac
    submit_job "$track" "$name" "$cards" "$hold" "$script" "$replaces"
}

# install: the farm scripts of wt/card-farm onto the node (each replaced by a rename, so a supervisor
# that runs the old cardq.sh keeps reading the old file) and bin/farm.
cmd_install() {
    local src="$G/wt/card-farm/tools/flash_next/rental/farm" f list=""
    for f in $FARM_FILES $FARM_TOOLS; do
        [ -f "$src/$f" ] || die "no $src/$f"
        list="$list $src/$f"
    done
    node "mkdir -p ~/farm/.incoming ~/farm/tools"
    # shellcheck disable=SC2086  # a list of files
    "$SCP" $list ":farm/.incoming/"
    node "cd ~/farm && for f in $FARM_FILES; do install -m 755 .incoming/\$f .\$f.new && mv -f .\$f.new \$f; done && for f in $FARM_TOOLS; do install -m 644 .incoming/\$f tools/.\$f.new && mv -f tools/.\$f.new tools/\$f; done && rm -rf .incoming && ls -la ~/farm ~/farm/tools"
    install -m 755 "$src/farm.sh" "$G/bin/.farm.new" && mv -f "$G/bin/.farm.new" "$G/bin/farm"
    echo "farm: installed on the node (~/farm, ~/farm/tools) and in $G/bin/farm"
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
    sync)
        shift
        cmd_sync "$@"
        echo "farm: $1 synced at $SYNCED (image farm-$1:$SYNCED)"
        ;;
    job | tiny | serve)
        kind="$1"
        shift
        cmd_stage2 "$kind" "$@"
        ;;
    install)
        cmd_install
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
