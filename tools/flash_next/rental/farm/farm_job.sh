#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# farm_job.sh: inside a job of the card queue (cardq.sh), run one command in a container of the
# track's image on exactly the cards the queue granted. The workstation's `farm job`, `farm tiny` and
# `farm serve` write job scripts that call it; it can also be called from a job script by hand.
#
#   farm_job.sh [--sha SHA12] [--run-id ID] [--env KEY=VALUE]... [--image IMG] [--suffix S]
#               [--cards LIST] [--port-base N] [--no-model] [--dry-run] -- COMMAND...
#
# --sha: the track's commit as `farm sync` left it on the node: the bundle ~/farm/<track>/b/<sha12>
# and the image farm-<track>:<sha12> (the stack installed, so the container starts in seconds; made
# here, holding the cards for about two minutes, when it is missing). Default: the commit the track
# synced last (~/farm/<track>/current). --image runs another image with the same bundle.
# --cards LIST runs the container on part of the job's cards only (several containers side by side in one
# job: give each its own --cards, --suffix and --run-id); its ports then start at 8100 + 100 x its
# lowest card, as a queue job of those cards would get them (--port-base N sets them).
#
# The container is named farm-<job id>[-S], labelled cardq.job=<job id> (the queue removes it when the
# job ends), mounts only the granted cards (HABANA_VISIBLE_DEVICES = the card indices,
# HABANA_VISIBLE_MODULES = their module ids), shares the host network and IPC, and sees:
#   /bundle        the track's bundle, read-only; the work folder is /bundle/plugin (the track's tree)
#   /results       ~/farm/<track>/results; FARM_OUT=/results/<run id> (default run id: <job id>-<job name>)
#   /recipe-cache  a compiled-recipe cache of this commit and card count, in RAM (FARM_RECIPE_CACHE)
#   /farm          the farm's tools (serve_probe.sh, decode_scaling.py, farm_tiny.py, kernel_test.sh), read-only
#   /models/Qwen3.8-Flash-Next-FP8   the checkpoint, read-only (left out by --no-model)
# Environment: FARM_OUT, FARM_RUN_ID, FARM_CARDS, FARM_MODULES, FARM_NCARDS, FARM_PORT_BASE (this job
# owns the 100 ports from there; with --net=host every server must take its port from there),
# FARM_TRACK, FARM_JOB, FARM_SHA, FARM_RECIPE_CACHE, and PT_HPU_LAZY_MODE=1,
# PT_HPU_LAZY_ACC_PAR_MODE=0, PYTHONFAULTHANDLER=1, HABANA_LOGS=$FARM_OUT/habana_logs. Every --env
# KEY=VALUE comes last and wins over these. The exit code is the command's.
#
# CPUs: a container of fewer than all cards gets its cards' share of the host CPUs (module m: CPUs
# m x C/8 .. (m+1) x C/8 - 1, so modules 0-3 get the first NUMA half on the 2026-11-20 node). vLLM sizes
# its workers' weight-loading threads by the CPUs a process may use (available_cpu_count: affinity and
# cgroup quota), so two 4-card servers then share the host instead of each giving its 4 workers 152/4
# threads: that oversubscription made a checkpoint shard take 55 s instead of 4 s (engine-step 00030 and
# 00031, 18:39 UTC). OMP_NUM_THREADS stays unset on purpose: set from outside, vLLM would keep that many
# torch threads for serving instead of 1. --all-cpus keeps the whole host; a job of all cards has it.

set -euo pipefail

die() {
    echo "farm_job.sh: error: $*" >&2
    exit 2
}

SHA=""
RUN_ID=""
IMAGE=""
SUFFIX=""
CARDS=""
PORT_BASE=""
MODEL_DIR="${FARM_MODEL_DIR:-/models/Qwen3.8-Flash-Next-FP8}"
WITH_MODEL=1
ALL_CPUS=0
DRY_RUN=0
EXTRA_ENV=()
while [ $# -gt 0 ]; do
    case "$1" in
        --sha)
            [ $# -ge 2 ] || die "--sha needs a value"
            SHA="$2"
            shift 2
            ;;
        --run-id)
            [ $# -ge 2 ] || die "--run-id needs a value"
            RUN_ID="$2"
            shift 2
            ;;
        --env)
            [ $# -ge 2 ] || die "--env needs KEY=VALUE"
            case "$2" in [A-Za-z_]*=*) ;; *) die "--env takes KEY=VALUE, got '$2'" ;; esac
            EXTRA_ENV+=("$2")
            shift 2
            ;;
        --image)
            [ $# -ge 2 ] || die "--image needs a value"
            IMAGE="$2"
            shift 2
            ;;
        --suffix)
            [ $# -ge 2 ] || die "--suffix needs a value"
            SUFFIX="$2"
            shift 2
            ;;
        --cards)
            [ $# -ge 2 ] || die "--cards needs a list of the job's cards"
            CARDS="$2"
            shift 2
            ;;
        --port-base)
            [ $# -ge 2 ] || die "--port-base needs a port"
            PORT_BASE="$2"
            shift 2
            ;;
        --no-model)
            WITH_MODEL=0
            shift
            ;;
        --all-cpus)
            ALL_CPUS=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *) die "unknown option: $1 (the command follows --)" ;;
    esac
done
[ $# -ge 1 ] || die "no command after --"

[ -n "${CARDQ_JOB:-}" ] && [ -n "${CARDQ_CARDS:-}" ] && [ -n "${CARDQ_TRACK:-}" ] \
    || die "not inside a job of the card queue (CARDQ_JOB, CARDQ_CARDS and CARDQ_TRACK are not set): submit it through the queue (farm job ...)"
case "$CARDQ_TRACK" in *[!abcdefghijklmnopqrstuvwxyz0123456789-]*) die "bad track name '$CARDQ_TRACK'" ;; esac
case "$CARDQ_CARDS" in *[!0-9,]*) die "bad card list '$CARDQ_CARDS'" ;; esac
case "$CARDQ_JOB" in *[!0-9]*) die "bad job id '$CARDQ_JOB'" ;; esac
case "$SHA" in *[!0123456789abcdef]*) die "--sha takes the 12 hex digits of a synced commit, got '$SHA'" ;; esac
case "$SUFFIX" in *[!A-Za-z0-9._-]*) die "--suffix: letters, digits, '.', '_' and '-' only" ;; esac

case "$PORT_BASE" in *[!0-9]*) die "--port-base takes a port number, got '$PORT_BASE'" ;; esac

FARM="${FARM_DIR:-$HOME/farm}"
TRACK_DIR="$FARM/$CARDQ_TRACK"
DOCKER="${FARM_DOCKER:-sudo -n docker}"

# The cards of this container: the job's, or the part --cards names (each one granted to the job).
if [ -n "$CARDS" ]; then
    case "$CARDS" in *[!0-9,]* | ,* | *, | *,,*) die "--cards takes a list such as 0,1,2,7, got '$CARDS'" ;; esac
    for c in ${CARDS//,/ }; do
        case ",$CARDQ_CARDS," in *",$c,"*) ;; *) die "--cards $CARDS: card $c is not one of this job's cards ($CARDQ_CARDS)" ;; esac
    done
else
    CARDS="$CARDQ_CARDS"
fi
NCARDS=0
lowest=""
for c in ${CARDS//,/ }; do
    NCARDS=$((NCARDS + 1))
    if [ -z "$lowest" ] || [ "$c" -lt "$lowest" ]; then lowest="$c"; fi
done
if [ -z "$PORT_BASE" ]; then
    if [ "$CARDS" = "$CARDQ_CARDS" ] && [ -n "${CARDQ_PORT_BASE:-}" ]; then
        PORT_BASE="$CARDQ_PORT_BASE"
    else
        PORT_BASE=$((${CARDQ_PORT_FIRST:-8100} + 100 * lowest))
    fi
fi
# The module id of each card (cardq.sh writes the map of the node to its queue folder); the module ids of
# this container's cards, ascending, are what the habana runtime sets in it as HABANA_VISIBLE_MODULES.
MAP="${CARDQ_MODULE_MAP:-$(head -n 1 "${CARDQ_DIR:-$FARM/q}/module_map" 2>/dev/null || true)}"
MODULES=""
for ((m = 0; m < 64; m++)); do
    i=0
    for mod in ${MAP//,/ }; do
        if [ "$mod" = "$m" ] && case ",$CARDS," in *",$i,"*) true ;; *) false ;; esac; then
            MODULES="$MODULES${MODULES:+,}$m"
        fi
        i=$((i + 1))
    done
done
if [ -z "$MAP" ]; then MODULES="$CARDS"; fi  # no map: card N is module N

# The container's CPUs (see the header): its modules' slices of the host's CPUs.
HOST_CPUS="${FARM_HOST_CPUS:-$(nproc 2>/dev/null || echo 0)}"
TOTAL_CARDS="${CARDQ_NCARDS_TOTAL:-8}"
CPUSET=""
PER_CARD_CPUS=0
if [ "$ALL_CPUS" -eq 0 ] && [ "$NCARDS" -lt "$TOTAL_CARDS" ] && [ "$HOST_CPUS" -ge "$TOTAL_CARDS" ]; then
    PER_CARD_CPUS=$((HOST_CPUS / TOTAL_CARDS))
    for m in ${MODULES//,/ }; do
        CPUSET="$CPUSET${CPUSET:+,}$((m * PER_CARD_CPUS))-$(((m + 1) * PER_CARD_CPUS - 1))"
    done
fi

if [ -z "$SHA" ]; then
    if [ -L "$TRACK_DIR/current" ]; then
        SHA="$(basename "$(readlink "$TRACK_DIR/current")")"
    fi
    [ -n "$SHA" ] || die "no commit of track $CARDQ_TRACK is synced: run 'farm sync $CARDQ_TRACK' on the workstation"
fi
BUNDLE="$TRACK_DIR/b/$SHA"
[ -f "$BUNDLE/BUNDLE.txt" ] || die "no bundle $BUNDLE: run 'farm sync $CARDQ_TRACK' on the workstation (it syncs the committed HEAD)"

RUN_ID="${RUN_ID:-$CARDQ_JOB-${CARDQ_NAME:-job}}"
case "$RUN_ID" in '' | *[!A-Za-z0-9._-]*) die "--run-id: letters, digits, '.', '_' and '-' only, got '$RUN_ID'" ;; esac
IMAGE="${IMAGE:-farm-$CARDQ_TRACK:$SHA}"
NAME="farm-$CARDQ_JOB${SUFFIX:+-$SUFFIX}"
RECIPE_DIR="${FARM_RECIPE_ROOT:-/dev/shm/farm-rc}/$CARDQ_TRACK/$SHA-${NCARDS}c"

# has_env KEY: 0 when an --env sets KEY (then a default of this script for KEY is left out).
has_env() {
    local kv
    for kv in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
        if [ "${kv%%=*}" = "$1" ]; then return 0; fi
    done
    return 1
}

CMD=($DOCKER run --rm --name "$NAME"
    --label "cardq.job=$CARDQ_JOB" --label "cardq.track=$CARDQ_TRACK"
    --init --runtime=habana --cap-add=sys_nice --net=host --ipc=host
    -w /bundle/plugin
    -v "$BUNDLE:/bundle:ro"
    -v "$TRACK_DIR/results:/results"
    -v "$RECIPE_DIR:/recipe-cache"
    -v "$FARM/tools:/farm:ro")
if [ "$WITH_MODEL" -eq 1 ]; then
    CMD+=(-v "$MODEL_DIR:$MODEL_DIR:ro")
fi
if [ -n "$CPUSET" ]; then
    CMD+=(--cpuset-cpus "$CPUSET")
fi
for kv in "HABANA_VISIBLE_DEVICES=$CARDS" "HABANA_VISIBLE_MODULES=$MODULES" \
    OMPI_MCA_btl_vader_single_copy_mechanism=none PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 \
    "HABANA_LOGS=/results/$RUN_ID/habana_logs" "FARM_OUT=/results/$RUN_ID" "FARM_RUN_ID=$RUN_ID" \
    "FARM_CARDS=$CARDS" "FARM_MODULES=$MODULES" "FARM_NCARDS=$NCARDS" \
    "FARM_PORT_BASE=$PORT_BASE" "FARM_TRACK=$CARDQ_TRACK" "FARM_JOB=$CARDQ_JOB" \
    "FARM_SHA=$SHA" FARM_RECIPE_CACHE=/recipe-cache; do
    case "${kv%%=*}" in
        HABANA_VISIBLE_DEVICES | HABANA_VISIBLE_MODULES | FARM_*) ;; # the job's cards and folders: never replaced
        *) if has_env "${kv%%=*}"; then continue; fi ;;
    esac
    CMD+=(-e "$kv")
done
for kv in ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}; do
    case "${kv%%=*}" in
        HABANA_VISIBLE_DEVICES | HABANA_VISIBLE_MODULES | FARM_*)
            die "--env ${kv%%=*}: set by the farm from the job's cards and folders"
            ;;
    esac
    CMD+=(-e "$kv")
done
CMD+=("$IMAGE" "$@")

if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

if ! $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; then
    case "$IMAGE" in
        "farm-$CARDQ_TRACK:$SHA")
            echo "farm_job: no image $IMAGE yet: building it now (about two minutes of this job's time; 'farm sync' builds it before the job)"
            bash "$FARM/build_track_image.sh" "$CARDQ_TRACK" "$SHA"
            ;;
        *) die "no image $IMAGE on the node" ;;
    esac
fi
mkdir -p "$TRACK_DIR/results/$RUN_ID/habana_logs" "$RECIPE_DIR"
echo "farm_job: container $NAME, image $IMAGE, cards $CARDS (modules $MODULES), ports $PORT_BASE+, CPUs ${CPUSET:-all}, results ~/farm/$CARDQ_TRACK/results/$RUN_ID"
exec "${CMD[@]}" </dev/null
