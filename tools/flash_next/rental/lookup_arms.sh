#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# lookup_arms.sh: several lookup_serve.sh servers one after the other in ONE farm job (the spec-lookup track's
# A/B on the same cards and build: no speculation, n-gram, the lookup + MTP hybrid), with hl-smi's device memory
# every 15 s. Each arm writes to $OUT/<arm>/ (lookup_serve.sh's files); an arm that fails does not stop the next.
#
#   bash tools/flash_next/rental/lookup_arms.sh [--dry-run] ARM...
#     nospec          no speculation, rounds LOOKUP_ARMS_ROUNDS (default
#                     edit:1:384:-:100,edit:32:384:-:0,mix:32:384:-:400: no two rounds share a prompt)
#     ngram<K>        n-gram k = K, the same rounds
#     ngram<K>-nospec n-gram k = K and the no-speculation reference on ONE server: every round twice, n-gram (no
#                     mode) on the prompts FIRST + 300 on, then with the request's lookup off (vllm_xargs
#                     spec_lookup 0: it drafts nothing) on the round's own prompts, with
#                     VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS=1 unless the job set it, so a step of such requests is a
#                     plain decode step. The reference rounds use the prompts of the hybrid's mode 1 rounds.
#     hybrid<K>       MTP k = K with VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP=1; every round twice, the request's mode 1
#                     (hybrid, the rounds' own prompts) then 0 (MTP alone, prompts FIRST + 300 on), on the same
#                     server. A second request on a prompt crashed the real model's n-gram server in job 00133
#                     (fused and torch verification alike) once it decoded past the first recurrent block.
#     tiny-off        the tiny-model gate of that reference at the job's TP (lookup_tiny.py --configs ngram5-off):
#                     requests with the lookup off on an n-gram server give the tokens of a server without
#                     speculation, run no verification step (the step timer's kinds) and the same decode graph
#                     shapes. When it fails, every later ngram<K>-nospec arm runs as nospec and ngram<K> (two
#                     servers); LOOKUP_ARMS_SPLIT=1 does that from the start.
# Verification: VLLM_HPU_GDN_CORE_VERIFY as the job sets it (lookup_serve.sh's preset: 0, the torch path). The MTP
# arms also get VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1 unless the job set them
# (the integration build's MTP server). Every other flag comes from the job's environment (farm --env).
#
# Worker placement, switched late (async-sched's switch, coordinator 07:02 UTC): the first word of
# LOOKUP_ARMS_PLACE_SWITCH (default /results/switches/place, read once here at the job's start), "on" or "a^1"
# (rule a^1) or "a+N/2"; anything else or no file: off. On: every real-model server runs with
# VLLM_HPU_WORKER_CPU_PIN=1 under LOOKUP_ARMS_PLACE_WRAP LOOKUP_ARMS_PLACE_SCRIPT (default
# /results/bench/place_wrap.sh /results/bench/place.sh: collectives' wrapper, native-replay's sibling-aware
# placement), its output in the arm's placement.log. The tiny gate is never placed.
# Exit code: the number of arms that did not end with 0 (2: a usage error).

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FARM_PYTHON:-python3}"
OUT="${FARM_OUT:-/results/lookup-arms}"
ROUNDS="${LOOKUP_ARMS_ROUNDS:-edit:1:384:-:100,edit:32:384:-:0,mix:32:384:-:400}"
PLACE_SWITCH="${LOOKUP_ARMS_PLACE_SWITCH:-/results/switches/place}"
PLACE_WRAP="${LOOKUP_ARMS_PLACE_WRAP:-/results/bench/place_wrap.sh}"
PLACE_SCRIPT="${LOOKUP_ARMS_PLACE_SCRIPT:-/results/bench/place.sh}"
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
if [ $# -eq 0 ]; then
    echo "usage: lookup_arms.sh [--dry-run] ARM... (nospec, ngram<K>, ngram<K>-nospec, hybrid<K>, tiny-off)" >&2
    exit 2
fi
for arm in "$@"; do
    case "$arm" in
        nospec | ngram[1-9] | ngram[1-9]-nospec | hybrid[1-5] | tiny-off) ;;
        *) echo "lookup_arms: unknown arm $arm" >&2; exit 2 ;;
    esac
done

# a round list with every round twice: the request's mode 1 on the round's prompts, then 0 (MTP alone) on the
# prompts 300 on (KIND:STREAMS:TOKENS:MODE[:FIRST])
modes10() {
    local out="" r kind n ntok mode first
    IFS=',' read -r -a parts <<<"$1"
    for r in "${parts[@]}"; do
        IFS=':' read -r kind n ntok mode first <<<"$r"
        first="${first:-0}"
        out="$out,$kind:$n:$ntok:1:$first,$kind:$n:$ntok:0:$((first + 300))"
    done
    echo "${out#,}"
}

# a round list with every round twice: n-gram (no mode) on the prompts 300 on, then the lookup off (mode 0) on the
# round's own prompts
ngram_off() {
    local out="" r kind n ntok mode first
    IFS=',' read -r -a parts <<<"$1"
    for r in "${parts[@]}"; do
        IFS=':' read -r kind n ntok mode first <<<"$r"
        first="${first:-0}"
        out="$out,$kind:$n:$ntok:-:$((first + 300)),$kind:$n:$ntok:0:$first"
    done
    echo "${out#,}"
}

arm_command() {  # sets CMD to the arm's command and EXTRA_ENV to its extra environment
    local arm="$1" k
    EXTRA_ENV=(LOOKUP_ARM="$arm")
    case "$arm" in
        tiny-off)
            CMD=("$PYTHON" "$HERE/lookup_tiny.py" --configs ngram5-off --out "$OUT/$arm")
            return
            ;;
        nospec) CMD=(bash "$HERE/lookup_serve.sh" --label nospec --rounds "$ROUNDS") ;;
        ngram[1-9]) CMD=(bash "$HERE/lookup_serve.sh" --label "$arm" --ngram "${arm#ngram}" --rounds "$ROUNDS") ;;
        ngram*-nospec)
            k="${arm%-nospec}"
            CMD=(bash "$HERE/lookup_serve.sh" --label "$arm" --ngram "${k#ngram}" --rounds "$(ngram_off "$ROUNDS")")
            if [ -z "${VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS+x}" ]; then
                EXTRA_ENV+=(VLLM_HPU_FLASH_NEXT_NGRAM_PLAIN_STEPS=1)
            fi
            ;;
        hybrid*)
            CMD=(bash "$HERE/lookup_serve.sh" --label "$arm" --mtp "${arm#hybrid}" --lookup 1 --rounds
                "$(modes10 "$ROUNDS")")
            for kv in VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1; do
                key="${kv%%=*}"
                if [ -z "${!key+x}" ]; then EXTRA_ENV+=("$kv"); fi
            done
            ;;
    esac
    CMD+=(--out "$OUT/$arm")
}

PLACE_RULE=""
if [ -r "$PLACE_SWITCH" ]; then
    case "$(awk 'NR == 1 { print $1 }' "$PLACE_SWITCH")" in
        on | "a^1") PLACE_RULE="a^1" ;;
        "a+N/2") PLACE_RULE="a+N/2" ;;
    esac
fi
if [ -n "$PLACE_RULE" ] && { [ ! -r "$PLACE_WRAP" ] || [ ! -r "$PLACE_SCRIPT" ]; }; then
    echo "lookup_arms: placement switch on but $PLACE_WRAP or $PLACE_SCRIPT is missing: placement OFF"
    PLACE_RULE=""
fi
if [ -n "$PLACE_RULE" ]; then
    echo "lookup_arms: placement ON (rule $PLACE_RULE; switch $PLACE_SWITCH; md5" \
        "$(md5sum "$PLACE_WRAP" "$PLACE_SCRIPT" 2>/dev/null | awk '{ printf "%s %s ", $2, substr($1, 1, 8) }'))"
else
    echo "lookup_arms: placement off (switch $PLACE_SWITCH: $(head -c 40 "$PLACE_SWITCH" 2>/dev/null || echo none))"
fi

MON=""
if [ "$DRY" -eq 0 ]; then
    mkdir -p "$OUT" || { echo "lookup_arms: cannot use $OUT" >&2; exit 2; }
    if command -v hl-smi >/dev/null 2>&1; then
        (while :; do
            printf '%s %s\n' "$(date -u +%H:%M:%S)" \
                "$(hl-smi -Q index,memory.used,memory.total -f csv,noheader 2>&1 | tr '\n' ';')"
            sleep 15
        done) >"$OUT/hlsmi.log" 2>&1 &
        MON=$!
    fi
fi
stop_monitor() {
    if [ -n "$MON" ]; then kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null; MON=""; fi
}
trap stop_monitor EXIT

failed=0
first=1
gate_failed=0
if [ "${LOOKUP_ARMS_SPLIT:-0}" = 1 ]; then gate_failed=1; fi
queue=("$@")
i=0
while [ "$i" -lt "${#queue[@]}" ]; do
    arm="${queue[$i]}"
    i=$((i + 1))
    case "$arm" in
        ngram*-nospec)
            if [ "$gate_failed" -eq 1 ]; then
                k="${arm%-nospec}"
                echo "lookup_arms: the tiny gate failed or LOOKUP_ARMS_SPLIT=1: $arm runs as nospec and $k (two servers)"
                queue=("${queue[@]:0:$i}" nospec "$k" "${queue[@]:$i}")
                continue
            fi
            ;;
    esac
    arm_command "$arm"
    placed=()
    if [ -n "$PLACE_RULE" ] && [ "$arm" != tiny-off ]; then
        EXTRA_ENV+=(VLLM_HPU_WORKER_CPU_PIN=1 PLACE_SIBLING_RULE="$PLACE_RULE" PLACE_ARGS="--log $OUT/$arm/server0.log"
            PLACE_LOG="$OUT/$arm/placement.log")
        placed=(bash "$PLACE_WRAP" "$PLACE_SCRIPT" --)
    fi
    if [ "$DRY" -eq 1 ]; then
        printf 'arm %s:' "$arm"
        printf ' %q' env "${EXTRA_ENV[@]}" ${placed[@]+"${placed[@]}"} "${CMD[@]}"
        printf '\n'
        continue
    fi
    if [ "$first" -eq 0 ]; then
        # the previous server's processes must be gone before the next takes the cards (CARD.md, point 5); the
        # bracket keeps this command line itself out of the pattern
        pkill -9 -f "VLL[M]::" 2>/dev/null
        sleep 15
    fi
    first=0
    echo "lookup_arms: arm $arm: $(date -u +%H:%M:%S)"
    mkdir -p "$OUT/$arm"
    env "${EXTRA_ENV[@]}" ${placed[@]+"${placed[@]}"} "${CMD[@]}"
    rc=$?
    echo "lookup_arms: arm $arm ended with rc $rc: $(date -u +%H:%M:%S)"
    if [ "$rc" -ne 0 ]; then
        failed=$((failed + 1))
        if [ "$arm" = tiny-off ]; then gate_failed=1; fi
    fi
done
exit "$failed"
