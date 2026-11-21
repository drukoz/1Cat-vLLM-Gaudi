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
#     hybrid<K>       MTP k = K with VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP=1; every round twice, the request's mode 1
#                     (hybrid, the rounds' own prompts) then 0 (MTP alone, prompts FIRST + 300 on), on the same
#                     server. A second request on a prompt crashed the real model's n-gram server in job 00133
#                     (fused and torch verification alike) once it decoded past the first recurrent block.
# Verification: VLLM_HPU_GDN_CORE_VERIFY as the job sets it (lookup_serve.sh's preset: 0, the torch path). The MTP
# arms also get VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1 unless the job set them
# (the integration build's MTP server). Every other flag comes from the job's environment (farm --env).
# Exit code: the number of arms whose lookup_serve.sh did not end with 0 (2: a usage error).

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${FARM_OUT:-/results/lookup-arms}"
ROUNDS="${LOOKUP_ARMS_ROUNDS:-edit:1:384:-:100,edit:32:384:-:0,mix:32:384:-:400}"
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
if [ $# -eq 0 ]; then
    echo "usage: lookup_arms.sh [--dry-run] ARM... (nospec, ngram<K>, hybrid<K>)" >&2
    exit 2
fi
for arm in "$@"; do
    case "$arm" in nospec | ngram[1-9] | hybrid[1-5]) ;; *) echo "lookup_arms: unknown arm $arm" >&2; exit 2 ;; esac
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

arm_words() {  # sets WORDS to the lookup_serve.sh arguments of an arm
    local arm="$1"
    case "$arm" in
        nospec) WORDS=(--label nospec --rounds "$ROUNDS") ;;
        ngram*) WORDS=(--label "$arm" --ngram "${arm#ngram}" --rounds "$ROUNDS") ;;
        hybrid*) WORDS=(--label "$arm" --mtp "${arm#hybrid}" --lookup 1 --rounds "$(modes10 "$ROUNDS")") ;;
    esac
}

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
for arm in "$@"; do
    arm_words "$arm"
    extra_env=(LOOKUP_ARM="$arm")
    case "$arm" in
        hybrid*)
            for kv in VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1; do
                key="${kv%%=*}"
                if [ -z "${!key+x}" ]; then extra_env+=("$kv"); fi
            done
            ;;
    esac
    if [ "$DRY" -eq 1 ]; then
        printf 'arm %s:' "$arm"
        printf ' %q' env "${extra_env[@]}" bash "$HERE/lookup_serve.sh" "${WORDS[@]}" --out "$OUT/$arm"
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
    env "${extra_env[@]}" bash "$HERE/lookup_serve.sh" "${WORDS[@]}" --out "$OUT/$arm"
    rc=$?
    echo "lookup_arms: arm $arm ended with rc $rc: $(date -u +%H:%M:%S)"
    if [ "$rc" -ne 0 ]; then failed=$((failed + 1)); fi
done
exit "$failed"
