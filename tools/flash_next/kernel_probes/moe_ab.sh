#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# moe_ab.sh KIT_DIR ARM...: kernel-fixes' TP8 A/B of the routed-expert forms on the integration build, as ONE
# all-card farm job of the integration track (its image holds the build's plugin code; for the dense arms the
# build must contain kernel-fixes 86104c9d, VLLM_HPU_MOE_DENSE). This script only drives the integration kit's
# headline_job.sh, which must sit in KIT_DIR next to its serve_probe.sh and decode_scaling.py:
#
#   farm job integration moe-ab all 40m --env VLLM_HPU_PREFIX_ALIGN_SPLIT=1 --env VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 \
#       -- bash /results/late/kit/moe_ab.sh /results/late/kit dense-mtp3 dense-nospec base-mtp3
#
# ARMs, one server each, in the order given (each: 32 streams greedy, a warm round then the measured rounds, then 1
# stream: the text to compare between arms):
#   base-mtp3     MTP k = 3 (headline_job.sh arm mtp3), the vendor MoE op as served today
#   chunk-mtp3    MTP k = 3 with VLLM_MOE_CHUNK=64,8192 VLLM_MOE_TOKEN_BOUNDARY=512,1000000 (chunk_size 64 for steps of
#                 at most 512 tokens, 8192 above)
#   dense-mtp3    MTP k = 3 with VLLM_HPU_MOE_DENSE=1 (two batched FP8 GEMMs over all local experts for steps of at
#                 most 512 tokens, the vendor op above)
#   base-nospec   no speculation (headline_job.sh arm nospec), the vendor op
#   dense-nospec  no speculation with VLLM_HPU_MOE_DENSE=1
# Measured rounds: MOE_AB_ROUNDS_MTP (default 2) on MTP arms, MOE_AB_ROUNDS_NOSPEC (default 1) on the others.
# MOE_AB_CTX=N:TOKENS adds one long-prompt round at the end of the MTP arms (default none; e.g. 1:16000 for TTFT).
# Results: $FARM_OUT/<arm>/ (headline_job.sh's layout with its headline.md). Exit code: arms that did not end with 0.
set -u
KIT="${1:-/results/late/kit}"
shift || true
OUT="${FARM_OUT:-/results/moe-ab}"
[ -f "$KIT/headline_job.sh" ] || { echo "moe_ab: no headline_job.sh in $KIT" >&2; exit 2; }
[ $# -gt 0 ] || set -- base-mtp3 dense-mtp3 dense-nospec
unset VLLM_MOE_CHUNK VLLM_MOE_TOKEN_BOUNDARY VLLM_HPU_MOE_DENSE

workers_left() {
    local n=0 pid cmd
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)" || continue
        case "$cmd" in VLLM::* | *"vllm serve"*) n=$((n + 1)) ;; esac
    done
    echo "$n"
}

clear_cards() {  # kills only processes of this container (its own PID namespace)
    local i used
    if [ "$(workers_left)" -gt 0 ]; then
        pkill -9 -f "VLLM::" 2>/dev/null
        pkill -9 -f "vllm serve" 2>/dev/null
    fi
    for ((i = 0; i < 60; i++)); do [ "$(workers_left)" -eq 0 ] && break; sleep 1; done
    sleep 8
    if command -v hl-smi >/dev/null 2>&1; then
        for ((i = 0; i < 120; i += 3)); do
            used="$(hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null |
                awk -F', *' '{ gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }')"
            [ "${used:-0}" -lt 4096 ] && break
            sleep 3
        done
    fi
    echo "moe_ab: cards cleared ($(workers_left) vLLM processes left) $(date -u +%T)"
}

failed=0
for arm in "$@"; do
    extra=()
    case "$arm" in
        base-*) ;;
        chunk-*) extra=(VLLM_MOE_CHUNK=64,8192 VLLM_MOE_TOKEN_BOUNDARY=512,1000000) ;;
        dense-*) extra=(VLLM_HPU_MOE_DENSE=1) ;;
        *) echo "moe_ab: unknown arm $arm" >&2; failed=$((failed + 1)); continue ;;
    esac
    case "$arm" in
        *-mtp3) headline=mtp3; rounds="${MOE_AB_ROUNDS_MTP:-2}"; ctx="${MOE_AB_CTX:-}" ;;
        *-nospec) headline=nospec; rounds="${MOE_AB_ROUNDS_NOSPEC:-1}"; ctx="" ;;
        *) echo "moe_ab: unknown arm $arm" >&2; failed=$((failed + 1)); continue ;;
    esac
    echo "=== moe_ab: arm $arm (headline_job.sh $headline, ${extra[*]:-no extra env}, rounds $rounds) $(date -u +%T)"
    env ${extra[@]+"${extra[@]}"} HEADLINE_LEVELS=32,1 HEADLINE_ROUNDS="$rounds" HEADLINE_MTP_FALLBACK=0 \
        ${ctx:+HEADLINE_MTP_CTX="$ctx"} FARM_OUT="$OUT/$arm" bash "$KIT/headline_job.sh" "$headline"
    rc=$?
    echo "moe_ab: arm $arm ended with exit code $rc $(date -u +%T)"
    [ "$rc" -eq 0 ] || failed=$((failed + 1))
    clear_cards
done
for arm in "$@"; do
    [ -f "$OUT/$arm/headline.md" ] && { echo "--- moe_ab: $arm/headline.md"; cat "$OUT/$arm/headline.md"; }
done
exit "$failed"
