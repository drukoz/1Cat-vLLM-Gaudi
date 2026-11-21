#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# mtp_arms.sh: several real-model arms of mtp_real_card.sh in one farm job, one after another on the job's cards.
#
#   farm job mtp-card-ready NAME any:4 50m [--env KEY=VALUE]... -- bash tools/flash_next/rental/mtp_arms.sh \
#       'NAME|K|KEY=VALUE KEY=VALUE|spec_scaling.py options' ...
#
# Each arm gets its own folder $FARM_OUT/NAME, the arm's variables on top of the job's (--env), K drafted tokens
# (0 = a server without speculation) and its probe options. An arm whose server dies at the start (exit 3) or
# whose probe measured nothing (exit 5) runs once more as NAME-fallback with MTP_ARMS_FALLBACK on top (default:
# the flags that had not run with MTP on the real model on a card set to 0). While the arms run, hl-smi's
# memory.used of the job's cards is logged every 15 s to $FARM_OUT/hlsmi.log (a PT_DEVMEM line in a server log
# is the graphs made at run time running out of device memory: VLLM_GRAPH_RESERVED_MEM).
# Exit code: 0 when every arm (or its fallback) ran its probe, else the last failing arm's code.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARD_SCRIPT="${MTP_ARMS_CARD_SCRIPT:-$HERE/mtp_real_card.sh}"
BASE="${FARM_OUT:-/results/mtp-arms}"
FALLBACK="${MTP_ARMS_FALLBACK-VLLM_HPU_HC_FUSED_COPY_MODE=0 VLLM_HPU_MTP_DRAFT_LOOP=0 VLLM_HPU_MTP_CHUNK_KV=0}"
PAUSE="${MTP_ARMS_PAUSE:-10}"  # seconds between two servers: the stopped server's cards are released (CARD.md)

mkdir -p "$BASE" || { echo "mtp_arms: cannot use $BASE" >&2; exit 2; }
if [ $# -eq 0 ]; then
    echo "usage: mtp_arms.sh 'NAME|K|KEY=VALUE ...|probe options' ..." >&2
    exit 2
fi

MON=""
if command -v hl-smi >/dev/null 2>&1; then
    (while :; do
        printf '%s %s\n' "$(date -u +%H:%M:%S)" \
            "$(hl-smi -Q index,memory.used,memory.total -f csv,noheader 2>&1 | tr '\n' ';')"
        sleep 15
    done) >"$BASE/hlsmi.log" 2>&1 &
    MON=$!
fi
stop_monitor() {
    if [ -n "$MON" ]; then kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null; MON=""; fi
}
trap stop_monitor EXIT

run_arm() {  # NAME K "KEY=VALUE ..." "probe options"
    local name="$1" k="$2" envs="$3" opts="$4" rc
    echo "=== mtp_arms: arm $name (k = $k) env: ${envs:-none} | probe: ${opts:-default}  $(date -u +%H:%M:%S)"
    # shellcheck disable=SC2086 - the variables and the probe options are word lists
    (
        export FARM_OUT="$BASE/$name"
        for kv in $envs; do export "$kv"; done
        exec bash "$CARD_SCRIPT" "$k" -- $opts
    )
    rc=$?
    echo "=== mtp_arms: arm $name rc=$rc  $(date -u +%H:%M:%S)"
    sleep "$PAUSE"
    return "$rc"
}

FINAL=0
for spec in "$@"; do
    IFS='|' read -r name k envs opts <<<"$spec"
    if [ -z "${name:-}" ] || ! [[ "${k:-}" =~ ^[0-9]$ ]]; then
        echo "mtp_arms: bad arm '$spec' (NAME|K|KEY=VALUE ...|probe options)" >&2
        FINAL=2
        continue
    fi
    run_arm "$name" "$k" "${envs:-}" "${opts:-}"
    rc=$?
    if { [ "$rc" = 3 ] || [ "$rc" = 5 ]; } && [ -n "$FALLBACK" ]; then
        run_arm "$name-fallback" "$k" "${envs:-} $FALLBACK" "${opts:-}"
        rc=$?
    fi
    [ "$rc" = 0 ] || FINAL="$rc"
done
stop_monitor
echo "mtp_arms: done, exit $FINAL"
exit "$FINAL"
