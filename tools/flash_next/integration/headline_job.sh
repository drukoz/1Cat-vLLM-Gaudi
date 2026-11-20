#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# headline_job.sh: the integration card build's headline numbers in ONE all-card farm job (TP8 + EP8):
#
#   farm job integration NAME all 40m [--env K=V]... -- bash tools/flash_next/integration/headline_job.sh [ARM...]
#
# ARMs run one after the other in the order given (default: nospec mtp3):
#   nospec  server 1: the farm preset + the build's flags below, no speculation. Probe: 1, 8 and 32 streams
#           greedy (a warm round before every measured round, two measured rounds at 32), then a SAMPLED round
#           (no temperature in the request: the checkpoint's default sampling) at 1, 8 and 32 streams.
#   mtp3    server 2: the same + MTP k = 3 (--speculative-config method mtp, VLLM_HPU_GDN_CORE_VERIFY=1,
#           VLLM_HPU_MTP_TIMING=1). Probe: 32, 8 and 1 streams greedy (32 first: the number that decides),
#           --no-sampled-check (sampled verification is off in this build: a sampled request gets HTTP 400);
#           acceptance per draft position per level from /metrics (SPEC lines).
#   mtp3-32 server 2 with the 32-stream level only (a short hold).
#   tiny    the tiny model at TP8 with the job's flags (farm_tiny.py --gdn hc; --spec 3 with HEADLINE_TINY_SPEC=1):
#           tokens against the CPU reference. A failure ends the job (exit 7) unless HEADLINE_TINY_ABORT=0.
#   mtp3-noshard  server 2 with VLLM_HPU_QSA_INDEXER_SHARD=0 (isolates the request-sharded indexer, never run in a
#           served speculative step before this build). Run by itself after mtp3 when mtp3 ended with the server
#           dead (3) or the probe failed (5) and HEADLINE_HOLD_S (default 2400, the job's hold) leaves 900 s or more;
#           HEADLINE_MTP_FALLBACK=0 turns that off.
# Between two servers: the leftover vLLM workers of THIS container are killed (the container has its own
# process namespace) and the script waits until no worker is left, then 8 s more for the cards (CARD.md).
#
# The build's flags (each only when the job did not set it with --env; serve_probe.sh adds the preset
# PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 VLLM_HPU_RECURRENT_STATE_POOL=1
# VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0
# VLLM_SKIP_WARMUP=true):
#   qsa-fused     VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1 (torch data flow, request-sharded indexer)
#   kernel-fixes  VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2
#   native-replay VLLM_HPU_DECODE_CUT_INTERVAL=1000 (decode graphs without the per-layer cuts)
#   engine-step   VLLM_HPU_CANDIDATE_SAMPLING=1 (sampled requests draw from the 64 largest logits; greedy rows and
#                 speculation keep their paths)
#   memory        VLLM_GRAPH_RESERVED_MEM=0.4 (graphs made at run time need room: warmup is skipped)
#   MTP server    VLLM_HPU_GDN_CORE_VERIFY=1 VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1
# Never set here: VLLM_HPU_GRAPH_ASYNC_REPLAY (never faster), HCCL_PRIM_* (HCCL_PRIM_COLLECTIVE_MASK=2 kills
# every rank), VLLM_HPU_MOE_GLUE (no gain), VLLM_HPU_QSA_FUSED=2/3 and VLLM_HPU_QSA_TABLE_WIDTH (not card-proven).
# The card memory of every card is logged every 15 s to $FARM_OUT/hlsmi_mem.log.
#
# Environment knobs: HEADLINE_TOKENS (measured tokens per stream, default 128), HEADLINE_WARM_TOKENS (default 48),
# HEADLINE_ROUNDS (measured rounds per level, greedy and sampled, after the warm round; default 0 = one per level
# and two at the top level without speculation, as in job 00128).
# Results: $FARM_OUT/<arm>/ (versions.txt, server0.log, scaling.json, probe.log, summary.json) and
# $FARM_OUT/headline.md + headline.json (headline_table.py), written after every arm.
# Exit code: the number of arms whose serve_probe.sh did not end with 0.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FARM_PYTHON:-python3}"
OUT="${FARM_OUT:-/results/headline}"
TOKENS="${HEADLINE_TOKENS:-128}"
WARM="${HEADLINE_WARM_TOKENS:-48}"
ROUNDS="${HEADLINE_ROUNDS:-0}"
T0="$(date -u +%s)"
HOLD_S="${HEADLINE_HOLD_S:-2400}"
FALLBACK="${HEADLINE_MTP_FALLBACK:-1}"

for kv in VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1 \
    VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2 \
    VLLM_HPU_DECODE_CUT_INTERVAL=1000 VLLM_HPU_CANDIDATE_SAMPLING=1 VLLM_GRAPH_RESERVED_MEM=0.4; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done

if [ $# -eq 0 ]; then set -- nospec mtp3; fi
mkdir -p "$OUT"

# The card memory every 15 s (coordinator 23:50 UTC: graphs made at run time can run out of device memory).
MEMLOG_PID=""
if command -v hl-smi >/dev/null 2>&1; then
    (
        while :; do
            echo "$(date -u +%T) $(hl-smi -Q index,memory.used,memory.total -f csv,noheader 2>&1 | tr '\n' ';')"
            sleep 15
        done
    ) >>"$OUT/hlsmi_mem.log" 2>&1 &
    MEMLOG_PID=$!
fi
stop_memlog() { if [ -n "$MEMLOG_PID" ]; then kill "$MEMLOG_PID" 2>/dev/null; MEMLOG_PID=""; fi; }
trap 'stop_memlog' EXIT
echo "headline_job: arms $*; tokens $TOKENS (warm $WARM); out $OUT; $(date -u +%T)"
env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FARM_)' | sort | sed 's/^/headline_job: env /'

# workers_left: how many vLLM processes are left in this container (their titles start with VLLM::, the API
# server's command line names vllm serve).
workers_left() {
    local n=0 pid cmd
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)" || continue
        case "$cmd" in
            VLLM::* | *"vllm serve"* | *"bin/vllm serve"*) n=$((n + 1)) ;;
        esac
    done
    echo "$n"
}

# clear_cards: after a server stop. Kills only processes of this container (own PID namespace).
clear_cards() {
    local i left
    left="$(workers_left)"
    echo "headline_job: after the server stop: $left vLLM processes left in this container"
    if [ "$left" -gt 0 ]; then
        pkill -9 -f "VLLM::" 2>/dev/null
        pkill -9 -f "vllm serve" 2>/dev/null
    fi
    for ((i = 0; i < 60; i++)); do
        [ "$(workers_left)" -eq 0 ] && break
        sleep 1
    done
    echo "headline_job: $(workers_left) left after ${i} s; waiting 8 s for the cards"
    sleep 8
    # The card memory of a stopped server is released some seconds after its processes are gone (00128: 65 GB per
    # card 10 s after the stop, 0.8 GB 25 s after): wait until every card holds less than 4 GB, at most 120 s.
    if command -v hl-smi >/dev/null 2>&1; then
        for ((i = 0; i < 120; i += 3)); do
            used="$(hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null |
                awk -F', *' '{ gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }')"
            [ "${used:-0}" -lt 4096 ] && break
            sleep 3
        done
        hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/^/headline_job: card memory: /'
        echo " (after ${i} s more)"
    fi
}

failed=0
first=1
ARMS=("$@")
i_arm=0
while [ "$i_arm" -lt "${#ARMS[@]}" ]; do
    arm="${ARMS[$i_arm]}"
    i_arm=$((i_arm + 1))
    if [ "$first" -eq 0 ]; then clear_cards; fi
    first=0
    echo "=== headline_job: arm $arm start $(date -u +%T)"
    case "$arm" in
        tiny)
            spec=()
            if [ "${HEADLINE_TINY_SPEC:-0}" = 1 ]; then spec=(--spec 3); fi
            "$PYTHON" /farm/farm_tiny.py --gdn hc ${spec[@]+"${spec[@]}"} --out "$OUT/tiny"
            rc=$?
            if [ "$rc" -ne 0 ] && [ "${HEADLINE_TINY_ABORT:-1}" = 1 ]; then
                echo "headline_job: the tiny gate failed (rc $rc): no real-model server with these flags"
                exit 7
            fi
            echo "=== headline_job: arm tiny rc=$rc end $(date -u +%T)"
            continue
            ;;
        nospec)
            bash "$HERE/serve_probe.sh" --out "$OUT/nospec" --levels 1,8,32 --tokens "$TOKENS" --warm-tokens "$WARM" \
                --repeat-top 2 --no-mintok-check --probe-arg --sampled-levels=1,8,32 --probe-arg "--rounds=$ROUNDS"
            ;;
        mtp3 | mtp3-32 | mtp3-noshard)
            levels=32,8,1
            if [ "$arm" = mtp3-32 ]; then levels=32; fi
            shard="$VLLM_HPU_QSA_INDEXER_SHARD"
            if [ "$arm" = mtp3-noshard ]; then shard=0; fi
            VLLM_HPU_QSA_INDEXER_SHARD="$shard" \
            VLLM_HPU_GDN_CORE_VERIFY="${VLLM_HPU_GDN_CORE_VERIFY:-1}" VLLM_HPU_MTP_TIMING="${VLLM_HPU_MTP_TIMING:-1}" \
                VLLM_HPU_MTP_DRAFT_LOOP="${VLLM_HPU_MTP_DRAFT_LOOP:-1}" VLLM_HPU_MTP_CHUNK_KV="${VLLM_HPU_MTP_CHUNK_KV:-1}" \
                bash "$HERE/serve_probe.sh" --out "$OUT/$arm" --mtp 3 --levels "$levels" --tokens "$TOKENS" \
                --warm-tokens "$WARM" --repeat-top 1 --no-mintok-check --probe-arg --no-sampled-check \
                --probe-arg "--rounds=$ROUNDS"
            ;;
        *)
            echo "headline_job: unknown arm $arm (nospec, mtp3, mtp3-32)"
            failed=$((failed + 1))
            continue
            ;;
    esac
    rc=$?
    # The probe ends with 0 even when rounds failed (it records them): a round with errors, or no measured round
    # at all, makes the arm a probe failure (5).
    if [ "$rc" -eq 0 ] && ! "$PYTHON" - "$OUT/$arm/scaling.json" <<'PY2'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(1)
rounds = (data.get("levels") or []) + (data.get("sampled") or [])
sys.exit(0 if rounds and all(r.get("ok") and not r.get("errors") for r in rounds) else 1)
PY2
    then
        echo "headline_job: arm $arm: a round failed or none was measured (scaling.json): counted as a probe failure"
        rc=5
    fi
    echo "=== headline_job: arm $arm rc=$rc end $(date -u +%T)"
    if [ "$rc" -ne 0 ]; then failed=$((failed + 1)); fi
    "$PYTHON" "$HERE/headline_table.py" "$OUT" --md "$OUT/headline.md" --json "$OUT/headline.json" || true
    left=$((HOLD_S - ($(date -u +%s) - T0)))
    if [ "$arm" = mtp3 ] && [ "$FALLBACK" = 1 ] && { [ "$rc" -eq 3 ] || [ "$rc" -eq 5 ]; }; then
        if [ "$left" -ge 900 ]; then
            echo "headline_job: mtp3 ended with $rc: the same server with VLLM_HPU_QSA_INDEXER_SHARD=0 next ($left s left)"
            ARMS+=(mtp3-noshard)
        else
            echo "headline_job: mtp3 ended with $rc; only $left s left: no run without the request shard"
        fi
    fi
done
echo "headline_job: $failed of ${#ARMS[@]} arms failed; $(date -u +%T)"
exit "$failed"
