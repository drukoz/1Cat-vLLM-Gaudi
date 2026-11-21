#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# ab_job.sh: one real-model server, VLLM_HPU_PREFILL_BATCH switched at run time, plan lines against it (card farm).
#
#   farm job prefill-batch NAME all 75m [--env K=V]... -- \
#       bash /bundle/plugin/tools/flash_next/prefill_batch/ab_job.sh PLAN [-- EXTRA_VLLM_SERVE_ARGS...]
#
# The server is prefill's prefill_job.sh server (the same `vllm serve` arguments: TP = the job's cards, expert
# parallel, --max-model-len 131072 --max-num-seqs 32, prefix caching, --max-num-batched-tokens 8192
# --long-prefill-token-threshold 2048, prompt token details; the same preset env, VLLM_PROFILER_ENABLED=true and the
# runtime knobs file /tmp/prefill_knobs.json), started with VLLM_HPU_PREFILL_BATCH=${PB_ROWS:-8} and the runtime
# switch file $FARM_OUT/switches (VLLM_HPU_RUNTIME_SWITCH_FILE), which holds VLLM_HPU_PREFILL_BATCH=0 until a plan
# line changes it. PLAN lines ('#' comments):
#   NAME switch V       write VLLM_HPU_PREFILL_BATCH=V to the switch file (0: every prompt chunk its own forward)
#   NAME sweep ARGS...  python -m tools.flash_next.rental.prefill_sweep --url --model fn --out $FARM_OUT/NAME ARGS
#   NAME burst ARGS...  python -m tools.flash_next.prefill_batch.burst_probe --url --model fn --out $FARM_OUT/NAME
#                       --switch-file $FARM_OUT/switches --rows $PB_ROWS ARGS (the probe switches the arms itself)
# PREFILL_MAX_SECONDS / PREFILL_STOP_RESERVE / PREFILL_MIN_LINE_SECONDS as in prefill_job.sh. Run it under the
# placement wrap (place_wrap.sh place.sh -- bash ab_job.sh ...). Writes versions.txt, server0.log, <NAME>/,
# <NAME>.log, server_events_*.json, merged.log (the VLLM_HPU_PREFILL_BATCH lines of the server log after each
# line). Exit: 0 every line ran, 3 the server died, 4 not healthy in time, 5 a line failed, 6 the time ran out.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../.." && pwd)"
PLAN="${1:-}"
[ -n "$PLAN" ] || { echo "ab_job: usage: ab_job.sh PLAN [-- EXTRA_VLLM_SERVE_ARGS...]" >&2; exit 2; }
shift
[ "${1:-}" = "--" ] && shift
[ -f "$PLAN" ] || PLAN="$HERE/plans/$PLAN"
[ -f "$PLAN" ] || { echo "ab_job: no plan file $PLAN" >&2; exit 2; }
OUT="${FARM_OUT:-/tmp/prefill_batch_ab}"
PORT="${FARM_PORT_BASE:-8000}"
TP="${FARM_NCARDS:-8}"
PYTHON="${FARM_PYTHON:-python3}"
ROWS="${PB_ROWS:-8}"
HEALTH_TIMEOUT="${PREFILL_HEALTH_TIMEOUT:-1500}"
MAX_SECONDS="${PREFILL_MAX_SECONDS:-0}"
STOP_RESERVE="${PREFILL_STOP_RESERVE:-75}"
MIN_LINE_SECONDS="${PREFILL_MIN_LINE_SECONDS:-30}"
T0=$(date +%s)
log() { echo "[$(date -u +%T)] ab_job: $*"; }

for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true \
    VLLM_PROFILER_ENABLED=true VLLM_HPU_SCHED_KNOBS_FILE=/tmp/prefill_knobs.json PYTHONDONTWRITEBYTECODE=1; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done
mkdir -p "$OUT" && cd "$OUT" || exit 2
SWITCH="$OUT/switches"
switch() { printf 'VLLM_HPU_PREFILL_BATCH=%s\n' "$1" >"$SWITCH.tmp" && mv -f "$SWITCH.tmp" "$SWITCH"; sleep 1; }
switch 0
export VLLM_HPU_PREFILL_BATCH="$ROWS" VLLM_HPU_RUNTIME_SWITCH_FILE="$SWITCH"

SERVE=(vllm serve /models/Qwen3.8-Flash-Next-FP8 --served-model-name fn --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size "$TP" --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32
    --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92
    --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 --enable-prompt-tokens-details "$@")
{
    "$PYTHON" -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1
    echo "image: $(cat /farm-image 2>/dev/null || echo "base $(cat /farm-base-sha 2>/dev/null)")"
    echo "tree: $TREE (commit $(cat "$TREE/.farm-sha" 2>/dev/null || echo "${FARM_SHA:-?}"))"
    echo "cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT, plan $PLAN, rows $ROWS"
    printf 'serve:'; printf ' %q' "${SERVE[@]}"; printf '\n'
    env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FLASHINFER|HCCL_|GC_|FARM_)' | sort
    echo "--- plan"; cat "$PLAN"
} >versions.txt

"${SERVE[@]}" >server0.log 2>&1 &
SP=$!
stop_server() {
    kill -TERM "$SP" 2>/dev/null
    for _ in $(seq 1 18); do kill -0 "$SP" 2>/dev/null || return 0; sleep 5; done
    kill -KILL "$SP" 2>/dev/null
}
trap 'stop_server; exit 130' INT TERM
while :; do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SP" 2>/dev/null; then
        log "SERVER-DIED after $(( $(date +%s) - T0 )) s"
        grep -nE "Error|Traceback|error:" server0.log | tail -n 25 | cut -c1-400
        exit 3
    fi
    if [ $(( $(date +%s) - T0 )) -ge "$HEALTH_TIMEOUT" ]; then
        log "SERVER-TIMEOUT"; tail -n 30 server0.log | cut -c1-300; stop_server; exit 4
    fi
    sleep 5
done
log "SERVER-HEALTHY after $(( $(date +%s) - T0 )) s"
grep -E "Maximum concurrency|KV cache size|Prompt bucket config|replay HPU graphs|VLLM_HPU_PREFILL_BATCH" \
    server0.log | sort -u | tail -n 10 | cut -c1-300

rc=0
while read -r NAME TOOL REST; do
    [ -z "$NAME" ] && continue
    case "$NAME" in \#*) continue ;; esac
    if ! kill -0 "$SP" 2>/dev/null; then log "the server is gone before $NAME"; rc=3; break; fi
    if [ "$TOOL" = switch ]; then
        switch "$REST"
        log "line $NAME: VLLM_HPU_PREFILL_BATCH=$REST"
        continue
    fi
    LIMIT=()
    if [ "$MAX_SECONDS" -gt 0 ]; then
        left=$(( MAX_SECONDS - STOP_RESERVE - ($(date +%s) - T0) ))
        if [ "$left" -lt "$MIN_LINE_SECONDS" ]; then
            log "no time left for $NAME and the lines after it ($left s)"; rc=6; break
        fi
        LIMIT=(timeout -s INT -k 20 "$left")
    fi
    eval "ARGS=($REST)"
    case "$TOOL" in
        sweep) CMD=(-m tools.flash_next.rental.prefill_sweep --url "http://127.0.0.1:$PORT" --model fn --out "$OUT/$NAME") ;;
        burst) CMD=(-m tools.flash_next.prefill_batch.burst_probe --url "http://127.0.0.1:$PORT" --model fn
                    --out "$OUT/$NAME" --switch-file "$SWITCH" --rows "$ROWS") ;;
        *) log "line $NAME: unknown tool $TOOL"; rc=5; continue ;;
    esac
    log "line $NAME ($TOOL): ${ARGS[*]} [switch: $(cat "$SWITCH")]"
    (cd "$TREE" && ${LIMIT[@]+"${LIMIT[@]}"} "$PYTHON" "${CMD[@]}" "${ARGS[@]}") </dev/null >"$OUT/$NAME.log" 2>&1
    prc=$?
    log "line $NAME rc $prc"
    grep -vE "^\s*$|[Ww]arning" "$OUT/$NAME.log" | tail -n 25 | cut -c1-330
    grep -E "VLLM_HPU_PREFILL_BATCH" server0.log | grep -E "merged prompt forwards" | tail -n 2 \
        | sed "s/^/[$NAME] /" | cut -c1-300 | tee -a merged.log
    if [ "$prc" -eq 124 ] || [ "$prc" -eq 137 ]; then log "the time ran out in $NAME"; rc=6; break; fi
    [ "$prc" -ne 0 ] && rc=5
done <"$PLAN"
grep -ciE "fallback|is off for|has no HPU kernel" server0.log | xargs echo "fallback/off lines:"
grep -cE "PT_DEVMEM|Allocation failed" server0.log | xargs echo "device memory failures:"
stop_server
trap - INT TERM
log "done after $(( $(date +%s) - T0 )) s, rc $rc"
exit "$rc"
