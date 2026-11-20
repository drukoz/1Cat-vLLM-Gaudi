#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# prefill_job.sh: a card-farm job (stage 2, `farm job`) that starts the real-model server on the job's
# cards, runs prefill_sweep.py plans against it and stops it. Results in FARM_OUT.
#
#   farm job prefill <name> any:4 40m [--env K=V]... -- \
#       bash /bundle/plugin/tools/flash_next/rental/prefill_job.sh PLAN [-- EXTRA_VLLM_SERVE_ARGS...]
#
# PLAN: a file of prefill_sweep runs, one per line, "NAME ARGS..." (ARGS without --url, --model, --out;
# '#' lines are comments); a plain name is looked up in prefill_plans/ next to this script: arm.sweeps (one
# arm of an A/B: a cold 32,768 and a cold 122,880 prompt twice each, four streams decoding during a prefill,
# four prompts at once), base_then_graphs.sweeps (the arm, then prompt HPU graphs through the runtime knob),
# arm_then_pacing_cap.sweeps (the arm, then two pacing shares and the 8,192-token prompt cap).
# Server: the rental's command (serve.py) at TP = the job's card count with expert parallelism,
# --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 unless EXTRA args override them, and the
# program head's preset environment of serve_probe.sh (each value only when the job did not set it), plus
# VLLM_PROFILER_ENABLED=true (rank-0 server_events_*.json: one event per prompt forward) and
# VLLM_HPU_SCHED_KNOBS_FILE=/tmp/prefill_knobs.json (the runtime knobs file: a plan line changes the step
# budget, the prompt cap, the pacing or the prompt HPU-graph threshold of the running server with
# `--knobs-file /tmp/prefill_knobs.json --knobs JSON`; no file = the server's own settings).
# PREFILL_MAX_SECONDS=N (0 = no limit): the whole job ends inside N seconds: a plan line is stopped (SIGINT,
# its report is written) and the lines after it are left out, so the server stops and every file is there
# before the queue's maximum hold kills the job (PREFILL_STOP_RESERVE seconds, default 75, are kept for the
# stop; a line is not started with less than PREFILL_MIN_LINE_SECONDS, default 30, left).
# Writes: versions.txt, server0.log, <NAME>/ (prefill_sweep.json, prefill_sweep.md, cases.jsonl), <NAME>.log,
# server_events_*.json. Exit: 0 every plan ran, 3 the server died, 4 not healthy in time, 5 a plan failed,
# 6 the time ran out before the last plan line ended.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../.." && pwd)"
PLAN="${1:-}"
[ -n "$PLAN" ] || { echo "prefill_job: usage: prefill_job.sh PLAN [-- EXTRA_VLLM_SERVE_ARGS...]" >&2; exit 2; }
shift
[ "${1:-}" = "--" ] && shift
[ -f "$PLAN" ] || PLAN="$HERE/prefill_plans/$PLAN"
[ -f "$PLAN" ] || { echo "prefill_job: no plan file $PLAN" >&2; exit 2; }
OUT="${FARM_OUT:-/tmp/prefill_job}"
PORT="${FARM_PORT_BASE:-8000}"
TP="${FARM_NCARDS:-8}"
PYTHON="${FARM_PYTHON:-python3}"
HEALTH_TIMEOUT="${PREFILL_HEALTH_TIMEOUT:-1500}"
MAX_SECONDS="${PREFILL_MAX_SECONDS:-0}"
STOP_RESERVE="${PREFILL_STOP_RESERVE:-75}"  # seconds kept for stopping the server at the end
MIN_LINE_SECONDS="${PREFILL_MIN_LINE_SECONDS:-30}"  # a plan line with less time than this is not started
T0=$(date +%s)
log() { echo "[$(date -u +%T)] prefill_job: $*"; }

for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true \
    VLLM_PROFILER_ENABLED=true VLLM_HPU_SCHED_KNOBS_FILE=/tmp/prefill_knobs.json PYTHONDONTWRITEBYTECODE=1; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done
mkdir -p "$OUT" && cd "$OUT" || exit 2

SERVE=(vllm serve /models/Qwen3.8-Flash-Next-FP8 --served-model-name fn --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size "$TP" --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32
    --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92
    --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 --enable-prompt-tokens-details "$@")
{
    "$PYTHON" -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1
    echo "image: $(cat /farm-image 2>/dev/null || echo "base $(cat /farm-base-sha 2>/dev/null)")"
    echo "tree: $TREE (commit $(cat "$TREE/.farm-sha" 2>/dev/null || echo "${FARM_SHA:-?}"))"
    echo "cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT, plan $PLAN"
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
grep -E "Maximum concurrency|KV cache size|Prompt bucket config|replay HPU graphs|Prefill pacing|Scheduler knobs" \
    server0.log | sort -u | tail -n 10 | cut -c1-300

rc=0
while read -r NAME REST; do
    [ -z "$NAME" ] && continue
    case "$NAME" in \#*) continue ;; esac
    if ! kill -0 "$SP" 2>/dev/null; then log "the server is gone before $NAME"; rc=3; break; fi
    LIMIT=()
    if [ "$MAX_SECONDS" -gt 0 ]; then
        left=$(( MAX_SECONDS - STOP_RESERVE - ($(date +%s) - T0) ))
        if [ "$left" -lt "$MIN_LINE_SECONDS" ]; then
            log "no time left for $NAME and the lines after it ($left s)"; rc=6; break
        fi
        LIMIT=(timeout -s INT -k 20 "$left")
    fi
    eval "ARGS=($REST)"
    log "plan $NAME: ${ARGS[*]}"
    (cd "$TREE" && ${LIMIT[@]+"${LIMIT[@]}"} "$PYTHON" -m tools.flash_next.rental.prefill_sweep --url "http://127.0.0.1:$PORT" \
        --model fn --out "$OUT/$NAME" "${ARGS[@]}") </dev/null >"$OUT/$NAME.log" 2>&1
    prc=$?
    log "plan $NAME rc $prc"
    grep -vE "^\s*$|[Ww]arning" "$OUT/$NAME.log" | tail -n 25 | cut -c1-330
    if [ "$prc" -eq 124 ] || [ "$prc" -eq 137 ]; then log "the time ran out in $NAME"; rc=6; break; fi
    [ "$prc" -ne 0 ] && rc=5
done <"$PLAN"
grep -ciE "fallback|is off for|has no HPU kernel" server0.log | xargs echo "fallback/off lines:"
stop_server
trap - INT TERM
log "done after $(( $(date +%s) - T0 )) s, rc $rc"
exit "$rc"
