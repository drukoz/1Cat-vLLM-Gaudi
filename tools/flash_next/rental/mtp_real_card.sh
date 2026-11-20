#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# mtp_real_card.sh: the real model with MTP on a card-farm job's cards and the speculative probe of this tree.
#
#   farm job mtp-card-ready NAME any:4 35m [--env KEY=VALUE]... -- bash tools/flash_next/rental/mtp_real_card.sh \
#       [K] [-- spec_scaling.py options...]
#
# K is the number of drafted tokens (default 3; 0 = a server without speculation, the reference). The server is
# the farm's (serve_probe.sh): the same command and the same preset environment, each value only when the job
# did not set it; tensor parallel size = the job's card count, expert parallel on. The probe is
# tools/flash_next/rental/spec_scaling.py (step time, tokens per step and acceptance per draft position; rounds
# with the server's default sampling with --sampled-levels; a long-prompt round with --ctx). Default probe
# options: --levels 1,8,32 --tokens 256. Flags of a speculative arm go in with --env, for example
#   --env VLLM_HPU_GDN_CORE_VERIFY=1 --env VLLM_HPU_MTP_TIMING=1 --env VLLM_HPU_SPEC_SAMPLED_VERIFY=1
#
# Writes to $FARM_OUT: versions.txt, server0.log, scaling.json, probe.log, mtp_step_timing_r*.jsonl (with the
# timer). Exit code: 0 the probe ran, 3 the server died, 4 not healthy in time, 5 the probe failed.

set -u

K=3
case "${1:-}" in
    [0-9]) K="$1"; shift ;;
esac
if [ "${1:-}" = "--" ]; then shift; fi
if [ $# -eq 0 ]; then set -- --levels 1,8,32 --tokens 256; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FARM_PYTHON:-python3}"
MODEL="${MTP_MODEL:-/models/Qwen3.8-Flash-Next-FP8}"
TP="${FARM_NCARDS:-8}"
PORT="${FARM_PORT_BASE:-8000}"
OUT="${FARM_OUT:-/results/mtp-real}"
HEALTH_TIMEOUT="${MTP_HEALTH_TIMEOUT:-1500}"

for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done

mkdir -p "$OUT" && cd "$OUT" || { echo "mtp_real_card: cannot use $OUT" >&2; exit 2; }

SERVE=(vllm serve "$MODEL" --served-model-name fn --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size "$TP" --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32
    --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92
    --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 --enable-prompt-tokens-details)
if [ "$K" -gt 0 ]; then
    SERVE+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $K}")
fi

{
    "$PYTHON" -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1 | tail -n 2
    echo "image: $(cat /farm-image 2>/dev/null || echo "base $(cat /farm-base-sha 2>/dev/null)")"
    echo "cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT, k $K"
    printf 'serve:'; printf ' %q' "${SERVE[@]}"; printf '\n'
    printf 'probe:'; printf ' %q' "$@"; printf '\n'
    env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FLASHINFER|HCCL_|GC_)' | sort
} >versions.txt

SP=""
stop_server() {
    local i
    [ -n "$SP" ] || return 0
    if kill -0 "$SP" 2>/dev/null; then
        kill -TERM -- "-$SP" 2>/dev/null || kill -TERM "$SP" 2>/dev/null
        for ((i = 0; i < 90; i++)); do
            kill -0 "$SP" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$SP" 2>/dev/null || kill -KILL "$SP" 2>/dev/null
    fi
    wait "$SP" 2>/dev/null
    SP=""
    pkill -9 -f "VLLM::" 2>/dev/null  # a worker that outlived the server keeps its card
}
CHILD=""
on_signal() {
    trap '' INT TERM
    echo "mtp_real_card: interrupted (SIG$1): stopping the server"
    if [ -n "$CHILD" ]; then pkill -TERM -P "$CHILD" 2>/dev/null; kill -TERM "$CHILD" 2>/dev/null; fi
    stop_server
    exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

STARTED="$(date -u +%s)"
setsid "${SERVE[@]}" >server0.log 2>&1 &
SP=$!
echo "mtp_real_card: server starting: tp $TP, k $K, port $PORT, cards ${FARM_CARDS:-?}, log $OUT/server0.log"
while :; do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SP" 2>/dev/null; then
        echo "mtp_real_card: SERVER-DIED after $(($(date -u +%s) - STARTED)) s"
        grep -nE "Error|Traceback|error:" server0.log | grep -v "vllm._C" | tail -n 25 | cut -c1-400
        wait "$SP" 2>/dev/null
        SP=""
        exit 3
    fi
    if [ $(($(date -u +%s) - STARTED)) -ge "$HEALTH_TIMEOUT" ]; then
        echo "mtp_real_card: SERVER-TIMEOUT after $HEALTH_TIMEOUT s"
        tail -n 20 server0.log | cut -c1-300
        stop_server
        exit 4
    fi
    sleep 5
done
echo "mtp_real_card: SERVER-HEALTHY after $(($(date -u +%s) - STARTED)) s"
grep -E "Maximum concurrency|Recurrent state pool: [0-9]+ slots x|GPU KV cache size|Free device memory" server0.log |
    sed 's/^.*\] //' | sort -u | tail -n 6 | cut -c1-300

probe() {
    "$PYTHON" "$HERE/spec_scaling.py" --port "$PORT" --model fn --out "$OUT/scaling.json" "$@" 2>&1 | tee probe.log
    return "${PIPESTATUS[0]}"
}
probe "$@" &
CHILD=$!
wait "$CHILD"
PROBE_RC=$?
CHILD=""
echo "mtp_real_card: PROBE-RC $PROBE_RC"
grep -E "SpecDecoding metrics" server0.log | tail -n 6 | cut -c1-320
grep -E "MTP step timing" server0.log | tail -n 12 | sed 's/^.*MTP step timing: //' | cut -c1-320
echo "mtp_real_card: fallback/off lines: $(grep -ciE "fallback|is off for|has no HPU kernel" server0.log)"
grep -E "Neither storage|Empty tensor optional|RuntimeError" server0.log | sed 's/^.*\] //' | sort | uniq -c | sort -rn | head -n 5 | cut -c1-300
stop_server
if [ "$PROBE_RC" -ne 0 ]; then
    echo "mtp_real_card: DONE (probe-failed)"
    exit 5
fi
echo "mtp_real_card: DONE (ok)"
exit 0
