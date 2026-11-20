#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# serve_long_probe.sh: an engine-step card job (farm job ... -- bash tools/flash_next/engine_step/serve_long_probe.sh).
# Starts the real-model server on the job's cards with the program head's best decode preset (the job's --env
# wins), waits until it is healthy, runs long_context_probe.py (decode at 1 and N streams, short prompts and a
# shared long prefix served from the prefix cache), counts the decode graphs compiled at run time, stops the
# server. Options: --prefix TOKENS (default 65536), --streams N (default 32), --tokens N (default 128).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${FARM_OUT:-/results/serve}"
PORT="${FARM_PORT_BASE:-8000}"
TP="${FARM_NCARDS:-8}"
PROBE_ARGS=("$@")
for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done
mkdir -p "$OUT" && cd "$OUT" || exit 2
{
    python3 -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1
    echo "image: $(cat /farm-image 2>/dev/null)"; echo "cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT, tp $TP"
    env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FARM_)' | sort
} > versions.txt
setsid vllm serve /models/Qwen3.8-Flash-Next-FP8 --served-model-name fn --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32 \
    --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 --enable-prompt-tokens-details \
    > server0.log 2>&1 &
SP=$!
stop_server() {
    kill -TERM -- "-$SP" 2>/dev/null
    for _ in $(seq 1 18); do kill -0 "$SP" 2>/dev/null || return 0; sleep 5; done
    kill -KILL -- "-$SP" 2>/dev/null
}
CHILD=""
trap '[ -n "$CHILD" ] && pkill -TERM -P "$CHILD"; stop_server; exit 143' TERM INT
t0=$(date +%s)
while :; do
    if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then break; fi
    if ! kill -0 $SP 2>/dev/null; then
        echo "SERVER-DIED after $(( $(date +%s) - t0 )) s"
        grep -nE "Error|Traceback|error:" server0.log | tail -n 25 | cut -c1-400; exit 3
    fi
    if [ $(( $(date +%s) - t0 )) -ge 1800 ]; then echo "SERVER-TIMEOUT"; tail -n 20 server0.log | cut -c1-300; stop_server; exit 4; fi
    sleep 5
done
echo "SERVER-HEALTHY after $(( $(date +%s) - t0 )) s"
grep -hE "sparse-QSA decode metadata|FLASH_NEXT_DECODE_METADATA" server0.log | sort | uniq -c | head -3
# The probe in the background and waited for: bash runs the TERM trap only after a foreground command ends.
probe() {
    python3 "$HERE/long_context_probe.py" --port "$PORT" --model fn --out "$OUT/probe.json" "${PROBE_ARGS[@]}" 2>&1 \
        | tee probe.log
    return "${PIPESTATUS[0]}"
}
probe &
CHILD=$!
wait "$CHILD"
RC=$?
echo "PROBE-RC $RC"
echo "decode configurations compiled at run time: $(grep -o "Configuration: ('decode'[^)]*)" server0.log | sort -u | wc -l)"
grep -o "Configuration: ('decode'[^)]*)" server0.log | sort | uniq -c | head -40
grep -ciE "fallback|is off for|has no HPU kernel" server0.log | xargs echo "fallback/off lines:"
grep -hiE "is off for|Traceback" server0.log | head -5 | cut -c1-300
stop_server
[ "$RC" -eq 0 ] || exit 5
