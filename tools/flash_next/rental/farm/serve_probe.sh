#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# serve_probe.sh: the real-model server job of the card farm. Inside a farm container (farm_job.sh), it
# starts vLLM on the job's cards, waits until the server is healthy, runs the decode probe
# decode_scaling.py (next to this script) and stops the server. The generalized form of the
# coordinator's serve_and_probe.sh: same server command, same probe, same scaling.json.
#
#   bash /farm/serve_probe.sh [options] [-- EXTRA_VLLM_SERVE_ARGS...]
#     --tp N              tensor parallel size (default: the job's card count, FARM_NCARDS, else 8)
#     --no-ep             no --enable-expert-parallel (default: on)
#     --mtp K             MTP speculative decoding with K tokens (--speculative-config method mtp)
#     --spec JSON         another --speculative-config, verbatim
#     --levels LIST       probe stream counts (default 1,8,16,32); each level: a warm round, then a measured one
#     --tokens N          tokens per measured stream (default 256); --warm-tokens N (default 48)
#     --repeat-top N      measured rounds at the largest level (default 2)
#     --ctx N:TOKENS      one more measured round: N streams with about TOKENS-token prompts
#     --no-probe          start, check health and stop (a start test)
#     --max-model-len N   (default 131072); --max-num-seqs N (default 32)
#     --health-timeout S  seconds to wait for the server (default 2400)
#     --bare              no preset environment (below); only what the job sets
#     --port N            (default: the job's port base FARM_PORT_BASE, else 8000)
#     --model DIR         (default /models/Qwen3.8-Flash-Next-FP8); --out DIR (default FARM_OUT)
#
# Preset environment, each value only when the job did not set it (farm --env KEY=VALUE wins): the
# program head's best decode setup of 2026-11-20: PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0
# VLLM_HPU_FLASH_NEXT_QSA_MODE=1 VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1
# VLLM_HPU_HC_FUSED=1 VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true.
#
# Writes to OUT: versions.txt, server0.log, scaling.json (decode_scaling.py), probe.log, summary.json.
# Exit code: 0 the probe ran, 3 the server died, 4 it did not get healthy in time, 5 the probe failed,
# 2 a usage error, 130/143 interrupted (the server is stopped first).

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FARM_PYTHON:-python3}"
MODEL="/models/Qwen3.8-Flash-Next-FP8"
TP="${FARM_NCARDS:-8}"
EP=1
SPEC=""
LEVELS="1,8,16,32"
TOKENS=256
WARM_TOKENS=48
REPEAT_TOP=2
CTX=""
PROBE=1
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=32
HEALTH_TIMEOUT=2400
BARE=0
PORT="${FARM_PORT_BASE:-8000}"
OUT="${FARM_OUT:-/results/serve}"

die() {
    echo "serve_probe: error: $*" >&2
    exit 2
}
num() { case "$2" in '' | *[!0-9]*) die "$1 takes a whole number, got '$2'" ;; esac; }

while [ $# -gt 0 ]; do
    case "$1" in
        --tp) num "$1" "${2:-}"; TP="$2"; shift 2 ;;
        --no-ep) EP=0; shift ;;
        --mtp)
            num "$1" "${2:-}"
            SPEC="{\"method\": \"mtp\", \"num_speculative_tokens\": $2}"
            shift 2
            ;;
        --spec) [ $# -ge 2 ] || die "--spec needs a JSON value"; SPEC="$2"; shift 2 ;;
        --levels)
            case "${2:-}" in '' | *[!0-9,]*) die "--levels takes stream counts such as 1,8,32, got '${2:-}'" ;; esac
            LEVELS="$2"
            shift 2
            ;;
        --tokens) num "$1" "${2:-}"; TOKENS="$2"; shift 2 ;;
        --warm-tokens) num "$1" "${2:-}"; WARM_TOKENS="$2"; shift 2 ;;
        --repeat-top) num "$1" "${2:-}"; REPEAT_TOP="$2"; shift 2 ;;
        --ctx)
            case "${2:-}" in [0-9]*:[0-9]*) ;; *) die "--ctx takes N:TOKENS such as 32:4000, got '${2:-}'" ;; esac
            CTX="$2"
            shift 2
            ;;
        --no-probe) PROBE=0; shift ;;
        --max-model-len) num "$1" "${2:-}"; MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs) num "$1" "${2:-}"; MAX_NUM_SEQS="$2"; shift 2 ;;
        --health-timeout) num "$1" "${2:-}"; HEALTH_TIMEOUT="$2"; shift 2 ;;
        --bare) BARE=1; shift ;;
        --port) num "$1" "${2:-}"; PORT="$2"; shift 2 ;;
        --model) [ $# -ge 2 ] || die "--model needs a folder"; MODEL="$2"; shift 2 ;;
        --out) [ $# -ge 2 ] || die "--out needs a folder"; OUT="$2"; shift 2 ;;
        --) shift; break ;;
        -h | --help) sed -n '3,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1 (extra vllm serve arguments follow --)" ;;
    esac
done

if [ "$BARE" -eq 0 ]; then
    for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
        VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
        VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true; do
        key="${kv%%=*}"
        if [ -z "${!key+x}" ]; then export "$kv"; fi
    done
fi

mkdir -p "$OUT" || die "cannot create $OUT"
cd "$OUT" || die "cannot enter $OUT"

SERVE=(vllm serve "$MODEL" --served-model-name fn --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size "$TP")
if [ "$EP" -eq 1 ]; then SERVE+=(--enable-expert-parallel); fi
SERVE+=(--max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --enable-prefix-caching
    --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192
    --long-prefill-token-threshold 2048 --enable-prompt-tokens-details)
if [ -n "$SPEC" ]; then SERVE+=(--speculative-config "$SPEC"); fi
SERVE+=("$@")

{
    "$PYTHON" -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1
    echo "image: $(cat /farm-image 2>/dev/null || echo "base $(cat /farm-base-sha 2>/dev/null)")"
    echo "cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT"
    printf 'serve:'
    printf ' %q' "${SERVE[@]}"
    printf '\n'
    env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FLASHINFER|HCCL_|GC_)' | sort
} >versions.txt

STARTED="$(date -u +%s)"
SP=""
RESULT="starting"
HEALTHY_S=""
PROBE_RC=""

json_summary() {
    "$PYTHON" - "$OUT/summary.json" "$RESULT" "$TP" "$EP" "$PORT" "${HEALTHY_S:-}" "${PROBE_RC:-}" "$SPEC" "$STARTED" <<'PY'
import json, os, sys, time
out, result, tp, ep, port, healthy, probe_rc, spec, started = sys.argv[1:10]
env = {k: v for k, v in sorted(os.environ.items())
       if k.startswith(("VLLM_", "PT_HPU", "HABANA_VISIBLE", "FLASHINFER", "HCCL_", "GC_", "FARM_"))}
json.dump({"result": result, "tp": int(tp), "ep": ep == "1", "port": int(port),
           "healthy_after_s": float(healthy) if healthy else None,
           "probe_rc": int(probe_rc) if probe_rc else None, "speculative_config": spec or None,
           "seconds": round(time.time() - float(started), 1), "env": env},
          open(out, "w"), indent=1)
PY
}

# stop_server: SIGTERM to the server's process group, SIGKILL 90 s later. Not SIGINT: a command a
# script starts with "&" inherits an ignored SIGINT, and Python then installs no handler for it.
stop_server() {
    local i
    [ -n "$SP" ] || return 0
    if kill -0 "$SP" 2>/dev/null; then
        kill -TERM -- "-$SP" 2>/dev/null || kill -TERM "$SP" 2>/dev/null
        for ((i = 0; i < 90; i++)); do
            kill -0 "$SP" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SP" 2>/dev/null; then echo "serve_probe: the server did not stop in 90 s: SIGKILL"; fi
        kill -KILL -- "-$SP" 2>/dev/null || kill -KILL "$SP" 2>/dev/null
    fi
    wait "$SP" 2>/dev/null
    SP=""
}

# CHILD: the probe while it runs. Long steps run in the background and the script waits for them,
# because bash runs a trap only after the foreground command has ended.
CHILD=""

on_signal() {
    trap '' INT TERM
    echo "serve_probe: interrupted (SIG$1): stopping the server"
    if [ -n "$CHILD" ]; then
        pkill -TERM -P "$CHILD" 2>/dev/null
        kill -TERM "$CHILD" 2>/dev/null
    fi
    stop_server
    RESULT="interrupted"
    json_summary
    exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

# The server in its own process group (setsid), so stop_server reaches its workers too.
setsid "${SERVE[@]}" >server0.log 2>&1 &
SP=$!
echo "serve_probe: server starting: tp $TP, ep $EP, port $PORT, cards ${FARM_CARDS:-?}, log $OUT/server0.log"
while :; do
    if "$PYTHON" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SP" 2>/dev/null; then
        echo "serve_probe: SERVER-DIED after $(($(date -u +%s) - STARTED)) s"
        grep -nE "Error|Traceback|error:" server0.log | tail -n 25 | cut -c1-400
        tail -n 25 server0.log | cut -c1-400
        wait "$SP" 2>/dev/null
        SP=""
        RESULT="server-died"
        json_summary
        exit 3
    fi
    if [ $(($(date -u +%s) - STARTED)) -ge "$HEALTH_TIMEOUT" ]; then
        echo "serve_probe: SERVER-TIMEOUT after $HEALTH_TIMEOUT s"
        tail -n 30 server0.log | cut -c1-300
        stop_server
        RESULT="server-timeout"
        json_summary
        exit 4
    fi
    sleep 5
done
HEALTHY_S=$(($(date -u +%s) - STARTED))
echo "serve_probe: SERVER-HEALTHY after $HEALTHY_S s"
grep -E "Maximum concurrency|num_blocks|KV cache|speculative|Speculative" server0.log | tail -n 12 | cut -c1-300

RC=0
if [ "$PROBE" -eq 1 ]; then
    PROBE_ARGS=(--port "$PORT" --model fn --out "$OUT/scaling.json" --levels "$LEVELS" --tokens "$TOKENS"
        --warm-tokens "$WARM_TOKENS" --repeat-top "$REPEAT_TOP")
    if [ -n "$CTX" ]; then PROBE_ARGS+=(--ctx "$CTX"); fi
    probe() {
        "$PYTHON" "$HERE/decode_scaling.py" "${PROBE_ARGS[@]}" 2>&1 | tee probe.log
        return "${PIPESTATUS[0]}"
    }
    probe &
    CHILD=$!
    wait "$CHILD"
    PROBE_RC=$?
    CHILD=""
    echo "serve_probe: PROBE-RC $PROBE_RC"
    grep -E "generation throughput|acceptance|SpecDecoding" server0.log | tail -n 16 | cut -c1-300
    if [ "$PROBE_RC" -ne 0 ]; then RC=5; fi
fi
echo "serve_probe: fallback/off lines: $(grep -ciE "fallback|is off for|has no HPU kernel" server0.log)"
stop_server
if [ "$RC" -eq 0 ]; then RESULT="ok"; else RESULT="probe-failed"; fi
json_summary
echo "serve_probe: DONE ($RESULT)"
exit "$RC"
