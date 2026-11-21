#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# lookup_serve.sh: the card job of the spec-lookup track. Inside a farm container (bin/farm job, CARD.md) it
# starts one real-model server on the job's cards with a speculative configuration, waits until it is healthy,
# runs the lookup probe (lookup_probe.py next to this script: edit-and-return and fresh-prose rounds with the
# speculative counters) and stops the server. The server command and its preset environment are the card
# farm's serve_probe.sh (the program head's best decode setup), so the numbers compare with `farm serve` jobs.
#
#   bash tools/flash_next/rental/lookup_serve.sh --label L --rounds ROUNDS [options] [-- EXTRA_VLLM_SERVE_ARGS...]
#     --label L           names the run in the log and in probe.json
#     --rounds ROUNDS     lookup_probe.py rounds: KIND:STREAMS:TOKENS:MODE,... (KIND edit, editl or prose;
#                         MODE -, or the request's spec_lookup mode 0 MTP, 1 hybrid, 2 lookup only)
#     --mtp K             MTP speculative decoding with K tokens
#     --ngram K           n-gram prompt lookup with K tokens (--ngram-min N, default 3; --ngram-max N, default 8)
#                         (neither: no speculation, the reference)
#     --lookup MODE       VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP for the server (with --mtp: 1 hybrid, 2 lookup only)
#     --warm-tokens N     lookup_probe.py --warm-tokens (default -1: a warm round as long as the round)
#     --tp N              tensor parallel size (default: the job's card count FARM_NCARDS, else 8); --no-ep
#     --max-model-len N   (default 131072); --max-num-seqs N (default 32)
#     --health-timeout S  seconds to wait for the server (default 1500)
#     --port N            (default FARM_PORT_BASE, else 8000); --model DIR; --out DIR (default FARM_OUT)
#     --dry-run           print the server command, the environment and the probe command, start nothing
#
# Preset environment, each value only when the job did not set it (farm --env KEY=VALUE wins):
# PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 VLLM_HPU_RECURRENT_STATE_POOL=1
# VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0
# VLLM_SKIP_WARMUP=true, and VLLM_HPU_GDN_CORE_VERIFY=0 (the torch GDN verification: the fused verify op is the
# mtp-card-ready track's to prove on a card).
#
# Writes to OUT: versions.txt, server0.log, probe.json, probe.log, summary.json. Exit code: 0 the probe ran,
# 3 the server died, 4 it did not get healthy in time, 5 the probe failed, 2 a usage error, 130/143 interrupted.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FARM_PYTHON:-python3}"
MODEL="/models/Qwen3.8-Flash-Next-FP8"
TP="${FARM_NCARDS:-8}"
EP=1
LABEL=""
ROUNDS=""
SPEC=""
LOOKUP=""
NGRAM_K=""
NGRAM_MIN=3
NGRAM_MAX=8
WARM_TOKENS=-1
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=32
HEALTH_TIMEOUT=1500
PORT="${FARM_PORT_BASE:-8000}"
OUT="${FARM_OUT:-/results/lookup}"
DRY_RUN=0

die() {
    echo "lookup_serve: error: $*" >&2
    exit 2
}
num() { case "$2" in '' | *[!0-9]*) die "$1 takes a whole number, got '$2'" ;; esac; }

while [ $# -gt 0 ]; do
    case "$1" in
        --label) [ $# -ge 2 ] || die "--label needs a value"; LABEL="$2"; shift 2 ;;
        --rounds) [ $# -ge 2 ] || die "--rounds needs a value"; ROUNDS="$2"; shift 2 ;;
        --mtp)
            num "$1" "${2:-}"
            SPEC="{\"method\": \"mtp\", \"num_speculative_tokens\": $2}"
            shift 2
            ;;
        --ngram) num "$1" "${2:-}"; NGRAM_K="$2"; shift 2 ;;
        --ngram-min) num "$1" "${2:-}"; NGRAM_MIN="$2"; shift 2 ;;
        --ngram-max) num "$1" "${2:-}"; NGRAM_MAX="$2"; shift 2 ;;
        --lookup) num "$1" "${2:-}"; LOOKUP="$2"; shift 2 ;;
        --warm-tokens)
            case "${2:-}" in -1 | [0-9]*) ;; *) die "--warm-tokens takes -1 or a whole number" ;; esac
            WARM_TOKENS="$2"
            shift 2
            ;;
        --tp) num "$1" "${2:-}"; TP="$2"; shift 2 ;;
        --no-ep) EP=0; shift ;;
        --max-model-len) num "$1" "${2:-}"; MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs) num "$1" "${2:-}"; MAX_NUM_SEQS="$2"; shift 2 ;;
        --health-timeout) num "$1" "${2:-}"; HEALTH_TIMEOUT="$2"; shift 2 ;;
        --port) num "$1" "${2:-}"; PORT="$2"; shift 2 ;;
        --model) [ $# -ge 2 ] || die "--model needs a folder"; MODEL="$2"; shift 2 ;;
        --out) [ $# -ge 2 ] || die "--out needs a folder"; OUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; break ;;
        -h | --help) sed -n '3,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1 (extra vllm serve arguments follow --)" ;;
    esac
done
[ -n "$LABEL" ] || die "--label is required"
case "$LABEL" in *[!A-Za-z0-9._-]*) die "--label: letters, digits, '.', '_' and '-' only" ;; esac
[ -n "$ROUNDS" ] || die "--rounds is required"
if [ -n "$NGRAM_K" ]; then
    [ -z "$SPEC" ] || die "--mtp and --ngram exclude each other"
    SPEC="{\"method\": \"ngram\", \"num_speculative_tokens\": $NGRAM_K, \"prompt_lookup_max\": $NGRAM_MAX, \"prompt_lookup_min\": $NGRAM_MIN}"
fi
if [ -n "$LOOKUP" ]; then export VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP="$LOOKUP"; fi

for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true \
    VLLM_HPU_GDN_CORE_VERIFY=0; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done

SERVE=(vllm serve "$MODEL" --served-model-name fn --host 127.0.0.1 --port "$PORT"
    --tensor-parallel-size "$TP")
if [ "$EP" -eq 1 ]; then SERVE+=(--enable-expert-parallel); fi
SERVE+=(--max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --enable-prefix-caching
    --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192
    --long-prefill-token-threshold 2048 --enable-prompt-tokens-details)
if [ -n "$SPEC" ]; then SERVE+=(--speculative-config "$SPEC"); fi
SERVE+=("$@")
PROBE=("$PYTHON" "$HERE/lookup_probe.py" --port "$PORT" --model fn --label "$LABEL" --out "$OUT/probe.json"
    --rounds "$ROUNDS" --warm-tokens "$WARM_TOKENS" --timeout 600)

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'serve:'
    printf ' %q' "${SERVE[@]}"
    printf '\nprobe:'
    printf ' %q' "${PROBE[@]}"
    printf '\n'
    env | grep -E '^(VLLM_|PT_HPU)' | sort
    exit 0
fi

mkdir -p "$OUT" || die "cannot create $OUT"
# The server starts in OUT, not in the tree: the installed plugin (with its native libraries) is the one that
# serves, not the tree's copy.
cd "$OUT" || die "cannot enter $OUT"

{
    "$PYTHON" -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1
    echo "image: $(cat /farm-image 2>/dev/null || echo "base $(cat /farm-base-sha 2>/dev/null)")"
    echo "cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT, label $LABEL"
    printf 'serve:'
    printf ' %q' "${SERVE[@]}"
    printf '\nprobe:'
    printf ' %q' "${PROBE[@]}"
    printf '\n'
    env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FLASHINFER|HCCL_|GC_)' | sort
} >versions.txt

STARTED="$(date -u +%s)"
SP=""
RESULT="starting"
HEALTHY_S=""
PROBE_RC=""

json_summary() {
    "$PYTHON" - "$OUT/summary.json" "$RESULT" "$TP" "$EP" "$PORT" "${HEALTHY_S:-}" "${PROBE_RC:-}" "$SPEC" "$STARTED" "$LABEL" "$ROUNDS" <<'PY'
import json, os, sys, time
out, result, tp, ep, port, healthy, probe_rc, spec, started, label, rounds = sys.argv[1:12]
env = {k: v for k, v in sorted(os.environ.items())
       if k.startswith(("VLLM_", "PT_HPU", "HABANA_VISIBLE", "FLASHINFER", "HCCL_", "GC_", "FARM_"))}
json.dump({"result": result, "label": label, "rounds": rounds, "tp": int(tp), "ep": ep == "1", "port": int(port),
           "healthy_after_s": float(healthy) if healthy else None,
           "probe_rc": int(probe_rc) if probe_rc else None, "speculative_config": spec or None,
           "seconds": round(time.time() - float(started), 1), "env": env},
          open(out, "w"), indent=1)
PY
}

# stop_server: SIGTERM to the server's process group, SIGKILL 90 s later (serve_probe.sh's rule: not SIGINT,
# which a command started with "&" inherits as ignored).
stop_server() {
    local i
    [ -n "$SP" ] || return 0
    if kill -0 "$SP" 2>/dev/null; then
        kill -TERM -- "-$SP" 2>/dev/null || kill -TERM "$SP" 2>/dev/null
        for ((i = 0; i < 90; i++)); do
            kill -0 "$SP" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SP" 2>/dev/null; then echo "lookup_serve: the server did not stop in 90 s: SIGKILL"; fi
        kill -KILL -- "-$SP" 2>/dev/null || kill -KILL "$SP" 2>/dev/null
    fi
    wait "$SP" 2>/dev/null
    SP=""
}

CHILD=""

on_signal() {
    trap '' INT TERM
    echo "lookup_serve: interrupted (SIG$1): stopping the server"
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

setsid "${SERVE[@]}" >server0.log 2>&1 &
SP=$!
echo "lookup_serve: $LABEL: server starting: tp $TP, ep $EP, port $PORT, cards ${FARM_CARDS:-?}, spec ${SPEC:-none}, lookup ${VLLM_HPU_FLASH_NEXT_SPEC_LOOKUP:-0}, log $OUT/server0.log"
while :; do
    if "$PYTHON" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SP" 2>/dev/null; then
        echo "lookup_serve: $LABEL: SERVER-DIED after $(($(date -u +%s) - STARTED)) s"
        grep -nE "Error|Traceback|error:" server0.log | grep -v "vllm._C" | tail -n 25 | cut -c1-400
        tail -n 25 server0.log | cut -c1-400
        wait "$SP" 2>/dev/null
        SP=""
        RESULT="server-died"
        json_summary
        exit 3
    fi
    if [ $(($(date -u +%s) - STARTED)) -ge "$HEALTH_TIMEOUT" ]; then
        echo "lookup_serve: $LABEL: SERVER-TIMEOUT after $HEALTH_TIMEOUT s"
        tail -n 30 server0.log | cut -c1-300
        stop_server
        RESULT="server-timeout"
        json_summary
        exit 4
    fi
    sleep 5
done
HEALTHY_S=$(($(date -u +%s) - STARTED))
echo "lookup_serve: $LABEL: SERVER-HEALTHY after $HEALTHY_S s"
grep -E "Maximum concurrency|Recurrent state pool: [0-9]+ slots x|speculative|Speculative|lookup beside MTP" server0.log \
    | sort -u | tail -n 12 | cut -c1-300

probe() {
    "${PROBE[@]}" 2>&1 | tee probe.log
    return "${PIPESTATUS[0]}"
}
probe &
CHILD=$!
wait "$CHILD"
PROBE_RC=$?
CHILD=""
echo "lookup_serve: $LABEL: PROBE-RC $PROBE_RC"
grep -E "Flash-Next lookup hybrid" server0.log | tail -n 40 | cut -c1-400
grep -E "SpecDecoding metrics" server0.log | tail -n 6 | cut -c1-300
echo "lookup_serve: $LABEL: fallback/off lines: $(grep -ciE "fallback|is off for|has no HPU kernel" server0.log)"
grep -E "is off for|has no HPU kernel" server0.log | sort -u | head -n 8 | cut -c1-300
grep -nE "Traceback|Error" server0.log | grep -v "vllm._C" | head -n 10 | cut -c1-300
stop_server
RC=0
if [ "$PROBE_RC" -ne 0 ]; then RC=5; fi
if [ "$RC" -eq 0 ]; then RESULT="ok"; else RESULT="probe-failed"; fi
json_summary
echo "lookup_serve: $LABEL: DONE ($RESULT)"
exit "$RC"
