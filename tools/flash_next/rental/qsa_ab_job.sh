#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# qsa_ab_job.sh ARM: one arm of the sparse-QSA A/B on the real model, inside a card-farm container
# (farm_job.sh ... -- bash tools/flash_next/rental/qsa_ab_job.sh off|mode1). Both arms run the same steps
# on the same commit; the arm's flags (VLLM_HPU_QSA_FUSED, VLLM_HPU_QSA_INDEXER_SHARD) come from the job's
# --env. Every result is a plain line on standard output (the job's log) as soon as it exists.
#
#   1. The card check of the block selection (card rule 9), tests/unit_tests/qsa_select/test_hpu_select_ids.py,
#      on one of the job's cards: QSAIDS lines, then one "RULE9 VERDICT" line. The server then runs the
#      narrowing form the check found right (VLLM_HPU_QSA_TOPK_IDS; 0, the default, unless only another is).
#   2. The real-model server on the job's cards: the farm's serve command and preset (serve_probe.sh).
#   3. qsa_ab_probe.py: decode at 1, 16 and 32 streams with short prompts; the needle at about 32,000 tokens (then 64,000);
#      32 streams at that context (the prefix served from the cache); every stream's text saved.
#   4. Arm mode1 only, after its server stopped: the card test of the four QSA TPC kernels
#      (tests/unit_tests/qsa_fused/test_hpu_op.py) with the installed libraries given by path.
#
# QSA_AB_DEADLINE (epoch seconds) is when the job's maximum hold ends; steps that do not fit are skipped and
# say so. Results: $FARM_OUT (select_ids.json, select_ids.log, server0.log, qsa_ab.json, probe.log,
# versions.txt, qsa_hpu_op.log, summary.txt). Exit code: 0 the probe ran, 3 the server died, 4 it was not
# healthy in time, 5 the probe failed; the checks of steps 1 and 4 are reported, not part of the exit code.

set -u

ARM="${1:-off}"
PY="${FARM_PYTHON:-python3}"
OUT="${FARM_OUT:-/tmp/qsa-ab-$ARM}"
PORT="${FARM_PORT_BASE:-8000}"
TP="${FARM_NCARDS:-4}"
MODEL="/models/Qwen3.8-Flash-Next-FP8"
TREE="$PWD"
NOW="$(date -u +%s)"
DEADLINE="${QSA_AB_DEADLINE:-$((NOW + 2400))}"
LEVELS="${QSA_AB_LEVELS:-1,16,32}"
CTX="${QSA_AB_CTX:-32000,64000}"
TOKENS="${QSA_AB_TOKENS:-128}"

mkdir -p "$OUT" || exit 2
say() { echo "QSAAB [$ARM $(date -u +%H:%M:%S)] $*"; }
left() { echo $((DEADLINE - $(date -u +%s))); }

say "arm $ARM, cards ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), commit ${FARM_SHA:-?}, $(left) s to the deadline;" \
    "VLLM_HPU_QSA_FUSED=${VLLM_HPU_QSA_FUSED:-0} VLLM_HPU_QSA_INDEXER_SHARD=${VLLM_HPU_QSA_INDEXER_SHARD:-0}"

# The installed package's lib folder (the tree's copy is empty): the kernel libraries of the op tests.
LIB="$(cd /tmp && env -u PYTHONPATH "$PY" -c "import flashinfer_gaudi, pathlib; print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')" 2>/dev/null | tail -n 1)"

# ---------------------------------------------------------------- 1. the selection check on one card
TOPK_IDS=0
if [ "${QSA_AB_SKIP_IDS:-0}" != 1 ]; then
    say "step 1: the block-selection check on one card (rule 9)"
    (
        cd "$TREE" || exit 2
        unset PT_HPU_RECIPE_CACHE_CONFIG VLLM_HPU_QSA_FUSED VLLM_HPU_QSA_INDEXER_SHARD VLLM_HPU_QSA_TOPK_IDS
        export PYTHONPATH="$TREE${PYTHONPATH:+:$PYTHONPATH}" PT_HPU_LAZY_MODE=1 FARM_OUT="$OUT"
        rm -f "$OUT/select_ids.json"
        timeout 360 "$PY" -m pytest -q -s -rs -p no:cacheprovider tests/unit_tests/qsa_select/test_hpu_select_ids.py 2>&1 \
            | tee "$OUT/select_ids.log" | grep --line-buffered -E "QSAIDS|passed|failed|skipped|rror" \
            | awk '{ print substr($0, 1, 600); fflush() }'
    )
    VERDICT="$("$PY" - "$OUT/select_ids.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("0|RULE9 VERDICT: no result file (%r): the check did not run; the server runs the default form" % (e,))
    sys.exit(0)
def right(name):
    return d.get("variant/" + name, {}).get("in_graph_consumer_wrong_rows") == 0
names = [k[len("variant/"):] for k in d if k.startswith("variant/") and "in_graph_consumer_wrong_rows" in d[k]]
cast = d.get("variant/cast", {})
plugin = {k[len("plugin/"):]: ("ok" if v.get("ok") else "WRONG") for k, v in d.items() if k.startswith("plugin/")}
mode = 0
if not (d.get("plugin/select_blocks", {}).get("ok") or right("gather_arange")):
    mode = 1 if right("cast") else (2 if right("via_fp32") else 0)
rows = cast.get("rows")
direct = d.get("variant/cast_direct_read", {})
print("%d|RULE9 VERDICT: the cast topk(...).indices.to(int32) in one graph: read directly as the graph's output "
      "(job 00038's read) %s of %s ids differ; as select_blocks uses it (mask, expansion) the host read is wrong on "
      "%s of %s rows and the in-graph consumer on %s of %s rows%s; forms right in the graph: %s; forms wrong in the "
      "graph: %s; the plugin as it is (a gather from an int32 arange): %s; the server runs "
      "VLLM_HPU_QSA_TOPK_IDS=%d" % (
          mode, direct.get("differ", direct.get("raised")), direct.get("elements"),
          cast.get("host_read_wrong_rows", cast.get("host_read_raised")), rows,
          cast.get("in_graph_consumer_wrong_rows", cast.get("in_graph_consumer_raised")), rows,
          (" (" + cast["host_read_detail"] + ")") if cast.get("host_read_detail") else "",
          ",".join(n for n in names if right(n)) or "none", ",".join(n for n in names if not right(n)) or "none",
          json.dumps(plugin, sort_keys=True), mode))
PY
)"
    TOPK_IDS="${VERDICT%%|*}"
    echo "${VERDICT#*|}" | tee "$OUT/summary.txt"
    case "$TOPK_IDS" in 0 | 1 | 2) ;; *) TOPK_IDS=0 ;; esac
    sleep 8 # the check's process has released its card before the server takes all of them
fi
export VLLM_HPU_QSA_TOPK_IDS="$TOPK_IDS"

# ---------------------------------------------------------------- 2. the server (serve_probe.sh's command)
for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done
cd "$OUT" || exit 2
SERVE=(vllm serve "$MODEL" --served-model-name fn --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP"
    --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32 --enable-prefix-caching
    --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192
    --long-prefill-token-threshold 2048 --enable-prompt-tokens-details)
{
    "$PY" -c "import vllm, vllm_gaudi; print('vllm', vllm.__version__); print('vllm_gaudi', vllm_gaudi.__file__)" 2>&1
    echo "image: $(cat /farm-image 2>/dev/null)"
    echo "arm: $ARM; cards: ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), port $PORT"
    printf 'serve:'
    printf ' %q' "${SERVE[@]}"
    printf '\n'
    env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FLASHINFER|HCCL_|GC_|QSA_)' | sort
} >versions.txt

SP=""
CHILD=""
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
}
on_signal() {
    trap '' INT TERM
    say "interrupted (SIG$1): stopping the server"
    if [ -n "$CHILD" ]; then
        pkill -TERM -P "$CHILD" 2>/dev/null
        kill -TERM "$CHILD" 2>/dev/null
    fi
    stop_server
    exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

STARTED="$(date -u +%s)"
setsid "${SERVE[@]}" >server0.log 2>&1 &
SP=$!
say "step 2: server starting (tp $TP, expert parallel, port $PORT, VLLM_HPU_QSA_TOPK_IDS=$TOPK_IDS)"
while :; do
    if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SP" 2>/dev/null; then
        say "SERVER-DIED after $(($(date -u +%s) - STARTED)) s"
        grep -nE "Error|Traceback|error:" server0.log | tail -n 25 | cut -c1-400
        tail -n 25 server0.log | cut -c1-400
        wait "$SP" 2>/dev/null
        exit 3
    fi
    if [ "$(left)" -lt 420 ]; then
        say "SERVER-TIMEOUT: not healthy after $(($(date -u +%s) - STARTED)) s, $(left) s to the deadline"
        tail -n 30 server0.log | cut -c1-300
        stop_server
        exit 4
    fi
    sleep 5
done
say "SERVER-HEALTHY after $(($(date -u +%s) - STARTED)) s"
grep -E "Maximum concurrency|QSA fused" server0.log | tail -n 6 | cut -c1-300

# ---------------------------------------------------------------- 3. the probe
RESERVE=150 # stopping the server, the summary
if [ "$ARM" = mode1 ]; then RESERVE=420; fi # and the kernel op test
probe() {
    "$PY" "$TREE/tools/flash_next/rental/qsa_ab_probe.py" --port "$PORT" --model fn --out "$OUT/qsa_ab.json" \
        --levels "$LEVELS" --tokens "$TOKENS" --ctx "$CTX" --docs "$TREE/docs/design/flash-next" \
        --deadline "$((DEADLINE - RESERVE))" 2>&1 | tee probe.log
    return "${PIPESTATUS[0]}"
}
probe &
CHILD=$!
wait "$CHILD"
PROBE_RC=$?
CHILD=""
say "PROBE-RC $PROBE_RC"
grep -E "generation throughput" server0.log | tail -n 4 | cut -c1-300
say "fallback/off lines in the server log: $(grep -ciE "fallback|is off for|has no HPU kernel|kernels are not loaded" server0.log)"
grep -hE "is off for|kernels are not loaded|Traceback" server0.log | sort | uniq -c | head -n 6 | cut -c1-300
say "decode configurations compiled at run time: $(grep -o "Configuration: ('decode'[^)]*)" server0.log | sort -u | wc -l)," \
    "prompt configurations: $(grep -o "Configuration: ('prompt'[^)]*)" server0.log | sort -u | wc -l)"
stop_server
RC=0
if [ "$PROBE_RC" -ne 0 ]; then RC=5; fi

# ---------------------------------------------------------------- 4. the kernels' card test (arm mode1)
if [ "$ARM" = mode1 ] && [ "${QSA_AB_SKIP_OPS:-0}" != 1 ]; then
    if [ "$(left)" -lt 150 ]; then
        say "step 4 SKIPPED: $(left) s to the deadline (the QSA kernel op test wants 150 s)"
    elif [ -z "$LIB" ] || [ ! -f "$LIB/libqsa_fused_gaudi2.so" ]; then
        say "step 4 SKIPPED: no installed libqsa_fused_gaudi2.so (lib folder '${LIB:-none}')"
    else
        say "step 4: the QSA TPC kernels' card test on one card"
        sleep 8
        (
            cd "$TREE" || exit 2
            unset PT_HPU_RECIPE_CACHE_CONFIG VLLM_HPU_QSA_FUSED VLLM_HPU_QSA_INDEXER_SHARD VLLM_HPU_HC_FUSED VLLM_HPU_MOE_GLUE
            export PYTHONPATH="$TREE${PYTHONPATH:+:$PYTHONPATH}" PT_HPU_LAZY_MODE=1
            export QSA_FUSED_GAUDI_KERNEL_LIBRARY="$LIB/libqsa_fused_gaudi2.so"
            for f in "$LIB"/qsa_fused_ops*.so; do [ -f "$f" ] && export QSA_FUSED_GAUDI_NATIVE_LIBRARY="$f"; done
            [ -f "$LIB/libgdn_fused_gaudi2.so" ] && export GDN_FUSED_GAUDI_KERNEL_LIBRARY="$LIB/libgdn_fused_gaudi2.so"
            timeout "$(($(left) - 20))" "$PY" -m pytest -q -s -rs -p no:cacheprovider tests/unit_tests/qsa_fused/test_hpu_op.py 2>&1 \
                | tee "$OUT/qsa_hpu_op.log" | grep --line-buffered -E "QSAHPU|passed|failed|skipped|rror" \
                | awk '{ print substr($0, 1, 400); fflush() }'
        )
        say "QSA KERNEL OP TEST: $(grep -E "passed|failed|skipped|error" "$OUT/qsa_hpu_op.log" | tail -n 1)"
    fi
fi
say "DONE (exit $RC)"
exit "$RC"
