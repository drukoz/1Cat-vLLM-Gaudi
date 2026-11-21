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

# The probe and the selection check: this tree's, or the copies next to this script when it runs from another
# folder (a later version of the three files on an image built from an earlier commit: the job is late-bound).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$TREE/tools/flash_next/rental/qsa_ab_probe.py"
IDS_TEST="tests/unit_tests/qsa_select/test_hpu_select_ids.py"
if [ -f "$HERE/test_hpu_select_ids.py" ]; then IDS_TEST="$HERE/test_hpu_select_ids.py"; fi
if [ "$HERE" != "$TREE/tools/flash_next/rental" ] && [ -f "$HERE/qsa_ab_probe.py" ]; then PROBE="$HERE/qsa_ab_probe.py"; fi

say "arm $ARM, cards ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}), commit ${FARM_SHA:-?}, $(left) s to the deadline;" \
    "script $(md5sum "${BASH_SOURCE[0]}" | cut -c1-8), probe $(md5sum "$PROBE" | cut -c1-8), check $(cd "$TREE" && md5sum "$IDS_TEST" | cut -c1-8);" \
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
        timeout 420 "$PY" -m pytest -q -s -rs -p no:cacheprovider "$IDS_TEST" 2>&1 \
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
# QSA_AB_SERVE_EXTRA: more vllm serve words, comma-separated, one word per argument (--kv-cache-dtype=fp8_inc).
if [ -n "${QSA_AB_SERVE_EXTRA:-}" ]; then
    IFS=, read -r -a SERVE_EXTRA <<<"$QSA_AB_SERVE_EXTRA"
    SERVE+=("${SERVE_EXTRA[@]}")
fi
# QSA_AB_MTP=K (K > 0): MTP speculative decoding with K draft tokens.
case "${QSA_AB_MTP:-0}" in
    '' | 0) ;;
    *[!0-9]*) echo "QSA_AB_MTP must be a number of draft tokens, got '$QSA_AB_MTP'" >&2 ;;
    *) SERVE+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $QSA_AB_MTP}") ;;
esac
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

RESERVE=150 # stopping the server, the summary
if [ "$ARM" = mode1 ]; then RESERVE=420; fi # and the kernel op test

# serve_and_probe TAG: start the server (log server<TAG>.log), wait until it is healthy, run the probe
# (qsa_ab<TAG>.json, probe<TAG>.log), stop the server. Returns 0 the probe ran, 3 the server died before it was
# healthy, 4 it was not healthy in time, 5 the probe failed. RUN_OK=1 when every round of the probe had all its
# streams and the server was still alive at the end.
RUN_OK=0
serve_and_probe() {
    local tag="$1" started log="server$1.log" json="qsa_ab$1.json" rc
    RUN_OK=0
    started="$(date -u +%s)"
    if [ "${QSA_AB_PLACE:-0}" = 1 ] && [ -f "$HERE/place_wrap.sh" ] && [ -f "$HERE/place.sh" ]; then
        # BRIEF 09:17 UTC: every real-model server under collectives' place_wrap.sh (it resets the CPU mask first)
        # and native-replay's place.sh after it is healthy; no VLLM_HPU_WORKER_CPU_PIN; one wrap per server start.
        say "placement: place_wrap.sh $(md5sum "$HERE/place_wrap.sh" | cut -c1-8), place.sh" \
            "$(md5sum "$HERE/place.sh" | cut -c1-8), log placement$tag.log"
        PLACE_PORT="$PORT" PLACE_LOG="$PWD/placement$tag.log" PLACE_ARGS="--log $PWD/$log" \
            setsid bash "$HERE/place_wrap.sh" "$HERE/place.sh" -- "${SERVE[@]}" >"$log" 2>&1 &
    else
        setsid "${SERVE[@]}" >"$log" 2>&1 &
    fi
    SP=$!
    say "server starting (tp $TP, expert parallel, port $PORT, VLLM_HPU_QSA_TOPK_IDS=$TOPK_IDS," \
        "VLLM_HPU_QSA_FUSED=${VLLM_HPU_QSA_FUSED:-0} VLLM_HPU_QSA_INDEXER_SHARD=${VLLM_HPU_QSA_INDEXER_SHARD:-0}), log $log"
    while :; do
        if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$SP" 2>/dev/null; then
            say "SERVER-DIED after $(($(date -u +%s) - started)) s"
            grep -nE "Error|Traceback|error:" "$log" | tail -n 25 | cut -c1-400
            tail -n 25 "$log" | cut -c1-400
            wait "$SP" 2>/dev/null
            SP=""
            return 3
        fi
        if [ "$(left)" -lt 420 ]; then
            say "SERVER-TIMEOUT: not healthy after $(($(date -u +%s) - started)) s, $(left) s to the deadline"
            tail -n 30 "$log" | cut -c1-300
            stop_server
            return 4
        fi
        sleep 5
    done
    say "SERVER-HEALTHY after $(($(date -u +%s) - started)) s"
    grep -E "Maximum concurrency|QSA fused" "$log" | tail -n 6 | cut -c1-300
    probe() {
        "$PY" "$PROBE" --port "$PORT" --model fn --out "$OUT/$json" \
            --levels "$LEVELS" --tokens "$TOKENS" --ctx "$CTX" --docs "$TREE/docs/design/flash-next" \
            ${QSA_AB_LONG_STREAMS:+--long-streams "$QSA_AB_LONG_STREAMS"} \
            ${QSA_AB_LONG_REPEAT:+--long-repeat "$QSA_AB_LONG_REPEAT"} \
            ${QSA_AB_REPEAT_TOP:+--repeat-top "$QSA_AB_REPEAT_TOP"} \
            --deadline "$((DEADLINE - RESERVE))" 2>&1 | tee "probe$tag.log"
        return "${PIPESTATUS[0]}"
    }
    probe &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    CHILD=""
    say "PROBE-RC $rc"
    local alive=1
    kill -0 "$SP" 2>/dev/null || alive=0
    if [ "$alive" -eq 0 ]; then
        say "THE SERVER DIED DURING THE PROBE"
        grep -nE "Error|Traceback|error:" "$log" | tail -n 20 | cut -c1-400
    fi
    grep -E "generation throughput" "$log" | tail -n 4 | cut -c1-300
    grep -E "GPU KV cache size|reserved for usable KV cache|Recurrent state pool: .* slots x" "$log" | head -n 4 \
        | cut -c1-300
    say "fallback/off lines in the server log: $(grep -ciE "fallback|is off for|has no HPU kernel|kernels are not loaded" "$log")"
    grep -hE "is off for|kernels are not loaded|Traceback" "$log" | sort | uniq -c | head -n 6 | cut -c1-300
    say "decode configurations compiled at run time: $(grep -o "Configuration: ('decode'[^)]*)" "$log" | sort -u | wc -l)," \
        "prompt configurations: $(grep -o "Configuration: ('prompt'[^)]*)" "$log" | sort -u | wc -l)"
    stop_server
    RUN_OK="$("$PY" - "$OUT/$json" "$alive" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(0)
    sys.exit(0)
rounds = d.get("levels", []) + d.get("long", [])
bad = [r for r in rounds if not r.get("ok") or r.get("errors")]
print(1 if rounds and not bad and sys.argv[2] == "1" else 0)
PY
)"
    if [ "$rc" -ne 0 ]; then return 5; fi
    return 0
}

# Device memory every 15 s (BRIEF.md 23:50 UTC: a server can run out of device memory at run time): hlsmi.log.
MEMLOG=""
if command -v hl-smi >/dev/null 2>&1; then
    (while :; do
        echo "$(date -u +%H:%M:%S) $(hl-smi --query-aip=index,memory.used,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
        sleep 15
    done) >"$OUT/hlsmi.log" 2>&1 &
    MEMLOG=$!
fi
say "step 2 and 3: the server and the probe (VLLM_GRAPH_RESERVED_MEM=${VLLM_GRAPH_RESERVED_MEM:-default}," \
    "long streams ${QSA_AB_LONG_STREAMS:-auto}, long repeats ${QSA_AB_LONG_REPEAT:-default})"
serve_and_probe ""
RC=$?
# The request-sharded indexer on its first run in a served model: when its server dies or a round loses streams,
# the arm runs once more without it (mode 1 replicated), so the arm still gives the request-major and grouped
# attention numbers.
if [ "$RUN_OK" != 1 ] && [ "${VLLM_HPU_QSA_INDEXER_SHARD:-0}" != 0 ] && [ "${QSA_AB_NO_RETRY:-0}" != 1 ]; then
    if [ "$(left)" -ge 1080 ]; then
        say "FALLBACK: the run with VLLM_HPU_QSA_INDEXER_SHARD=$VLLM_HPU_QSA_INDEXER_SHARD did not complete (exit $RC): once more with it at 0"
        export VLLM_HPU_QSA_INDEXER_SHARD=0
        sleep 10
        serve_and_probe "_noshard"
        RC=$?
    else
        say "NO FALLBACK RUN: $(left) s to the deadline (1080 s wanted)"
    fi
fi
if [ -n "$MEMLOG" ]; then kill "$MEMLOG" 2>/dev/null; fi
say "device memory, highest used per card (hlsmi.log): $(sed 's/^[0-9][0-9]:[0-9][0-9]:[0-9][0-9] //' "$OUT/hlsmi.log" 2>/dev/null | tr ';' '\n' | awk -F', *' 'NF >= 3 {g = $2; sub(/ .*/, "", g); if (g + 0 > m[$1]) m[$1] = g + 0} END {for (k in m) printf "%s:%d MiB ", k, m[k]}')"
if [ "$RC" -eq 3 ] || [ "$RC" -eq 4 ]; then
    say "DONE (exit $RC)"
    exit "$RC"
fi

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
