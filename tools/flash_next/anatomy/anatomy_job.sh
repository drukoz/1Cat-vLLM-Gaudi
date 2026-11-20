#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# anatomy_job.sh PLAN: the step-anatomy card job, run inside the farm-base container (CARD.md).
#
# Mounts: /bundle = the track's unpacked rental bundle (plugin/, vllm/), /results = the job's results folder,
# /models read-only. The script installs the stack from the bundle (container_entry.sh --install-only), then
# runs the plan: one or more real-model servers (TP8 + EP8, lazy, accumulation mode 0, fused GDN decode and
# fused hyper-connection kernels, PLE host n-gram ids, no collective mark_step, warmup skipped: the flags of
# the coordinator's scale-base job), each with the step timer on and anatomy_probe.py against it, and the
# one-card benchmarks the plan names.
#
# Plans: S = one server with the timer: spans at 1, 8, 32 streams (one round with min_tokens, three tagged for a
# perf sampler on the node), device waits (mode 3), host sync points (mode 2). M = S plus decode rounds at longer
# contexts, then a second server with the torch profiler (HPU activities) and one capture of steady decode steps
# at 32 and at 1 stream. T8 = S, then the component server (the 8-card form). L = VLLM_CONFIG_HIDDEN_LAYERS=4
# against the default 1 (I10 / G4-2), then the component server: no HPU graphs (--enforce-eager), mode 3 with
# VLLM_HPU_STEP_TIMING_COMPONENTS=1, the device time of every model component at 1, 8 and 32 streams. R = a
# rehearsal of S and the component server on the tiny model (2 or 4 cards, minutes). A = S with the capture server and the one-card benches (job 00013's plan). B = the one-card
# MME row sweep (G1-1) and the small lazy-op cases, then VLLM_CONFIG_HIDDEN_LAYERS 4 (and 12 with time left).
# Q = one server, one short phase (a check of this script's plumbing). T = the profiler capture with the Habana
# SDK in API-controlled mode (HABANA_PROFILE=1 and a profiler configuration file).
# P = the device time per model component against the rows (MTP's verification rows): one server with room for
# 128 sequences and the torch profiler, timer rounds, then one capture each in a 32- and a 128-stream decode round
# (read on the workstation with trace_components.py).
#
# A stage-2 job reads the file anatomy_late.sh next to its results folder (on the node:
# ~/farm/<track>/results/anatomy_late.sh) before its plan starts: it may set ANATOMY_S_PHASES, ANATOMY_C_PHASES,
# ANATOMY_TRACE_PHASES, ANATOMY_TRACE_LEVELS, ANATOMY_L_PHASES, ANATOMY_E_PHASES, ANATOMY_LAYERS, ANATOMY_ONE_SERVER,
# ANATOMY_SECOND (plan T8's second server: eager or trace),
# so a job that waits in the queue can still be given what an earlier job taught.
#
# Every server: $RESULTS/<label>/{server0.log, probe.json, timing/step_timing_*.jsonl, versions.txt}.
set -u

PLAN="${1:-A}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/anatomy_probe.py"
# Stage 1 (CARD.md stage 1: own docker run, the bundle mounted at /bundle, results at /results, 8 cards) or stage 2
# (`farm job`: the plugin installed in the job's image, results in $FARM_OUT, the job's cards and port base).
if [ -n "${FARM_OUT:-}" ]; then
    STAGE=2
    RESULTS="$FARM_OUT"
    PORT="${ANATOMY_PORT:-${FARM_PORT_BASE:-8100}}"
    TP="${FARM_NCARDS:-8}"
else
    STAGE=1
    RESULTS=/results
    PORT="${ANATOMY_PORT:-8100}"
    TP=8
fi
MODEL="${ANATOMY_MODEL:-/models/Qwen3.8-Flash-Next-FP8}"
TINY=0  # plan R: the tiny model of the rental gate stands in for the real one
# What every server of the real model is started with (job 00007's arguments).
MODEL_ARGS=(--enable-expert-parallel --max-model-len 131072 --max-num-seqs 32 --enable-prefix-caching
    --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192
    --long-prefill-token-threshold 2048 --enable-prompt-tokens-details)
ENTRY=/bundle/plugin/tools/flash_next/rental/container_entry.sh
mkdir -p "$RESULTS" && cd "$RESULTS" || exit 2

say() { echo "anatomy: $* [$(date -u +%H:%M:%S)]"; }

install_stack() {
    if [ "$STAGE" -eq 2 ]; then
        python3 -c "from vllm_gaudi.extension import step_timing; print('stage 2: the image has the plugin;', step_timing.__file__)" \
            || { say "the job's image has no step timer"; exit 3; }
        return 0
    fi
    say "install from the bundle ($(cat /bundle/BUNDLE.txt 2>/dev/null | tr '\n' ' ' | cut -c1-200))"
    if ! bash "$ENTRY" --install-only --results "$RESULTS" --run-id install > "$RESULTS/install.log" 2>&1; then
        say "INSTALL FAILED"; tail -n 30 "$RESULTS/install.log"; exit 3
    fi
    python3 -c "import vllm, vllm_gaudi, os; print('vllm', vllm.__version__, 'vllm_gaudi', os.path.dirname(vllm_gaudi.__file__))"
    python3 -c "from vllm_gaudi.extension import step_timing; print('step timer module:', step_timing.__file__)" \
        || { say "the installed plugin has no step timer"; exit 3; }
    say "install done"
}

# The coordinator's scale-base flags (coord/serve_and_probe.sh, job 00007).
base_env() {
    export PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1
    export VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1
    export VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true
    [ "$STAGE" -eq 1 ] && export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7  # stage 2: the job's own modules
    unset HABANA_PROFILE HABANA_PROF_CONFIG VLLM_CONFIG_HIDDEN_LAYERS VLLM_TORCH_PROFILER_DIR VLLM_HPU_FUSED_GREEDY_LOGITS
    unset VLLM_HPU_STEP_TIMING_COMPONENTS
    # The tiny model runs with the flags of the rental gate's hc mode only.
    if [ "$TINY" -eq 1 ]; then
        unset VLLM_HPU_FLASH_NEXT_QSA_MODE VLLM_HPU_RECURRENT_STATE_POOL VLLM_HPU_PLE_HOST_NGRAM_IDS
    fi
}

SP=""
stop_server() {
    [ -n "$SP" ] || return 0
    kill -INT "$SP" 2>/dev/null
    for _ in $(seq 1 24); do kill -0 "$SP" 2>/dev/null || break; sleep 5; done
    kill -9 "$SP" 2>/dev/null
    pkill -9 -f "vllm serve" 2>/dev/null
    # A worker that hangs in a step outlives its server and keeps its card: its process title is VLLM::Worker_TP<n>,
    # which "vllm serve" does not match (job 00013: the next server and the one-card benches failed with "Device
    # acquire failed"). Kill engine and workers by title and wait until none is left.
    pkill -9 -f "VLLM::" 2>/dev/null
    for _ in $(seq 1 30); do pgrep -f "VLLM::|vllm serve" > /dev/null 2>&1 || break; sleep 1; done
    if pgrep -f "VLLM::|vllm serve" > /dev/null 2>&1; then say "processes of the stopped server are still alive:"; pgrep -af "VLLM::|vllm serve" | cut -c1-120; fi
    sleep 5
    SP=""
}
trap stop_server EXIT

# serve LABEL PHASES TRACE [extra vllm serve args...]: one server, the probe, the records. The caller sets the
# server environment (base_env plus its own exports) before the call.
serve() {
    local label="$1" phases="$2" trace="$3"
    shift 3
    local out="$RESULTS/$label" t0 rc
    mkdir -p "$out/timing"
    cd "$out" || return 2
    export VLLM_HPU_STEP_TIMING="${VLLM_HPU_STEP_TIMING:-1}" VLLM_HPU_STEP_TIMING_DIR="$out/timing"
    export VLLM_HPU_STEP_TIMING_EVERY="${VLLM_HPU_STEP_TIMING_EVERY:-4000}"
    echo "1" > "$out/timing/step_timing.mode"
    { echo "label $label"; echo "phases $phases"; echo "trace $trace"; echo "extra args: $*"
      env | grep -E '^(VLLM_|PT_HPU|HABANA_|FLASHINFER)' | sort; } > versions.txt
    say "$label: starting the server"
    # shellcheck disable=SC2086  # ANATOMY_SERVE_EXTRA: vllm serve options of the plan (P: --max-num-seqs 128)
    vllm serve "$MODEL" --served-model-name fn --host 127.0.0.1 --port "$PORT" \
        --tensor-parallel-size "$TP" "${MODEL_ARGS[@]}" "$@" ${ANATOMY_SERVE_EXTRA:-} > server0.log 2>&1 &
    SP=$!
    t0=$(date +%s)
    while :; do
        if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$SP" 2>/dev/null; then
            say "$label: SERVER-DIED after $(( $(date +%s) - t0 )) s"
            grep -nE "Error|Traceback|error:" server0.log | tail -n 25 | cut -c1-400
            tail -n 20 server0.log | cut -c1-400
            SP=""; cd "$RESULTS" || exit 2; return 3
        fi
        if [ $(( $(date +%s) - t0 )) -ge 1800 ]; then
            say "$label: SERVER-TIMEOUT"; tail -n 20 server0.log | cut -c1-300; stop_server; cd "$RESULTS" || exit 2; return 4
        fi
        sleep 5
    done
    say "$label: SERVER-HEALTHY after $(( $(date +%s) - t0 )) s"
    grep -E "Maximum concurrency|Usable num_blocks|Loading model weights took|Wrapping in HPUGraph|step timing: on" \
        server0.log | head -n 8 | cut -c1-260
    # shellcheck disable=SC2086  # PROBE_EXTRA is a list of probe options
    python3 "$PROBE" --port "$PORT" --model fn --timing-dir "$out/timing" --phases "$phases" \
        ${trace:+--trace "$trace"} ${PROBE_EXTRA:-} ${ANATOMY_PROBE_ARGS:-} --out "$out/probe.json" 2>&1 | tee "$out/probe.log"
    rc=${PIPESTATUS[0]}
    say "$label: probe rc $rc"
    [ -f "$HERE/summarize_timing.py" ] && python3 "$HERE/summarize_timing.py" "$out" --json "$out/timing_summary.json" \
        > "$out/timing_summary.md" 2>&1 && head -n 70 "$out/timing_summary.md" | cut -c1-400
    grep -E "step timing: worker rank 0 decode" server0.log | tail -n 14 | cut -c1-700
    grep -ciE "fallback|is off for|has no HPU kernel" server0.log | xargs echo "$label: fallback/off lines:"
    grep -c "was not warmed-up" server0.log | xargs echo "$label: live bucket compiles (not warmed-up lines, all ranks):"
    stop_server
    ls -la "$out/timing" | tail -n 20
    du -sh "$out"/* 2>/dev/null | sort -h | tail -n 8
    cd "$RESULTS" || exit 2
    return "$rc"
}

plan_q() {
    base_env
    serve q-smoke "warm:1:16;1:1:16" ""
}

# The timer rounds of one server. Mode 4 (dispatched ops) is not among them: in job 00013 its 23rd one-stream step
# never returned (sample_tokens RPC timeout after 300 s, the engine died) and took the rest of the job along.
# One :min round only: on HPU the min_tokens mask never runs (engine-step, hpu_input_batch.py:554-562), the round
# confirms it. The rounds tagged perf and strace are for a sampler on the node (perf record of the workers, the
# system calls of the rank-0 worker: reports/step-anatomy/perf/watch.sh); mode 3 on a replayed graph splits the wait
# into the forward's device time and the sampler's.
S_PHASES="warm:1,8,32:128;1:1,8,32:128;1:32:128:min;1:32,8,1:128:tag=perf;1:1:128:tag=strace;3:32,8,1:64;2:32:24"
# Decode at longer contexts (target-model prediction 1): 8 streams with about 4,000- and 16,000-token prompts; the
# warm round fills the prefix cache, so the measured round decodes almost at once.
C_PHASES="warm:8:64:ctx=4000;1:8:64:ctx=4000;warm:8:64:ctx=16000;1:8:64:ctx=16000"

plan_s() {
    base_env
    serve s-timer "${ANATOMY_S_PHASES:-$S_PHASES}" ""
}

# eager_server: the device time of every model component. No HPU graphs (--enforce-eager: the forward runs in
# Python every step), timer mode 3 with VLLM_HPU_STEP_TIMING_COMPONENTS=1: the host waits for the device before and
# after every collective and where a component starts and ends (hyper-connection, GDN, QSA indexer and attention,
# MoE and its expert op, PLE), in the warm rounds too, so the measured rounds compile nothing. The step itself is
# several times slower than a replayed one: the component times, not the total, are the result. (A synthetic
# 8-rank replay with a mark_step before every collective stalled once, job 00012; the morning's real-model runs
# had one before every collective and did not.)
E_PHASES="warm:1,8,32:24;3:1,8,32:32"
eager_server() {
    base_env
    export VLLM_HPU_STEP_TIMING_COMPONENTS=1
    PROBE_EXTRA="--warm-mode 3" serve e-eager "${ANATOMY_E_PHASES:-$E_PHASES}" "" --enforce-eager
    local rc=$?
    grep -E "step timing: component cuts" "$RESULTS/e-eager/server0.log" | head -n 2 | cut -c1-300
    unset VLLM_HPU_STEP_TIMING_COMPONENTS
    return "$rc"
}

# trace_server LABEL: the torch profiler with HPU activities; mode 1 again (does the profiler setting cost time
# when it is not capturing?), then one capture in the middle of a 32-stream and a 1-stream decode round. The
# capture rounds are as long as the warm rounds (160 tokens): no bucket compiles inside a capture, and every
# stream still decodes for seconds after the capture starts.
trace_server() {
    local label="$1"
    base_env
    export HABANA_PROFILE="${ANATOMY_HABANA_PROFILE:-profile_api_light}"
    PROBE_EXTRA="--trace-tokens 160" serve "$label" "${ANATOMY_TRACE_PHASES:-warm:32,1:160;1:32,1:128}" \
        "${ANATOMY_TRACE_LEVELS:-32,1}" \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "'"$RESULTS/$label"'/trace", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": true, "torch_profiler_record_shapes": false}'
    local rc=$?
    ls -la "$RESULTS/$label/trace" 2>/dev/null | head -n 20
    return "$rc"
}

plan_m() {
    base_env
    if [ "${ANATOMY_ONE_SERVER:-0}" = "1" ]; then
        # One server for the timer rounds and the capture (when an earlier job showed that the profiler setting
        # costs nothing while it is not capturing).
        ANATOMY_TRACE_PHASES="${ANATOMY_S_PHASES:-$S_PHASES};${ANATOMY_C_PHASES-$C_PHASES}" trace_server m-timer-trace
        return
    fi
    serve s-timer "${ANATOMY_S_PHASES:-$S_PHASES};${ANATOMY_C_PHASES-$C_PHASES}" ""
    trace_server m-trace
}

plan_t8() {
    # The 8-card form: the timer rounds on the replayed graphs first, then (ANATOMY_SECOND, default eager) the
    # component server or the capture server when the hold leaves the eleven minutes it needs.
    if [ "${ANATOMY_ONE_SERVER:-0}" = "1" ]; then
        ANATOMY_TRACE_PHASES="${ANATOMY_S_PHASES:-$S_PHASES}" trace_server s-timer-trace
        return
    fi
    plan_s
    if [ $(( ${ANATOMY_HOLD_S:-1500} - SECONDS )) -lt 660 ]; then
        say "no time left for a second server ($SECONDS s used of ${ANATOMY_HOLD_S:-1500})"
    elif [ "${ANATOMY_SECOND:-eager}" = "trace" ]; then
        ANATOMY_TRACE_PHASES="${ANATOMY_TRACE_PHASES:-warm:32,1:160}" trace_server m-trace
    else
        eager_server
    fi
}

plan_l() {
    # I10 / G4-2: the per-decoder-layer mark_step every 4 layers (default: every layer). Compare with plan S's
    # s-timer of the same commit on the same card count.
    base_env
    export VLLM_CONFIG_HIDDEN_LAYERS="${ANATOMY_LAYERS:-4}"
    serve "l-layers$VLLM_CONFIG_HIDDEN_LAYERS" "${ANATOMY_L_PHASES:-warm:1,8,32:128;1:1,8,32:128;1:32,1:128;3:32,1:64}" ""
    if [ $(( ${ANATOMY_HOLD_S:-2400} - SECONDS )) -lt 840 ]; then
        say "no time left for the eager mode-3 server ($SECONDS s used of ${ANATOMY_HOLD_S:-2400})"
        return
    fi
    eager_server
}

plan_r() {
    # R: a rehearsal of plans S and E on the tiny model of the rental gate (prebuilt in the job's image), on 2 or
    # 4 cards in minutes: the timer rounds on replayed graphs, a round tagged for the perf sampler, then the
    # component server (--enforce-eager, mode 3, component cuts). It checks the mechanics, not the numbers.
    TINY=1
    MODEL="${ANATOMY_MODEL:-/tmp/fn-smoke-ckpt}"
    local len
    len=$(python3 -c "import sys; from pathlib import Path; from tools.flash_next.rental.main import tiny_model_len; print(tiny_model_len(Path(sys.argv[1])))" "$MODEL" 2> /dev/null)
    MODEL_ARGS=(--max-model-len "${len:-512}" --max-num-seqs 4 --language-model-only --trust-remote-code)
    say "tiny model $MODEL, model length ${len:-512}"
    export ANATOMY_PROBE_ARGS="--prompt-ids 8"
    # 400-token rounds: a tiny step takes milliseconds, and the node's sampler needs seconds of steady decode.
    ANATOMY_S_PHASES="${ANATOMY_S_PHASES:-warm:1,4:400;1:1,4:400;1:4:24:min;1:4,1:400:tag=perf;1:1:400:tag=strace;3:4,1:16;2:4:8}" plan_s
    ANATOMY_E_PHASES="${ANATOMY_E_PHASES:-warm:1,4:12;3:1,4:16}" eager_server
}

plan_a() {
    # Job 00013's plan with what it taught: the timer server, the capture server, then the one-card benches.
    plan_s
    local a1=$?
    if [ "$a1" -eq 3 ] || [ "$a1" -eq 4 ]; then
        say "s-timer did not serve (rc $a1): the profiler server runs with VLLM_HPU_STEP_TIMING=0"
        export VLLM_HPU_STEP_TIMING=0
    fi
    trace_server a2-trace
    # The one-card benches with what is left of the job's 50-minute hold (SIGTERM at 3,000 s), 2 minutes spare.
    local left=$(( ${ANATOMY_HOLD_S:-3000} - SECONDS - 120 ))
    if [ "$left" -ge 240 ]; then
        run_benches $(( left > 600 ? 600 : left ))
    else
        say "no time left for the one-card benches ($SECONDS s used)"
    fi
}

plan_t() {
    # T: the profiler capture again with the Habana SDK in API-controlled mode (HABANA_PROFILE=1, host profiler
    # started disabled, hardware trace in the profiler-API phase: the configuration of tools/run_deepseek_v41.py),
    # for a run where profile_api_light gave no device events.
    base_env
    mkdir -p $RESULTS/t-trace
    cat > $RESULTS/t-trace/prof-config.json <<'JSON'
{"Plugins": [
 {"enable": true, "lib": "libhost_profiler.so", "name": "HostProfiler",
  "values": {"api_group": {"HCCL": {"value": true}, "HLTHUNK": {"value": true}, "SYNAPSE": {"value": true}},
             "start_disabled": {"value": true}}},
 {"enable": true, "lib": "libhw_trace.so", "name": "HwTrace",
  "values": {"generalOptions": {"profilePhase": {"value": "profileApi"}, "traceBufferSize": {"value": "0x80000000"}},
             "parseOptions": {"addFuserMetadata": {"value": false}, "showNullDescs": {"value": false}}}}]}
JSON
    export HABANA_PROFILE=1 HABANA_PROF_CONFIG=$RESULTS/t-trace/prof-config.json
    PROBE_EXTRA="--trace-tokens 160" serve t-trace "warm:32,1:160" "32,1" \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "'"$RESULTS"'/t-trace/trace", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": true, "torch_profiler_record_shapes": false}'
    ls -la $RESULTS/t-trace/trace 2>/dev/null | head -n 20
}

# run_benches SECONDS: the one-card benches side by side, each stopped after SECONDS: the MME row sweep (G1-1,
# predictions 3 and 4) one case per card on cards 0-4, and on card 5 the lazy-op cases that job 00009 lost (mix family,
# cuts, mark_step alone, copies, LM head; accumulation mode 0, sizes up to 3,000 under the script's compile budget).
# The copies next to this script kill their children when they are stopped (job 00009: an orphan kept card 0).
run_benches() {
    local limit="$1" pids=() mme_cards="0,1,2,3,4" lazy_card=5
    if [ "$STAGE" -eq 2 ]; then mme_cards=""; lazy_card=""; fi  # a 1-card job: everything on its card, in turn
    say "one-card benches for at most $limit s: MME row sweep (cards ${mme_cards:-of the job}), lazy-op cases (card ${lazy_card:-of the job})"
    PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 timeout "$limit" \
        python3 "$HERE/bench_mme_rows.py" ${mme_cards:+--cards "$mme_cards"} --child-timeout "$limit" \
        --out "$RESULTS/mme_rows.json" > "$RESULTS/mme_rows.log" 2>&1 &
    pids+=($!)
    if [ "$STAGE" -eq 2 ]; then wait "${pids[0]}"; fi  # one card: the two benches in turn, never side by side
    HABANA_VISIBLE_MODULES="${lazy_card:-${HABANA_VISIBLE_MODULES:-}}" timeout "$limit" \
        python3 "$HERE/bench_lazy_ops.py" --acc-modes 0 --child-budget $(( limit > 300 ? limit - 120 : limit / 2 )) \
        --out "$RESULTS/lazy_ops_mode0.json" > "$RESULTS/lazy_ops_mode0.log" 2>&1 &
    pids+=($!)
    for pid in "${pids[@]}"; do wait "$pid"; done
    pkill -9 -f "bench_(mme_rows|lazy_ops).py" 2>/dev/null
    say "one-card benches done"
    tail -n 45 "$RESULTS/mme_rows.log" 2>/dev/null | cut -c1-250
    tail -n 30 "$RESULTS/lazy_ops_mode0.log" 2>/dev/null | cut -c1-250
}

plan_p() {
    # P: what a decode row costs per model component, at 32 and 128 rows (MTP k = 3 verifies 4 rows per stream:
    # 128 rows at 32 streams). One server with room for 128 sequences and the torch profiler (HPU activities): warm
    # rounds at the capture's length (no bucket compile inside a capture), timer rounds, then one capture in the
    # middle of a 32-stream and of a 128-stream round. The traces are read on the workstation with
    # trace_components.py (device time per component, graph launches per step). The profiler's stop takes minutes
    # (job 00113: 140 s at 32 streams on 4 cards): the probe waits up to 600 s for a stream's next chunk.
    ANATOMY_SERVE_EXTRA="--max-num-seqs ${ANATOMY_MAX_SEQS:-128}" \
        ANATOMY_PROBE_ARGS="${ANATOMY_PROBE_ARGS:---measured-timeout 600 --trace-seconds 1.0}" \
        ANATOMY_TRACE_PHASES="${ANATOMY_P_PHASES:-warm:32,128:160;1:32,128:128}" \
        ANATOMY_TRACE_LEVELS="${ANATOMY_P_LEVELS:-32,128}" trace_server p-trace
}

plan_b() {
    run_benches 600
    # I10 / G4-2: the per-decoder-layer mark_step every 4 layers (default every layer). A bigger recipe compiles
    # far longer (00009: 1,000 ops 14 s, 3,000 ops 135 s), so 4 first; 12 only when the job has time left.
    base_env
    export VLLM_CONFIG_HIDDEN_LAYERS=4
    serve b-layers4 "warm:1,32:128;1:1,32:128" ""
    # D4 rider: the in-graph greedy LM head (VLLM_HPU_FUSED_GREEDY_LOGITS=1) engages only for requests without
    # min_tokens, so one server gives both arms: rounds with min_tokens (today's path) and without (fused).
    if [ "$SECONDS" -lt 1800 ]; then
        base_env
        export VLLM_HPU_FUSED_GREEDY_LOGITS=1
        serve b-fused-greedy "warm:1,32:128:min;warm:1,32:128;1:1,32:128:min;1:1,32:128" ""
        unset VLLM_HPU_FUSED_GREEDY_LOGITS
    else
        say "no time left for the fused greedy rider ($SECONDS s used)"
    fi
    if [ "$SECONDS" -lt 1700 ]; then
        base_env
        export VLLM_CONFIG_HIDDEN_LAYERS=12
        serve b-layers12 "warm:1,32:128;1:1,32:128" ""
    else
        say "no time left for VLLM_CONFIG_HIDDEN_LAYERS=12 ($SECONDS s used)"
    fi
}

say "plan $PLAN, stage $STAGE, $TP cards, bundle $(sed -n 's/^plugin sha: //p' /bundle/BUNDLE.txt 2>/dev/null)"
install_stack
LATE="$(dirname "$RESULTS")/anatomy_late.sh"
if [ "$STAGE" -eq 2 ] && [ -f "$LATE" ]; then
    say "late settings from $LATE:"
    cat "$LATE"
    # shellcheck disable=SC1090  # the track's own file on the node
    . "$LATE"
fi
case "$PLAN" in
    A) plan_a ;;
    S) plan_s ;;
    M) plan_m ;;
    T8) plan_t8 ;;
    L) plan_l ;;
    R) plan_r ;;
    B) plan_b ;;
    Q) plan_q ;;
    T) plan_t ;;
    P) plan_p ;;
    *) say "unknown plan $PLAN"; exit 2 ;;
esac
say "plan $PLAN done"
