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
# Plans: A = timer anatomy (modes 1, 4, 2, 3 at 1, 8, 32 streams), then a second server with the torch profiler
# (HPU activities) and one capture of steady decode steps at 32 and at 1 stream. B = the one-card MME row sweep
# (G1-1) and the small lazy-op cases side by side, then VLLM_CONFIG_HIDDEN_LAYERS 4 (and 12 with time left) at
# 1 and 32 streams (I10 / G4-2). Q = one server, one short phase (a check of this script's plumbing). T = the
# profiler capture with the Habana SDK in API-controlled mode (HABANA_PROFILE=1 and a profiler configuration file).
#
# Every server: /results/<label>/{server0.log, probe.json, timing/step_timing_*.jsonl, versions.txt}.
set -u

PLAN="${1:-A}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/anatomy_probe.py"
PORT="${ANATOMY_PORT:-8100}"
MODEL="${ANATOMY_MODEL:-/models/Qwen3.8-Flash-Next-FP8}"
ENTRY=/bundle/plugin/tools/flash_next/rental/container_entry.sh
cd /results || exit 2

say() { echo "anatomy: $* [$(date -u +%H:%M:%S)]"; }

install_stack() {
    say "install from the bundle ($(cat /bundle/BUNDLE.txt 2>/dev/null | tr '\n' ' ' | cut -c1-200))"
    if ! bash "$ENTRY" --install-only --results /results --run-id install > /results/install.log 2>&1; then
        say "INSTALL FAILED"; tail -n 30 /results/install.log; exit 3
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
    export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7
    unset HABANA_PROFILE HABANA_PROF_CONFIG VLLM_CONFIG_HIDDEN_LAYERS VLLM_TORCH_PROFILER_DIR VLLM_HPU_FUSED_GREEDY_LOGITS
}

SP=""
stop_server() {
    [ -n "$SP" ] || return 0
    kill -INT "$SP" 2>/dev/null
    for _ in $(seq 1 36); do kill -0 "$SP" 2>/dev/null || break; sleep 5; done
    kill -9 "$SP" 2>/dev/null
    pkill -9 -f "vllm serve" 2>/dev/null
    sleep 5
    SP=""
}
trap stop_server EXIT

# serve LABEL PHASES TRACE [extra vllm serve args...]: one server, the probe, the records. The caller sets the
# server environment (base_env plus its own exports) before the call.
serve() {
    local label="$1" phases="$2" trace="$3"
    shift 3
    local out="/results/$label" t0 rc
    mkdir -p "$out/timing"
    cd "$out" || return 2
    export VLLM_HPU_STEP_TIMING="${VLLM_HPU_STEP_TIMING:-1}" VLLM_HPU_STEP_TIMING_DIR="$out/timing"
    export VLLM_HPU_STEP_TIMING_EVERY="${VLLM_HPU_STEP_TIMING_EVERY:-4000}"
    echo "1" > "$out/timing/step_timing.mode"
    { echo "label $label"; echo "phases $phases"; echo "trace $trace"; echo "extra args: $*"
      env | grep -E '^(VLLM_|PT_HPU|HABANA_|FLASHINFER)' | sort; } > versions.txt
    say "$label: starting the server"
    vllm serve "$MODEL" --served-model-name fn --host 127.0.0.1 --port "$PORT" \
        --tensor-parallel-size 8 --enable-expert-parallel --max-model-len 131072 --max-num-seqs 32 \
        --enable-prefix-caching --language-model-only --trust-remote-code --gpu-memory-utilization 0.92 \
        --max-num-batched-tokens 8192 --long-prefill-token-threshold 2048 --enable-prompt-tokens-details "$@" \
        > server0.log 2>&1 &
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
            SP=""; cd /results; return 3
        fi
        if [ $(( $(date +%s) - t0 )) -ge 1800 ]; then
            say "$label: SERVER-TIMEOUT"; tail -n 20 server0.log | cut -c1-300; stop_server; cd /results; return 4
        fi
        sleep 5
    done
    say "$label: SERVER-HEALTHY after $(( $(date +%s) - t0 )) s"
    grep -E "Maximum concurrency|Usable num_blocks|Loading model weights took|Wrapping in HPUGraph|step timing: on" \
        server0.log | head -n 8 | cut -c1-260
    # shellcheck disable=SC2086  # PROBE_EXTRA is a list of probe options
    python3 "$PROBE" --port "$PORT" --model fn --timing-dir "$out/timing" --phases "$phases" \
        ${trace:+--trace "$trace"} ${PROBE_EXTRA:-} --out "$out/probe.json" 2>&1 | tee "$out/probe.log"
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
    cd /results
    return "$rc"
}

plan_q() {
    base_env
    serve q-smoke "warm:1:16;1:1:16" ""
}

plan_a() {
    # A1: the timer modes on one server. Warm every level at the measured length first, then mode 1 (spans),
    # 4 (dispatched ops), 2 (host sync points), 3 (device waits around the forward, the sampler and, when the
    # forward runs in Python, every collective). Riskiest last.
    base_env
    serve a1-timer "warm:1,8,32:128;1:1,8,32:128;4:1,32:24;2:32:24;3:1,32:64" ""
    local a1=$?
    # A2: the torch profiler with HPU activities; mode 1 again (does the profiler setting cost time when it is
    # not capturing?), then one capture in the middle of a 32-stream and a 1-stream decode round. When the
    # timed server did not come up, the capture runs without the timer.
    base_env
    if [ "$a1" -eq 3 ] || [ "$a1" -eq 4 ]; then
        say "a1-timer did not serve (rc $a1): the profiler server runs with VLLM_HPU_STEP_TIMING=0"
        export VLLM_HPU_STEP_TIMING=0
    fi
    export HABANA_PROFILE=profile_api_light
    # The capture rounds are as long as the warm rounds (160 tokens): no bucket compiles inside a capture, and
    # every stream still decodes for seconds after the capture starts.
    PROBE_EXTRA="--trace-tokens 160" serve a2-trace "warm:32,1:160;1:32,1:96" "32,1" \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/results/a2-trace/trace", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": true, "torch_profiler_record_shapes": false}'
    ls -la /results/a2-trace/trace 2>/dev/null | head -n 20
}

plan_t() {
    # T: the profiler capture again with the Habana SDK in API-controlled mode (HABANA_PROFILE=1, host profiler
    # started disabled, hardware trace in the profiler-API phase: the configuration of tools/run_deepseek_v41.py),
    # for a run where profile_api_light gave no device events.
    base_env
    mkdir -p /results/t-trace
    cat > /results/t-trace/prof-config.json <<'JSON'
{"Plugins": [
 {"enable": true, "lib": "libhost_profiler.so", "name": "HostProfiler",
  "values": {"api_group": {"HCCL": {"value": true}, "HLTHUNK": {"value": true}, "SYNAPSE": {"value": true}},
             "start_disabled": {"value": true}}},
 {"enable": true, "lib": "libhw_trace.so", "name": "HwTrace",
  "values": {"generalOptions": {"profilePhase": {"value": "profileApi"}, "traceBufferSize": {"value": "0x80000000"}},
             "parseOptions": {"addFuserMetadata": {"value": false}, "showNullDescs": {"value": false}}}}]}
JSON
    export HABANA_PROFILE=1 HABANA_PROF_CONFIG=/results/t-trace/prof-config.json
    PROBE_EXTRA="--trace-tokens 160" serve t-trace "warm:32,1:160" "32,1" \
        --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/results/t-trace/trace", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": true, "torch_profiler_record_shapes": false}'
    ls -la /results/t-trace/trace 2>/dev/null | head -n 20
}

plan_b() {
    # G1-1 while no server holds the cards: one case per card on cards 0-4; on card 5 the lazy-op cases that
    # job 00009 lost (its 3,000- and 10,000-op graphs compile for minutes and time its children out).
    local pids=()
    if [ -f "$HERE/bench_mme_rows.py" ]; then
        say "MME row sweep on cards 0-4"
        PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 timeout 900 \
            python3 "$HERE/bench_mme_rows.py" --cards 0,1,2,3,4 --out /results/mme_rows.json > /results/mme_rows.log 2>&1 &
        pids+=($!)
    fi
    if [ -f /bundle/plugin/tools/flash_next/anatomy/bench_lazy_ops.py ]; then
        say "lazy-op cases on card 5 (sizes 100, 300, 1000; accumulation mode 0)"
        HABANA_VISIBLE_MODULES=5 timeout 900 python3 /bundle/plugin/tools/flash_next/anatomy/bench_lazy_ops.py \
            --sizes 100,300,1000 --acc-modes 0 --child-timeout 800 --out /results/lazy_ops_small.json \
            > /results/lazy_ops_small.log 2>&1 &
        pids+=($!)
    fi
    for pid in "${pids[@]}"; do wait "$pid"; done
    say "one-card benches done"
    tail -n 45 /results/mme_rows.log 2>/dev/null | cut -c1-250
    tail -n 30 /results/lazy_ops_small.log 2>/dev/null | cut -c1-250
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
        serve b-fused-greedy "warm:1,32:128;warm:1,32:128:nomin;1:1,32:128;1:1,32:128:nomin" ""
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

say "plan $PLAN, bundle $(sed -n 's/^plugin sha: //p' /bundle/BUNDLE.txt 2>/dev/null)"
install_stack
case "$PLAN" in
    A) plan_a ;;
    B) plan_b ;;
    Q) plan_q ;;
    T) plan_t ;;
    *) say "unknown plan $PLAN"; exit 2 ;;
esac
say "plan $PLAN done"
