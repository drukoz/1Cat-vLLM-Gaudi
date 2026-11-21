#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# moe_rt_ab.sh [ARM...]: kernel-fixes' TP8 A/B of the MoE dense range on ONE server per arm, switched at run time
# (VLLM_HPU_RUNTIME_SWITCH_FILE; the runner drops the captured HPU graphs when the range changes), in a tree that
# holds the integration kit (tools/flash_next/integration/) and step-anatomy's probe. One all-card farm job:
#
#   farm job kernel-fixes moe-rt-ab all 40m -- bash tools/flash_next/kernel_probes/moe_rt_ab.sh mtp3 nospec
#
# ARMs (default mtp3 nospec), one server each, both with VLLM_HPU_MOE_DENSE=1 (the stacked experts kept) and
# VLLM_MOE_CHUNK=0,64,8192 VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000 (the vendor op: its default up to 16 tokens,
# chunk_size 64 from 17 to 512, 8192 above):
#   mtp3    sync MTP k = 3 (the kit's headline_job.sh arm mtp3)
#   nospec  no speculation (the kit's arm nospec), then the prompt-logprob comparison (MOE_RT_ACCURACY=1)
# On each server moe_rt_driver.py runs the settings off (the vendor op at every size), 1 and 17 (the dense form from
# that many tokens to 512) at 1, 8 and 32 streams, two prompt sets, two rounds each, and one device-trace capture at
# 1 stream for the settings off and 1 (step-anatomy's anatomy_probe.py; the torch profiler of --profiler-config,
# HABANA_PROFILE=profile_api_light). The driver stops the job (exit 9) when the server did not keep the stacked experts
# or a switch reached no worker's graphs.
# The kit is copied to /tmp and changed there only: serve_probe.sh adds the profiler setting, decode_scaling.py is the
# driver.
# MOE_RT_ROUTE_STATS=1: the servers also log the real router's per-expert pair counts of every MoE call after every
# step (VLLM_HPU_MOE_ROUTE_STATS=$FARM_OUT/<arm>/<arm>/route_stats/rank<r>.jsonl; read with the workstation's
# reports/kernel-fixes/tools/route_stats.py). A stats run: one device-to-host read per step, its times are not the
# served model's.
# Results: $FARM_OUT/<arm>/<arm>/{moe_rt.json,scaling.json,probe.log,server0.log,trace/}; the probe's lines start
# with MEAS, TEXT, ACC and "moe_rt:".
set -u
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$TREE/tools/flash_next/integration"
PROBE="$TREE/tools/flash_next/anatomy/anatomy_probe.py"
DRIVER="$TREE/tools/flash_next/kernel_probes/moe_rt_driver.py"
OUT="${FARM_OUT:-/results/moe-rt-ab}"
KIT=/tmp/moe_rt_kit
SWITCH=/tmp/moe_rt_switches
[ -f "$SRC/headline_job.sh" ] && [ -f "$SRC/serve_probe.sh" ] && [ -f "$SRC/decode_scaling.py" ] ||
    { echo "moe_rt_ab: no integration kit in $SRC" >&2; exit 2; }
[ -f "$PROBE" ] && [ -f "$DRIVER" ] || { echo "moe_rt_ab: no anatomy_probe.py or moe_rt_driver.py in $TREE" >&2; exit 2; }
[ $# -gt 0 ] || set -- mtp3 nospec

rm -rf "$KIT" && cp -r "$SRC" "$KIT"
sed -i 's|^SERVE+=("\$@")$|SERVE+=(--profiler-config "$MOE_AB_PROFILER_CONFIG" "$@")|' "$KIT/serve_probe.sh"
grep -q 'MOE_AB_PROFILER_CONFIG' "$KIT/serve_probe.sh" || { echo "moe_rt_ab: serve_probe.sh not patched" >&2; exit 2; }
rm -f "$KIT/decode_scaling.py"
cp "$DRIVER" "$KIT/decode_scaling.py"

workers_left() {
    local n=0 pid cmd
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)" || continue
        case "$cmd" in VLLM::* | *"vllm serve"*) n=$((n + 1)) ;; esac
    done
    echo "$n"
}

clear_cards() {  # kills only processes of this container (its own PID namespace)
    local i used
    if [ "$(workers_left)" -gt 0 ]; then
        pkill -9 -f "VLLM::" 2>/dev/null
        pkill -9 -f "vllm serve" 2>/dev/null
    fi
    for ((i = 0; i < 60; i++)); do [ "$(workers_left)" -eq 0 ] && break; sleep 1; done
    sleep 8
    if command -v hl-smi >/dev/null 2>&1; then
        for ((i = 0; i < 120; i += 3)); do
            used="$(hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null |
                awk -F', *' '{ gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }')"
            [ "${used:-0}" -lt 4096 ] && break
            sleep 3
        done
    fi
    echo "moe_rt_ab: cards cleared ($(workers_left) vLLM processes left) $(date -u +%T)"
}

export VLLM_HPU_PREFIX_ALIGN_SPLIT=1 VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1
export VLLM_MOE_CHUNK=0,64,8192 VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000
export VLLM_HPU_MOE_DENSE=1 VLLM_HPU_RUNTIME_SWITCH_FILE="$SWITCH" MOE_RT_SWITCH_FILE="$SWITCH"
export HABANA_PROFILE=profile_api_light VLLM_HPU_STEP_TIMING=1 MOE_RT_ANATOMY_PROBE="$PROBE"
export HEADLINE_LEVELS=1,8,32 HEADLINE_ROUNDS=2 HEADLINE_MTP_FALLBACK=0 HEADLINE_PROBE_TIMEOUT=1500
export MOE_RT_SETTINGS="${MOE_RT_SETTINGS:-off,1,17}" MOE_RT_LEVELS="${MOE_RT_LEVELS:-1,8,32}"
export MOE_RT_TRACE_STREAMS="${MOE_RT_TRACE_STREAMS:-1}" MOE_RT_TRACE_SETTINGS="${MOE_RT_TRACE_SETTINGS:-off,1}"
failed=0
for arm in "$@"; do
    case "$arm" in
        mtp3) accuracy=0 ;;
        nospec) accuracy=1 ;;
        *) echo "moe_rt_ab: unknown arm $arm" >&2; failed=$((failed + 1)); continue ;;
    esac
    mkdir -p "$OUT/$arm/$arm"
    # The server starts on the vendor op everywhere; the driver switches from there.
    printf 'VLLM_HPU_MOE_DENSE_MIN_TOKENS=513\nVLLM_HPU_MOE_DENSE_MAX_TOKENS=512\n' >"$SWITCH.new" && mv "$SWITCH.new" "$SWITCH"
    profiler='{"profiler": "torch", "torch_profiler_dir": "'"$OUT/$arm/$arm/trace"'", "torch_profiler_with_stack": false,'
    profiler+=' "torch_profiler_use_gzip": true, "torch_profiler_record_shapes": false}'
    echo "=== moe_rt_ab: arm $arm (accuracy $accuracy) $(date -u +%T)"
    stats=()
    if [ "${MOE_RT_ROUTE_STATS:-0}" = 1 ]; then stats=(VLLM_HPU_MOE_ROUTE_STATS="$OUT/$arm/$arm/route_stats"); fi
    env ${stats[@]+"${stats[@]}"} MOE_AB_PROFILER_CONFIG="$profiler" MOE_RT_TRACE_DIR="$OUT/$arm/$arm/trace" \
        MOE_RT_ACCURACY="$accuracy" VLLM_HPU_STEP_TIMING_DIR="$OUT/$arm/$arm/timing" FARM_OUT="$OUT/$arm" \
        bash "$KIT/headline_job.sh" "$arm"
    rc=$?
    if [ ${#stats[@]} -gt 0 ]; then
        wc -l "$OUT/$arm/$arm/route_stats"/rank*.jsonl 2>/dev/null | sed "s/^/moe_rt_ab: $arm route stats /"
    fi
    echo "moe_rt_ab: arm $arm ended with exit code $rc $(date -u +%T)"
    [ "$rc" -eq 0 ] || failed=$((failed + 1))
    grep -hE "^(MEAS|TEXT|ACC|moe_rt:|=== moe_rt)" "$OUT/$arm/$arm/probe.log" 2>/dev/null | sed "s/^/moe_rt_ab: $arm: /" | cut -c1-330
    grep -h "runs the dense MoE form\|runs the vendor op\|HPU graphs dropped" "$OUT/$arm/$arm/server0.log" 2>/dev/null |
        grep "TP0" | sed 's/^.*\] //' | sort | uniq -c | head -n 40
    ls "$OUT/$arm/$arm/trace" 2>/dev/null | head -n 20 | sed "s/^/moe_rt_ab: $arm trace file /"
    if grep -q "moe_rt: FAIL" "$OUT/$arm/$arm/probe.log" 2>/dev/null; then
        echo "moe_rt_ab: FAIL in arm $arm (see above); the job stops here"
        clear_cards
        exit 9
    fi
    clear_cards
done
exit "$failed"
