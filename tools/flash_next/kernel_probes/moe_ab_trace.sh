#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# moe_ab_trace.sh ARM...: kernel-fixes' TP8 A/B of the MoE forms with one device-trace capture per arm, in a tree
# that holds the integration kit (tools/flash_next/integration/) and step-anatomy's probe
# (tools/flash_next/anatomy/anatomy_probe.py): the build-4b test branch = integration build 4 (30f4e37e) +
# kernel-fixes 11bc2f6d (VLLM_HPU_MOE_DENSE: 17-token minimum, folded routing weight) + a5786c55 (a 0 in
# VLLM_MOE_CHUNK). One all-card farm job:
#
#   farm job kernel-fixes moe-ab-trace all 45m -- bash tools/flash_next/kernel_probes/moe_ab_trace.sh base dense
#
# ARMs, one MTP k = 3 server each (the kit's headline_job.sh arm mtp3: levels 32,1, two rounds each, greedy), both
# with VLLM_MOE_CHUNK=0,64,8192 VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000:
#   base   the vendor op (chunk_size 64 from 17 to 512 tokens, the op's default up to 16, 8192 above)
#   dense  + VLLM_HPU_MOE_DENSE=1 (17 to 512 tokens as two batched FP8 GEMMs; the vendor op outside that range)
# The only difference between the arms is 17 to 512 tokens: chunk 64 against dense.
# The capture (step-anatomy's recipe, jobs 00113 / 00143): the server gets the torch profiler (--profiler-config,
# HABANA_PROFILE=profile_api_light; no measurable cost while not capturing) and the step timer
# (VLLM_HPU_STEP_TIMING=1); after the kit's rounds, anatomy_probe.py runs one 32-stream round on the same live
# server and captures 0.6 s of it (the stop takes minutes; --measured-timeout 600). The kit is copied to /tmp and
# changed there only: serve_probe.sh adds the profiler setting, decode_scaling.py runs the kit's probe and then the
# capture round.
# Results: $FARM_OUT/<arm>/ (headline_job.sh's layout, headline.md), $FARM_OUT/<arm>/mtp3/trace/*.pt.trace.json.gz,
# $FARM_OUT/<arm>/mtp3/timing/, $FARM_OUT/<arm>/mtp3/trace_probe.json. Read the traces on the workstation with
# step-anatomy's trace_components.py (one rank).
set -u
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$TREE/tools/flash_next/integration"
PROBE="$TREE/tools/flash_next/anatomy/anatomy_probe.py"
OUT="${FARM_OUT:-/results/moe-ab-trace}"
KIT=/tmp/moe_ab_trace_kit
[ -f "$SRC/headline_job.sh" ] && [ -f "$SRC/serve_probe.sh" ] && [ -f "$SRC/decode_scaling.py" ] ||
    { echo "moe_ab_trace: no integration kit in $SRC" >&2; exit 2; }
[ -f "$PROBE" ] || { echo "moe_ab_trace: no anatomy_probe.py in $TREE" >&2; exit 2; }
[ $# -gt 0 ] || set -- base dense

rm -rf "$KIT" && cp -r "$SRC" "$KIT"
sed -i 's|^SERVE+=("\$@")$|SERVE+=(--profiler-config "$MOE_AB_PROFILER_CONFIG" "$@")|' "$KIT/serve_probe.sh"
grep -q 'MOE_AB_PROFILER_CONFIG' "$KIT/serve_probe.sh" || { echo "moe_ab_trace: serve_probe.sh not patched" >&2; exit 2; }
mv "$KIT/decode_scaling.py" "$KIT/decode_scaling_kit.py"
cat >"$KIT/decode_scaling.py" <<'PY'
#!/usr/bin/env python3
"""The kit's decode probe, then one capture round of step-anatomy's probe against the same live server."""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
rc = subprocess.call([sys.executable, os.path.join(here, "decode_scaling_kit.py"), *args])


def value(flag):
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


port, out, probe = value("--port"), value("--out"), os.environ.get("MOE_AB_ANATOMY_PROBE")
if probe and port and out:
    target = os.path.join(os.path.dirname(out), "trace_probe.json")
    print(f"moe_ab_trace: capture round on port {port} -> {target}", flush=True)
    rc2 = subprocess.call([sys.executable, probe, "--port", port, "--model", "fn", "--phases", "", "--trace", "32",
                           "--trace-tokens", "64", "--trace-after", "12", "--trace-seconds", "0.6",
                           "--measured-timeout", "600", "--out", target])
    print(f"moe_ab_trace: capture round ended with exit code {rc2}", flush=True)
sys.exit(rc)
PY

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
    echo "moe_ab_trace: cards cleared ($(workers_left) vLLM processes left) $(date -u +%T)"
}

export VLLM_HPU_PREFIX_ALIGN_SPLIT=1 VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1
export VLLM_MOE_CHUNK=0,64,8192 VLLM_MOE_TOKEN_BOUNDARY=16,512,1000000
export HABANA_PROFILE=profile_api_light VLLM_HPU_STEP_TIMING=1 MOE_AB_ANATOMY_PROBE="$PROBE"
export HEADLINE_LEVELS=32,1 HEADLINE_ROUNDS=2 HEADLINE_MTP_FALLBACK=0 HEADLINE_PROBE_TIMEOUT=1500
unset VLLM_HPU_MOE_DENSE
failed=0
for arm in "$@"; do
    extra=()
    case "$arm" in
        base) ;;
        dense) extra=(VLLM_HPU_MOE_DENSE=1) ;;
        *) echo "moe_ab_trace: unknown arm $arm" >&2; failed=$((failed + 1)); continue ;;
    esac
    mkdir -p "$OUT/$arm/mtp3"
    profiler='{"profiler": "torch", "torch_profiler_dir": "'"$OUT/$arm/mtp3/trace"'", "torch_profiler_with_stack": false,'
    profiler+=' "torch_profiler_use_gzip": true, "torch_profiler_record_shapes": false}'
    echo "=== moe_ab_trace: arm $arm (${extra[*]:-vendor op}) $(date -u +%T)"
    env ${extra[@]+"${extra[@]}"} MOE_AB_PROFILER_CONFIG="$profiler" VLLM_HPU_STEP_TIMING_DIR="$OUT/$arm/mtp3/timing" \
        FARM_OUT="$OUT/$arm" bash "$KIT/headline_job.sh" mtp3
    rc=$?
    echo "moe_ab_trace: arm $arm ended with exit code $rc $(date -u +%T)"
    [ "$rc" -eq 0 ] || failed=$((failed + 1))
    ls "$OUT/$arm/mtp3/trace" 2>/dev/null | head -n 10 | sed "s/^/moe_ab_trace: $arm trace file /"
    clear_cards
done
for arm in "$@"; do
    [ -f "$OUT/$arm/headline.md" ] && { echo "--- moe_ab_trace: $arm/headline.md"; cat "$OUT/$arm/headline.md"; }
done
exit "$failed"
