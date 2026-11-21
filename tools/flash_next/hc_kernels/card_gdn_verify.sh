#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# card_gdn_verify.sh: the one-card A/B of the GDN verify variants (VLLM_HPU_GDN_VERIFY_VARIANT 0 to 7 with
# VLLM_HPU_GDN_KERNEL_INTERLEAVE=1; docs/design/flash-next/hc-kernels.md, "GDN verify"), run from the tree inside a
# farm container (farm job hc-kernels <name> any:1 <hold> -- bash tools/flash_next/hc_kernels/card_gdn_verify.sh):
# most important first, so that a hold that runs out loses the least:
#   1. the verify op card tests with variant 6 (the fold build)
#   2. bench_gdn_verify.py (us per call at batch 1, 8, 32, T = 4, TP8 geometry, chained calls; the bits of every
#      variant compared with variant 0's) for variants 0 and 6, then the index-space layouts 1, 4 and 7
#   3. profiler traces of variants 0, 1, 4 at batch 1 and 32 (the TPC engines a call runs on)
#   4. the lean variants 2, 3, 5 (bench), and the op card tests with variant 3
# One process per variant: the glue reads the variables when the graph compiler instantiates the op. Logs and
# results go to $FARM_OUT. Every step runs; the exit code is the first failing step's.
set -u
OUT="${FARM_OUT:-/tmp/hc-kernels-gdn}"
mkdir -p "$OUT"
WITH=tools/flash_next/kernel_probes/with_kernels.sh
rc=0
step() {  # step NAME ENV... -- COMMAND...: run with the variables, log to $OUT/NAME.log, keep the first failure
    local name="$1"
    shift
    local vars=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do vars+=("$1"); shift; done
    shift
    echo "=== hc-kernels step $name: ${vars[*]} $*"
    env "${vars[@]}" bash "$WITH" "$@" 2>&1 | tee "$OUT/$name.log"
    local r="${PIPESTATUS[0]}"
    echo "=== hc-kernels step $name: exit $r"
    if [ "$rc" -eq 0 ] && [ "$r" -ne 0 ]; then rc="$r"; fi
    sleep 3
}
IL=VLLM_HPU_GDN_KERNEL_INTERLEAVE=1
optest() {
    step "gdn_optest_variant$1" "$IL" "VLLM_HPU_GDN_VERIFY_VARIANT=$1" -- \
        python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py
}
bench() {
    local compare=()
    if [ "$1" != 0 ]; then compare=(--compare-to "$OUT/variant0"); fi
    step "gdn_bench_variant$1" "$IL" "VLLM_HPU_GDN_VERIFY_VARIANT=$1" -- \
        python tools/flash_next/hc_kernels/bench_gdn_verify.py --label "variant$1" --out "$OUT" ${compare[@]+"${compare[@]}"}
}
optest 6
for v in 0 6 1 4 7; do bench "$v"; done
# device traces (engines per call) of the layouts: today's (0), request-major (1), preferred key-head split (4)
for v in 0 1 4; do
    step "gdn_trace_variant$v" "$IL" "VLLM_HPU_GDN_VERIFY_VARIANT=$v" HABANA_PROFILE=profile_api_light -- \
        python tools/flash_next/hc_kernels/bench_gdn_verify.py --label "trace_variant$v" --out "$OUT" --trace \
        --batches 1,32 --iters 3
done
for v in 2 3 5; do bench "$v"; done
optest 3
grep -h "passed\|failed\|VERDICT\|DIFFERS" "$OUT"/gdn_*.log 2>/dev/null | tail -n 14
exit "$rc"
