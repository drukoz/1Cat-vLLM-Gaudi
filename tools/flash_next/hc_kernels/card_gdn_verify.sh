#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# card_gdn_verify.sh: the one-card A/B of the GDN verify variants (VLLM_HPU_GDN_VERIFY_VARIANT 0 to 3 with
# VLLM_HPU_GDN_KERNEL_INTERLEAVE=1; docs/design/flash-next/hc-kernels.md, "GDN verify"), run from the tree inside a
# farm container (farm job hc-kernels <name> any:1 <hold> -- bash tools/flash_next/hc_kernels/card_gdn_verify.sh):
#   1. the verify op card tests with variant 3 (the request-major index space and the lean broadcasts)
#   2. bench_gdn_verify.py per variant 0, 1, 2, 3 (us per call at batch 1, 8, 32, T = 4, TP8 geometry), the bits
#      of variants 1 to 3 compared with variant 0's
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
step gdn_optest_variant3 "$IL" VLLM_HPU_GDN_VERIFY_VARIANT=3 -- \
    python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py
step gdn_bench_variant0 "$IL" VLLM_HPU_GDN_VERIFY_VARIANT=0 -- \
    python tools/flash_next/hc_kernels/bench_gdn_verify.py --label variant0 --out "$OUT"
for v in 1 2 3; do
    step "gdn_bench_variant$v" "$IL" "VLLM_HPU_GDN_VERIFY_VARIANT=$v" -- \
        python tools/flash_next/hc_kernels/bench_gdn_verify.py --label "variant$v" --out "$OUT" \
        --compare-to "$OUT/variant0"
done
grep -h "passed\|failed\|VERDICT\|DIFFERS" "$OUT"/gdn_*.log 2>/dev/null | tail -n 10
exit "$rc"
