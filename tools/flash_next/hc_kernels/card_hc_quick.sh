#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# card_hc_quick.sh: the hc-kernels track's short one-card job for the interleaved hyper-connection kernels
# (docs/design/flash-next/hc-kernels.md), sized for a 12-minute hold, steps in order of importance (each writes its
# results as it goes, so a hold that ends early loses only the last ones):
#   1. tests/unit_tests/hc_fused/test_hpu_op_il.py: the interleaved ops against the oracle and the shipped ops
#   2. the tiny-model token gate at TP1 with VLLM_HPU_HC_FUSED_IL=1 and copy mode 2 (farm_tiny.py)
#   3. bench_hc.py at 4, 32, 96, 128 rows (parity, us per op call and per hyper-connection, chained calls)
#   4. profiler traces of the per-op graphs at 128 rows (HABANA_PROFILE=profile_api_light)
# Run from the tree in a farm container. Logs and results in $FARM_OUT; the exit code is the first failing step's.
set -u
OUT="${FARM_OUT:-/tmp/hc-kernels}"
mkdir -p "$OUT"
WITH=tools/flash_next/kernel_probes/with_kernels.sh
rc=0
note() {
    local name="$1" r="$2"
    echo "=== hc-kernels step $name: exit $r"
    if [ "$rc" -eq 0 ] && [ "$r" -ne 0 ]; then rc="$r"; fi
}
echo "=== hc-kernels step pytest_il"
bash "$WITH" python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op_il.py 2>&1 \
    | tee "$OUT/pytest_il.log"
note pytest_il "${PIPESTATUS[0]}"
sleep 3
echo "=== hc-kernels step tiny"
if [ -f /farm/farm_tiny.py ]; then
    python /farm/farm_tiny.py --tp 1 --gdn hc --env VLLM_HPU_HC_FUSED_IL=1 --env VLLM_HPU_HC_FUSED_COPY_MODE=2 \
        --out "$OUT/tiny" 2>&1 | tee "$OUT/tiny.log"
    note tiny "${PIPESTATUS[0]}"
else
    echo "card_hc_quick: /farm/farm_tiny.py is missing: no tiny gate" | tee "$OUT/tiny.log"
fi
sleep 8
echo "=== hc-kernels step bench"
bash "$WITH" python tools/flash_next/hc_kernels/bench_hc.py --out "$OUT" --k 8,24 --chain-k 4,12 --iters 6 2>&1 \
    | tee "$OUT/bench.log"
note bench "${PIPESTATUS[0]}"
sleep 3
echo "=== hc-kernels step bench_trace"
HABANA_PROFILE=profile_api_light bash "$WITH" python tools/flash_next/hc_kernels/bench_hc.py --out "$OUT/trace" \
    --rows 128 --parts ops --k 8,24 --iters 2 --trace 2>&1 | tee "$OUT/bench_trace.log"
note bench_trace "${PIPESTATUS[0]}"
grep -h "passed\|failed\|tiny smoke\|VERDICT" "$OUT"/pytest_il.log "$OUT"/tiny.log "$OUT"/bench.log 2>/dev/null | tail -n 6
exit "$rc"
