#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# card_1card.sh: the one-card job of the hc-kernels track (docs/design/flash-next/hc-kernels.md), run from the
# tree inside a farm container (farm job hc-kernels <name> any:1 <hold> -- bash tools/flash_next/hc_kernels/card_1card.sh):
#   1. tests/unit_tests/hc_fused/test_hpu_op_il.py  the interleaved ops against the oracle and the shipped ops
#   2. tests/unit_tests/hc_fused/test_hpu_op.py     the shipped ops of the rebuilt library (skipped with --no-shipped)
#   3. tools/flash_next/hc_kernels/bench_hc.py       time per call and per hyper-connection at 4, 32, 96, 128 rows,
#                                                    then profiler traces at 128 rows (HABANA_PROFILE=profile_api_light)
# Each step gets the installed kernel libraries by path (tools/flash_next/kernel_probes/with_kernels.sh). Logs and
# results go to $FARM_OUT. Every step runs; the exit code is the first failing step's.
set -u
OUT="${FARM_OUT:-/tmp/hc-kernels}"
mkdir -p "$OUT"
WITH=tools/flash_next/kernel_probes/with_kernels.sh
SHIPPED=1
BENCH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-shipped) SHIPPED=0 ;;
        *) BENCH_ARGS+=("$arg") ;;
    esac
done
rc=0
step() {  # step NAME COMMAND...: run, log to $OUT/NAME.log, keep the first failure
    local name="$1"
    shift
    echo "=== hc-kernels step $name: $*"
    bash "$WITH" "$@" 2>&1 | tee "$OUT/$name.log"
    local r="${PIPESTATUS[0]}"
    echo "=== hc-kernels step $name: exit $r"
    if [ "$rc" -eq 0 ] && [ "$r" -ne 0 ]; then rc="$r"; fi
    sleep 3  # the step's process has released the card before the next one opens it
}
step pytest_il python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op_il.py
if [ "$SHIPPED" -eq 1 ]; then
    step pytest_shipped python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op.py
fi
step bench python tools/flash_next/hc_kernels/bench_hc.py --out "$OUT" "${BENCH_ARGS[@]}"
# device traces at 128 rows (the profiler needs its variable before torch is imported): engines per call
HABANA_PROFILE=profile_api_light step bench_trace python tools/flash_next/hc_kernels/bench_hc.py --rows 128 --trace \
    --iters 3 --out "$OUT/trace"
grep -h "passed\|failed\|VERDICT" "$OUT"/pytest_il.log "$OUT"/pytest_shipped.log "$OUT"/bench.log 2>/dev/null | tail -n 6
exit "$rc"
