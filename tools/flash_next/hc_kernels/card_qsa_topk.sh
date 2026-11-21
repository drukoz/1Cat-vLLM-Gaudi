#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# card_qsa_topk.sh: the one-card job of the one-node QSA top-k kernel (qsa_index_topk, VLLM_HPU_QSA_TOPK_FORM=4;
# docs/design/flash-next/hc-kernels.md, "QSA indexer top-k"), run from the tree inside a farm container
# (farm job hc-kernels <name> any:1 <hold> -- bash tools/flash_next/hc_kernels/card_qsa_topk.sh), most important
# first:
#   1. the QSA op card tests of the top-k cases (ids bit-exact against the CPU reference; twice the same bits); if they
#      fail, the job stops here
#   2. bench_qsa_topk.py at 4, 16, 32, 128 rows (N = 32,768, k = 512): the vendor sort against the kernel and the
#      kernel + order by value (form 4 in the served path); the selected sets compared on the card; the vendor sort's
#      tie probe
#   3. profiler traces at 16 rows (one rank's share at 32-stream MTP k = 3) of the three forms
# Logs and results go to $FARM_OUT. The exit code is the first failing step's.
set -u
OUT="${FARM_OUT:-/tmp/hc-kernels-qsa-topk}"
mkdir -p "$OUT"
WITH=tools/flash_next/kernel_probes/with_kernels.sh
# with_kernels.sh exports the HC and GDN libraries of the image; the QSA ones by the same rule (the tree on PYTHONPATH
# has no built lib/ folder, so flashinfer_gaudi._native needs their paths)
lib="$(cd /tmp && env -u PYTHONPATH python -c "import flashinfer_gaudi, pathlib; print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')" 2>/dev/null | tail -n 1)"
if [ -n "$lib" ] && [ -f "$lib/libqsa_fused_gaudi2.so" ]; then
    export QSA_FUSED_GAUDI_KERNEL_LIBRARY="$lib/libqsa_fused_gaudi2.so"
    for f in "$lib"/qsa_fused_ops*.so; do [ -f "$f" ] && export QSA_FUSED_GAUDI_NATIVE_LIBRARY="$f"; done
else
    echo "card_qsa_topk: no libqsa_fused_gaudi2.so in the image's flashinfer_gaudi/lib ($lib)"
fi
rc=0
step() {  # step NAME ENV... -- COMMAND...
    local name="$1"
    shift
    local vars=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do vars+=("$1"); shift; done
    shift
    echo "=== hc-kernels step $name: ${vars[*]} $*"
    env ${vars[@]+"${vars[@]}"} bash "$WITH" "$@" 2>&1 | tee "$OUT/$name.log"
    local r="${PIPESTATUS[0]}"
    echo "=== hc-kernels step $name: exit $r"
    if [ "$rc" -eq 0 ] && [ "$r" -ne 0 ]; then rc="$r"; fi
    sleep 3
}
step qsa_optest_topk -- python -m pytest -q -rs -p no:cacheprovider tests/unit_tests/qsa_fused/test_hpu_op.py \
    -k "topk or every_op or twice"
if [ "$rc" -ne 0 ]; then
    echo "=== hc-kernels: the top-k op card test failed; no bench"
    exit "$rc"
fi
step qsa_bench_topk -- python tools/flash_next/hc_kernels/bench_qsa_topk.py --out "$OUT/bench" --ties
step qsa_trace_topk HABANA_PROFILE=profile_api_light -- python tools/flash_next/hc_kernels/bench_qsa_topk.py \
    --out "$OUT/trace" --rows 16 --forms sorted,kernel,kernel_sorted --iters 3 --trace
grep -h "passed\|failed\|VERDICT\|same_set\|the set of sorted" "$OUT"/qsa_*.log 2>/dev/null | tail -n 14
exit "$rc"
