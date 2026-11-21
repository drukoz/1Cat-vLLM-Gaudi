#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# native_tp4_job.sh RUNTIME_TARBALL: the 30-minute 4-card job of the next rental, stock HPU-graph replay against the
# fork's native joint replay on the same TP4 body (bench_native_tp4.py). STATUS: DRAFT, NEVER RUN.
# Inside a farm job's container on 4 cards of one module group (the repository tree is the working folder):
#   farm job native-replay native-tp4 any:4 30m --no-model -- bash tools/flash_next/replay/native_tp4_job.sh \
#       /results/<run>/native-runtime-1.24.1.tar.gz
# (copy the tarball from build_native_runtime.sh into ~/farm/native-replay/results/<run>/ on the node first).
# Order: stock with and without collectives (the image's own runtime), then the patched runtime installed in this
# container only, native with and without collectives. Results: $FARM_OUT/<mode>[_nocoll]_rank<r>.json, logs.
set -u
TARBALL="${1:?usage: native_tp4_job.sh RUNTIME_TARBALL}"
OUT="${FARM_OUT:-/tmp/native-tp4}"
mkdir -p "$OUT"
B=tools/flash_next/replay/bench_native_tp4.py
run() {  # run LABEL ENV... -- ARGS...
    local label="$1"
    shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    echo "== $label $(date -u +%T)"
    env "${envs[@]}" timeout 600 torchrun --nproc-per-node 4 "$B" --out "$OUT" "$@" >"$OUT/$label.log" 2>&1
    echo "== $label rc $? $(date -u +%T)"
    grep -h '"records"' "$OUT/$label.log" | head -n 1 | cut -c1-300
    sleep 15  # the cards are released before the next process takes them
}
run stock PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=1 -- --mode stock
run stock_nocoll PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=1 -- --mode stock --no-collectives
NR=/tmp/native-runtime
mkdir -p "$NR"
tar -xzf "$TARBALL" -C "$NR" || exit 5
pip install --no-deps --force-reinstall "$NR"/wheels/*.whl >"$OUT/pip.log" 2>&1 || exit 6
EXT=$(find "$NR/ext" -name '*.so' | head -n 1)
run native PT_HPU_LAZY_MODE=0 LD_LIBRARY_PATH="$NR/lib:${LD_LIBRARY_PATH:-}" -- --mode native --bridge "$EXT"
run native_nocoll PT_HPU_LAZY_MODE=0 LD_LIBRARY_PATH="$NR/lib:${LD_LIBRARY_PATH:-}" -- --mode native --bridge "$EXT" \
    --no-collectives
