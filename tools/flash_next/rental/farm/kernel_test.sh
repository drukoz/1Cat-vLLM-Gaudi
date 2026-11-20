#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# kernel_test.sh: a kernel card test inside a farm container, run the way the rental's gates run them
# (gates.py: fused_gdn, hc_fused, moe_glue): pytest in lazy mode from the track's tree (/bundle/plugin)
# with the installed native libraries of the track's image given by path.
#
#   bash /farm/kernel_test.sh TEST_FILE[::NAME]... [pytest options]
#
# Sets, from the installed flashinfer_gaudi lib folder, whichever exist: HC_FUSED_GAUDI_KERNEL_LIBRARY,
# HC_FUSED_GAUDI_NATIVE_LIBRARY, GDN_FUSED_GAUDI_KERNEL_LIBRARY, GDN_FUSED_GAUDI_NATIVE_LIBRARY; then
# PYTHONPATH=<the tree>, PT_HPU_LAZY_MODE=1 (unless the job set it), and drops VLLM_HPU_HC_FUSED,
# VLLM_HPU_MOE_GLUE and PT_HPU_RECIPE_CACHE_CONFIG (the tests load the kernels themselves; a recipe cache
# blocks the kernel registration). Writes $FARM_OUT/kernel-env.txt and $FARM_OUT/pytest.log. The exit
# code is pytest's.

set -u
PY="${FARM_PYTHON:-python}"
OUT="${FARM_OUT:-/tmp/kernel-test}"
mkdir -p "$OUT"
# The installed package, not the tree's copy (its lib/ is empty): look it up from /tmp, as the gates do.
lib="$(cd /tmp && env -u PYTHONPATH "$PY" -c "import flashinfer_gaudi, pathlib; print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')" 2>/dev/null | tail -n 1)"
if [ -n "$lib" ] && [ -d "$lib" ]; then
    [ -f "$lib/libhc_fused_gaudi2.so" ] && export HC_FUSED_GAUDI_KERNEL_LIBRARY="$lib/libhc_fused_gaudi2.so"
    for f in "$lib"/hc_fused_ops*.so; do [ -f "$f" ] && export HC_FUSED_GAUDI_NATIVE_LIBRARY="$f"; done
    [ -f "$lib/libgdn_fused_gaudi2.so" ] && export GDN_FUSED_GAUDI_KERNEL_LIBRARY="$lib/libgdn_fused_gaudi2.so"
    for f in "$lib"/gdn_fused_ops*.so; do [ -f "$f" ] && export GDN_FUSED_GAUDI_NATIVE_LIBRARY="$f"; done
else
    echo "kernel_test: no installed flashinfer_gaudi lib folder: the native library variables are not set"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"
unset VLLM_HPU_HC_FUSED VLLM_HPU_MOE_GLUE PT_HPU_RECIPE_CACHE_CONFIG
env | grep -E '_GAUDI_|^PT_HPU|^PYTHONPATH=|^HABANA_VISIBLE' | sort >"$OUT/kernel-env.txt"
"$PY" -m pytest -q -rs -p no:cacheprovider "$@" 2>&1 | tee "$OUT/pytest.log"
exit "${PIPESTATUS[0]}"
