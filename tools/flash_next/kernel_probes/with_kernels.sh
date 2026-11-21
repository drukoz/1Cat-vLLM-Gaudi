#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# with_kernels.sh COMMAND...: run COMMAND inside a farm container (stage 2) with the environment the kernel card
# tests get from the farm's kernel_test.sh: the installed native libraries of the image by path
# (HC_FUSED_GAUDI_*, GDN_FUSED_GAUDI_*), the tree on PYTHONPATH, lazy mode, and no recipe cache (it blocks the
# kernel registration). Run from the tree (/bundle/plugin). For the probes of this folder, e.g.
#   farm job kernel-fixes moe-route-probe any:1 15m -- bash tools/flash_next/kernel_probes/with_kernels.sh \
#       python tools/flash_next/kernel_probes/probe_moe_route.py
set -u
PY="${FARM_PYTHON:-python}"
lib="$(cd /tmp && env -u PYTHONPATH "$PY" -c "import flashinfer_gaudi, pathlib; print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')" 2>/dev/null | tail -n 1)"
if [ -n "$lib" ] && [ -d "$lib" ]; then
    [ -f "$lib/libhc_fused_gaudi2.so" ] && export HC_FUSED_GAUDI_KERNEL_LIBRARY="$lib/libhc_fused_gaudi2.so"
    for f in "$lib"/hc_fused_ops*.so; do [ -f "$f" ] && export HC_FUSED_GAUDI_NATIVE_LIBRARY="$f"; done
    [ -f "$lib/libgdn_fused_gaudi2.so" ] && export GDN_FUSED_GAUDI_KERNEL_LIBRARY="$lib/libgdn_fused_gaudi2.so"
    for f in "$lib"/gdn_fused_ops*.so; do [ -f "$f" ] && export GDN_FUSED_GAUDI_NATIVE_LIBRARY="$f"; done
else
    echo "with_kernels: no installed flashinfer_gaudi lib folder: the native library variables are not set"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"
export PT_HPU_LAZY_ACC_PAR_MODE="${PT_HPU_LAZY_ACC_PAR_MODE:-0}"
unset VLLM_HPU_HC_FUSED VLLM_HPU_MOE_GLUE PT_HPU_RECIPE_CACHE_CONFIG
mkdir -p "${FARM_OUT:-/tmp/kernel-probe}"
env | grep -E '_GAUDI_|^PT_HPU|^PYTHONPATH=|^HABANA_VISIBLE|^VLLM_HPU' | sort > "${FARM_OUT:-/tmp/kernel-probe}/probe-env.txt"
exec "$@"
