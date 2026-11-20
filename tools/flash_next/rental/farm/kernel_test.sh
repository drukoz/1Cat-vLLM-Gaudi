#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# kernel_test.sh: a kernel card test inside a farm container, run the way the rental's gates run them
# (gates.py: fused_gdn, hc_fused, moe_glue): pytest in lazy mode from the track's tree (/bundle/plugin)
# with the installed native libraries of the track's image given by path.
#
#   bash /farm/kernel_test.sh TEST_FILE[::NAME]... [pytest options]
#
# Library paths. For every kernel library of the installed flashinfer_gaudi lib folder, by its name:
#   lib<name>_gaudi2.so  ->  <NAME>_GAUDI_KERNEL_LIBRARY      (libqsa_fused_gaudi2.so -> QSA_FUSED_GAUDI_KERNEL_LIBRARY)
#   <name>_ops*.so       ->  <NAME>_GAUDI_NATIVE_LIBRARY      (hc_fused_ops.cpython-310...so -> HC_FUSED_GAUDI_NATIVE_LIBRARY)
# so a kernel a track adds (hc_fused, gdn_fused, qsa_fused, ...) is found without a change here. A variable
# the job already set (farm job --env QSA_FUSED_GAUDI_KERNEL_LIBRARY=/path) is kept: the job's value wins.
#
# Then PYTHONPATH=<the tree>, PT_HPU_LAZY_MODE=1 (unless the job set it), and VLLM_HPU_HC_FUSED,
# VLLM_HPU_MOE_GLUE and PT_HPU_RECIPE_CACHE_CONFIG are dropped (the tests load the kernels themselves; a
# recipe cache blocks the kernel registration). Writes $FARM_OUT/kernel-env.txt and $FARM_OUT/pytest.log.
#
# Exit code: pytest's; and 6 when pytest ended with 0 but every test was skipped (a card test skips itself
# when its library path is missing: nothing ran on the card, so the job must not end "ok"). A job that
# expects that sets FARM_ALLOW_ALL_SKIPPED=1.

set -u
PY="${FARM_PYTHON:-python}"
OUT="${FARM_OUT:-/tmp/kernel-test}"
mkdir -p "$OUT"

# set_lib VARIABLE FILE: export it unless the job set it already.
KEPT=""
set_lib() {
    if [ -n "${!1:-}" ]; then
        case " $KEPT " in *" $1 "*) ;; *) KEPT="$KEPT${KEPT:+ }$1" ;; esac
    else
        export "$1=$2"
    fi
}

# The installed package, not the tree's copy (its lib/ is empty): look it up from /tmp, as the gates do.
lib="$(cd /tmp && env -u PYTHONPATH "$PY" -c "import flashinfer_gaudi, pathlib; print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')" 2>/dev/null | tail -n 1)"
if [ -n "$lib" ] && [ -d "$lib" ]; then
    for f in "$lib"/lib*_gaudi2.so; do
        [ -f "$f" ] || continue
        name="$(basename "$f")"
        name="${name#lib}"
        name="${name%_gaudi2.so}"
        case "$name" in '' | *[!A-Za-z0-9_]*) continue ;; esac
        set_lib "$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')_GAUDI_KERNEL_LIBRARY" "$f"
    done
    for f in "$lib"/*_ops*.so; do
        [ -f "$f" ] || continue
        name="$(basename "$f")"
        name="${name%%_ops*}"
        case "$name" in '' | *[!A-Za-z0-9_]*) continue ;; esac
        set_lib "$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')_GAUDI_NATIVE_LIBRARY" "$f"
    done
else
    echo "kernel_test: no installed flashinfer_gaudi lib folder: the native library variables are not set"
fi
if [ -n "$KEPT" ]; then
    # shellcheck disable=SC2086  # a list of names
    echo "kernel_test: kept from the job: $(printf '%s\n' $KEPT | sort | tr '\n' ' ' | sed 's/ $//')"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-1}"
unset VLLM_HPU_HC_FUSED VLLM_HPU_MOE_GLUE PT_HPU_RECIPE_CACHE_CONFIG
env | grep -E '_GAUDI_|^PT_HPU|^PYTHONPATH=|^HABANA_VISIBLE' | sort >"$OUT/kernel-env.txt"
"$PY" -m pytest -q -rs -p no:cacheprovider "$@" 2>&1 | tee "$OUT/pytest.log"
rc="${PIPESTATUS[0]}"

# pytest's last summary line ("3 skipped in 0.10s", "=== 2 passed, 1 skipped in 3.2s ==="): skips and
# nothing that ran means no test ran.
summary="$(grep -E '[0-9]+ (passed|failed|skipped|errors?|xfailed|xpassed|deselected).* in [0-9.]+s' "$OUT/pytest.log" | tail -n 1 | sed 's/^=* *//; s/ *=*$//')"
if [ "$rc" -eq 0 ] && [ "${FARM_ALLOW_ALL_SKIPPED:-0}" != 1 ]; then
    case "$summary" in
        *passed* | *failed* | *error* | *xfailed* | *xpassed*) ;;
        *skipped*)
            echo "kernel_test: ALL-SKIPPED: every test was skipped ($summary): nothing ran on the card. The reasons are in the SKIPPED lines above (a missing library path: see kernel-env.txt; give it with farm job --env NAME=/path). Exit code 6; FARM_ALLOW_ALL_SKIPPED=1 accepts it." | tee -a "$OUT/pytest.log"
            rc=6
            ;;
    esac
fi
exit "$rc"
