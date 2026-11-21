#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# One-step crash probe for the GDN verify op on the card (RUK-3105 R1-e3).
#
# Runs six short card checks, each in its own process under `gdb -batch` with a time
# limit, and writes OUT_DIR/summary.txt with one status line per check and the
# crashing thread's backtrace. Run it inside the rental container; `--dry-run` and
# `--classify` also work on the Mac (bash 3.2) and touch no card, gdb, cmake or python.
#
# pytest runs with `-s`: its capture would swallow what the graph compiler writes to
# stderr, and that is the only place its own error message appears. When gdb is absent
# and cannot be installed, every check still runs, without gdb: the log then holds the
# same output and the Python stack of the fault handler (also for a check that hits the
# time limit, which is ended with SIGABRT for that), and the summary quotes that stack.
#
# usage: gdn_verify_crash_probe.sh [--dry-run] OUT_DIR
#        gdn_verify_crash_probe.sh --classify LOG_FILE EXIT_CODE
set -uo pipefail

usage() {
    printf 'usage: gdn_verify_crash_probe.sh [--dry-run] OUT_DIR | --classify LOG_FILE EXIT_CODE\n' >&2
}

classify() {
    # $1: log file, $2: exit code of the run -> prints one status word.
    local log="$1" rc="$2" sig last
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        printf 'timed_out\n'
        return 0
    fi
    sig=$(grep -m1 -E -o 'received signal SIG[A-Z]+' "$log" 2>/dev/null | sed 's/^received signal //' || true)
    if [ -n "$sig" ]; then
        printf 'crashed:%s\n' "$sig"
        return 0
    fi
    # without gdb the exit code is the signal: SIGILL, SIGABRT, SIGBUS, SIGFPE, SIGSEGV
    case "$rc" in
        132 | 134 | 135 | 136 | 139)
            printf 'crashed:exit%s\n' "$rc"
            return 0
            ;;
    esac
    # pytest's last summary ("1 failed, 2 passed in 3.21s"). With -s, text a library wrote without
    # a newline can sit on the same line in front of it, so the match is not tied to the line start.
    last=$(grep -E -o '[0-9]+ (passed|failed|skipped|error)[a-z, 0-9]* in [0-9.]+s' "$log" 2>/dev/null | tail -1 || true)
    case "$last" in
        *failed*|*error*|*skipped*) printf 'failed\n'; return 0 ;;
    esac
    case "$last" in
        *" passed"*) printf 'passed\n'; return 0 ;;
    esac
    printf 'failed\n'
}

PYTHON="${PYTHON:-python}"
PLUGIN_ROOT="${PLUGIN_ROOT:-}"
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-300}"

NODE_DECODE='tests/unit_tests/gdn_core_decode/test_hpu_op.py::test_hpu_op_matches_reference[tp8_mixed]'
NODE_T1='tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py::test_hpu_verify_op_matches_oracle[ver_tp4_t1]'
NODE_CHAIN='tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py::test_hpu_verify_op_matches_oracle[ver_tp8_chain]'

NAMES=(control_decode_lazy verify_t1_lazy verify_chain_lazy verify_chain_eager strip_verify_t1_lazy strip_verify_chain_lazy)
LAZYS=(1 1 1 0 1 1)
LIBS=(shipped shipped shipped shipped strip strip)
NODES=("$NODE_DECODE" "$NODE_T1" "$NODE_CHAIN" "$NODE_CHAIN" "$NODE_T1" "$NODE_CHAIN")

if [ "${1:-}" = "--classify" ]; then
    shift
    if [ $# -ne 2 ]; then usage; exit 2; fi
    classify "$1" "$2"
    exit 0
fi

DRY=0
case "${1:-}" in
    --dry-run) DRY=1; shift ;;
    "" | --*) usage; exit 2 ;;
esac
if [ $# -ne 1 ]; then usage; exit 2; fi
OUT_DIR="$1"

script_dir=$(cd "$(dirname "$0")" && pwd)
if [ -z "$PLUGIN_ROOT" ]; then
    PLUGIN_ROOT=$(cd "$script_dir/../.." && pwd)
fi
STRIP_LIB="$OUT_DIR/build_strip/libgdn_fused_gaudi2.so"

# the gdb command each check runs under, printed by --dry-run and executed by a real run
gdb_tail() {
    printf 'timeout -k 10 %s gdb -q -batch -nx -ex "set pagination off" -ex "set confirm off" -ex "set disable-randomization off" -ex "handle SIGPIPE nostop noprint pass" -ex "handle SIGUSR1 nostop noprint pass" -ex "handle SIGUSR2 nostop noprint pass" -ex run -ex "bt 40" -ex "info threads" -ex "thread apply all bt 25" -ex "info sharedlibrary" -ex kill' "$PROBE_TIMEOUT_S"
}

kernel_placeholder() {
    if [ -n "${GDN_FUSED_GAUDI_KERNEL_LIBRARY:-}" ]; then
        printf '%s' "$GDN_FUSED_GAUDI_KERNEL_LIBRARY"
    else
        printf '<installed>'
    fi
}

if [ "$DRY" = "1" ]; then
    py_display=$(command -v "$PYTHON" 2>/dev/null || printf '%s' "$PYTHON")
    i=0
    while [ $i -lt ${#NAMES[@]} ]; do
        lib=$(kernel_placeholder)
        if [ "${LIBS[$i]}" = "strip" ]; then lib="$STRIP_LIB"; fi
        printf 'run %s: PT_HPU_LAZY_MODE=%s GDN_FUSED_GAUDI_KERNEL_LIBRARY=%s PYTHONFAULTHANDLER=1 %s --args %s -X faulthandler -m pytest -q -p no:cacheprovider -rs -s %s\n' \
            "${NAMES[$i]}" "${LAZYS[$i]}" "$lib" "$(gdb_tail)" "$py_display" "${NODES[$i]}"
        i=$((i + 1))
    done
    exit 0
fi

# ---- real run (rental container) ----------------------------------------------------
if [ ! -d "$PLUGIN_ROOT/csrc/gdn_fused" ]; then
    printf 'error: PLUGIN_ROOT %s has no csrc/gdn_fused\n' "$PLUGIN_ROOT" >&2
    exit 2
fi
mkdir -p "$OUT_DIR" || exit 2
OUT_DIR=$(cd "$OUT_DIR" && pwd)
STRIP_LIB="$OUT_DIR/build_strip/libgdn_fused_gaudi2.so"  # absolute: the runs cd to $PLUGIN_ROOT
SUMMARY="$OUT_DIR/summary.txt"
: > "$SUMMARY"

# the two libraries the loader reads, from the env or from the installed plugin
lookup_lib_dir() {
    ( cd "${TMPDIR:-/tmp}" && "$PYTHON" -c "import flashinfer_gaudi, pathlib; print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')" 2>/dev/null ) | tail -1
}
KERNEL_LIB="${GDN_FUSED_GAUDI_KERNEL_LIBRARY:-}"
NATIVE_LIB="${GDN_FUSED_GAUDI_NATIVE_LIBRARY:-}"
if [ -z "$KERNEL_LIB" ] || [ -z "$NATIVE_LIB" ]; then
    lib_dir=$(lookup_lib_dir)
    if [ -z "$lib_dir" ] || [ ! -d "$lib_dir" ]; then
        printf 'error: cannot locate the installed flashinfer_gaudi/lib; set GDN_FUSED_GAUDI_KERNEL_LIBRARY and GDN_FUSED_GAUDI_NATIVE_LIBRARY\n' >&2
        exit 2
    fi
    if [ -z "$KERNEL_LIB" ]; then KERNEL_LIB="$lib_dir/libgdn_fused_gaudi2.so"; fi
    if [ -z "$NATIVE_LIB" ]; then
        ops_candidates=("$lib_dir"/gdn_fused_ops*.so)
        if [ ${#ops_candidates[@]} -ne 1 ] || [ ! -f "${ops_candidates[0]}" ]; then
            printf 'error: expected one gdn_fused_ops*.so in %s; set GDN_FUSED_GAUDI_NATIVE_LIBRARY\n' "$lib_dir" >&2
            exit 2
        fi
        NATIVE_LIB="${ops_candidates[0]}"
    fi
fi

{
    date
    uname -a
    "$PYTHON" -c 'import torch; print(torch.__version__)' 2>&1
    tpc-clang --version 2>&1 | head -1
    printf 'kernel library: %s\n' "$KERNEL_LIB"
    sha256sum "$KERNEL_LIB" 2>&1
    printf 'native library: %s\n' "$NATIVE_LIB"
    sha256sum "$NATIVE_LIB" 2>&1
} > "$OUT_DIR/env.txt"

GDB_OK=0
if command -v gdb >/dev/null 2>&1; then
    printf 'gdb: present\n' >> "$SUMMARY"
    GDB_OK=1
else
    # The rental container's apt lists are usually warm (the toolchain step refreshed them):
    # try the install first and pay for a list refresh only when the bare install fails.
    DEBIAN_FRONTEND=noninteractive timeout 300 apt-get install -y -qq gdb >/dev/null 2>&1 || true
    if ! command -v gdb >/dev/null 2>&1; then
        if DEBIAN_FRONTEND=noninteractive timeout 300 apt-get update -qq >/dev/null 2>&1; then
            timeout 300 apt-get install -y -qq gdb >/dev/null 2>&1 || true
        fi
    fi
    if command -v gdb >/dev/null 2>&1; then
        printf 'gdb: installed\n' >> "$SUMMARY"
        GDB_OK=1
    else
        printf 'gdb: missing\n' >> "$SUMMARY"
    fi
fi

# the strip build (R1-e2): the verify kernel ELF without its .llvmbc section
strip_ok=0
strip_reason=""
timeout 900 cmake -S "$PLUGIN_ROOT/csrc/gdn_fused" -B "$OUT_DIR/build_strip" -DCMAKE_BUILD_TYPE=Release -DGDN_FUSED_VERIFY_STRIP_BITCODE=ON > "$OUT_DIR/build_strip.log" 2>&1
cmake_rc=$?
if [ $cmake_rc -ne 0 ]; then
    strip_reason="cmake configure rc=$cmake_rc"
else
    timeout 900 cmake --build "$OUT_DIR/build_strip" --target gdn_fused_gaudi2 --parallel >> "$OUT_DIR/build_strip.log" 2>&1
    cmake_rc=$?
    if [ $cmake_rc -ne 0 ]; then strip_reason="cmake build rc=$cmake_rc"; fi
fi
if [ -z "$strip_reason" ] && [ ! -f "$OUT_DIR/build_strip/gdn_core_verify_gaudi2.o" ]; then
    strip_reason="gdn_core_verify_gaudi2.o missing"
fi
if [ -z "$strip_reason" ]; then
    llvmbc_count=$(readelf -S -W "$OUT_DIR/build_strip/gdn_core_verify_gaudi2.o" 2>/dev/null | grep -c llvmbc || true)
    if [ "${llvmbc_count:-1}" != "0" ]; then strip_reason=".llvmbc section present"; fi
fi
if [ -z "$strip_reason" ] && [ ! -f "$STRIP_LIB" ]; then
    strip_reason="libgdn_fused_gaudi2.so missing"
fi
if [ -z "$strip_reason" ]; then
    strip_ok=1
    printf 'strip build: ok\n' >> "$SUMMARY"
else
    printf 'strip build: failed (%s)\n' "$strip_reason" >> "$SUMMARY"
fi

run_one() {
    local name="$1" lazy="$2" lib="$3" node="$4" rc verdict log
    log="$OUT_DIR/$name.log"
    # -s (no capture) on both paths: the graph compiler's stderr must reach the log.
    if [ "$GDB_OK" = "1" ]; then
        (
            cd "$PLUGIN_ROOT" || exit 2
            env PYTHONPATH="$PLUGIN_ROOT" PT_HPU_LAZY_MODE="$lazy" \
                GDN_FUSED_GAUDI_KERNEL_LIBRARY="$lib" GDN_FUSED_GAUDI_NATIVE_LIBRARY="$NATIVE_LIB" \
                HABANA_LOGS="$OUT_DIR/$name.habana_logs" PYTHONFAULTHANDLER=1 \
                timeout -k 10 "$PROBE_TIMEOUT_S" gdb -q -batch -nx \
                -ex "set pagination off" -ex "set confirm off" -ex "set disable-randomization off" \
                -ex "handle SIGPIPE nostop noprint pass" -ex "handle SIGUSR1 nostop noprint pass" \
                -ex "handle SIGUSR2 nostop noprint pass" \
                -ex run -ex "bt 40" -ex "info threads" -ex "thread apply all bt 25" -ex "info sharedlibrary" -ex kill \
                --args "$(command -v "$PYTHON")" -X faulthandler -m pytest -q -p no:cacheprovider -rs -s "$node"
        ) > "$log" 2>&1
        rc=$?
    else
        # No gdb: at the time limit the check gets SIGABRT, which makes the fault handler write the
        # Python stack of every thread before the process dies (no core file: the limit is set to 0).
        (
            cd "$PLUGIN_ROOT" || exit 2
            ulimit -c 0 2>/dev/null
            env PYTHONPATH="$PLUGIN_ROOT" PT_HPU_LAZY_MODE="$lazy" \
                GDN_FUSED_GAUDI_KERNEL_LIBRARY="$lib" GDN_FUSED_GAUDI_NATIVE_LIBRARY="$NATIVE_LIB" \
                HABANA_LOGS="$OUT_DIR/$name.habana_logs" PYTHONFAULTHANDLER=1 \
                timeout -s ABRT -k 10 "$PROBE_TIMEOUT_S" "$PYTHON" -X faulthandler -m pytest -q -p no:cacheprovider -rs -s "$node"
        ) > "$log" 2>&1
        rc=$?
    fi
    verdict=$(classify "$log" "$rc")
    printf '%s: %s\n' "$name" "$verdict" >> "$SUMMARY"
    case "$verdict" in
        crashed:* | timed_out)
            if grep -q 'received signal' "$log" 2>/dev/null; then
                grep -m1 -A 16 'received signal' "$log" 2>/dev/null | sed 's/^/    /' >> "$SUMMARY" || true
            else
                # no gdb backtrace: the Python stack the fault handler wrote stands in for it
                grep -m1 -A 16 'Fatal Python error' "$log" 2>/dev/null | sed 's/^/    /' >> "$SUMMARY" || true
            fi
            ;;
    esac
    LAST_VERDICT="$verdict"
}

run_one "${NAMES[0]}" "${LAZYS[0]}" "$KERNEL_LIB" "${NODES[0]}"
if [ "$LAST_VERDICT" != "passed" ]; then
    printf 'control failed: verify runs skipped\n' >> "$SUMMARY"
    cat "$SUMMARY"
    exit 3
fi

i=1
while [ $i -lt ${#NAMES[@]} ]; do
    lib="$KERNEL_LIB"
    if [ "${LIBS[$i]}" = "strip" ]; then
        if [ "$strip_ok" = "0" ]; then
            printf '%s: not_run (strip build failed)\n' "${NAMES[$i]}" >> "$SUMMARY"
            i=$((i + 1))
            continue
        fi
        lib="$STRIP_LIB"
    fi
    run_one "${NAMES[$i]}" "${LAZYS[$i]}" "$lib" "${NODES[$i]}"
    i=$((i + 1))
done

cat "$SUMMARY"
exit 0
