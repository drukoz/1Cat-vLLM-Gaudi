#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run every unit-test file of the plugin on CPU, one file per pytest process.
#
# Usage: run_cpu_tests.sh <target-dir> [pytest args...]
#
#   <target-dir>   directory prepared by setup.sh (contains venv/)
#   [pytest args]  extra arguments forwarded to every pytest invocation
#
# Writes <target-dir>/cpu_tests.tsv with rows:
#   file<TAB>outcome(pass|fail|skip|error|timeout)<TAB>passed<TAB>skipped<TAB>first error line
#
# Resumable: a second run skips files already present in the TSV.
# Use FRESH=1 run_cpu_tests.sh <target-dir> to start over.

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <target-dir> [pytest args]" >&2
    exit 2
fi

TARGET_DIR="$(mkdir -p -- "$1" && cd -- "$1" && pwd)"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV="$TARGET_DIR/venv"
OUT="$TARGET_DIR/cpu_tests.tsv"
LOG_DIR="$TARGET_DIR/test_logs"
PER_FILE_TIMEOUT="${PER_FILE_TIMEOUT:-600}"

[[ -x "$VENV/bin/python" ]] || {
    echo "no venv at $VENV; run setup.sh first" >&2
    exit 1
}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$LOG_DIR"

TIMEOUT_BIN="$(command -v timeout || command -v gtimeout || true)"

# Minimal timeout(1) replacement for platforms without coreutils.
py_timeout() {
    python - "$PER_FILE_TIMEOUT" "$@" <<'PY'
import os, signal, subprocess, sys
secs = float(sys.argv[1]); cmd = sys.argv[2:]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, start_new_session=True)
def on_alarm(signum, frame):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    print(f"TIMEOUT after {secs:g}s", flush=True)
    sys.exit(124)
signal.signal(signal.SIGALRM, on_alarm)
signal.signal(signal.SIGTERM, on_alarm)
signal.setitimer(signal.ITIMER_REAL, secs)
for line in proc.stdout:
    sys.stdout.write(line); sys.stdout.flush()
proc.wait()
signal.setitimer(signal.ITIMER_REAL, 0)
sys.exit(proc.returncode)
PY
}

cd "$PLUGIN_ROOT"
FILES=()
while IFS= read -r f; do
    FILES+=("$f")
done < <(find tests/unit_tests -name 'test_*.py' | sort)

DONE=0
TOTAL=${#FILES[@]}
STARTED=$(date +%s)
echo "[cpu-tests] $TOTAL test files, logs in $LOG_DIR"

if [[ "${FRESH:-0}" == "1" ]]; then
    : >"$OUT"
elif [[ ! -f "$OUT" ]]; then
    : >"$OUT"
fi
done_files() {
    [[ -f "$OUT" ]] && cut -f1 "$OUT" || true
}
SKIP_LIST="$(done_files)"

record() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >>"$OUT"
}

for f in "${FILES[@]}"; do
    DONE=$((DONE + 1))
    if printf '%s\n' "$SKIP_LIST" | grep -qxF "$f"; then
        echo "[cpu-tests] $DONE/$TOTAL $f (already recorded, skipped)"
        continue
    fi
    log="$LOG_DIR/$(printf '%s' "${f#tests/unit_tests/}" | tr '/' '_' | sed 's/\.py$/.log/')"
    if [[ -n "$TIMEOUT_BIN" ]]; then
        "$TIMEOUT_BIN" "$PER_FILE_TIMEOUT" python -m pytest -q -p no:cacheprovider "$f" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >"$log" 2>&1
        rc=$?
    else
        py_timeout python -m pytest -q -p no:cacheprovider "$f" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >"$log" 2>&1
        rc=$?
    fi

    summary="$(grep -E '^(=+[^\n]*=+|[0-9]+ (passed|failed|error|skipped|warnings))' "$log" | tail -1)"
    [[ -n "$summary" ]] || summary="$(grep -E 'passed|failed|error|skipped|no tests ran' "$log" | tail -1)"
    npass="$(printf '%s\n' "$summary" | awk '{for(i=2;i<=NF;i++){w=$i; sub(/,$/,"",w); if (w=="passed"){v=$(i-1); sub(/[^0-9].*/,"",v); print v; exit}}}')"
    nskip="$(printf '%s\n' "$summary" | awk '{for(i=2;i<=NF;i++){w=$i; sub(/,$/,"",w); if (w=="skipped"){v=$(i-1); sub(/[^0-9].*/,"",v); print v; exit}}}')"
    npass="${npass:-0}"
    nskip="${nskip:-0}"

    case "$rc" in
        124 | 137)
            outcome=timeout
            ;;
        0)
            if [[ "$npass" -eq 0 && "$nskip" -gt 0 ]]; then
                outcome=skip
            else
                outcome=pass
            fi
            ;;
        5) # pytest: no tests collected
            outcome=skip
            ;;
        1)
            if printf '%s' "$summary" | grep -qE '[0-9]+ error' &&
                ! printf '%s' "$summary" | grep -qE '[0-9]+ failed'; then
                outcome=error
            else
                outcome=fail
            fi
            ;;
        *)
            outcome=error
            ;;
    esac

    err="$(grep -m1 -E '^(E  +|ERROR [^ ]|ImportError|ModuleNotFoundError)' "$log" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "$err" ]] || err="$summary"
    err="$(printf '%s' "$err" | tr '\t\n' '  ')"

    record "$f" "$outcome" "$npass" "$nskip" "$err"
    echo "[cpu-tests] $DONE/$TOTAL $f -> $outcome (passed=$npass skipped=$nskip)"
done

echo "[cpu-tests] finished in $((($(date +%s) - STARTED) / 60))m$((($(date +%s) - STARTED) % 60))s; results in $OUT"
exit 0
