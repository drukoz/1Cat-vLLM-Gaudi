#!/usr/bin/env bash
# qsa_next_job.sh: several arms of the sparse-QSA A/B one after the other in one card-farm container (the job's
# cards serve one arm at a time). Late-bound: ~/farm/qsa-fused/results/bench/ on the node, /results/bench/ in the
# container. The arms are the lines of /results/bench/next_arms.txt, read when the job starts:
#     NAME KEY=VALUE [KEY=VALUE]...        (# starts a comment)
# Each arm runs qsa_ab_job.sh NAME with those variables set (VLLM_HPU_QSA_FUSED, VLLM_HPU_QSA_INDEXER_SHARD,
# VLLM_HPU_QSA_TABLE_WIDTH, QSA_AB_LEVELS, QSA_AB_CTX, ...), its results in $FARM_OUT/NAME, and an equal share
# of the time that is left to QSA_AB_DEADLINE. The selection check and the kernels' op test are left out
# (QSA_AB_SKIP_IDS=1, QSA_AB_SKIP_OPS=1) unless a line sets them to 0.
# Words this script reads itself (VERSION 2, 2026-11-21 08:30 UTC):
#   QSA_TINY=MODES         the arm is the tiny-model token gate (/farm/farm_tiny.py --gdn MODES, the job's cards) instead
#                          of a real-model server; QSA_TINY_SPEC=K adds its MTP mode, QSA_TINY_ENV=K=V,K=V,... its
#                          server variables (--env); a pass leaves $FARM_OUT/NAME.PASSED.
#   QSA_AFTER_TINY=NAME    run the arm only if the tiny arm NAME passed.
#   QSA_NEEDS_ENV=FLAG     run the arm only if this build's vllm_gaudi.envs knows FLAG.
#   QSA_MIN_SECONDS=S      the smallest share the arm runs with (default 780).
set -u
BASE_OUT="${FARM_OUT:-/tmp/qsa-next}"
DEADLINE="${QSA_AB_DEADLINE:-$(($(date -u +%s) + 2300))}"
ARMS="${QSA_NEXT_ARMS:-/results/bench/next_arms.txt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BASE_OUT"
say() { echo "QSANEXT [$(date -u +%H:%M:%S)] $*"; }
LINES=()
while IFS= read -r line; do
    case "$line" in '' | '#'* | ' '*'#'*) continue ;; esac
    LINES+=("$line")
done <"$ARMS"
N="${#LINES[@]}"
if [ "$N" -eq 0 ]; then
    say "no arms in $ARMS"
    exit 2
fi
cp "$ARMS" "$BASE_OUT/next_arms.txt" 2>/dev/null
say "$N arm(s) from $ARMS, $((DEADLINE - $(date -u +%s))) s to the deadline"
RC=0
i=0
for line in "${LINES[@]}"; do
    i=$((i + 1))
    # shellcheck disable=SC2206  # words of one line: the arm's name, then KEY=VALUE pairs
    words=($line)
    name="${words[0]}"
    now="$(date -u +%s)"
    share=$(((DEADLINE - now) / (N - i + 1)))
    # QSA_ARM_SECONDS=S on an arm's line: that arm may take up to S seconds (at most what is left), not an equal share
    for kv in "${words[@]:1}"; do
        case "$kv" in QSA_ARM_SECONDS=[0-9]*) share="${kv#QSA_ARM_SECONDS=}" ;; esac
    done
    if [ "$share" -gt $((DEADLINE - now)) ]; then share=$((DEADLINE - now)); fi
    tiny="" after="" needs="" minsec=780 tspec="" tenv=""
    for kv in "${words[@]:1}"; do
        case "$kv" in
            QSA_TINY=*) tiny="${kv#QSA_TINY=}" ;;
            QSA_TINY_SPEC=[0-9]*) tspec="${kv#QSA_TINY_SPEC=}" ;;
            QSA_TINY_ENV=*) tenv="${kv#QSA_TINY_ENV=}" ;;
            QSA_AFTER_TINY=*) after="${kv#QSA_AFTER_TINY=}" ;;
            QSA_NEEDS_ENV=*) needs="${kv#QSA_NEEDS_ENV=}" ;;
            QSA_MIN_SECONDS=[0-9]*) minsec="${kv#QSA_MIN_SECONDS=}" ;;
        esac
    done
    if [ "$share" -lt "$minsec" ]; then
        say "arm $name SKIPPED: $((DEADLINE - now)) s left for $((N - i + 1)) arm(s)"
        continue
    fi
    if [ -n "$needs" ] && ! "${FARM_PYTHON:-python3}" -c "import sys, vllm_gaudi.envs as e; sys.exit(0 if hasattr(e, '$needs') else 1)" >/dev/null 2>&1; then
        say "arm $name SKIPPED: this build's vllm_gaudi.envs has no $needs"
        continue
    fi
    if [ -n "$after" ] && [ ! -f "$BASE_OUT/$after.PASSED" ]; then
        say "arm $name SKIPPED: the tiny gate $after did not pass"
        continue
    fi
    if [ -n "$tiny" ]; then
        targs=(--gdn "$tiny" --out "$BASE_OUT/$name" --no-recipe-cache)
        if [ -n "$tspec" ]; then targs+=(--spec "$tspec"); fi
        if [ -n "$tenv" ]; then
            IFS=',' read -r -a tenvs <<<"$tenv"
            for e in "${tenvs[@]}"; do targs+=(--env "$e"); done
        fi
        say "arm $i of $N: $name: tiny smoke ${targs[*]}, $share s"
        timeout "$share" "${FARM_PYTHON:-python3}" /farm/farm_tiny.py "${targs[@]}"
        rc=$?
        if [ "$rc" -eq 0 ]; then touch "$BASE_OUT/$name.PASSED"; fi
        say "arm $name (tiny gate) ended with exit code $rc"
        if [ "$rc" -ne 0 ]; then RC="$rc"; fi
        pkill -9 -f "VLLM::" 2>/dev/null
        sleep 10
        continue
    fi
    say "arm $i of $N: $name (${words[*]:1}), $share s"
    (
        export FARM_OUT="$BASE_OUT/$name" QSA_AB_SKIP_IDS=1 QSA_AB_SKIP_OPS=1 QSA_AB_DEADLINE="$((now + share))"
        for kv in "${words[@]:1}"; do
            case "$kv" in [A-Za-z_]*=*) export "$kv" ;; *) echo "QSANEXT: bad word '$kv' in the line of $name" ;; esac
        done
        bash "$HERE/qsa_ab_job.sh" "$name"
    )
    rc=$?
    say "arm $name ended with exit code $rc"
    if [ "$rc" -ne 0 ]; then RC="$rc"; fi
    sleep 10
done
say "DONE (exit $RC)"
exit "$RC"
