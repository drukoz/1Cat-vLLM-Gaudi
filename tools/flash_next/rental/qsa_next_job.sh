#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# qsa_next_job.sh: several arms of the sparse-QSA A/B one after the other in one card-farm container (the job's
# cards serve one arm at a time), each arm a run of qsa_ab_job.sh with its own variables. The arms are the lines of
# $QSA_NEXT_ARMS (default: qsa_next_arms_tp8.txt next to this script), read when the job starts:
#     NAME KEY=VALUE [KEY=VALUE]...        (# starts a comment)
# Each arm runs qsa_ab_job.sh NAME with those variables set (VLLM_HPU_QSA_FUSED, VLLM_HPU_QSA_INDEXER_SHARD,
# VLLM_HPU_QSA_TABLE_WIDTH, VLLM_GRAPH_RESERVED_MEM, QSA_AB_LEVELS, QSA_AB_CTX, QSA_AB_LONG_STREAMS, ...), its
# results in $FARM_OUT/NAME, and an equal share of the time left to the deadline, or at most QSA_ARM_SECONDS when
# its line sets that. The selection check and the kernels' op test are left out (QSA_AB_SKIP_IDS=1,
# QSA_AB_SKIP_OPS=1) unless a line sets them to 0. The deadline: QSA_AB_DEADLINE (epoch seconds), else the start
# plus QSA_NEXT_SECONDS (default 2300); give the job's maximum hold minus about a minute.
#
#     bin/farm job qsa-fused real8-qsa-long all 60m --env QSA_NEXT_SECONDS=3480 -- \
#         bash tools/flash_next/rental/qsa_next_job.sh
set -u
BASE_OUT="${FARM_OUT:-/tmp/qsa-next}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEADLINE="${QSA_AB_DEADLINE:-$(($(date -u +%s) + ${QSA_NEXT_SECONDS:-2300}))}"
ARMS="${QSA_NEXT_ARMS:-$HERE/qsa_next_arms_tp8.txt}"
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
    if [ "$share" -lt 780 ]; then
        say "arm $name SKIPPED: $((DEADLINE - now)) s left for $((N - i + 1)) arm(s)"
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
