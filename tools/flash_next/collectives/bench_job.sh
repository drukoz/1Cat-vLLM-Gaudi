#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# bench_job.sh: the collectives track's card bench as one farm job (all cards, or one module group).
#   farm job collectives c4-bench-r8-r4 all   18m -- bash tools/flash_next/collectives/bench_job.sh
#   farm job collectives c4-bench-r4    any:4 12m -- bash tools/flash_next/collectives/bench_job.sh
# (the hold covers every step at its limit; a run without a stalled step takes about 7 minutes on 8 cards)
# Steps, each with its own time limit, the cards released between them:
#   1. id_check.py on every card of the job (the vocabulary-parallel arg max through the all-gather, card rule 9);
#   2. bench_c4.py on every card of the job (all-reduce against all-gather + sum, lazy and HPU-graph replay);
#   3. with 8 cards: bench_c4.py on the module group 0-3 (the 4-rank cost per collective, same script, same hour);
#   4. with 8 cards and BENCH_PAIR=1: the replay regime on the groups 0-3 and 4-7 at the same time (two TP4
#      servers side by side); last, so that a failure costs nothing above.
# Results in $FARM_OUT: idcheck_r<N>.json, c4_r<N>.json, c4_r4.json, c4_r4_pair_{a,b}.json, g41.json, the rank logs.
set -u
OUT="${FARM_OUT:-/tmp/c4}"
N="${FARM_NCARDS:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROWS="${BENCH_ROWS:-1,8,32,128}"
REGIMES="${BENCH_REGIMES:-lazy,replay}"
mkdir -p "$OUT"
cd "$OUT" || exit 2
now() { date -u +%T; }

# cards_free PATTERN: every process of the step gone, then a pause (a card is not free the moment its process ends)
cards_free() {
    local i
    for i in $(seq 1 20); do pgrep -f "$1" >/dev/null || break; sleep 1; done
    pkill -9 -f "$1" 2>/dev/null
    sleep "${CARD_RELEASE_S:-12}"
}

echo "bench_job: $N card(s) ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}) [$(now)]"
rc_all=0

if [ "${BENCH_IDCHECK:-1}" = 1 ]; then
    timeout "${IDCHECK_S:-150}" python3 "$HERE/id_check.py" --ranks "$N" --budget-s 40 --timeout-s 130 \
        --cases "${IDCHECK_CASES:-vp_argmax,vp_topk,fp32_ids_row}" \
        --out "$OUT/idcheck_r$N.json" --logdir "$OUT/idcheck_r$N" > "$OUT/idcheck_r$N.log" 2>&1
    rc=$?
    echo "bench_job: id check on $N ranks rc $rc [$(now)]"
    grep -E "^idcheck:" "$OUT/idcheck_r$N.log" | cut -c1-260
    [ "$rc" -eq 0 ] || rc_all=$rc
    cards_free id_check.py
fi

timeout "${BENCH_S:-300}" python3 "$HERE/bench_c4.py" --ranks "$N" --rows "$ROWS" --regimes "$REGIMES" \
    --timeout-s 280 --out "$OUT/c4_r$N.json" --logdir "$OUT/c4_r$N" > "$OUT/c4_r$N.log" 2>&1
rc=$?
echo "bench_job: bench on $N ranks rc $rc [$(now)]"
grep -E "^c4|^rank0:" "$OUT/c4_r$N.log" | cut -c1-260
[ "$rc" -eq 0 ] || rc_all=$rc
cards_free bench_c4.py

if [ "$N" -eq 8 ]; then
    timeout "${BENCH_S:-300}" python3 "$HERE/bench_c4.py" --ranks 4 --cards 0,1,2,3 --rows "$ROWS" --regimes "$REGIMES" \
        --timeout-s 280 --out "$OUT/c4_r4.json" --logdir "$OUT/c4_r4" > "$OUT/c4_r4.log" 2>&1
    rc=$?
    echo "bench_job: bench on 4 ranks (modules 0-3) rc $rc [$(now)]"
    grep -E "^c4|^rank0:" "$OUT/c4_r4.log" | cut -c1-260
    [ "$rc" -eq 0 ] || rc_all=$rc
    cards_free bench_c4.py
    if [ "${BENCH_PAIR:-1}" = 1 ]; then
        for g in a:0,1,2,3 b:4,5,6,7; do
            timeout "${BENCH_PAIR_S:-120}" python3 "$HERE/bench_c4.py" --ranks 4 --cards "${g#*:}" --rows 1,32 \
                --regimes replay --forms none,all_reduce,gather_sum --tail-rows "" --timeout-s 100 \
                --out "$OUT/c4_r4_pair_${g%%:*}.json" --logdir "$OUT/c4_r4_pair_${g%%:*}" \
                > "$OUT/c4_r4_pair_${g%%:*}.log" 2>&1 &
        done
        wait
        echo "bench_job: two 4-rank groups at once (modules 0-3 and 4-7), replay [$(now)]"
        grep -H -E "^c4-fit|^rank0:|^c4: rank" "$OUT"/c4_r4_pair_?.log | cut -c1-260
    fi
fi

python3 "$HERE/decide_g41.py" "$OUT" --json "$OUT/g41.json" 2>&1 | cut -c1-260
echo "bench_job: DONE rc $rc_all [$(now)]"
exit "$rc_all"
