#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# idcheck_job.sh: the card-against-CPU id check (id_check.py, card rule 9) as a farm job on the job's cards.
#   farm job collectives idcheck-rule9 any:1 10m -- bash tools/flash_next/collectives/idcheck_job.sh
# One card: one process, a rank's vocabulary shard (31,040 columns) and then a whole row of logits (248,320).
# Several cards: one hccl rank per card, the shard width only (the gather of the (value, id) pairs is in it).
# Results in $FARM_OUT (idcheck*.json, the rank logs). Exit code 0: every serving case equals the CPU.
set -u
OUT="${FARM_OUT:-/tmp/idcheck}"
N="${FARM_NCARDS:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
echo "idcheck_job: $N card(s) ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}) [$(date -u +%T)]"
python3 "$HERE/id_check.py" --ranks "$N" --out "$OUT/idcheck_r$N.json" --logdir "$OUT/logs_shard" --timeout-s "${IDCHECK_TIMEOUT_S:-300}"
rc=$?
echo "idcheck_job: shard width rc $rc [$(date -u +%T)]"
if [ "$N" -eq 1 ]; then
    python3 "$HERE/id_check.py" --ranks 1 --width 248320 --rows 1,2,32 \
        --cases vp_argmax,fp32_ids_row,argmax_to_int32,argmax_int,argmax_int64,topk_to_int32,topk_gather_fp32 \
        --out "$OUT/idcheck_wide_r1.json" --logdir "$OUT/logs_wide" --timeout-s "${IDCHECK_TIMEOUT_S:-300}"
    rc2=$?
    echo "idcheck_job: whole row rc $rc2 [$(date -u +%T)]"
    [ "$rc" -eq 0 ] && rc=$rc2
fi
exit "$rc"
