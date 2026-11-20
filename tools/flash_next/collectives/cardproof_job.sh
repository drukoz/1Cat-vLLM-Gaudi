#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# cardproof_job.sh: the card proof of the collective cuts and of the top-k head, as one farm job on one module
# group (tensor parallel = the job's card count).
#   farm job collectives cuts-cardproof-tp4 any:4 18m --env VLLM_HPU_COLLECTIVE_MARK_STEP=0 \
#        --env VLLM_HPU_PLE_HOST_NGRAM_IDS=1 -- bash tools/flash_next/collectives/cardproof_job.sh
# 1. id_check.py on the job's cards as hccl ranks: the vocabulary-parallel arg max and top-k (k = 20 and 64) through
#    the all-gather, against the CPU, rows 1, 2, 32, 128, one lazy graph and a replayed one.
# 2. The tiny model, fused hyper-connections, tokens against its CPU reference (farm_tiny.py), once per arm:
#    logits-rider   VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 VLLM_HPU_PLE_TABLE_RIDER=1
#    logits2-embed  VLLM_HPU_VOCAB_PARALLEL_LOGITS=2 VLLM_HPU_REPLICATED_EMBEDDING=1
#    (CARDPROOF_ARMS="name:K=V,K=V name:..." replaces the arms.)
# Results in $FARM_OUT: idcheck_r<N>.json and .log, tiny-<arm>/ (summary.json, hc/server0.log), tiny-<arm>.log.
# Exit code 0: the serving id cases equal the CPU and every arm's tokens match the reference.
# Run from the tree's root (farm job does): farm_tiny.py imports the tree's tools.flash_next.rental.
set -u
OUT="${FARM_OUT:-/tmp/cardproof}"
N="${FARM_NCARDS:-4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FARM_TOOLS="${FARM_TOOLS_DIR:-/farm}"
ARMS="${CARDPROOF_ARMS:-logits-rider:VLLM_HPU_VOCAB_PARALLEL_LOGITS=1,VLLM_HPU_PLE_TABLE_RIDER=1 logits2-embed:VLLM_HPU_VOCAB_PARALLEL_LOGITS=2,VLLM_HPU_REPLICATED_EMBEDDING=1}"
mkdir -p "$OUT"
now() { date -u +%T; }
rc_all=0
echo "cardproof: $N card(s) ${FARM_CARDS:-?} (modules ${FARM_MODULES:-?}) [$(now)]"

if [ "${CARDPROOF_IDCHECK:-1}" = 1 ]; then
    timeout "${IDCHECK_S:-220}" python3 "$HERE/id_check.py" --ranks "$N" --top 20,64 --rows 1,2,32,128 --budget-s 60 \
        --cases vp_argmax,vp_topk,fp32_ids_row,topk_gather_fp32,stage2_topk_gather,first_maximum \
        --timeout-s 200 --out "$OUT/idcheck_r$N.json" --logdir "$OUT/idcheck_r$N" > "$OUT/idcheck_r$N.log" 2>&1
    rc=$?
    echo "cardproof: id check on $N ranks rc $rc [$(now)]"
    grep -E "^idcheck:" "$OUT/idcheck_r$N.log" | cut -c1-260
    [ "$rc" -eq 0 ] || rc_all=$rc
    for i in $(seq 1 20); do pgrep -f id_check.py >/dev/null || break; sleep 1; done
    sleep "${CARD_RELEASE_S:-12}"  # a card is not free the moment its process ends
fi

for arm in $ARMS; do
    name="${arm%%:*}"
    envs=()
    IFS=',' read -r -a pairs <<< "${arm#*:}"
    for kv in "${pairs[@]}"; do envs+=(--env "$kv"); done
    echo "cardproof: tiny $name: ${pairs[*]} [$(now)]"
    python3 "$FARM_TOOLS/farm_tiny.py" --gdn hc --out "$OUT/tiny-$name" "${envs[@]}" > "$OUT/tiny-$name.log" 2>&1
    rc=$?
    echo "cardproof: tiny $name rc $rc [$(now)]"
    grep -E "PASSED|FAILED|TIMED_OUT|decode_ms_per_token|fell back|is off for|Error" "$OUT/tiny-$name.log" | tail -n 8 | cut -c1-260
    [ "$rc" -eq 0 ] || rc_all=$rc
    sleep "${CARD_RELEASE_S:-12}"
done
echo "cardproof: DONE rc $rc_all [$(now)]"
exit "$rc_all"
