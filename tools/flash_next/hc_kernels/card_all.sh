#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# card_all.sh: the hc-kernels track's one-card job, three parts on the same card, one after the other:
#   1. card_1card.sh: the interleaved hyper-connection ops (card op tests, shipped op tests, bench)  -> $FARM_OUT/hc
#   2. the tiny-model token gate at TP1 with VLLM_HPU_HC_FUSED_IL=1 and copy mode 2 (the farm's farm_tiny.py: tokens
#      against the CPU reference, and any "HC fused ... is off for" line fails it)               -> $FARM_OUT/tiny
#   3. card_gdn_verify.sh: the GDN verify variants (op test, bench per variant, bits per variant) -> $FARM_OUT/gdn
# Exit code: the first failing part's.
set -u
OUT="${FARM_OUT:-/tmp/hc-kernels}"
FARM_OUT="$OUT/hc" bash tools/flash_next/hc_kernels/card_1card.sh
r1=$?
sleep 3
if [ -f /farm/farm_tiny.py ]; then
    python /farm/farm_tiny.py --tp 1 --gdn hc --env VLLM_HPU_HC_FUSED_IL=1 --env VLLM_HPU_HC_FUSED_COPY_MODE=2 \
        --out "$OUT/tiny" 2>&1 | tee "$OUT/tiny.log"
    r2="${PIPESTATUS[0]}"
else
    echo "card_all: /farm/farm_tiny.py is missing: no tiny gate" | tee "$OUT/tiny.log"
    r2=0
fi
sleep 8
FARM_OUT="$OUT/gdn" bash tools/flash_next/hc_kernels/card_gdn_verify.sh
r3=$?
echo "=== hc-kernels card_all: hc exit $r1, tiny exit $r2, gdn exit $r3"
for r in "$r1" "$r2" "$r3"; do
    if [ "$r" -ne 0 ]; then exit "$r"; fi
done
exit 0
