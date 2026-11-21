#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# mtp_tiny_card.sh: the tiny model's speculative mode on a card-farm job's cards, one arm after the other.
# A tiny server with speculation starts in about a minute, a real-model server in seven: every MTP change
# goes through this before a real-model job.
#
#   farm job mtp-card-ready NAME any:4 15m -- bash tools/flash_next/rental/mtp_tiny_card.sh [K] [ARM...]
#
# K is the number of drafted tokens (default 3). An ARM is NAME or NAME:KEY=VALUE[,KEY=VALUE...]: the extra
# server environment of that arm. Without arms: "base" (the tree's defaults) and
# "loop-chunk:VLLM_HPU_MTP_DRAFT_LOOP=1,VLLM_HPU_MTP_CHUNK_KV=1".
#
# Every arm runs the farm's tiny smoke (python /farm/farm_tiny.py, tensor parallel size = the job's cards) with
# --gdn hc --spec K: the mode hc without speculation, then hc+mtpK with its rungs (fused verification, torch
# verification, no speculation); the tokens are compared with the reference and with the mode without
# speculation. Results: $FARM_OUT/<arm>/summary.json and server logs. Exit code: the number of arms whose
# speculative mode did not pass on a speculative rung.

set -u

K=3
case "${1:-}" in
    [1-9]) K="$1"; shift ;;
esac
if [ $# -eq 0 ]; then
    set -- base "loop-chunk:VLLM_HPU_MTP_DRAFT_LOOP=1,VLLM_HPU_MTP_CHUNK_KV=1"
fi
OUT="${FARM_OUT:-/results/mtp-tiny}"
PYTHON="${FARM_PYTHON:-python3}"
FARM_TINY="${FARM_TINY:-/farm/farm_tiny.py}"
COMMON=(--env VLLM_HPU_PLE_HOST_NGRAM_IDS=1 --env PT_HPU_LAZY_ACC_PAR_MODE=0 --env VLLM_HPU_COLLECTIVE_MARK_STEP=0)

failed=0
for arm in "$@"; do
    name="${arm%%:*}"
    extra=()
    if [ "$arm" != "$name" ]; then
        IFS=',' read -r -a pairs <<<"${arm#*:}"
        for kv in "${pairs[@]}"; do extra+=(--env "$kv"); done
    fi
    echo "=== mtp_tiny_card: arm $name (k = $K) ${extra[*]:-} $(date -u +%T)"
    "$PYTHON" "$FARM_TINY" --gdn hc --spec "$K" --out "$OUT/$name" "${COMMON[@]}" ${extra[@]+"${extra[@]}"}
    rc=$?
    "$PYTHON" - "$OUT/$name" "$name" "$K" <<'PY'
import json
import pathlib
import sys

out, name, k = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
try:
    modes = json.loads((out / "summary.json").read_text()).get("modes", [])
except (OSError, ValueError) as error:
    print("ARM %s: no summary (%r)" % (name, error))
    sys.exit(0)
for mode in modes:
    acceptance = mode.get("acceptance") or {}
    print("ARM %s: %s %s rung=%s tokens-equal-to-plain=%s acceptance-length=%s per-position=%s decode %s ms/token" % (
        name, mode.get("mode"), mode.get("status"), mode.get("rung"), (mode.get("spec_vs_plain") or {}).get("equal"),
        acceptance.get("acceptance_length"), acceptance.get("position_rates"), mode.get("decode_ms_per_token")))
    for line in (mode.get("errors") or [])[:4]:
        print("   error: %s" % str(line)[:300])
PY
    passed="$("$PYTHON" -c "
import json, sys
try:
    modes = json.load(open(sys.argv[1])).get('modes', [])
    spec = [m for m in modes if '+mtp' in str(m.get('mode'))]
    print(1 if spec and all(m.get('status') == 'passed' for m in spec) else 0)
except Exception:
    print(0)
" "$OUT/$name/summary.json")"
    echo "=== mtp_tiny_card: arm $name rc=$rc speculative-pass=$passed $(date -u +%T)"
    if [ "$passed" != 1 ]; then failed=$((failed + 1)); fi
done
echo "mtp_tiny_card: $failed of $# arms failed"
exit "$failed"
