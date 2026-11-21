#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# moe_chunk_ab.sh KIT_DIR: kernel-fixes' TP8 A/B of the vendor MoE op's chunk_size on the integration build, as ONE
# all-card farm job of the integration track (its image holds the build's plugin code; this script only drives
# the integration kit's headline_job.sh, which must sit in KIT_DIR next to its serve_probe.sh and decode_scaling.py):
#
#   farm job integration moe-chunk-ab all 40m -- bash /results/late/kit/moe_chunk_ab.sh /results/late/kit
#
# Two MTP k = 3 servers (headline_job.sh arm mtp3, the build's flags), one after the other:
#   base   no VLLM_MOE_CHUNK: torch.ops.hpu.mixture_of_experts gets no chunk_size (today's form)
#   chunk  VLLM_MOE_CHUNK=64,8192 VLLM_MOE_TOKEN_BOUNDARY=512,1000000: chunk_size 64 up to 512 tokens a step (decode,
#          MTP drafts and verification), 8192 above (prefill chunks: a chunk at least the token count behaved like no
#          chunk_size at 32 and 128 rows on one card)
# Each server: 32 streams greedy (a warm round, 2 measured), then 1 stream (warm, 2 measured: the text to compare),
# then one 1-stream round with a 16,000-token prompt (time to first token) at the very end.
# Why (1-card jobs 00141 and 00149, HPU-graph replay of 16 layers with their own experts): chunk_size 64 cuts the op
# per layer at 128 rows from about 1,200 to 670 us at TP8 geometry (64 local experts) and from 2,270 to 1,110 us at
# TP4, at 512 rows from 1,590 to 910 us (TP8); at 32 rows it changes nothing (608 against 634 us).
# Results: $FARM_OUT/base/ and $FARM_OUT/chunk/ (each headline_job.sh's own layout and headline.md), and the
# COMPARE lines at the end (32-stream ms per token, accepted tokens per step, 1-stream ms per token, TTFT of the 16K
# round, the 1-stream text).
set -u
KIT="${1:-/results/late/kit}"
OUT="${FARM_OUT:-/results/moe-chunk-ab}"
PYTHON="${FARM_PYTHON:-python3}"
[ -f "$KIT/headline_job.sh" ] || { echo "moe_chunk_ab: no headline_job.sh in $KIT" >&2; exit 2; }
export HEADLINE_LEVELS=32,1 HEADLINE_ROUNDS=2 HEADLINE_MTP_CTX=1:16000 HEADLINE_MTP_FALLBACK=0
unset VLLM_MOE_CHUNK VLLM_MOE_TOKEN_BOUNDARY

# vLLM processes left in this container (own PID namespace): the same test as headline_job.sh's.
workers_left() {
    local n=0 pid cmd
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)" || continue
        case "$cmd" in VLLM::* | *"vllm serve"*) n=$((n + 1)) ;; esac
    done
    echo "$n"
}

clear_cards() {
    local i used
    if [ "$(workers_left)" -gt 0 ]; then
        pkill -9 -f "VLLM::" 2>/dev/null
        pkill -9 -f "vllm serve" 2>/dev/null
    fi
    for ((i = 0; i < 60; i++)); do [ "$(workers_left)" -eq 0 ] && break; sleep 1; done
    sleep 8
    if command -v hl-smi >/dev/null 2>&1; then
        for ((i = 0; i < 120; i += 3)); do
            used="$(hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null |
                awk -F', *' '{ gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }')"
            [ "${used:-0}" -lt 4096 ] && break
            sleep 3
        done
    fi
    echo "moe_chunk_ab: cards cleared ($(workers_left) vLLM processes left) $(date -u +%T)"
}

echo "=== moe_chunk_ab: arm base (no VLLM_MOE_CHUNK) $(date -u +%T)"
FARM_OUT="$OUT/base" bash "$KIT/headline_job.sh" mtp3
rc_base=$?
clear_cards
echo "=== moe_chunk_ab: arm chunk (VLLM_MOE_CHUNK=64,8192 VLLM_MOE_TOKEN_BOUNDARY=512,1000000) $(date -u +%T)"
VLLM_MOE_CHUNK=64,8192 VLLM_MOE_TOKEN_BOUNDARY=512,1000000 FARM_OUT="$OUT/chunk" bash "$KIT/headline_job.sh" mtp3
rc_chunk=$?
echo "moe_chunk_ab: exit codes base $rc_base, chunk $rc_chunk $(date -u +%T)"
for arm in base chunk; do
    grep -rh "moe_chunk\|VLLM_MOE_CHUNK\|moe_token_boundary" "$OUT/$arm" 2>/dev/null | head -n 3 | sed "s/^/moe_chunk_ab: $arm: /"
done
"$PYTHON" - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])


def rounds(arm):
    found = []
    for path in sorted((out / arm).rglob("scaling.json")):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for item in data if isinstance(data, list) else data.get("rounds", data.get("results", [])):
            if isinstance(item, dict):
                found.append(item)
    return found


for arm in ("base", "chunk"):
    for r in rounds(arm):
        keys = {k: r.get(k) for k in ("label", "n", "streams", "ms_per_token_p50", "steady_ms_per_token_p50",
                                      "ttft_p50", "tokens_per_chunk", "ok") if k in r}
        print(f"COMPARE {arm}: {json.dumps(keys)}")
PY
exit $((rc_base + rc_chunk))
