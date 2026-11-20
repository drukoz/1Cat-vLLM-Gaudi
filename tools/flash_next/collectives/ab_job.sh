#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# ab_job.sh: one arm of a real-model A/B of the collective cuts, as a farm job on the job's cards (TP = card count).
#   farm job collectives real4-ab-off    any:4 22m -- bash tools/flash_next/collectives/ab_job.sh --levels 1,32
#   farm job collectives real4-ab-logits any:4 22m --env VLLM_HPU_VOCAB_PARALLEL_LOGITS=1 \
#        --env VLLM_HPU_PLE_TABLE_RIDER=1 -- bash tools/flash_next/collectives/ab_job.sh --levels 1,32
# The farm's serve_probe.sh (server start, health, stop, summary.json, the program head's decode preset) with this
# track's probe in place of decode_scaling.py: probe_ab.py takes the same arguments, sends no min_tokens (a request
# with min_tokens stays out of the greedy region and so out of VLLM_HPU_VOCAB_PARALLEL_LOGITS) and records the text
# of every stream, so that two arms can be compared token for token (compare_arms.py). Options: serve_probe.sh's.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FARM_TOOLS="${FARM_TOOLS_DIR:-/farm}"
WORK="$(mktemp -d /tmp/ab_job.XXXXXX)" || exit 2
[ -f "$FARM_TOOLS/serve_probe.sh" ] || { echo "ab_job: no $FARM_TOOLS/serve_probe.sh (not a farm job container?)"; exit 2; }
cp "$FARM_TOOLS/serve_probe.sh" "$WORK/serve_probe.sh" && cp "$HERE/probe_ab.py" "$WORK/decode_scaling.py" || exit 2
# One more round at the top level with the server's default sampling (the checkpoint's generation_config: what a
# client that sets no temperature gets); PROBE_SAMPLED_ROUND=0 leaves it out.
export PROBE_SAMPLED_ROUND="${PROBE_SAMPLED_ROUND:-1}"
# Lone requests, one prompt each: their greedy text is the same on every server that computes the same model, so
# they are the token-for-token test of an arm (texts of batched streams change with the batch around them).
export PROBE_SERIAL_PROMPTS="${PROBE_SERIAL_PROMPTS:-6}"
echo "ab_job: collective flags: $(env | grep -E '^VLLM_HPU_(VOCAB_PARALLEL_LOGITS|PLE_TABLE_RIDER|ALL_REDUCE_AS_GATHER_ROWS|REPLICATED_EMBEDDING|MTP_REPLICATED_FC)=' | sort | tr '\n' ' ')"
exec bash "$WORK/serve_probe.sh" "$@"
