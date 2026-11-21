#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# bridge_headline_job.sh LEAN IMAGE [SHA12]: native-replay's prepared-replay build LEAN against the image's own
# bridge, both in the integration build IMAGE (for example farm-integration:d00c0a9a3e18, build 3) with that build's
# env and the accumulation thread off (PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1), probed as the integration headline
# probes (its serve_probe.sh + decode_scaling.py, staged at /results/bench/intprobe/: levels 32, 8, 1, 128 tokens,
# two measured rounds per level, greedy and sampled interleaved). One all-card job of the card queue (host side,
# deployed to ~/farm/collectives/bridge/):
#   tiny-stock  tiny model at TP8, the image's bridge                 (the reference)
#   tiny-lean   LEAN, PT_HPU_HPUGRAPH_PREPARED_REPLAY=1, timing 2000  (must give tiny-stock's tokens)
#   real-stock  the real model, the image's bridge                    (only if both tiny arms passed)
#   real-lean   LEAN, PT_HPU_HPUGRAPH_PREPARED_REPLAY=1, timing 5000
# Exit code 0 when all four passed; verdicts in ~/farm/<track>/results/<job>-<name>-verdicts.txt.
# Late-bound settings, read when the job starts: ~/farm/collectives/bridge/late/<job id>.env with KEY=VALUE lines
#   LEAN=<build id>   another prepared-replay build than the job's command names
#   INLINE=1          two more arms, tiny-lean-inline and real-lean-inline: LEAN with PT_HPU_DISABLE_ASYNC_COLLECTIVE=1
#                     (the stock bridge setting that issues each collective inline instead of through a job thread);
#                     the real one only if its tiny one passed (about 16 more minutes: give the job the hold)
set -u
L="${1:?LEAN build id}"
INLINE="${BRIDGE_INLINE:-0}"
LATE="$HOME/farm/collectives/bridge/late/${CARDQ_JOB:-none}.env"
if [ -s "$LATE" ]; then
    while IFS='=' read -r k v; do
        v="${v//[[:space:]]/}"
        case "$v" in *[!A-Za-z0-9._-]*) echo "bridge_headline: ignored a bad value in $LATE: $k"; continue ;; esac
        case "$k" in LEAN) [ -n "$v" ] && L="$v" ;; INLINE) INLINE="$v" ;; esac
    done <"$LATE"
    echo "bridge_headline: late-bound settings from $LATE: LEAN=$L INLINE=$INLINE"
fi
IMG="${2:?image, e.g. farm-collectives-int:d00c0a9a3e18 (a second tag of integration build 3, kept from pruning)}"
S="${3:-$(basename "$(readlink "$HOME/farm/${CARDQ_TRACK:?}/current")")}"
J="$HOME/farm/collectives/bridge/ab_bridge_job.sh"
V="$HOME/farm/${CARDQ_TRACK:?}/results/${CARDQ_JOB:?}-${CARDQ_NAME:?}-verdicts.txt"
P=/results/bench/intprobe
[ -f "$HOME/farm/$CARDQ_TRACK/results/bench/intprobe/serve_probe.sh" ] || { echo "bridge_headline: no staged probe"; exit 2; }
# the integration headline's no-speculation env (headline_job.sh: the build's flags; job 00178: the prefix flags)
# and the accumulation thread off (coordinator, 03:15 UTC: into build 3's run3.sh)
ENV=(--env VLLM_HPU_QSA_FUSED=1 --env VLLM_HPU_QSA_INDEXER_SHARD=1 --env VLLM_HPU_GDN_KERNEL_INTERLEAVE=1
    --env VLLM_HPU_GDN_GLUE_CACHE=1 --env VLLM_HPU_HC_FUSED_COPY_MODE=2 --env VLLM_HPU_DECODE_CUT_INTERVAL=1000
    --env VLLM_HPU_CANDIDATE_SAMPLING=1 --env VLLM_GRAPH_RESERVED_MEM=0.4 --env VLLM_HPU_PREFIX_ALIGN_SPLIT=1
    --env VLLM_HPU_PREFIX_KEEP_PROMPT_END=1 --env PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1)
TINY=("tiny-stock:-" "tiny-lean:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1,PT_HPU_HPUGRAPH_REPLAY_TIMING=2000")
[ "$INLINE" = 1 ] && TINY+=("tiny-lean-inline:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1,PT_HPU_HPUGRAPH_REPLAY_TIMING=2000,PT_HPU_DISABLE_ASYNC_COLLECTIVE=1")
bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd 'python /farm/farm_tiny.py --gdn hc' "${TINY[@]}"
if ! grep -q '^tiny-stock ok' "$V" || ! grep -q '^tiny-lean ok' "$V"; then
    echo "bridge_headline: a tiny arm failed: no real arm"
    cat "$V"
    exit 3
fi
REAL=("real-stock:-" "real-lean:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1,PT_HPU_HPUGRAPH_REPLAY_TIMING=5000")
if [ "$INLINE" = 1 ] && grep -q '^tiny-lean-inline ok' "$V"; then
    REAL+=("real-lean-inline:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1,PT_HPU_HPUGRAPH_REPLAY_TIMING=5000,PT_HPU_DISABLE_ASYNC_COLLECTIVE=1")
fi
bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" \
    --cmd "bash $P/serve_probe.sh --levels 32,8,1 --tokens 128 --warm-tokens 48 --repeat-top 2 --no-mintok-check --probe-arg --sampled-levels=32,8,1 --probe-arg --interleave-sampled --probe-arg --rounds=2" \
    "${REAL[@]}"
echo "bridge_headline: verdicts:"
cat "$V"
! grep -q ' fail' "$V"
