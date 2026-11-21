#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# run_bench.sh: native-replay's bench_replay.py in a farm job's container on all of its cards (it starts its own
# rank processes), the bridge's prepared-replay flag off and on as two processes of one run (paired), so one arm of
# ab_bridge_job.sh --cmd 'bash /results/bench/run_bench.sh' gives the launch and collective host cost under the
# bridge that arm installed. The bench file is staged at /results/bench/bench_replay.py (BENCH_FILE) at submit
# time. Writes $FARM_OUT/bench_r<N>.json and .log; summary on the workstation:
#   python3 reports/native-replay/probes/knob_table.py bench_r8.json
set -u
N="${FARM_NCARDS:-8}"
B="${BENCH_FILE:-/results/bench/bench_replay.py}"
[ -f "$B" ] || { echo "run_bench: no $B (stage it before the job)"; exit 2; }
echo "run_bench: $(md5sum "$B" | cut -c1-32) $B, $N ranks, bridge ${BRIDGE_BUILD_ID:-of the image} $(date -u +%T)"
python3 "$B" --processes lazy --ranks "$N" --collectives both --rows "${BENCH_ROWS:-1,32}" \
    --regimes replay,replay_seg --repeats "${BENCH_REPEATS:-40}" --timeout 600 \
    --env-sets "prep:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1" --env-regimes replay,replay_seg --env-timeout 300 \
    --out "$FARM_OUT/bench_r$N.json" >"$FARM_OUT/bench_r$N.log" 2>&1
rc=$?
tail -n 40 "$FARM_OUT/bench_r$N.log" | cut -c1-240
echo "run_bench: rc $rc $(date -u +%T)"
exit $rc
