#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# bridge_phase_job.sh TIMERS LEAN [SHA12]: native-replay's phase-timer build and its prepared-replay build in one
# all-card job of the card queue (host side, deployed to ~/farm/collectives/bridge/), six arms through
# ab_bridge_job.sh, the most important first:
#   tiny-timers   tiny model (TP = cards), TIMERS, PT_HPU_HPUGRAPH_REPLAY_TIMING=2000: the reference
#   tiny-lean     LEAN with PT_HPU_HPUGRAPH_PREPARED_REPLAY=1: must pass with tiny-timers' tokens
#   tiny-accsync  TIMERS with PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1 (no accumulation thread, no spinner)
#   real-timers   the real model, 1 and 32 streams, TIMERS, PT_HPU_HPUGRAPH_REPLAY_TIMING=5000
#   real-lean     only if tiny-lean passed
#   real-accsync  only if tiny-accsync passed
# No real arm when tiny-timers fails; no further real arm when real-timers fails. A missing LEAN build only drops
# the lean arms. Verdicts: ~/farm/<track>/results/<job>-<name>-verdicts.txt. Exit code 0 when all six passed.
set -u
T="${1:?TIMERS build id}"
L="${2:?LEAN build id}"
S="${3:-$(basename "$(readlink "$HOME/farm/${CARDQ_TRACK:?}/current")")}"
J="$HOME/farm/collectives/bridge/ab_bridge_job.sh"
V="$HOME/farm/${CARDQ_TRACK:?}/results/${CARDQ_JOB:?}-${CARDQ_NAME:?}-verdicts.txt"
bash "$J" --sha "$S" --cmd 'python /farm/farm_tiny.py --gdn hc' \
    "tiny-timers:$T:PT_HPU_HPUGRAPH_REPLAY_TIMING=2000" \
    "tiny-lean:$L:PT_HPU_HPUGRAPH_REPLAY_TIMING=2000,PT_HPU_HPUGRAPH_PREPARED_REPLAY=1" \
    "tiny-accsync:$T:PT_HPU_HPUGRAPH_REPLAY_TIMING=2000,PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1"
grep -q '^tiny-timers ok' "$V" || { echo "bridge_phase: the reference tiny arm failed: no real arm"; cat "$V"; exit 3; }
bash "$J" --sha "$S" "real-timers:$T:PT_HPU_HPUGRAPH_REPLAY_TIMING=5000" \
    || { echo "bridge_phase: real-timers failed: no further real arm"; cat "$V"; exit 4; }
more=()
if grep -q '^tiny-lean ok' "$V"; then
    more+=("real-lean:$L:PT_HPU_HPUGRAPH_REPLAY_TIMING=5000,PT_HPU_HPUGRAPH_PREPARED_REPLAY=1")
fi
if grep -q '^tiny-accsync ok' "$V"; then
    more+=("real-accsync:$T:PT_HPU_HPUGRAPH_REPLAY_TIMING=5000,PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1")
fi
[ ${#more[@]} -eq 0 ] || bash "$J" --sha "$S" "${more[@]}"
echo "bridge_phase: verdicts:"
cat "$V"
! grep -q ' fail' "$V"
