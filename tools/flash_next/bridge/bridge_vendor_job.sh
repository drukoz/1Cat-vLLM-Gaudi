#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# bridge_vendor_job.sh IMAGE PROBE_DIR REBUILD LEAN [SHA12]: in an integration build's image (IMAGE) with that build's
# no-speculation env (BRIDGE_ENV="K=V K=V ...", the job's environment; without the accumulation-thread setting), one
# all-card job of the card queue (host side, deployed to ~/farm/collectives/bridge/), in this order (coordinator,
# 05:00 UTC):
#   tiny gate    tiny model at TP8: tiny-stock (the image's bridge), tiny-rebuild (REBUILD, our unpatched build),
#                tiny-lean (LEAN, PT_HPU_HPUGRAPH_PREPARED_REPLAY=1): tokens against tiny-stock's
#   1. real-spinon / real-spinoff: the vendor's bridge with the accumulation thread (its spinner) on, the default,
#      and off (PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1)
#   2. the spinner setting with the lower sum of the greedy steady ms per step at 32, 8 and 1 streams (each the
#      median of its rounds) is kept for: real-rebuild (REBUILD against the vendor's), then real-lean (LEAN, timing 5000)
# The probe is the build's own pair at PROBE_DIR (e.g. /results/bench/intprobe4): levels 32, 8, 1, 128 tokens, two
# rounds per level, greedy and sampled interleaved. Exit code 0 when every arm that ran passed; verdicts in
# ~/farm/<track>/results/<job>-<name>-verdicts.txt; the spinner decision in <job>-<name>-spinner.txt.
# Settings (job environment; ~/farm/collectives/bridge/late/<job id>.env with KEY=VALUE lines wins, read at the start):
#   BRIDGE_REBUILD_KV / REBUILD_KV  the rebuild arm's env, ';'-separated (default PT_HPU_HPUGRAPH_REPLAY_TIMING=0;
#                                   with REBUILD = a lean build: PT_HPU_HPUGRAPH_PREPARED_REPLAY=0 and timers, our build
#                                   on the stock path); REBUILD overrides the third argument
#   BRIDGE_PIN_KV / PIN_KV          a fifth real arm, real-pin: LEAN, flag 1, timing 5000, plus this env (';'-separated,
#                                   e.g. VLLM_HPU_WORKER_CPU_PIN=1), last
#   BRIDGE_PIN_PLACE / PIN_PLACE    a placement script (a path inside the container, e.g. /results/bench/place.sh)
#                                   that place_wrap.sh runs once the pin arm's server is healthy; the pin arm then
#                                   runs in a call of its own (even with PIN_KV empty)
set -u
IMG="${1:?image}"
P="${2:?probe dir inside the container, e.g. /results/bench/intprobe4}"
RB="${3:?REBUILD build id}"
L="${4:?LEAN build id}"
S="${5:-$(basename "$(readlink "$HOME/farm/${CARDQ_TRACK:?}/current")")}"
RB_KV="${BRIDGE_REBUILD_KV:-PT_HPU_HPUGRAPH_REPLAY_TIMING=0}"
PIN_KV="${BRIDGE_PIN_KV:-}"
PIN_PLACE="${BRIDGE_PIN_PLACE:-}"
LATE="$HOME/farm/collectives/bridge/late/${CARDQ_JOB:-none}.env"
if [ -s "$LATE" ]; then
    while IFS='=' read -r k v; do
        case "$v" in *[!A-Za-z0-9._,\;=/-]*) echo "bridge_vendor: ignored a bad value in $LATE: $k"; continue ;; esac
        case "$k" in REBUILD) [ -n "$v" ] && RB="$v" ;; REBUILD_KV) RB_KV="$v" ;; PIN_KV) PIN_KV="$v" ;;
            PIN_PLACE) PIN_PLACE="$v" ;; esac
    done <"$LATE"
    echo "bridge_vendor: late-bound settings from $LATE"
fi
J="$HOME/farm/collectives/bridge/ab_bridge_job.sh"
R="$HOME/farm/${CARDQ_TRACK:?}/results"
PREFIX="${CARDQ_JOB:?}-${CARDQ_NAME:?}"
V="$R/$PREFIX-verdicts.txt"
[ -f "$R/${P#/results/}/serve_probe.sh" ] || { echo "bridge_vendor: no staged probe at $P"; exit 2; }
ENV=()
for kv in ${BRIDGE_ENV:-}; do
    case "$kv" in
        [A-Za-z_]*=*) ENV+=(--env "$kv") ;;
        *) echo "bridge_vendor: not KEY=VALUE in BRIDGE_ENV: $kv"; exit 2 ;;
    esac
done
case " ${BRIDGE_ENV:-} " in *" PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING="*)
    echo "bridge_vendor: BRIDGE_ENV must not set PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING (the job decides it)"; exit 2 ;;
esac
PROBE="bash $P/serve_probe.sh --levels 32,8,1 --tokens 128 --warm-tokens 48 --repeat-top 2 --no-mintok-check --probe-arg --sampled-levels=32,8,1 --probe-arg --interleave-sampled --probe-arg --rounds=2"
echo "bridge_vendor: image $IMG, rebuild $RB ($RB_KV), lean $L, pin arm ${PIN_KV:-none}, env ${BRIDGE_ENV:-none}"

bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd 'python /farm/farm_tiny.py --gdn hc' \
    "tiny-stock:-" "tiny-rebuild:$RB:$RB_KV" "tiny-lean:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1,PT_HPU_HPUGRAPH_REPLAY_TIMING=2000"
grep -q '^tiny-stock ok' "$V" || { echo "bridge_vendor: the reference tiny arm failed: no real arm"; cat "$V"; exit 3; }

bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd "$PROBE" \
    "real-spinon:-" "real-spinoff:-:PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1"

SPIN="$(python3 - "$R" "$PREFIX" <<'PY'
import json, os, statistics, sys
root, prefix = sys.argv[1], sys.argv[2]
sums = {}
for arm in ("spinon", "spinoff"):
    try:
        d = json.load(open(os.path.join(root, "%s-real-%s" % (prefix, arm), "scaling.json")))
    except Exception as exc:
        print("undecided: no scaling.json for %s (%s)" % (arm, exc), file=sys.stderr)
        sys.exit(0)
    per = {}
    for lv in d.get("levels", []):
        if lv.get("ok") and lv.get("steady_ms_per_token_p50"):
            per.setdefault(lv["streams"], []).append(lv["steady_ms_per_token_p50"] * (lv.get("tokens_per_chunk_p50") or 1.0))
    if sorted(per) != [1, 8, 32]:
        print("undecided: %s has levels %s" % (arm, sorted(per)), file=sys.stderr)
        sys.exit(0)
    meds = {n: statistics.median(v) for n, v in per.items()}
    sums[arm] = (sum(meds.values()), meds)
    print("%s: greedy ms per step %s, sum %.1f" % (arm, {n: round(m, 1) for n, m in sorted(meds.items())},
                                                   sums[arm][0]), file=sys.stderr)
print("off" if sums["spinoff"][0] <= sums["spinon"][0] else "on")
PY
)" 2>"$R/$PREFIX-spinner.txt"
cat "$R/$PREFIX-spinner.txt"
case "$SPIN" in
    off) SPINKV=";PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1" ;;
    *) SPIN="on"; SPINKV="" ;;
esac
echo "bridge_vendor: spinner $SPIN for the rebuild and lean arms" | tee -a "$R/$PREFIX-spinner.txt"

MORE=()
grep -q '^tiny-rebuild ok' "$V" && MORE+=("real-rebuild:$RB:$RB_KV$SPINKV")
if grep -q '^tiny-lean ok' "$V"; then
    MORE+=("real-lean:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1;PT_HPU_HPUGRAPH_REPLAY_TIMING=5000$SPINKV")
    if [ -n "$PIN_KV" ] && [ -z "$PIN_PLACE" ]; then
        MORE+=("real-pin:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1;PT_HPU_HPUGRAPH_REPLAY_TIMING=5000;$PIN_KV$SPINKV")
    fi
fi
[ ${#MORE[@]} -eq 0 ] || bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd "$PROBE" "${MORE[@]}"
if [ -n "$PIN_PLACE" ] && grep -q '^tiny-lean ok' "$V"; then
    bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd "bash /results/bench/place_wrap.sh $PIN_PLACE -- $PROBE" \
        "real-pin:$L:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1;PT_HPU_HPUGRAPH_REPLAY_TIMING=5000${PIN_KV:+;$PIN_KV}$SPINKV"
fi
echo "bridge_vendor: verdicts:"
cat "$V"
! grep -q ' fail' "$V"
