#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# bridge_reuse_job.sh IMAGE PROBE_DIR LEAN4 [SHA12]: the qualification of native-replay's output-buffer reuse
# (PT_HPU_HPUGRAPH_PREPARED_REUSE, with PT_HPU_HPUGRAPH_PREPARED_REPLAY=1) in an integration build's image (IMAGE) with
# that build's no-speculation env (BRIDGE_ENV="K=V K=V ...", the job's environment, without the accumulation-thread
# setting). One all-card job of the card queue (host side, deployed to ~/farm/collectives/bridge/):
#   tiny-stock      tiny model at TP8, the image's bridge: the reference
#   tiny-lean4-nan  LEAN4, REUSE=2 (reuse + a NaN fill of every reused buffer before its launch), timing 2000
#   tiny-lean4      LEAN4, REUSE=1                                                                 (both: tokens = reference)
#   real-ref        LEAN4, REUSE=0 (exactly lean3a), timing 5000            (if tiny-stock passed)
#   real-lean4-nan  LEAN4, REUSE=2                                            (if tiny-lean4-nan passed)
#   real-lean4      LEAN4, REUSE=1                                            (if tiny-lean4 passed)
# Each real arm runs PROBE_DIR's serve_probe.sh, whose decode_scaling.py (decode_scaling_textcheck.py) runs the build's
# probe and then text_check.py at 1 stream (6 prompts) and 32 streams, 600 tokens each (texts_1.json, texts_32.json).
# Afterwards: texts_1 of real-lean4-nan and real-lean4 must equal real-ref's (text_check.py --compare), and no
# texts_32 may show NaN garbage (replacement characters above 0.5 %, or a run of 40 or more of one character).
# Settings (the job's environment; ~/farm/collectives/bridge/late/<job id>.env with KEY=VALUE lines wins, read when
# the job starts): SPIN=on|off (BRIDGE_SPIN, default off: PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1 in every arm),
# PIN_KV (BRIDGE_PIN_KV, ';'-separated env of every lean real arm, e.g. VLLM_HPU_WORKER_CPU_PIN=1), PLACE
# (BRIDGE_PLACE, a placement script inside the container that place_wrap.sh runs in every real arm once its server
# is healthy). Verdicts: ~/farm/<track>/results/<job>-<name>-verdicts.txt; exit code 0 when all passed.
set -u
IMG="${1:?image}"
P="${2:?probe dir inside the container, e.g. /results/bench/intprobe4r}"
L="${3:?LEAN4 build id}"
S="${4:-$(basename "$(readlink "$HOME/farm/${CARDQ_TRACK:?}/current")")}"
SPIN="${BRIDGE_SPIN:-off}"
PIN_KV="${BRIDGE_PIN_KV:-}"
PLACE="${BRIDGE_PLACE:-}"
LATE="$HOME/farm/collectives/bridge/late/${CARDQ_JOB:-none}.env"
if [ -s "$LATE" ]; then
    while IFS='=' read -r k v; do
        case "$v" in *[!A-Za-z0-9._,\;=/-]*) echo "bridge_reuse: ignored a bad value in $LATE: $k"; continue ;; esac
        case "$k" in SPIN) SPIN="$v" ;; PIN_KV) PIN_KV="$v" ;; PLACE) PLACE="$v" ;; esac
    done <"$LATE"
    echo "bridge_reuse: late-bound settings from $LATE"
fi
J="$HOME/farm/collectives/bridge/ab_bridge_job.sh"
R="$HOME/farm/${CARDQ_TRACK:?}/results"
PREFIX="${CARDQ_JOB:?}-${CARDQ_NAME:?}"
V="$R/$PREFIX-verdicts.txt"
[ -f "$R/${P#/results/}/text_check.py" ] || { echo "bridge_reuse: no staged probe with text_check.py at $P"; exit 2; }
ENV=()
for kv in ${BRIDGE_ENV:-}; do
    case "$kv" in
        PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=*) echo "bridge_reuse: SPIN decides the accumulation thread"; exit 2 ;;
        [A-Za-z_]*=*) ENV+=(--env "$kv") ;;
        *) echo "bridge_reuse: not KEY=VALUE in BRIDGE_ENV: $kv"; exit 2 ;;
    esac
done
SPINKV=""
[ "$SPIN" = off ] && SPINKV=";PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1"
F1="PT_HPU_HPUGRAPH_PREPARED_REPLAY=1"
echo "bridge_reuse: image $IMG, lean4 $L, spinner $SPIN, pin ${PIN_KV:-none}, placement ${PLACE:-none}"

bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd 'python /farm/farm_tiny.py --gdn hc' \
    "tiny-stock:-:PT_HPU_HPUGRAPH_REPLAY_TIMING=0$SPINKV" \
    "tiny-lean4-nan:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV" \
    "tiny-lean4:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV"
grep -q '^tiny-stock ok' "$V" || { echo "bridge_reuse: the reference tiny arm failed: no real arm"; cat "$V"; exit 3; }

PROBE="bash $P/serve_probe.sh --levels 32,8,1 --tokens 128 --warm-tokens 48 --repeat-top 2 --no-mintok-check --probe-arg --sampled-levels=32,8,1 --probe-arg --interleave-sampled --probe-arg --rounds=2"
CMD="$PROBE"
[ -z "$PLACE" ] || CMD="bash /results/bench/place_wrap.sh $PLACE -- $PROBE"
LEANKV="${PIN_KV:+;$PIN_KV}$SPINKV"
REAL=("real-ref:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=0;PT_HPU_HPUGRAPH_REPLAY_TIMING=5000$LEANKV")
grep -q '^tiny-lean4-nan ok' "$V" && REAL+=("real-lean4-nan:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_REPLAY_TIMING=5000$LEANKV")
grep -q '^tiny-lean4 ok' "$V" && REAL+=("real-lean4:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;PT_HPU_HPUGRAPH_REPLAY_TIMING=5000$LEANKV")
bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd "$CMD" "${REAL[@]}"

# the long texts: texts_1 of the reuse arms against real-ref's; no NaN garbage in any texts_32
TC="$R/${P#/results/}/text_check.py"
REF="$R/$PREFIX-real-ref/texts_1.json"
for arm in real-lean4-nan real-lean4; do
    f="$R/$PREFIX-$arm/texts_1.json"
    if [ -f "$REF" ] && [ -f "$f" ]; then
        python3 "$TC" --compare "$REF" "$f" >"$R/$PREFIX-$arm/texts_1_vs_ref.txt" 2>&1
        crc=$?  # the comparison's own exit code (1: a text differs), not a pipe's
        cut -c1-200 "$R/$PREFIX-$arm/texts_1_vs_ref.txt"
        if [ "$crc" -eq 0 ]; then
            echo "texts1-$arm ok" >>"$V"
        else
            echo "texts1-$arm fail differs from real-ref (rc $crc)" >>"$V"
        fi
    elif grep -q "^$arm " "$V"; then
        echo "texts1-$arm fail no texts_1.json" >>"$V"
    fi
done
python3 - "$R" "$PREFIX" real-ref real-lean4-nan real-lean4 >>"$V" <<'PY'
import json, os, sys
root, prefix, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
for arm in arms:
    path = os.path.join(root, "%s-%s" % (prefix, arm), "texts_32.json")
    if not os.path.exists(path):
        continue
    worst_run, worst_rep = 0, 0.0
    for r in json.load(open(path))["results"]:
        text, run, best, prev = r["text"], 0, 0, None
        for ch in text:
            run = run + 1 if ch == prev else 1
            prev = ch
            best = max(best, run)
        worst_run = max(worst_run, best)
        worst_rep = max(worst_rep, text.count("�") / max(1, len(text)))
    bad = worst_rep > 0.005 or worst_run >= 40
    print("texts32-%s %s longest run %d, replacement share %.4f" % (arm, "fail garbage:" if bad else "ok",
                                                                    worst_run, worst_rep))
PY
echo "bridge_reuse: verdicts:"
cat "$V"
! grep -q ' fail' "$V"
