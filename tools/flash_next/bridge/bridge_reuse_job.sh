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
# texts_32 may show NaN garbage (replacement characters above 0.5 %; a run of 40 or more of one character without
# them is reported as "inspect" with the snippet: it can be a markdown rule or indentation).
# Settings (the job's environment; ~/farm/collectives/bridge/late/<job id>.env with KEY=VALUE lines wins, read when
# the job starts): SPIN=on|off (BRIDGE_SPIN, default off: PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1 in every arm),
# PIN_KV (BRIDGE_PIN_KV, ';'-separated env of every lean real arm, e.g. VLLM_HPU_WORKER_CPU_PIN=1), PLACE
# (BRIDGE_PLACE, a placement script inside the container that place_wrap.sh runs in every real arm once its server
# is healthy), ARMSET=lean4|lean5 (BRIDGE_ARMSET, default lean4) and LEAN5 (BRIDGE_LEAN5, a build id): with
# ARMSET=lean5 every lean arm runs on that wheel (all flags off = lean3a, PREPARED_REPLAY=1 + REUSE = lean4, + STATIC_PLAN
# = lean5): tiny-stock, tiny-lean4-nan, tiny-lean5-verify (REUSE=2, STATIC_PLAN=2, BUMP_EVERY=500), then real-ref (all
# flags off), real-lean5 (REUSE=1, STATIC_PLAN=1, inline collectives), real-lean5-verify (REUSE=2, STATIC_PLAN=2,
# BUMP_EVERY=5000, inline collectives), real-lean4 (REUSE=1), most important first; the lean5 arms also run
# text_check --churn, whose texts must equal their own texts_1 and real-ref's, and their plan_bad must be 0.
# REF_PAIR=1 (BRIDGE_REF_PAIR) adds real-ref-alt right after real-ref: the same arm with the other accumulation-thread
# setting (two servers that differ only in the spinner; its texts_1 must equal real-ref's); LEAN4=0 (BRIDGE_LEAN4)
# leaves real-lean4 out (the hold).
# Verdicts: ~/farm/<track>/results/<job>-<name>-verdicts.txt; exit code 0 when all passed.
set -u
IMG="${1:?image}"
P="${2:?probe dir inside the container, e.g. /results/bench/intprobe4r}"
L="${3:?LEAN4 build id}"
S="${4:-$(basename "$(readlink "$HOME/farm/${CARDQ_TRACK:?}/current")")}"
SPIN="${BRIDGE_SPIN:-off}"
PIN_KV="${BRIDGE_PIN_KV:-}"
PLACE="${BRIDGE_PLACE:-}"
ARMSET="${BRIDGE_ARMSET:-lean4}"
L5="${BRIDGE_LEAN5:-}"
REF_PAIR="${BRIDGE_REF_PAIR:-0}"
WITH_LEAN4="${BRIDGE_LEAN4:-1}"
LATE="$HOME/farm/collectives/bridge/late/${CARDQ_JOB:-none}.env"
if [ -s "$LATE" ]; then
    while IFS='=' read -r k v; do
        case "$v" in *[!A-Za-z0-9._,\;=/-]*) echo "bridge_reuse: ignored a bad value in $LATE: $k"; continue ;; esac
        case "$k" in SPIN) SPIN="$v" ;; PIN_KV) PIN_KV="$v" ;; PLACE) PLACE="$v" ;; ARMSET) ARMSET="$v" ;;
            LEAN5) L5="$v" ;; REF_PAIR) REF_PAIR="$v" ;; LEAN4) WITH_LEAN4="$v" ;; esac
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
echo "bridge_reuse: image $IMG, arm set $ARMSET, lean4 $L, lean5 ${L5:-none}, spinner $SPIN, pin ${PIN_KV:-none}, placement ${PLACE:-none}, ref pair $REF_PAIR, lean4 arm $WITH_LEAN4"
PROBE="bash $P/serve_probe.sh --levels 32,8,1 --tokens 128 --warm-tokens 48 --repeat-top 2 --no-mintok-check --probe-arg --sampled-levels=32,8,1 --probe-arg --interleave-sampled --probe-arg --rounds=2"
CMD="$PROBE"
[ -z "$PLACE" ] || CMD="bash /results/bench/place_wrap.sh $PLACE -- $PROBE"
LEANKV="${PIN_KV:+;$PIN_KV}$SPINKV"
T5="PT_HPU_HPUGRAPH_REPLAY_TIMING=5000"
if [ "$ARMSET" = lean5 ]; then
    [ -n "$L5" ] || { echo "bridge_reuse: ARMSET=lean5 needs LEAN5"; exit 2; }
    bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd 'python /farm/farm_tiny.py --gdn hc' \
        "tiny-stock:-:PT_HPU_HPUGRAPH_REPLAY_TIMING=0$SPINKV" \
        "tiny-lean4-nan:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV" \
        "tiny-lean5-verify:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_STATIC_PLAN=2;PT_HPU_HPUGRAPH_STATIC_PLAN_BUMP_EVERY=500;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV"
    grep -q '^tiny-stock ok' "$V" || { echo "bridge_reuse: the reference tiny arm failed: no real arm"; cat "$V"; exit 3; }
    REAL=("real-ref:$L5:$T5$LEANKV")
    if [ "$REF_PAIR" = 1 ]; then  # the other spinner setting, everything else the same
        if [ "$SPIN" = off ]; then ALTKV="${PIN_KV:+;$PIN_KV}"; else ALTKV="${PIN_KV:+;$PIN_KV};PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1"; fi
        REAL+=("real-ref-alt:$L5:$T5$ALTKV")
    fi
    if grep -q '^tiny-lean5-verify ok' "$V"; then
        REAL+=("real-lean5:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;PT_HPU_HPUGRAPH_STATIC_PLAN=1;PT_HPU_DISABLE_ASYNC_COLLECTIVE=1;TEXT_CHECK_CHURN=1;$T5$LEANKV")
        REAL+=("real-lean5-verify:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_STATIC_PLAN=2;PT_HPU_HPUGRAPH_STATIC_PLAN_BUMP_EVERY=5000;PT_HPU_DISABLE_ASYNC_COLLECTIVE=1;TEXT_CHECK_CHURN=1;$T5$LEANKV")
    fi
    if [ "$WITH_LEAN4" = 1 ] && grep -q '^tiny-lean4-nan ok' "$V"; then
        REAL+=("real-lean4:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;$T5$LEANKV")
    fi
    COMPARE=(real-ref-alt real-lean5 real-lean5-verify real-lean4)
    CHURN=(real-lean5 real-lean5-verify)
    T32=(real-ref real-ref-alt real-lean5 real-lean5-verify real-lean4)
    PLAN=(tiny-lean5-verify real-lean5 real-lean5-verify)
else
    bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd 'python /farm/farm_tiny.py --gdn hc' \
        "tiny-stock:-:PT_HPU_HPUGRAPH_REPLAY_TIMING=0$SPINKV" \
        "tiny-lean4-nan:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV" \
        "tiny-lean4:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV"
    grep -q '^tiny-stock ok' "$V" || { echo "bridge_reuse: the reference tiny arm failed: no real arm"; cat "$V"; exit 3; }
    REAL=("real-ref:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=0;$T5$LEANKV")
    grep -q '^tiny-lean4-nan ok' "$V" && REAL+=("real-lean4-nan:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;$T5$LEANKV")
    grep -q '^tiny-lean4 ok' "$V" && REAL+=("real-lean4:$L:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;$T5$LEANKV")
    COMPARE=(real-lean4-nan real-lean4)
    CHURN=()
    T32=(real-ref real-lean4-nan real-lean4)
    PLAN=()
fi
bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd "$CMD" "${REAL[@]}"

# the long texts: texts_1 of the reuse arms against real-ref's; no NaN garbage in any texts_32
TC="$R/${P#/results/}/text_check.py"
REF="$R/$PREFIX-real-ref/texts_1.json"
for arm in "${COMPARE[@]}"; do
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
# the churn round: each prompt alone after bursts and a long prefill must give its plain 1-stream text, the arm's own
# and real-ref's
for arm in ${CHURN[@]+"${CHURN[@]}"}; do
    c="$R/$PREFIX-$arm/texts_churn.json"
    if [ ! -f "$c" ]; then
        grep -q "^$arm " "$V" && echo "churn-$arm fail no texts_churn.json" >>"$V"
        continue
    fi
    for pair in "own:$R/$PREFIX-$arm/texts_1.json" "ref:$REF"; do
        what="${pair%%:*}"
        b="${pair#*:}"
        [ -f "$b" ] || { echo "churn-$arm-$what fail no texts_1.json" >>"$V"; continue; }
        python3 "$TC" --compare "$b" "$c" >"$R/$PREFIX-$arm/texts_churn_vs_$what.txt" 2>&1
        crc=$?
        cut -c1-200 "$R/$PREFIX-$arm/texts_churn_vs_$what.txt"
        if [ "$crc" -eq 0 ]; then echo "churn-$arm-$what ok" >>"$V"; else echo "churn-$arm-$what fail differs (rc $crc)" >>"$V"; fi
    done
done
# the static plans: plan_bad must be 0 in every timing line of the lean5 arms
for arm in ${PLAN[@]+"${PLAN[@]}"}; do
    log="$R/$PREFIX-$arm/server0.log"
    [ -f "$log" ] || log="$R/$PREFIX-$arm/hc/server0.log"
    [ -f "$log" ] || { grep -q "^$arm " "$V" && echo "plan-$arm fail no server0.log" >>"$V"; continue; }
    python3 - "$log" "$arm" >>"$V" <<'PY'
import re, sys
bad, tries, hits, lines = 0.0, [], [], 0
for line in open(sys.argv[1], errors="replace"):
    if "HPUGRAPH_REPLAY_TIMING" not in line:
        continue
    f = dict((k, float(v)) for k, v in re.findall(r"(plan_\w+) (-?\d+(?:\.\d+)?)", line))
    if "plan_try" in f:
        lines += 1
        bad = max(bad, f.get("plan_bad", 0.0))
        tries.append(f["plan_try"])
        hits.append(f.get("plan_hit", 0.0))
share = (sum(hits) / sum(tries)) if sum(tries) else 0.0
if not lines:
    print("plan-%s fail no plan fields in the timing lines" % sys.argv[2])
elif bad > 0:
    print("plan-%s fail plan_bad %.4f per launch (lines %d, plan_hit share %.3f)" % (sys.argv[2], bad, lines, share))
else:
    print("plan-%s ok plan_bad 0 (lines %d, plan_hit share %.3f)" % (sys.argv[2], lines, share))
PY
done
python3 - "$R" "$PREFIX" "${T32[@]}" >>"$V" <<'PY'
import json, os, sys
root, prefix, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
for arm in arms:
    path = os.path.join(root, "%s-%s" % (prefix, arm), "texts_32.json")
    if not os.path.exists(path):
        continue
    worst_run, worst_rep, where = 0, 0.0, ""
    for r in json.load(open(path))["results"]:
        text, run, prev = r["text"], 0, None
        for i, ch in enumerate(text):
            run = run + 1 if ch == prev else 1
            prev = ch
            if run > worst_run:
                worst_run, where = run, text[max(0, i - run - 20):i + 20]
        worst_rep = max(worst_rep, text.count("\ufffd") / max(1, len(text)))
    # NaN garbage shows replacement characters (with one token repeated); a long run alone can be a markdown rule
    # or indentation (native-replay): then "inspect" with the snippet, not a failure
    if worst_rep > 0.005:
        verdict = "fail garbage:"
    elif worst_run >= 40:
        verdict = "inspect (a long run, no replacement characters) %r:" % where
    else:
        verdict = "ok"
    print("texts32-%s %s longest run %d, replacement share %.4f" % (arm, verdict, worst_run, worst_rep))
PY
echo "bridge_reuse: verdicts:"
cat "$V"
! grep -q ' fail' "$V"
