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
# Right after each real arm (so the verdicts exist when the hold ends early): its texts_1 must equal real-ref's
# (text_check.py --compare), and its texts_32 must show no NaN garbage (replacement characters above 0.5 %; a run of 40
# or more of one character without them is reported as "inspect" with the snippet: it can be a markdown rule or
# indentation). 32-stream texts are never compared: on one server two greedy rounds of the same 32 prompts gave equal
# texts in only 3 to 5 of 32 streams (qsa-fused, 00184). A real arm that cannot end before CARDQ_DEADLINE (it takes
# the longest real arm so far, at least 20 min, 27 with the pin or the placement: INFERRED from 00189, a pinned server
# loads in 15 min against 8; BRIDGE_ARM_EST_S sets the floor) is skipped ("<arm> skipped hold ...": the exit code is
# then nonzero).
# Settings (the job's environment; ~/farm/collectives/bridge/late/<job id>.env with KEY=VALUE lines wins, read when
# the job starts): SPIN=on|off (BRIDGE_SPIN, default off: PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1 in every arm),
# PIN_KV (BRIDGE_PIN_KV, ';'-separated env of every lean real arm, e.g. VLLM_HPU_WORKER_CPU_PIN=1), PLACE
# (BRIDGE_PLACE, a placement script inside the container that place_wrap.sh runs in every real arm once its server
# is healthy), ARMSET=lean4|lean5 (BRIDGE_ARMSET, default lean4) and LEAN5 (BRIDGE_LEAN5, a build id): with
# ARMSET=lean5 every lean arm runs on that wheel (all flags off = lean3a, PREPARED_REPLAY=1 + REUSE = lean4, + STATIC_PLAN
# = lean5): tiny-stock, tiny-lean4-nan, tiny-lean5-verify (REUSE=2, STATIC_PLAN=2, BUMP_EVERY=500), then real-ref (all
# flags off), real-lean5 (REUSE=1, STATIC_PLAN=1, inline collectives), real-lean5-verify (REUSE=2, STATIC_PLAN=2,
# BUMP_EVERY=5000, inline collectives), real-ref-alt (with REF_PAIR=1), real-lean4 (REUSE=1), most important first.
# real-ref, real-ref-alt and the lean5 arms also run text_check --churn (each prompt alone after a 32-stream burst, a
# 20K prefill and an 8-stream burst: texts_churn.json). Hard: each lean5 arm's churn texts equal real-ref's churn texts,
# request by request, and its plan_bad is 0 in every timing line. Reported only ("report" lines, never a failure): how
# many churn texts equal the arm's own plain 1-stream texts (the prompt's KV can come from the prefix cache, written
# under another batch: real-ref's own count is the noise floor), and how many of real-ref-alt's churn texts equal
# real-ref's (two servers, all flags off: the floor of the hard rule).
# REF_PAIR=1 (BRIDGE_REF_PAIR) adds real-ref-alt: real-ref with the other accumulation-thread setting (two servers that
# differ only in the spinner; its texts_1 must equal real-ref's); LEAN4=0 (BRIDGE_LEAN4) leaves real-lean4 out;
# VERIFY_KV (BRIDGE_VERIFY_KV, ';'-separated) is extra env of real-lean5-verify alone, the arm that is not timed (e.g.
# PT_HPU_HPUGRAPH_STATIC_PLAN_MISS_LOG=200 with native-replay's lean6m wheel: one line per missing launcher).
# Verdicts: ~/farm/<track>/results/<job>-<name>-verdicts.txt; exit code 0 when all passed and no arm was skipped.
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
VERIFY_KV="${BRIDGE_VERIFY_KV:-}"
LATE="$HOME/farm/collectives/bridge/late/${CARDQ_JOB:-none}.env"
if [ -s "$LATE" ]; then
    while IFS='=' read -r k v; do
        case "$v" in *[!A-Za-z0-9._,\;=/-]*) echo "bridge_reuse: ignored a bad value in $LATE: $k"; continue ;; esac
        case "$k" in SPIN) SPIN="$v" ;; PIN_KV) PIN_KV="$v" ;; PLACE) PLACE="$v" ;; ARMSET) ARMSET="$v" ;;
            LEAN5) L5="$v" ;; REF_PAIR) REF_PAIR="$v" ;; LEAN4) WITH_LEAN4="$v" ;; VERIFY_KV) VERIFY_KV="$v" ;; esac
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
echo "bridge_reuse: image $IMG, arm set $ARMSET, lean4 $L, lean5 ${L5:-none}, spinner $SPIN, pin ${PIN_KV:-none}, placement ${PLACE:-none}, ref pair $REF_PAIR, lean4 arm $WITH_LEAN4, verify arm extra ${VERIFY_KV:-none}"
PROBE="bash $P/serve_probe.sh --levels 32,8,1 --tokens 128 --warm-tokens 48 --repeat-top 2 --no-mintok-check --probe-arg --sampled-levels=32,8,1 --probe-arg --interleave-sampled --probe-arg --rounds=2"
CMD="$PROBE"
[ -z "$PLACE" ] || CMD="bash /results/bench/place_wrap.sh $PLACE -- $PROBE"
LEANKV="${PIN_KV:+;$PIN_KV}$SPINKV"
T5="PT_HPU_HPUGRAPH_REPLAY_TIMING=5000"
CH="TEXT_CHECK_CHURN=1"
TC="$R/${P#/results/}/text_check.py"
REF="$R/$PREFIX-real-ref/texts_1.json"
RCH="$R/$PREFIX-real-ref/texts_churn.json"

in_list() {  # in_list X ITEM...: X is one of the items
    local x="$1" y
    shift
    for y in "$@"; do [ "$x" = "$y" ] && return 0; done
    return 1
}

cmp_texts() {  # cmp_texts A B OUT: text_check --compare A B into OUT; CRC = its exit code, EQ = "K of N" equal
    python3 "$TC" --compare "$1" "$2" >"$3" 2>&1
    CRC=$?  # the comparison's own exit code (1: a text differs), not a pipe's
    cut -c1-200 "$3"
    EQ=$(awk '/^text_check compare:/ { n = $5 - $3; print n " of " $5 }' "$3")
    [ -n "$EQ" ] || EQ="unknown"
}

plan_check() {  # plan_bad must be 0 in every timing line of a lean5 arm
    local arm="$1" log="$R/$PREFIX-$1/server0.log"
    [ -f "$log" ] || log="$R/$PREFIX-$arm/hc/server0.log"
    [ -f "$log" ] || { echo "plan-$arm fail no server0.log" >>"$V"; return 0; }
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
}

t32_check() {  # no NaN garbage in the arm's texts_32 (never compared with another arm's)
    local f="$R/$PREFIX-$1/texts_32.json"
    [ -f "$f" ] || { echo "texts32-$1 fail no texts_32.json" >>"$V"; return 0; }
    python3 - "$f" "$1" >>"$V" <<'PY'
import json, sys
worst_run, worst_rep, where = 0, 0.0, ""
for r in json.load(open(sys.argv[1]))["results"]:
    text, run, prev = r["text"], 0, None
    for i, ch in enumerate(text):
        run = run + 1 if ch == prev else 1
        prev = ch
        if run > worst_run:
            worst_run, where = run, text[max(0, i - run - 20):i + 20]
    worst_rep = max(worst_rep, text.count("�") / max(1, len(text)))
# NaN garbage shows replacement characters (with one token repeated); a long run alone can be a markdown rule
# or indentation (native-replay): then "inspect" with the snippet, not a failure
if worst_rep > 0.005:
    verdict = "fail garbage:"
elif worst_run >= 40:
    verdict = "inspect (a long run, no replacement characters) %r:" % where
else:
    verdict = "ok"
print("texts32-%s %s longest run %d, replacement share %.4f" % (sys.argv[2], verdict, worst_run, worst_rep))
PY
}

check_arm() {  # the checks of one real arm, right after it ended
    local arm="$1" d="$R/$PREFIX-$1" c hard
    if in_list "$arm" ${COMPARE[@]+"${COMPARE[@]}"}; then  # hard: texts_1 = real-ref's
        if [ ! -f "$REF" ]; then
            echo "texts1-$arm fail no real-ref texts_1.json" >>"$V"
        elif [ ! -f "$d/texts_1.json" ]; then
            echo "texts1-$arm fail no texts_1.json" >>"$V"
        else
            cmp_texts "$REF" "$d/texts_1.json" "$d/texts_1_vs_ref.txt"
            if [ "$CRC" -eq 0 ]; then
                echo "texts1-$arm ok $EQ equal real-ref's"
            else
                echo "texts1-$arm fail $EQ equal real-ref's (rc $CRC)"
            fi >>"$V"
        fi
    fi
    if in_list "$arm" ${CHURN[@]+"${CHURN[@]}"}; then
        c="$d/texts_churn.json"
        hard=0
        in_list "$arm" ${CHURN_HARD[@]+"${CHURN_HARD[@]}"} && hard=1
        if [ ! -f "$c" ]; then
            if [ "$hard" = 1 ]; then echo "churn-$arm fail no texts_churn.json"; else echo "churnown-$arm report no texts_churn.json"; fi >>"$V"
        else
            if [ -f "$d/texts_1.json" ]; then  # reported only: the noise floor of the prefix cache
                cmp_texts "$d/texts_1.json" "$c" "$d/texts_churn_vs_own.txt"
                echo "churnown-$arm report $EQ churn texts equal its plain 1-stream texts" >>"$V"
            fi
            if [ "$arm" != real-ref ]; then
                if [ ! -f "$RCH" ]; then
                    if [ "$hard" = 1 ]; then echo "churn-$arm fail no real-ref texts_churn.json"; else echo "churnfloor-$arm report no real-ref texts_churn.json"; fi >>"$V"
                else
                    cmp_texts "$RCH" "$c" "$d/texts_churn_vs_ref.txt"
                    if [ "$hard" = 0 ]; then
                        echo "churnfloor-$arm report $EQ churn texts equal real-ref's churn texts (two servers, all flags off)"
                    elif [ "$CRC" -eq 0 ]; then
                        echo "churn-$arm ok $EQ churn texts equal real-ref's churn texts"
                    else
                        echo "churn-$arm fail $EQ churn texts equal real-ref's churn texts (rc $CRC)"
                    fi >>"$V"
                fi
            fi
        fi
    fi
    if in_list "$arm" ${PLAN[@]+"${PLAN[@]}"}; then plan_check "$arm"; fi
    if in_list "$arm" ${T32[@]+"${T32[@]}"}; then t32_check "$arm"; fi
    return 0
}

if [ "$ARMSET" = lean5 ]; then
    [ -n "$L5" ] || { echo "bridge_reuse: ARMSET=lean5 needs LEAN5"; exit 2; }
    bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd 'python /farm/farm_tiny.py --gdn hc' \
        "tiny-stock:-:PT_HPU_HPUGRAPH_REPLAY_TIMING=0$SPINKV" \
        "tiny-lean4-nan:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV" \
        "tiny-lean5-verify:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_STATIC_PLAN=2;PT_HPU_HPUGRAPH_STATIC_PLAN_BUMP_EVERY=500;PT_HPU_HPUGRAPH_REPLAY_TIMING=2000$SPINKV"
    if grep -q '^tiny-lean5-verify ' "$V"; then plan_check tiny-lean5-verify; fi
    grep -q '^tiny-stock ok' "$V" || { echo "bridge_reuse: the reference tiny arm failed: no real arm"; cat "$V"; exit 3; }
    REAL=("real-ref:$L5:$CH;$T5$LEANKV")
    if grep -q '^tiny-lean5-verify ok' "$V"; then
        REAL+=("real-lean5:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;PT_HPU_HPUGRAPH_STATIC_PLAN=1;PT_HPU_DISABLE_ASYNC_COLLECTIVE=1;$CH;$T5$LEANKV")
        REAL+=("real-lean5-verify:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=2;PT_HPU_HPUGRAPH_STATIC_PLAN=2;PT_HPU_HPUGRAPH_STATIC_PLAN_BUMP_EVERY=5000;PT_HPU_DISABLE_ASYNC_COLLECTIVE=1;$CH;$T5$LEANKV${VERIFY_KV:+;$VERIFY_KV}")
    fi
    if [ "$REF_PAIR" = 1 ]; then  # real-ref with the other spinner setting, everything else the same
        if [ "$SPIN" = off ]; then ALTKV="${PIN_KV:+;$PIN_KV}"; else ALTKV="${PIN_KV:+;$PIN_KV};PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1"; fi
        REAL+=("real-ref-alt:$L5:$CH;$T5$ALTKV")
    fi
    if [ "$WITH_LEAN4" = 1 ] && grep -q '^tiny-lean4-nan ok' "$V"; then
        REAL+=("real-lean4:$L5:$F1;PT_HPU_HPUGRAPH_PREPARED_REUSE=1;$T5$LEANKV")
    fi
    COMPARE=(real-ref-alt real-lean5 real-lean5-verify real-lean4)
    CHURN=(real-ref real-lean5 real-lean5-verify real-ref-alt)
    CHURN_HARD=(real-lean5 real-lean5-verify)
    T32=(real-ref real-ref-alt real-lean5 real-lean5-verify real-lean4)
    PLAN=(real-lean5 real-lean5-verify)
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
    CHURN_HARD=()
    T32=(real-ref real-lean4-nan real-lean4)
    PLAN=()
fi

DEADLINE="${CARDQ_DEADLINE:-0}"
EST=1200
[ -z "$PIN_KV$PLACE" ] || EST=1620
EST="${BRIDGE_ARM_EST_S:-$EST}"
for spec in "${REAL[@]}"; do
    arm="${spec%%:*}"
    t0=$(date +%s)
    if [ "$DEADLINE" -gt 0 ] && [ $((DEADLINE - t0)) -lt $((EST + 120)) ]; then
        echo "$arm skipped hold: $(((DEADLINE - t0) / 60)) min left, a real arm takes about $(((EST + 59) / 60))" >>"$V"
        echo "bridge_reuse: $arm skipped, $((DEADLINE - t0)) s left of the hold"
        continue
    fi
    bash "$J" --sha "$S" --image "$IMG" "${ENV[@]}" --cmd "$CMD" "$spec"
    took=$(($(date +%s) - t0))
    [ "$took" -le "$EST" ] || EST="$took"
    check_arm "$arm"
done
echo "bridge_reuse: verdicts:"
cat "$V"
! grep -qE ' (fail|skipped)' "$V"
