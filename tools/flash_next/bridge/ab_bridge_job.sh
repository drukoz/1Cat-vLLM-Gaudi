#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# ab_bridge_job.sh: the real model with the image's PyTorch bridge against our own builds of it, one server per
# arm, one after another on all of the job's cards. Runs on the node's host inside a job of the card queue
# (CARDQ_* set); each arm is one container started by ~/farm/farm_job.sh (cards, CPUs, ports, labels as for every
# job), probed by the farm's serve_probe.sh + decode_scaling.py. Deployed to ~/farm/collectives/bridge/.
#
#   ab_bridge_job.sh [--sha SHA12] [--cmd 'COMMAND WORDS'] [--stop-on-fail] ARM...
#     ARM = name:BUILD[:KEY=VALUE[,KEY=VALUE...]]; --stop-on-fail: no further arm after one that failed
#     BUILD "-" = the image's bridge; otherwise a build id of ~/farm/native-replay/bridge-build/out/, installed by
#     with_bridge.py at the container's start (exit 7 or 8 and the arm fails if it is not what loads).
#   e.g. stock:-  lean0:lean-0123456789ab:PT_HPU_HPUGRAPH_PREPARED_REPLAY=0
#        lean1:lean-0123456789ab:PT_HPU_HPUGRAPH_PREPARED_REPLAY=1
#
# PROBE_OPTS: serve_probe.sh's options (default: 1 and 32 streams, 256 tokens, two measured rounds at 32, no
# MINTOK round). Every arm also gets VLLM_GRAPH_RESERVED_MEM (default 0.4 at 8 cards, 0.5 below; the 23:50 UTC
# rule) unless its KEY=VALUE list sets it. --cmd runs that command in every arm instead of the server and probe
# (for example 'python /farm/farm_tiny.py --gdn hc': the tiny model at TP = the job's cards; its arms are then
# compared prompt by prompt, and the job fails unless every arm passed with the same tokens). Results: ~/farm/<track>/results/<job>-<name>-<arm>/ (scaling.json,
# server0.log, probe.log, versions.txt, summary.json, bridge.json), and one summary line per arm and level here.
# Exit code 0 when every arm's server answered and its probe ended well. Every call also appends one line per arm,
# "<arm> ok" or "<arm> fail <reason>", to ~/farm/<track>/results/<job>-<name>-verdicts.txt (a tiny arm is ok when
# its smoke passed with the tokens of the call's first tiny arm; a server arm when it ended with exit code 0 and
# left a scaling.json), so that a job script can run a later arm only where an earlier one passed.
set -u
F="$HOME/farm"
TRACK="${CARDQ_TRACK:?not inside a job of the card queue}"
JOB="${CARDQ_JOB:?}"
NAME="${CARDQ_NAME:-bridge-ab}"
SHA=""
CMD=""
STOP_ON_FAIL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sha) SHA="$2"; shift 2 ;;
        --cmd) CMD="$2"; shift 2 ;;
        --stop-on-fail) STOP_ON_FAIL=1; shift ;;
        *) break ;;
    esac
done
[ -n "$SHA" ] || SHA="$(basename "$(readlink "$F/$TRACK/current")")"
[ $# -ge 1 ] || { echo "usage: ab_bridge_job.sh [--sha SHA12] name:BUILD[:K=V,...]..."; exit 2; }
PROBE_OPTS="${PROBE_OPTS:---levels 1,32 --tokens 256 --warm-tokens 64 --repeat-top 2 --no-mintok-check}"
NCARDS="$(echo "${CARDQ_CARDS:-}" | tr ',' '\n' | grep -c .)"
RESERVED="$([ "$NCARDS" -ge 8 ] && echo 0.4 || echo 0.5)"
WRAP='exec python3 "$FARM_OUT/bridge/with_bridge.py" "$FARM_OUT/bridge" -- "$@"'
now() { date -u +%T; }
rc_all=0
RCS="$F/$TRACK/results/$JOB-$NAME-arms.rc"
VERDICTS="$F/$TRACK/results/$JOB-$NAME-verdicts.txt"
echo "ab_bridge: job $JOB, commit $SHA, cards ${CARDQ_CARDS:-?}, arms: $* [$(now)]"
for arm in "$@"; do
    name="${arm%%:*}"
    rest="${arm#*:}"
    build="${rest%%:*}"
    kvs=""
    [ "$rest" != "$build" ] && kvs="${rest#*:}"
    case "$name" in '' | *[!A-Za-z0-9_-]*) echo "ab_bridge: bad arm name '$name'"; exit 2 ;; esac
    run="$JOB-$NAME-$name"
    out="$F/$TRACK/results/$run"
    mkdir -p "$out"
    envs=()
    has_reserved=0
    if [ -n "$kvs" ]; then
        IFS=',' read -r -a pairs <<<"$kvs"
        for kv in "${pairs[@]}"; do
            envs+=(--env "$kv")
            case "$kv" in VLLM_GRAPH_RESERVED_MEM=*) has_reserved=1 ;; esac
        done
    fi
    [ "$has_reserved" -eq 1 ] || [ -n "$CMD" ] || envs+=(--env "VLLM_GRAPH_RESERVED_MEM=$RESERVED")
    wrap=()
    if [ "$build" != "-" ]; then
        src="$F/native-replay/bridge-build/out/$build"
        if [ ! -f "$src/manifest.json" ]; then
            echo "ab_bridge: arm $name: no bridge build $build on the node"
            rc_all=2
            continue
        fi
        mkdir -p "$out/bridge" && cp -r "$src/." "$out/bridge/" && cp "$F/collectives/bridge/with_bridge.py" "$out/bridge/" \
            || { echo "ab_bridge: arm $name: cannot copy build $build"; rc_all=2; continue; }
        wrap=(bash -c "$WRAP" with-bridge)
    fi
    label="$build"
    [ "$build" = "-" ] && label="the image's"
    echo "ab_bridge: arm $name: bridge $label, env ${kvs:-none} [$(now)]"
    if [ -n "$CMD" ]; then
        # shellcheck disable=SC2086  # CMD is a list of words
        bash "$F/farm_job.sh" --sha "$SHA" --run-id "$run" --suffix "$name" ${envs[@]+"${envs[@]}"} -- \
            ${wrap[@]+"${wrap[@]}"} $CMD >"$out/arm.log" 2>&1
    else
        # shellcheck disable=SC2086  # PROBE_OPTS is a list of options
        bash "$F/farm_job.sh" --sha "$SHA" --run-id "$run" --suffix "$name" ${envs[@]+"${envs[@]}"} -- \
            ${wrap[@]+"${wrap[@]}"} bash /farm/serve_probe.sh $PROBE_OPTS >"$out/arm.log" 2>&1
    fi
    rc=$?
    grep -E "^BRIDGE |serve_probe: (SERVER-|PROBE-RC|DONE|fallback)|^MEAS|^SAMPLED|^TEXT|tiny smoke|decode_ms" "$out/arm.log" \
        | cut -c1-220 | head -40
    echo "ab_bridge: arm $name ended rc $rc [$(now)]"
    printf '%s %s\n' "$name" "$rc" >>"$RCS"
    [ "$rc" -eq 0 ] || rc_all=$rc
    sleep "${CARD_RELEASE_S:-15}"  # the cards are not free the moment the server's processes end
    if [ "$rc" -ne 0 ] && [ "$STOP_ON_FAIL" -eq 1 ]; then
        echo "ab_bridge: --stop-on-fail: no further arm"
        break
    fi
done
# one line per arm and level: the steady ms per token (median gap between chunks) of the measured rounds; with
# --cmd and the tiny smoke: each arm's result and its tokens against the first arm's, prompt by prompt
python3 - "$F/$TRACK/results" "$JOB-$NAME" "$RCS" "$VERDICTS" "$@" <<'PY' || rc_all=$?
import json, sys, os, glob
root, prefix, rcs_path, verdicts_path, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
tiny, first, bad = {}, None, False
rcs = {}
if os.path.exists(rcs_path):
    for line in open(rcs_path):
        if line.strip():
            k, v = line.split()[:2]
            rcs[k] = v
verdicts = []
for arm in arms:
    name = arm.split(":")[0]
    folder = os.path.join(root, "%s-%s" % (prefix, name))
    if name not in rcs:
        verdicts.append("%s fail not run" % name)
        continue
    details = sorted(glob.glob(os.path.join(folder, "*", "details.json")))
    if os.path.exists(os.path.join(folder, "summary.json")) and details:
        s = json.load(open(os.path.join(folder, "summary.json")))
        modes = {}
        for path in details:
            d = json.load(open(path))
            modes[d.get("mode") or os.path.basename(os.path.dirname(path))] = {
                p["index"]: (p.get("match"), p.get("expected"), tuple(p.get("got") or ()), bool(p.get("near_tie")))
                for p in d.get("prompts", [])}
        print("ab_bridge: %-8s tiny smoke passed=%s; %s" % (name, s.get("passed"), "; ".join(
            "%s: %s" % (m, " ".join("%s:%s/%s%s" % (i, v[0], v[1], "~" if v[3] else "") for i, v in sorted(r.items())))
            for m, r in modes.items())))
        bad = bad or not s.get("passed")
        same = True
        if first is None:
            first = (name, modes)
        else:
            same = modes == first[1]
            print("ab_bridge: %-8s tokens %s the %s arm's, prompt by prompt" % (
                name, "EQUAL to" if same else "DIFFER from", first[0]))
            bad = bad or not same
        if not s.get("passed"):
            verdicts.append("%s fail smoke rc %s" % (name, rcs.get(name)))
        elif not same:
            verdicts.append("%s fail tokens differ from %s" % (name, first[0]))
        else:
            verdicts.append("%s ok" % name)
        continue
    if rcs.get(name) != "0":
        verdicts.append("%s fail rc %s" % (name, rcs.get(name)))
    try:
        d = json.load(open(os.path.join(folder, "scaling.json")))
    except Exception as exc:
        print("ab_bridge: %-8s no scaling.json (%s)" % (name, exc))
        if rcs.get(name) == "0":
            verdicts.append("%s fail no scaling.json" % name)
        continue
    if rcs.get(name) == "0":
        verdicts.append("%s ok" % name)
    for lv in d.get("levels", []):
        print("ab_bridge: %-8s N=%-3s rep %s: steady ms/token p50 %s x tokens/chunk %s, ms/token p50 %s, "
              "total tok/s %s, ok %s err %s" % (
                  name, lv.get("streams"), lv.get("rep"), lv.get("steady_ms_per_token_p50"),
                  lv.get("tokens_per_chunk_p50"), lv.get("ms_per_token_p50"), lv.get("aggregate_tok_s"),
                  lv.get("ok"), lv.get("errors")))
with open(verdicts_path, "a") as f:
    for v in verdicts:
        f.write(v + "\n")
        print("ab_bridge: VERDICT " + v)
sys.exit(3 if bad else 0)
PY
echo "ab_bridge: DONE rc $rc_all [$(now)]"
exit "$rc_all"
