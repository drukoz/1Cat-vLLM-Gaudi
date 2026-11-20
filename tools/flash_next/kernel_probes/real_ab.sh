#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# real_ab.sh: the arms of a real-model A/B inside ONE farm job, one server after the other on the job's cards.
#
#   farm job kernel-fixes NAME any:4 40m -- bash tools/flash_next/kernel_probes/real_ab.sh \
#       [--tiny [MODE:]K=V,K=V...]... [--levels 1,8,32] [--tokens 128] [--warm-tokens 48] [--repeat-top 2] [ARM...]
#
# ARM is NAME or NAME:K=V,K=V,...  For each arm: the farm's serve_probe.sh, unchanged (its `vllm serve` command and
# its preset environment; the arm's K=V are exported first, so they win), with decode_ab_client.py of this folder as
# its probe client: no min_tokens in any request, and texts.json with what every stream generated. An arm's
# results are in $FARM_OUT/<NAME>/ (server0.log, scaling.json, texts.json, probe.log, summary.json, versions.txt).
#
# --tiny [MODE:]K=V,... (may be given more than once; with no ARM the job is these smokes only): before the arms,
# the farm's tiny smoke in MODE (hc: fused GDN decode + fused hyper-connections; moe, the default: fused GDN decode
# + MoE glue) with these variables added, without a recipe cache (a cached recipe would hide which kernel build a
# flag selects). The smokes are the gate of the arms that set variables: when one fails, those arms are skipped (in
# lazy mode a custom op that is handed a view does not fall back, the graph's launch raises and the step dies: ten
# minutes of loading the real model for nothing) and only the arms without variables run. Results: $FARM_OUT/tiny<N>-<MODE>/.
#
# After the arms: one ARM line per arm (steady ms per token per level, lines of the server log that name a
# fallback), and the texts of every arm against the first arm's (COMPARE lines; only the 1-stream rounds are a
# test). The exit code is the first non-zero one of the arms; 7 when a tiny smoke failed.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FARM_PYTHON:-python3}"
FARM_TOOLS="${FARM_TOOLS:-/farm}"  # the farm's tools as a farm container mounts them
OUT="${FARM_OUT:-/tmp/real-ab}"
PORT_BASE="${FARM_PORT_BASE:-8000}"
LEVELS="1,8,32"
TOKENS=128
WARM=48
REPEAT=2
TINIES=()
ARMS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --tiny) TINIES+=("$2"); shift 2 ;;
        --levels) LEVELS="$2"; shift 2 ;;
        --tokens) TOKENS="$2"; shift 2 ;;
        --warm-tokens) WARM="$2"; shift 2 ;;
        --repeat-top) REPEAT="$2"; shift 2 ;;
        -*) echo "real_ab: unknown option $1" >&2; exit 2 ;;
        *) ARMS+=("$1"); shift ;;
    esac
done
[ $((${#ARMS[@]} + ${#TINIES[@]})) -ge 1 ] || { echo "real_ab: no arm and no --tiny given" >&2; exit 2; }
mkdir -p "$OUT"

# The farm's server script with this folder's client next to it under the name the script runs.
SP=/tmp/kf-serve
mkdir -p "$SP"
cp "$FARM_TOOLS/serve_probe.sh" "$SP/serve_probe.sh" || { echo "real_ab: no $FARM_TOOLS/serve_probe.sh (not a farm container)" >&2; exit 2; }
cp "$HERE/decode_ab_client.py" "$SP/decode_scaling.py"

CHILD=""
on_signal() {
    trap '' INT TERM
    echo "real_ab: interrupted (SIG$1): stopping the running step"
    if [ -n "$CHILD" ]; then
        kill -TERM "$CHILD" 2>/dev/null
        wait "$CHILD" 2>/dev/null
    fi
    exit "$2"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

# run_with ENVLIST COMMAND...: COMMAND with the K=V of ENVLIST (comma separated) exported, in the background, and
# wait for it (bash runs a trap only after the foreground command has ended).
run_with() {
    local envlist="$1" kv
    shift
    (
        IFS=','
        for kv in $envlist; do
            # shellcheck disable=SC2163  # the K=V in the variable is what is exported
            [ -n "$kv" ] && export "$kv"
        done
        unset IFS
        exec "$@"
    ) &
    CHILD=$!
    wait "$CHILD"
    local rc=$?
    CHILD=""
    return "$rc"
}

# cards_free: after a server of this container was stopped, make sure none of its workers still holds a card
# (CARD.md, "two steps in one container on the same card": the next step fails with "Device acquire failed").
cards_free() {
    [ -n "${FARM_JOB:-}" ] || return 0 # only inside a farm container: there every such process is this job's
    pkill -9 -f "VLLM::" 2>/dev/null
    sleep 12
}

RC=0
TINY_OK=1
t=0
for TINY in ${TINIES[@]+"${TINIES[@]}"}; do
    t=$((t + 1))
    TINY_MODE=moe
    case "$TINY" in
        hc:* | moe:* | fused:*)
            TINY_MODE="${TINY%%:*}"
            TINY="${TINY#*:}"
            ;;
    esac
    TINY_OUT="$OUT/tiny$t-$TINY_MODE"
    echo "real_ab: === tiny smoke $t, mode $TINY_MODE, with $TINY, $(date -u +%H:%M:%S) UTC"
    # `farm tiny` runs it with `python`
    run_with "$TINY" "${FARM_PYTHON:-$(command -v python || command -v python3)}" "$FARM_TOOLS/farm_tiny.py" \
        --gdn "$TINY_MODE" --out "$TINY_OUT" --no-recipe-cache
    TINY_RC=$?
    echo "real_ab: tiny smoke $t ($TINY_MODE) exit code $TINY_RC"
    for f in "$TINY_OUT"/*/server0.log; do
        [ -f "$f" ] || continue
        echo "real_ab: tiny smoke $t ($TINY_MODE): $(grep -ciE 'is off for|has no HPU kernel|keeps the separate gate_up|a fused op raised|torch decode path' "$f") fallback line(s) in its server log"
        grep -iE 'is off for|has no HPU kernel|keeps the separate gate_up|a fused op raised|torch decode path' "$f" | head -n 8 | cut -c1-400
    done
    if [ "$TINY_RC" -ne 0 ]; then
        TINY_OK=0
        if [ "$RC" -eq 0 ]; then RC=7; fi
        echo "real_ab: tiny smoke $t ($TINY_MODE) FAILED: the arms that set variables are skipped"
        grep -hE "Error|Traceback|Empty tensor optional" "$TINY_OUT"/*/server0.log 2>/dev/null | head -n 12 | cut -c1-400
    fi
    cards_free
done

i=0
NAMES=()
for arm in ${ARMS[@]+"${ARMS[@]}"}; do
    name="${arm%%:*}"
    envlist=""
    case "$arm" in *:*) envlist="${arm#*:}" ;; esac
    case "$name" in '' | *[!A-Za-z0-9._-]*) echo "real_ab: bad arm name '$name'" >&2; exit 2 ;; esac
    port=$((PORT_BASE + 20 + 10 * i))
    i=$((i + 1))
    if [ "$TINY_OK" -eq 0 ] && [ -n "$envlist" ]; then
        echo "real_ab: === arm $name SKIPPED (a tiny smoke failed)"
        continue
    fi
    NAMES+=("$name")
    echo "real_ab: === arm $name (${envlist:-the preset}) on port $port, $(date -u +%H:%M:%S) UTC"
    run_with "$envlist" bash "$SP/serve_probe.sh" --out "$OUT/$name" --port "$port" --levels "$LEVELS" \
        --tokens "$TOKENS" --warm-tokens "$WARM" --repeat-top "$REPEAT" --
    arm_rc=$?
    echo "real_ab: arm $name ended with exit code $arm_rc, $(date -u +%H:%M:%S) UTC"
    if [ "$arm_rc" -ne 0 ] && [ "$RC" -eq 0 ]; then RC="$arm_rc"; fi
    cards_free
done

for name in ${NAMES[@]+"${NAMES[@]}"}; do
    "$PY" - "$OUT/$name" "$name" <<'PYEOF'
import json, os, re, sys
folder, name = sys.argv[1:3]
cells = []
try:
    data = json.load(open(os.path.join(folder, "scaling.json")))
    for level in data.get("levels", []):
        if level.get("ok"):
            cells.append("N=%d rep %d: steady %.2f, span %.2f ms/token, %.1f tok/s total" % (
                level["streams"], level.get("rep", 0), level["steady_ms_per_token_p50"], level["ms_per_token_p50"],
                level.get("aggregate_tok_s") or 0))
        else:
            cells.append("N=%d rep %d: FAILED" % (level["streams"], level.get("rep", 0)))
except Exception as error:  # noqa: BLE001
    cells.append("no scaling.json (%r)" % error)
lines = []
try:
    log = open(os.path.join(folder, "server0.log"), errors="replace").read()
    pattern = re.compile(r"is off for|has no HPU kernel|keeps the separate gate_up|a fused op raised|torch decode path",
                         re.IGNORECASE)
    lines = [line.strip()[:300] for line in log.splitlines() if pattern.search(line)]
except OSError:
    pass
summary = {}
try:
    summary = json.load(open(os.path.join(folder, "summary.json")))
except Exception:  # noqa: BLE001
    pass
print("ARM %s: %s | healthy after %s s | %d fallback line(s)" % (name, "; ".join(cells) or "no measured round",
                                                                 summary.get("healthy_after_s"), len(lines)))
for line in lines[:6]:
    print("    " + line)
PYEOF
done
first="${NAMES[0]:-}"
for name in ${NAMES[@]+"${NAMES[@]:1}"}; do
    if [ -f "$OUT/$first/texts.json" ] && [ -f "$OUT/$name/texts.json" ]; then
        echo "real_ab: texts of $name against $first"
        "$PY" "$HERE/decode_ab_client.py" --compare "$OUT/$first/texts.json" "$OUT/$name/texts.json"
    fi
done
echo "real_ab: DONE (exit code $RC)"
exit "$RC"
