#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# tiny_kvargs_gate.sh: the card check of VLLM_HPU_KV_CACHES_IN_GRAPH_ARGS on the tiny model (one card is enough).
# Two servers in turn on the same cards, the flag at 1 (default) and then at 0, each with MTP k = 3, prefix caching
# and the unmatched-input log (VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS=1). Each server answers mtp-card-ready's
# commit_turns_probe.py: requests whose decode crosses a recurrent block (align-mode commits), new batch shapes
# after them, the same prompt again, 8 crossings at once and a next turn on the prefix cache. The verdict:
#   * the copied-input sets: with 0 no cache tensor is copied before a replay (with 1: every conv / SSM / KV tensor);
#   * the probe's own comparisons (again = first, eight_again = eight) hold on both servers;
#   * every round's tokens are equal across the two servers.
# Output in $FARM_OUT: arg1/, arg0/ (server0.log, turns.json, probe.log) and kvargs_gate.json; exit 0 when it passes.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${FARM_OUT:-/results/tiny-kvargs}"
CKPT="${TINY_CKPT:-/tmp/fn-smoke-ckpt}"
PORT=$(( ${FARM_PORT_BASE:-8000} + 10 ))
TP="${FARM_NCARDS:-1}"
# The tiny model's attention block (the prefix cache's and align mode's unit) is 12800 tokens (its mamba page):
# prompts of 0.8 x that and 2700 decoded tokens cross it; the next turn then hits the cached block (job 00250 with
# 512 crossed nothing and hit nothing).
BLOCK="${KVARGS_BLOCK:-12800}"
DECODE="${KVARGS_DECODE:-2700}"
MAX_LEN="${KVARGS_MAX_LEN:-32768}"
for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_GDN_CORE_VERIFY=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PREFIX_ALIGN_SPLIT=1 VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 \
    VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_HPU_DECODE_CUT_INTERVAL=1000 VLLM_SKIP_WARMUP=true; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done
export VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  # the tiny config says 4096 positions; the gate needs 2.5 attention blocks
mkdir -p "$OUT"
rc_all=0
for FLAG in 1 0; do
    D="$OUT/arg$FLAG"
    mkdir -p "$D"
    echo "=== kvargs gate: VLLM_HPU_KV_CACHES_IN_GRAPH_ARGS=$FLAG $(date -u +%T)"
    VLLM_HPU_KV_CACHES_IN_GRAPH_ARGS=$FLAG setsid vllm serve "$CKPT" --served-model-name tiny \
        --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" --max-num-seqs 8 --language-model-only --trust-remote-code \
        --enable-prefix-caching --enable-prompt-tokens-details --port "$PORT" --host 127.0.0.1 \
        --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' > "$D/server0.log" 2>&1 &
    SP=$!
    healthy=0
    for _ in $(seq 1 300); do
        if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" \
            2>/dev/null; then healthy=1; break; fi
        kill -0 "$SP" 2>/dev/null || break
        sleep 2
    done
    if [ "$healthy" = 1 ]; then
        python3 "$ROOT/tools/flash_next/rental/commit_turns_probe.py" --port "$PORT" --model tiny --block "$BLOCK" \
            --decode "$DECODE" --out "$D/turns.json" > "$D/probe.log" 2>&1
        echo "probe rc $? (arg$FLAG)"
        grep "^ROUND\|^COMPARE\|equal" "$D/probe.log" | head -20
    else
        echo "server of arg$FLAG never healthy"
        rc_all=1
    fi
    kill -TERM -- "-$SP" 2>/dev/null
    for _ in $(seq 1 18); do kill -0 "$SP" 2>/dev/null || break; sleep 5; done
    kill -KILL -- "-$SP" 2>/dev/null
done
python3 - "$OUT" <<'PY'
import json, os, re, sys
out = sys.argv[1]
verdict = {"passed": True, "reasons": []}
def fail(reason):
    verdict["passed"] = False
    verdict["reasons"].append(reason)
CACHE = re.compile(r"\[\d+, \d+, \d+(, \d+)?\] (bfloat16|float32|float8_e4m3fn|float8_e5m2|int8)")
sets = {}
for flag in ("1", "0"):
    path = os.path.join(out, "arg" + flag, "server0.log")
    found = []
    try:
        for line in open(path, errors="replace"):
            m = re.search(r"HPU graph replay copies (\d+) unmatched input tensor\(s\) \(x(\d+)\): (.*)$", line)
            if m:
                tensors = m.group(3).split("; ")
                caches = [t for t in tensors if CACHE.search(t) and int(re.split(r"[,\]]", t.split("[", 2)[2])[0]) > 512]
                found.append({"n": int(m.group(1)), "seen": int(m.group(2)), "caches": len(caches),
                              "shapes": [t.split(" offset")[0] for t in tensors][:8]})
    except OSError:
        fail("no server log for arg" + flag)
    sets[flag] = found
verdict["copied_sets"] = {flag: [{"n": s["n"], "caches": s["caches"], "seen": s["seen"]} for s in v][:40]
                          for flag, v in sets.items()}
if not sets.get("1"):
    fail("arg1: no copied-input line (the log flag did not take?)")
if any(s["caches"] for s in sets.get("0", [])):
    fail("arg0: cache tensors still copied before a replay")
turns = {}
for flag in ("1", "0"):
    try:
        turns[flag] = json.load(open(os.path.join(out, "arg" + flag, "turns.json")))
    except (OSError, ValueError):
        fail("no turns.json for arg" + flag)
for flag, data in turns.items():
    for name, rows in (data.get("comparisons") or {}).items():
        for row in rows if isinstance(rows, list) else [rows]:
            if isinstance(row, dict) and row.get("equal") is False:
                fail("arg%s: %s differs (first difference %s)" % (flag, name, row.get("first_diff")))
    if not all(r.get("ok") for r in data.get("rounds", [])):
        fail("arg%s: a round failed" % flag)
    hits = [x.get("cached_tokens") or 0 for r in data.get("rounds", []) if r.get("round") == "next_turn"
            for x in r.get("results", []) if x]
    verdict.setdefault("next_turn_cached", {})[flag] = hits
    if not any(hits):
        fail("arg%s: the next turn hit no cached block" % flag)
if len(turns) == 2:
    by = [{r["round"]: [x.get("token_ids") for x in r.get("results", []) if x] for r in turns[f].get("rounds", [])}
          for f in ("1", "0")]
    for name in by[0]:
        if by[0][name] != by[1].get(name):
            fail("round %s: tokens differ between arg1 and arg0" % name)
    verdict["rounds"] = sorted(by[0])
json.dump(verdict, open(os.path.join(out, "kvargs_gate.json"), "w"), indent=1)
print("KVARGS GATE %s %s" % ("PASSED" if verdict["passed"] else "FAILED", "; ".join(verdict["reasons"])))
for flag in ("1", "0"):
    sizes = sorted({(s["n"], s["caches"]) for s in sets.get(flag, [])})
    print("arg%s copied sets (tensors, of them caches): %s" % (flag, sizes[:20]))
sys.exit(0 if verdict["passed"] else 1)
PY
rc=$?
[ "$rc_all" = 0 ] || rc=1
exit $rc
