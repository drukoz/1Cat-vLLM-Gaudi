#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# headline_job.sh: the integration card build's headline numbers in ONE all-card farm job (TP8 + EP8):
#
#   farm job integration NAME all 40m [--env K=V]... -- bash tools/flash_next/integration/headline_job.sh [ARM...]
#
# ARMs run one after the other in the order given (default: nospec mtp3):
#   nospec  server 1: the farm preset + the build's flags below, no speculation. Probe: 1, 8 and 32 streams
#           greedy (a warm round before every measured round, two measured rounds at 32), then a SAMPLED round
#           (no temperature in the request: the checkpoint's default sampling) at 1, 8 and 32 streams.
#   mtp3    server 2: the same + MTP k = 3 (--speculative-config method mtp, VLLM_HPU_GDN_CORE_VERIFY=1,
#           VLLM_HPU_MTP_TIMING=1). Probe: 32, 8 and 1 streams greedy (32 first: the number that decides),
#           --no-sampled-check (sampled verification is off in this build: a sampled request gets HTTP 400);
#           acceptance per draft position per level from /metrics (SPEC lines).
#   mtp3-32 server 2 with the 32-stream level only (a short hold).
#   mtp<K>-graph  server 2 with VLLM_HPU_MTP_DRAFT_GRAPH=1 (the k draft passes of a decode step in one HPU graph,
#           track mtp-draft-graph); the plain mtp<K> arm runs with VLLM_HPU_MTP_DRAFT_GRAPH=0 unless the job sets it.
#           The tiny arm takes HEADLINE_TINY_DRAFT_GRAPH (default: the job's VLLM_HPU_MTP_DRAFT_GRAPH, else 0).
#   mtp<K>-switch  ONE MTP server, VLLM_HPU_MTP_DRAFT_GRAPH switched per round through its runtime switch file
#           (VLLM_HPU_RUNTIME_SWITCH_FILE=$OUT/<arm>/switches): per level a warm round per arm, then the
#           measured rounds off, on, off, on (decode_scaling.py --switch-arm): an A/B without the
#           server-to-server spread. Rows of scaling.json carry "arm" (off / on); the MTP timer's records of
#           the one-graph drafts have the kind verify+graph.
#           HEADLINE_SWITCH_SET=graph-md: four arms, the 2 x 2 of VLLM_HPU_MTP_DRAFT_GRAPH and
#           VLLM_HPU_FLASH_NEXT_DECODE_METADATA (off, on, md, on-md).
#   mtp<K>-rfc, mtp<K>-rfc-switch  the MTP server of mtp<K> / mtp<K>-switch built with VLLM_HPU_MTP_REPLICATED_FC=1 (the
#           draft's two fc linears unsharded: two all-gathers fewer per draft pass); other MTP arms take the job's
#           value, else 0. The tiny arm takes HEADLINE_TINY_MTP_REPLICATED_FC and HEADLINE_TINY_DECODE_METADATA
#           (VLLM_HPU_FLASH_NEXT_DECODE_METADATA) the same way as HEADLINE_TINY_DRAFT_GRAPH.
#   mtp<K>-async-switch  the same A/B on the mtp<K>-async server (VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2, the MTP timer
#           at HEADLINE_ASYNC_MTP_TIMING: set 0 for this A/B, the timer's device wait hides async's overlap).
#   mtp<K>-WORD-WORD...  (build 5) any MTP arm is mtp<K> and words in any order: server words async, graph, rfc,
#           noshard, spin, inline, fp8kv, pin3, hpulog; probe words 32, greedy, switch; and the steps after the probe
#           on the same server sw, dg32, cap, kvlp, kvlpfb, long (the comment of the MTP arms below says what each
#           does). E.g. build 5's arms: tiny mtp2-async-sw-cap-kvlp mtp2-async-spin mtp3-async mtp2-kvlpfb
#           mtp2-async-inline-greedy mtp2-async-fp8kv-hpulog-32-greedy-dg32-kvlp-long. Knobs: HEADLINE_TRACE_ENV=1 (every MTP server with
#           step-anatomy's capture setup), HEADLINE_REQUIRE_DENSE=1 (the dense MoE form must run: else exit 8, the
#           job stops), HEADLINE_SW_ARMS / HEADLINE_SW_LEVELS / HEADLINE_SW_ROUNDS, HEADLINE_CAPTURE_LEVELS,
#           HEADLINE_KVLP_LENGTHS / HEADLINE_KVLP_REF, HEADLINE_LONG_CTX / HEADLINE_LONG_STREAMS / HEADLINE_LONG_REPEAT,
#           HEADLINE_INTERLEAVE_SAMPLED=1 (each level's SAMPLED rounds right after its greedy ones).
#   mtp-draft-graph's job on build 5 (coordinator, 09:05 UTC): server word cuts (VLLM_HPU_MTP_REPLICATED_FC=1 and
#           VLLM_HPU_REPLICATED_EMBEDDING=1: the whole token table on every rank), probe word chainsw (the whole probe
#           as an A/B of the chained draft, VLLM_HPU_MTP_DRAFT_CHAIN, on the draft graph and I01: on-md / chain-md
#           (stage A), vlg / chain-vlg (VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1, stage B on the verify graph's ids), all /
#           chain-all (VLLM_HPU_VOCAB_PARALLEL_LOGITS=1, stage B on its ids)), HEADLINE_SWITCH_SET=graph-md-all (the
#           four graph-md arms and "all": draft graph, I01 and VP), the tiny gates tiny-chain, tiny-chain-vlg and
#           tiny-chain-vp (the chain's stages with the draft graph and I01; a failure drops that stage's arms of
#           chainsw, the job goes on; VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS=1 logs the chain graph's copied inputs),
#           the tiny knobs HEADLINE_TINY_REPLICATED_EMBEDDING / _DRAFT_CHAIN / _VOCAB_PARALLEL / _VERIFY_LOGITS_IN_GRAPH,
#           and HEADLINE_PLACE_SWITCH (async-sched's placement switch, read with async-sched's case statement).
#           async-sched's M1 (VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS, the verification inputs made on the device) rides
#           along: tiny-m1 (its tiny gate at 1 with the chain, stage A) and the probe word m1sw (the server starts at 2,
#           the whole probe as an A/B of 0 against 2 with the chain on; skipped when tiny-m1 or tiny-chain failed).
#   nospec-noshard, nospec-noprefix, nospec-noprefix-nocut  the low-stream bisect: A's env with
#           VLLM_HPU_QSA_INDEXER_SHARD=0; A's env without VLLM_HPU_PREFIX_ALIGN_SPLIT and VLLM_HPU_PREFIX_KEEP_PROMPT_END
#           (then also VLLM_HPU_DECODE_CUT_INTERVAL=0); greedy 8 and 1 streams only.
#   nospec-async, mtp<K>-async  the same servers with async scheduling (VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=1 / 2;
#           the MTP timer at HEADLINE_ASYNC_MTP_TIMING, default 2).
#   nospec-pin  the no-speculation server with VLLM_HPU_WORKER_CPU_PIN=1 (HEADLINE_PIN_MODE) and
#           PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1 (native-replay's host levers): greedy 1 and 32 streams.
#   tiny    the tiny model at TP8 with the job's flags (farm_tiny.py --gdn hc; --spec K with HEADLINE_TINY_SPEC=K):
#           tokens against the CPU reference. A failure ends the job (exit 7) unless HEADLINE_TINY_ABORT=0.
#   mtp3-noshard  server 2 with VLLM_HPU_QSA_INDEXER_SHARD=0 (isolates the request-sharded indexer, never run in a
#           served speculative step before this build). Run by itself after mtp3 when mtp3 ended with the server
#           dead (3) or the probe failed (5) and HEADLINE_HOLD_S (default 2400, the job's hold) leaves 900 s or more;
#           HEADLINE_MTP_FALLBACK=0 turns that off.
# Between two servers: the leftover vLLM workers of THIS container are killed (the container has its own
# process namespace) and the script waits until no worker is left, then 8 s more for the cards (CARD.md).
#
# The build's flags (each only when the job did not set it with --env; serve_probe.sh adds the preset
# PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 VLLM_HPU_RECURRENT_STATE_POOL=1
# VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0
# VLLM_SKIP_WARMUP=true):
#   qsa-fused     VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1 (torch data flow, request-sharded indexer)
#   kernel-fixes  VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2
#   native-replay VLLM_HPU_DECODE_CUT_INTERVAL=1000 (decode graphs without the per-layer cuts)
#   engine-step   VLLM_HPU_CANDIDATE_SAMPLING=1 (sampled requests draw from the 64 largest logits; greedy rows and
#                 speculation keep their paths)
#   memory        VLLM_GRAPH_RESERVED_MEM=0.4 (graphs made at run time need room: warmup is skipped)
#   MTP server    VLLM_HPU_GDN_CORE_VERIFY=1 VLLM_HPU_MTP_DRAFT_LOOP=1 VLLM_HPU_MTP_CHUNK_KV=1 VLLM_HPU_MTP_TIMING=1
# Never set here: VLLM_HPU_GRAPH_ASYNC_REPLAY (never faster), HCCL_PRIM_* (HCCL_PRIM_COLLECTIVE_MASK=2 kills
# every rank), VLLM_HPU_MOE_GLUE (no gain), VLLM_HPU_QSA_FUSED=2/3 and VLLM_HPU_QSA_TABLE_WIDTH (not card-proven).
# The card memory of every card is logged every 15 s to $FARM_OUT/hlsmi_mem.log.
#
# Environment knobs: HEADLINE_TOKENS (measured tokens per stream, default 128), HEADLINE_WARM_TOKENS (default 48),
# HEADLINE_ROUNDS (measured rounds per level, greedy and sampled, after the warm round; default 2; 0 = one per level
# and two at the top level, as in job 00128), HEADLINE_LEVELS (default 32,8,1: the 32-stream numbers first),
# HEADLINE_MTP_SAMPLED=1 (the MTP server with VLLM_HPU_SPEC_SAMPLED_VERIFY=1 and a SAMPLED 32-stream level last;
# HEADLINE_MTP_SAMPLED_LEVELS), HEADLINE_MTP_CTX=N:TOKENS (one greedy round of N streams with long prompts on
# the MTP server, at the very end: a long-prompt crash (spec-lookup 00133) must not cost the sampled rounds),
# HEADLINE_TINY_SPEC=K (the tiny gate with --spec K). MTP arms are mtp<K>
# (k = K drafted tokens), mtp<K>-32 (32 streams only) and mtp<K>-noshard.
# Results: $FARM_OUT/<arm>/ (versions.txt, server0.log, scaling.json, probe.log, summary.json) and
# $FARM_OUT/headline.md + headline.json (headline_table.py), written after every arm.
# Exit code: the number of arms whose serve_probe.sh did not end with 0.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../.." && pwd)"
PYTHON="${FARM_PYTHON:-python3}"
OUT="${FARM_OUT:-/results/headline}"
TOKENS="${HEADLINE_TOKENS:-128}"
WARM="${HEADLINE_WARM_TOKENS:-48}"
ROUNDS="${HEADLINE_ROUNDS:-2}"
LEVELS="${HEADLINE_LEVELS:-32,8,1}"
T0="$(date -u +%s)"
HOLD_S="${HEADLINE_HOLD_S:-2400}"
FALLBACK="${HEADLINE_MTP_FALLBACK:-1}"

# The farm preset first (serve_probe.sh sets the same for its servers; the tiny gate needs it here).
for kv in PT_HPU_LAZY_MODE=1 PT_HPU_LAZY_ACC_PAR_MODE=0 VLLM_HPU_FLASH_NEXT_QSA_MODE=1 \
    VLLM_HPU_RECURRENT_STATE_POOL=1 VLLM_HPU_GDN_CORE_DECODE=1 VLLM_HPU_HC_FUSED=1 \
    VLLM_HPU_PLE_HOST_NGRAM_IDS=1 VLLM_HPU_COLLECTIVE_MARK_STEP=0 VLLM_SKIP_WARMUP=true; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done
for kv in VLLM_HPU_QSA_FUSED=1 VLLM_HPU_QSA_INDEXER_SHARD=1 \
    VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 VLLM_HPU_GDN_GLUE_CACHE=1 VLLM_HPU_HC_FUSED_COPY_MODE=2 \
    VLLM_HPU_DECODE_CUT_INTERVAL=1000 VLLM_HPU_CANDIDATE_SAMPLING=1 VLLM_GRAPH_RESERVED_MEM=0.4; do
    key="${kv%%=*}"
    if [ -z "${!key+x}" ]; then export "$kv"; fi
done

# The worker placement, switched late (coordinator, 2026-11-21 08:28 and 08:49 UTC): the first word of
# HEADLINE_PIN_SWITCH (default /results/switches/pin = ~/farm/integration/results/switches/pin on the node), read once
# here at the job's start:
#   place  every real-model server runs under collectives' place_wrap.sh with native-replay's place.sh (rule a^1 of
#          job 00193; PLACE_SIBLING_RULE): once the server is healthy, each worker's main thread gets a vCPU of its
#          own with its hyperthread sibling kept free, re-applied every 2 s; NO VLLM_HPU_WORKER_CPU_PIN, so the load
#          and warm-up keep every CPU (00189: the pin flag doubled the time to the first throughput line). The
#          placement's output: <arm>/placement.log. The scripts: /results/late/place/{place_wrap.sh,place.sh} when
#          both exist (late-bound), else this kit's place/ copies (place_wrap.sh md5 7b159d11 = collectives' 98213f8c,
#          which also widens its mask to the cgroup cpuset; place.sh 44273de4 = native-replay ba96f64b, which keeps
#          shells and the server's ancestors on the whole set);
#          HEADLINE_PLACE_WRAP / HEADLINE_PLACE_SCRIPT override. The tiny gate is not placed.
#   pin3   (or "on") every server of the job, the tiny gate too, with VLLM_HPU_WORKER_CPU_PIN=3
#          (native-replay's pin with the model thread's sibling kept free; restricts the load to the worker's share)
#   0..3   VLLM_HPU_WORKER_CPU_PIN=<digit>
#   anything else or no file: no placement. A VLLM_HPU_WORKER_CPU_PIN the job sets itself (--env) wins over the file.
PIN_SWITCH="${HEADLINE_PIN_SWITCH:-/results/switches/pin}"
# HEADLINE_PLACE_SWITCH (mtp-draft-graph's job, coordinator 09:05 UTC): async-sched's switch file instead, read with
# async-sched's case statement: "on" or a rule ("a^1", "a+N/2") = the placement with VLLM_HPU_WORKER_CPU_PIN=1, "place"
# = the placement alone (rule a^1, no pin flag), "off", anything else or no file = no placement. The word is handed to
# the block below as this kit's "place" or "off" (the same scripts: late/place, else this kit's place/ copies).
PLACE_EXTRA_PIN=0
if [ -n "${HEADLINE_PLACE_SWITCH:-}" ]; then
    as_word="$(head -c 64 "$HEADLINE_PLACE_SWITCH" 2>/dev/null | awk 'NR == 1 { print $1 }')"
    echo "headline_job: async-sched's placement switch $HEADLINE_PLACE_SWITCH says '${as_word:-nothing (no file)}'"
    case "$as_word" in
        on) as_word=place; PLACE_SIBLING_RULE="${PLACE_SIBLING_RULE:-a^1}"; PLACE_EXTRA_PIN=1 ;;
        place) PLACE_SIBLING_RULE="${PLACE_SIBLING_RULE:-a^1}" ;;
        "a^1" | "a+N/2") PLACE_SIBLING_RULE="$as_word"; as_word=place; PLACE_EXTRA_PIN=1 ;;
        *) as_word=off ;;
    esac
    PIN_SWITCH="$(mktemp)"
    echo "$as_word" >"$PIN_SWITCH"
fi
pin_word="$(head -c 64 "$PIN_SWITCH" 2>/dev/null | awk 'NR == 1 { print $1 }')"
PLACE_ON=0
if [ -n "${VLLM_HPU_WORKER_CPU_PIN+x}" ]; then
    echo "headline_job: placement from the job's env: VLLM_HPU_WORKER_CPU_PIN=$VLLM_HPU_WORKER_CPU_PIN" \
        "(the switch $PIN_SWITCH is not read)"
else
    case "$pin_word" in
        place) PLACE_ON=1 ;;
        on | pin3) export VLLM_HPU_WORKER_CPU_PIN=3 ;;
        [0-3]) export VLLM_HPU_WORKER_CPU_PIN="$pin_word" ;;
    esac
    if [ "$PLACE_ON" = 1 ]; then
        PLACE_WRAP="${HEADLINE_PLACE_WRAP:-}"
        PLACE_SCRIPT="${HEADLINE_PLACE_SCRIPT:-}"
        if [ -z "$PLACE_WRAP" ] && [ -z "$PLACE_SCRIPT" ] && [ -r /results/late/place/place_wrap.sh ] &&
            [ -r /results/late/place/place.sh ]; then
            PLACE_WRAP=/results/late/place/place_wrap.sh
            PLACE_SCRIPT=/results/late/place/place.sh
        fi
        PLACE_WRAP="${PLACE_WRAP:-$HERE/place/place_wrap.sh}"
        PLACE_SCRIPT="${PLACE_SCRIPT:-$HERE/place/place.sh}"
        export PLACE_SIBLING_RULE="${PLACE_SIBLING_RULE:-a^1}"
        if [ -r "$PLACE_WRAP" ] && [ -r "$PLACE_SCRIPT" ]; then
            echo "headline_job: placement switch $PIN_SWITCH says 'place': every real-model server under" \
                "$PLACE_WRAP $PLACE_SCRIPT (rule $PLACE_SIBLING_RULE, no VLLM_HPU_WORKER_CPU_PIN; md5" \
                "$(md5sum "$PLACE_WRAP" "$PLACE_SCRIPT" 2>/dev/null | awk '{ printf "%s %s ", $2, substr($1, 1, 8) }'))"
        else
            echo "headline_job: placement switch says 'place' but $PLACE_WRAP or $PLACE_SCRIPT is missing: NO placement"
            PLACE_ON=0
        fi
    else
        echo "headline_job: placement switch $PIN_SWITCH says '${pin_word:-nothing (no file)}':" \
            "VLLM_HPU_WORKER_CPU_PIN=${VLLM_HPU_WORKER_CPU_PIN:-unset (no pin)}, no place.sh"
    fi
fi

if [ "$PLACE_ON" = 1 ] && [ "$PLACE_EXTRA_PIN" = 1 ] && [ -z "${VLLM_HPU_WORKER_CPU_PIN+x}" ]; then
    export VLLM_HPU_WORKER_CPU_PIN=1
    echo "headline_job: async-sched's switch: the placement with VLLM_HPU_WORKER_CPU_PIN=1"
fi

if [ $# -eq 0 ]; then set -- nospec mtp3; fi
# Optional arms, switched late: an arm written if-NAME:ARM runs only when the first word of HEADLINE_SWITCH_DIR/NAME
# (default /results/switches/NAME = ~/farm/<track>/results/switches/NAME on the node) is "on", one written
# ifnot-NAME:ARM only when it is not; each file is read once, here at the job's start (coordinator, build 5: the
# stage-submission arms in place of the inline-collective arm).
SWITCH_DIR="${HEADLINE_SWITCH_DIR:-/results/switches}"
kept_arms=()
for a in "$@"; do
    case "$a" in
        if-*:* | ifnot-*:*)
            opt_kind="${a%%-*}"
            opt_name="${a%%:*}"
            opt_name="${opt_name#*-}"
            opt_word="$(head -c 64 "$SWITCH_DIR/$opt_name" 2>/dev/null | awk 'NR == 1 { print $1 }')"
            if { [ "$opt_kind" = if ] && [ "$opt_word" = on ]; } || { [ "$opt_kind" = ifnot ] && [ "$opt_word" != on ]; }; then
                kept_arms+=("${a#*:}")
                echo "headline_job: optional arm ${a#*:} runs ($a; $SWITCH_DIR/$opt_name says '${opt_word:-nothing (no file)}')"
            else
                echo "headline_job: optional arm ${a#*:} skipped ($a; $SWITCH_DIR/$opt_name says '${opt_word:-nothing (no file)}')"
            fi
            ;;
        *) kept_arms+=("$a") ;;
    esac
done
set -- ${kept_arms[@]+"${kept_arms[@]}"}
mkdir -p "$OUT"

# The card memory every 15 s (coordinator 23:50 UTC: graphs made at run time can run out of device memory).
MEMLOG_PID=""
if command -v hl-smi >/dev/null 2>&1; then
    (
        while :; do
            echo "$(date -u +%T) $(hl-smi -Q index,memory.used,memory.total -f csv,noheader 2>&1 | tr '\n' ';')"
            sleep 15
        done
    ) >>"$OUT/hlsmi_mem.log" 2>&1 &
    MEMLOG_PID=$!
fi
stop_memlog() { if [ -n "$MEMLOG_PID" ]; then kill "$MEMLOG_PID" 2>/dev/null; MEMLOG_PID=""; fi; }
trap 'stop_memlog' EXIT
echo "headline_job: arms $*; tokens $TOKENS (warm $WARM); out $OUT; $(date -u +%T)"
env | grep -E '^(VLLM_|PT_HPU|HABANA_VISIBLE|FARM_)' | sort | sed 's/^/headline_job: env /'

# workers_left: how many vLLM processes are left in this container (their titles start with VLLM::, the API
# server's command line names vllm serve).
workers_left() {
    local n=0 pid cmd
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null)" || continue
        case "$cmd" in
            VLLM::* | *"vllm serve"* | *"bin/vllm serve"*) n=$((n + 1)) ;;
        esac
    done
    echo "$n"
}

# clear_cards: after a server stop. Kills only processes of this container (own PID namespace).
clear_cards() {
    local i left
    left="$(workers_left)"
    echo "headline_job: after the server stop: $left vLLM processes left in this container"
    if [ "$left" -gt 0 ]; then
        pkill -9 -f "VLLM::" 2>/dev/null
        pkill -9 -f "vllm serve" 2>/dev/null
    fi
    for ((i = 0; i < 60; i++)); do
        [ "$(workers_left)" -eq 0 ] && break
        sleep 1
    done
    echo "headline_job: $(workers_left) left after ${i} s; waiting 8 s for the cards"
    sleep 8
    # The card memory of a stopped server is released some seconds after its processes are gone (00128: 65 GB per
    # card 10 s after the stop, 0.8 GB 25 s after): wait until every card holds less than 4 GB, at most 120 s.
    if command -v hl-smi >/dev/null 2>&1; then
        for ((i = 0; i < 120; i += 3)); do
            used="$(hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null |
                awk -F', *' '{ gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0 } END { print m + 0 }')"
            [ "${used:-0}" -lt 4096 ] && break
            sleep 3
        done
        hl-smi -Q index,memory.used -f csv,noheader 2>/dev/null | tr '\n' ';' | sed 's/^/headline_job: card memory: /'
        echo " (after ${i} s more)"
    fi
}

# HEADLINE_HEALTH_TIMEOUT (seconds a server may take to become healthy, default 2400) and HEADLINE_PROBE_TIMEOUT
# (seconds per probe request, default 1800): an arm whose server hangs gives up sooner and the next arm runs.
guard_args() {
    echo "--health-timeout ${HEADLINE_HEALTH_TIMEOUT:-2400} --probe-arg --timeout=${HEADLINE_PROBE_TIMEOUT:-1800}"
}

# HEADLINE_BARRIER_DIR: align every probe round with the other servers of the job (layout_2xtp4.sh):
# HEADLINE_BARRIER_N servers, this one named HEADLINE_BARRIER_NAME.
barrier_args() {
    if [ -n "${HEADLINE_BARRIER_DIR:-}" ]; then
        echo "--probe-arg --barrier-dir=$HEADLINE_BARRIER_DIR --probe-arg --barrier-n=${HEADLINE_BARRIER_N:-2}" \
            "--probe-arg --barrier-name=${HEADLINE_BARRIER_NAME:-probe} --probe-arg --barrier-tag=$1"
    fi
}

# The job's CPUs, read once here before any server: native-replay's place.sh moves EVERY process of the container off
# the vCPUs it reserves (each worker's main thread and its sibling: 16 at TP8), this script included, so every later
# server inherited 16 CPUs fewer (job 00222: 144, 128, 112, 96, 80, 64 CPUs over its six servers, 16 to 6 helper CPUs
# per rank). restore_cpus gives this shell its whole set back before each server starts (a no-op where the platform
# has no sched_getaffinity, e.g. a Mac dry run).
JOB_CPUS="$("$PYTHON" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))' 2>/dev/null)"
restore_cpus() {
    [ -n "$JOB_CPUS" ] || return 0
    "$PYTHON" - "$$" "$JOB_CPUS" <<'PY4' || echo "headline_job: could not give this shell its CPUs back"
import os
import sys
pid, cpus = int(sys.argv[1]), {int(c) for c in sys.argv[2].split(",")}
before = os.sched_getaffinity(pid)
if before != cpus:
    os.sched_setaffinity(pid, cpus)
    print("headline_job: this shell's CPUs %d -> %d (given back after the last placement)" % (len(before), len(cpus)))
PY4
}

# placed DIR COMMAND...: one real-model server (COMMAND, a serve_probe.sh run whose out folder is DIR), under the
# placement when the pin switch said "place" (above): place_wrap.sh runs COMMAND and, once the server answers /health
# on the job's port, place.sh with the server's log; its output goes to DIR/placement.log. Every server starts with
# the job's whole CPU set (restore_cpus).
placed() {
    local dir="$1"
    shift
    restore_cpus
    if [ "$PLACE_ON" = 1 ]; then
        mkdir -p "$dir"
        PLACE_ARGS="--log $dir/server0.log" PLACE_LOG="$dir/placement.log" bash "$PLACE_WRAP" "$PLACE_SCRIPT" -- "$@"
    else
        "$@"
    fi
}

# good_kvlp_ref: the job's first kvlp_bf16.json in which every length ran without an error (the kvlp post step of
# an fp8kv arm compares against it); prints nothing when there is none.
good_kvlp_ref() {
    "$PYTHON" - "$OUT" <<'PY3'
import glob, json, os, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*", "kvlp_bf16.json")), key=os.path.getmtime):
    try:
        lengths = json.load(open(path)).get("lengths") or {}
    except (OSError, ValueError):
        continue
    if lengths and all("error" not in r and r.get("mean_teacher_logprob") is not None for r in lengths.values()):
        print(path)
        break
PY3
}

failed=0
first=1
ARMS=("$@")
i_arm=0
while [ "$i_arm" -lt "${#ARMS[@]}" ]; do
    arm="${ARMS[$i_arm]}"
    i_arm=$((i_arm + 1))
    if [ "$first" -eq 0 ]; then clear_cards; fi
    first=0
    echo "=== headline_job: arm $arm start $(date -u +%T)"
    case "$arm" in
        tiny | tiny-chain | tiny-chain-vlg | tiny-chain-vp | tiny-m1 | tiny-m3 | tiny-m3pc1 | tiny-m3pc2 | tiny-il | \
            tiny-shift | tiny-gv | tiny-pi)
            spec=()
            case "${HEADLINE_TINY_SPEC:-0}" in [1-9]) spec=(--spec "$HEADLINE_TINY_SPEC") ;; esac
            tiny_graph="${HEADLINE_TINY_DRAFT_GRAPH:-${VLLM_HPU_MTP_DRAFT_GRAPH:-0}}"
            tiny_md="${HEADLINE_TINY_DECODE_METADATA:-${VLLM_HPU_FLASH_NEXT_DECODE_METADATA:-0}}"
            tiny_chain="${HEADLINE_TINY_DRAFT_CHAIN:-${VLLM_HPU_MTP_DRAFT_CHAIN:-0}}"
            tiny_vp="${HEADLINE_TINY_VOCAB_PARALLEL:-${VLLM_HPU_VOCAB_PARALLEL_LOGITS:-0}}"
            tiny_vlg="${HEADLINE_TINY_VERIFY_LOGITS_IN_GRAPH:-${VLLM_HPU_VERIFY_LOGITS_IN_GRAPH:-0}}"
            tiny_log="${VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS:-0}"
            tiny_m1="${VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS:-0}"
            tiny_async="${VLLM_HPU_FLASH_NEXT_ASYNC_SCHED:-0}"
            tiny_nopc="${FARM_SERVE_NO_PREFIX_CACHING:-0}"
            tiny_align="${VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN:-0}"
            # the chain's gates: the chained draft (stage A; -vlg and -vp: stage B) on the draft graph and I01, with
            # the chain graph's copied inputs logged (a tiny gate is never a measured arm)
            case "$arm" in
                tiny-chain) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=0; tiny_vlg=0; tiny_log=1 ;;
                tiny-chain-vlg) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=0; tiny_vlg=1; tiny_log=1 ;;
                tiny-chain-vp) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=1; tiny_vlg=0; tiny_log=1 ;;
                # async-sched's M1 gate: the device-made inputs made and checked against the host's every step
                tiny-m1) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=0; tiny_vlg=0; tiny_m1=1 ;;
                # async-sched's M2 gate: mode 3 (a chained step issued before the previous one's tokens reach the
                # host) with the device-made inputs, prefix caching off (rental/serve.py's opt-out)
                tiny-m3) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=0; tiny_vlg=0; tiny_m1=2; tiny_async=3
                    tiny_nopc=1 ;;
                # async-sched's M3 gates, prefix caching on: the align mode on the device checked against the host's
                # (mode 2, =1), then used by mode 3 (=2)
                tiny-m3pc1) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=0; tiny_vlg=0; tiny_m1=2; tiny_async=2
                    tiny_nopc=0; tiny_align=1 ;;
                tiny-m3pc2) tiny_graph=1; tiny_md=1; tiny_chain=1; tiny_vp=0; tiny_vlg=0; tiny_m1=2; tiny_async=3
                    tiny_nopc=0; tiny_align=2 ;;
            esac
            # build 6's gates: one new flag each on the job's flags (a failure drops the arms with that word)
            tiny_b6=()
            case "$arm" in
                tiny-il) tiny_b6=(VLLM_HPU_HC_FUSED_IL=1) ;;
                tiny-shift) tiny_b6=(VLLM_HPU_QSA_SLOT_SHIFT=1) ;;
                tiny-gv) tiny_b6=(VLLM_HPU_GDN_VERIFY_VARIANT="${HEADLINE_TINY_GV:-6}") ;;
                tiny-pi) tiny_b6=(VLLM_HPU_PERSISTENT_INPUTS=1) ;;
            esac
            # the MTP server's own flags too (they act only on the speculative mode)
            tiny_sv="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-0}"
            if [ "${HEADLINE_MTP_SAMPLED:-0}" = 1 ]; then tiny_sv="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-1}"; fi
            env ${tiny_b6[@]+"${tiny_b6[@]}"} \
                VLLM_HPU_MTP_DRAFT_LOOP="${VLLM_HPU_MTP_DRAFT_LOOP:-1}" VLLM_HPU_MTP_CHUNK_KV="${VLLM_HPU_MTP_CHUNK_KV:-1}" \
                VLLM_HPU_SPEC_SAMPLED_VERIFY="$tiny_sv" \
                VLLM_HPU_MTP_DRAFT_GRAPH="$tiny_graph" \
                VLLM_HPU_MTP_REPLICATED_FC="${HEADLINE_TINY_MTP_REPLICATED_FC:-${VLLM_HPU_MTP_REPLICATED_FC:-0}}" \
                VLLM_HPU_REPLICATED_EMBEDDING="${HEADLINE_TINY_REPLICATED_EMBEDDING:-${VLLM_HPU_REPLICATED_EMBEDDING:-0}}" \
                VLLM_HPU_FLASH_NEXT_DECODE_METADATA="$tiny_md" VLLM_HPU_MTP_DRAFT_CHAIN="$tiny_chain" \
                VLLM_HPU_VOCAB_PARALLEL_LOGITS="$tiny_vp" VLLM_HPU_VERIFY_LOGITS_IN_GRAPH="$tiny_vlg" \
                VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS="$tiny_log" VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS="$tiny_m1" \
                VLLM_HPU_FLASH_NEXT_ASYNC_SCHED="$tiny_async" FARM_SERVE_NO_PREFIX_CACHING="$tiny_nopc" \
                VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN="$tiny_align" \
                "$PYTHON" /farm/farm_tiny.py --gdn hc ${spec[@]+"${spec[@]}"} --out "$OUT/$arm"
            rc=$?
            if [ "$arm" != tiny ]; then
                # a chain gate: a failure drops that stage's chain arms, the job goes on
                if [ "$rc" -ne 0 ]; then
                    echo "headline_job: $arm failed (rc $rc): its chain arms are left out"
                    case "$arm" in
                        tiny-chain) chain_a_ok=0 ;;
                        tiny-chain-vlg) chain_vlg_ok=0 ;;
                        tiny-chain-vp) chain_vp_ok=0 ;;
                        tiny-m1) m1_ok=0 ;;
                        tiny-m3) m3_ok=0 ;;
                        tiny-m3pc1 | tiny-m3pc2) m3_align_ok=0 ;;
                        tiny-il) il_ok=0 ;;
                        tiny-shift) shift_ok=0 ;;
                        tiny-gv) gv_ok=0 ;;
                        tiny-pi) pi_ok=0 ;;
                    esac
                fi
                if [ "$arm" = tiny-m3 ] && [ "$rc" -eq 0 ]; then
                    # M2's own check: steps were issued before the previous step's tokens reached the host
                    last="$(grep -rhoE "VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3: [0-9]+ steps issued.*" "$OUT/$arm" 2>/dev/null | tail -n 1)"
                    echo "headline_job: tiny-m3: ${last:-no mode-3 summary line}"
                    if [ -z "$last" ] || grep -rqE "ASYNC_SCHED=3 acts as 2|device-made inputs differ" "$OUT/$arm" 2>/dev/null; then
                        echo "headline_job: tiny-m3: mode 3 did not serve the tiny gate"
                        m3_ok=0
                    fi
                fi
                if [ "$arm" = tiny-m3pc1 ] && [ "$rc" -eq 0 ]; then
                    last="$(grep -rhoE "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN=1: [0-9]+ verification steps made.*" "$OUT/$arm" 2>/dev/null | tail -n 1)"
                    echo "headline_job: tiny-m3pc1: ${last:-no M3 summary line}"
                    if [ -z "$last" ] || ! printf '%s' "$last" | grep -q " 0 with differences" ||
                        grep -rqE "align mode differs|device-made inputs differ" "$OUT/$arm" 2>/dev/null; then
                        echo "headline_job: tiny-m3pc1: M3's check did not pass: the m3a arms are left out"
                        m3_align_ok=0
                    fi
                fi
                if [ "$arm" = tiny-m3pc2 ] && [ "$rc" -eq 0 ]; then
                    last="$(grep -rhoE "VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3: [0-9]+ steps issued.*" "$OUT/$arm" 2>/dev/null | tail -n 1)"
                    echo "headline_job: tiny-m3pc2: ${last:-no mode-3 summary line}"
                    if [ -z "$last" ] || grep -rqE "ASYNC_SCHED=3 acts as 2|align mode differs" "$OUT/$arm" 2>/dev/null; then
                        echo "headline_job: tiny-m3pc2: mode 3 with M3 did not serve the tiny gate: the m3a arms are left out"
                        m3_align_ok=0
                    fi
                fi
                if [ "$arm" = tiny-m1 ] && [ "$rc" -eq 0 ]; then
                    # M1's own check: every advanced step checked, no difference (async-sched)
                    last="$(grep -rhoE "VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS=1: [0-9]+ verification steps.*" "$OUT/$arm" 2>/dev/null | tail -n 1)"
                    echo "headline_job: tiny-m1: ${last:-no M1 summary line}"
                    if [ -z "$last" ] || ! printf '%s' "$last" | grep -q " 0 with differences" ||
                        grep -rqE "device-made inputs differ|keeps the host's inputs" "$OUT/$arm" 2>/dev/null; then
                        echo "headline_job: tiny-m1: M1's check did not pass: the m1sw arm is left out"
                        m1_ok=0
                    fi
                fi
                # whether the chain served the tiny's decode steps (the runner logs its stage once), what it copied
                grep -rhoE "chained behind its verification, stage [AB]|keeps the draft after the host read: .*|HPU graph replay copies .*" \
                    "$OUT/$arm" 2>/dev/null | cut -c1-400 | sort | uniq -c | head -20
                want_stage=B
                case "$arm" in tiny-chain | tiny-m1 | tiny-m3 | tiny-m3pc1 | tiny-m3pc2) want_stage=A ;; esac
                case "$arm" in tiny-il | tiny-shift | tiny-gv | tiny-pi) want_stage="" ;; esac
                if [ -n "$want_stage" ] && [ "$rc" -eq 0 ] && ! grep -rqE "chained behind its verification, stage $want_stage" "$OUT/$arm" 2>/dev/null; then
                    echo "headline_job: WARNING: $arm passed but no step was chained in stage $want_stage (the gate did not test it)"
                fi
                echo "=== headline_job: arm $arm rc=$rc end $(date -u +%T)"
                continue
            fi
            if [ "$rc" -ne 0 ] && [ "${HEADLINE_TINY_ABORT:-1}" = 1 ]; then
                echo "headline_job: the tiny gate failed (rc $rc): no real-model server with these flags"
                exit 7
            fi
            echo "=== headline_job: arm tiny rc=$rc end $(date -u +%T)"
            continue
            ;;
        nospec | nospec-async)
            # Levels in HEADLINE_LEVELS order (default 32,8,1: the 32-stream numbers first); each level's sampled
            # rounds right after its greedy rounds. nospec-async: VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=1 (async-sched).
            async_env=()
            if [ "$arm" = nospec-async ]; then async_env=(VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=1); fi
            placed "$OUT/$arm" env ${async_env[@]+"${async_env[@]}"} \
                bash "$HERE/serve_probe.sh" --out "$OUT/$arm" --levels "$LEVELS" --tokens "$TOKENS" \
                --warm-tokens "$WARM" --repeat-top 2 --no-mintok-check --probe-arg "--sampled-levels=$LEVELS" \
                --probe-arg --interleave-sampled --probe-arg "--rounds=$ROUNDS" $(barrier_args "$arm") $(guard_args)
            ;;
        nospec-noshard | nospec-noprefix | nospec-noprefix-nocut)
            # The low-stream bisect (coordinator, build 3): A's env without the QSA request shard; A's env without the
            # prefix flags, then also without the decode cut interval; greedy only, HEADLINE_BISECT_LEVELS (default
            # 8,1), HEADLINE_BISECT_TOKENS (default 96).
            change=()
            case "$arm" in
                nospec-noshard) change=(VLLM_HPU_QSA_INDEXER_SHARD=0) ;;
                nospec-noprefix) change=(VLLM_HPU_PREFIX_ALIGN_SPLIT=0 VLLM_HPU_PREFIX_KEEP_PROMPT_END=0) ;;
                nospec-noprefix-nocut)
                    change=(VLLM_HPU_PREFIX_ALIGN_SPLIT=0 VLLM_HPU_PREFIX_KEEP_PROMPT_END=0 VLLM_HPU_DECODE_CUT_INTERVAL=0) ;;
            esac
            placed "$OUT/$arm" env "${change[@]}" \
                bash "$HERE/serve_probe.sh" --out "$OUT/$arm" --levels "${HEADLINE_BISECT_LEVELS:-8,1}" \
                --tokens "${HEADLINE_BISECT_TOKENS:-96}" --warm-tokens "$WARM" --repeat-top 2 --no-mintok-check \
                --probe-arg --no-sampled-check --probe-arg "--rounds=$ROUNDS"
            ;;
        nospec-pin)
            # B: native-replay's two host levers on A's env (never run at TP8 before): greedy 1 and 32 streams only;
            # its 1-stream text is compared with A's.
            VLLM_HPU_WORKER_CPU_PIN="${HEADLINE_PIN_MODE:-1}" PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1 \
                bash "$HERE/serve_probe.sh" --out "$OUT/$arm" --levels "${HEADLINE_PIN_LEVELS:-1,32}" --tokens "$TOKENS" \
                --warm-tokens "$WARM" --repeat-top 2 --no-mintok-check --probe-arg --no-sampled-check \
                --probe-arg "--rounds=$ROUNDS"
            ;;
        mtp[1-9] | mtp[1-9]-*)
            k="${arm:3:1}"
            # The words after mtp<K>, separated by "-", each one change of the server or of its probe (any order):
            #   server: async (VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2, the MTP timer at HEADLINE_ASYNC_MTP_TIMING), graph
            #     (VLLM_HPU_MTP_DRAFT_GRAPH=1), rfc (VLLM_HPU_MTP_REPLICATED_FC=1), noshard (VLLM_HPU_QSA_INDEXER_SHARD=0,
            #     greedy only), spin (the bridge's spinner ON: PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING removed from the
            #     server's environment), inline (PT_HPU_DISABLE_ASYNC_COLLECTIVE=1: collectives issued inline), fp8kv
            #     (VLLM_HPU_QSA_KV_FP8=1 and --kv-cache-dtype fp8_inc), pin3 (VLLM_HPU_WORKER_CPU_PIN=3), hpulog
            #     (engine-step's copy_to logger: HPU_GRAPH_INPUT_LOG=1 and hpulog/ first on PYTHONPATH; it walks the
            #     forward's inputs at every replay: never on an arm whose speed is compared), ss0
            #     (PT_HPU_ENABLE_STAGE_SUBMISSION=0: the bridge's stage submission off), irlog (LOG_LEVEL_PT_IRGRAPH=1
            #     LOG_LEVEL_PT_LAZY=1 into <arm>/habana_logs: a chatty bridge log, never on a measured arm)
            #   probe: 32 (the 32-stream level only), lv<N>_<N>... (these levels, e.g. lv32_1), greedy (no SAMPLED
            #     rounds), switch (the whole probe as an A/B of
            #     the draft graph through the runtime switch file, HEADLINE_SWITCH_SET)
            #   after the probe, on the same server (serve_probe.sh --post-cmd), in this order: sw (an A/B through the
            #     runtime switch file: HEADLINE_SW_ARMS at HEADLINE_SW_LEVELS, HEADLINE_SW_ROUNDS rounds each, greedy,
            #     into scaling_sw.json; afterwards the file holds the first arm's values again), dg32 (the same with
            #     the draft graph off / on at 32 streams, one round each: scaling_dg32.json), cap (step-anatomy's
            #     device trace: anatomy_probe.py --trace HEADLINE_CAPTURE_LEVELS), kvlp (qsa-fused's teacher-forced
            #     prompt-logprob probe: kvlp_bf16.json, or with fp8kv kvlp_fp8.json compared with the job's first good
            #     kvlp_bf16.json), kvlpfb (kvlp only when the job has no good kvlp_bf16.json yet), long (qsa-fused's
            #     long rounds: the needle at HEADLINE_LONG_CTX tokens, then HEADLINE_LONG_STREAMS streams on its cached
            #     prefix, long.json)
            # HEADLINE_TRACE_ENV=1: every MTP server starts with step-anatomy's capture setup (the torch profiler by
            # --profiler-config into <arm>/trace, HABANA_PROFILE=profile_api_light, the step timer VLLM_HPU_STEP_TIMING=1
            # into <arm>/timing); only an arm with cap captures. HEADLINE_REQUIRE_DENSE=1 (with VLLM_HPU_MOE_DENSE=1):
            # the server log must say at load that the MoE layers keep their stacked experts and, after the first
            # warm round, that a step ran the dense form; else the arm ends with 8 and the job stops (all its arms
            # depend on it).
            draft_graph="${VLLM_HPU_MTP_DRAFT_GRAPH:-0}"
            replicated_fc="${VLLM_HPU_MTP_REPLICATED_FC:-0}"
            shard="$VLLM_HPU_QSA_INDEXER_SHARD"
            timing="${VLLM_HPU_MTP_TIMING:-1}"
            levels="$LEVELS"
            async_env=()
            arm_env=()
            unset_env=()
            serve_extra=()
            switch_env=()
            switch_args=()
            post_args=()
            post_words=()
            whole_switch=0
            switch_set="${HEADLINE_SWITCH_SET:-graph}"
            greedy_only=0
            fp8kv=0
            irlog=0
            bad_word=""
            needs_m3=0
            needs_async3=0
            needs_b6=()
            words="${arm:4}"
            words="${words#-}"
            arm_words=()
            if [ -n "$words" ]; then IFS=- read -r -a arm_words <<<"$words"; fi
            for w in ${arm_words[@]+"${arm_words[@]}"}; do
                case "$w" in
                    async) async_env=(VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2); timing="${HEADLINE_ASYNC_MTP_TIMING:-2}" ;;
                    # async-sched job branch: async3 = stage 2's mode 3 (a chained step issued before the previous
                    # one's tokens reach the host; needs the chain, SPEC_DEVICE_INPUTS=2, I01 and nopc)
                    async3) async_env=(VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=3); timing="${HEADLINE_ASYNC_MTP_TIMING:-2}"
                        needs_async3=1 ;;
                    # async-sched job branch: nopc = prefix caching off (after serve_probe.sh's --enable-prefix-caching)
                    nopc) serve_extra+=(--no-enable-prefix-caching) ;;
                    # async-sched job branch: m3a = stage 2's M3 in use (VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN=2; the
                    # arm is left out when a tiny-m3pc gate failed); turns = commit_turns_probe.py after the probe
                    m3a) arm_env+=(VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_ALIGN=2); needs_m3=1 ;;
                    graph) draft_graph=1 ;;
                    rfc) replicated_fc=1 ;;
                    noshard) shard=0; greedy_only=1 ;;
                    32) levels=32 ;;
                    switch) whole_switch=1 ;;
                    chainsw) whole_switch=1; switch_set=chain ;;
                    m1sw) whole_switch=1; switch_set=m1; arm_env+=(VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS=2) ;;
                    cuts) replicated_fc=1; arm_env+=(VLLM_HPU_REPLICATED_EMBEDDING=1) ;;
                    spin) unset_env+=(-u PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING) ;;
                    inline) arm_env+=(PT_HPU_DISABLE_ASYNC_COLLECTIVE=1) ;;
                    fp8kv) fp8kv=1; arm_env+=(VLLM_HPU_QSA_KV_FP8=1); serve_extra+=(--kv-cache-dtype fp8_inc) ;;
                    pin3) arm_env+=(VLLM_HPU_WORKER_CPU_PIN=3) ;;
                    # build 6: one flag each (a server start; each needs its tiny gate, if run, to have passed)
                    il) arm_env+=(VLLM_HPU_HC_FUSED_IL=1); needs_b6+=(il) ;;
                    shift) arm_env+=(VLLM_HPU_QSA_SLOT_SHIFT=1); needs_b6+=(shift) ;;
                    gv[0-9]) arm_env+=(VLLM_HPU_GDN_VERIFY_VARIANT="${w#gv}"); needs_b6+=(gv) ;;
                    pi) arm_env+=(VLLM_HPU_PERSISTENT_INPUTS=1); needs_b6+=(pi) ;;
                    # build 6: the chain at the server's start (stage A on the draft graph and I01; with vlg stage B on
                    # the verify graph's ids), I01 alone, the verify graph's LM head alone, M1 at start
                    chainon) draft_graph=1; arm_env+=(VLLM_HPU_MTP_DRAFT_CHAIN=1 VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1)
                        needs_b6+=(chain) ;;
                    vlg) arm_env+=(VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1) ;;
                    md) arm_env+=(VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1) ;;
                    m1on) arm_env+=(VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS=2) ;;
                    ss0) arm_env+=(PT_HPU_ENABLE_STAGE_SUBMISSION=0) ;;
                    irlog) irlog=1; arm_env+=(LOG_LEVEL_PT_IRGRAPH=1 LOG_LEVEL_PT_LAZY=1) ;;
                    lv[0-9]*)
                        levels="${w#lv}"
                        levels="${levels//_/,}"
                        case "$levels" in *[!0-9,]*) bad_word="$w" ;; esac
                        ;;
                    hpulog) arm_env+=(HPU_GRAPH_INPUT_LOG=1 "PYTHONPATH=$HERE/hpulog${PYTHONPATH:+:$PYTHONPATH}") ;;
                    greedy) greedy_only=1 ;;
                    sw | dg32 | cap | kvlp | kvlpfb | kvlpc | kvlpfloor | long | turns) post_words+=("$w") ;;
                    *) bad_word="$w" ;;
                esac
            done
            if [ -n "$bad_word" ]; then
                echo "headline_job: arm $arm: unknown word '$bad_word' (see the comment of the MTP arms)"
                failed=$((failed + 1))
                continue
            fi
            if [ "$needs_m3" = 1 ] && [ "${m3_align_ok:-1}" != 1 ]; then
                echo "headline_job: arm $arm left out: a tiny-m3pc gate failed (async-sched's M3)"
                continue
            fi
            if [ "$needs_async3" = 1 ] && [ "${m3_ok:-1}" != 1 ]; then
                echo "headline_job: arm $arm left out: the tiny-m3 gate failed (async-sched's M2, mode 3)"
                continue
            fi
            b6_skip=""
            for g in ${needs_b6[@]+"${needs_b6[@]}"}; do
                case "$g" in
                    il) [ "${il_ok:-1}" = 1 ] || b6_skip="tiny-il" ;;
                    shift) [ "${shift_ok:-1}" = 1 ] || b6_skip="tiny-shift" ;;
                    gv) [ "${gv_ok:-1}" = 1 ] || b6_skip="tiny-gv" ;;
                    pi) [ "${pi_ok:-1}" = 1 ] || b6_skip="tiny-pi" ;;
                    chain) [ "${chain_a_ok:-1}" = 1 ] || b6_skip="tiny-chain" ;;
                esac
            done
            if [ -n "$b6_skip" ]; then
                echo "headline_job: arm $arm left out: its gate $b6_skip failed"
                continue
            fi
            ARM_DIR="${OUT:?}/${arm:?}"
            mkdir -p "$ARM_DIR"
            # irlog: the bridge's debug log of this server in its own folder (the job's HABANA_LOGS stays quiet)
            if [ "$irlog" = 1 ]; then arm_env+=(HABANA_LOGS="$ARM_DIR/habana_logs"); mkdir -p "$ARM_DIR/habana_logs"; fi
            # The runtime switch file (the whole-probe switch arms and sw): a path at the server's start, absent then.
            if [ "$whole_switch" = 1 ] || [[ " ${post_words[*]:-} " == *" sw "* ]] ||
                [[ " ${post_words[*]:-} " == *" dg32 "* ]]; then
                rm -f "$ARM_DIR/switches" "$ARM_DIR/switches.new"
                switch_env=(VLLM_HPU_RUNTIME_SWITCH_FILE="$ARM_DIR/switches")
            fi
            if [ "$whole_switch" = 1 ]; then
                draft_graph=0
                switch_args=(--probe-arg "--switch-file=$ARM_DIR/switches" --probe-arg "--server-log=$ARM_DIR/server0.log")
                # HEADLINE_SWITCH_SET: graph (default) = off / on; graph-md = the 2 x 2 with engine-step's
                # VLLM_HPU_FLASH_NEXT_DECODE_METADATA (the sparse-QSA decode metadata, also switchable)
                case "$switch_set" in
                    chain)
                        # the chained draft on the draft graph and I01, behind each of the three verifications it can
                        # follow; a stage whose tiny gate failed is left out (with its non-chained twin for VLG)
                        g="VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1"
                        switch_args+=(--probe-arg "--switch-arm=on-md=$g;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=0;VLLM_HPU_MTP_DRAFT_CHAIN=0")
                        if [ "${chain_a_ok:-1}" = 1 ]; then
                            switch_args+=(--probe-arg "--switch-arm=chain-md=$g;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=0;VLLM_HPU_MTP_DRAFT_CHAIN=1")
                        fi
                        if [ "${chain_vlg_ok:-1}" = 1 ]; then
                            switch_args+=(--probe-arg "--switch-arm=vlg=$g;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1;VLLM_HPU_MTP_DRAFT_CHAIN=0"
                                --probe-arg "--switch-arm=chain-vlg=$g;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=1;VLLM_HPU_MTP_DRAFT_CHAIN=1")
                        fi
                        switch_args+=(--probe-arg "--switch-arm=all=$g;VLLM_HPU_VOCAB_PARALLEL_LOGITS=1;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=0;VLLM_HPU_MTP_DRAFT_CHAIN=0")
                        if [ "${chain_vp_ok:-1}" = 1 ]; then
                            switch_args+=(--probe-arg "--switch-arm=chain-all=$g;VLLM_HPU_VOCAB_PARALLEL_LOGITS=1;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=0;VLLM_HPU_MTP_DRAFT_CHAIN=1")
                        fi
                        if [ "${chain_a_ok:-1}" != 1 ] && [ "${chain_vlg_ok:-1}" != 1 ] && [ "${chain_vp_ok:-1}" != 1 ]; then
                            echo "headline_job: arm $arm: every chain stage failed its tiny gate, the arm is skipped"
                            failed=$((failed + 1))
                            continue
                        fi
                        ;;
                    m1)
                        # async-sched's M1 off (0) against on (2), with the chain on (stage A, the pair CPU-proven)
                        if [ "${m1_ok:-1}" != 1 ] || [ "${chain_a_ok:-1}" != 1 ]; then
                            echo "headline_job: arm $arm: tiny-m1 or tiny-chain failed, the arm is skipped"
                            failed=$((failed + 1))
                            continue
                        fi
                        g="VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0;VLLM_HPU_VERIFY_LOGITS_IN_GRAPH=0;VLLM_HPU_MTP_DRAFT_CHAIN=1"
                        switch_args+=(--probe-arg "--switch-arm=m1off=$g;VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS=0"
                            --probe-arg "--switch-arm=m1on=$g;VLLM_HPU_FLASH_NEXT_SPEC_DEVICE_INPUTS=2")
                        ;;
                    graph-md-all)
                        switch_args+=(--probe-arg "--switch-arm=off=VLLM_HPU_MTP_DRAFT_GRAPH=0;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0"
                            --probe-arg "--switch-arm=on=VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0"
                            --probe-arg "--switch-arm=md=VLLM_HPU_MTP_DRAFT_GRAPH=0;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0"
                            --probe-arg "--switch-arm=on-md=VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1;VLLM_HPU_VOCAB_PARALLEL_LOGITS=0"
                            --probe-arg "--switch-arm=all=VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1;VLLM_HPU_VOCAB_PARALLEL_LOGITS=1")
                        ;;
                    graph-md)
                        switch_args+=(--probe-arg "--switch-arm=off=VLLM_HPU_MTP_DRAFT_GRAPH=0;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0"
                            --probe-arg "--switch-arm=on=VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0"
                            --probe-arg "--switch-arm=md=VLLM_HPU_MTP_DRAFT_GRAPH=0;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1"
                            --probe-arg "--switch-arm=on-md=VLLM_HPU_MTP_DRAFT_GRAPH=1;VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1")
                        ;;
                    *)
                        switch_args+=(--probe-arg "--switch-arm=off=VLLM_HPU_MTP_DRAFT_GRAPH=0"
                            --probe-arg "--switch-arm=on=VLLM_HPU_MTP_DRAFT_GRAPH=1")
                        ;;
                esac
            fi
            # The post steps (serve_probe.sh --post-cmd), in the order sw, cap, kvlp / kvlpfb, long. Each command is
            # one bash -c string; serve_probe.sh runs it in the arm's folder with SERVE_PORT set.
            for w in sw dg32 cap kvlp kvlpfb kvlpfloor kvlpc long turns; do
                [[ " ${post_words[*]:-} " == *" $w "* ]] || continue
                case "$w" in
                    sw | dg32)
                        # sw: HEADLINE_SW_ARMS at HEADLINE_SW_LEVELS, HEADLINE_SW_ROUNDS rounds (scaling_sw.json);
                        # dg32: the draft graph off / on at 32 streams, one round each (scaling_dg32.json: engine-step's
                        # copy_to log of both forms, with hpulog)
                        sw_levels="${HEADLINE_SW_LEVELS:-32,8,1}"
                        sw_rounds="${HEADLINE_SW_ROUNDS:-2}"
                        sw_spec="${HEADLINE_SW_ARMS:-off=VLLM_HPU_MTP_DRAFT_GRAPH=0,VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0;dg=VLLM_HPU_MTP_DRAFT_GRAPH=1,VLLM_HPU_FLASH_NEXT_DECODE_METADATA=0;md=VLLM_HPU_MTP_DRAFT_GRAPH=0,VLLM_HPU_FLASH_NEXT_DECODE_METADATA=1}"
                        if [ "$w" = dg32 ]; then
                            sw_levels=32
                            sw_rounds=1
                            sw_spec="off=VLLM_HPU_MTP_DRAFT_GRAPH=0;dg=VLLM_HPU_MTP_DRAFT_GRAPH=1"
                        fi
                        sw_cmd=("$PYTHON" "$HERE/decode_scaling.py" --model fn --levels "$sw_levels" --tokens "$TOKENS"
                            --warm-tokens "$WARM" --rounds "$sw_rounds" --no-sampled-check
                            --switch-file "$ARM_DIR/switches" --server-log "$ARM_DIR/server0.log"
                            --timeout "${HEADLINE_PROBE_TIMEOUT:-1800}" --out "$ARM_DIR/scaling_$w.json")
                        # NAME=FLAG=V,FLAG=V;NAME=FLAG=V,... (the first arm's values are the server's start values)
                        IFS=';' read -r -a sw_arms <<<"$sw_spec"
                        for a in "${sw_arms[@]}"; do sw_cmd+=(--switch-arm "${a//,/;}"); done
                        first="${sw_arms[0]#*=}"
                        restore="$(printf '%s\n' "${first//,/ }" | tr ' ' '\n')"
                        post_args+=(--post-cmd "$(printf '%q ' "${sw_cmd[@]}") --port \"\$SERVE_PORT\" 2>&1 | tee probe_$w.log; rc=\${PIPESTATUS[0]}; printf '%s\n' $(printf '%q' "$restore") > switches.new && mv switches.new switches; sleep 4; echo \"switches back to: \$(tr '\n' ' ' < switches)\"; exit \$rc")
                        ;;
                    cap)
                        cap_cmd=("$PYTHON" "$TREE/tools/flash_next/anatomy/anatomy_probe.py" --model fn --phases ""
                            --trace "${HEADLINE_CAPTURE_LEVELS:-32,1}" --trace-tokens 64 --trace-after 12 --trace-seconds 0.6
                            --measured-timeout 600 --out "$ARM_DIR/trace_probe.json")
                        post_args+=(--post-cmd "$(printf '%q ' "${cap_cmd[@]}") --port \"\$SERVE_PORT\" 2>&1 | tee trace_probe.log; rc=\${PIPESTATUS[0]}; ls -la trace 2>/dev/null | tail -n 12; du -sh trace 2>/dev/null; exit \$rc")
                        ;;
                    turns)
                        # async-sched job branch: requests across the recurrent block, new batch shapes after the
                        # commits, the same prompts again (exact tokens), the agent's next turn on the committed blocks
                        turns_cmd=("$PYTHON" "$TREE/tools/flash_next/rental/commit_turns_probe.py" --model fn
                            --block "${HEADLINE_TURNS_BLOCK:-896}" --out "$ARM_DIR/turns.json")
                        post_args+=(--post-cmd "$(printf '%q ' "${turns_cmd[@]}") --port \"\$SERVE_PORT\" 2>&1 | tee turns.log; exit \${PIPESTATUS[0]}")
                        ;;
                    kvlp | kvlpfb | kvlpfloor | kvlpc)
                        # kvlpfloor (build 6): a second bf16 server's raw run against the job's first good bf16 file,
                        # the cross-server floor of the FP8 comparison; kvlpc: the probe's --chat mode (the prompt in
                        # the chat template, so the 24 tokens answer the needle), bf16c against nothing, fp8c against
                        # the job's bf16c file (chat and raw prompts differ: never compared with each other)
                        label=bf16
                        ref_args=()
                        chat_args=()
                        if [ "$w" = kvlpc ]; then
                            chat_args=(--chat)
                            label=bf16c
                            if [ "$fp8kv" = 1 ]; then
                                label=fp8c
                                ref="$(ls -1 "$OUT"/*/kvlp_bf16c.json 2>/dev/null | head -n 1)"
                                if [ -n "$ref" ]; then ref_args=(--reference "$ref"); fi
                            fi
                        elif [ "$w" = kvlpfloor ]; then
                            label=bf16floor
                            ref="${HEADLINE_KVLP_REF:-$(good_kvlp_ref)}"
                            if [ -n "$ref" ]; then ref_args=(--reference "$ref"); fi
                            echo "headline_job: arm $arm: kvlpfloor reference ${ref:-none (no good kvlp_bf16.json in the job)}"
                        elif [ "$fp8kv" = 1 ]; then
                            label=fp8
                            ref="${HEADLINE_KVLP_REF:-$(good_kvlp_ref)}"
                            if [ -n "$ref" ]; then ref_args=(--reference "$ref"); fi
                            echo "headline_job: arm $arm: kvlp reference ${ref:-none (no good kvlp_bf16.json in the job)}"
                        elif [ "$w" = kvlpfb ] && [ -n "$(good_kvlp_ref)" ]; then
                            echo "headline_job: arm $arm: kvlpfb not needed, the job has a reference: $(good_kvlp_ref)"
                            continue
                        fi
                        kvlp_cmd=("$PYTHON" "$TREE/tools/flash_next/rental/kv_logprob_probe.py" --model fn --docs "$TREE"
                            --label "$label" --lengths "${HEADLINE_KVLP_LENGTHS:-16000,64000,120000}"
                            --timeout "${HEADLINE_KVLP_TIMEOUT:-1200}" --out "$ARM_DIR/kvlp_$label.json"
                            ${ref_args[@]+"${ref_args[@]}"} ${chat_args[@]+"${chat_args[@]}"})
                        post_args+=(--post-cmd "$(printf '%q ' "${kvlp_cmd[@]}") --port \"\$SERVE_PORT\" 2>&1 | tee kvlp_$label.log; exit \${PIPESTATUS[0]}")
                        ;;
                    long)
                        long_cmd=("$PYTHON" "$TREE/tools/flash_next/rental/qsa_ab_probe.py" --model fn --levels ""
                            --ctx "${HEADLINE_LONG_CTX:-64000}" --long-streams "${HEADLINE_LONG_STREAMS:-32}"
                            --long-repeat "${HEADLINE_LONG_REPEAT:-1}" --tokens "$TOKENS" --docs "$TREE/docs/design/flash-next"
                            --timeout "${HEADLINE_LONG_TIMEOUT:-1500}" --out "$ARM_DIR/long.json")
                        post_args+=(--post-cmd "$(printf '%q ' "${long_cmd[@]}") --port \"\$SERVE_PORT\" 2>&1 | tee long.log; exit \${PIPESTATUS[0]}")
                        ;;
                esac
            done
            # HEADLINE_TRACE_ENV=1: step-anatomy's capture setup at every MTP server's start (00205's recipe)
            if [ "${HEADLINE_TRACE_ENV:-0}" = 1 ]; then
                mkdir -p "$ARM_DIR/trace" "$ARM_DIR/timing"
                arm_env+=(HABANA_PROFILE=profile_api_light VLLM_HPU_STEP_TIMING="${VLLM_HPU_STEP_TIMING:-1}"
                    VLLM_HPU_STEP_TIMING_DIR="$ARM_DIR/timing")
                serve_extra+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$ARM_DIR/trace\", \"torch_profiler_with_stack\": false, \"torch_profiler_use_gzip\": true, \"torch_profiler_record_shapes\": false}")
            fi
            # HEADLINE_MTP_SAMPLED=1: the server also verifies sampled requests (VLLM_HPU_SPEC_SAMPLED_VERIFY=1, the
            # exact path, unless the job sets it) and SAMPLED rounds follow each level of HEADLINE_MTP_SAMPLED_LEVELS
            # (default 32) right after its greedy rounds.
            sampled_args=(--probe-arg --no-sampled-check)
            sampled_verify="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-0}"
            if [ "${HEADLINE_MTP_SAMPLED:-0}" = 1 ] && [ "$greedy_only" = 0 ]; then
                s_levels="${HEADLINE_MTP_SAMPLED_LEVELS:-32}"
                if [ "$levels" != "$LEVELS" ]; then s_levels="$levels"; fi
                sampled_args=(--probe-arg "--sampled-levels=$s_levels")
                if [ "${HEADLINE_INTERLEAVE_SAMPLED:-0}" = 1 ]; then sampled_args+=(--probe-arg --interleave-sampled); fi
                sampled_verify="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-1}"
            fi
            dense_args=()
            if [ "${HEADLINE_REQUIRE_DENSE:-0}" = 1 ] && [ "${VLLM_HPU_MOE_DENSE:-0}" = 1 ]; then
                dense_args=(--require-log "MOE_DENSE=1: the per-channel FP8 MoE layers keep their stacked experts"
                    --forbid-log "MOE_DENSE=1: a device move after the weights were prepared"
                    --probe-require-log "runs the dense MoE form")
            fi
            echo "headline_job: arm $arm: k $k, levels $levels, draft graph $draft_graph, ${async_env[*]:-sync}," \
                "MTP timer $timing, env ${arm_env[*]:-} ${unset_env[*]:-}, serve args ${serve_extra[*]:-}," \
                "after the probe: ${post_words[*]:-nothing}"
            placed "$ARM_DIR" env ${unset_env[@]+"${unset_env[@]}"} ${async_env[@]+"${async_env[@]}"} \
                ${switch_env[@]+"${switch_env[@]}"} ${arm_env[@]+"${arm_env[@]}"} \
                VLLM_HPU_QSA_INDEXER_SHARD="$shard" VLLM_HPU_SPEC_SAMPLED_VERIFY="$sampled_verify" \
                VLLM_HPU_GDN_CORE_VERIFY="${VLLM_HPU_GDN_CORE_VERIFY:-1}" VLLM_HPU_MTP_TIMING="$timing" \
                VLLM_HPU_MTP_DRAFT_LOOP="${VLLM_HPU_MTP_DRAFT_LOOP:-1}" VLLM_HPU_MTP_CHUNK_KV="${VLLM_HPU_MTP_CHUNK_KV:-1}" \
                VLLM_HPU_MTP_DRAFT_GRAPH="$draft_graph" VLLM_HPU_MTP_REPLICATED_FC="$replicated_fc" \
                bash "$HERE/serve_probe.sh" --out "$ARM_DIR" --mtp "$k" --levels "$levels" --tokens "$TOKENS" \
                --warm-tokens "$WARM" --repeat-top 1 --no-mintok-check "${sampled_args[@]}" \
                ${switch_args[@]+"${switch_args[@]}"} ${dense_args[@]+"${dense_args[@]}"} \
                ${post_args[@]+"${post_args[@]}"} --probe-arg "--rounds=$ROUNDS" \
                ${HEADLINE_MTP_CTX:+--ctx "$HEADLINE_MTP_CTX"} $(barrier_args "$arm") $(guard_args) \
                ${serve_extra[@]+-- "${serve_extra[@]}"}
            ;;
        *)
            echo "headline_job: unknown arm $arm (nospec, mtp3, mtp3-32)"
            failed=$((failed + 1))
            continue
            ;;
    esac
    rc=$?
    # The probe ends with 0 even when rounds failed (it records them): a round with errors, or no measured round
    # at all, makes the arm a probe failure (5).
    if [ "$rc" -eq 0 ] && ! "$PYTHON" - "$OUT/$arm/scaling.json" <<'PY2'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(1)
rounds = data.get("levels") or []  # greedy rounds; a failed sampled round is reported, not a server failure
sys.exit(0 if rounds and all(r.get("ok") and not r.get("errors") for r in rounds) else 1)
PY2
    then
        echo "headline_job: arm $arm: a greedy round failed or none was measured (scaling.json): a probe failure"
        rc=5
    fi
    echo "=== headline_job: arm $arm rc=$rc end $(date -u +%T)"
    if [ "$rc" -ne 0 ]; then failed=$((failed + 1)); fi
    "$PYTHON" "$HERE/headline_table.py" "$OUT" --md "$OUT/headline.md" --json "$OUT/headline.json" || true
    if [ "$rc" -eq 8 ]; then
        # a required server log line is missing (HEADLINE_REQUIRE_DENSE: the dense MoE form did not run): every
        # later arm of the build depends on the same flag, so the job stops here instead of measuring a fallback
        echo "headline_job: arm $arm ended with 8 (a required server log line is missing): the job stops"
        clear_cards
        exit 8
    fi
    left=$((HOLD_S - ($(date -u +%s) - T0)))
    case "$arm" in mtp[1-9]) is_mtp=1 ;; *) is_mtp=0 ;; esac
    if [ "$is_mtp" = 1 ] && [ "$FALLBACK" = 1 ] && { [ "$rc" -eq 3 ] || [ "$rc" -eq 5 ]; }; then
        if [ "$left" -ge 900 ]; then
            echo "headline_job: $arm ended with $rc: the same server with VLLM_HPU_QSA_INDEXER_SHARD=0 next ($left s left)"
            ARMS+=("$arm-noshard")
        else
            echo "headline_job: $arm ended with $rc; only $left s left: no run without the request shard"
        fi
    fi
done
echo "headline_job: $failed of ${#ARMS[@]} arms failed; $(date -u +%T)"
exit "$failed"
