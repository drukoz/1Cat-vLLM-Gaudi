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
#   mtp<K>-cuts, mtp<K>-cuts-switch  as -rfc, and also VLLM_HPU_REPLICATED_EMBEDDING=1 (every rank holds the whole
#           token table, about 1.1 GiB more per card: the embedding all-reduce of every forward is gone). The tiny
#           arm takes HEADLINE_TINY_REPLICATED_EMBEDDING. HEADLINE_SWITCH_SET=graph-md-all: the four graph-md arms
#           and "all" (draft graph, I01 and VLLM_HPU_VOCAB_PARALLEL_LOGITS=1).
#   mtp<K>-async-switch  the same A/B on the mtp<K>-async server (VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2, the MTP timer
#           at HEADLINE_ASYNC_MTP_TIMING: set 0 for this A/B, the timer's device wait hides async's overlap).
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

if [ $# -eq 0 ]; then set -- nospec mtp3; fi
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
        tiny)
            spec=()
            case "${HEADLINE_TINY_SPEC:-0}" in [1-9]) spec=(--spec "$HEADLINE_TINY_SPEC") ;; esac
            # the MTP server's own flags too (they act only on the speculative mode)
            tiny_sv="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-0}"
            if [ "${HEADLINE_MTP_SAMPLED:-0}" = 1 ]; then tiny_sv="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-1}"; fi
            VLLM_HPU_MTP_DRAFT_LOOP="${VLLM_HPU_MTP_DRAFT_LOOP:-1}" VLLM_HPU_MTP_CHUNK_KV="${VLLM_HPU_MTP_CHUNK_KV:-1}" \
                VLLM_HPU_SPEC_SAMPLED_VERIFY="$tiny_sv" \
                VLLM_HPU_MTP_DRAFT_GRAPH="${HEADLINE_TINY_DRAFT_GRAPH:-${VLLM_HPU_MTP_DRAFT_GRAPH:-0}}" \
                VLLM_HPU_MTP_REPLICATED_FC="${HEADLINE_TINY_MTP_REPLICATED_FC:-${VLLM_HPU_MTP_REPLICATED_FC:-0}}" \
                VLLM_HPU_REPLICATED_EMBEDDING="${HEADLINE_TINY_REPLICATED_EMBEDDING:-${VLLM_HPU_REPLICATED_EMBEDDING:-0}}" \
                VLLM_HPU_FLASH_NEXT_DECODE_METADATA="${HEADLINE_TINY_DECODE_METADATA:-${VLLM_HPU_FLASH_NEXT_DECODE_METADATA:-0}}" \
                "$PYTHON" /farm/farm_tiny.py --gdn hc ${spec[@]+"${spec[@]}"} --out "$OUT/tiny"
            rc=$?
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
            env ${async_env[@]+"${async_env[@]}"} \
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
            env "${change[@]}" \
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
        mtp[1-9] | mtp[1-9]-32 | mtp[1-9]-noshard | mtp[1-9]-async | mtp[1-9]-graph | mtp[1-9]-switch | \
            mtp[1-9]-async-switch | mtp[1-9]-rfc | mtp[1-9]-rfc-switch | mtp[1-9]-cuts | mtp[1-9]-cuts-switch)
            k="${arm:3:1}"
            # mtp<K>-graph: the k draft passes of a decode step in one HPU graph (VLLM_HPU_MTP_DRAFT_GRAPH=1)
            draft_graph="${VLLM_HPU_MTP_DRAFT_GRAPH:-0}"
            case "$arm" in *-graph) draft_graph=1 ;; esac
            # mtp<K>-rfc*: the draft's fc linears unsharded (built at the server's start, not switchable)
            replicated_fc="${VLLM_HPU_MTP_REPLICATED_FC:-0}"
            replicated_embedding="${VLLM_HPU_REPLICATED_EMBEDDING:-0}"
            case "$arm" in *-rfc | *-rfc-*) replicated_fc=1 ;; esac
            # mtp<K>-cuts*: the draft's fc linears unsharded and the whole token table on every rank (server start)
            case "$arm" in *-cuts | *-cuts-*) replicated_fc=1; replicated_embedding=1 ;; esac
            # mtp<K>-switch: the flag per round on one server (the server starts with it off)
            switch_env=()
            switch_args=()
            case "$arm" in
                *-switch)
                    draft_graph=0
                    mkdir -p "$OUT/$arm"
                    rm -f "$OUT/$arm/switches" "$OUT/$arm/switches.new"
                    switch_env=(VLLM_HPU_RUNTIME_SWITCH_FILE="$OUT/$arm/switches")
                    switch_args=(--probe-arg "--switch-file=$OUT/$arm/switches"
                        --probe-arg "--server-log=$OUT/$arm/server0.log")
                    # HEADLINE_SWITCH_SET: graph (default) = off / on; graph-md = the 2 x 2 with engine-step's
                    # VLLM_HPU_FLASH_NEXT_DECODE_METADATA (the sparse-QSA decode metadata, also switchable)
                    case "${HEADLINE_SWITCH_SET:-graph}" in
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
                    ;;
            esac
            # mtp<K>-async: VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2 (async scheduling with MTP) and the MTP timer at
            # HEADLINE_ASYNC_MTP_TIMING (default 2, the coordinator's; async-sched advises 0: the timer's device wait
            # at the draft's end hides the overlap).
            timing="${VLLM_HPU_MTP_TIMING:-1}"
            async_env=()
            # (mtp<K>-async-switch: the async server of mtp<K>-async with the draft-graph flag switched per round)
            if [ "${arm%-async}" != "$arm" ] || [ "${arm%-async-switch}" != "$arm" ]; then
                async_env=(VLLM_HPU_FLASH_NEXT_ASYNC_SCHED=2)
                timing="${HEADLINE_ASYNC_MTP_TIMING:-2}"
            fi
            levels="$LEVELS"
            case "$arm" in *-32) levels=32 ;; esac
            shard="$VLLM_HPU_QSA_INDEXER_SHARD"
            case "$arm" in *-noshard) shard=0 ;; esac
            # HEADLINE_MTP_SAMPLED=1: the server also verifies sampled requests (VLLM_HPU_SPEC_SAMPLED_VERIFY=1, the
            # exact path) and one SAMPLED level (32 streams) runs LAST, after every greedy level.
            sampled_args=(--probe-arg --no-sampled-check)
            sampled_verify="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-0}"
            if [ "${HEADLINE_MTP_SAMPLED:-0}" = 1 ] && [ "${arm%-noshard}" = "$arm" ]; then
                sampled_args=(--probe-arg "--sampled-levels=${HEADLINE_MTP_SAMPLED_LEVELS:-32}")
                sampled_verify="${VLLM_HPU_SPEC_SAMPLED_VERIFY:-1}"
            fi
            env ${async_env[@]+"${async_env[@]}"} ${switch_env[@]+"${switch_env[@]}"} \
            VLLM_HPU_QSA_INDEXER_SHARD="$shard" VLLM_HPU_SPEC_SAMPLED_VERIFY="$sampled_verify" \
            VLLM_HPU_GDN_CORE_VERIFY="${VLLM_HPU_GDN_CORE_VERIFY:-1}" VLLM_HPU_MTP_TIMING="$timing" \
                VLLM_HPU_MTP_DRAFT_LOOP="${VLLM_HPU_MTP_DRAFT_LOOP:-1}" VLLM_HPU_MTP_CHUNK_KV="${VLLM_HPU_MTP_CHUNK_KV:-1}" \
                VLLM_HPU_MTP_DRAFT_GRAPH="$draft_graph" VLLM_HPU_MTP_REPLICATED_FC="$replicated_fc" \
                VLLM_HPU_REPLICATED_EMBEDDING="$replicated_embedding" \
                bash "$HERE/serve_probe.sh" --out "$OUT/$arm" --mtp "$k" --levels "$levels" --tokens "$TOKENS" \
                --warm-tokens "$WARM" --repeat-top 1 --no-mintok-check "${sampled_args[@]}" \
                ${switch_args[@]+"${switch_args[@]}"} --probe-arg "--rounds=$ROUNDS" ${HEADLINE_MTP_CTX:+--ctx "$HEADLINE_MTP_CTX"} $(barrier_args "$arm") $(guard_args)
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
