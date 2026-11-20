#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# perf_watch.sh JOB_ID: on the rented node, sample the vLLM workers of a farm job with perf while its probe runs
# the rounds tagged "perf" (anatomy_probe.py prints "round start phase 1 N=<n> ... tag=perf" before each round),
# and count the system calls of the rank-0 worker during the rounds tagged "strace". It only reads the job: perf
# and strace attach to the job's own worker processes from the host (the container has no privilege for either);
# everything it writes is under ~/farm/<track>/. Start it before the job starts; it waits for the job:
#
#   nohup ~/farm/step-anatomy/perf/watch.sh 00069 > /dev/null 2>&1 &
#
# perf is not installed on the node: unpack it once into ~/farm/<track>/perf/root with
#   cd ~/farm/<track>/perf && apt-get download linux-tools-$(uname -r) linux-tools-$(uname -r | sed 's/-generic//') \
#       && for d in *.deb; do dpkg -x "$d" root; done
# Output: ~/farm/<track>/results/perf-<JOB_ID>/<server>-N<n>.{data,dso.txt,sym.txt,tid.txt,calls.txt},
# <server>-N<n>-strace.{count.txt,calls.txt}, load.log (the node's load at the start and the end of every round) and
# loadavg.log; fetch with `farm fetch <track> <JOB_ID> perf-<JOB_ID>`. Run a copy of this script, not the file a
# later update overwrites: `cp watch.sh run-<id>.sh && nohup bash run-<id>.sh <id> &`.
JOB="$1"
T=$HOME/farm/${WATCH_TRACK:-step-anatomy}
P=$(ls "$T"/perf/root/usr/lib/linux-tools-*/perf 2>/dev/null | head -n 1)
BD=$T/perf/buildid
OUT=$T/results/perf-$JOB
JD=${WATCH_JD:-$HOME/farm/q/jobs/$JOB}  # WATCH_JD: a stand-in job folder (a test of this script)
mkdir -p "$OUT" "$BD"
log() { echo "$(date -u +%H:%M:%S) $*" >> "$OUT/watch.log"; }
log "watching job $JOB"
# wait until the job runs (at most 6 h), then until its results folder exists
for _ in $(seq 1 4320); do
    st=$(cat "$JD/state" 2>/dev/null)
    [ "$st" = running ] && break
    case "$st" in ended | cancelled) log "job $st before it ran"; exit 0 ;; esac
    sleep 5
done
RUN=$(ls -d "$T"/results/"$JOB"-* 2>/dev/null | head -n 1)
for _ in $(seq 1 120); do [ -n "$RUN" ] && break; sleep 2; RUN=$(ls -d "$T"/results/"$JOB"-* 2>/dev/null | head -n 1); done
log "run folder: $RUN"
reports() {  # reports FILE LABEL: the text reports, while the container (and so its binaries) still exists
    local f="$1"
    sudo -n "$P" --buildid-dir "$BD" report -i "$f.data" --stdio --no-children -g none --sort pid,dso --percent-limit 0.2 2>/dev/null | grep -v "^$" | head -n 150 > "$f.dso.txt"
    sudo -n "$P" --buildid-dir "$BD" report -i "$f.data" --stdio --no-children -g none --sort dso,sym --percent-limit 0.15 2>/dev/null | grep -v "^$" | head -n 220 > "$f.sym.txt"
    sudo -n "$P" --buildid-dir "$BD" report -i "$f.data" --stdio --no-children -g none --sort pid,tid,dso --percent-limit 0.3 2>/dev/null | grep -v "^$" | head -n 200 > "$f.tid.txt"
    sudo -n "$P" --buildid-dir "$BD" report -i "$f.data" --stdio --children -g caller,0.5,callee --sort sym --percent-limit 2 2>/dev/null | grep -v "^$" | head -n 400 > "$f.calls.txt"
    sudo -n chown -R "$(id -un):" "$OUT" 2>/dev/null
    log "$2: reports written ($(wc -l < "$f.sym.txt") symbol lines)"
}
sample() {  # sample SERVER N: perf record of every worker of the job's container
    local server="$1" n="$2" delay dur pids f
    case "$n" in 1) delay=0.9; dur=3 ;; 8) delay=1.6; dur=3.5 ;; *) delay=3.8; dur=5 ;; esac
    sleep "$delay"
    pids=$(sudo -n docker top "farm-$JOB" -eo pid,args 2>/dev/null | awk '/VLLM::Worker/ {print $1}' | paste -sd, -)
    [ -n "$pids" ] || { log "$server N=$n: no worker processes"; return; }
    f="$OUT/$server-N$n"
    log "$server N=$n: perf record $dur s of pids $pids"
    # Next to the samples: per thread the CPU time, the context switches and the CPU migrations of the same seconds
    # (a worker whose main thread is switched out or moved issues its launches later than the other ranks), and on
    # which CPU every thread runs.
    local side=()
    sudo -n "$P" stat -e task-clock,context-switches,cpu-migrations --per-thread -p "$pids" -o "$f.stat.txt" -- sleep "$dur" > /dev/null 2>&1 &
    side+=($!)
    # thread_stat.py (next to this script): CPU time, run-queue wait, voluntary and involuntary switches and the
    # CPUs of every busy thread, from /proc.
    if [ -f "$T/perf/thread_stat.py" ]; then
        python3 "$T/perf/thread_stat.py" --pids "$pids" --seconds "$dur" --out "$f.threadstat.json" > "$f.threadstat.txt" 2>&1 &
        side+=($!)
    fi
    sudo -n "$P" --buildid-dir "$BD" record -F 999 -g -p "$pids" -o "$f.data" -- sleep "$dur" > "$f.record.log" 2>&1
    wait "${side[@]}" 2>/dev/null
    ps -L -o pid,tid,psr,pcpu,stat,comm -p "$pids" 2>/dev/null | awk 'NR==1 || $4 >= 1.0' > "$f.threads.txt"
    for pid in ${pids//,/ }; do taskset -cp "$pid" 2>/dev/null; done > "$f.affinity.txt"
    sudo -n docker top "farm-$JOB" -eo pid,tid,pcpu,comm,args 2>/dev/null | cut -c1-150 > "$f.procs.txt"
    reports "$f" "$server N=$n" &
}
trace_calls() {  # trace_calls SERVER N: system calls of the rank-0 worker (strace slows it: its own tagged round)
    local server="$1" n="$2" pid f
    sleep 0.9
    pid=$(sudo -n docker top "farm-$JOB" -eo pid,args 2>/dev/null | awk '/VLLM::Worker_TP0/ {print $1; exit}')
    [ -n "$pid" ] || { log "$server N=$n: no rank-0 worker"; return; }
    f="$OUT/$server-N$n-strace"
    log "$server N=$n: strace of pid $pid"
    sudo -n timeout 2.5 strace -f -c -p "$pid" -o "$f.count.txt" > /dev/null 2>&1
    sudo -n timeout 0.4 strace -f -tt -T -p "$pid" -o "$f.calls.txt" > /dev/null 2>&1
    sudo -n chown -R "$(id -un):" "$OUT" 2>/dev/null
    log "$server N=$n: strace written ($(wc -l < "$f.calls.txt" 2>/dev/null) call lines)"
}
# The node's load next to every round (a one-stream step is host-bound: a loaded node makes it slower): one line in
# load.log when a round starts and when it ends, with the load averages, the runnable tasks and the image builds
# that run at that moment; and one line every five seconds in loadavg.log while the job runs.
load_line() {
    echo "$(date -u +%H:%M:%S) load $(cut -d' ' -f1-4 /proc/loadavg) image-builds $(sudo -n docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^farm-img') | $1" >> "$OUT/load.log"
}
declare -A HANDLED  # per server: the lines of its probe.log that are handled
TICK=0
while :; do
    st=$(cat "$JD/state" 2>/dev/null)
    for plog in "$RUN"/*/probe.log; do
        [ -f "$plog" ] || continue
        server=$(basename "$(dirname "$plog")")
        total=$(wc -l < "$plog")
        from=${HANDLED[$server]:-0}
        [ "$total" -gt "$from" ] || continue
        while IFS= read -r line; do
            case "$line" in
                "round start phase "*)
                    load_line "$server | $line"
                    n=$(echo "$line" | sed -n 's/.* N=\([0-9]*\) .*/\1/p')
                    case "$line" in
                        *tag=strace*) trace_calls "$server" "$n" ;;
                        *tag=perf*) sample "$server" "$n" ;;
                    esac
                    ;;
                "phase "* | "trace "* | "flush "*) load_line "$server | end of: ${line:0:100}" ;;
            esac
        done < <(sed -n "$((from + 1)),${total}p" "$plog")
        HANDLED[$server]=$total
    done
    TICK=$((TICK + 1))
    if [ $((TICK % 16)) -eq 0 ]; then echo "$(date -u +%H:%M:%S) $(cat /proc/loadavg)" >> "$OUT/loadavg.log"; fi
    [ "$st" = running ] || break
    sleep 0.3
done
wait
sudo -n chown -R "$(id -un):" "$OUT" 2>/dev/null
log "job state $st: done"
