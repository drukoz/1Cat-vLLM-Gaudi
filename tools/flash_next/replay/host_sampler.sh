#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# host_sampler.sh: what the host does inside a replay, seen from outside the process. Run it on the node
# (not in the container), inside the card job that runs bench_replay.py with --hold-file:
#
#   host_sampler.sh HOLD_FILE OUT_DIR [STACKS] [WAIT_S]
#
# bench_replay.py's hold loop replays the body back to back; rank 0 writes HOLD_FILE (its pid inside the
# container) before the loop and HOLD_FILE.done after it. This script waits for HOLD_FILE (up to WAIT_S
# seconds, default 1500), finds rank 0's pid on the host and records into OUT_DIR:
#   threads.txt      CPU time and context switches of every thread of the process over a 2 s window
#                    (/proc/PID/task/*/stat and status): which threads work during replays, in user or
#                    system time, and which sleep and wake (voluntary switches)
#   stacks.txt       STACKS (default 12) native stacks of every thread (gdb, attach and detach each time;
#                    only the symbols of the Habana libraries, the bridge and libc are loaded): where the
#                    calling thread is inside replayV3, launch or collective, and in which library
#   stacks_full.txt  one more sample with every library's symbols (a slower attach)
#   strace_c.txt     the syscalls of every thread during 3 s (strace -c -f): does a launch or a collective
#                    enter the kernel (ioctl), does a thread sleep (futex, poll)
#   strace_main.txt  the calling thread's syscalls with their times during 0.3 s
#   status.txt       what was done when (and what was skipped because the loop had ended)
# SAMPLER_MATCH (environment): a string rank 0's command line must contain (the hold file's path as the
# container sees it), so that no other job's process is ever attached to.
# gdb and strace need root (sudo -n). Nothing is changed in the process; while a tool is attached the
# process is slower, so the loop's own timing (the bench's "hold" record) is not a measurement.
# The exit code is 0 unless HOLD_FILE never appeared (3) or no host pid matched (4).

set -u

HOLD="${1:?usage: host_sampler.sh HOLD_FILE OUT_DIR [STACKS] [WAIT_S]}"
OUT="${2:?usage: host_sampler.sh HOLD_FILE OUT_DIR [STACKS] [WAIT_S]}"
STACKS="${3:-12}"
WAIT_S="${4:-1500}"
SUDO="${SAMPLER_SUDO-sudo -n}"

mkdir -p "$OUT"
STATUS="$OUT/status.txt"
note() { echo "$(date -u +%H:%M:%S.%N | cut -c1-12) $*" | tee -a "$STATUS"; }
loop_done() { [ -f "$HOLD.done" ]; }

note "waiting for $HOLD (up to $WAIT_S s)"
waited=0
while [ ! -f "$HOLD" ]; do
    sleep 0.5
    waited=$((waited + 1))
    if [ "$waited" -ge $((WAIT_S * 2)) ]; then
        note "no hold file after $WAIT_S s: nothing sampled"
        exit 3
    fi
done
INNER="$(sed -n 's/.*"pid": *\([0-9][0-9]*\).*/\1/p' "$HOLD" | head -n 1)"
note "hold file: $(cat "$HOLD")"

# Rank 0's pid on the host: the bench child of rank 0 whose innermost namespace pid is the one in the file
# and whose command line names this run (SAMPLER_MATCH, the hold file's path inside the container). Nothing
# else is ever attached to: other jobs run on the node at the same time.
PID=""
for p in $(pgrep -f -- 'bench_replay.py --child --process lazy --rank 0 ' 2>/dev/null); do
    inner="$(awk '/^NSpid:/ {print $NF}' "/proc/$p/status" 2>/dev/null)"
    [ "$inner" = "$INNER" ] || continue
    if [ -n "${SAMPLER_MATCH:-}" ] && ! tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null | grep -q -F -- "$SAMPLER_MATCH"; then
        continue
    fi
    PID="$p"
    break
done
if [ -z "$PID" ]; then
    note "no process matches rank 0 (pid $INNER in the container, command line with '${SAMPLER_MATCH:-}')"
    exit 4
fi
note "rank 0: host pid $PID (in the container: $INNER), $(find "/proc/$PID/task" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l) threads"

# 1. CPU time per thread over a window. /proc/PID/task/TID/stat: fields 14 and 15 after the name are utime, stime.
snapshot() {
    local t vol invol
    for t in /proc/"$PID"/task/*; do
        [ -r "$t/stat" ] || continue
        vol="$(awk '/^voluntary_ctxt_switches:/ {print $2}' "$t/status" 2>/dev/null)"
        invol="$(awk '/^nonvoluntary_ctxt_switches:/ {print $2}' "$t/status" 2>/dev/null)"
        awk -v tid="${t##*/}" -v comm="$(cat "$t/comm" 2>/dev/null)" -v vol="${vol:-0}" -v invol="${invol:-0}" '{
            line = $0; sub(/^.*\) /, "", line); n = split(line, f, " ");
            print tid, f[12], f[13], vol, invol, comm
        }' "$t/stat" 2>/dev/null
    done
}
TICK="$(getconf CLK_TCK 2>/dev/null || echo 100)"
snapshot >"$OUT/.threads_a"
t_a="$(date +%s.%N)"
sleep 2
snapshot >"$OUT/.threads_b"
t_b="$(date +%s.%N)"
{
    echo "# pid $PID, window $(echo "$t_b - $t_a" | bc 2>/dev/null || echo 2) s, clock tick 1/$TICK s; loop ended before the window closed: $(loop_done && echo yes || echo no)"
    echo "# tid user_ms system_ms voluntary_switches involuntary_switches name   (threads that ran or switched in the window)"
    awk -v tick="$TICK" 'NR == FNR { u[$1] = $2; s[$1] = $3; v[$1] = $4; w[$1] = $5; next }
        ($1 in u) { du = ($2 - u[$1]) * 1000 / tick; ds = ($3 - s[$1]) * 1000 / tick;
                    name = $6; for (i = 7; i <= NF; i++) name = name " " $i;
                    if (du + ds > 0 || $4 - v[$1] > 0)
                        printf "%s %d %d %d %d %s\n", $1, du, ds, $4 - v[$1], $5 - w[$1], name }' \
        "$OUT/.threads_a" "$OUT/.threads_b" | sort -k2,2nr -k3,3nr -k4,4nr
    echo "# every thread (tid name):"
    awk '{ name = $6; for (i = 7; i <= NF; i++) name = name " " $i; print $1, name }' "$OUT/.threads_b" | tr '\n' ';'
    echo
} >"$OUT/threads.txt"
rm -f "$OUT/.threads_a" "$OUT/.threads_b"
note "threads.txt written"

# 2. Native stacks. Only the libraries that can hold the launch and collective path get symbols: the attach
# stays short, and a frame in another library ends the backtrace there.
: >"$OUT/stacks.txt"
taken=0
for i in $(seq 1 "$STACKS"); do
    if loop_done; then
        note "loop ended after $taken stacks"
        break
    fi
    {
        echo "=== sample $i $(date -u +%H:%M:%S.%N | cut -c1-12)"
        $SUDO timeout 40 gdb -nx -batch \
            -iex "set auto-solib-add off" -iex "set sysroot /proc/$PID/root" -iex "set pagination off" \
            -iex "set print thread-events off" \
            -p "$PID" \
            -ex "sharedlibrary libhcl" -ex "sharedlibrary libSynapse" -ex "sharedlibrary libscal" \
            -ex "sharedlibrary libhabana_pytorch" -ex "sharedlibrary _hpu_C" -ex "sharedlibrary libhl" \
            -ex "sharedlibrary libc\\.so" -ex "sharedlibrary libpthread" -ex "sharedlibrary libstdc" \
            -ex "sharedlibrary libgcc_s" -ex "sharedlibrary librdmacm" -ex "sharedlibrary libibverbs" \
            -ex "sharedlibrary libhbl" -ex "sharedlibrary libm\\.so" \
            -ex "thread apply all bt 28" 2>&1 | grep -v -E '^\[New LWP|^warning: (Could not load|.*separate debug)|^Reading symbols|^\(No debugging'
    } >>"$OUT/stacks.txt" 2>&1
    taken=$((taken + 1))
    sleep 0.15
done
note "stacks.txt: $taken samples"
# One more sample with the symbols of every library (a slower attach, at most 45 s): the whole stack of each
# thread, for the samples above whose backtrace ends at a library without symbols.
if ! loop_done; then
    {
        echo "=== full sample $(date -u +%H:%M:%S.%N | cut -c1-12)"
        $SUDO timeout 45 gdb -nx -batch -iex "set sysroot /proc/$PID/root" -iex "set pagination off" \
            -iex "set print thread-events off" -p "$PID" -ex "thread apply all bt 48" 2>&1 \
            | grep -v -E '^\[New LWP|^warning: (Could not load|.*separate debug)|^Reading symbols|^\(No debugging'
    } >"$OUT/stacks_full.txt" 2>&1
    note "stacks_full.txt written ($(grep -c '^#' "$OUT/stacks_full.txt") frames)"
fi

# 3. Syscalls of every thread for 3 s, then the calling thread's with times for 0.3 s.
if loop_done; then
    note "loop ended: no strace"
else
    $SUDO timeout -s INT 3 strace -f -c -p "$PID" -o "$OUT/strace_c.txt" >/dev/null 2>"$OUT/strace_c.err"
    note "strace_c.txt written (loop ended meanwhile: $(loop_done && echo yes || echo no))"
    if ! loop_done; then
        $SUDO timeout -s INT 0.3 strace -T -tt -p "$PID" -o "$OUT/strace_main.raw" >/dev/null 2>>"$OUT/strace_c.err"
        head -n 4000 "$OUT/strace_main.raw" >"$OUT/strace_main.txt" 2>/dev/null
        $SUDO rm -f "$OUT/strace_main.raw"
        note "strace_main.txt written ($(wc -l <"$OUT/strace_main.txt" 2>/dev/null) lines)"
    fi
fi
$SUDO chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null
note "done"
exit 0
