#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One command on the workstation for a whole paid Flash-Next rental session (RUK-2709), given only
# the node's IP address:
#
#     bash tools/flash_next/rental/rent.sh <ip>
#
# It replaces the six manual steps of the run sheet's "Cheapest session order" with eight stages:
# connect, bundle, host (prepare_host.sh), download (started, not waited for), smoke
# (tiny_smoke.sh, one mode per call), wait (for the download), run (run_rental.sh), fetch.
# Page: docs/design/flash-next/run-sheets/one-command.md.
#
# The session folder (default ~/rental-sessions/<ip>-<UTC date>/) holds session.log (everything
# printed, with UTC timestamps), state (the finished stages and the clock), known_hosts (the node's
# host key, for this session only), jobs/ (copies of the node-side job logs) and results/. The same
# command with the same address continues the session: a finished stage is skipped after a check on
# the node that its result is still there.
#
# Long work on the node (prepare_host.sh, tiny_smoke.sh, run_rental.sh) runs as a named job that is
# detached from the SSH connection (setsid and nohup): its output is in ~/rental/.jobs/<name>.log
# and its exit code in ~/rental/.jobs/<name>.rc. This command follows the log, reconnects after a
# dropped connection, and Ctrl-C only detaches. --status prints the jobs, --stop interrupts the run
# (SIGINT to run_rental.sh, which stops its servers and writes the tarball) and fetches.
#
# It cannot release the node: the billing stops only in the provider's console.
#
# Exit codes: 0 done; 1 a stage failed; 2 usage; 3 stopped for a decision (data disk, reboot);
# 130 detached by Ctrl-C. Runs under the bash 3.2 of macOS.
#
# Timing knobs (seconds), for tests: FN_RENTAL_POLL_S (30, download progress), FN_RENTAL_FOLLOW_S
# (1, job log poll on the node), FN_RENTAL_HEARTBEAT_S (15), FN_RENTAL_RETRY_S (2, first retry
# delay), FN_RENTAL_RETRY_MAX_S (30), FN_RENTAL_SSH_TRIES (6, attempts of a short command).

# shellcheck disable=SC2329  # on_exit, on_signal and what they call run from traps

set -euo pipefail

SELF="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL=8
DOWNLOAD_NEED_GB=250  # free space run_rental.sh --download-only wants under --model
DISK_MIN_GB=300       # the smallest blank disk this command picks by itself
FS_LABEL="fn-models"  # the label prepare_host.sh gives the data disk it makes
CKPT_GB="${FN_RENTAL_CKPT_GB:-185.5}"        # size of the checkpoint, for the ETA
CKPT_MIN_GB="${FN_RENTAL_CKPT_MIN_GB:-180}"  # a folder below this is not a complete checkpoint
POLL_S="${FN_RENTAL_POLL_S:-30}"
FOLLOW_S="${FN_RENTAL_FOLLOW_S:-1}"
HEARTBEAT_S="${FN_RENTAL_HEARTBEAT_S:-15}"
RETRY_S="${FN_RENTAL_RETRY_S:-2}"
RETRY_MAX_S="${FN_RENTAL_RETRY_MAX_S:-30}"
SSH_TRIES="${FN_RENTAL_SSH_TRIES:-6}"
SESSION_MAX_AGE_MIN=1200  # a session of an earlier UTC day is continued while its state is younger
NL=$'\n'
CR=$'\r'
BEGIN_MARK="FN-BEGIN"

usage() {
    cat <<EOF
usage: $(basename "$SELF") <ip> [options]
connection and session
  --user NAME        login user (default: sesterce; env FN_RENTAL_SSH_USER)
  --key FILE         private key (default: ~/.ssh/sesterce_ed25519; env FN_RENTAL_SSH_KEY)
  --session DIR      session folder (default: ~/rental-sessions/<ip>-<UTC date>)
  --rate N           dollars per hour for the cost lines (default: 10)
stage 2, bundle
  --bundle FILE      use this bundle (default: build one with make_rental_bundle.sh into dist/)
  --vllm-src PATH    vLLM checkout for the build (env FN_VLLM_SRC)
  --allow-dirty      build from a working tree that is not clean
stage 3, host
  --data-disk DEV    the blank disk prepare_host.sh formats and mounts for the checkpoint (default:
                     the only blank disk of at least $DISK_MIN_GB GB; none when there is room already)
  --model DIR        checkpoint folder on the node (default: /models/Qwen3.8-Flash-Next-FP8)
stages 4 and 6, download
  --no-download      the checkpoint is in place: start no download and wait for none
stage 5, smoke
  --smoke-modes LIST modes of tiny_smoke.sh, one call each: pytorch, fused, hc, moe (default: pytorch,fused)
  --skip-smoke       no smoke test
  --smoke-arg ARG    extra argument for tiny_smoke.sh (repeatable)
  --continue-after-smoke-failure LIST
                     start the run although these modes failed: fused, hc, moe (never pytorch)
stage 7, run
  --variants LIST    variants of the run (default: A,C)
  --no-quick-gates   run without --quick-gates
  --hc-fused         pass --hc-fused to run_rental.sh
  --moe-glue         pass --moe-glue to run_rental.sh
  --run-arg ARG      extra argument for run_rental.sh (repeatable)
  --from PHASE       resume this session's last run from PHASE (same run id)
  --new-run          start another run although this session's run has ended
other ways to call it
  --status           print each job's state and the last lines of its log; changes nothing
  --stop             interrupt the run (it writes its report and tarball), then fetch
  --dry-run          print every local and remote command in order; contacts nothing
  -h, --help         this text
EOF
}

# ------------------------------------------------------------------ printing, the log, the state

LOG=""
STATE=""
TS=""
TS_AT=-1
REMIND=1
CUR_N=""
CUR_NAME=""
FOLLOW_PID=""
NAP_PID=""
FOLLOWING=""
FIFO=""
IP=""
RATE=10
RERUN=""

log_line() {
    [ -n "$LOG" ] || return 0
    if [ "$SECONDS" != "$TS_AT" ]; then
        TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        TS_AT="$SECONDS"
    fi
    printf '%s %s\n' "$TS" "$1" >>"$LOG" 2>/dev/null || true
}

say() {
    printf '%s\n' "$*"
    log_line "$*"
}

warn() {
    printf '%s\n' "$*" >&2
    log_line "$*"
}

# mark TEXT — a line the operator must not miss (bold on a terminal).
mark() {
    if [ -t 1 ]; then printf '\033[1m%s\033[0m\n' "$1"; else printf '%s\n' "$1"; fi
    log_line "$1"
}

# say_lines TEXT [PREFIX] — TEXT line by line.
say_lines() {
    local line
    [ -n "$1" ] || return 0
    while IFS= read -r line; do say "${2:-}$line"; done <<<"$1"
}

die() {
    warn "error: $*"
    exit 1
}

state_get() {
    local line
    if [ -z "$STATE" ] || [ ! -f "$STATE" ]; then return 0; fi
    while IFS= read -r line; do
        case "$line" in
            "$1="*)
                printf '%s\n' "${line#*=}"
                return 0
                ;;
        esac
    done <"$STATE"
}

state_set() {
    local line
    [ -n "$STATE" ] || return 0
    {
        if [ -f "$STATE" ]; then
            while IFS= read -r line; do
                case "$line" in
                    "$1="*) ;;
                    *) printf '%s\n' "$line" ;;
                esac
            done <"$STATE"
        fi
        printf '%s=%s\n' "$1" "$2"
    } >"$STATE.tmp"
    mv -f "$STATE.tmp" "$STATE"
}

# Seconds since the first successful connection of this session (0 before it).
elapsed_s() {
    local start
    start="$(state_get clock_start)"
    case "$start" in
        '' | *[!0-9]*) echo 0 ;;
        *) echo $(($(date -u +%s) - start)) ;;
    esac
}

fmt_dur() {
    printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60))
}

cost_of() {
    awk -v s="$1" -v r="$RATE" 'BEGIN { printf "%.2f", s * r / 3600 }'
}

stage_line() {
    local e
    e="$(elapsed_s)"
    say "stage $CUR_N/$TOTAL $CUR_NAME: $1${2:+ ($2)}  [session $(fmt_dur "$e"), \$$(cost_of "$e") at \$$RATE/h]"
}

stage_start() {
    CUR_N="$1"
    CUR_NAME="$2"
    if [ "$DRY_RUN" -eq 1 ]; then
        say "stage $CUR_N/$TOTAL $CUR_NAME: these commands"
    else
        stage_line start ""
    fi
}

stage_ok() {
    if [ "$DRY_RUN" -eq 0 ]; then
        stage_line ok "${1:-}"
    elif [ -n "${1:-}" ]; then
        say "[dry-run] $1"
    fi
    CUR_N=""
}

stage_failed() {
    stage_line FAILED "${1:-}"
    CUR_N=""
}

# The standing reminder, printed at every stop.
reminder() {
    local e start
    start="$(state_get clock_start)"
    say ""
    if [ -n "$IP" ] && [ -n "$start" ]; then
        e="$(elapsed_s)"
        say "REMINDER: the node $IP is still billing: $(fmt_dur "$e") since this session's first connection, about \$$(cost_of "$e") at \$$RATE/h."
    else
        say "REMINDER: a rented node is billing from its first minute, whatever this command did (this session has no clock yet)."
    fi
    say "This command cannot release the node: stop the node in the provider's console when you are done."
}

stop_children() {
    if [ -n "$FOLLOW_PID" ]; then kill "$FOLLOW_PID" 2>/dev/null || true; fi
    if [ -n "$NAP_PID" ]; then kill "$NAP_PID" 2>/dev/null || true; fi
    if [ -n "$FIFO" ]; then rm -f "$FIFO" 2>/dev/null || true; fi
    FOLLOW_PID=""
    NAP_PID=""
}

on_exit() {
    local rc=$?
    trap - EXIT
    trap '' INT TERM HUP
    set +e
    stop_children
    if [ -n "$CUR_N" ] && [ "$rc" -ne 0 ] && [ "$rc" -lt 129 ]; then stage_line FAILED ""; fi
    if [ "$REMIND" -eq 1 ]; then reminder; fi
    exit "$rc"
}

# Ctrl-C, SIGTERM or a closed terminal: leave, and leave the node alone.
on_signal() {
    trap '' INT TERM HUP
    set +e
    stop_children
    say ""
    say "interrupted (SIG$1): detached; nothing on the node was stopped."
    if [ -n "$FOLLOWING" ]; then
        say "the job '$FOLLOWING' goes on on the node (log: ~/rental/.jobs/$FOLLOWING.log)."
    fi
    if [ -n "$RERUN" ]; then
        say "to re-attach and go on, run the same command:"
        say "  $RERUN"
    fi
    exit "$2"
}

trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

# nap SECONDS — a sleep that a signal ends at once.
nap() {
    sleep "$1" &
    NAP_PID=$!
    wait "$NAP_PID" 2>/dev/null || true
    NAP_PID=""
}

next_delay() {
    awk -v d="$1" -v m="$RETRY_MAX_S" 'BEGIN { d *= 2; if (d > m) d = m; print d }'
}

# first_value PREFIX TEXT — what follows PREFIX on the first line of TEXT that starts with it.
first_value() {
    local line
    while IFS= read -r line; do
        case "$line" in
            "$1"*)
                printf '%s\n' "${line#"$1"}"
                return 0
                ;;
        esac
    done <<<"$2"
}

last_line() {
    local text="$1"
    text=${text%"$NL"}
    text=${text##*"$NL"}
    printf '%s\n' "$text"
}

# ------------------------------------------------------------------ quoting and the SSH arguments

# rq WORD... — the one quoting helper: the words as one shell-safe string. Every remote command is
# built with it and handed to the node as ONE argument.
rq() {
    local out="" a sq="'" rep="'\\''"
    for a in "$@"; do
        case "$a" in
            '' | *[!A-Za-z0-9_./:=,@%+-]*)
                a=${a//"$sq"/$rep}
                a="'$a'"
                ;;
        esac
        out="$out${out:+ }$a"
    done
    printf '%s\n' "$out"
}

# The one place that names connection options; every call below takes "${SSH_OPTS[@]}" from here.
# The known-hosts file is the session's own: the provider reuses addresses between rentals, and an
# old entry of the user's file must neither block nor vouch for this node. No prompt ever
# (BatchMode), a connect timeout, and keep-alives that notice a dead connection within a minute.
# In the option's value a "%" starts a token and a space ends a file name: the first is doubled,
# and a path with a space goes in double quotes, as ssh_config(5) has it.
ssh_opts() {
    local known="" pct='%'
    known=${KNOWN_HOSTS//"$pct"/%%}
    case "$known" in
        *[[:space:]]*) known="\"$known\"" ;;
    esac
    SSH_OPTS=(
        -i "$SSH_KEY"
        -o IdentitiesOnly=yes
        -o "UserKnownHostsFile=$known"
        -o StrictHostKeyChecking=accept-new
        -o BatchMode=yes
        -o ConnectTimeout=15
        -o ServerAliveInterval=15
        -o ServerAliveCountMax=4
        -o LogLevel=ERROR
    )
}

# node_path REL — a path under the node's home, as the copy command names it.
node_path() {
    case "$IP" in
        *:*) printf '%s\n' "$SSH_USER@[$IP]:$1" ;;
        *) printf '%s\n' "$TARGET:$1" ;;
    esac
}

# ------------------------------------------------------------------ the node helpers
# Sent in front of every remote command (bash -c '<helpers> <command>'), so the node needs nothing
# installed and holds no state of ours but ~/rental/.jobs. Plain bash; the test node runs 3.2.

IFS= read -r -d '' PRELUDE <<'EOF' || true
echo FN-BEGIN
R="$HOME/rental"; J="$R/.jobs"

# fn_job_state NAME: running PID | ended RC | lost (no process, no exit code) | none
fn_job_state() {
    local n="$1" pid=""
    if [ -f "$J/$n.rc" ]; then echo "ended $(cat "$J/$n.rc")"; return 0; fi
    if [ ! -f "$J/$n.pid" ]; then echo none; return 0; fi
    pid="$(cat "$J/$n.pid" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && ps -ww -o args= -p "$pid" 2>/dev/null | grep -q "/$n\.run $n"; then
        echo "running $pid"; return 0
    fi
    if [ -f "$J/$n.rc" ]; then echo "ended $(cat "$J/$n.rc")"; else echo lost; fi
}

# fn_job_start NAME COMMAND [ID]: start COMMAND detached from this connection, or attach when NAME
# is running. Output in NAME.log, exit code in NAME.rc, the command's pid in NAME.cpid.
# No "&" anywhere: a background start would hand the job an ignored SIGINT, and run_rental.sh
# could then not be interrupted by --stop.
fn_job_start() {
    local n="$1" cmd="$2" id="${3:--}" st="" i=0
    mkdir -p "$J" || return 1
    st="$(fn_job_state "$n")"
    case "$st" in
        running*) echo "attached ${st#running } $(cat "$J/$n.id" 2>/dev/null)"; return 0 ;;
    esac
    command -v setsid >/dev/null 2>&1 || { echo "no setsid on the node" >&2; return 1; }
    rm -f "$J/$n.rc" "$J/$n.pid" "$J/$n.cpid"
    if [ -f "$J/$n.log" ]; then mv -f "$J/$n.log" "$J/$n.log.prev"; fi
    : >"$J/$n.log"
    printf '%s\n' "$id" >"$J/$n.id"
    printf '%s.%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "$$" >"$J/$n.gen"
    printf '%s\n' 'echo $$ >"$FN_JOB_CPID"' "$cmd" >"$J/$n.cmd"
    printf '%s\n' \
        'J="$(cd "$(dirname "$0")" && pwd)"; n="$1"' \
        'echo $$ >"$J/$n.pid"' \
        'FN_JOB_CPID="$J/$n.cpid" bash "$J/$n.cmd" >>"$J/$n.log" 2>&1 </dev/null' \
        'echo $? >"$J/$n.rc.tmp"; mv -f "$J/$n.rc.tmp" "$J/$n.rc"' >"$J/$n.run"
    setsid -f nohup bash "$J/$n.run" "$n" </dev/null >/dev/null 2>&1 || { echo "setsid -f failed on the node" >&2; return 1; }
    while [ ! -s "$J/$n.pid" ] && [ "$i" -lt 100 ]; do sleep 0.1; i=$((i + 1)); done
    [ -s "$J/$n.pid" ] || { echo "job $n did not start (no $J/$n.pid)" >&2; return 1; }
    echo "started $(cat "$J/$n.pid") $id"
}

# fn_job_follow NAME FROM GEN EVERY HEARTBEAT: the log from line FROM on as "L <line>", "H" when
# nothing came for HEARTBEAT seconds, then "E <exit code>" or "X <state>". "G <gen>" first when the
# log is not the one the caller has lines of (the job was started again): then from line 1.
fn_job_follow() {
    local n="$1" from="$2" gen="$3" every="$4" hb="$5" st="" total=0 cur="" last="$SECONDS"
    cur="$(cat "$J/$n.gen" 2>/dev/null)"
    if [ "$cur" != "$gen" ]; then echo "G ${cur:-none}"; from=1; fi
    while :; do
        st="$(fn_job_state "$n")"
        if [ -f "$J/$n.log" ]; then
            total="$(wc -l <"$J/$n.log" | tr -d ' ')"
            if [ "$total" -ge "$from" ]; then
                sed -n "${from},${total}p" "$J/$n.log" | sed 's/^/L /'
                from=$((total + 1)); last="$SECONDS"
            fi
        fi
        case "$st" in
            running*) ;;
            ended*)
                if [ -s "$J/$n.log" ] && [ "$(tail -c 1 "$J/$n.log" | wc -l | tr -d ' ')" = 0 ]; then
                    printf 'L %s\n' "$(tail -n 1 "$J/$n.log")"
                fi
                echo "E ${st#ended }"; return 0 ;;
            *) echo "X $st"; return 0 ;;
        esac
        if [ $((SECONDS - last)) -ge "$hb" ]; then echo H; last="$SECONDS"; fi
        sleep "$every"
    done
}

# fn_ckpt MODEL: the download container and the checkpoint folder, on one line.
fn_ckpt() {
    local m="$1" c="" status=none rc=0 kb=0 config=no inc=no
    c="$(sudo -n docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' fn-download 2>/dev/null)" || c=""
    if [ -n "$c" ]; then status="${c%% *}"; rc="${c##* }"; fi
    if [ -d "$m" ]; then
        kb="$(sudo -n du -sk "$m" 2>/dev/null | awk 'NR==1 {print $1}')"; kb="${kb:-0}"
        if [ -f "$m/config.json" ]; then config=yes; fi
        if [ -n "$(sudo -n find "$m" -name '*.incomplete' 2>/dev/null | head -n 1)" ]; then inc=yes; fi
    fi
    echo "container=$status rc=$rc kb=$kb config=$config incomplete=$inc"
}

# fn_host_facts MODEL: what stage 3 decides on. Reads only.
fn_host_facts() {
    local m="$1" d="$1"
    if sudo -n docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"'; then echo "runtime=yes"; else echo "runtime=no"; fi
    if command -v hl-smi >/dev/null 2>&1; then echo "cards=$(hl-smi -Q index,name -f csv,noheader 2>/dev/null | grep -c .)"; else echo "cards=0"; fi
    while [ ! -d "$d" ] && [ "$d" != / ]; do d="$(dirname "$d")"; done
    echo "space_dir=$d"
    df -Pk "$d" 2>/dev/null | awk 'NR==2 {print "free_kb=" $4}'
    echo "ckpt=$(fn_ckpt "$m")"
    sudo -n lsblk -b -P -o NAME,TYPE,SIZE,FSTYPE,LABEL,MOUNTPOINT,PKNAME 2>/dev/null | sed 's/^/P /'
    lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT 2>/dev/null | sed 's/^/T /'
}
# --- end of the node helpers
EOF

# The first command on the node: who we are, what the machine is, what is already there.
IFS= read -r -d '' CMD_FACTS <<'EOF' || true
echo "user=$(id -un)"
if sudo -n true 2>/dev/null; then echo "sudo=ok"; else echo "sudo=password"; fi
echo "os=$(lsb_release -ds 2>/dev/null || echo unknown)"
echo "kernel=$(uname -r)"
echo "cpus=$(nproc 2>/dev/null || echo unknown)"
echo "mem_gb=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}')"
if command -v hl-smi >/dev/null 2>&1; then echo "cards=$(hl-smi -Q index,name -f csv,noheader 2>/dev/null | grep -c .)"; else echo "cards=none"; fi
echo "bundle=$(sed -n 's/^plugin sha: //p' "$R/BUNDLE.txt" 2>/dev/null | head -n 1)"
for n in host smoke-pytorch smoke-fused smoke-hc smoke-moe run; do echo "job.$n=$(fn_job_state "$n")"; done
echo "--- disks"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT 2>&1
EOF
CMD_FACTS=${CMD_FACTS%"$NL"}

# shellcheck disable=SC2016  # $R is the node's
CMD_BUNDLE_SHA='sed -n "s/^plugin sha: //p" "$R/BUNDLE.txt" 2>/dev/null | head -n 1'

# ------------------------------------------------------------------ remote commands and copies

REMOTE_OUT=""
REMOTE_ERR=""
REMOTE_RC=0
CONNECTED=0
ALLOW_SWITCH=0

# The helpers are quoted once (quoting 5 KB costs bash 3.2 some 40 ms); two quoted strings side by
# side are one word for the node's shell.
PRELUDE_Q="$(rq "$PRELUDE")"

remote_cmd() {
    printf 'bash -c %s%s\n' "$PRELUDE_Q" "$(rq "$1")"
}

# The clock starts at the first successful connection of the session and is kept in the state file.
connected() {
    if [ "$CONNECTED" -eq 1 ]; then return 0; fi
    CONNECTED=1
    if [ "$MODE" != status ] && [ -z "$(state_get clock_start)" ]; then
        state_set clock_start "$(date -u +%s)"
    fi
}

# connection_fatal TEXT — exits on a connection error that no retry can fix. Returns 0 after it
# moved to a fresh session (the caller tries again at once), 1 for an error worth a retry.
connection_fatal() {
    case "$1" in
        *"HOST IDENTIFICATION HAS CHANGED"* | *"Host key verification failed"*)
            if [ "$ALLOW_SWITCH" -eq 1 ] && [ "$SESSION_AUTO" -eq 1 ]; then
                fresh_session
                return 0
            fi
            warn "error: the host key of $IP is not the one this session first saw ($KNOWN_HOSTS): another node answers at this address now (the provider reuses addresses), and nothing of this session is true for it."
            warn "for a new rental at this address, start a new session:"
            warn "  $RERUN --session $(rq "$SESSION-new")"
            exit 1
            ;;
        *"Permission denied ("*)
            warn "error: $TARGET refused the key $SSH_KEY: $(last_line "$1")"
            warn "check --user and --key. A key with a passphrase must be loaded into the agent first: this command never asks for one."
            exit 1
            ;;
    esac
    return 1
}

# remote SCRIPT — run SCRIPT on the node as one quoted string. Sets REMOTE_OUT, REMOTE_ERR and
# REMOTE_RC (the script's exit code). A failed connection is retried with a growing, bounded delay,
# SSH_TRIES times; after that the command stops and says so.
remote() {
    local script="$1" tries=0 delay="$RETRY_S" out="" rc=0 err="" started=0 now=0
    REMOTE_OUT=""
    REMOTE_ERR=""
    REMOTE_RC=0
    if [ "$DRY_RUN" -eq 1 ]; then
        say_lines "$script" "[dry-run] remote: "
        return 0
    fi
    while :; do
        # shellcheck disable=SC2029  # the command is built here on purpose: one quoted string for the node
        if out="$(ssh "${SSH_OPTS[@]}" "$TARGET" "$(remote_cmd "$script")" </dev/null 2>"$SSH_ERR")"; then rc=0; else rc=$?; fi
        err="$(cat "$SSH_ERR" 2>/dev/null || true)"
        if [ "$rc" -ne 255 ]; then break; fi
        if connection_fatal "$err"; then continue; fi
        tries=$((tries + 1))
        now="$(date -u +%s)"
        if [ "$started" -eq 0 ]; then started="$now"; fi
        if [ "$tries" -ge "$SSH_TRIES" ]; then
            warn "error: cannot reach $TARGET: $tries attempts over $((now - started)) s failed ($(last_line "$err"))."
            warn "nothing on the node was changed by this; when the node answers again, run the same command:"
            warn "  $RERUN"
            exit 1
        fi
        warn "connection lost, retrying in $delay s (outage so far: $((now - started)) s): $(last_line "$err")"
        nap "$delay"
        delay="$(next_delay "$delay")"
    done
    if [ "$started" -ne 0 ]; then say "connection is back after $(($(date -u +%s) - started)) s"; fi
    connected
    case "$out" in
        *"$BEGIN_MARK") out="" ;;
        *"$BEGIN_MARK$NL"*) out=${out#*"$BEGIN_MARK$NL"} ;;
    esac
    REMOTE_OUT="$out"
    REMOTE_ERR="$err"
    REMOTE_RC="$rc"
}

# copy SRC DST — one file to or from the node; a failure is retried a few times.
copy() {
    local tries=0 delay="$RETRY_S" err=""
    if [ "$DRY_RUN" -eq 1 ]; then
        say "[dry-run] local: scp $(rq "${SSH_OPTS[@]}" "$1" "$2")"
        return 0
    fi
    while :; do
        if scp "${SSH_OPTS[@]}" "$1" "$2" </dev/null >/dev/null 2>"$SSH_ERR"; then return 0; fi
        err="$(cat "$SSH_ERR" 2>/dev/null || true)"
        connection_fatal "$err" || true
        tries=$((tries + 1))
        if [ "$tries" -ge 3 ]; then
            warn "copy failed after $tries attempts: $1 -> $2 ($(last_line "$err"))"
            return 1
        fi
        warn "copy failed, retrying in $delay s: $(last_line "$err")"
        nap "$delay"
        delay="$(next_delay "$delay")"
    done
}

# ------------------------------------------------------------------ jobs on the node

JOB_ID=""
JOB_RC=""
JOB_END=""
OUTAGE_START=0
FOLLOW_BAD=0

# job_start NAME COMMAND [ID] — start COMMAND as the detached job NAME, or attach to the running
# one. Sets JOB_ID (on attach: the id the running job was started with).
job_start() {
    local name="$1" cmd="$2" id="${3:--}" rest=""
    JOB_ID="$id"
    if [ "$DRY_RUN" -eq 1 ]; then
        say "[dry-run] remote, as the detached job '$name' (log ~/rental/.jobs/$name.log, exit code ~/rental/.jobs/$name.rc), then followed: $cmd"
        return 0
    fi
    remote "fn_job_start $(rq "$name" "$cmd" "$id")"
    if [ "$REMOTE_RC" -ne 0 ]; then
        die "cannot start the job '$name' on the node: $REMOTE_OUT $REMOTE_ERR"
    fi
    case "$REMOTE_OUT" in
        attached*)
            rest="${REMOTE_OUT#attached }"
            case "$rest" in
                *" "*) JOB_ID="${rest#* }" ;;
            esac
            say "job $name: already running on the node (pid ${rest%% *}): attaching to it, not starting a second one"
            ;;
        *)
            say "job $name: started on the node, detached from this connection (log ~/rental/.jobs/$name.log)"
            ;;
    esac
}

# show_job_line TEXT — one line of a job's log; the lines that decide something are marked.
show_job_line() {
    local text=""
    text=${1//"$CR"/}
    case "$text" in
        *"no first token after "*) mark ">>> NO FIRST TOKEN >>> $text" ;;
        *"first token after "*) mark ">>> FIRST TOKEN >>> $text" ;;
        "phase: "* | "RUN "*": PASSED"* | "RUN "*": FAILED at "* | *": PASSED after "*" s"* | *": FAILED after "*" s"* | \
            *": TIMED_OUT after "*" s"* | *": SKIPPED after "*" s"* | "container exit code: "* | "PASTE THIS FILE: "* | \
            "SEND THIS FILE: "* | "tiny smoke"* | "host step "*)
            mark ">>> $text"
            ;;
        *) say "$text" ;;
    esac
}

alive() {
    FOLLOW_BAD=0
    if [ "$OUTAGE_START" -ne 0 ]; then
        say "connection is back after $(($(date -u +%s) - OUTAGE_START)) s"
        OUTAGE_START=0
    fi
}

# job_follow NAME — stream the job's new log lines until it ends; sets JOB_RC (its exit code), or
# JOB_END when it ended without one. A dropped connection is retried for as long as it takes, with
# a growing delay capped at RETRY_MAX_S and one line per attempt; the lines already shown are not
# shown again (the copy in jobs/NAME.log counts them).
job_follow() {
    local name="$1" mirror="$SESSION/jobs/$1.log" genf="$SESSION/jobs/$1.gen"
    local from=1 gen="" line="" got="" begun=0 delay="$RETRY_S" rc=0 err="" now=0
    JOB_RC=""
    JOB_END=""
    OUTAGE_START=0
    FOLLOW_BAD=0
    FOLLOWING="$name"
    while :; do
        if [ ! -f "$mirror" ]; then : >"$mirror"; fi
        from=$(($(wc -l <"$mirror") + 1))
        gen="$(cat "$genf" 2>/dev/null || true)"
        rm -f "$FIFO"
        mkfifo "$FIFO"
        # shellcheck disable=SC2029  # as in remote(): built here, one quoted string for the node
        ssh "${SSH_OPTS[@]}" "$TARGET" "$(remote_cmd "fn_job_follow $(rq "$name" "$from" "${gen:-none}" "$FOLLOW_S" "$HEARTBEAT_S")")" </dev/null >"$FIFO" 2>"$SSH_ERR.follow" &
        FOLLOW_PID=$!
        got=""
        begun=0
        while IFS= read -r line; do
            if [ "$begun" -eq 0 ]; then
                if [ "$line" = "$BEGIN_MARK" ]; then begun=1; fi
                continue
            fi
            case "$line" in
                "L "*)
                    alive
                    printf '%s\n' "${line#L }" >>"$mirror"
                    show_job_line "${line#L }"
                    ;;
                "L")
                    alive
                    echo >>"$mirror"
                    say ""
                    ;;
                "G "*)
                    alive
                    printf '%s\n' "${line#G }" >"$genf"
                    : >"$mirror"
                    ;;
                "H") alive ;;
                "E "*)
                    JOB_RC="${line#E }"
                    got=end
                    ;;
                "X "*)
                    JOB_END="${line#X }"
                    got=end
                    ;;
            esac
        done <"$FIFO"
        if wait "$FOLLOW_PID" 2>/dev/null; then rc=0; else rc=$?; fi
        FOLLOW_PID=""
        rm -f "$FIFO"
        if [ "$got" = end ]; then break; fi
        err="$(cat "$SSH_ERR.follow" 2>/dev/null || true)"
        connection_fatal "$err" || true
        if [ "$rc" -ne 255 ]; then
            FOLLOW_BAD=$((FOLLOW_BAD + 1))
            if [ "$FOLLOW_BAD" -ge 3 ]; then
                die "following the job '$name' fails on the node (exit $rc): $(last_line "$err"). The job itself is not touched; see: $SELF $IP --status"
            fi
        fi
        now="$(date -u +%s)"
        if [ "$OUTAGE_START" -eq 0 ]; then OUTAGE_START="$now"; fi
        warn "connection lost, retrying in $delay s (outage so far: $((now - OUTAGE_START)) s): $(last_line "$err")"
        nap "$delay"
        delay="$(next_delay "$delay")"
    done
    FOLLOWING=""
}

# ------------------------------------------------------------------ options

need_value() {
    if [ "$2" -lt 2 ]; then
        usage_error "$1 needs a value"
    fi
}

usage_error() {
    echo "error: $1" >&2
    usage >&2
    exit 2
}

ORIG_ARGS=("$@")
MODE=run
DRY_RUN=0
SSH_USER="${FN_RENTAL_SSH_USER:-}"
SSH_KEY="${FN_RENTAL_SSH_KEY:-}"
SESSION=""
SESSION_AUTO=0
BUNDLE=""
VLLM_SRC="${FN_VLLM_SRC:-}"
ALLOW_DIRTY=0
DATA_DISK=""
MODEL="/models/Qwen3.8-Flash-Next-FP8"
NO_DOWNLOAD=0
SMOKE_MODES="pytorch,fused"
SKIP_SMOKE=0
SMOKE_ARGS=()
SMOKE_OVERRIDE=""
VARIANTS="A,C"
QUICK_GATES=1
HC_FUSED=0
MOE_GLUE=0
RUN_ARGS=()
FROM_PHASE=""
NEW_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --user)
            need_value "$1" $#
            SSH_USER="$2"
            shift 2
            ;;
        --key)
            need_value "$1" $#
            SSH_KEY="$2"
            shift 2
            ;;
        --session)
            need_value "$1" $#
            SESSION="$2"
            shift 2
            ;;
        --rate)
            need_value "$1" $#
            RATE="$2"
            shift 2
            ;;
        --bundle)
            need_value "$1" $#
            BUNDLE="$2"
            shift 2
            ;;
        --vllm-src)
            need_value "$1" $#
            VLLM_SRC="$2"
            shift 2
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
            shift
            ;;
        --data-disk)
            need_value "$1" $#
            DATA_DISK="$2"
            shift 2
            ;;
        --model)
            need_value "$1" $#
            MODEL="$2"
            shift 2
            ;;
        --no-download)
            NO_DOWNLOAD=1
            shift
            ;;
        --smoke-modes)
            need_value "$1" $#
            SMOKE_MODES="$2"
            shift 2
            ;;
        --skip-smoke)
            SKIP_SMOKE=1
            shift
            ;;
        --smoke-arg)
            need_value "$1" $#
            SMOKE_ARGS+=("$2")
            shift 2
            ;;
        --continue-after-smoke-failure)
            need_value "$1" $#
            SMOKE_OVERRIDE="$2"
            shift 2
            ;;
        --variants)
            need_value "$1" $#
            VARIANTS="$2"
            shift 2
            ;;
        --no-quick-gates)
            QUICK_GATES=0
            shift
            ;;
        --hc-fused)
            HC_FUSED=1
            shift
            ;;
        --moe-glue)
            MOE_GLUE=1
            shift
            ;;
        --run-arg)
            need_value "$1" $#
            RUN_ARGS+=("$2")
            shift 2
            ;;
        --from)
            need_value "$1" $#
            FROM_PHASE="$2"
            shift 2
            ;;
        --new-run)
            NEW_RUN=1
            shift
            ;;
        --status)
            MODE=status
            shift
            ;;
        --stop)
            MODE=stop
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            REMIND=0
            exit 0
            ;;
        -*)
            usage_error "unknown option: $1"
            ;;
        *)
            if [ -n "$IP" ]; then usage_error "unexpected argument: $1"; fi
            IP="$1"
            shift
            ;;
    esac
done

if [ -z "$IP" ]; then usage_error "the node's IP address is missing"; fi
case "$IP" in
    *[!A-Za-z0-9.:_-]*) usage_error "not an address: $IP" ;;
esac
case "$RATE" in
    '' | *[!0-9.]* | *.*.* | .) usage_error "--rate takes dollars per hour, got '$RATE'" ;;
esac
case "$MODEL" in
    /*/*) ;;
    *) usage_error "--model must be an absolute path below a folder of its own, got '$MODEL'" ;;
esac
SMOKE_MODE_WORDS=""
OLD_IFS="$IFS"
IFS=','
for word in $SMOKE_MODES; do
    case "$word" in
        pytorch | fused | hc | moe) SMOKE_MODE_WORDS="$SMOKE_MODE_WORDS${SMOKE_MODE_WORDS:+ }$word" ;;
        *)
            IFS="$OLD_IFS"
            usage_error "--smoke-modes: unknown mode '$word' (the modes: pytorch, fused, hc, moe)"
            ;;
    esac
done
IFS="$OLD_IFS"
if [ -z "$SMOKE_MODE_WORDS" ]; then usage_error "--smoke-modes is empty (use --skip-smoke for no smoke test)"; fi
SMOKE_OVERRIDE_WORDS=""
IFS=','
for word in $SMOKE_OVERRIDE; do
    case "$word" in
        fused | hc | moe) SMOKE_OVERRIDE_WORDS="$SMOKE_OVERRIDE_WORDS${SMOKE_OVERRIDE_WORDS:+ }$word" ;;
        *)
            IFS="$OLD_IFS"
            usage_error "--continue-after-smoke-failure takes fused, hc and moe only: a failed pytorch mode means no variant can start, and nothing overrides it (got '$SMOKE_OVERRIDE')"
            ;;
    esac
done
IFS="$OLD_IFS"
for word in ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}; do
    case "$word" in
        --run-id | --model | --download-only | --download | --dry-run | --results | --from | --variants | --quick-gates)
            usage_error "--run-arg $word: this command sets that itself (see --model, --variants, --from, --no-quick-gates, --no-download)"
            ;;
    esac
done
if [ -n "$FROM_PHASE" ] && [ "$NEW_RUN" -eq 1 ]; then usage_error "--from resumes the last run and --new-run starts another: pass one"; fi

SSH_USER="${SSH_USER:-sesterce}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/sesterce_ed25519}"
TARGET="$SSH_USER@$IP"
MODEL_PARENT="$(dirname "$MODEL")"
RERUN="bash $(rq "$SELF")"
if [ "${#ORIG_ARGS[@]}" -gt 0 ]; then RERUN="$RERUN $(rq "${ORIG_ARGS[@]}")"; fi

# ------------------------------------------------------------------ the session folder

SESSION_TODAY=""
KNOWN_HOSTS=""
SSH_ERR=""

choose_session() {
    local base="${FN_RENTAL_SESSIONS:-$HOME/rental-sessions}" ip_safe="" newest=""
    ip_safe="$(printf '%s' "$IP" | tr ':' '-')"
    SESSION_TODAY="$base/$ip_safe-$(date -u +%Y-%m-%d)"
    if [ -n "$SESSION" ]; then
        case "$SESSION" in
            /*) ;;
            *) SESSION="$PWD/$SESSION" ;;
        esac
        return 0
    fi
    SESSION_AUTO=1
    # The newest folder of this address: today's, or an earlier day's whose state is still young
    # (a session that began before midnight UTC is the same rental after it).
    # shellcheck disable=SC2012  # ls for its order by time; the names are an address and a date
    newest="$(ls -1dt "$base/$ip_safe"-* 2>/dev/null | head -n 1 || true)"
    if [ -n "$newest" ] && [ -d "$newest" ]; then
        case "$newest" in
            "$SESSION_TODAY"*)
                SESSION="$newest"
                return 0
                ;;
        esac
        if [ -n "$(find "$newest/state" -mmin "-$SESSION_MAX_AGE_MIN" 2>/dev/null || true)" ]; then
            SESSION="$newest"
            return 0
        fi
    fi
    SESSION="$SESSION_TODAY"
}

# The paths of the session; with MAKE=1 the folder is created and the log opened.
open_session() {
    KNOWN_HOSTS="$SESSION/known_hosts"
    SSH_ERR="$SESSION/.connection-errors"
    FIFO="$SESSION/.follow.fifo"
    ssh_opts
    if [ "$DRY_RUN" -eq 1 ]; then
        FIFO=""
        return 0
    fi
    mkdir -p "$SESSION/jobs" "$SESSION/results"
    LOG="$SESSION/session.log"
    STATE="$SESSION/state"
    log_line "command: $RERUN"
}

# Another node answers at the address than the one the chosen session knows: leave that session as
# it is and start a fresh one, with its own empty known-hosts file.
fresh_session() {
    local old="$SESSION" n=2 new="$SESSION_TODAY"
    while [ -e "$new" ]; do
        new="$SESSION_TODAY-$n"
        n=$((n + 1))
    done
    warn "note: another node answers at $IP than the one of the session $old (the host key differs; the provider reuses addresses). Starting the fresh session $new."
    SESSION="$new"
    CONNECTED=0
    open_session
}

# ------------------------------------------------------------------ stage 1: connect

FACTS=""
NODE_BUNDLE=""
JOB_RUNNING=""

fact() {
    local line
    while IFS= read -r line; do
        case "$line" in
            "--- disks") return 0 ;;
            "$1="*)
                printf '%s\n' "${line#*=}"
                return 0
                ;;
        esac
    done <<<"$FACTS"
}

stage_connect() {
    local fp="" cards="" name="" st="" jobs="" table="" marker=""
    stage_start 1 connect
    if [ "$DRY_RUN" -eq 1 ]; then
        remote "$CMD_FACTS"
        say "[dry-run] local: ssh-keygen -l -f $(rq "$KNOWN_HOSTS")   # the host key fingerprint, printed once"
        return 0
    fi
    ALLOW_SWITCH=1
    remote "$CMD_FACTS"
    ALLOW_SWITCH=0
    FACTS="$REMOTE_OUT"
    if [ "$REMOTE_RC" -ne 0 ] || [ -z "$(fact user)" ]; then
        stage_failed "the first command on the node failed (exit $REMOTE_RC)"
        say_lines "$REMOTE_OUT" "  "
        say_lines "$REMOTE_ERR" "  "
        exit 1
    fi
    if [ -z "$(state_get hostkey)" ]; then
        fp="$(ssh-keygen -l -f "$KNOWN_HOSTS" 2>/dev/null | head -n 1 || true)"
        fp="${fp:-not recorded}"
        say "host key: $fp  (trusted for this session only: $KNOWN_HOSTS)"
        state_set hostkey "$fp"
    fi
    cards="$(fact cards)"
    case "$cards" in
        none) cards="Gaudi driver not installed yet" ;;
        *) cards="$cards Gaudi cards (driver present)" ;;
    esac
    say "node: $(fact user)@$IP: $(fact os), kernel $(fact kernel), $(fact cpus) CPUs, $(fact mem_gb) GB RAM, $cards"
    NODE_BUNDLE="$(fact bundle)"
    say "bundle on the node: ${NODE_BUNDLE:-none}"
    for name in host smoke-pytorch smoke-fused smoke-hc smoke-moe run; do
        st="$(fact "job.$name")"
        case "$st" in
            none | '') ;;
            running*)
                jobs="$jobs${jobs:+, }$name $st"
                JOB_RUNNING="${JOB_RUNNING:-$name}"
                ;;
            *) jobs="$jobs${jobs:+, }$name $st" ;;
        esac
    done
    if [ -n "$jobs" ]; then say "jobs on the node: $jobs"; fi
    marker="--- disks$NL"
    table=${FACTS#*"$marker"}
    say "disks:"
    say_lines "$table" "  "
    if [ "$(fact sudo)" != ok ]; then
        stage_failed "sudo asks for a password"
        say "the stages run prepare_host.sh, tiny_smoke.sh and run_rental.sh as root without a prompt (sudo -n); give $(fact user) passwordless sudo, then run the same command again."
        exit 1
    fi
    stage_ok
}

# ------------------------------------------------------------------ stage 2: bundle

REPO_ROOT=""
HEAD_SHA=""
TREE_DIRTY=0
BUNDLE_SHA=""
BUNDLE_CHANGED=0

bundle_sha_of() {
    local text
    text="$(tar xzOf "$1" BUNDLE.txt 2>/dev/null || tar xzOf "$1" ./BUNDLE.txt 2>/dev/null || true)"
    first_value "plugin sha: " "$text"
}

repo_facts() {
    REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -z "$REPO_ROOT" ]; then return 1; fi
    HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [ -n "$(git --no-optional-locks -C "$REPO_ROOT" status --porcelain)" ]; then TREE_DIRTY=1; else TREE_DIRTY=0; fi
}

build_command() {
    BUILD_CMD=(bash "$SCRIPT_DIR/make_rental_bundle.sh" --out "${REPO_ROOT:-<checkout>}/dist")
    if [ -n "$VLLM_SRC" ]; then BUILD_CMD+=(--vllm-src "$VLLM_SRC"); fi
    if [ "$ALLOW_DIRTY" -eq 1 ]; then BUILD_CMD+=(--allow-dirty); fi
}

cmd_unpack() {
    printf '%s\n' "mkdir -p \"\$R\" && tar xzf \"\$HOME\"/flash-next-rental-$1.tar.gz -C \"\$R\" && cat \"\$R/BUNDLE.txt\""
}

stage_bundle() {
    local want="" file="" have="" out="" short="" forced=0
    stage_start 2 bundle
    if [ "$DRY_RUN" -eq 1 ]; then
        file="$BUNDLE"
        short="SHA12"
        if [ -z "$BUNDLE" ]; then
            repo_facts || true
            build_command
            say "[dry-run] local: $(rq "${BUILD_CMD[@]}")   # refused when the tree is not clean, without --allow-dirty"
            if [ -n "$HEAD_SHA" ]; then short="${HEAD_SHA:0:12}"; fi
            file="${REPO_ROOT:-<checkout>}/dist/flash-next-rental-$short.tar.gz"
        fi
        remote "$CMD_BUNDLE_SHA   # the copy is skipped when the node has this bundle"
        copy "$file" "$(node_path "flash-next-rental-$short.tar.gz")"
        remote "$(cmd_unpack "$short")"
        return 0
    fi
    if [ -n "$JOB_RUNNING" ]; then
        BUNDLE_SHA="$NODE_BUNDLE"
        stage_ok "skipped: the job '$JOB_RUNNING' is running on the node, and its bundle is left alone"
        return 0
    fi
    case "$(fact job.run)" in
        ended*)
            if [ "$NEW_RUN" -eq 0 ] && [ -z "$FROM_PHASE" ]; then
                BUNDLE_SHA="$NODE_BUNDLE"
                if [ -z "$BUNDLE" ] && repo_facts; then
                    if [ "$TREE_DIRTY" -eq 1 ]; then
                        say "note: this checkout is at ${HEAD_SHA:0:12} with uncommitted changes; the node ran ${NODE_BUNDLE:0:12}."
                    elif [ "$HEAD_SHA" != "$NODE_BUNDLE" ]; then
                        say "note: this checkout is at ${HEAD_SHA:0:12}; the node ran ${NODE_BUNDLE:0:12}."
                    fi
                fi
                stage_ok "skipped: the run on the node has ended, and the bundle it ran (${NODE_BUNDLE:0:12}) is left there; --new-run sends this checkout and runs again"
                return 0
            fi
            ;;
    esac
    if [ -n "$BUNDLE" ]; then
        if [ ! -f "$BUNDLE" ]; then
            stage_failed "no file $BUNDLE"
            exit 1
        fi
        file="$BUNDLE"
        want="$(bundle_sha_of "$file")"
        if [ -z "$want" ]; then
            stage_failed "$BUNDLE is not a rental bundle: no BUNDLE.txt with a 'plugin sha:' line in it"
            exit 1
        fi
    else
        if ! repo_facts; then
            stage_failed "$SCRIPT_DIR is not in a git checkout: pass --bundle FILE"
            exit 1
        fi
        want="$HEAD_SHA"
        if [ "$TREE_DIRTY" -eq 1 ] && [ "$ALLOW_DIRTY" -eq 1 ]; then forced=1; fi
    fi
    remote "$CMD_BUNDLE_SHA"
    have="$REMOTE_OUT"
    if [ "$have" = "$want" ] && [ "$forced" -eq 0 ]; then
        if [ -z "$BUNDLE" ] && [ "$TREE_DIRTY" -eq 1 ]; then
            say "note: the working tree has uncommitted changes; they are NOT on the node, which has commit ${want:0:12}. Commit and run the same command to send them."
        fi
        BUNDLE_SHA="$want"
        state_set bundle_sha "$want"
        stage_ok "the node already has bundle ${want:0:12}"
        return 0
    fi
    if [ -z "$BUNDLE" ]; then
        if [ "$TREE_DIRTY" -eq 1 ] && [ "$ALLOW_DIRTY" -eq 0 ]; then
            stage_failed "the working tree is not clean"
            say "the node has ${have:-no bundle} and this checkout is at ${want:0:12} with uncommitted changes: commit first, or run again with --allow-dirty (the three top-level scripts then come from the working tree, plugin/ from the commit)."
            exit 1
        fi
        build_command
        say "building the bundle: $(rq "${BUILD_CMD[@]}")"
        if out="$("${BUILD_CMD[@]}" 2>&1)"; then
            say_lines "$out" "  "
        else
            stage_failed "make_rental_bundle.sh failed"
            say_lines "$out" "  "
            say "the build needs a clean tree and a vLLM checkout that has the engine pin (--vllm-src PATH, or FN_VLLM_SRC); a bundle built earlier goes in with --bundle FILE. Then run the same command."
            exit 1
        fi
        file="$(first_value "bundle: " "$out")"
        if [ -z "$file" ] || [ ! -f "$file" ]; then
            stage_failed "make_rental_bundle.sh printed no 'bundle: FILE' line"
            exit 1
        fi
        want="$(bundle_sha_of "$file")"
    fi
    short="${want:0:12}"
    say "copying $file to the node ($(du -h "$file" | cut -f1 | tr -d ' '))"
    if ! copy "$file" "$(node_path "flash-next-rental-$short.tar.gz")"; then
        stage_failed "the bundle did not reach the node"
        exit 1
    fi
    remote "$(cmd_unpack "$short")"
    say_lines "$REMOTE_OUT" "  "
    have="$(first_value "plugin sha: " "$REMOTE_OUT")"
    if [ "$REMOTE_RC" -ne 0 ] || [ "$have" != "$want" ]; then
        stage_failed "unpacking the bundle on the node failed (exit $REMOTE_RC)"
        say_lines "$REMOTE_ERR" "  "
        exit 1
    fi
    if [ -n "$NODE_BUNDLE" ] && [ "$NODE_BUNDLE" != "$want" ]; then BUNDLE_CHANGED=1; fi
    BUNDLE_SHA="$want"
    state_set bundle_sha "$want"
    stage_ok "bundle $short unpacked in ~/rental"
}

# ------------------------------------------------------------------ stage 3: host

HOSTFACTS=""
CKPT=""
CKPT_KB=0
CKPT_CONTAINER=none
CKPT_RC=0
D_N=0
PAIR=""
DISK_USE=""
DISK_WHY=""
DISK_STOP=""
DISK_CANDS=""

host_cards() {
    local n
    n="$(hfact cards)"
    case "$n" in
        '' | *[!0-9]*) n=0 ;;
    esac
    echo "$n"
}

hfact() {
    local line
    while IFS= read -r line; do
        case "$line" in
            "$1="*)
                printf '%s\n' "${line#*=}"
                return 0
                ;;
        esac
    done <<<"$HOSTFACTS"
}

# kv LINE KEY — the value of KEY in a "key=value key=value" line.
kv() {
    local s=" $1"
    case "$s" in
        *" $2="*)
            s=${s#*" $2="}
            printf '%s\n' "${s%% *}"
            ;;
    esac
}

# ckpt_verdict LINE — the line of fn_ckpt as one word in CKPT:
# complete | downloading | failed (the download container ended badly) | partial | absent.
ckpt_verdict() {
    local config="" inc=""
    CKPT_CONTAINER="$(kv "$1" container)"
    CKPT_RC="$(kv "$1" rc)"
    CKPT_KB="$(kv "$1" kb)"
    config="$(kv "$1" config)"
    inc="$(kv "$1" incomplete)"
    case "$CKPT_KB" in
        '' | *[!0-9]*) CKPT_KB=0 ;;
    esac
    case "$CKPT_CONTAINER" in
        running | created | restarting | paused)
            CKPT=downloading
            return 0
            ;;
        exited | dead)
            if [ "$CKPT_RC" != 0 ]; then
                CKPT=failed
                return 0
            fi
            if [ "$config" = yes ]; then
                CKPT=complete
                return 0
            fi
            ;;
    esac
    if [ "$config" = yes ] && [ "$inc" = no ] && [ $((CKPT_KB * 1024)) -ge $((CKPT_MIN_GB * 1000000000)) ]; then
        CKPT=complete
    elif [ "$CKPT_KB" -gt 0 ]; then
        CKPT=partial
    else
        CKPT=absent
    fi
}

kb_gb() {
    awk -v kb="$1" 'BEGIN { printf "%.1f", kb * 1024 / 1e9 }'
}

# pair LINE KEY — the value of KEY="..." in a line of lsblk -P, into PAIR. No eval: the text comes
# from the node.
pair() {
    local s=" $1"
    case "$s" in
        *" $2=\""*)
            s=${s#*" $2=\""}
            PAIR="${s%%\"*}"
            ;;
        *) PAIR="" ;;
    esac
}

parse_disks() {
    local line
    D_N=0
    D_NAME=()
    D_TYPE=()
    D_SIZE=()
    D_FS=()
    D_LABEL=()
    D_MNT=()
    D_PK=()
    while IFS= read -r line; do
        case "$line" in
            "P "*) line="${line#P }" ;;
            *) continue ;;
        esac
        pair "$line" NAME
        D_NAME[D_N]="$PAIR"
        pair "$line" TYPE
        D_TYPE[D_N]="$PAIR"
        pair "$line" SIZE
        case "$PAIR" in
            '' | *[!0-9]*) PAIR=0 ;;
        esac
        D_SIZE[D_N]="$PAIR"
        pair "$line" FSTYPE
        D_FS[D_N]="$PAIR"
        pair "$line" LABEL
        D_LABEL[D_N]="$PAIR"
        pair "$line" MOUNTPOINT
        D_MNT[D_N]="$PAIR"
        pair "$line" PKNAME
        D_PK[D_N]="$PAIR"
        D_N=$((D_N + 1))
    done <<<"$HOSTFACTS"
}

has_children() {
    local j=0
    while [ "$j" -lt "$D_N" ]; do
        if [ "${D_PK[$j]}" = "$1" ]; then return 0; fi
        j=$((j + 1))
    done
    return 1
}

# decide_data_disk FREE_GB DIR — sets DISK_USE (a device, or empty for none) and DISK_WHY, or
# DISK_STOP (why nothing may be chosen) with DISK_CANDS. It only reads what the node listed;
# formatting is prepare_host.sh's, which checks the device again as root (blkid, wipefs).
decide_data_disk() {
    local free_gb="$1" dir="$2" i=0 n=0 name="" found=-1
    DISK_USE=""
    DISK_WHY=""
    DISK_STOP=""
    DISK_CANDS=""
    parse_disks
    if [ -n "$DATA_DISK" ]; then
        name="${DATA_DISK#/dev/}"
        while [ "$i" -lt "$D_N" ]; do
            if [ "${D_NAME[$i]}" = "$name" ]; then found="$i"; fi
            i=$((i + 1))
        done
        if [ "$D_N" -eq 0 ]; then
            # no listing to check against (lsblk gave nothing this command can read): the named
            # device is handed on, and prepare_host.sh checks it as root before it formats anything
            DISK_USE="/dev/$name"
            DISK_WHY="named with --data-disk; the node's lsblk listing could not be read here, so prepare_host.sh alone checks that it is blank (it refuses a device with a filesystem, partitions or a mount point)"
        elif [ "$found" -lt 0 ]; then
            DISK_STOP="/dev/$name is not a disk the node lists"
        elif [ "${D_TYPE[$found]}" != disk ] && [ "${D_TYPE[$found]}" != part ]; then
            DISK_STOP="/dev/$name is not a disk or partition (lsblk type '${D_TYPE[$found]}')"
        elif [ "${D_FS[$found]}" = ext4 ] && [ "${D_LABEL[$found]}" = "$FS_LABEL" ]; then
            DISK_USE="/dev/$name"
            DISK_WHY="named with --data-disk; it already holds the $FS_LABEL filesystem prepare_host.sh made, which is mounted, not formatted"
        elif has_children "$name"; then
            DISK_STOP="/dev/$name has partitions"
        elif [ -n "${D_FS[$found]}" ]; then
            DISK_STOP="/dev/$name has a filesystem (${D_FS[$found]})"
        elif [ -n "${D_MNT[$found]}" ]; then
            DISK_STOP="/dev/$name is mounted at ${D_MNT[$found]}"
        else
            DISK_USE="/dev/$name"
            DISK_WHY="named with --data-disk; blank: no partitions, no filesystem, not mounted"
        fi
        return 0
    fi
    if [ "$NO_DOWNLOAD" -eq 1 ]; then
        DISK_WHY="no data disk is needed: --no-download says the checkpoint is in place"
        return 0
    fi
    if [ "$CKPT" = complete ]; then
        DISK_WHY="no data disk is needed: the checkpoint is complete already ($(kb_gb "$CKPT_KB") GB in $MODEL)"
        return 0
    fi
    if [ "$free_gb" -ge "$DOWNLOAD_NEED_GB" ]; then
        DISK_WHY="no data disk is needed: $dir has $free_gb GB for the checkpoint (free space and what is already downloaded), and the download needs $DOWNLOAD_NEED_GB GB"
        return 0
    fi
    i=0
    while [ "$i" -lt "$D_N" ]; do
        if [ "${D_TYPE[$i]}" = disk ] && [ -z "${D_FS[$i]}" ] && [ -z "${D_MNT[$i]}" ] &&
            [ "${D_SIZE[$i]}" -ge $((DISK_MIN_GB * 1000000000)) ] && ! has_children "${D_NAME[$i]}"; then
            n=$((n + 1))
            found="$i"
            DISK_CANDS="$DISK_CANDS${DISK_CANDS:+ }/dev/${D_NAME[$i]}"
        fi
        i=$((i + 1))
    done
    if [ "$n" -eq 1 ]; then
        DISK_USE="/dev/${D_NAME[$found]}"
        DISK_WHY="($((D_SIZE[found] / 1000000000)) GB) chosen automatically: it is the only blank disk of at least $DISK_MIN_GB GB (no partitions, no filesystem, not mounted), and $dir has only $free_gb GB free of the $DOWNLOAD_NEED_GB GB the checkpoint download needs"
    elif [ "$n" -eq 0 ]; then
        DISK_STOP="$dir has only $free_gb GB free of the $DOWNLOAD_NEED_GB GB the checkpoint download needs, and there is no blank disk of at least $DISK_MIN_GB GB (no partitions, no filesystem, not mounted) to choose"
    else
        DISK_STOP="$dir has only $free_gb GB free of the $DOWNLOAD_NEED_GB GB the checkpoint download needs, and there are $n blank disks of at least $DISK_MIN_GB GB ($DISK_CANDS): which one is yours to choose"
    fi
}

cmd_host() {
    local tail=""
    if [ -n "$1" ]; then tail=" $(rq --data-disk "$1" --mount "$MODEL_PARENT")"; fi
    printf '%s\n' "cd \"\$HOME/rental\" && exec sudo -n bash prepare_host.sh$tail"
}

host_facts() {
    remote "fn_host_facts $(rq "$MODEL")"
    HOSTFACTS="$REMOTE_OUT"
    ckpt_verdict "$(hfact ckpt)"
}

# Free space for the checkpoint, in GB: the free space of the model folder's filesystem plus what
# is already downloaded there.
space_gb() {
    local free_kb
    free_kb="$(hfact free_kb)"
    case "$free_kb" in
        '' | *[!0-9]*) free_kb=0 ;;
    esac
    echo $(((free_kb + CKPT_KB) * 1024 / 1000000000))
}

show_disk_table() {
    local line
    while IFS= read -r line; do
        case "$line" in
            "T "*) say "  ${line#T }" ;;
        esac
    done <<<"$HOSTFACTS"
}

stage_host() {
    local free_gb=0 dir="" cand="" lines="" enough=0
    stage_start 3 host
    if [ "$DRY_RUN" -eq 1 ]; then
        remote "fn_host_facts $(rq "$MODEL")   # the runtime, the cards, the free space, the disks (lsblk)"
        say "[dry-run] the data disk: --data-disk when given; else the only blank disk of at least $DISK_MIN_GB GB; none when $MODEL_PARENT has $DOWNLOAD_NEED_GB GB free; in every other case the command stops here"
        if [ -n "$DATA_DISK" ]; then
            job_start host "$(cmd_host "/dev/${DATA_DISK#/dev/}")"
        else
            job_start host "$(cmd_host "/dev/THE-BLANK-DISK")"
        fi
        return 0
    fi
    case "$(fact job.host)" in
        running*)
            say "job host: already running on the node: attaching to it, not starting a second one"
            ;;
        *)
            host_facts
            free_gb="$(space_gb)"
            dir="$(hfact space_dir)"
            enough=0
            if [ "$NO_DOWNLOAD" -eq 1 ] || [ "$CKPT" = complete ] || [ "$CKPT" = downloading ] || [ "$free_gb" -ge "$DOWNLOAD_NEED_GB" ]; then
                enough=1
            fi
            if [ "$(state_get host)" = "done" ] && [ "$(hfact runtime)" = yes ] && [ "$(host_cards)" -gt 0 ] && [ "$enough" -eq 1 ]; then
                stage_ok "already prepared, checked on the node: the habana runtime is registered, hl-smi lists $(host_cards) cards, $dir has room"
                return 0
            fi
            decide_data_disk "$free_gb" "$dir"
            if [ -n "$DISK_STOP" ]; then
                stage_failed "no data disk chosen; nothing was changed on the node"
                say "$DISK_STOP."
                if [ -n "$DATA_DISK" ]; then
                    say "this command never formats a disk that has a filesystem or partitions, or is mounted."
                fi
                say "the disks of the node:"
                show_disk_table
                say "run again with the data disk named (prepare_host.sh formats it only when it is blank):"
                if [ -n "$DISK_CANDS" ] && [ -z "$DATA_DISK" ]; then
                    for cand in $DISK_CANDS; do say "  $RERUN --data-disk $cand"; done
                else
                    say "  $RERUN --data-disk /dev/DEVICE"
                fi
                say "or with --model DIR on a filesystem with $DOWNLOAD_NEED_GB GB free, or with --no-download when the checkpoint is in place."
                exit 3
            fi
            if [ -n "$DISK_USE" ]; then
                say "data disk: $DISK_USE $DISK_WHY"
                state_set data_disk "$DISK_USE"
            else
                say "data disk: $DISK_WHY"
            fi
            job_start host "$(cmd_host "$DISK_USE")"
            ;;
    esac
    job_follow host
    if [ "$JOB_RC" != 0 ]; then
        lines="$(grep -i '^remedy:.*reboot' "$SESSION/jobs/host.log" 2>/dev/null || true)"
        if [ -n "$lines" ]; then
            stage_failed "prepare_host.sh asks for a reboot"
            say_lines "$(grep -i -E '^(remedy|note):.*reboot|^host step .*FAILED' "$SESSION/jobs/host.log" 2>/dev/null || true)" "  "
            say "this command never reboots the node. Reboot it yourself (on the node: sudo reboot), wait for it to answer, then run the same command; prepare_host.sh goes on where it stopped:"
            say "  $RERUN"
            exit 3
        fi
        stage_failed "prepare_host.sh ended with ${JOB_RC:+exit $JOB_RC}${JOB_END:+no exit code (job $JOB_END)}"
        lines="$(grep -E '^(remedy:|host step .*(FAILED|REFUSED))' "$SESSION/jobs/host.log" 2>/dev/null || true)"
        say_lines "$lines" "  "
        say "its output: $SESSION/jobs/host.log (step logs on the node: ~/rental/prepare-host-logs/). After the remedy, run the same command:"
        say "  $RERUN"
        exit 1
    fi
    host_facts
    if [ "$(hfact runtime)" != yes ] || [ "$(host_cards)" -le 0 ]; then
        stage_failed "prepare_host.sh ended with exit 0, but the node check does not agree: habana runtime registered: $(hfact runtime), cards in hl-smi: $(host_cards)"
        exit 1
    fi
    state_set host "done"
    stage_ok "$(host_cards) cards, $(hfact space_dir) has $(space_gb) GB for the checkpoint"
}

# ------------------------------------------------------------------ stages 4 and 6: the download

ckpt_now() {
    remote "fn_ckpt $(rq "$MODEL")"
    ckpt_verdict "$REMOTE_OUT"
}

cmd_download() {
    printf '%s\n' "cd \"\$R\" && sudo -n bash run_rental.sh $(rq --model "$MODEL" --download-only) 2>&1"
}

stage_download() {
    stage_start 4 download
    if [ "$NO_DOWNLOAD" -eq 1 ]; then
        stage_ok "skipped: --no-download"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        remote "fn_ckpt $(rq "$MODEL")   # no start when the checkpoint is complete or the download is running"
        remote "$(cmd_download)"
        return 0
    fi
    ckpt_now
    case "$CKPT" in
        complete)
            stage_ok "the checkpoint is complete already: $(kb_gb "$CKPT_KB") GB in $MODEL"
            return 0
            ;;
        downloading)
            stage_ok "the download is already running in the container fn-download: $(kb_gb "$CKPT_KB") GB so far"
            return 0
            ;;
        failed) say "the earlier download ended with exit code $CKPT_RC; starting it again (it resumes in $MODEL)" ;;
    esac
    remote "$(cmd_download)"
    say_lines "$REMOTE_OUT" "  "
    if [ "$REMOTE_RC" -ne 0 ]; then
        stage_failed "run_rental.sh --download-only ended with exit $REMOTE_RC"
        say_lines "$REMOTE_ERR" "  "
        say "nothing is downloading. When the message is about free space, name a data disk (--data-disk DEV) or another --model DIR; then run the same command:"
        say "  $RERUN"
        exit 1
    fi
    stage_ok "started, not waited for: stage 6 waits"
}

stage_wait() {
    local prev_kb="" prev_t=0 now=0 line="" waited=0
    stage_start 6 wait
    if [ "$NO_DOWNLOAD" -eq 1 ]; then
        stage_ok "skipped: --no-download"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        remote "fn_ckpt $(rq "$MODEL")   # every $POLL_S s: bytes on disk, rate, ETA; until the container fn-download has ended"
        remote "sudo -n docker logs --tail 30 fn-download 2>&1   # only when it ended with an error"
        return 0
    fi
    while :; do
        ckpt_now
        now="$(date -u +%s)"
        case "$CKPT" in
            complete) break ;;
            failed)
                stage_failed "the download container fn-download ended with exit code $CKPT_RC"
                remote "sudo -n docker logs --tail 30 fn-download 2>&1"
                say "the last lines of its log:"
                say_lines "$REMOTE_OUT" "  "
                say "$(kb_gb "$CKPT_KB") GB are in $MODEL. The run was not started. Run the same command to start the download again (it resumes):"
                say "  $RERUN"
                exit 1
                ;;
            downloading)
                if [ -n "$prev_kb" ] && [ "$now" -gt "$prev_t" ]; then
                    line="$(awk -v kb="$CKPT_KB" -v pk="$prev_kb" -v dt="$((now - prev_t))" -v total="$CKPT_GB" 'BEGIN {
                        rate = (kb - pk) * 1024 / 1e9 / dt; left = total - kb * 1024 / 1e9; if (left < 0) left = 0
                        if (rate > 0) printf "%.2f GB/s, ETA %dm%02ds", rate, int(left / rate / 60), int(left / rate) % 60
                        else printf "0.00 GB/s, ETA unknown" }')"
                    say "download: $(kb_gb "$CKPT_KB") GB on disk of about $CKPT_GB GB, $line"
                elif [ -z "$prev_kb" ]; then
                    say "download: $(kb_gb "$CKPT_KB") GB on disk of about $CKPT_GB GB, measuring the rate"
                fi
                if [ -z "$prev_kb" ] || [ "$now" -gt "$prev_t" ]; then
                    prev_kb="$CKPT_KB"
                    prev_t="$now"
                fi
                ;;
            *)
                stage_failed "no download is running and the checkpoint is not complete"
                say "$MODEL holds $(kb_gb "$CKPT_KB") GB (container fn-download: $CKPT_CONTAINER). Run the same command to start the download again:"
                say "  $RERUN"
                exit 1
                ;;
        esac
        waited=1
        nap "$POLL_S"
    done
    if [ "$waited" -eq 1 ]; then
        stage_ok "the checkpoint is complete: $(kb_gb "$CKPT_KB") GB in $MODEL"
    else
        stage_ok "the checkpoint is complete already: $(kb_gb "$CKPT_KB") GB in $MODEL"
    fi
}

# ------------------------------------------------------------------ stage 5: smoke

cmd_smoke() {
    local args=(--gdn "$1" --run-id "$2")
    if [ "$3" -eq 1 ]; then args+=(--reinstall); fi
    printf '%s\n' "cd \"\$HOME/rental\" && exec sudo -n bash tiny_smoke.sh $(rq "${args[@]}" ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"})"
}

# fetch_smoke ID — the smoke folder ID, without *.safetensors, into results/ID of the session.
fetch_smoke() {
    local id="$1" dest="$SESSION/results"
    remote "cd \"\$R/results\" && [ -d $(rq "$id") ] && sudo -n tar czf - --exclude='*.safetensors' $(rq "$id") >\"\$J/$id.tar.gz\""
    if [ "$REMOTE_RC" -ne 0 ]; then
        say "note: no folder results/$id on the node to fetch"
        return 1
    fi
    copy "$(node_path "rental/.jobs/$id.tar.gz")" "$dest/$id.tar.gz" || return 1
    tar xzf "$dest/$id.tar.gz" -C "$dest" || return 1
    rm -f "$dest/$id.tar.gz"
}

# overridden MODE — true when --continue-after-smoke-failure names MODE.
overridden() {
    case " $SMOKE_OVERRIDE_WORDS " in
        *" $1 "*) return 0 ;;
    esac
    return 1
}

smoke_report() {
    local mode="$1" id="$2" lines="" log="$SESSION/results/$2/$1/server0.log"
    lines="$(awk '/^tiny smoke.*(FAILED|TIMED OUT)/ { on = 1; print; next } on && /^    / { print; next } { on = 0 } /^error:/ { print }' \
        "$SESSION/jobs/smoke-$mode.log" 2>/dev/null || true)"
    if [ -z "$lines" ]; then lines="$(tail -n 20 "$SESSION/jobs/smoke-$mode.log" 2>/dev/null || true)"; fi
    say "what tiny_smoke.sh printed:"
    say_lines "$lines" "  "
    if [ -f "$log" ]; then
        say "server log: $log"
    else
        say "server log: none in the fetched folder (the server may not have started); the console of the call: $SESSION/jobs/smoke-$mode.log"
    fi
    say "the smoke folder, without *.safetensors: $SESSION/results/$id"
}

stage_smoke() {
    local stamp="" mode="" v="" st="" sha="" id="" rest="" reinstall=0 stack="" over="" note=""
    stage_start 5 smoke
    if [ "$SKIP_SMOKE" -eq 1 ]; then
        stage_ok "skipped: --skip-smoke"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        for mode in $SMOKE_MODE_WORDS; do
            job_start "smoke-$mode" "$(cmd_smoke "$mode" "smoke-UTC-$mode" 0)"
        done
        say "[dry-run] a failed mode stops the command here, before the run (the fused, hc and moe modes can be overridden, the pytorch mode never)"
        return 0
    fi
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    for mode in $SMOKE_MODE_WORDS; do
        v="$(state_get "smoke_$mode")"
        st="${v%%:*}"
        rest="${v#*:}"
        sha="${rest%%:*}"
        id="${rest#*:}"
        case "$(fact "job.smoke-$mode")" in
            running*)
                remote "cat \"\$J/smoke-$mode.id\" 2>/dev/null"
                id="${REMOTE_OUT:-smoke-$stamp-$mode}"
                say "job smoke-$mode: already running on the node: attaching to it, not starting a second one"
                st=running
                ;;
            ended*)
                # started by this session with this bundle, ended while nobody watched: its result counts
                remote "cat \"\$J/smoke-$mode.id\" 2>/dev/null"
                if [ -n "$REMOTE_OUT" ] && [ "$REMOTE_OUT" != "$id" ] && [ "$BUNDLE_CHANGED" -eq 0 ] &&
                    [ "$(state_get smoke_stack_sha)" = "$BUNDLE_SHA" ]; then
                    id="$REMOTE_OUT"
                    say "job smoke-$mode: it ended on the node while nobody watched (results/$id): reading its result, not running it again"
                    st=running
                fi
                ;;
        esac
        if [ "$st" != running ] && [ -n "$v" ] && [ "$sha" = "$BUNDLE_SHA" ]; then
            if [ "$st" = passed ]; then
                remote "[ -f \"\$R/results/\"$(rq "$id")/summary.json ]"
                if [ "$REMOTE_RC" -eq 0 ]; then
                    say "smoke $mode: already passed with this bundle, and its result is still on the node (results/$id)"
                    continue
                fi
            elif [ "$st" = failed ] && overridden "$mode"; then
                remote "[ -d \"\$R/results/\"$(rq "$id") ]"
                if [ "$REMOTE_RC" -eq 0 ]; then
                    say "smoke $mode: failed before with this bundle (results/$id); not run again, overridden by --continue-after-smoke-failure $mode"
                    over="$over${over:+, }$mode"
                    continue
                fi
            fi
        fi
        if [ "$st" != running ]; then
            id="smoke-$stamp-$mode"
            reinstall=0
            stack="$(state_get smoke_stack_sha)"
            if [ -n "$stack" ] && [ "$stack" != "$BUNDLE_SHA" ]; then reinstall=1; fi
            job_start "smoke-$mode" "$(cmd_smoke "$mode" "$id" "$reinstall")" "$id"
            id="$JOB_ID"
            state_set smoke_stack_sha "$BUNDLE_SHA"
        fi
        job_follow "smoke-$mode"
        if [ "$JOB_RC" = 0 ]; then
            state_set "smoke_$mode" "passed:$BUNDLE_SHA:$id"
            continue
        fi
        state_set "smoke_$mode" "failed:$BUNDLE_SHA:$id"
        fetch_smoke "$id" || true
        if overridden "$mode"; then
            say "smoke $mode: FAILED, overridden by --continue-after-smoke-failure $mode"
            smoke_report "$mode" "$id"
            over="$over${over:+, }$mode"
            continue
        fi
        stage_failed "the $mode mode failed: ${JOB_RC:+exit $JOB_RC}${JOB_END:+no exit code, job $JOB_END}"
        smoke_report "$mode" "$id"
        say "the run was NOT started, and the download goes on by itself. Fix the failure (a new commit is sent by the same command), then run it again:"
        say "  $RERUN"
        case "$mode" in
            fused)
                say "a failed fused mode means the run would work on the PyTorch GDN path; to start the run all the same:"
                say "  $RERUN --continue-after-smoke-failure fused"
                ;;
            hc)
                say "a failed hc mode means the run would go on with the torch hyper-connections; to start the run all the same:"
                say "  $RERUN --continue-after-smoke-failure hc"
                ;;
            moe)
                say "a failed moe mode means the run would go on with the torch path of the MoE blocks; to start the run all the same:"
                say "  $RERUN --continue-after-smoke-failure moe"
                ;;
        esac
        exit 1
    done
    if [ -n "$over" ]; then
        note="FAILED and overridden: $over. The run goes on without what failed there: the PyTorch GDN path for fused, the torch hyper-connections for hc, the torch path of the MoE blocks for moe (its own gates decide)"
    fi
    stage_ok "$note"
}

# ------------------------------------------------------------------ stage 7: run

RUN_ID=""
RUN_FAILED=0
RAN_NOW=0

cmd_run() {
    local args=(--model "$MODEL" --run-id "$1")
    if [ "$QUICK_GATES" -eq 1 ]; then args+=(--quick-gates); fi
    args+=(--variants "$VARIANTS")
    if [ "$HC_FUSED" -eq 1 ]; then args+=(--hc-fused); fi
    if [ "$MOE_GLUE" -eq 1 ]; then args+=(--moe-glue); fi
    if [ "${#RUN_ARGS[@]}" -gt 0 ]; then args+=("${RUN_ARGS[@]}"); fi
    if [ -n "$FROM_PHASE" ]; then args+=(--from "$FROM_PHASE"); fi
    printf '%s\n' "cd \"\$HOME/rental\" && exec sudo -n bash run_rental.sh $(rq "${args[@]}")"
}

run_ended() {
    state_set run_state "ended:${JOB_RC:-lost}"
    if [ "$JOB_RC" = 0 ]; then
        stage_ok "run $RUN_ID"
    elif [ -n "$JOB_RC" ]; then
        stage_failed "run $RUN_ID: run_rental.sh ended with exit $JOB_RC; its results are fetched next"
        RUN_FAILED=1
    else
        stage_failed "run $RUN_ID: the job ended without an exit code (job $JOB_END: the node rebooted, or the job was killed); what it left is fetched next"
        RUN_FAILED=1
    fi
}

# need_run_option OPTION — stage 7 stops here when the run_rental.sh of the bundle on the node does
# not know OPTION, one of the switches this command passes on to it (--hc-fused, --moe-glue).
need_run_option() {
    remote "grep -q -e $1 \"\$R/run_rental.sh\""
    if [ "$REMOTE_RC" -ne 0 ]; then
        stage_failed "$1: the run_rental.sh of the bundle on the node does not know this option"
        say "send a bundle that has it (a commit with $1 in run_rental.sh), or run the same command without $1."
        exit 1
    fi
}

stage_run() {
    local st="" prev="" id="" node_id="" rc=""
    stage_start 7 run
    if [ "$DRY_RUN" -eq 1 ]; then
        if [ "$HC_FUSED" -eq 1 ]; then remote "grep -q -e --hc-fused \"\$R/run_rental.sh\"   # the bundle must know the option"; fi
        if [ "$MOE_GLUE" -eq 1 ]; then remote "grep -q -e --moe-glue \"\$R/run_rental.sh\"   # the bundle must know the option"; fi
        job_start run "$(cmd_run "UTC-TIMESTAMP")"
        say "[dry-run] the run takes hours; it goes on when this command or the connection ends, and the same command attaches to it again"
        return 0
    fi
    prev="$(state_get run_state)"
    id="$(state_get run_id)"
    remote "echo \"\$(fn_job_state run)|\$(cat \"\$J/run.id\" 2>/dev/null)\""
    st="${REMOTE_OUT%%|*}"
    node_id="${REMOTE_OUT#*|}"
    case "$st" in
        running*)
            say "job run: already running on the node (${st#running } is its pid): attaching to it, not starting a second one"
            JOB_ID="${node_id:-unknown}"
            ;;
        *)
            if [ "$NEW_RUN" -eq 0 ] && [ -z "$FROM_PHASE" ]; then
                case "$st" in
                    ended*)
                        # ended on the node while nobody watched, or a run this folder does not know
                        if [ "$prev" != "ended:${st#ended }" ] || [ "$id" != "$node_id" ]; then
                            RUN_ID="$node_id"
                            state_set run_id "$RUN_ID"
                            state_set fetch ""
                            RAN_NOW=1
                            say "run $RUN_ID ended on the node while nobody watched: reading its end, not starting another (--new-run does that)."
                            job_follow run
                            run_ended
                            return 0
                        fi
                        ;;
                esac
                case "$prev" in
                    ended:*)
                        remote "[ -f \"\$R/results/$id.tar.gz\" ] || [ -d \"\$R/results/$id\" ]"
                        if [ "$REMOTE_RC" -eq 0 ]; then
                            RUN_ID="$id"
                            rc="${prev#ended:}"
                            if [ "$rc" = 0 ]; then
                                stage_ok "run $id already ended with exit 0, and its results are still on the node"
                            else
                                stage_failed "run $id already ended with exit $rc"
                                RUN_FAILED=1
                            fi
                            say "a run is not started twice by itself. To resume this one from a phase: add --from PHASE; to start another run: add --new-run."
                            return 0
                        fi
                        say "note: the results of run $id are not on the node any more (another machine?): a new run starts"
                        ;;
                esac
            fi
            if [ -n "$FROM_PHASE" ]; then
                id="${id:-$node_id}"
                if [ -z "$id" ]; then
                    stage_failed "--from $FROM_PHASE: this session has no run to resume"
                    exit 1
                fi
            else
                id="$(date -u +%Y%m%dT%H%M%SZ)"
            fi
            if [ "$HC_FUSED" -eq 1 ]; then need_run_option --hc-fused; fi
            if [ "$MOE_GLUE" -eq 1 ]; then need_run_option --moe-glue; fi
            job_start run "$(cmd_run "$id")" "$id"
            ;;
    esac
    RUN_ID="$JOB_ID"
    state_set run_id "$RUN_ID"
    state_set run_state started
    state_set fetch ""
    RAN_NOW=1
    say "run $RUN_ID: it takes hours and needs neither this command nor the connection; Ctrl-C detaches, the same command attaches again."
    job_follow run
    run_ended
}

# ------------------------------------------------------------------ stage 8: fetch

cmd_fetch_pack() {
    local id="$1"
    printf '%s\n' "cd \"\$R\" || exit 1
for f in results/$id.tar.gz results/$id/summary.md results/$id/console.log; do if [ -f \"\$f\" ]; then echo \"have \$f\"; fi; done
if ls -d results/smoke-* >/dev/null 2>&1; then (cd results && sudo -n tar czf - --exclude='*.safetensors' smoke-*) >\"\$J/smoke-results.tar.gz\" && echo \"have .jobs/smoke-results.tar.gz\"; fi
if [ -d prepare-host-logs ]; then sudo -n tar czf - prepare-host-logs >\"\$J/prepare-host-logs.tar.gz\" && echo \"have .jobs/prepare-host-logs.tar.gz\"; fi
exit 0"
}

stage_fetch() {
    local id="$RUN_ID" dest="$SESSION/results" line="" rel="" to="" got="" missed=""
    stage_start 8 fetch
    if [ "$DRY_RUN" -eq 1 ]; then
        remote "$(cmd_fetch_pack "RUN-ID")"
        copy "$(node_path "rental/results/RUN-ID.tar.gz")" "$dest/RUN-ID.tar.gz"
        copy "$(node_path "rental/results/RUN-ID/summary.md")" "$dest/RUN-ID/summary.md"
        copy "$(node_path "rental/results/RUN-ID/console.log")" "$dest/RUN-ID/console.log"
        copy "$(node_path "rental/.jobs/smoke-results.tar.gz")" "$dest/smoke-results.tar.gz"
        copy "$(node_path "rental/.jobs/prepare-host-logs.tar.gz")" "$dest/prepare-host-logs.tar.gz"
        say "[dry-run] local: tar xzf $(rq "$dest/smoke-results.tar.gz" -C "$dest"); tar xzf $(rq "$dest/prepare-host-logs.tar.gz" -C "$dest"); cat $(rq "$dest/RUN-ID/summary.md")"
        return 0
    fi
    id="${id:-none}"
    if [ "$RAN_NOW" -eq 0 ] && [ "$(state_get fetch)" = "$id" ] && [ -f "$dest/$id.tar.gz" ]; then
        stage_ok "already here: $dest/$id.tar.gz and $dest/$id/summary.md"
        return 0
    fi
    remote "$(cmd_fetch_pack "$id")"
    if [ "$REMOTE_RC" -ne 0 ]; then
        stage_failed "no ~/rental on the node to fetch from (exit $REMOTE_RC)"
        exit 1
    fi
    while IFS= read -r line; do
        case "$line" in
            "have "*) rel="${line#have }" ;;
            *) continue ;;
        esac
        case "$rel" in
            results/*) to="$dest/${rel#results/}" ;;
            .jobs/*) to="$dest/${rel#.jobs/}" ;;
            *) continue ;;
        esac
        mkdir -p "$(dirname "$to")"
        if copy "$(node_path "rental/$rel")" "$to"; then
            got="$got$to$NL"
            case "$rel" in
                .jobs/*.tar.gz)
                    if ! tar xzf "$to" -C "$dest"; then say "note: $to did not unpack"; fi
                    ;;
            esac
        else
            missed="$missed${missed:+, }$rel"
        fi
    done <<<"$REMOTE_OUT"
    if [ -f "$dest/$id/summary.md" ]; then
        say "----- summary.md of run $id (the file to paste) -----"
        say_lines "$(cat "$dest/$id/summary.md")"
        say "----- end of summary.md -----"
    else
        say "note: run ${id} left no summary.md on the node"
    fi
    say "fetched into $dest:"
    got=${got%"$NL"}
    say_lines "$got" "  "
    if [ -f "$dest/smoke-results.tar.gz" ] || [ -f "$dest/prepare-host-logs.tar.gz" ]; then
        say "smoke-results.tar.gz and prepare-host-logs.tar.gz are unpacked next to them (the smoke folders hold no *.safetensors)."
    fi
    if [ -n "$missed" ]; then
        stage_failed "not copied: $missed"
        exit 1
    fi
    if [ ! -f "$dest/$id.tar.gz" ]; then
        if [ "$id" = none ]; then
            stage_ok "no run in this session; the smoke folders and the host logs are here"
        else
            stage_failed "the node has no results/$id.tar.gz"
            RUN_FAILED=1
        fi
        return 0
    fi
    state_set fetch "$id"
    stage_ok "send $dest/$id.tar.gz"
}

# ------------------------------------------------------------------ --status and --stop

do_status() {
    local line=""
    remote "for n in host smoke-pytorch smoke-fused smoke-hc smoke-moe run; do
    echo \"job \$n: \$(fn_job_state \"\$n\")\"
    if [ -f \"\$J/\$n.log\" ]; then tail -n 5 \"\$J/\$n.log\" | sed 's/^/    /'; fi
done
echo \"ckpt \$(fn_ckpt $(rq "$MODEL"))\""
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    if [ "$REMOTE_RC" -ne 0 ]; then die "the node did not answer the status command (exit $REMOTE_RC): $REMOTE_ERR"; fi
    say "node $TARGET, session $SESSION"
    while IFS= read -r line; do
        case "$line" in
            "ckpt "*)
                ckpt_verdict "${line#ckpt }"
                say "download: checkpoint $CKPT, $(kb_gb "$CKPT_KB") GB in $MODEL (container fn-download: $CKPT_CONTAINER)"
                ;;
            *) say "$line" ;;
        esac
    done <<<"$REMOTE_OUT"
}

do_stop() {
    local st="" node_id=""
    remote "echo \"\$(fn_job_state run)|\$(cat \"\$J/run.id\" 2>/dev/null)\""
    if [ "$DRY_RUN" -eq 1 ]; then
        remote "p=\"\$(cat \"\$J/run.cpid\" 2>/dev/null)\"; [ -n \"\$p\" ] && sudo -n kill -INT \"\$p\" && echo \"signalled \$p\""
        stage_fetch
        return 0
    fi
    st="${REMOTE_OUT%%|*}"
    node_id="${REMOTE_OUT#*|}"
    RUN_ID="$(state_get run_id)"
    RUN_ID="${RUN_ID:-$node_id}"
    case "$st" in
        running*)
            RUN_ID="${node_id:-$RUN_ID}"
            remote "p=\"\$(cat \"\$J/run.cpid\" 2>/dev/null)\"; [ -n \"\$p\" ] && sudo -n kill -INT \"\$p\" && echo \"signalled \$p\""
            if [ "$REMOTE_RC" -ne 0 ]; then
                die "could not signal the run on the node (exit $REMOTE_RC): $REMOTE_OUT $REMOTE_ERR. By hand, on the node: sudo docker stop fn-rental"
            fi
            say "asked run $RUN_ID to stop: SIGINT to run_rental.sh ($REMOTE_OUT). It stops its servers, writes the report and the tarball; that takes up to about two minutes."
            say "its output follows. If it goes on for longer than that, the signal did not arrive; then, on the node: sudo docker stop fn-rental"
            state_set run_id "$RUN_ID"
            state_set fetch ""
            RAN_NOW=1
            job_follow run
            state_set run_state "ended:${JOB_RC:-lost}"
            say "the run has ended (exit ${JOB_RC:-unknown})."
            ;;
        *)
            say "no run is running on the node (job run: $st): nothing to stop."
            ;;
    esac
    stage_fetch
}

# ------------------------------------------------------------------ main

choose_session
open_session

if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry-run] nothing is contacted and nothing is written. The commands of a session on a fresh node, in order."
    say "[dry-run] every 'remote:' command would be sent as ONE argument: ssh $(rq "${SSH_OPTS[@]}" "$TARGET") bash -c '<node helpers> <command>'"
    say "[dry-run] session folder: $SESSION"
else
    if [ ! -f "$SSH_KEY" ]; then
        die "no SSH key at $SSH_KEY (--key FILE, or FN_RENTAL_SSH_KEY)"
    fi
fi

case "$MODE" in
    status)
        do_status
        exit 0
        ;;
    stop)
        do_stop
        exit 0
        ;;
esac

stage_connect
case "$(fact job.run)" in
    running*)
        say "the run is in progress on the node: stages 2 to 6 are behind it, attaching."
        BUNDLE_SHA="$NODE_BUNDLE"
        ;;
    *)
        stage_bundle
        stage_host
        stage_download
        stage_smoke
        stage_wait
        ;;
esac
stage_run
stage_fetch

if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry-run] the node helpers, sent in front of every remote command:"
    PRELUDE=${PRELUDE%"$NL"}
    say_lines "$PRELUDE" "[dry-run]   "
    exit 0
fi
if [ "$RUN_FAILED" -eq 1 ]; then
    say "the session is over, and the run did not pass: read summary.md above."
    exit 1
fi
say "the session is complete."
exit 0
