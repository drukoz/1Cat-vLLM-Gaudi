#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# cardq.sh: the card queue of the rented Gaudi node. Every use of a card goes through it.
#
#   cardq.sh submit --track T --name NAME --cards SPEC [--max-hold DUR] [--replaces ID]
#                   [--script FILE | -- COMMAND...]
#       Queue a job. Without --script and without "--" the job's script is read from standard input
#       (the safe form over SSH: nothing is quoted twice). Prints the job id on standard output.
#       --replaces ID: the new job takes the place in the queue of the track's own job ID, which is
#       cancelled if it still waits (or was cancelled by the track in the last 90 minutes); the jobs
#       that replace one job may hold at most as many cards in total as it asked for.
#       SPEC is "all", "any:N" (N cards chosen by the queue; any:2 and any:4 give one module group),
#       or a list such as "3,4,5,6".
#       DUR is seconds, or a number with s, m or h (default 45m).
#   cardq.sh wait ID [--timeout S]   wait for the job's end; the exit code is the job's
#                                    (124: --timeout passed, 125: cancelled before it ran, 126: lost)
#   cardq.sh status                  who holds which card, who waits, the last jobs that ended
#   cardq.sh log ID [-f]             the job's log; -f follows it until the job ends
#   cardq.sh cancel --track T ID     stop a job of track T (a track stops only its own jobs)
#   cardq.sh ledger                  one line per ended job: track, cards, start, end, seconds
#   cardq.sh health                  the habanalabs kernel lines since boot per card and class, and the
#                                    jobs that had some
#   cardq.sh bump --by coord ID [WHY]  (the coordinator only) move a waiting job to the front
#   cardq.sh place --by coord NEW OLD  (the coordinator only) the waiting job NEW takes the place of
#                                    the job OLD its track cancelled (as submit --replaces would have)
#   cardq.sh adopt                   (after an update of this file) the waiting jobs get supervisors
#                                    that run it, so its rules apply to them too
#
# How it works. One lock file per card, held with flock(1) by the job's supervisor for as long as
# the job runs, so a dead supervisor frees its cards by itself. Locks are only ever tried, never
# waited for, and tried in ascending card order under one short queue lock, so two jobs cannot
# deadlock. Jobs start in submission order; a later job starts ahead of an earlier waiting one only
# on cards that job cannot use yet and only when its own maximum hold ends before that job can start.
# Two exceptions. Fairness: a job that wants all cards for more than 45 minutes (CARDQ_LONG_HOLD_S)
# counts as submitted 10 minutes (CARDQ_LONG_DELAY_S) later, so shorter jobs submitted in those 10
# minutes start before it (for jobs submitted from 2026-11-20 17:55 UTC on). Bump: a job the
# coordinator bumped starts before every other waiting job.
#
# Cards and modules. A card number is the card's index (hl-smi's, the one HABANA_VISIBLE_DEVICES and
# /dev/accel/accel<N> take). Its module id can differ (on the 2026-11-20 node index 0..7 is module
# 3,0,1,6,7,4,5,2), and Habana supports partial workloads on the module groups 0,1 2,3 4,5 6,7 and
# 0,1,2,3 4,5,6,7 only, so any:2 and any:4 hand out the cards of one such group. The map comes from
# CARDQ_MODULE_MAP (the module of card 0, 1, ..., comma-separated) or from $Q/module_map, which is
# written from hl-smi on first use; without either, card N is module N.
#
# A job runs detached from the SSH connection (setsid, nohup) in its own session and process group,
# with CARDQ_JOB, CARDQ_TRACK, CARDQ_NAME, CARDQ_CARDS (the granted cards, for example 3,4,5,6),
# CARDQ_MODULES (their module ids, ascending: the HABANA_VISIBLE_MODULES a container of these cards
# sees), CARDQ_NCARDS, CARDQ_PORT_BASE (100 free ports from there), CARDQ_DEADLINE and CARDQ_JOB_DIR set.
# At the maximum hold, or on cancel, the process group gets SIGTERM and, CARDQ_GRACE_S seconds
# later, SIGKILL. When a job ends, whatever is left of its process group is killed and every
# container labelled cardq.job=<id> is removed; a job that held every card and did not end by itself
# also loses every running container of the habana runtime. Nothing else is ever stopped.
#
# Health: when a job ends, the habanalabs kernel lines since its start (dmesg, read with sudo) are
# counted per PCI address and card, by class. Two classes follow from what a job did and are no card
# fault: "killed while waiting" (user process got signal while waiting for interrupt: a process was
# killed while it waited for the card) and "failure analysis" (Received unexpected user error interrupt:
# Synapse started its device failure analysis in a process of the job, after a signal that process does
# not handle or 600 s without progress; measured 8 ms apart, twice, on 2026-11-20). Every other line is
# an ERROR. All of them go to the job's log and to $Q/health.tsv; the lines of the job's own cards go to
# its end line and ledger reason ("ok; habanalabs: card 1 (module 0, 0000:0b:00.0) 2 lines (1 failure
# analysis, 1 killed while waiting)", "habanalabs ERROR: ..." when one is an error). Routine lines (link
# renames, IB registration) are left out.
#
# State: $CARDQ_DIR (default ~/farm/q): jobs/<id>/, locks/, ledger.tsv, health.tsv. Plain bash 3.2 and later.

set -u
set -o pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
Q="${CARDQ_DIR:-$HOME/farm/q}"
NC="${CARDQ_NCARDS:-8}"
POLL_S="${CARDQ_POLL_S:-2}"
GRACE_S="${CARDQ_GRACE_S:-120}"
MARGIN_S="${CARDQ_MARGIN_S:-15}"
DEFAULT_HOLD_S="${CARDQ_DEFAULT_HOLD_S:-2700}"
HOLD_CAP_S="${CARDQ_HOLD_CAP_S:-10800}"
LONG_HOLD_S="${CARDQ_LONG_HOLD_S:-2700}"
LONG_DELAY_S="${CARDQ_LONG_DELAY_S:-600}"
PORT_BASE="${CARDQ_PORT_FIRST:-8100}"
INF=99999999999
FD_BASE=20

note() { printf '%s\n' "$*" >&2; }
die() {
    note "error: $*"
    exit "${DIE_RC:-1}"
}

# rd FILE: the first line of FILE in R ("" when there is none). No fork.
R=""
rd() {
    R=""
    if [ -f "$1" ]; then IFS= read -r R <"$1" 2>/dev/null || true; fi
    return 0
}

put() { printf '%s\n' "$2" >"$1.tmp.$$" && mv -f "$1.tmp.$$" "$1"; }

iso() {
    case "$1" in
        '' | *[!0-9]*) printf '%s\n' "-" ;;
        *) date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ ;;
    esac
}

dur() {
    local s="$1"
    if [ "$s" -ge 3600 ]; then
        printf '%dh%02dm%02ds\n' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
    elif [ "$s" -ge 60 ]; then
        printf '%dm%02ds\n' $((s / 60)) $((s % 60))
    else
        printf '%ds\n' "$s"
    fi
}

# parse_hold TEXT: seconds in HOLD, or return 1.
HOLD=0
parse_hold() {
    local t="$1" n unit=1
    case "$t" in
        *s) n="${t%s}" ;;
        *m) n="${t%m}" unit=60 ;;
        *h) n="${t%h}" unit=3600 ;;
        *) n="$t" ;;
    esac
    case "$n" in '' | *[!0-9]*) return 1 ;; esac
    HOLD=$((10#$n * unit))
    [ "$HOLD" -ge 1 ]
}

# parse_spec TEXT: the card set in SPEC ("all", "any:N", or an ascending list), or return 1.
SPEC=""
parse_spec() {
    local t="$1" n c seen=" " out=""
    case "$t" in
        all) SPEC=all ;;
        any:*)
            n="${t#any:}"
            case "$n" in '' | *[!0-9]*) return 1 ;; esac
            n=$((10#$n))
            if [ "$n" -lt 1 ] || [ "$n" -gt "$NC" ]; then return 1; fi
            if [ "$n" -eq "$NC" ]; then SPEC=all; else SPEC="any:$n"; fi
            ;;
        *)
            case "$t" in '' | *[!0-9,]* | ,* | *, | *,,*) return 1 ;; esac
            for ((c = 0; c < NC; c++)); do
                case ",$t," in
                    *",$c,"*) out="$out${out:+,}$c" ;;
                esac
            done
            for n in ${t//,/ }; do
                n=$((10#$n))
                if [ "$n" -ge "$NC" ]; then return 1; fi
                case "$seen" in *" $n "*) return 1 ;; esac
                seen="$seen$n "
            done
            SPEC="$out"
            ;;
    esac
    return 0
}

# norm_id TEXT: the job id in ID (five digits), or exit.
ID=""
norm_id() {
    case "${1:-}" in '' | *[!0-9]*) die "not a job id: '${1:-}'" ;; esac
    ID="$(printf '%05d' "$((10#$1))")"
    [ -d "$Q/jobs/$ID" ] || die "no job $ID under $Q/jobs"
}

qlock() {
    exec 9>>"$Q/queue.lock" || die "cannot open $Q/queue.lock"
    flock 9 || die "cannot take the queue lock"
}

# ------------------------------------------------------------------ cards and modules

MOD=() # MOD[card] = the card's module id
IDX=() # IDX[module] = the module's card

# set_map TEXT: MOD and IDX from "m0,m1,...", or return 1 when TEXT is not a permutation of 0..NC-1.
set_map() {
    local t="$1" i=0 m
    MOD=()
    IDX=()
    case "$t" in '' | *[!0-9,]* | ,* | *, | *,,*) return 1 ;; esac
    for m in ${t//,/ }; do
        m=$((10#$m))
        if [ "$i" -ge "$NC" ] || [ "$m" -ge "$NC" ] || [ -n "${IDX[m]:-}" ]; then return 1; fi
        MOD[i]="$m"
        IDX[m]="$i"
        i=$((i + 1))
    done
    [ "$i" -eq "$NC" ]
}

identity_map() {
    local c
    MOD=()
    IDX=()
    for ((c = 0; c < NC; c++)); do
        MOD[c]="$c"
        IDX[c]="$c"
    done
}

# pci_of CARD: the card's PCI address (from $Q/pci_map, written from hl-smi with the module map), in R.
pci_of() {
    local line
    R=""
    if [ -f "$Q/pci_map" ]; then
        while read -r line; do
            case "$line" in "$1 "*) R="${line#* }" ;; esac
        done <"$Q/pci_map"
    fi
}

# card_of_pci BUS: the card index of a PCI address, in R ("" when unknown).
card_of_pci() {
    local line
    R=""
    if [ -f "$Q/pci_map" ]; then
        while read -r line; do
            case "$line" in *" $1") R="${line%% *}" ;; esac
        done <"$Q/pci_map"
    fi
}

# map_from_hlsmi: the map as hl-smi reports it ("m0,m1,..."), or nothing. Also writes $Q/pci_map
# ("index bus" lines) once.
write_pci_map() {
    if [ ! -s "$Q/pci_map" ] && [ -d "$Q" ] && command -v hl-smi >/dev/null 2>&1; then
        hl-smi -Q index,bus_id -f csv,noheader 2>/dev/null | tr -d ' ' | sort -t, -k1,1n \
            | awk -F, '{ b = tolower($2); sub(/^0000:/, "", b); print $1 " 0000:" b }' >"$Q/pci_map.tmp.$$" \
            && mv -f "$Q/pci_map.tmp.$$" "$Q/pci_map"
        rm -f "$Q/pci_map.tmp.$$"
    fi
}

map_from_hlsmi() {
    command -v hl-smi >/dev/null 2>&1 || return 0
    hl-smi -Q index,module_id -f csv,noheader 2>/dev/null | tr -d ' ' | sort -t, -k1,1n | cut -d, -f2 | paste -sd, -
}

# load_map: MOD and IDX for this node (see the header). A bad map is reported and the identity used.
load_map() {
    local t="${CARDQ_MODULE_MAP:-}" from="CARDQ_MODULE_MAP"
    if [ -z "$t" ]; then
        from="$Q/module_map"
        if [ ! -s "$Q/module_map" ] && [ -d "$Q" ]; then
            t="$(map_from_hlsmi)"
            if [ -n "$t" ] && set_map "$t"; then put "$Q/module_map" "$t" 2>/dev/null || true; fi
        fi
        rd "$Q/module_map"
        t="$R"
    fi
    write_pci_map
    if [ -z "$t" ]; then
        identity_map
    elif ! set_map "$t"; then
        note "warning: $from is '$t', not the module id of each of the $NC cards: card N is taken as module N"
        identity_map
    fi
}

# modules_of "C C C": the module ids of those cards, ascending and comma-separated, in R.
modules_of() {
    local c m out=""
    for ((m = 0; m < NC; m++)); do
        for c in $1; do
            if [ "${MOD[c]}" = "$m" ]; then out="$out${out:+,}$m"; fi
        done
    done
    R="$out"
}

# cards_of_modules FIRST COUNT: the cards of modules FIRST..FIRST+COUNT-1, ascending, space-separated, in R.
cards_of_modules() {
    local first="$1" count="$2" c m out=""
    for ((c = 0; c < NC; c++)); do
        m="${MOD[c]}"
        if [ "$m" -ge "$first" ] && [ "$m" -lt $((first + count)) ]; then out="$out${out:+ }$c"; fi
    done
    R="$out"
}

qunlock() {
    flock -u 9 2>/dev/null
    exec 9>&-
}

# try_card N: 0 when this process now holds the lock of card N.
try_card() {
    local fd=$((FD_BASE + $1))
    eval "exec $fd>>\"\$Q/locks/card$1.lock\"" 2>/dev/null || return 1
    if flock -n "$fd" 2>/dev/null; then return 0; fi
    eval "exec $fd>&-"
    return 1
}

drop_card() {
    local fd=$((FD_BASE + $1))
    flock -u "$fd" 2>/dev/null
    eval "exec $fd>&-"
}

# sup_alive DIR: 0 while the job's supervisor lives (a job submitted in the last 30 s counts).
sup_alive() {
    local pid sub now
    rd "$1/pid"
    pid="$R"
    case "$pid" in
        '' | *[!0-9]*)
            rd "$1/submitted"
            sub="${R:-0}"
            now="$(date -u +%s)"
            [ $((now - sub)) -lt 30 ]
            return
            ;;
    esac
    kill -0 "$pid" 2>/dev/null
}

ledger() {
    # ledger ID TRACK NAME SPEC CARDS SUBMITTED START END SECONDS RC REASON
    if [ ! -s "$Q/ledger.tsv" ]; then
        printf 'id\ttrack\tname\twanted\tcards\tsubmitted\tstart\tend\tseconds\trc\treason\n' >>"$Q/ledger.tsv"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >>"$Q/ledger.tsv"
}

# ------------------------------------------------------------------ choosing cards

U=()
FREE=()
AVAIL=()
LIMIT=()
BLOCKS=()
PICK=""
BLOCK=""
SHADOW=0

# blocks_for SPEC: the card sets SPEC accepts, best first, in BLOCKS. Returns 1 for an "any:N"
# that has no fixed sets (N other than 1, 2, 4). A set of 2 or 4 is the cards of one module group
# (modules 0,1 2,3 4,5 6,7 and 0,1,2,3 4,5,6,7, the partial workloads Habana supports). Fours come
# from the bottom modules and ones and twos from the top, so small jobs leave a whole group free for
# a four-card job.
blocks_for() {
    local spec="$1" n i b
    BLOCKS=()
    case "$spec" in
        all)
            b=""
            for ((i = 0; i < NC; i++)); do b="$b $i"; done
            BLOCKS[0]="${b# }"
            ;;
        any:*)
            n="${spec#any:}"
            if [ "$n" -eq 1 ]; then
                for ((i = NC - 1; i >= 0; i--)); do BLOCKS[${#BLOCKS[@]}]="${IDX[i]}"; done
            elif [ "$n" -eq 2 ] && [ $((NC % 2)) -eq 0 ]; then
                for ((i = NC - 2; i >= 0; i -= 2)); do
                    cards_of_modules "$i" 2
                    BLOCKS[${#BLOCKS[@]}]="$R"
                done
            elif [ "$n" -eq 4 ] && [ $((NC % 4)) -eq 0 ]; then
                for ((i = 0; i < NC; i += 4)); do
                    cards_of_modules "$i" 4
                    BLOCKS[${#BLOCKS[@]}]="$R"
                done
            else
                return 1
            fi
            ;;
        *) BLOCKS[0]="${spec//,/ }" ;;
    esac
    return 0
}

# is_group "C C": 0 when the cards are exactly one module group of 2 or 4 (or all cards, or one card).
is_group() {
    local n=0 c b
    for c in $1; do n=$((n + 1)); done
    if [ "$n" -eq 1 ] || [ "$n" -eq "$NC" ]; then return 0; fi
    blocks_for "any:$n" || return 1
    for b in "${BLOCKS[@]}"; do
        if [ "$b" = "$1" ]; then return 0; fi
    done
    return 1
}

# pick_cards SPEC: the cards SPEC gets from the usable ones (U[c] = 1) in PICK, or return 1.
pick_cards() {
    local spec="$1" n b c ok got=0
    PICK=""
    if blocks_for "$spec"; then
        for b in "${BLOCKS[@]}"; do
            ok=1
            for c in $b; do
                if [ "${U[c]}" != 1 ]; then
                    ok=0
                    break
                fi
            done
            if [ "$ok" -eq 1 ]; then
                PICK="$b"
                return 0
            fi
        done
        return 1
    fi
    n="${spec#any:}"
    for ((c = 0; c < NC && got < n; c++)); do
        if [ "${U[c]}" = 1 ]; then
            PICK="$PICK $c"
            got=$((got + 1))
        fi
    done
    PICK="${PICK# }"
    if [ "$got" -lt "$n" ]; then
        PICK=""
        return 1
    fi
    return 0
}

# plan_block SPEC: for a job that cannot start now, the card set it is certain to get first (BLOCK)
# and the time by which that set is certain to be free (SHADOW), from AVAIL.
plan_block() {
    local spec="$1" b c t bt=0 best="" n
    if blocks_for "$spec"; then
        for b in "${BLOCKS[@]}"; do
            t=0
            for c in $b; do
                if [ "${AVAIL[c]}" -gt "$t" ]; then t="${AVAIL[c]}"; fi
            done
            if [ -z "$best" ] || [ "$t" -lt "$bt" ]; then
                best="$b"
                bt="$t"
            fi
        done
        BLOCK="$best"
        SHADOW="$bt"
        return 0
    fi
    n="${spec#any:}"
    BLOCK=""
    SHADOW=0
    while read -r t c; do
        BLOCK="$BLOCK $c"
        if [ "$t" -gt "$SHADOW" ]; then SHADOW="$t"; fi
    done < <(for ((c = 0; c < NC; c++)); do echo "${AVAIL[c]} $c"; done | sort -n -k1,1 -k2,2 | head -n "$n")
    BLOCK="${BLOCK# }"
}

# start_order: one "KEY ID" line per waiting job, in the order the jobs may start: by submission time,
# except that a job that wants every card for more than LONG_HOLD_S counts as submitted LONG_DELAY_S
# later, so shorter jobs submitted within that time go before it (and nothing waits more than that
# on its account).
start_order() {
    local d id sub spec hold key tie
    for d in "$Q"/jobs/*/; do
        d="${d%/}"
        rd "$d/state"
        [ "$R" = waiting ] || continue
        id="${d##*/}"
        rd "$d/submitted"
        sub="$R"
        case "$sub" in '' | *[!0-9]*) sub=0 ;; esac
        rd "$d/spec"
        spec="$R"
        rd "$d/maxhold"
        hold="${R:-$DEFAULT_HOLD_S}"
        key="$sub"
        # the rule holds for jobs submitted under it (a "rules" file), not for jobs queued before it
        if [ "$spec" = all ] && [ "$hold" -gt "$LONG_HOLD_S" ] && [ -f "$d/rules" ]; then
            key=$((sub + LONG_DELAY_S))
        fi
        tie="$id"
        rd "$d/place" # the place of the job this one replaces (submit --replaces), its id breaking ties
        case "$R" in
            '' | *[!0-9]*) ;;
            *)
                key="$R"
                rd "$d/replaces"
                case "$R" in [0-9]*) tie="$R" ;; esac
                ;;
        esac
        rd "$d/bump"
        case "${R%% *}" in
            '' | *[!0-9]*) ;;
            *) key=$((${R%% *} - 10000000000)) ;; # bumped: before every other job, the first bump first
        esac
        printf '%s %s %s\n' "$key" "$tie" "$id"
    done | sort -n -k1,1 -k2,2 -k3,3
}

# plan MYID: with the queue lock held and FREE[] filled, decide whether job MYID may start now.
# PICK holds its cards on return 0. Waiting jobs are walked in start order (start_order): one that can
# start claims its cards; one that cannot reserves the set it will get first from the time that set
# is free, and a later job may use a reserved card only if its own maximum hold ends before that time.
# With MYID "status" nothing is decided and one line per waiting job is printed instead.
plan() {
    local me="$1" now d id st spec hold total c dl cards
    now="$(date -u +%s)"
    for ((c = 0; c < NC; c++)); do
        LIMIT[c]="$INF"
        # A locked card that no running job claims below counts as free now: a waiting job then
        # reserves it from now on, and nobody slips in front of that job on its account.
        AVAIL[c]="$now"
    done
    for d in "$Q"/jobs/*/; do
        d="${d%/}"
        rd "$d/state"
        [ "$R" = running ] || continue
        rd "$d/deadline"
        dl="$R"
        rd "$d/cards"
        cards="$R"
        case "$dl" in '' | *[!0-9]*) continue ;; esac
        for c in ${cards//,/ }; do
            if [ "${FREE[c]}" != 1 ]; then AVAIL[c]=$((dl + GRACE_S + MARGIN_S)); fi
        done
    done
    while read -r _ _ id; do
        d="$Q/jobs/$id"
        rd "$d/state"
        st="$R"
        [ "$st" = waiting ] || continue
        if [ "$id" != "$me" ]; then
            if ! sup_alive "$d"; then
                if [ "$me" != status ]; then lose_job "$d" "$id"; fi
                continue
            fi
            if [ -e "$d/cancel" ]; then continue; fi
        fi
        rd "$d/spec"
        spec="$R"
        rd "$d/maxhold"
        hold="${R:-$DEFAULT_HOLD_S}"
        total=$((hold + GRACE_S + MARGIN_S))
        for ((c = 0; c < NC; c++)); do
            if [ "${FREE[c]}" = 1 ] && [ $((now + total)) -le "${LIMIT[c]}" ]; then U[c]=1; else U[c]=0; fi
        done
        if pick_cards "$spec"; then
            if [ "$id" = "$me" ]; then return 0; fi
            if [ "$me" = status ]; then
                printf '  %s  %-14s %-28s wants %-9s starting now on %s\n' "$id" "$(cat "$d/track")" "$(cat "$d/name")" "$spec" "${PICK// /,}"
            fi
            for c in $PICK; do
                LIMIT[c]=0
                AVAIL[c]=$((now + total))
            done
        else
            if [ "$id" = "$me" ]; then
                PICK=""
                return 1
            fi
            plan_block "$spec"
            if [ "$me" = status ]; then
                rd "$d/submitted"
                printf '  %s  %-14s %-28s wants %-9s waiting %s, max hold %s, starts by %s at the latest (cards %s)%s\n' \
                    "$id" "$(cat "$d/track")" "$(cat "$d/name")" "$spec" "$(dur $((now - ${R:-$now})))" "$(dur "$hold")" \
                    "$(iso "$SHADOW")" "${BLOCK// /,}" "$(if [ -f "$d/bump" ]; then printf '  [bumped: %s]' "$(cut -d' ' -f2- "$d/bump")"; fi)"
            fi
            for c in $BLOCK; do
                if [ "${LIMIT[c]}" -gt "$SHADOW" ]; then LIMIT[c]="$SHADOW"; fi
                AVAIL[c]=$((SHADOW + total))
            done
        fi
    done < <(start_order)
    PICK=""
    return 1
}

# lose_job DIR ID: a job whose supervisor is gone (the node restarted, or it was killed).
lose_job() {
    local d="$1" id="$2" track name spec sub start cards now
    now="$(date -u +%s)"
    rd "$d/track"
    track="$R"
    rd "$d/name"
    name="$R"
    rd "$d/spec"
    spec="$R"
    rd "$d/submitted"
    sub="$R"
    rd "$d/start"
    start="$R"
    rd "$d/cards"
    cards="$R"
    put "$d/reason" lost
    put "$d/end" "$now"
    put "$d/state" lost
    ledger "$id" "$track" "$name" "$spec" "${cards:--}" "$(iso "$sub")" "$(iso "$start")" "$(iso "$now")" \
        "$(if [ -n "$start" ]; then echo $((now - start)); else echo 0; fi)" - lost
}

# ------------------------------------------------------------------ submit

# cards_in SPEC: the number of cards SPEC asks for, in R.
cards_in() {
    case "$1" in
        all) R="$NC" ;;
        any:*) R="${1#any:}" ;;
        *)
            local c n=0
            for c in ${1//,/ }; do n=$((n + 1)); done
            R="$n"
            ;;
    esac
}

# check_replace OLD TRACK SPEC: with the queue lock held, whether a new job of TRACK on SPEC may take the
# place of job OLD; sets PLACE_AT (OLD's submission time) and OLD_STATE, or dies (the lock released).
PLACE_AT=""
OLD_STATE=""
check_replace() {
    local old="$1" track="$2" spec="$3" od used=0 want have now ended line
    case "$old" in '' | *[!0-9]*) qunlock; die "--replaces takes a job id, got '$old'" ;; esac
    old="$(printf '%05d' "$((10#$old))")"
    od="$Q/jobs/$old"
    if [ ! -d "$od" ]; then qunlock; die "--replaces: no job $old"; fi
    rd "$od/track"
    if [ "$R" != "$track" ]; then qunlock; DIE_RC=4 die "--replaces: job $old belongs to track $R, not to $track"; fi
    rd "$od/state"
    OLD_STATE="$R"
    case "$OLD_STATE" in
        waiting) ;;
        cancelled)
            rd "$od/end"
            ended="$R"
            now="$(date -u +%s)"
            case "$ended" in '' | *[!0-9]*) ended=0 ;; esac
            if [ $((now - ended)) -gt 5400 ]; then
                qunlock
                die "--replaces: job $old was cancelled more than 90 minutes ago; its place is gone"
            fi
            ;;
        *) qunlock; die "--replaces: job $old is $OLD_STATE: only a waiting job (or one cancelled in the last 90 minutes) can be replaced" ;;
    esac
    rd "$od/spec"
    cards_in "$R"
    have="$R"
    if [ -f "$od/replaced_by" ]; then
        while read -r _ line; do used=$((used + line)); done <"$od/replaced_by"
    fi
    cards_in "$spec"
    want="$R"
    if [ $((used + want)) -gt "$have" ]; then
        qunlock
        die "--replaces: job $old asked for $have cards and jobs on $used of them replace it already; $want more do not fit"
    fi
    rd "$od/place"
    case "$R" in
        '' | *[!0-9]*)
            rd "$od/submitted"
            PLACE_AT="$R"
            ;;
        *) PLACE_AT="$R" ;; # the old job held a place of its own: pass it on
    esac
    REPLACES_ID="$old"
    REPLACES_CARDS="$want"
}
REPLACES_ID=""
REPLACES_CARDS=0

cmd_submit() {
    local track="" name="" spec="" hold="$DEFAULT_HOLD_S" script="" body="" have_cmd=0 d id seq st i=0 replaces=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --track)
                [ $# -ge 2 ] || die "--track needs a value"
                track="$2"
                shift 2
                ;;
            --name)
                [ $# -ge 2 ] || die "--name needs a value"
                name="$2"
                shift 2
                ;;
            --cards)
                [ $# -ge 2 ] || die "--cards needs a value"
                spec="$2"
                shift 2
                ;;
            --max-hold)
                [ $# -ge 2 ] || die "--max-hold needs a value"
                parse_hold "$2" || die "--max-hold takes seconds or a number with s, m or h, got '$2'"
                hold="$HOLD"
                shift 2
                ;;
            --script)
                [ $# -ge 2 ] || die "--script needs a file"
                script="$2"
                shift 2
                ;;
            --replaces)
                [ $# -ge 2 ] || die "--replaces needs a job id"
                replaces="$2"
                shift 2
                ;;
            --)
                shift
                have_cmd=1
                body="$*"
                break
                ;;
            *) die "submit: unknown option: $1" ;;
        esac
    done
    case "$track" in '' | *[!abcdefghijklmnopqrstuvwxyz0123456789-]*) die "--track is required: lower-case letters, digits and '-' (your track's name)" ;; esac
    case "$name" in '' | *[!A-Za-z0-9._-]*) die "--name is required: letters, digits, '.', '_' and '-' (say what the job is for)" ;; esac
    [ -n "$spec" ] || die "--cards is required: all, a list such as 0,1,2,3, or any:N"
    parse_spec "$spec" || die "--cards: '$spec' is not all, a list of cards 0..$((NC - 1)) without repeats, or any:1..any:$NC"
    spec="$SPEC"
    if [ "$hold" -gt "$HOLD_CAP_S" ]; then
        die "--max-hold $(dur "$hold") is above the limit of $(dur "$HOLD_CAP_S") for one job"
    fi
    if [ "$have_cmd" -eq 0 ]; then
        if [ -n "$script" ]; then
            [ -f "$script" ] || die "--script: no file $script"
            body="$(cat "$script")"
        else
            if [ -t 0 ]; then die "no command: give it after --, in --script FILE, or on standard input"; fi
            body="$(cat)"
        fi
    fi
    case "$body" in *[![:space:]]*) ;; *) die "the job's command is empty" ;; esac

    command -v flock >/dev/null 2>&1 || die "no flock(1) on this host"
    command -v setsid >/dev/null 2>&1 || die "no setsid(1) on this host"
    mkdir -p "$Q/jobs" "$Q/locks" || die "cannot create $Q"
    load_map
    case "$spec" in
        all | any:*) ;;
        *)
            if ! is_group "${spec//,/ }"; then
                note "note: cards $spec are not one module group (modules of cards 0..$((NC - 1)): ${MOD[*]}); HCCL across them is not a setup Habana supports: use any:2 or any:4 for a job on several cards"
            fi
            ;;
    esac
    qlock
    for d in "$Q"/jobs/*/; do
        d="${d%/}"
        rd "$d/state"
        st="$R"
        case "$st" in waiting | running) ;; *) continue ;; esac
        rd "$d/track"
        [ "$R" = "$track" ] || continue
        rd "$d/name"
        [ "$R" = "$name" ] || continue
        sup_alive "$d" || continue
        qunlock
        DIE_RC=3 die "track $track already has the job '$name' $st as ${d##*/}; wait for it (cardq.sh wait ${d##*/}), cancel it, or use another name"
    done
    if [ -n "$replaces" ]; then check_replace "$replaces" "$track" "$spec"; fi
    rd "$Q/seq"
    seq=$((10#${R:-0} + 1))
    put "$Q/seq" "$seq"
    id="$(printf '%05d' "$seq")"
    d="$Q/jobs/$id"
    mkdir "$d" || {
        qunlock
        die "cannot create $d"
    }
    if [ -n "$REPLACES_ID" ]; then
        put "$d/place" "$PLACE_AT"
        put "$d/replaces" "$REPLACES_ID"
        printf '%s %s\n' "$id" "$REPLACES_CARDS" >>"$Q/jobs/$REPLACES_ID/replaced_by"
        if [ "$OLD_STATE" = waiting ]; then
            put "$Q/jobs/$REPLACES_ID/cancel" "by $track at $(iso "$(date -u +%s)"): replaced by job $id"
        fi
        echo "cardq: replaced by job $id ($name, $spec), which takes this job's place in the queue" >>"$Q/jobs/$REPLACES_ID/log"
    fi
    put "$d/track" "$track"
    put "$d/name" "$name"
    put "$d/spec" "$spec"
    put "$d/maxhold" "$hold"
    put "$d/submitted" "$(date -u +%s)"
    put "$d/rules" "fair-long-all-card-jobs"
    printf '%s\n' "$body" >"$d/cmd.sh"
    : >"$d/log"
    put "$d/state" waiting
    qunlock

    setsid -f nohup bash "$SELF" _run "$id" </dev/null >/dev/null 2>&1 || die "cannot start the supervisor of job $id"
    while [ ! -s "$d/pid" ] && [ "$i" -lt 100 ]; do
        sleep 0.1
        i=$((i + 1))
    done
    [ -s "$d/pid" ] || die "the supervisor of job $id did not start (no $d/pid)"
    if [ -n "$REPLACES_ID" ]; then
        echo "cardq: takes the place of job $REPLACES_ID (submitted $(iso "$PLACE_AT"))" >>"$d/log"
        note "job $id replaces job $REPLACES_ID and takes its place in the queue (submitted $(iso "$PLACE_AT"))"
    fi
    note "job $id queued: track $track, name $name, cards $spec, max hold $(dur "$hold"); log: $d/log"
    printf '%s\n' "$id"
}

# ------------------------------------------------------------------ the supervisor of one job

SUDO=""
set_sudo() {
    if [ -n "${CARDQ_SUDO+x}" ]; then
        SUDO="$CARDQ_SUDO"
    elif [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        SUDO="sudo -n"
    fi
}

JOB_DIR=""

# kill_group SIGNAL: the job's process group, and what the job started with sudo in it.
kill_group() {
    local pg
    rd "$JOB_DIR/pgid"
    pg="$R"
    case "$pg" in '' | *[!0-9]*) return 0 ;; esac
    if [ "$pg" -le 1 ] || [ "$pg" -eq "$$" ]; then return 0; fi
    kill "-$1" -- "-$pg" 2>/dev/null
    if [ -n "$SUDO" ]; then $SUDO kill "-$1" -- "-$pg" 2>/dev/null; fi
    return 0
}

group_alive() {
    local pg
    rd "$JOB_DIR/pgid"
    pg="$R"
    case "$pg" in '' | *[!0-9]*) return 1 ;; esac
    if [ "$pg" -le 1 ]; then return 1; fi
    if kill -0 -- "-$pg" 2>/dev/null; then return 0; fi
    if [ -n "$SUDO" ] && $SUDO kill -0 -- "-$pg" 2>/dev/null; then return 0; fi
    return 1
}

# sweep_containers ID ALL_CARDS ABNORMAL: the job's containers, by its label; and after a job that
# held every card and was stopped, every running container of the habana runtime (nobody else can
# have been using a card).
sweep_containers() {
    local id="$1" allc="$2" abnormal="$3" ids="" cid rt nm
    command -v docker >/dev/null 2>&1 || return 0
    ids="$($SUDO docker ps -aq --filter "label=cardq.job=$id" 2>/dev/null | tr '\n' ' ')" || ids=""
    case "$ids" in
        *[![:space:]]*)
            echo "cardq: removing the containers of job $id: $ids" >>"$JOB_DIR/log"
            # shellcheck disable=SC2086  # a list of container ids
            $SUDO docker rm -f $ids >/dev/null 2>&1
            ;;
    esac
    if [ "$allc" -eq 1 ] && [ "$abnormal" -eq 1 ]; then
        for cid in $($SUDO docker ps -q 2>/dev/null); do
            rt="$($SUDO docker inspect --format '{{.HostConfig.Runtime}}' "$cid" 2>/dev/null)" || rt=""
            [ "$rt" = habana ] || continue
            nm="$($SUDO docker inspect --format '{{.Name}}' "$cid" 2>/dev/null)" || nm="$cid"
            echo "cardq: job $id held every card and was stopped: removing the container ${nm#/}" >>"$JOB_DIR/log"
            $SUDO docker rm -f "$cid" >/dev/null 2>&1
        done
    fi
}

# watchdog SUPERVISOR_PID DEADLINE: ends the job at its maximum hold, on cancel, or when the
# supervisor is gone. It holds the card locks too, so they outlive a killed supervisor until the
# job is stopped.
watchdog() {
    local sup="$1" deadline="$2" why="" now waited=0
    while :; do
        sleep "$POLL_S"
        if [ -e "$JOB_DIR/done" ]; then exit 0; fi
        now="$(date -u +%s)"
        if [ -e "$JOB_DIR/cancel" ]; then
            why=cancelled
        elif [ "$now" -ge "$deadline" ]; then
            why=timeout
        elif ! kill -0 "$sup" 2>/dev/null; then
            why=lost
        fi
        if [ -n "$why" ]; then break; fi
    done
    put "$JOB_DIR/reason" "$why"
    echo "cardq: $why at $(iso "$now"): SIGTERM to the job's process group, SIGKILL $GRACE_S s later" >>"$JOB_DIR/log"
    kill_group TERM
    while [ ! -e "$JOB_DIR/done" ] && group_alive && [ "$waited" -lt "$GRACE_S" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if [ ! -e "$JOB_DIR/done" ] && group_alive; then
        echo "cardq: the job did not end in $GRACE_S s: SIGKILL" >>"$JOB_DIR/log"
    fi
    kill_group KILL
    if [ "$why" = lost ]; then
        # nobody is left to close the job: do it here
        sweep_containers "${JOB_DIR##*/}" "$3" 1
        qlock
        lose_job "$JOB_DIR" "${JOB_DIR##*/}"
        rd "$JOB_DIR/cards"
        for now in ${R//,/ }; do drop_card "$now"; done
        qunlock
    fi
    exit 0
}

# uptime_now: seconds since boot (the clock of dmesg), in R ("" where there is no /proc/uptime).
uptime_now() {
    local f="${CARDQ_UPTIME_FILE:-/proc/uptime}"
    R=""
    if [ -r "$f" ]; then read -r R _ <"$f" 2>/dev/null || R=""; fi
    R="${R%%.*}"
}

# kernel_lines T0: the habanalabs kernel lines since uptime T0 (seconds), routine ones left out.
kernel_lines() {
    (${SUDO:+$SUDO }dmesg 2>/dev/null || dmesg 2>/dev/null) | awk -v t0="$1" '
        /habanalabs/ && !/renamed from|IB device registered|hwmon|Successfully|firmware|version|device found|Found GAUDI/ {
            if (match($0, /^\[ *[0-9]+\.[0-9]+\]/)) {
                t = substr($0, 2, RLENGTH - 2) + 0
                if (t >= t0) print
            }
        }'
}

# classes LINES: the lines of one card by class: HK killed while waiting, HD failure analysis, HE any
# other line (a card error) with the text of the first in HE_TEXT; HC is "2 ERROR, 1 failure analysis".
HK=0
HD=0
HE=0
HE_TEXT=""
HC=""
classes() {
    local n
    n="$(printf '%s\n' "$1" | grep -c .)"
    HK="$(printf '%s\n' "$1" | grep -c 'got signal while waiting for interrupt')"
    HD="$(printf '%s\n' "$1" | grep -c 'unexpected user error interrupt')"
    HE=$((n - HK - HD))
    HE_TEXT=""
    HC=""
    if [ "$HE" -gt 0 ]; then
        HE_TEXT="$(printf '%s\n' "$1" | grep -v -e 'got signal while waiting for interrupt' -e 'unexpected user error interrupt' \
            | head -n 1 | sed -E 's/^\[[^]]*\] *//; s/^habanalabs [0-9a-f:.]+: *//' | tr '\t' ' ' | cut -c1-80)"
        HC="$HE ERROR"
    fi
    if [ "$HD" -gt 0 ]; then HC="$HC${HC:+, }$HD failure analysis"; fi
    if [ "$HK" -gt 0 ]; then HC="$HC${HC:+, }$HK killed while waiting"; fi
}

# health_scan JOB_DIR CARDS: the habanalabs kernel lines since the job started, per PCI address and class
# (see the header); one log line per address, the lines of the job's own cards as a short summary in HEALTH
# ("" when there is none or no dmesg; HEALTH_BAD is 1 when one of them is an error), rows in health.tsv.
HEALTH=""
HEALTH_BAD=0
health_scan() {
    local d="$1" cards="$2" t0 lines bus n card mod mine own first of
    HEALTH=""
    HEALTH_BAD=0
    rd "$d/uptime_start"
    t0="$R"
    case "$t0" in '' | *[!0-9]*) return 0 ;; esac
    lines="$(kernel_lines "$t0")" || lines=""
    [ -n "$lines" ] || return 0
    for bus in $(printf '%s\n' "$lines" | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | sort -u); do
        of="$(printf '%s\n' "$lines" | grep "$bus")"
        n="$(printf '%s\n' "$of" | grep -c .)"
        first="$(printf '%s\n' "$of" | head -n 1 | cut -c1-160)"
        classes "$of"
        card_of_pci "$bus"
        card="${R:-?}"
        mod="?"
        if [ -n "$R" ]; then mod="${MOD[R]:-?}"; fi
        mine=" (not a card of this job)"
        own=other
        case ",$cards," in
            *",$card,"*)
                mine=" (one of this job's cards)"
                own=own
                HEALTH="$HEALTH${HEALTH:+, }card $card (module $mod, $bus) $n lines ($(
                    if [ "$HE" -gt 0 ]; then printf '%s' "${HC/ERROR/ERROR: $HE_TEXT}"; else printf '%s' "$HC"; fi
                ))"
                if [ "$HE" -gt 0 ]; then HEALTH_BAD=1; fi
                ;;
        esac
        echo "cardq: habanalabs kernel lines during the job: card $card (module $mod, $bus): $n$mine: $HC; first: $first" >>"$d/log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${d##*/}" "$(cat "$d/track")" "$(cat "$d/name")" \
            "$cards" "$card" "$mod" "$bus" "$n" "$own" "$HK" "$HD" "$HE" "$HE_TEXT" >>"$Q/health.tsv"
    done
    return 0
}

# health: the kernel lines since boot per card and class, and the jobs that had some (health.tsv).
cmd_health() {
    local lines bus of n card mod bad=0
    mkdir -p "$Q/jobs" "$Q/locks" || die "cannot create $Q"
    set_sudo
    load_map
    lines="$(kernel_lines 0)" || lines=""
    echo "habanalabs kernel lines since boot, per card and class (routine lines left out):"
    if [ -z "$lines" ]; then
        echo "  none"
    else
        for bus in $(printf '%s\n' "$lines" | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | sort -u); do
            of="$(printf '%s\n' "$lines" | grep "$bus")"
            n="$(printf '%s\n' "$of" | grep -c .)"
            classes "$of"
            card_of_pci "$bus"
            card="${R:-?}"
            mod="?"
            if [ -n "$R" ]; then mod="${MOD[R]:-?}"; fi
            if [ "$HE" -gt 0 ]; then
                bad=1
                echo "  card $card (module $mod, $bus): $n lines: ${HC/ERROR/ERROR (first: $HE_TEXT)}"
            else
                echo "  card $card (module $mod, $bus): $n lines: $HC"
            fi
        done
    fi
    if [ "$bad" -eq 1 ]; then
        echo "CARD ERROR: a line of an unknown class above: tell the card-farm track and the coordinator (sudo dmesg | grep habanalabs)"
    else
        echo "no line of an unknown class: no card error"
    fi
    echo "  killed while waiting: 'user process got signal while waiting for interrupt', a process killed while it waited for the card"
    echo "  failure analysis: 'Received unexpected user error interrupt', Synapse started its failure analysis in a job's process"
    echo "    (a signal the process does not handle, or 600 s without progress): see habana_logs/dfa_log.txt in the job's results"
    echo "jobs with kernel lines (job, track, name, cards, card, module, pci, lines, own/other card, killed, failure analysis, error, first error):"
    if [ -s "$Q/health.tsv" ]; then cat "$Q/health.tsv"; else echo "  (none)"; fi
}

cmd_run() {
    local id="$1" d="$Q/jobs/$1" track name spec hold sub now c start deadline cards modules rc reason wd fds="" n=0 allc=0 abnormal=0 lowest
    [ -d "$d" ] || exit 1
    JOB_DIR="$d"
    exec >>"$d/supervisor.log" 2>&1
    put "$d/pid" "$$"
    set_sudo
    load_map
    rd "$d/track"
    track="$R"
    rd "$d/name"
    name="$R"
    rd "$d/spec"
    spec="$R"
    rd "$d/maxhold"
    hold="$R"
    rd "$d/submitted"
    sub="$R"

    # 1. wait for the cards
    while :; do
        if [ -e "$d/cancel" ]; then
            now="$(date -u +%s)"
            qlock
            put "$d/reason" cancelled
            put "$d/end" "$now"
            put "$d/state" cancelled
            ledger "$id" "$track" "$name" "$spec" - "$(iso "$sub")" - "$(iso "$now")" 0 - cancelled
            qunlock
            echo "cardq: job $id cancelled before it got cards ($(cat "$d/cancel" 2>/dev/null))" >>"$d/log"
            exit 0
        fi
        qlock
        for ((c = 0; c < NC; c++)); do
            if try_card "$c"; then FREE[c]=1; else FREE[c]=0; fi
        done
        if plan "$id"; then
            for ((c = 0; c < NC; c++)); do
                case " $PICK " in
                    *" $c "*) ;;
                    *) if [ "${FREE[c]}" = 1 ]; then drop_card "$c"; fi ;;
                esac
            done
            cards="${PICK// /,}"
            modules_of "$PICK"
            modules="$R"
            start="$(date -u +%s)"
            deadline=$((start + hold))
            put "$d/modules" "$modules"
            uptime_now
            if [ -n "$R" ]; then put "$d/uptime_start" "$R"; fi
            put "$d/cards" "$cards"
            put "$d/start" "$start"
            put "$d/deadline" "$deadline"
            put "$d/state" running
            qunlock
            break
        fi
        for ((c = 0; c < NC; c++)); do
            if [ "${FREE[c]}" = 1 ]; then drop_card "$c"; fi
        done
        qunlock
        sleep "$POLL_S"
    done

    # 2. run the job: its own session and process group, the lock descriptors closed, and no "&"
    #    (a background start would hand it an ignored SIGINT)
    for c in $PICK; do
        fds="$fds $((FD_BASE + c))>&-"
        n=$((n + 1))
    done
    if [ "$n" -eq "$NC" ]; then allc=1; fi
    lowest="${PICK%% *}"
    export CARDQ_JOB="$id" CARDQ_TRACK="$track" CARDQ_NAME="$name" CARDQ_CARDS="$cards" CARDQ_NCARDS="$n"
    export CARDQ_MODULES="$modules"
    export CARDQ_DEADLINE="$deadline" CARDQ_JOB_DIR="$d" CARDQ_PORT_BASE=$((PORT_BASE + 100 * lowest))
    {
        echo "cardq: job $id (track $track, name $name) on cards $cards (modules $modules) from $(iso "$start"), waited $(dur $((start - sub)))"
        echo "cardq: max hold $(dur "$hold"): SIGTERM at $(iso "$deadline"), SIGKILL $GRACE_S s later; ports $CARDQ_PORT_BASE..$((CARDQ_PORT_BASE + 99))"
    } >>"$d/log"
    watchdog "$$" "$deadline" "$allc" &
    wd=$!
    cd "$HOME" 2>/dev/null || true
    # shellcheck disable=SC2016  # $$ and $1 belong to the job's shell
    eval "setsid bash -c 'echo \$\$ >\"\$1/pgid\"; exec bash \"\$1/cmd.sh\"' cardq-job \"\$d\" </dev/null >>\"\$d/log\" 2>&1 $fds"
    rc=$?

    # 3. close the job
    : >"$d/done"
    kill "$wd" 2>/dev/null
    wait "$wd" 2>/dev/null
    rd "$d/reason"
    reason="$R"
    if [ -n "$reason" ]; then
        abnormal=1
    elif [ "$rc" -eq 0 ]; then
        reason=ok
    else
        reason=failed
    fi
    kill_group KILL
    sweep_containers "$id" "$allc" "$abnormal"
    health_scan "$d" "$cards"
    if [ -n "$HEALTH" ]; then
        if [ "$HEALTH_BAD" -eq 1 ]; then reason="$reason; habanalabs ERROR: $HEALTH"; else reason="$reason; habanalabs: $HEALTH"; fi
    fi
    now="$(date -u +%s)"
    echo "cardq: job $id ended at $(iso "$now"): exit code $rc, $((now - start)) s on cards $cards, $reason" >>"$d/log"
    qlock
    put "$d/reason" "$reason"
    put "$d/rc" "$rc"
    put "$d/end" "$now"
    put "$d/state" ended
    ledger "$id" "$track" "$name" "$spec" "$cards" "$(iso "$sub")" "$(iso "$start")" "$(iso "$now")" $((now - start)) "$rc" "$reason"
    # The cards are unlocked under the queue lock, in the same step as the state change: nobody may
    # see an ended job whose cards are still locked. Unlocked by hand, because a sleep the watchdog
    # left behind still has the descriptors open for a moment.
    for c in $PICK; do drop_card "$c"; done
    qunlock
    exit 0
}

# ------------------------------------------------------------------ wait, log, cancel

# end_line DIR ID: one line for an ended job.
end_line() {
    local d="$1" rc reason start end cards
    rd "$d/rc"
    rc="${R:--}"
    rd "$d/reason"
    reason="$R"
    rd "$d/start"
    start="$R"
    rd "$d/end"
    end="$R"
    rd "$d/cards"
    cards="${R:--}"
    if [ -n "$start" ] && [ -n "$end" ]; then
        echo "job $2 ended: exit code $rc, $((end - start)) s on cards $cards, $reason"
    else
        echo "job $2 ended without running: $reason"
    fi
}

cmd_wait() {
    local timeout="" d st last="" began="$SECONDS" rc
    [ $# -ge 1 ] || die "usage: cardq.sh wait ID [--timeout S]"
    norm_id "$1"
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --timeout)
                [ $# -ge 2 ] || die "--timeout needs seconds"
                timeout="$2"
                case "$timeout" in '' | *[!0-9]*) die "--timeout takes whole seconds" ;; esac
                shift 2
                ;;
            *) die "wait: unknown option: $1" ;;
        esac
    done
    d="$Q/jobs/$ID"
    while :; do
        rd "$d/state"
        st="$R"
        if [ "$st" != "$last" ]; then
            case "$st" in
                waiting)
                    rd "$d/spec"
                    note "job $ID: waiting for cards ($R)"
                    ;;
                running)
                    rd "$d/cards"
                    note "job $ID: running on cards $R"
                    ;;
            esac
            last="$st"
        fi
        case "$st" in
            ended)
                note "$(end_line "$d" "$ID")"
                rd "$d/rc"
                rc="$R"
                case "$rc" in '' | *[!0-9]*) exit 126 ;; esac
                exit "$rc"
                ;;
            cancelled)
                note "$(end_line "$d" "$ID")"
                exit 125
                ;;
            lost)
                note "job $ID is lost: its supervisor is gone (was the node restarted?)"
                exit 126
                ;;
            waiting | running)
                if ! sup_alive "$d"; then
                    rd "$d/state"
                    case "$R" in
                        waiting | running)
                            note "job $ID is lost: its supervisor is gone (was the node restarted?)"
                            exit 126
                            ;;
                    esac
                    continue
                fi
                ;;
        esac
        if [ -n "$timeout" ] && [ $((SECONDS - began)) -ge "$timeout" ]; then
            note "job $ID: still $st after $timeout s (the job goes on; run the same command again)"
            exit 124
        fi
        sleep 1
    done
}

cmd_log() {
    local follow=0 d off=0 size st
    [ $# -ge 1 ] || die "usage: cardq.sh log ID [-f]"
    norm_id "$1"
    shift
    if [ "${1:-}" = -f ]; then follow=1; fi
    d="$Q/jobs/$ID"
    if [ "$follow" -eq 0 ]; then
        cat "$d/log"
        return 0
    fi
    while :; do
        rd "$d/state"
        st="$R"
        size="$(wc -c <"$d/log" | tr -d ' ')"
        if [ "$size" -gt "$off" ]; then
            tail -c "+$((off + 1))" "$d/log" | head -c "$((size - off))"
            off="$size"
        fi
        case "$st" in
            waiting | running) sup_alive "$d" || break ;;
            *) break ;;
        esac
        sleep 1
    done
    size="$(wc -c <"$d/log" | tr -d ' ')"
    if [ "$size" -gt "$off" ]; then tail -c "+$((off + 1))" "$d/log"; fi
}

cmd_cancel() {
    local track="" d st i=0 limit
    while [ $# -gt 0 ]; do
        case "$1" in
            --track)
                [ $# -ge 2 ] || die "--track needs a value"
                track="$2"
                shift 2
                ;;
            -*) die "cancel: unknown option: $1" ;;
            *) break ;;
        esac
    done
    [ -n "$track" ] || die "usage: cardq.sh cancel --track T ID (your own track's name)"
    [ $# -ge 1 ] || die "usage: cardq.sh cancel --track T ID"
    norm_id "$1"
    d="$Q/jobs/$ID"
    rd "$d/track"
    if [ "$R" != "$track" ]; then
        DIE_RC=4 die "job $ID belongs to track $R, not to $track: a track stops only its own jobs"
    fi
    rd "$d/state"
    st="$R"
    case "$st" in
        waiting | running) ;;
        *)
            note "job $ID is not waiting or running (state: $st): nothing to cancel"
            return 0
            ;;
    esac
    put "$d/cancel" "by $track at $(iso "$(date -u +%s)")"
    note "job $ID: cancel requested ($st); it stops within about $((GRACE_S + 10)) s"
    limit=$((GRACE_S + 30))
    while [ "$i" -lt "$limit" ]; do
        rd "$d/state"
        case "$R" in
            waiting | running) sup_alive "$d" || break ;;
            *) break ;;
        esac
        sleep 1
        i=$((i + 1))
    done
    rd "$d/state"
    note "job $ID: $R"
}

# ------------------------------------------------------------------ bump, adopt

# bump --by coord ID [WHY]: the waiting job ID goes before every other waiting job (bumped jobs among
# themselves in bump order). Only the coordinator decides a bump; each one is a line in the ledger
# (end column: the bump time, reason: "bump by WHO: WHY") and in the job's log.
cmd_bump() {
    local by="" why d now
    while [ $# -gt 0 ]; do
        case "$1" in
            --by)
                [ $# -ge 2 ] || die "--by needs a value"
                by="$2"
                shift 2
                ;;
            -*) die "bump: unknown option: $1" ;;
            *) break ;;
        esac
    done
    [ "$by" = coord ] || DIE_RC=4 die "bump: only the coordinator moves jobs: cardq.sh bump --by coord ID [WHY]"
    [ $# -ge 1 ] || die "usage: cardq.sh bump --by coord ID [WHY]"
    norm_id "$1"
    shift
    why="${*:-no reason given}"
    d="$Q/jobs/$ID"
    qlock
    rd "$d/state"
    if [ "$R" != waiting ]; then
        qunlock
        die "job $ID is $R, not waiting: nothing to bump"
    fi
    now="$(date -u +%s)"
    put "$d/bump" "$now $by: $why"
    rd "$d/submitted"
    ledger "$ID" "$(cat "$d/track")" "$(cat "$d/name")" "$(cat "$d/spec")" - "$(iso "$R")" - "$(iso "$now")" 0 - \
        "bump by $by: $why"
    qunlock
    echo "cardq: bumped to the front of the queue at $(iso "$now") by $by: $why" >>"$d/log"
    note "job $ID: bumped to the front by $by ($why)"
}

# place --by coord NEW OLD: what submit --replaces OLD would have done for the waiting job NEW, for a job
# its track submitted after cancelling OLD itself. OLD must be cancelled (never cancelled here).
cmd_place() {
    local by="" new old nd spec track
    while [ $# -gt 0 ]; do
        case "$1" in
            --by)
                [ $# -ge 2 ] || die "--by needs a value"
                by="$2"
                shift 2
                ;;
            -*) die "place: unknown option: $1" ;;
            *) break ;;
        esac
    done
    [ "$by" = coord ] || DIE_RC=4 die "place: only the coordinator moves jobs: cardq.sh place --by coord NEW OLD"
    [ $# -eq 2 ] || die "usage: cardq.sh place --by coord NEW OLD"
    norm_id "$1"
    new="$ID"
    norm_id "$2"
    old="$ID"
    nd="$Q/jobs/$new"
    qlock
    rd "$nd/state"
    if [ "$R" != waiting ]; then qunlock; die "job $new is $R, not waiting"; fi
    rd "$Q/jobs/$old/state"
    if [ "$R" != cancelled ]; then qunlock; die "job $old is $R: place only gives the place of a job its track cancelled"; fi
    rd "$nd/track"
    track="$R"
    rd "$nd/spec"
    spec="$R"
    check_replace "$old" "$track" "$spec"
    put "$nd/place" "$PLACE_AT"
    put "$nd/replaces" "$REPLACES_ID"
    printf '%s %s\n' "$new" "$REPLACES_CARDS" >>"$Q/jobs/$REPLACES_ID/replaced_by"
    ledger "$new" "$track" "$(cat "$nd/name")" "$spec" - "$(iso "$(cat "$nd/submitted")")" - "$(iso "$(date -u +%s)")" 0 - \
        "place of $old by $by"
    qunlock
    echo "cardq: takes the place of job $old (submitted $(iso "$PLACE_AT")), given by $by" >>"$nd/log"
    note "job $new: takes the place of job $old (submitted $(iso "$PLACE_AT")), given by $by"
}

# adopt: every waiting job gets a supervisor that runs this file. The new supervisor starts and records
# its pid before the old one is stopped, under the queue lock, so the job never looks lost and nobody
# plans meanwhile. A waiting supervisor holds no card. Running jobs are not touched.
cmd_adopt() {
    local d id old i n=0
    mkdir -p "$Q/jobs" "$Q/locks" || die "cannot create $Q"
    qlock
    for d in "$Q"/jobs/*/; do
        d="${d%/}"
        id="${d##*/}"
        rd "$d/state"
        [ "$R" = waiting ] || continue
        rd "$d/pid"
        old="$R"
        case "$old" in '' | *[!0-9]*) continue ;; esac
        kill -0 "$old" 2>/dev/null || continue # a dead supervisor: the next plan marks the job lost
        setsid -f nohup bash "$SELF" _run "$id" </dev/null >/dev/null 2>&1 || {
            note "job $id: cannot start a new supervisor; the old one ($old) stays"
            continue
        }
        for ((i = 0; i < 100; i++)); do
            rd "$d/pid"
            if [ "$R" != "$old" ] && [ -n "$R" ]; then break; fi
            sleep 0.1
        done
        if [ "$R" = "$old" ] || [ -z "$R" ]; then
            note "job $id: the new supervisor did not start; the old one ($old) stays"
            continue
        fi
        kill -KILL "$old" 2>/dev/null
        echo "cardq: supervisor $old replaced by $R (cardq.sh adopt at $(iso "$(date -u +%s)"))" >>"$d/log"
        n=$((n + 1))
    done
    qunlock
    note "adopt: $n waiting jobs now have supervisors that run $SELF"
}

# ------------------------------------------------------------------ status, ledger

# One line per job that is over, oldest first. (A function of its own: bash 3.2 cannot read a
# case statement inside a command substitution.)
ended_lines() {
    local d id
    for d in "$Q"/jobs/*/; do
        d="${d%/}"
        rd "$d/state"
        case "$R" in
            ended | cancelled | lost) ;;
            *) continue ;;
        esac
        id="${d##*/}"
        rd "$d/end"
        printf '  %s  %-14s %-28s %s  %s\n' "$id" "$(cat "$d/track")" "$(cat "$d/name")" "$(iso "$R")" \
            "$(end_line "$d" "$id" | sed 's/^job [0-9]* //')"
    done
}

cmd_status() {
    local now c d id st cards hold start dl owner ended n=0 mem
    mkdir -p "$Q/jobs" "$Q/locks" || die "cannot create $Q"
    load_map
    now="$(date -u +%s)"
    echo "cardq status at $(iso "$now") ($NC cards, queue $Q)"
    mem=""
    for ((c = 0; c < NC; c++)); do mem="$mem $c:${MOD[c]}"; done
    echo "card:module${mem} (any:4 and any:2 take whole module groups)"
    qlock
    for ((c = 0; c < NC; c++)); do
        if try_card "$c"; then
            FREE[c]=1
            drop_card "$c"
        else
            FREE[c]=0
        fi
    done
    echo "cards:"
    for ((c = 0; c < NC; c++)); do
        owner=""
        for d in "$Q"/jobs/*/; do
            d="${d%/}"
            rd "$d/state"
            [ "$R" = running ] || continue
            rd "$d/cards"
            case ",$R," in *",$c,"*) ;; *) continue ;; esac
            id="${d##*/}"
            rd "$d/start"
            start="$R"
            rd "$d/deadline"
            dl="$R"
            owner="job $id  $(cat "$d/track")/$(cat "$d/name")  running $(dur $((now - start))), SIGTERM at $(iso "$dl")"
            if ! sup_alive "$d"; then owner="$owner  (supervisor gone: being stopped)"; fi
        done
        if [ -n "$owner" ]; then
            if [ "${FREE[c]}" = 1 ]; then owner="$owner  (lock free: the job is lost)"; fi
            printf '  %s  %s\n' "$c" "$owner"
        elif [ "${FREE[c]}" = 1 ]; then
            printf '  %s  free\n' "$c"
        else
            printf '  %s  locked, by no running job of this queue\n' "$c"
        fi
    done
    echo "waiting (in the order they start):"
    { plan status || true; } | grep . || echo "  (none)"
    qunlock
    echo "ended (the last 8):"
    ended="$(ended_lines | tail -n 8)"
    if [ -n "$ended" ]; then printf '%s\n' "$ended"; else echo "  (none)"; fi
    if command -v hl-smi >/dev/null 2>&1; then
        mem="$(hl-smi -Q index,memory.used,memory.total -f csv,noheader 2>/dev/null | sort -n | tr '\n' ';')" || mem=""
        if [ -n "$mem" ]; then echo "card memory (index, used, total): $mem"; fi
    fi
    return "$n"
}

cmd_ledger() {
    if [ -f "$Q/ledger.tsv" ]; then cat "$Q/ledger.tsv"; else echo "no job has ended yet"; fi
}

usage() {
    sed -n '3,29p' "$SELF" | sed 's/^# \{0,1\}//'
}

if [ "$(id -u)" -eq 0 ] && [ "${CARDQ_ALLOW_ROOT:-0}" != 1 ]; then
    die "do not run cardq.sh as root: jobs run as the login user and call sudo themselves"
fi
case "$NC" in '' | *[!0-9]* | 0) die "CARDQ_NCARDS must be a positive number" ;; esac

case "${1:-}" in
    submit)
        shift
        cmd_submit "$@"
        ;;
    wait)
        shift
        cmd_wait "$@"
        ;;
    status)
        shift
        cmd_status "$@"
        ;;
    log)
        shift
        cmd_log "$@"
        ;;
    cancel)
        shift
        cmd_cancel "$@"
        ;;
    ledger)
        shift
        cmd_ledger "$@"
        ;;
    health)
        shift
        cmd_health "$@"
        ;;
    bump)
        shift
        cmd_bump "$@"
        ;;
    adopt)
        shift
        cmd_adopt "$@"
        ;;
    place)
        shift
        cmd_place "$@"
        ;;
    _run)
        shift
        cmd_run "$@"
        ;;
    -h | --help | help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
