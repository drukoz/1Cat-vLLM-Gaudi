#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# node_bringup.sh: on the rented Gaudi2 node, everything between "Ubuntu 22.04 with eight cards" and
# "the card farm takes jobs". A node start gives a new machine (no driver, no image, no checkpoint), so
# this runs once per node start; every step is idempotent, so it can be run again after a failure.
#
#   bash ~/farm/node_bringup.sh [--ram-gb N] [--repo ID] [--min-gb N] [--no-download] [--dry-run]
#
#   1 checks      Ubuntu 22.04, eight Gaudi2 cards on PCI, memory for the checkpoint in RAM, free root disk
#   2 host        sudo bash ~/rental/prepare_host.sh: driver stack, Docker, the habana runtime, the image
#   3 ram-disk    /models as a RAM disk (tmpfs) of --ram-gb GB (default 320); no disk is touched
#   4 download    hf download --repo (default Qwen/Qwen3.8-Flash-Next-FP8) into /models, in the background
#   5 base-image  ~/farm/build_base.sh ~/rental (needs no card)
#   6 checkpoint  waits for step 4 and checks that /models holds at least --min-gb GB (default 170)
#   7 cards       the queue's first status: the card:module map of THIS machine and its module groups
#
# Needs ~/rental (an unpacked rental bundle of the program head) and the farm scripts in ~/farm; the
# workstation's `farm bringup ADDRESS` puts both there and starts this script. Never formats, partitions
# or mounts a disk: /dev/vdc (26 TB, blank) stays as it is. Log lines: "STEP n name: ok in S s"; the last
# line is BRINGUP-DONE or BRINGUP-FAILED with the step. On 2026-11-20 the steps took: host 9 min, download
# 100 s (173 GB), base image 4 min.

set -u
set -o pipefail

RAM_GB=320
REPO="Qwen/Qwen3.8-Flash-Next-FP8"
MIN_GB=170
DOWNLOAD=1
DRY=0
MODELS="${BRINGUP_MODELS_DIR:-/models}"
RENTAL="${BRINGUP_RENTAL_DIR:-$HOME/rental}"
FARM="${FARM_DIR:-$HOME/farm}"
WORK="${BRINGUP_WORK_DIR:-$HOME/bringup}"
SUDO="${BRINGUP_SUDO-sudo -n}"
NCARDS_WANT="${BRINGUP_NCARDS:-8}"
HF_VERSION="${BRINGUP_HF_VERSION:-1.33.0}"
MIN_ROOT_GB="${BRINGUP_MIN_ROOT_GB:-40}"

die() {
    echo "node_bringup: error: $*" >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --ram-gb) [ $# -ge 2 ] || die "--ram-gb needs a number"; RAM_GB="$2"; shift 2 ;;
        --repo) [ $# -ge 2 ] || die "--repo needs a Hugging Face repository id"; REPO="$2"; shift 2 ;;
        --min-gb) [ $# -ge 2 ] || die "--min-gb needs a number"; MIN_GB="$2"; shift 2 ;;
        --no-download) DOWNLOAD=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) sed -n '3,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
case "$RAM_GB$MIN_GB" in *[!0-9]*) die "--ram-gb and --min-gb take whole numbers" ;; esac
NAME="${REPO##*/}"
DEST="$MODELS/$NAME"

say() { echo "$*"; }
# run COMMAND...: print it; run it unless --dry-run.
run() {
    say "\$ $*"
    if [ "$DRY" -eq 1 ]; then return 0; fi
    "$@"
}
# run_logged LOG COMMAND...: the same, with the command's output in LOG.
run_logged() {
    local log="$1"
    shift
    say "\$ $* >$log 2>&1"
    if [ "$DRY" -eq 1 ]; then return 0; fi
    "$@" >"$log" 2>&1
}
STEP_N=0
STEP_NAME=""
STEP_T=0
step() {
    STEP_N="$1"
    STEP_NAME="$2"
    STEP_T="$(date -u +%s)"
    say "STEP $STEP_N $STEP_NAME: start [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
}
ok() { say "STEP $STEP_N $STEP_NAME: ok in $(($(date -u +%s) - STEP_T)) s${1:+ ($1)}"; }
fail() {
    say "STEP $STEP_N $STEP_NAME: FAILED: $*"
    say "BRINGUP-FAILED at step $STEP_N ($STEP_NAME): $*"
    exit 1
}

mkdir -p "$WORK" || die "cannot create $WORK"

# ------------------------------------------------------------------ 1 checks
step 1 checks
os="$(. "${BRINGUP_OS_RELEASE:-/etc/os-release}" 2>/dev/null && echo "${ID:-?} ${VERSION_ID:-?}")"
cards="$(lspci -d 1da3: 2>/dev/null | grep -c . || true)"
mem_gb="$(awk '/^MemTotal:/ {printf "%d", $2 / 1048576}' "${BRINGUP_MEMINFO:-/proc/meminfo}" 2>/dev/null || echo 0)"
root_gb="$(df -Pk / 2>/dev/null | awk 'NR == 2 {printf "%d", $4 / 1048576}')"
cpus="$(nproc 2>/dev/null || echo 0)"
say "os: $os; Gaudi cards on PCI: $cards; memory: $mem_gb GB; free on /: ${root_gb:-?} GB; CPUs: $cpus"
[ -f "$RENTAL/BUNDLE.txt" ] || fail "no $RENTAL/BUNDLE.txt: copy and unpack the rental bundle first (the workstation's farm bringup does it)"
[ -f "$FARM/build_base.sh" ] || fail "no $FARM/build_base.sh: the farm scripts are not installed (farm install)"
if [ "$DRY" -eq 0 ]; then
    [ "$os" = "ubuntu 22.04" ] || fail "this is '$os', not Ubuntu 22.04"
    [ "$cards" -eq "$NCARDS_WANT" ] || fail "$cards Gaudi cards on PCI, $NCARDS_WANT expected"
    # The checkpoint in RAM, the servers' host memory and the page cache of the images must fit.
    [ "$mem_gb" -ge $((RAM_GB + 200)) ] || fail "only $mem_gb GB of memory: a $RAM_GB GB RAM disk needs $((RAM_GB + 200)) GB. Do not use a disk without the owner's word (prepare_host.sh --data-disk formats it)"
    [ "${root_gb:-0}" -ge "$MIN_ROOT_GB" ] || fail "only ${root_gb:-0} GB free on /: the images need $MIN_ROOT_GB GB"
fi
ok "$cards cards, $mem_gb GB, $cpus CPUs"

# ------------------------------------------------------------------ 2 host
step 2 host
if [ "$DRY" -eq 0 ] && command -v hl-smi >/dev/null 2>&1 \
    && [ "$(hl-smi -Q index -f csv,noheader 2>/dev/null | grep -c .)" -eq "$NCARDS_WANT" ] \
    && $SUDO docker info 2>/dev/null | grep -q 'habana'; then
    ok "driver, Docker and the habana runtime are there already"
else
    # shellcheck disable=SC2086  # SUDO is a command with its option
    run_logged "$WORK/prepare_host.log" $SUDO bash "$RENTAL/prepare_host.sh" || {
        tail -n 25 "$WORK/prepare_host.log"
        fail "prepare_host.sh failed; log: $WORK/prepare_host.log and $RENTAL/prepare-host-logs/"
    }
    if [ "$DRY" -eq 0 ]; then
        n="$(hl-smi -Q index -f csv,noheader 2>/dev/null | grep -c . || true)"
        [ "$n" -eq "$NCARDS_WANT" ] || fail "hl-smi lists $n cards after prepare_host.sh, $NCARDS_WANT expected"
    fi
    ok "prepare_host.sh"
fi

# ------------------------------------------------------------------ 3 ram-disk
step 3 ram-disk
if [ "$DRY" -eq 0 ] && mountpoint -q "$MODELS" 2>/dev/null; then
    ok "$MODELS is mounted already ($(df -h "$MODELS" | awk 'NR == 2 {print $1 ", " $2}'))"
else
    # shellcheck disable=SC2086
    run $SUDO mkdir -p "$MODELS" || fail "cannot create $MODELS"
    # shellcheck disable=SC2086
    run $SUDO mount -t tmpfs -o "size=${RAM_GB}G" tmpfs "$MODELS" || fail "cannot mount a tmpfs on $MODELS"
    # shellcheck disable=SC2086
    run $SUDO chown "$(id -u):$(id -g)" "$MODELS" || fail "cannot chown $MODELS"
    ok "tmpfs of $RAM_GB GB on $MODELS"
fi

# ------------------------------------------------------------------ 4 download (background)
step 4 download
dl_running() {
    local pid
    pid="$(cat "$WORK/dl.pid" 2>/dev/null || true)"
    case "$pid" in '' | *[!0-9]*) return 1 ;; esac
    kill -0 "$pid" 2>/dev/null
}
if [ "$DOWNLOAD" -eq 0 ]; then
    ok "skipped (--no-download)"
elif [ -f "$WORK/download.done" ] && [ -d "$DEST" ]; then
    ok "done already ($(cat "$WORK/download.done"))"
elif dl_running; then
    ok "a download started earlier still runs (pid $(cat "$WORK/dl.pid"))"
elif [ "$DRY" -eq 1 ]; then
    say "\$ nohup bash $WORK/dl.sh >$WORK/dl.log 2>&1 &   (hf download $REPO -> $DEST)"
    ok "dry run"
else
    cat >"$WORK/dl.sh" <<EOF
#!/bin/bash
# The checkpoint into the RAM disk; restartable (hf download resumes).
export HF_HUB_ENABLE_HF_TRANSFER=0
cd "$WORK" || exit 1
if [ ! -x "\$HOME/.local/bin/uvx" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1; fi
for try in 1 2 3 4 5 6; do
  date -u +%FT%TZ
  "\$HOME/.local/bin/uvx" --from "huggingface_hub==$HF_VERSION" hf download "$REPO" --local-dir "$DEST" && { echo DOWNLOAD-DONE; date -u +%FT%TZ >"$WORK/download.done"; exit 0; }
  echo "download try \$try failed, retrying in 10 s"; sleep 10
done
echo DOWNLOAD-FAILED; exit 1
EOF
    chmod +x "$WORK/dl.sh"
    say "\$ nohup bash $WORK/dl.sh >$WORK/dl.log 2>&1 &   (hf download $REPO -> $DEST)"
    nohup bash "$WORK/dl.sh" >"$WORK/dl.log" 2>&1 &
    echo $! >"$WORK/dl.pid"
    ok "started in the background"
fi

# ------------------------------------------------------------------ 5 base-image
step 5 base-image
run_logged "$WORK/build_base.log" bash "$FARM/build_base.sh" "$RENTAL" || {
    tail -n 25 "$WORK/build_base.log"
    fail "build_base.sh failed; log: $WORK/build_base.log and $FARM/base/"
}
ok "$(tail -n 1 "$WORK/build_base.log" 2>/dev/null | cut -c1-80)"

# ------------------------------------------------------------------ 6 checkpoint
step 6 checkpoint
if [ "$DOWNLOAD" -eq 0 ]; then
    ok "skipped (--no-download)"
elif [ "$DRY" -eq 1 ]; then
    say "\$ wait for $WORK/dl.sh; du -s $DEST >= $MIN_GB GB"
    ok "dry run"
else
    while dl_running; do sleep 5; done
    [ -f "$WORK/download.done" ] || fail "the download failed; log: $WORK/dl.log (run this script again: hf download resumes)"
    got_gb="$(du -sk "$DEST" 2>/dev/null | awk '{printf "%d", $1 / 1048576}')"
    [ "${got_gb:-0}" -ge "$MIN_GB" ] || fail "$DEST holds ${got_gb:-0} GB, at least $MIN_GB expected; log: $WORK/dl.log"
    ok "$got_gb GB in $DEST, $(find "$DEST" -maxdepth 1 -type f | wc -l | tr -d ' ') files"
fi

# ------------------------------------------------------------------ 7 cards
step 7 cards
if [ "$DRY" -eq 1 ]; then
    say "\$ $FARM/cardq.sh status"
    ok "dry run"
else
    # The card order of this machine, from hl-smi; a map file that says otherwise is from another machine
    # (never the case on a fresh node) and is replaced, with the PCI map that belongs to it.
    map="$(hl-smi -Q index,module_id -f csv,noheader 2>/dev/null | tr -d ' ' | sort -t, -k1,1n | cut -d, -f2 | paste -sd, -)"
    [ -n "$map" ] || fail "hl-smi -Q index,module_id gave no card:module map"
    mkdir -p "$FARM/q"
    if [ "$(head -n 1 "$FARM/q/module_map" 2>/dev/null || true)" != "$map" ]; then
        printf '%s\n' "$map" >"$FARM/q/module_map.tmp.$$" && mv -f "$FARM/q/module_map.tmp.$$" "$FARM/q/module_map"
        rm -f "$FARM/q/pci_map"
    fi
    bash "$FARM/cardq.sh" status >"$WORK/status.txt" 2>&1 || fail "cardq.sh status failed: $(tail -n 3 "$WORK/status.txt")"
    say "card:module map of this machine (module of card index 0, 1, ...): $map"
    # The card indices of each module group and their CPU sets, for CARD.md ("A job that starts its own containers").
    per=$((cpus / NCARDS_WANT))
    idx_of() { # idx_of MODULE: the card index of that module
        local i=0 m
        for m in ${map//,/ }; do
            if [ "$m" = "$1" ]; then printf '%s' "$i"; fi
            i=$((i + 1))
        done
    }
    group() { # group FIRST COUNT: "indices | CPUs"
        local m out="" lo=$(($1 * per)) hi=$((($1 + $2) * per - 1))
        for ((m = $1; m < $1 + $2; m++)); do out="$out${out:+,}$(idx_of "$m")"; done
        printf '%s | %s-%s' "$(printf '%s\n' ${out//,/ } | sort -n | paste -sd, -)" "$lo" "$hi"
    }
    say "module groups of this machine (modules | card indices for HABANA_VISIBLE_DEVICES | CPUs for --cpuset-cpus):"
    half=$((NCARDS_WANT / 2))
    say "  0-$((half - 1)) | $(group 0 "$half")"
    say "  $half-$((NCARDS_WANT - 1)) | $(group "$half" "$half")"
    for ((m = 0; m < NCARDS_WANT; m += 2)); do say "  $m,$((m + 1)) | $(group "$m" 2)"; done
    ok "map $map"
fi

say "BRINGUP-DONE [$(date -u +%Y-%m-%dT%H:%M:%SZ)]: the queue takes jobs. If the map above differs from CARD.md's table, correct the table."
exit 0
