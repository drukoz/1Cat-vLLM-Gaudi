#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Prepare a bare Ubuntu 22.04 Gaudi2 node for run_rental.sh (RUK-2709 H9a). The rental ships with
# nothing installed but Ubuntu 22.04; this script installs the four things the run needs, in order:
# the Intel Gaudi driver stack, Docker, the Habana container runtime (registered with Docker as
# runtime "habana"), and the pulled container image. Every step is idempotent; rerun it after a
# reboot. It is pinned to the Gaudi software release the image uses (1.24.1, build 482).
#
# Run it from the unpacked rental bundle, as root:
#     sudo bash prepare_host.sh
#
# Options:
#     --version V (default 1.24.1)  --build N (default 482)  --image IMG
#     --data-disk DEV  --mount DIR (default /models)
#     --no-pull  --reinstall-driver  --force  --dry-run  -h/--help

set -euo pipefail

VAULT="https://vault.habana.ai/artifactory"
DEFAULT_IMAGE="vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest"
FS_LABEL="fn-models"       # ext4 label of the data disk this script makes
DOWNLOAD_NEED_GB=250       # free GB a run_rental.sh --download needs under --model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/prepare-host-logs"
LOG_DIR_READY=0

VERSION="1.24.1"
BUILD="482"
IMAGE="$DEFAULT_IMAGE"
NO_PULL=0
REINSTALL_DRIVER=0
FORCE=0
DRY_RUN=0
DATA_DISK=""
MOUNT="/models"
MOUNT_SET=0
ORIG_ARGS=("$@")  # kept for the sudo re-exec; the option loop below consumes "$@"

# Ubuntu 22.04 apt runs needrestart, whose dialog would hang unseen (output goes to the step logs).
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") [options]"
    echo "  --version V          Gaudi software release (default: $VERSION)"
    echo "  --build N            release build id (default: $BUILD); the apt pin is V-N"
    echo "  --image IMG          container image (default: $DEFAULT_IMAGE)"
    echo "  --data-disk DEV      make ext4 (label $FS_LABEL) on a blank, unmounted DEV, mount it and add an fstab line"
    echo "  --mount DIR          data-disk mount point (default: $MOUNT); needs --data-disk"
    echo "  --no-pull            skip the image pull and the smoke test"
    echo "  --reinstall-driver   replace a driver whose version differs from V-N"
    echo "  --force              continue on a non-22.04 Ubuntu"
    echo "  --dry-run            print every step's commands and exit 0 (no root/apt/docker/network)"
    echo "  -h, --help           show this help"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --build)
            BUILD="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --data-disk)
            DATA_DISK="$2"
            shift 2
            ;;
        --mount)
            MOUNT="$2"
            MOUNT_SET=1
            shift 2
            ;;
        --no-pull)
            NO_PULL=1
            shift
            ;;
        --reinstall-driver)
            REINSTALL_DRIVER=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ "$MOUNT_SET" -eq 1 ] && [ -z "$DATA_DISK" ]; then
    echo "error: --mount needs --data-disk" >&2
    exit 2
fi
if [ "$MOUNT_SET" -eq 1 ]; then
    case "$MOUNT" in
        /*) ;;
        *)
            echo "error: --mount must be an absolute path: $MOUNT" >&2
            exit 2
            ;;
    esac
fi

PIN="$VERSION-$BUILD"
INSTALLER_URL="$VAULT/gaudi-installer/$VERSION/habanalabs-installer.sh"
HCR_DEB_URL="$VAULT/debian/jammy/pool/main/h/habanacontainerruntime/habanalabs-container-runtime-$PIN.amd64.deb"
KEYRING="/usr/share/keyrings/habana-artifactory.gpg"
REPO_LIST="/etc/apt/sources.list.d/habanalabs.list"
SYNAPSEAI_LIST="/etc/apt/sources.list.d/habanalabs_synapseai.list"
DAEMON_JSON="/etc/docker/daemon.json"
CARDS=0
APT_UPDATED=0

# --- step logging -------------------------------------------------------------

ensure_log_dir() {
    if [ "$LOG_DIR_READY" -eq 0 ]; then
        mkdir -p "$LOG_DIR"
        LOG_DIR_READY=1
    fi
}

STEP_N=""
STEP_NAME=""
STEP_LOG=""

step_start() {
    STEP_N="$1"
    STEP_NAME="$2"
    STEP_LOG="$LOG_DIR/step-$STEP_N.log"
    : >"$STEP_LOG"
}

step_ok() { echo "host step $STEP_N ($STEP_NAME): ok"; }
step_skipped() { echo "host step $STEP_N ($STEP_NAME): skipped"; }

step_failed() {
    echo "host step $STEP_N ($STEP_NAME): FAILED" >&2
    if [ -n "$STEP_LOG" ] && [ -f "$STEP_LOG" ]; then
        echo "----- last 30 lines of $STEP_LOG -----" >&2
        tail -n 30 "$STEP_LOG" >&2 || true
        echo "----------------------------------------" >&2
    fi
    echo "remedy: $1" >&2
    exit 1
}

run_logged() {
    {
        printf '\n$ %s\n' "$*"
        bash -c "$*" </dev/null  # no tty: a prompt fails instead of hanging
    } >>"$STEP_LOG" 2>&1
}

# apt_update_once — `apt-get update` at most once per run (it is slow on the rental node). Steps call
# it only right before they install something.
apt_update_once() {
    [ "$APT_UPDATED" -eq 1 ] && return 0
    run_logged "apt-get update" || return 1
    APT_UPDATED=1
}

# apt_update_list FILE — refresh only the repo in FILE (one this run just added), not every repo.
apt_update_list() {
    run_logged "apt-get update -o Dir::Etc::sourcelist=$1 -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0"
}

# pkg_installed NAME — true when dpkg has NAME fully installed.
pkg_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'install ok installed'
}

# --- data disk (step 1b) --------------------------------------------------------

DISK_STATE=""
DISK_UUID=""

disk_refuse() {
    echo "host step 1b (disks): REFUSED: $1" >&2
    echo "remedy: this script formats only a blank, unmounted, unpartitioned device; pick another device from lsblk, or pass --mount DIR" >&2
    exit 1
}

nearest_existing_dir() {
    local d="$1"
    while [ ! -d "$d" ] && [ "$d" != "/" ]; do
        d="$(dirname "$d")"
    done
    printf '%s\n' "$d"
}

disk_free_gb() {
    # whole GB free on the filesystem of the closest existing folder at or above $1 (df -Pk, 1K blocks).
    local base
    base="$(nearest_existing_dir "$1")"
    df -Pk "$base" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}'
}

check_data_disk() {
    # Read-only checks on $DATA_DISK: sets DISK_STATE to blank | ours | ready, or refuses (exit 1).
    # Never writes. A check that cannot run, or answers unexpectedly, is a refusal.
    local dev="$DATA_DISK"
    local t="" nlines="" out="" rc=0 mps="" mp_first="" w_out="" w_rc=0 tline=""
    t="$(lsblk -dn -o TYPE "$dev" 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    if [ "$t" != "disk" ] && [ "$t" != "part" ]; then
        disk_refuse "$dev is not a disk or partition (lsblk TYPE '$t')"
    fi
    nlines="$(lsblk -n -o NAME "$dev" 2>/dev/null | grep -c '[^[:space:]]' || true)"
    if [ "${nlines:-0}" -gt 1 ]; then
        disk_refuse "$dev has partitions"
    fi
    out="$(blkid -p -o export "$dev" 2>/dev/null)" && rc=0 || rc=$?
    DISK_STATE=""
    if [ "$rc" -eq 2 ]; then
        DISK_STATE="blank"
    elif [ "$rc" -eq 0 ]; then
        if printf '%s\n' "$out" | grep -qx 'TYPE=ext4' && printf '%s\n' "$out" | grep -qx "LABEL=$FS_LABEL"; then
            DISK_STATE="ours"
            DISK_UUID="$(printf '%s\n' "$out" | sed -n 's/^UUID=//p' | head -1 || true)"
        else
            tline="$(printf '%s\n' "$out" | grep '^TYPE=' | head -1 || true)"
            disk_refuse "$dev already has a signature (blkid: $tline); it may hold data"
        fi
    else
        disk_refuse "blkid could not read $dev (exit $rc); run as root"
    fi
    mps="$(lsblk -n -o MOUNTPOINT "$dev" 2>/dev/null | grep '[^[:space:]]' || true)"
    if [ -n "$mps" ]; then
        mp_first="$(printf '%s\n' "$mps" | head -1)"
        if [ "$DISK_STATE" = "ours" ] && printf '%s\n' "$mps" | grep -qxF "$MOUNT"; then
            DISK_STATE="ready"
            return 0
        fi
        disk_refuse "$dev is mounted at $mp_first"
    fi
    if [ "$DISK_STATE" = "blank" ]; then
        w_out="$(wipefs -n "$dev" 2>/dev/null)" && w_rc=0 || w_rc=$?
        if [ "$w_rc" -ne 0 ]; then
            disk_refuse "wipefs could not read $dev"
        fi
        if [ -n "$w_out" ]; then
            disk_refuse "wipefs -n sees signatures on $dev"
        fi
    fi
    if [ "$DISK_STATE" != "ready" ]; then
        if [ -d "$MOUNT" ] && [ -n "$(ls -A "$MOUNT" 2>/dev/null)" ]; then
            disk_refuse "$MOUNT is not empty; mounting over it would hide its files: move them, or pass --mount DIR"
        fi
        if grep -q " $MOUNT " /proc/mounts 2>/dev/null; then
            disk_refuse "$MOUNT is already a mount point in /proc/mounts; pass --mount DIR"
        fi
    fi
    return 0
}

list_unused_disks() {
    # Whole disks with no partitions, no filesystem and no mount point, as the operator sees them.
    local n="" t="" parts="" fsty="" mnts="" size=""
    lsblk -dn -o NAME,TYPE 2>/dev/null | while read -r n t; do
        [ "$t" = "disk" ] || continue
        parts="$(lsblk -n -o NAME "/dev/$n" 2>/dev/null | grep -c '[^[:space:]]' || true)"
        fsty="$(lsblk -dn -o FSTYPE "/dev/$n" 2>/dev/null | tr -d '[:space:]' || true)"
        mnts="$(lsblk -n -o MOUNTPOINT "/dev/$n" 2>/dev/null | grep -c '[^[:space:]]' || true)"
        if [ "${parts:-1}" -le 1 ] && [ -z "$fsty" ] && [ "${mnts:-0}" -eq 0 ]; then
            size="$(lsblk -dn -o SIZE "/dev/$n" 2>/dev/null | tr -d '[:space:]' || true)"
            echo "unused disk: /dev/$n ($size): no filesystem, not mounted; to use it: sudo bash $0 --data-disk /dev/$n"
        fi
    done
}

step1b_disks() {
    ensure_log_dir
    step_start 1b disks
    echo "disks:"
    lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT 2>/dev/null || echo "(lsblk failed)"
    echo "free space:"
    df -h / "$(nearest_existing_dir "$MOUNT")" 2>/dev/null || true
    if [ -d /var/lib/docker ]; then
        df -h /var/lib/docker 2>/dev/null || true
    fi
    list_unused_disks || true
    local free_gb uuid
    if [ -z "$DATA_DISK" ]; then
        free_gb="$(disk_free_gb "$MOUNT" || true)"
        if [ -n "$free_gb" ] && [ "$free_gb" -lt "$DOWNLOAD_NEED_GB" ]; then
            echo "note: $MOUNT has $free_gb GB free; run_rental.sh --download needs $DOWNLOAD_NEED_GB GB: use --data-disk with an unused disk above, or the RAM-disk fallback in the run sheet" >&2
        fi
        step_ok
        return 0
    fi
    check_data_disk
    if [ "$DISK_STATE" = "ready" ]; then
        echo "host step 1b (disks): $DATA_DISK already mounted at $MOUNT"
        step_skipped
        return 0
    fi
    if [ "$DISK_STATE" = "blank" ]; then
        if ! run_logged "mkfs.ext4 -F -L $FS_LABEL -m 0 $DATA_DISK"; then
            step_failed "mkfs.ext4 on $DATA_DISK failed; see $STEP_LOG"
        fi
    fi
    uuid="$(blkid -p -o value -s UUID "$DATA_DISK" 2>/dev/null || true)"
    if [ -z "$uuid" ]; then
        step_failed "cannot read the UUID of $DATA_DISK (blkid -p -o value -s UUID) after mkfs"
    fi
    mkdir -p "$MOUNT" || step_failed "cannot create $MOUNT"
    if ! grep -q "UUID=$uuid" /etc/fstab 2>/dev/null; then
        if awk -v m="$MOUNT" '!/^[[:space:]]*(#|$)/ && $2==m {found=1} END {exit found?0:1}' /etc/fstab 2>/dev/null; then
            step_failed "/etc/fstab already mounts something at $MOUNT"
        fi
        if [ ! -f /etc/fstab.fn-backup ]; then
            cp -p /etc/fstab /etc/fstab.fn-backup || step_failed "cannot back up /etc/fstab to /etc/fstab.fn-backup"
        fi
        printf '%s\n' "UUID=$uuid $MOUNT ext4 defaults,nofail,x-systemd.device-timeout=30s 0 2" >>/etc/fstab ||
            step_failed "cannot append the fstab line to /etc/fstab"
    fi
    systemctl daemon-reload 2>/dev/null || true
    if ! run_logged "mount $MOUNT"; then
        step_failed "mount $MOUNT failed; check the fstab line (UUID=$uuid) in $STEP_LOG"
    fi
    if ! lsblk -n -o MOUNTPOINT "$DATA_DISK" 2>/dev/null | grep -qxF "$MOUNT"; then
        step_failed "$DATA_DISK is not mounted at $MOUNT after mount; check lsblk and /etc/fstab"
    fi
    free_gb="$(disk_free_gb "$MOUNT" || true)"
    echo "data disk: $DATA_DISK mounted at $MOUNT, $free_gb GB free"
    step_ok
}

# --- dry run ------------------------------------------------------------------

dry_run() {
    echo "prepare_host.sh --dry-run: nothing is installed; no root, network, apt or docker is used."
    echo "pins: driver/runtime $PIN, image $IMAGE"
    echo
    echo "host step 1 (checks):"
    printf '%s\n' \
        "id -u  # must be 0, else: exec sudo bash $0" \
        "grep -qx 'ID=ubuntu' /etc/os-release; grep -qx 'VERSION_ID=\"22.04\"' /etc/os-release  # --force to override" \
        "uname -m  # must be x86_64" \
        "lspci -d 1da3: | wc -l  # CARDS; 0 is a stop (no Gaudi devices on the PCI bus)" \
        "curl -fsI $INSTALLER_URL  # vault reachable, version $VERSION exists"
    echo
    echo "host step 1b (disks):"
    printf '%s\n' \
        "lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT" \
        "df -h / $(nearest_existing_dir "$MOUNT")  # free space: /, the closest existing folder at or above $MOUNT, and /var/lib/docker" \
        "unused disks: lsblk -dn -o NAME,TYPE  # printed as 'unused disk: /dev/<n> (<size>): ... to use it: sudo bash prepare_host.sh --data-disk /dev/<n>'"
    if [ -n "$DATA_DISK" ]; then
        if [ "$(id -u)" -ne 0 ]; then
            echo "note: not root, so blkid/wipefs may not read the device; the real run repeats these checks as root"
        fi
        check_data_disk
        case "$DISK_STATE" in
            blank)
                printf '%s\n' \
                    "mkfs.ext4 -F -L $FS_LABEL -m 0 $DATA_DISK" \
                    "mkdir -p $MOUNT" \
                    "UUID=<uuid after mkfs> $MOUNT ext4 defaults,nofail,x-systemd.device-timeout=30s 0 2  # appended to /etc/fstab (backed up to /etc/fstab.fn-backup once)" \
                    "systemctl daemon-reload" \
                    "mount $MOUNT"
                ;;
            ours)
                printf '%s\n' \
                    "already ext4 label $FS_LABEL: no mkfs" \
                    "mkdir -p $MOUNT" \
                    "UUID=${DISK_UUID:-<uuid after mkfs>} $MOUNT ext4 defaults,nofail,x-systemd.device-timeout=30s 0 2  # appended to /etc/fstab when missing (backed up once)" \
                    "systemctl daemon-reload" \
                    "mount $MOUNT"
                ;;
            ready)
                printf '%s\n' "already mounted at $MOUNT: nothing to do"
                ;;
        esac
    else
        echo "--data-disk DEV [--mount DIR]: formats a blank DEV as ext4 (label $FS_LABEL) and mounts it at $MOUNT; never without the option"
    fi
    echo
    echo "host step 2 (apt prerequisites):"
    printf '%s\n' \
        "apt-get update  # only when a prerequisite below is missing; at most once per run" \
        "apt-get install -y linux-headers-$(uname -r)  # remedy if absent: apt-get install -y linux-headers-generic linux-image-generic, reboot, rerun" \
        "apt-get install -y curl wget gnupg ca-certificates pciutils sudo python3"
    echo
    echo "host step 3 (driver): skipped when hl-smi exists and habanalabs-dkms == $PIN (and not --reinstall-driver)"
    printf '%s\n' \
        "dpkg-query -W -f='\${Version}' habanalabs-dkms" \
        "curl -fsSL $VAULT/api/gpg/key/public | gpg --dearmor -o $TRUSTED_KEY  # before the installer: its own apt update fails on the unsigned repo otherwise" \
        "curl -fsSL -o $LOG_DIR/habanalabs-installer.sh $INSTALLER_URL" \
        "chmod +x $LOG_DIR/habanalabs-installer.sh" \
        "cd $LOG_DIR && ./habanalabs-installer.sh install --type base -y  # 5-15 min"
    echo
    echo "host step 4 (verify driver):"
    printf '%s\n' \
        "hl-smi -Q index,name -f csv,noheader | grep -c .  # must equal CARDS"
    echo
    echo "host step 5 (docker): skipped when 'docker info' succeeds"
    printf '%s\n' \
        "apt-get update  # only when no earlier step ran it" \
        "apt-get install -y docker.io" \
        "systemctl enable --now docker"
    echo
    echo "host step 6 (container runtime): skipped when 'dpkg -s habanalabs-container-runtime' succeeds"
    printf '%s\n' \
        "curl -fsSL $VAULT/api/gpg/key/public | gpg --dearmor -o $KEYRING  # only when $SYNAPSEAI_LIST is absent" \
        "echo 'deb [signed-by=$KEYRING] $VAULT/debian jammy main' > $REPO_LIST" \
        "apt-get update -o Dir::Etc::sourcelist=$REPO_LIST -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0  # only the Habana list, when this step added it; otherwise the once-per-run update" \
        "apt-get install -y --no-install-recommends habanalabs-container-runtime=$PIN" \
        "curl -fsSL -o /tmp/hcr.deb $HCR_DEB_URL && apt-get install -y --no-install-recommends /tmp/hcr.deb  # direct-deb fallback"
    echo
    echo "host step 7 (register the runtime): merges runtimes.habana into $DAEMON_JSON (keeps other keys)"
    printf '%s\n' \
        "docker info --format '{{json .Runtimes}}' | grep -q '\"habana\"'  # skip when present" \
        "python3 -  # merge runtimes.habana={path:/usr/bin/habana-container-runtime,runtimeArgs:[]} into $DAEMON_JSON" \
        "systemctl restart docker" \
        "docker info --format '{{json .Runtimes}}'  # re-check, must contain habana"
    echo
    echo "host step 8 (image): skipped with --no-pull"
    printf '%s\n' \
        "docker pull $IMAGE"
    echo
    echo "host step 9 (smoke test): skipped with --no-pull"
    printf '%s\n' \
        "docker run --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all $IMAGE hl-smi -Q index,name -f csv,noheader | grep -c .  # must equal CARDS"
    echo
    echo "summary, then: bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8"
}

# --- steps --------------------------------------------------------------------

step1_checks() {
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            exec sudo bash "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
        fi
        echo "error: run as root: sudo bash $0 $*" >&2
        exit 1
    fi
    ensure_log_dir
    step_start 1 checks

    local os_id="" os_ver=""
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release 2>/dev/null || true
        os_id="${ID:-}"
        os_ver="${VERSION_ID:-}"
    fi
    echo "os-release: ID=$os_id VERSION_ID=$os_ver" >>"$STEP_LOG"
    if [ "$os_id" != "ubuntu" ] || [ "$os_ver" != "22.04" ]; then
        if [ "$FORCE" -eq 1 ]; then
            echo "warning: not Ubuntu 22.04 (ID=$os_id VERSION_ID=$os_ver); continuing due to --force" >&2
        else
            step_failed "this script targets Ubuntu 22.04; pass --force to continue on '$os_id $os_ver'"
        fi
    fi

    if [ "$(uname -m)" != "x86_64" ]; then
        step_failed "the Gaudi2 driver is x86-64 only; this host is $(uname -m)"
    fi

    if ! command -v lspci >/dev/null 2>&1; then
        step_failed "lspci is not installed; run: apt-get install -y pciutils, then rerun"
    fi
    CARDS="$(lspci -d 1da3: 2>/dev/null | wc -l | tr -d ' ' || true)"
    echo "lspci 1da3 cards: $CARDS" >>"$STEP_LOG"
    if [ "${CARDS:-0}" -eq 0 ]; then
        step_failed "no Gaudi devices on the PCI bus (lspci -d 1da3: found 0)"
    fi

    if ! command -v curl >/dev/null 2>&1; then
        step_failed "curl is not installed; step 1 needs it to reach the vault (apt-get install -y curl), then rerun"
    fi
    if ! curl -fsI "$INSTALLER_URL" >>"$STEP_LOG" 2>&1; then
        step_failed "cannot reach $INSTALLER_URL (no network, or version $VERSION is wrong)"
    fi
    step_ok
}

# The Habana installer writes its apt repo and runs `apt update` before importing the signing key, so the
# update fails on its own unsigned repo (NO_PUBKEY 1E676CE836A2DE65; seen on 2026-11-19) and the key import
# never runs. Importing the key first, into the installer's own location, makes both steps 2 and 3 succeed.
TRUSTED_KEY="/etc/apt/trusted.gpg.d/habanalabs.gpg"
ensure_habana_key() {
    [ -s "$TRUSTED_KEY" ] && return 0
    run_logged "curl -fsSL $VAULT/api/gpg/key/public | gpg --dearmor -o $TRUSTED_KEY" \
        || step_failed "cannot fetch and dearmor the Habana GPG key from $VAULT/api/gpg/key/public into $TRUSTED_KEY"
}

step2_apt() {
    step_start 2 "apt prerequisites"
    command -v gpg >/dev/null 2>&1 && ensure_habana_key
    local headers="linux-headers-$(uname -r)"
    local missing=()
    local pkg
    for pkg in curl wget gnupg ca-certificates pciutils sudo python3; do
        pkg_installed "$pkg" || missing+=("$pkg")
    done
    if pkg_installed "$headers" && [ "${#missing[@]}" -eq 0 ]; then
        echo "all prerequisites installed; no apt-get update" >>"$STEP_LOG"
    else
        if ! apt_update_once; then
            step_failed "apt-get update failed; check the network and /etc/apt/sources.list.d"
        fi
        if ! pkg_installed "$headers"; then
            if ! run_logged "apt-get install -y $headers"; then
                step_failed "cannot install linux-headers-$(uname -r) (custom cloud kernel?). Remedy: apt-get install -y linux-headers-generic linux-image-generic, reboot into the new kernel, then rerun this script"
            fi
        fi
        if [ "${#missing[@]}" -gt 0 ]; then
            if ! run_logged "apt-get install -y ${missing[*]}"; then
                step_failed "apt-get install of the prerequisites (curl wget gnupg ca-certificates pciutils sudo python3) failed"
            fi
        fi
    fi
    local kver
    kver="$(uname -r)"
    case "$kver" in
        6.8*)
            if ! grep -q 'iommu=pt' /proc/cmdline 2>/dev/null; then
                echo "note: on Ubuntu 22.04.5 / kernel 6.8, set GRUB_CMDLINE_LINUX_DEFAULT to include 'iommu=pt intel_iommu=on' in /etc/default/grub, run update-grub, and reboot (this script does not edit GRUB)." >&2
            fi
            ;;
    esac
    step_ok
}

step3_driver() {
    step_start 3 driver
    local installed=""
    installed="$(dpkg-query -W -f='${Version}' habanalabs-dkms 2>/dev/null || true)"
    echo "habanalabs-dkms installed: '${installed}'  target: '$PIN'  reinstall-driver: $REINSTALL_DRIVER" >>"$STEP_LOG"
    if [ "$REINSTALL_DRIVER" -eq 0 ] && command -v hl-smi >/dev/null 2>&1 && [ "$installed" = "$PIN" ]; then
        step_skipped
        return 0
    fi
    if [ "$REINSTALL_DRIVER" -eq 0 ] && [ -n "$installed" ] && [ "$installed" != "$PIN" ]; then
        echo "driver $installed differs from the image's $VERSION userspace (the previous session ran on such a mismatch); pass --reinstall-driver to replace it" >&2
        step_skipped
        return 0
    fi
    ensure_log_dir
    if ! run_logged "curl -fsSL -o $LOG_DIR/habanalabs-installer.sh $INSTALLER_URL"; then
        step_failed "cannot download $INSTALLER_URL (no network, or version $VERSION is wrong)"
    fi
    run_logged "chmod +x $LOG_DIR/habanalabs-installer.sh" || true
    ensure_habana_key
    echo "installing the driver with habanalabs-installer.sh (this takes 5-15 min)..." >&2
    if ! run_logged "cd $LOG_DIR && ./habanalabs-installer.sh install --type base -y"; then
        step_failed "habanalabs-installer.sh install --type base -y failed; see $STEP_LOG"
    fi
    APT_UPDATED=1  # the installer ran apt-get update itself
    step_ok
}

step4_verify_driver() {
    step_start 4 "verify driver"
    if ! command -v hl-smi >/dev/null 2>&1; then
        step_failed "hl-smi is not installed after the driver step; see $LOG_DIR/step-3.log"
    fi
    local seen
    seen="$(hl-smi -Q index,name -f csv,noheader 2>>"$STEP_LOG" | grep -c . || true)"
    echo "hl-smi cards: $seen  expected (PCI): $CARDS" >>"$STEP_LOG"
    if [ "${seen:-0}" -ne "$CARDS" ]; then
        echo "host step 4 (verify driver): FAILED" >&2
        echo "----- dmesg | grep -i habana | tail -n 20 -----" >&2
        dmesg 2>/dev/null | grep -i habana | tail -n 20 >&2 || true
        echo "-----------------------------------------------" >&2
        echo "remedy: the driver loaded but hl-smi does not see all cards: reboot and rerun this script" >&2
        exit 1
    fi
    step_ok
}

step5_docker() {
    step_start 5 docker
    if docker info >/dev/null 2>&1; then
        step_skipped
        return 0
    fi
    apt_update_once || step_failed "apt-get update failed before installing docker.io"
    if ! run_logged "apt-get install -y docker.io"; then
        step_failed "apt-get install -y docker.io failed"
    fi
    if ! run_logged "systemctl enable --now docker"; then
        step_failed "systemctl enable --now docker failed"
    fi
    if ! docker info >/dev/null 2>&1; then
        step_failed "docker is installed but 'docker info' still fails; check 'systemctl status docker'"
    fi
    step_ok
}

step6_runtime() {
    step_start 6 "container runtime"
    if dpkg -s habanalabs-container-runtime >/dev/null 2>&1; then
        step_skipped
        return 0
    fi
    local added_list=0
    if [ ! -f "$SYNAPSEAI_LIST" ]; then
        if ! run_logged "curl -fsSL $VAULT/api/gpg/key/public | gpg --dearmor -o $KEYRING"; then
            step_failed "cannot fetch and dearmor the Habana GPG key from $VAULT/api/gpg/key/public"
        fi
        if ! run_logged "echo 'deb [signed-by=$KEYRING] $VAULT/debian jammy main' > $REPO_LIST"; then
            step_failed "cannot write $REPO_LIST"
        fi
        added_list=1
    fi
    if [ "$added_list" -eq 1 ]; then
        if ! apt_update_list "$REPO_LIST"; then
            step_failed "apt-get update of $REPO_LIST failed before installing the container runtime"
        fi
    else
        if ! apt_update_once; then
            step_failed "apt-get update failed before installing the container runtime"
        fi
    fi
    if ! run_logged "apt-get install -y --no-install-recommends habanalabs-container-runtime=$PIN"; then
        echo "apt install of habanalabs-container-runtime=$PIN failed; trying the direct deb" >>"$STEP_LOG"
        if ! run_logged "curl -fsSL -o /tmp/hcr.deb $HCR_DEB_URL && apt-get install -y --no-install-recommends /tmp/hcr.deb"; then
            step_failed "cannot install habanalabs-container-runtime=$PIN from apt or $HCR_DEB_URL"
        fi
    fi
    step_ok
}

step7_register() {
    step_start 7 "register the runtime"
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"'; then
        step_skipped
        return 0
    fi
    if ! python3 - >>"$STEP_LOG" 2>&1 <<'PYEOF'
import json
import os

path = "/etc/docker/daemon.json"
data = {}
if os.path.exists(path):
    with open(path) as f:
        txt = f.read().strip()
    if txt:
        data = json.loads(txt)
if not isinstance(data, dict):
    data = {}
runtimes = data.get("runtimes")
if not isinstance(runtimes, dict):
    runtimes = {}
runtimes["habana"] = {"path": "/usr/bin/habana-container-runtime", "runtimeArgs": []}
data["runtimes"] = runtimes
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print("merged runtimes.habana into " + path)
PYEOF
    then
        step_failed "cannot merge runtimes.habana into $DAEMON_JSON (is the existing file valid JSON?)"
    fi
    if ! run_logged "systemctl restart docker"; then
        step_failed "systemctl restart docker failed"
    fi
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"'; then
        step_ok
    else
        step_failed "the habana runtime is still not registered after merging $DAEMON_JSON and restarting Docker"
    fi
}

step8_image() {
    step_start 8 image
    if [ "$NO_PULL" -eq 1 ]; then
        step_skipped
        return 0
    fi
    if ! run_logged "docker pull $IMAGE"; then
        step_failed "docker pull $IMAGE failed (about 10 GB; check the network and vault.habana.ai access)"
    fi
    step_ok
}

step9_smoke() {
    step_start 9 "smoke test"
    if [ "$NO_PULL" -eq 1 ]; then
        step_skipped
        return 0
    fi
    local seen
    seen="$(docker run --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all "$IMAGE" hl-smi -Q index,name -f csv,noheader 2>>"$STEP_LOG" | grep -c . || true)"
    echo "smoke test hl-smi cards: $seen  expected (PCI): $CARDS" >>"$STEP_LOG"
    if [ "${seen:-0}" -ne "$CARDS" ]; then
        step_failed "the smoke test sees $seen cards via --runtime=habana, expected $CARDS; check the runtime registration (step 7) and the driver (step 4)"
    fi
    step_ok
}

summary() {
    local drv hlv dkr rt img free_gb
    drv="$(dpkg-query -W -f='${Version}' habanalabs-dkms 2>/dev/null || echo 'not installed')"
    hlv="-"
    if command -v hl-smi >/dev/null 2>&1; then
        hlv="$(hl-smi -v 2>/dev/null | head -1 || true)"
    fi
    dkr="$(docker --version 2>/dev/null || echo '-')"
    rt="no"
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"habana"'; then
        rt="yes"
    fi
    img="no"
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        img="yes"
    fi
    echo
    echo "host preparation summary"
    echo "----------------------"
    printf '%-20s %s\n' "driver (dkms)" "$drv"
    printf '%-20s %s\n' "hl-smi" "$hlv"
    printf '%-20s %s\n' "docker" "$dkr"
    printf '%-20s %s\n' "runtime habana" "$rt"
    printf '%-20s %s\n' "image pulled" "$img"
    printf '%-20s %s\n' "cards" "$CARDS"
    free_gb="$(disk_free_gb "$MOUNT" || true)"
    printf '%-20s %s\n' "free at $MOUNT" "${free_gb:-?} GB"
    printf '%-20s %s\n' "data disk" "${DATA_DISK:-none}"
    echo
    echo "next: bash run_rental.sh --model /models/Qwen3.8-Flash-Next-FP8"
    echo "  (add --download if the checkpoint is not on the node)"
    echo "run run_rental.sh as root, or add your user to the docker group."
}

main() {
    if [ "$DRY_RUN" -eq 1 ]; then
        dry_run
        exit 0
    fi
    step1_checks
    step1b_disks
    step2_apt
    step3_driver
    step4_verify_driver
    step5_docker
    step6_runtime
    step7_register
    step8_image
    step9_smoke
    summary
}

main "$@"
