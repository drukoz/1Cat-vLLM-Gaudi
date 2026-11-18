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
#     --no-pull  --reinstall-driver  --force  --dry-run  -h/--help

set -euo pipefail

VAULT="https://vault.habana.ai/artifactory"
DEFAULT_IMAGE="vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest"

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
ORIG_ARGS=("$@")  # kept for the sudo re-exec; the option loop below consumes "$@"

# Ubuntu 22.04 apt runs needrestart, whose dialog would hang unseen (output goes to the step logs).
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") [options]"
    echo "  --version V          Gaudi software release (default: $VERSION)"
    echo "  --build N            release build id (default: $BUILD); the apt pin is V-N"
    echo "  --image IMG          container image (default: $DEFAULT_IMAGE)"
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

PIN="$VERSION-$BUILD"
INSTALLER_URL="$VAULT/gaudi-installer/$VERSION/habanalabs-installer.sh"
HCR_DEB_URL="$VAULT/debian/jammy/pool/main/h/habanacontainerruntime/habanalabs-container-runtime-$PIN.amd64.deb"
KEYRING="/usr/share/keyrings/habana-artifactory.gpg"
REPO_LIST="/etc/apt/sources.list.d/habanalabs.list"
SYNAPSEAI_LIST="/etc/apt/sources.list.d/habanalabs_synapseai.list"
DAEMON_JSON="/etc/docker/daemon.json"
CARDS=0

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
    echo "host step 2 (apt prerequisites):"
    printf '%s\n' \
        "apt-get update" \
        "apt-get install -y linux-headers-$(uname -r)  # remedy if absent: apt-get install -y linux-headers-generic linux-image-generic, reboot, rerun" \
        "apt-get install -y curl wget gnupg ca-certificates pciutils sudo python3"
    echo
    echo "host step 3 (driver): skipped when hl-smi exists and habanalabs-dkms == $PIN (and not --reinstall-driver)"
    printf '%s\n' \
        "dpkg-query -W -f='\${Version}' habanalabs-dkms" \
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
        "apt-get install -y docker.io" \
        "systemctl enable --now docker"
    echo
    echo "host step 6 (container runtime): skipped when 'dpkg -s habanalabs-container-runtime' succeeds"
    printf '%s\n' \
        "curl -fsSL $VAULT/api/gpg/key/public | gpg --dearmor -o $KEYRING  # only when $SYNAPSEAI_LIST is absent" \
        "echo 'deb [signed-by=$KEYRING] $VAULT/debian jammy main' > $REPO_LIST" \
        "apt-get update" \
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

step2_apt() {
    step_start 2 "apt prerequisites"
    if ! run_logged "apt-get update"; then
        step_failed "apt-get update failed; check the network and /etc/apt/sources.list.d"
    fi
    if ! run_logged "apt-get install -y linux-headers-$(uname -r)"; then
        step_failed "cannot install linux-headers-$(uname -r) (custom cloud kernel?). Remedy: apt-get install -y linux-headers-generic linux-image-generic, reboot into the new kernel, then rerun this script"
    fi
    if ! run_logged "apt-get install -y curl wget gnupg ca-certificates pciutils sudo python3"; then
        step_failed "apt-get install of the prerequisites (curl wget gnupg ca-certificates pciutils sudo python3) failed"
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
    echo "installing the driver with habanalabs-installer.sh (this takes 5-15 min)..." >&2
    if ! run_logged "cd $LOG_DIR && ./habanalabs-installer.sh install --type base -y"; then
        step_failed "habanalabs-installer.sh install --type base -y failed; see $STEP_LOG"
    fi
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
    if [ ! -f "$SYNAPSEAI_LIST" ]; then
        if ! run_logged "curl -fsSL $VAULT/api/gpg/key/public | gpg --dearmor -o $KEYRING"; then
            step_failed "cannot fetch and dearmor the Habana GPG key from $VAULT/api/gpg/key/public"
        fi
        if ! run_logged "echo 'deb [signed-by=$KEYRING] $VAULT/debian jammy main' > $REPO_LIST"; then
            step_failed "cannot write $REPO_LIST"
        fi
    fi
    if ! run_logged "apt-get update"; then
        step_failed "apt-get update failed before installing the container runtime"
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
    local drv hlv dkr rt img
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
