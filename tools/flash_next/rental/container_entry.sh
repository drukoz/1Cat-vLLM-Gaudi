#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# In-container entry for the Flash-Next rental run (RUK-2709 M4). Installs the stack from the
# bundle (unless --skip-install), then runs the orchestrator (tools.flash_next.rental.main).
# Started by run_rental.sh inside the Habana container; every phase is printed.

set -euo pipefail

REPO_ID="Qwen/Qwen3.8-Flash-Next-FP8"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") --model DIR [--results DIR] [--run-id ID]"
    echo "       [--variants LIST] [--with-profile] [--from PHASE] [--download] [--skip-install]"
}

MODEL=""
RESULTS="/results"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
VARIANTS=""
WITH_PROFILE=0
FROM_PHASE=""
DOWNLOAD=0
SKIP_INSTALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --results)
            RESULTS="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --variants)
            VARIANTS="$2"
            shift 2
            ;;
        --with-profile)
            WITH_PROFILE=1
            shift
            ;;
        --from)
            FROM_PHASE="$2"
            shift 2
            ;;
        --download)
            DOWNLOAD=1
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=1
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

if [ -z "$MODEL" ]; then
    echo "error: --model is required" >&2
    usage >&2
    exit 2
fi

LOG_DIR="$RESULTS/$RUN_ID"
mkdir -p "$LOG_DIR"

install_step() {
    # install_step N NAME CMD... — runs the command, logs to install-N.log, stops on failure.
    local n="$1"
    shift
    local name="$1"
    shift
    echo "phase: install $n ($name)"
    if "$@" >"$LOG_DIR/install-$n.log" 2>&1; then
        echo "install $n ($name): ok"
    else
        local rc=$?
        echo "install $n ($name): FAILED (see $LOG_DIR/install-$n.log)" >&2
        tail -n 30 "$LOG_DIR/install-$n.log" >&2
        exit "$rc"
    fi
}

if [ "$SKIP_INSTALL" -eq 0 ]; then
    GDN_FUSED=1
    if command -v tpc-clang >/dev/null 2>&1; then
        echo "tpc-clang found: building the fused GDN kernel"
    else
        GDN_FUSED=0
        echo "WARNING: tpc-clang not found; building without the fused GDN kernel" \
            "(VLLM_GAUDI_BUILD_GDN_FUSED=0). The plugin runs the PyTorch GDN path." >&2
    fi
    install_step 1 "vllm (engine pin)" env VLLM_TARGET_DEVICE=empty \
        pip install --no-build-isolation /bundle/vllm
    install_step 2 "plugin (vllm-gaudi)" env VLLM_TARGET_DEVICE=hpu \
        VLLM_GAUDI_BUILD_FLASHINFER=1 VLLM_GAUDI_BUILD_GDN_FUSED="$GDN_FUSED" \
        pip install /bundle/plugin --no-build-isolation
    install_step 3 "model deps" \
        pip install --no-deps transformers==5.16.1 huggingface_hub==1.33.0 \
        tokenizers==0.23.2 safetensors==0.8.0
fi

if [ "$DOWNLOAD" -eq 1 ]; then
    echo "phase: download checkpoint ($REPO_ID)"
    if ! command -v hf >/dev/null 2>&1; then
        echo "error: the hf CLI is not installed (run the install steps first)" >&2
        exit 1
    fi
    if ! hf download "$REPO_ID" --local-dir "$MODEL" >"$LOG_DIR/install-4-download.log" 2>&1; then
        echo "error: checkpoint download failed (see $LOG_DIR/install-4-download.log)" >&2
        tail -n 30 "$LOG_DIR/install-4-download.log" >&2
        exit 1
    fi
    echo "checkpoint download: ok"
fi

MAIN_ARGS=(--model "$MODEL" --results "$RESULTS" --run-id "$RUN_ID")
if [ -n "$VARIANTS" ]; then MAIN_ARGS+=(--variants "$VARIANTS"); fi
if [ "$WITH_PROFILE" -eq 1 ]; then MAIN_ARGS+=(--with-profile); fi
if [ -n "$FROM_PHASE" ]; then MAIN_ARGS+=(--from "$FROM_PHASE"); fi

echo "phase: orchestrator (python -m tools.flash_next.rental.main)"
cd /bundle/plugin
exec python -m tools.flash_next.rental.main "${MAIN_ARGS[@]}"
