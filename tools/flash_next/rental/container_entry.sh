#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# In-container entry for the Flash-Next rental run (RUK-2709 M4). Installs the stack from the
# bundle (unless --skip-install), then runs the orchestrator (tools.flash_next.rental.main).
# Started by run_rental.sh inside the Habana container; every phase is printed.

set -euo pipefail

REPO_ID="Qwen/Qwen3.8-Flash-Next-FP8"
# TPC toolchain package (tpc-clang 2.7.33), matching the 1.24.1 release of the base image.
HABANA_TOOLS_VERSION="1.24.1-482"
# The bundle folder as the container sees it (run_rental.sh and tiny_smoke.sh mount it at /bundle).
# The tests point FN_RENTAL_BUNDLE_ROOT at a stand-in folder.
BUNDLE_ROOT="${FN_RENTAL_BUNDLE_ROOT:-/bundle}"

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") --model DIR [--results DIR] [--run-id ID]"
    echo "       [--variants LIST] [--with-profile] [--from PHASE] [--download] [--skip-install]"
    echo "       [--full-warmup] [--server-timeout S] [--recipe-cache DIR] [--no-warm-pass]"
    echo "       [--quick-gates] [--no-verify-probe] [--load-read-timeout S] [--warm-request-timeout S]"
    echo "       [--hc-fused] [--moe-glue]"
    echo "       $(basename "${BASH_SOURCE[0]}") --install-only [--results DIR] [--run-id ID]"
    echo "  --quick-gates, --no-verify-probe, --load-read-timeout S, --warm-request-timeout S, --hc-fused and"
    echo "  --moe-glue are handed to the orchestrator unchanged (python -m tools.flash_next.rental.main --help)."
}

need_value() {
    # need_value OPTION ARGC — a usage error when OPTION is the last word, so no value follows it.
    if [ "$2" -lt 2 ]; then
        echo "error: $1 needs a value" >&2
        usage >&2
        exit 2
    fi
}

phase() {
    # phase TEXT — the console line of a phase, with the UTC time, so console.log can be timed.
    echo "phase: $* [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
}

MODEL=""
RESULTS="/results"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
VARIANTS=""
WITH_PROFILE=0
FROM_PHASE=""
DOWNLOAD=0
SKIP_INSTALL=0
INSTALL_ONLY=0
FULL_WARMUP=0
SERVER_TIMEOUT=""
RECIPE_CACHE=""
NO_WARM_PASS=0
QUICK_GATES=0
NO_VERIFY_PROBE=0
LOAD_READ_TIMEOUT=""
WARM_REQUEST_TIMEOUT=""
HC_FUSED=0
MOE_GLUE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            need_value "$1" $#
            MODEL="$2"
            shift 2
            ;;
        --results)
            need_value "$1" $#
            RESULTS="$2"
            shift 2
            ;;
        --run-id)
            need_value "$1" $#
            RUN_ID="$2"
            shift 2
            ;;
        --variants)
            need_value "$1" $#
            VARIANTS="$2"
            shift 2
            ;;
        --with-profile)
            WITH_PROFILE=1
            shift
            ;;
        --from)
            need_value "$1" $#
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
        --install-only)
            INSTALL_ONLY=1
            shift
            ;;
        --full-warmup)
            FULL_WARMUP=1
            shift
            ;;
        --server-timeout)
            need_value "$1" $#
            SERVER_TIMEOUT="$2"
            shift 2
            ;;
        --recipe-cache)
            need_value "$1" $#
            RECIPE_CACHE="$2"
            shift 2
            ;;
        --no-warm-pass)
            NO_WARM_PASS=1
            shift
            ;;
        --quick-gates)
            QUICK_GATES=1
            shift
            ;;
        --no-verify-probe)
            NO_VERIFY_PROBE=1
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
        --load-read-timeout)
            need_value "$1" $#
            LOAD_READ_TIMEOUT="$2"
            shift 2
            ;;
        --warm-request-timeout)
            need_value "$1" $#
            WARM_REQUEST_TIMEOUT="$2"
            shift 2
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

if [ -z "$MODEL" ] && [ "$INSTALL_ONLY" -eq 0 ]; then
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
    phase "install $n ($name)"
    if "$@" >"$LOG_DIR/install-$n.log" 2>&1; then
        echo "install $n ($name): ok"
    else
        local rc=$?
        echo "install $n ($name): FAILED (see $LOG_DIR/install-$n.log)" >&2
        tail -n 30 "$LOG_DIR/install-$n.log" >&2
        exit "$rc"
    fi
}

json_str() {
    # json_str VALUE — a JSON string for VALUE, or null when VALUE is empty.
    if [ -z "$1" ]; then
        printf 'null'
    else
        printf '"%s"' "$1"
    fi
}

TOOLS_INSTALLED_BY_SCRIPT=false

# Debs for habanalabs-tools and the packages the image lacks (libboost-dev, ...) are kept on the data
# disk (/models, the checkpoint's parent folder), so only the first run on a node pays for
# `apt-get update`; later containers install from this cache without it. Delete the folder to refresh.
APT_DEB_CACHE="/models/apt-cache/habanalabs-tools-$HABANA_TOOLS_VERSION"

install_toolchain() {
    # install_toolchain — installs habanalabs-tools (tpc-clang) when the image lacks it (the
    # proven tpcsim recipe). Never exits: on failure the run continues with NATIVE=0.
    if command -v tpc-clang >/dev/null 2>&1; then
        TOOLS_INSTALLED_BY_SCRIPT=false
        return 0
    fi
    phase "toolchain (tpc-clang absent, installing habanalabs-tools $HABANA_TOOLS_VERSION)"
    # errexit is off in here (the caller runs `install_toolchain || true`), so the group's status
    # would be ldconfig's; success is judged by tpc-clang being on PATH afterwards.
    {
        export DEBIAN_FRONTEND=noninteractive  # no debconf prompt can wait on stdin
        local cache_ok=0
        if [ -d /models ] && mkdir -p "$APT_DEB_CACHE/partial"; then
            cache_ok=1
        fi
        # Rerun on the same node: the data-disk cache carries the debs, so no apt-get update.
        if [ "$cache_ok" -eq 1 ] && [ -f "$APT_DEB_CACHE/complete" ] \
            && ls "$APT_DEB_CACHE"/*.deb >/dev/null 2>&1; then
            echo "installing habanalabs-tools from the deb cache $APT_DEB_CACHE (no apt-get update)"
            apt-get install -y --no-install-recommends "$APT_DEB_CACHE"/*.deb \
                || { echo "deb cache install failed; falling back to apt-get update"; \
                     rm -f "$APT_DEB_CACHE/complete"; }
            hash -r
        fi
        if ! command -v tpc-clang >/dev/null 2>&1; then
            # The image configures the Habana repo for habanalabs-graph; add it only when absent.
            if ! grep -qs vault.habana.ai /etc/apt/sources.list.d/*; then
                curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public \
                    | gpg --dearmor -o /usr/share/keyrings/habana-artifactory.gpg
                echo "deb [signed-by=/usr/share/keyrings/habana-artifactory.gpg] https://vault.habana.ai/artifactory/debian jammy main" \
                    > /etc/apt/sources.list.d/habanalabs.list
            fi
            # An unrelated stale repo of the image can fail the update; the install decides.
            apt-get update || true
            # First run (or a broken cache): download the debs into the cache, then install them
            # from there, so the next container on this node needs no update and no download.
            if [ "$cache_ok" -eq 1 ]; then
                rm -f "$APT_DEB_CACHE"/*.deb
                if apt-get install -y --no-install-recommends --download-only \
                        -o Dir::Cache::Archives="$APT_DEB_CACHE" habanalabs-tools=1.24.1-482 \
                    && apt-get install -y --no-install-recommends "$APT_DEB_CACHE"/*.deb; then
                    touch "$APT_DEB_CACHE/complete"
                fi
                hash -r
            fi
            if ! command -v tpc-clang >/dev/null 2>&1; then
                apt-get install -y --no-install-recommends habanalabs-tools=1.24.1-482 \
                    || (echo "apt install failed, falling back to the direct deb" \
                        && curl -fsSL -o /tmp/habanalabs-tools.deb \
                           https://vault.habana.ai/artifactory/debian/jammy/pool/main/h/habanalabs-tools/habanalabs-tools_1.24.1-482_amd64.deb \
                        && apt-get install -y --no-install-recommends /tmp/habanalabs-tools.deb \
                        && rm -f /tmp/habanalabs-tools.deb)
            fi
        fi
        # habanalabs-tools ships its .so files without RPATH, so the loader needs:
        echo "/usr/lib/habanatools" > /etc/ld.so.conf.d/habanatools.conf
        ldconfig
    } >"$LOG_DIR/install-0-toolchain.log" 2>&1
    hash -r  # bash cached the failed tpc-clang lookup; forget it before the next command -v
    if command -v tpc-clang >/dev/null 2>&1; then
        TOOLS_INSTALLED_BY_SCRIPT=true
        return 0
    fi
    echo "WARNING: the habanalabs-tools install failed (see $LOG_DIR/install-0-toolchain.log)" >&2
    tail -n 30 "$LOG_DIR/install-0-toolchain.log" >&2
    return 1
}

# plugin/build sits in the bundle folder on the host, so it outlives the container and also a
# `tar xzf` of a newer bundle over the same folder. Its rebuilds go by file time (cmake, setuptools),
# and the files of a bundle carry their commit's time, so objects built from another commit can pass
# for current. The folder is therefore kept only for the bundle that made it: the plugin install
# records the `plugin sha:` of BUNDLE.txt in it, and a start with any other sha (or with none)
# removes the folder first. The same bundle keeps its build, so its later starts stay fast.
PLUGIN_BUILD_DIR="$BUNDLE_ROOT/plugin/build"
BUILD_SHA_FILE="$PLUGIN_BUILD_DIR/.bundle-sha"
BUNDLE_SHA="$(sed -n 's/^plugin sha: *//p' "$BUNDLE_ROOT/BUNDLE.txt" 2>/dev/null | head -n 1 || true)"

remove_stale_build() {
    if [ ! -d "$PLUGIN_BUILD_DIR" ]; then
        return 0
    fi
    local recorded=""
    recorded="$(head -n 1 "$BUILD_SHA_FILE" 2>/dev/null || true)"
    if [ -n "$BUNDLE_SHA" ] && [ "$recorded" = "$BUNDLE_SHA" ]; then
        return 0
    fi
    phase "stale build (plugin/build was made for bundle ${recorded:-<not recorded>}, this bundle is" \
        "${BUNDLE_SHA:-<no BUNDLE.txt sha>}: removing it, the plugin builds from scratch)"
    rm -rf "$PLUGIN_BUILD_DIR"
}

record_build_sha() {
    # After a successful plugin install only: a failed one leaves no record, so the next start
    # removes what it left behind.
    if [ -n "$BUNDLE_SHA" ] && [ -d "$PLUGIN_BUILD_DIR" ]; then
        printf '%s\n' "$BUNDLE_SHA" >"$BUILD_SHA_FILE" \
            || echo "warning: cannot write $BUILD_SHA_FILE; the next start rebuilds the plugin from scratch" >&2
    fi
}

if [ "$SKIP_INSTALL" -eq 0 ]; then
    install_toolchain || true
    TPC_CLANG_PATH="$(command -v tpc-clang 2>/dev/null || true)"
    if [ -n "$TPC_CLANG_PATH" ]; then
        NATIVE=1
        TPC_CLANG=true
        TPC_CLANG_VERSION="$("$TPC_CLANG_PATH" --version 2>/dev/null | head -n 1)"
        phase "toolchain (tpc-clang at $TPC_CLANG_PATH, building the native kernels)"
    else
        NATIVE=0
        TPC_CLANG=false
        TPC_CLANG_VERSION=""
        echo "WARNING: tpc-clang unavailable after the habanalabs-tools install attempt;" \
            "building WITHOUT native kernels (fused GDN, fused hyper-connections and flashinfer):" \
            "the servers run the PyTorch GDN path" >&2
    fi
    # toolchain.json records what the toolchain step did; the G1b gate reads the libraries, this
    # records why they are (or are not) there.
    HABANA_TOOLS_DPKG="$(dpkg-query -W -f='${Version}' habanalabs-tools 2>/dev/null || true)"
    printf '{"tpc_clang": %s, "tpc_clang_path": %s, "tpc_clang_version": %s, "habanalabs_tools_version": %s, "habanalabs_tools_installed_by_script": %s, "native_build": %d}\n' \
        "$TPC_CLANG" "$(json_str "$TPC_CLANG_PATH")" "$(json_str "$TPC_CLANG_VERSION")" \
        "$(json_str "$HABANA_TOOLS_DPKG")" "$TOOLS_INSTALLED_BY_SCRIPT" "$NATIVE" \
        >"$LOG_DIR/toolchain.json"
    # --no-build-isolation skips the build-system requires of vLLM and the plugin, and the Habana
    # image lacks some (setuptools-rust broke step 1 on the 2026-11-18 rental): install them here,
    # never torch (the image's Habana torch must stay).
    install_step 0b "build tools" pip install "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" "packaging>=24.2" wheel jinja2
    # The bundle's vllm/ is a git archive (no .git), so setuptools_scm cannot find a version and
    # the install fails; pin the version G1 checks (vLLM's own override hook in its setup.py).
    install_step 1 "vllm (engine pin)" env VLLM_TARGET_DEVICE=empty \
        VLLM_VERSION_OVERRIDE=0.28.1rc1.dev392+g8f816a3f6 \
        pip install --no-build-isolation "$BUNDLE_ROOT/vllm"
    remove_stale_build
    # One NATIVE value for the three native builds. The fused hyper-connection library is always built
    # with the others (a failed build of it removes only its own two files and the install goes on);
    # whether a server uses it is the orchestrator's decision (--hc-fused and the card test), not this
    # script's. The MoE glue kernels are in that library too: nothing more is built for --moe-glue.
    install_step 2 "plugin (vllm-gaudi)" env VLLM_TARGET_DEVICE=hpu \
        VLLM_GAUDI_BUILD_FLASHINFER="$NATIVE" VLLM_GAUDI_BUILD_GDN_FUSED="$NATIVE" \
        VLLM_GAUDI_BUILD_HC_FUSED="$NATIVE" VLLM_GAUDI_BUILD_QSA_FUSED="$NATIVE" \
        pip install "$BUNDLE_ROOT/plugin" --no-build-isolation
    record_build_sha
    install_step 3 "model deps" \
        pip install --no-deps transformers==5.16.1 huggingface_hub==1.33.0 \
        tokenizers==0.23.2 safetensors==0.8.0
    # Gate G1b runs the fused GDN card checks with "python -m pytest", and the image ships no pytest
    # (on the 2026-11-18 rental the gate failed on that and turned the fused kernel off). Pinned to
    # the version that rental resolved (its install-3b.log), so a new pytest release cannot change a run.
    install_step 3b "test tools" pip install pytest==9.2.0
    # What the install steps left in the image's Python, for the report-back; never stops the run.
    pip freeze >"$LOG_DIR/pip-freeze.txt" 2>/dev/null \
        || echo "warning: pip freeze failed; $LOG_DIR/pip-freeze.txt is incomplete" >&2
fi

if [ "$DOWNLOAD" -eq 1 ]; then
    phase "download checkpoint ($REPO_ID)"
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

if [ "$INSTALL_ONLY" -eq 1 ]; then
    # tiny_smoke.sh stops here: the stack is installed, and it starts its own check in this container.
    echo "install only: done"
    exit 0
fi

MAIN_ARGS=(--model "$MODEL" --results "$RESULTS" --run-id "$RUN_ID")
if [ "$FULL_WARMUP" -eq 1 ]; then MAIN_ARGS+=(--full-warmup); fi
if [ -n "$SERVER_TIMEOUT" ]; then MAIN_ARGS+=(--server-timeout "$SERVER_TIMEOUT"); fi
if [ -n "$VARIANTS" ]; then MAIN_ARGS+=(--variants "$VARIANTS"); fi
if [ "$WITH_PROFILE" -eq 1 ]; then MAIN_ARGS+=(--with-profile); fi
if [ -n "$FROM_PHASE" ]; then MAIN_ARGS+=(--from "$FROM_PHASE"); fi
if [ -n "$RECIPE_CACHE" ]; then MAIN_ARGS+=(--recipe-cache "$RECIPE_CACHE"); fi
if [ "$QUICK_GATES" -eq 1 ]; then MAIN_ARGS+=(--quick-gates); fi
if [ "$NO_VERIFY_PROBE" -eq 1 ]; then MAIN_ARGS+=(--no-verify-probe); fi
if [ -n "$LOAD_READ_TIMEOUT" ]; then MAIN_ARGS+=(--load-read-timeout "$LOAD_READ_TIMEOUT"); fi
if [ -n "$WARM_REQUEST_TIMEOUT" ]; then MAIN_ARGS+=(--warm-request-timeout "$WARM_REQUEST_TIMEOUT"); fi
if [ "$HC_FUSED" -eq 1 ]; then MAIN_ARGS+=(--hc-fused); fi
if [ "$MOE_GLUE" -eq 1 ]; then MAIN_ARGS+=(--moe-glue); fi
if [ "$NO_WARM_PASS" -eq 1 ]; then MAIN_ARGS+=(--no-warm-pass); fi

phase "orchestrator (python -m tools.flash_next.rental.main)"
cd "$BUNDLE_ROOT/plugin"
exec python -m tools.flash_next.rental.main "${MAIN_ARGS[@]}"
