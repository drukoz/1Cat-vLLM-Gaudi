#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build a CPU-only Python environment for developing/testing the vllm-gaudi
# plugin on a machine without Gaudi hardware (e.g. macOS arm64).
#
# Usage: setup.sh <target-dir> <vllm-commit> [python]
#
#   <target-dir>   directory (outside the worktree) that will hold
#                  venv/ and vllm-src/
#   <vllm-commit>  vLLM git commit to pin
#   [python]       python interpreter for the venv (default: newest of
#                  /opt/homebrew/bin/python3.12, /opt/homebrew/bin/python3.11)
#
# The script is idempotent: a second run with the same target dir and vLLM
# commit skips the clone and every install that is already satisfied, and
# exits quickly when PINS.txt matches the current state.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <target-dir> <vllm-commit> [python]" >&2
    exit 2
fi

TARGET_DIR="$(mkdir -p -- "$1" && cd -- "$1" && pwd)"
VLLM_COMMIT="$2"
PYTHON_BIN="${3:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV="$TARGET_DIR/venv"
VLLM_SRC="$TARGET_DIR/vllm-src"
PINS="$TARGET_DIR/PINS.txt"
SETUP_LOG="$TARGET_DIR/setup.log"
VLLM_REPO="https://github.com/vllm-project/vllm.git"

log() { printf '[setup] %s\n' "$*"; }

die() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

# ---------------------------------------------------------------- python ----
pick_python() {
    local candidates=(
        /opt/homebrew/bin/python3.12
        /opt/homebrew/bin/python3.11
        python3.12
        python3.11
    )
    local c
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}

PLUGIN_COMMIT="$(git -C "$PLUGIN_ROOT" rev-parse HEAD)"

# ------------------------------------------------------- idempotent exit ----
venv_python_version() {
    "$VENV/bin/python" -c 'import platform; print(platform.python_version())' 2>/dev/null || true
}

pins_match() {
    [[ -f "$PINS" && -x "$VENV/bin/python" ]] || return 1
    local py_ver
    py_ver="$(venv_python_version)"
    [[ -n "$py_ver" ]] || return 1
    grep -qx "vllm_commit=$VLLM_COMMIT" "$PINS" &&
        grep -qx "plugin_commit=$PLUGIN_COMMIT" "$PINS" &&
        grep -qx "python=$py_ver" "$PINS"
}

if [[ -z "$PYTHON_BIN" ]] && pins_match; then
    if "$VENV/bin/python" -c "import vllm" >/dev/null 2>&1; then
        log "environment already set up for vllm commit $VLLM_COMMIT (see $PINS); nothing to do"
        exit 0
    fi
    log "PINS.txt matches but 'import vllm' fails; continuing with repairs"
fi

if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(pick_python)" ||
        die "no python3.12/python3.11 found; pass an interpreter explicitly"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found: $PYTHON_BIN"
log "python: $("$PYTHON_BIN" --version 2>&1) ($PYTHON_BIN)"
log "target dir: $TARGET_DIR"
log "vllm commit: $VLLM_COMMIT"
log "plugin root: $PLUGIN_ROOT ($PLUGIN_COMMIT)"

# ------------------------------------------------------------------- venv ----
if [[ ! -x "$VENV/bin/python" ]]; then
    log "creating venv at $VENV"
    "$PYTHON_BIN" -m venv "$VENV"
else
    log "reusing venv at $VENV ($(venv_python_version))"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "pip bootstrap"
python -m pip install --quiet --upgrade pip >/dev/null
python -m pip install --quiet 'packaging>=24.2' 'setuptools>=77.0.3,<85.0.0' \
    'setuptools-scm>=8.0' wheel jinja2 >/dev/null

# --------------------------------------------------------------- vllm src ----
if [[ -d "$VLLM_SRC/.git" ]]; then
    log "reusing $VLLM_SRC"
    if [[ "$(git -C "$VLLM_SRC" rev-parse HEAD 2>/dev/null || true)" != "$VLLM_COMMIT" ]]; then
        log "fetching vllm commit $VLLM_COMMIT"
        git -C "$VLLM_SRC" fetch --depth 1 origin "$VLLM_COMMIT"
        git -C "$VLLM_SRC" checkout --detach --force "$VLLM_COMMIT"
    fi
else
    log "cloning vllm (blobless) into $VLLM_SRC"
    git clone --filter=blob:none --no-checkout "$VLLM_REPO" "$VLLM_SRC"
    git -C "$VLLM_SRC" fetch --depth 1 origin "$VLLM_COMMIT"
    git -C "$VLLM_SRC" checkout --detach FETCH_HEAD
fi
git -C "$VLLM_SRC" rev-parse HEAD | grep -qx "$VLLM_COMMIT" ||
    die "vllm-src HEAD is not $VLLM_COMMIT"

# ------------------------------------------------------------------ torch ----
TORCH_PIN=""
for f in "$VLLM_SRC"/requirements/*.txt "$VLLM_SRC"/requirements/*/*.txt; do
    [[ -f "$f" ]] || continue
    line="$(grep -h -E '^torch==' "$f" | head -1 || true)"
    if [[ -n "$line" ]]; then
        TORCH_PIN="${line#torch==}"
        TORCH_PIN="${TORCH_PIN%%;*}"     # drop env markers (linux-only +cpu pins)
        TORCH_PIN="${TORCH_PIN%%+*}"     # drop local version (PyPI has plain wheels)
        TORCH_PIN="${TORCH_PIN//[[:space:]]/}"
        break
    fi
done
[[ -n "$TORCH_PIN" ]] || die "no torch== pin found in $VLLM_SRC/requirements/"
log "torch pin from vllm requirements: $TORCH_PIN"

INSTALLED_TORCH="$(python - <<'PY'
try:
    import torch
    print(torch.__version__.split('+')[0])
except ImportError:
    pass
PY
)"
if [[ "$INSTALLED_TORCH" == "$TORCH_PIN" ]]; then
    log "torch $TORCH_PIN already installed"
else
    log "installing torch==$TORCH_PIN from PyPI"
    if ! python -m pip install --quiet "torch==$TORCH_PIN"; then
        NEAREST="$(python -m pip index versions torch 2>/dev/null |
            sed -n 's/^Available versions: \(.*\)/\1/p' |
            tr ',' '\n' |
            sed 's/^ *//;s/ *$//' |
            grep -vE '[a-zA-Z]' |
            head -1)"
        [[ -n "$NEAREST" ]] || die "pip cannot find torch==$TORCH_PIN nor any stable version for this platform"
        log "WARNING: torch==$TORCH_PIN unavailable; installing nearest available version $NEAREST"
        python -m pip install "torch==$NEAREST"
    fi
fi
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
log "torch: $TORCH_VER"

# -------------------------------------------------- vllm build requires ----
BUILD_REQ=""
for f in "$VLLM_SRC/requirements/build/cuda.txt" "$VLLM_SRC/requirements/build.txt"; do
    if [[ -f "$f" ]]; then
        BUILD_REQ="$f"
        break
    fi
done
[[ -n "$BUILD_REQ" ]] || die "no build requirements file at $VLLM_SRC/requirements/"
log "installing vllm build requirements from ${BUILD_REQ#$VLLM_SRC/} (torch lines removed, per-line)"
# Per-line installs: CUDA-only packages (nvidia-*) have no macOS wheels;
# lines that pip cannot satisfy on this platform are skipped and logged.
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* || "$line" == -* ]] && continue
    python -m pip install --quiet -- "$line" >/dev/null 2>&1 ||
        log "skipped build requirement (unavailable on this platform): $line"
done < <(sed '/^torch/d' "$BUILD_REQ")

# ------------------------------------------------------------- vllm empty ----
# "Fresh" means: importable, from this checkout, AND the installed version
# metadata carries this commit's hash. Checking the hash makes a re-pin
# (checkout of a new commit in a reused clone) reinstall vLLM, which
# regenerates the setuptools-scm version file and refreshes dependencies.
vllm_fresh() {
    python - "$VLLM_SRC" "$VLLM_COMMIT" <<'PY'
import os
import pathlib
import sys

import vllm

src = os.path.realpath(sys.argv[1])
path = pathlib.Path(vllm.__file__).resolve()
if not str(path).startswith(src + os.sep):
    sys.exit(1)
sys.exit(0 if f"g{sys.argv[2][:7]}" in vllm.__version__ else 1)
PY
}
if vllm_fresh >/dev/null 2>&1; then
    log "vllm (editable, from $VLLM_SRC at $VLLM_COMMIT) already importable"
else
    log "installing vllm from source with VLLM_TARGET_DEVICE=empty"
    VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e "$VLLM_SRC"
fi
# plugin banners go to stdout; the version is the last line
VLLM_VER="$(python -c 'import vllm; print(vllm.__version__)' | tail -1)"
log "vllm: $VLLM_VER"

# ---------------------------------------------------------------- plugin ----
log "installing plugin from $PLUGIN_ROOT"
python -m pip install --no-build-isolation -e "$PLUGIN_ROOT"

# ----------------------------------------------------------- test deps ----
log "installing test dependencies"
python -m pip install --quiet pytest pytest-asyncio pytest-timeout pyyaml >/dev/null

# -------------------------------------------------------------- PINS.txt ----
PY_VER="$(venv_python_version)"
{
    echo "python=$PY_VER"
    echo "torch=$TORCH_VER"
    echo "vllm_commit=$VLLM_COMMIT"
    echo "vllm_version=$VLLM_VER"
    echo "plugin_commit=$PLUGIN_COMMIT"
    echo "setup_finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$PINS"
log "wrote $PINS"

# ---------------------------------------------------------------- report ----
log "import checks:"
if python -c "import torch, vllm; print('  torch', torch.__version__, '| vllm', vllm.__version__)"; then
    :
else
    log "WARNING: import torch/vllm failed"
fi
if python -c "import vllm_gaudi" 2>"$TARGET_DIR/import_vllm_gaudi.err"; then
    log "import vllm_gaudi: OK"
else
    log "import vllm_gaudi: FAILED (expected on machines without habana_frameworks; first line: $(head -1 "$TARGET_DIR/import_vllm_gaudi.err" 2>/dev/null | tail -c 200))"
fi
log "done"
