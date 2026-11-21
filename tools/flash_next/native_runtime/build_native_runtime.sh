#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# build_native_runtime.sh: build the fork's patched native runtime (HCL, Synapse, the PyTorch bridge and the
# plugin's bridge adapter) for SDK 1.24.1, unattended, inside a container of the SDK's PyTorch image, so that a
# 4-card job can compare stock HPU-graph replay with the native joint replay (bench_native_tp4.py) on the next
# rental. It never touches a card. Recipe: tools/communication/patches/native-runtime/README.md and the three
# upstream READMEs (gaudi-pytorch-bridge v1.24.1, HabanaAI/HCL, the Synapse source snapshot).
#
# STATUS: DRAFT, NEVER RUN (written 2026-11-21 00:10 UTC, track native-replay). Stage 4 (Synapse from the
# third-party snapshot of the whole stack, with the fork's backports) is the uncertain one; the script stops at the
# first failing stage and leaves every stage's log in OUT/logs/.
#
# On any x86-64 Linux machine with Docker (no Gaudi card needed; about 50 GB of disk; more CPUs, faster):
#   docker run --rm -v "$PWD":/plugin:ro -v /some/out:/out \
#       vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest \
#       bash /plugin/tools/flash_next/native_runtime/build_native_runtime.sh /out
# ($PWD = a checkout of this repository at the commit whose patches and adapter are wanted.)
#
# Result: OUT/native-runtime-1.24.1.tar.gz with lib/ (patched libhcl.so, libSynapse.so and the libraries built
# with it), wheels/ (the bridge), ext/ (the plugin's adapter and its ABI sidecar), manifest.json (pins, patch
# hashes, library hashes, versions), logs/. Use (on the node, inside the job's container): unpack under /tmp,
# pip install wheels/*.whl --no-deps --force-reinstall, LD_LIBRARY_PATH=<lib>, and pass ext/ to the benchmark.
set -euo pipefail

OUT="${1:?usage: build_native_runtime.sh OUT_DIR}"
PLUGIN="${PLUGIN:-/plugin}"
WORK="${WORK:-/tmp/native-runtime}"
JOBS="${JOBS:-$(nproc)}"
BRIDGE_REPO=https://github.com/HabanaAI/gaudi-pytorch-bridge.git
BRIDGE_SHA=5176b1b2d6865608514e83a2b1fd66459920676e   # head of branch v1.24.1
SYNAPSE_REPO=https://github.com/heyuanliu-intel/Intel_Gaudi3_Software.git
SYNAPSE_SHA=6de1f66a4ab658685f23dbe007bdc9eb8fe8dbdf  # head of main (a 2024-06 snapshot of the whole stack)
HCL_REPO=https://github.com/HabanaAI/HCL.git
HCL_SHA=11f9114b88038174d8b2c9ac09e2295b75455078      # head of main
mkdir -p "$OUT/logs" "$WORK"
SRC="$WORK/src"
B="$WORK/build"
mkdir -p "$SRC" "$B"

stage() {  # stage NAME COMMAND...: run, log to OUT/logs/NAME.log, stop on failure
    local name="$1"
    shift
    echo "== $(date -u +%T) stage $name"
    if ! "$@" >"$OUT/logs/$name.log" 2>&1; then
        echo "== $(date -u +%T) stage $name FAILED; the end of OUT/logs/$name.log:"
        tail -n 40 "$OUT/logs/$name.log"
        exit 1
    fi
    echo "== $(date -u +%T) stage $name ok"
}

fetch() {  # fetch REPO SHA DIR: exactly that commit, shallow
    local repo="$1" sha="$2" dir="$3"
    rm -rf "$dir"
    git init -q "$dir"
    git -C "$dir" remote add origin "$repo"
    git -C "$dir" fetch -q --depth 1 origin "$sha"
    git -C "$dir" checkout -q FETCH_HEAD
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]
}

s0_versions() {
    python3 -c 'import torch, sys; print("python", sys.version.split()[0]); print("torch", torch.__version__)'
    python3 -c 'import habana_frameworks.torch as h; print("bridge", getattr(h, "__version__", "?"))' || true
    ls -l /usr/lib/habanalabs/ | head -n 60
    ls /usr/include/habanalabs/
    cmake --version | head -n 1
    g++ --version | head -n 1
    protoc --version || true
}

s1_sources() {
    apt-get update -q && apt-get install -y -q git ccache || true  # git is in the image; ccache helps reruns
    fetch "$HCL_REPO" "$HCL_SHA" "$SRC/hcl"
    fetch "$BRIDGE_REPO" "$BRIDGE_SHA" "$SRC/gaudi-pytorch-bridge"
    fetch "$SYNAPSE_REPO" "$SYNAPSE_SHA" "$SRC/trees"
    du -sh "$SRC"/*
}

s2_patches() {  # the fork's tool checks the base commit, the patch hash and a clean tree, then applies
    cd "$PLUGIN"
    python3 tools/communication/apply_native_runtime.py hcl --source "$SRC/hcl" --apply
    python3 tools/communication/apply_native_runtime.py synapse --source "$SRC/trees" --apply
    python3 tools/communication/apply_native_runtime.py bridge --source "$SRC/gaudi-pytorch-bridge" --apply
}

s3_hcl() {  # HabanaAI/HCL README: cmake in hcl/src with HCL_SRC_PKG_DIR and HCL_LIB_DIR; the fork's patch writes
    # the library to the build directory's lib/ (HCL_OUTPUT_DIR) instead of /usr/lib/habanalabs
    mkdir -p "$B/hcl"
    cd "$B/hcl"
    HCL_SRC_PKG_DIR="$SRC/hcl" HCL_LIB_DIR=/usr/lib/habanalabs cmake -DCMAKE_BUILD_TYPE=Release "$SRC/hcl/hcl/src"
    make -j "$JOBS"
    find "$B/hcl" -name 'libhcl*.so*' -printf '%p %s\n'
}

s4_synapse() {  # the snapshot's README: source its environment, then build in this order; the fork's README:
    # Release, TESTS_ENABLED=OFF, AUTO_GENERATED_TESTS=OFF, the patched HCL's headers (HCL_ROOT), protobuf 3.9.0
    # from the snapshot's own CMake rules, keep the MME and rotator products
    mkdir -p "$B/syn"
    cd "$SRC/trees"
    set +u
    # shellcheck disable=SC1091
    source ./habana_set_env_val.sh "$SRC/trees" "$B/syn"
    export HCL_ROOT="$SRC/hcl"
    export TESTS_ENABLED=OFF AUTO_GENERATED_TESTS=OFF
    build_hl_logger -r
    build_hl_gcfg -r
    build_scal -r
    build_mme -r
    build_synapse -r
    set -u
    find "$B/syn" -name 'libSynapse*.so*' -printf '%p %s\n'
}

s5_bridge() {  # gaudi-pytorch-bridge v1.24.1 README, "Code Build", against the image's torch 2.11.0
    export HABANA_SOFTWARE_STACK="$SRC"
    export PYTORCH_MODULES_ROOT_PATH="$SRC/gaudi-pytorch-bridge"
    export HCL_INCLUDE_DIR=/usr/include/habanalabs/
    export MEDIA_ROOT
    MEDIA_ROOT=$(python3 -c "import habana_frameworks.mediapipe, os; print(os.path.dirname(habana_frameworks.mediapipe.__file__))")
    export SPECS_EXT_ROOT=/usr/include/habanalabs/ SYNAPSE_INCLUDE_DIR=/usr/include/habanalabs/
    export SYNAPSE_UTILS_INCLUDE_DIR=/usr/include/habanalabs/ SWTOOLS_SDK_ROOT=/usr/include/habanalabs/
    export BUILD_ROOT="$B/bridge" BUILD_ROOT_LATEST=/usr/lib/habanalabs/
    export PYTORCH_MODULES_RELEASE_BUILD="$B/bridge/pytorch_modules_release"
    export PYTORCH_MODULES_DEBUG_BUILD="$B/bridge/pytorch_modules_debug"
    chmod +xw /usr/lib/habanalabs
    [ -e /usr/include/habanalabs/hl_logger/include ] || ln -s /usr/include/habanalabs/hl_logger /usr/include/habanalabs/hl_logger/include
    pip install -q -r "$PYTORCH_MODULES_ROOT_PATH/requirements.txt"
    "$PYTORCH_MODULES_ROOT_PATH/.devops/build.py" -cr -j "$JOBS"  # -c configure, -r release (no -i: wheels only)
    find "$B/bridge" -name '*.whl' -printf '%p %s\n'
}

s6_adapter() {  # the plugin's adapter against the patched libraries (fork README, "Source and build")
    local backend hccl synapse hcl
    backend=$(find "$B/bridge" -name 'libhabana_pytorch_backend.so' | head -n 1)
    hccl=$(find "$B/bridge" -name '*hccl*.so' -path '*distributed*' | head -n 1)
    synapse=$(find "$B/syn" -name 'libSynapse.so' | head -n 1)
    hcl=$(find "$B/hcl" -name 'libhcl.so' | head -n 1)
    echo "backend $backend hccl $hccl synapse $synapse hcl $hcl"
    cd "$PLUGIN"
    python3 tools/communication/build_tp2_fused_ar_norm_bridge.py \
        --bridge-source "$SRC/gaudi-pytorch-bridge" \
        --cmake-build-directory "$PYTORCH_MODULES_RELEASE_BUILD" \
        --build-directory "$B/ext" \
        --backend-library "$backend" --native-hccl-library "$hccl" \
        --synapse-library "$synapse" --hcl-library "$hcl"
}

s7_pack() {
    local pack="$WORK/pack"
    rm -rf "$pack"
    mkdir -p "$pack/lib" "$pack/wheels" "$pack/ext"
    find "$B/hcl" -name 'libhcl*.so*' -exec cp -a {} "$pack/lib/" \;
    find "$B/syn" \( -name 'libSynapse*.so*' -o -name 'libscal*.so*' -o -name 'libhl_gcfg*.so*' \
        -o -name 'libhl_logger*.so*' -o -name 'libmme*.so*' -o -name 'librotator*.so*' \) -exec cp -a {} "$pack/lib/" \;
    find "$B/bridge" -name '*.whl' -exec cp -a {} "$pack/wheels/" \;
    cp -a "$B/ext/." "$pack/ext/"
    python3 - "$pack" "$PLUGIN" <<'PY'
import hashlib, json, os, subprocess, sys
pack, plugin = sys.argv[1], sys.argv[2]
files = {}
for root, _, names in os.walk(pack):
    for name in names:
        path = os.path.join(root, name)
        files[os.path.relpath(path, pack)] = hashlib.sha256(open(path, "rb").read()).hexdigest()
manifest = json.load(open(os.path.join(plugin, "tools/communication/patches/native-runtime/manifest.json")))
commit = subprocess.run(["git", "-C", plugin, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"sdk": "1.24.1", "plugin_commit": commit or None, "pins": manifest, "files": files},
          open(os.path.join(pack, "manifest.json"), "w"), indent=1)
PY
    cp -a "$OUT/logs" "$pack/"
    tar -C "$pack" -czf "$OUT/native-runtime-1.24.1.tar.gz" .
    ls -l "$OUT/native-runtime-1.24.1.tar.gz"
}

echo "build_native_runtime: $(date -u +%FT%TZ), $JOBS jobs, work $WORK, out $OUT"
stage 0-versions s0_versions
stage 1-sources s1_sources
stage 2-patches s2_patches
stage 3-hcl s3_hcl
stage 4-synapse s4_synapse
stage 5-bridge s5_bridge
stage 6-adapter s6_adapter
stage 7-pack s7_pack
echo "build_native_runtime: done $(date -u +%FT%TZ): $OUT/native-runtime-1.24.1.tar.gz"
