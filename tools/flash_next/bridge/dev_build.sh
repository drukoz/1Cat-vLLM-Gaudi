#!/bin/bash
# native-replay: an incremental build of the Gaudi PyTorch bridge with one patch, inside the long-lived dev container
# nr-bridge-dev (image farm-bridge-build:5176b1b2: the stock build tree of 5176b1b2 in /tmp; CPUs 18,37,56,75, nice 19,
# no card). Run on the node:
#   dev_build.sh BUILD_ID PATCH_FILE [NOTE]      PATCH_FILE under ~/farm/native-replay/bridge-build/patches/
# The tree is reset to 5176b1b2 (sources only; the build directory is kept, so only what the patch touches is
# recompiled), the patch is applied, build.py builds the release wheels, and package_out.py puts them into
# ~/farm/native-replay/bridge-build/out/BUILD_ID/ (atomically, manifest.json last). Log: out-logs/BUILD_ID.log.
set -u
ID="${1:?BUILD_ID}"
PATCH="${2:?PATCH_FILE}"
NOTE="${3:-}"
H="$HOME/farm/native-replay/bridge-build"
mkdir -p "$H/out-logs" "$H/out" "$H/dev-wheels/$ID"
LOG="$H/out-logs/$ID.log"
echo "dev_build: $ID patch $PATCH start $(date -u +%T)" | tee "$LOG"
sudo -n docker exec nr-bridge-dev bash -c "
set -u
cd /tmp/gaudi-pytorch-bridge && git checkout -q -- . && git clean -fdq -- backend habana_lazy && git status --short | head -n 5
if [ -s '/out/patches/$PATCH' ]; then git apply --whitespace=nowarn '/out/patches/$PATCH' || exit 11; fi
git status --short | head -n 20
export HABANA_SOFTWARE_STACK=/tmp PYTORCH_MODULES_ROOT_PATH=/tmp/gaudi-pytorch-bridge
export HCL_INCLUDE_DIR=/usr/include/habanalabs/ SPECS_EXT_ROOT=/usr/include/habanalabs/
export SYNAPSE_INCLUDE_DIR=/usr/include/habanalabs/ SYNAPSE_UTILS_INCLUDE_DIR=/usr/include/habanalabs/
export SWTOOLS_SDK_ROOT=/usr/include/habanalabs/ BUILD_ROOT=/tmp/builds BUILD_ROOT_LATEST=/usr/lib/habanalabs/
export PYTORCH_MODULES_RELEASE_BUILD=/tmp/builds/pytorch_modules_release PYTORCH_MODULES_DEBUG_BUILD=/tmp/builds/pytorch_modules_debug
export MEDIA_ROOT=\$(python3 -c 'import habana_frameworks.mediapipe, os; print(os.path.dirname(habana_frameworks.mediapipe.__file__))')
touch /tmp/dev_marker
nice -n 19 python3 .devops/build.py -r -l -j 4 --no-iwyu --no-swig || exit 12
rm -rf /out/dev-wheels/$ID && mkdir -p /out/dev-wheels/$ID
find /tmp/builds /tmp/gaudi-pytorch-bridge -name 'habana*.whl' -newer /tmp/dev_marker -exec cp -v {} /out/dev-wheels/$ID/ \;
" >>"$LOG" 2>&1
rc=$?
echo "dev_build: container rc $rc $(date -u +%T)" | tee -a "$LOG"
[ "$rc" -eq 0 ] || exit "$rc"
sudo -n chown -R "$(id -u):$(id -g)" "$H/dev-wheels/$ID" 2>/dev/null
python3 "$H/package_out.py" "$ID" "$H/dev-wheels/$ID" "$LOG" --patch "$H/patches/$PATCH" --note "$NOTE" | tee -a "$LOG"
