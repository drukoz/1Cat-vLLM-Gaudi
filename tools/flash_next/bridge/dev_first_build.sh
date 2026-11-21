#!/bin/bash
# Inside the long-lived dev container nr-bridge-dev (farm-base, CPUs 18,37,56,75, no card, /out = the node's
# ~/farm/native-replay/bridge-build): the bridge source at 5176b1b2 in /tmp/gaudi-pytorch-bridge (fetched once), one
# patch applied, a full configure + release build (later builds: dev_build.sh, incremental), the wheels packaged as
# /out/out/BUILD_ID by package_out.py.
#   dev_first_build.sh BUILD_ID PATCH_FILE [NOTE]
set -u
ID="${1:?BUILD_ID}"
PATCH="${2:?PATCH}"
NOTE="${3:-}"
LOG="/out/out-logs/$ID.log"
mkdir -p /out/out-logs /out/out "/out/dev-wheels/$ID"
exec >>"$LOG" 2>&1
echo "dev_first_build: $ID start $(date -u +%FT%TZ) cpus $(nproc)"
if [ ! -d /tmp/gaudi-pytorch-bridge/.git ]; then
    cd /tmp && git init -q gaudi-pytorch-bridge && cd gaudi-pytorch-bridge \
        && git remote add origin https://github.com/HabanaAI/gaudi-pytorch-bridge.git \
        && git fetch -q --depth 1 origin 5176b1b2d6865608514e83a2b1fd66459920676e && git checkout -q FETCH_HEAD \
        || { echo "FETCH FAILED"; exit 2; }
    pip install -q -r /tmp/gaudi-pytorch-bridge/requirements.txt || echo "requirements: pip rc $?"
fi
cd /tmp/gaudi-pytorch-bridge && git checkout -q -- . && git clean -fdq -- backend habana_lazy
git apply --whitespace=nowarn "/out/patches/$PATCH" || { echo "PATCH FAILED"; exit 11; }
git status --short | head -n 20
export HABANA_SOFTWARE_STACK=/tmp PYTORCH_MODULES_ROOT_PATH=/tmp/gaudi-pytorch-bridge
export HCL_INCLUDE_DIR=/usr/include/habanalabs/ SPECS_EXT_ROOT=/usr/include/habanalabs/
export SYNAPSE_INCLUDE_DIR=/usr/include/habanalabs/ SYNAPSE_UTILS_INCLUDE_DIR=/usr/include/habanalabs/
export SWTOOLS_SDK_ROOT=/usr/include/habanalabs/ BUILD_ROOT=/tmp/builds BUILD_ROOT_LATEST=/usr/lib/habanalabs/
export PYTORCH_MODULES_RELEASE_BUILD=/tmp/builds/pytorch_modules_release PYTORCH_MODULES_DEBUG_BUILD=/tmp/builds/pytorch_modules_debug
export MEDIA_ROOT
MEDIA_ROOT=$(python3 -c 'import habana_frameworks.mediapipe, os; print(os.path.dirname(habana_frameworks.mediapipe.__file__))')
chmod +xw /usr/lib/habanalabs
[ -e /usr/include/habanalabs/hl_logger/include ] || ln -s /usr/include/habanalabs/hl_logger /usr/include/habanalabs/hl_logger/include
touch /tmp/dev_marker
echo "build start $(date -u +%T)"
nice -n 19 python3 .devops/build.py -c -r -l -j "$(nproc)" --no-iwyu --no-swig
rc=$?
echo "build end $(date -u +%T) rc $rc"
[ "$rc" -eq 0 ] || exit 12
rm -rf "/out/dev-wheels/$ID" && mkdir -p "/out/dev-wheels/$ID"
find /tmp/builds -name 'habana*.whl' -newer /tmp/dev_marker -exec cp -v {} "/out/dev-wheels/$ID/" \;
python3 /out/package_out.py "$ID" "/out/dev-wheels/$ID" "$LOG" --patch "/out/patches/$PATCH" --note "$NOTE" --root /out/out
echo "dev_first_build: end $(date -u +%FT%TZ)"
