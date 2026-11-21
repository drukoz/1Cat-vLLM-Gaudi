#!/bin/bash
# native-replay, 2026-11-21 (coordinator's ask, 00:05 UTC): build the UNPATCHED Gaudi PyTorch bridge at 5176b1b2
# (head of v1.24.1, the node's version) inside farm-base, on 4 image-build CPUs at nice 19, no card; collect the
# wheels and the libraries it writes. Run by: docker run -d --rm --name nr-bridge-build --net=host
#   --cpuset-cpus 18,37,56,75 -v ~/farm/native-replay/bridge-build:/out farm-base:latest bash /out/build_bridge.sh
set -uo pipefail
OUT=/out
mkdir -p "$OUT/wheels" "$OUT/lib"
exec >>"$OUT/build.log" 2>&1
echo "start $(date -u +%FT%TZ) cpus $(nproc) python $(python3 -V 2>&1) torch $(python3 -c 'import torch;print(torch.__version__)')"
touch /tmp/marker
cd /tmp && git init -q gaudi-pytorch-bridge && cd gaudi-pytorch-bridge \
    && git remote add origin https://github.com/HabanaAI/gaudi-pytorch-bridge.git \
    && git fetch -q --depth 1 origin 5176b1b2d6865608514e83a2b1fd66459920676e && git checkout -q FETCH_HEAD \
    || { echo "FETCH FAILED"; exit 2; }
echo "fetched $(git rev-parse HEAD) $(date -u +%T)"
export HABANA_SOFTWARE_STACK=/tmp
export PYTORCH_MODULES_ROOT_PATH=/tmp/gaudi-pytorch-bridge
export HCL_INCLUDE_DIR=/usr/include/habanalabs/
export MEDIA_ROOT
MEDIA_ROOT=$(python3 -c "import habana_frameworks.mediapipe, os; print(os.path.dirname(habana_frameworks.mediapipe.__file__))")
export SPECS_EXT_ROOT=/usr/include/habanalabs/ SYNAPSE_INCLUDE_DIR=/usr/include/habanalabs/
export SYNAPSE_UTILS_INCLUDE_DIR=/usr/include/habanalabs/ SWTOOLS_SDK_ROOT=/usr/include/habanalabs/
export BUILD_ROOT=/tmp/builds BUILD_ROOT_LATEST=/usr/lib/habanalabs/
export PYTORCH_MODULES_RELEASE_BUILD=/tmp/builds/pytorch_modules_release
export PYTORCH_MODULES_DEBUG_BUILD=/tmp/builds/pytorch_modules_debug
chmod +xw /usr/lib/habanalabs
[ -e /usr/include/habanalabs/hl_logger/include ] || ln -s /usr/include/habanalabs/hl_logger /usr/include/habanalabs/hl_logger/include
pip install -q -r "$PYTORCH_MODULES_ROOT_PATH/requirements.txt" || echo "requirements: pip rc $?"
echo "build start $(date -u +%T)"
nice -n 19 python3 "$PYTORCH_MODULES_ROOT_PATH/.devops/build.py" -c -r -l -j 4 --no-iwyu --no-swig
rc=$?
echo "build end $(date -u +%T) rc $rc"
find / -xdev -name '*.whl' -newer /tmp/marker -not -path '/proc/*' -exec cp -v {} "$OUT/wheels/" \; 2>/dev/null
find /usr/lib/habanalabs -newer /tmp/marker -type f -exec cp -v {} "$OUT/lib/" \; 2>/dev/null
du -sh /tmp/builds 2>/dev/null
echo "end $(date -u +%FT%TZ) rc $rc"
