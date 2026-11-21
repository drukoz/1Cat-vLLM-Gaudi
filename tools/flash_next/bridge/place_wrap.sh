#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# place_wrap.sh PLACEMENT -- COMMAND...: inside a farm job's container, run COMMAND (a server and its probe, e.g. the
# integration build's serve_probe.sh) and, in the background, wait until that server answers /health on its port
# ($PLACE_PORT, default $FARM_PORT_BASE as serve_probe.sh uses it), then run the script PLACEMENT once (for example
# native-replay's sibling-aware placement of the workers' busy threads), with the words of $PLACE_ARGS as its arguments
# (e.g. "--log <the server's out folder>/server0.log" when the server's log is not $FARM_OUT/server0.log; native-replay's
# place.sh takes --log). Output: $PLACE_LOG, default $FARM_OUT/placement.log. The waiting stops, without a placement,
# when COMMAND ends first or after 60 min, so a server that never got healthy leaves nothing behind that would place
# the next server of the same container (several servers in sequence: one wrap, one --log and one $PLACE_LOG each).
# COMMAND and the placement start with the container's whole CPU set (its cgroup's cpuset): the placement of an
# earlier server moves every other process of the container off that server's reserved vCPUs, the job's shells too,
# and a later server would inherit the smaller set (integration's 00222: 144, 128, 112, ... CPUs; PLACE_KEEP_MASK=1
# keeps the mask the wrap was started with).
set -u
PLACE="${1:?placement script}"
shift
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "place_wrap: no command after --"; exit 2; }
PORT="${PLACE_PORT:-${FARM_PORT_BASE:-8000}}"
LOG="${PLACE_LOG:-${FARM_OUT:-/tmp}/placement.log}"
WRAP_PID=$$  # after the exec below, the pid of COMMAND
MASK_NOTE=""
if [ "${PLACE_KEEP_MASK:-0}" != 1 ]; then
    MASK_NOTE="$(python3 - "$WRAP_PID" 2>&1 <<'PY'
import os, sys
pid = int(sys.argv[1])
for path in ("/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpuset/cpuset.effective_cpus"):
    try:
        text = open(path).read().strip()
    except OSError:
        continue
    cpus = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    before = len(os.sched_getaffinity(pid))
    if cpus:
        os.sched_setaffinity(pid, cpus)
    print("CPU mask %d -> %d CPUs (%s)" % (before, len(os.sched_getaffinity(pid)), path))
    break
else:
    print("CPU mask kept: no cpuset file")
PY
)" || MASK_NOTE="CPU mask kept: ${MASK_NOTE:-the reset failed}"
fi
(
    [ -z "$MASK_NOTE" ] || echo "place_wrap: $MASK_NOTE"
    healthy=0
    for _ in $(seq 1 1800); do
        if curl --noproxy "*" -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then healthy=1; break; fi
        kill -0 "$WRAP_PID" 2>/dev/null || break
        sleep 2
    done
    if [ "$healthy" != 1 ]; then
        echo "place_wrap: no healthy server on port $PORT (the command ended or 60 min passed) at $(date -u +%T): no placement"
        exit 0
    fi
    echo "place_wrap: server on port $PORT healthy at $(date -u +%T); running $PLACE ${PLACE_ARGS:-}"
    # shellcheck disable=SC2086  # PLACE_ARGS is a list of words
    bash "$PLACE" ${PLACE_ARGS:-}
    echo "place_wrap: placement rc $? at $(date -u +%T)"
) >"$LOG" 2>&1 &
exec "$@"
