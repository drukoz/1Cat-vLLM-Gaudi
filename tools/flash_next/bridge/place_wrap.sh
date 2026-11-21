#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# place_wrap.sh PLACEMENT -- COMMAND...: inside a farm job's container, run COMMAND (a server and its probe, e.g. the
# integration build's serve_probe.sh) and, in the background, wait until that server answers /health on its port
# ($PLACE_PORT, default $FARM_PORT_BASE as serve_probe.sh uses it), then run the script PLACEMENT once (for example
# native-replay's sibling-aware placement of the workers' busy threads), with the words of $PLACE_ARGS as its arguments
# (e.g. "--log <the server's out folder>/server0.log" when the server's log is not $FARM_OUT/server0.log; native-replay's
# place.sh takes --log). Output: $PLACE_LOG, default $FARM_OUT/placement.log.
set -u
PLACE="${1:?placement script}"
shift
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "place_wrap: no command after --"; exit 2; }
PORT="${PLACE_PORT:-${FARM_PORT_BASE:-8000}}"
LOG="${PLACE_LOG:-${FARM_OUT:-/tmp}/placement.log}"
(
    for _ in $(seq 1 1800); do
        if curl --noproxy "*" -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        sleep 2
    done
    echo "place_wrap: server on port $PORT healthy at $(date -u +%T); running $PLACE ${PLACE_ARGS:-}"
    # shellcheck disable=SC2086  # PLACE_ARGS is a list of words
    bash "$PLACE" ${PLACE_ARGS:-}
    echo "place_wrap: placement rc $? at $(date -u +%T)"
) >"$LOG" 2>&1 &
exec "$@"
