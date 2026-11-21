#!/bin/bash
# Build place.sh: the self-contained bash wrapper around place_workers.py that card jobs run once, as root inside a
# server's container, right after the server is healthy (the collectives track's place_wrap.sh calls it).
#   tools/flash_next/replay/make_place_sh.sh [RULE] > place.sh      RULE: a^1 (default; the rented node, job 00193)
# The wrapper embeds the script, so a job needs only the one file; its second line names the commit it came from.
set -eu
RULE="${1:-a^1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SHA="$(git -C "$HERE" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
echo '#!/bin/bash'
echo "# native-replay: sibling-aware placement of the HPU workers (tools/flash_next/replay/place_workers.py at $SHA)."
echo "# Rule $RULE (job 00193: vCPUs 2k and 2k+1 share a host core). Every process but shells and the server's ancestors keeps off the reserved vCPUs."
echo "# Run once as root in the server's container after health; the background copy re-applies every 2 s until the workers exit."
echo "export PLACE_SIBLING_RULE=\"\${PLACE_SIBLING_RULE:-$RULE}\""
echo "exec python3 - \"\$@\" <<'PY'"
cat "$HERE/place_workers.py"
echo 'PY'
