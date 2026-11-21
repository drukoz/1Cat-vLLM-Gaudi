#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""decode_scaling.py stand-in for serve_probe.sh: the build's own decode_scaling (staged next to it as
decode_scaling_real.py) with the arguments serve_probe.sh passes, then, against the same server while it still runs,
native-replay's text_check.py (next to it) at 1 stream (6 prompts) and at 32 streams, 600 tokens each, into
$FARM_OUT/texts_1.json and texts_32.json. Exit code: the probe's, else 4 when a text check failed.

Staged as <probe dir>/decode_scaling.py with serve_probe.sh, decode_scaling_real.py and text_check.py beside it.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FARM_OUT", ".")
TOKENS = os.environ.get("TEXT_CHECK_TOKENS", "600")


def main() -> int:
    rc = subprocess.run([sys.executable, os.path.join(HERE, "decode_scaling_real.py")] + sys.argv[1:]).returncode
    port = None
    if "--port" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1]
    check = [sys.executable, os.path.join(HERE, "text_check.py"), "--tokens", TOKENS] + (["--port", port] if port else [])
    failed = 0
    for streams in ("1", "32"):
        r = subprocess.run(check + ["--streams", streams, "--out", os.path.join(OUT, "texts_%s.json" % streams)])
        print("TEXTCHECK streams %s rc %d" % (streams, r.returncode), flush=True)
        failed = failed or r.returncode
    return rc if rc else (4 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
