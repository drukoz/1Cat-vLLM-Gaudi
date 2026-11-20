# SPDX-License-Identifier: Apache-2.0
"""The tiny-model smoke of the card farm: the track's own tools.flash_next.rental.tiny_smoke (its tiny
checkpoint, its CPU reference, its GDN modes) served on this farm job's cards.

Run inside a farm container (farm_job.sh), from the track's tree (/bundle/plugin):

    python /farm/farm_tiny.py [--tp N] [tiny_smoke options...]

--tp N is the tensor parallel size of the tiny server (default: the job's card count, FARM_NCARDS). The
track's tiny_smoke.py may predate partial cards (its server command has TP 8 built in): the command is
taken from it and only the value after --tensor-parallel-size is replaced, so every track's tree since
2026-11-19 works. Defaults filled in from the job unless given: --out FARM_OUT, --port FARM_PORT_BASE + 10
(this job's own port; with --net=host two servers on one port would collide), --ckpt /tmp/fn-smoke-ckpt
(the image's prebuilt tiny checkpoint, rebuilt by tiny_smoke when it is stale) and --recipe-cache
FARM_RECIPE_CACHE (--no-recipe-cache leaves it out). The exit code is tiny_smoke's.
"""

from __future__ import annotations

import os
import sys

TP_FLAG = "--tensor-parallel-size"


def with_tp(cmd, tp: int) -> list:
    """`cmd` with the value after --tensor-parallel-size set to `tp` (added when there is none)."""
    cmd = [str(word) for word in cmd]
    if TP_FLAG in cmd:
        i = cmd.index(TP_FLAG)
        cmd[i + 1] = str(tp)
    else:
        cmd += [TP_FLAG, str(tp)]
    return cmd


def split_tp(argv: list) -> tuple:
    """(tp or None, the other arguments)."""
    rest, tp, i = [], None, 0
    while i < len(argv):
        if argv[i] == "--tp":
            if i + 1 >= len(argv) or not argv[i + 1].isdigit() or int(argv[i + 1]) < 1:
                raise SystemExit("farm_tiny: --tp takes a positive whole number")
            tp = int(argv[i + 1])
            i += 2
            continue
        if argv[i].startswith("--tp="):
            tp = int(argv[i].split("=", 1)[1])
            i += 1
            continue
        rest.append(argv[i])
        i += 1
    return tp, rest


def has_option(argv: list, name: str) -> bool:
    return any(word == name or word.startswith(name + "=") for word in argv)


def job_defaults(argv: list, env=os.environ) -> list:
    """`argv` with the job's --out, --port, --ckpt and --recipe-cache added where they are missing
    (--no-recipe-cache: no recipe cache)."""
    argv = list(argv)
    no_cache = "--no-recipe-cache" in argv
    argv = [word for word in argv if word != "--no-recipe-cache"]
    if not has_option(argv, "--out"):
        argv += ["--out", env.get("FARM_OUT") or "/results/tiny"]
    if not has_option(argv, "--port"):
        argv += ["--port", str(int(env.get("FARM_PORT_BASE") or 8000) + 10)]
    if not has_option(argv, "--ckpt"):
        argv += ["--ckpt", "/tmp/fn-smoke-ckpt"]
    if not no_cache and not has_option(argv, "--recipe-cache") and env.get("FARM_RECIPE_CACHE"):
        argv += ["--recipe-cache", env["FARM_RECIPE_CACHE"]]
    return argv


def main(argv=None, env=os.environ) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tp, rest = split_tp(argv)
    if tp is None:
        tp = int(env.get("FARM_NCARDS") or 8)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())  # the track's tree: tools.flash_next.rental is its own
    from tools.flash_next.rental import tiny_smoke

    original = tiny_smoke.serve_command

    def serve_command(*args, **kwargs):
        return with_tp(original(*args, **kwargs), tp)

    tiny_smoke.serve_command = serve_command
    rest = job_defaults(rest, env)
    print("farm_tiny: tiny smoke at TP %d on cards %s (modules %s): %s" %
          (tp, env.get("FARM_CARDS", "?"), env.get("FARM_MODULES", "?"), " ".join(rest)), flush=True)
    return tiny_smoke.main(rest)


if __name__ == "__main__":
    sys.exit(main())
