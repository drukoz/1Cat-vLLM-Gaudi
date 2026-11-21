#!/usr/bin/env python3
"""Put a bridge build's wheels into ~/farm/native-replay/bridge-build/out/<build-id>/ atomically (the deployment
interface agreed with the collectives track): wheels/*.whl, build.log, then manifest.json last, all written under
out/.tmp-<build-id> and renamed into place.

    package_out.py BUILD_ID WHEEL_DIR BUILD_LOG [--patch FILE] [--env-flag NAME] [--note TEXT]

Only the bridge's own wheels (names starting with habana) are taken.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_id")
    ap.add_argument("wheel_dir")
    ap.add_argument("build_log")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--env-flag", default="PT_HPU_HPUGRAPH_PREPARED_REPLAY")
    ap.add_argument("--note", default="")
    ap.add_argument("--root", default=os.path.expanduser("~/farm/native-replay/bridge-build/out"))
    args = ap.parse_args()
    final = os.path.join(args.root, args.build_id)
    if os.path.exists(final):
        print("package_out: %s exists already" % final)
        return 2
    tmp = os.path.join(args.root, ".tmp-" + args.build_id)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "wheels"))
    wheels = {}
    for name in sorted(os.listdir(args.wheel_dir)):
        if name.endswith(".whl") and name.lower().startswith("habana"):
            dst = os.path.join(tmp, "wheels", name)
            shutil.copy2(os.path.join(args.wheel_dir, name), dst)
            wheels[name] = sha256(dst)
    if not wheels:
        print("package_out: no habana*.whl in %s" % args.wheel_dir)
        shutil.rmtree(tmp, ignore_errors=True)
        return 3
    shutil.copy2(args.build_log, os.path.join(tmp, "build.log"))
    manifest = {
        "build_id": args.build_id,
        "bridge_repository": "https://github.com/HabanaAI/gaudi-pytorch-bridge.git",
        "bridge_commit": "5176b1b2d6865608514e83a2b1fd66459920676e",
        "patch": os.path.basename(args.patch) if args.patch else None,
        "patch_sha256": sha256(args.patch) if args.patch else None,
        "built_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_image": "farm-base:latest (vault.habana.ai 1.24.1 pytorch-installer-2.11.0)",
        "wheels": wheels,
        "wheel_bytes": {n: os.path.getsize(os.path.join(tmp, "wheels", n)) for n in wheels},
        "env_flag": args.env_flag,
        "note": args.note,
    }
    if args.patch:
        shutil.copy2(args.patch, os.path.join(tmp, os.path.basename(args.patch)))
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:  # last
        json.dump(manifest, fh, indent=1)
    os.rename(tmp, final)
    print("package_out: %s %s" % (final, json.dumps(manifest["wheel_bytes"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
