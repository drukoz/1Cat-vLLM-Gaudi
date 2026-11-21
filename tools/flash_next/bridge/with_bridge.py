#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a command with our own build of the Gaudi PyTorch bridge installed over the image's.

    python3 with_bridge.py BUILD_DIR [options] -- COMMAND...

Inside a farm job's container (the card farm's ``bin/farm job|tiny|serve ... --bridge <build-id>`` copies the build
into ``$FARM_OUT/bridge`` and runs this script there). ``BUILD_DIR`` holds the build as native-replay's bridge
builds leave it (``~/farm/native-replay/bridge-build/out/<build-id>/``):

* ``manifest.json``: ``{"build_id", "bridge_commit", "patch_sha256", "built_at", "wheels": {file: sha256},
  "files": {file: [sha256, destination]} (optional: libraries outside the wheels), "env_flag"}``;
* ``wheels/*.whl``: exactly the wheels the manifest names.

Steps, each one fatal when it fails (exit 7: the build cannot be installed; exit 8: the installed bridge does not
load, or what the processes load is not the build):

1. every wheel's (and listed file's) sha256 equals the manifest's; no wheel that the manifest does not name;
2. a probe process records the image's bridge: the package's ``__file__`` and version, the md5 of every ``.so``
   under the package, and the ``.so`` files a process maps after the import (``/proc/self/maps``);
3. ``pip install --no-deps --force-reinstall --no-index`` of the wheels (and the listed files copied to their
   destinations, which must lie under the package or ``/usr/lib/habanalabs``);
4. a fresh probe process imports the bridge, runs one HPU op (``--no-op`` leaves it out) and records the same;
   every ``.so`` of the wheels that lies under the package must be installed with the wheel's md5, and every
   mapped ``.so`` under the package must be one of the wheels' files with the wheel's md5 (at least one must be
   mapped: the bridge was loaded from where pip put it), and a bridge library mapped from anywhere else (a copy
   on ``LD_LIBRARY_PATH``, such as ``/usr/lib/habanalabs``) must have the md5 of the wheel's file of that name;
5. ``bridge.json`` (``--out``, default ``$FARM_OUT/bridge.json``) and one ``BRIDGE <build-id> ok: ...`` line;
   the copied wheels are removed (``--keep-wheels`` keeps them) so results stay small;
6. ``FARM_RECIPE_CACHE`` moves to its subfolder ``bridge-<build-id>`` (a compiled recipe of another bridge build
   is never reused) and ``BRIDGE_BUILD_ID`` is set; then the command replaces this process.

Stdlib only. ``BRIDGE_PYTHON`` names the interpreter that pip and the probes use (default: this one) and
``--package`` the bridge's top-level module (default ``habana_frameworks.torch``): the unit test installs a fake
package into a scratch venv that way.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

EXIT_INSTALL = 7
EXIT_LOAD = 8
PROBE_TAG = "BRIDGE_PROBE "
ALLOWED_FILE_ROOTS = ("/usr/lib/habanalabs/", )

# Runs in a fresh interpreter: argv = [package, "1" to run one HPU op | "0"]. Prints one tagged JSON line.
PROBE = r'''
import hashlib, importlib, json, os, sys
pkg, op = sys.argv[1], sys.argv[2] == "1"
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
rec = {"package": pkg, "python": sys.executable}
try:
    m = importlib.import_module(pkg)
except Exception as exc:
    rec["error"] = "import %s: %s: %s" % (pkg, type(exc).__name__, str(exc)[:400])
    print("BRIDGE_PROBE " + json.dumps(rec), flush=True)
    sys.exit(0)
root = os.path.dirname(os.path.realpath(m.__file__))
rec.update(file=m.__file__, root=root, version=str(getattr(m, "__version__", None)))
so = {}
for d, _, names in os.walk(root):
    for n in names:
        if ".so" in n:
            p = os.path.join(d, n)
            if os.path.isfile(p) and not os.path.islink(p):
                so[os.path.relpath(p, root)] = md5(p)
rec["so"] = so
mapped, other = None, []
if os.path.exists("/proc/self/maps"):
    mapped, seen = {}, set()
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6 or not parts[-1].startswith("/") or ".so" not in parts[-1]:
                continue
            p = os.path.realpath(parts[-1])
            if p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            if p.startswith(root + os.sep):
                mapped[os.path.relpath(p, root)] = md5(p)
            elif os.path.basename(p).startswith(("libhabana_pytorch", "libfake")):
                other.append([p, md5(p)])  # a bridge library loaded from outside the package
            elif "habana" in p or "synapse" in p.lower() or "torch" in p:
                other.append([p, None])
rec["mapped"] = mapped
rec["mapped_other"] = sorted(other)[:120]
try:
    import torch
    rec["torch"] = torch.__version__
except Exception as exc:
    rec["torch"] = "import failed: %s" % exc
if op:
    try:
        import torch
        x = torch.ones(4, device="hpu") * 3
        rec["op"] = float(x.sum().item())
    except Exception as exc:
        rec["op_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:400])
print("BRIDGE_PROBE " + json.dumps(rec), flush=True)
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(code, message, record=None, out=None):
    print("BRIDGE %s FAILED: %s" % ((record or {}).get("build_id", "?"), message), flush=True)
    if record is not None and out:
        record.update(ok=False, error=message)
        write_json(out, record)
    sys.exit(code)


def write_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def probe(python, package, op):
    proc = subprocess.run([python, "-c", PROBE, package, "1" if op else "0"], capture_output=True, text=True,
                          timeout=600)
    for line in proc.stdout.splitlines():
        if line.startswith(PROBE_TAG):
            rec = json.loads(line[len(PROBE_TAG):])
            rec["rc"] = proc.returncode
            return rec
    return {"error": "the probe printed no record (rc %s): %s" % (proc.returncode, (proc.stderr or "")[-800:])}


def wheel_members(path, package):
    """md5 of every ``.so`` member of a wheel that lies under the package's folder, keyed by the path relative to
    that folder (``habana_frameworks/torch/lib/x.so`` -> ``lib/x.so`` for ``habana_frameworks.torch``)."""
    prefix = package.replace(".", "/") + "/"
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith(prefix) and ".so" in name.rsplit("/", 1)[-1] and not name.endswith("/"):
                out[name[len(prefix):]] = hashlib.md5(z.read(name)).hexdigest()
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: with_bridge.py BUILD_DIR [options] -- COMMAND...", file=sys.stderr)
        return 2
    cut = argv.index("--")
    own, command = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(prog="with_bridge.py")
    ap.add_argument("build_dir")
    ap.add_argument("--package", default="habana_frameworks.torch")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-op", action="store_true", help="the probe after the install runs no HPU op")
    ap.add_argument("--keep-wheels", action="store_true")
    ap.add_argument("--release-s", type=float, default=5.0,
                    help="pause after an HPU op probe, so its card is free before the command takes it")
    args = ap.parse_args(own)
    if not command:
        ap.error("no command after --")
    python = os.environ.get("BRIDGE_PYTHON") or sys.executable
    build = os.path.abspath(args.build_dir)
    out = args.out or os.path.join(os.environ.get("FARM_OUT", build), "bridge.json")
    record = {"build_dir": build, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": command}

    # 1. the build is what its manifest says
    try:
        with open(os.path.join(build, "manifest.json")) as f:
            manifest = json.load(f)
    except Exception as exc:  # noqa: BLE001
        fail(EXIT_INSTALL, "no readable manifest.json in %s: %s" % (build, exc), record, out)
    record["build_id"] = str(manifest.get("build_id") or os.path.basename(build))
    record["manifest"] = manifest
    wheels = manifest.get("wheels") or {}
    if not wheels:
        fail(EXIT_INSTALL, "the manifest names no wheel", record, out)
    wheel_dir = os.path.join(build, "wheels")
    present = sorted(n for n in os.listdir(wheel_dir) if n.endswith(".whl")) if os.path.isdir(wheel_dir) else []
    if sorted(wheels) != present:
        fail(EXIT_INSTALL, "wheels present %s, manifest names %s" % (present, sorted(wheels)), record, out)
    for name, want in wheels.items():
        got = sha256(os.path.join(wheel_dir, name))
        if got != want:
            fail(EXIT_INSTALL, "%s: sha256 %s, manifest %s" % (name, got, want), record, out)
    files = manifest.get("files") or {}
    for name, (want, dest) in files.items():
        src = os.path.join(build, "files", name)
        if not os.path.isfile(src) or sha256(src) != want:
            fail(EXIT_INSTALL, "file %s missing or not the manifest's sha256" % name, record, out)
        if not os.path.isabs(dest) or os.path.normpath(dest) != dest:
            fail(EXIT_INSTALL, "file %s: destination %r is not a plain absolute path" % (name, dest), record, out)

    # 2. the image's bridge
    record["before"] = probe(python, args.package, op=False)

    # 3. the install
    paths = [os.path.join(wheel_dir, n) for n in sorted(wheels)]
    pip = subprocess.run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", "--no-index",
                          "--disable-pip-version-check"] + paths, capture_output=True, text=True, timeout=1200)
    with open(os.path.join(build, "pip.log"), "w") as f:
        f.write(pip.stdout + pip.stderr)
    if pip.returncode != 0:
        fail(EXIT_INSTALL, "pip install rc %d: %s" % (pip.returncode, (pip.stderr or pip.stdout)[-600:]), record, out)
    root = (record["before"] or {}).get("root") or ""
    for name, (want, dest) in files.items():
        if not (dest.startswith(ALLOWED_FILE_ROOTS) or (root and dest.startswith(root + os.sep))):
            fail(EXIT_INSTALL, "file %s: destination %s is outside the package and /usr/lib/habanalabs" % (name, dest),
                 record, out)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(build, "files", name), dest + ".bridge-new")
        os.replace(dest + ".bridge-new", dest)

    # 4. what a fresh process loads
    after = probe(python, args.package, op=not args.no_op)
    record["after"] = after
    if after.get("error"):
        fail(EXIT_LOAD, after["error"], record, out)
    if not args.no_op:
        if after.get("op_error") or after.get("op") != 12.0:
            fail(EXIT_LOAD, "the HPU op failed: %s" % (after.get("op_error") or after.get("op")), record, out)
        time.sleep(max(0.0, args.release_s))
    expected = {}
    for path in paths:
        expected.update(wheel_members(path, args.package))
    if not expected:
        fail(EXIT_LOAD, "no wheel holds a .so under %s" % args.package, record, out)
    installed = after.get("so") or {}
    wrong = sorted(rel for rel, md5 in expected.items() if installed.get(rel) != md5)
    if wrong:
        fail(EXIT_LOAD, "%d .so of the wheels not installed with their md5 under %s: %s" % (
            len(wrong), after.get("root"), wrong[:5]), record, out)
    mapped = after.get("mapped")
    if mapped is not None:  # Linux: what the process really loaded
        if not mapped:
            fail(EXIT_LOAD, "no .so under %s is mapped after the import" % after.get("root"), record, out)
        stray = sorted(rel for rel, md5 in mapped.items() if expected.get(rel) != md5)
        if stray:
            fail(EXIT_LOAD, "mapped .so under the package that are not the wheels': %s" % stray[:5], record, out)
        by_name = {os.path.basename(rel): md5 for rel, md5 in expected.items()}
        outside = sorted(path for path, md5 in after.get("mapped_other") or []
                         if md5 is not None and os.path.basename(path) in by_name
                         and by_name[os.path.basename(path)] != md5)
        if outside:  # e.g. a stale copy found through LD_LIBRARY_PATH (/usr/lib/habanalabs)
            fail(EXIT_LOAD, "a bridge library is loaded from outside the package and is not the wheel's: %s"
                 % outside[:3], record, out)
    before_so = (record["before"] or {}).get("so") or {}
    record["changed_so"] = sorted(rel for rel in expected if before_so.get(rel) != expected[rel])
    record["ok"] = True
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not args.keep_wheels:
        shutil.rmtree(wheel_dir, ignore_errors=True)
    write_json(out, record)
    print("BRIDGE %s ok: %s %s at %s; %d .so from the wheels, %d mapped, %d differ from the image's; torch %s" % (
        record["build_id"], args.package, after.get("version"), after.get("root"), len(expected),
        len(mapped or {}), len(record["changed_so"]), after.get("torch")), flush=True)

    # 6. the command, with a recipe cache of this build's own
    cache = os.environ.get("FARM_RECIPE_CACHE")
    if cache:
        sub = os.path.join(cache, "bridge-" + "".join(c if c.isalnum() or c in "-_." else "_"
                                                      for c in record["build_id"]))
        os.makedirs(sub, exist_ok=True)
        os.environ["FARM_RECIPE_CACHE"] = sub
    os.environ["BRIDGE_BUILD_ID"] = record["build_id"]
    sys.stdout.flush()
    os.execvp(command[0], command)
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main())
