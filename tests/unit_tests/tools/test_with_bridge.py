# SPDX-License-Identifier: Apache-2.0
"""tools/flash_next/bridge/with_bridge.py on the CPU: a fake bridge package in a scratch venv stands for the image's
bridge, another build of it for ours. The HPU op and /proc/self/maps are the card's part (macOS has no maps file;
on Linux the mapped check runs too)."""
import base64
import hashlib
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "flash_next" / "bridge" / "with_bridge.py"


def make_wheel(folder: Path, payload: bytes, package: str = "fakebridge", version: str = "1.0") -> Path:
    """A minimal wheel of ``package`` with ``lib/libfake.so`` = ``payload`` (pip accepts it offline)."""
    folder.mkdir(parents=True, exist_ok=True)
    dist = "%s-%s.dist-info" % (package, version)
    files = {
        package + "/__init__.py": ("__version__ = %r\n" % version).encode(),
        package + "/lib/libfake.so": payload,
        dist + "/METADATA": ("Metadata-Version: 2.1\nName: %s\nVersion: %s\n" % (package, version)).encode(),
        dist + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append("%s,sha256=%s,%d" % (name, digest, len(data)))
    record.append(dist + "/RECORD,,")
    path = folder / ("%s-%s-py3-none-any.whl" % (package, version))
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr(dist + "/RECORD", "\n".join(record) + "\n")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """A venv with the 'image's bridge' (fakebridge 1.0, libfake.so = b'stock') installed."""
    base = tmp_path_factory.mktemp("bridge")
    env_dir = base / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    python = env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    stock = make_wheel(base / "stock", b"stock bridge bytes")
    subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--disable-pip-version-check", "-q",
                    str(stock)], check=True)
    return base, python


def make_build(base: Path, name: str, payload: bytes, manifest_extra=None, tamper=False, package="fakebridge"):
    build = base / name
    wheel = make_wheel(build / "wheels", payload, package=package)
    manifest = {"build_id": name, "bridge_commit": "5176b1b2", "wheels": {wheel.name: sha256(wheel)}}
    manifest.update(manifest_extra or {})
    (build / "manifest.json").write_text(json.dumps(manifest))
    if tamper:
        wheel.write_bytes(wheel.read_bytes() + b"x")
    return build


def run(scratch, build: Path, *extra, command=None):
    base, python = scratch
    out = base / (build.name + "-bridge.json")
    env = dict(os.environ, BRIDGE_PYTHON=str(python), FARM_RECIPE_CACHE=str(base / "rc"))
    env.pop("FARM_OUT", None)
    cmd = command or [str(python), "-c",
                      "import os, fakebridge; print('CMD', os.environ['BRIDGE_BUILD_ID'], "
                      "os.environ['FARM_RECIPE_CACHE'], fakebridge.__version__)"]
    proc = subprocess.run([sys.executable, str(SCRIPT), str(build), "--package", "fakebridge", "--no-op", "--out",
                           str(out), *extra, "--", *cmd], capture_output=True, text=True, env=env, timeout=600)
    data = json.loads(out.read_text()) if out.exists() else None
    return proc, data


def test_installs_the_build_and_runs_the_command(scratch):
    base, _ = scratch
    build = make_build(base, "lean-test1", b"our bridge bytes")
    proc, data = run(scratch, build)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BRIDGE lean-test1 ok" in proc.stdout
    # the command ran with the build's id and a recipe cache of the build's own
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("CMD ")][0].split()
    assert line[1] == "lean-test1" and line[2] == str(base / "rc" / "bridge-lean-test1") and line[3] == "1.0"
    assert data["ok"] is True and data["changed_so"] == ["lib/libfake.so"]
    assert data["before"]["so"]["lib/libfake.so"] == hashlib.md5(b"stock bridge bytes").hexdigest()
    assert data["after"]["so"]["lib/libfake.so"] == hashlib.md5(b"our bridge bytes").hexdigest()
    assert not (build / "wheels").exists()  # removed after the install: results stay small
    assert (build / "pip.log").exists()


def test_a_wheel_that_is_not_the_manifests_is_refused(scratch):
    base, _ = scratch
    build = make_build(base, "lean-tampered", b"other bytes", tamper=True)
    proc, data = run(scratch, build)
    assert proc.returncode == 7, proc.stdout + proc.stderr
    assert "sha256" in proc.stdout and data["ok"] is False
    assert (build / "wheels").exists()  # nothing installed, nothing removed


def test_a_wheel_the_manifest_does_not_name_is_refused(scratch):
    base, _ = scratch
    build = make_build(base, "lean-extra", b"third bytes")
    make_wheel(build / "wheels", b"stray", package="strayplugin")
    proc, data = run(scratch, build)
    assert proc.returncode == 7 and "manifest names" in proc.stdout and data["ok"] is False


def test_a_build_without_the_bridges_libraries_is_refused(scratch):
    base, _ = scratch
    build = make_build(base, "lean-wrongpkg", b"elsewhere", package="otherplugin")
    proc, data = run(scratch, build)
    assert proc.returncode == 8, proc.stdout + proc.stderr
    assert "no wheel holds a .so under fakebridge" in proc.stdout and data["ok"] is False


def test_a_build_without_a_manifest_is_refused(scratch, tmp_path):
    (tmp_path / "wheels").mkdir()
    proc, data = run(scratch, tmp_path)
    assert proc.returncode == 7 and "manifest.json" in proc.stdout
