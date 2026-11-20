# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/kernel_test.sh: a kernel card test in a farm container gets the installed native
libraries by path (looked up outside the tree), lazy mode, the tree on PYTHONPATH, and no kernel flags or
recipe cache; a library path the job gives wins; a run in which every test was skipped fails. The
interpreter is a stand-in that prints what pytest would get. Runs on CPU."""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_TEST = REPO_ROOT / "tools" / "flash_next" / "rental" / "farm" / "kernel_test.sh"

FAKE_PY = r"""#!/bin/bash
if [ "$1" = -c ]; then
  case "$PWD" in "$FAKE_TREE"*) echo "$FAKE_TREE/flashinfer_gaudi/lib" ;; *) echo "$FAKE_LIB" ;; esac
  exit 0
fi
echo "pytest-args: $*"
echo "lazy=$PT_HPU_LAZY_MODE hc=${VLLM_HPU_HC_FUSED-unset} rc=${PT_HPU_RECIPE_CACHE_CONFIG-unset}"
echo "pythonpath=$PYTHONPATH"
echo "hck=${HC_FUSED_GAUDI_KERNEL_LIBRARY-} hcn=${HC_FUSED_GAUDI_NATIVE_LIBRARY-}"
echo "gdk=${GDN_FUSED_GAUDI_KERNEL_LIBRARY-} gdn=${GDN_FUSED_GAUDI_NATIVE_LIBRARY-}"
echo "qsk=${QSA_FUSED_GAUDI_KERNEL_LIBRARY-} qsn=${QSA_FUSED_GAUDI_NATIVE_LIBRARY-}"
if [ -n "${FAKE_SUMMARY:-}" ]; then echo "$FAKE_SUMMARY"; fi
exit "${FAKE_RC:-0}"
"""


def rig(tmp_path, names):
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    for name in names:
        (lib / name).touch()
    tree = tmp_path / "tree"
    tree.mkdir()
    py = tmp_path / "python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if "_GAUDI_" not in k and not k.startswith("FARM_")}
    env.update(FARM_PYTHON=str(py), FARM_OUT=str(tmp_path / "out"), FAKE_LIB=str(lib), FAKE_TREE=str(tree))
    return lib, tree, env


def run(tree, env, *args):
    return subprocess.run(["bash", str(KERNEL_TEST), *args], cwd=tree, env=env, capture_output=True, text=True,
                          timeout=60)


def test_every_installed_kernel_library_is_given_and_a_path_of_the_job_wins(tmp_path):
    # A kernel that came after this script (the QSA kernels) gets its two variables by the naming rule
    # lib<name>_gaudi2.so -> <NAME>_GAUDI_KERNEL_LIBRARY, <name>_ops*.so -> <NAME>_GAUDI_NATIVE_LIBRARY.
    lib, tree, env = rig(tmp_path, ("libqsa_fused_gaudi2.so", "qsa_fused_ops.cpython-310-x86_64-linux-gnu.so",
                                    "libhc_fused_gaudi2.so", "hc_fused_ops.cpython-310.so", "notes.txt"))
    proc = run(tree, env, "tests/unit_tests/qsa_fused/test_hpu_op.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "qsk=%s/libqsa_fused_gaudi2.so qsn=%s/qsa_fused_ops.cpython-310-x86_64-linux-gnu.so" % (lib, lib) in proc.stdout
    env_file = (tmp_path / "out" / "kernel-env.txt").read_text()
    assert "QSA_FUSED_GAUDI_KERNEL_LIBRARY=%s/libqsa_fused_gaudi2.so" % lib in env_file
    # the job's own value (farm job --env) is kept, for a kernel library built in the job
    env2 = dict(env, QSA_FUSED_GAUDI_KERNEL_LIBRARY="/job/libqsa.so", HC_FUSED_GAUDI_NATIVE_LIBRARY="/job/hc_ops.so")
    proc = run(tree, env2, "x.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "qsk=/job/libqsa.so qsn=%s/qsa_fused_ops.cpython-310-x86_64-linux-gnu.so" % lib in proc.stdout
    assert "hck=%s/libhc_fused_gaudi2.so hcn=/job/hc_ops.so" % lib in proc.stdout
    assert "kept from the job: HC_FUSED_GAUDI_NATIVE_LIBRARY QSA_FUSED_GAUDI_KERNEL_LIBRARY" in proc.stdout


def test_a_run_in_which_every_test_was_skipped_fails(tmp_path):
    # pytest exits 0 when every test skipped itself (a missing library path); on the card that is a test
    # that did not run, so the job must not end "ok".
    lib, tree, env = rig(tmp_path, ())
    proc = run(tree, dict(env, FAKE_SUMMARY="3 skipped in 0.10s"), "x.py")
    assert proc.returncode == 6, proc.stdout + proc.stderr
    assert "every test was skipped (3 skipped in 0.10s)" in proc.stdout
    assert "ALL-SKIPPED" in (tmp_path / "out" / "pytest.log").read_text()
    for summary in ("2 passed, 1 skipped in 0.20s", "1 passed in 0.20s", "1 skipped, 1 xfailed in 0.20s",
                    "======== 4 passed, 2 skipped, 1 warning in 3.20s ========"):
        proc = run(tree, dict(env, FAKE_SUMMARY=summary), "x.py")
        assert proc.returncode == 0, summary + proc.stdout
    proc = run(tree, dict(env, FAKE_SUMMARY="===== 2 skipped, 1 warning in 0.10s ====="), "x.py")
    assert proc.returncode == 6, proc.stdout
    proc = run(tree, dict(env, FAKE_SUMMARY="3 skipped in 0.10s", FARM_ALLOW_ALL_SKIPPED="1"), "x.py")
    assert proc.returncode == 0, proc.stdout
    proc = run(tree, dict(env, FAKE_SUMMARY="1 failed, 2 skipped in 0.10s", FAKE_RC="1"), "x.py")
    assert proc.returncode == 1, proc.stdout  # pytest's own failure code stays


def test_the_installed_libraries_are_given_by_path(tmp_path):
    lib = tmp_path / "site" / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    for name in ("libhc_fused_gaudi2.so", "hc_fused_ops.cpython-310.so", "libgdn_fused_gaudi2.so",
                 "gdn_fused_ops.cpython-310.so"):
        (lib / name).touch()
    tree = tmp_path / "tree"
    tree.mkdir()
    py = tmp_path / "python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    out = tmp_path / "out"
    env = dict(os.environ, FARM_PYTHON=str(py), FARM_OUT=str(out), FAKE_LIB=str(lib), FAKE_TREE=str(tree),
               VLLM_HPU_HC_FUSED="1", PT_HPU_RECIPE_CACHE_CONFIG="/x,false,1")
    env.pop("PT_HPU_LAZY_MODE", None)
    proc = subprocess.run(["bash", str(KERNEL_TEST), "tests/unit_tests/hc_fused/test_hpu_op.py", "-k", "fwd"],
                          cwd=tree, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    o = proc.stdout
    assert "pytest-args: -m pytest -q -rs -p no:cacheprovider tests/unit_tests/hc_fused/test_hpu_op.py -k fwd" in o
    assert "lazy=1 hc=unset rc=unset" in o
    assert "pythonpath=%s" % tree in o
    assert "hck=%s/libhc_fused_gaudi2.so hcn=%s/hc_fused_ops.cpython-310.so" % (lib, lib) in o
    assert "gdk=%s/libgdn_fused_gaudi2.so gdn=%s/gdn_fused_ops.cpython-310.so" % (lib, lib) in o
    assert "HC_FUSED_GAUDI_KERNEL_LIBRARY=" in (out / "kernel-env.txt").read_text()
    assert "pytest-args" in (out / "pytest.log").read_text()
    env["FAKE_RC"] = "1"
    proc = subprocess.run(["bash", str(KERNEL_TEST), "x.py"], cwd=tree, env=env, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 1
