# SPDX-License-Identifier: Apache-2.0
"""Tests for farm/kernel_test.sh: a kernel card test in a farm container gets the installed native
libraries by path (looked up outside the tree), lazy mode, the tree on PYTHONPATH, and no kernel flags or
recipe cache. The interpreter is a stand-in that prints what pytest would get. Runs on CPU."""

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
exit "${FAKE_RC:-0}"
"""


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
