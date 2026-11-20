# SPDX-License-Identifier: Apache-2.0
"""Host-side test of the fused hyper-connection ops' PyTorch callbacks under ASan/UBSan.

Compiles the unmodified ``csrc/hc_fused/pytorch/hpu_hc_fused.cpp`` against the test stub of
Habana's ``hpu_custom_op_pt2.h`` (``hpu_stub/``), calls the ``output_meta`` and ``fill_params``
callbacks of the six ops for every HPU test case, and checks each result against the Python schema
(the fake kernels on meta tensors), the GUID of the kernel library and ``struct.pack`` of the node
parameters.  No card, no HPU runtime: the ``registerUserCustomOp`` callbacks are plain C++ on
``at::Stack``.  What this cannot prove is listed in ``docs/design/flash-next/hc-fused.md``
("Unproven without a card").  Modelled on ``tests/unit_tests/gdn_core_decode/test_op_callbacks_host.py``.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import struct
import subprocess

import pytest
import torch
import torch.utils.cpp_extension

import tests.unit_tests.hc_fused.hc_ops  # noqa: F401  registers the six schemas and fake kernels
from tests.unit_tests.hc_fused.hc_case import (BF16, GUIDS, HPU_CASES, INPUT_ORDER, OP_NAME, input_shapes,
                                               kernel_params, scalars)

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="no c++ on PATH")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
NAMESPACE = "flashinfer_gaudi"

CASES = HPU_CASES
CASE_NAMES = [case.name for case in CASES]

# name -> (schema op, tensor specs, scalar args, text the error must contain)
NEGATIVES = {
    "neg/norm_hidden_not_vector_multiple":
    ("hc_norm", ["bf16:2,400", "bf16:400"], ["f:1e-06", "i:4"], "multiple of 128"),
    "neg/norm_weight_size": ("hc_norm", ["bf16:2,512", "bf16:256"], ["f:1e-06", "i:4"], "norm_weight"),
    "neg/norm_bad_eps": ("hc_norm", ["bf16:2,512", "bf16:512"], ["f:0.0", "i:4"], "eps"),
    "neg/norm_f32_input": ("hc_norm", ["f32:2,512", "bf16:512"], ["f:1e-06", "i:4"], "bf16"),
    "neg/norm_rank3": ("hc_norm", ["bf16:1,2,512", "bf16:512"], ["f:1e-06", "i:4"], "2-D"),
    "neg/norm_zero_streams": ("hc_norm", ["bf16:2,512", "bf16:512"], ["f:1e-06", "i:0"], "hc_count"),
    "neg/silu_split_rank_too_large": ("hc_silu_split", ["bf16:2,32"], ["i:4", "i:30"], "lora_rank"),
    "neg/gate_mix_shape_mismatch": ("hc_gate_mix", ["bf16:2,512", "bf16:2,256"], ["i:4"], "gate"),
    "neg/combine_block_width": ("hc_combine", ["bf16:2,512", "bf16:2,64", "bf16:2,4"], ["i:4"], "block_output"),
    "neg/combine_injection_width": ("hc_combine", ["bf16:2,512", "bf16:2,128", "bf16:2,3"], ["i:4"], "injection"),
    "neg/combine_norm_rows": ("hc_combine_norm", ["bf16:2,512", "bf16:3,128", "bf16:2,4", "bf16:512"],
                              ["f:1e-06", "i:4"], "block_output"),
    "neg/combine_norm_short_stack": ("hc_combine_norm", ["bf16:2,512", "bf16:2,128", "bf16:2,4", "bf16:512"],
                                     ["f:1e-06"], ""),
}


def _schema(kind_or_op: str) -> str:
    return f"{NAMESPACE}::{kind_or_op}"


def _arg_strings(case) -> list[str]:
    shapes = input_shapes(case)
    parts = [f"t:bf16:{','.join(str(d) for d in shapes[name])}" for name in INPUT_ORDER[case.kind]]
    parts += [f"i:{v}" if isinstance(v, int) else f"f:{v!r}" for v in scalars(case)]
    return parts


def _input_lines() -> list[str]:
    lines = [" ".join([_schema(OP_NAME[case.kind]), case.name] + _arg_strings(case)) for case in CASES]
    for name, (op, tensors, scalar_args, _) in NEGATIVES.items():
        lines.append(" ".join([_schema(op), name] + [f"t:{t}" for t in tensors] + scalar_args))
    return lines


def _expected(case) -> dict:
    """(output dtypes and dims, guid, param bytes) from the fake kernels and struct.pack."""
    shapes = input_shapes(case)
    op = getattr(torch.ops.flashinfer_gaudi, OP_NAME[case.kind])
    meta = op(*[torch.empty(shapes[name], dtype=BF16, device="meta") for name in INPUT_ORDER[case.kind]],
              *scalars(case))
    if not isinstance(meta, tuple):
        meta = (meta, )
    assert all(m.dtype == BF16 for m in meta)
    return {
        "outputs": [f"bf16:{','.join(str(d) for d in m.shape)}" for m in meta],
        "guid": GUIDS[case.kind],
        "bytes": struct.pack("<fii", *kernel_params(case)),
    }


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    """op_callbacks_host.cpp compiled with the command of the GDN host test (ASan + UBSan)."""
    exe = tmp_path_factory.mktemp("hc_op_callbacks_host") / "op_callbacks_host"
    cmd = [
        "c++", "-std=c++20", "-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=undefined", "-Wall", "-Werror", "-I",
        str(HERE / "hpu_stub"), "-I",
        str(REPO / "csrc" / "hc_fused" / "pytorch"),
        str(HERE / "op_callbacks_host.cpp")
    ]
    for inc in torch.utils.cpp_extension.include_paths():
        cmd += ["-isystem", inc]
    lib = torch.utils.cpp_extension.library_paths()[0]
    cmd += ["-L", lib, "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{lib}", "-o", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr[-3000:]}"
    return exe


@pytest.fixture(scope="module")
def parsed(program, tmp_path_factory):
    """One run over every case line; name -> parsed result line."""
    case_file = tmp_path_factory.mktemp("cases") / "cases.txt"
    case_file.write_text("\n".join(_input_lines()) + "\n")
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
    proc = subprocess.run([str(program), str(case_file)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    assert "runtime error:" not in proc.stderr, proc.stderr[-3000:]
    assert "AddressSanitizer" not in proc.stderr, proc.stderr[-3000:]
    lines = {}
    for line in proc.stdout.splitlines():
        if not line:
            continue
        fields = line.split(" ")
        if fields[1] == "ok":
            lines[fields[0]] = {
                "status": "ok",
                "n": int(fields[2]),
                "outputs": fields[3].split(";"),
                "guid": fields[4],
                "size": int(fields[5]),
                "hex": fields[6],
            }
        else:
            lines[fields[0]] = {"status": "error", "message": " ".join(fields[2:])}
    missing = [n for n in CASE_NAMES + list(NEGATIVES) if n not in lines]
    assert not missing, f"no output line for: {missing}\nstdout:\n{proc.stdout}"
    return lines


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_callbacks(case, parsed):
    line = parsed[case.name]
    exp = _expected(case)
    assert line["status"] == "ok", line.get("message")
    assert line["n"] == len(exp["outputs"])
    assert line["outputs"] == exp["outputs"]
    assert line["guid"] == exp["guid"]
    assert line["size"] == len(exp["bytes"]) == 12
    assert line["hex"] == exp["bytes"].hex()


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_rejections(name, parsed):
    line = parsed[name]
    assert line["status"] == "error", line
    assert NEGATIVES[name][3] in line["message"], line["message"]
