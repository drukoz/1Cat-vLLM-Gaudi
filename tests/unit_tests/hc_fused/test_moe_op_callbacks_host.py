# SPDX-License-Identifier: Apache-2.0
"""Host-side test of the fused MoE glue ops' PyTorch callbacks under ASan/UBSan.

The MoE ops are registered by the same file as the hyper-connection ops,
``csrc/hc_fused/pytorch/hpu_hc_fused.cpp``.  As ``test_op_callbacks_host.py`` does for those, this
compiles the unmodified file against the test stub of Habana's ``hpu_custom_op_pt2.h``, calls the
``output_meta`` and ``fill_params`` callbacks of the four ops for every HPU test case, and checks each
result against the Python schema (the fake kernels on meta tensors: shapes and dtypes, int32 ids and
fp8 values included), the GUID of the kernel library and ``struct.pack`` of the node parameters.
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

import tests.unit_tests.hc_fused.hc_ops  # noqa: F401  the file under test registers both families
import tests.unit_tests.hc_fused.moe_ops  # noqa: F401  registers the four schemas and fake kernels
from tests.unit_tests.hc_fused.moe_case import (BF16, GUIDS, HPU_CASES, INPUT_ORDER, OP_NAME, input_shapes,
                                                kernel_params, scalars)

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="no c++ on PATH")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
NAMESPACE = "flashinfer_gaudi"
DTYPE_NAME = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32", torch.float8_e4m3fn: "f8"}

CASES = HPU_CASES
CASE_NAMES = [case.name for case in CASES]

# name -> (schema op, tensor specs, scalar args, text the error must contain)
NEGATIVES = {
    "neg/route_top_k_zero": ("moe_route", ["bf16:2,513"], ["i:512", "i:0"], "top_k"),
    "neg/route_top_k_too_large": ("moe_route", ["bf16:2,513"], ["i:512", "i:65"], "top_k"),
    "neg/route_fewer_experts_than_top_k": ("moe_route", ["bf16:2,513"], ["i:9", "i:10"], "experts"),
    "neg/route_more_experts_than_columns": ("moe_route", ["bf16:2,513"], ["i:514", "i:10"], "experts"),
    "neg/route_f32_gates": ("moe_route", ["f32:2,513"], ["i:512", "i:10"], "bf16"),
    "neg/route_rank3": ("moe_route", ["bf16:1,2,513"], ["i:512", "i:10"], "2-D"),
    "neg/silu_mul_odd_columns": ("moe_silu_mul", ["bf16:2,161"], [], "even"),
    "neg/combine_hidden_not_vector_multiple": ("moe_combine", ["bf16:2,400", "bf16:2,513", "bf16:2,400"], ["i:512"],
                                               "multiple of 128"),
    "neg/combine_gate_column_outside": ("moe_combine", ["bf16:2,256", "bf16:2,513", "bf16:2,256"], ["i:513"],
                                        "gate_column"),
    "neg/combine_gate_rows": ("moe_combine", ["bf16:2,256", "bf16:3,513", "bf16:2,256"], ["i:512"], "gates"),
    "neg/combine_routed_shape": ("moe_combine", ["bf16:2,256", "bf16:2,513", "bf16:2,128"], ["i:512"], "routed"),
    "neg/row_quant_gaudi3_range": ("moe_row_quant", ["bf16:2,256"], ["f:448.0"], "fp8_max"),
    "neg/row_quant_hidden_not_vector_multiple": ("moe_row_quant", ["bf16:2,200"], ["f:240.0"], "multiple of 128"),
    "neg/row_quant_short_stack": ("moe_row_quant", ["bf16:2,256"], [], ""),
}


def _schema(op: str) -> str:
    return f"{NAMESPACE}::{op}"


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
    return {
        "outputs": [f"{DTYPE_NAME[m.dtype]}:{','.join(str(d) for d in m.shape)}" for m in meta],
        "guid": GUIDS[case.kind],
        "bytes": struct.pack("<iiif", *kernel_params(case)),
    }


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    """op_callbacks_host.cpp compiled with the command of the hyper-connection host test (ASan + UBSan)."""
    exe = tmp_path_factory.mktemp("moe_op_callbacks_host") / "op_callbacks_host"
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
    assert line["size"] == len(exp["bytes"]) == 16
    assert line["hex"] == exp["bytes"].hex()


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_rejections(name, parsed):
    line = parsed[name]
    assert line["status"] == "error", line
    assert NEGATIVES[name][3] in line["message"], line["message"]


def test_the_output_dtypes_are_the_vendor_op_inputs(parsed):
    """Weights bf16 and fp8 values with an fp32 scale go to the vendor op as they are; the ids are int32."""
    assert [out.split(":")[0] for out in parsed["route_real_r1"]["outputs"]] == ["i32", "bf16"]
    assert [out.split(":")[0] for out in parsed["row_quant_real_r1"]["outputs"]] == ["f8", "f32"]
    assert parsed["row_quant_real_r1"]["outputs"][1] == "f32:1,1"
