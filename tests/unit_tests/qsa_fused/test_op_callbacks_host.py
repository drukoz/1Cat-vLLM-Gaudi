# SPDX-License-Identifier: Apache-2.0
"""Host-side test of the fused QSA ops' PyTorch callbacks under ASan/UBSan (as tests/unit_tests/hc_fused).

Compiles the unmodified ``csrc/qsa_fused/pytorch/hpu_qsa_fused.cpp`` against the test stub of Habana's
``hpu_custom_op_pt2.h`` (``hpu_stub/``), calls the ``output_meta`` and ``fill_params`` callbacks of both ops,
and checks each result against the Python schema (the fake kernels on meta tensors), the GUID of the kernel
library and ``struct.pack`` of the node parameters. No card, no HPU runtime.
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

import flashinfer_gaudi.qsa_fused as qsa_fused  # noqa: F401  defines the schemas and the fake kernels

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="no c++ on PATH")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]

# name -> (op, tensor specs, scalar args, expected fill_params struct)
POSITIVES = {
    "prepare/real": ("qsa_index_prepare",
                     ["bf16:96,128", "bf16:1024,128", "bf16:32,640", "i32:32,4", "bf16:128", "bf16:128",
                      "bf16:4096,64"], ["f:1e-06", "i:4", "i:4", "i:4"], ("<fiiiiii", 1e-6, 4, 4, 4, 0, 32, 0)),
    "prepare/tiny_k3": ("qsa_index_prepare",
                        ["bf16:64,16", "bf16:256,16", "bf16:8,80", "i32:8,4", "bf16:16", "bf16:16", "bf16:4096,8"],
                        ["f:1e-06", "i:4", "i:8", "i:4"], ("<fiiiiii", 1e-6, 4, 8, 4, 0, 4, 0)),
    "emit/real": ("qsa_index_emit", ["i32:32,512", "i32:32", "i32:32,1024"], ["i:4", "i:128"],
                  ("<fiiiiii", 0.0, 0, 0, 4, 128, 0, 0)),
    "threshold/real": ("qsa_index_threshold", ["f32:32,32768", "i32:32"], ["i:4", "i:512"],
                       ("<fiiiiii", 0.0, 0, 0, 4, 0, 0, 512)),
    "select/real": ("qsa_index_select", ["i32:32,2052"], ["i:512"], ("<fiiiiii", 0.0, 0, 0, 0, 0, 0, 512)),
}
NEGATIVES = {
    "neg/prepare_dim_too_large": ("qsa_index_prepare", ["bf16:96,256", "bf16:1024,256", "bf16:32,1280", "i32:32,4",
                                                        "bf16:256", "bf16:256", "bf16:4096,64"],
                                  ["f:1e-06", "i:4", "i:4", "i:4"], "head dim"),
    "neg/prepare_qk_width": ("qsa_index_prepare", ["bf16:96,128", "bf16:1024,128", "bf16:32,600", "i32:32,4",
                                                   "bf16:128", "bf16:128", "bf16:4096,64"],
                             ["f:1e-06", "i:4", "i:4", "i:4"], "qk"),
    "neg/prepare_ratio3": ("qsa_index_prepare", ["bf16:96,128", "bf16:1024,128", "bf16:32,640", "i32:32,4",
                                                 "bf16:128", "bf16:128", "bf16:4096,64"],
                           ["f:1e-06", "i:4", "i:4", "i:3"], "ratio"),
    "neg/prepare_ring_rows": ("qsa_index_prepare", ["bf16:90,128", "bf16:1024,128", "bf16:32,640", "i32:32,4",
                                                    "bf16:128", "bf16:128", "bf16:4096,64"],
                              ["f:1e-06", "i:4", "i:4", "i:4"], "key_ring"),
    "neg/prepare_int64_rows": ("qsa_index_prepare", ["bf16:96,128", "bf16:1024,128", "bf16:32,640", "f32:32,4",
                                                     "bf16:128", "bf16:128", "bf16:4096,64"],
                               ["f:1e-06", "i:4", "i:4", "i:4"], "rows"),
    "neg/emit_rows": ("qsa_index_emit", ["i32:32,512", "i32:31", "i32:32,1024"], ["i:4", "i:128"], "one row"),
    "neg/emit_block_size": ("qsa_index_emit", ["i32:32,512", "i32:32", "i32:32,1024"], ["i:4", "i:100"],
                            "block_size"),
    "neg/emit_f32_blocks": ("qsa_index_emit", ["f32:32,512", "i32:32", "i32:32,1024"], ["i:4", "i:128"], "blocks"),
    "neg/threshold_columns": ("qsa_index_threshold", ["f32:32,1000", "i32:32"], ["i:4", "i:512"], "multiple of 128"),
    "neg/threshold_bf16": ("qsa_index_threshold", ["bf16:32,1024", "i32:32"], ["i:4", "i:512"], "scores"),
    "neg/select_width": ("qsa_index_select", ["i32:32,101"], ["i:512"], "meta"),
}


def _meta(spec: str):
    dt, dims = spec.split(":")
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}[dt]
    return torch.empty([int(d) for d in dims.split(",")], dtype=dtype, device="meta")


def _value(arg: str):
    kind, text = arg.split(":")
    return float(text) if kind == "f" else int(text)


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    exe = tmp_path_factory.mktemp("qsa_op_callbacks_host") / "op_callbacks_host"
    cmd = [
        "c++", "-std=c++20", "-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=undefined", "-Wall", "-Werror", "-I",
        str(HERE / "hpu_stub"), "-I",
        str(REPO / "csrc" / "qsa_fused" / "pytorch"),
        str(HERE / "op_callbacks_host.cpp")
    ]
    for inc in torch.utils.cpp_extension.include_paths():
        cmd += ["-isystem", inc]
    lib = torch.utils.cpp_extension.library_paths()[0]
    cmd += ["-L", lib, "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{lib}", "-o", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr[-3000:]}"
    return exe


@pytest.fixture(scope="module")
def parsed(program, tmp_path_factory):
    lines = []
    for name, (op, tensors, scalar_args, _) in {**POSITIVES, **NEGATIVES}.items():
        lines.append(" ".join([f"flashinfer_gaudi::{op}", name] + [f"t:{t}" for t in tensors] + scalar_args))
    case_file = tmp_path_factory.mktemp("cases") / "cases.txt"
    case_file.write_text("\n".join(lines) + "\n")
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
    proc = subprocess.run([str(program), str(case_file)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    assert "runtime error:" not in proc.stderr and "AddressSanitizer" not in proc.stderr, proc.stderr[-3000:]
    out = {}
    for line in proc.stdout.splitlines():
        fields = line.split(" ")
        if len(fields) < 2:
            continue
        if fields[1] == "ok":
            out[fields[0]] = {"status": "ok", "outputs": fields[3].split(";"), "guid": fields[4], "size": int(fields[5]),
                              "hex": fields[6]}
        else:
            out[fields[0]] = {"status": "error", "message": " ".join(fields[2:])}
    return out


@pytest.mark.parametrize("name", list(POSITIVES))
def test_callbacks(name, parsed):
    op, tensors, scalar_args, packed = POSITIVES[name]
    line = parsed[name]
    assert line["status"] == "ok", line.get("message")
    meta = getattr(torch.ops.flashinfer_gaudi, op)(*[_meta(t) for t in tensors], *[_value(a) for a in scalar_args])
    meta = meta if isinstance(meta, tuple) else (meta, )
    names = {torch.bfloat16: "bf16", torch.int32: "i32"}
    assert line["outputs"] == [f"{names[m.dtype]}:{','.join(str(d) for d in m.shape)}" for m in meta]
    assert line["guid"] == qsa_fused.GUIDS[op]
    assert line["size"] == 28
    assert line["hex"] == struct.pack(*packed).hex()


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_rejections(name, parsed):
    line = parsed[name]
    assert line["status"] == "error", line
    assert NEGATIVES[name][3] in line["message"], line["message"]
