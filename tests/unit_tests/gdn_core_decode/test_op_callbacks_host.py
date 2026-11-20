# SPDX-License-Identifier: Apache-2.0
"""Host-side test of the GDN ops' PyTorch callbacks under ASan/UBSan (RUK-3105 R1-e1).

Compiles the unmodified ``csrc/gdn_fused/pytorch/hpu_gdn_core_decode.cpp`` against the test
stub of Habana's ``hpu_custom_op_pt2.h`` (``hpu_stub/``), calls the ``output_meta`` and
``fill_params`` callbacks of the three ops for every HPU test case, and checks each result
against the Python schema (the fake kernels on meta tensors) and ``struct.pack`` of the
scalars.  No card, no HPU runtime: the ``registerUserCustomOp`` callbacks are plain C++ on
``at::Stack``.  Contract: ``docs/design/flash-next/gdn-core-decode.md``.
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

import flashinfer_gaudi.gdn_core_decode  # noqa: F401  registers gdn_core_decode + gdn_recurrent_decode
import flashinfer_gaudi.gdn_core_verify  # noqa: F401  registers gdn_core_verify
from tests.unit_tests.gdn_core_decode.recurrent_sim_case import (RECURRENT_ARG_ORDER, RECURRENT_HPU_CASES,
                                                                 make_recurrent_inputs, recurrent_scalars)
from tests.unit_tests.gdn_core_decode.sim_case import ARG_ORDER, HPU_CASES, make_inputs, scalars
from tests.unit_tests.gdn_core_decode.verify_sim_case import (MAX_NODES, VERIFY_ARG_ORDER, VERIFY_HPU_CASES,
                                                              VERIFY_TOO_MANY_NODES_CASE, make_verify_inputs,
                                                              verify_scalars)

pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="no c++ on PATH")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]

SCHEMA = {
    "decode": "flashinfer_gaudi::gdn_core_decode",
    "recurrent": "flashinfer_gaudi::gdn_recurrent_decode",
    "verify": "flashinfer_gaudi::gdn_core_verify",
}
_DTYPE = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32"}
_PACK = {"decode": "<ffi", "recurrent": "<f", "verify": "<ffi"}
_NEGATIVES = ("verify/bad_rows", "decode/short_stack", "verify/too_many_nodes")


def _case_list():
    return ([("decode", c) for c in HPU_CASES] + [("recurrent", c) for c in RECURRENT_HPU_CASES] +
            [("verify", c) for c in VERIFY_HPU_CASES])


def _args(kind, case):
    """(inputs, arg order, scalars) for one case."""
    if kind == "decode":
        return make_inputs(case), ARG_ORDER, scalars(case)
    if kind == "recurrent":
        return make_recurrent_inputs(case), RECURRENT_ARG_ORDER, recurrent_scalars(case)
    return make_verify_inputs(case), VERIFY_ARG_ORDER, verify_scalars(case)


def _arg_strings(inputs, order, scalar_values):
    parts = [f"t:{_DTYPE[t.dtype]}:{','.join(str(d) for d in t.shape)}" for t in (inputs[n] for n in order)]
    parts += [f"i:{v}" if isinstance(v, int) else f"f:{v!r}" for v in scalar_values]
    return parts


def _input_lines():
    lines = []
    for kind, case in _case_list():
        inputs, order, sc = _args(kind, case)
        lines.append(" ".join([SCHEMA[kind], f"{kind}/{case.name}"] + _arg_strings(inputs, order, sc)))
    ver = next(c for c in VERIFY_HPU_CASES if c.name == "ver_tp8_chain")
    inputs, order, sc = _args("verify", ver)
    bad = dict(inputs)
    mq = inputs["mixed_qkv"]
    bad["mixed_qkv"] = torch.zeros(mq.shape[0] - 1, mq.shape[1], dtype=torch.bfloat16)  # one row short
    lines.append(" ".join([SCHEMA["verify"], _NEGATIVES[0]] + _arg_strings(bad, order, sc)))
    dec = HPU_CASES[0]
    inputs, order, sc = _args("decode", dec)
    lines.append(" ".join([SCHEMA["decode"], _NEGATIVES[1]] + _arg_strings(inputs, order, sc)[:-1]))  # no gate
    inputs, order, sc = _args("verify", VERIFY_TOO_MANY_NODES_CASE)  # T = MAX_NODES + 1, otherwise well-formed
    lines.append(" ".join([SCHEMA["verify"], _NEGATIVES[2]] + _arg_strings(inputs, order, sc)))
    return lines


def _expected():
    """name -> (n_outputs, dtype, dims, param bytes) from the fake kernels and struct.pack."""
    exp = {}
    for kind, case in _case_list():
        inputs, order, sc = _args(kind, case)
        op = getattr(torch.ops.flashinfer_gaudi, SCHEMA[kind].split("::")[1])
        meta = op(*[inputs[n].to("meta") for n in order], *sc)
        exp[f"{kind}/{case.name}"] = {
            "n": 1,
            "dtype": _DTYPE[meta.dtype],
            "dims": [str(d) for d in meta.shape],
            "bytes": struct.pack(_PACK[kind], *sc),
        }
    return exp


EXPECTED = _expected()
CASE_NAMES = [f"{kind}/{case.name}" for kind, case in _case_list()]


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    """op_callbacks_host.cpp compiled with the exact command of the run sheet (RUK-3119)."""
    exe = tmp_path_factory.mktemp("op_callbacks_host") / "op_callbacks_host"
    cmd = [
        "c++", "-std=c++20", "-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined",
        "-fno-sanitize-recover=undefined", "-I",
        str(HERE / "hpu_stub"), "-I",
        str(REPO / "csrc" / "gdn_fused" / "pytorch"),
        str(HERE / "op_callbacks_host.cpp")
    ]
    for inc in torch.utils.cpp_extension.include_paths():
        cmd += ["-I", inc]
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
    proc = subprocess.run([str(program), str(case_file)],
                          capture_output=True,
                          text=True,
                          timeout=300,
                          env=env)
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
                "dtype": fields[3],
                "dims": fields[4].split(","),
                "size": int(fields[5]),
                "hex": fields[6],
            }
        else:
            lines[fields[0]] = {"status": "error", "message": " ".join(fields[2:])}
    missing = [n for n in CASE_NAMES + list(_NEGATIVES) if n not in lines]
    assert not missing, f"no output line for: {missing}\nstdout:\n{proc.stdout}"
    return lines


@pytest.mark.parametrize("name", CASE_NAMES)
def test_callbacks(name, parsed):
    line = parsed[name]
    exp = EXPECTED[name]
    assert line["status"] == "ok", line.get("message")
    assert line["n"] == exp["n"]
    assert line["dtype"] == exp["dtype"]
    assert line["dims"] == exp["dims"]
    assert line["size"] == len(exp["bytes"])
    assert line["hex"] == exp["bytes"].hex()


def test_negative_lines(parsed):
    bad = parsed["verify/bad_rows"]
    assert bad["status"] == "error"
    assert "mixed_qkv must be [B*T,C]" in bad["message"]
    short = parsed["decode/short_stack"]
    assert short["status"] == "error"


def test_verify_rejects_too_many_nodes(parsed):
    """T = MAX_NODES + 1 fails the op callback's check, so the HPU bridge never builds a node for the kernel."""
    assert VERIFY_TOO_MANY_NODES_CASE.T == MAX_NODES + 1
    line = parsed["verify/too_many_nodes"]
    assert line["status"] == "error"
    assert f"ssm_store_indices must be [B,T] with 1<=T<={MAX_NODES}" in line["message"]
