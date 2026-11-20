# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the fused MoE glue op definitions: schemas, fake kernels and the CPU stand-in.

``flashinfer_gaudi/moe_glue.py`` defines the four schemas and the fake kernels and nothing else: the
HPU kernels come from the ``hc_fused_ops`` extension, and a CPU process has no kernel at all.
``moe_ops.py`` (this folder) is that module plus the torch oracle registered as the CPU kernels, the
stand-in the CPU tests run the ops through.  No vLLM, no toolchain.  The HPU kernels are tested on a
card by ``test_moe_hpu_op.py``.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.unit_tests.hc_fused import moe_ops
from tests.unit_tests.hc_fused.moe_case import (GUIDS, INPUT_ORDER, OP_NAME, OUTPUT_DTYPE, OUTPUT_ORDER, SIM_CASES,
                                                expected, make_inputs, output_shapes, scalars)

SMALL = [case for case in SIM_CASES if case.rows <= 64]
IDS = [case.name for case in SMALL]


def _call(case, tensors):
    out = getattr(moe_ops, OP_NAME[case.kind])(*[tensors[name] for name in INPUT_ORDER[case.kind]], *scalars(case))
    return out if isinstance(out, tuple) else (out, )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()])


def test_schemas_are_registered():
    assert tuple(moe_ops.SCHEMAS) == moe_ops.OP_NAMES
    for name in moe_ops.OP_NAMES:
        schema = getattr(torch.ops.flashinfer_gaudi, name).default._schema
        assert str(schema).replace("flashinfer_gaudi::", "") == moe_ops.SCHEMAS[name]
        assert not any(argument.alias_info for argument in schema.arguments)  # no op writes an input


def test_guids_are_the_kernel_library_names():
    """The loader asks the kernel library for these names; the cases use the same ones."""
    assert moe_ops.GUIDS == {OP_NAME[kind]: guid for kind, guid in GUIDS.items()}


@pytest.mark.parametrize("case", SMALL, ids=IDS)
def test_cpu_kernel_is_the_oracle(case):
    inputs = make_inputs(case)
    before = {name: tensor.clone() for name, tensor in inputs.items()}
    got = _call(case, inputs)
    exp = expected(case, inputs)
    assert len(got) == len(OUTPUT_ORDER[case.kind])
    for tensor, name in zip(got, OUTPUT_ORDER[case.kind]):
        assert tensor.dtype == OUTPUT_DTYPE[name] and torch.equal(_bits(tensor), _bits(exp[name])), name
    for name, tensor in inputs.items():
        assert torch.equal(_bits(tensor), _bits(before[name]))


@pytest.mark.parametrize("case", SMALL, ids=IDS)
def test_fake_kernel_shapes_and_dtypes(case):
    inputs = {name: tensor.to("meta") for name, tensor in make_inputs(case).items()}
    got = _call(case, inputs)
    shapes = output_shapes(case)
    assert [tuple(t.shape) for t in got] == [shapes[name] for name in OUTPUT_ORDER[case.kind]]
    assert [t.dtype for t in got] == [OUTPUT_DTYPE[name] for name in OUTPUT_ORDER[case.kind]]
    assert all(t.device.type == "meta" for t in got)


def test_availability():
    assert moe_ops.moe_glue_available("cpu")
    assert moe_ops.moe_glue_missing_ops(torch.device("cpu")) == ()
    # No HPU kernel is registered in a process without the hc_fused_ops extension.
    assert moe_ops.moe_glue_missing_ops("cuda") == moe_ops.OP_NAMES
    if not moe_ops.moe_glue_available("hpu"):
        assert set(moe_ops.moe_glue_missing_ops("hpu")) <= set(moe_ops.OP_NAMES)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _python(code: str) -> subprocess.CompletedProcess:
    """A fresh interpreter from the repository root: this process already holds the CPU stand-in."""
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def test_the_package_module_is_the_only_definition():
    from flashinfer_gaudi import moe_glue

    assert moe_ops.SCHEMAS is moe_glue.SCHEMAS and moe_ops.OP_NAMES is moe_glue.OP_NAMES
    for name in moe_glue.OP_NAMES:
        assert getattr(moe_ops, name) is getattr(moe_glue, name)
    assert moe_ops.moe_glue_missing_ops is moe_glue.moe_glue_missing_ops


def test_the_package_module_registers_no_cpu_kernel():
    """In a process without the stand-in, no op has a kernel for any device: a layer then runs torch."""
    proc = _python("import torch\n"
                   "from flashinfer_gaudi import moe_glue as m\n"
                   "assert m.moe_glue_missing_ops('cpu') == m.OP_NAMES, m.moe_glue_missing_ops('cpu')\n"
                   "assert m.moe_glue_missing_ops(torch.device('hpu')) == m.OP_NAMES\n"
                   "assert not m.moe_glue_available('cpu')\n"
                   "ids, weights = m.moe_route(torch.empty(3, 513, dtype=torch.bfloat16, device='meta'), 512, 10)\n"
                   "assert tuple(ids.shape) == (3, 10) and ids.dtype == torch.int32\n"
                   "assert tuple(weights.shape) == (3, 10) and weights.dtype == torch.bfloat16\n"
                   "print('ok')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "ok"


def test_importing_the_package_does_not_define_the_ops():
    """With VLLM_HPU_MOE_GLUE=0 (and VLLM_HPU_HC_FUSED=0) nothing imports the op module."""
    proc = _python("import torch, flashinfer_gaudi\n"
                   "from flashinfer_gaudi import _native\n"
                   "import sys\n"
                   "assert 'flashinfer_gaudi.moe_glue' not in sys.modules\n"
                   "names = [n for n in torch._C._dispatch_get_all_op_names()\n"
                   "         if n.startswith('flashinfer_gaudi::moe_')]\n"
                   "assert names == [], names\n"
                   "print('ok')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "ok"
