# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the fused hyper-connection op definitions (``hc_ops.py``): schemas, CPU and fake kernels.

No vLLM, no toolchain.  The HPU kernels are tested on a card by ``test_hpu_op.py``.
"""

import pytest
import torch

from tests.unit_tests.hc_fused import hc_ops
from tests.unit_tests.hc_fused.hc_case import (BF16, INPUT_ORDER, OP_NAME, OUTPUT_ORDER, SIM_CASES, expected,
                                               make_inputs, output_shapes, scalars)

SMALL = [case for case in SIM_CASES if case.rows <= 32]
IDS = [case.name for case in SMALL]


def _call(case, tensors):
    out = getattr(hc_ops, OP_NAME[case.kind])(*[tensors[name] for name in INPUT_ORDER[case.kind]], *scalars(case))
    return out if isinstance(out, tuple) else (out, )


def test_schemas_are_registered():
    assert tuple(hc_ops.SCHEMAS) == hc_ops.OP_NAMES
    for name in hc_ops.OP_NAMES:
        schema = getattr(torch.ops.flashinfer_gaudi, name).default._schema
        assert str(schema).replace("flashinfer_gaudi::", "") == hc_ops.SCHEMAS[name]
        assert not any(argument.alias_info for argument in schema.arguments)  # no op writes an input


@pytest.mark.parametrize("case", SMALL, ids=IDS)
def test_cpu_kernel_is_the_oracle(case):
    inputs = make_inputs(case)
    before = {name: tensor.clone() for name, tensor in inputs.items()}
    got = _call(case, inputs)
    exp = expected(case, inputs)
    assert len(got) == len(OUTPUT_ORDER[case.kind])
    for tensor, name in zip(got, OUTPUT_ORDER[case.kind]):
        assert tensor.dtype == BF16 and torch.equal(tensor.view(torch.int16), exp[name].view(torch.int16))
    for name, tensor in inputs.items():
        assert torch.equal(tensor.view(torch.int16), before[name].view(torch.int16))


@pytest.mark.parametrize("case", SMALL, ids=IDS)
def test_fake_kernel_shapes(case):
    inputs = {name: tensor.to("meta") for name, tensor in make_inputs(case).items()}
    got = _call(case, inputs)
    shapes = output_shapes(case)
    assert [tuple(t.shape) for t in got] == [shapes[name] for name in OUTPUT_ORDER[case.kind]]
    assert all(t.dtype == BF16 and t.device.type == "meta" for t in got)


def test_silu_split_injection_is_a_copy_not_a_view():
    down = torch.randn(3, 32).to(BF16)
    lora, injection = hc_ops.hc_silu_split(down, 4, 16)
    assert torch.equal(injection, down[:, 16:20])
    down[:, 16:20] = 0
    assert bool(injection.float().abs().sum() > 0)
    assert lora.shape == (3, 16)


def test_availability():
    assert hc_ops.hc_fused_available("cpu")
    assert hc_ops.hc_fused_missing_ops(torch.device("cpu")) == ()
    # No HPU kernel is registered in a process without the hc_fused_ops extension.
    assert hc_ops.hc_fused_missing_ops("cuda") == hc_ops.OP_NAMES
    if not hc_ops.hc_fused_available("hpu"):
        assert set(hc_ops.hc_fused_missing_ops("hpu")) <= set(hc_ops.OP_NAMES)
