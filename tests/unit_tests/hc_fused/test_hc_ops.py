# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the fused hyper-connection op definitions: schemas, fake kernels and the CPU stand-in.

``flashinfer_gaudi/hc_fused.py`` defines the six schemas and the fake kernels and nothing else: the
HPU kernels come from the ``hc_fused_ops`` extension, and a CPU process has no kernel at all.
``hc_ops.py`` (this folder) is that module plus the torch oracle registered as the CPU kernels, the
stand-in the CPU tests run the ops through.  No vLLM, no toolchain.  The HPU kernels are tested on a
card by ``test_hpu_op.py``.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.unit_tests.hc_fused import hc_ops, hc_stress
from tests.unit_tests.hc_fused.hc_case import (BF16, INPUT_ORDER, OP_NAME, OUTPUT_ORDER, SIM_CASES, case_by_name,
                                               expected, make_inputs, output_shapes, scalars)

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


@pytest.mark.parametrize("name", hc_stress.STRESS_CASES)
def test_the_queue_stress_of_the_card_test_runs_on_cpu(name):
    """The body of test_hpu_op.py::test_hpu_op_behind_a_queue_of_lazy_ops, here with the oracle as the kernel:
    it checks that body (the calls, the comparison, the queued ops' own result), which no card ran yet."""
    assert hc_stress.run_behind_a_queue(case_by_name(name), "cpu", rounds=2) == []


def test_the_queue_stress_reports_a_wrong_result(monkeypatch):
    case = case_by_name("norm_real_r16")
    monkeypatch.setattr(hc_ops, "hc_norm", lambda x, weight, eps, hc: torch.zeros_like(x))

    messages = hc_stress.run_behind_a_queue(case, "cpu", rounds=2)

    assert len(messages) >= 2 and messages[0].startswith("round 0: ") and any(m.startswith("round 1: ") for m in messages)


def test_the_queue_stress_covers_the_five_kernels_of_the_decode_loop():
    assert [case_by_name(name).kind for name in hc_stress.STRESS_CASES] == ["norm", "silu_split", "gate_mix",
                                                                           "combine", "combine_norm"]
    assert hc_stress.QUEUED_PAIRS == 48 and hc_stress.ROUNDS == 8


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


REPO_ROOT = Path(__file__).resolve().parents[3]


def _python(code: str) -> subprocess.CompletedProcess:
    """A fresh interpreter from the repository root: this process already holds the CPU stand-in."""
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)


def test_the_package_module_is_the_only_definition():
    """One module defines the schemas (a second definition raises); hc_ops only adds the CPU kernels."""
    from flashinfer_gaudi import hc_fused

    assert hc_ops.SCHEMAS is hc_fused.SCHEMAS and hc_ops.OP_NAMES is hc_fused.OP_NAMES
    for name in hc_fused.OP_NAMES:
        assert getattr(hc_ops, name) is getattr(hc_fused, name)
    assert hc_ops.hc_fused_missing_ops is hc_fused.hc_fused_missing_ops
    assert hc_ops.hc_fused_available is hc_fused.hc_fused_available


def test_the_package_module_registers_no_cpu_kernel():
    """In a process without the stand-in, no op has a kernel for any device: a layer then runs torch."""
    proc = _python("import torch\n"
                   "from flashinfer_gaudi import hc_fused as m\n"
                   "assert m.hc_fused_missing_ops('cpu') == m.OP_NAMES, m.hc_fused_missing_ops('cpu')\n"
                   "assert m.hc_fused_missing_ops(torch.device('hpu')) == m.OP_NAMES\n"
                   "assert not m.hc_fused_available('cpu')\n"
                   "for name in m.OP_NAMES:\n"
                   "    schema = getattr(torch.ops.flashinfer_gaudi, name).default._schema\n"
                   "    assert str(schema).replace('flashinfer_gaudi::', '') == m.SCHEMAS[name]\n"
                   "out = m.hc_gate_mix(torch.empty(3, 512, dtype=torch.bfloat16, device='meta'),\n"
                   "                    torch.empty(3, 512, dtype=torch.bfloat16, device='meta'), 4)\n"
                   "assert tuple(out.shape) == (3, 128)\n"
                   "print('ok')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "ok"


def test_importing_the_package_does_not_define_the_ops():
    """With VLLM_HPU_HC_FUSED=0 nothing imports the op module, so nothing new is registered."""
    proc = _python("import torch, flashinfer_gaudi\n"
                   "from flashinfer_gaudi import _native\n"
                   "import sys\n"
                   "assert 'flashinfer_gaudi.hc_fused' not in sys.modules\n"
                   "names = [n for n in torch._C._dispatch_get_all_op_names()\n"
                   "         if n.startswith('flashinfer_gaudi::hc_')]\n"
                   "assert names == [], names\n"
                   "print('ok')\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "ok"
