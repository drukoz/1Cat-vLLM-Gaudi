# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the ``torch.library`` op ``flashinfer_gaudi::gdn_core_decode``.

The op wraps the T1 reference (contract: ``docs/design/flash-next/gdn-core-decode.md``):
the CPU kernel matches the reference bit-exactly (output and both mutated states),
``opcheck`` passes on schema and fake tensor, the fake kernel allocates only,
``gdn_core_decode_available`` reports kernel presence per device type, and the
``VLLM_HPU_GDN_CORE_DECODE`` flag resolves through ``GdnCoreDecodeMode``.
"""

import pytest
import torch

import vllm_gaudi.envs
from flashinfer_gaudi.gdn_core_decode import (GdnCoreDecodeMode, GdnGateActivation, gdn_core_decode,
                                              gdn_core_decode_available, gdn_core_decode_reference)
from tests.unit_tests.gdn_core_decode.test_reference import _fork, _make_inputs

GATE = int(GdnGateActivation.SIGMOID)
HEADS = [(2, 6), (16, 48)]
DTYPES = [torch.float32, torch.bfloat16]


def _op_args(inp):
    """Positional args for one op call, with its own state clones."""
    return tuple(_fork(inp) + [GATE])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H,HV", HEADS)
def test_op_matches_reference(H, HV, dtype):
    inp = _make_inputs(H, HV, 128, dtype)
    exp_args = _fork(inp)
    exp = gdn_core_decode_reference(*exp_args, GATE)
    args = _op_args(inp)
    out = gdn_core_decode(*args)
    assert torch.equal(out, exp)
    assert torch.equal(args[0], exp_args[0])
    assert torch.equal(args[1], exp_args[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("H,HV", HEADS)
def test_opcheck(H, HV, dtype):
    inp = _make_inputs(H, HV, 128, dtype)
    args = _op_args(inp)
    result = torch.library.opcheck(torch.ops.flashinfer_gaudi.gdn_core_decode.default,
                                   args,
                                   test_utils=("test_schema", "test_faketensor"))
    assert result == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}


@pytest.mark.parametrize("dtype", DTYPES)
def test_fake_shape(dtype):
    B, H, HV, D = 4, 16, 48, 128
    inp = _make_inputs(H, HV, D, dtype)
    args = _op_args(inp)
    mode = torch._subclasses.fake_tensor.FakeTensorMode()
    with mode:
        fargs = [mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        out = torch.ops.flashinfer_gaudi.gdn_core_decode(*fargs)
    assert out.shape == (B, HV, D)
    assert out.dtype == torch.float32


def test_available():
    assert gdn_core_decode_available("cpu") is True
    assert gdn_core_decode_available("meta") is False


def test_flag_default_and_value(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_GDN_CORE_DECODE", raising=False)
    assert vllm_gaudi.envs.VLLM_HPU_GDN_CORE_DECODE == 1
    assert GdnCoreDecodeMode(vllm_gaudi.envs.VLLM_HPU_GDN_CORE_DECODE) is GdnCoreDecodeMode.FUSED
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "0")
    assert GdnCoreDecodeMode(vllm_gaudi.envs.VLLM_HPU_GDN_CORE_DECODE) is GdnCoreDecodeMode.OFF
