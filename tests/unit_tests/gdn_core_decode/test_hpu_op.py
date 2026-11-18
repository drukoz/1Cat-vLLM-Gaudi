# SPDX-License-Identifier: Apache-2.0
"""HPU tests for ``flashinfer_gaudi::gdn_core_decode`` and ``gdn_recurrent_decode``.

Runs each op (HPU kernels from the ``gdn_fused_ops`` extension) on every case of its shared
cases (``sim_case.HPU_CASES`` and ``recurrent_sim_case.RECURRENT_HPU_CASES``) and checks the
output and the mutated states with ``compare`` / ``compare_recurrent``.  The whole module skips
without a Gaudi card (no ``habana_frameworks``) or without the built extension.  Contract:
``docs/design/flash-next/gdn-core-decode.md``.
"""

import pytest

from flashinfer_gaudi._native import load_gdn_fused_extension
from flashinfer_gaudi.gdn_core_decode import gdn_core_decode, gdn_recurrent_decode
from tests.unit_tests.gdn_core_decode.recurrent_sim_case import (RECURRENT_ARG_ORDER, RECURRENT_HPU_CASES,
                                                                 compare_recurrent, make_recurrent_inputs,
                                                                 recurrent_scalars)
from tests.unit_tests.gdn_core_decode.sim_case import ARG_ORDER, HPU_CASES, compare, make_inputs, scalars

pytest.importorskip("habana_frameworks.torch")

if not load_gdn_fused_extension():
    pytest.skip("gdn_fused_ops extension is not built", allow_module_level=True)


@pytest.mark.parametrize("case", HPU_CASES, ids=[case.name for case in HPU_CASES])
def test_hpu_op_matches_reference(case):
    inputs = make_inputs(case)
    hpu = {name: tensor.to("hpu") for name, tensor in inputs.items()}
    out = gdn_core_decode(*[hpu[name] for name in ARG_ORDER], *scalars(case))
    got = {
        "out": out.cpu(),
        "conv_state": hpu["conv_state"].cpu(),
        "ssm_state": hpu["ssm_state"].cpu(),
    }
    messages = compare(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []


@pytest.mark.parametrize("case", RECURRENT_HPU_CASES, ids=[case.name for case in RECURRENT_HPU_CASES])
def test_hpu_recurrent_op_matches_reference(case):
    inputs = make_recurrent_inputs(case)
    hpu = {name: tensor.to("hpu") for name, tensor in inputs.items()}
    out = gdn_recurrent_decode(*[hpu[name] for name in RECURRENT_ARG_ORDER], *recurrent_scalars(case))
    got = {"out": out.cpu(), "ssm_state": hpu["ssm_state"].cpu()}
    messages = compare_recurrent(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []
