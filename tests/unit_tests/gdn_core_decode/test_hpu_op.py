# SPDX-License-Identifier: Apache-2.0
"""HPU tests for ``flashinfer_gaudi::gdn_core_decode`` against the shared sim_case cases.

Runs the op (HPU kernel from the ``gdn_fused_ops`` extension) on every case of
``sim_case.HPU_CASES`` and checks the output and both mutated states with ``sim_case.compare``.
The whole module skips without a Gaudi card (no ``habana_frameworks``) or without the built
extension.  Contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

import pytest

from flashinfer_gaudi._native import load_gdn_fused_extension
from flashinfer_gaudi.gdn_core_decode import gdn_core_decode
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
