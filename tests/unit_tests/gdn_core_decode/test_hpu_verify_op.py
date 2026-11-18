# SPDX-License-Identifier: Apache-2.0
"""Card check of the fused GDN verification op for the rental gate G1b.

Runs ``flashinfer_gaudi::gdn_core_verify`` (the HPU kernel from the ``gdn_fused_ops`` extension) on
every ``verify_sim_case.VERIFY_HPU_CASES`` case and checks the output and the mutated states with
``compare_verify``.  The whole module skips without a Gaudi card (no ``habana_frameworks``) or
without the built extension.  Contract: ``docs/design/flash-next/gdn-core-decode.md``.
"""

import pytest

from flashinfer_gaudi._native import load_gdn_fused_extension
from flashinfer_gaudi.gdn_core_verify import gdn_core_verify
from tests.unit_tests.gdn_core_decode.verify_sim_case import (VERIFY_ARG_ORDER, VERIFY_HPU_CASES, compare_verify,
                                                              make_verify_inputs, verify_scalars)

pytest.importorskip("habana_frameworks.torch")

if not load_gdn_fused_extension():
    pytest.skip("gdn_fused_ops extension is not built", allow_module_level=True)


@pytest.mark.parametrize("case", VERIFY_HPU_CASES, ids=[case.name for case in VERIFY_HPU_CASES])
def test_hpu_verify_op_matches_oracle(case):
    inputs = make_verify_inputs(case)
    hpu = {name: tensor.to("hpu") for name, tensor in inputs.items()}
    out = gdn_core_verify(*[hpu[name] for name in VERIFY_ARG_ORDER], *verify_scalars(case))
    got = {"out": out.cpu(), "conv_state": hpu["conv_state"].cpu(), "ssm_state": hpu["ssm_state"].cpu()}
    messages = compare_verify(case, inputs, got)
    if messages:
        print("\n".join(messages))
    assert messages == []
