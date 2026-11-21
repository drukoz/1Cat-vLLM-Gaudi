# SPDX-License-Identifier: Apache-2.0
"""The fused MoE glue ops with a CPU stand-in: the torch oracle as their CPU kernels.

The schemas and the fake kernels are defined in ``flashinfer_gaudi/moe_glue.py`` and nowhere else.
That module registers no CPU kernel: in a server only the ``hc_fused_ops`` extension gives the ops a
kernel, for HPU.  Importing this module adds the torch oracle (``moe_reference.py``, copies of the
plugin's torch code) as the CPU kernels, so a CPU test can run the ops, and the MoE block's fused
path, and gets the torch path's bits by construction.  "The torch path" is the one of the device the
op runs on: the CPU kernel of ``moe_row_quant`` divides the way CPU torch does
(``moe_row_quant_cpu_reference``), the HPU kernel the way a card does (``moe_row_quant_reference``, the
oracle of ``moe_case.compare``).  For the other three ops the two are the same function.

Contract: ``docs/design/flash-next/hc-fused.md``, "MoE glue kernels".
"""

from __future__ import annotations

import torch

from flashinfer_gaudi.moe_glue import (GUIDS, NAMESPACE, OP_NAMES, SCHEMAS, moe_act_quant,  # noqa: F401
                                       moe_combine, moe_glue_available, moe_glue_missing_ops, moe_route,
                                       moe_row_quant, moe_silu_mul)
from tests.unit_tests.hc_fused.moe_reference import (moe_act_quant_cpu_reference, moe_combine_reference,
                                                     moe_route_reference, moe_row_quant_cpu_reference,
                                                     moe_silu_mul_reference)

CPU_KERNELS = {
    "moe_route": moe_route_reference,
    "moe_silu_mul": moe_silu_mul_reference,
    "moe_combine": moe_combine_reference,
    "moe_row_quant": moe_row_quant_cpu_reference,
    "moe_act_quant": moe_act_quant_cpu_reference,
}


def moe_glue_module():
    """The package module that defines the ops (its limits are checked against the C++ sources)."""
    from flashinfer_gaudi import moe_glue

    return moe_glue


# Kept for the life of the process: the registrations go away with the Library object.
_CPU_LIBRARY = torch.library.Library(NAMESPACE, "IMPL")
for _name, _kernel in CPU_KERNELS.items():
    _CPU_LIBRARY.impl(_name, _kernel, "CPU")
