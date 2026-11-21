# SPDX-License-Identifier: Apache-2.0
"""The fused hyper-connection ops with a CPU stand-in: the torch oracle as their CPU kernels.

The schemas and the fake kernels are defined in ``flashinfer_gaudi/hc_fused.py`` and nowhere else
(defining a schema twice in one process raises).  That module registers no CPU kernel: in a server
only the ``hc_fused_ops`` extension gives the ops a kernel, for HPU.  Importing this module adds the
torch oracle (``hc_reference.py``, verbatim copies of the plugin's functions) as the CPU kernels,
so a CPU test can run the ops, and the layer's fused path, and gets the oracle by construction.
On a card the same module serves ``test_hpu_op.py``: the CPU kernels are its reference.

Contract: ``docs/design/flash-next/hc-fused.md``.
"""

from __future__ import annotations

import torch

from flashinfer_gaudi.hc_fused import (IL_OP_NAMES, IL_SCHEMAS, NAMESPACE, OP_NAMES, SCHEMAS,  # noqa: F401
                                       hc_combine, hc_combine_norm, hc_combine_norm_il, hc_fused_available,
                                       hc_fused_il_missing_ops, hc_fused_missing_ops, hc_gate_mix, hc_gate_mix_il,
                                       hc_norm, hc_silu, hc_silu_split)
from tests.unit_tests.hc_fused.hc_reference import (hc_combine_norm_reference, hc_combine_reference,
                                                    hc_gate_mix_reference, hc_norm_reference, hc_silu_reference,
                                                    hc_silu_split_reference)

CPU_KERNELS = {
    "hc_norm": hc_norm_reference,
    "hc_silu": hc_silu_reference,
    "hc_silu_split": hc_silu_split_reference,
    "hc_gate_mix": hc_gate_mix_reference,
    "hc_combine": hc_combine_reference,
    "hc_combine_norm": hc_combine_norm_reference,
    # The interleaved kernels (docs/design/flash-next/hc-kernels.md) compute the same functions.
    "hc_gate_mix_il": hc_gate_mix_reference,
    "hc_combine_norm_il": hc_combine_norm_reference,
}

# Kept for the life of the process: the registrations go away with the Library object.
_CPU_LIBRARY = torch.library.Library(NAMESPACE, "IMPL")
for _name, _kernel in CPU_KERNELS.items():
    _CPU_LIBRARY.impl(_name, _kernel, "CPU")
