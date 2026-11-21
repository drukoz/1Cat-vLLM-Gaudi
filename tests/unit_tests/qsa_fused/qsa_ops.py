# SPDX-License-Identifier: Apache-2.0
"""The fused QSA decode ops with a CPU stand-in: the torch references as their CPU kernels (as hc_ops.py).

The schemas and the fake kernels are defined in ``flashinfer_gaudi/qsa_fused.py`` and nowhere else. Importing
this module adds ``qsa_reference``'s ``qsa_index_prepare_ref`` / ``qsa_index_emit_ref`` as the CPU kernels, so a
CPU test can run the ops and the layer's mode-2 path, and gets the reference by construction. On a card the
same module serves ``test_hpu_op.py``: the CPU kernels are its reference.
"""

from __future__ import annotations

import torch

from flashinfer_gaudi.qsa_fused import NAMESPACE, OP_NAMES, qsa_fused_missing_ops, qsa_index_emit  # noqa: F401
from flashinfer_gaudi.qsa_fused import qsa_index_prepare  # noqa: F401
from tests.unit_tests.qsa_fused.qsa_reference import (qsa_index_emit_ref, qsa_index_prepare_ref, qsa_index_select_ref,
                                                      qsa_index_threshold_ref, qsa_index_topk_masked_ref)

CPU_KERNELS = {"qsa_index_prepare": qsa_index_prepare_ref, "qsa_index_emit": qsa_index_emit_ref,
               "qsa_index_threshold": qsa_index_threshold_ref, "qsa_index_select": qsa_index_select_ref,
               "qsa_index_topk": qsa_index_topk_masked_ref}

# Kept for the life of the process: the registrations go away with the Library object.
_CPU_LIBRARY = torch.library.Library(NAMESPACE, "IMPL")
for _name, _kernel in CPU_KERNELS.items():
    _CPU_LIBRARY.impl(_name, _kernel, "CPU")
