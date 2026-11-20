# SPDX-License-Identifier: Apache-2.0
"""The loader of the fused hyper-connection kernel database and HPU op extension, for the tests here.

The loader itself is ``flashinfer_gaudi._native.load_hc_fused_extension`` (it was staged in this file
until the kernels were wired in; ``docs/design/flash-next/hc-fused.md``, "Loader"). This module keeps
the names the card test imports and forwards every one of them: there is one loader, and it

1. finds ``libhc_fused_gaudi2.so`` (``HC_FUSED_GAUDI_KERNEL_LIBRARY`` or ``flashinfer_gaudi/lib``) and
   ``hc_fused_ops*.so`` (``HC_FUSED_GAUDI_NATIVE_LIBRARY`` or ``flashinfer_gaudi/lib``); both must exist;
2. registers the kernel database for the graph compiler: through the combined kernel database when
   that file serves the library, otherwise first in the ``GC_KERNEL_PATH`` colon list, which it
   refuses when ``PT_HPU_RECIPE_CACHE_CONFIG`` is set;
3. only then imports the op schemas (``flashinfer_gaudi.hc_fused``) and loads the extension.

Its tests are ``tests/unit_tests/gdn_core_decode/test_hc_fused_loader.py``.
"""

from __future__ import annotations

from flashinfer_gaudi._native import (COMBINED_MEMBERS_ENV, HC_FUSED_EXTENSION_GLOB, HC_FUSED_KERNEL_DB,  # noqa: F401
                                      HC_FUSED_KERNEL_LIBRARY_ENV, HC_FUSED_NATIVE_LIBRARY_ENV,
                                      HC_FUSED_SCHEMA_MODULE, hc_fused_extension_errors, load_hc_fused_extension)
from flashinfer_gaudi._native import combined_db_serves_hc_fused as combined_db_serves  # noqa: F401
