# SPDX-License-Identifier: Apache-2.0
"""Loads the fused hyper-connection kernel database and HPU op extension.

Staged here because the track may only add files under ``tests/unit_tests/hc_fused``.  At
integration ``load_hc_fused_extension`` and ``hc_fused_extension_errors`` move into
``flashinfer_gaudi/_native.py`` (``docs/design/flash-next/hc-fused.md``, "Loader patch"); from then
on the two functions here only forward to them.  The staged loader does what
``load_gdn_fused_extension`` does for the fused GDN kernels, with one more check:

1. find ``libhc_fused_gaudi2.so`` (``HC_FUSED_GAUDI_KERNEL_LIBRARY`` or ``flashinfer_gaudi/lib``) and
   ``hc_fused_ops*.so`` (``HC_FUSED_GAUDI_NATIVE_LIBRARY`` or ``flashinfer_gaudi/lib``); both must exist;
2. register the kernel database for the graph compiler (SynapseAI reads ``GC_KERNEL_PATH`` when its
   graph compiler starts, so this must run before the first HPU op of the process):
   - through the combined kernel database when that file serves our library (the library sits next
     to it and the adapter's member list names it: an adapter built before this track does not),
   - otherwise first in the ``GC_KERNEL_PATH`` colon list, which the loader refuses when
     ``PT_HPU_RECIPE_CACHE_CONFIG`` is set (the bridge then accepts one file only);
3. only then import the op schemas and ``torch.ops.load_library`` the extension.

A failure at any step leaves the ops without an HPU kernel, which is what the layer's precondition
looks at: the model then runs the torch path instead of failing at the first graph compile on a
GUID no kernel database serves.
"""

from __future__ import annotations

import glob
import importlib
import os
import threading
from pathlib import Path

import torch

HC_FUSED_KERNEL_DB = "libhc_fused_gaudi2.so"  # the fused hyper-connection kernel database
HC_FUSED_KERNEL_LIBRARY_ENV = "HC_FUSED_GAUDI_KERNEL_LIBRARY"  # its path, when it is not in the package's lib/
HC_FUSED_NATIVE_LIBRARY_ENV = "HC_FUSED_GAUDI_NATIVE_LIBRARY"  # the path of hc_fused_ops*.so, same rule
HC_FUSED_EXTENSION_GLOB = "hc_fused_ops*.so"
HC_FUSED_SCHEMA_MODULE = "tests.unit_tests.hc_fused.hc_ops"  # flashinfer_gaudi.hc_fused after integration
COMBINED_MEMBERS_ENV = "FLASHINFER_GAUDI_COMBINED_MEMBERS"  # csrc/kernel_db_adapter/adapter.cpp (host tests)

_LOCK = threading.Lock()
_LOADED: bool | None = None  # None = not attempted
_ERRORS: list[str] = []


def _native():
    return importlib.import_module("flashinfer_gaudi._native")


def hc_fused_extension_errors() -> tuple[str, ...]:
    """Why ``load_hc_fused_extension`` failed (empty when it loaded or was never called)."""
    native = _native()
    if hasattr(native, "hc_fused_extension_errors"):
        return native.hc_fused_extension_errors()
    return tuple(_ERRORS)


def combined_db_serves(combined: Path, kernel_db: str) -> bool:
    """True when the combined kernel database `combined` forwards the GUIDs of the library `kernel_db`.

    The adapter serves the libraries of a fixed name list that sit next to it.  The list is compiled
    in, so an adapter built before ``libhc_fused_gaudi2.so`` was added to it does not serve the file
    even when it is there; the name is then absent from the adapter's bytes.  With the members
    override set (host tests) the adapter serves exactly the paths of the override.
    """
    override = os.environ.get(COMBINED_MEMBERS_ENV, "")
    if override:
        return os.path.abspath(kernel_db) in [os.path.abspath(entry) for entry in override.split(":") if entry]
    path = Path(os.path.abspath(kernel_db))
    if path.name != HC_FUSED_KERNEL_DB or path.parent != Path(os.path.abspath(combined)).parent:
        return False
    try:
        return HC_FUSED_KERNEL_DB.encode() in Path(combined).read_bytes()
    except OSError:
        return False


def _staged_load(native) -> bool:
    package_lib = Path(native.__file__).resolve().parent / "lib"
    kernel_db = os.environ.get(HC_FUSED_KERNEL_LIBRARY_ENV, "") or str(package_lib / HC_FUSED_KERNEL_DB)
    extension = os.environ.get(HC_FUSED_NATIVE_LIBRARY_ENV, "")
    if not extension:
        matches = sorted(glob.glob(str(package_lib / HC_FUSED_EXTENSION_GLOB)))
        extension = matches[0] if matches else str(package_lib / HC_FUSED_EXTENSION_GLOB)
    missing = [path for path in (kernel_db, extension) if not os.path.isfile(path)]
    if missing:
        _ERRORS.extend(f"{path}: file does not exist" for path in missing)
        return False

    combined = native._combined_kernel_db_path()
    if combined is not None and combined_db_serves(combined, kernel_db):
        if not native._activate_combined_kernel_db():
            _ERRORS.append(f"the combined kernel database {combined} serves {HC_FUSED_KERNEL_DB} but could not be "
                           f"made the GC_KERNEL_PATH file (GC_KERNEL_PATH={os.environ.get('GC_KERNEL_PATH', '')!r})")
            return False
    else:
        reason = native._prepend_kernel_database(kernel_db)
        if reason:
            if combined is not None:
                reason += (f" The combined kernel database {combined} does not serve {HC_FUSED_KERNEL_DB}: rebuild "
                           f"it with tools/build_flashinfer_gaudi.py --combined-only after building the library.")
            _ERRORS.append(reason)
            return False
    # The schemas must exist before the extension is loaded: the C++ side only implements the HPU kernels.
    try:
        importlib.import_module(HC_FUSED_SCHEMA_MODULE)
    except Exception as exc:  # pragma: no cover - the schema module is part of this package
        _ERRORS.append(f"{HC_FUSED_SCHEMA_MODULE}: {type(exc).__name__}: {exc}")
        return False
    try:
        torch.ops.load_library(extension)
    except Exception as exc:
        _ERRORS.append(f"{extension}: {type(exc).__name__}: {exc}")
        return False
    return True


def load_hc_fused_extension() -> bool:
    """Register libhc_fused_gaudi2.so and load hc_fused_ops*.so once; True when both are in place."""
    global _LOADED
    native = _native()
    if hasattr(native, "load_hc_fused_extension"):
        return native.load_hc_fused_extension()
    if _LOADED is not None:
        return _LOADED
    with _LOCK:
        if _LOADED is None:
            _LOADED = _staged_load(native)
    return _LOADED
