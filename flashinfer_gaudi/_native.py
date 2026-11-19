# SPDX-License-Identifier: Apache-2.0
"""Lazy loading and discovery of optional native Gaudi kernels."""

from __future__ import annotations

import glob
import importlib
import os
import threading
from enum import IntEnum
from pathlib import Path
from typing import Callable

import torch

_LOAD_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False
_LOADED_PATHS: list[str] = []
_LOAD_ERRORS: list[str] = []
_SILU_AND_MUL_OP: Callable | None = None
_SILU_MUL_QUANT_OP: Callable | None = None
_BLOCK_FP8_DEQUANT_OP: Callable | None = None
_BLOCK_FP8_LINEAR_OP: Callable | None = None
_ADD_RMSNORM_QUANT_OP: Callable | None = None
_GDN_FUSED_LOADED: bool | None = None  # None = not attempted
_GDN_FUSED_ERRORS: list[str] = []  # the fused GDN loader's own errors

RECIPE_CACHE_ENV = "PT_HPU_RECIPE_CACHE_CONFIG"
COMBINED_KERNEL_DB = "libflashinfer_gaudi_combined_gaudi2.so"  # built by a later task; named only in the reason


class KernelDbState(IntEnum):
    MISSING = 0  # the database file is not installed
    REGISTERED = 1  # prepended to GC_KERNEL_PATH, or already in it
    BLOCKED = 2  # left out because PT_HPU_RECIPE_CACHE_CONFIG is set


def kernel_database_blocked_reason(name: str) -> str | None:
    """Why the kernel database `name` must not be prepended to GC_KERNEL_PATH in this process, or None."""
    if os.environ.get(RECIPE_CACHE_ENV, "").strip():
        return (f"{RECIPE_CACHE_ENV} is set: with a recipe cache the Habana PyTorch bridge accepts only one regular "
                f"file in GC_KERNEL_PATH (Gaudi 1.24.1 aborts at init_device with 'PT_SYNHELPER Flag GC_KERNEL_PATH: "
                f"Path is not a regular file'), so {name} was not registered and its native TPC kernels are "
                f"unavailable in this process. Unset {RECIPE_CACHE_ENV} to use them, or install the combined kernel "
                f"database ({COMBINED_KERNEL_DB}).")
    return None


_FLASHINFER_DB_STATE = KernelDbState.MISSING


def flashinfer_kernel_database_state() -> KernelDbState:
    """The registration state of the packaged FlashInfer kernel database in this process."""
    return _FLASHINFER_DB_STATE


def _library_candidates() -> list[str]:
    candidates: list[str] = []
    configured = os.environ.get("FLASHINFER_GAUDI_NATIVE_LIBRARY", "")
    if configured:
        candidates.extend(path for path in configured.split(os.pathsep) if path)

    package_lib = Path(__file__).resolve().parent / "lib"
    candidates.extend(sorted(glob.glob(str(package_lib / "flashinfer_gaudi_ops*.so"))))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _prepend_kernel_database(path: str) -> str | None:
    reason = kernel_database_blocked_reason(Path(path).name)
    if reason:
        return reason

    current = [entry for entry in os.environ.get("GC_KERNEL_PATH", "").split(os.pathsep) if entry]
    if path in current:
        return None

    configured = [path, *current]
    # Setting GC_KERNEL_PATH replaces SynapseAI's implicit default. Preserve
    # the stock kernel database when this package is the first component to
    # configure the variable.
    system_kernel_db = Path("/usr/lib/habanalabs/libtpc_kernels.so")
    if not current and system_kernel_db.is_file():
        configured.append(str(system_kernel_db))
    os.environ["GC_KERNEL_PATH"] = os.pathsep.join(configured)
    return None


def _configure_kernel_database_path() -> KernelDbState:
    global _FLASHINFER_DB_STATE
    package_lib = Path(__file__).resolve().parent / "lib"
    kernel_db = package_lib / "libflashinfer_gaudi_kernels.so"
    if not kernel_db.is_file():
        _FLASHINFER_DB_STATE = KernelDbState.MISSING
        return KernelDbState.MISSING
    reason = _prepend_kernel_database(str(kernel_db))
    if reason is not None:
        if reason not in _LOAD_ERRORS:
            _LOAD_ERRORS.append(reason)
        _FLASHINFER_DB_STATE = KernelDbState.BLOCKED
        return KernelDbState.BLOCKED
    _FLASHINFER_DB_STATE = KernelDbState.REGISTERED
    return KernelDbState.REGISTERED


def _exception_text(exc: BaseException) -> str:
    """``Type: message`` of ``exc`` and of each exception it was raised from, joined by ``; caused by ``.

    ``torch.ops.load_library`` raises ``OSError("Could not load this library: <path>")`` from the ``ctypes.CDLL``
    error, so the dlopen reason (a missing NEEDED library, an undefined symbol) is only in ``__cause__``.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return "; caused by ".join(parts)


def _gdn_fused_error(message: str) -> None:
    """Record one fused GDN loader failure in the shared list and in the GDN-only list."""
    _LOAD_ERRORS.append(message)
    _GDN_FUSED_ERRORS.append(message)


def gdn_fused_extension_errors() -> tuple[str, ...]:
    """Why ``load_gdn_fused_extension`` failed (empty when it loaded or was never called)."""
    return tuple(_GDN_FUSED_ERRORS)


def load_gdn_fused_extension() -> bool:
    """Load libgdn_fused_gaudi2.so (kernel database) and gdn_fused_ops*.so (HPU op) once; True when loaded."""
    global _GDN_FUSED_LOADED
    if _GDN_FUSED_LOADED is not None:
        return _GDN_FUSED_LOADED
    with _LOAD_LOCK:
        if _GDN_FUSED_LOADED is not None:
            return _GDN_FUSED_LOADED
        package_lib = Path(__file__).resolve().parent / "lib"
        kernel_db = os.environ.get("GDN_FUSED_GAUDI_KERNEL_LIBRARY", "")
        if not kernel_db:
            kernel_db = str(package_lib / "libgdn_fused_gaudi2.so")
        extension = os.environ.get("GDN_FUSED_GAUDI_NATIVE_LIBRARY", "")
        if not extension:
            matches = sorted(glob.glob(str(package_lib / "gdn_fused_ops*.so")))
            extension = matches[0] if matches else str(package_lib / "gdn_fused_ops*.so")
        missing = False
        for path in (kernel_db, extension):
            if not os.path.isfile(path):
                _gdn_fused_error(f"{path}: file does not exist")
                missing = True
        if missing:
            _GDN_FUSED_LOADED = False
            return False
        reason = _prepend_kernel_database(kernel_db)
        if reason:
            _gdn_fused_error(reason)
            _GDN_FUSED_LOADED = False
            return False
        # The op's Python schema (T2) must exist before the extension is loaded,
        # because the C++ side only implements the HPU kernel of the fragment.
        try:
            importlib.import_module("flashinfer_gaudi.gdn_core_decode")
        except Exception as exc:  # pragma: no cover - requires the schema module
            _gdn_fused_error(f"flashinfer_gaudi.gdn_core_decode: {type(exc).__name__}: {exc}")
            _GDN_FUSED_LOADED = False
            return False
        try:
            torch.ops.load_library(extension)
        except Exception as exc:  # pragma: no cover - requires an HPU runtime
            _gdn_fused_error(f"{extension}: {_exception_text(exc)}")
            _GDN_FUSED_LOADED = False
            return False
        _LOADED_PATHS.append(extension)
        _GDN_FUSED_LOADED = True
        return True


def load_native_extensions() -> tuple[str, ...]:
    """Load packaged or explicitly configured native libraries once.

    Loading failures are recorded for diagnostics. They are not raised here so
    CPU-only imports and the PyTorch reference fallback remain usable.
    """
    global _LOAD_ATTEMPTED, _SILU_AND_MUL_OP, _ADD_RMSNORM_QUANT_OP
    if _LOAD_ATTEMPTED:
        return tuple(_LOADED_PATHS)
    with _LOAD_LOCK:
        if _LOAD_ATTEMPTED:
            return tuple(_LOADED_PATHS)
        state = _configure_kernel_database_path()
        if state is KernelDbState.BLOCKED:
            # The op extensions' TPC GUIDs have no discoverable kernel database,
            # so loading them would only move the failure to the first graph
            # compile. Leave the handles unset; the callers run the Python paths.
            _LOAD_ATTEMPTED = True
            return tuple(_LOADED_PATHS)
        for candidate in _library_candidates():
            if not os.path.isfile(candidate):
                _LOAD_ERRORS.append(f"{candidate}: file does not exist")
                continue
            try:
                torch.ops.load_library(candidate)
            except Exception as exc:  # pragma: no cover - requires an HPU runtime
                _LOAD_ERRORS.append(f"{candidate}: {_exception_text(exc)}")
            else:
                _LOADED_PATHS.append(candidate)
        try:
            _SILU_AND_MUL_OP = torch.ops.custom_op.flashinfer_gaudi_silu_and_mul
        except (AttributeError, RuntimeError):
            _SILU_AND_MUL_OP = None
        try:
            _ADD_RMSNORM_QUANT_OP = torch.ops.custom_op.flashinfer_gaudi_add_rmsnorm_quant
        except (AttributeError, RuntimeError):
            _ADD_RMSNORM_QUANT_OP = None
        # Private compound ops are published only by the verified Bridge
        # loader, not by probing an arbitrary externally registered symbol.
        _LOAD_ATTEMPTED = True
    return tuple(_LOADED_PATHS)


def _resolve_op(namespace: str, name: str) -> Callable | None:
    load_native_extensions()
    try:
        op_namespace = getattr(torch.ops, namespace)
        return getattr(op_namespace, name)
    except (AttributeError, RuntimeError):
        return None


def public_packed_gdn_op() -> Callable | None:
    """Return the preferred public packed-GDN op, including the legacy GUID."""
    return (_resolve_op("flashinfer_gaudi", "gdn_decode_packed")
            or _resolve_op("custom_op", "custom_gdn_packed_decode_f32_gaudi2"))


def bridge_packed_gdn_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi_bridge", "gdn_decode_packed")


def silu_and_mul_op() -> Callable | None:
    # The public CustomOp API requires custom_op for full torch.compile support.
    # Registration is immutable after the one-shot loader. Resolve it at
    # compilation rather than installing guards on loader locks/path lists.
    if not _LOAD_ATTEMPTED:
        load_native_extensions()
    return _SILU_AND_MUL_OP


def public_mtp_gdn_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "gdn_mtp_packed")


def public_mtp_prepared_op() -> Callable | None:
    return _resolve_op("custom_op", "flashinfer_gaudi_gdn_mtp_prepared")


def native_dflash2_grouped_conv_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_grouped_conv")


def native_dflash2_select_path_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_select_path")


def native_dflash2_score_select_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_score_select")


def native_dflash2_top_k_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_top_k")


def native_diagnostics() -> dict[str, object]:
    load_native_extensions()
    return {
        "loaded_libraries": tuple(_LOADED_PATHS),
        "load_errors": tuple(_LOAD_ERRORS),
        "public_packed_gdn": public_packed_gdn_op() is not None,
        "bridge_packed_gdn": bridge_packed_gdn_op() is not None,
        "silu_and_mul": silu_and_mul_op() is not None,
        "silu_and_mul_quant": silu_mul_quant_op() is not None,
        "block_fp8_dequant": block_fp8_dequant_op() is not None,
        "block_fp8_linear": block_fp8_linear_op() is not None,
        "fused_add_rmsnorm_quant": add_rmsnorm_quant_op() is not None,
        "public_mtp_gdn": public_mtp_gdn_op() is not None,
        "public_mtp_prepared": public_mtp_prepared_op() is not None,
        "dflash2_grouped_conv": native_dflash2_grouped_conv_op() is not None,
        "dflash2_select_path": native_dflash2_select_path_op() is not None,
        "dflash2_score_select": native_dflash2_score_select_op() is not None,
        "dflash2_top_k": native_dflash2_top_k_op() is not None,
    }


def _reset_native_state_for_tests() -> None:
    global _LOAD_ATTEMPTED, _SILU_AND_MUL_OP, _SILU_MUL_QUANT_OP
    global _BLOCK_FP8_DEQUANT_OP, _BLOCK_FP8_LINEAR_OP
    global _ADD_RMSNORM_QUANT_OP, _FLASHINFER_DB_STATE
    with _LOAD_LOCK:
        _LOAD_ATTEMPTED = False
        _SILU_AND_MUL_OP = None
        _SILU_MUL_QUANT_OP = None
        _BLOCK_FP8_DEQUANT_OP = None
        _BLOCK_FP8_LINEAR_OP = None
        _ADD_RMSNORM_QUANT_OP = None
        _FLASHINFER_DB_STATE = KernelDbState.MISSING
        _LOADED_PATHS.clear()
        _LOAD_ERRORS.clear()


def silu_mul_quant_op() -> Callable | None:
    if not _LOAD_ATTEMPTED:
        load_native_extensions()
    return _SILU_MUL_QUANT_OP


def block_fp8_dequant_op() -> Callable | None:
    # Private ops are published only by load_bridge_adapter before compile.
    return _BLOCK_FP8_DEQUANT_OP


def block_fp8_linear_op() -> Callable | None:
    return _BLOCK_FP8_LINEAR_OP


def add_rmsnorm_quant_op() -> Callable | None:
    if not _LOAD_ATTEMPTED:
        load_native_extensions()
    return _ADD_RMSNORM_QUANT_OP


# SynapseAI reads GC_KERNEL_PATH when its graph compiler is initialized. Set
# the packaged kernel database path as soon as this lightweight loader module
# is imported; the PyTorch registration extension itself remains lazy.
_configure_kernel_database_path()
