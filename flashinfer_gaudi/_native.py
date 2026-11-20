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
_COMBINED_DB_ACTIVE: str | None = None  # the combined kernel database this process activated (its path)

RECIPE_CACHE_ENV = "PT_HPU_RECIPE_CACHE_CONFIG"
COMBINED_KERNEL_DB = "libflashinfer_gaudi_combined_gaudi2.so"  # built by RUK-3243; one file carrying every GUID
COMBINED_KERNEL_DB_ENV = "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"  # kill switch: "0" ignores the combined database
FLASHINFER_KERNEL_DB = "libflashinfer_gaudi_kernels.so"  # the FlashInfer kernel database; a member of the combined one
GDN_FUSED_KERNEL_DB = "libgdn_fused_gaudi2.so"  # the fused GDN kernel database; a member of the combined one
GDN_FUSED_KERNEL_LIBRARY_ENV = "GDN_FUSED_GAUDI_KERNEL_LIBRARY"  # its path, when it is not in the package's lib/


class KernelDbState(IntEnum):
    MISSING = 0  # the database file is not installed
    REGISTERED = 1  # prepended to GC_KERNEL_PATH, or already in it
    BLOCKED = 2  # left out because PT_HPU_RECIPE_CACHE_CONFIG is set
    COMBINED = 3  # registered through the combined kernel database (the single GC_KERNEL_PATH file)


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


def _combined_kernel_db_path() -> Path | None:
    """The combined kernel database this process may use, or None when absent or switched off.

    It is looked for in this package's ``lib`` folder first. When it is not there, it is looked for next to the
    library named by ``GDN_FUSED_GAUDI_KERNEL_LIBRARY``: a process that imports this package from a source tree
    (an empty ``lib``) and takes the installed libraries by path, as the rental's kernel gate does, must load the
    same single file the servers load. The combined database serves only the libraries next to itself, so that
    second location counts only when it serves what this process would otherwise register one by one: the
    variable names an existing file called ``libgdn_fused_gaudi2.so``, and this package's ``lib`` holds no
    ``libflashinfer_gaudi_kernels.so`` of its own (which the other folder's combined file would leave out).
    ``FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0`` switches every location off.
    """
    if os.environ.get(COMBINED_KERNEL_DB_ENV, "1").strip() == "0":
        return None
    package_lib = Path(__file__).resolve().parent / "lib"
    path = package_lib / COMBINED_KERNEL_DB
    if path.is_file():
        return path
    if (package_lib / FLASHINFER_KERNEL_DB).is_file():
        return None
    configured = os.environ.get(GDN_FUSED_KERNEL_LIBRARY_ENV, "")
    if configured and os.path.isfile(configured) and os.path.basename(configured) == GDN_FUSED_KERNEL_DB:
        path = Path(os.path.abspath(configured)).parent / COMBINED_KERNEL_DB
        if path.is_file():
            return path
    return None


def _activate_combined_kernel_db() -> bool:
    """Make the combined kernel database the only GC_KERNEL_PATH entry; True when activated."""
    global _COMBINED_DB_ACTIVE
    path = _combined_kernel_db_path()
    if path is None:
        return False

    current = os.environ.get("GC_KERNEL_PATH", "")
    if current in ("", str(path), "/usr/lib/habanalabs/libtpc_kernels.so"):
        os.environ["GC_KERNEL_PATH"] = str(path)
        _COMBINED_DB_ACTIVE = str(path)
        return True

    error = (f"GC_KERNEL_PATH is already set to {current}: the combined kernel database {path} was not "
             f"activated; unset it or set {COMBINED_KERNEL_DB_ENV}=0")
    if error not in _LOAD_ERRORS:
        _LOAD_ERRORS.append(error)
    return False


def _configure_kernel_database_path() -> KernelDbState:
    global _FLASHINFER_DB_STATE
    if _activate_combined_kernel_db():
        # The FlashInfer TPC GUIDs are inside the combined file, so the database is registered.
        _FLASHINFER_DB_STATE = KernelDbState.COMBINED
        return KernelDbState.COMBINED
    package_lib = Path(__file__).resolve().parent / "lib"
    kernel_db = package_lib / FLASHINFER_KERNEL_DB
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
        kernel_db = os.environ.get(GDN_FUSED_KERNEL_LIBRARY_ENV, "")
        if not kernel_db:
            kernel_db = str(package_lib / GDN_FUSED_KERNEL_DB)
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
        # The combined database serves the fused GDN GUIDs, so its member only
        # needs to exist (checked above); with it active GC_KERNEL_PATH is left
        # as the single file and the recipe-cache block does not apply.
        if not _activate_combined_kernel_db():
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


def kernel_database_status() -> dict[str, object]:
    """What the loader decided about ``GC_KERNEL_PATH`` in this process, as plain (JSON-serializable) data.

    It only reports: it loads nothing and changes nothing, so it can be called at any point and logged.

    - ``state``: the name of the ``KernelDbState`` of the FlashInfer kernel database (``MISSING``, ``REGISTERED``,
      ``BLOCKED`` or ``COMBINED``).
    - ``gc_kernel_path``: the value of ``GC_KERNEL_PATH`` now, the one SynapseAI reads (``""`` when unset).
    - ``combined_active``: True only when this loader activated the combined kernel database and
      ``GC_KERNEL_PATH`` is still exactly that one file.
    - ``combined_kernel_db``: the combined kernel database this loader activated, or None.
    - ``gdn_fused_loaded``: the result of ``load_gdn_fused_extension`` (None when it was never called).
    - ``load_errors``: every recorded loader error, in order.

    The member libraries are opened by the combined database itself, inside SynapseAI: a member it cannot
    use is reported on stderr by that file and makes its ``GetKernelGuids`` fail; it is not visible here.
    """
    gc_kernel_path = os.environ.get("GC_KERNEL_PATH", "")
    return {
        "state": _FLASHINFER_DB_STATE.name,
        "gc_kernel_path": gc_kernel_path,
        "combined_active": _COMBINED_DB_ACTIVE is not None and gc_kernel_path == _COMBINED_DB_ACTIVE,
        "combined_kernel_db": _COMBINED_DB_ACTIVE,
        "gdn_fused_loaded": _GDN_FUSED_LOADED,
        "load_errors": list(_LOAD_ERRORS),
    }


def native_diagnostics() -> dict[str, object]:
    load_native_extensions()
    return {
        "loaded_libraries": tuple(_LOADED_PATHS),
        "load_errors": tuple(_LOAD_ERRORS),
        "combined_kernel_db": _COMBINED_DB_ACTIVE,
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
    global _ADD_RMSNORM_QUANT_OP, _FLASHINFER_DB_STATE, _COMBINED_DB_ACTIVE
    with _LOAD_LOCK:
        _LOAD_ATTEMPTED = False
        _SILU_AND_MUL_OP = None
        _SILU_MUL_QUANT_OP = None
        _BLOCK_FP8_DEQUANT_OP = None
        _BLOCK_FP8_LINEAR_OP = None
        _ADD_RMSNORM_QUANT_OP = None
        _FLASHINFER_DB_STATE = KernelDbState.MISSING
        _COMBINED_DB_ACTIVE = None
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
