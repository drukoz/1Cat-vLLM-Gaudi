# SPDX-License-Identifier: Apache-2.0
"""Lazy loading and discovery of optional native Gaudi kernels."""

from __future__ import annotations

import ctypes
import glob
import importlib
import os
import sys
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
_HC_FUSED_LOADED: bool | None = None  # None = not attempted (the flag VLLM_HPU_HC_FUSED is not 1)
_HC_FUSED_ERRORS: list[str] = []  # the fused hyper-connection loader's own errors
_MOE_GLUE_LOADED: bool | None = None  # None = not attempted (the flag VLLM_HPU_MOE_GLUE is not 1)
_MOE_GLUE_ERRORS: list[str] = []  # the MoE glue loader's own errors (the library's are the hyper-connection loader's)
_MOE_GLUE_MISSING: tuple[str, ...] = ()  # the MoE glue ops whose kernel the library does not serve
_COMBINED_DB_ACTIVE: str | None = None  # the combined kernel database this process activated (its path)

RECIPE_CACHE_ENV = "PT_HPU_RECIPE_CACHE_CONFIG"
COMBINED_KERNEL_DB = "libflashinfer_gaudi_combined_gaudi2.so"  # built by RUK-3243; one file carrying every GUID
COMBINED_KERNEL_DB_ENV = "FLASHINFER_GAUDI_COMBINED_KERNEL_DB"  # kill switch: "0" ignores the combined database
FLASHINFER_KERNEL_DB = "libflashinfer_gaudi_kernels.so"  # the FlashInfer kernel database; a member of the combined one
GDN_FUSED_KERNEL_DB = "libgdn_fused_gaudi2.so"  # the fused GDN kernel database; a member of the combined one
GDN_FUSED_KERNEL_LIBRARY_ENV = "GDN_FUSED_GAUDI_KERNEL_LIBRARY"  # its path, when it is not in the package's lib/
HC_FUSED_KERNEL_DB = "libhc_fused_gaudi2.so"  # the fused hyper-connection kernel database; an optional member
HC_FUSED_KERNEL_LIBRARY_ENV = "HC_FUSED_GAUDI_KERNEL_LIBRARY"  # its path, when it is not in the package's lib/
HC_FUSED_NATIVE_LIBRARY_ENV = "HC_FUSED_GAUDI_NATIVE_LIBRARY"  # the path of hc_fused_ops*.so, same rule
HC_FUSED_EXTENSION_GLOB = "hc_fused_ops*.so"
HC_FUSED_SCHEMA_MODULE = "flashinfer_gaudi.hc_fused"  # defines the six op schemas; imported only with the flag at 1
# The MoE glue ops are implemented by the same extension, so importing the module above imports this one too.
MOE_GLUE_SCHEMA_MODULE = "flashinfer_gaudi.moe_glue"
COMBINED_MEMBERS_ENV = "FLASHINFER_GAUDI_COMBINED_MEMBERS"  # csrc/kernel_db_adapter/adapter.cpp (host tests)
LAZY_ACC_PAR_ENV = "PT_HPU_LAZY_ACC_PAR_MODE"  # the lazy bridge's accumulation thread: 1 (its default) on, 0 off
ALLOW_LAZY_ACC_THREAD_ENV = "FLASHINFER_GAUDI_ALLOW_LAZY_ACC_THREAD"  # "1": load custom ops next to that thread


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


def lazy_bridge_loaded() -> bool:
    """True when the Habana PyTorch bridge is imported in this process and runs in lazy mode.

    A custom-op extension binds to the bridge's plugin library, so the bridge is imported before a loader
    here can load one; a process without it has no HPU op to protect.
    """
    bridge = sys.modules.get("habana_frameworks.torch")
    if bridge is None:
        return False
    try:
        return bool(bridge.utils.internal.is_lazy())
    except Exception:  # noqa: BLE001 - a bridge that cannot be asked: the variable that selects the mode decides
        return os.environ.get("PT_HPU_LAZY_MODE", "1").strip() != "0"


def lazy_accumulation_reason() -> str | None:
    """Why a custom HPU op must not run in this process, or None.

    In lazy mode with ``PT_HPU_LAZY_ACC_PAR_MODE=1`` (the bridge's default) the bridge builds the graph nodes
    of queued ops on a second thread, while ``UserCustomOpDescriptor::execute`` builds the node of a custom op
    at once on the calling thread. An input whose producer is still queued is then wired as a graph input
    that never gets data, and the graph fails when it runs. Measured on a Gaudi2 node on 2026-11-20 (bridge
    1.24.1): the hyper-connection card test failed 6 and 7 of 90 tests at 1, a different set each run, and
    passed 90 of 90 at 0 (``docs/design/flash-next/native-extensions.md``, "Custom ops and the lazy
    accumulation thread"). The second thread is read off the failing graphs and the effect of the variable;
    the bridge's source is not public. Only the exact value 0 counts as off.
    ``FLASHINFER_GAUDI_ALLOW_LAZY_ACC_THREAD=1`` lifts the refusal for the probes and control runs that
    reproduce the failure on purpose.
    """
    if os.environ.get(ALLOW_LAZY_ACC_THREAD_ENV, "").strip() == "1" or not lazy_bridge_loaded():
        return None
    value = os.environ.get(LAZY_ACC_PAR_ENV)
    if value is not None and value.strip() == "0":
        return None
    shown = "unset (the bridge's default is 1)" if value is None else repr(value)
    return (f"{LAZY_ACC_PAR_ENV} is {shown}: the lazy Habana bridge then builds queued ops on a second thread, and a "
            f"custom op called while the producer of one of its inputs is still queued takes that input as a graph "
            f"input without data. The graph fails when it runs, in a different place each time ('Empty tensor "
            f"optional', 'ValidateSyncInputTensors tensor_data is empty'; Gaudi 1.24.1). Start the process with "
            f"{LAZY_ACC_PAR_ENV}=0; the fused kernels are not loaded without it.")


def _hc_fused_error(message: str) -> None:
    """Record one fused hyper-connection loader failure in the shared list and in the HC-only list."""
    _LOAD_ERRORS.append(message)
    _HC_FUSED_ERRORS.append(message)


def hc_fused_extension_errors() -> tuple[str, ...]:
    """Why ``load_hc_fused_extension`` failed (empty when it loaded or was never called)."""
    return tuple(_HC_FUSED_ERRORS)


def hc_fused_extension_loaded() -> bool | None:
    """The result of ``load_hc_fused_extension``; None when it was never called (``VLLM_HPU_HC_FUSED`` is not 1)."""
    return _HC_FUSED_LOADED


def combined_db_serves_hc_fused(combined: Path | str, kernel_db: str) -> bool:
    """True when the combined kernel database `combined` forwards the GUIDs of the library `kernel_db`.

    The adapter serves the libraries of a fixed name list that sit next to it. The list is compiled in, so
    an adapter built before ``libhc_fused_gaudi2.so`` was added to it does not serve the file even when it is
    there; the name is then absent from the adapter's bytes. With the members override set (host tests) the
    adapter serves exactly the paths of the override. A combined file that cannot be read serves nothing.
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


def _load_hc_fused_locked() -> bool:
    """The one attempt of ``load_hc_fused_extension`` (the caller holds ``_LOAD_LOCK``).

    The order is what makes the fallback safe: both files exist; the bridge can run a custom op
    (``lazy_accumulation_reason``); then the kernel database is registered for the graph compiler; only then
    the schemas are imported and the extension, which registers the HPU kernels, is loaded. A failure at any
    step leaves the ops without an HPU kernel, and the layers then run torch.
    """
    package_lib = Path(__file__).resolve().parent / "lib"
    kernel_db = os.environ.get(HC_FUSED_KERNEL_LIBRARY_ENV, "") or str(package_lib / HC_FUSED_KERNEL_DB)
    extension = os.environ.get(HC_FUSED_NATIVE_LIBRARY_ENV, "")
    if not extension:
        matches = sorted(glob.glob(str(package_lib / HC_FUSED_EXTENSION_GLOB)))
        extension = matches[0] if matches else str(package_lib / HC_FUSED_EXTENSION_GLOB)
    missing = False
    for path in (kernel_db, extension):
        if not os.path.isfile(path):
            _hc_fused_error(f"{path}: file does not exist")
            missing = True
    if missing:
        return False
    # Before anything is registered: a lazy bridge with its accumulation thread on cannot run a custom op
    # reliably, and a process without the kernels runs the torch path.
    reason = lazy_accumulation_reason()
    if reason:
        _hc_fused_error(reason)
        return False
    # The rules of the fused GDN library: the combined database when it can be the single
    # GC_KERNEL_PATH file, otherwise first in the colon list, which is refused next to a recipe
    # cache. One more condition here: the combined file must serve this library (an adapter
    # built before the library was a member does not, and would leave its GUIDs unserved).
    combined = _combined_kernel_db_path()
    serves = combined is not None and combined_db_serves_hc_fused(combined, kernel_db)
    if not (serves and _activate_combined_kernel_db()):
        reason = _prepend_kernel_database(kernel_db)
        if reason:
            if combined is not None and not serves:
                reason += (f" The combined kernel database {combined} does not serve {HC_FUSED_KERNEL_DB}: rebuild "
                           f"it with tools/build_flashinfer_gaudi.py --combined-only after building the library.")
            _hc_fused_error(reason)
            return False
    # The schemas must exist before the extension is loaded: the C++ side only implements the HPU kernels.
    try:
        importlib.import_module(HC_FUSED_SCHEMA_MODULE)
    except Exception as exc:  # pragma: no cover - the schema module is part of this package
        _hc_fused_error(f"{HC_FUSED_SCHEMA_MODULE}: {type(exc).__name__}: {exc}")
        return False
    try:
        torch.ops.load_library(extension)
    except Exception as exc:
        _hc_fused_error(f"{extension}: {_exception_text(exc)}")
        return False
    _LOADED_PATHS.append(extension)
    return True


def load_hc_fused_extension() -> bool:
    """Register libhc_fused_gaudi2.so (kernel database) and load hc_fused_ops*.so (HPU ops) once; True when loaded.

    Called only with ``VLLM_HPU_HC_FUSED=1``, before the first HPU op of the process: SynapseAI reads
    ``GC_KERNEL_PATH`` when its graph compiler starts. It never raises; ``hc_fused_extension_errors`` says why
    it returned False.
    """
    global _HC_FUSED_LOADED
    if _HC_FUSED_LOADED is not None:
        return _HC_FUSED_LOADED
    with _LOAD_LOCK:
        if _HC_FUSED_LOADED is None:
            try:
                _HC_FUSED_LOADED = _load_hc_fused_locked()
            except Exception as exc:  # noqa: BLE001 - an optional kernel never stops a process
                _hc_fused_error(f"load_hc_fused_extension: {_exception_text(exc)}")
                _HC_FUSED_LOADED = False
        return _HC_FUSED_LOADED


class _GuidInfo(ctypes.Structure):
    """``tpc_lib_api::GuidInfo`` (tpc_kernel_lib_interface.h): the name, the name hash, the property bits."""

    _fields_ = [("name", ctypes.c_char * 64), ("name_hash", ctypes.c_uint64), ("kernel_properties", ctypes.c_uint32)]


def kernel_library_guids(path: Path | str) -> tuple[str, ...]:
    """The Gaudi2 kernel GUIDs the kernel library `path` lists, in its order.

    The library is opened the way the graph compiler opens it and asked through its ``GetKernelGuids`` entry
    point (the two-call protocol). Raises ``OSError`` when it does not load and ``RuntimeError`` when the entry
    point is missing or fails. A kernel library links nothing from the Gaudi runtime.
    """
    library = ctypes.CDLL(str(path))
    if not hasattr(library, "GetKernelGuids"):
        raise RuntimeError(f"{path} does not export GetKernelGuids")
    library.GetKernelGuids.restype = ctypes.c_int
    library.GetKernelGuids.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    gaudi2 = 3  # tpc_lib_api::DEVICE_ID_GAUDI2 (tpc_kernel_lib_interface.h)
    count = ctypes.c_uint32(0)
    status = library.GetKernelGuids(gaudi2, ctypes.byref(count), None)
    if status != 0:
        raise RuntimeError(f"{path}: GetKernelGuids returned {status}")
    guids = (_GuidInfo * max(count.value, 1))()
    status = library.GetKernelGuids(gaudi2, ctypes.byref(count), ctypes.cast(guids, ctypes.c_void_p))
    if status != 0:
        raise RuntimeError(f"{path}: GetKernelGuids returned {status}")
    return tuple(guids[index].name.decode("ascii", "replace") for index in range(count.value))


def moe_glue_extension_errors() -> tuple[str, ...]:
    """Why ``load_moe_glue_extension`` failed (empty when it loaded or was never called).

    The MoE glue kernels are in the fused hyper-connection library, so a failure to load that library is
    reported with that loader's own words.
    """
    return tuple(_HC_FUSED_ERRORS) + tuple(_MOE_GLUE_ERRORS) if _MOE_GLUE_LOADED is not None else ()


def moe_glue_extension_loaded() -> bool | None:
    """The result of ``load_moe_glue_extension``; None when it was never called (``VLLM_HPU_MOE_GLUE`` is not 1)."""
    return _MOE_GLUE_LOADED


def moe_glue_missing_kernels() -> tuple[str, ...]:
    """The MoE glue ops whose kernel is not served, as the last ``load_moe_glue_extension`` found it.

    Empty when every kernel is served and before the loader was called; every op when the library did not load.
    """
    return _MOE_GLUE_MISSING


MOE_GLUE_OPS = ("moe_route", "moe_silu_mul", "moe_combine", "moe_row_quant", "moe_act_quant")  # flashinfer_gaudi.moe_glue.OP_NAMES


def _load_moe_glue_locked() -> bool:
    """The one attempt of ``load_moe_glue_extension`` (the caller holds ``_MOE_GLUE_LOCK``)."""
    global _MOE_GLUE_MISSING
    _MOE_GLUE_MISSING = MOE_GLUE_OPS
    if not load_hc_fused_extension():
        return False
    # Imported by the hyper-connection schema module before the extension was loaded.
    from flashinfer_gaudi import moe_glue

    package_lib = Path(__file__).resolve().parent / "lib"
    kernel_db = os.environ.get(HC_FUSED_KERNEL_LIBRARY_ENV, "") or str(package_lib / HC_FUSED_KERNEL_DB)
    try:
        served = kernel_library_guids(kernel_db)
    except Exception as exc:  # noqa: BLE001 - a library that cannot be asked serves nothing
        message = f"{kernel_db}: cannot list its kernels ({_exception_text(exc)})"
        _LOAD_ERRORS.append(message)
        _MOE_GLUE_ERRORS.append(message)
        return False
    missing = tuple(name for name in MOE_GLUE_OPS if moe_glue.GUIDS[name] not in served)
    _MOE_GLUE_MISSING = missing
    if missing:
        named = ", ".join(f"{name} ({moe_glue.GUIDS[name]})" for name in missing)
        message = (f"{kernel_db} has no kernel for {named}: a kernel that does not compile is left out when the "
                   f"library is built (csrc/hc_fused/CMakeLists.txt)")
        _LOAD_ERRORS.append(message)
        _MOE_GLUE_ERRORS.append(message)
        return False
    return True


_MOE_GLUE_LOCK = threading.Lock()


def load_moe_glue_extension() -> bool:
    """Make the fused MoE glue ops usable on HPU, once; True when all four have a kernel.

    The kernels are in ``libhc_fused_gaudi2.so`` and the ops in ``hc_fused_ops*.so``, so this is
    ``load_hc_fused_extension`` (which registers the library and loads the extension, whichever of the two
    flags asks first) followed by one question to the library: which of the four MoE kernels it serves, each
    being optional at build time. Called only with ``VLLM_HPU_MOE_GLUE=1``, before the first HPU op of the
    process. It never raises; ``moe_glue_extension_errors`` says why it returned False and
    ``moe_glue_missing_kernels`` which ops have no kernel.
    """
    global _MOE_GLUE_LOADED, _MOE_GLUE_MISSING
    if _MOE_GLUE_LOADED is not None:
        return _MOE_GLUE_LOADED
    with _MOE_GLUE_LOCK:
        if _MOE_GLUE_LOADED is None:
            try:
                _MOE_GLUE_LOADED = _load_moe_glue_locked()
            except Exception as exc:  # noqa: BLE001 - an optional kernel never stops a process
                message = f"load_moe_glue_extension: {_exception_text(exc)}"
                _LOAD_ERRORS.append(message)
                _MOE_GLUE_ERRORS.append(message)
                _MOE_GLUE_MISSING = MOE_GLUE_OPS
                _MOE_GLUE_LOADED = False
        return _MOE_GLUE_LOADED


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
    - ``hc_fused_loaded``: the result of ``load_hc_fused_extension`` (None when it was never called, which is
      every process without ``VLLM_HPU_HC_FUSED=1``).
    - ``moe_glue_loaded``: the result of ``load_moe_glue_extension``. The key exists only when that loader was
      called (``VLLM_HPU_MOE_GLUE=1``); such a process also has a result under ``hc_fused_loaded``, because
      the MoE glue kernels are in the hyper-connection library.
    - ``load_errors``: every recorded loader error, in order.

    The member libraries are opened by the combined database itself, inside SynapseAI: a member it cannot
    use is reported on stderr by that file and makes its ``GetKernelGuids`` fail; it is not visible here.
    """
    gc_kernel_path = os.environ.get("GC_KERNEL_PATH", "")
    status: dict[str, object] = {
        "state": _FLASHINFER_DB_STATE.name,
        "gc_kernel_path": gc_kernel_path,
        "combined_active": _COMBINED_DB_ACTIVE is not None and gc_kernel_path == _COMBINED_DB_ACTIVE,
        "combined_kernel_db": _COMBINED_DB_ACTIVE,
        "gdn_fused_loaded": _GDN_FUSED_LOADED,
        "hc_fused_loaded": _HC_FUSED_LOADED,
        "load_errors": list(_LOAD_ERRORS),
    }
    if _MOE_GLUE_LOADED is not None:
        status["moe_glue_loaded"] = _MOE_GLUE_LOADED
    return status


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
    global _ADD_RMSNORM_QUANT_OP, _FLASHINFER_DB_STATE, _COMBINED_DB_ACTIVE, _HC_FUSED_LOADED
    global _MOE_GLUE_LOADED, _MOE_GLUE_MISSING
    with _LOAD_LOCK:
        _LOAD_ATTEMPTED = False
        _SILU_AND_MUL_OP = None
        _SILU_MUL_QUANT_OP = None
        _BLOCK_FP8_DEQUANT_OP = None
        _BLOCK_FP8_LINEAR_OP = None
        _ADD_RMSNORM_QUANT_OP = None
        _FLASHINFER_DB_STATE = KernelDbState.MISSING
        _COMBINED_DB_ACTIVE = None
        _HC_FUSED_LOADED = None
        _HC_FUSED_ERRORS.clear()
        _MOE_GLUE_LOADED = None
        _MOE_GLUE_MISSING = ()
        _MOE_GLUE_ERRORS.clear()
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


# ---------------------------------------------------------------------------------------------------------
# The fused QSA decode kernels (VLLM_HPU_QSA_FUSED=2): libqsa_fused_gaudi2.so + qsa_fused_ops*.so, loaded by the
# rules of the fused hyper-connection kernels above. docs/design/flash-next/qsa-fused.md.
# ---------------------------------------------------------------------------------------------------------
QSA_FUSED_KERNEL_DB = "libqsa_fused_gaudi2.so"  # an optional member of the combined kernel database
QSA_FUSED_KERNEL_LIBRARY_ENV = "QSA_FUSED_GAUDI_KERNEL_LIBRARY"  # its path, when it is not in the package's lib/
QSA_FUSED_NATIVE_LIBRARY_ENV = "QSA_FUSED_GAUDI_NATIVE_LIBRARY"  # the path of qsa_fused_ops*.so, same rule
QSA_FUSED_EXTENSION_GLOB = "qsa_fused_ops*.so"
QSA_FUSED_SCHEMA_MODULE = "flashinfer_gaudi.qsa_fused"  # defines the op schemas; imported only with the flag at 2 or 3
_QSA_FUSED_LOADED: bool | None = None  # None = not attempted
_QSA_FUSED_ERRORS: list[str] = []
_QSA_FUSED_LOCK = threading.Lock()


def _qsa_fused_error(message: str) -> None:
    _LOAD_ERRORS.append(message)
    _QSA_FUSED_ERRORS.append(message)


def qsa_fused_extension_errors() -> tuple[str, ...]:
    """Why ``load_qsa_fused_extension`` failed (empty when it loaded or was never called)."""
    return tuple(_QSA_FUSED_ERRORS)


def qsa_fused_extension_loaded() -> bool | None:
    """The result of ``load_qsa_fused_extension``; None when it was never called (``VLLM_HPU_QSA_FUSED`` is not 2)."""
    return _QSA_FUSED_LOADED


def combined_db_serves_qsa_fused(combined: Path | str, kernel_db: str) -> bool:
    """True when the combined kernel database `combined` forwards the GUIDs of `kernel_db` (as for hc_fused)."""
    override = os.environ.get(COMBINED_MEMBERS_ENV, "")
    if override:
        return os.path.abspath(kernel_db) in [os.path.abspath(entry) for entry in override.split(":") if entry]
    path = Path(os.path.abspath(kernel_db))
    if path.name != QSA_FUSED_KERNEL_DB or path.parent != Path(os.path.abspath(combined)).parent:
        return False
    try:
        return QSA_FUSED_KERNEL_DB.encode() in Path(combined).read_bytes()
    except OSError:
        return False


def _load_qsa_fused_locked() -> bool:
    """The one attempt of ``load_qsa_fused_extension``: files, the bridge's accumulation mode
    (``lazy_accumulation_reason``, as the hyper-connection loader), kernel database, schemas, then the extension."""
    package_lib = Path(__file__).resolve().parent / "lib"
    kernel_db = os.environ.get(QSA_FUSED_KERNEL_LIBRARY_ENV, "") or str(package_lib / QSA_FUSED_KERNEL_DB)
    extension = os.environ.get(QSA_FUSED_NATIVE_LIBRARY_ENV, "")
    if not extension:
        matches = sorted(glob.glob(str(package_lib / QSA_FUSED_EXTENSION_GLOB)))
        extension = matches[0] if matches else str(package_lib / QSA_FUSED_EXTENSION_GLOB)
    missing = False
    for path in (kernel_db, extension):
        if not os.path.isfile(path):
            _qsa_fused_error(f"{path}: file does not exist")
            missing = True
    if missing:
        return False
    # Before anything is registered: a lazy bridge with its accumulation thread on cannot run a custom op
    # reliably (card rule 4); a process without the kernels runs the mode-1 torch data flow.
    reason = lazy_accumulation_reason()
    if reason:
        _qsa_fused_error(reason)
        return False
    combined = _combined_kernel_db_path()
    serves = combined is not None and combined_db_serves_qsa_fused(combined, kernel_db)
    if not (serves and _activate_combined_kernel_db()):
        reason = _prepend_kernel_database(kernel_db)
        if reason:
            if combined is not None and not serves:
                reason += (f" The combined kernel database {combined} does not serve {QSA_FUSED_KERNEL_DB}: rebuild "
                           f"it with tools/build_flashinfer_gaudi.py --combined-only after building the library.")
            _qsa_fused_error(reason)
            return False
    try:
        importlib.import_module(QSA_FUSED_SCHEMA_MODULE)
    except Exception as exc:  # pragma: no cover - the schema module is part of this package
        _qsa_fused_error(f"{QSA_FUSED_SCHEMA_MODULE}: {type(exc).__name__}: {exc}")
        return False
    try:
        torch.ops.load_library(extension)
    except Exception as exc:
        _qsa_fused_error(f"{extension}: {_exception_text(exc)}")
        return False
    _LOADED_PATHS.append(extension)
    return True


def load_qsa_fused_extension() -> bool:
    """Register libqsa_fused_gaudi2.so and load qsa_fused_ops*.so once; True when loaded. Never raises.

    Called only with ``VLLM_HPU_QSA_FUSED=2`` or ``3``, before the first HPU op of the process (SynapseAI reads
    ``GC_KERNEL_PATH`` when its graph compiler starts). ``qsa_fused_extension_errors`` says why it returned False.
    """
    global _QSA_FUSED_LOADED
    if _QSA_FUSED_LOADED is not None:
        return _QSA_FUSED_LOADED
    with _QSA_FUSED_LOCK:
        if _QSA_FUSED_LOADED is None:
            try:
                with _LOAD_LOCK:
                    _QSA_FUSED_LOADED = _load_qsa_fused_locked()
            except Exception as exc:  # noqa: BLE001 - an optional kernel never stops a process
                _qsa_fused_error(f"load_qsa_fused_extension: {_exception_text(exc)}")
                _QSA_FUSED_LOADED = False
        return _QSA_FUSED_LOADED


# SynapseAI reads GC_KERNEL_PATH when its graph compiler is initialized. Set
# the packaged kernel database path as soon as this lightweight loader module
# is imported; the PyTorch registration extension itself remains lazy.
_configure_kernel_database_path()
