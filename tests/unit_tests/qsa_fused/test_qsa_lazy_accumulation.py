# SPDX-License-Identifier: Apache-2.0
"""CPU tests: the fused QSA kernels and the lazy bridge's accumulation thread (card rule 4).

The rule of the hyper-connection kernels (``tests/unit_tests/gdn_core_decode/test_hc_fused_lazy_accumulation.py``)
holds for every custom op: in a lazy process with ``PT_HPU_LAZY_ACC_PAR_MODE`` not at 0 the loader refuses the
kernels (the QSA layers then run the mode-1 torch data flow), and the plugin sets the variable to 0, unless the
user set it, in a process that asks for them (``VLLM_HPU_QSA_FUSED`` 2 or 3; 0 and 1 load no custom op).
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_gaudi
from flashinfer_gaudi import _native

ACC = "PT_HPU_LAZY_ACC_PAR_MODE"
KERNEL_DB = "libqsa_fused_gaudi2.so"
EXTENSION = "qsa_fused_ops.cpython-310-x86_64-linux-gnu.so"


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for name, value in (("_QSA_FUSED_LOADED", None), ("_QSA_FUSED_ERRORS", []), ("_LOAD_ERRORS", []),
                        ("_LOADED_PATHS", []), ("_COMBINED_DB_ACTIVE", None),
                        ("_FLASHINFER_DB_STATE", _native.KernelDbState.MISSING)):
        monkeypatch.setattr(_native, name, value)
    monkeypatch.setenv("GC_KERNEL_PATH", "")
    for name in ("QSA_FUSED_GAUDI_KERNEL_LIBRARY", "QSA_FUSED_GAUDI_NATIVE_LIBRARY", "PT_HPU_RECIPE_CACHE_CONFIG",
                 "FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "FLASHINFER_GAUDI_COMBINED_MEMBERS", "PT_HPU_LAZY_MODE", ACC,
                 "FLASHINFER_GAUDI_ALLOW_LAZY_ACC_THREAD"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    """The schema import and the extension load, in order; the load does not run for real."""
    log: list[str] = []
    real_import = _native.importlib.import_module

    def fake_import(name):
        log.append(f"import {name}")
        return real_import(name)

    monkeypatch.setattr(_native.importlib, "import_module", fake_import)
    monkeypatch.setattr(torch.ops, "load_library", lambda path: log.append(f"load {path}"))
    return log


def package(monkeypatch, tmp_path):
    lib = tmp_path / "flashinfer_gaudi" / "lib"
    lib.mkdir(parents=True)
    monkeypatch.setattr(_native, "__file__", str(lib.parent / "_native.py"))
    (lib / KERNEL_DB).write_bytes(b"")
    (lib / EXTENSION).write_bytes(b"")
    return lib


def bridge(monkeypatch, lazy: bool) -> None:
    module = ModuleType("habana_frameworks.torch")
    module.utils = SimpleNamespace(internal=SimpleNamespace(is_lazy=lambda: lazy))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", module)


@pytest.mark.parametrize("value", [None, "1"])
def test_a_lazy_process_with_the_accumulation_thread_gets_no_qsa_kernels(monkeypatch, tmp_path, events, value):
    package(monkeypatch, tmp_path)
    bridge(monkeypatch, True)
    if value is not None:
        monkeypatch.setenv(ACC, value)

    assert _native.load_qsa_fused_extension() is False

    assert _native.qsa_fused_extension_errors() == (_native.lazy_accumulation_reason(), )
    assert events == []  # no schema, no extension
    assert os.environ["GC_KERNEL_PATH"] == ""  # nothing registered for the graph compiler


def test_a_lazy_process_at_mode_0_loads_the_qsa_kernels(monkeypatch, tmp_path, events):
    lib = package(monkeypatch, tmp_path)
    bridge(monkeypatch, True)
    monkeypatch.setenv(ACC, "0")

    assert _native.load_qsa_fused_extension() is True

    assert _native.qsa_fused_extension_errors() == ()
    assert events == [f"import {_native.QSA_FUSED_SCHEMA_MODULE}", f"load {lib / EXTENSION}"]
    assert os.environ["GC_KERNEL_PATH"].split(os.pathsep)[0] == str(lib / KERNEL_DB)


def test_a_missing_build_is_reported_before_the_mode(monkeypatch, tmp_path, events):
    lib = package(monkeypatch, tmp_path)
    (lib / KERNEL_DB).unlink()
    bridge(monkeypatch, True)

    assert _native.load_qsa_fused_extension() is False
    errors = _native.qsa_fused_extension_errors()
    assert len(errors) == 1 and "file does not exist" in errors[0]


@pytest.mark.parametrize("flag,expected", [("2", "0"), ("3", "0"), ("1", None), ("0", None), ("x", None)])
def test_the_plugin_defaults_mode_0_for_the_kernel_modes_only(flag, expected):
    environ = {"VLLM_HPU_QSA_FUSED": flag}

    assert vllm_gaudi._default_serial_lazy_accumulation(environ) == expected
    assert environ.get(ACC) == expected


def test_the_plugin_keeps_a_value_the_user_set():
    environ = {"VLLM_HPU_QSA_FUSED": "2", ACC: "1"}

    assert vllm_gaudi._default_serial_lazy_accumulation(environ) is None
    assert environ[ACC] == "1"
