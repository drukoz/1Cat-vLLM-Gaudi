# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the startup side of VLLM_HPU_HC_FUSED: the flag, the loader call and the one warning."""

from __future__ import annotations

from pathlib import Path

import pytest

import flashinfer_gaudi
import vllm_gaudi
import vllm_gaudi.envs
from flashinfer_gaudi import _native

ERRORS = ("a.so: file does not exist", "b.so: file does not exist")


def test_flag_default_and_value(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_HC_FUSED", raising=False)
    assert vllm_gaudi.envs.VLLM_HPU_HC_FUSED == 0

    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    assert vllm_gaudi.envs.VLLM_HPU_HC_FUSED == 1


def test_flag_is_declared_with_the_flash_next_kernel_flags():
    """In the table right after the GDN core flags (another Flash-Next flag may sit between them), and in
    the type annotations."""
    names = list(vllm_gaudi.envs.environment_variables)
    verify, flag = names.index("VLLM_HPU_GDN_CORE_VERIFY"), names.index("VLLM_HPU_HC_FUSED")
    assert verify < flag <= verify + 2
    source = Path(vllm_gaudi.envs.__file__).read_text(encoding="utf-8")
    assert "    VLLM_HPU_HC_FUSED: int = 0" in source


def test_warns_when_kernel_missing_on_hpu():
    calls: list[str] = []

    message = vllm_gaudi._warn_hc_fused_without_kernel(1, False, True, ERRORS, warn=calls.append)

    assert message is not None
    assert calls == [message]
    assert message.startswith("HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel")
    assert "a.so: file does not exist; b.so: file does not exist" in message
    assert "torch path" in message
    assert "tools/build_flashinfer_gaudi.py --hc-fused-only" in message
    assert "set VLLM_HPU_HC_FUSED=0" in message


def test_message_without_recorded_errors():
    calls: list[str] = []

    message = vllm_gaudi._warn_hc_fused_without_kernel(1, False, True, (), warn=calls.append)

    assert calls == [message]
    assert "(no error recorded)" in message


@pytest.mark.parametrize("flag,loaded,on_hpu", [(1, True, True), (1, False, False), (0, False, True)])
def test_logs_nothing_when_not_applicable(flag, loaded, on_hpu):
    calls: list[str] = []

    message = vllm_gaudi._warn_hc_fused_without_kernel(flag, loaded, on_hpu, ERRORS, warn=calls.append)

    assert message is None
    assert calls == []


@pytest.fixture
def loaders(monkeypatch):
    """The loaders register_ops may call, recorded in call order; none of them runs."""
    calls: list[str] = []
    state = {"hc_loaded": False}
    monkeypatch.setattr(flashinfer_gaudi, "load_native_extensions", lambda: calls.append("native") or ())
    monkeypatch.setattr(_native, "load_gdn_fused_extension", lambda: calls.append("gdn") or True)
    monkeypatch.setattr(_native, "load_hc_fused_extension", lambda: calls.append("hc") or state["hc_loaded"])
    monkeypatch.setattr(_native, "hc_fused_extension_errors", lambda: () if state["hc_loaded"] else ERRORS)
    monkeypatch.setattr(_native, "kernel_database_status", lambda: calls.append("status") or {})
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "0")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_VERIFY", "0")
    monkeypatch.delenv("VLLM_HPU_HC_FUSED", raising=False)
    return calls, state


def _register_ops() -> None:
    try:
        vllm_gaudi.register_ops()
    except Exception:  # noqa: BLE001 - without habana_frameworks the imports after the loaders fail
        pass


def test_register_ops_does_not_touch_the_loader_by_default(loaders):
    calls, _ = loaders

    _register_ops()

    assert calls == ["native", "status"]


def test_register_ops_does_not_touch_the_loader_with_the_flag_at_zero(loaders, monkeypatch):
    calls, _ = loaders
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "0")

    _register_ops()

    assert calls == ["native", "status"]


def test_register_ops_loads_the_kernels_with_the_flag_before_the_status_line(loaders, monkeypatch):
    calls, _ = loaders
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")

    _register_ops()

    assert calls == ["native", "hc", "status"]


def test_register_ops_loads_the_gdn_kernels_first(loaders, monkeypatch):
    calls, _ = loaders
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "1")
    monkeypatch.setenv("VLLM_HPU_GDN_CORE_DECODE", "1")

    _register_ops()

    assert calls == ["native", "gdn", "hc", "status"]


def test_any_other_flag_value_loads_nothing(loaders, monkeypatch):
    calls, _ = loaders
    monkeypatch.setenv("VLLM_HPU_HC_FUSED", "2")

    _register_ops()

    assert calls == ["native", "status"]


def test_load_if_requested_warns_once_when_the_kernel_is_not_usable(loaders, monkeypatch):
    calls, _ = loaders
    warnings: list[str] = []
    monkeypatch.setattr(vllm_gaudi, "_hpu_visible", lambda: True)

    assert vllm_gaudi._load_hc_fused_if_requested(1, warn=warnings.append) is False
    assert calls == ["hc"]
    assert len(warnings) == 1 and "has no HPU kernel" in warnings[0] and ERRORS[0] in warnings[0]


def test_load_if_requested_is_silent_when_the_kernel_loaded(loaders, monkeypatch):
    calls, state = loaders
    state["hc_loaded"] = True
    warnings: list[str] = []
    monkeypatch.setattr(vllm_gaudi, "_hpu_visible", lambda: True)

    assert vllm_gaudi._load_hc_fused_if_requested(1, warn=warnings.append) is True
    assert calls == ["hc"] and warnings == []


@pytest.mark.parametrize("flag", [0, 2, -1])
def test_load_if_requested_does_nothing_without_the_flag(loaders, flag):
    calls, _ = loaders
    warnings: list[str] = []

    assert vllm_gaudi._load_hc_fused_if_requested(flag, warn=warnings.append) is None
    assert calls == [] and warnings == []
