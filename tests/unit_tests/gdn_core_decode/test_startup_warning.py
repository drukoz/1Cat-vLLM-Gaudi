# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the one-shot startup warning when the fused GDN kernel has no HPU extension."""

from __future__ import annotations

import pytest

import vllm_gaudi


def test_warns_when_kernel_missing_on_hpu():
    calls: list[str] = []
    errors = ("a.so: file does not exist", "b.so: file does not exist")

    message = vllm_gaudi._warn_gdn_core_decode_without_kernel(1, False, True, errors, warn=calls.append)

    assert message is not None
    assert calls == [message]
    assert "has no HPU kernel" in message
    assert "VLLM_HPU_GDN_CORE_DECODE=1" in message
    assert "a.so: file does not exist; b.so: file does not exist" in message
    assert "set VLLM_HPU_GDN_CORE_DECODE=0" in message


def test_message_uses_mode_value():
    calls: list[str] = []
    errors = ("a.so: file does not exist", "b.so: file does not exist")

    message = vllm_gaudi._warn_gdn_core_decode_without_kernel(2, False, True, errors, warn=calls.append)

    assert message is not None
    assert calls == [message]
    assert "VLLM_HPU_GDN_CORE_DECODE=2" in message


def test_message_without_recorded_errors():
    calls: list[str] = []

    message = vllm_gaudi._warn_gdn_core_decode_without_kernel(1, False, True, (), warn=calls.append)

    assert message is not None
    assert calls == [message]
    assert "(no error recorded)" in message


@pytest.mark.parametrize("mode,loaded,on_hpu", [(1, True, True), (1, False, False), (0, False, True)])
def test_logs_nothing_when_not_applicable(mode, loaded, on_hpu):
    calls: list[str] = []
    errors = ("a.so: file does not exist",)

    message = vllm_gaudi._warn_gdn_core_decode_without_kernel(mode, loaded, on_hpu, errors, warn=calls.append)

    assert message is None
    assert calls == []


def test_hpu_visible_false_without_habana():
    assert vllm_gaudi._hpu_visible() is False
