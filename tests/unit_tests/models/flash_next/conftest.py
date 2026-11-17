# SPDX-License-Identifier: Apache-2.0
"""Session fixtures for the Flash-Next whole-model CPU tests (M0-T7a and the T7b/T7c/T7d parity tasks)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point
os.environ["TMPDIR"] = "/tmp"  # keep the gloo init file off a sandboxed TMPDIR

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import build_tiny_checkpoint  # noqa: E402


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """The tiny Flash-Next checkpoint, built once per session in a subprocess (no distributed state)."""
    return build_tiny_checkpoint(tmp_path_factory.mktemp("flash_next_tiny"))
