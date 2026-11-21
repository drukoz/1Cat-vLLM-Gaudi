# SPDX-License-Identifier: Apache-2.0
"""Session fixtures for the Flash-Next whole-model CPU tests (M0-T7a and the T7b/T7c/T7d parity tasks)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # the plugin registers by hand, not through vLLM's entry point
os.environ["TMPDIR"] = "/tmp"  # keep the gloo init file off a sandboxed TMPDIR

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup, size_id  # noqa: E402
from tests.unit_tests.models.flash_next.cpu_model import build_tiny_checkpoint  # noqa: E402


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """The tiny Flash-Next checkpoint, built once per session in a subprocess (no distributed state)."""
    return build_tiny_checkpoint(tmp_path_factory.mktemp("flash_next_tiny"))


@pytest.fixture(scope="session")
def tiny_tp8_checkpoint(tmp_path_factory) -> Path:
    """The 8-attention-head tiny checkpoint (legal at TP8), with its own regenerated reference.json."""
    fixture = Path(__file__).resolve().parent / "tiny_tp8_config.json"
    return build_tiny_checkpoint(tmp_path_factory.mktemp("flash_next_tiny_tp8"), fixture=fixture)


@pytest.fixture(params=SPEC_SIZES, ids=size_id)
def spec_k(request) -> int:
    """Each speculative size ``k`` (drafted tokens per step) the MTP tests run at.

    The platform's startup validator is asked first (``flash_next_spec_sizes.require_startup``): a served
    size it refuses fails here with its message, and so does a size listed as proven ahead of the
    validator once the validator accepts it.
    """
    require_startup(request.param)
    return request.param
