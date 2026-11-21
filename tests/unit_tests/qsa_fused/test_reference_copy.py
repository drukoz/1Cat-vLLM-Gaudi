# SPDX-License-Identifier: Apache-2.0
"""The torch helpers copied into ``qsa_reference.py`` are the plugin's (same source text; needs vLLM)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import inspect  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("vllm")

from tests.unit_tests.qsa_fused import qsa_reference  # noqa: E402
from vllm_gaudi.models.qwen4_exp import qsa_hpu_attention, qsa_sparse_ops  # noqa: E402


@pytest.mark.parametrize("name", qsa_reference.COPIED)
def test_copy_has_the_plugin_source(name):
    plugin = getattr(qsa_sparse_ops, name, None) or getattr(qsa_hpu_attention, name)
    assert inspect.getsource(getattr(qsa_reference, name)) == inspect.getsource(plugin), (
        f"{name} in qsa_reference.py is not the plugin's: copy it again")
