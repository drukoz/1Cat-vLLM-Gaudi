# SPDX-License-Identifier: Apache-2.0
"""The torch reference copied into ``hc_reference.py`` is the plugin's torch code (needs vLLM).

The TPC simulator container has no ``vllm_gaudi``, so the simulator tests compare the kernels with
copies of the five glue functions of ``vllm_gaudi/models/qwen4_exp/hyperconnection.py``.  This test
ties the copies to the originals: the same bits on random inputs (both dtypes, both norm-weight
layouts, 2-D and 3-D inputs) and the same source text.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import inspect  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("vllm")

from tests.unit_tests.hc_fused import hc_reference  # noqa: E402
from vllm_gaudi.models.qwen4_exp import hyperconnection as plugin  # noqa: E402

HC = 4
SHAPES = [((3, ), 128), ((1, ), 2560), ((2, 5), 64)]  # (leading dims, hidden size)
DTYPES = [torch.bfloat16, torch.float32]


def _inputs(lead, hidden, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    width = HC * hidden

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    return {
        "x": randn(*lead, width, scale=3.0),
        "gate": randn(*lead, width, scale=4.0),
        "block": randn(*lead, hidden, scale=2.0),
        "injection": randn(*lead, HC, scale=5.0),
        "lora": randn(*lead, 320, scale=8.0),
        "weight_full": randn(width, scale=0.3),
        "weight_shared": randn(hidden, scale=0.3),
    }


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


@pytest.mark.parametrize("name", hc_reference.COPIED_FUNCTIONS)
def test_copy_has_the_plugin_source(name):
    assert inspect.getsource(getattr(hc_reference, name)) == inspect.getsource(getattr(plugin, name))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.parametrize("lead,hidden", SHAPES, ids=["3x128", "1x2560", "2x5x64"])
def test_copies_return_the_plugin_bits(lead, hidden, dtype):
    for seed in range(3):
        t = _inputs(lead, hidden, dtype, seed)
        for weight in (t["weight_full"], t["weight_shared"]):
            assert _same(hc_reference.grouped_gemma_rmsnorm_torch(t["x"], weight, 1e-6, HC),
                         plugin.grouped_gemma_rmsnorm_torch(t["x"], weight, 1e-6, HC))
            ours = hc_reference.hc_combine_norm_torch(t["x"], t["block"], t["injection"], weight, 1e-6, HC)
            theirs = plugin.hc_combine_norm_torch(t["x"], t["block"], t["injection"], weight, 1e-6, HC)
            assert _same(ours[0], theirs[0]) and _same(ours[1], theirs[1])
        assert _same(hc_reference.hc_silu_torch(t["lora"], HC), plugin.hc_silu_torch(t["lora"], HC))
        assert _same(hc_reference.hc_gate_mix_torch(t["x"], t["gate"], HC),
                     plugin.hc_gate_mix_torch(t["x"], t["gate"], HC))
        assert _same(hc_reference.hc_combine_torch(t["x"], t["block"], t["injection"], HC),
                     plugin.hc_combine_torch(t["x"], t["block"], t["injection"], HC))


def test_plugin_exports_no_glue_function_this_file_does_not_cover():
    glue = {name for name in plugin.__all__ if name != "HpuGatedResidual"}
    assert glue == set(hc_reference.COPIED_FUNCTIONS)
