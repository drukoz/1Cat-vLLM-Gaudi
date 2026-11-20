# SPDX-License-Identifier: Apache-2.0
"""The sparse QSA wrapper is built on the HPU platform, and a missing wrapper is an error there.

``HpuQwen4ExpQSAAttention.__init__`` decided "am I on HPU?" with
``getattr(current_platform, "is_hpu", None)``. No platform defines ``is_hpu``: vLLM's
``Platform.__getattr__`` logs ``Current platform hpu does not have 'is_hpu' attribute.`` (the line a
card log printed) and returns ``None``, so ``sparse_attn`` stayed ``None`` and ``QsaMode.SPARSE``
silently ran dense attention, which is wrong beyond about 2050 tokens of context.

The platform here is a real ``vllm.platforms.interface.Platform`` subclass with the attribute
surface of ``vllm_gaudi.platform.HpuPlatform`` (``vllm_gaudi.platform`` itself imports
``habana_frameworks`` and cannot be imported on a CPU-only machine).
"""

import contextlib
import os
import shutil
import sys
import tempfile

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel, init_distributed_environment, initialize_model_parallel,
)
from vllm.platforms.interface import Platform, PlatformEnum  # noqa: E402
from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402

import vllm_gaudi.models.qwen4_exp.qsa as qsa_module  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import HpuQsaSparseAttention  # noqa: E402

from tests.unit_tests.models.flash_next.test_qsa_dense import FIXTURE, LAYER_ID, PREFIX, _engine_config  # noqa: E402

WRAPPER_MODULE = "vllm_gaudi.models.qwen4_exp.qsa_hpu_attention"


@contextlib.contextmanager
def built_layer(tmp_path, monkeypatch, mode: QsaMode):
    """Build one QSA layer in ``mode`` on a world-1 gloo group; the group is destroyed even when the build raises."""
    monkeypatch.setenv("VLLM_HPU_FLASH_NEXT_QSA_MODE", str(int(mode)))
    shutil.copy(FIXTURE, tmp_path / "config.json")
    vllm_config = _engine_config(tmp_path)
    init_file = tempfile.mkstemp(dir="/tmp")[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1,
                                     rank=0,
                                     distributed_init_method=f"file://{init_file}",
                                     local_rank=0,
                                     backend="gloo")
        try:
            initialize_model_parallel(1, 1)
            with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
                layer = HpuQwen4ExpQSAAttention(vllm_config=vllm_config,
                                                config=vllm_config.model_config.hf_text_config,
                                                layer_id=LAYER_ID,
                                                prefix=PREFIX)
            yield layer
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(init_file)


class FakeHpuPlatform(Platform):
    """The attribute surface of ``HpuPlatform`` that the platform test may read (no ``is_hpu``)."""
    _enum = PlatformEnum.OOT
    device_name: str = "hpu"
    device_type: str = "hpu"
    dispatch_key: str = "HPU"


@pytest.fixture()
def hpu_platform(monkeypatch):
    platform = FakeHpuPlatform()
    monkeypatch.setattr(qsa_module, "current_platform", platform)
    return platform


def test_hpu_platform_has_no_is_hpu(hpu_platform):
    """The premise: the platform object exposes ``device_type``, and ``is_hpu`` resolves to ``None``."""
    assert hpu_platform.device_type == "hpu"
    assert getattr(hpu_platform, "is_hpu", None) is None


def test_sparse_mode_builds_the_wrapper_on_hpu(tmp_path, monkeypatch, hpu_platform):
    with built_layer(tmp_path, monkeypatch, QsaMode.SPARSE) as layer:
        assert layer.qsa_mode is QsaMode.SPARSE
        assert isinstance(layer.sparse_attn, HpuQsaSparseAttention), (
            "QsaMode.SPARSE on the HPU platform built no sparse wrapper: the forward would run dense attention")
        assert layer.sparse_attn.attn is layer.attn


def test_sparse_mode_builds_the_wrapper_while_the_load_device_is_overridden(tmp_path, monkeypatch, hpu_platform):
    """``HPUModelRunner.load_model`` sets ``current_platform.device_type`` to the load device (``"cpu"`` for an
    INC quantization run) while the model is built; ``device_name`` still says ``hpu``."""
    monkeypatch.setattr(hpu_platform, "device_type", "cpu")
    with built_layer(tmp_path, monkeypatch, QsaMode.SPARSE) as layer:
        assert isinstance(layer.sparse_attn, HpuQsaSparseAttention)


def test_sparse_mode_without_a_wrapper_is_an_error_on_hpu(tmp_path, monkeypatch, hpu_platform):
    """An unusable wrapper module must stop the build on HPU instead of leaving ``sparse_attn`` as ``None``."""
    monkeypatch.setitem(sys.modules, WRAPPER_MODULE, None)  # ``from <module> import ...`` raises ImportError
    with pytest.raises(RuntimeError, match="sparse"), built_layer(tmp_path, monkeypatch, QsaMode.SPARSE):
        pass


def test_dense_mode_builds_no_wrapper_on_hpu(tmp_path, monkeypatch, hpu_platform):
    with built_layer(tmp_path, monkeypatch, QsaMode.DENSE) as layer:
        assert layer.qsa_mode is QsaMode.DENSE
        assert layer.sparse_attn is None


def test_sparse_mode_keeps_the_stand_in_seam_on_cpu(tmp_path, monkeypatch):
    """Off HPU (the CPU tests) the seam stays ``None`` for the harness to fill, with no error."""
    assert qsa_module.current_platform.device_type != "hpu"
    with built_layer(tmp_path, monkeypatch, QsaMode.SPARSE) as layer:
        assert layer.qsa_mode is QsaMode.SPARSE
        assert layer.sparse_attn is None
