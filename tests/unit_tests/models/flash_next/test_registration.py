# SPDX-License-Identifier: Apache-2.0
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


class _HabanaStubFinder:
    """Answer every habana_frameworks import with a MagicMock module.

    vllm_gaudi.platform imports habana_frameworks at module level; only
    is_qwen3_5_hybrid_model (pure-python) is exercised here, so the runtime
    modules are stubbed to let the import succeed on CPU.
    """

    class _Loader:

        def create_module(self, spec):
            return MagicMock(name=spec.name)

        def exec_module(self, module):
            pass

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "habana_frameworks":
            return importlib.machinery.ModuleSpec(fullname, self._Loader(), is_package=True)
        return None


if not importlib.util.find_spec("habana_frameworks"):
    sys.meta_path.insert(0, _HabanaStubFinder())

# The qwen4_exp package imports vllm_gaudi.models.qwen3_5 and qwen3_next before
# upstream's model module, which is the order the GDN/MoE replacements need.
import vllm_gaudi  # noqa: E402
import vllm_gaudi.models  # noqa: E402
import vllm_gaudi.models.qwen3_5 as qwen3_5  # noqa: E402
import vllm_gaudi.models.qwen3_next as qwen3_next  # noqa: E402
import vllm_gaudi.models.qwen4_exp as flash_next  # noqa: E402
from vllm_gaudi.models.qwen4_exp import HpuQwen4ExpForCausalLM, HpuQwen4ExpForConditionalGeneration  # noqa: E402
from vllm_gaudi.platform import is_qwen3_5_hybrid_model  # noqa: E402
import vllm.models.qwen4_exp.nvidia.model as upstream_model  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.model_loader.utils import get_model_architecture  # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

TINY_CONFIG = Path(__file__).resolve().parent / "tiny_config.json"

HPU_CLASSES = {
    "Qwen4ExpForConditionalGeneration": (HpuQwen4ExpForConditionalGeneration,
                                         upstream_model.Qwen4ExpForConditionalGeneration),
    "Qwen4ExpForCausalLM": (HpuQwen4ExpForCausalLM, upstream_model.Qwen4ExpForCausalLM),
}


@pytest.fixture(scope="module")
def flash_next_vllm_config(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("flash_next_tiny")
    shutil.copy(TINY_CONFIG, model_dir / "config.json")
    args = EngineArgs(model=str(model_dir),
                      tokenizer=str(model_dir),
                      skip_tokenizer_init=True,
                      dtype="bfloat16",
                      trust_remote_code=True,
                      enable_prefix_caching=False,
                      max_model_len=512,
                      max_num_seqs=4,
                      enforce_eager=True,
                      language_model_only=True,
                      load_format="dummy")
    return args.create_engine_config()


def _register_flash_next_archs():
    # The full vllm_gaudi.models.register_model() imports HPU-only modules
    # (habana_frameworks, vllm.models.deepseek_v4_1) that are absent on CPU,
    # so drive the hook register_model() calls at its end. Its wiring is
    # asserted from source in test_registry_resolves_to_hpu_subclasses.
    flash_next.register()


def test_registry_resolves_to_hpu_subclasses():
    source = inspect.getsource(vllm_gaudi.models.register_model)
    assert "_register_flash_next" in source
    _register_flash_next_archs()
    for arch, (hpu_cls, upstream_cls) in HPU_CLASSES.items():
        loaded = ModelRegistry._try_load_model_cls(arch)
        assert loaded is hpu_cls, f"{arch} resolved to {loaded!r}"
        assert issubclass(hpu_cls, upstream_cls)


def test_gdn_replacement_reaches_upstream_model():
    assert upstream_model.QwenGatedDeltaNetAttention is qwen3_5.HPUGatedDeltaNetAttention
    assert upstream_model.Qwen4ExpSparseMoeBlock.forward is qwen3_next._hpu_qwen3next_sparse_moe_forward


def test_hybrid_allow_list(flash_next_vllm_config):
    model_config = flash_next_vllm_config.model_config
    assert model_config.is_hybrid
    assert model_config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert is_qwen3_5_hybrid_model(model_config) is True
    _register_flash_next_archs()
    assert get_model_architecture(model_config)[0] is HpuQwen4ExpForConditionalGeneration


def test_runner_allow_list_text():
    runner_path = Path(vllm_gaudi.__file__).resolve().parent / "v1" / "worker" / "hpu_model_runner.py"
    text = runner_path.read_text()
    start = text.index("mamba_like_arch = [")
    list_text = text[start:text.index("]", start)]
    assert "Qwen4ExpForConditionalGeneration" in list_text
    assert "Qwen4ExpForCausalLM" in list_text
