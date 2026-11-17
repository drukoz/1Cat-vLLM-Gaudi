# SPDX-License-Identifier: Apache-2.0
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


class _HabanaStubFinder:
    """Answer every habana_frameworks import with a MagicMock module.

    vllm_gaudi.platform imports habana_frameworks at module level; only
    the pure-python helpers (is_flash_next_model,
    _configure_recurrent_state) are exercised here, so the runtime modules
    are stubbed to let the import succeed on CPU.
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

from vllm_gaudi.platform import _configure_recurrent_state, is_flash_next_model  # noqa: E402
from vllm_gaudi.v1.core.recurrent_state_pool import ENV_CACHE_SLOTS, ENV_MODE, ENV_SLOTS  # noqa: E402

POOL_ENV_VARS = ("VLLM_COMPACT_GDN", ENV_MODE, ENV_SLOTS, ENV_CACHE_SLOTS)


def _config(architectures, architecture=None):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=architectures), architecture=architecture)


def test_flash_next_arch_in_hf_config():
    assert is_flash_next_model(_config(["Qwen4ExpForCausalLM"]))


def test_flash_next_arch_attribute():
    assert is_flash_next_model(_config([], architecture="Qwen4ExpForConditionalGeneration"))


def test_other_models_false():
    assert not is_flash_next_model(_config(["Qwen3_5MoeForConditionalGeneration"]))
    assert not is_flash_next_model(None)


def _recurrent_config(prefix_caching=True, is_hybrid=True, max_num_seqs=32):
    """A Flash-Next-shaped config: 12 attention + 36 GDN layers, one PLE layer."""
    layer_counts = {"attention": 12, "linear_attention": 36, "gdn_attention": 0}
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=is_hybrid,
            hf_text_config=SimpleNamespace(ple_layer_ids=[2]),
            get_num_layers_by_block_type=lambda pc, bt: layer_counts[bt],
        ),
        parallel_config=SimpleNamespace(),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _clear_pool_env(monkeypatch):
    for name in POOL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_compact_gdn_with_prefix_caching_raises(monkeypatch):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    with pytest.raises(ValueError, match="does not support prefix caching"):
        _configure_recurrent_state(_recurrent_config())


def test_compact_gdn_defaults_to_zero_under_prefix_caching(monkeypatch):
    _clear_pool_env(monkeypatch)
    _configure_recurrent_state(_recurrent_config())
    assert os.environ["VLLM_COMPACT_GDN"] == "0"


def test_pool_mode_resolves_slots(monkeypatch):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "1")
    _configure_recurrent_state(_recurrent_config())
    # groups = ceil(36 / 12) + 1 = 4; 32 * 4 * 3 + 512 = 896
    assert os.environ[ENV_SLOTS] == "896"
    assert os.environ["VLLM_COMPACT_GDN"] == "0"


def test_pool_mode_ignored_without_prefix_caching(monkeypatch):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "1")
    _configure_recurrent_state(_recurrent_config(prefix_caching=False))
    assert ENV_SLOTS not in os.environ


def test_pool_mode_keeps_preset_slots(monkeypatch):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "1")
    monkeypatch.setenv(ENV_SLOTS, "100")
    _configure_recurrent_state(_recurrent_config())
    assert os.environ[ENV_SLOTS] == "100"


def test_non_hybrid_config_untouched(monkeypatch):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "1")
    _configure_recurrent_state(_recurrent_config(is_hybrid=False))
    assert ENV_SLOTS not in os.environ
    assert "VLLM_COMPACT_GDN" not in os.environ
