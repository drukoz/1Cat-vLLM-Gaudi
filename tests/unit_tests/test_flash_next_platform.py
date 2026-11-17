# SPDX-License-Identifier: Apache-2.0
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


class _HabanaStubFinder:
    """Answer every habana_frameworks import with a MagicMock module.

    vllm_gaudi.platform imports habana_frameworks at module level; only
    is_flash_next_model (pure-python) is exercised here, so the runtime
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

from vllm_gaudi.platform import is_flash_next_model  # noqa: E402


def _config(architectures, architecture=None):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=architectures), architecture=architecture)


def test_flash_next_arch_in_hf_config():
    assert is_flash_next_model(_config(["Qwen4ExpForCausalLM"]))


def test_flash_next_arch_attribute():
    assert is_flash_next_model(_config([], architecture="Qwen4ExpForConditionalGeneration"))


def test_other_models_false():
    assert not is_flash_next_model(_config(["Qwen3_5MoeForConditionalGeneration"]))
    assert not is_flash_next_model(None)
