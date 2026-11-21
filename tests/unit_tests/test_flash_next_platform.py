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

from vllm_gaudi.platform import (_configure_recurrent_state, _validate_flash_next_speculative_config,  # noqa: E402
                                 is_flash_next_model)
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
    # setenv first so monkeypatch records the original value: the hook writes
    # os.environ directly, and delenv on an absent variable records nothing.
    for name in POOL_ENV_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


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


@pytest.mark.parametrize("k,slots", [(3, "1280"), (4, "1408"), (5, "1536")])
def test_pool_mode_resolves_slots_with_speculation(monkeypatch, k, slots):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "1")
    config = _recurrent_config()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=k)
    # groups = ceil(36 / 12) + 1 = 4; 32 * 4 * (3 + k) + 512: 1280, 1408, 1536
    _configure_recurrent_state(config)
    assert os.environ[ENV_SLOTS] == slots


def test_pool_mode_speculative_config_none(monkeypatch):
    _clear_pool_env(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "1")
    config = _recurrent_config()
    config.speculative_config = None
    _configure_recurrent_state(config)
    assert os.environ[ENV_SLOTS] == "896"


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


def _spec_config(method="mtp", draft_archs=("Qwen4ExpMTP", ), prefix_caching=False, k=3,
                 target_archs=("Qwen4ExpForCausalLM", )):
    """A Flash-Next-shaped vllm_config with a speculative config of the given shape."""
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=list(target_archs)), architecture=None),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        speculative_config=SimpleNamespace(
            method=method,
            num_speculative_tokens=k,
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=list(draft_archs))),
        ),
    )


def test_spec_supported_config_passes(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    _validate_flash_next_speculative_config(_spec_config())


def test_spec_eagle_method_raises(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    with pytest.raises(ValueError, match="its own MTP layer or n-gram"):
        _validate_flash_next_speculative_config(_spec_config(method="eagle"))


def test_spec_wrong_draft_arch_raises(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    with pytest.raises(ValueError, match="its own MTP layer or n-gram"):
        _validate_flash_next_speculative_config(_spec_config(draft_archs=["Qwen3_5MoeMTP"]))


def test_spec_ngram_passes(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    _validate_flash_next_speculative_config(_spec_config(method="ngram", draft_archs=()))


def test_spec_ngram_prefix_caching_passes(monkeypatch):
    """n-gram lookup with prefix caching runs on the align-mode lattices (the state pool, compact GDN off)."""
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0")
    _validate_flash_next_speculative_config(_spec_config(method="ngram", draft_archs=(), prefix_caching=True))
    monkeypatch.delenv("VLLM_COMPACT_GDN", raising=False)
    _validate_flash_next_speculative_config(_spec_config(method="ngram", draft_archs=(), prefix_caching=True))


def test_spec_ngram_too_many_tokens_raises(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    with pytest.raises(ValueError, match="n-gram lookup on HPU runs 1 to 8 speculative tokens"):
        _validate_flash_next_speculative_config(_spec_config(method="ngram", draft_archs=(), k=9))


def test_spec_ngram_compact_gdn_off_raises(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0")
    with pytest.raises(ValueError, match="needs compact GDN state"):
        _validate_flash_next_speculative_config(_spec_config(method="ngram", draft_archs=()))


def test_spec_prefix_caching_passes_with_compact_gdn_off(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0")
    _validate_flash_next_speculative_config(_spec_config(prefix_caching=True))


def test_spec_prefix_caching_passes_with_compact_gdn_unset(monkeypatch):
    monkeypatch.delenv("VLLM_COMPACT_GDN", raising=False)
    _validate_flash_next_speculative_config(_spec_config(prefix_caching=True))


def test_spec_compact_gdn_off_raises(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0")
    with pytest.raises(ValueError, match="needs compact GDN state"):
        _validate_flash_next_speculative_config(_spec_config())


def test_spec_too_many_tokens_raises(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    with pytest.raises(ValueError, match="1 to 5 speculative tokens"):
        _validate_flash_next_speculative_config(_spec_config(k=7))


SUPPORTED_SIZES = (1, 2, 3, 4, 5)  # MTP
NGRAM_SUPPORTED_SIZES = tuple(range(1, 9))
SERVER_KINDS = [("mtp", ("Qwen4ExpMTP", ), False), ("mtp", ("Qwen4ExpMTP", ), True), ("ngram", (), False),
                ("ngram", (), True)]
SERVER_KIND_IDS = ["mtp", "mtp_prefix_caching", "ngram", "ngram_prefix_caching"]
# (method, draft architectures, prefix caching, k) of every served size and of the sizes just outside the range
SERVED = [(m, a, pc, k) for (m, a, pc) in SERVER_KINDS
          for k in (NGRAM_SUPPORTED_SIZES if m == "ngram" else SUPPORTED_SIZES)]
REFUSED = [(m, a, pc, k) for (m, a, pc) in SERVER_KINDS for k in ((0, 9, 10, 16) if m == "ngram" else (0, 6, 7))]


def _validate_size(monkeypatch, method, draft_archs, prefix_caching, k):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0" if prefix_caching else "1")
    _validate_flash_next_speculative_config(
        _spec_config(method=method, draft_archs=draft_archs, prefix_caching=prefix_caching, k=k))


@pytest.mark.parametrize("method,draft_archs,prefix_caching,k", SERVED)
def test_spec_supported_sizes_pass(monkeypatch, method, draft_archs, prefix_caching, k):
    _validate_size(monkeypatch, method, draft_archs, prefix_caching, k)


@pytest.mark.parametrize("method,draft_archs,prefix_caching,k", REFUSED)
def test_spec_sizes_outside_the_supported_range_raise(monkeypatch, method, draft_archs, prefix_caching, k):
    limit = 8 if method == "ngram" else 5
    with pytest.raises(ValueError, match=f"1 to {limit} speculative tokens"):
        _validate_size(monkeypatch, method, draft_archs, prefix_caching, k)


def test_spec_bound_against_the_sizes_the_cpu_tests_run(monkeypatch):
    """The validator accepts 1 to 5: 5 is the largest lattice the fused verify kernel takes (1 + k = 6 nodes),
    and the CPU tests run 3, 4 and 5; a size they call served must be accepted."""
    from flashinfer_gaudi.gdn_core_verify import GDN_CORE_VERIFY_MAX_NODES
    from tests.unit_tests.flash_next_spec_sizes import PROVEN_NOT_SERVED, RECIPE_K, SPEC_SIZES

    def accepted(k):
        try:
            _validate_size(monkeypatch, "mtp", ("Qwen4ExpMTP", ), False, k)
        except ValueError:
            return False
        return True

    assert [k for k in range(0, 12) if accepted(k)] == list(SUPPORTED_SIZES)
    assert [k for k in SPEC_SIZES if accepted(k)] == [k for k in SPEC_SIZES if k not in PROVEN_NOT_SERVED]
    assert max(SPEC_SIZES) == GDN_CORE_VERIFY_MAX_NODES - 1 == max(SUPPORTED_SIZES)
    assert PROVEN_NOT_SERVED == ()
    assert RECIPE_K == 3 and RECIPE_K in SUPPORTED_SIZES


def test_spec_non_flash_next_model_untouched(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0")
    _validate_flash_next_speculative_config(
        _spec_config(method="eagle",
                     draft_archs=["DFlash2DraftModel"],
                     prefix_caching=True,
                     k=7,
                     target_archs=["Qwen3_5MoeForConditionalGeneration"]))


def test_spec_none_passes(monkeypatch):
    monkeypatch.setenv("VLLM_COMPACT_GDN", "0")
    config = _spec_config()
    config.speculative_config = None
    _validate_flash_next_speculative_config(config)


# ---- sampled requests on a speculative server (idea-sweep I07): refused unless VLLM_HPU_SPEC_SAMPLED_VERIFY=1


@pytest.fixture
def speculative_server(monkeypatch):
    import vllm_gaudi.platform as platform_module

    monkeypatch.setattr(platform_module, "_SPEC_GREEDY_ONLY", False)
    monkeypatch.setattr(platform_module, "_SPECULATIVE_SERVER", False)
    monkeypatch.setenv("VLLM_COMPACT_GDN", "1")
    return platform_module


def _params(temperature):
    from vllm import SamplingParams

    return SamplingParams(temperature=temperature, max_tokens=4)


def test_a_sampled_request_on_a_speculative_server_is_refused(speculative_server, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_SPEC_SAMPLED_VERIFY", "0")
    _validate_flash_next_speculative_config(_spec_config())
    with pytest.raises(ValueError, match="VLLM_HPU_SPEC_SAMPLED_VERIFY=1"):
        speculative_server.HpuPlatform.validate_request(None, _params(1.0))
    speculative_server.HpuPlatform.validate_request(None, _params(0.0))  # greedy: served


@pytest.mark.parametrize("mode", ["1", "2"])  # 2: over candidates, the exact path for the steps it refuses
def test_a_sampled_request_is_served_with_the_flag(speculative_server, monkeypatch, mode):
    monkeypatch.setenv("VLLM_HPU_SPEC_SAMPLED_VERIFY", mode)
    _validate_flash_next_speculative_config(_spec_config())
    speculative_server.HpuPlatform.validate_request(None, _params(0.7))


def test_a_server_without_speculation_serves_sampled_requests(speculative_server, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_SPEC_SAMPLED_VERIFY", "0")
    config = _spec_config()
    config.speculative_config = None
    _validate_flash_next_speculative_config(config)
    speculative_server.HpuPlatform.validate_request(None, _params(1.0))


@pytest.mark.parametrize("field,value", [("presence_penalty", 0.5), ("frequency_penalty", 0.2),
                                         ("repetition_penalty", 1.1), ("allowed_token_ids", [1, 2])])
def test_penalties_and_token_masks_are_refused_on_a_speculative_server(speculative_server, monkeypatch, field, value):
    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_HPU_SPEC_SAMPLED_VERIFY", "1")
    _validate_flash_next_speculative_config(_spec_config())
    with pytest.raises(ValueError, match=field):
        speculative_server.HpuPlatform.validate_request(None, SamplingParams(temperature=0.0, **{field: value}))
    speculative_server.HpuPlatform.validate_request(None, SamplingParams(temperature=0.0))


def test_penalties_are_served_without_speculation(speculative_server):
    from vllm import SamplingParams

    speculative_server.HpuPlatform.validate_request(None, SamplingParams(temperature=0.0, repetition_penalty=1.1))
