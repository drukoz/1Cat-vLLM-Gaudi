# SPDX-License-Identifier: Apache-2.0
"""HPUModelRunner._has_recurrent_state_caches must flag Mamba/GDN/short-conv groups."""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import torch
from types import SimpleNamespace

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)


def _mamba_spec() -> MambaSpec:
    # The Flash-Next PLE short-conv spec
    return MambaSpec(block_size=128, shapes=((9, 512), ), dtypes=(torch.bfloat16, ))


def _kv_cache_config(*specs) -> KVCacheConfig:
    groups = [KVCacheGroupSpec(layer_names=[f"layer_{i}"], kv_cache_spec=spec) for i, spec in enumerate(specs)]
    return KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=groups)


def test_no_kv_cache_config_is_not_recurrent():
    assert HPUModelRunner._has_recurrent_state_caches(SimpleNamespace()) is False


def test_full_attention_only_is_not_recurrent():
    cfg = _kv_cache_config(_full_attention_spec())
    runner = SimpleNamespace(kv_cache_config=cfg)
    assert HPUModelRunner._has_recurrent_state_caches(runner) is False


def test_mamba_group_is_recurrent():
    cfg = _kv_cache_config(_full_attention_spec(), _mamba_spec())
    runner = SimpleNamespace(kv_cache_config=cfg)
    assert HPUModelRunner._has_recurrent_state_caches(runner) is True
