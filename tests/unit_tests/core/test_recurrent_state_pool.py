# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the recurrent-state-pool contract (M1-T1)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec  # noqa: E402

import vllm.v1.core.kv_cache_utils as kv_cache_utils  # noqa: E402

from vllm_gaudi.v1.core.recurrent_state_pool import (  # noqa: E402
    ENV_MODE, ENV_SLOTS, LIVE_SLOTS_PER_REQUEST, RecurrentStatePoolMode, hpu_kv_cache_groups,
    install_hpu_kv_cache_grouping, is_recurrent_spec, minor_recurrent_layers, num_recurrent_groups,
    pool_mode_from_env, pool_pad_block_id, pool_slot_bytes, pool_slots_from_env, pool_tensor_slots,
    recurrent_positions, resolve_pool_slots)

GDN_PAGE_BYTES = 3 * 1280 * 2 + 6 * 128 * 128 * 4  # bf16 state + fp32 state = 400896
PLE_PAGE_BYTES = 9 * 10240 * 2  # bf16 short-conv state = 184320


def _flash_next_specs() -> dict[str, object]:
    """The Flash-Next shape: 12 attention layers, 36 GDN layers, 1 PLE short-conv layer."""
    attn = FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    gdn = MambaSpec(block_size=512,
                    shapes=((3, 1280), (6, 128, 128)),
                    dtypes=(torch.bfloat16, torch.float32),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="align",
                    page_size_padded=524288)
    ple = MambaSpec(block_size=512,
                    shapes=((9, 10240), ),
                    dtypes=(torch.bfloat16, ),
                    mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                    mamba_cache_mode="align",
                    page_size_padded=524288)
    specs: dict[str, object] = {}
    for i in range(48):
        specs[f"layers.{i}.attn"] = attn if i % 4 == 3 else gdn
    specs["layers.2.ple"] = ple
    return specs


def _qwen35_27b_specs() -> dict[str, object]:
    """The 27B shape: 16 attention + 48 GDN layers, no PLE."""
    attn = FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    gdn = MambaSpec(block_size=512,
                    shapes=((3, 1280), (6, 128, 128)),
                    dtypes=(torch.bfloat16, torch.float32),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="align",
                    page_size_padded=524288)
    specs: dict[str, object] = {}
    for i in range(64):
        specs[f"layers.{i}.attn"] = attn if i % 4 == 3 else gdn
    return specs


def test_flash_next_positions_and_groups():
    specs = _flash_next_specs()
    positions = recurrent_positions(specs)
    gdn_specs = [
        s for s in specs.values() if isinstance(s, MambaSpec) and s.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
    ]
    ple_specs = [
        s for s in specs.values() if isinstance(s, MambaSpec) and s.mamba_type == MambaAttentionBackendEnum.SHORT_CONV
    ]
    assert (len(gdn_specs), len(ple_specs)) == (36, 1)
    assert positions == {gdn_specs[0]: 12, ple_specs[0]: 1}
    assert num_recurrent_groups(specs) == 4


def test_flash_next_slot_bytes():
    specs = _flash_next_specs()
    assert pool_slot_bytes(specs) == 12 * GDN_PAGE_BYTES + PLE_PAGE_BYTES
    assert GDN_PAGE_BYTES == 400896 and PLE_PAGE_BYTES == 184320


def test_qwen35_27b_positions_and_groups():
    specs = _qwen35_27b_specs()
    positions = recurrent_positions(specs)
    gdn = next(s for s in specs.values() if is_recurrent_spec(s))
    assert positions == {gdn: 16}
    assert num_recurrent_groups(specs) == 3
    assert pool_slot_bytes(specs) == 16 * GDN_PAGE_BYTES


def test_slot_arithmetic():
    assert LIVE_SLOTS_PER_REQUEST == 3
    assert resolve_pool_slots(32, 4, 512) == 896
    assert pool_tensor_slots(896) == 898
    assert pool_pad_block_id(896) == 897


def test_resolve_pool_slots_with_speculation():
    # Default and explicit 0 keep the non-speculative formula.
    assert resolve_pool_slots(32, 4, 512) == 896
    assert resolve_pool_slots(32, 4, 512, 0) == 896
    # k=3: every live request owns LIVE_SLOTS_PER_REQUEST + 3 slots per group.
    assert resolve_pool_slots(32, 4, 512, 3) == 32 * 4 * 6 + 512
    assert resolve_pool_slots(32, 4, 512, 3) == 1280
    assert resolve_pool_slots(1, 1, 0, 4) == LIVE_SLOTS_PER_REQUEST + 4


def test_is_recurrent_spec():
    specs = _flash_next_specs()
    attn = next(s for s in specs.values() if isinstance(s, FullAttentionSpec))
    gdn = next(s for s in specs.values() if is_recurrent_spec(s))
    assert not is_recurrent_spec(attn)
    assert is_recurrent_spec(gdn)


def test_mode_enum_values():
    assert RecurrentStatePoolMode.OFF == 0
    assert RecurrentStatePoolMode.POOL == 1
    assert isinstance(RecurrentStatePoolMode.OFF, int)


def test_pool_from_env_off(monkeypatch):
    monkeypatch.delenv(ENV_MODE, raising=False)
    monkeypatch.delenv(ENV_SLOTS, raising=False)
    assert pool_slots_from_env() == 0
    assert pool_mode_from_env() == RecurrentStatePoolMode.OFF

    monkeypatch.setenv(ENV_SLOTS, "896")
    assert pool_slots_from_env() == 0  # slots set but mode unset: pool is off

    monkeypatch.setenv(ENV_MODE, "0")
    assert pool_slots_from_env() == 0
    assert pool_mode_from_env() == RecurrentStatePoolMode.OFF


def test_pool_from_env_pool_mode(monkeypatch):
    monkeypatch.setenv(ENV_MODE, "1")
    monkeypatch.setenv(ENV_SLOTS, "896")
    assert pool_mode_from_env() == RecurrentStatePoolMode.POOL
    assert pool_slots_from_env() == 896

    monkeypatch.delenv(ENV_SLOTS, raising=False)
    assert pool_slots_from_env() == 0


def test_pool_from_env_reads_at_call_time(monkeypatch):
    monkeypatch.delenv(ENV_MODE, raising=False)
    assert pool_slots_from_env() == 0
    monkeypatch.setenv(ENV_MODE, "1")
    monkeypatch.setenv(ENV_SLOTS, "64")
    assert pool_slots_from_env() == 64


def _engine_groups():
    """vLLM's own grouping of same-page-size layers (never the wrapper another test may have installed)."""
    installed = kv_cache_utils._get_kv_cache_groups_uniform_page_size
    return getattr(installed, "__wrapped__", installed)


def _group_sizes(groups, kind) -> list[int]:
    return [len(group.layer_names) for group in groups if isinstance(group.kv_cache_spec, kind)]


def test_engine_splits_flash_next_attention_into_single_layer_groups():
    """The cause of the 12x KV cache: one PLE layer makes vLLM's group size 1."""
    groups = _engine_groups()(_flash_next_specs())
    assert _group_sizes(groups, FullAttentionSpec) == [1] * 12
    assert _group_sizes(groups, MambaSpec) == [1] * 37


def test_flash_next_groups_keep_attention_whole():
    specs = _flash_next_specs()
    ple = specs["layers.2.ple"]
    assert minor_recurrent_layers(specs) == {ple: ["layers.2.ple"]}
    groups = hpu_kv_cache_groups(specs, _engine_groups())
    # The engine's order: by the first layer of each spec (GDN at layer 0, attention at layer 3, PLE last here).
    assert [len(group.layer_names) for group in groups] == [12, 12, 12, 12, 1]
    assert [type(group.kv_cache_spec) for group in groups] == [MambaSpec] * 3 + [FullAttentionSpec, MambaSpec]
    assert groups[3].layer_names == [f"layers.{i}.attn" for i in range(3, 48, 4)]
    assert groups[4].layer_names == ["layers.2.ple"] and groups[4].kv_cache_spec == ple
    gdn_layers = [name for group in groups[:3] for name in group.layer_names]
    assert sorted(gdn_layers) == sorted(name for name, spec in specs.items() if is_recurrent_spec(spec) and spec != ple)
    # What the platform hook and the worker assume: 4 recurrent groups, 12 GDN positions and 1 PLE position.
    assert len(_group_sizes(groups, MambaSpec)) == num_recurrent_groups(specs) == 4
    assert {group.kv_cache_spec: len(group.layer_names) for group in groups if is_recurrent_spec(group.kv_cache_spec)} \
        == recurrent_positions(specs)


def test_groups_without_a_minor_recurrent_spec_are_the_engines():
    engine_groups = _engine_groups()
    tiny = {"layers.0.linear_attn": None, "layers.1.ple": None, "layers.1.linear_attn": None,
            "layers.2.linear_attn": None, "layers.3.self_attn.attn": None}
    flash_next = _flash_next_specs()
    attn = flash_next["layers.3.attn"]
    gdn = flash_next["layers.0.attn"]
    for name in tiny:
        tiny[name] = attn if name.endswith(".attn") else flash_next["layers.2.ple"] if name.endswith(".ple") else gdn
    for specs in (_qwen35_27b_specs(), tiny, {"layers.0.attn": attn, "layers.1.attn": attn}, {"layers.0.mixer": gdn}):
        assert minor_recurrent_layers(specs) == {}
        assert hpu_kv_cache_groups(specs, engine_groups) == engine_groups(specs)
    assert _group_sizes(engine_groups(_qwen35_27b_specs()), FullAttentionSpec) == [16]
    assert _group_sizes(engine_groups(tiny), FullAttentionSpec) == [1]


def test_install_wraps_the_engine_grouping_once(monkeypatch):
    engine_groups = _engine_groups()
    monkeypatch.setattr(kv_cache_utils, "_get_kv_cache_groups_uniform_page_size", engine_groups)
    assert install_hpu_kv_cache_grouping()
    installed = kv_cache_utils._get_kv_cache_groups_uniform_page_size
    assert installed is not engine_groups and installed.__wrapped__ is engine_groups
    assert install_hpu_kv_cache_grouping()
    assert kv_cache_utils._get_kv_cache_groups_uniform_page_size is installed
    specs = _flash_next_specs()
    assert _group_sizes(installed(specs), FullAttentionSpec) == [12]
    assert installed(_qwen35_27b_specs()) == engine_groups(_qwen35_27b_specs())


def test_install_reports_a_missing_engine_function(monkeypatch):
    monkeypatch.delattr(kv_cache_utils, "_get_kv_cache_groups_uniform_page_size")
    assert not install_hpu_kv_cache_grouping()


if __name__ == "__main__":
    pytest.main([__file__])
