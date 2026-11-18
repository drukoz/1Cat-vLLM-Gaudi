# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the recurrent-state-pool contract (M1-T1)."""

import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec  # noqa: E402

from vllm_gaudi.v1.core.recurrent_state_pool import (  # noqa: E402
    ENV_MODE, ENV_SLOTS, LIVE_SLOTS_PER_REQUEST, RecurrentStatePoolMode, is_recurrent_spec, num_recurrent_groups,
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


if __name__ == "__main__":
    pytest.main([__file__])
