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

from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup, size_id  # noqa: E402

GDN_PAGE_BYTES = 3 * 1280 * 2 + 6 * 128 * 128 * 4  # bf16 state + fp32 state = 400896
PLE_PAGE_BYTES = 9 * 10240 * 2  # bf16 short-conv state = 184320
ATTN_PAGE_BYTES = 524288  # 512 tokens of one KV head pair at TP8 in bf16: the page every group is padded to


def _flash_next_specs(k: int = 0) -> dict[str, object]:
    """The Flash-Next shape at TP8: 12 attention layers, 36 GDN layers, 1 PLE short-conv layer.

    ``k`` speculative tokens add ``k`` rows to both conv states (upstream's ``conv_kernel - 1 + num_spec``) and
    ``k`` scratch blocks per request to both recurrent specs.
    """
    attn = FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    gdn = MambaSpec(block_size=512,
                    shapes=((3 + k, 1280), (6, 128, 128)),
                    dtypes=(torch.bfloat16, torch.float32),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="align",
                    page_size_padded=ATTN_PAGE_BYTES,
                    num_speculative_blocks=k)
    ple = MambaSpec(block_size=512,
                    shapes=((9 + k, 10240), ),
                    dtypes=(torch.bfloat16, ),
                    mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                    mamba_cache_mode="align",
                    page_size_padded=ATTN_PAGE_BYTES,
                    num_speculative_blocks=k)
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


@pytest.fixture(params=SPEC_SIZES, ids=size_id)
def spec_k(request) -> int:
    """Each speculative size k the Flash-Next tests run at, after flash_next_spec_sizes' startup check."""
    require_startup(request.param, prefix_caching=True)
    return request.param


def test_pool_grows_with_the_speculative_size(spec_k):
    """Slots and bytes per slot of the pool for 32 and 16 sequences with k speculative tokens (Flash-Next, TP8).

    Slots: ``max_num_seqs * 4 groups * (3 + k) + 512``. Bytes per slot: 12 GDN positions of
    ``(3 + k) * 1280 * 2 + 393216`` plus the PLE state ``(9 + k) * 10240 * 2``; both states still fit the
    524288-byte page the attention block sets, so the scheduler block stays 512 tokens.
    """
    specs = _flash_next_specs(spec_k)
    assert num_recurrent_groups(specs) == 4
    gdn_bytes = (3 + spec_k) * 1280 * 2 + 6 * 128 * 128 * 4
    ple_bytes = (9 + spec_k) * 10240 * 2
    assert pool_slot_bytes(specs) == 12 * gdn_bytes + ple_bytes
    assert max(gdn_bytes, ple_bytes) <= ATTN_PAGE_BYTES
    expected = {3: (1280, 896, 5148672), 4: (1408, 960, 5199872), 5: (1536, 1024, 5251072)}[spec_k]
    assert resolve_pool_slots(32, 4, 512, spec_k) == 32 * 4 * (LIVE_SLOTS_PER_REQUEST + spec_k) + 512 == expected[0]
    assert resolve_pool_slots(16, 4, 512, spec_k) == expected[1]
    assert pool_slot_bytes(specs) == expected[2]
    assert pool_tensor_slots(expected[0]) == expected[0] + 2
    # The k scratch blocks are per request and per group: every live request can hold its lattice at once.
    live = (resolve_pool_slots(32, 4, 512, spec_k) - 512) // (32 * 4)
    assert live - LIVE_SLOTS_PER_REQUEST == spec_k


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


@pytest.mark.parametrize("k", [0] + list(SPEC_SIZES))
def test_flash_next_mtp_positions_are_the_engines_group_sizes(k):
    # The MTP draft adds a 13th attention layer: the engine still makes 3 GDN groups, now of 12 layers each
    # (layers[i::3] of 36), so the runner allocates 12 GDN state tensors; one more position would charge every
    # pool slot one GDN page it never gets (I37 (f): 0.49 GiB per card at k = 3, 32 sequences).
    specs = _flash_next_specs(k)
    specs["mtp.layers.48.attn"] = specs["layers.3.attn"]
    groups = hpu_kv_cache_groups(specs, _engine_groups())
    assert _group_sizes(groups, FullAttentionSpec) == [13]
    per_group = {}
    for group in groups:
        if is_recurrent_spec(group.kv_cache_spec):
            per_group[group.kv_cache_spec] = max(per_group.get(group.kv_cache_spec, 0), len(group.layer_names))
    assert recurrent_positions(specs) == per_group
    gdn = specs["layers.0.attn"]
    assert per_group[gdn] == 12
    assert num_recurrent_groups(specs) == len(_group_sizes(groups, MambaSpec)) == 4


@pytest.mark.parametrize("n_attn,n_gdn", [(1, 3), (2, 3), (12, 36), (13, 36), (16, 48), (17, 48), (13, 18), (4, 5)])
def test_recurrent_positions_mirror_the_engine_grouping(n_attn, n_gdn):
    flash_next = _flash_next_specs()
    attn, gdn = flash_next["layers.3.attn"], flash_next["layers.0.attn"]
    specs = {f"g.{i}": gdn for i in range(n_gdn)}
    specs.update({f"a.{i}": attn for i in range(n_attn)})
    groups = _engine_groups()(specs)
    gdn_sizes = [len(group.layer_names) for group in groups if group.kv_cache_spec == gdn]
    assert recurrent_positions(specs) == {gdn: max(gdn_sizes)}
    assert num_recurrent_groups(specs) == len(gdn_sizes)
