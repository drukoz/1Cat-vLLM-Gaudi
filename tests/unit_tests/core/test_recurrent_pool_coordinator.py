# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the recurrent-state-pool scheduler coordinator (M1-T2).

Drives KVCacheManager like the scheduler does (get_computed_blocks / allocate_slots / free)
on a Flash-Next-shaped config: 1 full-attention group + 3 GDN + 1 short-conv (MambaSpec) groups.
"""
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402


class _HabanaStubFinder:
    """Answer every habana_frameworks import with a MagicMock module (CPU has no habana).

    vllm_gaudi.utils imports habana_frameworks.torch at module level; the factory patch
    installed here is what the tests exercise, so the runtime modules are stubbed.
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

import vllm_gaudi.utils  # noqa: E402, F401  (installs the factory patch)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

from vllm_gaudi.v1.core.recurrent_pool_coordinator import HpuRecurrentPoolCoordinator  # noqa: E402
from vllm_gaudi.v1.core.recurrent_state_pool import ENV_MODE, ENV_SLOTS  # noqa: E402

BLOCK, HASH_BLOCK = 512, 128
NUM_ATTN_BLOCKS = 200
NUM_RECURRENT_GROUPS = 4  # 3 GDN + 1 short-conv


def _build_config() -> KVCacheConfig:
    attn = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    gdn = MambaSpec(block_size=BLOCK,
                    shapes=((3, 1280), (6, 128, 128)),
                    dtypes=(torch.bfloat16, torch.float32),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    mamba_cache_mode="align",
                    page_size_padded=attn.page_size_bytes)
    ple = MambaSpec(block_size=BLOCK,
                    shapes=((9, 10240), ),
                    dtypes=(torch.bfloat16, ),
                    mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
                    mamba_cache_mode="align",
                    page_size_padded=attn.page_size_bytes)
    groups = [KVCacheGroupSpec([f"layers.{i}.attn" for i in range(12)], attn)]
    groups += [KVCacheGroupSpec([f"layers.{g * 12 + i}.linear_attn" for i in range(12)], gdn) for g in range(3)]
    groups.append(KVCacheGroupSpec(["layers.1.ple"], ple))
    return KVCacheConfig(num_blocks=NUM_ATTN_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups)


hash_fn = get_hash_fn_by_name("sha256")
init_none_hash(hash_fn)


def make_request(rid, n_tokens, offset=0):
    params = SamplingParams(max_tokens=17)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=rid,
                   prompt_token_ids=list(range(1 + offset, n_tokens + 1 + offset)),
                   mm_features=None,
                   sampling_params=params,
                   pooling_params=None,
                   block_hasher=get_request_block_hasher(HASH_BLOCK, hash_fn))


def build_manager(monkeypatch, slots=None, mode=None, enable_kv_cache_events=False):
    """Fresh KVCacheManager with the pool env vars set as the test states them."""
    monkeypatch.delenv(ENV_MODE, raising=False)
    monkeypatch.delenv(ENV_SLOTS, raising=False)
    if mode is not None:
        monkeypatch.setenv(ENV_MODE, mode)
    if slots is not None:
        monkeypatch.setenv(ENV_SLOTS, slots)
    return KVCacheManager(kv_cache_config=_build_config(),
                          max_model_len=8192,
                          scheduler_block_size=BLOCK,
                          hash_block_size=HASH_BLOCK,
                          enable_caching=True,
                          enable_kv_cache_events=enable_kv_cache_events)


def schedule_chunks(mgr, req, chunk, num_chunks):
    """Schedule req in `num_chunks` chunks of `chunk` tokens, scheduler-style."""
    blocks, hit, _ = mgr.get_computed_blocks(req)
    out = mgr.allocate_slots(req, chunk, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    assert out is not None
    req.num_computed_tokens += chunk
    for _ in range(num_chunks - 1):
        out = mgr.allocate_slots(req, chunk)
        assert out is not None
        req.num_computed_tokens += chunk
    return out


@pytest.fixture
def mgr_pool24(monkeypatch):
    return build_manager(monkeypatch, mode="1", slots="24")


def test_pool_off_keeps_stock_coordinator(monkeypatch):
    mgr = build_manager(monkeypatch)  # mode unset
    assert type(mgr.coordinator).__name__ == "HybridKVCacheCoordinator"

    mgr = build_manager(monkeypatch, mode="1", slots="0")
    assert type(mgr.coordinator).__name__ == "HybridKVCacheCoordinator"


def test_pool_on_binds_pools(monkeypatch):
    mgr = build_manager(monkeypatch, mode="1", slots="24")
    coordinator = mgr.coordinator
    assert type(coordinator).__name__ == "HpuRecurrentPoolCoordinator"
    assert isinstance(coordinator, HpuRecurrentPoolCoordinator)
    assert mgr.block_pool is coordinator.block_pool
    for gid, manager in enumerate(coordinator.single_type_managers):
        if gid in coordinator.recurrent_group_ids:
            assert manager.block_pool is coordinator.recurrent_pool
            assert manager._null_block is coordinator.recurrent_pool.null_block
        else:
            assert manager.block_pool is coordinator.block_pool
            assert manager._null_block is coordinator.block_pool.null_block
    assert coordinator.recurrent_pool.num_gpu_blocks == 25


def test_pool_on_allocates_ids_per_pool(mgr_pool24):
    mgr = mgr_pool24
    recurrent_free_at_start = mgr.coordinator.recurrent_pool.get_num_free_blocks()
    req = make_request("r1", 1536)
    schedule_chunks(mgr, req, BLOCK, 3)
    ids = mgr.get_block_ids(req.request_id)
    assert len(ids) == 5
    for block_id in ids[0]:
        assert 1 <= block_id < NUM_ATTN_BLOCKS
    for group_ids in ids[1:]:
        for block_id in group_ids:
            assert 0 <= block_id < 25
    # Each chunk costs each recurrent manager at most one block.
    assert mgr.coordinator.recurrent_pool.get_num_free_blocks() >= recurrent_free_at_start - 4 * 3


def test_prefix_cache_hit_from_recurrent_pool(mgr_pool24):
    mgr = mgr_pool24
    req = make_request("r1", 1536)
    schedule_chunks(mgr, req, BLOCK, 3)
    mgr.free(req)

    req2 = make_request("r2", 1536 + 300)
    blocks, hit, _ = mgr.get_computed_blocks(req2)
    assert hit == 1536
    coordinator = mgr.coordinator
    for gid in coordinator.recurrent_group_ids:
        non_null = [b for b in blocks.blocks[gid] if not b.is_null]
        assert len(non_null) == 1
        b = non_null[0]
        assert coordinator.recurrent_pool.blocks[b.block_id] is b


def test_recurrent_pool_exhaustion_preempts(monkeypatch):
    mgr = build_manager(monkeypatch, mode="1", slots="8")
    req1 = make_request("r1", 1024)
    schedule_chunks(mgr, req1, BLOCK, 2)
    # 4 recurrent groups x 2 chunks fill the 8-slot pool; attention has plenty left.
    assert mgr.block_pool.get_num_free_blocks() > 100

    req2 = make_request("r2", 1024, offset=100000)  # distinct prompt: no prefix hit
    blocks, hit, _ = mgr.get_computed_blocks(req2)
    assert hit == 0
    assert mgr.allocate_slots(req2, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is None

    mgr.free(req1)
    out = mgr.allocate_slots(req2, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks)
    assert out is not None


def test_deferred_free_routes_to_both_pools(mgr_pool24):
    mgr = mgr_pool24
    attn_free_at_start = mgr.block_pool.get_num_free_blocks()
    recurrent_free_at_start = mgr.coordinator.recurrent_pool.get_num_free_blocks()
    req = make_request("r1", 1536)
    schedule_chunks(mgr, req, BLOCK, 3)

    blocks = mgr.pop_blocks_for_free(req)
    mgr.block_pool.free_blocks(reversed(blocks))
    assert mgr.block_pool.get_num_free_blocks() == attn_free_at_start
    assert mgr.coordinator.recurrent_pool.get_num_free_blocks() == recurrent_free_at_start


def test_reset_prefix_cache_resets_both_pools(mgr_pool24):
    mgr = mgr_pool24
    req = make_request("r1", 1536)
    schedule_chunks(mgr, req, BLOCK, 3)
    mgr.free(req)

    assert mgr.reset_prefix_cache() is True
    blocks, hit, _ = mgr.get_computed_blocks(make_request("r2", 1536))
    assert hit == 0


def test_events_aggregated_from_both_pools(monkeypatch):
    mgr = build_manager(monkeypatch, mode="1", slots="24", enable_kv_cache_events=True)
    req = make_request("r1", 1536)
    schedule_chunks(mgr, req, BLOCK, 3)

    events = mgr.take_events()
    assert events
    assert mgr.coordinator.recurrent_pool.kv_event_queue == []
    assert mgr.block_pool.kv_event_queue == []
