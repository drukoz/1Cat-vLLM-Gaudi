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
from vllm_gaudi.v1.worker.recurrent_lattice import align_lattice  # noqa: E402

from tests.unit_tests.flash_next_spec_sizes import SPEC_SIZES, require_startup, size_id  # noqa: E402

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


def _spec_config(k: int) -> KVCacheConfig:
    """The same config with ``k`` speculative (align-mode scratch) blocks per recurrent group."""
    import dataclasses
    cfg = _build_config()
    cfg.kv_cache_groups = [
        dataclasses.replace(g, kv_cache_spec=dataclasses.replace(g.kv_cache_spec, num_speculative_blocks=k))
        if isinstance(g.kv_cache_spec, MambaSpec) else g for g in cfg.kv_cache_groups
    ]
    return cfg


def _spec3_config() -> KVCacheConfig:
    """The recipe: 3 speculative blocks per recurrent group."""
    return _spec_config(3)


def build_manager(monkeypatch, slots=None, mode=None, enable_kv_cache_events=False, config=None):
    """Fresh KVCacheManager with the pool env vars set as the test states them."""
    monkeypatch.delenv(ENV_MODE, raising=False)
    monkeypatch.delenv(ENV_SLOTS, raising=False)
    if mode is not None:
        monkeypatch.setenv(ENV_MODE, mode)
    if slots is not None:
        monkeypatch.setenv(ENV_SLOTS, slots)
    return KVCacheManager(kv_cache_config=config if config is not None else _build_config(),
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


def test_align_spec_scratch_blocks_route_to_recurrent_pool(monkeypatch):
    """num_speculative_blocks=3: each recurrent group carries a 1+3 align tail from the pool."""
    mgr = build_manager(monkeypatch, mode="1", slots="24", config=_spec3_config())
    req = make_request("r1", BLOCK)
    blocks, hit, _ = mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    req.num_computed_tokens += BLOCK

    ids = mgr.get_block_ids(req.request_id)
    assert len(ids[0]) == 1  # attention: only the position block
    for group_ids in ids[1:]:
        assert len(group_ids) == 1 + 3  # position block + 3 scratch blocks
        for block_id in group_ids:
            assert 0 <= block_id < 25

    # Decode step with 3 drafts: the position block grows, the scratch tail is reused.
    assert mgr.allocate_slots(req, 1 + 3) is not None
    req.num_computed_tokens += 1
    ids = mgr.get_block_ids(req.request_id)
    assert len(ids[0]) == 2
    for group_ids in ids[1:]:
        assert len(group_ids) == 2 + 3
        for block_id in group_ids:
            assert 0 <= block_id < 25


def test_align_spec_scratch_blocks_count_against_pool(monkeypatch):
    """The 3 scratch blocks per recurrent group consume pool slots and gate admission."""
    mgr = build_manager(monkeypatch, mode="1", slots="16", config=_spec3_config())
    req1 = make_request("r1", BLOCK)
    blocks, hit, _ = mgr.get_computed_blocks(req1)
    assert mgr.allocate_slots(req1, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    req1.num_computed_tokens += BLOCK

    req2 = make_request("r2", BLOCK, offset=100000)  # distinct prompt: no prefix hit
    blocks, hit, _ = mgr.get_computed_blocks(req2)
    assert mgr.allocate_slots(req2, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is None

    mgr.free(req1)
    assert mgr.allocate_slots(req2, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None


def test_align_spec_block_rows_across_steps(monkeypatch):
    """Spike RUK-2847: record the recurrent group's block-table row across MTP decode steps.

    Drives one request through the real KVCacheManager (pool mode, 3 speculative blocks, align
    cache mode), accepting 1..4 tokens per MTP verification step, across 3 block boundaries.
    Prints one line per step (ctx, row length, row tail, align_lattice load/store) for the
    report; asserts only the layout rules observed in this trace.
    """
    k = 3
    sched = 1 + k
    accepts = [4, 1, 2, 3, 4, 4, 1, 4, 3, 2] + [4] * 260

    mgr = build_manager(monkeypatch, mode="1", slots="64", config=_spec3_config())
    null_id = mgr.coordinator.recurrent_pool.null_block.block_id
    attn_free_at_start = mgr.block_pool.get_num_free_blocks()
    recurrent_free_at_start = mgr.coordinator.recurrent_pool.get_num_free_blocks()

    req = make_request("r1", BLOCK - 2)
    blocks, hit, _ = mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, BLOCK - 2, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    req.num_computed_tokens += BLOCK - 2

    print(f"\nprefill: row={list(mgr.get_block_ids(req.request_id)[1])}")
    prev_ctx = req.num_computed_tokens
    prev_window = None
    null_load_steps = []
    boundary_steps = []
    for step, a in enumerate(accepts):
        assert mgr.allocate_slots(req, sched) is not None
        ctx = req.num_computed_tokens
        row = list(mgr.get_block_ids(req.request_id)[1])
        load, store = align_lattice(row, ctx, sched, BLOCK, k)
        n_pos = -(-(ctx + sched) // BLOCK)
        print(f"step={step:3d} ctx={ctx:5d} a={a} len(row)={len(row):2d} n_pos={n_pos:2d} "
              f"tail={row[-6:]} load={load} store={store}")

        # Row length rule observed for align mode with 3 speculative blocks.
        assert len(row) == -(-(ctx + sched) // BLOCK) + k

        # The scratch window is exactly the last k entries, row[n_pos:n_pos+k], never null.
        window = row[n_pos:n_pos + k]
        assert window == row[-k:]
        assert all(b != null_id for b in window)
        if prev_window is not None:
            if window != prev_window:
                # Only steps that append one position block move the window: it shifts left
                # by one (the leftmost id becomes the new position block) and gains one id.
                assert window == prev_window[1:] + [window[-1]]
                assert window[-1] not in prev_window
        prev_window = window

        # store[0] (this step's running-state block) and load[0] are real blocks: the running
        # position block (index ctx//bs, and (ctx-1)//bs == ctx//bs for ctx % bs != 0) is
        # never freed; only strictly older blocks are nulled by remove_skipped_blocks.
        assert store[0] != null_id
        if load[0] == null_id:
            null_load_steps.append(step)
        assert load[0] != null_id

        # The step right after a crossing: the previous running position block is nulled.
        if prev_ctx // BLOCK < ctx // BLOCK:
            boundary_steps.append(step)
            assert row[(prev_ctx - 1) // BLOCK] == null_id
        prev_ctx = ctx
        req.num_computed_tokens += a
        if ctx > 3 * BLOCK:
            break

    print(f"summary: steps_done={step + 1} final_ctx={ctx} final_row={row}")
    print(f"summary: null_load_steps={null_load_steps} boundary_crossing_steps={boundary_steps}")
    assert len(boundary_steps) == 3

    mgr.free(req)
    assert mgr.block_pool.get_num_free_blocks() == attn_free_at_start
    assert mgr.coordinator.recurrent_pool.get_num_free_blocks() == recurrent_free_at_start



@pytest.fixture(params=SPEC_SIZES, ids=size_id)
def spec_k(request) -> int:
    """Each speculative size k the Flash-Next tests run at, after flash_next_spec_sizes' startup check."""
    require_startup(request.param, prefix_caching=True)
    return request.param


def test_align_tail_and_pool_admission_at_every_size(monkeypatch, spec_k):
    """k speculative blocks: each recurrent group of a request holds its position blocks plus k scratch blocks,
    all from the pool, and a pool one slot short of two such requests admits one."""
    per_request = NUM_RECURRENT_GROUPS * (1 + spec_k)  # one position block and k scratch blocks per group
    mgr = build_manager(monkeypatch, mode="1", slots=str(2 * per_request - 1), config=_spec_config(spec_k))
    req1 = make_request("r1", BLOCK)
    blocks, hit, _ = mgr.get_computed_blocks(req1)
    assert mgr.allocate_slots(req1, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    req1.num_computed_tokens += BLOCK
    ids = mgr.get_block_ids(req1.request_id)
    assert len(ids[0]) == 1
    assert [len(group_ids) for group_ids in ids[1:]] == [1 + spec_k] * NUM_RECURRENT_GROUPS
    assert len({block_id for group_ids in ids[1:] for block_id in group_ids}) == per_request

    req2 = make_request("r2", BLOCK, offset=100000)  # distinct prompt: no prefix hit
    blocks, hit, _ = mgr.get_computed_blocks(req2)
    assert mgr.allocate_slots(req2, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is None
    mgr.free(req1)
    assert mgr.allocate_slots(req2, BLOCK, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None


def test_align_lattice_on_the_manager_rows_at_every_size(monkeypatch, spec_k):
    """The real manager's block-table rows across verification steps of 1 + k tokens and three block ends.

    Every accepted count 1 .. 1 + k occurs. At each step the row has ``cdiv(ctx + 1 + k, block) + k`` entries,
    its last k entries are the scratch blocks, and ``align_lattice`` finds 1 + k distinct, non-null slots: the
    running block and the k scratch blocks (RUK-2847's rules, which the k = 3 trace above established).
    """
    k, sched = spec_k, 1 + spec_k
    accepts = list(range(1, sched + 1)) + list(range(sched, 0, -1)) + [sched] * 400
    mgr = build_manager(monkeypatch, mode="1", slots="64", config=_spec_config(k))
    null_id = mgr.coordinator.recurrent_pool.null_block.block_id
    attn_free_at_start = mgr.block_pool.get_num_free_blocks()
    recurrent_free_at_start = mgr.coordinator.recurrent_pool.get_num_free_blocks()

    req = make_request("r1", BLOCK - 2)
    blocks, hit, _ = mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, BLOCK - 2, num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
    req.num_computed_tokens += BLOCK - 2

    prev_ctx, crossings, seen = req.num_computed_tokens, 0, set()
    for a in accepts:
        assert mgr.allocate_slots(req, sched) is not None
        ctx = req.num_computed_tokens
        n_pos = -(-(ctx + sched) // BLOCK)
        for group in range(1, 1 + NUM_RECURRENT_GROUPS):
            row = list(mgr.get_block_ids(req.request_id)[group])
            assert len(row) == n_pos + k, f"ctx {ctx}: row of {len(row)} blocks, expected {n_pos} + {k}"
            load, store = align_lattice(row, ctx, sched, BLOCK, k, null_block=null_id)
            assert load == store == row[n_pos - 1:n_pos + k] and len(set(load)) == sched
            assert row[n_pos:] == row[-k:] and null_id not in row[-k:]
        if prev_ctx // BLOCK < ctx // BLOCK:
            crossings += 1
        prev_ctx = ctx
        seen.add(a)
        req.num_computed_tokens += a
        if ctx > 3 * BLOCK:
            break
    assert crossings == 3 and seen == set(range(1, sched + 1))

    mgr.free(req)
    assert mgr.block_pool.get_num_free_blocks() == attn_free_at_start
    assert mgr.coordinator.recurrent_pool.get_num_free_blocks() == recurrent_free_at_start
