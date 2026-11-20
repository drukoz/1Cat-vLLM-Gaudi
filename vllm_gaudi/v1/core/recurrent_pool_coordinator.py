# SPDX-License-Identifier: Apache-2.0
"""Recurrent state pool, scheduler side (M1-T2): MambaSpec groups allocate from their own BlockPool.

When VLLM_HPU_RECURRENT_STATE_POOL is on, the attention groups keep the stock budget of
`kv_cache_config.num_blocks` blocks while every MambaSpec group's manager allocates its (recurrent
state) blocks from a second, small BlockPool of `pool_slots + 1` blocks (id 0 is the null block),
so the worker (M1-T3) can size its recurrent state tensors to the pool. KVCacheManager only ever
talks to `coordinator.block_pool`, so the attention pool is a `RecurrentRoutingBlockPool` that
forwards the deferred frees, prefix-cache resets and events of recurrent blocks to the recurrent
pool.
"""
import os
from collections.abc import Iterable, Sequence

from vllm.distributed.kv_events import KVCacheEvent
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, MambaSpec


ENV_KEEP_PROMPT_END = "VLLM_HPU_PREFIX_KEEP_PROMPT_END"


def keep_prompt_end_requested(environ=None) -> bool:
    """``VLLM_HPU_PREFIX_KEEP_PROMPT_END``: 0 (default) upstream's sparse retention, 1 ``keep_block_aligned_prompt_end``."""
    value = int((os.environ if environ is None else environ).get(ENV_KEEP_PROMPT_END, "0"))
    if value not in (0, 1):
        raise ValueError(f"{ENV_KEEP_PROMPT_END} must be 0 or 1, got {value}")
    return value == 1


def keep_block_aligned_prompt_end(manager) -> None:
    """Make a recurrent group's sparse retention also keep the state at a prompt's end when the prompt ends
    on a scheduler block.

    Why. Under sparse retention (``--prefix-cache-retention-interval 0``, the default) a recurrent group
    keeps only its "reachable boundaries", and the one a prompt of ``N`` tokens gets is ``N - 1`` aligned
    down to the block (``SingleTypeKVCacheManager.cache_blocks``): the furthest point a replay of the same
    prompt can reuse, since a replay computes its last token again. The scheduler ends a chunk there
    (``Scheduler._mamba_block_aligned_split`` stops at ``N - N % block``), so a state exists at it, unless
    ``N`` is a multiple of the block: then the split rule's stop is ``N`` itself, no chunk ends one block
    earlier, the kept entry is a null block, and the state that does exist, after exactly ``N`` tokens, is
    masked out and freed while the request decodes. The next turn of that conversation then finds no
    recurrent state at all, the hybrid hit is 0 and the whole prompt is prefilled again (one turn in 512 at
    TP8, one in 896 at TP4; tens of seconds at 128K).

    What. ``N`` is a reuse point too: a longer prompt that starts with these ``N`` tokens (the next turn)
    can start right behind it. For every reachable boundary ``b`` with ``b + 1`` on a block boundary the
    mask is asked to keep ``b + 1`` as well. The only boundary of that form is the replay boundary
    ``N - 1`` of a block-aligned prompt (shared-prefix junctions are block-aligned themselves), so other
    prompts keep exactly what they kept. Cost: one more retained state per recurrent group for such a
    prompt, evicted like every cached block. A replay of the same aligned prompt still misses.
    """
    base = manager.reachable_block_mask

    def reachable_block_mask(*args, alignment_tokens=None, reachable_boundaries=(), **kwargs):
        if alignment_tokens:
            ends = [b + 1 for b in reachable_boundaries if (b + 1) % alignment_tokens == 0]
            if ends:
                reachable_boundaries = [*reachable_boundaries, *ends]
        return base(*args, alignment_tokens=alignment_tokens, reachable_boundaries=reachable_boundaries, **kwargs)

    manager.reachable_block_mask = reachable_block_mask


class RecurrentRoutingBlockPool(BlockPool):
    """The attention pool. KVCacheManager only ever talks to coordinator.block_pool, so deferred
    frees, resets and events of recurrent blocks arrive here and are forwarded to the recurrent
    pool.

    `evict_blocks` is deliberately NOT routed: it stays attention-only (KV connectors and
    offloading are out of scope for M1-T2; recurrent ids are outside this pool's id range).
    """

    def __init__(self, *args, recurrent_pool: BlockPool, **kwargs):
        super().__init__(*args, **kwargs)
        self.recurrent_pool = recurrent_pool

    def _owns_recurrent(self, block: KVCacheBlock) -> bool:
        """True when `block` is one of the recurrent pool's blocks (ids 0..pool_slots)."""
        blocks = self.recurrent_pool.blocks
        return block.block_id < len(blocks) and blocks[block.block_id] is block

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        own: list[KVCacheBlock] = []
        recurrent: list[KVCacheBlock] = []
        for block in ordered_blocks:
            (recurrent if self._owns_recurrent(block) else own).append(block)
        super().free_blocks(own)
        if recurrent:
            self.recurrent_pool.free_blocks(recurrent)

    def reset_prefix_cache(self) -> bool:
        attention_ok = super().reset_prefix_cache()
        recurrent_ok = self.recurrent_pool.reset_prefix_cache()
        return attention_ok and recurrent_ok

    def take_events(self) -> list[KVCacheEvent]:
        return super().take_events() + self.recurrent_pool.take_events()


class HpuRecurrentPoolCoordinator(HybridKVCacheCoordinator):
    """HybridKVCacheCoordinator whose MambaSpec groups draw block ids 0..pool_slots from a
    separate pool. Recurrent ids live in [0, pool_slots] in every MambaSpec group's block table;
    attention ids stay in [0, num_blocks - 1].
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        num_prefill_lookahead: int = 0,
        *,
        pool_slots: int,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            num_prefill_lookahead=num_prefill_lookahead,
        )
        self.pool_slots = pool_slots
        self.recurrent_group_ids = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)
        }
        # Block ids 0..pool_slots; id 0 is the null block. Events go to the routing pool
        # (take_events aggregates both pools), so this pool collects none of its own.
        self.recurrent_pool = BlockPool(
            num_gpu_blocks=pool_slots + 1,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=None,
        )
        # Replace the plain pool the base __init__ built; KVCacheManager reads
        # coordinator.block_pool after construction, so it sees the routing pool.
        self.block_pool = RecurrentRoutingBlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
            recurrent_pool=self.recurrent_pool,
        )
        keep_prompt_end = keep_prompt_end_requested()
        for gid, manager in enumerate(self.single_type_managers):
            manager.block_pool = self._pool_for_group(gid)
            manager._null_block = manager.block_pool.null_block
            if keep_prompt_end and gid in self.recurrent_group_ids:
                keep_block_aligned_prompt_end(manager)

    def _pool_for_group(self, group_id: int) -> BlockPool:
        return self.recurrent_pool if group_id in self.recurrent_group_ids else self.block_pool

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """The base-class loop (KVCacheCoordinator.get_num_blocks_to_allocate), summed per pool.

        Returns the attention need; when the recurrent need exceeds the recurrent pool's free
        blocks it returns `block_pool.num_gpu_blocks + 1`, which KVCacheManager compares against
        the (attention pool's) free count, so it sees "not enough" and waits or preempts.
        """
        attention_need = 0
        recurrent_need = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                need = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                need = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            if i in self.recurrent_group_ids:
                recurrent_need += need
            else:
                attention_need += need
        if recurrent_need > self.recurrent_pool.get_num_free_blocks():
            return self.block_pool.num_gpu_blocks + 1
        return attention_need

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the reconciled (combined) cache hit.
                - ``num_uncached_common_prefix_tokens``: a shared prefix that a
                  sparse-retention group has not cached yet (0 unless hybrid).
        """

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                # DCP/PCP shard each block's KV across ranks, so the manager's
                # effective block size may exceed the spec's.
                group_block_size = self.single_type_managers[first_group_id].block_size
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
                # Eagle matches one extra drop unit (one hash unit for
                # fine-grained managers, else one cache block) and then drops
                # it, landing back at the candidate length. No margin for
                # mamba: its finder never drops (draft models have no mamba
                # layers), so the hit would grow past the candidate.
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(
                        curr_hit_length + eagle_margin, max_cache_hit_length
                    )
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self._pool_for_group(first_group_id),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=self.single_type_managers[
                        first_group_id
                    ].dcp_world_size,
                    pcp_world_size=self.single_type_managers[
                        first_group_id
                    ].pcp_world_size,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate every full-attention group (target and draft) blocks
        # to final hit_length.
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            group_block_size = self.single_type_managers[group.group_ids[0]].block_size
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        # Uncached shared prefix detection: if any attn. group cached a longer
        # prefix than the reconciled hit, it is an uncached common prefix across
        # requests that a sparse-retention group hasn't cached yet.
        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        """Like find_longest_cache_hit but evaluates each group independently.

        Returns:
            (blocks_per_group, hit_lengths_per_group)
        """

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups

        for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
            manager = self.single_type_managers[group_ids[0]]
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self._pool_for_group(group_ids[0]),
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=manager.dcp_world_size,
                pcp_world_size=manager.pcp_world_size,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit

        return tuple(hit_blocks), tuple(hit_lengths)

