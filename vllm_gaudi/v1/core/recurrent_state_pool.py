# SPDX-License-Identifier: Apache-2.0
"""Recurrent (GDN / short-conv) state pool: the contract shared by the platform hook, the scheduler
coordinator (M1-T2) and the worker/runner allocation (M1-T3). Pure Python, importable on CPU."""
import enum
import os

from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec


class RecurrentStatePoolMode(enum.IntEnum):
    """How the HPU allocates recurrent-state slots (next free value: 2)."""
    OFF = 0   # today: one recurrent slot per attention block id (num_blocks + 1 rows per tensor)
    POOL = 1  # a separate pool of VLLM_HPU_RECURRENT_STATE_POOL_SLOTS slots shared by the recurrent groups


# align mode holds the last-computed block and the last-scheduled block per group per request,
# plus one extra for the chunk transition / copy-on-write of a partial hit.
LIVE_SLOTS_PER_REQUEST = 3
ENV_MODE = "VLLM_HPU_RECURRENT_STATE_POOL"
ENV_SLOTS = "VLLM_HPU_RECURRENT_STATE_POOL_SLOTS"
ENV_CACHE_SLOTS = "VLLM_HPU_RECURRENT_STATE_POOL_CACHE_SLOTS"


def is_recurrent_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, MambaSpec)


def _recurrent_layer_counts(kv_cache_spec: dict[str, KVCacheSpec]) -> dict[MambaSpec, int]:
    counts: dict[MambaSpec, int] = {}
    for spec in kv_cache_spec.values():
        if isinstance(spec, MambaSpec):
            counts[spec] = counts.get(spec, 0) + 1
    return counts


def recurrent_positions(kv_cache_spec: dict[str, KVCacheSpec]) -> dict[MambaSpec, int]:
    """State tensors the runner allocates per distinct recurrent spec: min(layers of that spec, attention layers).

    vLLM groups a hybrid model's layers into groups of the attention-layer count, and the runner shares one
    state tensor between the layers at the same position of every group (hpu_model_runner.py:8787-8797).
    """
    n_attn = sum(isinstance(s, AttentionSpec) for s in kv_cache_spec.values())
    return {spec: (min(n, n_attn) if n_attn > 0 else n) for spec, n in _recurrent_layer_counts(kv_cache_spec).items()}


def num_recurrent_groups(kv_cache_spec: dict[str, KVCacheSpec]) -> int:
    """Recurrent KV-cache groups: for each spec, ceil(layers / positions)."""
    positions = recurrent_positions(kv_cache_spec)
    return sum(-(-n // positions[spec]) for spec, n in _recurrent_layer_counts(kv_cache_spec).items())


def pool_slot_bytes(kv_cache_spec: dict[str, KVCacheSpec]) -> int:
    """Device bytes one pool slot costs over all recurrent state tensors (real page sizes, no padding)."""
    return sum(pos * spec.real_page_size_bytes for spec, pos in recurrent_positions(kv_cache_spec).items())


def resolve_pool_slots(max_num_seqs: int, num_groups: int, cache_slots: int) -> int:
    """Pool size: the live need of max_num_seqs requests plus cache_slots retained prefix states."""
    return max_num_seqs * num_groups * LIVE_SLOTS_PER_REQUEST + cache_slots


def pool_tensor_slots(pool_slots: int) -> int:
    """Rows of each recurrent state tensor: ids 0..pool_slots (0 is vLLM's null block) plus one padding row."""
    return pool_slots + 2


def pool_pad_block_id(pool_slots: int) -> int:
    """The padding row's index (the runner's _MAMBA_PAD_BLOCK_ID for pooled groups)."""
    return pool_slots + 1


def pool_mode_from_env() -> RecurrentStatePoolMode:
    return RecurrentStatePoolMode(int(os.environ.get(ENV_MODE, "0")))


def pool_slots_from_env() -> int:
    """The resolved pool size, 0 when the pool is off. Reads os.environ at call time (never at import)."""
    if pool_mode_from_env() != RecurrentStatePoolMode.POOL:
        return 0
    return max(0, int(os.environ.get(ENV_SLOTS, "0")))
