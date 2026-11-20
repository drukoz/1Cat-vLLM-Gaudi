# SPDX-License-Identifier: Apache-2.0
"""Recurrent (GDN / short-conv) state pool: the contract shared by the platform hook, the scheduler
coordinator (M1-T2) and the worker/runner allocation (M1-T3). Pure Python, importable on CPU."""
import enum
import functools
import os
from collections.abc import Callable

from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroupSpec, KVCacheSpec, MambaSpec


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

    The engine groups a hybrid model's layers into groups of the attention-layer count (a recurrent spec with
    fewer layers keeps a group of its own, see ``hpu_kv_cache_groups``), and the runner shares one state
    tensor between the layers at the same position of every group (``_mamba_state_tensors`` in
    ``HPUModelRunner.initialize_kv_cache``).
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


def minor_recurrent_layers(kv_cache_spec: dict[str, KVCacheSpec]) -> dict[MambaSpec, list[str]]:
    """Recurrent specs that have fewer layers than every attention spec, each with its layer names.

    The Flash-Next PLE short conv is one: 1 layer next to 12 attention and 36 GDN layers.
    """
    attention: dict[KVCacheSpec, int] = {}
    recurrent: dict[MambaSpec, list[str]] = {}
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, MambaSpec):
            recurrent.setdefault(spec, []).append(name)
        elif isinstance(spec, AttentionSpec):
            attention[spec] = attention.get(spec, 0) + 1
    if not attention:
        return {}
    return {spec: names for spec, names in recurrent.items() if len(names) < min(attention.values())}


def hpu_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
    engine_groups: Callable[[dict[str, KVCacheSpec]], list[KVCacheGroupSpec]],
) -> list[KVCacheGroupSpec]:
    """The engine's same-page-size KV cache groups, with every attention spec's layers in ONE group.

    ``engine_groups`` (vLLM's ``_get_kv_cache_groups_uniform_page_size``) makes every group as large as the
    spec with the fewest layers. One PLE layer therefore turns Flash-Next's 12 attention layers into 12
    single-layer groups: the scheduler then draws 12 block ids per block of tokens from a pool counted in
    single-layer pages. The HPU runner turns one block table (the first attention group's) into the slots of
    every attention layer, each layer in its own tensors, so it needs the attention layers in one group: one
    block id per block of tokens, a pool counted in 12-layer pages.

    Recurrent specs with fewer layers than the attention layers are therefore left out of the pattern: the
    engine groups the other layers (12 attention layers, 3 GDN groups of 12), and each left-out spec gets a
    group of its own, in the place the engine's ordering (first layer of each spec) gives it. Without such a
    spec the engine's groups are returned unchanged.
    """
    minor = minor_recurrent_layers(kv_cache_spec)
    if not minor:
        return engine_groups(kv_cache_spec)
    left_out = {name for names in minor.values() for name in names}
    groups = list(engine_groups({name: spec for name, spec in kv_cache_spec.items() if name not in left_out}))
    groups += [KVCacheGroupSpec(names, type(spec).merge([kv_cache_spec[name] for name in names]))
               for spec, names in minor.items()]
    first_layer: dict[KVCacheSpec, int] = {}
    for index, spec in enumerate(kv_cache_spec.values()):
        first_layer.setdefault(spec, index)
    groups.sort(key=lambda group: min(first_layer[kv_cache_spec[name]] for name in group.layer_names))
    return groups


def install_hpu_kv_cache_grouping() -> bool:
    """Make the engine group KV cache layers with ``hpu_kv_cache_groups``. Returns True when it is in place.

    Called from ``HpuPlatform.register_custom_kv_cache_specs``, which the engine core runs right before it
    groups the layers (``register_all_kvcache_specs`` in ``EngineCore._initialize_kv_caches``).
    ``get_kv_cache_groups`` looks the grouping function up in its module at call time, so rebinding it works.
    """
    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    engine_groups = getattr(kv_cache_utils, "_get_kv_cache_groups_uniform_page_size", None)
    if engine_groups is None:
        logger.error("[HPU] vLLM has no _get_kv_cache_groups_uniform_page_size any more: KV cache groups are not "
                     "adjusted, and a hybrid model with a short recurrent spec (the Flash-Next PLE layer) gets one "
                     "group per attention layer, which the HPU runner cannot serve")
        return False
    if getattr(engine_groups, "_vllm_gaudi_hpu_groups", False):
        return True

    @functools.wraps(engine_groups)
    def groups_for_hpu(kv_cache_spec: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
        groups = hpu_kv_cache_groups(kv_cache_spec, engine_groups)
        minor = minor_recurrent_layers(kv_cache_spec)
        if minor:
            names = ", ".join(name for layer_names in minor.values() for name in layer_names)
            attention = [len(group.layer_names) for group in groups if isinstance(group.kv_cache_spec, AttentionSpec)]
            logger.info(
                "[HPU] KV cache groups: %s grouped apart from the attention layers (fewer layers than they have); "
                "%d groups, attention layers per group: %s", names, len(groups), attention)
        return groups

    groups_for_hpu._vllm_gaudi_hpu_groups = True
    kv_cache_utils._get_kv_cache_groups_uniform_page_size = groups_for_hpu
    return True


def resolve_pool_slots(max_num_seqs: int, num_groups: int, cache_slots: int, num_speculative_tokens: int = 0) -> int:
    """Pool size: the live need of max_num_seqs requests plus cache_slots retained prefix states.

    Under speculation the MambaManager allocates ``num_speculative_tokens`` scratch blocks per
    request right after its position blocks, so each live request owns
    ``LIVE_SLOTS_PER_REQUEST + num_speculative_tokens`` slots per recurrent group.
    """
    return max_num_seqs * num_groups * (LIVE_SLOTS_PER_REQUEST + num_speculative_tokens) + cache_slots


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
