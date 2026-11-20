# SPDX-License-Identifier: Apache-2.0
"""CPU test of the KV cache sizing at the real Flash-Next geometry (no checkpoint, no memory touched).

On the 2026-11-20 rental the 8-card Flash-Next server died in ``initialize_kv_cache`` with
``PT_DEVMEM Allocation failed for size::27623489536``: one attention layer's K cache was being
allocated with 53,952,128 token rows while the engine reported a cache of 4,390,245 tokens.

Every start below runs the code a card runs, in the order a card runs it:

1. the engine config from the model's real ``config.json`` (``EngineArgs.create_engine_config``),
2. the platform hooks: ``_configure_recurrent_state`` (pool slots) and
   ``HpuPlatform.update_block_size_for_backend`` (attention block size, padded recurrent page),
3. ``HPUModelRunner.get_kv_cache_spec`` over the model's cache layers,
4. ``HPUWorker.determine_available_memory`` with the card's free memory,
5. the engine's ``register_all_kvcache_specs`` (it ends with the platform's
   ``register_custom_kv_cache_specs``) and ``get_kv_cache_configs``, the first two KV cache steps of
   ``EngineCore._initialize_kv_caches``,
6. ``HPUModelRunner.initialize_kv_cache`` with every ``torch.zeros`` recorded and redirected to the
   ``meta`` device (shapes and dtypes without storage),
7. the engine's ``KVCacheManager`` (with the plugin's coordinator) admitting full-length requests.

The model itself is replaced by weightless layers: an ``Attention`` per QSA layer and a ``MambaBase``
per GDN / PLE layer, registered under the real layer names with the state shapes the model class
computes from the config (``get_mamba_specs_from_config`` / ``get_mamba_state_shape_from_config``).
``test_weightless_layers_report_what_the_real_layers_report`` holds them against the tiny model built
for real.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import os  # noqa: E402

os.environ["TMPDIR"] = "/tmp"

import dataclasses  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import vllm_gaudi.utils  # noqa: E402, F401  (installs the recurrent-pool coordinator factory, as on a card)
import vllm.platforms  # noqa: E402
import vllm.v1.core.kv_cache_utils as kv_cache_utils  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.layers.attention import Attention  # noqa: E402
from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402
from vllm.model_executor.models import ModelRegistry  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.utils.math_utils import cdiv  # noqa: E402
from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    generate_scheduler_kv_cache_config, get_kv_cache_capacity, get_kv_cache_configs, get_request_block_hasher,
    init_none_hash, resolve_kv_cache_block_sizes)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, MambaSpec  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

import vllm_gaudi.models.qwen4_exp as flash_next  # noqa: E402
import vllm_gaudi.v1.worker.hpu_model_runner as runner_module  # noqa: E402
import vllm_gaudi.v1.worker.hpu_worker as hw  # noqa: E402
from vllm_gaudi.extension.profiler import format_bytes  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention  # noqa: E402
from vllm_gaudi.platform import HpuPlatform, _configure_recurrent_state, is_flash_next_model  # noqa: E402
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionBackendV1  # noqa: E402
from vllm_gaudi.v1.core.recurrent_state_pool import (  # noqa: E402
    ENV_CACHE_SLOTS, ENV_MODE, ENV_SLOTS, LIVE_SLOTS_PER_REQUEST, pool_pad_block_id, pool_slots_from_env,
    pool_tensor_slots)

pytestmark = pytest.mark.skipif(not hpu_cpu_stubs.STUBBED, reason="CPU-only: needs the stubbed Gaudi runtime")

REPO = Path(__file__).resolve().parents[3]
FLASH_NEXT_CONFIG = REPO / "spikes" / "kv-capacity" / "configs" / "Qwen3.8-Flash-Next.json"
QWEN_27B_CONFIG = REPO / "spikes" / "kv-capacity" / "configs" / "Qwen3.8-27B.json"
TINY_CONFIG = REPO / "tests" / "unit_tests" / "models" / "flash_next" / "tiny_tp8_config.json"

GIB = 1 << 30
MAX_MODEL_LEN = 131072
KERNEL_BLOCK = 128  # tokens per HPU paged-attention kernel block (HPUAttentionBackendV1.get_preferred_block_size)
CARD_TOTAL_BYTES = int(94.62 * GIB)  # server0.log: "23.51 GiB/94.62 GiB used"
# server0.log (2026-11-20, TP8): "Free device memory: 71.11 GiB". The log rounds; the failed allocation
# does not: 27,623,489,536 B = (4 * 105,375 + 1) kernel blocks * 128 rows * 512 B, so the engine
# computed num_blocks = 105,375, which pins the free memory to [76,356,000,068, 76,356,672,840) bytes.
FN_TP8_FREE_BYTES = 76_356_300_000
# Not measured: TP4 holds twice the weights per card (TP8 used 94.62 - 71.11 = 23.51 GiB).
FN_TP4_FREE_BYTES = CARD_TOTAL_BYTES - 2 * (CARD_TOTAL_BYTES - FN_TP8_FREE_BYTES)
# Run 1 of the 27B (spikes/kv-capacity/README.md): 72.98 GiB "reserved for usable KV cache", TP8, bf16,
# no pool; its log read "2,647,954 tokens", "20.20x", "Usable num_blocks: 5293".
QWEN_27B_USABLE_BYTES = int(72.98 * GIB)

# The environment the platform hook and the runner read or write (restored after every start).
ENV_VARS = (ENV_MODE, ENV_SLOTS, ENV_CACHE_SLOTS, "VLLM_COMPACT_GDN", "VLLM_HPU_FLASH_NEXT_QSA_MODE",
            "VLLM_USE_HYBRID_CACHE", "VLLM_USE_NAIVE_MAMBA_CACHE_SHARING", "VLLM_PROFILE_PROMPT", "VLLM_PROFILE_DECODE",
            "VLLM_GRAPH_RESERVED_MEM", "VLLM_HPU_GDN_ACTIVE_STATE_VIEWS")


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One server start: the model's config.json and the launch settings."""
    name: str
    config: Path
    tp: int
    max_num_seqs: int
    free_bytes: int  # torch.hpu.mem_get_info()[0] after the profile run
    pool: bool  # VLLM_HPU_RECURRENT_STATE_POOL=1
    sparse_qsa: bool  # VLLM_HPU_FLASH_NEXT_QSA_MODE=1
    max_model_len: int = MAX_MODEL_LEN


# The rental defaults (tools/flash_next/rental/serve.py): pool on, sparse QSA, prefix caching, 131072 tokens.
FN_TP8 = Geometry("flash-next-tp8x32", FLASH_NEXT_CONFIG, 8, 32, FN_TP8_FREE_BYTES, pool=True, sparse_qsa=True)
FN_TP4 = Geometry("flash-next-tp4x16", FLASH_NEXT_CONFIG, 4, 16, FN_TP4_FREE_BYTES, pool=True, sparse_qsa=True)
# The 4-layer test model (1 QSA, 3 GDN, 1 PLE layer), which starts on the card. 8 GiB of free memory keeps
# the block pool of its 16 KiB pages small; its context is 4096 tokens.
TINY = Geometry("tiny-tp8x32", TINY_CONFIG, 8, 32, 8 * GIB, pool=True, sparse_qsa=True, max_model_len=4096)
# 16 attention + 48 GDN layers, no PLE, no pool; free memory is set from the Run 1 budget in start(), and
# 20 sequences is what its log's 20.20x covers.
QWEN_27B = Geometry("27b-tp8-no-pool", QWEN_27B_CONFIG, 8, 20, 0, pool=False, sparse_qsa=False)

# Not one of the rental layouts: Flash-Next with prefix caching and without the pool (GDN and PLE state per
# block id). Same code, same geometry; it has not started on a card either.
FN_TP8_NO_POOL = Geometry("flash-next-tp8-no-pool", FLASH_NEXT_CONFIG, 8, 32, FN_TP8_FREE_BYTES, pool=False,
                          sparse_qsa=True)

GEOMETRIES = [FN_TP8, FN_TP4, TINY, QWEN_27B, FN_TP8_NO_POOL]
POOL_GEOMETRIES = [FN_TP8, FN_TP4, TINY]


class _AttentionLayer(Attention):
    """The KV-cache owner of a full-attention layer (``HpuQwen4ExpQSAAttention.attn``), without weights."""

    def __init__(self, layer_name: str, num_kv_heads: int, head_size: int) -> None:
        torch.nn.Module.__init__(self)
        self.layer_name = layer_name
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.attn_type = AttentionType.DECODER
        self.kv_sharing_target_layer_name = None


class _RecurrentLayer(torch.nn.Module, MambaBase):
    """A GDN or PLE layer's cache: the model class's state shapes, ``MambaBase.get_kv_cache_spec`` unchanged."""

    def __init__(self, state: MambaSpec, kind: MambaAttentionBackendEnum) -> None:
        super().__init__()
        self._state = state
        self._kind = kind

    def get_state_shape(self):
        return self._state.shapes

    def get_state_dtype(self):
        return self._state.dtypes

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return self._kind

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return self._state.tp_replicated


class _Indexer(torch.nn.Module):
    """Receives the sparse-QSA side caches the runner allocates."""

    def __init__(self) -> None:
        super().__init__()
        self.compressed_key_cache = None
        self.key_ring = None
        self.ring_rows = None

    def bind_key_ring(self, ring, rows) -> None:
        self.key_ring = ring
        self.ring_rows = rows


def _qsa_owner(attn: _AttentionLayer) -> HpuQwen4ExpQSAAttention:
    owner = HpuQwen4ExpQSAAttention.__new__(HpuQwen4ExpQSAAttention)
    torch.nn.Module.__init__(owner)
    owner.attn = attn
    owner.indexer = _Indexer()
    return owner


def _recurrent_states(vllm_config) -> tuple[MambaSpec, MambaSpec | None]:
    """(GDN state, PLE state or None) from the model class, as ``Platform._align_hybrid_block_size`` reads them."""
    model_config = vllm_config.model_config
    model_cls, _ = ModelRegistry.resolve_model_cls(model_config.architecture, model_config=model_config)
    if hasattr(model_cls, "get_mamba_specs_from_config"):
        gdn, ple = model_cls.get_mamba_specs_from_config(vllm_config)
        return gdn, ple
    gdn = MambaSpec(shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
                    dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
                    block_size=-1)
    return gdn, None


def _build_model(vllm_config) -> torch.nn.Module:
    """Register the weightless cache layers under the real names and return the model stand-in.

    Names and order follow ``Qwen4ExpDecoderLayer.__init__``: ``layers.N.ple`` when ``N + 1`` is in
    ``ple_layer_ids``, then ``layers.N.linear_attn`` or ``layers.N.self_attn.attn``.
    """
    model_config = vllm_config.model_config
    text = model_config.hf_text_config
    context = vllm_config.compilation_config.static_forward_context
    context.clear()
    gdn_state, ple_state = _recurrent_states(vllm_config)
    num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = model_config.get_head_size()
    ple_layer_ids = set(getattr(text, "ple_layer_ids", None) or []) if ple_state is not None else set()
    owners = torch.nn.ModuleList()
    for index, layer_type in enumerate(text.layer_types):
        prefix = f"language_model.model.layers.{index}"
        if index + 1 in ple_layer_ids:
            context[f"{prefix}.ple"] = _RecurrentLayer(ple_state, MambaAttentionBackendEnum.SHORT_CONV)
        if layer_type == "linear_attention":
            context[f"{prefix}.linear_attn"] = _RecurrentLayer(gdn_state, MambaAttentionBackendEnum.GDN_ATTN)
        else:
            assert layer_type == "full_attention", layer_type
            attn = _AttentionLayer(f"{prefix}.self_attn.attn", num_kv_heads, head_size)
            context[attn.layer_name] = attn
            if is_flash_next_model(model_config):
                owners.append(_qsa_owner(attn))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = owners
    model.model.config = model_config.hf_config
    return model


def _make_runner(vllm_config, model) -> runner_module.HPUModelRunner:
    """An ``HPUModelRunner`` with the attributes ``__init__`` derives for a GDN hybrid (no device, no bucketing)."""
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    runner = object.__new__(runner_module.HPUModelRunner)
    runner.vllm_config = vllm_config
    runner.model_config = model_config
    runner.cache_config = vllm_config.cache_config
    runner.parallel_config = parallel_config
    runner.scheduler_config = vllm_config.scheduler_config
    runner.speculative_config = vllm_config.speculative_config
    runner.model = model
    runner.device = torch.device("meta")
    runner.dtype = model_config.dtype
    runner.kv_cache_dtype = model_config.dtype  # cache_dtype == "auto"
    runner.block_size = vllm_config.cache_config.block_size
    runner.attn_block_size = runner.block_size
    runner._original_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    count = lambda kind: model_config.get_num_layers_by_block_type(parallel_config, kind)  # noqa: E731
    runner.num_mamba_like_layers = sum(count(kind) for kind in ("mamba", "gdn_attention", "linear_attention"))
    runner.num_gdn = sum(count(kind) for kind in ("gdn_attention", "linear_attention"))
    assert runner.num_gdn > 0
    runner.use_hybrid_cache = True  # __init__ sets VLLM_USE_HYBRID_CACHE=1 when GDN layers exist
    runner.use_naive_mamba_cache_sharing = False  # and VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=0
    runner._compact_gdn_enabled = os.environ.get("VLLM_COMPACT_GDN", "1").strip().lower() in ("1", "true")
    runner.attn_backend = HPUAttentionBackendV1  # the card log: "Setting kv cache block size to 128 for CUSTOM backend"
    runner.enable_bucketing = True
    runner.bucketing_manager = SimpleNamespace(num_hpu_blocks=None, block_size=runner.block_size)
    runner.runner_only_attn_layers = set()
    runner.shared_kv_cache_layers = {}
    runner.kv_caches = []
    runner.attn_groups = []
    runner._compact_gdn_group_ids = set()
    runner._compact_gdn_group_offset = {}
    runner._compact_gdn_state_tensors = []
    runner._gdn_state_slots_per_req = 1
    runner._recurrent_spec = False
    runner._recurrent_pool_slots = 0
    runner.use_prefix_caching = bool(vllm_config.cache_config.enable_prefix_caching)
    runner.may_reinitialize_input_batch = lambda *args, **kwargs: None
    runner.profile_run = lambda **kwargs: None
    return runner


def _engine_config(geometry: Geometry, model_dir: Path):
    kwargs = dict(model=str(model_dir),
                  tokenizer=str(model_dir),
                  skip_tokenizer_init=True,
                  dtype="bfloat16",
                  trust_remote_code=True,
                  language_model_only=True,
                  enable_prefix_caching=True,
                  tensor_parallel_size=geometry.tp,
                  max_model_len=geometry.max_model_len,
                  max_num_seqs=geometry.max_num_seqs,
                  max_num_batched_tokens=8192,
                  long_prefill_token_threshold=2048)
    return EngineArgs(**kwargs).create_engine_config()


@dataclasses.dataclass
class Allocation:
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclasses.dataclass(repr=False)
class Start:
    """Everything one server start computed and (would have) allocated."""
    geometry: Geometry
    vllm_config: object
    specs: dict
    worker_log: list[str]
    runner_log: list[str]
    kv_budget: int  # free * gpu_memory_utilization * (1 - VLLM_GRAPH_RESERVED_MEM): KV cache, pad block and pool
    dummy_block_bytes: int
    pool_slots: int
    pool_bytes: int
    available: int  # what determine_available_memory returned: the attention budget
    kv_cache_config: object
    scheduler_config: object
    engine_tokens: int  # "GPU KV cache size: N tokens"
    engine_concurrency: float  # "Maximum concurrency for max_model_len tokens per request"
    runner: object
    allocations: list[Allocation]
    kv_caches: dict
    qsa_owners: list
    admitted: int = 0
    admitted_block_ids: dict = dataclasses.field(default_factory=dict)

    # -- the engine's view -------------------------------------------------------------------------
    @property
    def groups(self):
        return self.scheduler_config.kv_cache_groups

    @property
    def attention_group_ids(self) -> list[int]:
        return [gid for gid, group in enumerate(self.groups) if isinstance(group.kv_cache_spec, AttentionSpec)]

    @property
    def recurrent_group_ids(self) -> list[int]:
        return [gid for gid, group in enumerate(self.groups) if isinstance(group.kv_cache_spec, MambaSpec)]

    @property
    def runner_attention_group_id(self) -> int:
        """The one group whose block table the runner turns into slots (hpu_model_runner.py:3823, :5422)."""
        return self.runner._get_attention_group_id_for_hybrid()

    @property
    def block_size(self) -> int:
        return self.groups[self.runner_attention_group_id].kv_cache_spec.block_size

    def blocks_per_request(self, group_ids) -> int:
        """Block ids the engine charges a max_model_len request in these groups (get_max_concurrency_...)."""
        return sum(
            cdiv(self.groups[gid].kv_cache_spec.max_memory_usage_bytes(self.vllm_config),
                 self.groups[gid].kv_cache_spec.page_size_bytes) for gid in group_ids)

    # -- what the runner allocated ------------------------------------------------------------------
    @property
    def attention_layers(self) -> dict:
        return {name: cache for name, cache in self.kv_caches.items() if name.endswith(".attn")}

    @property
    def recurrent_layers(self) -> dict:
        return {name: cache for name, cache in self.kv_caches.items() if not name.endswith(".attn")}

    @property
    def total_allocated(self) -> int:
        return sum(allocation.nbytes for allocation in self.allocations)

    def group_bytes(self) -> dict[str, int]:
        """Bytes per tensor group: attention K+V, compressed key caches, key rings, and the recurrent state
        storage (every other allocation: the state tensors, or the raw buffer a PLE layer's views sit on)."""
        nbytes = lambda tensor: tensor.numel() * tensor.element_size()  # noqa: E731
        groups = {
            "attention": sum(nbytes(cache[0]) + nbytes(cache[1]) for cache in self.attention_layers.values()),
            "compressed": sum(nbytes(owner.indexer.compressed_key_cache) for owner in self.qsa_owners),
            "ring": sum(nbytes(owner.indexer.key_ring) for owner in self.qsa_owners),
        }
        groups["recurrent"] = self.total_allocated - sum(groups.values())
        return groups

    @property
    def usable_rows(self) -> int:
        """Token rows of one attention layer that a block id can address (the last kernel block is padding)."""
        rows = {cache[0].shape[0] for cache in self.attention_layers.values()}
        rows |= {cache[1].shape[0] for cache in self.attention_layers.values()}
        assert len(rows) == 1, rows
        return rows.pop() - self.runner.attn_block_size


def _record_logs(logger):
    messages: list[str] = []

    def info(msg, *args, **kwargs):
        messages.append(str(msg) % args if args else str(msg))

    return messages, patch.object(logger, "info", info)


def _admit_full_length_requests(start: Start) -> None:
    """Admit max_model_len requests like the scheduler does, until max_num_seqs or the first refusal."""
    vllm_config = start.vllm_config
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(start.scheduler_config, vllm_config)
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    manager = KVCacheManager(kv_cache_config=start.scheduler_config,
                             max_model_len=start.geometry.max_model_len,
                             scheduler_block_size=scheduler_block_size,
                             hash_block_size=hash_block_size,
                             enable_caching=True)
    length = start.geometry.max_model_len
    for index in range(start.geometry.max_num_seqs):
        params = SamplingParams(max_tokens=1)
        params.update_from_generation_config({}, eos_token_id=0)
        request = Request(request_id=f"r{index}",
                          prompt_token_ids=list(range(1 + index * length, 1 + (index + 1) * length)),
                          mm_features=None,
                          sampling_params=params,
                          pooling_params=None,
                          block_hasher=get_request_block_hasher(hash_block_size, hash_fn))
        blocks, hit, _ = manager.get_computed_blocks(request)
        assert hit == 0
        scheduled = 0
        refused = False
        while scheduled < length:
            chunk = min(scheduler_block_size, length - scheduled)
            kwargs = dict(num_new_computed_tokens=hit, new_computed_blocks=blocks) if scheduled == 0 else {}
            if manager.allocate_slots(request, chunk, **kwargs) is None:
                refused = True
                break
            request.num_computed_tokens += chunk
            scheduled += chunk
        if refused:
            break
        start.admitted += 1
        start.admitted_block_ids[request.request_id] = [list(ids) for ids in manager.get_block_ids(request.request_id)]


def start(geometry: Geometry, model_dir: Path, stock_engine_grouping: bool = False) -> Start:
    """Run one server start on CPU. ``stock_engine_grouping`` skips the platform's engine-core hook."""
    env = pytest.MonkeyPatch()
    try:
        for name in ENV_VARS:  # setenv first, so the direct os.environ writes of the hooks are undone too
            env.setenv(name, "")
            env.delenv(name)
        if geometry.pool:
            env.setenv(ENV_MODE, "1")
        if geometry.sparse_qsa:
            env.setenv("VLLM_HPU_FLASH_NEXT_QSA_MODE", "1")
        flash_next.register()
        shutil.copyfile(geometry.config, model_dir / "config.json")
        vllm_config = _engine_config(geometry, model_dir)
        cache_config = vllm_config.cache_config

        # APIServer, HpuPlatform.check_and_update_config: compact-GDN rule and pool slots (written to os.environ).
        _configure_recurrent_state(vllm_config)
        # Worker, after the model is built: block size for the backend, then the hybrid page alignment.
        with patch.object(HpuPlatform, "_find_non_ssm_backend", classmethod(lambda cls, cfg: HPUAttentionBackendV1)):
            HpuPlatform.update_block_size_for_backend(vllm_config)
        # EngineCore resolves the layout before profiling; the card log reads "Using LBNHC KV cache layout."
        cache_config.kv_cache_layout = "LBNHC"

        model = _build_model(vllm_config)
        runner = _make_runner(vllm_config, model)
        specs = runner.get_kv_cache_spec()

        worker = object.__new__(hw.HPUWorker)
        worker.model_runner = runner
        worker.vllm_config = vllm_config
        worker.model_config = vllm_config.model_config
        worker.cache_config = cache_config
        worker.speculative_config = vllm_config.speculative_config
        worker.rank = 0
        dummy_block_bytes = sum(spec.page_size_bytes for spec in specs.values())
        usable_fraction = cache_config.gpu_memory_utilization * (1 - 0.1)  # VLLM_GRAPH_RESERVED_MEM default
        free_bytes = geometry.free_bytes
        if geometry is QWEN_27B:
            free_bytes = math.ceil((QWEN_27B_USABLE_BYTES + dummy_block_bytes) / usable_fraction)

        real_zeros, real_tensor = torch.zeros, torch.tensor
        allocations: list[Allocation] = []

        def meta_zeros(*args, **kwargs):
            kwargs["device"] = "meta"
            tensor = real_zeros(*args, **kwargs)
            allocations.append(Allocation(tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size()))
            return tensor

        def cpu_tensor(*args, **kwargs):
            if kwargs.get("device") == "hpu":
                kwargs["device"] = "cpu"
            return real_tensor(*args, **kwargs)

        worker_log, worker_logging = _record_logs(hw.logger)
        with worker_logging, patch.object(hw, "is_fake_hpu", lambda: False), \
                patch.object(hw, "HabanaMemoryProfiler", MagicMock()), \
                patch.object(hw, "bind_kv_cache", lambda caches, context, out: None), \
                patch.object(torch, "zeros", meta_zeros), patch.object(torch, "tensor", cpu_tensor), \
                patch.object(torch.hpu, "mem_get_info", lambda: (free_bytes, CARD_TOTAL_BYTES)):
            available = worker.determine_available_memory()
        allocations.clear()  # the profile run's one-block dummy caches are freed before the real allocation
        kv_budget = int(free_bytes * usable_fraction)
        pool_slots = pool_slots_from_env()

        # EngineCore._initialize_kv_caches: register_all_kvcache_specs, which ends with the current platform's
        # register_custom_kv_cache_specs, then get_kv_cache_configs. Each start begins from the engine's own
        # grouping function; stock_engine_grouping leaves the HPU platform out, as if it had no hook.
        installed = kv_cache_utils._get_kv_cache_groups_uniform_page_size
        stock = getattr(installed, "__wrapped__", installed)
        with patch.object(kv_cache_utils, "_get_kv_cache_groups_uniform_page_size", stock):
            if not stock_engine_grouping:
                with patch.object(vllm.platforms, "_current_platform", HpuPlatform()):
                    register_all_kvcache_specs(vllm_config)
            kv_cache_config = get_kv_cache_configs(vllm_config, [specs], [available])[0]
        kv_cache_config.kv_cache_layout = cache_config.kv_cache_layout
        scheduler_config = generate_scheduler_kv_cache_config([kv_cache_config])
        engine_tokens, engine_concurrency = get_kv_cache_capacity(vllm_config, scheduler_config)

        # Worker: initialize_from_config -> HPUModelRunner.initialize_kv_cache, every allocation recorded.
        runner.attn_groups = [[SimpleNamespace(backend=HPUAttentionBackendV1)]
                              if isinstance(group.kv_cache_spec, AttentionSpec) else []
                              for group in kv_cache_config.kv_cache_groups]
        kv_caches: dict = {}
        runner_log, runner_logging = _record_logs(runner_module.logger)
        with runner_logging, patch.object(torch, "zeros", meta_zeros), \
                patch.object(runner_module, "bind_kv_cache", lambda caches, context, out: kv_caches.update(caches)), \
                patch.object(runner_module, "maybe_set_mamba_kv_cache_groups_ids", lambda *args, **kwargs: None):
            runner.initialize_kv_cache(kv_cache_config)

        result = Start(geometry=geometry,
                       vllm_config=vllm_config,
                       specs=specs,
                       worker_log=worker_log,
                       runner_log=runner_log,
                       kv_budget=kv_budget,
                       dummy_block_bytes=dummy_block_bytes,
                       pool_slots=pool_slots,
                       pool_bytes=pool_tensor_slots(pool_slots) * hw.pool_slot_bytes(specs) if pool_slots else 0,
                       available=available,
                       kv_cache_config=kv_cache_config,
                       scheduler_config=scheduler_config,
                       engine_tokens=engine_tokens,
                       engine_concurrency=engine_concurrency,
                       runner=runner,
                       allocations=list(allocations),
                       kv_caches=kv_caches,
                       qsa_owners=list(model.model.layers))
        _admit_full_length_requests(result)  # while the pool environment is still in place
        return result
    finally:
        env.undo()


@pytest.fixture(scope="module")
def starts(tmp_path_factory):
    """Server starts by (geometry name, stock engine grouping), run once per module."""
    cache: dict[tuple[str, bool], Start] = {}

    def get(geometry: Geometry, stock_engine_grouping: bool = False) -> Start:
        key = (geometry.name, stock_engine_grouping)
        if key not in cache:
            model_dir = tmp_path_factory.mktemp(geometry.name)
            cache[key] = start(geometry, model_dir, stock_engine_grouping)
        return cache[key]

    return get


def _ids(geometries):
    return [geometry.name for geometry in geometries]


# --------------------------------------------------------------------------------------------------
# The card's failure, from the engine's own grouping
# --------------------------------------------------------------------------------------------------


def test_weightless_layers_report_what_the_real_layers_report(tmp_path_factory):
    """The tiny checkpoint, built for real on CPU: same layer names, same order, same specs as the stand-ins."""
    from tests.unit_tests.models.flash_next.cpu_model import build_tiny_checkpoint, cpu_model

    def reported(context, vllm_config):
        specs = {}
        for name, module in context.items():
            if isinstance(module, MambaBase):
                specs[name] = module.get_kv_cache_spec(vllm_config)
            elif isinstance(module, Attention):
                specs[name] = (module.num_kv_heads, module.head_size, module.attn_type)
        return specs

    checkpoint = build_tiny_checkpoint(tmp_path_factory.mktemp("tiny_real"), fixture=TINY_CONFIG)
    with cpu_model(checkpoint) as (vllm_config, model):
        context = vllm_config.compilation_config.static_forward_context
        real = reported(context, vllm_config)
        assert len(real) == 5 and isinstance(model, flash_next.HpuQwen4ExpForConditionalGeneration)
        _build_model(vllm_config)  # replaces the context's layers with the stand-ins
        stand_in = reported(context, vllm_config)
    assert list(stand_in) == list(real)
    assert stand_in == real


def test_real_config_gives_the_per_rank_shapes_the_card_logged(starts):
    """The specs come from the real config.json and the model code: 1024 B per token and layer at TP8."""
    card = starts(FN_TP8, stock_engine_grouping=True)
    text = card.vllm_config.model_config.hf_text_config
    by_kind: dict[str, list] = {"attn": [], "gdn": [], "ple": []}
    for name, spec in card.specs.items():
        by_kind["attn" if name.endswith(".attn") else "gdn" if name.endswith(".linear_attn") else "ple"].append(spec)
    assert {kind: len(specs) for kind, specs in by_kind.items()} == {"attn": 12, "gdn": 36, "ple": 1}
    attn, gdn, ple = by_kind["attn"][0], by_kind["gdn"][0], by_kind["ple"][0]
    # QSA: max(1, num_key_value_heads // tp) heads per rank (HpuQwen4ExpQSAAttention.__init__), head_dim wide.
    assert (attn.num_kv_heads, attn.head_size) == (max(1, text.num_key_value_heads // 8), text.head_dim) == (1, 256)
    assert isinstance(attn, FullAttentionSpec) and attn.dtype == torch.bfloat16
    # "Setting attention block size to 512 tokens", "Padding mamba page size by 30.78%".
    assert attn.block_size == 512 and attn.page_size_bytes == 512 * 1024
    assert (gdn.real_page_size_bytes, gdn.page_size_bytes) == (400896, 524288)
    assert (ple.real_page_size_bytes, ple.page_size_bytes) == (184320, 524288)
    assert round(100 * (524288 - 400896) / 400896, 2) == 30.78
    assert gdn.mamba_cache_mode == ple.mamba_cache_mode == "align"


def test_card_failure_is_reproduced_from_the_stock_engine_grouping(starts):
    """Without the platform's engine-core hook the CPU start prints the card's log and sizes its allocation."""
    card = starts(FN_TP8, stock_engine_grouping=True)
    worker_log = "\n".join(card.worker_log)
    assert "Free device memory: 71.11 GiB, 65.42 GiB usable (gpu_memory_utilization=0.92)" in worker_log
    assert "24.5 MiB reserved for KV cache dummy block 58.86 GiB reserved for usable KV cache" in worker_log
    assert ("Recurrent state pool: 896 slots x 4.764 MiB per slot = 4.178 GiB reserved; "
            "54.68 GiB left for attention blocks") in worker_log
    assert "Flash-Next sparse QSA side caches: attention budget x 0.9412, ring 10.52 MiB" in worker_log

    # The engine's own grouping: the single PLE layer makes the group size 1.
    counts = {
        kind: sum(isinstance(group.kv_cache_spec, kind) for group in card.groups)
        for kind in (FullAttentionSpec, MambaSpec)
    }
    assert counts == {FullAttentionSpec: 12, MambaSpec: 37}
    assert {len(group.layer_names) for group in card.groups} == {1}
    assert card.kv_cache_config.num_blocks == 105_375 == card.available // 524_288
    # 12 * 256 attention blocks + 37 * 2 recurrent blocks per 131072-token request.
    assert card.blocks_per_request(range(len(card.groups))) == 12 * 256 + 37 * 2 == 3146
    assert f"{card.engine_tokens:,}" == "4,390,245"
    assert f"{card.engine_concurrency:.2f}" == "33.49"

    # The runner: every attention layer gets num_blocks * 4 kernel blocks (+1 padding) of 128 rows.
    k_cache = next(iter(card.attention_layers.values()))[0]
    assert tuple(k_cache.shape) == (53_952_128, 1, 256) == ((105_375 * 4 + 1) * KERNEL_BLOCK, 1, 256)
    assert k_cache.numel() * k_cache.element_size() == 27_623_489_536  # "Allocation failed for size::27623489536"
    assert ("Flash-Next QSA side caches: 12 layers, compressed_key_cache (13488032, 128), "
            "key_ring (3592, 128), total 38.6 GiB") in "\n".join(card.runner_log)
    # 53,952,128 rows against 4,390,245 tokens: 3146 / 256 = 12.29, not 12.
    assert round(k_cache.shape[0] / card.engine_tokens, 2) == round(3146 / 256, 2) == 12.29
    # 24 K/V tensors of 25.7 GiB each against 58.88 GiB.
    assert card.group_bytes()["attention"] == 24 * 27_623_489_536 > 10 * card.kv_budget
    # The recurrent pool has 896 slots for 4 groups; 37 groups run it dry after a few requests.
    assert card.admitted < card.geometry.max_num_seqs


# --------------------------------------------------------------------------------------------------
# What every start must satisfy
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=_ids(GEOMETRIES))
def test_allocations_fit_the_budget_the_worker_computed(starts, geometry):
    """Everything initialize_kv_cache and the side-cache allocation allocate is inside the KV budget."""
    case = starts(geometry)
    groups = case.group_bytes()
    assert case.total_allocated <= case.kv_budget, (
        f"{geometry.name}: {format_bytes(case.total_allocated)} allocated against a KV budget of "
        f"{format_bytes(case.kv_budget)} ({case.total_allocated / case.kv_budget:.2f}x): "
        f"{ {name: format_bytes(size) for name, size in groups.items()} }")
    # The attention blocks stay inside what the worker told the engine; the padding kernel block and the
    # compressed cache's padding rows are covered by the dummy-block reservation.
    attention_like = groups["attention"] + groups["compressed"] + groups["ring"]
    assert attention_like <= case.kv_budget - case.pool_bytes
    if geometry.pool:
        assert groups["recurrent"] == case.pool_bytes


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=_ids(GEOMETRIES))
def test_attention_rows_equal_the_engine_token_capacity(starts, geometry):
    """One attention layer has as many token rows as the engine's block pool has tokens."""
    case = starts(geometry)
    num_blocks = case.kv_cache_config.num_blocks
    assert case.usable_rows == num_blocks * case.block_size
    # The engine reports tokens after charging every group's blocks to the pool, so a layer's rows may
    # exceed the reported tokens by the recurrent groups' share of a request and by nothing else.
    attention_blocks = cdiv(geometry.max_model_len, case.block_size)
    recurrent_blocks = case.blocks_per_request(case.recurrent_group_ids)
    ratio = case.usable_rows / case.engine_tokens
    allowed = (attention_blocks + recurrent_blocks) * case.block_size / geometry.max_model_len
    assert 1 <= ratio <= allowed + 1e-6, (
        f"{geometry.name}: each attention layer has {case.usable_rows:,} token rows, {ratio:.2f}x the "
        f"{case.engine_tokens:,} tokens the engine reports (at most {allowed:.3f}x is the recurrent charge)")
    # The decode buckets and the padding ids are sized from the same count.
    kernel_blocks = num_blocks * (case.block_size // KERNEL_BLOCK)
    assert case.runner.bucketing_manager.num_hpu_blocks == kernel_blocks
    assert case.runner._PAD_BLOCK_ID == kernel_blocks
    assert case.runner._PAD_SLOT_ID == case.usable_rows


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=_ids(GEOMETRIES))
def test_scheduler_block_ids_address_allocated_rows(starts, geometry):
    """Every attention block id the scheduler hands out is read by the runner and lies inside the tensors."""
    case = starts(geometry)
    # The runner turns ONE group's block table into slots for every attention layer, so a second
    # attention group would hold block ids that address no row.
    assert case.attention_group_ids == [case.runner_attention_group_id], (
        f"{geometry.name}: the scheduler keeps {len(case.attention_group_ids)} attention block tables per "
        "request; the runner reads the first one for all attention layers")
    assert set(case.groups[case.runner_attention_group_id].layer_names) == set(case.attention_layers)
    # Layers never share storage: the same block id is a different row in every layer's own tensors.
    tensors = [tensor for cache in case.attention_layers.values() for tensor in cache[:2]]
    tensors += [owner.indexer.compressed_key_cache for owner in case.qsa_owners]
    tensors += [owner.indexer.key_ring for owner in case.qsa_owners]
    assert len({id(tensor) for tensor in tensors}) == len(tensors)

    assert case.admitted > 0
    seen: set[int] = set()
    recurrent_rows = {state.shape[0] for cache in case.recurrent_layers.values() for state in cache}
    for block_ids in case.admitted_block_ids.values():
        attention_ids = block_ids[case.runner_attention_group_id]
        assert len(attention_ids) == cdiv(geometry.max_model_len, case.block_size)
        assert all(0 < block_id and (block_id + 1) * case.block_size <= case.usable_rows for block_id in attention_ids)
        assert seen.isdisjoint(attention_ids)  # no two requests write the same rows
        seen.update(attention_ids)
        for gid in case.recurrent_group_ids:
            assert all(0 <= block_id < min(recurrent_rows) - 1 for block_id in block_ids[gid])
    if geometry.pool:
        assert recurrent_rows == {pool_tensor_slots(case.pool_slots)}
        assert case.runner._MAMBA_PAD_BLOCK_ID == pool_pad_block_id(case.pool_slots)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=_ids(GEOMETRIES))
def test_reported_concurrency_is_what_the_tensors_hold(starts, geometry):
    """The engine's "Maximum concurrency" is admitted by the scheduler and fills the attention tensors."""
    case = starts(geometry)
    attention_blocks = cdiv(geometry.max_model_len, case.block_size)
    recurrent_blocks = case.blocks_per_request(case.recurrent_group_ids)
    # What one layer's rows hold, in full-length requests.
    tensor_concurrency = case.usable_rows / (attention_blocks * case.block_size)
    assert case.engine_concurrency <= tensor_concurrency <= case.engine_concurrency * (
        1 + recurrent_blocks / attention_blocks) + 1e-6, (
            f"{geometry.name}: the engine reports {case.engine_concurrency:.2f}x, the attention tensors hold "
            f"{tensor_concurrency:.2f} requests of {geometry.max_model_len} tokens")
    # One attention block id per block of tokens, plus the engine's charge for the recurrent groups.
    assert case.engine_concurrency == case.kv_cache_config.num_blocks / (attention_blocks + recurrent_blocks)
    # The scheduler really admits what the capacity gate reads from the log (up to max_num_seqs).
    expected = min(geometry.max_num_seqs, int(case.engine_concurrency))
    assert case.admitted >= expected, (
        f"{geometry.name}: the engine reports {case.engine_concurrency:.2f}x but the scheduler admitted "
        f"{case.admitted} requests of {geometry.max_model_len} tokens")


@pytest.mark.parametrize("geometry", POOL_GEOMETRIES, ids=_ids(POOL_GEOMETRIES))
def test_pool_is_sized_for_the_groups_the_engine_made(starts, geometry):
    """The platform's slot count and the worker's bytes per slot match the engine's recurrent groups."""
    case = starts(geometry)
    recurrent_groups = len(case.recurrent_group_ids)
    assert case.pool_slots == geometry.max_num_seqs * recurrent_groups * LIVE_SLOTS_PER_REQUEST + 512, (
        f"{geometry.name}: the pool has {case.pool_slots} slots, the engine made {recurrent_groups} recurrent groups")
    # One state tensor per (spec, position in its group), each with slots + 2 rows.
    positions: dict[MambaSpec, int] = {}
    for gid in case.recurrent_group_ids:
        group = case.groups[gid]
        positions[group.kv_cache_spec] = max(positions.get(group.kv_cache_spec, 0), len(group.layer_names))
    assert case.group_bytes()["recurrent"] == pool_tensor_slots(case.pool_slots) * sum(
        count * spec.real_page_size_bytes for spec, count in positions.items())
    # The key ring has one window per recurrent state row.
    for owner in case.qsa_owners:
        assert owner.indexer.key_ring.shape[0] == pool_tensor_slots(case.pool_slots) * owner.indexer.ring_rows
        k_rows = case.kv_caches[owner.attn.layer_name][0].shape[0]
        assert owner.indexer.compressed_key_cache.shape[0] * 4 == k_rows


# --------------------------------------------------------------------------------------------------
# The numbers of the supported starts
# --------------------------------------------------------------------------------------------------


def test_flash_next_tp8_capacity(starts):
    """TP8 x 32 sequences with the card's free memory: one attention group of 12 layers."""
    case = starts(FN_TP8)
    assert [len(case.groups[gid].layer_names) for gid in case.attention_group_ids] == [12]
    assert sorted(len(case.groups[gid].layer_names) for gid in case.recurrent_group_ids) == [1, 12, 12, 12]
    assert case.available == starts(FN_TP8, stock_engine_grouping=True).available  # the worker budget is unchanged
    assert case.kv_cache_config.num_blocks == 8781 == case.available // (12 * 524_288)
    assert case.usable_rows == 8781 * 512 == 4_495_872
    assert case.blocks_per_request(range(len(case.groups))) == 256 + 4 * 2
    assert f"{case.engine_tokens:,}" == "4,359,633"
    assert f"{case.engine_concurrency:.2f}" == "33.26"
    assert case.admitted == 32
    assert case.pool_slots == 896


def test_flash_next_tp4_capacity(starts):
    """TP4 x 16 sequences: 896-token blocks of 7 kernel blocks, one attention group of 12 layers."""
    case = starts(FN_TP4)
    attention = case.groups[case.runner_attention_group_id].kv_cache_spec
    assert (attention.block_size, attention.page_size_bytes) == (896, 896 * 1024)
    assert [len(case.groups[gid].layer_names) for gid in case.attention_group_ids] == [12]
    assert case.kv_cache_config.num_blocks == case.available // (12 * 896 * 1024)
    assert case.usable_rows == case.kv_cache_config.num_blocks * 896
    assert case.pool_slots == 16 * 4 * 3 + 512 == 704
    assert case.admitted == 16


def test_flash_next_without_the_pool_sizes_the_ple_buffer_to_its_views(starts):
    """Without the pool the PLE state lives in a raw buffer under strided views: one PLE page per block."""
    case = starts(FN_TP8_NO_POOL)
    num_blocks = case.kv_cache_config.num_blocks
    ple_name = next(name for name in case.kv_caches if name.endswith(".ple"))
    ple_spec = case.specs[ple_name]
    (ple_state, ) = case.kv_caches[ple_name]
    assert tuple(ple_state.shape) == (num_blocks + 1, *ple_spec.shapes[0])
    raw = [allocation for allocation in case.allocations if len(allocation.shape) == 1]
    assert [allocation.nbytes for allocation in raw] == [(num_blocks + 1) * ple_spec.page_size_bytes]
    # 12 GDN positions of num_blocks + 1 rows each, next to that buffer.
    gdn_spec = case.specs[next(name for name in case.kv_caches if name.endswith(".linear_attn"))]
    assert case.group_bytes()["recurrent"] == (num_blocks + 1) * (12 * gdn_spec.real_page_size_bytes +
                                                                   ple_spec.page_size_bytes)


def test_tiny_keeps_its_grouping(starts):
    """The tiny model has one attention layer: the hook changes nothing for it."""
    fixed, stock = starts(TINY), starts(TINY, stock_engine_grouping=True)
    assert [group.layer_names for group in fixed.groups] == [group.layer_names for group in stock.groups]
    assert fixed.kv_cache_config.num_blocks == stock.kv_cache_config.num_blocks
    assert [allocation.shape for allocation in fixed.allocations] == [a.shape for a in stock.allocations]
    assert len(fixed.attention_group_ids) == 1 and len(fixed.recurrent_group_ids) == 4


def test_27b_reproduces_its_card_log(starts):
    """The 27B without the pool keeps the numbers of its Run 1 log, hook or no hook."""
    fixed, stock = starts(QWEN_27B), starts(QWEN_27B, stock_engine_grouping=True)
    for case in (fixed, stock):
        assert [len(group.layer_names) for group in case.groups] == [16, 16, 16, 16]
        assert case.kv_cache_config.num_blocks == 5293
        assert f"{case.engine_tokens:,}" == "2,647,954"
        assert f"{case.engine_concurrency:.2f}" == "20.20"
        assert case.admitted == 20
    assert [allocation.shape for allocation in fixed.allocations] == [a.shape for a in stock.allocations]
