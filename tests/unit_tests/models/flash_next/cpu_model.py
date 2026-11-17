# SPDX-License-Identifier: Apache-2.0
"""Shared CPU harness for the Flash-Next whole-model tests (M0-T7a).

Plain functions (no pytest import): ``conftest.py`` and the parity tasks T7b (prefill),
T7c (decode) and T7d (TP) import these names, so the signatures below are a contract.

``cpu_model()`` builds ``HpuQwen4ExpForConditionalGeneration`` from the tiny checkpoint
through vLLM's real loader on world 1 (gloo), then patches what the CPU cannot run:

* the HPU paged-attention backend is replaced by :class:`SdpaAttention` (the HPU QSA
  layer keeps its ``.attn`` child name, so ``get_kv_cache_spec`` and the class checks
  still see a real attention layer),
* the ``MambaBase`` state tensors (GDN layers, PLE layer) are bound by hand, because
  the HPU model runner is the component that allocates them on a real engine.

The seams (``GatedResidual``, ``Qwen4ExpQSAAttention``) are wired by
``vllm_gaudi.models.qwen4_exp.apply_flash_next_patches()``, which ``register()`` calls.
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")  # before vLLM is imported: fn.register() is called by hand
os.environ.setdefault("TMPDIR", "/tmp")  # the gloo init file must not sit on a sandboxed TMPDIR

import contextlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel, init_distributed_environment, initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.model_executor.layers.mamba.abstract import MambaBase  # noqa: E402

import vllm_gaudi.models.qwen4_exp as fn  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention, QsaMode  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_side_caches import allocate_side_caches  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa_sparse_ops import selection_mask  # noqa: E402

_MAKER = Path(__file__).resolve().parents[4] / "tools" / "flash_next" / "make_tiny_checkpoint.py"

DENSE_EXACT_MAX_TOKENS = 2048  # dense attention equals QSA while every query sees <= 512 complete 4-token blocks


def dense_reference_prompts(reference: dict) -> list[dict]:
    """The reference prompts a dense-mode (QsaMode.DENSE) test may use: prompt plus the generated tokens
    fit the exact range."""
    limit = DENSE_EXACT_MAX_TOKENS - int(reference["max_new_tokens"])
    return [p for p in reference["prompts"] if len(p["prompt_ids"]) <= limit]


def build_tiny_checkpoint(out_dir: Path, fixture: Path | None = None) -> Path:
    """Write the tiny Flash-Next checkpoint into ``out_dir`` (a subdirectory is created) and return it.

    Runs ``tools/flash_next/make_tiny_checkpoint.py --out <out_dir> --skip-tokenizer`` as a
    subprocess with the current environment: the builder drives transformers, whose
    ``CONFIG_MAPPING["qwen4_exp"]`` entry vLLM's ``create_engine_config()`` replaces
    in-process, so builder and engine config must not share a process. When ``fixture`` is
    given, passes ``--fixture <path>`` to build from that config instead of the default
    ``tiny_config.json``.

    Writes ``config.json``, ``model.safetensors`` (132 tensors, 134,454,704 bytes: bf16
    weights plus the int64 PLE index tables) and ``reference.json`` (the deterministic HF
    CPU oracle, byte-equal to the committed ``tests/unit_tests/models/flash_next/reference.json``
    only for the default fixture; a non-default fixture regenerates its own reference).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_MAKER), "--out", str(out_dir), "--skip-tokenizer"]
    if fixture is not None:
        cmd += ["--fixture", str(fixture)]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return out_dir


def to_fp32(model: torch.nn.Module) -> None:
    """Convert a built model to fp32, rebuilding every rotary cache.

    ``model.float()`` casts parameters and buffers, which would quantize the RoPE
    ``cos_sin_cache`` built from bf16 inputs; HyperConnection parameters are always bf16
    upstream, so ``Qwen4ExpQSAAttention`` also rejects a non-bf16 engine config and an fp32
    engine config cannot be built directly.
    """
    model.float()
    for module in model.modules():
        if hasattr(module, "_compute_cos_sin_cache") and hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache = module._compute_cos_sin_cache().float()


def bind_cpu_states(model: torch.nn.Module, num_slots: int, fp32: bool, max_model_len: int = 512) -> None:
    """Bind hand-made cache tensors on every stateful module.

    Each ``MambaBase`` module (the 3 GDN layers, the PLE layer) gets
    ``m.kv_cache = tuple(zeros((num_slots, *shape)) for shape in m.get_state_shape())``,
    its own dtype per ``m.get_state_dtype()`` (fp32 when the model was converted
    with :func:`to_fp32`): the GDN layers get ``(num_slots, 3, 1280)`` and
    ``(num_slots, 48, 16, 16)``, the PLE layer ``(num_slots, 9, 512)``.

    In sparse QSA mode each QSA layer's indexer also gets its two side caches
    (``compressed_key_cache`` of ``max_model_len // compress_ratio`` block rows and a
    ``key_ring`` of ``num_slots`` raw-key rows) from ``allocate_side_caches``; the
    model runner allocates these on a real engine.
    """
    for module in model.modules():
        if isinstance(module, MambaBase):
            module.kv_cache = tuple(
                torch.zeros((num_slots, *shape), dtype=torch.float32 if fp32 else dtype)
                for shape, dtype in zip(module.get_state_shape(), module.get_state_dtype()))
        if isinstance(module, HpuQwen4ExpQSAAttention) and module.qsa_mode is QsaMode.SPARSE:
            config = module.config
            compressed, ring = allocate_side_caches(
                key_cache_rows=max_model_len,
                gdn_state_rows=num_slots,
                compress_ratio=int(config.indexer_compress_ratio),
                index_head_dim=int(config.indexer_head_dim),
                num_speculative_tokens=0,
                device=torch.device("cpu"),
                dtype=torch.float32 if fp32 else torch.bfloat16,
            )
            module.indexer.compressed_key_cache = compressed
            module.indexer.key_ring = ring


def reset_cpu_states(model: torch.nn.Module) -> None:
    """Zero every state tensor bound by :func:`bind_cpu_states` (state slot indices keep their meaning)."""
    for module in model.modules():
        for state in getattr(module, "kv_cache", None) or ():
            state.zero_()
        indexer = getattr(module, "indexer", None)
        for cache in (getattr(indexer, "compressed_key_cache", None), getattr(indexer, "key_ring", None)):
            if cache is not None:
                cache.zero_()


class SdpaAttention(torch.nn.Module):
    """Stands in for the HPU paged-attention backend of one QSA layer: causal SDPA.

    ``forward(query, key, value)`` takes the packed projections of a single call,
    ``(num_tokens, num_heads * head_dim)`` / ``(num_tokens, num_kv_heads * head_dim)``,
    and returns ``(num_tokens, num_heads * head_dim)``: causal attention over the tokens
    of that call, with GQA by ``repeat_interleave`` of the KV heads (dense mode, the same
    stand-in ``test_qsa_dense.py`` uses for one layer).
    """

    def __init__(self, qsa_layer: torch.nn.Module) -> None:
        super().__init__()
        self.num_heads = qsa_layer.num_heads
        self.num_kv_heads = qsa_layer.num_kv_heads
        self.head_dim = qsa_layer.head_dim
        self.scaling = qsa_layer.scaling

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        num_tokens = query.shape[0]
        qh = query.view(num_tokens, self.num_heads, self.head_dim).transpose(0, 1)
        kh = key.view(num_tokens, self.num_kv_heads, self.head_dim).transpose(0, 1)
        vh = value.view(num_tokens, self.num_kv_heads, self.head_dim).transpose(0, 1)
        kh = kh.repeat_interleave(self.num_heads // self.num_kv_heads, dim=0)
        vh = vh.repeat_interleave(self.num_heads // self.num_kv_heads, dim=0)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=self.scaling)
        return out.transpose(0, 1).reshape(num_tokens, -1)


class SparseSdpaAttention(torch.nn.Module):
    """CPU stand-in for the HPU sparse-QSA attention (the M2-T5c wrapper seam).

    Like :class:`CachedSdpaAttention` it keeps one K/V cache per slot (set ``.slot``
    before each forward), appending the call's K/V, but it attends only over the
    positions ``selected`` picks for each query: ``forward(query, key, value,
    selected, positions, metadata)`` builds a boolean mask from
    ``selection_mask(selected, L)`` over the whole cached K/V (``L`` = tokens seen so
    far) and runs SDPA with it. The selection already encodes causality, so no causal
    mask is applied; inverting the mask is the parity negative control.
    """

    def __init__(self, qsa_layer: torch.nn.Module) -> None:
        super().__init__()
        self.num_heads = qsa_layer.num_heads
        self.num_kv_heads = qsa_layer.num_kv_heads
        self.head_dim = qsa_layer.head_dim
        self.scaling = qsa_layer.scaling
        self.slot = 1
        self.cache = {}  # slot -> (K, V), each (tokens, kv_heads * head_dim)

    def reset(self) -> None:
        self.cache = {}

    def forward(self, query, key, value, selected, positions, metadata):
        n = query.shape[0]
        if self.slot in self.cache:
            k0, v0 = self.cache[self.slot]
            key, value = torch.cat([k0, key]), torch.cat([v0, value])
        self.cache[self.slot] = (key, value)
        L = key.shape[0]
        mask = selection_mask(selected, L)
        qh = query.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        kh = key.view(L, self.num_kv_heads, self.head_dim).transpose(0, 1)
        vh = value.view(L, self.num_kv_heads, self.head_dim).transpose(0, 1)
        kh = kh.repeat_interleave(self.num_heads // self.num_kv_heads, dim=0)
        vh = vh.repeat_interleave(self.num_heads // self.num_kv_heads, dim=0)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=self.scaling)
        return out.transpose(0, 1).reshape(n, -1)


def use_sdpa_attention(model: torch.nn.Module) -> int:
    """Replace the ``.attn`` child of every ``HpuQwen4ExpQSAAttention`` with :class:`SdpaAttention`.

    The HPU paged-attention backend does not run on CPU. In sparse QSA mode the layer's
    ``.sparse_attn`` seam is also filled with :class:`SparseSdpaAttention` (the HPU builds
    ``HpuQsaSparseAttention`` in ``__init__``; on CPU that seam is installed by the
    harness). Returns the number of layers replaced (1 on the tiny checkpoint,
    ``layers[3].self_attn``).
    """
    replaced = 0
    for module in model.modules():
        if isinstance(module, HpuQwen4ExpQSAAttention):
            module.attn = SdpaAttention(module)
            if module.qsa_mode is QsaMode.SPARSE:
                module.sparse_attn = SparseSdpaAttention(module)
            replaced += 1
    return replaced


def eos_token_id(vllm_config) -> int:
    """The checkpoint's eos token id as a plain int (the config may hold a list of them)."""
    eos = vllm_config.model_config.hf_text_config.eos_token_id
    return eos[0] if isinstance(eos, list) else eos


def prefill_metadata(num_tokens: int,
                     slot: int = 1,
                     *,
                     load_slot: int | None = None,
                     store_slot: int | None = None,
                     has_initial_state: bool = False,
                     context_len: int = 0,
                     block_size: int = 16) -> SimpleNamespace:
    """The HPU GDN forward metadata for one prefill row in state slot ``slot`` (int64 tensors).

    ``load_indices_tensor``/``store_indices_tensor`` are ``(1,)`` = ``[slot]``,
    ``query_start_loc_p`` is ``(2,)`` int32 ``[0, num_tokens]``, ``has_initial_states_p``
    is ``(1,)`` int32 ``[0]``: the single-row prefill the runner builds for a fresh prompt.

    ``load_slot``/``store_slot`` override the two indices separately (a chunked prefill under
    prefix caching continues from the load slot and writes the store slot); ``slot`` keeps
    covering both when they are omitted. ``has_initial_state`` sets ``has_initial_states_p``
    to ``[1]``, which is what tells the GDN/PLE layers to read the load slot at all.

    The sparse-QSA seam fields the indexer reads: ``slot_mapping`` is the attention slot of
    each token (``arange(context_len, context_len + num_tokens)``, so slot == position for a
    growing sequence), ``context_lens_tensor`` the ``context_len`` of the row, ``block_list``
    ``None`` (no prefix-cache block table, so the visible keys are the compressed-cache
    prefix) and ``block_size`` the KV block size.
    """
    load = slot if load_slot is None else load_slot
    store = slot if store_slot is None else store_slot
    return SimpleNamespace(
        is_prompt=True,
        load_indices_tensor=torch.tensor([load]),
        store_indices_tensor=torch.tensor([store]),
        query_start_loc_p=torch.tensor([0, num_tokens], dtype=torch.int32),
        has_initial_states_p=torch.tensor([1 if has_initial_state else 0], dtype=torch.int32),
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        dflash_full_query=False,
        slot_mapping=torch.arange(context_len, context_len + num_tokens, dtype=torch.int64),
        context_lens_tensor=torch.tensor([context_len], dtype=torch.int64),
        block_list=None,
        block_size=block_size,
    )


def ple_kwargs(num_tokens: int, eos: int, context=None) -> dict:
    """Keyword arguments for the model forward's PLE inputs on one prefill row.

    ``query_start_loc``: ``(2,)`` int32 ``[0, num_tokens]``. ``ngram_context``: ``(1, 2)``
    int32, filled with ``eos`` (the runner's no-context padding) or the two ids of
    ``context``: the left context the n-gram embedding pads the first tokens with.
    """
    if context is None:
        ngram_context = torch.full((1, 2), eos, dtype=torch.int32)
    else:
        ngram_context = torch.tensor([list(context)], dtype=torch.int32)
    return {
        "query_start_loc": torch.tensor([0, num_tokens], dtype=torch.int32),
        "ngram_context": ngram_context,
    }


def fp32_logits(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """fp32 logits ``(num_tokens, 248320)`` from hidden states ``(num_tokens, 128)``.

    ``hidden.float() @ lm_head.weight.float().T``: ``compute_logits`` returns bf16 even
    after :func:`to_fp32`, and bf16 ties flip greedy picks on the tiny model's flat logits.
    """
    return hidden.float() @ model.language_model.lm_head.weight.float().T


@contextlib.contextmanager
def cpu_model(ckpt_dir: Path,
              fp32: bool = False,
              num_slots: int = 4,
              qsa_mode: QsaMode | None = None,
              max_model_len: int = 512):
    """Build the tiny Flash-Next model on CPU and yield ``(vllm_config, model)``.

    Registers the plugin (``VLLM_PLUGINS`` is empty, so ``register()`` is called by hand),
    builds an engine config for ``ckpt_dir`` (bf16, eager, ``max_model_len`` tokens, 4
    sequences), initializes a world-1 gloo group inside ``set_current_vllm_config`` and
    loads the checkpoint with ``vllm.model_executor.model_loader.get_model``. Then, when
    ``fp32`` is set, converts with :func:`to_fp32`, and always binds the states
    (:func:`bind_cpu_states`, ``num_slots`` slots, side caches sized to ``max_model_len``)
    and swaps in the SDPA attention (:func:`use_sdpa_attention`).

    ``qsa_mode`` (an int-enum, e.g. ``QsaMode.SPARSE``) sets ``VLLM_HPU_FLASH_NEXT_QSA_MODE``
    for the duration of the build (envs is read lazily, so the layers pick it up), which
    builds sparse layers, binds their indexer side caches and installs the sparse-attention
    stand-in on the ``.sparse_attn`` seam. Use ``max_model_len=4096`` for the 2600-token
    sparse-prefill prompts (the compressed cache has one row per 4 tokens).

    The distributed group is destroyed on exit, always. Only one ``cpu_model`` may be open
    at a time: a second ``init_distributed_environment`` asserts "already initialized".
    A build takes a few seconds.
    """
    fn.register()
    prev_mode = os.environ.get("VLLM_HPU_FLASH_NEXT_QSA_MODE")
    if qsa_mode is not None:
        os.environ["VLLM_HPU_FLASH_NEXT_QSA_MODE"] = str(int(qsa_mode))
    vllm_config = EngineArgs(model=str(ckpt_dir),
                             tokenizer=str(ckpt_dir),
                             skip_tokenizer_init=True,
                             dtype="bfloat16",
                             enable_prefix_caching=False,
                             max_model_len=max_model_len,
                             max_num_seqs=4,
                             enforce_eager=True,
                             language_model_only=True).create_engine_config()
    fd, init_file = tempfile.mkstemp(dir="/tmp")
    os.close(fd)
    with set_current_vllm_config(vllm_config):
        try:
            init_distributed_environment(world_size=1,
                                         rank=0,
                                         distributed_init_method=f"file://{init_file}",
                                         local_rank=0,
                                         backend="gloo")
            initialize_model_parallel(1, 1)
            from vllm.model_executor.model_loader import get_model
            model = get_model(vllm_config=vllm_config)
            if fp32:
                to_fp32(model)
            bind_cpu_states(model, num_slots, fp32, max_model_len)
            use_sdpa_attention(model)
            yield vllm_config, model
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(init_file)
            if prev_mode is None:
                os.environ.pop("VLLM_HPU_FLASH_NEXT_QSA_MODE", None)
            else:
                os.environ["VLLM_HPU_FLASH_NEXT_QSA_MODE"] = prev_mode
