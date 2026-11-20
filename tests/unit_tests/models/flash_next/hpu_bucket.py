# SPDX-License-Identifier: Apache-2.0
"""CPU stand-ins for the HPU runner's padded buckets and for the HPU sparse-QSA wrapper's surroundings.

Plain functions (no pytest import), next to ``cpu_model.py``. Two things live here.

**The wrapper's surroundings.** ``HpuQsaSparseAttention`` (``vllm_gaudi/models/qwen4_exp/qsa_hpu_attention.py``)
is the code that runs sparse QSA on a card; the other CPU tests replace it with ``SparseSdpaAttention``, so its
``forward`` never ran anywhere before a card. :func:`install_hpu_sparse_wrapper` puts the REAL wrapper on every
QSA layer and gives it CPU stand-ins for the three HPU-only things it touches:

* ``attn.impl`` -> :class:`FakeHpuAttentionImpl` (``do_kv_cache_update``, ``common_attention_args``, ``scale``,
  ``prefill_impl``, ``k_cache`` / ``v_cache``),
* ``attn.kv_cache`` -> the ``(key_cache, value_cache, None, None)`` tuple, bound the way the runner binds it
  (``vllm_gaudi/ops/hpu_attention.py``: ``self_kv_cache = self.kv_cache``),
* ``vllm_gaudi.extension.ops.prompt_attention`` and ``hpu_attn._set_fetch_by_id`` -> :func:`fake_ops_module` /
  :func:`fake_hpu_attn_module`, placed in ``sys.modules`` (both real modules import ``habana_frameworks``).

The fake ``prompt_attention`` follows ``vllm_gaudi/extension/ops.py`` (``_include_past`` puts the context
blocks, fetched by id, in front of the chunk's keys; the bias is added to the scores; fp32 softmax).

**The runner's layout.** :func:`bucket_prefill` and :func:`bucket_decode` build the tensors
``HPUModelRunner._form_prefill_batch`` / ``_create_decode_input_data`` build for this model: token ids padded
with ``-1``, prefill slots padded with ``-1``, the padding mask in the model dtype, ``query_start_loc_p`` as the
cumulative REAL lengths, a decode batch padded to its bucket with rows on the pad block, the pad slots and the
pad state row. The positions come in the runner's two layouts:

* ``mrope=False``, the production setting. ``--language-model-only`` makes the engine strip the M-RoPE keys
  (``_strip_qwen4_exp_mrope``), so ``uses_mrope`` is False, the QSA layer builds the plain ``RotaryEmbedding``
  and the runner sends ``[bucket_rows, bucket]`` prompt positions padded with ``-1`` and ``[batch, 1]`` decode
  positions with ``0`` on the padding rows. This is also what ``cpu_model`` builds.
* ``mrope=True``, a server started without ``--language-model-only``: ``uses_mrope`` is True, the QSA layer
  builds ``MRotaryEmbedding`` (:func:`use_mrope_rotary` swaps it in) and the runner sends ``[3, N]`` positions
  with ``-1`` on every padding token, in prompts and in decode steps.

Layout of one test sequence (the convention of ``cpu_model.prefill_metadata``): token position ``p`` lives in
attention slot ``p`` (kernel block ``p // block_size``). The LAST kernel block is the pad block and the LAST
state row is the pad state row, as on a card (``num_blocks + 1`` rows, ``_PAD_BLOCK_ID`` / ``_MAMBA_PAD_BLOCK_ID``).
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_gaudi.models.qwen4_exp.qsa import HpuQwen4ExpQSAAttention
from vllm_gaudi.models.qwen4_exp.qsa_hpu_attention import HpuQsaSparseAttention

OPS_MODULE = "vllm_gaudi.extension.ops"
HPU_ATTN_MODULE = "vllm_gaudi.attention.backends.hpu_attn"
PROMPT_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 2944, 4096)  # docs/design/flash-next/plan.md
TINY_CONFIG = Path(__file__).resolve().parent / "tiny_config.json"


def prompt_bucket(num_tokens: int) -> int:
    """The smallest prompt query bucket that holds ``num_tokens``."""
    return next(b for b in PROMPT_BUCKETS if b >= num_tokens)


class FakeHpuAttentionImpl:
    """The part of ``HPUAttentionImpl`` the sparse wrapper uses, on CPU tensors."""

    prefill_impl = "naive_impl"

    def __init__(self, num_heads: int, num_kv_heads: int, head_size: int, scale: float) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = scale
        self.k_cache = SimpleNamespace(fetch_from_cache=self._fetch)
        self.v_cache = SimpleNamespace(fetch_from_cache=self._fetch)
        self.updates = 0

    @staticmethod
    def _fetch(cache: torch.Tensor, blocks: torch.Tensor, **_: object) -> torch.Tensor:
        """``VLLMKVCache.fetch_from_cache`` by id; a ``-1`` (the runner's context padding) reads the pad block."""
        blocks = blocks.long()
        return cache.index_select(0, torch.where(blocks < 0, torch.full_like(blocks, cache.shape[0] - 1), blocks))

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping) -> None:
        """``HPUAttentionImpl.do_kv_cache_update``: one ``index_copy_`` per cache at the flat slots.

        A ``-1`` slot (the runner's prefill padding) lands on the pad block's last slot.
        """
        key_cache, value_cache, _, _ = kv_cache
        slots = slot_mapping.flatten().long()
        slots = torch.where(slots < 0, torch.full_like(slots, key_cache.shape[0] - 1), slots)
        key_cache.index_copy_(0, slots, key.reshape(-1, self.num_kv_heads, self.head_size).to(key_cache.dtype))
        value_cache.index_copy_(0, slots, value.reshape(-1, self.num_kv_heads, self.head_size).to(value_cache.dtype))
        self.updates += 1

    def common_attention_args(self,
                              block_list=None,
                              key_cache=None,
                              value_cache=None,
                              block_size=None,
                              k_scales=None,
                              v_scales=None) -> dict:
        return {
            "scale": self.scale,
            "keys_fetch_func": self.k_cache.fetch_from_cache,
            "values_fetch_func": self.v_cache.fetch_from_cache,
            "block_list": block_list,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "block_size": block_size,
            "k_scales": k_scales,
            "v_scales": v_scales,
        }


class FakeHpuAttention:
    """The QSA layer's ``Attention`` child as the wrapper sees it: ``.impl`` and the bound ``.kv_cache`` tuple.

    Not an ``nn.Module``: ``cpu_model.reset_cpu_states`` zeroes every module's ``kv_cache`` entries and this
    tuple carries the two ``None`` scale slots of the runner's ``(key, value, k_scales, v_scales)``.
    """

    def __init__(self, qsa_layer: torch.nn.Module, cache_slots: int, dtype: torch.dtype) -> None:
        self.impl = FakeHpuAttentionImpl(qsa_layer.num_heads, qsa_layer.num_kv_heads, qsa_layer.head_dim,
                                         qsa_layer.scaling)
        shape = (cache_slots, qsa_layer.num_kv_heads, qsa_layer.head_dim)
        self.kv_cache = (torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype), None, None)

    def reset(self) -> None:
        self.kv_cache[0].zero_()
        self.kv_cache[1].zero_()


def _prompt_attention(impl, *, query, key, value, is_causal, attn_bias, position_bias, valid_seq_lengths, scale,
                      keys_fetch_func, values_fetch_func, block_list, key_cache, value_cache, block_size, **_):
    """``ops.prompt_attention`` for the sparse wrapper's call: ``[rows, T, H, D]`` queries over context + chunk."""
    assert impl == FakeHpuAttentionImpl.prefill_impl
    assert not is_causal and position_bias is None and valid_seq_lengths is None
    if block_list is not None:  # ops._include_past
        past_k = keys_fetch_func(key_cache.unflatten(0, (-1, block_size)), blocks=block_list)
        past_v = values_fetch_func(value_cache.unflatten(0, (-1, block_size)), blocks=block_list)
        key = torch.cat((past_k.reshape(key.size(0), -1, past_k.shape[2], past_k.shape[3]), key), dim=1)
        value = torch.cat((past_v.reshape(value.size(0), -1, past_v.shape[2], past_v.shape[3]), value), dim=1)
    repeat = query.shape[2] // key.shape[2]
    q = query.transpose(1, 2)  # [rows, H, T, D]
    k = key.repeat_interleave(repeat, dim=2).transpose(1, 2)  # [rows, H, cols, D]
    v = value.repeat_interleave(repeat, dim=2).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores + attn_bias.to(scores.dtype)  # [rows, 1, T, cols]
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, v).transpose(1, 2)  # [rows, T, H, D]


def fake_ops_module() -> ModuleType:
    module = ModuleType(OPS_MODULE)
    module.prompt_attention = _prompt_attention
    return module


def fake_hpu_attn_module() -> ModuleType:
    module = ModuleType(HPU_ATTN_MODULE)

    def _set_fetch_by_id(kv_cache, value: bool) -> None:
        kv_cache._fetch_by_id = value

    module._set_fetch_by_id = _set_fetch_by_id
    return module


def install_hpu_sparse_wrapper(model: torch.nn.Module, monkeypatch, cache_slots: int, dtype=torch.float32) -> list:
    """Put the real ``HpuQsaSparseAttention`` on every QSA layer of ``model`` and return the fake attention children.

    ``cache_slots`` is the flat K/V row count (a multiple of the kernel block size; its last block is the pad
    block). ``monkeypatch`` places the two HPU-only modules in ``sys.modules`` for the duration of the test.
    """
    monkeypatch.setitem(sys.modules, OPS_MODULE, fake_ops_module())
    monkeypatch.setitem(sys.modules, HPU_ATTN_MODULE, fake_hpu_attn_module())
    attns = []
    for module in model.modules():
        if isinstance(module, HpuQwen4ExpQSAAttention):
            attn = FakeHpuAttention(module, cache_slots, dtype)
            module.sparse_attn = HpuQsaSparseAttention(attn)
            attns.append(attn)
    return attns


def qsa_layers(model: torch.nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, HpuQwen4ExpQSAAttention)]


def use_mrope_rotary(model: torch.nn.Module, fp32: bool = True) -> None:
    """Give every QSA layer the ``MRotaryEmbedding`` a server without ``--language-model-only`` builds.

    ``cpu_model`` builds with ``language_model_only=True``, where the engine strips ``mrope_section`` and the
    layer gets the plain ``RotaryEmbedding``; this rebuilds the rotary embedding from the fixture's own
    ``rope_parameters`` (the cache recomputed in fp32, as ``cpu_model.to_fp32`` does).
    """
    rope_parameters = json.loads(TINY_CONFIG.read_text(encoding="utf-8"))["text_config"]["rope_parameters"]
    for layer in qsa_layers(model):
        rotary = get_rope(head_size=layer.head_dim,
                          max_position=layer.config.max_position_embeddings,
                          rope_parameters=dict(rope_parameters))
        assert getattr(rotary, "mrope_section", None), type(rotary)
        if fp32:
            rotary.cos_sin_cache = rotary._compute_cos_sin_cache().float()
        layer.rotary_emb = rotary


def clamp_rotary_padding(model: torch.nn.Module, monkeypatch) -> None:
    """Let the CPU rotary kernel take the ``-1`` positions of a padded bucket.

    The CPU ``RotaryEmbedding`` kernel reads ``cos_sin_cache[position]`` unchecked and wants int64; the HPU one
    is a different kernel. The rotation of a padding row is a don't-care value, so the stand-in rotates it at
    position 0. Only the rotary embedding's own kernel call is wrapped: the indexer still sees the ``-1``.
    """
    for layer in qsa_layers(model):
        rotary = layer.rotary_emb
        if getattr(rotary, "mrope_section", None):
            continue  # MRotaryEmbedding.forward_native indexes the cache in torch; -1 wraps
        kernel = rotary._forward_method

        def forward(positions, query, key=None, _kernel=kernel, **kwargs):
            # Flattened as HPURotaryEmbedding.prepare_cos_sin flattens the [rows, bucket] positions.
            return _kernel(positions.long().clamp_min(0).flatten(), query, key, **kwargs)

        monkeypatch.setattr(rotary, "_forward_method", forward)


def bucket_prefill(token_ids,
                   *,
                   slot: int,
                   eos: int,
                   bucket: int | None = None,
                   mrope: bool = True,
                   context_len: int = 0,
                   ngram_context=None,
                   has_initial_state: bool | None = None,
                   block_size: int = 128,
                   mask_dtype: torch.dtype = torch.bfloat16) -> SimpleNamespace:
    """One prefill row of ``len(token_ids)`` real tokens in a ``bucket``-token bucket, as the HPU runner lays it out.

    ``bucket=None`` is the unpadded row (``bucket == len``); with ``mrope=False`` that is the layout of the other
    CPU tests, used as the oracle. Returns a namespace with ``input_ids`` ``[bucket]`` (padding ``-1``; the model
    clamps it to id 0), ``positions`` (``[3, bucket]`` int32 under M-RoPE, ``[1, bucket]`` without; ``-1`` on the
    padding), ``metadata`` and ``ple`` (the two PLE kwargs: every padded row is one "request" of ``bucket``
    tokens, ``flash_next_inputs.build_ple_inputs``).
    """
    length = len(token_ids)
    bucket = length if bucket is None else bucket
    if bucket < length:
        raise ValueError(f"bucket {bucket} is smaller than the {length} tokens")
    pad = bucket - length
    input_ids = torch.tensor(list(token_ids) + [-1] * pad, dtype=torch.int32)
    real = torch.arange(context_len, context_len + length, dtype=torch.int32)
    line = torch.cat((real, torch.full((pad, ), -1, dtype=torch.int32)))
    # [3, bucket] under M-RoPE; [1, bucket] without (int64 here: the CPU rotary kernel takes no int32).
    positions = line.reshape(1, -1).expand(3, -1).contiguous() if mrope else line.long().reshape(1, -1)
    padding_mask = torch.zeros(bucket, 1, dtype=mask_dtype)
    padding_mask[:length] = 1.0
    n_ctx_blocks = -(-context_len // block_size)
    if ngram_context is None:
        context = torch.full((1, 2), eos, dtype=torch.int32)
    else:
        context = torch.tensor([list(ngram_context)], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prompt=True,
        seq_lens_tensor=torch.tensor([length], dtype=torch.int32),
        context_lens_tensor=torch.tensor([context_len], dtype=torch.int32),
        slot_mapping=torch.cat((real.long(), torch.full((pad, ), -1, dtype=torch.int64))).view(1, bucket),
        block_list=torch.arange(n_ctx_blocks, dtype=torch.int32) if n_ctx_blocks else None,
        block_size=block_size,
        block_table=None,
        load_indices_tensor=torch.tensor([slot], dtype=torch.int32),
        store_indices_tensor=torch.tensor([slot], dtype=torch.int32),
        query_start_loc_p=torch.tensor([0, length], dtype=torch.int32),
        has_initial_states_p=torch.tensor([int(context_len > 0 if has_initial_state is None else has_initial_state)],
                                          dtype=torch.int32),
        # The runner always sends the mask (all ones without padding).
        padding_mask_flat=padding_mask,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        dflash_full_query=False,
        lattice_store_differs=False,
    )
    ple = {"query_start_loc": torch.tensor([0, bucket], dtype=torch.int32), "ngram_context": context}
    return SimpleNamespace(input_ids=input_ids, positions=positions, metadata=metadata, ple=ple, length=length,
                           bucket=bucket)


def bucket_decode(tokens,
                  positions,
                  slots,
                  ngram_contexts,
                  *,
                  batch: int,
                  eos: int,
                  pad_state_slot: int,
                  pad_block: int,
                  mrope: bool = True,
                  block_size: int = 128,
                  max_blocks: int = 32) -> SimpleNamespace:
    """A single-token decode batch of ``len(tokens)`` requests padded to ``batch`` rows, as the HPU runner lays it out.

    Request ``i`` decodes ``tokens[i]`` at ``positions[i]`` with its state in slot ``slots[i]`` and the two
    preceding tokens ``ngram_contexts[i]``. The padding rows carry token 0, position ``-1`` (M-RoPE; 0 without),
    a block table full of ``pad_block``, slots inside the pad block, the pad state row and a zero length in
    ``query_start_loc_p``. ``block_table`` is int32 ``[batch, max_blocks]`` (``flash_next_inputs.build_block_table``:
    the request's blocks, then ``pad_block``).
    """
    real = len(tokens)
    if batch < real:
        raise ValueError(f"batch {batch} is smaller than the {real} requests")
    pad = batch - real
    input_ids = torch.tensor(list(tokens) + [0] * pad, dtype=torch.int32)
    line = torch.tensor(list(positions) + [-1 if mrope else 0] * pad, dtype=torch.int32)
    pos = line.reshape(1, -1).expand(3, -1).contiguous() if mrope else line.long().reshape(batch, 1)
    pad_slot_base = pad_block * block_size
    slot_mapping = torch.tensor(list(positions) + [pad_slot_base + i % block_size for i in range(pad)],
                                dtype=torch.int64).view(batch, 1)
    block_table = torch.full((batch, max_blocks), pad_block, dtype=torch.int32)
    for row, position in enumerate(positions):
        n = position // block_size + 1
        block_table[row, :n] = torch.arange(n, dtype=torch.int32)
    state = torch.tensor(list(slots) + [pad_state_slot] * pad, dtype=torch.int32)
    lengths = [1] * real + [0] * pad
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    metadata = SimpleNamespace(
        is_prompt=False,
        seq_lens_tensor=torch.tensor(lengths, dtype=torch.int32),
        context_lens_tensor=None,
        slot_mapping=slot_mapping,
        block_list=None,
        block_size=block_size,
        block_table=block_table,
        load_indices_tensor=state,
        store_indices_tensor=state.clone(),
        query_start_loc_p=torch.tensor(starts, dtype=torch.int32),
        has_initial_states_p=None,
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        dflash_full_query=False,
        lattice_store_differs=False,
    )
    context = torch.full((batch, 2), eos, dtype=torch.int32)
    for row, ctx in enumerate(ngram_contexts):
        context[row] = torch.tensor(list(ctx), dtype=torch.int32)
    ple = {"query_start_loc": torch.arange(batch + 1, dtype=torch.int32), "ngram_context": context}
    return SimpleNamespace(input_ids=input_ids, positions=pos, metadata=metadata, ple=ple, real=real, batch=batch)
